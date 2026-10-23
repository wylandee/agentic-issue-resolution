"""
tests/test_execution_nodes.py - Unit tests for the remaining Phase 5 execution nodes.

All Docker SDK interactions are mocked. No real Docker daemon is required.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langgraph.graph import END, START, StateGraph

from remediation_engine.contracts.schemas import (
    CommandResult,
    IssueSource,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    SystemContext,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.orchestration.graph_wrappers import (
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.state import (
    ChangedFilesProjection,
    OrchestratorState,
    initial_orchestrator_state,
    merge_changed_files_reducer,
)
from remediation_engine.orchestration.supervisor_node import instruction_digest
from remediation_engine.orchestration.teardown_node import (
    _build_diff,
    _revert_unfixable_packages_in_json,
    run_teardown_node,
)
from remediation_engine.orchestration.tools_manifest_python import (
    restore_python_package_changes,
)
from remediation_engine.orchestration.workspace_builder import run_workspace_builder_node


def _sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
    mock.run.return_value = CommandResult(exit_code=0, duration_seconds=0.0)
    return mock


def _capture_maven_subagent_state(dispatch, runner_name: str) -> dict:
    """Capture the language/cache projection passed through a worker bridge."""
    from remediation_engine.orchestration import graph_wrappers

    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-0001",
        package_name="org.example:library",
        package_version="1.0.0",
    )
    group = VulnerabilityGroup(
        group_id="sca:pom.xml:org.example:library",
        issue_type=IssueType.SCA,
        vulnerable_component="org.example:library",
        representative_issue_id=issue.id,
        issues=[issue],
    )
    task = RemediationTask(
        task_id="task-java",
        parent_group_id=group.group_id,
        strategy=(
            RoutingStrategy.VERSION_BUMP
            if runner_name == "run_update_subagent_node"
            else RoutingStrategy.CODE_WORKAROUND
        ),
    )
    state = initial_orchestrator_state("/tmp/repo", [group])
    state.update(
        {
            "project_language": ProjectLanguage.JAVA,
            "workspace_volume": "agent_workspace_test",
            "maven_cache_volume": "agent_maven_repository_test",
            "task_queue": {task.task_id: task},
            "active_target_task_ids": [task.task_id],
        }
    )
    captured: dict = {}

    def capture_runner_state(subagent_state: dict) -> dict:
        captured.update(subagent_state)
        return {}

    runner_module = SimpleNamespace(**{runner_name: capture_runner_state})
    with (
        patch.object(graph_wrappers, "_dispatch_boundary_rejection", return_value=None),
        patch.object(graph_wrappers, "_prepare_workspace_for_dispatch", return_value=(state, [])),
        patch.object(
            graph_wrappers,
            "_create_workspace_attempt_snapshot",
            return_value=("attempt-snapshot", []),
        ),
        patch.object(graph_wrappers, "_finalize_worker_workspace_snapshot", return_value=[]),
        patch.object(graph_wrappers, "_graph_module", return_value=runner_module),
    ):
        dispatch(state)
    return captured


def test_revert_unfixable_v3_lockfile_entries_and_descendants():
    original = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "app",
                "dependencies": {"foo": "1.0.0", "bar": "1.0.0"},
            },
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/foo/node_modules/child": {"version": "1.1.0"},
            "node_modules/bar": {"version": "1.0.0"},
        },
    }
    updated = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "app",
                "dependencies": {
                    "foo": "9.9.9",
                    "bar": "2.0.0",
                    "new-package": "9.9.9",
                },
            },
            "node_modules/foo": {"version": "9.9.9"},
            "node_modules/foo/node_modules/child": {"version": "9.9.9"},
            "node_modules/foo/node_modules/added": {"version": "1.0.0"},
            "node_modules/bar": {"version": "2.0.0"},
            "node_modules/new-package": {"version": "9.9.9"},
        },
    }

    reverted = json.loads(
        _revert_unfixable_packages_in_json(
            json.dumps(original),
            json.dumps(updated),
            {"foo", "new-package"},
        )
    )

    assert reverted["packages"]["node_modules/foo"] == {"version": "1.0.0"}
    assert reverted["packages"]["node_modules/foo/node_modules/child"] == {"version": "1.1.0"}
    assert "node_modules/foo/node_modules/added" not in reverted["packages"]
    assert "node_modules/new-package" not in reverted["packages"]
    assert "new-package" not in reverted["packages"][""]["dependencies"]
    assert reverted["packages"][""]["dependencies"] == {
        "foo": "1.0.0",
        "bar": "2.0.0",
    }
    assert reverted["packages"]["node_modules/bar"] == {"version": "2.0.0"}


def test_revert_unfixable_v1_v2_dependency_trees_preserve_other_packages():
    for lockfile_version in (1, 2):
        original = {
            "name": "app",
            "lockfileVersion": lockfile_version,
            "dependencies": {
                "foo": {
                    "version": "1.0.0",
                    "dependencies": {"child": {"version": "1.1.0"}},
                },
                "bar": {
                    "version": "1.0.0",
                    "dependencies": {"foo": {"version": "1.0.0"}},
                },
                "unrelated": {"version": "1.0.0"},
            },
        }
        updated = {
            "name": "app",
            "lockfileVersion": lockfile_version,
            "dependencies": {
                "foo": {
                    "version": "9.9.9",
                    "dependencies": {
                        "child": {"version": "9.9.9"},
                        "added": {"version": "1.0.0"},
                    },
                },
                "bar": {
                    "version": "2.0.0",
                    "dependencies": {"foo": {"version": "9.9.9"}},
                },
                "unrelated": {"version": "2.0.0"},
                "ghost": {"version": "9.9.9"},
                "new-parent": {
                    "version": "2.0.0",
                    "dependencies": {"foo": {"version": "9.9.9"}},
                },
            },
        }

        reverted = json.loads(
            _revert_unfixable_packages_in_json(
                json.dumps(original),
                json.dumps(updated),
                {"foo", "ghost"},
            )
        )

        assert reverted["dependencies"]["foo"] == original["dependencies"]["foo"]
        assert reverted["dependencies"]["bar"]["version"] == "2.0.0"
        assert reverted["dependencies"]["bar"]["dependencies"]["foo"] == {"version": "1.0.0"}
        assert "dependencies" not in reverted["dependencies"]["new-parent"]
        assert "ghost" not in reverted["dependencies"]
        assert reverted["dependencies"]["unrelated"]["version"] == "2.0.0"


def test_malformed_lockfile_reversion_uses_host_text_and_reports_error():
    original = '{"packages":{"node_modules/foo":{"version":"1.0.0"}}}\n'
    errors = []

    reverted = _revert_unfixable_packages_in_json(
        original,
        "{malformed json\n",
        {"foo"},
        errors,
    )

    assert reverted == original
    assert len(errors) == 1
    assert "malformed JSON" in errors[0]


def test_unfixable_lockfile_candidate_is_absent_from_final_patch(tmp_path):
    manifest_before = {
        "name": "app",
        "dependencies": {"foo": "1.0.0", "bar": "1.0.0"},
    }
    manifest_after = {
        "name": "app",
        "dependencies": {"foo": "9.9.9", "bar": "2.0.0"},
    }
    lockfile_before = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "app",
                "dependencies": {"foo": "1.0.0", "bar": "1.0.0"},
            },
            "node_modules/foo": {"version": "1.0.0"},
            "node_modules/foo/node_modules/child": {"version": "1.1.0"},
            "node_modules/bar": {"version": "1.0.0"},
        },
    }
    lockfile_after = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "app",
                "dependencies": {"foo": "9.9.9", "bar": "2.0.0"},
            },
            "node_modules/foo": {"version": "9.9.9"},
            "node_modules/foo/node_modules/child": {"version": "9.9.9"},
            "node_modules/bar": {"version": "2.0.0"},
        },
    }
    (tmp_path / "package.json").write_text(
        json.dumps(manifest_before, indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text(
        json.dumps(lockfile_before, indent=2) + "\n",
        encoding="utf-8",
    )
    groups = [
        VulnerabilityGroup(
            group_id="group-foo",
            issue_type=IssueType.SCA,
            vulnerable_component="foo",
            file_path="package.json",
            file_paths=["package.json"],
            representative_issue_id="00000000-0000-0000-0000-000000000001",
        ),
        VulnerabilityGroup(
            group_id="group-bar",
            issue_type=IssueType.SCA,
            vulnerable_component="bar",
            file_path="package.json",
            file_paths=["package.json"],
            representative_issue_id="00000000-0000-0000-0000-000000000002",
        ),
    ]
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "task_queue": {
                "task-foo": RemediationTask(
                    task_id="task-foo",
                    parent_group_id="group-foo",
                    strategy=RoutingStrategy.VERSION_BUMP,
                    target_package_name="foo",
                    instruction="Update foo.",
                    status=TaskStatus.UNFIXABLE,
                ),
                "task-bar": RemediationTask(
                    task_id="task-bar",
                    parent_group_id="group-bar",
                    strategy=RoutingStrategy.VERSION_BUMP,
                    target_package_name="bar",
                    instruction="Update bar.",
                    status=TaskStatus.QA_PASSED,
                ),
            },
            "workspace_volume": "agent_workspace_lockfile",
            "changed_files": ["package.json", "package-lock.json"],
            "final_full_scan_completed": True,
        }
    )
    sandbox = _sandbox_mock()
    sandbox.read_file.side_effect = {
        "package.json": json.dumps(manifest_after, indent=2) + "\n",
        "package-lock.json": json.dumps(lockfile_after, indent=2) + "\n",
    }.get
    client = MagicMock()

    with (
        patch(
            "remediation_engine.orchestration.teardown_node.DockerSandbox",
            return_value=sandbox,
        ),
        patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ),
    ):
        result = run_teardown_node(state)

    assert result["changed_files"] == ["package-lock.json", "package.json"]
    assert '"foo": "9.9.9"' not in result["diff"]
    assert '"version": "9.9.9"' not in result["diff"]
    assert '"bar": "2.0.0"' in result["diff"]
    assert '"version": "2.0.0"' in result["diff"]


def test_python_teardown_restores_only_unfixable_requirements_declaration(tmp_path):
    original = "foo==1.0.0\nbar==1.0.0\n"
    updated = "foo==2.0.0\nbar==2.0.0\n"
    (tmp_path / "requirements.txt").write_text(original, encoding="utf-8")
    groups = [
        VulnerabilityGroup(
            group_id="group-foo",
            issue_type=IssueType.SCA,
            vulnerable_component="foo",
            file_path="requirements.txt",
            file_paths=["requirements.txt"],
            representative_issue_id="00000000-0000-0000-0000-000000000003",
        ),
        VulnerabilityGroup(
            group_id="group-bar",
            issue_type=IssueType.SCA,
            vulnerable_component="bar",
            file_path="requirements.txt",
            file_paths=["requirements.txt"],
            representative_issue_id="00000000-0000-0000-0000-000000000004",
        ),
    ]
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "project_language": "python",
            "task_queue": {
                "task-foo": RemediationTask(
                    task_id="task-foo",
                    parent_group_id="group-foo",
                    strategy=RoutingStrategy.VERSION_BUMP,
                    target_package_name="foo",
                    instruction="Update foo.",
                    status=TaskStatus.UNFIXABLE,
                ),
                "task-bar": RemediationTask(
                    task_id="task-bar",
                    parent_group_id="group-bar",
                    strategy=RoutingStrategy.VERSION_BUMP,
                    target_package_name="bar",
                    instruction="Update bar.",
                    status=TaskStatus.QA_PASSED,
                ),
            },
            "workspace_volume": "agent_workspace_python_manifest",
            "changed_files": ["requirements.txt"],
        }
    )
    sandbox = _sandbox_mock()
    sandbox.read_file.side_effect = {"requirements.txt": updated}.get
    client = MagicMock()

    with (
        patch(
            "remediation_engine.orchestration.teardown_node.DockerSandbox",
            return_value=sandbox,
        ),
        patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ),
    ):
        result = run_teardown_node(state)

    assert "-foo==1.0.0" not in result["diff"]
    assert "-foo==2.0.0" not in result["diff"]
    assert "bar==2.0.0" in result["diff"]


def test_restore_python_package_changes_updates_pyproject_and_pipfile_lock():
    original_pyproject = '[project]\nname = "app"\ndependencies = ["foo==1.0.0", "bar==1.0.0"]\n'
    updated_pyproject = '[project]\nname = "app"\ndependencies = ["foo==2.0.0", "bar==2.0.0"]\n'

    restored_pyproject = restore_python_package_changes(
        "pyproject.toml",
        original_pyproject,
        updated_pyproject,
        {"foo"},
    )

    assert '"foo==1.0.0"' in restored_pyproject
    assert '"bar==2.0.0"' in restored_pyproject
    original_pipfile = '[packages]\nfoo = "==1.0.0"\nbar = "==1.0.0"\n'
    updated_pipfile = '[packages]\nfoo = "==2.0.0"\nbar = "==2.0.0"\n'
    restored_pipfile = restore_python_package_changes(
        "Pipfile",
        original_pipfile,
        updated_pipfile,
        {"foo"},
    )
    original_lock = (
        '{"_meta":{"hash":{"sha256":"old"}},'
        '"default":{"foo":{"version":"==1.0.0"},'
        '"bar":{"version":"==1.0.0"}}}\n'
    )
    updated_lock = (
        '{"_meta":{"hash":{"sha256":"stale"}},'
        '"default":{"foo":{"version":"==2.0.0"},'
        '"bar":{"version":"==2.0.0"}}}\n'
    )

    restored_lock = restore_python_package_changes(
        "Pipfile.lock",
        original_lock,
        updated_lock,
        {"foo"},
        companion_manifest_text=restored_pipfile,
    )
    lock = json.loads(restored_lock)
    assert lock["default"]["foo"]["version"] == "==1.0.0"
    assert lock["default"]["bar"]["version"] == "==2.0.0"
    assert lock["_meta"]["hash"]["sha256"] != "stale"


class TestStateDefaults:
    def test_orchestrator_state_initializes_master_state_fields(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])

        assert state["constraints_ledger"] == []
        assert state["retry_counts"] == {}
        assert state["group_strategies"] == {}
        assert state["qa_evaluations"] == {}
        assert state["action_summaries"] == []
        assert state["changed_files"] == []
        assert state["package_inventory_baseline_complete"] is False
        assert "changed_files" in OrchestratorState.__annotations__
        assert state["baseline_scan_complete"] is False
        assert state["accepted_package_inventory_by_lockfile"] == {}
        assert state["current_scan_observations_by_occurrence"] == {}
        assert state["latest_successful_scan_by_task"] == {}
        assert state["workspace_volume"] is None
        assert state["status"] == "pending"
        assert "messages" not in state
        assert state["maven_cache_volume"] is None

    def test_supplied_scan_baseline_is_marked_complete(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [], issues=[])

        assert state["baseline_scan_complete"] is True


class TestMavenWorkerCacheRouting:
    def test_update_dispatch_passes_run_owned_maven_cache(self):
        state = _capture_maven_subagent_state(
            run_update_subagent_from_orchestrator,
            "run_update_subagent_node",
        )

        assert state["project_language"] == ProjectLanguage.JAVA
        assert state["maven_cache_volume"] == "agent_maven_repository_test"

    def test_workaround_dispatch_passes_run_owned_maven_cache(self):
        state = _capture_maven_subagent_state(
            run_workaround_subagent_from_orchestrator,
            "run_workaround_subagent_node",
        )

        assert state["project_language"] == ProjectLanguage.JAVA
        assert state["maven_cache_volume"] == "agent_maven_repository_test"

    def test_package_scope_preserves_the_full_supplied_scan_baseline(self, tmp_path):
        def issue(package_name: str, cve_id: str) -> VulnerabilityIssue:
            return VulnerabilityIssue(
                source=IssueSource.ODC,
                issue_type=IssueType.SCA,
                severity=Severity.HIGH,
                cve_id=cve_id,
                package_name=package_name,
                package_version="1.0.0",
            )

        state = initial_orchestrator_state(
            str(tmp_path),
            [],
            issues=[issue("selected", "CVE-2025-0001"), issue("outside", "CVE-2025-0002")],
            system_context=SystemContext(environment="development"),
            target_packages=["selected"],
        )

        assert state["baseline_scan_identifiers"] == ["CVE-2025-0001", "CVE-2025-0002"]
        assert [item.package_name for item in state["issues"]] == ["selected"]

    def test_graph_preserves_changed_file_channel_and_teardown_projection(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node(
            "candidate",
            lambda _state: {"changed_files": ["frontend/package-lock.json"]},
        )
        workflow.add_node(
            "final",
            lambda _state: {"changed_files": ChangedFilesProjection(["frontend/package.json"])},
        )
        workflow.add_edge(START, "candidate")
        workflow.add_edge("candidate", "final")
        workflow.add_edge("final", END)

        result = workflow.compile().invoke({"changed_files": []})

        assert result["changed_files"] == ["frontend/package.json"]


class TestWorkspaceBuilderNode:
    def test_invalid_repo_root_returns_workspace_build_failed(self):
        result = run_workspace_builder_node({"repo_root": "/nonexistent/xyz"})

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None

    def test_success_creates_volume_and_copies_repo(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "frontend").mkdir()
        (tmp_path / "frontend" / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "package-lock.json").write_text(
            '{"packages":{"":{"name":"root","version":"1.0.0"}}}',
            encoding="utf-8",
        )
        (tmp_path / "frontend" / "package-lock.json").write_text(
            '{"packages":{"":{"name":"frontend","version":"1.0.0"}}}',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox = _sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ) as mock_sandbox,
        ):
            result = run_workspace_builder_node(state)

        client.volumes.create.assert_called_once()
        mock_sandbox.assert_called_once()
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "npm install --package-lock=true",
            "cd frontend && npm install --package-lock=true",
        ]
        assert result["status"] == "workspace_ready"
        assert result["workspace_volume"].startswith("agent_workspace_")
        inventory = result["accepted_package_inventory_by_lockfile"]
        assert result["package_inventory_baseline_complete"] is True
        assert len(inventory) == 2
        assert any(key.startswith('["package-lock.json",') for key in inventory)
        assert any(key.startswith('["frontend/package-lock.json",') for key in inventory)

    def test_no_package_repository_skips_install(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        client = MagicMock()
        sandbox = _sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(state)

        sandbox.run.assert_not_called()
        assert result["status"] == "workspace_ready"

    def test_install_failure_preserves_volume_and_diagnostics(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        sandbox = _sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            duration_seconds=0.0,
            stdout="install output",
            stderr="npm ERR! dependency failure",
        )

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(state)

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")
        assert "npm install failed in ." in result["errors"][0]
        assert "npm ERR! dependency failure" in result["errors"][0]

    def test_copy_failure_preserves_volume_for_teardown(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.workspace_builder.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace_builder.DockerSandbox",
                side_effect=RuntimeError("copy failed"),
            ),
        ):
            result = run_workspace_builder_node(state)

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"].startswith("agent_workspace_")


class TestTeardownNode:
    def test_final_changed_file_projection_replaces_historical_candidates(self):
        assert merge_changed_files_reducer(["stale.ts"], ChangedFilesProjection(["actual.ts"])) == [
            "actual.ts"
        ]

    def test_diff_marks_missing_final_newline(self):
        diff = _build_diff("src/app.js", "const before = true;", "const after = true;")

        assert "\\ No newline at end of file" in diff

    def test_final_state_barrier_detaches_terminal_worker_attempt(self, tmp_path):
        instruction = "Apply the workaround."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-terminal",
            task_id="task-1",
            state_revision=1,
            task_revision=1,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="workaround_subagent",
        )
        state = initial_orchestrator_state(str(tmp_path), [])
        state["task_queue"] = {
            "task-1": RemediationTask(
                task_id="task-1",
                parent_group_id="g1",
                strategy=RoutingStrategy.CODE_WORKAROUND,
                strategy_stage=SCARemediationStage.CODE_WORKAROUND,
                instruction=instruction,
                status=TaskStatus.UNFIXABLE,
                task_revision=1,
                current_attempt_id=snapshot.attempt_id,
            )
        }
        state["attempt_snapshots_by_id"] = {snapshot.attempt_id: snapshot}

        result = run_teardown_node(state)

        assert result["task_queue"]["task-1"].current_attempt_id is None
        assert result["active_target_task_ids"] == []
        assert any(
            event.error_code == "TERMINAL_TASK_FIELDS_NORMALIZED"
            for event in result["consistency_events"]
        )

    def test_final_state_barrier_repairs_passed_parent_with_failed_pivot_child(self, tmp_path):
        parent = RemediationTask(
            task_id="task-parent",
            parent_group_id="group-parent",
            strategy=RoutingStrategy.VERSION_BUMP,
            strategy_stage=SCARemediationStage.NPM_LATEST,
            instruction="Update the package.",
            status=TaskStatus.QA_PASSED,
        )
        child = RemediationTask(
            task_id="task-child",
            parent_group_id="group-child",
            parent_task_id="task-parent",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            instruction="Apply a workaround.",
            status=TaskStatus.UNFIXABLE,
        )
        state = initial_orchestrator_state(str(tmp_path), [])
        state["task_queue"] = {parent.task_id: parent, child.task_id: child}

        result = run_teardown_node(state)

        assert result["task_queue"]["task-parent"].status == TaskStatus.PIVOTED
        assert result["status"] == "completed_with_errors"
        assert any(
            event.error_code == "PIVOT_PARENT_STATUS_REPAIRED"
            for event in result["consistency_events"]
        )

    def test_changed_files_are_deduplicated_diffed_and_volume_removed(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["status"] = "edits_completed"
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["changed_files"] = ["routes/login.ts", "routes/login.ts"]

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = "const x = 2;\n"
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        sandbox.read_file.assert_called_once_with("routes/login.ts")
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None
        assert result["changed_files"] == ["routes/login.ts"]
        assert "a/routes/login.ts" in result["diff"]
        assert "b/routes/login.ts" in result["diff"]

    def test_teardown_removes_workspace_and_maven_cache_volumes(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        volumes = {
            state["workspace_volume"]: MagicMock(),
            state["maven_cache_volume"]: MagicMock(),
        }
        client.volumes.get.side_effect = volumes.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert client.volumes.get.call_args_list[0].args == ("agent_workspace_deadbeef",)
        assert client.volumes.get.call_args_list[1].args == ("agent_maven_repository_deadbeef",)
        for volume in volumes.values():
            volume.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] is None

    def test_teardown_reports_cache_failure_without_claiming_workspace_live(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        workspace = MagicMock()
        maven_cache = MagicMock()
        maven_cache.remove.side_effect = RuntimeError("permission denied")
        client.volumes.get.side_effect = {
            state["workspace_volume"]: workspace,
            state["maven_cache_volume"]: maven_cache,
        }.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert [call.args[0] for call in client.volumes.get.call_args_list] == [
            "agent_workspace_deadbeef",
            "agent_maven_repository_deadbeef",
        ]
        workspace.remove.assert_called_once_with(force=True)
        maven_cache.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] == "agent_maven_repository_deadbeef"
        assert result["status"] == "completed_with_errors"
        assert any("failed to remove Maven cache volume" in error for error in result["errors"])

    def test_teardown_still_removes_cache_when_workspace_removal_fails(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        workspace = MagicMock()
        workspace.remove.side_effect = RuntimeError("workspace volume locked")
        maven_cache = MagicMock()
        client.volumes.get.side_effect = {
            state["workspace_volume"]: workspace,
            state["maven_cache_volume"]: maven_cache,
        }.__getitem__

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        assert [call.args[0] for call in client.volumes.get.call_args_list] == [
            "agent_workspace_deadbeef",
            "agent_maven_repository_deadbeef",
        ]
        workspace.remove.assert_called_once_with(force=True)
        maven_cache.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] == "agent_workspace_deadbeef"
        assert result["maven_cache_volume"] is None
        assert any("failed to remove workspace volume" in error for error in result["errors"])

    def test_unexpected_teardown_failure_still_cleans_both_volumes(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["maven_cache_volume"] = "agent_maven_repository_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        volumes = {
            state["workspace_volume"]: MagicMock(),
            state["maven_cache_volume"]: MagicMock(),
        }
        client.volumes.get.side_effect = volumes.__getitem__

        with (
            patch(
                "remediation_engine.orchestration.supervisor_node.reconcile_phase5_state_before_teardown",
                side_effect=RuntimeError("terminal graph failure"),
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        for volume in volumes.values():
            volume.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None
        assert result["maven_cache_volume"] is None
        assert any("terminal graph failure" in error for error in result["errors"])

    def test_changed_files_are_derived_from_actual_final_content(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "insecurity.ts").write_text("const safe = true;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["changed_files"] = ["routes/login.ts", "lib/insecurity.ts"]

        sandbox = _sandbox_mock()
        sandbox.read_file.side_effect = lambda path: {
            "routes/login.ts": "const x = 2;\n",
            "lib/insecurity.ts": "const safe = true;\n",
        }.get(path)
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        assert result["changed_files"] == ["routes/login.ts"]
        assert "lib/insecurity.ts" not in result["diff"]

    def test_no_changed_files_still_removes_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
            ) as mock_sandbox,
        ):
            result = run_teardown_node(state)

        mock_sandbox.assert_not_called()
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["changed_files"] == []
        assert result["diff"] == ""

    def test_invalid_changed_file_type_is_reported(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["changed_files"] = [None]

        result = run_teardown_node(state)

        assert result["changed_files"] == []
        assert any("path must be a string" in error for error in result["errors"])

    def test_attached_container_is_removed_before_volume(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        attached = MagicMock()
        attached.name = "/adoring_hertz"
        client.containers.list.return_value = [attached]

        with patch(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            return_value=client,
        ):
            result = run_teardown_node(state)

        client.containers.list.assert_called_once_with(
            all=True,
            filters={"volume": "agent_workspace_deadbeef"},
        )
        attached.remove.assert_called_once_with(force=True)
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["workspace_volume"] is None

    def test_volume_conflict_is_retried_after_attached_container_cleanup(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        client.volumes.get.return_value.remove.side_effect = [
            RuntimeError("409 Client Error: Conflict (volume is in use)"),
            None,
        ]

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.teardown_node.time.sleep") as sleep,
        ):
            result = run_teardown_node(state)

        assert client.volumes.get.return_value.remove.call_count == 2
        sleep.assert_called_once()
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None

    def test_persistent_volume_conflict_is_reported_without_raising(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        client = MagicMock()
        client.containers.list.return_value = []
        client.volumes.get.return_value.remove.side_effect = RuntimeError(
            "409 Client Error: Conflict (volume is in use)"
        )

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.teardown_node.time.sleep"),
        ):
            result = run_teardown_node(state)

        assert result["status"] == "completed_with_errors"
        assert result["workspace_volume"] == "agent_workspace_deadbeef"
        assert any("failed to remove workspace volume" in error for error in result["errors"])

    def test_existing_errors_produce_completed_with_errors_status(self, tmp_path):
        """Teardown preserves a failed worker outcome in its terminal status."""
        state = initial_orchestrator_state(str(tmp_path), [])
        state["errors"] = ["worker surrendered"]

        result = run_teardown_node(state)

        assert result["status"] == "completed_with_errors"


# Stage-2 behavior retained during the architecture merge.
class TestTeardownNodeStage2Parity:
    def test_changed_files_are_deduplicated_diffed_and_volume_removed_python_stage2(self, tmp_path):
        route_dir = tmp_path / "routes"
        route_dir.mkdir()
        (route_dir / "login.ts").write_text("const x = 1;\n", encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [])
        state["status"] = "edits_completed"
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["changed_files"] = ["routes/login.ts", "routes/login.ts"]
        state["project_language"] = "python"

        sandbox = _sandbox_mock()
        sandbox.read_file.return_value = "const x = 2;\n"
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.teardown_node.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
            patch(
                "remediation_engine.orchestration.teardown_node.get_docker_client",
                return_value=client,
            ),
        ):
            result = run_teardown_node(state)

        sandbox.read_file.assert_called_once_with("routes/login.ts")
        sandbox_type.assert_called_once_with(
            repo_root=None,
            image=LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image,
            workspace_volume="agent_workspace_deadbeef",
        )
        client.volumes.get.assert_called_once_with("agent_workspace_deadbeef")
        client.volumes.get.return_value.remove.assert_called_once_with(force=True)
        assert result["status"] == "completed"
        assert result["workspace_volume"] is None
        assert result["changed_files"] == ["routes/login.ts"]
        assert "a/routes/login.ts" in result["diff"]
        assert "b/routes/login.ts" in result["diff"]
