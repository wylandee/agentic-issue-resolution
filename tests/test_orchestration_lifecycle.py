"""Focused tests for orchestration lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from langgraph.graph import END, START, StateGraph

import remediation_engine.orchestration.execution as graph_wrappers
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
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.orchestration.execution import (
    run_update_subagent_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.graph import (
    build_orchestrator_graph,
    route_after_post_qa_triage,
    route_after_triage,
    run_orchestrator,
    triage_node,
)
from remediation_engine.orchestration.state import (
    ChangedFilesProjection,
    OrchestratorState,
    initial_orchestrator_state,
    instruction_digest,
    merge_changed_files_reducer,
)
from remediation_engine.orchestration.workspace import (
    _build_diff,
    _revert_unfixable_packages_in_json,
    run_teardown_node,
    run_workspace_builder_node,
)
from remediation_engine.runtime.agent_runtime import MAX_SUBAGENT_TOOL_CALL_ROUNDS
from remediation_engine.runtime.telemetry import (
    TrajectoryRecorder,
    build_phase5_runnable_config,
    mark_phase5_trace_failed,
    resolve_phase5_trace_url,
    use_trajectory_recorder,
)
from remediation_engine.settings import (
    AppSettings,
    get_bound_runtime_settings,
    get_runtime_settings,
    openai_api_key_kwargs,
    use_runtime_settings,
)
from remediation_engine.tools.python_mutations import restore_python_package_changes


def _execution_node_sandbox_mock() -> MagicMock:
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)
    mock.run.return_value = CommandResult(exit_code=0, duration_seconds=0.0)
    return mock


def _capture_maven_subagent_state(dispatch, runner_name: str) -> dict:
    """Capture the language/cache projection passed through a worker bridge."""

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
    sandbox = _execution_node_sandbox_mock()
    sandbox.read_file.side_effect = {
        "package.json": json.dumps(manifest_after, indent=2) + "\n",
        "package-lock.json": json.dumps(lockfile_after, indent=2) + "\n",
    }.get
    client = MagicMock()

    with (
        patch(
            "remediation_engine.orchestration.workspace.DockerSandbox",
            return_value=sandbox,
        ),
        patch(
            "remediation_engine.orchestration.workspace.get_docker_client",
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
    sandbox = _execution_node_sandbox_mock()
    sandbox.read_file.side_effect = {"requirements.txt": updated}.get
    client = MagicMock()

    with (
        patch(
            "remediation_engine.orchestration.workspace.DockerSandbox",
            return_value=sandbox,
        ),
        patch(
            "remediation_engine.orchestration.workspace.get_docker_client",
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


class TestWorkspaceBuilderExecutionNode:
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
        sandbox = _execution_node_sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
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
        sandbox = _execution_node_sandbox_mock()

        with (
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
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
        sandbox = _execution_node_sandbox_mock()
        sandbox.run.return_value = CommandResult(
            exit_code=1,
            duration_seconds=0.0,
            stdout="install output",
            stderr="npm ERR! dependency failure",
        )

        with (
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
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
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
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

        sandbox = _execution_node_sandbox_mock()
        sandbox.read_file.return_value = "const x = 2;\n"
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
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
            "remediation_engine.orchestration.workspace.get_docker_client",
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
            "remediation_engine.orchestration.workspace.get_docker_client",
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
            "remediation_engine.orchestration.workspace.get_docker_client",
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
                "remediation_engine.orchestration.state.reconcile_phase5_state_before_teardown",
                side_effect=RuntimeError("terminal graph failure"),
            ),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
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

        sandbox = _execution_node_sandbox_mock()
        sandbox.read_file.side_effect = lambda path: {
            "routes/login.ts": "const x = 2;\n",
            "lib/insecurity.ts": "const safe = true;\n",
        }.get(path)
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
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
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
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
            "remediation_engine.orchestration.workspace.get_docker_client",
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
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.workspace.time.sleep") as sleep,
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
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch("remediation_engine.orchestration.workspace.time.sleep"),
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

        sandbox = _execution_node_sandbox_mock()
        sandbox.read_file.return_value = "const x = 2;\n"
        client = MagicMock()

        with (
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
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


_VOLUME_HEX = "deadbeefcafebabe"


def _workspace_builder_sandbox_mock(*, result: CommandResult | None = None) -> MagicMock:
    sandbox = MagicMock()
    sandbox.__enter__ = MagicMock(return_value=sandbox)
    sandbox.__exit__ = MagicMock(return_value=None)
    sandbox.run.return_value = result or CommandResult(exit_code=0, duration_seconds=0.0)
    return sandbox


def _volume_name_patch():
    return patch(
        "remediation_engine.orchestration.workspace.uuid.uuid4",
        return_value=SimpleNamespace(hex=_VOLUME_HEX),
    )


class TestWorkspaceBuilderNode:
    def test_created_volume_is_registered_on_the_active_run_recorder(self, tmp_path):
        recorder = TrajectoryRecorder()
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
            use_trajectory_recorder(recorder),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        assert result["workspace_volume"] == "agent_workspace_deadbeef"
        assert recorder.workspace_volume == result["workspace_volume"]

    def test_success_creates_named_volume_copies_repo_and_returns_ready_projection(self, tmp_path):
        """The builder provisions one volume and hands the repository to the sandbox."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("console.log('ready');", encoding="utf-8")
        (tmp_path / "packages" / "web").mkdir(parents=True)
        (tmp_path / "packages" / "web" / "package.json").write_text("{}", encoding="utf-8")

        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        client.volumes.create.assert_called_once_with(name=volume_name)
        client.close.assert_called_once_with()
        sandbox_type.assert_called_once_with(
            tmp_path,
            workspace_volume=volume_name,
            image="node:22",
            maven_repository_volume=None,
            archive_excluded_dirs=frozenset(),
        )
        sandbox.__enter__.assert_called_once_with()
        sandbox.__exit__.assert_called_once()
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "npm install --package-lock=true",
            "cd packages/web && npm install --package-lock=true",
        ]
        assert all(call.kwargs == {"timeout": 900} for call in sandbox.run.call_args_list)
        assert result["workspace_volume"] == volume_name
        assert result["status"] == "workspace_ready"
        assert result["accepted_package_inventory_by_lockfile"] == {}
        assert result["package_inventory_baseline_complete"] is True

    def test_volume_creation_failure_closes_client_and_does_not_start_sandbox(self, tmp_path):
        """A failed volume transaction reports failure and releases its client."""
        client = MagicMock()
        client.volumes.create.side_effect = RuntimeError("volume quota exceeded")
        sandbox_type = MagicMock()

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        client.close.assert_called_once_with()
        sandbox_type.assert_not_called()
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None
        assert "volume quota exceeded" in result["errors"][0]

    def test_sandbox_setup_failure_preserves_volume_for_teardown(self, tmp_path):
        """Once provisioned, a setup error retains the volume name for cleanup."""
        client = MagicMock()
        sandbox_type = MagicMock(side_effect=RuntimeError("repository copy failed"))

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        client.volumes.create.assert_called_once_with(name=volume_name)
        client.close.assert_called_once_with()
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] == volume_name
        assert "repository copy failed" in result["errors"][0]

    def test_dependency_install_failure_returns_diagnostics_and_keeps_volume(self, tmp_path):
        """A failed npm command is surfaced while retaining the initialized volume."""
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock(
            result=CommandResult(
                exit_code=1,
                stdout="npm output",
                stderr="npm ERR! install failed",
                duration_seconds=0.0,
            )
        )

        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node({"repo_root": str(tmp_path)})

        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"
        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] == volume_name
        assert "npm install failed in ." in result["errors"][0]
        assert "npm ERR! install failed" in result["errors"][0]
        sandbox.__exit__.assert_called_once()


class TestWorkspaceBuilderNodeStage2Parity:
    def test_incompatible_python_requirement_fails_before_volume_creation(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "future-only"\nrequires-python = ">=3.12"\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox_type = MagicMock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None
        assert "incompatible with Python 3.11" in result["errors"][0]
        client.volumes.create.assert_not_called()
        sandbox_type.assert_not_called()

    def test_pipfile_without_lock_runs_pipenv_install(self, tmp_path):
        (tmp_path / "Pipfile").write_text(
            '[requires]\npython_version = "3.11"\n[packages]\nrequests = "*"\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        commands = [call.args[0] for call in sandbox.run.call_args_list]
        assert "PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv install --dev" in commands

        assert ".remedy-pipenv/bin/python -m pip install --upgrade pip==26.2.1" in commands
        assert ".venv/bin/python -m pip install --upgrade pip==26.2.1" in commands
        assert not any(command.endswith("pipenv sync --dev") for command in commands)
        assert result["status"] == "workspace_ready"

    def test_poetry_only_project_fails_before_volume_creation(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.poetry.dependencies]\npython = "^3.11"\nrequests = "^2.31"\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox_type = MagicMock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None
        assert "Poetry-managed Python dependencies are unsupported" in result["errors"][0]
        client.volumes.create.assert_not_called()
        sandbox_type.assert_not_called()

    def test_python_pipfile_is_authoritative_and_uses_persistent_volume_environments(
        self, tmp_path
    ):
        """Pipenv provisions the app venv and ignores coexisting requirements files."""
        (tmp_path / "Pipfile").write_text(
            '[requires]\npython_version = "3.11"\n[packages]\nrequests = "*"\n',
            encoding="utf-8",
        )
        (tmp_path / "Pipfile.lock").write_text("{}", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
        (tmp_path / ".venv").mkdir()
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )
        volume_name = f"agent_workspace_{_VOLUME_HEX[:8]}"

        sandbox_type.assert_called_once_with(
            tmp_path,
            image=LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image,
            workspace_volume=volume_name,
            maven_repository_volume=None,
            archive_excluded_dirs=LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].excluded_dirs,
        )
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "rm -rf .venv .remedy-pipenv",
            "python -m venv .remedy-pipenv",
            ".remedy-pipenv/bin/python -m pip install --upgrade pip==26.2.1",
            ".remedy-pipenv/bin/python -m pip install pipenv",
            "PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv --python 3.11",
            ".venv/bin/python -m pip install --upgrade pip==26.2.1",
            "PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv sync --dev",
            (
                "if ! .venv/bin/python -c 'import pytest' >/dev/null 2>&1; "
                "then .venv/bin/python -m pip install pytest; fi"
            ),
        ]
        assert result["workspace_volume"] == volume_name
        assert result["status"] == "workspace_ready"

    def test_python_requirements_and_static_pep621_metadata_install_inside_volume(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests==2.31.0\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nname = "sample"\nrequires-python = ">=3.11"\ndependencies = []\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_type,
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": "python"}
            )

        sandbox_type.assert_called_once()
        assert (
            sandbox_type.call_args.kwargs["image"]
            == LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image
        )
        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "rm -rf .venv .remedy-pipenv",
            "python -m venv .venv",
            ".venv/bin/python -m pip install --upgrade pip==26.2.1",
            ".venv/bin/python -m pip install -r requirements.txt",
            ".venv/bin/python -m pip install -e .",
            (
                "if ! .venv/bin/python -c 'import pytest' >/dev/null 2>&1; "
                "then .venv/bin/python -m pip install pytest; fi"
            ),
        ]
        assert result["status"] == "workspace_ready"

    def test_python_setup_command_failure_keeps_volume_and_bounded_diagnostics(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\n", encoding="utf-8")
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock(
            result=CommandResult(
                exit_code=1,
                stdout="python setup output",
                stderr="pip installation failed",
                duration_seconds=0.0,
            )
        )
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] == f"agent_workspace_{_VOLUME_HEX[:8]}"
        assert "Python workspace setup command failed" in result["errors"][0]
        assert "pip installation failed" in result["errors"][0]
        sandbox.__exit__.assert_called_once()

    def test_python_tool_only_pyproject_installs_pytest_without_editable_install(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert [call.args[0] for call in sandbox.run.call_args_list] == [
            "rm -rf .venv .remedy-pipenv",
            "python -m venv .venv",
            ".venv/bin/python -m pip install --upgrade pip==26.2.1",
            (
                "if ! .venv/bin/python -c 'import pytest' >/dev/null 2>&1; "
                "then .venv/bin/python -m pip install pytest; fi"
            ),
        ]
        assert result["status"] == "workspace_ready"

    def test_static_setup_cfg_metadata_installs_project_editably(self, tmp_path):
        (tmp_path / "setup.cfg").write_text(
            "[metadata]\nname = sample\n[options]\npackages = find:\n",
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox = _workspace_builder_sandbox_mock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                return_value=sandbox,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert ".venv/bin/python -m pip install -e ." in [
            call.args[0] for call in sandbox.run.call_args_list
        ]
        assert result["status"] == "workspace_ready"

    def test_uv_only_dependencies_fail_before_volume_creation(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.uv]\ndev-dependencies = ["pytest"]\n',
            encoding="utf-8",
        )
        client = MagicMock()
        sandbox_type = MagicMock()
        with (
            _volume_name_patch(),
            patch(
                "remediation_engine.orchestration.workspace.get_docker_client",
                return_value=client,
            ),
            patch(
                "remediation_engine.orchestration.workspace.DockerSandbox",
                sandbox_type,
            ),
        ):
            result = run_workspace_builder_node(
                {"repo_root": str(tmp_path), "project_language": ProjectLanguage.PYTHON}
            )

        assert result["status"] == "workspace_build_failed"
        assert result["workspace_volume"] is None
        assert "uv-managed Python dependencies are unsupported" in result["errors"][0]
        client.volumes.create.assert_not_called()
        sandbox_type.assert_not_called()


def _issue(issue_id: str | None = None) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        id=issue_id or str(uuid4()),
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )


def _group(group_id: str | None = None) -> VulnerabilityGroup:
    issue = _issue()
    return VulnerabilityGroup(
        group_id=group_id or str(uuid4()),
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path="package.json",
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _initial_state(tmp_path: Path, issues: list[VulnerabilityIssue], system_context: SystemContext):
    return {
        "repo_root": str(tmp_path),
        "valid_groups": [],
        "issues": issues,
        "system_context": system_context,
        "constraints_ledger": [],
        "retry_counts": {},
        "group_strategies": {},
        "qa_evaluations": {},
        "action_summaries": [],
        "changed_files": [],
        "workspace_volume": None,
        "status": "pending",
        "errors": [],
    }


class TestTriageNodeIntegration:
    def test_triage_records_a_named_observability_span(self, tmp_path: Path):
        """Initial triage is visible in the local/LangSmith callback trace."""
        issues = [_issue()]
        state = _initial_state(tmp_path, issues, SystemContext(scan_id="trace-test"))
        recorder = TrajectoryRecorder()
        with (
            patch(
                "remediation_engine.orchestration.graph.run_triage_pipeline",
                return_value=[],
            ),
            use_trajectory_recorder(recorder),
        ):
            triage_node(state)

        assert any(span["name"] == "triage.pipeline" for span in recorder.spans())

    def test_route_after_triage(self):
        assert route_after_triage({"status": "triage_completed"}) == "workspace_builder"
        assert route_after_triage({"status": "triage_skipped"}) == "workspace_builder"
        assert route_after_triage({"status": "triage_completed_no_work"}) == "teardown"
        assert route_after_triage({"status": "failed"}) == "teardown"

    def test_route_after_post_qa_triage_only_dispatches_actionable_work(self):
        assert route_after_post_qa_triage({"status": "triage_completed"}) == "supervisor"
        assert route_after_post_qa_triage({"status": "triage_completed_no_work"}) == "teardown"
        assert route_after_post_qa_triage({"status": "triage_failed"}) == "teardown"

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_skipped_when_no_issues(self, mock_pipeline, tmp_path: Path):
        # No issues provided
        state = {
            "repo_root": str(tmp_path),
            "valid_groups": [],
            "constraints_ledger": [],
            "retry_counts": {},
            "group_strategies": {},
            "qa_evaluations": {},
            "action_summaries": [],
            "changed_files": [],
            "workspace_volume": None,
            "status": "pending",
            "errors": [],
        }

        # We need to mock the rest of the nodes so we don't actually run them
        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_ws.return_value = {"status": "workspace_ready"}
            mock_teardown.return_value = {"status": "completed"}

            graph.invoke(state)

            mock_pipeline.assert_not_called()
            mock_ws.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_completed_with_valid_groups(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        group1 = _group()
        triage_res = TriageResult(
            group_id=group1.group_id,
            is_valid=True,
            false_positive_reason=None,
            revised_priority=Severity.HIGH,
            recommended_issue_id=issues[0].id,
            priority_reasoning="test reasoning",
            validity_confidence_score=0.9,
            priority_confidence_score=0.9,
            triage_method="static",
        )

        # Mock run_triage_pipeline to return one valid group
        mock_pipeline.return_value = [(group1, triage_res)]

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_supervisor_node") as mock_supervisor,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_ws.return_value = {"status": "workspace_ready"}
            mock_supervisor.return_value = {
                "status": "supervisor_routed",
                "next_routing_step": "teardown",
                "feedback_by_group": {},
                "supervisor_instructions": "done",
            }
            mock_teardown.return_value = {"status": "phase5_refactor_blocked"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once_with(issues, system_context, str(tmp_path))
            assert group1.group_id in [g.group_id for g in result.get("valid_groups", [])]
            mock_ws.assert_called_once()
            mock_supervisor.assert_called_once()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_completed_no_work(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        group1 = _group()
        triage_res = TriageResult(
            group_id=group1.group_id,
            is_valid=False,
            false_positive_reason="test code",
            revised_priority=Severity.INFO,
            recommended_issue_id=issues[0].id,
            priority_reasoning="test reasoning",
            validity_confidence_score=0.9,
            priority_confidence_score=0.9,
            triage_method="static",
        )

        # Mock run_triage_pipeline to return only invalid groups
        mock_pipeline.return_value = [(group1, triage_res)]

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_teardown.return_value = {"status": "completed"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once_with(issues, system_context, str(tmp_path))
            assert len(result.get("valid_groups", [])) == 0
            # Should skip workspace builder and go directly to teardown
            mock_ws.assert_not_called()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.run_triage_pipeline")
    def test_triage_failure_routes_to_teardown(self, mock_pipeline, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        mock_pipeline.side_effect = RuntimeError("Something went wrong")

        state = _initial_state(tmp_path, issues, system_context)

        with (
            patch("remediation_engine.orchestration.graph.run_workspace_builder_node") as mock_ws,
            patch("remediation_engine.orchestration.graph.run_teardown_node") as mock_teardown,
        ):
            graph = build_orchestrator_graph()

            mock_teardown.return_value = {"status": "completed"}

            result = graph.invoke(state)

            mock_pipeline.assert_called_once()
            assert len(result.get("valid_groups", [])) == 0
            assert len(result.get("errors", [])) > 0
            mock_ws.assert_not_called()
            mock_teardown.assert_called_once()

    @patch("remediation_engine.orchestration.graph.orchestrator_engine")
    def test_run_orchestrator_with_triage_params(self, mock_engine, tmp_path: Path):
        issues = [_issue()]
        system_context = SystemContext(scan_id="test")

        mock_engine.invoke.return_value = {"status": "completed", "valid_groups": [_group()]}

        with patch(
            "remediation_engine.orchestration.graph.build_phase5_runnable_config",
            return_value=(None, None),
        ):
            result = run_orchestrator(
                str(tmp_path),
                valid_groups=[],
                issues=issues,
                system_context=system_context,
            )

        invoked_state = mock_engine.invoke.call_args[0][0]
        assert invoked_state["issues"] == issues
        assert invoked_state["system_context"] == system_context
        assert invoked_state["repo_root"] == str(tmp_path)
        assert result["status"] == "completed"


def test_use_runtime_settings_binds_settings_for_current_run() -> None:
    settings = AppSettings(openai_api_key="bound-key", remedy_llm_model="bound-model")

    assert get_bound_runtime_settings() is None

    with use_runtime_settings(settings):
        assert get_bound_runtime_settings() is settings
        assert get_runtime_settings() is settings

    assert get_bound_runtime_settings() is None


def test_use_runtime_settings_restores_outer_binding_for_nested_contexts() -> None:
    outer_settings = AppSettings(openai_api_key="outer-key")
    inner_settings = AppSettings(openai_api_key="inner-key")

    with use_runtime_settings(outer_settings):
        assert get_bound_runtime_settings() is outer_settings

        with use_runtime_settings(inner_settings):
            assert get_bound_runtime_settings() is inner_settings
            assert get_runtime_settings() is inner_settings

        assert get_bound_runtime_settings() is outer_settings

    assert get_bound_runtime_settings() is None


def test_use_runtime_settings_restores_binding_when_context_raises() -> None:
    outer_settings = AppSettings(openai_api_key="outer-key")
    inner_settings = AppSettings(openai_api_key="inner-key")

    def raise_from_inner_context() -> None:
        with use_runtime_settings(inner_settings):
            assert get_bound_runtime_settings() is inner_settings
            raise RuntimeError("context failure")

    with (
        use_runtime_settings(outer_settings),
        pytest.raises(RuntimeError, match="context failure"),
    ):
        raise_from_inner_context()
        assert get_bound_runtime_settings() is outer_settings

    assert get_bound_runtime_settings() is None


def test_openai_client_receives_bound_key_without_environment(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = AppSettings(openai_api_key="injected-test-key")

    with use_runtime_settings(settings):
        assert openai_api_key_kwargs() == {"api_key": "injected-test-key"}

    with use_runtime_settings(AppSettings(openai_api_key="")):
        assert openai_api_key_kwargs() == {}


def _langsmith_config_group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        id=str(uuid4()),
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )
    return VulnerabilityGroup(
        group_id=str(uuid4()),
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path="package.json",
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
    )


class TestPhase5LangsmithConfig:
    def test_build_phase5_runnable_config_returns_none_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "false")

        config, run_id = build_phase5_runnable_config(
            "D:/repos/juice-shop", [_langsmith_config_group()]
        )

        assert config is None
        assert run_id is None

    def test_build_phase5_runnable_config_includes_expected_metadata(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        groups = [_langsmith_config_group(), _langsmith_config_group()]
        config, run_id = build_phase5_runnable_config("D:/repos/juice-shop", groups)

        assert config is not None
        assert isinstance(run_id, UUID)
        assert config["run_id"] == run_id
        assert config["run_name"] == "phase5_orchestrator"
        assert config["tags"] == ["phase-5", "orchestrator", "langgraph"]
        assert config["metadata"] == {
            "repo_name": "juice-shop",
            "repo_root": "D:/repos/juice-shop",
            "vulnerability_group_count": 2,
            "max_tool_call_rounds": MAX_SUBAGENT_TOOL_CALL_ROUNDS,
        }


class TestPhase5TraceUrlResolution:
    @patch("remediation_engine.runtime.telemetry.wait_for_all_tracers")
    @patch("remediation_engine.runtime.telemetry.Client")
    def test_resolve_phase5_trace_url_returns_url(self, mock_client_cls, mock_wait, monkeypatch):
        monkeypatch.setenv("LANGSMITH_PROJECT", "AppSec-Remediation-Engine")
        run_id = uuid4()
        run = MagicMock()
        client = mock_client_cls.return_value
        client.read_run.return_value = run
        client.get_run_url.return_value = "https://smith.langchain.com/o/test/projects/p/runs/r"

        result = resolve_phase5_trace_url(run_id)

        assert result == "https://smith.langchain.com/o/test/projects/p/runs/r"
        mock_wait.assert_called_once_with()
        client.read_run.assert_called_once_with(run_id)
        client.get_run_url.assert_called_once_with(
            run=run,
            project_name="AppSec-Remediation-Engine",
        )

    @patch("remediation_engine.runtime.telemetry.wait_for_all_tracers")
    @patch("remediation_engine.runtime.telemetry.Client")
    def test_resolve_phase5_trace_url_returns_none_when_lookup_fails(
        self,
        mock_client_cls,
        mock_wait,
    ):
        run_id = uuid4()
        client = mock_client_cls.return_value
        client.read_run.side_effect = RuntimeError("boom")

        result = resolve_phase5_trace_url(run_id)

        assert result is None
        mock_wait.assert_called_once_with()

    @patch("remediation_engine.runtime.telemetry.wait_for_all_tracers")
    @patch("remediation_engine.runtime.telemetry.Client")
    def test_resolve_phase5_trace_url_returns_none_when_url_build_fails(
        self,
        mock_client_cls,
        mock_wait,
    ):
        run_id = uuid4()
        run = MagicMock()
        client = mock_client_cls.return_value
        client.read_run.return_value = run
        client.get_run_url.side_effect = RuntimeError("boom")

        result = resolve_phase5_trace_url(run_id)

        assert result is None
        mock_wait.assert_called_once_with()


class TestPhase5TraceFailure:
    @patch("remediation_engine.runtime.telemetry.Client")
    @patch("remediation_engine.runtime.telemetry.get_runtime_settings")
    def test_interrupted_trace_update_is_direct_and_bounded(
        self, mock_get_settings, mock_client_cls
    ):
        run_id = uuid4()
        mock_get_settings.return_value = AppSettings(
            langsmith_endpoint="https://apac.api.smith.langchain.com/",
            langsmith_api_key="test-key",
        )

        mark_phase5_trace_failed(run_id, KeyboardInterrupt())

        mock_client_cls.assert_called_once_with(
            api_url="https://apac.api.smith.langchain.com/",
            api_key="test-key",
            auto_batch_tracing=False,
            timeout_ms=(2_000, 5_000),
        )
        mock_client_cls.return_value.update_run.assert_called_once()
        update_kwargs = mock_client_cls.return_value.update_run.call_args.kwargs
        assert update_kwargs["error"] == "KeyboardInterrupt"
        assert update_kwargs["outputs"] == {
            "status": "completed_with_errors",
            "error": "KeyboardInterrupt",
        }
