from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    LocalizedIssue,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration import update_subagent as update_subagent_module
from remediation_engine.orchestration.state import initial_update_subagent_state
from remediation_engine.orchestration.subagent_runtime import ToolEvent
from remediation_engine.orchestration.update_subagent import (
    _attempted_versions_for_current_run,
    _changed_files_by_task,
    _effective_targets_for_current_run,
    _executed_versions_for_current_run,
    _resolve_manifest_targets,
    run_update_subagent_node,
)


def _group(ecosystem: str, manifest_path: str | None, *, issue_path: str | None = None):
    issue = SimpleNamespace(
        ecosystem=ecosystem,
        purl=f"pkg:{'pypi' if ecosystem == 'pypi' else 'npm'}/sample@1.0",
        file_path=issue_path,
    )
    localized = (
        [SimpleNamespace(issue=issue, manifest_file=manifest_path)]
        if manifest_path is not None
        else []
    )
    return SimpleNamespace(
        group_id=f"{ecosystem}-group",
        issues=[issue],
        localized_issues=localized,
        file_path=issue_path,
        file_paths=[issue_path] if issue_path else [],
        vulnerable_component="sample",
        parent_package_name=None,
    )


def _task(task_id: str, package_name: str, dependency_type: str):
    return SimpleNamespace(
        task_id=task_id,
        target_package_name=package_name,
        target_dependency_type=dependency_type,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        strategy=RoutingStrategy.VERSION_BUMP,
        parent_package_name=None,
        status=TaskStatus.PENDING,
        retry_count=0,
    )


def test_python_manifest_resolution_uses_localization_and_repository_containment(tmp_path) -> None:
    (tmp_path / "requirements-dev.txt").write_text("sample==1.0\n", encoding="utf-8")
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")

    localized_python_group = _group(
        "pypi",
        "requirements-dev.txt",
        issue_path="package.json",
    )
    paths, errors = _resolve_manifest_targets(localized_python_group, tmp_path)
    assert paths == ["requirements-dev.txt"]
    assert errors == []

    unlocalized_python_group = _group("pypi", None, issue_path="package.json")
    paths, errors = _resolve_manifest_targets(unlocalized_python_group, tmp_path)
    assert paths == []
    assert errors

    traversal_group = _group("pypi", "../requirements.txt")
    paths, errors = _resolve_manifest_targets(traversal_group, tmp_path)
    assert paths == []
    assert any("rejected manifest path" in error for error in errors)


def test_npm_manifest_resolution_remains_package_json_only(tmp_path) -> None:
    (tmp_path / "package.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("sample==1.0\n", encoding="utf-8")

    group = _group("npm", "package.json", issue_path="requirements.txt")
    paths, errors = _resolve_manifest_targets(group, tmp_path)

    assert paths == ["package.json"]
    assert errors


def test_python_attempt_and_effective_evidence_is_pep503_and_pep440_normalized() -> None:
    task = _task("task-python", "Zope_Interface", "packages")
    group = _group("pypi", "Pipfile")
    other_task = _task("task-other", "zope.interface", "dependencies")
    other_group = _group("pypi", "requirements.txt")
    resolved_tasks = [
        (task, group, ["Pipfile"]),
        (other_task, other_group, ["requirements.txt"]),
    ]
    events = [
        ToolEvent(
            "modify_and_validate_python_dependency",
            {
                "package_name": "zope.interface",
                "target_version": "v6.0.0",
                "dependency_type": "packages",
                "manifest_path": "Pipfile",
            },
            "ERROR_CODE: SYNC_FAILED: rolled back",
        ),
        ToolEvent(
            "modify_and_validate_python_dependency",
            {
                "package_name": "zope-interface",
                "target_version": "6.0",
                "dependency_type": "packages",
                "manifest_path": "Pipfile",
            },
            'SUCCESS: updated JSON: {"changed_files":["Pipfile","Pipfile.lock"]}',
        ),
    ]

    assert _attempted_versions_for_current_run(resolved_tasks, events) == {
        "task-python": ["6.0.0", "6.0"],
        "task-other": [],
    }
    assert _executed_versions_for_current_run(resolved_tasks, events) == {
        "task-python": ["6.0"],
        "task-other": [],
    }
    assert _effective_targets_for_current_run(resolved_tasks, events) == (
        {"task-python": "6.0", "task-other": None},
        {"task-python": "packages", "task-other": None},
    )
    assert _changed_files_by_task(
        resolved_tasks,
        ["Pipfile", "Pipfile.lock", "package.json"],
    ) == {
        "task-python": ["Pipfile", "Pipfile.lock"],
        "task-other": [],
    }


def test_npm_attempt_evidence_keeps_existing_version_and_file_partition() -> None:
    task = _task("task-node", "lodash", "dependencies")
    group = _group("npm", "package.json")
    resolved_tasks = [(task, group, ["package.json"])]
    event = ToolEvent(
        "modify_and_validate_npm_dependency",
        {
            "package_name": "lodash",
            "target_version": "v4.17.21",
            "dependency_type": "dependencies",
            "manifest_path": "package.json",
        },
        "SUCCESS: updated package",
    )

    assert _attempted_versions_for_current_run(resolved_tasks, [event]) == {
        "task-node": ["4.17.21"]
    }
    assert _changed_files_by_task(
        resolved_tasks,
        ["package.json", "Pipfile.lock"],
    ) == {"task-node": ["package.json"]}


def _dispatch_state(repo_root, language: ProjectLanguage, ecosystem: str, manifest: str):
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name="sample",
        package_version="1.0",
        purl=f"pkg:{'pypi' if ecosystem == 'pypi' else 'npm'}/sample@1.0",
        ecosystem=ecosystem,
    )
    dependency_type = "requirements" if ecosystem == "pypi" else "dependencies"
    localized_issue = LocalizedIssue(
        issue=issue,
        manifest_file=manifest,
        declaration_type=dependency_type,
    )
    group = VulnerabilityGroup(
        group_id=f"sca:{manifest}:sample",
        issue_type=IssueType.SCA,
        vulnerable_component="sample",
        file_path=manifest,
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized_issue],
    )
    task = RemediationTask(
        task_id=f"task-{ecosystem}",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        target_package_name="sample",
        target_dependency_type=dependency_type,
        selected_version="2.0.0",
        instruction=f"Update sample in {manifest}.",
    )
    return initial_update_subagent_state(
        str(repo_root),
        "test-workspace",
        [task],
        [group],
        project_language=language,
    )


def _stub_update_dispatch(monkeypatch, *, belt, runtime):
    sandbox_context = MagicMock()
    sandbox_context.__enter__.return_value = SimpleNamespace()
    sandbox_context.__exit__.return_value = False
    worker_loop = MagicMock(return_value=runtime)
    monkeypatch.setattr(
        update_subagent_module,
        "ChatOpenAI",
        lambda **_kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        update_subagent_module,
        "DockerSandbox",
        lambda **_kwargs: sandbox_context,
    )
    monkeypatch.setattr(
        update_subagent_module,
        "get_runtime_settings",
        lambda: SimpleNamespace(update_llm_model="test-model"),
    )
    monkeypatch.setattr(update_subagent_module, "build_repository_map", lambda _root: "map")
    monkeypatch.setattr(
        update_subagent_module, "_build_update_prompt", lambda *_args, **_kwargs: "task"
    )
    monkeypatch.setattr(
        update_subagent_module,
        "_create_skinny_subagent_group",
        lambda group: group,
    )
    monkeypatch.setattr(
        update_subagent_module,
        "_filter_constraints_ledger",
        lambda *_args: [],
    )
    monkeypatch.setattr(update_subagent_module, "build_update_toolbelt", belt)
    monkeypatch.setattr(update_subagent_module, "run_bounded_subagent_loop", worker_loop)
    monkeypatch.setattr(
        update_subagent_module,
        "rollback_pending_package_updates",
        lambda *_args: [],
    )
    return worker_loop


@pytest.mark.parametrize(
    ("language", "ecosystem", "manifest"),
    [
        (ProjectLanguage.PYTHON, "pypi", "requirements.txt"),
        (ProjectLanguage.NODEJS, "npm", "package.json"),
    ],
)
def test_update_dispatch_routes_language_and_canonical_ecosystem(
    monkeypatch,
    tmp_path,
    language: ProjectLanguage,
    ecosystem: str,
    manifest: str,
) -> None:
    (tmp_path / manifest).write_text("", encoding="utf-8")
    state = _dispatch_state(tmp_path, language, ecosystem, manifest)
    runtime = SimpleNamespace(final_text="done", tool_events=[], changed_files=[], errors=[])
    belt = MagicMock(return_value=[])
    worker_loop = _stub_update_dispatch(monkeypatch, belt=belt, runtime=runtime)

    run_update_subagent_node(state)

    assert belt.call_args.kwargs["language"] == language
    assert belt.call_args.kwargs["package_ecosystem"] == ecosystem
    assert belt.call_args.kwargs["target_manifest_paths"] == [manifest]
    worker_loop.assert_called_once()


def test_mismatched_update_dispatch_fails_before_worker_loop(monkeypatch, tmp_path) -> None:
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    state = _dispatch_state(
        tmp_path,
        ProjectLanguage.NODEJS,
        "pypi",
        "requirements.txt",
    )
    runtime = SimpleNamespace(final_text="done", tool_events=[], changed_files=[], errors=[])
    belt = MagicMock(
        side_effect=ValueError("Unsupported project-language/package-ecosystem pairing.")
    )
    worker_loop = _stub_update_dispatch(monkeypatch, belt=belt, runtime=runtime)

    result = run_update_subagent_node(state)

    assert belt.call_args.kwargs["language"] == ProjectLanguage.NODEJS
    assert belt.call_args.kwargs["package_ecosystem"] == "pypi"
    worker_loop.assert_not_called()
    assert any(
        "Unsupported project-language/package-ecosystem pairing." in error
        for error in result["errors"]
    )
