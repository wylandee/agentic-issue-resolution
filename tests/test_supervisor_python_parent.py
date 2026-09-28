from __future__ import annotations

from unittest.mock import patch

import pytest
from packaging.version import Version

from remediation_engine.contracts import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.supervisor_node import _plan_initial_transitive_task
from remediation_engine.triage.grouper import group_issues


def _group(*, ecosystem: str = "pypi", direct: bool | None = False) -> VulnerabilityGroup:
    package = "vulnerable-child"
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=package,
        package_version="1.0.0",
        purl=f"pkg:{ecosystem}/{package}@1.0.0",
        cve_id="CVE-2026-1000",
        ecosystem=ecosystem,
        file_path="Pipfile.lock" if ecosystem == "pypi" else "package-lock.json",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="Pipfile" if ecosystem == "pypi" else "package.json",
        package_manager="pipenv" if ecosystem == "pypi" else "npm",
        is_direct_dependency=direct,
        declaration_type="packages" if ecosystem == "pypi" else "dependencies",
        dependency_ancestry=[] if ecosystem == "pypi" else ["direct-parent", package],
        dependency_versions={"direct-parent": "1.0.0"},
        parent_package_name=None if ecosystem == "pypi" else "direct-parent",
        parent_package_version=None if ecosystem == "pypi" else "1.0.0",
        parent_declaration_type=None if ecosystem == "pypi" else "dependencies",
        localization_confidence=1.0,
    )
    fix_plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="1.2.0",
        instruction="Install the fixed release.",
        strategy_used="osv_api",
    )
    return group_issues([issue], sca_issue_plans=[(localized, fix_plan)])[0]


def _task(group: VulnerabilityGroup, *, ecosystem: str = "pypi") -> RemediationTask:
    return RemediationTask(
        task_id=f"{ecosystem}-task",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        target_package_name=None,
        target_dependency_type=None,
        parent_package_name=None,
        parent_package_version=None,
        instruction="Await supervisor planning.",
        status=TaskStatus.PENDING,
    )


def _report(version: str, other_candidate: str) -> str:
    pypi_latest = "3.1.0.post1"
    return (
        f"- Selected Version: {version}\n"
        f"- Eligible Candidates: {version}, {other_candidate}, 99.0.0\n"
        f"- Compatible Parent Versions: {version}, {other_candidate}\n"
        "- Latest Stable: 99.0.0\n"
        f"- PyPI Latest: {pypi_latest}"
    )


@pytest.mark.parametrize("direct", [True, None], ids=["direct", "non-transitive"])
def test_direct_or_non_transitive_pypi_task_is_not_replanned_as_a_parent_update(direct):
    group = _group(direct=direct)
    task = _task(group)

    with (
        patch(
            "remediation_engine.orchestration.supervisor_spawn._resolve_pipfile_parent_context"
        ) as proof,
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_python_parent_version"
        ) as planner,
    ):
        planned = _plan_initial_transitive_task(task, group, workspace_volume="workspace")

    assert planned == task
    proof.assert_not_called()
    planner.invoke.assert_not_called()


@pytest.mark.parametrize(
    ("selection", "stage", "version"),
    [
        ("minimum", SCARemediationStage.OSV_MINIMUM, "1.2.0.post1"),
        ("same_major", SCARemediationStage.PYPI_SAME_MAJOR, "1.8.0.post1"),
        ("latest", SCARemediationStage.PYPI_LATEST, "3.1.0.post1"),
    ],
)
def test_proven_one_hop_pipfile_parent_commits_selected_parent_only(selection, stage, version):
    group = _group()
    task = _task(group)
    candidates: list[str] = ["stale-version"]

    def plan_parent(inputs: dict[str, str]) -> str:
        if inputs["selection"] == selection:
            return _report(
                version,
                {"minimum": "2.0.0", "same_major": "1.9.0", "latest": "3.0.0"}[selection],
            )
        return "- Status: no eligible candidate"

    with (
        patch(
            "remediation_engine.orchestration.supervisor_spawn._resolve_pipfile_parent_context",
            return_value=("direct-parent", "1.0.0", ["direct-parent", "vulnerable-child"]),
        ),
        patch(
            "remediation_engine.orchestration.supervisor_spawn._pipfile_parent_declaration_type",
            return_value="dev-packages",
        ) as category,
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_python_parent_version"
        ) as planner,
    ):
        planner.invoke.side_effect = plan_parent
        planned = _plan_initial_transitive_task(
            task,
            group,
            candidate_versions=candidates,
            workspace_volume="workspace",
        )

    assert planned.strategy == RoutingStrategy.VERSION_BUMP
    assert planned.strategy_stage == stage
    assert planned.target_package_name == "direct-parent"
    assert planned.target_dependency_type == "dev-packages"
    assert planned.parent_package_name == "direct-parent"
    assert planned.parent_package_version == "1.0.0"
    assert planned.selected_version == version
    assert planned.target_package_name != group.vulnerable_component
    assert "modify_and_validate_python_dependency" in planned.instruction
    assert "Do not edit the vulnerable child declaration or add an override." in planned.instruction
    assert candidates == sorted(
        [
            version,
            {"minimum": "2.0.0", "same_major": "1.9.0", "latest": "3.0.0"}[selection],
        ],
        key=Version,
    )
    assert "99.0.0" not in candidates
    category.assert_called_once_with("workspace", "direct-parent")
    calls = planner.invoke.call_args_list
    assert [call.args[0]["selection"] for call in calls] == [
        "minimum",
        "same_major",
        "latest",
    ][: ["minimum", "same_major", "latest"].index(selection) + 1]
    chosen = calls[-1].args[0]
    assert chosen["parent_package_name"] == "direct-parent"
    assert chosen["child_package_name"] == "vulnerable-child"
    assert chosen["child_fixed_version"] == "1.2.0"
    assert chosen["installed_parent_version"] == "1.0.0"
    assert chosen["dependency_ancestry"] == "direct-parent,vulnerable-child"


@pytest.mark.parametrize("proof_failure", ["missing", "ambiguous"])
def test_missing_or_ambiguous_pipfile_parent_fails_closed(proof_failure):
    group = _group()
    task = _task(group)
    proof_options = (
        {"return_value": None}
        if proof_failure == "missing"
        else {"side_effect": ValueError("ambiguous direct parents")}
    )

    with (
        patch(
            "remediation_engine.orchestration.supervisor_spawn._resolve_pipfile_parent_context",
            **proof_options,
        ) as proof,
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_python_parent_version"
        ) as planner,
    ):
        planned = _plan_initial_transitive_task(task, group, workspace_volume="workspace")

    assert planned.strategy == RoutingStrategy.CODE_WORKAROUND
    assert planned.strategy_stage == SCARemediationStage.CODE_WORKAROUND
    assert planned.target_package_name is None
    assert planned.target_dependency_type is None
    assert planned.parent_package_name is None
    assert planned.selected_version is None
    assert "Do not pin the vulnerable child" in planned.instruction
    proof.assert_called_once_with("workspace", group)
    planner.invoke.assert_not_called()


def test_missing_workspace_volume_fails_closed_without_parent_lookup():
    group = _group()
    task = _task(group)
    candidates: list[str] = ["stale-version"]

    with (
        patch(
            "remediation_engine.orchestration.supervisor_spawn._resolve_pipfile_parent_context"
        ) as proof,
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_python_parent_version"
        ) as planner,
    ):
        planned = _plan_initial_transitive_task(task, group, candidate_versions=candidates)

    assert planned.strategy == RoutingStrategy.CODE_WORKAROUND
    assert planned.strategy_stage == SCARemediationStage.CODE_WORKAROUND
    assert planned.target_package_name is None
    assert planned.target_dependency_type is None
    assert planned.selected_version is None
    assert candidates == []
    proof.assert_not_called()
    planner.invoke.assert_not_called()


def test_npm_parent_planning_keeps_existing_override_progression_contract():
    group = _group(ecosystem="npm")
    task = _task(group, ecosystem="npm").model_copy(
        update={"parent_package_name": "direct-parent", "parent_package_version": "1.0.0"}
    )
    candidates: list[str] = []

    with (
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_npm_parent_version"
        ) as npm_planner,
        patch(
            "remediation_engine.orchestration.supervisor_spawn._resolve_pipfile_parent_context"
        ) as proof,
        patch(
            "remediation_engine.orchestration.supervisor_spawn.plan_python_parent_version"
        ) as python_planner,
    ):
        npm_planner.invoke.return_value = "- Selected Version: 1.0.1\n- Eligible Candidates: 1.0.1"
        planned = _plan_initial_transitive_task(task, group, candidate_versions=candidates)

    assert planned.strategy_stage == SCARemediationStage.OSV_MINIMUM
    assert planned.target_package_name == "direct-parent"
    assert planned.target_dependency_type == "dependencies"
    assert planned.selected_version == "1.0.1"
    assert candidates == ["1.0.1"]
    npm_planner.invoke.assert_called_once()
    proof.assert_not_called()
    python_planner.invoke.assert_not_called()
