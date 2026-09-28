"""Ecosystem-specific Supervisor strategy progression and fixed-version policy."""

from __future__ import annotations

from uuid import uuid4

from packaging.version import Version

from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.supervisor_policy import (
    _canonical_security_floor,
    _is_exhausted_update_pivot_candidate,
    _next_sca_stage,
    _selection_for_stage,
)


def _group(
    *,
    ecosystem: str,
    issue_fixed_version: str | None,
    plan_fixed_version: str | None,
) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        ecosystem=ecosystem,
        purl="pkg:pypi/example-package@1.0.0" if ecosystem == "pypi" else None,
        package_name="example-package",
        fixed_version=issue_fixed_version,
    )
    plan = (
        FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version=plan_fixed_version,
            instruction="Upgrade to the fixed version.",
            strategy_used="osv_api",
        )
        if plan_fixed_version is not None
        else None
    )
    return VulnerabilityGroup(
        group_id="test-group",
        issue_type=IssueType.SCA,
        representative_issue_id=issue.id or uuid4(),
        issues=[issue],
        fix_plan=plan,
    )


def test_pypi_stage_chain_never_enters_npm_override():
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM, ecosystem="pypi") == (
        SCARemediationStage.PYPI_SAME_MAJOR
    )
    assert _next_sca_stage(SCARemediationStage.PYPI_SAME_MAJOR, ecosystem="pypi") == (
        SCARemediationStage.PYPI_LATEST
    )
    assert _next_sca_stage(SCARemediationStage.PYPI_LATEST, transitive=True, ecosystem="pypi") == (
        SCARemediationStage.CODE_WORKAROUND
    )
    assert _next_sca_stage(SCARemediationStage.PACKAGE_OVERRIDE, ecosystem="pypi") == (
        SCARemediationStage.CODE_WORKAROUND
    )
    assert _selection_for_stage(SCARemediationStage.PYPI_SAME_MAJOR) == "same_major"
    assert _selection_for_stage(SCARemediationStage.PYPI_LATEST) == "latest"


def test_default_npm_stage_progression_is_unchanged():
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM) == SCARemediationStage.NPM_SAME_MAJOR
    assert _next_sca_stage(SCARemediationStage.NPM_SAME_MAJOR) == SCARemediationStage.NPM_LATEST
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST, transitive=True) == (
        SCARemediationStage.PACKAGE_OVERRIDE
    )
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST) == SCARemediationStage.CODE_WORKAROUND


def test_exhausted_pypi_latest_parent_can_pivot_but_npm_parent_cannot():
    pypi_task = RemediationTask(
        task_id="pypi-task",
        parent_group_id="group",
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        status=TaskStatus.NEEDS_RETRY,
        parent_package_name="direct-parent",
    )
    npm_task = pypi_task.model_copy(
        update={
            "task_id": "npm-task",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id="pypi-task",
        exhausted_update_path=True,
    )

    assert _is_exhausted_update_pivot_candidate(pypi_task, diagnostics) is True
    assert (
        _is_exhausted_update_pivot_candidate(
            npm_task,
            diagnostics.model_copy(update={"task_id": "npm-task"}),
        )
        is False
    )


def test_canonical_security_floor_uses_stable_pep440_for_pypi():
    group = _group(
        ecosystem="pypi",
        issue_fixed_version="1.4",
        plan_fixed_version="1.4.0",
    )
    assert _canonical_security_floor(group) == ("1.4.0", None)

    group.fix_plan = group.fix_plan.model_copy(update={"fixed_version": "1.4.0rc1"})
    floor, error = _canonical_security_floor(group)
    assert floor is None
    assert error is not None and "PEP 440" in error


def test_pypi_member_floor_conflicts_compare_with_pep440_ordering():
    group = _group(
        ecosystem="python",
        issue_fixed_version="1.10.0",
        plan_fixed_version="1.9.0",
    )
    floor, error = _canonical_security_floor(group)
    assert floor is None
    assert error is not None and "exceeds" in error
    assert Version("1.10.0") > Version("1.9.0")


def test_npm_floor_remains_strict_semver():
    assert _canonical_security_floor(
        _group(ecosystem="npm", issue_fixed_version="1.4.0", plan_fixed_version="1.4.0")
    ) == ("1.4.0", None)
    floor, error = _canonical_security_floor(
        _group(ecosystem="npm", issue_fixed_version=None, plan_fixed_version="1.4")
    )
    assert floor is None
    assert error is not None and "semver" in error
