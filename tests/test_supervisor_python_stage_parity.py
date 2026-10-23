"""Python retry planning must retain the shared ecosystem-specific stages."""

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    SupervisorRetryPlan,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.supervisor_planner import (
    _planner_plan_violations,
    _repair_invalid_planner_plans,
)


def _python_group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="certifi",
        package_version="1.0.0",
        purl="pkg:pypi/certifi@1.0.0",
        ecosystem="pypi",
    )
    return VulnerabilityGroup(
        group_id="sca:Pipfile:certifi",
        issue_type=IssueType.SCA,
        vulnerable_component="certifi",
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _python_task(group: VulnerabilityGroup) -> RemediationTask:
    return RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        status=TaskStatus.NEEDS_RETRY,
    )


def test_python_pivot_must_use_pypi_latest_stage() -> None:
    group = _python_group()
    task = _python_task(group)
    plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        action="pivot_workaround",
        exact_instruction="Implement a code workaround.",
    )

    violations = _planner_plan_violations(
        {task.task_id: plan},
        {task.task_id: task},
        {},
        {group.group_id: group},
    )

    assert any("not valid for pypi" in item for item in violations)
    assert any("must be committed at pypi_latest" in item for item in violations)


def test_python_exhaustion_repair_pivots_at_pypi_latest() -> None:
    group = _python_group()
    task = _python_task(group)
    plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        selected_version="1.2.0",
        latest_version_seen="1.2.0",
        attempted_versions=["1.2.0"],
        action="retry_update",
        exact_instruction="Update certifi to 1.2.0.",
    )

    diagnostics, repaired = _repair_invalid_planner_plans(
        {task.task_id: plan},
        {},
        {task.task_id: task},
        {group.group_id: group},
        violations=[f"task {task.task_id}: selected version 1.2.0 was already attempted"],
    )

    assert repaired[task.task_id].action == "pivot_workaround"
    assert repaired[task.task_id].strategy_stage == SCARemediationStage.PYPI_LATEST
    assert repaired[task.task_id].selected_version is None
    assert diagnostics[task.task_id].strategy_stage == SCARemediationStage.PYPI_LATEST
