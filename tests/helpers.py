"""Shared factories for focused engine tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from unittest.mock import MagicMock

from remediation_engine.agents.supervisor_reasoning import apply_issue_fix_plans_to_group
from remediation_engine.agents.triage import group_issues
from remediation_engine.contracts.schemas import (
    CommandResult,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    PackageFixPlanCandidate,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionCertificate,
    PackageResolutionStatus,
    SolverRuntimeFingerprint,
)
from remediation_engine.orchestration.state import select_package_fix_plan
from remediation_engine.planning.portfolio import (
    apply_portfolio_plan,
    build_portfolio_plan,
    prepare_portfolio_inputs,
)
from remediation_engine.tools.qa_scan import _QAExecutionResults, _QALogRecord, _SecurityScanResult
from remediation_engine.tools.test_execution import _QAInstallOutcome, _QATestExecutionOutcome

_MISSING = object()


def qa_policy_for_task(strategy, no_fix_stage=None) -> QAPolicy:
    """Return the explicit QA policy expected for a test task fixture."""
    stage = getattr(no_fix_stage, "value", no_fix_stage)
    if stage == "package_removal":
        return QAPolicy.NO_FIX_PACKAGE_REMOVAL
    if stage == "vulnerable_code_removal":
        return QAPolicy.NO_FIX_CODE_REMOVAL
    strategy_value = getattr(strategy, "value", strategy)
    if strategy_value == RoutingStrategy.VERSION_BUMP.value:
        return QAPolicy.VERSION_BUMP
    return QAPolicy.INITIAL_CODE_WORKAROUND


def selected_group_plan(group: VulnerabilityGroup) -> FixPlan | None:
    """Return the plan selected from a group's typed issue candidates."""
    return select_package_fix_plan(group).plan


def attach_test_resolution_certificate(plan):
    """Add internally consistent resolver evidence to an offline test plan.

    Args:
        plan: A solver plan built from an isolated test repository.

    Returns:
        A copy of the plan with a deterministic test certificate attached.

    Raises:
        AssertionError: If the solver did not select a fully covered assignment.

    This fixture does not run a package manager. It marks the candidate catalog
    complete so Supervisor unit tests can exercise committed-plan behavior
    without Docker, registry access, or network services.
    """
    solver_plan = plan.solver_plan.model_copy(update={"candidate_catalog_complete": True})
    selected = solver_plan.selected_plan
    assert selected is not None
    assignment_digest = hashlib.sha256(
        json.dumps(
            dict(sorted(selected.selected_candidate_versions.items())),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    covered = sorted(
        coverage_id for batch in selected.batches for coverage_id in batch.resolved_coverage_ids
    )
    workaround = sorted(
        coverage_id for batch in selected.batches for coverage_id in batch.workaround_coverage_ids
    )
    assert not any(batch.unresolved_coverage_ids for batch in selected.batches)
    certificate = PackageResolutionCertificate(
        status=PackageResolutionStatus.CERTIFIED,
        portfolio_plan_id=plan.portfolio_plan_id,
        candidate_plan_id=selected.candidate_plan_id,
        solver_input_digest=plan.solver_input_digest,
        repository_fingerprint=plan.repository_fingerprint,
        task_revisions=plan.task_revisions,
        workspace_graph_digest=plan.workspace_graph_digest,
        candidate_catalog_digest=solver_plan.candidate_catalog_digest,
        candidate_assignment_digest=assignment_digest,
        resolved_graph_digest="test-resolved-graph-digest",
        runtime_fingerprint=SolverRuntimeFingerprint(
            node_version="v22.0.0",
            npm_version="10.0.0",
            platform="linux",
            architecture="x64",
        ),
        covered_coverage_ids=covered,
        workaround_coverage_ids=workaround,
    )
    return plan.model_copy(
        update={"solver_plan": solver_plan, "resolution_certificate": certificate}
    )


def commit_test_portfolio_plan(repo_root, groups, task_queue):
    """Prepare and commit an offline solver plan for Supervisor tests.

    Args:
        repo_root: Isolated repository used by the test.
        groups: Vulnerability groups represented in the test state.
        task_queue: Pending tasks to include in the plan.

    Returns:
        The certified test plan, committed groups, and committed task queue.

    Raises:
        AssertionError: If preparation or plan application reports diagnostics.

    All planning inputs are local. Registry candidates and resolver behavior
    are not queried by this test fixture.
    """
    prepared_groups, prepared_queue, prepare_diagnostics = prepare_portfolio_inputs(
        repo_root,
        groups,
        task_queue,
    )
    assert prepare_diagnostics == []
    plan = build_portfolio_plan(repo_root, prepared_groups, prepared_queue)
    plan = attach_test_resolution_certificate(plan)
    committed_groups, committed_queue, apply_diagnostics = apply_portfolio_plan(
        plan,
        prepared_groups,
        prepared_queue,
    )
    assert apply_diagnostics == []
    return plan, committed_groups, committed_queue


def _make_group(
    group_id: str = "sca:package.json:lodash",
    cve_ids=None,
    ghsa_ids=None,
    fix_plan_status: FixPlanStatus = FixPlanStatus.VERSION_FOUND,
) -> VulnerabilityGroup:
    cve_ids = cve_ids or ["CVE-2021-23337"]
    ghsa_ids = ghsa_ids or ["GHSA-35JH-R3H4-6JV8"]
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name="lodash",
        cve_id=cve_ids[0] if cve_ids else None,
        ghsa_id=ghsa_ids[0] if ghsa_ids else None,
        severity=Severity.HIGH,
    )
    fix_plan = FixPlan(
        status=fix_plan_status,
        fixed_version="4.17.21" if fix_plan_status == FixPlanStatus.VERSION_FOUND else None,
        workaround_snippets=(
            ["// workaround: sanitize input"]
            if fix_plan_status == FixPlanStatus.WORKAROUND_FOUND
            else None
        ),
        instruction="Upgrade lodash to 4.17.21.",
        strategy_used="osv_api",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_paths=["package.json"],
        cve_ids=cve_ids,
        ghsa_ids=ghsa_ids,
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan_candidates=[PackageFixPlanCandidate(issue_id=issue.id, plan=fix_plan)],
    )


def _make_sandbox(
    run_side_effects=None,
    read_file_return=None,
    workspace_volume="sandbox-vol",
):
    """Return a pre-configured mock ``DockerSandbox``."""
    sandbox = MagicMock()
    sandbox._workspace_volume = workspace_volume
    if run_side_effects is not None:
        sandbox.run.side_effect = run_side_effects
    else:
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="ok", stderr="", duration_seconds=0.5
        )
    sandbox.read_file.return_value = read_file_return
    return sandbox


def _install_outcome(
    ok: bool = True,
    summary: str = "ok",
    *,
    exit_code: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    error_category: str | None = None,
) -> _QAInstallOutcome:
    """Build a typed install outcome for global-execution tests."""
    return _QAInstallOutcome(
        ok=ok,
        summary=summary,
        exit_code=exit_code,
        error_category=error_category,
        raw_stdout=stdout,
        raw_stderr=stderr,
        log_record=_QALogRecord(
            phase="install",
            label="npm install",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
        ),
    )


def _test_outcome(
    ok: bool = True,
    summary: str = "ok",
    *,
    exit_code: int | None = 0,
    failure_count: int | None = 0,
    stdout: str = "",
    stderr: str = "",
) -> _QATestExecutionOutcome:
    """Build a typed test outcome for global-execution tests."""
    return _QATestExecutionOutcome(
        ok=ok,
        summary=summary,
        exit_code=exit_code,
        failure_count=failure_count,
        raw_stdout=stdout,
        raw_stderr=stderr,
        log_records=(
            _QALogRecord(
                phase="tests",
                label="npm test",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            ),
        ),
    )


def _scan_outcome(
    ok: bool = True,
    summary: str = "ok",
    *,
    remaining: set[str] | None = None,
) -> _SecurityScanResult:
    """Build a typed scanner outcome for global-execution tests."""
    return _SecurityScanResult(
        ok=ok,
        summary=summary,
        remaining_identifiers=remaining or set(),
        found_identifiers=set(),
        new_identifiers=set(),
    )


def _make_minimal_state(
    groups=_MISSING,
    workspace_volume="test-vol",
    repo_root="/tmp/repo",
    group_strategies=None,
    changed_files=None,
):
    resolved_groups = [_make_group()] if groups is _MISSING else groups
    task_queue = {}
    attempt_snapshots_by_id = {}
    for index, group in enumerate(resolved_groups, start=1):
        task_id = f"task-{index}"
        attempt_id = f"{task_id}-attempt"
        instruction = f"Remediate {group.group_id}."
        task = RemediationTask(
            task_id=task_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            qa_policy=QAPolicy.VERSION_BUMP,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
            current_attempt_id=attempt_id,
            target_package_name=group.vulnerable_component,
            selected_version=(
                selected_group_plan(group).fixed_version if selected_group_plan(group) else None
            ),
            instruction=instruction,
        )
        task_queue[task_id] = task
        attempt_snapshots_by_id[attempt_id] = TaskAttemptSnapshot(
            attempt_id=attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=instruction,
            instruction_digest=f"digest-{task_id}",
            dispatch_node="qa_critic",
            allowed_target_versions=[task.selected_version]
            if task.selected_version is not None
            else [],
            allowed_dependency_types=[],
        )
    worker_results_by_attempt = {
        task.current_attempt_id: {"changed_files": list(changed_files or [])}
        for task in task_queue.values()
        if task.current_attempt_id
    }
    return {
        "valid_groups": resolved_groups,
        "workspace_volume": workspace_volume,
        "repo_root": repo_root,
        "action_summaries": [],
        "group_strategies": group_strategies or {},
        "changed_files": changed_files or [],
        "task_queue": task_queue,
        "active_target_task_ids": list(task_queue),
        "attempt_snapshots_by_id": attempt_snapshots_by_id,
        "worker_results_by_attempt": worker_results_by_attempt,
    }


def _make_fully_populated_results(ok=True):
    """Build a _QAExecutionResults with all three phases filled in."""
    r = _QAExecutionResults()
    r.install = (ok, "npm install succeeded." if ok else "npm install FAILED")
    r.scan = _scan_outcome(ok=ok, summary="Dependency-Check OK." if ok else "scan FAILED")
    r.tests = (ok, "npm test passed." if ok else "npm test FAILED")
    return r


def group_with_fix_plans(
    issues: list[VulnerabilityIssue],
    *,
    plans_by_finding: Iterable[tuple[LocalizedIssue, FixPlan]],
):
    """Build plan-bearing groups without invoking advisory or registry services."""
    pairs = list(plans_by_finding)
    groups = group_issues(
        issues,
        localized_sca_issues=[localized for localized, _ in pairs],
    )
    plans_by_issue = {localized.issue.id: (localized, plan) for localized, plan in pairs}
    result = []
    for group in groups:
        group_pairs = [
            plans_by_issue[issue.id] for issue in group.issues if issue.id in plans_by_issue
        ]
        result.append(apply_issue_fix_plans_to_group(group, group_pairs) if group_pairs else group)
    return result
