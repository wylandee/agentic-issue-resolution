"""Shared fixtures for focused QA critic tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import (
    CommandResult,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.qa_test_parsing import (
    _QAInstallOutcome,
    _QATestExecutionOutcome,
)
from remediation_engine.orchestration.qa_types import (
    _QAExecutionResults,
    _QALogRecord,
    _SecurityScanResult,
)

_MISSING = object()


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
        fix_plan=fix_plan,
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


def _make_workspace_tmpdir() -> Path:
    """Create a writable temp directory inside the workspace."""
    base_dir = Path("data/cache")
    base_dir.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base_dir))


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
            selected_version=group.fix_plan.fixed_version if group.fix_plan else None,
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


def _make_loop_result(
    final_text="",
    tool_events=None,
    errors=None,
    structured_output=None,
):
    """Build a mocked SubagentRuntimeResult."""
    from remediation_engine.orchestration.subagent_runtime import SubagentRuntimeResult

    return SubagentRuntimeResult(
        final_text=final_text,
        tool_events=tool_events or [],
        changed_files=[],
        errors=errors or [],
        structured_output=structured_output,
    )


def _make_fully_populated_results(ok=True):
    """Build a _QAExecutionResults with all three phases filled in."""
    r = _QAExecutionResults()
    r.install = (ok, "npm install succeeded." if ok else "npm install FAILED")
    r.scan = _scan_outcome(ok=ok, summary="Dependency-Check OK." if ok else "scan FAILED")
    r.tests = (ok, "npm test passed." if ok else "npm test FAILED")
    return r
