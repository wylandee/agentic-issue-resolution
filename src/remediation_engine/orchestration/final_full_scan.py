"""Optional authoritative Dependency-Check pass before final reconciliation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from remediation_engine.contracts.schemas import FinalFullScanResult, ScannerExecutionStatus
from remediation_engine.orchestration.qa_odc import _run_security_scan
from remediation_engine.orchestration.qa_types import _SecurityScanResult
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)


def _state_value(value: Any, field: str, default: Any = None) -> Any:
    """Read one field from either a typed model or a state dictionary."""
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def run_final_full_scan_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run an optional full-workspace ODC scan on the modified Docker volume.

    The scan is performed only after the Supervisor routes a terminal run to
    this node. It never scans or modifies the host repository. If execution or
    report parsing fails, previously accumulated incremental identifiers are
    preserved and the full-scan result records the failure.
    """
    workspace_volume = state.get("workspace_volume")
    baseline = {
        identifier.strip().upper()
        for identifier in (state.get("baseline_scan_identifiers") or [])
        if isinstance(identifier, str) and identifier.strip()
    }
    if not workspace_volume:
        result = _SecurityScanResult(
            False,
            "FAILURE: final full scan skipped because the workspace volume is unavailable.",
            set(),
            set(),
            set(),
            execution_status=ScannerExecutionStatus.NOT_RUN,
        )
    else:
        try:
            with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
                result = _run_security_scan(
                    sandbox,
                    workspace_volume,
                    baseline,
                    baseline,
                )
        except Exception as exc:  # noqa: BLE001 - teardown must still run
            logger.exception("final_full_scan: Dependency-Check execution failed.")
            result = _SecurityScanResult(
                False,
                f"FAILURE: final full scan could not complete: {exc}",
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            )

    completed = result.execution_status == ScannerExecutionStatus.SUCCESS
    prior_result = state.get("final_full_scan_result")
    if completed:
        found_identifiers = sorted(result.found_identifiers)
        new_identifiers = sorted(result.new_identifiers)
        remaining_identifiers = sorted(result.remaining_identifiers)
        found_issues = list(result.found_issues)
    else:
        # Failed final scans must not erase identifiers already accumulated by
        # successful targeted scans.
        found_identifiers = list(state.get("post_remediation_scan_identifiers", []) or [])
        new_identifiers = list(state.get("new_vulnerability_identifiers", []) or [])
        remaining_identifiers = list(
            _state_value(prior_result, "remaining_target_identifiers", []) or []
        )
        found_issues = list(state.get("post_remediation_scan_issues", []) or [])

    status = (
        "scan_failed"
        if not completed
        else "detected"
        if new_identifiers
        else "unresolved"
        if remaining_identifiers
        else "none"
    )
    triage_required = bool(
        completed
        and state.get("post_qa_triage_enabled", True)
        and (new_identifiers or remaining_identifiers)
    )
    error = None if completed else result.summary
    final_result = FinalFullScanResult(
        completed=completed,
        authoritative=completed,
        coverage_scope="full_workspace",
        coverage_complete=completed,
        found_identifiers=found_identifiers,
        remaining_target_identifiers=remaining_identifiers,
        new_identifiers=new_identifiers,
        found_issues=found_issues,
        status=status,
        triage_required=triage_required,
        error=error,
    )
    if error:
        logger.warning("final_full_scan: %s", error)

    return {
        "status": "final_scan_completed" if completed else "final_scan_failed",
        "final_full_scan_result": final_result,
        "post_remediation_scan_identifiers": found_identifiers,
        "post_remediation_scan_issues": found_issues,
        "new_vulnerability_identifiers": new_identifiers,
        "new_vulnerability_status": status,
        "triage_required": triage_required,
        "errors": [f"final_full_scan: {error}"] if error else [],
    }
