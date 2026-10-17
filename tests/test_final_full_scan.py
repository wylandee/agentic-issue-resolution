"""Tests for the optional final full-workspace scan node."""

from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import ScannerExecutionStatus
from remediation_engine.orchestration.final_full_scan import run_final_full_scan_node
from remediation_engine.orchestration.qa_types import _SecurityScanResult


def test_final_full_scan_projects_authoritative_findings():
    workspace_sandbox = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = workspace_sandbox
    result = _SecurityScanResult(
        ok=False,
        summary="target remains and a new identifier was found",
        remaining_identifiers={"CVE-BASE"},
        found_identifiers={"CVE-BASE", "CVE-NEW"},
        new_identifiers={"CVE-NEW"},
        found_issues=[],
        execution_status=ScannerExecutionStatus.SUCCESS,
    )
    with (
        patch(
            "remediation_engine.orchestration.final_full_scan.DockerSandbox",
            return_value=context,
        ),
        patch(
            "remediation_engine.orchestration.final_full_scan._run_security_scan",
            return_value=result,
        ) as run_scan,
    ):
        output = run_final_full_scan_node(
            {
                "workspace_volume": "workspace-volume",
                "baseline_scan_identifiers": ["cve-base"],
                "post_qa_triage_enabled": True,
            }
        )

    run_scan.assert_called_once_with(
        workspace_sandbox,
        "workspace-volume",
        {"CVE-BASE"},
        {"CVE-BASE"},
    )
    final_result = output["final_full_scan_result"]
    assert final_result.completed is True
    assert final_result.authoritative is True
    assert final_result.coverage_scope == "full_workspace"
    assert final_result.coverage_complete is True
    assert final_result.found_identifiers == ["CVE-BASE", "CVE-NEW"]
    assert final_result.remaining_target_identifiers == ["CVE-BASE"]
    assert final_result.new_identifiers == ["CVE-NEW"]
    assert final_result.status == "detected"
    assert final_result.triage_required is True
    assert output["new_vulnerability_identifiers"] == ["CVE-NEW"]


def test_failed_final_full_scan_preserves_incremental_identifiers():
    workspace_sandbox = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = workspace_sandbox
    result = _SecurityScanResult(
        ok=False,
        summary="Dependency-Check timed out",
        remaining_identifiers=set(),
        found_identifiers=set(),
        new_identifiers=set(),
        execution_status=ScannerExecutionStatus.TIMEOUT,
    )
    with (
        patch(
            "remediation_engine.orchestration.final_full_scan.DockerSandbox",
            return_value=context,
        ),
        patch(
            "remediation_engine.orchestration.final_full_scan._run_security_scan",
            return_value=result,
        ),
    ):
        output = run_final_full_scan_node(
            {
                "workspace_volume": "workspace-volume",
                "baseline_scan_identifiers": ["CVE-BASE"],
                "post_remediation_scan_identifiers": ["CVE-BASE"],
                "new_vulnerability_identifiers": ["CVE-KNOWN-NEW"],
                "new_vulnerability_status": "detected",
            }
        )

    final_result = output["final_full_scan_result"]
    assert final_result.completed is False
    assert final_result.authoritative is False
    assert final_result.coverage_complete is False
    assert final_result.status == "scan_failed"
    assert final_result.new_identifiers == ["CVE-KNOWN-NEW"]
    assert output["new_vulnerability_identifiers"] == ["CVE-KNOWN-NEW"]
    assert output["new_vulnerability_status"] == "scan_failed"
    assert output["errors"] == ["final_full_scan: Dependency-Check timed out"]
