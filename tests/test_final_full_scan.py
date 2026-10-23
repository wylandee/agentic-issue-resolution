"""Tests for the optional final full-workspace scan node."""

from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import ScannerExecutionStatus
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
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


def test_python_final_full_scan_uses_python_runtime_and_typed_coverage():
    workspace_sandbox = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = workspace_sandbox
    result = _SecurityScanResult(
        ok=True,
        summary="Dependency-Check found no remaining target vulnerability identifiers.",
        remaining_identifiers=set(),
        found_identifiers=set(),
        new_identifiers=set(),
        execution_status=ScannerExecutionStatus.SUCCESS,
    )
    state = {
        "workspace_volume": "python-workspace",
        "baseline_scan_identifiers": [],
        "project_language": ProjectLanguage.PYTHON,
    }

    with (
        patch(
            "remediation_engine.orchestration.final_full_scan.DockerSandbox",
            return_value=context,
        ) as sandbox_factory,
        patch(
            "remediation_engine.orchestration.final_full_scan._run_security_scan",
            return_value=result,
        ) as run_scan,
    ):
        output = run_final_full_scan_node(state)

    sandbox_factory.assert_called_once_with(
        repo_root=None,
        workspace_volume="python-workspace",
        image=LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image,
    )
    run_scan.assert_called_once_with(
        workspace_sandbox,
        "python-workspace",
        set(),
        set(),
        project_language=ProjectLanguage.PYTHON,
    )
    context.__exit__.assert_called_once()
    final_result = output["final_full_scan_result"]
    assert output["status"] == "final_scan_completed"
    assert final_result.completed is True
    assert final_result.authoritative is True
    assert final_result.coverage_scope == "full_workspace"
    assert final_result.coverage_complete is True
    assert final_result.status == "none"


def test_final_full_scan_docker_error_returns_typed_failure():
    with patch(
        "remediation_engine.orchestration.final_full_scan.DockerSandbox",
        side_effect=Exception("409 Client Error: Conflict (volume is in use)"),
    ):
        output = run_final_full_scan_node(
            {"workspace_volume": "agent_workspace_deadbeef", "baseline_scan_identifiers": []}
        )

    assert output["status"] == "final_scan_failed"
    final_result = output["final_full_scan_result"]
    assert final_result.completed is False
    assert final_result.authoritative is False
    assert final_result.coverage_complete is False
    assert final_result.status == "scan_failed"
    assert "409" in final_result.error
    assert "409" in output["errors"][0]


def test_final_full_scan_skips_docker_when_workspace_is_unavailable():
    with patch("remediation_engine.orchestration.final_full_scan.DockerSandbox") as sandbox_factory:
        output = run_final_full_scan_node({"workspace_volume": None})

    sandbox_factory.assert_not_called()
    final_result = output["final_full_scan_result"]
    assert output["status"] == "final_scan_failed"
    assert final_result.completed is False
    assert final_result.coverage_complete is False
    assert final_result.status == "scan_failed"
    assert "workspace volume is unavailable" in final_result.error
