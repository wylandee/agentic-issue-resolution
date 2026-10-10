"""Offline tests for task-scoped ODC execution and fallback behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    ODCScanEvidence,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
)
from remediation_engine.orchestration.qa_critic import _run_global_execution
from remediation_engine.orchestration.qa_test_parsing import (
    _QAInstallOutcome,
    _QATestExecutionOutcome,
)
from remediation_engine.orchestration.qa_types import (
    QAScanTarget,
    _QALogRecord,
    _SecurityScanResult,
)


def _target() -> QAScanTarget:
    """Build one task-owned npm target for execution tests."""
    return QAScanTarget(
        task_id="task-1",
        group_id="group-1",
        target_package="a",
        expected_version="1.0.0",
        manifest_paths=("package-lock.json",),
        dependency_ancestry=(),
        target_identifiers=frozenset({"CVE-2026-0001"}),
    )


def _lockfile() -> str:
    """Return a minimal live npm lockfile."""
    return json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {},
                "node_modules/a": {"version": "1.0.0"},
            },
        }
    )


def _install_outcome(ok: bool = True) -> _QAInstallOutcome:
    """Build a typed install fixture for global execution tests."""
    return _QAInstallOutcome(
        ok=ok,
        summary="ok" if ok else "install failed",
        exit_code=0 if ok else 1,
        error_category=None if ok else "PEER_CONFLICT",
        raw_stdout="",
        raw_stderr="",
        log_record=_QALogRecord(
            phase="install",
            label="npm install",
            exit_code=0 if ok else 1,
            stdout="",
            stderr="",
        ),
    )


def _test_outcome(ok: bool = True) -> _QATestExecutionOutcome:
    """Build a typed test fixture for global execution tests."""
    return _QATestExecutionOutcome(
        ok=ok,
        summary="ok" if ok else "tests failed",
        exit_code=0 if ok else 1,
        failure_count=0 if ok else None,
        raw_stdout="",
        raw_stderr="",
        log_records=(
            _QALogRecord(
                phase="tests",
                label="npm test",
                exit_code=0 if ok else 1,
                stdout="",
                stderr="",
            ),
        ),
    )


def test_supported_target_runs_targeted_scan_and_attaches_evidence() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    targeted_result = _SecurityScanResult(
        True,
        "ok",
        set(),
        {"CVE-2026-0001"},
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ) as targeted_scan,
        patch("remediation_engine.orchestration.qa_odc._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            {"CVE-2026-0001"},
            {"CVE-2026-0001"},
            scan_targets=[_target()],
        )

    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    assert results.scan is not None
    assert results.scan.ok is True
    assert [record.label for record in results.scan.scan_records] == ["odc:targeted"]
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
    assert results.scan_evidence.authoritative is False
    assert results.scan_evidence.covered_task_ids == ["task-1"]
    sandbox.write_file.assert_any_call(
        ".odc-targeted/000/package-lock.json",
        sandbox.write_file.call_args_list[1].args[1],
    )
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)


def test_no_matching_target_skips_only_that_package_from_batch() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    missing_target = QAScanTarget(
        task_id="task-2",
        group_id="group-2",
        target_package="missing",
        expected_version="1.0.0",
        manifest_paths=("package-lock.json",),
        dependency_ancestry=(),
        target_identifiers=frozenset({"CVE-2026-0002"}),
    )
    targeted_result = _SecurityScanResult(
        True,
        "ok",
        set(),
        {"CVE-2026-0001"},
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ) as targeted_scan,
        patch("remediation_engine.orchestration.qa_odc._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            {"CVE-2026-0001", "CVE-2026-0002"},
            {"CVE-2026-0001", "CVE-2026-0002"},
            scan_targets=[_target(), missing_target],
        )

    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    assert results.scan is not None
    assert results.scan.execution_status == ScannerExecutionStatus.SUCCESS
    assert results.scan_target_failures_by_task["task-2"][0] == (
        ScanFallbackReason.NO_MATCHING_TARGET
    )
    assert results.scan_evidence is not None
    assert results.scan_evidence.covered_task_ids == ["task-1"]
    assert results.scan_evidence.complete is False
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.NO_MATCHING_TARGET


def test_multiple_target_occurrences_are_scanned_independently() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {},
                "node_modules/a": {"version": "1.0.0"},
                "node_modules/nested/node_modules/a": {"version": "1.0.0"},
            },
        }
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        {"CVE-2026-0001"},
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0"), ("a", "1.0.0")],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ) as targeted_scan,
        patch("remediation_engine.orchestration.qa_odc._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            {"CVE-2026-0001"},
            {"CVE-2026-0001"},
            scan_targets=[_target()],
        )

    assert results.scan is not None
    assert results.scan.execution_status == ScannerExecutionStatus.SUCCESS
    assert [record.label for record in results.scan.scan_records] == ["odc:targeted"]
    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    evidence: ODCScanEvidence = results.scan_evidence
    assert evidence.effective_scope == ScanScope.TARGETED
    assert evidence.fallback_reason is None
    assert evidence.complete is True
    assert evidence.covered_task_ids == ["task-1"]
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)
