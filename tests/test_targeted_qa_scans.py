"""Offline tests for task-scoped ODC execution and fallback behavior."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    ODCScanEvidence,
    ScanFallbackReason,
    ScanScope,
)
from remediation_engine.orchestration.qa_critic import (
    QAScanTarget,
    _QAInstallOutcome,
    _QALogRecord,
    _QATestExecutionOutcome,
    _run_global_execution,
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
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_critic._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._run_targeted_security_scan",
            return_value=targeted_result,
        ) as targeted_scan,
        patch("remediation_engine.orchestration.qa_critic._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_unit_tests",
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


def test_multiple_targets_falls_back_to_existing_full_scan() -> None:
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
    full_result = _SecurityScanResult(True, "full", set(), set(), set(), [])
    with (
        patch(
            "remediation_engine.orchestration.qa_critic._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._run_targeted_security_scan"
        ) as targeted_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_security_scan",
            return_value=full_result,
        ) as full_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_unit_tests",
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
    assert [record.label for record in results.scan.scan_records] == ["odc:fallback-full"]
    targeted_scan.assert_not_called()
    full_scan.assert_called_once()
    evidence: ODCScanEvidence = results.scan_evidence
    assert evidence.effective_scope == ScanScope.FULL
    assert evidence.fallback_reason == ScanFallbackReason.MULTIPLE_TARGETS
    assert evidence.complete is False
    sandbox.run.assert_not_called()
