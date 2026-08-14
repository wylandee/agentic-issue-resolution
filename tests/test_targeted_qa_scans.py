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
        patch("remediation_engine.orchestration.qa_critic._run_install", return_value=(True, "ok")),
        patch(
            "remediation_engine.orchestration.qa_critic._run_targeted_security_scan",
            return_value=targeted_result,
        ) as targeted_scan,
        patch("remediation_engine.orchestration.qa_critic._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_unit_tests", return_value=(True, "ok")
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
    assert results.scan == targeted_result
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
    assert results.scan_evidence.authoritative is False
    assert results.scan_evidence.covered_task_ids == ["task-1"]
    sandbox.write_file.assert_any_call(
        ".odc-targeted/000/package-lock.json",
        sandbox.write_file.call_args_list[1].args[1],
    )
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)


def test_ambiguous_target_falls_back_to_existing_full_scan() -> None:
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
        patch("remediation_engine.orchestration.qa_critic._run_install", return_value=(True, "ok")),
        patch(
            "remediation_engine.orchestration.qa_critic._run_targeted_security_scan"
        ) as targeted_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_security_scan",
            return_value=full_result,
        ) as full_scan,
        patch(
            "remediation_engine.orchestration.qa_critic._run_unit_tests", return_value=(True, "ok")
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            {"CVE-2026-0001"},
            {"CVE-2026-0001"},
            scan_targets=[_target()],
        )

    targeted_scan.assert_not_called()
    full_scan.assert_called_once()
    evidence: ODCScanEvidence = results.scan_evidence
    assert evidence.effective_scope == ScanScope.FULL
    assert evidence.fallback_reason == ScanFallbackReason.AMBIGUOUS_TARGET
    assert evidence.complete is False
    sandbox.run.assert_not_called()
