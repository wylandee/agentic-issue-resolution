"""Offline tests for task-scoped ODC execution and fallback behavior."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    ODCScanEvidence,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
)
from remediation_engine.language import ProjectLanguage
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


def _python_target(
    package: str = "demo-root",
    version: str = "2.0.0",
) -> QAScanTarget:
    """Build one task-owned Python target with no lockfile requirement."""
    return QAScanTarget(
        task_id="task-1",
        group_id="group-1",
        target_package=package,
        expected_version=version,
        manifest_paths=(),
        dependency_ancestry=(),
        target_identifiers=frozenset({"CVE-2026-0001"}),
    )


def _pip_inspect_report(*, missing_leaf: bool = False) -> str:
    """Return a deterministic pip-inspect graph for targeted-scan tests."""
    installed = [
        {
            "metadata": {
                "name": "demo-root",
                "version": "2.0.0",
                "requires_dist": ["leaf>=1; python_version >= '3.10'"],
                "provides_extra": [],
            }
        },
        {
            "metadata": {
                "name": "unrelated",
                "version": "9.0.0",
                "requires_dist": [],
                "provides_extra": [],
            }
        },
    ]
    if not missing_leaf:
        installed.insert(
            1,
            {
                "metadata": {
                    "name": "leaf",
                    "version": "1.5.0",
                    "requires_dist": [],
                    "provides_extra": [],
                }
            },
        )
    return json.dumps(
        {
            "version": "1",
            "pip_version": "26.2.1",
            "environment": {
                "implementation_name": "cpython",
                "implementation_version": "3.11.8",
                "os_name": "posix",
                "platform_machine": "x86_64",
                "platform_python_implementation": "CPython",
                "python_full_version": "3.11.8",
                "python_version": "3.11",
                "sys_platform": "linux",
            },
            "installed": installed,
        }
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
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
        ) as targeted_scan,
        patch(
            "remediation_engine.orchestration.qa_odc._run_security_scan",
            return_value=full_result,
        ) as full_scan,
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
    assert [record.label for record in results.scan.scan_records] == ["odc:fallback-full"]
    targeted_scan.assert_not_called()
    full_scan.assert_called_once()
    evidence: ODCScanEvidence = results.scan_evidence
    assert evidence.effective_scope == ScanScope.FULL
    assert evidence.fallback_reason == ScanFallbackReason.MULTIPLE_TARGETS
    assert evidence.complete is False
    sandbox.run.assert_not_called()


def test_python_targeted_request_scans_installed_dependency_closure() -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(
            exit_code=0,
            stdout=_pip_inspect_report(),
            stderr="",
        ),
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
    ]
    targeted_result = _SecurityScanResult(
        True,
        "python targeted scan",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        scanned_packages={("demo-root", "2.0.0"), ("leaf", "1.5.0")},
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
            scan_targets=[_python_target()],
            project_language=ProjectLanguage.PYTHON,
        )

    full_scan.assert_not_called()
    targeted_scan.assert_called_once()
    assert targeted_scan.call_args.kwargs["project_language"] == ProjectLanguage.PYTHON
    assert sandbox.run.call_args_list[0].args[0] == (".venv/bin/python -m pip inspect --local")
    sandbox.write_file.assert_called_once_with(
        ".odc-targeted/000/requirements.txt",
        "demo-root==2.0.0\nleaf==1.5.0\n",
    )
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.closure_package_names == ["demo-root", "leaf"]
    assert results.scan_evidence.scanned_package_specs == [
        "demo-root==2.0.0",
        "leaf==1.5.0",
    ]


def test_python_targeted_scan_falls_back_for_unresolved_dependency() -> None:
    sandbox = MagicMock()
    sandbox.run.return_value = SimpleNamespace(
        exit_code=0,
        stdout=_pip_inspect_report(missing_leaf=True),
        stderr="",
    )
    full_result = _SecurityScanResult(
        True,
        "full fallback",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
    )

    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
        ) as targeted_scan,
        patch(
            "remediation_engine.orchestration.qa_odc._run_security_scan",
            return_value=full_result,
        ) as full_scan,
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
            scan_targets=[_python_target()],
            project_language=ProjectLanguage.PYTHON,
        )

    targeted_scan.assert_not_called()
    full_scan.assert_called_once()
    sandbox.write_file.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.FULL
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.INCOMPLETE_CLOSURE
    assert results.scan_evidence.complete is False


def test_python_targeted_scan_falls_back_when_odc_misses_closure_packages() -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(
            exit_code=0,
            stdout=_pip_inspect_report(),
            stderr="",
        ),
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
    ]
    targeted_result = _SecurityScanResult(
        True,
        "python scan omitted a dependency",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        scanned_packages={("demo-root", "2.0.0")},
    )
    full_result = _SecurityScanResult(
        True,
        "full fallback",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
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
        patch(
            "remediation_engine.orchestration.qa_odc._run_security_scan",
            return_value=full_result,
        ) as full_scan,
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
            scan_targets=[_python_target()],
            project_language=ProjectLanguage.PYTHON,
        )

    targeted_scan.assert_called_once()
    full_scan.assert_called_once()
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.FULL
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.INCOMPLETE_CLOSURE
    assert results.scan_evidence.complete is False


def test_unparseable_targeted_scan_is_not_misclassified_as_incomplete_closure() -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(exit_code=0, stdout=_pip_inspect_report(), stderr=""),
        SimpleNamespace(exit_code=0, stdout="", stderr=""),
    ]
    targeted_result = _SecurityScanResult(
        False,
        "FAILURE: targeted Dependency-Check exited 14.",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.UNPARSEABLE,
        scanned_packages=set(),
    )
    full_result = _SecurityScanResult(
        True,
        "full fallback",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
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
        patch(
            "remediation_engine.orchestration.qa_odc._run_security_scan",
            return_value=full_result,
        ) as full_scan,
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
            scan_targets=[_python_target()],
            project_language=ProjectLanguage.PYTHON,
        )

    targeted_scan.assert_called_once()
    full_scan.assert_called_once()
    assert results.scan_evidence is not None
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.TARGETED_SCAN_FAILED
    assert results.scan_evidence.complete is False
