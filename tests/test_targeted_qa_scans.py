"""Offline tests for task-scoped ODC execution and fallback behavior."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts.schemas import (
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    ODCScanEvidence,
    PackageScanOccurrence,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration import qa_odc
from remediation_engine.orchestration._qa_runtime import (
    _attribute_reported_occurrences,
    _maven_dependency_copy_command,
    _maven_odc_artifact_parity,
    _NonNpmInputProject,
    _package_lock_inventory,
    _package_occurrence_key,
    _resolve_targeted_packages,
    _scan_evidence_by_task,
    _scan_state_projection,
    _targeted_project_for_report_path,
    _TargetedArtifactManifest,
    _TargetedPackageBatch,
    _write_targeted_artifacts,
)
from remediation_engine.orchestration.qa_critic import (
    _missing_targeted_input_paths,
    _run_global_execution,
    run_workspace_setup_scan_node,
)
from remediation_engine.orchestration.qa_odc import (
    _ODCExecution,
    _ODCScanTimeout,
    _record_scan_result,
    _redact_odc_command,
    _run_odc_process,
)
from remediation_engine.orchestration.qa_policy_engine import (
    _apply_policy_decision,
    _evaluate_policy_gates,
)
from remediation_engine.orchestration.qa_test_parsing import (
    _QAInstallOutcome,
    _QATestExecutionOutcome,
)
from remediation_engine.orchestration.qa_types import (
    QAScanTarget,
    QATaskContext,
    _ODCReportedPackageOccurrence,
    _QAExecutionResults,
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


def _target_qa_context() -> QATaskContext:
    """Build a VERSION_BUMP task matching the targeted scan fixture."""
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-0001",
        package_name="a",
        package_version="1.0.0",
    )
    group = VulnerabilityGroup(
        group_id="group-1",
        issue_type=IssueType.SCA,
        vulnerable_component="a",
        file_paths=["package.json"],
        cve_ids=["CVE-2026-0001"],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="2.0.0",
            instruction="Upgrade a to 2.0.0.",
            strategy_used="test",
        ),
    )
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        qa_policy=QAPolicy.VERSION_BUMP,
        target_package_name="a",
    )
    return QATaskContext(task=task, group=group)


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


def test_odc_redacts_secret_arguments_from_logs_diagnostics_and_scan_records(
    tmp_path, caplog, monkeypatch
) -> None:
    """Keep ODC credentials out of command logs, diagnostic files, and QA evidence."""
    secrets = (
        "odc-api-key-test-only",
        "odc-nvd-key-test-only",
        "odc-proxy-pass-test-only",
        "odc-output-key-test-only",
    )
    command = [
        "docker",
        "run",
        "--apiKey",
        secrets[0],
        f"--nvdApiKey={secrets[1]}",
        "--proxyPass",
        secrets[2],
    ]
    execution = _ODCExecution(
        run_id="run-id",
        scope="targeted",
        container_name="remedy-odc-run-id",
        log_path=tmp_path / "odc.json",
        started_at_utc="2026-10-15T00:00:00+00:00",
    )
    process = subprocess.CompletedProcess(
        command,
        0,
        stdout=f"report complete --apiKey={secrets[3]}",
        stderr=f"proxy --proxyPass {secrets[2]}",
    )
    monkeypatch.setattr(qa_odc, "_new_odc_execution", lambda _scope: execution)
    monkeypatch.setattr(qa_odc, "_odc_command", lambda *_args: command)
    monkeypatch.setattr(qa_odc.subprocess, "run", lambda *_args, **_kwargs: process)

    result = _run_odc_process("workspace-volume", "targeted-dir")

    assert _redact_odc_command(command) == [
        "docker",
        "run",
        "--apiKey",
        "<redacted>",
        "--nvdApiKey=<redacted>",
        "--proxyPass",
        "<redacted>",
    ]
    assert _redact_odc_command(["echo", "tokenizer", "bypass", "still-visible"]) == [
        "echo",
        "tokenizer",
        "bypass",
        "still-visible",
    ]
    assert all(secret not in caplog.text for secret in secrets)
    diagnostic = json.loads(execution.log_path.read_text(encoding="utf-8"))
    assert all(secret not in json.dumps(diagnostic) for secret in secrets)

    scan = _record_scan_result(
        _SecurityScanResult(False, f"scan failed --apiKey {secrets[0]}", set(), set(), set()),
        label="odc:targeted",
        exit_code=1,
        stdout=f"--nvdApiKey={secrets[1]}",
        stderr=f"--proxyPass {secrets[2]}",
        error=f"subprocess failed --password {secrets[3]}",
    )
    scan_payload = {
        "summary": scan.summary,
        "stdout": scan.raw_stdout,
        "stderr": scan.raw_stderr,
        "record": scan.scan_records[0].__dict__,
    }
    assert all(secret not in json.dumps(scan_payload) for secret in secrets)
    assert result.returncode == 0


def test_odc_timeout_exception_redacts_command_and_streams(tmp_path) -> None:
    """A timeout's string form and retained streams must not expose credentials."""
    secret = "odc-timeout-secret-test-only"
    original = subprocess.TimeoutExpired(
        ["docker", "run", "--nvdApiKey", secret],
        10,
        output=f"retrying --apiKey={secret}",
        stderr=f"proxy --proxyPass {secret}",
    )
    execution = _ODCExecution(
        run_id="run-id",
        scope="full",
        container_name="remedy-odc-run-id",
        log_path=tmp_path / "odc-timeout.json",
        started_at_utc="2026-10-15T00:00:00+00:00",
    )
    cleanup = qa_odc._ODCCleanupResult(
        container_name=execution.container_name,
        succeeded=True,
        already_absent=False,
        returncode=0,
        stdout="",
        stderr="",
    )

    wrapped = _ODCScanTimeout(original, execution, cleanup, None)

    assert secret not in str(wrapped)
    assert secret not in str(wrapped.cmd)
    assert secret not in str(wrapped.output)
    assert secret not in str(wrapped.stderr)

    qa_odc._write_odc_diagnostic_log(
        execution,
        original.cmd,
        status="timeout",
        error=original,
    )
    diagnostic = execution.log_path.read_text(encoding="utf-8")
    assert secret not in diagnostic


def test_odc_exception_redacts_quoted_and_escaped_command_values() -> None:
    """Replace the exact command representation before scrubbing other text."""
    from remediation_engine.orchestration.qa_odc import _redact_odc_exception

    secret = "odc-quoted-'secret'\\path"
    errors = (
        subprocess.TimeoutExpired(["docker", "run", "--password", secret], 10),
        subprocess.CalledProcessError(1, ["docker", "run", "--nvdApiKey", secret]),
    )

    for error in errors:
        safe_message = _redact_odc_exception(error)
        assert secret not in safe_message
        assert "<redacted>" in safe_message


def test_package_lock_inventory_fingerprints_root_edges_and_occurrence_metadata() -> None:
    lockfile = {
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
            "node_modules/a": {
                "version": "1.0.0",
                "peerDependencies": {"b": "^2.0.0"},
                "peerDependenciesMeta": {"b": {"optional": True}},
            },
        }
    }
    inventory = _package_lock_inventory("package-lock.json", lockfile)
    root_key = _package_occurrence_key(
        "package-lock.json",
        "",
        "sample-project",
        "1.0.0",
    )
    package_key = _package_occurrence_key(
        "package-lock.json",
        "node_modules/a",
        "a",
        "1.0.0",
    )
    changed_root = {
        "packages": {
            **lockfile["packages"],
            "": {
                **lockfile["packages"][""],
                "dependencies": {"a": "2.0.0"},
            },
        }
    }
    changed_peer_metadata = {
        "packages": {
            **lockfile["packages"],
            "node_modules/a": {
                **lockfile["packages"]["node_modules/a"],
                "peerDependenciesMeta": {"b": {"optional": False}},
            },
        }
    }

    assert set(inventory) == {root_key, package_key}
    assert (
        inventory[root_key]
        != _package_lock_inventory(
            "package-lock.json",
            changed_root,
        )[root_key]
    )
    assert (
        inventory[package_key]
        != _package_lock_inventory(
            "package-lock.json",
            changed_peer_metadata,
        )[package_key]
    )
    assert root_key == json.dumps(
        ["package-lock.json", "", "sample-project", "1.0.0", 0],
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="packages mapping"):
        _package_lock_inventory("package-lock.json", {})


def test_package_lock_inventory_delta_builds_exact_changed_occurrence_closures() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"dep-b": "1.0.0"},
            },
            "node_modules/dep-b": {"version": "1.0.0"},
        },
    }
    current_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
            "node_modules/a": {
                "version": "1.0.0",
                "dependencies": {"dep-b": "2.0.0"},
            },
            "node_modules/dep-b": {"version": "2.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(current_lockfile)
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }

    batch = _resolve_targeted_packages(sandbox, [_target()], state)

    old_dependency_key = _package_occurrence_key(
        "package-lock.json",
        "node_modules/dep-b",
        "dep-b",
        "1.0.0",
    )
    new_dependency_key = _package_occurrence_key(
        "package-lock.json",
        "node_modules/dep-b",
        "dep-b",
        "2.0.0",
    )
    assert batch.failures_by_task == {}
    assert old_dependency_key in batch.inventory_removals
    assert new_dependency_key in batch.inventory_updates
    assert batch.covered_input_paths == ("package-lock.json",)
    assert {closure.root_keys[0] for closure in batch.closures_by_task["task-1"]} == {
        "node_modules/a",
        "node_modules/dep-b",
    }
    assert {closure.root_keys[0] for closure in batch.task_scoped_closures_by_task["task-1"]} == {
        "node_modules/a",
        "node_modules/dep-b",
    }
    scoped_nodes = {
        (node.lockfile_key, node.version)
        for closure in batch.task_scoped_closures_by_task["task-1"]
        for node in closure.nodes
    }
    assert scoped_nodes == {
        ("node_modules/a", "1.0.0"),
        ("node_modules/dep-b", "2.0.0"),
    }


def test_modified_root_dependency_edges_build_root_and_dependency_closures() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
            "node_modules/a": {"version": "1.0.0"},
        },
    }
    current_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "2.0.0"},
            },
            "node_modules/a": {"version": "2.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(current_lockfile)
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }

    batch = _resolve_targeted_packages(
        sandbox,
        [replace(_target(), expected_version="2.0.0")],
        state,
    )

    root_occurrence_key = _package_occurrence_key(
        "package-lock.json",
        "",
        "sample-project",
        "1.0.0",
    )
    assert batch.failures_by_task == {}
    assert root_occurrence_key in batch.inventory_updates
    assert {closure.root_keys[0] for closure in batch.closures_by_task["task-1"]} == {
        "",
        "node_modules/a",
    }


def test_targeted_resolution_uses_root_npm_override_for_lockfile_closure() -> None:
    source_path = "frontend/package-lock.json"
    packages = {
        "": {
            "name": "frontend",
            "version": "20.0.0",
            "dependencies": {"jsdom": "29.1.1"},
        },
        "node_modules/jsdom": {
            "version": "29.1.1",
            "dependencies": {"lru-cache": "^11.3.5"},
        },
        "node_modules/jsdom/node_modules/lru-cache": {"version": "11.2.7"},
        "node_modules/lru-cache": {"version": "5.1.1"},
    }
    package_lock = json.dumps({"lockfileVersion": 3, "packages": packages})
    package_json = json.dumps(
        {
            "name": "frontend",
            "version": "20.0.0",
            "overrides": {"lru-cache@^11": "~11.2.7"},
        }
    )
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: {
        source_path: package_lock,
        "frontend/package.json": package_json,
    }.get(path)
    target = QAScanTarget(
        task_id="task-jsdom",
        group_id="group-jsdom",
        target_package="jsdom",
        expected_version="29.1.1",
        manifest_paths=(source_path,),
        dependency_ancestry=(),
        target_identifiers=frozenset(),
    )

    batch = _resolve_targeted_packages(
        sandbox,
        [target],
        {
            "active_target_task_ids": [target.task_id],
            "accepted_package_inventory_by_lockfile": {},
            "task_queue": {},
            "worker_results_by_attempt": {},
            "attempt_snapshots_by_id": {},
        },
    )

    assert batch.failures_by_task == {}
    assert target.task_id in batch.covered_task_ids
    assert source_path in batch.covered_input_paths
    closure_nodes = {
        (node.lockfile_key, node.version)
        for closure in batch.closures_by_task[target.task_id]
        for node in closure.nodes
    }
    assert ("node_modules/jsdom/node_modules/lru-cache", "11.2.7") in closure_nodes


def test_retained_pivot_parent_lockfile_is_scanned_for_active_child() -> None:
    root_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
        },
    }
    child_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "child-project",
                "version": "1.0.0",
                "dependencies": {"child-lib": "1.0.0"},
            },
            "node_modules/child-lib": {"version": "1.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: json.dumps(
        child_lockfile if path == "packages/app/package-lock.json" else root_lockfile
    )
    baseline_inventory = _package_lock_inventory("package-lock.json", root_lockfile)
    state = {
        "accepted_package_inventory_by_lockfile": baseline_inventory,
        "task_queue": {
            "task-parent": SimpleNamespace(
                current_attempt_id="parent-attempt",
                parent_task_id=None,
                status="pivoted",
            ),
            "task-child": SimpleNamespace(
                current_attempt_id="child-attempt",
                parent_task_id="task-parent",
                status="optimistically_fixed",
            ),
        },
        "worker_results_by_attempt": {
            "parent-attempt": SimpleNamespace(
                task_id="task-parent",
                attempt_id="parent-attempt",
                task_revision=1,
                changed_files=["packages/app/package-lock.json"],
            ),
            "child-attempt": SimpleNamespace(
                task_id="task-child",
                attempt_id="child-attempt",
                task_revision=1,
                changed_files=[],
            ),
        },
    }
    target = replace(
        _target(),
        task_id="task-child",
        manifest_paths=("package-lock.json",),
    )

    batch = _resolve_targeted_packages(sandbox, [target], state)

    sandbox.read_file.assert_any_call("packages/app/package-lock.json")
    assert batch.failures_by_task == {}
    assert "packages/app/package-lock.json" in batch.covered_input_paths
    assert any(
        closure.source_lockfile == "packages/app/package-lock.json"
        for closure in batch.closures_by_task["task-child"]
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
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
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
    staged_files = sandbox.write_files.call_args.args[0]
    assert ".odc-targeted/000/package-lock.json" in staged_files
    sandbox.write_files.assert_called_once()
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)


def test_python_target_runs_pip_inspect_closure_and_pypi_targeted_scan() -> None:
    target = QAScanTarget(
        task_id="python-task",
        group_id="python-group",
        target_package="certifi",
        expected_version="2026.2.25",
        manifest_paths=("requirements.txt",),
        dependency_ancestry=(),
        target_identifiers=frozenset({"CVE-2026-0001"}),
    )
    inspect_report = {
        "version": "1",
        "environment": {"python_version": "3.11", "sys_platform": "linux"},
        "installed": [
            {
                "metadata": {
                    "name": "certifi",
                    "version": "2026.2.25",
                    "requires_dist": [],
                    "provides_extra": [],
                }
            }
        ],
    }
    targeted_result = _SecurityScanResult(
        True,
        "ok",
        set(),
        {"CVE-2026-0001"},
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_paths=[".odc-targeted/000/requirements.txt"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="certifi",
                version="2026.2.25",
                ecosystem="pypi",
                file_path=".odc-targeted/000/requirements.txt",
            )
        ],
    )
    sandbox = MagicMock()
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration._qa_runtime._run_readonly",
            return_value=SimpleNamespace(exit_code=0, stdout=json.dumps(inspect_report)),
        ) as pip_inspect,
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
            scan_targets=[target],
            project_language=ProjectLanguage.PYTHON,
        )

    pip_inspect.assert_called_once_with(
        sandbox,
        ".venv/bin/python -m pip inspect --local",
        timeout=60,
    )
    targeted_scan.assert_called_once()
    assert targeted_scan.call_args.kwargs["project_language"] == ProjectLanguage.PYTHON
    full_scan.assert_not_called()
    assert sandbox.write_file.call_args.args == (
        ".odc-targeted/000/requirements.txt",
        "certifi==2026.2.25\n",
    )
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.covered_task_ids == ["python-task"]
    assert results.scan_evidence.closure_package_names == ["certifi"]
    assert results.scan_evidence.covered_occurrences[0].source_input_path == "requirements.txt"


def test_successful_scan_records_removed_package_occurrence_tombstones() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
            "node_modules/removed": {"version": "1.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }
    targeted_result = _SecurityScanResult(
        True,
        "ok",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
    )
    removed_key = _package_occurrence_key(
        "package-lock.json",
        "node_modules/removed",
        "removed",
        "1.0.0",
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ),
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
            scan_state=state,
        )

    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.scan_performed is True
    assert results.scan_evidence.inventory_removals == [removed_key]
    assert results.scan_evidence.removed_occurrence_keys == [removed_key]
    assert results.scan_evidence.covered_input_paths == ["package-lock.json"]


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
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-0001",
        package_name="a",
        package_version="1.0.0",
        file_path=".odc-targeted/000/node_modules/a/package.json",
    )
    targeted_result = _SecurityScanResult(
        True,
        "ok",
        set(),
        {"CVE-2026-0001"},
        set(),
        [issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
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
    assert results.scan_target_failures_by_task == {}
    assert results.scan_target_skipped_by_task == {"task-2": "no_matching_target"}
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.found_identifiers == ["CVE-2026-0001"]


def test_multiple_physical_targets_are_scanned_without_full_workspace_fallback() -> None:
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

    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-0001",
        package_name="a",
        package_version="1.0.0",
        file_path=".odc-targeted/000/package-lock.json",
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        {"CVE-2026-0001"},
        set(),
        [issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/package-lock.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/package-lock.json",
            )
        ],
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
    assert results.scan_evidence is not None
    evidence: ODCScanEvidence = results.scan_evidence
    assert evidence.effective_scope == ScanScope.TARGETED
    assert evidence.fallback_reason is None
    assert evidence.complete is True
    assert evidence.covered_task_ids == ["task-1"]
    assert len(evidence.covered_occurrences) == 2
    assert all(
        occurrence.found_identifiers == ["CVE-2026-0001"]
        for occurrence in evidence.covered_occurrences
    )
    staged_files = sandbox.write_files.call_args.args[0]
    assert sum(path.endswith("/package-lock.json") for path in staged_files) == 1
    sandbox.write_files.assert_called_once()
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)


def test_missing_distinct_package_identity_remains_inconclusive() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {"dependencies": {"a": "1.0.0", "b": "2.0.0"}},
                "node_modules/a": {"version": "1.0.0"},
                "node_modules/nested/node_modules/a": {"version": "1.0.0"},
                "node_modules/b": {"version": "2.0.0"},
            },
        }
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/package-lock.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/package-lock.json",
            )
        ],
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

    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is False
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
    targeted_scan.assert_called_once()
    full_scan.assert_not_called()


def test_missing_target_package_skips_odc_and_is_complete_evidence() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {},
                "node_modules/other": {"version": "2.0.0"},
            },
        }
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
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

    targeted_scan.assert_not_called()
    full_scan.assert_not_called()
    assert results.scan is not None
    assert results.scan.ok is True
    assert results.scan.execution_status == ScannerExecutionStatus.NOT_RUN
    assert results.scan_skipped is True
    assert results.scan_target_skipped_by_task == {"task-1": "no_matching_target"}
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.scan_performed is False
    assert results.scan_evidence.covered_task_ids == ["task-1"]
    task_evidence = results.scan_evidence_by_task["task-1"]
    assert task_evidence.complete is True
    assert task_evidence.scan_performed is False
    assert task_evidence.scan_skip_reason == "no_matching_target"
    assert _scan_state_projection(results, {"CVE-2026-0001"})["new_vulnerability_status"] == "none"
    context = _target_qa_context()
    task_policies = {context.task_id: QAPolicy.VERSION_BUMP}
    gates, _ = _evaluate_policy_gates([context], results, task_policies)
    assert gates[context.task_id].target_package_absent is True
    evaluation_by_task, _ = _apply_policy_decision(
        [context],
        [
            QAEvaluation(
                task_id=context.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Package-specific review failed.",
            )
        ],
        gates,
        task_policies,
    )
    assert evaluation_by_task[context.task_id].passed is True


def test_target_package_at_another_version_is_not_absent() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "": {},
                "node_modules/a": {"version": "0.9.0"},
            },
        }
    )

    batch = _resolve_targeted_packages(sandbox, [_target()])

    assert batch.skipped_task_ids == ()
    assert batch.failures_by_task["task-1"].reason == ScanFallbackReason.TARGETED_SCAN_FAILED


@pytest.mark.parametrize(
    ("lockfile_text", "reason"),
    [
        (None, ScanFallbackReason.MISSING_LOCKFILE),
        ("{", ScanFallbackReason.INVALID_LOCKFILE),
        (json.dumps({"lockfileVersion": 3, "packages": []}), ScanFallbackReason.INVALID_LOCKFILE),
    ],
    ids=["unreadable", "malformed-json", "invalid-packages-map"],
)
def test_unreadable_or_invalid_lockfile_is_not_absence_evidence(
    lockfile_text: str | None,
    reason: ScanFallbackReason,
) -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = lockfile_text

    batch = _resolve_targeted_packages(sandbox, [_target()])

    assert batch.skipped_task_ids == ()
    assert batch.failures_by_task["task-1"].reason == reason


def test_absent_target_does_not_attribute_other_changed_closures_and_findings() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {},
            "node_modules/other": {"version": "1.0.0"},
        },
    }
    current_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {},
            "node_modules/other": {"version": "2.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(current_lockfile)
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }

    batch = _resolve_targeted_packages(sandbox, [_target()], state)

    assert batch.skipped_task_ids == ("task-1",)
    assert batch.failures_by_task == {}
    assert any(
        node.package_name == "other"
        for closure in batch.closures_by_task["task-1"]
        for node in closure.nodes
    )
    other = next(
        node
        for closure in batch.closures_by_task["task-1"]
        for node in closure.nodes
        if node.package_name == "other"
    )
    occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="other",
        version=other.version,
        lockfile_key=other.lockfile_key,
        found_identifiers=["CVE-2026-70001"],
    )
    project_path = ".odc-targeted/000"
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("other", "2.0.0"),),
        occurrences_by_project={project_path: (occurrence,)},
        task_ids_by_project={project_path: ("task-1",)},
        input_paths_by_project={project_path: ("package-lock.json",)},
    )
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        complete=True,
    )

    task_evidence = _scan_evidence_by_task(
        evidence,
        targets=[_target()],
        baseline_identifiers=set(),
        batch=batch,
        artifact_manifest=manifest,
    )["task-1"]

    assert task_evidence.complete is True
    assert task_evidence.scan_performed is True
    assert task_evidence.scan_skip_reason == "no_matching_target"
    assert task_evidence.covered_occurrences == []
    assert task_evidence.new_identifiers == []


def test_missing_target_does_not_invalidate_other_targets_new_identifier_list() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    target_b = replace(
        _target(),
        task_id="task-2",
        group_id="group-2",
        target_package="b",
        expected_version="2.0.0",
        target_identifiers=frozenset({"CVE-2026-0002"}),
    )
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-20001",
        package_name="a",
        package_version="1.0.0",
        file_path=".odc-targeted/000/node_modules/a/package.json",
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        {"CVE-2026-20001"},
        {"CVE-2026-20001"},
        [issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
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
            scan_targets=[_target(), target_b],
        )

    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.new_identifiers == ["CVE-2026-20001"]
    assert results.scan_target_failures_by_task == {}
    assert results.scan_target_skipped_by_task == {"task-2": "no_matching_target"}
    assert results.scan_evidence_by_task["task-1"].new_identifiers == ["CVE-2026-20001"]
    skipped = results.scan_evidence_by_task["task-2"]
    assert skipped.complete is True
    assert skipped.scan_performed is False
    assert skipped.scan_skip_reason == "no_matching_target"


def test_changed_non_npm_input_is_staged_and_scanned_without_full_workspace_fallback() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    sandbox.run.side_effect = lambda _command, timeout=30: SimpleNamespace(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )
    lockfile_document = json.loads(_lockfile())
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            lockfile_document,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["frontend/pom.xml"],
            )
        },
    }
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[
            ".odc-targeted/000/node_modules/a/package.json",
            ".odc-targeted/001/pom.xml",
        ],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            ),
            SimpleNamespace(
                package_name="org.example:library",
                version="2.1.0",
                ecosystem="maven",
                file_path=".odc-targeted/001/pom.xml",
            ),
        ],
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
            scan_state=state,
        )

    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    assert results.scan_evidence is not None
    assert "frontend/pom.xml" in results.scan_evidence.covered_input_paths
    copy_commands = [call.args[0] for call in sandbox.run.call_args_list if "cp --" in call.args[0]]
    assert copy_commands == [
        "mkdir -p -- .odc-targeted/001 && cp -- frontend/pom.xml .odc-targeted/001/pom.xml"
    ]


def test_changed_non_npm_input_without_companion_is_incomplete() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()

    def run(command: str, timeout: int = 30) -> SimpleNamespace:
        exit_code = 1 if "frontend/pyproject.toml" in command else 0
        return SimpleNamespace(
            exit_code=exit_code,
            stdout="",
            stderr="",
            duration_seconds=0.0,
        )

    sandbox.run.side_effect = run
    state = {
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            json.loads(_lockfile()),
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["frontend/poetry.lock"],
            )
        },
    }
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
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
            scan_state=state,
        )

    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is False
    assert results.scan_evidence.fallback_reason is not None
    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
    assert not any("cp --" in call.args[0] for call in sandbox.run.call_args_list)


def test_targeted_manifest_scans_each_package_lock_once_and_preserves_occurrences() -> None:
    sandbox = MagicMock()
    lockfile = {
        "lockfileVersion": 3,
        "name": "sample",
        "version": "1.0.0",
        "packages": {
            "": {"name": "sample", "version": "1.0.0", "dependencies": {"a": "1.0.0"}},
            "node_modules/a": {"version": "1.0.0"},
            "node_modules/nested/node_modules/a": {"version": "1.0.0"},
            "node_modules/b": {"version": "2.0.0"},
        },
    }
    sandbox.read_file.return_value = json.dumps(lockfile)
    targets = [
        _target(),
        replace(_target(), task_id="task-2", target_package="b", expected_version="2.0.0"),
    ]
    batch = _resolve_targeted_packages(sandbox, targets)

    manifest = _write_targeted_artifacts(sandbox, batch.closures_by_task)

    assert manifest.scan_subdir == ".odc-targeted"
    assert manifest.expected_package_occurrences == (
        ("a", "1.0.0"),
        ("a", "1.0.0"),
        ("b", "2.0.0"),
    )
    assert set(manifest.occurrences_by_project) == {".odc-targeted/000"}
    assert set(manifest.task_ids_by_project.values()) == {("task-1", "task-2")}
    assert {
        occurrence.lockfile_key
        for occurrence in manifest.occurrences_by_project[".odc-targeted/000"]
    } == {
        "node_modules/a",
        "node_modules/nested/node_modules/a",
        "node_modules/b",
    }
    assert manifest.input_paths_by_project == {".odc-targeted/000": ("package-lock.json",)}
    sandbox.write_files.assert_called_once()
    sandbox.write_file.assert_not_called()
    staged_files = sandbox.write_files.call_args.args[0]
    assert set(staged_files) == {
        ".odc-targeted/000/package.json",
        ".odc-targeted/000/package-lock.json",
        ".odc-targeted/000/node_modules/a/package.json",
        ".odc-targeted/000/node_modules/b/package.json",
        ".odc-targeted/000/node_modules/nested/node_modules/a/package.json",
    }
    assert sum(path.endswith("/package-lock.json") for path in staged_files) == 1
    assert json.loads(staged_files[".odc-targeted/000/node_modules/a/package.json"]) == {
        "name": "a",
        "version": "1.0.0",
    }
    assert json.loads(
        staged_files[".odc-targeted/000/node_modules/nested/node_modules/a/package.json"]
    ) == {"name": "a", "version": "1.0.0"}


def test_trace_esbuild_target_stages_files_in_one_batch() -> None:
    sandbox = MagicMock()
    lockfile = {
        "lockfileVersion": 3,
        "name": "juice-shop",
        "version": "1.0.0",
        "packages": {
            "": {
                "name": "juice-shop",
                "version": "1.0.0",
                "dependencies": {"esbuild": "0.28.1"},
            },
            "node_modules/esbuild": {"version": "0.28.1"},
        },
    }
    sandbox.read_file.return_value = json.dumps(lockfile)
    trace_target = replace(
        _target(),
        task_id="task-3",
        target_package="esbuild",
        expected_version="0.28.1",
    )

    batch = _resolve_targeted_packages(sandbox, [trace_target])
    manifest = _write_targeted_artifacts(sandbox, batch.closures_by_task)

    assert batch.failures_by_task == {}
    assert manifest.expected_package_occurrences == (("esbuild", "0.28.1"),)
    sandbox.write_files.assert_called_once()
    sandbox.write_file.assert_not_called()
    staged_files = sandbox.write_files.call_args.args[0]
    assert set(staged_files) == {
        ".odc-targeted/000/package.json",
        ".odc-targeted/000/package-lock.json",
        ".odc-targeted/000/node_modules/esbuild/package.json",
    }
    assert json.loads(staged_files[".odc-targeted/000/node_modules/esbuild/package.json"]) == {
        "name": "esbuild",
        "version": "0.28.1",
    }


def test_missing_targeted_project_path_keeps_coverage_inconclusive() -> None:
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
        set(),
        set(),
        [],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0"), ("a", "1.0.0")],
        reported_paths=[],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/package-lock.json",
            ),
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/package-lock.json",
            ),
        ],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ),
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

    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is False
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.TARGETED_SCAN_FAILED
    assert results.scan is not None and results.scan.ok is False
    full_scan.assert_not_called()


def test_report_coverage_requires_every_non_npm_companion_path() -> None:
    manifest = SimpleNamespace(
        occurrences_by_project={".odc-targeted/001": ()},
        input_paths_by_project={".odc-targeted/001": ("frontend/pom.xml", "frontend/Cargo.lock")},
    )

    assert _missing_targeted_input_paths(
        manifest,
        [".odc-targeted/001/pom.xml"],
    ) == ["frontend/Cargo.lock"]


def test_maven_project_root_report_path_keeps_targeted_project_identity() -> None:
    assert _targeted_project_for_report_path("/scan/.odc-targeted/000") == ".odc-targeted/000"
    assert (
        _targeted_project_for_report_path("/scan/.odc-targeted/000/dependencies/org/yaml/a.jar")
        == ".odc-targeted/000"
    )
    assert _targeted_project_for_report_path("/scan/.odc-targeted") is None


def test_maven_project_root_report_does_not_claim_standalone_pom_coverage() -> None:
    manifest = SimpleNamespace(
        occurrences_by_project={".odc-targeted/000": ()},
        input_paths_by_project={".odc-targeted/000": ("pom.xml",)},
    )

    assert _missing_targeted_input_paths(manifest, [".odc-targeted/000"]) == ["pom.xml"]


def test_maven_dependency_tree_parser_preserves_resolved_ancestry() -> None:
    from remediation_engine.orchestration._qa_runtime import (
        _parse_maven_dependency_tree_paths,
    )

    output = r"""[INFO] --- maven-dependency-plugin:3.6.1:tree (default-cli) @ demo ---
[INFO] +- org.springframework.boot:spring-boot-starter:jar:2.1.2.RELEASE:compile
[INFO] |  +- ch.qos.logback:logback-classic:jar:1.2.3:compile
[INFO] |  |  \- ch.qos.logback:logback-core:jar:1.2.3:compile
[INFO] |  \- org.example:omitted:jar:1.0:compile (omitted for conflict with 2.0)
"""

    assert _parse_maven_dependency_tree_paths(output) == (
        ("org.springframework.boot:spring-boot-starter",),
        (
            "org.springframework.boot:spring-boot-starter",
            "ch.qos.logback:logback-classic",
        ),
        (
            "org.springframework.boot:spring-boot-starter",
            "ch.qos.logback:logback-classic",
            "ch.qos.logback:logback-core",
        ),
    )


def test_maven_artifact_copy_command_is_scoped_to_pom_and_target_directory() -> None:
    command = _maven_dependency_copy_command(
        "services/api/pom.xml", ".odc-targeted/003/dependencies/000"
    )

    assert "cd services/api && mvn" in command
    assert "-DoutputDirectory=/workspace/.odc-targeted/003/dependencies/000" in command
    assert "-Dmdep.useRepositoryLayout=true" in command
    assert "-DincludeScope=test" in command
    assert "-DmarkersDirectory=/workspace/.odc-targeted/003/dependencies/000/.markers" in command
    assert "org.apache.maven.plugins:maven-dependency-plugin:3.6.1:copy-dependencies" in command


@pytest.mark.parametrize(
    ("pom_path", "output_directory"),
    [
        ("pom.xml", "../outside"),
        ("module/build.gradle", ".odc-targeted/000/dependencies"),
    ],
)
def test_maven_artifact_copy_command_rejects_unscoped_paths(pom_path, output_directory) -> None:
    with pytest.raises(ValueError):
        _maven_dependency_copy_command(pom_path, output_directory)


def test_maven_artifacts_are_copied_from_original_pom_into_targeted_project() -> None:
    sandbox = MagicMock()
    artifact_path = ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar"
    sandbox.run.side_effect = [
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0, stdout=artifact_path + "\n"),
        SimpleNamespace(exit_code=0, stdout=artifact_path + "\n"),
    ]
    project = _NonNpmInputProject(
        source_input_path="services/api/pom.xml",
        staged_input_paths=("services/api/pom.xml",),
        task_ids=("task-1",),
        occurrences=(
            PackageScanOccurrence(
                source_input_path="services/api/pom.xml",
                package_name="org.yaml:snakeyaml",
                version="2.0",
                ecosystem="maven",
            ),
        ),
        preserve_relative_paths=True,
        package_manager="maven",
    )

    manifest = _write_targeted_artifacts(sandbox, {}, (project,))

    assert manifest.input_paths_by_project[".odc-targeted/000"] == ("services/api/pom.xml",)
    assert sandbox.run.call_count == 4
    staged_pom_command = sandbox.run.call_args_list[0].args[0]
    copy_dependencies_command = sandbox.run.call_args_list[1].args[0]
    assert "cp -- services/api/pom.xml .odc-targeted/000/services/api/pom.xml" in staged_pom_command
    assert "cd services/api && mvn" in copy_dependencies_command
    assert "/workspace/.odc-targeted/000/dependencies/000" in copy_dependencies_command
    assert manifest.maven_artifacts_by_project[".odc-targeted/000"] == {
        artifact_path: ("org.yaml:snakeyaml", "2.0")
    }


def _maven_project_with_one_artifact() -> _NonNpmInputProject:
    return _NonNpmInputProject(
        source_input_path="pom.xml",
        staged_input_paths=("pom.xml",),
        task_ids=("task-1",),
        occurrences=(
            PackageScanOccurrence(
                source_input_path="pom.xml",
                package_name="org.yaml:snakeyaml",
                version="2.0",
                ecosystem="maven",
            ),
        ),
        preserve_relative_paths=True,
        package_manager="maven",
    )


def test_maven_artifact_pruning_removes_only_unlisted_files_in_target_directory() -> None:
    expected_path = ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar"
    extra_path = ".odc-targeted/000/dependencies/000/org/ow2/asm/asm/5.0.4/asm-5.0.4.jar"
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0, stdout=f"{expected_path}\n{extra_path}\n"),
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0, stdout=expected_path + "\n"),
    ]

    _write_targeted_artifacts(sandbox, {}, (_maven_project_with_one_artifact(),))

    prune_command = sandbox.run.call_args_list[3].args[0]
    assert prune_command.startswith("rm -- ")
    assert extra_path in prune_command
    assert expected_path not in prune_command


@pytest.mark.parametrize(
    ("listed_paths", "message"),
    [
        ("", "missing=.*snakeyaml"),
        (
            ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar\n"
            ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0-sources.jar",
            "ambiguous=.*snakeyaml",
        ),
        (
            ".odc-targeted/001/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar",
            "escaped its targeted directory",
        ),
    ],
)
def test_maven_artifact_staging_rejects_missing_ambiguous_or_escaped_paths(
    listed_paths: str,
    message: str,
) -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=0, stdout=listed_paths),
    ]

    with pytest.raises((RuntimeError, ValueError), match=message):
        _write_targeted_artifacts(sandbox, {}, (_maven_project_with_one_artifact(),))


def test_maven_odc_parity_requires_report_paths_to_match_staged_artifacts() -> None:
    project_path = ".odc-targeted/000"
    artifact_path = ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar"
    occurrence = PackageScanOccurrence(
        source_input_path="pom.xml",
        package_name="org.yaml:snakeyaml",
        version="2.0",
        ecosystem="maven",
        lockfile_key="org.yaml:snakeyaml:jar:compile",
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(),
        occurrences_by_project={project_path: (occurrence, occurrence)},
        task_ids_by_project={project_path: ("task-1",)},
        input_paths_by_project={project_path: ("pom.xml",)},
        maven_artifacts_by_project={project_path: {artifact_path: ("org.yaml:snakeyaml", "2.0")}},
    )
    direct = _ODCReportedPackageOccurrence(
        package_name="org.yaml:snakeyaml",
        version="2.0",
        ecosystem="maven",
        file_path=artifact_path,
        evidence_source="maven_repository_layout",
    )
    embedded = _ODCReportedPackageOccurrence(
        package_name="org.yaml:metadata",
        version="1.0",
        ecosystem="maven",
        file_path=artifact_path + "/META-INF/maven/org.yaml/metadata/pom.xml",
        evidence_source="embedded_maven_pom",
        owner_file_path=artifact_path,
    )

    assert _maven_odc_artifact_parity(manifest, [direct, embedded]) == (True, ())
    unlisted = replace(direct, file_path=artifact_path.replace("snakeyaml", "other"))
    complete, diagnostics = _maven_odc_artifact_parity(manifest, [unlisted])
    assert complete is False
    assert any("staged artifact" in diagnostic for diagnostic in diagnostics)


def test_maven_task_qa_maps_filename_issues_and_validates_embedded_metadata() -> None:
    project_path = ".odc-targeted/000"
    artifacts = {
        "org.yaml:snakeyaml": (
            "2.0",
            ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/2.0/snakeyaml-2.0.jar",
        ),
        "org.apache.tomcat.embed:tomcat-embed-core": (
            "9.0.14",
            ".odc-targeted/000/dependencies/000/org/apache/tomcat/embed/"
            "tomcat-embed-core/9.0.14/tomcat-embed-core-9.0.14.jar",
        ),
        "net.bytebuddy:byte-buddy": (
            "1.9.7",
            ".odc-targeted/000/dependencies/000/net/bytebuddy/byte-buddy/"
            "1.9.7/byte-buddy-1.9.7.jar",
        ),
        "org.postgresql:postgresql": (
            "42.2.5",
            ".odc-targeted/000/dependencies/000/org/postgresql/postgresql/"
            "42.2.5/postgresql-42.2.5.jar",
        ),
    }
    occurrences = tuple(
        PackageScanOccurrence(
            source_input_path="pom.xml",
            package_name=package_name,
            version=version,
            ecosystem="maven",
            lockfile_key=f"{package_name}:jar:compile",
        )
        for package_name, (version, _path) in artifacts.items()
    )
    artifact_paths = {
        path: (package_name, version) for package_name, (version, path) in artifacts.items()
    }
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(),
        occurrences_by_project={project_path: occurrences},
        task_ids_by_project={project_path: ("task-1",)},
        input_paths_by_project={project_path: ("pom.xml",)},
        maven_artifacts_by_project={project_path: artifact_paths},
    )
    tomcat_version, tomcat_path = artifacts["org.apache.tomcat.embed:tomcat-embed-core"]
    issues = [
        VulnerabilityIssue(
            source=IssueSource.ODC,
            issue_type=IssueType.SCA,
            severity=Severity.HIGH,
            cve_id=f"CVE-2026-{10000 + index}",
            package_name="tomcat-embed-core-9.0.14.jar",
            package_version="",
            file_path=tomcat_path,
            ecosystem="maven",
        )
        for index in range(147)
    ]
    direct_reports = [
        _ODCReportedPackageOccurrence(
            package_name=package_name,
            version=version,
            ecosystem="maven",
            file_path=path,
            evidence_source="maven_repository_layout",
        )
        for package_name, (version, path) in artifacts.items()
    ]
    embedded_reports = [
        _ODCReportedPackageOccurrence(
            package_name="net.bytebuddy:byte-buddy-dep",
            version="1.9.7",
            ecosystem="maven",
            file_path=(
                artifacts["net.bytebuddy:byte-buddy"][1]
                + "/META-INF/maven/net.bytebuddy/byte-buddy-dep/pom.xml"
            ),
            evidence_source="embedded_maven_pom",
            owner_file_path=artifacts["net.bytebuddy:byte-buddy"][1],
        ),
        _ODCReportedPackageOccurrence(
            package_name="com.ongres.scram:client",
            version="1.0.0-beta.2",
            ecosystem="maven",
            file_path=(
                artifacts["org.postgresql:postgresql"][1]
                + "/META-INF/maven/com.ongres.scram/client/pom.xml"
            ),
            evidence_source="embedded_maven_pom",
            owner_file_path=artifacts["org.postgresql:postgresql"][1],
        ),
        _ODCReportedPackageOccurrence(
            package_name="com.ongres.scram:common",
            version="1.0.0-beta.2",
            ecosystem="maven",
            file_path=(
                artifacts["org.postgresql:postgresql"][1]
                + "/META-INF/maven/com.ongres.scram/common/pom.xml"
            ),
            evidence_source="embedded_maven_pom",
            owner_file_path=artifacts["org.postgresql:postgresql"][1],
        ),
    ]
    scan_result = _SecurityScanResult(
        ok=True,
        summary="Dependency-Check completed",
        remaining_identifiers=set(),
        found_identifiers={issue.cve_id for issue in issues if issue.cve_id},
        new_identifiers=set(),
        found_issues=issues,
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_paths=[".odc-targeted/000/pom.xml", *artifact_paths],
        reported_package_occurrences=[*direct_reports, *embedded_reports],
    )
    project = _NonNpmInputProject(
        source_input_path="pom.xml",
        staged_input_paths=("pom.xml",),
        task_ids=("task-1",),
        occurrences=occurrences,
        preserve_relative_paths=True,
        package_manager="maven",
    )
    batch = _TargetedPackageBatch(
        target_packages=(),
        closures_by_task={},
        covered_task_ids=("task-1",),
        failures_by_task={},
        inventory_updates={},
        inventory_removals=(),
        covered_input_paths=("pom.xml",),
        non_npm_input_projects=(project,),
    )
    target = QAScanTarget(
        task_id="task-1",
        group_id="group-1",
        target_package="org.yaml:snakeyaml",
        expected_version="2.0",
        manifest_paths=("pom.xml",),
        dependency_ancestry=(),
        target_identifiers=frozenset(),
        package_manager="maven",
    )

    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._resolve_targeted_packages",
            return_value=batch,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts",
            return_value=manifest,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._qa_odc_module._run_incremental_security_scan",
            return_value=scan_result,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._cleanup_targeted_artifacts",
        ),
    ):
        results = _run_global_execution(
            MagicMock(),
            "workspace-volume",
            set(),
            set(),
            scan_targets=[target],
            project_language=ProjectLanguage.JAVA,
        )

    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.fallback_reason is None
    tomcat_occurrence = next(
        occurrence
        for occurrence in results.scan_evidence.covered_occurrences
        if occurrence.package_name == "org.apache.tomcat.embed:tomcat-embed-core"
    )
    assert len(tomcat_occurrence.found_issues) == 147
    assert {issue.package_name for issue in tomcat_occurrence.found_issues} == {
        "org.apache.tomcat.embed:tomcat-embed-core"
    }
    assert {issue.package_version for issue in tomcat_occurrence.found_issues} == {tomcat_version}
    assert {issue.file_path for issue in tomcat_occurrence.found_issues} == {tomcat_path}


def test_maven_artifact_copy_failure_fails_targeted_staging_closed() -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = [
        SimpleNamespace(exit_code=0),
        SimpleNamespace(exit_code=1),
    ]
    project = _NonNpmInputProject(
        source_input_path="pom.xml",
        staged_input_paths=("pom.xml",),
        task_ids=("task-1",),
        package_manager="maven",
    )

    with pytest.raises(RuntimeError, match="Maven dependency artifact staging failed"):
        _write_targeted_artifacts(sandbox, {}, (project,))
def test_report_coverage_accepts_odc_package_suffix_but_keeps_missing_paths() -> None:
    manifest = SimpleNamespace(
        occurrences_by_project={".odc-targeted/001": ()},
        input_paths_by_project={".odc-targeted/001": ("requirements.txt", "pyproject.toml")},
    )

    assert _missing_targeted_input_paths(
        manifest,
        [".odc-targeted/001/requirements.txt:certifi/2024.7.4"],
    ) == ["pyproject.toml"]

    requirements_only = SimpleNamespace(
        occurrences_by_project={".odc-targeted/001": ()},
        input_paths_by_project={".odc-targeted/001": ("requirements.txt",)},
    )
    assert (
        _missing_targeted_input_paths(
            requirements_only,
            [".odc-targeted/001/requirements.txt:certifi/2024.7.4"],
        )
        == []
    )

    assert _missing_targeted_input_paths(
        requirements_only,
        [".odc-targeted/001/requirements.txt:unrecognized"],
    ) == ["requirements.txt"]


def test_active_task_without_package_target_uses_incremental_input_scanner() -> None:
    sandbox = MagicMock()
    sandbox.run.side_effect = lambda _command, timeout=30: SimpleNamespace(
        exit_code=0,
        stdout="",
        stderr="",
        duration_seconds=0.0,
    )
    state = {
        "active_target_task_ids": ["task-1"],
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["requirements.txt"],
            )
        },
    }
    incremental_issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-10001",
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        package_name="sample",
        package_version="1.0.0",
        file_path=".odc-targeted/000/requirements.txt",
    )
    scan_result = _SecurityScanResult(
        True,
        "incremental",
        set(),
        {"CVE-BASE", "CVE-2026-10001", "GHSA-ABCD-EFGH-IJKL"},
        {"CVE-2026-10001", "GHSA-ABCD-EFGH-IJKL"},
        [incremental_issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[],
        reported_paths=[".odc-targeted/000/requirements.txt"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="sample",
                version="1.0.0",
                ecosystem="pypi",
                file_path=".odc-targeted/000/requirements.txt",
            )
        ],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_incremental_security_scan",
            return_value=scan_result,
        ) as incremental_scan,
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
        ) as task_targeted_scan,
        patch("remediation_engine.orchestration.qa_odc._run_security_scan") as full_scan,
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            set(),
            {"CVE-BASE"},
            scan_state=state,
        )

    incremental_scan.assert_called_once()
    task_targeted_scan.assert_not_called()
    full_scan.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.new_identifiers == [
        "CVE-2026-10001",
        "GHSA-ABCD-EFGH-IJKL",
    ]
    assert results.scan_evidence.covered_task_ids == ["task-1"]
    occurrence = results.scan_evidence.covered_occurrences[0]
    assert occurrence.source_input_path == "requirements.txt"
    assert occurrence.lockfile_key is None
    assert occurrence.occurrence_index == 0
    assert occurrence.found_identifiers == ["CVE-2026-10001", "GHSA-ABCD-EFGH-IJKL"]
    assert occurrence.found_issues == [incremental_issue]


def test_non_npm_repeated_package_occurrences_receive_stable_indices() -> None:
    project_path = ".odc-targeted/000"
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(),
        occurrences_by_project={project_path: ()},
        task_ids_by_project={project_path: ("task-1",)},
        input_paths_by_project={project_path: ("requirements.txt",)},
    )
    reported_occurrence = SimpleNamespace(
        package_name="library",
        version="2.0.0",
        ecosystem="pypi",
        file_path=f"{project_path}/requirements.txt",
    )
    scan_result = SimpleNamespace(
        reported_package_occurrences=[reported_occurrence, reported_occurrence],
        found_issues=[],
    )

    attributed_manifest, complete = _attribute_reported_occurrences(manifest, scan_result)

    assert complete is True
    occurrences = attributed_manifest.occurrences_by_project[project_path]
    assert [occurrence.occurrence_index for occurrence in occurrences] == [0, 1]
    assert all(occurrence.source_input_path == "requirements.txt" for occurrence in occurrences)
    assert all(occurrence.lockfile_key is None for occurrence in occurrences)


def test_task_evidence_keeps_new_cve_and_ghsa_outside_target_identifiers() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-20001",
        ghsa_id="GHSA-WXYZ-1234-ABCD",
        package_name="a",
        package_version="1.0.0",
        file_path=".odc-targeted/000/node_modules/a/package.json",
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        {"CVE-2026-20001", "GHSA-WXYZ-1234-ABCD"},
        {"CVE-2026-20001", "GHSA-WXYZ-1234-ABCD"},
        [issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("a", "1.0.0")],
        reported_paths=[".odc-targeted/000/node_modules/a/package.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="a",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/node_modules/a/package.json",
            )
        ],
    )
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan",
            return_value=targeted_result,
        ),
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
            return_value=_test_outcome(),
        ),
    ):
        results = _run_global_execution(
            sandbox,
            "workspace-volume",
            {"CVE-2026-0001"},
            {"CVE-BASE"},
            scan_targets=[_target()],
        )

    evidence = results.scan_evidence_by_task["task-1"]
    assert evidence.found_identifiers == [
        "CVE-2026-20001",
        "GHSA-WXYZ-1234-ABCD",
    ]
    assert evidence.new_identifiers == [
        "CVE-2026-20001",
        "GHSA-WXYZ-1234-ABCD",
    ]
    assert evidence.remaining_target_identifiers == []
    assert evidence.found_issues == [issue]


def test_failed_artifact_write_still_cleans_temporary_targeted_directory() -> None:
    sandbox = MagicMock()
    sandbox.read_file.return_value = _lockfile()
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts",
            side_effect=ValueError("cannot materialize closure"),
        ),
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
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

    targeted_scan.assert_not_called()
    full_scan.assert_not_called()
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is False


def test_manifest_task_projection_preserves_new_ghsa_with_existing_cve() -> None:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2025-0001",
        ghsa_id="GHSA-ABCD-EFGH-IJKL",
        package_name="express",
        package_version="4.0.0",
        file_path=".odc-targeted/000/node_modules/express/package.json",
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("express", "4.0.0"), ("helmet", "7.0.0")),
        occurrences_by_project={
            ".odc-targeted/000": (
                PackageScanOccurrence(
                    source_input_path="package-lock.json",
                    package_name="express",
                    version="4.0.0",
                    lockfile_key="node_modules/express",
                    found_identifiers=["CVE-2025-0001", "GHSA-ABCD-EFGH-IJKL"],
                    found_issues=[issue],
                ),
            ),
            ".odc-targeted/001": (
                PackageScanOccurrence(
                    source_input_path="frontend/package-lock.json",
                    package_name="helmet",
                    version="7.0.0",
                    lockfile_key="node_modules/helmet",
                ),
            ),
        },
        task_ids_by_project={
            ".odc-targeted/000": ("task-1",),
            ".odc-targeted/001": ("task-2",),
        },
        input_paths_by_project={
            ".odc-targeted/000": ("package-lock.json",),
            ".odc-targeted/001": ("frontend/package-lock.json",),
        },
    )
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1", "task-2"],
        found_identifiers=["CVE-2025-0001", "GHSA-ABCD-EFGH-IJKL"],
        new_identifiers=["GHSA-ABCD-EFGH-IJKL"],
        complete=True,
    )
    targets = [
        replace(
            _target(),
            task_id="task-1",
            target_identifiers=frozenset({"CVE-2025-0001"}),
        ),
        replace(
            _target(),
            task_id="task-2",
            target_identifiers=frozenset({"CVE-2025-0002"}),
        ),
    ]

    task_evidence = _scan_evidence_by_task(
        evidence,
        targets=targets,
        baseline_identifiers={"CVE-2025-0001"},
        batch=None,
        artifact_manifest=manifest,
    )

    assert task_evidence["task-1"].closure_package_names == ["express"]
    assert task_evidence["task-1"].found_identifiers == [
        "CVE-2025-0001",
        "GHSA-ABCD-EFGH-IJKL",
    ]
    assert task_evidence["task-1"].new_identifiers == ["GHSA-ABCD-EFGH-IJKL"]
    assert task_evidence["task-1"].remaining_target_identifiers == ["CVE-2025-0001"]
    assert task_evidence["task-1"].found_issues == [issue]
    assert task_evidence["task-2"].closure_package_names == ["helmet"]
    assert task_evidence["task-2"].found_identifiers == []


def test_shared_maven_pom_projection_uses_exact_gav_for_task_verdicts() -> None:
    issue_for_other_group = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-10001",
        package_name="org.beta:common",
        package_version="2.0.0",
    )
    issue_for_beta = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-10002",
        package_name="org.beta:common",
        package_version="2.0.0",
    )
    alpha_occurrence = PackageScanOccurrence(
        source_input_path="pom.xml",
        package_name="org.alpha:common",
        version="1.0.0",
        ecosystem="maven",
        lockfile_key="org.alpha:common:jar:compile",
    )
    beta_occurrence = PackageScanOccurrence(
        source_input_path="pom.xml",
        package_name="org.beta:common",
        version="2.0.0",
        ecosystem="maven",
        lockfile_key="org.beta:common:jar:compile",
        found_identifiers=["CVE-2026-10001", "CVE-2026-10002"],
        found_issues=[issue_for_other_group, issue_for_beta],
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(
            ("org.alpha:common", "1.0.0"),
            ("org.beta:common", "2.0.0"),
        ),
        occurrences_by_project={".odc-targeted/000": (alpha_occurrence, beta_occurrence)},
        task_ids_by_project={".odc-targeted/000": ("task-alpha", "task-beta")},
        input_paths_by_project={".odc-targeted/000": ("pom.xml",)},
    )
    cumulative_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-alpha", "task-beta"],
        found_identifiers=["CVE-2026-10001", "CVE-2026-10002"],
        new_identifiers=["CVE-2026-10001", "CVE-2026-10002"],
        found_issues=[issue_for_other_group, issue_for_beta],
        complete=True,
    )
    targets = [
        QAScanTarget(
            task_id="task-alpha",
            group_id="group-alpha",
            target_package="org.alpha:common",
            expected_version="1.0.0",
            manifest_paths=("pom.xml",),
            dependency_ancestry=(),
            target_identifiers=frozenset({"CVE-2026-10001"}),
            package_manager="maven",
        ),
        QAScanTarget(
            task_id="task-beta",
            group_id="group-beta",
            target_package="org.beta:common",
            expected_version="2.0.0",
            manifest_paths=("pom.xml",),
            dependency_ancestry=(),
            target_identifiers=frozenset({"CVE-2026-10002"}),
            package_manager="maven",
        ),
    ]

    task_evidence = _scan_evidence_by_task(
        cumulative_evidence,
        targets=targets,
        baseline_identifiers=set(),
        batch=None,
        artifact_manifest=manifest,
    )

    assert cumulative_evidence.found_identifiers == [
        "CVE-2026-10001",
        "CVE-2026-10002",
    ]
    assert task_evidence["task-alpha"].closure_package_names == ["org.alpha:common"]
    assert task_evidence["task-alpha"].found_identifiers == []
    assert task_evidence["task-alpha"].remaining_target_identifiers == []
    assert task_evidence["task-alpha"].found_issues == []
    assert task_evidence["task-beta"].closure_package_names == ["org.beta:common"]
    assert task_evidence["task-beta"].remaining_target_identifiers == ["CVE-2026-10002"]
    assert task_evidence["task-beta"].found_issues == [issue_for_other_group, issue_for_beta]


def test_shared_lockfile_report_projects_only_task_owned_occurrences() -> None:
    issue_a = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-10001",
        package_name="a",
        package_version="1.0.0",
    )
    issue_b = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-10002",
        package_name="b",
        package_version="2.0.0",
    )
    occurrence_a = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="a",
        version="1.0.0",
        lockfile_key="node_modules/a",
        found_identifiers=["CVE-2026-10001"],
        found_issues=[issue_a],
    )
    occurrence_b = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="b",
        version="2.0.0",
        lockfile_key="node_modules/b",
        found_identifiers=["CVE-2026-10002"],
        found_issues=[issue_b],
    )
    occurrence_extra = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="ip-address",
        version="10.2.0",
        lockfile_key="node_modules/ip-address",
        found_identifiers=["CVE-2026-10003"],
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(
            ("a", "1.0.0"),
            ("b", "2.0.0"),
            ("ip-address", "10.2.0"),
        ),
        occurrences_by_project={
            ".odc-targeted/000": (occurrence_a, occurrence_b, occurrence_extra)
        },
        task_ids_by_project={".odc-targeted/000": ("task-1", "task-2")},
        input_paths_by_project={".odc-targeted/000": ("package-lock.json",)},
    )
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1", "task-2"],
        found_identifiers=["CVE-2026-10001", "CVE-2026-10002", "CVE-2026-10003"],
        new_identifiers=["CVE-2026-10001", "CVE-2026-10002", "CVE-2026-10003"],
        complete=True,
    )
    closure_a = SimpleNamespace(
        source_lockfile="package-lock.json",
        root_keys=("node_modules/a",),
        nodes=(
            SimpleNamespace(
                lockfile_key="node_modules/a",
                package_name="a",
                version="1.0.0",
            ),
        ),
    )
    closure_b = SimpleNamespace(
        source_lockfile="package-lock.json",
        root_keys=("node_modules/b",),
        nodes=(
            SimpleNamespace(
                lockfile_key="node_modules/b",
                package_name="b",
                version="2.0.0",
            ),
        ),
    )
    changed_closure = SimpleNamespace(
        source_lockfile="package-lock.json",
        root_keys=("node_modules/ip-address",),
        nodes=(
            SimpleNamespace(
                lockfile_key="node_modules/ip-address",
                package_name="ip-address",
                version="10.2.0",
            ),
        ),
    )
    batch = SimpleNamespace(
        covered_task_ids=("task-1", "task-2"),
        failures_by_task={},
        # All affected closures are visible to the shared scan; each task keeps
        # only its own target and the changed dependency closure.
        closures_by_task={
            "task-1": (closure_a, closure_b, changed_closure),
            "task-2": (closure_a, closure_b, changed_closure),
        },
        task_scoped_closures_by_task={
            "task-1": (closure_a, changed_closure),
            "task-2": (closure_b, changed_closure),
        },
        covered_input_paths=("package-lock.json",),
        inventory_updates={},
        inventory_removals=(),
    )
    targets = [
        replace(_target(), target_identifiers=frozenset({"CVE-2026-10001"})),
        replace(
            _target(),
            task_id="task-2",
            target_package="b",
            expected_version="2.0.0",
            target_identifiers=frozenset({"CVE-2026-10002"}),
        ),
    ]

    task_evidence = _scan_evidence_by_task(
        evidence,
        targets=targets,
        baseline_identifiers=set(),
        batch=batch,
        artifact_manifest=manifest,
    )

    assert task_evidence["task-1"].closure_package_names == ["a"]
    assert task_evidence["task-1"].found_identifiers == ["CVE-2026-10001"]
    assert task_evidence["task-1"].found_issues == [issue_a]
    assert task_evidence["task-2"].closure_package_names == ["b"]
    assert task_evidence["task-2"].found_identifiers == ["CVE-2026-10002"]
    assert task_evidence["task-2"].found_issues == [issue_b]


def test_targeted_extra_args_conflict_does_not_start_odc() -> None:
    sandbox = MagicMock()
    with (
        patch(
            "remediation_engine.orchestration.qa_test_parsing._run_install",
            return_value=_install_outcome(),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._targeted_extra_args_conflict",
            return_value=True,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts"
        ) as write_artifacts,
        patch(
            "remediation_engine.orchestration.qa_odc._run_targeted_security_scan"
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

    write_artifacts.assert_not_called()
    targeted_scan.assert_not_called()
    full_scan.assert_not_called()
    sandbox.run.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is False


def test_nofix_deletion_only_delta_emits_complete_unscanned_tombstones() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
        },
    }
    current_lockfile = {"lockfileVersion": 3, "packages": {"": {}}}
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(current_lockfile)
    state = {
        "active_target_task_ids": ["task-1"],
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }
    target = replace(
        _target(),
        target_identifiers=frozenset({"CVE-2026-0001"}),
    )
    removed_key = _package_occurrence_key(
        "package-lock.json",
        "node_modules/a",
        "a",
        "1.0.0",
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
            "remediation_engine.orchestration.qa_odc._run_incremental_security_scan"
        ) as incremental_scan,
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
            {"CVE-BASE"},
            scan_targets=[target],
            skip_scan=True,
            scan_skip_reason="no_fix_package_removal",
            scan_state=state,
        )

    targeted_scan.assert_not_called()
    incremental_scan.assert_not_called()
    full_scan.assert_not_called()
    assert results.scan_skipped is True
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.scan_performed is False
    assert results.scan_evidence.inventory_removals == [removed_key]
    assert results.scan_evidence.removed_occurrence_keys == [removed_key]
    assert results.scan_evidence_by_task["task-1"].complete is True
    assert results.scan_evidence_by_task["task-1"].removed_occurrence_keys == [removed_key]


def test_nofix_addition_delta_runs_incremental_scan_before_qa() -> None:
    baseline_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"a": "1.0.0"},
            },
            "node_modules/a": {"version": "1.0.0"},
        },
    }
    current_lockfile = {
        "lockfileVersion": 3,
        "packages": {
            "": {
                "name": "sample-project",
                "version": "1.0.0",
                "dependencies": {"new-dep": "1.0.0"},
            },
            "node_modules/new-dep": {"version": "1.0.0"},
        },
    }
    sandbox = MagicMock()
    sandbox.read_file.return_value = json.dumps(current_lockfile)
    state = {
        "active_target_task_ids": ["task-1"],
        "accepted_package_inventory_by_lockfile": _package_lock_inventory(
            "package-lock.json",
            baseline_lockfile,
        ),
        "task_queue": {
            "task-1": SimpleNamespace(
                current_attempt_id="attempt-1",
                parent_task_id=None,
                status="optimistically_fixed",
            )
        },
        "worker_results_by_attempt": {
            "attempt-1": SimpleNamespace(
                task_id="task-1",
                attempt_id="attempt-1",
                task_revision=1,
                changed_files=["package-lock.json"],
            )
        },
    }
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-20001",
        package_name="new-dep",
        package_version="1.0.0",
        file_path=".odc-targeted/000/package-lock.json",
    )
    targeted_result = _SecurityScanResult(
        True,
        "targeted",
        set(),
        {"CVE-2026-20001"},
        {"CVE-2026-20001"},
        [issue],
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_packages=[("new-dep", "1.0.0")],
        reported_paths=[".odc-targeted/000/package-lock.json"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="new-dep",
                version="1.0.0",
                ecosystem="npm",
                file_path=".odc-targeted/000/package-lock.json",
            )
        ],
    )
    target = replace(_target(), target_identifiers=frozenset({"CVE-2026-0001"}))
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
            "remediation_engine.orchestration.qa_odc._run_incremental_security_scan"
        ) as incremental_scan,
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
            {"CVE-BASE"},
            scan_targets=[target],
            skip_scan=True,
            scan_skip_reason="no_fix_package_removal",
            scan_state=state,
        )

    targeted_scan.assert_called_once()
    incremental_scan.assert_not_called()
    full_scan.assert_not_called()
    assert results.scan_skipped is False
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.scan_performed is True
    assert results.scan_evidence.new_identifiers == ["CVE-2026-20001"]
    assert results.scan_evidence_by_task["task-1"].new_identifiers == ["CVE-2026-20001"]


def test_workspace_setup_scan_stages_changed_package_lockfile_inputs(tmp_path):
    (tmp_path / "package.json").write_text('{"name":"sample"}', encoding="utf-8")
    issue = VulnerabilityIssue(
        id="00000000-0000-4000-8000-000000000001",
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        message="setup scan finding",
        cve_id="CVE-2026-70001",
        package_name="lodash",
        package_version="4.17.21",
        file_path="package-lock.json",
    )
    occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="lodash",
        version="4.17.21",
        lockfile_key="node_modules/lodash",
        found_identifiers=["CVE-2026-70001"],
        found_issues=[issue],
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("lodash", "4.17.21"),),
        occurrences_by_project={".odc-targeted/000": (occurrence,)},
        task_ids_by_project={".odc-targeted/000": ("workspace_setup",)},
        input_paths_by_project={".odc-targeted/000": ("package-lock.json",)},
    )
    change_batch = SimpleNamespace(
        target_packages=(),
        closures_by_task={"workspace_setup": (object(),)},
        failures_by_task={},
        inventory_updates={
            '["package-lock.json","node_modules/lodash","lodash","4.17.21",0]': "fp"
        },
        inventory_removals=(),
        covered_input_paths=("package-lock.json",),
    )
    scan_result = SimpleNamespace(
        ok=True,
        summary="scan completed",
        execution_status=ScannerExecutionStatus.SUCCESS,
        reported_paths=[".odc-targeted/000/package-lock.json"],
        reported_packages=[("lodash", "4.17.21")],
        found_identifiers={"CVE-2026-70001"},
        new_identifiers={"CVE-2026-70001"},
        found_issues=[issue],
        remaining_identifiers=set(),
    )
    sandbox = MagicMock()
    sandbox_context = MagicMock()
    sandbox_context.__enter__.return_value = sandbox
    sandbox_context.__exit__.return_value = False

    with (
        patch(
            "remediation_engine.orchestration.qa_critic.DockerSandbox",
            return_value=sandbox_context,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._targeted_extra_args_conflict",
            return_value=False,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._changed_package_lock_inputs",
            return_value=change_batch,
        ) as compare_inventory,
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts",
            return_value=manifest,
        ) as write_artifacts,
        patch(
            "remediation_engine.orchestration.qa_critic._qa_odc_module._run_incremental_security_scan",
            return_value=scan_result,
        ) as scan,
        patch(
            "remediation_engine.orchestration.qa_critic._missing_targeted_input_paths",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._attribute_reported_occurrences",
            return_value=(manifest, True),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._cleanup_targeted_artifacts",
        ) as cleanup,
    ):
        result = run_workspace_setup_scan_node(
            {
                "repo_root": str(tmp_path),
                "workspace_volume": "workspace-volume",
                "package_inventory_baseline_complete": True,
                "accepted_package_inventory_by_lockfile": {},
                "baseline_scan_identifiers": ["CVE-BASE"],
            }
        )

    compare_inventory.assert_called_once()
    assert compare_inventory.call_args.args[1][0].manifest_paths == ("package-lock.json",)
    write_artifacts.assert_called_once_with(
        sandbox,
        change_batch.closures_by_task,
    )
    scan.assert_called_once_with(
        sandbox,
        "workspace-volume",
        ".odc-targeted",
        {"CVE-BASE"},
    )
    cleanup.assert_called_once_with(sandbox)
    evidence = result["setup_scan_evidence"]
    assert evidence.complete is True
    assert evidence.scan_performed is True
    assert evidence.new_identifiers == ["CVE-2026-70001"]
    assert evidence.covered_occurrences == [occurrence]
    assert result["errors"] == []


def test_incremental_scan_matches_full_scan_oracle_for_changed_closure():
    baseline = {"CVE-2024-70001"}
    baseline_issue = VulnerabilityIssue(
        id="00000000-0000-4000-8000-000000000002",
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        message="baseline finding",
        cve_id="CVE-2024-70001",
        package_name="a",
        package_version="1.0.0",
        file_path="package-lock.json",
    )
    introduced_issue = VulnerabilityIssue(
        id="00000000-0000-4000-8000-000000000003",
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        message="introduced finding",
        cve_id="CVE-2026-70010",
        package_name="a",
        package_version="1.0.0",
        file_path="package-lock.json",
    )
    occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="a",
        version="1.0.0",
        lockfile_key="node_modules/a",
        found_identifiers=["CVE-2024-70001", "CVE-2026-70010"],
        found_issues=[baseline_issue, introduced_issue],
    )
    target = replace(
        _target(),
        target_identifiers=frozenset({"CVE-2024-70001"}),
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("a", "1.0.0"),),
        occurrences_by_project={".odc-targeted/000": (occurrence,)},
        task_ids_by_project={".odc-targeted/000": ("task-1",)},
        input_paths_by_project={".odc-targeted/000": ("package-lock.json",)},
    )
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        complete=True,
        covered_occurrences=[occurrence],
    )
    full_scan_projection = _scan_state_projection(
        _QAExecutionResults(
            scan=_SecurityScanResult(
                ok=True,
                summary="oracle scan completed",
                remaining_identifiers=set(),
                found_identifiers={"CVE-2024-70001", "CVE-2026-70010"},
                new_identifiers={"CVE-2026-70010"},
                found_issues=[baseline_issue, introduced_issue],
            )
        ),
        baseline,
        authoritative=True,
    )
    incremental_projection = _scan_evidence_by_task(
        evidence,
        targets=[target],
        baseline_identifiers=baseline,
        batch=None,
        artifact_manifest=manifest,
    )["task-1"]

    assert (
        incremental_projection.found_identifiers
        == full_scan_projection["post_remediation_scan_identifiers"]
    )
    assert (
        incremental_projection.new_identifiers
        == full_scan_projection["new_vulnerability_identifiers"]
    )
    assert (
        incremental_projection.found_issues == full_scan_projection["post_remediation_scan_issues"]
    )


def test_unmapped_setup_lockfile_delta_prevents_clean_coverage(tmp_path):
    (tmp_path / "package.json").write_text('{"name":"sample"}', encoding="utf-8")
    failure = SimpleNamespace(
        reason=ScanFallbackReason.INVALID_LOCKFILE,
        detail="changed occurrence could not be mapped to a package closure",
    )
    change_batch = SimpleNamespace(
        target_packages=(),
        closures_by_task={},
        failures_by_task={"workspace_setup": failure},
        inventory_updates={},
        inventory_removals=(),
        covered_input_paths=(),
    )
    sandbox_context = MagicMock()
    sandbox_context.__enter__.return_value = MagicMock()
    sandbox_context.__exit__.return_value = False

    with (
        patch(
            "remediation_engine.orchestration.qa_critic.DockerSandbox",
            return_value=sandbox_context,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._targeted_extra_args_conflict",
            return_value=False,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._changed_package_lock_inputs",
            return_value=change_batch,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._qa_odc_module._run_incremental_security_scan",
        ) as scan,
    ):
        result = run_workspace_setup_scan_node(
            {
                "repo_root": str(tmp_path),
                "workspace_volume": "workspace-volume",
                "package_inventory_baseline_complete": True,
                "accepted_package_inventory_by_lockfile": {},
            }
        )

    scan.assert_not_called()
    assert result["setup_scan_evidence"].complete is False
    assert result["setup_scan_evidence"].fallback_reason == ScanFallbackReason.INVALID_LOCKFILE


def test_maven_workspace_setup_scan_uses_native_gav_and_java_sandbox(tmp_path):
    pom_path = tmp_path / "pom.xml"
    pom_path.write_text("<project/>", encoding="utf-8")
    occurrence = PackageScanOccurrence(
        source_input_path="pom.xml",
        package_name="org.yaml:snakeyaml",
        version="1.33",
        ecosystem="maven",
        lockfile_key="org.yaml:snakeyaml:jar:compile",
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("org.yaml:snakeyaml", "1.33"),),
        occurrences_by_project={".odc-targeted/000": (occurrence,)},
        task_ids_by_project={".odc-targeted/000": ("workspace_setup",)},
        input_paths_by_project={".odc-targeted/000": ("pom.xml",)},
        maven_artifacts_by_project={
            ".odc-targeted/000": {
                ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/1.33/snakeyaml-1.33.jar": (
                    "org.yaml:snakeyaml",
                    "1.33",
                )
            }
        },
    )
    artifact_path = ".odc-targeted/000/dependencies/000/org/yaml/snakeyaml/1.33/snakeyaml-1.33.jar"
    reported = _ODCReportedPackageOccurrence(
        package_name="org.yaml:snakeyaml",
        version="1.33",
        ecosystem="maven",
        file_path=artifact_path,
        evidence_source="maven_repository_layout",
    )
    scan_result = SimpleNamespace(
        execution_status=ScannerExecutionStatus.SUCCESS,
        summary="Dependency-Check completed",
        reported_paths=[".odc-targeted/000/pom.xml", artifact_path],
        reported_package_occurrences=[reported],
        found_identifiers=set(),
        new_identifiers=set(),
        remaining_identifiers=set(),
        found_issues=[],
    )
    sandbox = MagicMock()
    sandbox_context = MagicMock()
    sandbox_context.__enter__.return_value = sandbox
    sandbox_context.__exit__.return_value = False

    with (
        patch(
            "remediation_engine.orchestration.qa_critic.DockerSandbox",
            return_value=sandbox_context,
        ) as docker_sandbox,
        patch(
            "remediation_engine.orchestration.qa_critic.locate_maven_manifests",
            return_value=[pom_path],
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._maven_dependency_tree_occurrences",
            return_value=(occurrence,),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts",
            return_value=manifest,
        ) as write_artifacts,
        patch(
            "remediation_engine.orchestration.qa_critic._qa_odc_module._run_incremental_security_scan",
            return_value=scan_result,
        ) as scan,
        patch(
            "remediation_engine.orchestration.qa_critic._cleanup_targeted_artifacts",
        ) as cleanup,
    ):
        result = run_workspace_setup_scan_node(
            {
                "repo_root": str(tmp_path),
                "workspace_volume": "workspace-volume",
                "maven_cache_volume": "maven-cache-volume",
                "project_language": "java",
                # Maven setup evidence is independent of npm inventory state.
                "package_inventory_baseline_complete": False,
            }
        )

    sandbox_options = docker_sandbox.call_args.kwargs
    assert sandbox_options["image"] == "maven:3.9-eclipse-temurin-17"
    assert sandbox_options["maven_repository_volume"] == "maven-cache-volume"
    staged_project = write_artifacts.call_args.args[2][0]
    assert staged_project.staged_input_paths == ("pom.xml",)
    assert staged_project.preserve_relative_paths is True
    assert staged_project.package_manager == "maven"
    scan.assert_called_once_with(
        sandbox,
        "workspace-volume",
        ".odc-targeted",
        set(),
    )
    cleanup.assert_called_once_with(sandbox)
    evidence = result["setup_scan_evidence"]
    assert evidence.complete is True
    assert evidence.scan_performed is True
    assert evidence.covered_input_paths == ["pom.xml"]
    assert evidence.covered_occurrences == [occurrence]
    assert result["errors"] == []


def test_maven_workspace_setup_scan_fails_closed_on_odc_gav_mismatch(tmp_path):
    pom_path = tmp_path / "pom.xml"
    pom_path.write_text("<project/>", encoding="utf-8")
    occurrence = PackageScanOccurrence(
        source_input_path="pom.xml",
        package_name="org.yaml:snakeyaml",
        version="1.33",
        ecosystem="maven",
        lockfile_key="org.yaml:snakeyaml:jar:compile",
    )
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("org.yaml:snakeyaml", "1.33"),),
        occurrences_by_project={".odc-targeted/000": (occurrence,)},
        task_ids_by_project={".odc-targeted/000": ("workspace_setup",)},
        input_paths_by_project={".odc-targeted/000": ("pom.xml",)},
    )
    scan_result = SimpleNamespace(
        execution_status=ScannerExecutionStatus.SUCCESS,
        summary="Dependency-Check completed",
        reported_paths=[".odc-targeted/000/pom.xml"],
        reported_package_occurrences=[
            _ODCReportedPackageOccurrence(
                package_name="org.yaml:snakeyaml",
                version="1.32",
                ecosystem="maven",
                file_path=".odc-targeted/000/pom.xml",
            )
        ],
        found_identifiers=set(),
        new_identifiers=set(),
        remaining_identifiers=set(),
        found_issues=[],
    )
    sandbox_context = MagicMock()
    sandbox_context.__enter__.return_value = MagicMock()
    sandbox_context.__exit__.return_value = False

    with (
        patch(
            "remediation_engine.orchestration.qa_critic.DockerSandbox",
            return_value=sandbox_context,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic.locate_maven_manifests",
            return_value=[pom_path],
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._maven_dependency_tree_occurrences",
            return_value=(occurrence,),
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._write_targeted_artifacts",
            return_value=manifest,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._qa_odc_module._run_incremental_security_scan",
            return_value=scan_result,
        ),
        patch(
            "remediation_engine.orchestration.qa_critic._cleanup_targeted_artifacts",
        ),
    ):
        result = run_workspace_setup_scan_node(
            {
                "repo_root": str(tmp_path),
                "workspace_volume": "workspace-volume",
                "project_language": "java",
            }
        )

    evidence = result["setup_scan_evidence"]
    assert evidence.complete is False
    assert evidence.fallback_reason == ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
    assert result["errors"]
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


def test_multiple_target_occurrences_are_retained_in_the_task_batch() -> None:
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
    batch = _resolve_targeted_packages(sandbox, [_target()])

    assert batch.failures_by_task == {}
    assert {
        resolution.target_node.lockfile_key
        for resolution in batch.target_packages
        if resolution.target_node is not None
    } == {"node_modules/a", "node_modules/nested/node_modules/a"}
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
        reported_paths=[".odc-targeted/000/requirements.txt"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="demo-root",
                version="2.0.0",
                ecosystem="pypi",
                file_path=".odc-targeted/000/requirements.txt",
            ),
            SimpleNamespace(
                package_name="leaf",
                version="1.5.0",
                ecosystem="pypi",
                file_path=".odc-targeted/000/requirements.txt",
            ),
        ],
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
    assert sandbox.run.call_args_list[0].args[0] == ".venv/bin/python -m pip inspect --local"
    sandbox.write_file.assert_called_once_with(
        ".odc-targeted/000/requirements.txt",
        "demo-root==2.0.0\nleaf==1.5.0\n",
    )
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
    assert results.scan_evidence.complete is True
    assert results.scan_evidence.closure_package_names == ["demo-root", "leaf"]
    assert [
        (item.package_name, item.version) for item in results.scan_evidence.covered_occurrences
    ] == [("demo-root", "2.0.0"), ("leaf", "1.5.0")]


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
    full_scan.assert_not_called()
    sandbox.write_file.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
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
        reported_paths=[".odc-targeted/000/requirements.txt"],
        reported_package_occurrences=[
            SimpleNamespace(
                package_name="demo-root",
                version="2.0.0",
                ecosystem="pypi",
                file_path=".odc-targeted/000/requirements.txt",
            )
        ],
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
    full_scan.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.effective_scope == ScanScope.TARGETED
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.TARGET_PACKAGE_NOT_REPORTED
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
    full_scan.assert_not_called()
    assert results.scan_evidence is not None
    assert results.scan_evidence.fallback_reason == ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
    assert results.scan_evidence.complete is False
