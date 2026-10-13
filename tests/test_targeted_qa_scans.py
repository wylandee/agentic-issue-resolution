"""Offline tests for task-scoped ODC execution and fallback behavior."""

from __future__ import annotations

import json
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
from remediation_engine.orchestration._qa_runtime import (
    _attribute_reported_occurrences,
    _package_lock_inventory,
    _package_occurrence_key,
    _resolve_targeted_packages,
    _scan_evidence_by_task,
    _scan_state_projection,
    _TargetedArtifactManifest,
    _write_targeted_artifacts,
)
from remediation_engine.orchestration.qa_critic import (
    _missing_targeted_input_paths,
    _run_global_execution,
    run_workspace_setup_scan_node,
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
    assert any(
        call.args[0] == ".odc-targeted/000/package-lock.json"
        for call in sandbox.write_file.call_args_list
    )
    sandbox.run.assert_called_once_with("rm -rf -- .odc-targeted", timeout=30)


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
    assert [record.label for record in results.scan.scan_records] == ["odc:targeted"]
    assert results.scan_evidence is not None
    assert results.scan_evidence.complete is True
    assert len(results.scan_evidence.covered_occurrences) == 2
    assert all(
        occurrence.found_identifiers == ["CVE-2026-0001"]
        for occurrence in results.scan_evidence.covered_occurrences
    )
    assert (
        len(
            [
                call
                for call in sandbox.write_file.call_args_list
                if call.args[0].endswith("/package-lock.json")
            ]
        )
        == 1
    )
    targeted_scan.assert_called_once()
    full_scan.assert_not_called()
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


def test_absent_target_keeps_other_changed_closures_and_findings() -> None:
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
    assert task_evidence.covered_occurrences == [occurrence]
    assert task_evidence.new_identifiers == ["CVE-2026-70001"]


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
    written_files = {call.args[0]: call.args[1] for call in sandbox.write_file.call_args_list}
    assert set(written_files) == {
        ".odc-targeted/000/package.json",
        ".odc-targeted/000/package-lock.json",
        ".odc-targeted/000/node_modules/a/package.json",
        ".odc-targeted/000/node_modules/b/package.json",
        ".odc-targeted/000/node_modules/nested/node_modules/a/package.json",
    }
    assert sum(path.endswith("/package-lock.json") for path in written_files) == 1
    assert json.loads(written_files[".odc-targeted/000/node_modules/a/package.json"]) == {
        "name": "a",
        "version": "1.0.0",
    }
    assert json.loads(
        written_files[".odc-targeted/000/node_modules/nested/node_modules/a/package.json"]
    ) == {"name": "a", "version": "1.0.0"}


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
    manifest = _TargetedArtifactManifest(
        scan_subdir=".odc-targeted",
        expected_package_occurrences=(("a", "1.0.0"), ("b", "2.0.0")),
        occurrences_by_project={".odc-targeted/000": (occurrence_a, occurrence_b)},
        task_ids_by_project={".odc-targeted/000": ("task-1", "task-2")},
        input_paths_by_project={".odc-targeted/000": ("package-lock.json",)},
    )
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1", "task-2"],
        found_identifiers=["CVE-2026-10001", "CVE-2026-10002"],
        new_identifiers=["CVE-2026-10001", "CVE-2026-10002"],
        complete=True,
    )
    closure_a = SimpleNamespace(
        source_lockfile="package-lock.json",
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
        nodes=(
            SimpleNamespace(
                lockfile_key="node_modules/b",
                package_name="b",
                version="2.0.0",
            ),
        ),
    )
    batch = SimpleNamespace(
        covered_task_ids=("task-1", "task-2"),
        failures_by_task={},
        closures_by_task={"task-1": (closure_a,), "task-2": (closure_b,)},
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
    cleanup.assert_called_once_with(sandbox, ".odc-targeted")
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
