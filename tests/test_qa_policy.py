"""Focused tests for qa policy."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from remediation_engine.agents.qa_evaluation import (
    _LOG_QUERY_MAX_CHARS,
    GroupInvestigation,
    _apply_guardrails,
    _apply_policy_decision,
    _attach_scan_evidence_to_evaluations,
    _evaluate_policy_gates,
    _package_state_checks_enabled,
    _query_qa_logs,
    build_qa_review_toolbelt,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    CommandResult,
    DependencyEvidenceStatus,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    ODCScanEvidence,
    PackageScanOccurrence,
    QAAttemptResult,
    QADependencyEvidence,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    QASemanticSecurityReview,
    QATestAttribution,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    SecurityReviewVerdict,
    Severity,
    TaskAttemptSnapshot,
    TestAttributionVerdict,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.state import (
    build_initial_remediation_task,
    derive_missing_task_qa_policy,
)
from remediation_engine.tools.qa_scan import (
    QATaskContext,
    _QAExecutionResults,
    _QALogRecord,
    _QAPackageState,
    _SecurityScanResult,
    _validate_qa_path,
)
from remediation_engine.tools.qa_workspace import (
    _attempt_version_evidence,
    _build_qa_scan_targets,
    _collect_group_package_state,
    _dependency_tree_versions,
    _derive_qa_task_policies,
    _generate_workspace_diff,
)


def _group(group_id: str, identifier: str = "CVE-2025-0001") -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name="demo-package",
        cve_id=identifier,
        severity=Severity.HIGH,
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="demo-package",
        file_paths=["package.json"],
        cve_ids=[identifier],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="2.0.0",
            instruction="Upgrade demo-package to 2.0.0.",
            strategy_used="test",
        ),
    )


def _python_group(package: str, manifest: str, manager: str) -> VulnerabilityGroup:
    """Build one localized PyPI group for deterministic QA evidence tests."""
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name=package,
        purl=f"pkg:pypi/{package}@1.0.0",
        ecosystem="pypi",
        severity=Severity.HIGH,
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest,
        package_manager=manager,
        declaration_type="packages" if manager == "pipenv" else "requirements",
    )
    return VulnerabilityGroup(
        group_id=f"sca:{manifest}:{package}",
        issue_type=IssueType.SCA,
        vulnerable_component=package,
        file_paths=[manifest],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
    )


def _results(
    *,
    install: tuple[bool, str] = (True, "install ok"),
    remaining: set[str] | None = None,
    scanner_status: ScannerExecutionStatus = ScannerExecutionStatus.SUCCESS,
    tests: tuple[bool, str] | None = (True, "tests ok"),
    install_error_category: str | None = None,
) -> _QAExecutionResults:
    remaining = set(remaining or set())
    results = _QAExecutionResults(
        install=install,
        tests=tests,
        scan=_SecurityScanResult(
            ok=scanner_status == ScannerExecutionStatus.SUCCESS,
            summary="scanner",
            remaining_identifiers=remaining,
            found_identifiers=remaining,
            new_identifiers=set(),
            execution_status=scanner_status,
        ),
    )
    results.package_state_by_task["task-1"] = _QAPackageState(
        manifest_state="present",
        graph_state="present",
        dependency_evidence=QADependencyEvidence(
            status=DependencyEvidenceStatus.VERIFIED,
            target_package="demo-package",
            expected_version="2.0.0",
            manifest_paths=["package.json"],
            lockfile_paths=["package-lock.json"],
            declarations={"package.json#/dependencies/demo-package": "2.0.0"},
            resolved_versions=["2.0.0"],
            evidence_refs=[
                "package.json#/dependencies/demo-package",
                "package-lock.json#resolved/demo-package",
            ],
        ),
    )
    results.install_error_category = install_error_category
    return results


def _absence_evidence(task_id: str = "task-1") -> ODCScanEvidence:
    """Build complete task-owned evidence for an absent SCA target."""
    return ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=[task_id],
        scan_performed=False,
        scan_skip_reason="no_matching_target",
        complete=True,
    )


def _absence_results(
    *,
    install: tuple[bool, str] = (True, "install ok"),
    remaining: set[str] | None = None,
    tests: tuple[bool, str] | None = (True, "tests ok"),
    evidence: ODCScanEvidence | None = None,
) -> _QAExecutionResults:
    """Build execution results whose scanner work was skipped for a proven absent target."""
    results = _results(
        install=install,
        scanner_status=ScannerExecutionStatus.NOT_RUN,
        tests=tests,
    )
    remaining_ids = set(remaining or set())
    results.scan = _SecurityScanResult(
        ok=True,
        summary="Targeted ODC skipped because the target package was absent.",
        remaining_identifiers=remaining_ids,
        found_identifiers=remaining_ids,
        new_identifiers=set(),
        execution_status=ScannerExecutionStatus.NOT_RUN,
    )
    results.scan_evidence_by_task["task-1"] = evidence or _absence_evidence()
    results.scan_target_skipped_by_task["task-1"] = "no_matching_target"
    return results


def _semantic_review() -> QASemanticSecurityReview:
    return QASemanticSecurityReview(
        verdict=SecurityReviewVerdict.PASS,
        reasoning="The changed call site no longer reaches the vulnerable sink.",
        evidence_refs=["src/index.js:42", "search:vulnerable_api"],
    )


def _task_context(
    group: VulnerabilityGroup,
    task_id: str = "task-1",
    policy: QAPolicy = QAPolicy.VERSION_BUMP,
    strategy: RoutingStrategy = RoutingStrategy.VERSION_BUMP,
) -> QATaskContext:
    return QATaskContext(
        task=RemediationTask(
            task_id=task_id,
            parent_group_id=group.group_id,
            strategy=strategy,
            qa_policy=policy,
            current_attempt_id=f"{task_id}-attempt",
            instruction="Apply the remediation.",
        ),
        group=group,
    )


def test_scan_evidence_projection_is_isolated_by_task_id() -> None:
    """Project one scan's identifiers into independent task-owned evidence."""
    group_one = _group("group-1", identifier="CVE-2025-0001")
    group_two = _group("group-2", identifier="CVE-2025-0002")
    context_one = _task_context(group_one, task_id="task-1")
    context_two = _task_context(group_two, task_id="task-2")
    evaluations = {
        "task-1": QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Retry task one.",
        ),
        "task-2": QAEvaluation(
            task_id="task-2",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Retry task two.",
        ),
    }
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.FULL,
        effective_scope=ScanScope.FULL,
        covered_task_ids=["task-1", "task-2"],
        found_identifiers=["CVE-2025-0001", "CVE-2025-0002"],
        remaining_target_identifiers=["CVE-2025-0001", "CVE-2025-0002"],
        complete=True,
    )

    enriched = _attach_scan_evidence_to_evaluations(
        evaluations,
        evidence,
        [context_one, context_two],
    )

    assert enriched["task-1"].scan_evidence is not None
    assert enriched["task-1"].scan_evidence.covered_task_ids == ["task-1"]
    assert enriched["task-1"].scan_evidence.remaining_target_identifiers == ["CVE-2025-0001"]
    assert enriched["task-2"].scan_evidence is not None
    assert enriched["task-2"].scan_evidence.covered_task_ids == ["task-2"]
    assert enriched["task-2"].scan_evidence.remaining_target_identifiers == ["CVE-2025-0002"]


def test_task_specific_scan_evidence_preserves_new_ids_and_closure_metadata() -> None:
    group_one = _group("group-1", identifier="CVE-2025-0001")
    group_two = _group("group-2", identifier="CVE-2025-0002")
    context_one = _task_context(group_one, task_id="task-1")
    context_two = _task_context(group_two, task_id="task-2")
    evaluations = {
        "task-1": QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Retry task one.",
        ),
        "task-2": QAEvaluation(
            task_id="task-2",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Retry task two.",
        ),
    }
    shared_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1", "task-2"],
        complete=True,
    )
    evidence_by_task = {
        "task-1": ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            covered_task_ids=["task-1"],
            closure_package_names=["express", "body-parser"],
            closure_lockfile_keys=["node_modules/express", "node_modules/body-parser"],
            found_identifiers=["CVE-2025-0001", "GHSA-2026-ABCD-EFGH-IJKL"],
            new_identifiers=["GHSA-2026-ABCD-EFGH-IJKL"],
            remaining_target_identifiers=["CVE-2025-0001"],
            covered_occurrences=[
                PackageScanOccurrence(
                    source_input_path="package-lock.json",
                    package_name="body-parser",
                    version="2.2.0",
                    lockfile_key="node_modules/body-parser",
                    found_identifiers=["GHSA-2026-ABCD-EFGH-IJKL"],
                )
            ],
            complete=True,
        ),
        "task-2": ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            covered_task_ids=["task-2"],
            closure_package_names=["helmet"],
            closure_lockfile_keys=["node_modules/helmet"],
            found_identifiers=["CVE-2025-0002"],
            remaining_target_identifiers=["CVE-2025-0002"],
            complete=True,
        ),
    }

    enriched = _attach_scan_evidence_to_evaluations(
        evaluations,
        shared_evidence,
        [context_one, context_two],
        evidence_by_task=evidence_by_task,
    )

    task_one_evidence = enriched["task-1"].scan_evidence
    task_two_evidence = enriched["task-2"].scan_evidence
    assert task_one_evidence is not None
    assert task_one_evidence.found_identifiers == [
        "CVE-2025-0001",
        "GHSA-2026-ABCD-EFGH-IJKL",
    ]
    assert task_one_evidence.new_identifiers == ["GHSA-2026-ABCD-EFGH-IJKL"]
    assert task_one_evidence.closure_package_names == ["express", "body-parser"]
    assert task_two_evidence is not None
    assert task_two_evidence.closure_package_names == ["helmet"]
    assert task_two_evidence.found_identifiers == ["CVE-2025-0002"]


def _investigation() -> GroupInvestigation:
    return GroupInvestigation(
        group_id="g1",
        task_id="task-1",
        investigation_text="Reviewed the diff.\nStructured Review Verdict: PASS",
        tool_transcript="[TOOL: read_file_context]",
        review_tools_used=["read_file_context"],
        source_review_evidence=True,
        structured_review_verdict=SecurityReviewVerdict.PASS,
    )


def _evaluate(
    group: VulnerabilityGroup,
    policy: QAPolicy,
    results: _QAExecutionResults,
    evaluation: QAEvaluation,
    *,
    investigation: GroupInvestigation | None = None,
    package_state: _QAPackageState | None = None,
) -> QAEvaluation:
    context = _task_context(group, policy=policy)
    if package_state is not None:
        results.package_state_by_task[context.task_id] = package_state
    with (
        patch(
            "remediation_engine.agents.qa_evaluation._QA_PACKAGE_STATE_CHECKS_ENABLED",
            package_state is not None,
        ),
        patch(
            "remediation_engine.agents.qa_evaluation._QA_VERSION_BUMP_PACKAGE_STATE_CHECKS_ENABLED",
            package_state is not None,
        ),
    ):
        gates, _ = _evaluate_policy_gates(
            [context],
            results,
            {context.task_id: policy},
        )
        evaluations, _ = _apply_policy_decision(
            [context],
            [evaluation],
            gates,
            {context.task_id: policy},
            {context.task_id: investigation} if investigation is not None else None,
        )
    return evaluations[context.task_id]


def test_installed_tree_checks_are_scoped_to_version_bump_policy() -> None:
    """Version bumps verify the installed tree without enabling unrelated gates."""
    assert _package_state_checks_enabled(QAPolicy.VERSION_BUMP) is True
    assert _package_state_checks_enabled(QAPolicy.NO_FIX_PACKAGE_REMOVAL) is False


def test_version_bump_scanner_is_group_scoped() -> None:
    g1 = _group("g1", "CVE-2025-0001")
    g2 = _group("g2", "CVE-2025-0002")
    contexts = [_task_context(g1, "task-1"), _task_context(g2, "task-2")]
    results = _results(remaining={"CVE-2025-0002"})
    policies = {context.task_id: QAPolicy.VERSION_BUMP for context in contexts}
    gates, _ = _evaluate_policy_gates(contexts, results, policies)
    evaluations, _ = _apply_policy_decision(
        contexts,
        [
            QAEvaluation(task_id="task-1", passed=True),
            QAEvaluation(task_id="task-2", passed=True),
        ],
        gates,
        policies,
    )
    assert evaluations["task-1"].passed is True
    assert evaluations["task-2"].failure_category == FailureCategory.SECURITY_FLAG


def test_task_scan_projection_overrides_unrelated_global_identifier() -> None:
    """A report-wide occurrence cannot make a clean task fail its scanner gate."""
    group = _group("g1", "CVE-2025-0001")
    context = _task_context(group, "task-1")
    results = _results(remaining={"CVE-2025-0001"})
    results.scan_evidence_by_task["task-1"] = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        complete=True,
        found_identifiers=[],
        new_identifiers=[],
        remaining_target_identifiers=[],
    )

    gates, _ = _evaluate_policy_gates(
        [context],
        results,
        {context.task_id: QAPolicy.VERSION_BUMP},
    )

    assert gates["task-1"].target_scanner_cleared is True
    assert gates["task-1"].target_remaining_identifiers == []


def test_missing_task_scan_projection_fails_closed_instead_of_using_global_ids() -> None:
    """Aggregate scan identifiers cannot stand in for a missing task projection."""
    group = _group("g1", "CVE-2025-0001")
    context = _task_context(group, "task-1")
    results = _results(remaining={"CVE-2025-0001"})
    results.scan_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        complete=True,
        found_identifiers=["CVE-2025-0001"],
        remaining_target_identifiers=["CVE-2025-0001"],
    )

    gates, _ = _evaluate_policy_gates(
        [context],
        results,
        {context.task_id: QAPolicy.VERSION_BUMP},
    )

    assert gates["task-1"].target_scanner_cleared is False
    assert gates["task-1"].target_remaining_identifiers == []
    assert "Task-specific scanner occurrence coverage is incomplete." in gates["task-1"].diagnostics


def test_missing_target_skip_satisfies_policy_and_python_owned_qa() -> None:
    group = _group("g1")
    context = _task_context(group)
    results = _absence_results()
    gates, _ = _evaluate_policy_gates(
        [context],
        results,
        {context.task_id: QAPolicy.VERSION_BUMP},
    )

    assert gates[context.task_id].status == "pass"
    assert gates[context.task_id].scanner_execution_status == ScannerExecutionStatus.SUCCESS
    assert gates[context.task_id].target_scanner_cleared is True
    assert gates[context.task_id].target_package_absent is True
    assert any("target package absent" in item for item in gates[context.task_id].diagnostics)

    evaluations, _ = _apply_policy_decision(
        [context],
        [
            QAEvaluation(
                task_id=context.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Package-specific evidence cannot be collected.",
            )
        ],
        gates,
        {context.task_id: QAPolicy.VERSION_BUMP},
    )
    assert evaluations[context.task_id].passed is True


@pytest.mark.parametrize(
    ("evidence_updates", "has_task_failure"),
    [
        ({"complete": False}, False),
        ({"covered_task_ids": []}, False),
        ({"scan_skip_reason": None}, False),
        ({}, True),
    ],
    ids=["incomplete", "uncovered", "wrong-skip-reason", "task-failure"],
)
def test_target_absence_requires_complete_task_owned_evidence(
    evidence_updates: dict[str, object],
    has_task_failure: bool,
) -> None:
    context = _task_context(_group("g1"))
    evidence = _absence_evidence().model_copy(update=evidence_updates)
    results = _absence_results(evidence=evidence)
    if has_task_failure:
        results.scan_target_failures_by_task[context.task_id] = (
            ScanFallbackReason.INVALID_LOCKFILE,
            "A supported lockfile could not be resolved.",
        )

    gates, _ = _evaluate_policy_gates(
        [context],
        results,
        {context.task_id: QAPolicy.VERSION_BUMP},
    )

    assert gates[context.task_id].target_package_absent is False


@pytest.mark.parametrize(
    ("policy", "install", "tests"),
    [
        (QAPolicy.VERSION_BUMP, (False, "install failed"), (True, "tests ok")),
        (QAPolicy.VERSION_BUMP, (True, "install ok"), (False, "tests failed")),
        (QAPolicy.INITIAL_CODE_WORKAROUND, (True, "install ok"), (False, "tests failed")),
    ],
    ids=["install-failed", "version-bump-tests-failed", "hard-tests-failed"],
)
def test_target_absence_does_not_waive_install_or_required_tests(
    policy: QAPolicy,
    install: tuple[bool, str],
    tests: tuple[bool, str],
) -> None:
    context = _task_context(_group("g1"), policy=policy)
    results = _absence_results(install=install, tests=tests)
    gates, _ = _evaluate_policy_gates([context], results, {context.task_id: policy})

    evaluations, _ = _apply_policy_decision(
        [context],
        [QAEvaluation(task_id=context.task_id, passed=True)],
        gates,
        {context.task_id: policy},
    )

    assert evaluations[context.task_id].passed is False


def test_target_absence_does_not_clear_remaining_target_identifiers() -> None:
    context = _task_context(_group("g1"))
    results = _absence_results(remaining={"CVE-2025-0001"})
    policy = QAPolicy.VERSION_BUMP
    gates, _ = _evaluate_policy_gates([context], results, {context.task_id: policy})

    evaluations, _ = _apply_policy_decision(
        [context],
        [QAEvaluation(task_id=context.task_id, passed=True)],
        gates,
        {context.task_id: policy},
    )

    assert gates[context.task_id].target_package_absent is True
    assert gates[context.task_id].target_scanner_cleared is False
    assert evaluations[context.task_id].passed is False


def test_version_bump_ignores_compacted_manifest_false_negative() -> None:
    """Verified Python evidence overrides a judge blocked by compaction."""
    result = _evaluate(
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        _results(),
        QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback=(
                "Manifest and lockfile evidence is unavailable because the "
                "workspace tool output was compacted."
            ),
        ),
    )

    assert result.passed is True
    assert result.deterministic_gates is not None
    assert result.deterministic_gates.dependency_evidence is not None
    assert (
        result.deterministic_gates.dependency_evidence.status == DependencyEvidenceStatus.VERIFIED
    )


def test_version_bump_dependency_mismatch_remains_security_failure() -> None:
    """A deterministic target mismatch cannot be exonerated by the judge."""
    result = _evaluate(
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        _results(),
        QAEvaluation(task_id="task-1", passed=True),
        package_state=_QAPackageState(
            manifest_state="present",
            graph_state="present",
            dependency_evidence=QADependencyEvidence(
                status=DependencyEvidenceStatus.MISMATCH,
                target_package="demo-package",
                expected_version="2.0.0",
                manifest_paths=["package.json"],
                lockfile_paths=["package-lock.json"],
                declarations={"package.json#/dependencies/demo-package": "1.0.0"},
                resolved_versions=["1.0.0"],
            ),
        ),
    )

    assert result.passed is False
    assert result.failure_category == FailureCategory.SECURITY_FLAG
    assert "does not match" in (result.retry_feedback or "")


def test_version_bump_unavailable_dependency_evidence_is_inconclusive() -> None:
    """Unavailable deterministic evidence requests QA again without worker retry."""
    result = _evaluate(
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        _results(),
        QAEvaluation(task_id="task-1", passed=True),
        package_state=_QAPackageState(
            manifest_state="unknown",
            graph_state="unknown",
            dependency_evidence=QADependencyEvidence(
                status=DependencyEvidenceStatus.INCONCLUSIVE,
                target_package="demo-package",
                expected_version="2.0.0",
                diagnostics=["package.json was unavailable"],
            ),
        ),
    )

    assert result.passed is False
    assert result.evidence_inconclusive is True
    assert result.failure_category == FailureCategory.SECURITY_FLAG


def test_version_bump_incomplete_scan_evidence_is_inconclusive() -> None:
    """A successful empty fallback report must not clear a target."""
    results = _results()
    results.scan_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.FULL,
        complete=False,
        fallback_reason=ScanFallbackReason.INCOMPLETE_CLOSURE,
    )

    result = _evaluate(
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        results,
        QAEvaluation(task_id="task-1", passed=True),
    )

    assert result.passed is False
    assert result.evidence_inconclusive is True
    assert result.deterministic_gates is not None
    assert result.deterministic_gates.scan_evidence_complete is False
    assert result.deterministic_gates.target_scanner_cleared is None
    assert "ODC scan evidence was incomplete" in (result.retry_feedback or "")


def test_version_bump_collects_task_keyed_dependency_evidence() -> None:
    """Dependency evidence is parsed in Python before evaluator compaction."""
    sandbox = MagicMock()
    manifest_payload = '{"dependencies":{"demo-package":"^2.0.0"}}'
    lockfile_payload = (
        '{"lockfileVersion":3,"packages":{"":{"dependencies":'
        '{"demo-package":"^2.0.0"}},'
        '"node_modules/demo-package":{"version":"2.0.0"}}}'
    )
    sandbox.read_file.side_effect = lambda path: (
        lockfile_payload if str(path).endswith("package-lock.json") else manifest_payload
    )
    sandbox.run.return_value.stdout = (
        '{"name":"workspace","dependencies":{"demo-package":{"version":"2.0.0"}}}'
    )
    package_state = _collect_group_package_state(
        sandbox,
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        task=_task_context(_group("g1")).task,
        expected_version="2.0.0",
    )

    assert package_state.dependency_evidence is not None
    assert package_state.dependency_evidence.status == DependencyEvidenceStatus.VERIFIED
    assert package_state.dependency_evidence.target_package == "demo-package"
    assert package_state.dependency_evidence.resolved_versions == ["2.0.0"]
    assert package_state.dependency_evidence.lockfile_versions == ["2.0.0"]
    assert package_state.dependency_evidence.manifest_paths == ["package.json"]
    assert "--package-lock-only" not in sandbox.run.call_args.args[0]
    sandbox.run.assert_called_once()


def test_pypi_dependency_evidence_uses_venv_and_pep440_equivalence() -> None:
    group = _python_group("requests", "requirements.txt", "pip")
    task = _task_context(group).task.model_copy(
        update={"target_package_name": "requests", "selected_version": "2.32.0"}
    )
    sandbox = MagicMock()
    sandbox.read_file.return_value = "Requests==2.32.0\n"
    sandbox.run.return_value = MagicMock(
        exit_code=0,
        stdout="Name: requests\nVersion: 2.32\n",
        stderr="",
    )

    state = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.VERSION_BUMP,
        task=task,
        expected_version="2.32.0",
    )

    evidence = state.dependency_evidence
    assert evidence is not None
    assert evidence.status == DependencyEvidenceStatus.VERIFIED
    assert evidence.target_package == "requests"
    assert evidence.resolved_versions == ["2.32"]
    assert evidence.lockfile_paths == []
    command = sandbox.run.call_args.args[0]
    assert command == ".venv/bin/python -m pip show requests"
    assert "npm" not in command
    sandbox.read_file.assert_called_once_with("requirements.txt")


def test_pipfile_parent_lock_version_mismatch_is_reported() -> None:
    group = _python_group("urllib3", "Pipfile", "pipenv").model_copy(
        update={"file_paths": ["Pipfile.lock"]}
    )
    task = _task_context(group).task.model_copy(
        update={"target_package_name": "requests", "selected_version": "2.32.0"}
    )
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: {
        "Pipfile": '[packages]\nrequests = "*"\n',
        "Pipfile.lock": '{"default":{"requests":{"version":"==2.31.0"}},"develop":{}}',
    }.get(path)
    sandbox.run.return_value = MagicMock(
        exit_code=0,
        stdout="Name: requests\nVersion: 2.32.0\n",
        stderr="",
    )

    state = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.VERSION_BUMP,
        task=task,
        expected_version="2.32.0",
    )

    evidence = state.dependency_evidence
    assert evidence is not None
    assert evidence.status == DependencyEvidenceStatus.MISMATCH
    assert evidence.target_package == "requests"
    assert evidence.lockfile_versions == ["2.31.0"]
    assert sandbox.run.call_args.args[0] == ".venv/bin/python -m pip show requests"
    assert all("npm" not in call.args[0] for call in sandbox.run.call_args_list)


def test_missing_pipfile_lock_is_inconclusive_without_npm_fallback() -> None:
    group = _python_group("requests", "Pipfile", "pipenv")
    task = _task_context(group).task.model_copy(
        update={"target_package_name": "requests", "selected_version": "2.32.0"}
    )
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        '[packages]\\nrequests = "*"\\n' if path == "Pipfile" else None
    )
    sandbox.run.return_value = MagicMock(
        exit_code=0,
        stdout="Name: requests\\nVersion: 2.32.0\\n",
        stderr="",
    )

    state = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.VERSION_BUMP,
        task=task,
        expected_version="2.32.0",
    )

    evidence = state.dependency_evidence
    assert evidence is not None
    assert evidence.status == DependencyEvidenceStatus.INCONCLUSIVE
    assert evidence.lockfile_paths == ["Pipfile.lock"]
    assert any("Pipfile.lock was unavailable" in item for item in evidence.diagnostics)
    assert sandbox.run.call_args.args[0] == ".venv/bin/python -m pip show requests"


def test_python_no_fix_package_removal_uses_manifest_and_installed_state() -> None:
    group = _python_group("requests", "requirements.txt", "pip")
    task = _task_context(group, policy=QAPolicy.NO_FIX_PACKAGE_REMOVAL).task
    sandbox = MagicMock()
    sandbox.read_file.return_value = ""
    sandbox.run.return_value = MagicMock(
        exit_code=1,
        stdout="WARNING: Package(s) not found: requests",
        stderr="",
    )

    removed = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        task=task,
    )

    assert removed.manifest_state == "absent"
    assert removed.graph_state == "absent"
    assert removed.dependency_evidence is not None
    assert removed.dependency_evidence.status == DependencyEvidenceStatus.VERIFIED
    assert sandbox.run.call_args.args[0] == ".venv/bin/python -m pip show requests"


def test_python_no_fix_package_still_present_is_a_mismatch() -> None:
    group = _python_group("requests", "requirements.txt", "pip")
    task = _task_context(group, policy=QAPolicy.NO_FIX_PACKAGE_REMOVAL).task
    sandbox = MagicMock()
    sandbox.read_file.return_value = "requests==2.31.0\n"
    sandbox.run.return_value = MagicMock(
        exit_code=0,
        stdout="Name: requests\nVersion: 2.31.0\n",
        stderr="",
    )

    present = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        task=task,
    )

    assert present.manifest_state == "present"
    assert present.graph_state == "present"
    assert present.dependency_evidence is not None
    assert present.dependency_evidence.status == DependencyEvidenceStatus.MISMATCH


def test_version_bump_collects_deeply_nested_npm_dependency_evidence() -> None:
    """Nested npm dependency nodes contribute to resolved graph evidence."""
    nested_tree = {
        "name": "workspace",
        "dependencies": {
            "sqlite3": {
                "version": "5.1.7",
                "dependencies": {
                    "node-gyp": {
                        "version": "8.4.1",
                        "dependencies": {
                            "make-fetch-happen": {
                                "version": "9.1.0",
                                "dependencies": {
                                    "http-proxy-agent": {
                                        "version": "4.0.1",
                                        "dependencies": {
                                            "@tootallnate/once": {
                                                "version": "2.0.1",
                                                "overridden": True,
                                            }
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
    }

    assert _dependency_tree_versions(nested_tree, "@tootallnate/once") == {"2.0.1"}

    sandbox = MagicMock()
    manifest_payload = '{"overrides":{"@tootallnate/once":"2.0.1"}}'
    lockfile_payload = (
        '{"lockfileVersion":3,"packages":{"node_modules/@tootallnate/once":{"version":"2.0.1"}}}'
    )
    sandbox.read_file.side_effect = lambda path: (
        manifest_payload if str(path).endswith("package.json") else lockfile_payload
    )
    sandbox.run.return_value.exit_code = 0
    sandbox.run.return_value.stdout = json.dumps(nested_tree)
    group = _group("g1").model_copy(update={"vulnerable_component": "@tootallnate/once"})

    package_state = _collect_group_package_state(
        sandbox,
        group,
        QAPolicy.VERSION_BUMP,
        task=_task_context(group),
        expected_version="2.0.1",
    )

    evidence = package_state.dependency_evidence
    assert evidence is not None
    assert evidence.status == DependencyEvidenceStatus.VERIFIED
    assert evidence.resolved_versions == ["2.0.1"]
    assert evidence.lockfile_versions == ["2.0.1"]


def test_version_bump_missing_lockfile_is_inconclusive() -> None:
    """A missing lockfile is unavailable evidence, not a target mismatch."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        '{"dependencies":{"demo-package":"^2.0.0"}}' if str(path).endswith("package.json") else None
    )
    sandbox.run.return_value.stdout = (
        '{"name":"workspace","dependencies":{"demo-package":{"version":"2.0.0"}}}'
    )

    package_state = _collect_group_package_state(
        sandbox,
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        task=_task_context(_group("g1")).task,
        expected_version="2.0.0",
    )

    assert package_state.dependency_evidence is not None
    assert package_state.dependency_evidence.status == DependencyEvidenceStatus.INCONCLUSIVE
    assert any("lockfile" in diagnostic.lower() for diagnostic in package_state.diagnostics)


def test_version_bump_rejects_mixed_resolved_dependency_versions() -> None:
    """A target is not verified when npm resolves stale and selected copies."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        '{"dependencies":{"demo-package":"^2.0.0"}}'
        if str(path).endswith("package.json")
        else (
            '{"lockfileVersion":3,"packages":'
            '{"node_modules/demo-package":{"version":"2.0.0"},'
            '"node_modules/parent/node_modules/demo-package":{"version":"1.0.0"}}}'
        )
    )
    sandbox.run.return_value.stdout = (
        '{"name":"workspace","dependencies":{"demo-package":{"version":"2.0.0",'
        '"dependencies":{"demo-package":{"version":"1.0.0"}}}}}'
    )

    package_state = _collect_group_package_state(
        sandbox,
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        task=_task_context(_group("g1")).task,
        expected_version="2.0.0",
    )

    assert package_state.dependency_evidence is not None
    evidence = package_state.dependency_evidence
    assert evidence.status == DependencyEvidenceStatus.MISMATCH
    assert evidence.resolved_versions == ["1.0.0", "2.0.0"]
    assert evidence.lockfile_versions == ["1.0.0", "2.0.0"]


def test_version_bump_treats_failed_dependency_graph_as_inconclusive() -> None:
    """A nonzero npm ls result cannot be verified from its partial stdout."""
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        '{"dependencies":{"demo-package":"^2.0.0"}}'
        if str(path).endswith("package.json")
        else ('{"lockfileVersion":3,"packages":{"node_modules/demo-package":{"version":"2.0.0"}}}')
    )
    sandbox.run.return_value.exit_code = 1
    sandbox.run.return_value.stdout = (
        '{"name":"workspace","dependencies":{"demo-package":{"version":"2.0.0"}}}'
    )

    package_state = _collect_group_package_state(
        sandbox,
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        task=_task_context(_group("g1")).task,
        expected_version="2.0.0",
    )

    assert package_state.dependency_evidence is not None
    assert package_state.dependency_evidence.status == DependencyEvidenceStatus.INCONCLUSIVE
    assert any("exit code 1" in diagnostic for diagnostic in package_state.diagnostics)


def test_attempt_version_evidence_prefers_allowed_effective_candidate() -> None:
    """QA follows the version the worker actually executed when authorized."""
    group = _group("g1")
    task = _task_context(group).task.model_copy(update={"selected_version": "1.0.0"})
    snapshot = TaskAttemptSnapshot(
        attempt_id=task.current_attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        selected_version="1.0.0",
        allowed_target_versions=["1.0.0", "2.0.0"],
        instruction="Update demo-package.",
        instruction_digest="digest",
        dispatch_node="update_subagent",
    )
    worker_result = WorkerAttemptResult(
        attempt_id=task.current_attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["2.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            executed_versions=["2.0.0"],
            effective_target_version="2.0.0",
        ),
        instruction_digest="digest",
    )

    version, has_evidence = _attempt_version_evidence(
        {
            "attempt_snapshots_by_id": {task.current_attempt_id: snapshot},
            "worker_results_by_attempt": {task.current_attempt_id: worker_result},
        },
        task,
    )

    assert version == "2.0.0"
    assert has_evidence is True


def test_qa_scan_target_uses_authorized_attempt_version_over_task_hint() -> None:
    """Current execution evidence is authoritative over an older task hint."""
    group = _group("g1")
    task = _task_context(group).task.model_copy(update={"selected_version": "1.0.0"})
    snapshot = TaskAttemptSnapshot(
        attempt_id=task.current_attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        selected_version="1.0.0",
        allowed_target_versions=["1.0.0", "2.0.0"],
        instruction="Update demo-package.",
        instruction_digest="digest",
        dispatch_node="update_subagent",
    )
    worker_result = WorkerAttemptResult(
        attempt_id=task.current_attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["2.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            executed_versions=["2.0.0"],
            effective_target_version="2.0.0",
        ),
        instruction_digest="digest",
    )
    state = {
        "active_target_task_ids": [task.task_id],
        "task_queue": {task.task_id: task},
        "attempt_snapshots_by_id": {task.current_attempt_id: snapshot},
        "worker_results_by_attempt": {task.current_attempt_id: worker_result},
    }

    target = _build_qa_scan_targets(state, [group])[0]

    assert target.expected_version == "2.0.0"

    invalid_result = worker_result.model_copy(
        update={
            "executed_versions": ["9.9.9"],
            "execution_diagnostics": WorkerExecutionDiagnostics(
                executed_versions=["9.9.9"],
                effective_target_version="9.9.9",
            ),
        }
    )
    invalid_state = {
        **state,
        "worker_results_by_attempt": {task.current_attempt_id: invalid_result},
    }
    invalid_target = _build_qa_scan_targets(invalid_state, [group])[0]

    assert invalid_target.expected_version is None


def test_attempt_version_evidence_rejects_foreign_snapshot() -> None:
    """A snapshot keyed by an attempt cannot override its task provenance."""
    group = _group("g1")
    task = _task_context(group).task
    snapshot = TaskAttemptSnapshot(
        attempt_id=task.current_attempt_id,
        task_id="other-task",
        task_revision=task.task_revision,
        selected_version="9.9.9",
        instruction="Update demo-package.",
        instruction_digest="digest",
        dispatch_node="update_subagent",
    )

    version, has_evidence = _attempt_version_evidence(
        {"attempt_snapshots_by_id": {task.current_attempt_id: snapshot}},
        task,
    )

    assert version is None
    assert has_evidence is True


def test_version_bump_may_exonerate_unrelated_test_failure() -> None:
    group = _group("g1")
    unrelated_group = _group("g2", "CVE-2025-0002")
    contexts = [_task_context(group, "task-1"), _task_context(unrelated_group, "task-2")]
    results = _results(tests=(False, "test failed"))
    results.scan_evidence_by_task["task-1"] = _absence_evidence()
    results.scan_target_skipped_by_task["task-1"] = "no_matching_target"
    policies = {context.task_id: QAPolicy.VERSION_BUMP for context in contexts}
    gates, _ = _evaluate_policy_gates(contexts, results, policies)
    evaluations, _ = _apply_policy_decision(
        contexts,
        [
            QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.BREAKING_CHANGE,
                retry_feedback="The other task owns the failure.",
                test_attribution=QATestAttribution(
                    verdict=TestAttributionVerdict.EXONERATED,
                    responsible_group_ids=["g2"],
                    failed_tests=["tests/other.test.js::fails"],
                    reasoning="The failure does not touch this diff.",
                ),
            ),
            QAEvaluation(task_id="task-2", passed=True),
        ],
        gates,
        policies,
    )
    assert evaluations["task-1"].passed is True
    assert (
        evaluations["task-1"].deterministic_gates is not None
        and evaluations["task-1"].deterministic_gates.target_package_absent is True
    )


def test_version_bump_exonerates_only_the_known_pastebin_test_failure() -> None:
    group = _group("g1")
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The PasteBin Internet resource was unavailable.",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["expected 200, received 503 from https://pastebin.com/example"],
            failed_tests=[
                "for default configuration (https://pastebin.com/example)",
                'PasteBin paste for "Leaked Unsafe Product" challenge available',
                "Required Internet resource",
            ],
            source_locations=[
                "test/api/internet-resources.test.ts:2:1060",
                "test/api/internet-resources.test.ts:14:14",
            ],
            affected_files=["test/api/internet-resources.test.ts"],
        ),
    )

    result = _evaluate(
        group,
        QAPolicy.VERSION_BUMP,
        _results(tests=(False, "The external PasteBin test failed.")),
        evaluation,
    )

    assert result.passed is True


def test_version_bump_does_not_exonerate_other_tests_with_pastebin_failure() -> None:
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="PasteBin and an application test failed.",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["expected 200, received 503 from https://pastebin.com/example"],
            failed_tests=[
                "PasteBin paste for a security challenge was unavailable",
                "application behavior changed",
            ],
            source_locations=[
                "test/api/internet-resources.test.ts:2:1060",
                "test/api/auth.test.ts:55:10",
            ],
            affected_files=["test/api/internet-resources.test.ts", "test/api/auth.test.ts"],
        ),
    )

    result = _evaluate(
        _group("g1"),
        QAPolicy.VERSION_BUMP,
        _results(tests=(False, "PasteBin and an application test failed.")),
        evaluation,
    )

    assert result.passed is False


def test_hard_test_policy_ignores_llm_exoneration() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(tests=(False, "test failed")),
        QAEvaluation(
            task_id="task-1",
            passed=True,
            semantic_security_review=_semantic_review(),
            test_attribution=QATestAttribution(
                verdict=TestAttributionVerdict.EXONERATED,
                responsible_group_ids=["g2"],
                failed_tests=["tests/other.test.js::fails"],
                reasoning="The other task owns the failure.",
            ),
        ),
        investigation=_investigation(),
    )
    assert result.passed is False
    assert result.failure_category == FailureCategory.BREAKING_CHANGE


def test_required_semantic_review_requires_source_evidence() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(remaining={"CVE-2025-0001"}),
        QAEvaluation(
            task_id="task-1",
            passed=True,
            semantic_security_review=_semantic_review(),
        ),
        investigation=_investigation(),
    )
    assert result.passed is True
    assert result.deterministic_gates is not None
    assert result.deterministic_gates.target_scanner_cleared is False


def test_nonblocking_scanner_failure_can_pass_with_semantic_review() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        _results(scanner_status=ScannerExecutionStatus.TIMEOUT),
        QAEvaluation(
            task_id="task-1",
            passed=True,
            semantic_security_review=_semantic_review(),
        ),
        investigation=_investigation(),
    )
    assert result.passed is True


def test_no_fix_package_removal_requires_manifest_and_graph_absence() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        _results(),
        QAEvaluation(task_id="task-1", passed=True),
        package_state=_QAPackageState(manifest_state="absent", graph_state="absent"),
    )
    assert result.passed is True


def test_no_fix_package_state_fails_closed_for_unsupported_manager() -> None:
    group = _group("g1").model_copy(
        update={
            "localized_issues": [
                LocalizedIssue(
                    issue=_group("localized").issues[0],
                    manifest_file="package.json",
                    package_manager="pnpm",
                )
            ]
        }
    )
    package_state = _collect_group_package_state(
        MagicMock(), group, QAPolicy.NO_FIX_PACKAGE_REMOVAL
    )
    assert package_state.manifest_state == "unknown"
    assert package_state.graph_state == "unknown"
    assert "pnpm" in package_state.diagnostics[0]


def test_no_fix_code_removal_requires_package_in_manifest_and_graph() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.NO_FIX_CODE_REMOVAL,
        _results(remaining={"CVE-2025-0001"}),
        QAEvaluation(
            task_id="task-1",
            passed=True,
            semantic_security_review=_semantic_review(),
        ),
        investigation=_investigation(),
        package_state=_QAPackageState(manifest_state="present", graph_state="present"),
    )
    assert result.passed is True


def test_shared_install_failure_fails_every_task() -> None:
    groups = [_group("g1", "CVE-2025-0001"), _group("g2", "CVE-2025-0002")]
    contexts = [_task_context(group, f"task-{index}") for index, group in enumerate(groups, 1)]
    policies = {context.task_id: QAPolicy.VERSION_BUMP for context in contexts}
    results = _results(install=(False, "npm install ERESOLVE peer conflict"))
    gates, _ = _evaluate_policy_gates(contexts, results, policies)
    evaluations, _ = _apply_policy_decision(
        contexts,
        [QAEvaluation(task_id=context.task_id, passed=True) for context in contexts],
        gates,
        policies,
    )
    assert all(not evaluation.passed for evaluation in evaluations.values())
    assert all(
        evaluation.failure_category == FailureCategory.PEER_CONFLICT
        for evaluation in evaluations.values()
    )


@pytest.mark.parametrize(
    "policy",
    [
        QAPolicy.VERSION_BUMP,
        QAPolicy.INITIAL_CODE_WORKAROUND,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    ],
)
def test_dependency_install_conflict_is_terminal(policy: QAPolicy) -> None:
    """A dependency conflict wins over post-install gates that could not run."""
    group = _group("g1")
    context = _task_context(group, policy=policy)
    results = _results(
        install=(False, "npm install EOVERRIDE peer conflict"),
        scanner_status=ScannerExecutionStatus.UNPARSEABLE,
        tests=None,
        install_error_category="PEER_CONFLICT",
    )
    evaluations, errors = _apply_guardrails(
        task_contexts=[context],
        batch_result=[
            QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="The evaluator reported a generic failure.",
            )
        ],
        results=results,
        task_policies={"task-1": policy},
    )

    evaluation = evaluations["task-1"]
    assert not errors
    assert evaluation.passed is False
    assert evaluation.failure_category == FailureCategory.PEER_CONFLICT
    assert evaluation.semantic_security_review is None
    assert evaluation.test_attribution is None
    assert "resolve the dependency conflict" in (evaluation.retry_feedback or "")
    assert "scanner" not in (evaluation.retry_feedback or "").lower()
    assert "tests" not in (evaluation.retry_feedback or "").lower()


def test_missing_initial_task_policy_is_recoverable_deterministically() -> None:
    group = _group("g1")
    task = build_initial_remediation_task(group, "task-1").model_copy(update={"qa_policy": None})

    assert derive_missing_task_qa_policy(task, group) == QAPolicy.VERSION_BUMP


def test_missing_no_fix_policy_recovers_package_removal_stage() -> None:
    group = _group("g1").model_copy(
        update={
            "fix_plan": FixPlan(
                status=FixPlanStatus.NO_FIX,
                instruction="Remove the vulnerable package.",
                strategy_used="test",
            )
        }
    )
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Remove the vulnerable package.",
    )

    assert derive_missing_task_qa_policy(task, group) == QAPolicy.NO_FIX_PACKAGE_REMOVAL


def test_missing_pivot_policy_is_not_guessed() -> None:
    group = _group("g1")
    task = RemediationTask(
        task_id="task-child",
        parent_group_id=group.group_id,
        parent_task_id="task-parent",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply workaround.",
    )

    assert derive_missing_task_qa_policy(task, group) is None


def test_active_qa_policy_must_match_attempt_snapshot() -> None:
    group = _group("g1")
    task = build_initial_remediation_task(group, "task-1").model_copy(
        update={"task_revision": 1, "current_attempt_id": "attempt-1"}
    )

    policies = _derive_qa_task_policies(
        [group],
        {task.task_id: task},
        [task.task_id],
        {"attempt-1": task.model_copy(update={"current_attempt_id": None, "qa_policy": None})},
    )

    assert policies["task-1"] is None


def test_qa_attempt_result_requires_consistent_policy_provenance() -> None:
    evaluation = QAEvaluation(task_id="g1", passed=True)

    with pytest.raises(ValidationError, match="qa_policy_source"):
        QAAttemptResult(
            attempt_id="attempt-1",
            task_id="task-1",
            qa_policy=QAPolicy.VERSION_BUMP,
            evaluation=evaluation,
        )

    with pytest.raises(ValidationError, match="qa_policy_source"):
        QAAttemptResult(
            attempt_id="attempt-1",
            task_id="task-1",
            qa_policy_source="task_queue",
            evaluation=evaluation,
        )


class TestGenerateWorkspaceDiff:
    def test_detects_modified_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/agents/qa.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: "const x = 2;\n" if path == rel_path else None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert rel_path in changed
        assert rel_path in diff_text

    def test_detects_deleted_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/agents/qa.py"

        sandbox = MagicMock()
        # read_file returns None â†’ file deleted in workspace
        sandbox.read_file.return_value = None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])

        assert rel_path in changed
        assert "deleted" in diff_text.lower()

    def test_ignores_node_modules(self):
        tmp_path = Path.cwd()

        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        _, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["node_modules/evil.js"])

        assert not any("node_modules" in f for f in changed)

    def test_diff_text_is_capped(self, tmp_path):
        from remediation_engine.tools.qa_workspace import _DIFF_CHAR_BUDGET

        rel_path = "src/remediation_engine/agents/qa.py"

        big_content = "x" * (_DIFF_CHAR_BUDGET * 3)
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            big_content + "\nextra line\n" if path == rel_path else None
        )
        diff_text, _ = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert len(diff_text) <= _DIFF_CHAR_BUDGET + len("\n... (diff truncated)")

    def test_optimized_changed_files_path(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/agents/qa.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const x = 2;\n" if path == rel_path else "const y = 2;\n"
        )

        diff_text, changed = _generate_workspace_diff(
            str(tmp_path), sandbox, candidate_changed_files=[rel_path]
        )

        assert rel_path in changed
        assert "other.js" not in changed
        assert rel_path in diff_text
        assert "other.js" not in diff_text

    def test_empty_candidates_returns_empty_diff_note(self, tmp_path):
        """When no candidate files are given, diff returns an empty-diff note."""
        sandbox = MagicMock()
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [])
        assert changed == []
        assert "empty" in diff_text.lower() or "no changed files" in diff_text.lower()
        sandbox.read_file.assert_not_called()

    def test_blocked_candidate_path_is_reported_explicitly(self, tmp_path):
        """Traversal candidates are errors, not unchanged-file observations."""
        sandbox = MagicMock()

        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["../outside.js"])

        assert changed == []
        assert "ERROR: blocked candidate path" in diff_text
        sandbox.read_file.assert_not_called()


class TestQAReviewToolSafety:
    """Verify read_file_context rejects dangerous paths."""

    def _get_read_file_tool(self, sandbox=None):
        if sandbox is None:
            sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root=None,
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        for t in tools:
            if t.name == "read_file_context":
                return t
        raise KeyError("read_file_context not found")

    def test_rejects_absolute_path(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "/etc/passwd"})
        assert "ERROR" in result
        assert "absolute" in result.lower()

    def test_rejects_path_traversal(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "../../etc/passwd"})
        assert "ERROR" in result

    def test_returns_file_content_for_valid_path(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "const x = 1;\n"
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "src/app.js"})
        assert "const x = 1;" in result
        sandbox.read_file.assert_called_once_with("src/app.js")

    def test_returns_error_when_file_not_found(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "nonexistent.js"})
        assert "ERROR" in result
        assert "not found" in result.lower()


class TestValidateQaPath:
    """Unit tests for the _validate_qa_path helper."""

    def test_accepts_relative_path(self):
        assert _validate_qa_path("src/app.js") == "src/app.js"

    def test_accepts_nested_relative_path(self):
        assert _validate_qa_path("a/b/c.ts") == "a/b/c.ts"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="required"):
            _validate_qa_path("")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="absolute"):
            _validate_qa_path("/etc/passwd")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_qa_path("../../secret")

    def test_normalizes_backslashes(self):
        assert _validate_qa_path("src\\app.js") == "src/app.js"


class TestQADiffToolNoCandidates:
    """Verify generate_workspace_diff tool returns an informative note when no candidates exist."""

    def test_empty_candidates_returns_informative_note(self):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root="/tmp/repo",
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        # Should mention no changed files, not crash
        assert "empty" in result.lower() or "no changed files" in result.lower()
        sandbox.read_file.assert_not_called()

    def test_no_host_repo_root_returns_error(self):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.js"],
            host_repo_root=None,
            results=results,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        assert "ERROR" in result


class TestQueryQaLogs:
    """Verify query_qa_logs returns cached data and handles missing runs correctly."""

    def _get_query_tool(self, prepopulate: bool = False):
        sandbox = MagicMock()
        results = _QAExecutionResults()
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=[],
            host_repo_root=None,
            results=results,
        )
        if prepopulate:
            results.install = (True, "install ready")
            results.scan = _SecurityScanResult(True, "scan ready", set(), set(), set())
            results.tests = (True, "tests ready")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        return tool, results

    def test_returns_error_when_install_not_run(self):
        tool, _ = self._get_query_tool()
        result = tool.invoke({"log_type": "install"})
        assert "ERROR" in result

    def test_returns_cached_install_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.install = (True, "npm install output here")
        result = tool.invoke({"log_type": "install"})
        assert "npm install output here" in result

    def test_returns_cached_scan_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.scan = _SecurityScanResult(True, "scan output here", set(), set(), set())
        result = tool.invoke({"log_type": "scan"})
        assert "scan output here" in result

    def test_returns_cached_tests_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.tests = (True, "test output here")
        result = tool.invoke({"log_type": "tests"})
        assert "test output here" in result

    def test_invalid_log_type_returns_error(self):
        tool, _ = self._get_query_tool(prepopulate=True)
        result = tool.invoke({"log_type": "unknown"})
        assert "ERROR" in result


class TestQAToolsNotInSubagentToolbelts:
    """
    Verify that the heavy QA commands are not present in the update or
    workaround toolbelts that are given to subagents.

    This duplicates the guards in test_remedy_tools.py but is kept here for
    documentation purposes and test isolation.
    """

    def test_run_dependency_install_not_in_update_toolbelt(self):
        from remediation_engine.tools.workers import build_update_toolbelt

        sandbox = MagicMock()
        tools = build_update_toolbelt(
            sandbox,
            touched_files=set(),
            target_manifest_paths=["package.json"],
            package_manifest_paths={"lodash": ["package.json"]},
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names

    def test_run_dependency_install_not_in_workaround_toolbelt(self):
        from remediation_engine.tools.workers import build_workaround_toolbelt

        sandbox = MagicMock()
        tools = build_workaround_toolbelt(
            sandbox,
            touched_files=set(),
            host_repo_root=Path("/repo"),
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names


class TestBuildQaReviewToolbelt:
    def _build(self, prepopulate=True):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "file content"
        results = _QAExecutionResults()
        if prepopulate:
            results.install = (True, "ok")
            results.scan = _SecurityScanResult(True, "ok", set(), set(), set())
            results.tests = (True, "ok")
        return build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.ts"],
            host_repo_root="/tmp/repo",
            results=results,
        ), results

    def test_no_execution_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        assert "run_dependency_install" not in names
        assert "run_security_scan" not in names
        assert "run_unit_tests" not in names

    def test_all_six_review_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        for n in [
            "list_changed_files",
            "generate_workspace_diff",
            "read_file_context",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "query_qa_logs",
        ]:
            assert n in names

    def test_review_tools_locked_when_results_empty(self):
        tools, _ = self._build(prepopulate=False)
        tool = next(t for t in tools if t.name == "list_changed_files")
        result = tool.invoke({})
        assert "ERROR" in result

    def test_list_changed_files_works_when_populated(self):
        tools, _ = self._build(prepopulate=True)
        tool = next(t for t in tools if t.name == "list_changed_files")
        assert "src/app.ts" in tool.invoke({})

    def test_query_qa_logs_returns_install_log(self):
        tools, results = self._build(prepopulate=True)
        results.install = (True, "npm install output here")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        assert "npm install output here" in tool.invoke({"log_type": "install"})

    def test_query_qa_logs_returns_condensed_test_summary(self):
        tools, results = self._build(prepopulate=True)
        results.tests = (
            False,
            "npm test FAILED (exit 1).\nDetected Failures: 1\n\nFailing Tests:\n1. jwt challenge\nAssertionError: boom",
        )
        tool = next(t for t in tools if t.name == "query_qa_logs")
        result = tool.invoke({"log_type": "tests"})
        assert "Detected Failures: 1" in result
        assert "jwt challenge" in result

    def test_query_qa_logs_supports_tail_errors_and_filter_views(self):
        tools, results = self._build(prepopulate=True)
        record = _QALogRecord(
            phase="install",
            label="npm install",
            exit_code=1,
            stdout="first stdout line\nsecond stdout line",
            stderr="ERESOLVE peer conflict\nsecondary diagnostic",
            error="install failed",
        )
        results.install = (False, "install failed")
        results.log_records["install"] = (record,)
        tool = next(t for t in tools if t.name == "query_qa_logs")

        tail = tool.invoke({"log_type": "install", "view": "tail", "tail_lines": 1})
        errors = tool.invoke({"log_type": "install", "view": "errors"})
        filtered = tool.invoke(
            {
                "log_type": "install",
                "view": "filter",
                "filter_pattern": "secondary diagnostic",
            }
        )
        full = tool.invoke({"log_type": "install", "view": "full"})

        assert "[npm install stdout]" in full
        assert "exit_code=1" in full
        assert "ERESOLVE peer conflict" in full
        assert "secondary diagnostic" in tail
        assert "first stdout line" not in tail
        assert "ERESOLVE peer conflict" in errors
        assert "install failed" in errors
        assert "secondary diagnostic" in filtered
        assert "first stdout line" not in filtered

    def test_query_qa_logs_caps_full_output_and_validates_view(self):
        results = _QAExecutionResults(
            install=(True, "install ok"),
            log_records={
                "install": (
                    _QALogRecord(
                        phase="install",
                        label="npm install",
                        exit_code=0,
                        stdout="x" * (_LOG_QUERY_MAX_CHARS + 100),
                        stderr="",
                    ),
                )
            },
        )

        capped = _query_qa_logs(results, "install", view="full")
        invalid = _query_qa_logs(results, "install", view="unknown")
        missing = _query_qa_logs(_QAExecutionResults(), "install", view="tail")

        assert len(capped) <= _LOG_QUERY_MAX_CHARS
        assert capped.endswith("(log output truncated)")
        assert invalid.startswith("ERROR: view")
        assert missing.startswith("ERROR: run_dependency_install")

    def test_query_qa_logs_rejects_invalid_filter_regex(self):
        tools, _ = self._build(prepopulate=True)
        tool = next(t for t in tools if t.name == "query_qa_logs")

        result = tool.invoke(
            {
                "log_type": "install",
                "view": "filter",
                "filter_pattern": "[unterminated",
            }
        )

        assert result.startswith("ERROR:")

    def test_search_codebase_pattern_forwards_shared_argument_contract(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="src/app.ts:1:foo\n",
            stderr="",
            duration_seconds=0.1,
        )
        results = _QAExecutionResults(
            install=(True, "install ok"),
            scan=(True, "scan ok", set()),
            tests=(True, "tests ok"),
        )
        tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.ts"],
            host_repo_root="/tmp/repo",
            results=results,
        )
        search_tool = next(t for t in tools if t.name == "search_codebase_pattern")

        assert (
            search_tool.invoke({"search_pattern": "foo", "target_directory": "src"})
            == "src/app.ts:1:foo"
        )
        command = sandbox.run.call_args.args[0]
        assert "'foo'" in command
        assert "'src'" in command
        assert set(search_tool.args) == {"search_pattern", "target_directory"}
