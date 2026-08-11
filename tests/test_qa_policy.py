"""Focused tests for the strategy-aware deterministic QA policy matrix."""

from __future__ import annotations

from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import (
    BatchQAResult,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    QAEvaluation,
    QAPolicy,
    QASemanticSecurityReview,
    QATestAttribution,
    ScannerExecutionStatus,
    SecurityReviewVerdict,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.schemas import TestAttributionVerdict as AttributionVerdict
from remediation_engine.orchestration.qa_critic import (
    GroupInvestigation,
    _apply_policy_decision,
    _collect_group_package_state,
    _evaluate_policy_gates,
    _QAExecutionResults,
    _QAPackageState,
    _SecurityScanResult,
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


def _results(
    *,
    install: tuple[bool, str] = (True, "install ok"),
    remaining: set[str] | None = None,
    scanner_status: ScannerExecutionStatus = ScannerExecutionStatus.SUCCESS,
    tests: tuple[bool, str] = (True, "tests ok"),
) -> _QAExecutionResults:
    result = _QAExecutionResults(
        install=install,
        tests=tests,
        scan=_SecurityScanResult(
            ok=scanner_status == ScannerExecutionStatus.SUCCESS,
            summary="scanner",
            remaining_identifiers=set(remaining or set()),
            found_identifiers=set(remaining or set()),
            new_identifiers=set(),
            execution_status=scanner_status,
        ),
    )
    return result


def _semantic() -> QASemanticSecurityReview:
    return QASemanticSecurityReview(
        verdict=SecurityReviewVerdict.PASS,
        reasoning="The changed call site no longer reaches the vulnerable sink.",
        evidence_refs=["src/index.js:42", "search:vulnerable_api", "CVE-2025-0001"],
    )


def _investigation() -> GroupInvestigation:
    return GroupInvestigation(
        group_id="g1",
        investigation_text="Reviewed the diff and affected call site.\nVerdict: PASS",
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
    if package_state is not None:
        results.package_state_by_group[group.group_id] = package_state
    groups = [group]
    if evaluation.test_attribution is not None:
        groups.extend(
            _group(identifier)
            for identifier in evaluation.test_attribution.responsible_group_ids
            if identifier != group.group_id
        )
    policies = {candidate.group_id: policy for candidate in groups}
    gates, _ = _evaluate_policy_gates(groups, results, policies)
    evaluations, _ = _apply_policy_decision(
        groups,
        BatchQAResult(holistic_report="test", evaluations=[evaluation]),
        gates,
        policies,
        {group.group_id: investigation} if investigation is not None else None,
    )
    return evaluations[group.group_id]


def test_version_bump_scanner_is_group_scoped():
    g1 = _group("g1", "CVE-2025-0001")
    g2 = _group("g2", "CVE-2025-0002")
    results = _results(remaining={"CVE-2025-0002"})
    gates, _ = _evaluate_policy_gates(
        [g1, g2], results, {"g1": QAPolicy.VERSION_BUMP, "g2": QAPolicy.VERSION_BUMP}
    )
    evaluations, _ = _apply_policy_decision(
        [g1, g2],
        BatchQAResult(
            holistic_report="test",
            evaluations=[
                QAEvaluation(task_id="g1", passed=True),
                QAEvaluation(task_id="g2", passed=True),
            ],
        ),
        gates,
        {"g1": QAPolicy.VERSION_BUMP, "g2": QAPolicy.VERSION_BUMP},
    )
    assert evaluations["g1"].passed is True
    assert evaluations["g2"].failure_category == FailureCategory.SECURITY_FLAG


def test_version_bump_may_exonerate_unrelated_test_failure():
    group = _group("g1")
    evaluation = QAEvaluation(
        task_id="g1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The other group owns the failure.",
        test_attribution=QATestAttribution(
            verdict=AttributionVerdict.EXONERATED,
            responsible_group_ids=["g2"],
            failed_tests=["tests/other.test.js::fails"],
            reasoning="The failure names the other package and does not touch this diff.",
        ),
    )
    result = _evaluate(
        group,
        QAPolicy.VERSION_BUMP,
        _results(tests=(False, "test failed")),
        evaluation,
    )
    assert result.passed is True


def test_hard_test_policies_ignore_llm_exoneration():
    group = _group("g1")
    evaluation = QAEvaluation(
        task_id="g1",
        passed=True,
        semantic_security_review=_semantic(),
        test_attribution=QATestAttribution(
            verdict=AttributionVerdict.EXONERATED,
            responsible_group_ids=["g2"],
            failed_tests=["tests/other.test.js::fails"],
            reasoning="The other group owns the failure.",
        ),
    )
    result = _evaluate(
        group,
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(tests=(False, "test failed")),
        evaluation,
        investigation=_investigation(),
    )
    assert result.passed is False
    assert result.failure_category == FailureCategory.BREAKING_CHANGE


def test_required_semantic_review_is_evidence_backed():
    group = _group("g1")
    evaluation = QAEvaluation(
        task_id="g1",
        passed=True,
        semantic_security_review=_semantic(),
    )
    result = _evaluate(
        group,
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(remaining={"CVE-2025-0001"}),
        evaluation,
        investigation=_investigation(),
    )
    assert result.passed is True
    assert result.deterministic_gates is not None
    assert result.deterministic_gates.target_scanner_cleared is False


def test_missing_semantic_review_fails_nonblocking_workaround_policy():
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        _results(remaining={"CVE-2025-0001"}),
        QAEvaluation(task_id="g1", passed=True),
        investigation=_investigation(),
    )
    assert result.failure_category == FailureCategory.SECURITY_FLAG


def test_semantic_pass_without_structured_investigator_verdict_fails_closed():
    group = _group("g1")
    investigation = _investigation()
    investigation.structured_review_verdict = None
    result = _evaluate(
        group,
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(),
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic()),
        investigation=investigation,
    )
    assert result.passed is False
    assert result.failure_category == FailureCategory.SECURITY_FLAG
    assert result.semantic_security_review is not None
    assert result.semantic_security_review.verdict == SecurityReviewVerdict.INCONCLUSIVE


def test_nonblocking_scanner_execution_failure_can_pass_with_semantic_review():
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(scanner_status=ScannerExecutionStatus.TIMEOUT),
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic()),
        investigation=_investigation(),
    )
    assert result.passed is True
    assert result.deterministic_gates is not None
    assert result.deterministic_gates.scanner_execution_status == ScannerExecutionStatus.TIMEOUT


def test_migration_policy_keeps_scanner_and_tests_hard():
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.MIGRATION_CODE_WORKAROUND,
        _results(remaining={"CVE-2025-0001"}, tests=(False, "test failed")),
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic()),
    )
    assert result.passed is False
    assert result.failure_category == FailureCategory.SECURITY_FLAG


def test_no_fix_package_removal_requires_manifest_and_graph_absence():
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        _results(),
        QAEvaluation(task_id="g1", passed=True),
        package_state=_QAPackageState(manifest_state="absent", graph_state="absent"),
    )
    assert result.passed is True


def test_no_fix_package_state_fails_closed_for_unsupported_manager():
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
        MagicMock(),
        group,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
    )
    assert package_state.manifest_state == "unknown"
    assert package_state.graph_state == "unknown"
    assert "pnpm" in package_state.diagnostics[0]


def test_no_fix_code_removal_requires_package_in_manifest_and_graph():
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.NO_FIX_CODE_REMOVAL,
        _results(remaining={"CVE-2025-0001"}),
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic()),
        investigation=_investigation(),
        package_state=_QAPackageState(
            manifest_state="present",
            graph_state="present",
        ),
    )
    assert result.passed is True


def test_shared_install_failure_fails_every_group():
    groups = [_group("g1", "CVE-2025-0001"), _group("g2", "CVE-2025-0002")]
    results = _results(install=(False, "npm install ERESOLVE peer conflict"))
    gates, _ = _evaluate_policy_gates(
        groups, results, {group.group_id: QAPolicy.VERSION_BUMP for group in groups}
    )
    evaluations, _ = _apply_policy_decision(
        groups,
        BatchQAResult(
            holistic_report="test",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True) for group in groups],
        ),
        gates,
        {group.group_id: QAPolicy.VERSION_BUMP for group in groups},
    )
    assert all(not evaluation.passed for evaluation in evaluations.values())
    assert all(
        evaluation.failure_category == FailureCategory.PEER_CONFLICT
        for evaluation in evaluations.values()
    )
