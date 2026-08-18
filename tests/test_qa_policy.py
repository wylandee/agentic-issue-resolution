"""Focused tests for the supervisor-owned deterministic QA policy matrix."""

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
    TestAttributionVerdict,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
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
    remaining = set(remaining or set())
    return _QAExecutionResults(
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


def _semantic_review() -> QASemanticSecurityReview:
    return QASemanticSecurityReview(
        verdict=SecurityReviewVerdict.PASS,
        reasoning="The changed call site no longer reaches the vulnerable sink.",
        evidence_refs=["src/index.js:42", "search:vulnerable_api"],
    )


def _investigation() -> GroupInvestigation:
    return GroupInvestigation(
        group_id="g1",
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
    if package_state is not None:
        results.package_state_by_group[group.group_id] = package_state
    gates, _ = _evaluate_policy_gates([group], results, {group.group_id: policy})
    evaluations, _ = _apply_policy_decision(
        [group],
        BatchQAResult(evaluations=[evaluation]),
        gates,
        {group.group_id: policy},
        {group.group_id: investigation} if investigation is not None else None,
    )
    return evaluations[group.group_id]


def test_version_bump_scanner_is_group_scoped() -> None:
    g1 = _group("g1", "CVE-2025-0001")
    g2 = _group("g2", "CVE-2025-0002")
    results = _results(remaining={"CVE-2025-0002"})
    gates, _ = _evaluate_policy_gates(
        [g1, g2], results, {"g1": QAPolicy.VERSION_BUMP, "g2": QAPolicy.VERSION_BUMP}
    )
    evaluations, _ = _apply_policy_decision(
        [g1, g2],
        BatchQAResult(
            evaluations=[
                QAEvaluation(task_id="g1", passed=True),
                QAEvaluation(task_id="g2", passed=True),
            ]
        ),
        gates,
        {"g1": QAPolicy.VERSION_BUMP, "g2": QAPolicy.VERSION_BUMP},
    )
    assert evaluations["g1"].passed is True
    assert evaluations["g2"].failure_category == FailureCategory.SECURITY_FLAG


def test_version_bump_may_exonerate_unrelated_test_failure() -> None:
    group = _group("g1")
    result = _evaluate(
        group,
        QAPolicy.VERSION_BUMP,
        _results(tests=(False, "test failed")),
        QAEvaluation(
            task_id="g1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The other group owns the failure.",
            test_attribution=QATestAttribution(
                verdict=TestAttributionVerdict.EXONERATED,
                responsible_group_ids=["g2"],
                failed_tests=["tests/other.test.js::fails"],
                reasoning="The failure does not touch this diff.",
            ),
        ),
    )
    assert result.passed is True


def test_hard_test_policy_ignores_llm_exoneration() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.INITIAL_CODE_WORKAROUND,
        _results(tests=(False, "test failed")),
        QAEvaluation(
            task_id="g1",
            passed=True,
            semantic_security_review=_semantic_review(),
            test_attribution=QATestAttribution(
                verdict=TestAttributionVerdict.EXONERATED,
                responsible_group_ids=["g2"],
                failed_tests=["tests/other.test.js::fails"],
                reasoning="The other group owns the failure.",
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
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic_review()),
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
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic_review()),
        investigation=_investigation(),
    )
    assert result.passed is True


def test_no_fix_package_removal_requires_manifest_and_graph_absence() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        _results(),
        QAEvaluation(task_id="g1", passed=True),
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
    package_state = _collect_group_package_state(MagicMock(), group, QAPolicy.NO_FIX_PACKAGE_REMOVAL)
    assert package_state.manifest_state == "unknown"
    assert package_state.graph_state == "unknown"
    assert "pnpm" in package_state.diagnostics[0]


def test_no_fix_code_removal_requires_package_in_manifest_and_graph() -> None:
    result = _evaluate(
        _group("g1"),
        QAPolicy.NO_FIX_CODE_REMOVAL,
        _results(remaining={"CVE-2025-0001"}),
        QAEvaluation(task_id="g1", passed=True, semantic_security_review=_semantic_review()),
        investigation=_investigation(),
        package_state=_QAPackageState(manifest_state="present", graph_state="present"),
    )
    assert result.passed is True


def test_shared_install_failure_fails_every_group() -> None:
    groups = [_group("g1", "CVE-2025-0001"), _group("g2", "CVE-2025-0002")]
    results = _results(install=(False, "npm install ERESOLVE peer conflict"))
    gates, _ = _evaluate_policy_gates(
        groups, results, {group.group_id: QAPolicy.VERSION_BUMP for group in groups}
    )
    evaluations, _ = _apply_policy_decision(
        groups,
        BatchQAResult(evaluations=[QAEvaluation(task_id=group.group_id, passed=True) for group in groups]),
        gates,
        {group.group_id: QAPolicy.VERSION_BUMP for group in groups},
    )
    assert all(not evaluation.passed for evaluation in evaluations.values())
    assert all(
        evaluation.failure_category == FailureCategory.PEER_CONFLICT
        for evaluation in evaluations.values()
    )
