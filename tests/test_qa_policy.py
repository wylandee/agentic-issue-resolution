"""Focused tests for the supervisor-owned deterministic QA policy matrix."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    DependencyEvidenceStatus,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    ODCScanEvidence,
    QAAttemptResult,
    QADependencyEvidence,
    QAEvaluation,
    QAPolicy,
    QASemanticSecurityReview,
    QATestAttribution,
    RemediationTask,
    RoutingStrategy,
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
from remediation_engine.orchestration._qa_runtime import (
    _attempt_version_evidence,
    _collect_group_package_state,
    _derive_qa_task_policies,
)
from remediation_engine.orchestration.qa_evaluator import GroupInvestigation
from remediation_engine.orchestration.qa_policy_engine import (
    _apply_guardrails,
    _apply_policy_decision,
    _attach_scan_evidence_to_evaluations,
    _evaluate_policy_gates,
)
from remediation_engine.orchestration.qa_types import (
    QATaskContext,
    _QAExecutionResults,
    _QAPackageState,
    _SecurityScanResult,
)
from remediation_engine.orchestration.task_utils import (
    build_initial_remediation_task,
    derive_missing_task_qa_policy,
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
    sandbox.run.assert_called_once()


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
