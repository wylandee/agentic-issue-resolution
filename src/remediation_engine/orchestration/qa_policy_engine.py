"""Deterministic QA policy gates, guardrails, and evidence attachment."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from remediation_engine.contracts.schemas import (
    DependencyEvidenceStatus,
    FailureCategory,
    ODCScanEvidence,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    QASemanticSecurityReview,
    ScannerExecutionStatus,
    SecurityReviewVerdict,
    TestAttributionVerdict,
)
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

from ._qa_runtime import group_target_identifiers
from .qa_types import QATaskContext, _QAExecutionResults, _QAPackageState, _scan_result_value

if TYPE_CHECKING:
    from .qa_evaluator import GroupInvestigation


logger = logging.getLogger(__name__)


_PEER_CONFLICT_PATTERNS = ("ERESOLVE", "EOVERRIDE", "peer dep", "peer tree")
_ENGINE_CONFLICT_PATTERNS = ("EBADENGINE",)


@dataclass(frozen=True)
class _QAPolicyPromptSpec:
    """Prompt contract for one deterministic QA policy."""

    scanner_rule: str
    package_rule: str
    tests_rule: str
    semantic_rule: str
    review_focus: str
    prohibited_conclusions: str


_QA_POLICY_PROMPT_SPECS: dict[QAPolicy, _QAPolicyPromptSpec] = {
    QAPolicy.VERSION_BUMP: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared. Treat remaining target identifiers as a blocking security failure.",
        "Python verifies the authorized manifest and resolved dependency state. Use the supplied typed dependency evidence; do not fail because raw manifest or lockfile output is unavailable.",
        "Review each failed test independently. Assign responsibility only with positive evidence tied to this group's changed package or behavior; otherwise use structured exoneration or INCONCLUSIVE attribution.",
        "A code-path semantic review is not required by this policy.",
        "Validate the Python-owned dependency evidence, target scanner clearance, and any test attribution.",
        "Do not assign blame because a group was updated in the same batch, and do not exonerate a group without naming exact failed tests and evidence.",
    ),
    QAPolicy.INITIAL_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers are non-blocking for this policy. They are evidence to interpret, not proof that the code workaround failed.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the initial workaround blocks the vulnerable execution path.",
        "Identify the advisory mechanism, affected call sites, protected call sites, and concrete file/symbol evidence showing whether the workaround is effective.",
        "Do not fail solely because the original scanner identifier remains, and do not pass using test logs without source or diff evidence.",
    ),
    QAPolicy.MIGRATION_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared. Treat remaining target identifiers as a blocking security failure.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A required semantic security review is not part of this policy.",
        "Validate target scanner clearance, migration/install behavior, and whether the remediation caused any test or build regression.",
        "Do not treat a remaining target identifier as non-blocking or use LLM attribution to override hard scanner or test gates.",
    ),
    QAPolicy.MITIGATION_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers are non-blocking for this policy. They are evidence to interpret, not proof that the mitigation failed.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the mitigation blocks or materially constrains the vulnerable behavior.",
        "Identify the advisory mechanism, affected call sites, protected call sites, and concrete file/symbol evidence showing whether the mitigation is effective.",
        "Do not fail solely because the original scanner identifier remains, and do not pass using test logs without source or diff evidence.",
    ),
    QAPolicy.NO_FIX_PACKAGE_REMOVAL: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared after package removal. Treat remaining target identifiers as a blocking security failure.",
        "The vulnerable package must be absent from every authorized direct manifest and from the resolved dependency graph.",
        "The required test suite must pass.",
        "A code-path semantic security review is not required by this policy.",
        "Confirm the authorized package-removal operation, manifest state, resolved graph, scanner clearance, and test results.",
        "Do not pass while the package remains in an authorized manifest or resolved graph, and do not treat residual target findings as non-blocking.",
    ),
    QAPolicy.NO_FIX_CODE_REMOVAL: _QAPolicyPromptSpec(
        "Target scanner identifiers are expected to remain because the vulnerable package is intentionally retained. They are non-blocking for this policy.",
        "The vulnerable package must remain present in the direct manifest and resolved dependency graph. The workaround must not remove or change dependency metadata.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the vulnerable code path was removed while the package remained installed.",
        "Identify the advisory mechanism, affected and protected call sites, and concrete file/symbol evidence showing that the vulnerable execution path is no longer used.",
        "Do not fail solely because the target package or scanner identifiers remain. Do not pass using test logs without source or diff evidence.",
    ),
}
_STRICT_SCANNER_QA_POLICIES = frozenset(
    {
        QAPolicy.VERSION_BUMP,
        QAPolicy.MIGRATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
    }
)
_HARD_TEST_QA_POLICIES = frozenset(
    {
        QAPolicy.INITIAL_CODE_WORKAROUND,
        QAPolicy.MIGRATION_CODE_WORKAROUND,
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }
)
_REQUIRED_SEMANTIC_QA_POLICIES = frozenset(
    {
        QAPolicy.INITIAL_CODE_WORKAROUND,
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }
)


def _qa_policy_prompt_block(policy: QAPolicy | None) -> str:
    """Render Supervisor-owned QA rules for the structured evaluator."""
    if policy is None:
        return (
            "## QA Policy\n"
            "- Policy: unavailable; do not invent a policy-specific pass.\n"
            "- Treat missing policy as an inconclusive contract condition."
        )
    spec = _QA_POLICY_PROMPT_SPECS[policy]
    return "\n".join(
        [
            "## QA Policy",
            f"- Policy: {policy.value}",
            f"- Scanner rule: {spec.scanner_rule}",
            f"- Package rule: {spec.package_rule}",
            f"- Test rule: {spec.tests_rule}",
            f"- Semantic review rule: {spec.semantic_rule}",
            f"- Review focus: {spec.review_focus}",
            f"- Prohibited conclusions: {spec.prohibited_conclusions}",
        ]
    )


def _valid_test_attribution(
    evaluation: QAEvaluation,
    group_id: str,
    known_group_ids: set[str],
) -> bool:
    """Return whether a structured test attribution contains usable evidence."""
    attribution = evaluation.test_attribution
    if attribution is None or not attribution.failed_tests or not attribution.reasoning.strip():
        return False
    if any(identifier not in known_group_ids for identifier in attribution.responsible_group_ids):
        return False
    if attribution.verdict == TestAttributionVerdict.INCONCLUSIVE:
        return True
    if not attribution.responsible_group_ids:
        return False
    if attribution.verdict == TestAttributionVerdict.RESPONSIBLE:
        return group_id in attribution.responsible_group_ids
    if attribution.verdict == TestAttributionVerdict.EXONERATED:
        return group_id not in attribution.responsible_group_ids
    return False


def _evaluate_policy_gates(
    task_contexts: list[QATaskContext],
    results: _QAExecutionResults,
    task_policies: dict[str, QAPolicy | None],
) -> tuple[dict[str, QADeterministicGates], list[str]]:
    """Record deterministic execution evidence and policy gates per task."""
    gates_by_task: dict[str, QADeterministicGates] = {}
    errors: list[str] = []
    install_passed = bool(results.install and results.install[0])
    install_summary = results.install[1] if results.install else "install did not run"
    tests_passed = results.tests[0] if results.tests is not None else None
    remaining_global = set(
        _scan_result_value(results.scan, "remaining_identifiers", set()) or set()
    )
    scanner_status = _scan_result_value(
        results.scan, "execution_status", ScannerExecutionStatus.NOT_RUN
    )
    if not isinstance(scanner_status, ScannerExecutionStatus):
        try:
            scanner_status = ScannerExecutionStatus(str(scanner_status))
        except ValueError:
            scanner_status = ScannerExecutionStatus.UNPARSEABLE
    if scanner_status == ScannerExecutionStatus.NOT_RUN and results.scan is not None:
        scanner_status = (
            ScannerExecutionStatus.SUCCESS
            if bool(_scan_result_value(results.scan, "ok", False))
            else ScannerExecutionStatus.UNPARSEABLE
        )

    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        policy = task_policies.get(task_id)
        effective_scanner_status = (
            ScannerExecutionStatus.SUCCESS
            if results.scan_skipped and policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL
            else scanner_status
        )
        remaining = sorted(group_target_identifiers(group) & remaining_global)
        package_state = results.package_state_by_task.get(task_id, _QAPackageState())
        dependency_evidence = package_state.dependency_evidence
        target_cleared = (
            effective_scanner_status == ScannerExecutionStatus.SUCCESS and not remaining
        )
        diagnostics: list[str] = []
        if not install_passed:
            diagnostics.append(install_summary)
        if effective_scanner_status != ScannerExecutionStatus.SUCCESS:
            diagnostics.append(
                f"Scanner execution status: {effective_scanner_status.value}. "
                f"{_scan_result_value(results.scan, 'summary', 'scanner did not run')}"
            )
        diagnostics.extend(package_state.diagnostics)
        if policy == QAPolicy.VERSION_BUMP and dependency_evidence is None:
            diagnostics.append("Deterministic dependency evidence was not collected.")
        if policy is None:
            diagnostics.append("QA policy provenance is missing or invalid.")
            errors.append(
                f"qa_critic policy evaluator: task '{task_id}' has missing or invalid policy provenance."
            )

        deterministic_pass = policy is not None and install_passed
        if policy in _STRICT_SCANNER_QA_POLICIES:
            deterministic_pass = deterministic_pass and target_cleared
        if policy in _HARD_TEST_QA_POLICIES:
            deterministic_pass = deterministic_pass and tests_passed is True
        if policy == QAPolicy.VERSION_BUMP:
            deterministic_pass = (
                deterministic_pass
                and dependency_evidence is not None
                and dependency_evidence.status == DependencyEvidenceStatus.VERIFIED
            )
        if policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL:
            deterministic_pass = deterministic_pass and package_state.manifest_state == "absent"
            deterministic_pass = deterministic_pass and package_state.graph_state == "absent"
        if policy == QAPolicy.NO_FIX_CODE_REMOVAL:
            deterministic_pass = deterministic_pass and package_state.manifest_state == "present"
            deterministic_pass = deterministic_pass and package_state.graph_state == "present"

        gates_by_task[task_id] = QADeterministicGates(
            dependency_evidence=dependency_evidence,
            status="pass" if deterministic_pass else "fail",
            install_passed=install_passed,
            scanner_execution_status=effective_scanner_status,
            target_remaining_identifiers=remaining,
            target_scanner_cleared=target_cleared,
            tests_passed=tests_passed,
            package_manifest_state=package_state.manifest_state,
            package_graph_state=package_state.graph_state,
            install_error_category=results.install_error_category,
            peer_conflicts=list(results.peer_conflicts),
            diagnostics=diagnostics,
        )
    return gates_by_task, errors


def _semantic_review_is_evidence_backed(
    review: QASemanticSecurityReview | None,
    investigation: GroupInvestigation | None,
) -> bool:
    """Require structured review fields plus actual successful review-tool evidence."""
    return bool(
        review is not None
        and review.verdict == SecurityReviewVerdict.PASS
        and review.reasoning.strip()
        and review.evidence_refs
        and all(reference.strip() for reference in review.evidence_refs)
        and investigation is not None
        and investigation.source_review_evidence
        and not investigation.fallback
        and investigation.structured_review_verdict == SecurityReviewVerdict.PASS
    )


def _qa_evaluation_items(
    source: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
) -> list[QAEvaluation]:
    """Normalize evaluator output into a list of QAEvaluation items."""
    if isinstance(source, Mapping):
        return list(source.values())
    return list(source)


def _install_conflict_is_terminal(
    gates: QADeterministicGates,
    install_error_category: str | None,
) -> bool:
    """Return whether a failed install has a deterministic dependency conflict."""
    if gates.install_passed:
        return False
    if str(install_error_category or "").upper() in {"PEER_CONFLICT", "ENGINE_CONFLICT"}:
        return True
    install_summary = gates.diagnostics[0] if gates.diagnostics else ""
    conflict_markers = (*_PEER_CONFLICT_PATTERNS, *_ENGINE_CONFLICT_PATTERNS)
    return any(marker.casefold() in install_summary.casefold() for marker in conflict_markers)


def _install_conflict_retry_feedback(gates: QADeterministicGates) -> str:
    """Build retry guidance without claiming unavailable post-install validation."""
    install_summary = gates.diagnostics[0] if gates.diagnostics else "npm install failed."
    return (
        f"{install_summary} Dependency installation failed before post-install QA validation; "
        "resolve the dependency conflict before retrying."
    )


def _version_bump_llm_failure_is_relevant(
    evaluation: QAEvaluation,
    gates: QADeterministicGates,
    *,
    test_exonerated: bool,
) -> bool:
    """Return whether an LLM failure adds a version-bump policy failure."""
    if evaluation.passed:
        return False
    if evaluation.failure_category == FailureCategory.BREAKING_CHANGE:
        return gates.tests_passed is False and not test_exonerated
    if evaluation.failure_category == FailureCategory.SECURITY_FLAG:
        feedback = (evaluation.retry_feedback or "").casefold()
        dependency_terms = (
            "manifest",
            "lockfile",
            "package.json",
            "package-lock.json",
            "dependency",
        )
        unavailable_terms = (
            "missing",
            "unavailable",
            "could not",
            "cannot",
            "unable",
            "no deterministic",
            "not found",
            "no match",
            "compacted",
            "omitted",
            "not provided",
            "not visible",
            "truncated",
            "not shown",
            "insufficient",
        )
        if any(term in feedback for term in dependency_terms) and any(
            term in feedback for term in unavailable_terms
        ):
            return False
    return True


def _apply_policy_decision(
    task_contexts: list[QATaskContext],
    batch_result: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
    gates_by_task: dict[str, QADeterministicGates],
    task_policies: dict[str, QAPolicy | None],
    investigations_by_task: dict[str, GroupInvestigation] | None = None,
    install_error_category: str | None = None,
    *,
    terminal_install_conflict: bool = True,
) -> tuple[dict[str, QAEvaluation], list[str]]:
    """Apply the Supervisor-owned policy matrix to task-keyed evaluations."""
    known_task_ids = {context.task_id for context in task_contexts}
    known_group_ids = {context.group.group_id for context in task_contexts}
    errors: list[str] = []
    normalized: dict[str, QAEvaluation] = {}
    missing_ids: set[str] = set()
    for evaluation in _qa_evaluation_items(batch_result):
        task_id = evaluation.task_id
        if task_id not in known_task_ids:
            errors.append(f"qa_critic policy evaluator: unknown task '{task_id}' dropped.")
        elif task_id in normalized:
            errors.append(
                f"qa_critic policy evaluator: duplicate evaluation for '{task_id}' dropped."
            )
        else:
            normalized[task_id] = evaluation
    for context in task_contexts:
        task_id = context.task_id
        if task_id not in normalized:
            missing_ids.add(task_id)
            errors.append(f"qa_critic policy evaluator: missing evaluation for '{task_id}'.")
            normalized[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                contract_error=True,
                contract_error_reason="Structured QA evaluator omitted this task.",
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Structured QA evaluator omitted this task; retry required.",
            )

    final: dict[str, QAEvaluation] = {}
    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        policy = task_policies.get(task_id)
        current = normalized[task_id]
        gates = gates_by_task[task_id]
        if current.contract_error:
            reason = (
                current.contract_error_reason.strip()
                or "The structured QA result could not be validated."
            )
            errors.append(f"qa_critic contract error for '{task_id}': {reason}")
            final[task_id] = current.model_copy(
                update={
                    "task_id": task_id,
                    "passed": False,
                    "deterministic_gates": gates,
                    "failure_category": FailureCategory.SECURITY_FLAG,
                    "retry_feedback": (
                        "QA result is inconclusive because the structured QA result could not be validated. "
                        "No remediation retry was consumed. Re-run QA after correcting the judge contract."
                    ),
                }
            )
            continue

        if terminal_install_conflict and _install_conflict_is_terminal(
            gates, install_error_category
        ):
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=FailureCategory.PEER_CONFLICT,
                retry_feedback=_install_conflict_retry_feedback(gates),
                deterministic_gates=gates,
            )
            continue

        failures: list[tuple[FailureCategory, str]] = []
        evaluator_test_exonerated = (
            policy == QAPolicy.VERSION_BUMP
            and gates.tests_passed is False
            and current.failure_category == FailureCategory.BREAKING_CHANGE
            and current.test_attribution is not None
            and _valid_test_attribution(current, group.group_id, known_group_ids)
        )
        dependency_evidence_inconclusive = policy == QAPolicy.VERSION_BUMP and (
            gates.dependency_evidence is None
            or gates.dependency_evidence.status == DependencyEvidenceStatus.INCONCLUSIVE
        )
        if policy == QAPolicy.VERSION_BUMP:
            if _version_bump_llm_failure_is_relevant(
                current,
                gates,
                test_exonerated=evaluator_test_exonerated,
            ):
                failures.append(
                    (
                        current.failure_category or FailureCategory.SECURITY_FLAG,
                        current.retry_feedback or "The structured QA evaluator failed this task.",
                    )
                )
        elif not current.passed and not evaluator_test_exonerated:
            failures.append(
                (
                    current.failure_category or FailureCategory.SECURITY_FLAG,
                    current.retry_feedback or "The structured QA evaluator failed this task.",
                )
            )
        if task_id in missing_ids:
            failures.append(
                (FailureCategory.SECURITY_FLAG, "Structured QA evaluator omitted this task.")
            )
        if not gates.install_passed:
            install_text = " ".join(gates.diagnostics)
            category = (
                FailureCategory.PEER_CONFLICT
                if any(
                    pattern.lower() in install_text.lower() for pattern in _PEER_CONFLICT_PATTERNS
                )
                else FailureCategory.SECURITY_FLAG
            )
            failures.append(
                (category, "Shared npm install failed; every task in this QA batch is failed.")
            )
        if policy is None:
            failures.append(
                (FailureCategory.SECURITY_FLAG, "Missing or invalid QA policy provenance.")
            )
        if policy in _STRICT_SCANNER_QA_POLICIES:
            if gates.scanner_execution_status != ScannerExecutionStatus.SUCCESS:
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "Strict scanner policy requires a trustworthy parseable scanner report.",
                    )
                )
            if gates.target_remaining_identifiers:
                if current.passed or current.failure_category == FailureCategory.BREAKING_CHANGE:
                    errors.append(
                        f"qa_critic guardrail: task '{task_id}' has remaining scanner "
                        "identifiers but judge did not prioritize SECURITY_FLAG; forcing "
                        "passed=False / SECURITY_FLAG."
                    )
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "Remaining target scanner identifiers: "
                        + ", ".join(gates.target_remaining_identifiers),
                    )
                )
        if (
            policy == QAPolicy.VERSION_BUMP
            and gates.dependency_evidence is not None
            and gates.dependency_evidence.status == DependencyEvidenceStatus.MISMATCH
        ):
            failures.append(
                (
                    FailureCategory.SECURITY_FLAG,
                    "Deterministic dependency evidence does not match the Supervisor-selected target.",
                )
            )
        if policy in _HARD_TEST_QA_POLICIES and gates.tests_passed is not True:
            failures.append(
                (FailureCategory.BREAKING_CHANGE, "The required global test suite did not pass.")
            )
        elif policy == QAPolicy.VERSION_BUMP and gates.tests_passed is not True:
            attribution = current.test_attribution
            valid_exoneration = bool(
                attribution
                and attribution.verdict == TestAttributionVerdict.EXONERATED
                and attribution.responsible_group_ids
                and all(
                    identifier in known_group_ids
                    for identifier in attribution.responsible_group_ids
                )
                and gates.tests_passed is False
                and attribution.failed_tests
                and attribution.reasoning.strip()
                and group.group_id not in attribution.responsible_group_ids
            )
            if not valid_exoneration:
                failures.append(
                    (
                        FailureCategory.BREAKING_CHANGE,
                        "VERSION_BUMP tests failed without valid structured exoneration evidence.",
                    )
                )
        if policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL:
            if gates.package_manifest_state != "absent":
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "NO_FIX Stage 1 requires the package to be absent from every authorized direct manifest.",
                    )
                )
            if gates.package_graph_state != "absent":
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "NO_FIX Stage 1 requires the package to be absent from the resolved dependency graph.",
                    )
                )
        if policy == QAPolicy.NO_FIX_CODE_REMOVAL:
            if gates.package_manifest_state != "present":
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "NO_FIX Stage 2 requires the package to remain present in the direct manifest.",
                    )
                )
            if gates.package_graph_state != "present":
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "NO_FIX Stage 2 requires the package to remain present in the resolved dependency graph.",
                    )
                )

        semantic_review = current.semantic_security_review
        if policy in _REQUIRED_SEMANTIC_QA_POLICIES and not _semantic_review_is_evidence_backed(
            semantic_review, (investigations_by_task or {}).get(task_id)
        ):
            if semantic_review is None or semantic_review.verdict == SecurityReviewVerdict.PASS:
                semantic_review = QASemanticSecurityReview(
                    verdict=SecurityReviewVerdict.INCONCLUSIVE,
                    reasoning="Required semantic review could not be verified from investigator evidence and structured review output.",
                )
            failures.append(
                (
                    FailureCategory.SECURITY_FLAG,
                    "Required semantic security review is missing, inconclusive, or lacks source/diff evidence.",
                )
            )

        # An unavailable dependency graph is a QA-only rerun when it is the
        # sole blocker. Preserve actionable install, scanner, test, and
        # semantic failures so the Supervisor can repair the candidate.
        if dependency_evidence_inconclusive and not failures:
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    "Deterministic dependency evidence was unavailable or incomplete. "
                    "Re-run QA evidence collection; no remediation retry should be consumed."
                ),
                failure_evidence=current.failure_evidence,
                deterministic_gates=gates,
                evidence_inconclusive=True,
                test_attribution=current.test_attribution,
            )
            continue

        category = next(
            (
                candidate
                for candidate in (
                    FailureCategory.SECURITY_FLAG,
                    FailureCategory.PEER_CONFLICT,
                    FailureCategory.BREAKING_CHANGE,
                )
                if any(item[0] == candidate for item in failures)
            ),
            None,
        )
        diagnostics = [message for _category, message in failures]
        if current.retry_feedback and not current.passed:
            diagnostics.append(current.retry_feedback)
        if not failures:
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=True,
                deterministic_gates=gates,
                semantic_security_review=semantic_review,
                test_attribution=current.test_attribution,
            )
        else:
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=category or FailureCategory.SECURITY_FLAG,
                retry_feedback=" ".join(dict.fromkeys(diagnostics))
                or "QA policy gate failed; retry required.",
                failure_evidence=current.failure_evidence,
                deterministic_gates=gates,
                semantic_security_review=semantic_review,
                test_attribution=current.test_attribution,
            )
    return final, errors


def _apply_guardrails(
    task_contexts: list[QATaskContext],
    batch_result: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
    results: _QAExecutionResults,
    task_policies: dict[str, QAPolicy | None],
    investigations_by_task: dict[str, GroupInvestigation] | None = None,
) -> tuple[dict[str, QAEvaluation], list[str]]:
    """Apply deterministic QA guardrails to task-keyed evaluations.

    Unknown or duplicate task evaluations are rejected, missing task results
    become inconclusive failures, and policy gates enforce scanner,
    package-state, test, and review requirements.
    """
    gates, gate_errors = _evaluate_policy_gates(task_contexts, results, task_policies)
    evaluations, decision_errors = _apply_policy_decision(
        task_contexts,
        batch_result,
        gates,
        task_policies,
        investigations_by_task,
        install_error_category=results.install_error_category,
    )
    return evaluations, gate_errors + decision_errors


def _extract_deterministic_test_evidence(
    results: _QAExecutionResults,
    *,
    sandbox: DockerSandbox | None = None,
) -> QAFailureEvidence | None:
    """Extract failure evidence from the captured test process streams."""
    if results.tests is None or results.tests[0]:
        return None
    exit_code = results.test_exit_code
    stdout = results.test_raw_stdout
    stderr = results.test_raw_stderr
    if stdout is None:
        stdout = results.tests[1]
    if stderr is None:
        stderr = ""
    from .qa_test_parsing import extract_qa_failure_evidence

    return extract_qa_failure_evidence(
        exit_code if exit_code is not None else 1,
        stdout,
        stderr,
        sandbox=sandbox,
    )


def _attach_failure_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    results: _QAExecutionResults,
    state: OrchestratorState,
    *,
    deterministic_evidence: QAFailureEvidence | None = None,
    sandbox: DockerSandbox | None = None,
) -> dict[str, QAEvaluation]:
    """Attach deterministic failure evidence and committed attempt provenance.

    The structured evaluator classifies the outcome, while deterministic test
    parsing supplies failing output and source paths. The task's committed
    attempt envelope supplies the authoritative ``attempt_id`` so retry
    routing receives actionable evidence instead of inferred locations.

    Args:
        evaluations: Structured evaluations keyed by task identifier.
        results: Deterministic QA execution results.
        state: Current orchestration state with task and attempt projections.
        deterministic_evidence: Previously extracted test evidence, if any.
        sandbox: Optional active sandbox for lazy evidence extraction.

    Returns:
        Evaluations enriched with deterministic evidence and attempt identity.
    """
    test_evidence = deterministic_evidence
    if test_evidence is None:
        test_evidence = _extract_deterministic_test_evidence(results, sandbox=sandbox)

    task_queue = state.get("task_queue", {}) or {}
    enriched: dict[str, QAEvaluation] = {}
    for task_id, evaluation in evaluations.items():
        if evaluation.task_id != task_id:
            logger.warning(
                "QA discarded evaluation evidence for key %r because the "
                "evaluation identifies task %r.",
                task_id,
                evaluation.task_id,
            )
            enriched[task_id] = evaluation.model_copy(update={"failure_evidence": None})
            continue
        if evaluation.passed:
            enriched[task_id] = evaluation
            continue
        llm_evidence = evaluation.failure_evidence
        evidence = test_evidence

        task = task_queue.get(task_id)
        attempt_id = getattr(task, "current_attempt_id", None) if task is not None else None
        task_revision = int(getattr(task, "task_revision", 0) or 0) if task is not None else 0
        snapshots = state.get("attempt_snapshots_by_id", {}) or {}
        snapshot = snapshots.get(attempt_id) if attempt_id else None
        snapshot_task_id = (
            snapshot.get("task_id")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "task_id", None)
            if snapshot is not None
            else None
        )
        snapshot_attempt_id = (
            snapshot.get("attempt_id")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "attempt_id", None)
            if snapshot is not None
            else None
        )
        snapshot_revision = (
            snapshot.get("task_revision")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "task_revision", None)
            if snapshot is not None
            else None
        )
        snapshot_matches = False
        if snapshot is not None:
            try:
                snapshot_matches = int(snapshot_revision) == task_revision
            except (TypeError, ValueError):
                snapshot_matches = False
        if (
            task is None
            or not attempt_id
            or snapshot is None
            or snapshot_task_id != task_id
            or snapshot_attempt_id != attempt_id
            or not snapshot_matches
        ):
            if evidence is not None or llm_evidence is not None:
                logger.warning(
                    "QA discarded failure evidence for task %r because its current "
                    "attempt snapshot was missing or contradictory.",
                    task_id,
                )
            enriched[task_id] = evaluation.model_copy(update={"failure_evidence": None})
            continue
        if evidence is not None and llm_evidence is not None and llm_evidence.source_locations:
            discarded_locations = ", ".join(llm_evidence.source_locations[:5])
            if len(llm_evidence.source_locations) > 5:
                discarded_locations += ", ..."
            diagnostic = (
                "QA discarded LLM-supplied source location(s) because deterministic "
                f"test evidence is authoritative: {discarded_locations}."
            )
            logger.warning(diagnostic)
            evidence = evidence.model_copy(
                update={
                    "exact_diagnostics": [
                        *evidence.exact_diagnostics,
                        diagnostic,
                    ][:15]
                }
            )
        elif evidence is None and llm_evidence is not None:
            diagnostic = (
                "QA discarded LLM-supplied source locations because deterministic "
                "test evidence was unavailable."
            )
            logger.warning(diagnostic)
            evidence = llm_evidence.model_copy(
                update={
                    "source_locations": [],
                    "affected_files": [],
                    "exact_diagnostics": [
                        *llm_evidence.exact_diagnostics,
                        diagnostic,
                    ][:15],
                }
            )
        if evidence is not None:
            evidence = evidence.model_copy(
                update={
                    "attempt_id": attempt_id,
                    "task_revision": task_revision,
                }
            )
            evaluation = evaluation.model_copy(update={"failure_evidence": evidence})
        enriched[task_id] = evaluation
    return enriched


def _attach_scan_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    evidence: ODCScanEvidence | None,
    task_contexts: list[QATaskContext],
) -> dict[str, QAEvaluation]:
    """Attach task-scoped projections of attempt-local scan evidence."""
    if evidence is None:
        return evaluations
    groups_by_task = {context.task_id: context.group for context in task_contexts}
    enriched: dict[str, QAEvaluation] = {}
    for task_id, evaluation in evaluations.items():
        if evaluation.task_id != task_id:
            logger.warning(
                "QA skipped scan evidence for key %r because the evaluation identifies task %r.",
                task_id,
                evaluation.task_id,
            )
            enriched[task_id] = evaluation
            continue
        group = groups_by_task.get(task_id)
        if group is None:
            enriched[task_id] = evaluation
            continue
        target_ids = group_target_identifiers(group)
        scoped_evidence = evidence.model_copy(
            update={
                "covered_task_ids": [task_id],
                "found_identifiers": sorted(set(evidence.found_identifiers) & target_ids),
                "remaining_target_identifiers": sorted(
                    set(evidence.remaining_target_identifiers) & target_ids
                ),
            }
        )
        enriched[task_id] = evaluation.model_copy(update={"scan_evidence": scoped_evidence})
    return enriched
