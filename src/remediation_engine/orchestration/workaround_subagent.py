"""
Sequential Workaround Subagent for Phase 5 code-security rewrites.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundEdit,
    WorkaroundPhase,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.remedy_tools import (
    _is_prohibited_target,
    build_workaround_toolbelt,
)
from remediation_engine.orchestration.state import SubagentState, _derive_legacy_task_from_group
from remediation_engine.orchestration.subagent_runtime import (
    has_successful_validation_gate,
    has_tool_call_before_first_successful_edit,
    run_bounded_subagent_loop,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]


def _create_skinny_subagent_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for execution agents while preserving compact vulnerability identifiers."""
    return group.model_copy(
        update={
            "cve_ids": group.cve_ids[:5],
            "ghsa_ids": group.ghsa_ids[:5],
            "versions": group.versions[:5],
            "issues": [],
        }
    )


def _filter_constraints_ledger(
    constraints_ledger: list[str], target_group: VulnerabilityGroup
) -> list[str]:
    """Filter ledger to only include constraints matching the target component."""
    comp = target_group.vulnerable_component
    if not comp:
        return list(constraints_ledger)

    filtered = []
    for constraint in constraints_ledger:
        if comp in constraint:
            filtered.append(constraint)
    return filtered


_QA_ERROR_MARKER = re.compile(
    r"(?:error|exception|failed|failure|not\s+a\s+function|undefined|cannot|invalid|missing|required\s+option|not\s+exported|cannot\s+find)",
    re.IGNORECASE,
)


def _clean_prompt_snippet(value: str, max_chars: int = 240) -> str:
    """Normalize one extracted QA or vulnerability snippet for prompt use."""
    cleaned = re.sub(r"\s+", " ", value).strip(" `\"'")
    cleaned = cleaned.replace("`", "").replace('"', "")
    return cleaned[:max_chars].strip()


def _extract_qa_error_snippet(previous_feedback: str) -> str:
    """Extract the most diagnostic error text from QA feedback for web search."""
    feedback = str(previous_feedback or "")

    explicit_error = re.search(
        r"\b[A-Za-z_][\w.]*(?:Error|Exception)\s*:\s*[^;\n]{1,240}",
        feedback,
        re.IGNORECASE,
    )
    if explicit_error:
        return _clean_prompt_snippet(explicit_error.group(0))

    package_diagnostic = re.search(
        r"\b[A-Za-z][\w-]*[.-][\w.-]*:\s*(?:[^\n]{1,240}?\b(?:required\s+option|not\s+exported|not\s+a\s+function|cannot\s+find|undefined|invalid)\b[^\n]{0,160})",
        feedback,
        re.IGNORECASE,
    )
    if package_diagnostic:
        return _clean_prompt_snippet(package_diagnostic.group(0))

    for match in re.finditer(r"`([^`\n]{1,240})`|\"([^\"\n]{1,240})\"", feedback):
        candidate = match.group(1) or match.group(2) or ""
        if _QA_ERROR_MARKER.search(candidate):
            return _clean_prompt_snippet(candidate)

    not_a_function = re.search(
        r"\([^\n)]{1,240}\)\s+is\s+not\s+a\s+function",
        feedback,
        re.IGNORECASE,
    )
    if not_a_function:
        return _clean_prompt_snippet(not_a_function.group(0))

    for line in feedback.splitlines():
        if _QA_ERROR_MARKER.search(line):
            return _clean_prompt_snippet(line)

    return _clean_prompt_snippet(feedback, max_chars=240)


@dataclasses.dataclass(frozen=True)
class _SearchQueryRecommendation:
    """Scenario-specific first query and rationale for workaround research."""

    scenario: str
    initial_query: str
    rationale: str
    follow_up_query: str = ""


def _search_query_term(value: Any, *, max_chars: int = 160) -> str:
    """Return a compact unquoted term suitable for a web-search query."""
    cleaned = _clean_prompt_snippet(str(value or ""), max_chars=max_chars)
    return cleaned.replace('"', "").replace("'", "")


def _query_parts(*parts: str) -> str:
    """Join non-empty search terms without introducing duplicate whitespace or quotation marks."""
    joined = " ".join(part.strip() for part in parts if part and part.strip())
    return joined.replace('"', "").replace("'", "")


def _workaround_evidence_text(
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> str:
    """Flatten QA and retry evidence used to classify the research scenario."""
    qa_evidence = workaround_context.qa_evidence if workaround_context else None
    values = [str(previous_feedback or "")]
    if qa_evidence:
        values.extend(
            [
                *(qa_evidence.exact_diagnostics or []),
                *(qa_evidence.failed_tests or []),
                qa_evidence.raw_excerpt or "",
            ]
        )
    return " ".join(value for value in values if value).lower()


def _has_test_failure_evidence(
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> bool:
    """Return whether the supplied evidence indicates a test or build regression."""
    qa_evidence = workaround_context.qa_evidence if workaround_context else None
    if qa_evidence and qa_evidence.failed_tests:
        return True

    evidence_text = _workaround_evidence_text(workaround_context, previous_feedback)
    return any(
        marker in evidence_text
        for marker in (
            "test failed",
            "tests failed",
            "npm test",
            "typecheck",
            "compile failed",
            "build failed",
            "is not a function",
            "typeerror",
            "exception",
        )
    )


def _workaround_search_recommendation(
    target_task: Any,
    target_group: VulnerabilityGroup,
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> _SearchQueryRecommendation:
    """Build a first web query from the current remediation evidence.

    Scanner failures need advisory and source-level mitigation research, while
    test failures need package migration and compatibility research.  Keeping
    this classification deterministic gives the LLM a useful first query
    without preventing it from refining that query after inspecting the code.
    """
    fix_plan = getattr(target_group, "fix_plan", None)
    plan_status = getattr(getattr(fix_plan, "status", None), "value", "")
    evidence_text = _workaround_evidence_text(workaround_context, previous_feedback)
    scanner_failure = any(
        marker in evidence_text
        for marker in (
            "scanner",
            "remaining findings",
            "remaining scanner",
            "unresolved identifier",
            "still vulnerable",
            "security scan",
            "dependency-check",
            "semgrep",
        )
    )
    test_failure = _has_test_failure_evidence(workaround_context, previous_feedback)

    component = _search_query_term(
        getattr(target_group, "vulnerable_component", "") or "component",
        max_chars=100,
    )
    identifiers = [
        *(getattr(target_group, "cve_ids", []) or [])[:1],
        *(getattr(target_group, "ghsa_ids", []) or [])[:1],
    ]
    identifier_terms = " ".join(
        _search_query_term(identifier, max_chars=80) for identifier in identifiers
    )
    mechanism = _search_query_term(
        _extract_vulnerability_mechanism(target_group),
        max_chars=140,
    )
    diagnostic = ""
    qa_evidence = workaround_context.qa_evidence if workaround_context else None
    if qa_evidence and qa_evidence.exact_diagnostics:
        diagnostic = _extract_qa_error_snippet(qa_evidence.exact_diagnostics[0])
    elif previous_feedback:
        diagnostic = _extract_qa_error_snippet(previous_feedback)
    diagnostic_term = _search_query_term(diagnostic, max_chars=180)

    selected_version = getattr(target_task, "selected_version", None)
    if not selected_version and fix_plan is not None:
        selected_version = getattr(fix_plan, "fixed_version", None)
    version_term = f"version {selected_version}" if selected_version else ""

    if scanner_failure:
        return _SearchQueryRecommendation(
            scenario="update_does_not_resolve_scanner_findings",
            initial_query=_query_parts(
                component,
                identifier_terms,
                "still vulnerable",
                "scanner remediation",
                mechanism,
            ),
            rationale=(
                "Lead with the advisory identifier and vulnerable mechanism so the worker "
                "can determine why the scanner still flags the package."
            ),
            follow_up_query=_query_parts(
                component,
                identifier_terms,
                "source-level mitigation",
                "maintainer advisory",
            ),
        )

    if test_failure or (
        workaround_context and workaround_context.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
    ):
        return _SearchQueryRecommendation(
            scenario="update_mitigates_cve_but_breaks_tests",
            initial_query=_query_parts(
                component,
                version_term,
                diagnostic_term,
                "migration breaking changes compatibility",
            ),
            rationale=(
                "Lead with the exact QA diagnostic and attempted version so the worker "
                "finds the package migration or API compatibility guidance."
            ),
            follow_up_query=_query_parts(
                component,
                version_term,
                "migration guide",
                "breaking changes",
            ),
        )

    if plan_status in {"no_fix", "workaround_found"}:
        return _SearchQueryRecommendation(
            scenario="no_update_available_to_resolve_cve",
            initial_query=_query_parts(
                component,
                identifier_terms,
                "no upstream fix",
                "compensating control",
                mechanism,
            ),
            rationale=(
                "Search for a defensible source-level isolation or compensating control "
                "because the planner has no usable upstream version."
            ),
            follow_up_query=_query_parts(
                component,
                identifier_terms,
                "source-level mitigation",
                "security advisory",
            ),
        )

    return _SearchQueryRecommendation(
        scenario="initial_code_workaround_or_isolation",
        initial_query=_query_parts(
            component,
            identifier_terms,
            mechanism,
            "security advisory",
            "mitigation",
        ),
        rationale=(
            "Start with the vulnerability identifier and mechanism, then adapt the "
            "advisory guidance to the local source code."
        ),
        follow_up_query=_query_parts(
            component,
            identifier_terms,
            "maintainer guidance",
            "workaround",
        ),
    )


def _preferred_targeted_test_files(qa_evidence: Any) -> list[str]:
    """Extract source test files from structured QA locations and affected files."""
    if qa_evidence is None:
        return []

    candidates: list[str] = []
    for raw_value in [
        *(getattr(qa_evidence, "source_locations", []) or []),
        *(getattr(qa_evidence, "affected_files", []) or []),
    ]:
        value = str(raw_value or "").strip().replace("\\", "/")
        value = re.sub(r":\d+(?::\d+)?(?:$|\b).*", "", value)
        if value.startswith("/workspace/"):
            value = value[len("/workspace/") :]
        value = value.strip(" `\"'()[]{}.,;")
        if value.startswith("build/"):
            value = value[len("build/") :]
        if not value or not (
            "/test/" in f"/{value}/"
            or "/tests/" in f"/{value}/"
            or "/__tests__/" in f"/{value}/"
            or re.search(r"\.(?:test|spec)\.[^.]+$", value, re.IGNORECASE)
        ):
            continue
        if value not in candidates:
            candidates.append(value)
    return candidates[:5]


def _extract_vulnerability_mechanism(group: VulnerabilityGroup) -> str:
    """Extract a compact vulnerability mechanism before the detailed fix guidance."""
    for issue in getattr(group, "issues", []) or []:
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message.strip():
            continue

        mechanism = message
        for section_marker in ("### Am I affected?", "### How to fix that?"):
            mechanism = mechanism.split(section_marker, 1)[0]
        mechanism = _clean_prompt_snippet(mechanism, max_chars=1200)
        if mechanism:
            return mechanism

    fix_plan = getattr(group, "fix_plan", None)
    instruction = getattr(fix_plan, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return _clean_prompt_snippet(instruction, max_chars=1200)
    return ""


def _workaround_search_strategy(
    target_task: Any,
    target_group: VulnerabilityGroup,
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> str:
    """Explain the scenario and recommended first query for this workaround."""
    recommendation = _workaround_search_recommendation(
        target_task,
        target_group,
        workaround_context,
        previous_feedback,
    )

    return "\n".join(
        [
            "=== WORKAROUND SEARCH STRATEGY ===",
            "Classify the workaround from the evidence before calling search_web.",
            "  - update_mitigates_cve_but_breaks_tests: search the package's migration/API change, exact runtime error, and affected test behavior.",
            "  - update_does_not_resolve_scanner_findings: search the advisory's vulnerable mechanism, scanner identifier, and source-level mitigation requirements.",
            "  - no_update_available_to_resolve_cve: search vendor advisories, maintainer guidance, and safe code-level isolation or compensating controls.",
            "  - initial_code_workaround_or_isolation: search the exact component, vulnerability mechanism, and authoritative implementation guidance.",
            f"Initial evidence-based classification to confirm or reject: {recommendation.scenario}.",
            "=== RECOMMENDED INITIAL SEARCH QUERY ===",
            f"Scenario: {recommendation.scenario}",
            f"Query: {recommendation.initial_query}",
            f"Why this query: {recommendation.rationale}",
            (
                f"If the first results are insufficient, follow-up query: {recommendation.follow_up_query}"
                if recommendation.follow_up_query
                else ""
            ),
            "Use the recommended query for the first search_web call; construct the query yourself only when refining it, and do not blindly copy a fixed template.",
            "Prefer authoritative advisory, maintainer, migration-guide, or source-repository results.",
        ]
    )


def _build_workaround_prompt(
    target_task: Any,  # RemediationTask
    target_group: VulnerabilityGroup | list[str],
    constraints_ledger: list[str] | None = None,
    previous_feedback: str | None = None,
    current_replay_plan: WorkaroundReplayPlan | None = None,
    vulnerability_mechanism: str | None = None,
    workaround_context: WorkaroundContext | None = None,
) -> str:
    if isinstance(target_group, list):
        constraints_ledger = list(target_group)
        target_group = target_task
        target_task = _derive_legacy_task_from_group(target_group)

    constraints_ledger = list(constraints_ledger or [])
    fix_plan = getattr(target_group, "fix_plan", None)
    vulnerability_mechanism = (
        _clean_prompt_snippet(vulnerability_mechanism, max_chars=1200)
        if vulnerability_mechanism
        else _extract_vulnerability_mechanism(target_group)
    )

    phase = (
        workaround_context.phase
        if workaround_context
        else (
            WorkaroundPhase.QA_REGRESSION_REPAIR
            if (previous_feedback or current_replay_plan)
            else WorkaroundPhase.INITIAL_MITIGATION
        )
    )

    sections = [
        "You are a code security specialist operating inside a shared Docker workspace.",
        f"WORKFLOW PHASE: {phase.value.upper()}",
    ]

    if phase == WorkaroundPhase.INITIAL_MITIGATION:
        sections.append(
            "\n".join(
                [
                    "=== OPERATING PRINCIPLES ===",
                    "  1. MINIMAL SURGICAL EDITS: Make only the changes necessary to implement a code workaround or isolate the targeted vulnerability. Do not rewrite surrounding unchanged code, alter formatting styles unnecessarily, or remove comments.",
                    "  2. NO ASSUMPTIONS: Always inspect the codebase using search_codebase_pattern and inspect_ast_symbol to confirm file paths, signatures, and dependencies. DO NOT guess security configurations; rely on search_web for authoritative CVE mitigation guidance.",
                    "  3. VALIDATION-DRIVEN CONFIRMATION: Never assume a fix is successful without verifying it through validate_workaround. It is the only public validation gate and short-circuits on the first failure.",
                    "  4. ADAPT, DO NOT COPY: NEVER blindly copy-paste code snippets from CVE advisories or documentation. Advisories use placeholders (like [REDACTED]) and external libraries that may not exist in our codebase. You must extract the underlying security invariant and apply it seamlessly to the existing variables and logic found in the workspace.",
                    "",
                    "=== EXECUTION LIFECYCLE ===",
                    "  1. EXPLORE & INSPECT",
                    "     - Read the TARGET section to understand the vulnerability mechanism.",
                    "     - FIRST: Use search_codebase_pattern to locate vulnerable package/pattern usage across source files.",
                    "     - FIRST: Inspect relevant JS/TS AST symbols using inspect_ast_symbol or read lines with read_workspace_file BEFORE searching the web. You must understand the local context (variable names, existing parameters) first.",
                    "     - THEN: Use search_web to find authoritative advisory or mitigation information. Adapt the findings to the local context you just discovered. If you need to read a specific result, use read_web_page.",
                    "  2. PLAN",
                    "     - Form a security-preserving causal hypothesis based on authoritative documentation.",
                    "     - Record your plan using record_plan, explicitly identifying the affected files, symbols, security invariant, and the exact intended edits.",
                    "  3. IMPLEMENT",
                    "     - Apply required security changes incrementally using deterministic_search_replace or deterministic_replace_ast_symbol.",
                    "     - If you make a mistake, you may use revert_workspace_file to reset a file back to its initial state.",
                    "  4. VERIFY & ITERATE",
                    "     - Call validate_workaround with every modified source file and the most relevant runtime smoke file.",
                    "     - If validation fails, analyze the exact first-gate diagnostic. Do not blindly guess fixes. If you encounter a TypeScript or runtime error, you MUST search the web for the exact error string.",
                    "     - Revise your hypothesis, re-record the complete plan, and repeat until the validation gate passes cleanly.",
                    "",
                    "=== TOOL USE RULES ===",
                    "  - Batching: Batch independent tool calls (e.g., calling read_workspace_file on multiple files, or doing multiple search_codebase_pattern searches) into a single turn to save latency.",
                    "  - Editing: Build the complete cumulative patch first before calling validate_workaround.",
                    "  - Debugging Errors: If a tool call or validation fails, analyze the error output directly rather than repeating the exact same tool invocation or guessing syntax.",
                ]
            )
        )
    else:  # QA_REGRESSION_REPAIR
        sections.append(
            "\n".join(
                [
                    "=== OPERATING PRINCIPLES ===",
                    "  1. PRESERVE INTENT: Make only the changes necessary to resolve the QA regression caused by a dependency update. Preserve prior replayed edits listed in CUMULATIVE REPLAY CONTEXT.",
                    "  2. NO API ASSUMPTIONS: When fixing regressions following a package update, DO NOT guess the new API syntax. Always use search_web to find the library's migration guide or breaking changes.",
                    "  3. TEST-DRIVEN CONFIRMATION: Return control to QA only after the combined validation gate passes cleanly.",
                    "  4. ADAPT, DO NOT COPY: NEVER blindly copy-paste code snippets from CVE advisories or documentation. You must extract the underlying security invariant and apply it seamlessly to the existing variables and logic found in the workspace.",
                    "",
                    "=== EXECUTION LIFECYCLE ===",
                    "  1. EXPLORE & INSPECT",
                    "     - Quote or explicitly acknowledge the exact QA diagnostic reported below.",
                    "     - FIRST: Trace failing behavior from the reported test location to the modified source using search_codebase_pattern, read_workspace_file, and inspect_ast_symbol.",
                    '     - THEN: If the diagnostic is a TypeError (e.g., "is not a function") or an import failure, use search_web to find the library\'s "migration guide" or exact error string. Read the guide using read_web_page.',
                    "  2. PLAN",
                    "     - Form a causal hypothesis based on authoritative documentation, NOT hallucination.",
                    "     - Record all required changes as one coherent plan using record_plan, including every causally related file and symbol.",
                    "  3. IMPLEMENT",
                    "     - Apply the complete repair incrementally using deterministic_search_replace or deterministic_replace_ast_symbol. Ensure imports, call sites, and control-flow changes exactly match the updated library API.",
                    "     - If you make a mistake, use revert_workspace_file to discard your local edits.",
                    "  4. VERIFY & ITERATE",
                    "     - Call validate_workaround with the complete cumulative modified-file list and the failing test target.",
                    "     - If the gate fails again, use its exact diagnostic to revise your hypothesis. Do not flip-flop between syntax guesses. Re-evaluate your documentation search, re-record the plan, and try again.",
                    "",
                    "=== TOOL USE RULES ===",
                    "  - Batching: Batch independent tool calls (e.g., calling inspect_ast_symbol on multiple files) into a single turn to save latency.",
                    "  - Editing: Build the complete cumulative patch first before calling validate_workaround.",
                    "  - Debugging Errors: If a tool call or validation fails, analyze the error output directly rather than repeating the exact same tool invocation or guessing syntax.",
                ]
            )
        )

    sections.append(
        "\n".join(
            [
                "=== PROHIBITIONS & ANTI-PATTERNS ===",
                "- ❌ NEVER modify package.json, package-lock.json, yarn.lock, pnpm-lock.yaml, pom.xml, or any dependency manifest.",
                "- ❌ NEVER modify test files to make assertions pass.",
                "- ❌ NEVER bump library versions — version selection is strictly the update_subagent's job.",
                "- ❌ NEVER remove vulnerability mitigations without explicit instruction.",
                "- ❌ NEVER declare success based only on syntax/typecheck.",
                "- ALWAYS use relative file paths.",
                "- MUST call record_plan before making code edits.",
            ]
        )
    )

    sections.append(
        _workaround_search_strategy(
            target_task,
            target_group,
            workaround_context,
            previous_feedback,
        )
    )

    if current_replay_plan and current_replay_plan.successful_edits:
        replay_lines = [
            "=== CUMULATIVE REPLAY CONTEXT ===",
            f"The following {len(current_replay_plan.successful_edits)} valid code workaround edit(s) from prior attempt(s) have been automatically replayed onto your baseline:",
        ]
        for edit in current_replay_plan.successful_edits:
            replay_lines.append(
                f"  - File: {edit.file_path} (symbol: {edit.symbol_name or 'n/a'}, edit #{edit.edit_index})"
            )
            old_preview = _clean_prompt_snippet(edit.old_text, max_chars=260)
            new_preview = _clean_prompt_snippet(edit.new_text, max_chars=260)
            replay_lines.append(f"    Prior exact replacement: {old_preview} -> {new_preview}")
        if current_replay_plan.security_invariants:
            replay_lines.append("Preserved Security Invariants:")
            for inv in current_replay_plan.security_invariants:
                replay_lines.append(f"  - {inv}")
        if current_replay_plan.diagnosed_root_causes:
            replay_lines.append("Previously Diagnosed Root Causes:")
            for cause in current_replay_plan.diagnosed_root_causes:
                replay_lines.append(f"  - {cause}")
        if current_replay_plan.planned_targets:
            replay_lines.append("Previously Planned Targets:")
            for target in current_replay_plan.planned_targets:
                replay_lines.append(f"  - {target}")
        findings = current_replay_plan.investigation_findings or {}
        validation_feedback = findings.get("validation_feedback")
        if validation_feedback:
            replay_lines.append("Latest Validation Feedback:")
            replay_lines.append(f"  {str(validation_feedback)[:2400]}")
        if current_replay_plan.validated_files:
            replay_lines.append("Previously Validated Files:")
            for file_path in current_replay_plan.validated_files:
                replay_lines.append(f"  - {file_path}")
        replay_lines.append(
            "Build directly on top of these replayed edits without removing security mitigations."
        )
        sections.append("\n".join(replay_lines))

    comp_name = getattr(target_group, "vulnerable_component", "") or "component"
    cve_label = (
        target_group.cve_ids[0]
        if getattr(target_group, "cve_ids", None)
        else (target_group.ghsa_ids[0] if getattr(target_group, "ghsa_ids", None) else "")
    )

    qa_evidence = workaround_context.qa_evidence if workaround_context else None
    # Structured QA evidence is authoritative. The free-form retry message
    # often wraps the real diagnostic in a large generic report.
    err_text = (
        qa_evidence.exact_diagnostics[0]
        if qa_evidence and qa_evidence.exact_diagnostics
        else (previous_feedback or "")
    )
    if err_text:
        err_snippet = _extract_qa_error_snippet(err_text)
        search_evidence = err_snippet
    else:
        search_evidence = ""

    if qa_evidence:
        ev_lines = [
            "=== QA FAILURE EVIDENCE ===",
            f"Source Attempt ID : {qa_evidence.attempt_id}",
            f"Source Revision   : {qa_evidence.task_revision}",
            "Exact Diagnostics :",
        ]
        for diag in qa_evidence.exact_diagnostics:
            ev_lines.append(f"  - {diag}")
        if qa_evidence.failed_tests:
            ev_lines.append("Failed Tests:")
            for ft in qa_evidence.failed_tests:
                ev_lines.append(f"  - {ft}")
        if qa_evidence.source_locations:
            ev_lines.append("Source Locations:")
            for sl in qa_evidence.source_locations:
                ev_lines.append(f"  - {sl}")
        if qa_evidence.affected_files:
            ev_lines.append("Affected Files:")
            for af in qa_evidence.affected_files:
                ev_lines.append(f"  - {af}")
        preferred_tests = _preferred_targeted_test_files(qa_evidence)
        if preferred_tests:
            ev_lines.append(
                "RECOMMENDED TARGETED TEST FILES (use these before any unrelated test):"
            )
            for test_file in preferred_tests:
                ev_lines.append(f"  - {test_file}")
        if qa_evidence.raw_excerpt:
            ev_lines.append(f"Raw Excerpt:\n{qa_evidence.raw_excerpt[:1000]}")
        if search_evidence:
            ev_lines.extend(
                [
                    "SEARCH INPUT EVIDENCE (construct the query yourself; do not copy a fixed template):",
                    f"  - Component: {comp_name}",
                    f"  - Vulnerability identifier: {cve_label or 'not provided'}",
                    f"  - Exact diagnostic candidate: {search_evidence}",
                ]
            )
        sections.append("\n".join(ev_lines))
    elif previous_feedback:
        retry_lines = [
            "=== RETRY CONTEXT ===",
            f"QA Feedback: {previous_feedback}",
        ]
        if search_evidence:
            retry_lines.extend(
                [
                    "SEARCH INPUT EVIDENCE (construct the query yourself; do not copy a fixed template):",
                    f"  - Component: {comp_name}",
                    f"  - Vulnerability identifier: {cve_label or 'not provided'}",
                    f"  - Exact diagnostic candidate: {search_evidence}",
                ]
            )
        sections.append("\n".join(retry_lines))

    if constraints_ledger:
        sections.append(
            "Constraints ledger:\n" + "\n".join(f"- {item}" for item in constraints_ledger)
        )

    target_lines = [
        "=== TARGET ===",
        f"Task ID       : {target_task.task_id}",
        f"Issue Type    : {getattr(target_group.issue_type, 'value', str(target_group.issue_type))}",
        f"Component     : {getattr(target_group, 'vulnerable_component', '') or 'unknown'}",
        f"Initial File  : {getattr(target_group, 'file_path', '') or 'none'}",
        f"Vulnerability Mechanism: {vulnerability_mechanism or 'not provided'}",
        f"Instruction   : {target_task.instruction or 'Apply defensive code fix.'}",
    ]
    sections.append("\n".join(target_lines))

    if fix_plan and getattr(fix_plan, "workaround_snippets", None):
        sections.append(
            "\n".join(
                [
                    "=== WORKAROUND SNIPPETS ===",
                    "Reference code patterns from security advisories:",
                    *[
                        f"  {i + 1}. {snippet}"
                        for i, snippet in enumerate(fix_plan.workaround_snippets)
                    ],
                ]
            )
        )

    return "\n\n".join(sections)


def _build_action_summaries(
    task_id: str,
    changed_files: list[str],
    final_text: str,
    succeeded: bool,
) -> list[AgentActionSummary]:
    summary_status = AgentActionStatus.SUCCESS if succeeded else AgentActionStatus.SURRENDER
    changed_label = ", ".join(changed_files) if changed_files else "no files"
    outcome = (
        "Completed validated code workaround edits"
        if succeeded
        else "Stopped without a validated code workaround"
    )
    final_note = final_text.strip()
    if final_note:
        summary_text = f"{outcome}; changed files: {changed_label}. Final note: {final_note}"
    else:
        summary_text = f"{outcome}; changed files: {changed_label}."
    return [AgentActionSummary(task_id=task_id, status=summary_status, summary=summary_text)]


def _latest_validation_feedback(tool_events: list[Any]) -> str:
    """Return the latest failed validation diagnostic for cumulative replay context."""
    for event in reversed(tool_events):
        if getattr(event, "name", "") in {"validate_workaround", "run_targeted_test"} and str(
            getattr(event, "content", "")
        ).startswith("FAILURE:"):
            return str(event.content)[:4000]
    return ""


def _workaround_attempt_succeeded(
    runtime: Any,
    *,
    has_all_validated: bool,
    has_recorded_plan: bool,
    requires_targeted_test: bool,
    targeted_test_passed: bool,
    validation_gate_passed: bool | None = None,
) -> bool:
    """Apply the deterministic success contract for one workaround attempt."""
    validation_passed = (
        validation_gate_passed if validation_gate_passed is not None else has_all_validated
    )
    return bool(
        runtime.changed_files
        and not runtime.errors
        and validation_passed
        and has_recorded_plan
        and (not requires_targeted_test or targeted_test_passed)
    )


def _build_surrender_summaries(task_id: str, message: str) -> list[AgentActionSummary]:
    return [
        AgentActionSummary(task_id=task_id, status=AgentActionStatus.SURRENDER, summary=message)
    ]


def _build_attempt_result(
    state: SubagentState,
    summary: AgentActionSummary,
    *,
    succeeded: bool,
    errors: list[str],
    changed_files: list[str],
    replay_plan: WorkaroundReplayPlan | None = None,
) -> dict[str, WorkerAttemptResult]:
    snapshot = state.get("attempt_snapshot")
    if snapshot is None:
        return {}
    tagged_summary = summary.model_copy(
        update={
            "attempt_id": snapshot.attempt_id,
            "task_revision": snapshot.task_revision,
            "instruction_digest": snapshot.instruction_digest,
        }
    )
    return {
        snapshot.attempt_id: WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=snapshot.task_id,
            task_revision=snapshot.task_revision,
            status=tagged_summary.status,
            changed_files=changed_files,
            action_summary=tagged_summary,
            execution_diagnostics=WorkerExecutionDiagnostics(
                validation_passed=succeeded,
                failure_reason=" | ".join(errors),
            ),
            instruction_digest=snapshot.instruction_digest,
            replay_plan=replay_plan,
            errors=errors,
        )
    }


@traceable(name="Workaround_Subagent_Test_Run")
def run_workaround_subagent_node(state: SubagentState) -> dict[str, Any]:
    """Run the single-group workaround subagent on ``SubagentState``."""
    repo_root_str = state.get("repo_root", "")
    workspace_volume = state.get("workspace_volume", "")
    target_task = state.get("target_task")
    target_group = state.get("target_group")
    constraints_ledger = list(state.get("constraints_ledger", []))
    previous_feedback = state.get("previous_feedback")
    current_replay_plan = state.get("current_replay_plan")

    t_id = target_task.task_id if target_task else "unknown"

    if os.environ.get("REMEDY_BYPASS_WORKAROUND_SUBAGENT", "false").lower() in ("1", "true", "yes"):
        summaries = _build_surrender_summaries(
            t_id,
            "Workaround subagent bypassed: marked unfixable (workaround functionality currently inactive).",
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "worker_results_by_attempt": _build_attempt_result(
                state,
                summaries[0],
                succeeded=False,
                errors=[],
                changed_files=[],
            ),
            "errors": [],
        }

    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        summaries = _build_surrender_summaries(
            t_id, "Stopped before execution because repo_root was invalid."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": [
                f"Workaround Subagent: repo_root '{repo_root_str}' is not a valid directory."
            ],
        }

    if not workspace_volume:
        summaries = _build_surrender_summaries(
            t_id, "Stopped before execution because workspace_volume was missing."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": ["Workaround Subagent: workspace_volume is missing from state."],
        }

    if target_task is None or target_group is None:
        summaries = _build_surrender_summaries(
            t_id, "Stopped before execution because no target task/group was provided."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": ["Workaround Subagent: target_task or target_group is missing from state."],
        }

    if ChatOpenAI is None:
        summaries = _build_surrender_summaries(
            t_id, "Stopped before execution because the LLM client is unavailable."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": ["Workaround Subagent: 'langchain-openai' is not installed."],
        }

    model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(
            t_id, "Stopped before execution because the LLM failed to initialize."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": [],
            "errors": [f"Workaround Subagent: failed to initialize LLM - {exc}."],
        }

    touched_files: set[str] = set()
    filtered_ledger = _filter_constraints_ledger(constraints_ledger, target_group)
    vulnerability_mechanism = _extract_vulnerability_mechanism(target_group)
    skinny_group = _create_skinny_subagent_group(target_group)
    plan_state: dict[str, Any] = {"recorded": False}

    pre_attempt_snapshots: dict[str, str] = {}
    replayed_edits: list[WorkaroundEdit] = []
    if current_replay_plan is not None:
        pre_attempt_snapshots = dict(current_replay_plan.pre_attempt_snapshots)
        replayed_edits = list(current_replay_plan.successful_edits)

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # 1. Restore pre-attempt snapshots if replay plan present
            if pre_attempt_snapshots:
                for rel_p, orig_content in pre_attempt_snapshots.items():
                    try:
                        sandbox.write_file(rel_p, orig_content)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Workaround subagent: failed to restore snapshot for %s: %s", rel_p, exc
                        )

            # 2. Replay prior successful edits sequentially
            for redit in replayed_edits:
                try:
                    curr = sandbox.read_file(redit.file_path)
                    if curr and redit.old_text in curr:
                        updated = curr.replace(redit.old_text, redit.new_text, 1)
                        sandbox.write_file(redit.file_path, updated)
                        touched_files.add(redit.file_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Workaround subagent: failed to replay edit on %s: %s", redit.file_path, exc
                    )

            snapshot = state.get("attempt_snapshot")
            workaround_ctx = getattr(snapshot, "workaround_context", None) if snapshot else None

            qa_ev = workaround_ctx.qa_evidence if workaround_ctx else None
            prompt = _build_workaround_prompt(
                target_task,
                skinny_group,
                filtered_ledger,
                previous_feedback,
                current_replay_plan,
                vulnerability_mechanism=vulnerability_mechanism,
                workaround_context=workaround_ctx,
            )
            initial_messages = [
                SystemMessage(
                    content=(
                        "Use only source-code tools. Build the complete cumulative patch first, "
                        "then call validate_workaround once; it is the only public validation gate "
                        "and short-circuits on the first failure."
                    )
                ),
                HumanMessage(content=prompt),
            ]

            toolbelt = build_workaround_toolbelt(
                sandbox,
                touched_files,
                repo_root,
                plan_state=plan_state,
                preferred_test_files=_preferred_targeted_test_files(qa_ev),
            )
            runtime = run_bounded_subagent_loop(llm, toolbelt, initial_messages, touched_files)

            # Revert any illegally modified manifest files or test files
            prohibited_modified = {f for f in touched_files if _is_prohibited_target(f)}
            if prohibited_modified:
                logger.warning(
                    "Workaround subagent modified prohibited files %s. Reverting.",
                    prohibited_modified,
                )
                for f in prohibited_modified:
                    try:
                        sandbox.revert_file(f)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to revert %s: %s", f, exc)
                touched_files -= prohibited_modified
                runtime = dataclasses.replace(runtime, changed_files=sorted(touched_files))

            validation_gate_passed = has_successful_validation_gate(
                runtime.tool_events,
                has_prior_edits=bool(replayed_edits),
            )
            has_all_validated = validation_gate_passed
            is_record_plan_in_toolbelt = any(
                getattr(t, "name", "") == "record_plan" for t in toolbelt
            )
            has_recorded_plan = (
                (not is_record_plan_in_toolbelt)
                or plan_state.get("recorded", False)
                or has_tool_call_before_first_successful_edit(
                    runtime.tool_events,
                    lookup_tool_name="record_plan",
                    edit_tool_name="deterministic_search_replace",
                )
                or has_tool_call_before_first_successful_edit(
                    runtime.tool_events,
                    lookup_tool_name="record_plan",
                    edit_tool_name="deterministic_replace_ast_symbol",
                )
            )

            requires_targeted_test = False
            targeted_test_passed = True
            if (
                workaround_ctx
                and workaround_ctx.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
                and qa_ev
                and (
                    qa_ev.failed_tests
                    or qa_ev.source_locations
                    or qa_ev.exact_diagnostics
                    or qa_ev.affected_files
                    or qa_ev.raw_excerpt
                )
            ):
                requires_targeted_test = True
                targeted_test_calls = [
                    e
                    for e in runtime.tool_events
                    if e.name == "validate_workaround" and e.content.startswith("SUCCESS:")
                ]
                targeted_test_passed = len(targeted_test_calls) > 0

            succeeded = _workaround_attempt_succeeded(
                runtime,
                has_all_validated=has_all_validated,
                has_recorded_plan=has_recorded_plan,
                requires_targeted_test=requires_targeted_test,
                targeted_test_passed=targeted_test_passed,
                validation_gate_passed=validation_gate_passed,
            )

            all_edits: list[WorkaroundEdit] = list(replayed_edits)
            if succeeded:
                new_edits = plan_state.get("successful_edits", [])
                all_edits.extend(new_edits)
            else:
                replayed_files = {e.file_path for e in replayed_edits}
                current_attempt_files = set(runtime.changed_files) - replayed_files
                for f in current_attempt_files:
                    if f in pre_attempt_snapshots:
                        with contextlib.suppress(Exception):
                            sandbox.write_file(f, pre_attempt_snapshots[f])
                    else:
                        with contextlib.suppress(Exception):
                            sandbox.revert_file(f)
                touched_files -= current_attempt_files
                runtime = dataclasses.replace(
                    runtime, changed_files=sorted(replayed_files & set(touched_files))
                )

    except Exception as exc:  # noqa: BLE001
        summaries = _build_surrender_summaries(
            t_id, "Stopped because the sandbox or tool loop failed."
        )
        return {
            "action_summaries": summaries,
            "action_summary": summaries[0],
            "changed_files": sorted(touched_files),
            "errors": [f"Workaround Subagent: sandbox or tool loop failed - {exc}"],
        }

    snapshot = state.get("attempt_snapshot")

    planned_files = plan_state.get("planned_files", [])
    planned_symbols = plan_state.get("planned_symbols", [])
    sec_inv = plan_state.get("security_invariant", "")
    causal_hyp = plan_state.get("causal_hypothesis", "")

    new_replay_plan = WorkaroundReplayPlan(
        task_id=t_id,
        pre_attempt_snapshots=pre_attempt_snapshots,
        successful_edits=all_edits,
        investigation_findings={
            "changed_files": list(touched_files),
            "validation_feedback": _latest_validation_feedback(runtime.tool_events),
        },
        source_attempt_id=snapshot.attempt_id if snapshot else "",
        security_invariants=[sec_inv] if sec_inv else [],
        diagnosed_root_causes=[causal_hyp] if causal_hyp else [],
        planned_targets=planned_files + planned_symbols,
        validated_files=list(runtime.changed_files),
    )

    summaries = _build_action_summaries(
        t_id,
        runtime.changed_files,
        runtime.final_text,
        succeeded,
    )
    tagged_summaries = [
        summaries[0].model_copy(
            update={
                "attempt_id": snapshot.attempt_id,
                "task_revision": snapshot.task_revision,
                "instruction_digest": snapshot.instruction_digest,
            }
        )
        if snapshot is not None
        else summaries[0]
    ]
    return {
        "action_summaries": tagged_summaries,
        "action_summary": tagged_summaries[0],
        "changed_files": runtime.changed_files,
        "worker_results_by_attempt": _build_attempt_result(
            state,
            tagged_summaries[0],
            succeeded=succeeded,
            errors=list(runtime.errors),
            changed_files=list(runtime.changed_files),
            replay_plan=new_replay_plan,
        ),
        "errors": runtime.errors,
    }
