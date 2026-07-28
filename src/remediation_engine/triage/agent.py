"""
agent.py - Triage agent for scanner findings.

Public API
----------
run_triage(group, context) -> TriageResult
    Produce a triage verdict for one ``VulnerabilityGroup``.

    Deterministic path (always active):
      - Always valid by default (unknown evidence != false positive).
      - Dev/test scope explicitly set -> may mark is_valid=False with reason.
      - Deterministic RBVM guardrails are always applied afterwards.

    LLM path (active when TRIAGE_LLM_ENABLED=true):
      - Uses LangChain ``with_structured_output(TriageResult)`` via OpenAI.
      - Deterministic guardrails are applied *after* LLM output.
      - triage_method is set to "llm"; falls back to "deterministic" on error.

Environment variables
---------------------
TRIAGE_LLM_ENABLED   : "true" to activate LLM path (default: off)
OPENAI_API_KEY       : Required when LLM path is active
TRIAGE_LLM_MODEL     : OpenAI model name (default: "gpt-4o-mini")
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langsmith import traceable

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.trajectory_exporter import invoke_with_trajectory

logger = logging.getLogger(__name__)

_UNREACHABLE_CODE_NOTE = (
    "Reachability analysis shows the package is not imported by the application "
    "source code."
)

# ---------------------------------------------------------------------------
# Priority ordering helpers
# ---------------------------------------------------------------------------

_SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 4,
    Severity.MEDIUM: 3,
    Severity.LOW: 2,
    Severity.INFO: 1,
    Severity.UNKNOWN: 0,
}


def _max_severity(*severities: Severity) -> Severity:
    """Return the highest severity from the provided set."""
    return max(severities, key=lambda s: _SEVERITY_RANK[s])


# ---------------------------------------------------------------------------
# Deterministic triage core
# ---------------------------------------------------------------------------


def _original_severity(group: VulnerabilityGroup) -> Severity:
    """Return the highest original severity across all member issues."""
    if not group.issues:
        return Severity.UNKNOWN
    return _max_severity(*(i.severity for i in group.issues))


def _apply_guardrails(
    priority: Severity,
    is_valid: bool,
    false_positive_reason: Optional[str],
    context: SystemContext,
    enrichment: Optional[CVEEnrichment],
    original_severity: Severity,
) -> tuple[Severity, str, bool, Optional[str], bool]:
    """
    Apply deterministic RBVM guardrails to a candidate triage verdict.

    Returns
    -------
    tuple
        (revised_priority, guardrail_notes, is_valid, false_positive_reason,
        priority_overridden)
    """
    reasoning_parts: list[str] = []
    starting_priority = priority
    epss_score = enrichment.epss if enrichment else 0.0
    environment = (context.environment or "").strip().lower()
    data_sensitivity = (context.data_sensitivity or "").strip().lower()

    # 1. Drop Everything Override
    if enrichment and enrichment.in_kev and environment == "production":
        priority = Severity.CRITICAL
        is_valid = True
        false_positive_reason = None
        if starting_priority != Severity.CRITICAL:
            reasoning_parts.append(
                "[Guardrail] Forced to CRITICAL: CVE is in CISA KEV and the "
                "environment is production."
            )

    # 2. Imminent Threat Override
    elif epss_score >= 0.36:
        if context.public_facing is True or data_sensitivity == "high":
            priority = Severity.CRITICAL
            if starting_priority != Severity.CRITICAL:
                reasoning_parts.append(
                    "[Guardrail] Forced to CRITICAL: EPSS >= 0.36 and the app "
                    "is public-facing or handles high-sensitivity data."
                )
        else:
            upgraded_priority = _max_severity(priority, Severity.HIGH)
            if upgraded_priority != starting_priority:
                reasoning_parts.append(
                    "[Guardrail] Forced to at least HIGH: EPSS >= 0.36 indicates "
                    "imminent exploit risk."
                )
            priority = upgraded_priority

    # 3. Floor Downgrade
    elif (
        epss_score < 0.01
        and context.public_facing is False
        and original_severity in (Severity.HIGH, Severity.CRITICAL)
        and _SEVERITY_RANK[priority] > _SEVERITY_RANK[Severity.MEDIUM]
    ):
        priority = Severity.MEDIUM
        reasoning_parts.append(
            "[Guardrail] Downgraded to MEDIUM: EPSS < 0.01 for an internal app "
            "indicates extremely low exploitability."
        )

    priority_overridden = priority != starting_priority
    return priority, " ".join(reasoning_parts), is_valid, false_positive_reason, priority_overridden


def _deterministic_triage(
    group: VulnerabilityGroup,
    context: SystemContext,
) -> TriageResult:
    """
    Produce a fully deterministic ``TriageResult`` without any LLM call.

    This is the baseline path and the final guardrail layer that wraps the
    LLM path.
    """
    original_sev = _original_severity(group)
    enrichment = group.enrichment
    is_unreachable_code = group.is_reachable is False

    # Start optimistic
    is_valid = True
    false_positive_reason: Optional[str] = None
    base_priority = original_sev if original_sev != Severity.UNKNOWN else Severity.MEDIUM

    reasoning_parts: list[str] = [
        f"Original severity: {original_sev.value}."
    ]
    if is_unreachable_code:
        reasoning_parts.append(_UNREACHABLE_CODE_NOTE)

    # Dev/test scope: only mark FP when there is clear evidence
    is_dev_env = (context.environment or "").lower() in {"dev", "test", "ci", "development"}
    all_dev_only = group.issues and all(
        (i.file_path or "").startswith(("test/", "tests/", "spec/", "dev/"))
        or i.file_path is None
        for i in group.issues
    )
    if is_dev_env and all_dev_only:
        is_valid = False
        false_positive_reason = (
            f"All findings are in dev/test paths and environment is '{context.environment}'. "
            "Treating as out-of-scope for production remediation."
        )
        base_priority = Severity.LOW
        reasoning_parts.append(false_positive_reason)

    final_priority, guardrail_note, is_valid, false_positive_reason, _ = _apply_guardrails(
        priority=base_priority,
        is_valid=is_valid,
        false_positive_reason=false_positive_reason,
        context=context,
        enrichment=enrichment,
        original_severity=original_sev,
    )
    if guardrail_note:
        reasoning_parts.append(guardrail_note)

    if enrichment:
        if enrichment.epss > 0.0:
            reasoning_parts.append(
                f"EPSS score: {enrichment.epss:.3f} "
                f"(percentile {enrichment.epss_percentile:.3f})."
            )
        if enrichment.in_kev:
            reasoning_parts.append("CVE is in CISA KEV; active exploitation risk is known.")

    return TriageResult(
        chain_of_thought="Deterministic fallback path; no LLM reasoning used.",
        group_id=group.group_id,
        is_valid=is_valid,
        false_positive_reason=false_positive_reason,
        original_severity=original_sev,
        revised_priority=final_priority,
        is_unreachable_code=is_unreachable_code,
        priority_reasoning=" ".join(reasoning_parts),
        validity_confidence_score=1.0,
        priority_confidence_score=1.0,
        recommended_issue_id=group.representative_issue_id,
        triage_method="deterministic",
    )


# ---------------------------------------------------------------------------
# LLM triage path
# ---------------------------------------------------------------------------


def _build_cve_details_prompt(group: VulnerabilityGroup) -> str:
    if not group.cve_ids and not group.ghsa_ids:
        return "  (No CVE or GHSA identifiers in this group)"

    # Map CVE IDs (case-insensitive keys) to their descriptions and CVSS scores.
    cve_details: dict[str, tuple[list[str], list[str]]] = {}
    for issue in group.issues:
        cve = issue.cve_id
        if not cve:
            continue
        cve = cve.upper().strip()
        if cve not in cve_details:
            cve_details[cve] = ([], [])

        desc = issue.message
        if desc:
            cve_details[cve][0].append(desc.strip())

        raw = issue.raw_payload or {}
        vuln = raw.get("vulnerability") or {}
        cvssv3 = vuln.get("cvssv3") or {}
        cvssv2 = vuln.get("cvssv2") or {}
        if cvssv3 and cvssv3.get("baseScore") is not None:
            score = cvssv3.get("baseScore")
            severity = cvssv3.get("baseSeverity") or cvssv3.get("severity") or "UNKNOWN"
            cve_details[cve][1].append(f"CVSS v3: {score} ({severity})")
        elif cvssv2 and cvssv2.get("score") is not None:
            score = cvssv2.get("score")
            severity = cvssv2.get("severity") or "UNKNOWN"
            cve_details[cve][1].append(f"CVSS v2: {score} ({severity})")

    lines = []
    for cve in sorted(group.cve_ids):
        cve_upper = cve.upper().strip()
        desc_list, cvss_list = cve_details.get(cve_upper, ([], []))
        desc = desc_list[0] if desc_list else "unavailable"
        cvss = cvss_list[0] if cvss_list else "unavailable"
        lines.append(f"  - CVE ID: {cve_upper}")
        lines.append(f"    CVSS Score: {cvss}")
        lines.append(f"    Description: {desc}")
    for ghsa in sorted(group.ghsa_ids):
        lines.append(f"  - GHSA ID: {ghsa.upper().strip()}")
    return "\n".join(lines)


def _build_triage_prompt(group: VulnerabilityGroup, context: SystemContext) -> str:
    """Build a structured prompt summarising the group for the LLM."""
    enrichment = group.enrichment
    epss_info = (
        f"EPSS score: {enrichment.epss:.3f} (percentile: {enrichment.epss_percentile:.3f})"
        if enrichment and enrichment.epss > 0.0
        else "EPSS: unavailable"
    )
    kev_info = (
        f"In CISA KEV: YES (added {enrichment.kev_date_added or 'unknown'})"
        if enrichment and enrichment.in_kev
        else "In CISA KEV: NO"
    )

    cve_list = ", ".join(group.cve_ids) if group.cve_ids else "none"
    ghsa_list = ", ".join(group.ghsa_ids) if group.ghsa_ids else "none"
    sources = ", ".join(s.value for s in group.sources)
    original_sev = _original_severity(group)
    reachability_info = (
        "TRUE (Package is explicitly imported in application code)"
        if group.is_reachable is True
        else (
            "FALSE (Package is a direct dependency but is NEVER imported in the application source code)"
            if group.is_reachable is False
            else "UNKNOWN (Likely a transitive dependency; cannot reliably determine reachability)"
        )
    )

    sys_os = context.deployment_os or "unknown"
    sys_public = "yes" if context.public_facing is True else ("no" if context.public_facing is False else "unknown")
    sys_lang = context.primary_language or "unknown"
    sys_arch = context.deployment_architecture or "unknown"
    sys_sens = context.data_sensitivity or "unknown"

    lines = [
        "You are a senior application security engineer triaging a vulnerability group.",
        "",
        "=== DATA ===",
        f"Group ID: {group.group_id}",
        f"Issue type: {group.issue_type.value}",
        f"Component: {group.vulnerable_component or 'unknown'}",
        f"File: {group.file_path or 'N/A'}",
        f"CVEs: {cve_list}",
        f"GHSAs: {ghsa_list}",
        f"Installed versions: {', '.join(group.versions) or 'unknown'}",
        f"Sources: {sources}",
        f"Original severity: {original_sev.value}",
        f"Reachability Analysis: {reachability_info}",
        f"{epss_info}",
        f"{kev_info}",
        "",
        "=== CVE DETAILS ===",
        _build_cve_details_prompt(group),
        "",
        "=== SYSTEM CONTEXT ===",
        f"Environment: {context.environment or 'unknown'}",
        f"Deployment OS: {sys_os}",
        f"Public Facing: {sys_public}",
        f"Primary Language: {sys_lang}",
        f"Deployment Architecture: {sys_arch}",
        f"Data Sensitivity: {sys_sens}",
        "",
        "=== TRIAGE INSTRUCTIONS & FALSE POSITIVE RULES ===",
        "Assess whether this finding is a genuine, in-scope vulnerability that should be remediated.",
        "You must follow strict rules to determine if a finding is a false positive.",
        "",
        "CHAIN OF THOUGHT REASONING:",
        "You MUST first use the 'chain_of_thought' field to think step-by-step before deciding on the final fields:",
        "1. Start by analyzing the details of the finding (vulnerable component, files, packages, CVE descriptions, and CVSS scores).",
        "2. Systematically evaluate the strict rules one-by-one (Rule A, Rule B, Rule C, Rule D) against the finding and system context. Output YES or NO for each rule.",
        "3. Conclude whether the finding is a false positive using only Rule A, Rule B, and Rule C. If Rule D is YES, set is_unreachable_code=True and call that out clearly, but do not set is_valid=False from reachability alone.",
        "4. Always return original_severity exactly as shown in the DATA section. If the finding is valid, assign revised_priority by starting from Original Severity / CVSS and then adjusting it strictly using the CVE Description and System Context.",
        "5. Set is_unreachable_code=True only when Reachability Analysis is explicitly FALSE. Otherwise set is_unreachable_code=False.",
        "6. Assign validity_confidence_score and priority_confidence_score on a 0.0 to 1.0 scale using the confidence rubric below.",
        "",
        "STRICT FALSE POSITIVE CRITERIA:",
        "Set an issue as a false positive (is_valid=False) ONLY if it satisfies at least one of the following three rules:",
        "  Rule A: The vulnerability is invalid due to identification error by the scanner.",
        "   - Look at the CVE description and identify the exact software product, vendor, operating system, or language runtime that the CVE actually applies to",
        "   - Compare that to the File Path and System Context fields to determine if the CVE is truly relevant to this finding",
        "  Rule B: The vulnerability is valid, but the exploit strictly requires an OS or architecture that contradicts the SystemContext.",
        "  Rule C: The vulnerability is valid, but is only present in development code (e.g. tests, specs, dev-only tools) or code that is removed before production.",
        "  Rule D: If Reachability Analysis is explicitly FALSE, the package is unreachable in this application. Set is_unreachable_code=True, but do not treat that fact alone as a false positive.",
        "",
        "If the finding does NOT satisfy any of Rule A, Rule B, or Rule C, it MUST be considered a valid vulnerability (is_valid=True).",
        "For valid vulnerabilities, decide revised_priority by starting with Original Severity / CVSS, then:",
        "  - Downgrade if the CVE requires conditions such as public network exposure that contradict the System Context.",
        "  - Upgrade if Data Sensitivity is high and the CVE Description indicates data leakage or exposure risk.",
        "  - Otherwise keep the priority aligned with the original severity and the concrete exploit conditions in the description.",
        "",
        "CONFIDENCE SCORING RUBRIC:",
        "Assign validity_confidence_score and priority_confidence_score as numbers from 0.0 to 1.0.",
        "  - validity_confidence_score = 1.0 for absolute proof such as a definitive CVE-to-technology mismatch or explicit out-of-scope evidence.",
        "  - validity_confidence_score = 0.8 for strong evidence such as a clear CVE-description mismatch.",
        "  - validity_confidence_score = 0.5 for ambiguous data, guesses, or transitive dependencies with uncertain reachability.",
        "  - priority_confidence_score = 1.0 when hard threat intel like EPSS or KEV clearly drives the priority.",
        "  - priority_confidence_score = 0.8 for strong contextual alignment between the description and System Context.",
        "  - priority_confidence_score = 0.5 when descriptions are vague and threat intel is unavailable.",
        "",
        "Assign a revised_priority using one of: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN.",
        "Always return original_severity, revised_priority, is_unreachable_code, validity_confidence_score, and priority_confidence_score.",
        "Set triage_method to 'llm'.",
    ]
    return "\n".join(lines)


def _llm_triage(
    group: VulnerabilityGroup,
    context: SystemContext,
) -> Optional[TriageResult]:
    """
    Attempt an LLM-based triage using LangChain structured output.

    Returns None if the LLM call fails or is unavailable, so the caller can
    fall back to deterministic triage.
    """
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
    except ImportError:
        logger.warning(
            "LangChain/OpenAI packages not installed.  "
            "Set TRIAGE_LLM_ENABLED=false or install langchain-openai."
        )
        return None

    model_name = os.environ.get("TRIAGE_LLM_MODEL", "gpt-4o-mini")
    try:
        llm = ChatOpenAI(model=model_name, temperature=0)
        structured_llm = llm.with_structured_output(TriageResult)
        prompt_text = _build_triage_prompt(group, context)
        result: TriageResult = invoke_with_trajectory(
            "triage.llm",
            lambda: structured_llm.invoke(prompt_text),
            prompt_text,
        )
        result.group_id = group.group_id
        result.recommended_issue_id = group.representative_issue_id
        result.triage_method = "llm"
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM triage failed (%s); falling back to deterministic.", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@traceable(name="triage.verdict", run_type="chain")
def run_triage(group: VulnerabilityGroup, context: SystemContext) -> TriageResult:
    """
    Produce a ``TriageResult`` for one ``VulnerabilityGroup``.

    When ``TRIAGE_LLM_ENABLED=true``, the LLM path is attempted first.
    Deterministic guardrails are applied afterwards regardless of which path
    produced the initial verdict.
    """
    llm_enabled = os.environ.get("TRIAGE_LLM_ENABLED", "false").lower() == "true"
    result: Optional[TriageResult] = None

    if llm_enabled:
        result = _llm_triage(group, context)

    if result is None:
        return _deterministic_triage(group, context)

    original_sev = _original_severity(group)
    is_unreachable_code = group.is_reachable is False
    final_priority, guardrail_note, is_valid, fp_reason, priority_overridden = _apply_guardrails(
        priority=result.revised_priority,
        is_valid=result.is_valid,
        false_positive_reason=result.false_positive_reason,
        context=context,
        enrichment=group.enrichment,
        original_severity=original_sev,
    )

    result.original_severity = original_sev
    result.revised_priority = final_priority
    result.is_unreachable_code = is_unreachable_code
    result.is_valid = is_valid
    result.false_positive_reason = fp_reason
    if is_unreachable_code and _UNREACHABLE_CODE_NOTE not in result.priority_reasoning:
        result.priority_reasoning = f"{result.priority_reasoning} {_UNREACHABLE_CODE_NOTE}".strip()
    if guardrail_note:
        result.priority_reasoning = result.priority_reasoning + " " + guardrail_note
    if priority_overridden:
        result.priority_confidence_score = 1.0

    return result


