"""
agent.py — Triage agent for the Phase 4.0 triage layer.

Public API
----------
run_triage(group, context) -> TriageResult
    Produce a triage verdict for one ``VulnerabilityGroup``.

    Deterministic path (always active):
      - Always valid by default (unknown evidence ≠ false positive).
      - KEV membership → clamp revised_priority to CRITICAL.
      - EPSS ≥ 0.5 or original severity HIGH/CRITICAL → at least HIGH.
      - Dev/test scope explicitly set → may mark is_valid=False with reason.

    LLM path (active when TRIAGE_LLM_ENABLED=true):
      - Uses LangChain ``with_structured_output(TriageResult)`` via OpenAI.
      - Deterministic guardrails are applied *after* LLM output.
      - triage_method is set to "llm"; falls back to "deterministic" on error.

Environment variables
---------------------
TRIAGE_LLM_ENABLED   : "true" to activate LLM path (default: off)
OPENAI_API_KEY       : Required when LLM path is active
TRIAGE_LLM_MODEL     : OpenAI model name (default: "gpt-4o-mini")
EPSS_CRITICAL_THRESHOLD : EPSS score that clamps to at least HIGH (default: 0.5)
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from uuid import UUID

from src.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
)

logger = logging.getLogger(__name__)

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


def _epss_threshold() -> float:
    try:
        return float(os.environ.get("EPSS_CRITICAL_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


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
    enrichment: Optional[CVEEnrichment],
    original_severity: Severity,
) -> tuple[Severity, str, bool, Optional[str]]:
    """
    Apply deterministic guardrails to a candidate triage verdict.

    Returns (revised_priority, reasoning_suffix, is_valid, false_positive_reason).
    """
    reasoning_parts: list[str] = []
    epss_threshold = _epss_threshold()

    # Rule 1: KEV → CRITICAL always
    if enrichment and enrichment.in_kev:
        if _SEVERITY_RANK[priority] < _SEVERITY_RANK[Severity.CRITICAL]:
            reasoning_parts.append(
                f"Clamped to CRITICAL: CVE appears in CISA KEV "
                f"(added {enrichment.kev_date_added or 'unknown date'})."
            )
        priority = Severity.CRITICAL
        # KEV membership means it IS actively exploited → must be valid
        is_valid = True
        false_positive_reason = None

    # Rule 2: EPSS ≥ threshold or original HIGH/CRITICAL → at least HIGH
    elif (enrichment and enrichment.epss >= epss_threshold) or original_severity in (
        Severity.HIGH,
        Severity.CRITICAL,
    ):
        if _SEVERITY_RANK[priority] < _SEVERITY_RANK[Severity.HIGH]:
            parts = []
            if enrichment and enrichment.epss >= epss_threshold:
                parts.append(f"EPSS={enrichment.epss:.3f} ≥ {epss_threshold}")
            if original_severity in (Severity.HIGH, Severity.CRITICAL):
                parts.append(f"original severity={original_severity.value}")
            reasoning_parts.append(
                f"Clamped to HIGH: {'; '.join(parts)}."
            )
        priority = _max_severity(priority, Severity.HIGH)

    return priority, " ".join(reasoning_parts), is_valid, false_positive_reason


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

    # Start optimistic
    is_valid = True
    false_positive_reason: Optional[str] = None
    base_priority = original_sev if original_sev != Severity.UNKNOWN else Severity.MEDIUM

    reasoning_parts: list[str] = [
        f"Original severity: {original_sev.value}."
    ]

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

    # Apply guardrails
    final_priority, guardrail_note, is_valid, false_positive_reason = _apply_guardrails(
        priority=base_priority,
        is_valid=is_valid,
        false_positive_reason=false_positive_reason,
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
            reasoning_parts.append("CVE is in CISA KEV — active exploitation confirmed.")

    return TriageResult(
        group_id=group.group_id,
        is_valid=is_valid,
        false_positive_reason=false_positive_reason,
        revised_priority=final_priority,
        priority_reasoning=" ".join(reasoning_parts),
        recommended_issue_id=group.representative_issue_id,
        triage_method="deterministic",
    )


# ---------------------------------------------------------------------------
# LLM triage path
# ---------------------------------------------------------------------------


def _build_cve_details_prompt(group: VulnerabilityGroup) -> str:
    if not group.cve_ids:
        return "  (No CVEs in this group)"

    # Map CVE IDs (case-insensitive keys) to their descriptions and CVSS scores
    cve_details: dict[str, tuple[list[str], list[str]]] = {}
    for issue in group.issues:
        cve = issue.cve_id
        if not cve:
            continue
        cve = cve.upper().strip()
        if cve not in cve_details:
            cve_details[cve] = ([], [])
        
        # Description
        desc = issue.message
        if desc:
            cve_details[cve][0].append(desc.strip())
        
        # CVSS
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
        "2. Systematically evaluate the STRICT FALSE POSITIVE CRITERIA one-by-one (Rule A, Rule B, Rule C, Rule D) against the finding and system context. Output YES or NO for each rule.",
        "3. Conclude if the finding qualifies as a false positive under any of these four rules. If you answered YES to any rule, the finding is a false positive (is_valid=False) with the corresponding reason. If you answered NO to all rules, it is a valid vulnerability (is_valid=True).",
        "4. If it is valid, evaluate the appropriate priority based on original severity, EPSS, KEV, CVSS, and the system context details.",
        "",
        "STRICT FALSE POSITIVE CRITERIA:",
        "Set an issue as a false positive (is_valid=False) ONLY if it satisfies at least one of the following four rules:",
        "  Rule A: The vulnerability is invalid due to identification error by the scanner.",
        "   - Look at the CVE description and identify the exact software product, vendor, operating system, or language runtime that the CVE actually applies to",
        "   - Compare that to the File Path and System Context fields to determine if the CVE is truly relevant to this finding",
        "  Rule B: The vulnerability is valid, but the exploit strictly requires an OS or architecture that contradicts the SystemContext.",
        "  Rule C: The vulnerability is valid, but is only present in development code (e.g. tests, specs, dev-only tools) or code that is removed before production.",
        "  Rule D: If Reachability Analysis is explicitly FALSE, the package is dead code in this application and this is a confirmed false positive.",
        "",
        "If the finding does NOT satisfy any of the above four rules, it MUST be considered a valid vulnerability (is_valid=True).",
        "For valid vulnerabilities, decide an appropriate priority score (revised_priority) based on all the provided information (original severity, CVSS, EPSS, KEV, System Context, data sensitivity, public-facing, etc.).",
        "",
        "Assign a revised_priority using one of: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN.",
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
        from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate  # type: ignore[import]
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
        result: TriageResult = structured_llm.invoke(prompt_text)
        # Ensure group_id and recommended_issue_id are consistent
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


def run_triage(group: VulnerabilityGroup, context: SystemContext) -> TriageResult:
    """
    Produce a ``TriageResult`` for one ``VulnerabilityGroup``.

    When ``TRIAGE_LLM_ENABLED=true``, the LLM path is attempted first.
    Deterministic guardrails are applied afterwards regardless of which path
    produced the initial verdict.

    Parameters
    ----------
    group:
        The vulnerability group to triage.  ``group.enrichment`` should be
        populated before calling this function.
    context:
        System-level metadata for contextualising the verdict.

    Returns
    -------
    TriageResult
        Always returns a valid ``TriageResult``; never raises.
    """
    llm_enabled = os.environ.get("TRIAGE_LLM_ENABLED", "false").lower() == "true"
    result: Optional[TriageResult] = None

    if llm_enabled:
        result = _llm_triage(group, context)

    if result is None:
        # Pure deterministic path (or LLM fallback)
        return _deterministic_triage(group, context)

    # LLM produced a result — apply deterministic guardrails on top
    original_sev = _original_severity(group)
    final_priority, guardrail_note, is_valid, fp_reason = _apply_guardrails(
        priority=result.revised_priority,
        is_valid=result.is_valid,
        false_positive_reason=result.false_positive_reason,
        enrichment=group.enrichment,
        original_severity=original_sev,
    )

    if guardrail_note:
        # Guardrail overrode the LLM — append the note to the reasoning
        result.revised_priority = final_priority
        result.priority_reasoning = result.priority_reasoning + " [Guardrail] " + guardrail_note
        result.is_valid = is_valid
        result.false_positive_reason = fp_reason

    return result
