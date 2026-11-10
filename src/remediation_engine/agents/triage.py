"""Finding grouping, enrichment, reachability, and triage."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from langsmith import traceable

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    DependencyParentContext,
    IssueSource,
    IssueType,
    LocalizedIssue,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import ProjectLanguage, resolve_project_language
from remediation_engine.runtime.agent_runtime import create_chat_model
from remediation_engine.runtime.path_policy import WorkspacePathError, resolve_repository_path
from remediation_engine.runtime.telemetry import invoke_with_trajectory
from remediation_engine.settings import AppSettings, get_runtime_settings
from remediation_engine.tools.advisories import enrich_cves
from remediation_engine.tools.identity import normalize_python_package_name, package_name_from_purl
from remediation_engine.tools.source import (
    extract_all_imports,
    language_for_path,
    load_source_bytes,
    parse_source,
)

logger = logging.getLogger(__name__)

_UNREACHABLE_CODE_NOTE = (
    "Reachability analysis shows the package is not imported by the application source code."
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
    false_positive_reason: str | None,
    context: SystemContext,
    enrichment: CVEEnrichment | None,
    original_severity: Severity,
) -> tuple[Severity, str, bool, str | None, bool]:
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
    false_positive_reason: str | None = None
    base_priority = original_sev if original_sev != Severity.UNKNOWN else Severity.MEDIUM

    reasoning_parts: list[str] = [f"Original severity: {original_sev.value}."]
    if is_unreachable_code:
        reasoning_parts.append(_UNREACHABLE_CODE_NOTE)

    # Dev/test scope: only mark FP when there is clear evidence
    is_dev_env = (context.environment or "").lower() in {"dev", "test", "ci", "development"}
    all_dev_only = group.issues and all(
        (i.file_path or "").startswith(("test/", "tests/", "spec/", "dev/")) for i in group.issues
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
                f"EPSS score: {enrichment.epss:.3f} (percentile {enrichment.epss_percentile:.3f})."
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
    sys_public = (
        "yes"
        if context.public_facing is True
        else ("no" if context.public_facing is False else "unknown")
    )
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
    settings: AppSettings | None = None,
) -> TriageResult | None:
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

    resolved_settings = settings or get_runtime_settings()
    model_name = resolved_settings.triage_llm_model
    try:
        llm = create_chat_model(
            model_name,
            settings=resolved_settings,
            reasoning_effort=getattr(resolved_settings, "triage_llm_reasoning_effort", None),
            chat_openai_class=ChatOpenAI,
        )
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
def run_triage(
    group: VulnerabilityGroup,
    context: SystemContext,
    settings: AppSettings | None = None,
) -> TriageResult:
    """
    Produce a ``TriageResult`` for one ``VulnerabilityGroup``.

    When ``TRIAGE_LLM_ENABLED=true``, the LLM path is attempted first.
    Deterministic guardrails are applied afterwards regardless of which path
    produced the initial verdict.
    """
    resolved_settings = settings or get_runtime_settings()
    llm_enabled = resolved_settings.triage_llm_enabled
    result: TriageResult | None = None

    if llm_enabled:
        result = _llm_triage(group, context, resolved_settings)

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


def _dedupe_paths(paths: list[str | None]) -> list[str]:
    """Return stable, deduplicated non-empty repo-relative paths."""
    result: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        if path in seen:
            continue
        result.append(path)
        seen.add(path)
    return result


# ---------------------------------------------------------------------------
# Key builders
# ---------------------------------------------------------------------------


def _component_from_issue(issue: VulnerabilityIssue) -> str:
    """Return the canonical SCA component identifier for an issue."""
    purl = issue.purl or ""
    is_pypi = (issue.ecosystem or "").strip().casefold() in {"python", "pypi"} or (
        purl.strip().casefold().startswith("pkg:pypi/")
    )
    component = issue.package_name or package_name_from_purl(purl) or "unknown"
    return normalize_python_package_name(component) if is_pypi else component


def _sca_key(localized_issue: LocalizedIssue) -> str:
    """Return the normalised package-occurrence key for an SCA issue."""
    issue = localized_issue.issue
    component = _component_from_issue(issue)
    file_part = localized_issue.manifest_file or issue.file_path or ""
    return f"sca:{file_part}:{component}"


def _parent_contexts(
    localized_issues: list[LocalizedIssue],
) -> list[DependencyParentContext]:
    """Return stable, deduplicated parent context for grouped SCA evidence."""
    contexts: dict[tuple[str, str, str], DependencyParentContext] = {}
    for localized in localized_issues:
        name = (localized.parent_package_name or "").strip()
        if not name:
            continue
        context = DependencyParentContext(
            package_name=name,
            package_version=localized.parent_package_version
            or localized.dependency_versions.get(name),
            declaration_type=localized.parent_declaration_type,
            manifest_file=localized.manifest_file,
            manifest_line=localized.parent_manifest_line,
            manifest_snippet=localized.parent_manifest_snippet,
            dependency_ancestry=list(localized.dependency_ancestry),
            dependency_versions=dict(localized.dependency_versions),
        )
        key = (
            context.package_name,
            context.declaration_type or "",
            context.manifest_file or "",
        )
        existing = contexts.get(key)
        if existing is None or (
            (context.package_version and not existing.package_version)
            or (context.manifest_line is not None and existing.manifest_line is None)
            or (context.manifest_snippet and not existing.manifest_snippet)
        ):
            contexts[key] = context
    return sorted(
        contexts.values(),
        key=lambda context: (
            context.package_name,
            context.declaration_type or "",
            context.package_version or "",
            context.manifest_file or "",
        ),
    )


def _merged_dependency_versions(localized_issues: list[LocalizedIssue]) -> dict[str, str]:
    """Merge resolved dependency versions without discarding populated evidence."""
    versions: dict[str, str] = {}
    for localized in localized_issues:
        for package_name, version in localized.dependency_versions.items():
            if package_name not in versions or not versions[package_name]:
                versions[package_name] = version
    return versions


def _sast_key(issue: VulnerabilityIssue) -> str:
    """Return the normalised grouping key for a SAST issue."""
    file_part = issue.file_path or ""
    rule_part = issue.rule_id or "unknown_rule"
    line_part = f"{issue.line_range.start}-{issue.line_range.end}" if issue.line_range else "0-0"
    return f"sast:{file_part}:{rule_part}:{line_part}"


# ---------------------------------------------------------------------------
# Representative-issue selection
# ---------------------------------------------------------------------------


def _field_richness(issue: VulnerabilityIssue) -> int:
    """Count non-None fields as a proxy for data richness."""
    return sum(1 for v in issue.model_dump().values() if v is not None and v != [] and v != {})


def _choose_representative(issues: list[VulnerabilityIssue]) -> VulnerabilityIssue:
    """
    Choose the most informative issue from a group.

    Priority:
    1. Has fixed_version set (most actionable for SCA).
    2. Highest field richness (most complete record).
    3. First in insertion order (stable tie-break).
    """
    with_fix = [i for i in issues if i.fixed_version]
    pool = with_fix if with_fix else issues
    return max(pool, key=_field_richness)


# ---------------------------------------------------------------------------
# SCA grouping
# ---------------------------------------------------------------------------


def _group_sca(localized_issues: list[LocalizedIssue]) -> dict[str, VulnerabilityGroup]:
    """Return triage groups from localized SCA findings without fix plans."""
    buckets: dict[str, list[LocalizedIssue]] = defaultdict(list)
    for localized_issue in localized_issues:
        buckets[_sca_key(localized_issue)].append(localized_issue)

    groups: dict[str, VulnerabilityGroup] = {}
    for key, members in buckets.items():
        seen_cves: list[str] = []
        seen_ghsas: list[str] = []
        seen_versions: list[str] = []
        seen_cve_set: set[str] = set()
        seen_ghsa_set: set[str] = set()
        seen_ver_set: set[str] = set()
        sources_set: set[IssueSource] = set()
        member_issues: list[VulnerabilityIssue] = []

        for localized_issue in members:
            issue = localized_issue.issue
            member_issues.append(issue)
            if issue.cve_id and issue.cve_id not in seen_cve_set:
                seen_cves.append(issue.cve_id)
                seen_cve_set.add(issue.cve_id)
            if issue.ghsa_id and issue.ghsa_id not in seen_ghsa_set:
                seen_ghsas.append(issue.ghsa_id)
                seen_ghsa_set.add(issue.ghsa_id)
            if issue.package_version and issue.package_version not in seen_ver_set:
                seen_versions.append(issue.package_version)
                seen_ver_set.add(issue.package_version)
            sources_set.add(issue.source)

        representative = _choose_representative(member_issues)
        group_file_paths = _dedupe_paths(
            [localized.manifest_file or localized.issue.file_path for localized in members]
        )
        group_file_path = (
            group_file_paths[0]
            if group_file_paths
            else (members[0].manifest_file or representative.file_path)
        )
        parent_contexts = _parent_contexts(members)
        exemplar_localized = members[0] if members else None
        parent_localized = next(
            (
                localized
                for context in parent_contexts
                for localized in members
                if localized.parent_package_name == context.package_name
                and (
                    localized.parent_package_version
                    or localized.dependency_versions.get(localized.parent_package_name)
                    or ""
                )
                == (context.package_version or "")
                and (localized.parent_declaration_type or "") == (context.declaration_type or "")
                and (localized.manifest_file or "") == (context.manifest_file or "")
            ),
            None,
        )
        context_localized = parent_localized or exemplar_localized
        context_parent_name = context_localized.parent_package_name if context_localized else None
        context_parent_version = context_localized.parent_package_version or (
            context_localized.dependency_versions.get(context_parent_name)
            if context_localized and context_parent_name
            else None
        )

        groups[key] = VulnerabilityGroup(
            group_id=key,
            issue_type=IssueType.SCA,
            vulnerable_component=_component_from_issue(representative),
            file_path=group_file_path,
            file_paths=group_file_paths,
            cve_ids=seen_cves,
            ghsa_ids=seen_ghsas,
            versions=seen_versions,
            dependency_ancestry=(
                list(context_localized.dependency_ancestry) if context_localized else []
            ),
            dependency_versions=_merged_dependency_versions(members),
            parent_package_name=(
                context_localized.parent_package_name if context_localized else None
            ),
            parent_package_version=context_parent_version,
            parent_declaration_type=(
                context_localized.parent_declaration_type if context_localized else None
            ),
            parent_contexts=parent_contexts,
            sources=sorted(sources_set, key=lambda source: source.value),
            representative_issue_id=representative.id,
            issues=member_issues,
            localized_issues=list(members),
            fix_plan_candidates=[],
        )
    return groups


# ---------------------------------------------------------------------------
# SAST grouping
# ---------------------------------------------------------------------------


def _group_sast(issues: list[VulnerabilityIssue]) -> dict[str, VulnerabilityGroup]:
    """Return a dict of group_id → VulnerabilityGroup for SAST issues."""
    groups: dict[str, VulnerabilityGroup] = {}
    for issue in issues:
        key = _sast_key(issue)
        if key in groups:
            # Same SAST location seen twice (e.g. duplicate rule run) — merge sources
            existing = groups[key]
            if issue.source not in existing.sources:
                existing.sources.append(issue.source)
            if issue not in existing.issues:
                existing.issues.append(issue)
            # Re-evaluate representative
            rep = _choose_representative(existing.issues)
            existing.representative_issue_id = rep.id
        else:
            groups[key] = VulnerabilityGroup(
                group_id=key,
                issue_type=IssueType.SAST,
                vulnerable_component=issue.rule_id,
                file_path=issue.file_path,
                file_paths=[issue.file_path] if issue.file_path else [],
                cve_ids=[],  # SAST findings rarely carry a CVE
                ghsa_ids=[],
                versions=[],
                sources=[issue.source],
                representative_issue_id=issue.id,
                issues=[issue],
            )

    return groups


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def group_issues(
    issues: list[VulnerabilityIssue],
    localized_sca_issues: list[LocalizedIssue] | None = None,
) -> list[VulnerabilityGroup]:
    """Group findings, retaining SCA localization and deferring fix planning.

    Args:
        issues: Scanner findings from any supported source.
        localized_sca_issues: Optional localization results keyed by issue ID.
            Missing results receive a conservative issue-path fallback.

    Returns:
        Groups ordered by deterministic group ID. SCA plan fields remain empty.
    """
    if not issues:
        return []

    sca_issues = [issue for issue in issues if issue.issue_type == IssueType.SCA]
    sast_issues = [issue for issue in issues if issue.issue_type == IssueType.SAST]
    localized_by_id = {localized.issue.id: localized for localized in (localized_sca_issues or [])}
    effective_localized = [
        localized_by_id.get(issue.id)
        or LocalizedIssue(issue=issue, manifest_file=issue.file_path, localization_confidence=0.0)
        for issue in sca_issues
    ]

    logger.debug(
        "Grouping %d issues (%d SCA, %d SAST).",
        len(issues),
        len(sca_issues),
        len(sast_issues),
    )
    all_groups: dict[str, VulnerabilityGroup] = {}
    all_groups.update(_group_sca(effective_localized))
    all_groups.update(_group_sast(sast_issues))
    result = sorted(all_groups.values(), key=lambda group: group.group_id)
    logger.debug("Produced %d groups.", len(result))
    return result


# ---------------------------------------------------------------------------
# Priority sort order
# ---------------------------------------------------------------------------

_PRIORITY_ORDER: dict[Severity, int] = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
    Severity.INFO: 4,
    Severity.UNKNOWN: 5,
}


class TriagePipelineError(RuntimeError):
    """Raised when one or more groups cannot receive a triage outcome."""

    def __init__(self, failed_group_ids: list[str], cause: BaseException) -> None:
        """Store failed group IDs and the first underlying triage exception."""
        self.failed_group_ids = tuple(failed_group_ids)
        self.cause = cause
        groups = ", ".join(self.failed_group_ids)
        super().__init__(f"triage failed for groups [{groups}]: {cause}")


class TriageSelectionError(RuntimeError):
    """Raised when a valid triage result cannot resolve an issue to execute."""

    def __init__(self, group_id: str) -> None:
        """Store the group whose valid result was internally inconsistent."""
        self.group_id = group_id
        super().__init__(
            f"triage result for group {group_id!r} has no resolvable recommended issue"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _attach_enrichment(
    groups: list[VulnerabilityGroup],
    enrichment_map: dict[str, CVEEnrichment],
) -> None:
    """
    Mutate groups in-place: attach the enrichment for the primary CVE.

    If a group has multiple CVEs, the one with the highest EPSS score (or the
    first KEV entry) is chosen as the primary enrichment.
    """
    for group in groups:
        if not group.cve_ids:
            continue

        candidates = [enrichment_map[c] for c in group.cve_ids if c in enrichment_map]
        if not candidates:
            continue

        # Prefer KEV; then highest EPSS; then first
        kev_hits = [c for c in candidates if c.in_kev]
        if kev_hits:
            group.enrichment = kev_hits[0]
        else:
            group.enrichment = max(candidates, key=lambda c: c.epss)


def _find_issue_by_id(
    groups: list[VulnerabilityGroup],
    issue_id,
) -> VulnerabilityIssue | None:
    """Locate a VulnerabilityIssue by UUID across all group members."""
    for group in groups:
        for issue in group.issues:
            if issue.id == issue_id:
                return issue
    return None


def _fallback_localized_issue(
    issue: VulnerabilityIssue,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> LocalizedIssue:
    """Build a minimal issue localization while retaining Maven fail-closed routing."""
    is_maven = project_language == ProjectLanguage.JAVA and (
        (issue.ecosystem or "").strip().lower() == "maven"
        or (issue.purl or "").strip().lower().startswith("pkg:maven/")
    )
    return LocalizedIssue(
        issue=issue,
        manifest_file=None if is_maven else issue.file_path,
        package_manager="maven" if is_maven else None,
        localization_confidence=0.0,
    )


@traceable(name="triage.grouping", run_type="chain")
def _run_grouping(
    issues: list[VulnerabilityIssue],
    localized_sca_issues: list[LocalizedIssue],
) -> list[VulnerabilityGroup]:
    """Group normalized findings with their localization evidence."""
    return group_issues(issues, localized_sca_issues=localized_sca_issues)


@traceable(name="triage.cve_enrichment", run_type="chain")
def _run_enrichment(
    cve_ids: list[str],
    settings: AppSettings | None = None,
) -> dict[str, CVEEnrichment]:
    """Fetch threat-intelligence enrichment for the pipeline's CVE set."""
    if settings is None:
        return enrich_cves(cve_ids)
    return enrich_cves(cve_ids, settings=settings)


@traceable(name="triage.reachability", run_type="chain")
def _run_reachability(
    groups: list[VulnerabilityGroup],
    repo_root: str,
    project_language: ProjectLanguage,
) -> None:
    """Annotate SCA groups with deterministic reachability evidence."""
    analyze_reachability(groups, repo_root, project_language=project_language)


@traceable(name="triage.sca_localization", run_type="chain")
def _localize_sca_issues(
    sca_issues: list[VulnerabilityIssue],
    repo_root: str | None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> list[LocalizedIssue]:
    """Locate SCA findings without invoking advisory fix planning."""
    if not sca_issues:
        return []

    from remediation_engine.tools.manifest_locator import locate_from_issue

    repo_path = Path(repo_root) if repo_root and Path(repo_root).exists() else None
    localized_issues: list[LocalizedIssue] = []
    for original_issue in sca_issues:
        issue = original_issue
        ecosystem = (issue.ecosystem or "").strip().casefold()
        purl = (issue.purl or "").strip().casefold()
        is_pypi = (
            project_language == ProjectLanguage.PYTHON
            or ecosystem in {"python", "pypi"}
            or purl.startswith("pkg:pypi/")
        )
        if is_pypi:
            package_name = issue.package_name or package_name_from_purl(issue.purl) or ""
            issue = issue.model_copy(
                update={
                    "ecosystem": "pypi",
                    "package_name": normalize_python_package_name(package_name),
                }
            )

        localized_issue = _fallback_localized_issue(issue, project_language)
        if repo_path is not None:
            try:
                if is_pypi:
                    from remediation_engine.tools.python_manifest_locator import (
                        locate_from_issue as locate_python_from_issue,
                    )

                    localized_issue = locate_python_from_issue(issue, repo_path)
                else:
                    localized_issue = locate_from_issue(
                        issue,
                        repo_path,
                        project_language=project_language,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SCA localization failed for %s (%s); using fallback localization.",
                    issue.id,
                    exc,
                )
        localized_issues.append(localized_issue)
    return localized_issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@traceable(name="triage.pipeline", run_type="chain")
def run_triage_pipeline(
    issues: list[VulnerabilityIssue],
    system_context: SystemContext,
    repo_root: str | None = None,
    settings: AppSettings | None = None,
) -> list[tuple[VulnerabilityGroup, TriageResult]]:
    """
    Run the full triage pipeline on a flat list of ``VulnerabilityIssue`` records.

    Parameters
    ----------
    issues:
        All issues from any combination of scanners.
    system_context:
        Caller-supplied scan session metadata.
    repo_root:
        Optional absolute path to the repository. When present and it exists,
        SCA reachability analysis is run before triage.

    Returns
    -------
    List[Tuple[VulnerabilityGroup, TriageResult]]
        One pair per group.  Includes *all* groups — valid and invalid.
        Use ``select_issues_for_remediation`` to filter to actionable ones.
    """
    if not issues:
        logger.info("Triage pipeline: no issues provided, returning empty.")
        return []

    logger.info("Triage pipeline: processing %d issues.", len(issues))

    sca_issues = [issue for issue in issues if issue.issue_type == IssueType.SCA]

    primary_language = system_context.primary_language
    project_language = (
        resolve_project_language(Path(repo_root or "."), primary_language)
        if primary_language is not None
        else ProjectLanguage.NODEJS
    )
    localized_sca_issues = _localize_sca_issues(sca_issues, repo_root, project_language)

    # Step 1: Group localized findings; fix plans are materialized by the Supervisor.
    groups = _run_grouping(issues, localized_sca_issues)
    logger.info("Triage pipeline: produced %d groups.", len(groups))

    # Step 2: Collect all unique CVE IDs for bulk enrichment
    all_cve_ids: list[str] = []
    seen: set = set()
    for group in groups:
        for cve in group.cve_ids:
            if cve not in seen:
                all_cve_ids.append(cve)
                seen.add(cve)

    # Step 3: Enrich CVEs (failure-safe)
    enrichment_map: dict[str, CVEEnrichment] = {}
    if all_cve_ids:
        enrichment_map = _run_enrichment(all_cve_ids, settings=settings)
        logger.info(
            "Triage pipeline: enriched %d/%d CVEs.",
            len(enrichment_map),
            len(all_cve_ids),
        )

    # Step 4: Attach enrichment to groups
    _attach_enrichment(groups, enrichment_map)

    # Step 5: Reachability analysis for SCA groups (failure-safe)
    if repo_root and Path(repo_root).exists():
        _run_reachability(groups, repo_root, project_language)

    # Step 6: Triage each group
    results: list[tuple[VulnerabilityGroup, TriageResult]] = []
    failed_group_ids: list[str] = []
    first_error: BaseException | None = None
    for group in groups:
        try:
            triage_result = (
                run_triage(group, system_context)
                if settings is None
                else run_triage(group, system_context, settings=settings)
            )
            results.append((group, triage_result))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Triage failed for group %s (%s); aborting result construction.",
                group.group_id,
                exc,
            )
            failed_group_ids.append(group.group_id)
            first_error = first_error or exc

    if failed_group_ids and first_error is not None:
        raise TriagePipelineError(failed_group_ids, first_error) from first_error

    valid_count = sum(1 for _, r in results if r.is_valid)
    logger.info("Triage pipeline: %d/%d groups are valid.", valid_count, len(groups))
    return results


def select_issues_for_remediation(
    results: list[tuple[VulnerabilityGroup, TriageResult]],
) -> list[VulnerabilityIssue]:
    """
    Select one representative ``VulnerabilityIssue`` per valid group.

    Invalid (false-positive) groups are excluded.  The remaining issues are
    sorted by revised priority (CRITICAL first).

    Parameters
    ----------
    results:
        Output of ``run_triage_pipeline``.

    Returns
    -------
    List[VulnerabilityIssue]
        Ready to pass one-by-one to ``run_remediation`` in the existing graph.
    """
    selected: list[tuple[Severity, VulnerabilityIssue]] = []

    for group, triage in results:
        if not triage.is_valid:
            logger.debug(
                "Skipping group %s (false positive: %s).",
                group.group_id,
                triage.false_positive_reason,
            )
            continue

        # Find the recommended issue among group members
        issue = _find_issue_by_id([group], triage.recommended_issue_id)
        if issue is None:
            # Fallback: use the representative issue
            issue = _find_issue_by_id([group], group.representative_issue_id)
        if issue is None and group.issues:
            issue = group.issues[0]

        if issue is None:
            raise TriageSelectionError(group.group_id)
        selected.append((triage.revised_priority, issue))

    # Sort by priority (lower _PRIORITY_ORDER value = higher urgency)
    selected.sort(key=lambda t: _PRIORITY_ORDER.get(t[0], 99))
    return [issue for _, issue in selected]


_SOURCE_SUFFIXES = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
_EXCLUDED_PARTS = {"node_modules", ".git", "dist", "build"}
_MODULE_SPECIFIER_PATTERNS = (
    re.compile(r"\bfrom\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"\bimport\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]"),
)


def _imported_package_names(statement: str) -> set[str]:
    """Return all npm package identities referenced by one import statement."""
    packages: set[str] = set()
    for pattern in _MODULE_SPECIFIER_PATTERNS:
        for match in pattern.finditer(statement):
            module = match.group(1)
            if not module or module.startswith((".", "/", "#", "node:")):
                continue
            segments = module.split("/")
            if segments[0].startswith("@"):
                package_name = "/".join(segments[:2]) if len(segments) > 1 else None
            else:
                package_name = segments[0]
            if package_name:
                packages.add(package_name)
    return packages


def _is_pypi_issue_group(group: VulnerabilityGroup) -> bool:
    """Return whether an SCA group is explicitly identified as PyPI."""
    for issue in group.issues:
        ecosystem = str(issue.ecosystem or "").strip().casefold()
        purl = str(issue.purl or "").strip().casefold()
        if ecosystem in {"pypi", "python"} or purl.startswith("pkg:pypi/"):
            return True
    return False


def _load_direct_dependencies(repo_root: Path) -> set[str]:
    """Best-effort load of direct dependencies from package.json."""
    try:
        package_json = resolve_repository_path(repo_root, "package.json")
    except WorkspacePathError:
        return set()
    if not package_json.is_file():
        return set()

    try:
        raw = package_json.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reachability: failed to parse %s: %s", package_json, exc)
        return set()

    if not isinstance(data, dict):
        return set()

    direct_deps: set[str] = set()
    for key in ("dependencies", "devDependencies"):
        deps = data.get(key)
        if isinstance(deps, dict):
            direct_deps.update(str(name) for name in deps)
    return direct_deps


def _iter_source_files(repo_root: Path) -> Iterable[Path]:
    """Yield application source files while skipping generated/vendor paths."""
    for path in repo_root.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


def _collect_global_imports(repo_root: Path) -> set[str]:
    """Aggregate all import/require statements across application source files."""
    global_imports: set[str] = set()

    for source_file in _iter_source_files(repo_root):
        try:
            language = language_for_path(str(source_file))
            if language is None:
                continue

            source_bytes = load_source_bytes(source_file)
            if source_bytes is None:
                continue

            tree = parse_source(source_bytes, language)
            if tree is None:
                logger.warning(
                    "Reachability: parse returned no AST for %s; skipping.",
                    source_file,
                )
                continue

            imports = extract_all_imports(tree.root_node, source_bytes)
            global_imports.update(
                package_name
                for statement in imports
                for package_name in _imported_package_names(statement)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reachability: failed to analyze %s: %s",
                source_file,
                exc,
            )

    return global_imports


def analyze_reachability(
    groups: list[VulnerabilityGroup],
    repo_root: str | Path,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> None:
    """Mutate SCA groups in place with a deterministic reachability signal."""
    root = Path(repo_root).resolve()
    if not root.exists():
        logger.warning("Reachability: repo root does not exist: %s", root)
        return

    maven_groups: set[int] = set()
    if project_language == ProjectLanguage.JAVA:
        for group in groups:
            if group.issue_type != IssueType.SCA:
                continue
            if any(
                (issue.ecosystem or "").strip().lower() == "maven"
                or (issue.purl or "").strip().lower().startswith("pkg:maven/")
                for issue in group.issues
            ):
                group.is_reachable = None
                maven_groups.add(id(group))
        if all(group.issue_type != IssueType.SCA or id(group) in maven_groups for group in groups):
            return

    npm_groups: list[VulnerabilityGroup] = []
    for group in groups:
        if group.issue_type != IssueType.SCA or id(group) in maven_groups:
            continue
        if _is_pypi_issue_group(group):
            # JavaScript import names and package.json declarations do not
            # establish whether a PyPI distribution is used by Python code.
            group.is_reachable = None
        else:
            npm_groups.append(group)
    if not npm_groups:
        return

    root = Path(repo_root)
    direct_deps = _load_direct_dependencies(root)
    global_imports = _collect_global_imports(root)

    for group in npm_groups:
        component = (group.vulnerable_component or "").strip()
        if component and component in global_imports:
            group.is_reachable = True
        elif component in direct_deps:
            group.is_reachable = False
        else:
            group.is_reachable = None
