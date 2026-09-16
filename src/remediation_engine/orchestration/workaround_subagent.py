"""
Sequential Workaround Subagent for Phase 5 code-security rewrites.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    NoFixMitigationStage,
    RemediationTask,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundEditSet,
    WorkaroundPhase,
    WorkaroundReplayPlan,
    WorkaroundValidationStatus,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration._tool_support import (
    _detect_newline_style,
    _normalise_newlines,
    _restore_newlines,
)
from remediation_engine.orchestration.context_manager import ContextManager
from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.state import SubagentState
from remediation_engine.orchestration.subagent_runtime import (
    has_successful_validation_gate,
    has_tool_call_before_first_successful_edit,
    run_bounded_subagent_loop,
)
from remediation_engine.orchestration.task_utils import (
    create_skinny_subagent_group,
    filter_constraints_ledger,
    select_package_fix_plan,
)
from remediation_engine.orchestration.tools_manifest import (
    _is_allowlisted_no_fix_package_file,
    _is_prohibited_target,
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
    return create_skinny_subagent_group(group, keep_identifiers=5)


def _filter_constraints_ledger(
    constraints_ledger: list[str], target_group: VulnerabilityGroup
) -> list[str]:
    """Filter ledger to only include constraints matching the target component."""
    return filter_constraints_ledger(constraints_ledger, target_group)


_QA_ERROR_MARKER = re.compile(
    r"(?:error|exception|failed|failure|not\s+a\s+function|undefined|cannot|invalid|missing|required\s+option|not\s+exported|cannot\s+find)",
    re.IGNORECASE,
)


def _clean_prompt_snippet(value: str, max_chars: int = 240) -> str:
    """Normalize one extracted QA or vulnerability snippet for prompt use."""
    cleaned = re.sub(r"\s+", " ", value).strip(" `\"'")
    cleaned = cleaned.replace("`", "").replace('"', "")
    return cleaned[:max_chars].strip()


def _clean_prompt_log(value: str, max_chars: int = 1600) -> str:
    """Normalize multiline QA logs while retaining the details of each line."""
    lines = []
    for raw_line in str(value or "").splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if line:
            lines.append(line)
    cleaned = "\n".join(lines)
    if len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip() + "\n... (truncated)"
    return cleaned


def _restore_attempt_file(
    sandbox: DockerSandbox,
    file_path: str,
    plan_state: Mapping[str, Any],
    *,
    stage_snapshots: Mapping[str, str] | None = None,
    stage_absent_paths: Sequence[str] = (),
    allow_stage_baseline: bool = False,
) -> None:
    """Restore one file from the current attempt's recorded pre-edit state.

    The sandbox exposes read, write, and command operations but no generic
    ``revert_file`` method. Restore from the attempt snapshot first so a
    failed retry cannot erase another group's accepted edit. The stage
    baseline is used only for an explicit stage reset or when it is the only
    compatible anchor available.

    Args:
        sandbox: Active Docker workspace sandbox.
        file_path: Workspace-relative file path to restore.
        plan_state: Workaround execution state containing snapshots.
        stage_snapshots: Optional pre-stage file contents for compatibility.
        stage_absent_paths: Optional paths that did not exist at stage start.
        allow_stage_baseline: Whether the stage baseline may be used.

    Raises:
        RuntimeError: If no safe pre-edit snapshot is available.
    """
    normalized = str(file_path).replace("\\", "/").lstrip("/")
    attempt_snapshots = plan_state.get("attempt_file_snapshots", {}) or {}
    if isinstance(attempt_snapshots, Mapping):
        snapshot = attempt_snapshots.get(normalized)
        if isinstance(snapshot, str):
            sandbox.write_file(normalized, snapshot)
            return

    attempt_absent_paths = {
        str(path).replace("\\", "/").lstrip("/")
        for path in plan_state.get("attempt_absent_paths", []) or []
    }
    if normalized in attempt_absent_paths:
        result = sandbox.run(f"rm -f -- {shlex.quote(normalized)}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"could not remove newly created file '{normalized}' "
                f"(exit {result.exit_code}): {result.stderr}"
            )
        return

    if allow_stage_baseline:
        if isinstance(stage_snapshots, Mapping):
            snapshot = stage_snapshots.get(normalized)
            if isinstance(snapshot, str):
                sandbox.write_file(normalized, snapshot)
                return
        if normalized in {str(path).replace("\\", "/").lstrip("/") for path in stage_absent_paths}:
            result = sandbox.run(f"rm -f -- {shlex.quote(normalized)}")
            if result.exit_code != 0:
                raise RuntimeError(
                    f"could not remove absent stage-baseline file '{normalized}' "
                    f"(exit {result.exit_code}): {result.stderr}"
                )
            return

    raise RuntimeError(f"no pre-edit snapshot recorded for '{normalized}'")


def _qa_failure_log_snippet(
    qa_evidence: Any,
    previous_feedback: str | None,
    max_chars: int = 1600,
) -> str:
    """Return concrete QA failure logs for the workaround prompt.

    The raw excerpt is preferred because it contains the test names and
    expected/actual output. Structured fields are used as a fallback when QA
    could not capture a raw log excerpt.
    """
    if qa_evidence:
        raw_excerpt = str(getattr(qa_evidence, "raw_excerpt", "") or "").strip()
        if raw_excerpt:
            return _clean_prompt_log(raw_excerpt, max_chars=max_chars)

        fallback_values = [
            *(getattr(qa_evidence, "failed_tests", None) or []),
            *(getattr(qa_evidence, "exact_diagnostics", None) or []),
        ]
        fallback = "\n".join(str(value) for value in fallback_values if value)
        if fallback:
            return _clean_prompt_log(fallback, max_chars=max_chars)

    return _clean_prompt_log(previous_feedback or "", max_chars=max_chars)


_QA_GENERIC_STATUS_RE = re.compile(
    r"^(?:"
    r"(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b.*\b(?:failed|passed)\b|"
    r"(?:detected\s+failures|failing\s+tests|stdout\s+tail|stderr\s+tail)\s*:|"
    r"(?:\d+\s+)?(?:tests?|specs?)\s+(?:failed|failing|passed)\b|"
    r"(?:\d+\s+)?(?:failed|failing|passed)(?:\s+(?:tests?|specs?))?\s*$|"
    r"exit(?:\s+code)?\s*[:=]?\s*\d+\b"
    r")",
    re.IGNORECASE,
)
_QA_SEARCH_DETAIL_RE = re.compile(
    r"(?:\b[A-Za-z_][\w.]*(?:Error|Exception)\s*:|"
    r"\bassert(?:ion)?(?:error)?\b|\bexpected\b|\bactual\b|"
    r"not\s+a\s+function|cannot\s+find|\bundefined\b|\binvalid\b|"
    r"required\s+option|not\s+exported)",
    re.IGNORECASE,
)


def _normalise_qa_search_line(value: str) -> str:
    """Normalize one QA line before using it as a search term."""
    line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", str(value or "")).strip()
    return re.sub(r"\s+", " ", line)


def _is_generic_qa_status(value: str) -> bool:
    """Return whether a QA line only describes the test runner outcome."""
    return bool(_QA_GENERIC_STATUS_RE.search(_normalise_qa_search_line(value)))


def _extract_qa_search_diagnostic(
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> str:
    """Extract a substantive test failure detail for the first web query.

    Runner summaries such as ``npm test FAILED (exit 2)`` are intentionally
    ignored. The query instead uses the failing test title and the first
    assertion or exception detail available in the structured QA evidence or
    raw test excerpt.
    """
    qa_evidence = workaround_context.qa_evidence if workaround_context else None
    failed_tests = []
    candidate_values = [previous_feedback or ""]
    if qa_evidence:
        failed_tests = [
            _normalise_qa_search_line(value)
            for value in (qa_evidence.failed_tests or [])
            if value and not _is_generic_qa_status(value)
        ]
        candidate_values = [
            *(qa_evidence.exact_diagnostics or []),
            qa_evidence.raw_excerpt or "",
            previous_feedback or "",
        ]

    detail = ""
    for value in candidate_values:
        for raw_line in str(value or "").splitlines():
            line = _normalise_qa_search_line(raw_line)
            if not line or _is_generic_qa_status(line):
                continue
            if _QA_SEARCH_DETAIL_RE.search(line):
                detail = _clean_prompt_snippet(line, max_chars=180)
                break
        if detail:
            break

    title = failed_tests[0] if failed_tests else ""
    if not title:
        for value in candidate_values:
            for raw_line in str(value or "").splitlines():
                line = _normalise_qa_search_line(raw_line)
                if line and not _is_generic_qa_status(line):
                    title = _clean_prompt_snippet(line, max_chars=180)
                    break
            if title:
                break

    if title and detail and title.casefold() != detail.casefold():
        return _clean_prompt_snippet(f"{title}; {detail}", max_chars=180)
    return detail or title


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


def _preferred_targeted_test_files(qa_evidence: Any) -> list[str]:
    """Extract source test files from structured QA locations and affected files."""
    if qa_evidence is None:
        return []

    candidates: list[str] = []
    for raw_value in [
        *(getattr(qa_evidence, "failed_tests", []) or []),
        *(getattr(qa_evidence, "source_locations", []) or []),
        *(getattr(qa_evidence, "affected_files", []) or []),
    ]:
        value = str(raw_value or "").strip().replace("\\", "/")
        value = re.sub(r":\d+(?::\d+)?(?:$|\b).*", "", value)
        # QA failure summaries may encode a test as ``path: test title``.
        # Keep only the repository-relative path here; the title is supplied
        # separately to the validation tool by the worker.  Without this
        # split, the title becomes part of the preferred filesystem path and
        # every validation attempt is rejected as a non-canonical target.
        path_prefix, separator, _test_title = value.partition(":")
        if separator and re.search(
            r"(?:^|/)(?:test|tests|__tests__)/|\.(?:test|spec)\.[^.]+$",
            path_prefix,
            re.IGNORECASE,
        ):
            value = path_prefix
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


def _primary_non_test_source_files(qa_evidence: Any) -> list[str]:
    """Extract primary non-test source files from structured QA locations."""
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
        elif value.startswith("workspace/"):
            value = value[len("workspace/") :]
        value = value.strip(" `\"'()[]{}.,;")
        if value.startswith("build/"):
            value = value[len("build/") :]
        if not value:
            continue
        if (
            "/test/" in f"/{value}/"
            or "/tests/" in f"/{value}/"
            or "/__tests__/" in f"/{value}/"
            or re.search(r"\.(?:test|spec)\.[^.]+$", value, re.IGNORECASE)
        ):
            continue
        if value not in candidates:
            candidates.append(value)
    return candidates[:3]


def _extract_vulnerability_mechanism(group: VulnerabilityGroup) -> str:
    """Extract a compact vulnerability mechanism before the detailed fix guidance."""
    for issue in getattr(group, "issues", []) or []:
        message = getattr(issue, "message", None)
        if not isinstance(message, str) or not message.strip():
            continue

        mechanism = message
        for section_marker in ("### Am I affected?", "### How to fix that?"):
            mechanism = mechanism.split(section_marker, 1)[0]
        mechanism = _clean_prompt_snippet(mechanism, max_chars=600)
        if mechanism:
            return mechanism

    fix_plan = select_package_fix_plan(group).plan
    instruction = getattr(fix_plan, "instruction", None)
    if isinstance(instruction, str) and instruction.strip():
        return _clean_prompt_snippet(instruction, max_chars=600)
    return ""


def _workaround_search_recommendation(
    target_task: Any,
    target_group: VulnerabilityGroup,
    workaround_context: WorkaroundContext | None,
    previous_feedback: str | None,
) -> _SearchQueryRecommendation:
    """Build a first web query from the current remediation evidence."""
    fix_plan = select_package_fix_plan(target_group, target_task.strategy).plan
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
    is_qa_phase = (
        workaround_context is not None
        and workaround_context.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
    )

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
    diagnostic = _extract_qa_search_diagnostic(workaround_context, previous_feedback)
    diagnostic_term = _search_query_term(diagnostic, max_chars=180)

    selected_version = getattr(target_task, "selected_version", None)
    if not selected_version and fix_plan is not None:
        selected_version = getattr(fix_plan, "fixed_version", None)
    version_term = f"version {selected_version}" if selected_version else ""

    no_fix_stage = getattr(workaround_context, "no_fix_stage", None)
    if isinstance(no_fix_stage, NoFixMitigationStage):
        no_fix_stage = no_fix_stage.value
    if no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL.value:
        return _SearchQueryRecommendation(
            scenario="no_fix_package_removal",
            initial_query=_query_parts(
                component,
                "package.json manifest imports call sites",
                "package removal",
            ),
            rationale=(
                "Prioritize local manifests, imports, and call sites so the worker can "
                "remove only the authorized direct declaration and its dependent usage."
            ),
            follow_up_query="",
        )
    if no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value:
        return _SearchQueryRecommendation(
            scenario="no_fix_vulnerable_code_removal",
            initial_query=_query_parts(
                component,
                identifier_terms,
                mechanism,
                "installed package source vulnerable API call path",
            ),
            rationale=(
                "Use the advisory identifiers, vulnerable mechanism, and installed-package "
                "source evidence to trace and remove direct and indirect vulnerable call paths."
            ),
            follow_up_query=diagnostic_term,
        )

    if is_qa_phase or test_failure:
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
            follow_up_query="",
        )

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
            follow_up_query="",
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
            follow_up_query="",
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
        follow_up_query="",
    )


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
            "=== RECOMMENDED INITIAL SEARCH QUERY ===",
            f"Scenario: {recommendation.scenario}",
            f"Query: {recommendation.initial_query}",
            f"Rationale: {recommendation.rationale}",
            "Use the recommended query for the first search_web call. Refine your query only after receiving an inadequate result or validation failure.",
            "Prefer authoritative advisory, maintainer, migration-guide, or source-repository results.",
        ]
    )


_WORKAROUND_COMMON_STATIC_INSTRUCTIONS = """You are a code security specialist
operating inside a shared Docker workspace. Use only the provided scoped workspace
tools and relative paths. Perform local inspection before external research, and
use authoritative advisory, maintainer, migration-guide, or source-repository
guidance when web research is necessary.

Follow Investigate -> Plan -> Execute -> Validate. Record an evidence-backed plan
before edits. Make one minimal semantic source patch per iteration with
deterministic_apply_edit_set, and include an import plus all causally related
call-site changes in the same atomic edit set. Call validate_workaround
immediately after each edit set. Do not weaken the security invariant or modify
unrelated code."""

_WORKAROUND_INITIAL_MITIGATION_INSTRUCTIONS = """For INITIAL_MITIGATION, inspect
local code first, then perform the initial search_web call once and read specific
authoritative results with read_web_page. Adapt the security invariant to local
code; do not copy an external patch blindly."""

_WORKAROUND_QA_REGRESSION_REPAIR_INSTRUCTIONS = """For QA_REGRESSION_REPAIR, the
dependency update and replayed edits are already present. Trace the failure from
the test evidence to modified source. Use search_web only for migration guidance
or exact breaking-change diagnostics. If a targeted test has a proven
infrastructure-only failure, one existing alternative test may be substituted
once with a recorded mapping; never substitute for an assertion, syntax, type, or
application-runtime failure."""

_WORKAROUND_NO_FIX_PACKAGE_REMOVAL_INSTRUCTIONS = """For NO_FIX PACKAGE_REMOVAL,
record a package-removal plan with the configured authorized manifest paths.
remove_no_fix_dependency is the only manifest or lockfile operation. Remove
source imports and dependent usage when local inspection shows them, then
validate the cumulative source patch. Never manually edit lockfile nodes, bump
versions, or modify tests. If no removable direct declaration exists, report
NOT_APPLICABLE and surrender for vulnerable-code removal."""

_WORKAROUND_NO_FIX_VULNERABLE_CODE_INSTRUCTIONS = """For NO_FIX
VULNERABLE_CODE_REMOVAL, keep the vulnerable package installed and perform a
source-only removal of the vulnerable API and its callers. Never modify manifests,
lockfiles, dependency versions, or test files. Validate with runtime smoke and a
separate targeted test where available."""

_WORKAROUND_EDIT_CHECKPOINT_INSTRUCTIONS = """The workspace is the baseline plus
previously validated or replayed edit sets. Each deterministic_apply_edit_set is
provisional until validate_workaround returns PASS. CODE_FAILURE rolls back the
whole pending set; the next plan must re-include every required change. PASS
promotes it into the cumulative patch. INFRA_FAILURE or BLOCKED retains the
pending set for recovery; never re-apply the same patch. Always report
modified_files cumulatively."""

_WORKAROUND_VALIDATION_INSTRUCTIONS = """Runtime smoke must import a lightweight
source module, never a test/spec file or build/dist artifact, and remain separate
from the targeted test. Supply the complete cumulative modified-file list to
validate_workaround. Resolve infrastructure-only validation failures through the
permitted alternative targeted-test path; do not edit for infrastructure noise."""

_WORKAROUND_PROHIBITIONS = """Never modify tests to make assertions pass. Never
manually edit or delete lockfile nodes, bump library versions, or use absolute
paths. Before source edits or package removal, call record_plan. For ordinary
workarounds dependency manifests remain prohibited; for NO_FIX PACKAGE_REMOVAL
only the configured package-removal operation may change its authorized manifest
and lockfile paths."""

_WORKAROUND_STATIC_INSTRUCTIONS = "\n\n".join(
    [
        _WORKAROUND_COMMON_STATIC_INSTRUCTIONS,
        _WORKAROUND_INITIAL_MITIGATION_INSTRUCTIONS,
        _WORKAROUND_QA_REGRESSION_REPAIR_INSTRUCTIONS,
        _WORKAROUND_NO_FIX_PACKAGE_REMOVAL_INSTRUCTIONS,
        _WORKAROUND_NO_FIX_VULNERABLE_CODE_INSTRUCTIONS,
        _WORKAROUND_EDIT_CHECKPOINT_INSTRUCTIONS,
        _WORKAROUND_VALIDATION_INSTRUCTIONS,
        _WORKAROUND_PROHIBITIONS,
    ]
)


def _build_workaround_prompt(
    target_task: RemediationTask,
    target_group: VulnerabilityGroup,
    constraints_ledger: list[str] | None = None,
    previous_feedback: str | None = None,
    current_replay_plan: WorkaroundReplayPlan | None = None,
    vulnerability_mechanism: str | None = None,
    workaround_context: WorkaroundContext | None = None,
) -> str:
    """Build the dynamic human context for one workaround attempt."""
    constraints_ledger = list(constraints_ledger or [])
    fix_plan = select_package_fix_plan(target_group, target_task.strategy).plan
    vulnerability_mechanism = (
        _clean_prompt_snippet(vulnerability_mechanism, max_chars=600)
        if vulnerability_mechanism
        else _clean_prompt_snippet(_extract_vulnerability_mechanism(target_group), max_chars=600)
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
    no_fix_stage = getattr(workaround_context, "no_fix_stage", None)
    if no_fix_stage is None:
        no_fix_stage = getattr(target_task, "no_fix_stage", None)
    if isinstance(no_fix_stage, NoFixMitigationStage):
        no_fix_stage = no_fix_stage.value
    if no_fix_stage not in {
        None,
        NoFixMitigationStage.PACKAGE_REMOVAL.value,
        NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value,
        NoFixMitigationStage.UNFIXABLE.value,
    }:
        no_fix_stage = None

    comp_name = getattr(target_group, "vulnerable_component", "") or "component"
    cve_label = (
        target_group.cve_ids[0]
        if getattr(target_group, "cve_ids", None)
        else (target_group.ghsa_ids[0] if getattr(target_group, "ghsa_ids", None) else "")
    )
    cves = ", ".join(getattr(target_group, "cve_ids", []) or []) or "(none)"
    ghsas = ", ".join(getattr(target_group, "ghsa_ids", []) or []) or "(none)"
    selected_version = getattr(target_task, "selected_version", None)
    if not selected_version and fix_plan is not None:
        selected_version = getattr(fix_plan, "fixed_version", None)

    sections = [
        "DYNAMIC WORKAROUND CONTEXT (current attempt data):",
        f"WORKFLOW PHASE: {phase.value.upper()}",
        f"NO_FIX MITIGATION STAGE: {no_fix_stage or 'not applicable'}",
        f"Target Package: {comp_name}"
        + (f" (version: {selected_version})" if selected_version else ""),
        f"CVEs: {cves}",
        f"GHSAs: {ghsas}",
        f"Vulnerability Identifier: {cve_label or 'none'}",
        f"Vulnerability Mechanism: {vulnerability_mechanism or 'not provided'}",
        f"Task Instruction: {_clean_prompt_snippet(getattr(target_task, 'instruction', '') or 'Apply defensive code fix.', max_chars=600)}",
    ]

    if no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL.value:
        manifest_paths = [
            str(path).replace("\\", "/").lstrip("/")
            for path in [
                *(getattr(target_group, "file_paths", []) or []),
                getattr(target_group, "file_path", None),
                *(
                    getattr(issue, "manifest_file", None)
                    for issue in getattr(target_group, "localized_issues", []) or []
                ),
            ]
            if str(path).strip()
        ]
        sections.extend(
            [
                "=== ACTIVE NO_FIX PACKAGE REMOVAL DATA ===",
                f"Authorized manifest paths: {', '.join(dict.fromkeys(manifest_paths)) or 'none supplied'}",
                f"Configured vulnerable package: {comp_name}",
            ]
        )
    elif no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value:
        sections.append("=== ACTIVE NO_FIX VULNERABLE-CODE REMOVAL DATA ===")

    if current_replay_plan is not None:
        replay_findings = sorted(str(key) for key in current_replay_plan.investigation_findings)
        replay_lines = [
            "=== REPLAY PLAN ===",
            f"Source attempt: {current_replay_plan.source_attempt_id or 'unknown'}",
            f"Successful edit sets already replayed: {len(current_replay_plan.successful_edit_sets)}",
            f"Validated files: {', '.join(current_replay_plan.validated_files[:12]) or 'none'}",
            f"Planned targets: {', '.join(current_replay_plan.planned_targets[:12]) or 'none'}",
            f"Security invariants: {'; '.join(current_replay_plan.security_invariants[:4]) or 'none'}",
            f"Diagnosed root causes: {'; '.join(current_replay_plan.diagnosed_root_causes[:4]) or 'none'}",
            f"Prior finding keys: {', '.join(replay_findings[:12]) or 'none'}",
            f"Validation calls: {current_replay_plan.validation_calls}",
        ]
        if current_replay_plan.final_selected_targeted_test:
            replay_lines.append(
                f"Selected targeted test: {current_replay_plan.final_selected_targeted_test}"
            )
        sections.append("\n".join(replay_lines))

    if previous_feedback:
        sections.append(
            "=== PREVIOUS FEEDBACK ===\n" + _clean_prompt_log(previous_feedback, max_chars=1200)
        )

    if phase == WorkaroundPhase.QA_REGRESSION_REPAIR:
        qa_evidence = workaround_context.qa_evidence if workaround_context else None
        qa_lines = ["=== QA FAILURE EVIDENCE ==="]
        diagnostic_logs = _qa_failure_log_snippet(qa_evidence, previous_feedback)
        if diagnostic_logs:
            qa_lines.append(f"Diagnostic:\n{diagnostic_logs}")
        primary_sources = _primary_non_test_source_files(qa_evidence)
        if primary_sources:
            qa_lines.append("Primary Source Files:")
            qa_lines.extend(f"  - {path}" for path in primary_sources)
        preferred_tests = _preferred_targeted_test_files(qa_evidence)[:1]
        if preferred_tests:
            qa_lines.append(f"Targeted Test File: {preferred_tests[0]}")
        tool_events = getattr(workaround_context, "tool_events", []) or []
        latest_validation = _clean_prompt_snippet(
            _latest_validation_feedback(tool_events),
            max_chars=300,
        )
        if latest_validation:
            qa_lines.append(f"Latest Validation Excerpt: {latest_validation}")
        sections.append("\n".join(qa_lines))

    if fix_plan and getattr(fix_plan, "workaround_snippets", None):
        sections.append(
            "\n".join(
                [
                    "=== WORKAROUND SNIPPETS ===",
                    "Reference code patterns from security advisories:",
                    *[
                        f"  {index + 1}. {_clean_prompt_snippet(snippet, max_chars=300)}"
                        for index, snippet in enumerate(fix_plan.workaround_snippets[:3])
                    ],
                ]
            )
        )

    if constraints_ledger:
        cleaned_constraints = [
            _clean_prompt_snippet(constraint, max_chars=120)
            for constraint in constraints_ledger[:5]
        ]
        sections.append("Constraints:\n" + "\n".join(f"- {item}" for item in cleaned_constraints))

    sections.append(
        _workaround_search_strategy(
            target_task,
            target_group,
            workaround_context,
            previous_feedback,
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
    plan_state: dict[str, Any] | None = None,
) -> bool:
    """Apply the deterministic success contract for one workaround attempt."""
    validation_passed = (
        validation_gate_passed if validation_gate_passed is not None else has_all_validated
    )
    if not (
        runtime.changed_files and not runtime.errors and validation_passed and has_recorded_plan
    ):
        return False
    if requires_targeted_test and not targeted_test_passed:
        return False

    if plan_state:
        val_calls = int(plan_state.get("validation_calls", 0))
        if val_calls < 1:
            return False

        val_files = set(plan_state.get("validated_files", []))
        runtime_files = set(runtime.changed_files)
        if val_files != runtime_files:
            no_fix_package_files = set(plan_state.get("no_fix_package_files", []))
            missing_runtime_files = val_files - runtime_files
            if not (
                plan_state.get("package_removal_completed")
                and missing_runtime_files
                and missing_runtime_files <= no_fix_package_files
            ):
                return False

        last_val = plan_state.get("last_validation_result")
        if last_val is None:
            return False
        status = getattr(last_val, "overall_status", None) or (
            last_val.get("overall_status") if isinstance(last_val, dict) else None
        )
        if status not in ("PASS", WorkaroundValidationStatus.PASS):
            return False
        validation_state = plan_state.get("validation_passed")
        if validation_state is False:
            return False
        if plan_state.get("targeted_test_required"):
            targeted_file = getattr(last_val, "targeted_test_file", None) or (
                last_val.get("targeted_test_file") if isinstance(last_val, dict) else None
            )
            if not targeted_file:
                return False

    return True


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
    plan_state: dict[str, Any] | None = None,
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

    p_state = plan_state or {}
    val_calls = int(p_state.get("validation_calls", 0))
    val_files = list(p_state.get("validated_files", [])) if succeeded else []
    last_val = p_state.get("last_validation_result")
    per_gate = {}
    if last_val:
        per_gate = (
            last_val.model_dump()
            if hasattr(last_val, "model_dump")
            else dict(last_val)
            if isinstance(last_val, dict)
            else {}
        )

    selected_test = p_state.get("accepted_alternative_test") or p_state.get(
        "original_targeted_test"
    )
    mapping = dict(p_state.get("original_to_alternative_test_mapping", {}))
    mapping_evidence = dict(p_state.get("original_to_alternative_test_evidence", {}))
    mapping_details = dict(p_state.get("original_to_alternative_test_details", {}))
    infra_details = (
        p_state.get("latest_infra_diagnostics")
        or p_state.get("last_infrastructure_diagnostics")
        or p_state.get("infrastructure_failure_details")
    )

    diag = WorkerExecutionDiagnostics(
        validation_calls=val_calls,
        validation_input_errors=int(p_state.get("validation_input_errors", 0)),
        validation_passed=succeeded,
        failure_reason=" | ".join(errors),
        per_gate_results=per_gate,
        final_selected_targeted_test=selected_test,
        original_to_alternative_test_mapping=mapping,
        alternative_test_mapping_evidence=mapping_evidence,
        alternative_test_mapping_details=mapping_details,
        validated_files=val_files,
        infrastructure_failure_details=infra_details,
    )

    return {
        snapshot.attempt_id: WorkerAttemptResult(
            attempt_id=snapshot.attempt_id,
            task_id=snapshot.task_id,
            task_revision=snapshot.task_revision,
            cluster_id=snapshot.cluster_id,
            dispatch_batch_id=snapshot.dispatch_batch_id,
            action_digest=snapshot.action_digest,
            status=tagged_summary.status,
            changed_files=changed_files,
            action_summary=tagged_summary,
            execution_diagnostics=diag,
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
    t_id = target_task.task_id if target_task else "unknown"

    if get_runtime_settings().remedy_bypass_workaround_subagent:
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

    constraints_ledger = state.get("constraints_ledger", [])
    previous_feedback = state.get("previous_feedback", "")
    current_replay_plan = state.get("current_replay_plan")

    repo_root = Path(repo_root_str) if repo_root_str else Path.cwd()

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

    model_name = get_runtime_settings().workaround_llm_model
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
    pre_attempt_absent_paths: list[str] = []
    replayed_edit_sets: list[WorkaroundEditSet] = []
    if current_replay_plan is not None:
        pre_attempt_snapshots = dict(current_replay_plan.pre_attempt_snapshots)
        pre_attempt_absent_paths = list(current_replay_plan.pre_attempt_absent_paths)
        replayed_edit_sets = list(current_replay_plan.successful_edit_sets)
        plan_state["stage_baseline_snapshots"] = dict(pre_attempt_snapshots)
        plan_state["stage_baseline_absent_paths"] = list(pre_attempt_absent_paths)
    reset_context = (
        getattr(state.get("attempt_snapshot"), "workaround_context", None)
        if state.get("attempt_snapshot") is not None
        else None
    )
    if reset_context is not None and getattr(reset_context, "reset_prior_stage_workspace", False):
        # The supervisor normally clears these edit sets when advancing a
        # NO_FIX stage. Keep the worker fail-closed if an older replay plan
        # arrives before that reducer update is visible.
        replayed_edit_sets = []

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # A replay plan's stage baseline is task-local. Restoring the
            # whole file from that baseline on every retry can erase a
            # different vulnerability group's already-validated edit when
            # both groups share a source file. Stage resets are explicit; an
            # ordinary retry rebases on the live cumulative workspace.
            reset_prior_stage_workspace = bool(
                reset_context is not None
                and getattr(reset_context, "reset_prior_stage_workspace", False)
            )
            if reset_prior_stage_workspace:
                for rel_p in pre_attempt_absent_paths:
                    try:
                        sandbox.run(f"rm -f -- {shlex.quote(rel_p)}")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Workaround subagent: failed to remove absent-baseline path %s: %s",
                            rel_p,
                            exc,
                        )
                if pre_attempt_snapshots:
                    for rel_p, orig_content in pre_attempt_snapshots.items():
                        try:
                            sandbox.write_file(rel_p, orig_content)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Workaround subagent: failed to restore snapshot for %s: %s",
                                rel_p,
                                exc,
                            )

            # Replay prior successful edit sets on the live workspace. A
            # retry may already contain an accepted replacement, so treating
            # the replacement as idempotent avoids duplicate edits and a
            # destructive restore to an old task-local baseline.
            replay_failed = False
            replay_error_msg = ""
            for edit_set in replayed_edit_sets:
                replacements_to_apply: list[Any] = []
                # Pre-verify all replacements in edit_set
                for redit in edit_set.replacements:
                    try:
                        curr = sandbox.read_file(redit.file_path)
                        if curr is None:
                            replay_failed = True
                            replay_error_msg = f"Replay failed: Could not read '{redit.file_path}' for patch '{edit_set.patch_id}'."
                            break
                        curr_norm = _normalise_newlines(curr)
                        old_norm = _normalise_newlines(redit.old_text)
                        count = curr_norm.count(old_norm)
                        expected = (
                            redit.expected_occurrences
                            if hasattr(redit, "expected_occurrences")
                            and redit.expected_occurrences > 0
                            else 1
                        )
                        new_count = curr_norm.count(_normalise_newlines(redit.new_text))
                        if count == expected:
                            replacements_to_apply.append(redit)
                        elif count == 0 and new_count >= expected:
                            # The replacement is already present in the
                            # cumulative workspace.
                            touched_files.add(redit.file_path)
                        else:
                            replay_failed = True
                            replay_error_msg = (
                                f"Replay failed: Occurrence count mismatch for anchor in '{redit.file_path}' "
                                f"during patch '{edit_set.patch_id}' (expected {expected}, found {count})."
                            )
                            break
                    except Exception as exc:  # noqa: BLE001
                        replay_failed = True
                        replay_error_msg = f"Replay failed on file '{redit.file_path}': {exc}"
                        break

                if replay_failed:
                    break

                # Apply edit set replacements
                for redit in replacements_to_apply:
                    curr = sandbox.read_file(redit.file_path)
                    newline_style = _detect_newline_style(curr)
                    curr_norm = _normalise_newlines(curr)
                    old_norm = _normalise_newlines(redit.old_text)
                    new_norm = _normalise_newlines(redit.new_text)
                    expected = (
                        redit.expected_occurrences
                        if hasattr(redit, "expected_occurrences") and redit.expected_occurrences > 0
                        else 1
                    )
                    updated_norm = curr_norm.replace(old_norm, new_norm, expected)
                    sandbox.write_file(
                        redit.file_path, _restore_newlines(updated_norm, newline_style)
                    )
                    touched_files.add(redit.file_path)

            if replay_failed:
                logger.error(replay_error_msg)
                summaries = _build_surrender_summaries(t_id, replay_error_msg)
                return {
                    "action_summaries": summaries,
                    "action_summary": summaries[0],
                    "changed_files": [],
                    "errors": [replay_error_msg],
                }

            snapshot = state.get("attempt_snapshot")
            workaround_ctx = getattr(snapshot, "workaround_context", None) if snapshot else None
            if (
                workaround_ctx
                and getattr(workaround_ctx, "phase", None) == WorkaroundPhase.QA_REGRESSION_REPAIR
            ):
                plan_state["require_authoritative_evidence"] = True

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
                SystemMessage(content=_WORKAROUND_STATIC_INSTRUCTIONS),
                HumanMessage(content=prompt),
            ]

            no_fix_manifest_paths: list[str] = []
            if getattr(target_task, "no_fix_stage", None) is not None:
                for candidate in [
                    *(getattr(target_group, "file_paths", []) or []),
                    getattr(target_group, "file_path", None),
                    *(
                        getattr(issue, "manifest_file", None)
                        for issue in getattr(target_group, "localized_issues", []) or []
                    ),
                ]:
                    normalized = str(candidate or "").replace("\\", "/").lstrip("/")
                    if normalized and normalized not in no_fix_manifest_paths:
                        no_fix_manifest_paths.append(normalized)

            toolbelt = build_workaround_toolbelt(
                sandbox,
                touched_files,
                repo_root,
                plan_state=plan_state,
                preferred_test_files=_preferred_targeted_test_files(qa_ev),
                no_fix_stage=(
                    getattr(workaround_ctx, "no_fix_stage", None)
                    if workaround_ctx is not None
                    else getattr(target_task, "no_fix_stage", None)
                ),
                no_fix_package_name=(
                    getattr(target_group, "vulnerable_component", None)
                    if getattr(target_task, "no_fix_stage", None) is not None
                    else None
                ),
                no_fix_manifest_paths=(
                    no_fix_manifest_paths
                    if getattr(target_task, "no_fix_stage", None) is not None
                    else []
                ),
                no_fix_package_manager=(
                    next(
                        (
                            issue.package_manager
                            for issue in getattr(target_group, "localized_issues", []) or []
                            if getattr(issue, "package_manager", None)
                        ),
                        "npm",
                    )
                ),
            )
            context_manager = ContextManager(toolbelt)
            runtime = run_bounded_subagent_loop(
                llm,
                toolbelt,
                initial_messages,
                touched_files,
                execution_state=plan_state,
                context_manager=context_manager,
            )

            # Revert any illegally modified manifest files or test files
            prohibited_modified = {
                f
                for f in touched_files
                if _is_prohibited_target(f)
                and not _is_allowlisted_no_fix_package_file(f, plan_state)
            }
            if prohibited_modified:
                logger.warning(
                    "Workaround subagent modified prohibited files %s. Reverting.",
                    prohibited_modified,
                )
                restored_prohibited: set[str] = set()
                for f in prohibited_modified:
                    try:
                        _restore_attempt_file(
                            sandbox,
                            f,
                            plan_state,
                            stage_snapshots=pre_attempt_snapshots,
                            stage_absent_paths=pre_attempt_absent_paths,
                            allow_stage_baseline=reset_prior_stage_workspace,
                        )
                        restored_prohibited.add(f)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to revert %s: %s", f, exc)
                touched_files -= restored_prohibited
                runtime = dataclasses.replace(runtime, changed_files=sorted(touched_files))

            validation_gate_passed = has_successful_validation_gate(
                runtime.tool_events,
                has_prior_edits=bool(replayed_edit_sets)
                or bool(plan_state.get("package_removal_completed")),
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
                    edit_tool_names="deterministic_search_replace",
                )
                or has_tool_call_before_first_successful_edit(
                    runtime.tool_events,
                    lookup_tool_name="record_plan",
                    edit_tool_names="deterministic_replace_ast_symbol",
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
                last_validation = plan_state.get("last_validation_result")
                targeted_test_passed = bool(
                    last_validation
                    and (
                        getattr(last_validation, "overall_status", None)
                        or (
                            last_validation.get("overall_status")
                            if isinstance(last_validation, dict)
                            else None
                        )
                    )
                    in ("PASS", WorkaroundValidationStatus.PASS)
                    and (
                        getattr(last_validation, "targeted_test_file", None)
                        or (
                            last_validation.get("targeted_test_file")
                            if isinstance(last_validation, dict)
                            else None
                        )
                    )
                )

            succeeded = _workaround_attempt_succeeded(
                runtime,
                has_all_validated=has_all_validated,
                has_recorded_plan=has_recorded_plan,
                requires_targeted_test=requires_targeted_test,
                targeted_test_passed=targeted_test_passed,
                validation_gate_passed=validation_gate_passed,
                plan_state=plan_state,
            )

            all_edit_sets: list[WorkaroundEditSet] = list(replayed_edit_sets)
            if succeeded:
                new_edit_sets = plan_state.get("successful_edit_sets", [])
                all_edit_sets.extend(new_edit_sets)
            else:
                replayed_files = {
                    edit.file_path
                    for edit_set in replayed_edit_sets
                    for edit in edit_set.replacements
                }
                current_attempt_files = set(runtime.changed_files) - replayed_files
                restored_files: set[str] = set()
                for f in current_attempt_files:
                    try:
                        _restore_attempt_file(
                            sandbox,
                            f,
                            plan_state,
                            stage_snapshots=pre_attempt_snapshots,
                            stage_absent_paths=pre_attempt_absent_paths,
                            allow_stage_baseline=reset_prior_stage_workspace,
                        )
                        restored_files.add(f)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to restore failed-attempt file %s: %s", f, exc)
                touched_files -= restored_files
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

    p_state = plan_state or {}
    last_val = p_state.get("last_validation_result")
    per_gate = (
        last_val.model_dump()
        if hasattr(last_val, "model_dump")
        else dict(last_val)
        if isinstance(last_val, dict)
        else {}
    )
    selected_test = p_state.get("accepted_alternative_test") or p_state.get(
        "original_targeted_test"
    )
    mapping = dict(p_state.get("original_to_alternative_test_mapping", {}))
    mapping_evidence = dict(p_state.get("original_to_alternative_test_evidence", {}))
    mapping_details = dict(p_state.get("original_to_alternative_test_details", {}))
    infra_details = (
        p_state.get("latest_infra_diagnostics")
        or p_state.get("last_infrastructure_diagnostics")
        or p_state.get("infrastructure_failure_details")
    )

    pre_attempt_snapshots = dict(plan_state.get("stage_baseline_snapshots", pre_attempt_snapshots))
    pre_attempt_absent_paths = list(
        plan_state.get("stage_baseline_absent_paths", pre_attempt_absent_paths)
    )

    new_replay_plan = WorkaroundReplayPlan(
        task_id=t_id,
        pre_attempt_snapshots=pre_attempt_snapshots,
        pre_attempt_absent_paths=sorted(
            set(plan_state.get("stage_baseline_absent_paths", pre_attempt_absent_paths))
        ),
        successful_edit_sets=all_edit_sets,
        investigation_findings={
            "changed_files": list(touched_files),
            "validation_feedback": _latest_validation_feedback(runtime.tool_events),
        },
        source_attempt_id=snapshot.attempt_id if snapshot else "",
        security_invariants=[sec_inv] if sec_inv else [],
        diagnosed_root_causes=[causal_hyp] if causal_hyp else [],
        planned_targets=planned_files + planned_symbols,
        validated_files=list(plan_state.get("validated_files", [])) if succeeded else [],
        validation_calls=int(p_state.get("validation_calls", 0)),
        validation_input_errors=int(p_state.get("validation_input_errors", 0)),
        per_gate_results=per_gate,
        final_selected_targeted_test=selected_test,
        original_to_alternative_test_mapping=mapping,
        alternative_test_mapping_evidence=mapping_evidence,
        alternative_test_mapping_details=mapping_details,
        infrastructure_failure_details=infra_details,
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
            plan_state=plan_state,
        ),
        "errors": runtime.errors,
    }
