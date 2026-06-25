"""
qa_critic.py - Agentic QA evaluator node for the Phase 5 orchestrator.

The QA Critic now follows a map-reduce architecture:

  Step 0 — Global Execution (deterministic Python):
    run_dependency_install → run_security_scan → run_unit_tests, called exactly
    once via direct Python helpers, with no LLM tools involved.

  Map — Individual Investigators:
    One bounded ReAct agent per vulnerability group, given a group-scoped
    prompt and a read-only review toolbelt.  Each agent answers only for its
    assigned group.

  Reduce — Batch Judge:
    One ChatOpenAI.with_structured_output(BatchQAResult) call across all
    group investigation texts, producing a holistic_report and exactly one
    QAEvaluation per group.

  Python Guardrails:
    Normalize, validate, and fill missing/duplicate/unknown evaluations
    deterministically.  Enforce scanner and install-error policies.

The node's external contract (run_qa_critic_node, run_qa_critic_from_orchestrator,
state key outputs) is unchanged — graph wiring requires no modifications.

Heavy QA commands (install, scan, tests) are intentionally *not* exposed as
tools to the update or workaround subagents; they live here only.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langsmith import traceable

from src.contracts.schemas import (
    AgentActionSummary,
    BatchQAResult,
    FailureCategory,
    QAEvaluation,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.orchestrator.subagent_runtime import run_bounded_subagent_loop
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ported legacy constants
# ---------------------------------------------------------------------------

_NPM_INSTALL_TIMEOUT_SECONDS = 900
_NPM_TEST_TIMEOUT_SECONDS = 600
_ODC_TIMEOUT_SECONDS = 300
_ODC_REPORT_NAME = "dependency-check-report.json"
_ODC_HTML_REPORT_NAME = "dependency-check-report.html"
_ODC_CACHE_VOLUME = "odc-cache"
_ODC_DEBUG_DIR = Path("data/cache/qa_reports")

# LLM context budget limits
_DIFF_CHAR_BUDGET = 8_000
_TEST_LOG_TAIL_LINES = 60
_STDERR_TAIL_LINES = 30
_INSTALL_LOG_TAIL_LINES = 80
_FILE_READ_MAX_CHARS = 8_000
_LOG_QUERY_MAX_CHARS = 6_000

# Exclusion patterns for workspace diff
_DIFF_EXCLUDE_DIRS = frozenset({
    "node_modules",
    ".git",
    "dependency-check-data",
    "coverage",
    ".nyc_output",
    ".cache",
})
_DIFF_EXCLUDE_SUFFIXES = frozenset({".map", ".lock"})
_DIFF_EXCLUDE_NAMES = frozenset({_ODC_REPORT_NAME, _ODC_HTML_REPORT_NAME})

# Install error patterns that indicate peer/engine conflicts
_PEER_CONFLICT_PATTERNS = ("ERESOLVE", "EBADENGINE", "peer dep", "peer tree")


# ---------------------------------------------------------------------------
# Legacy ODC helpers (ported from old remedy_tools.py)
# ---------------------------------------------------------------------------


def _read_report_from_workspace(sandbox: DockerSandbox) -> Optional[str]:
    """Read the ODC JSON report from the sandbox workspace."""
    try:
        return sandbox.read_file(_ODC_REPORT_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to read ODC report from workspace — %s", exc)
        return None

def _persist_workspace_report_to_host(
    sandbox: DockerSandbox,
    workspace_name: str,
    host_path: Path,
) -> Optional[Path]:
    """Copy a Dependency-Check report from the workspace volume onto the host."""
    try:
        content = sandbox.read_file(workspace_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to read %s from workspace - %s", workspace_name, exc)
        return None

    if content is None:
        return None

    try:
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "qa_critic: failed to persist %s to host path %s - %s",
            workspace_name,
            host_path,
            exc,
        )
        return None

    return host_path.resolve()


def _next_html_report_host_path() -> Path:
    """Return a unique host-side HTML report path for one QA scan."""
    timestamp_ms = int(time.time() * 1000)
    return _ODC_DEBUG_DIR / f"dependency-check-report-{timestamp_ms}.html"


def _parse_report_identifiers(report_text: str) -> Optional[Set[str]]:
    """Parse CVE/GHSA identifiers from the ODC JSON report text."""
    try:
        report = json.loads(report_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to decode ODC report JSON — %s", exc)
        return None

    try:
        from src.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning("qa_critic: src.tools.odc_parser not importable.")
        return None

    identifiers: Set[str] = set()
    for issue in parse_vulnerabilities(report):
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _run_odc(workspace_volume: str) -> "subprocess.CompletedProcess[str]":
    """Execute OWASP Dependency-Check in Docker against the shared workspace volume."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        "root",
        "-v",
        f"{workspace_volume}:/scan",
        "-v",
        f"{_ODC_CACHE_VOLUME}:/usr/share/dependency-check/data",
        "owasp/dependency-check:latest",
        "--project",
        "sandbox_scan",
        "--scan",
        "/scan",
        "--format",
        "JSON",
        "--format",
        "HTML",
        "--out",
        "/scan",
        "--noupdate",
    ]

    extra_args = os.environ.get("ODC_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    logger.info("qa_critic: running ODC in Docker: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_ODC_TIMEOUT_SECONDS,
    )


# ---------------------------------------------------------------------------
# QA runner helpers (deterministic, called by tool wrappers and global execution)
# ---------------------------------------------------------------------------


def _run_install(sandbox: DockerSandbox) -> Tuple[bool, str]:
    """
    Run ``npm install --package-lock=true`` inside the workspace.

    Returns:
        (success, summary_text)
    """
    result = sandbox.run(
        "npm install --package-lock=true",
        timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
    )
    if result.exit_code == 0:
        return True, "npm install succeeded."

    # Surface the tail of the output so the LLM context stays bounded.
    stdout_tail = "\n".join(result.stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
    stderr_tail = "\n".join(result.stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
    summary = (
        f"npm install FAILED (exit {result.exit_code}).\n"
        f"stdout tail:\n{stdout_tail}\n"
        f"stderr tail:\n{stderr_tail}"
    )
    return False, summary


def _run_security_scan(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: Set[str],
) -> Tuple[bool, str, Set[str]]:
    """
    Run OWASP Dependency-Check and identify remaining target identifiers.

    Returns:
        (success, summary_text, remaining_identifiers)
        ``success=False`` when Docker is absent, ODC times out, or no
        parseable report is produced.
        ``remaining_identifiers`` is empty on success or on hard failure.
    """
    if shutil.which("docker") is None:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        logger.warning("qa_critic: %s", msg)
        return False, msg, set()

    try:
        proc = _run_odc(workspace_volume)
    except FileNotFoundError:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        return False, msg, set()
    except subprocess.TimeoutExpired:
        msg = f"FAILURE: Dependency-Check timed out after {_ODC_TIMEOUT_SECONDS}s."
        return False, msg, set()
    except Exception as exc:  # noqa: BLE001
        msg = f"FAILURE: Dependency-Check subprocess error — {exc}"
        return False, msg, set()

    saved_html_report = _persist_workspace_report_to_host(
        sandbox,
        _ODC_HTML_REPORT_NAME,
        _next_html_report_host_path(),
    )
    saved_json_report = _persist_workspace_report_to_host(
        sandbox,
        _ODC_REPORT_NAME,
        _ODC_DEBUG_DIR / _ODC_REPORT_NAME,
    )
    report_location_note = ""
    if saved_html_report is not None:
        report_location_note = f"\nHTML report saved to: {saved_html_report}"
        if saved_json_report is not None:
            report_location_note += f"\nJSON report saved to: {saved_json_report}"

    report_text = _read_report_from_workspace(sandbox)
    found_identifiers = (
        _parse_report_identifiers(report_text) if report_text is not None else None
    )

    if proc.returncode != 0 and found_identifiers is None:
        summary = (
            f"FAILURE: Dependency-Check exited {proc.returncode} and produced "
            "no parseable report.\n"
            f"stdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        summary += report_location_note
        return False, summary, set()

    if found_identifiers is None:
        summary = (
            f"FAILURE: Dependency-Check report was not parseable "
            f"(exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        summary += report_location_note
        return False, summary, set()

    remaining = {ident.upper().strip() for ident in target_identifiers if ident}
    remaining &= found_identifiers

    if remaining:
        remaining_text = ", ".join(sorted(remaining))
        summary = (
            "FAILURE: Dependency-Check found unresolved target vulnerabilities. "
            f"Remaining identifiers: {remaining_text}"
        )
        summary += report_location_note
        return False, summary, remaining

    summary = "Dependency-Check found no remaining target vulnerability identifiers."
    summary += report_location_note
    return True, summary, set()


def _run_unit_tests(sandbox: DockerSandbox) -> Tuple[bool, str]:
    """
    Run ``npm test`` inside the workspace.

    Returns:
        (success, summary_text)
    """
    result = sandbox.run("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)
    if result.exit_code == 0:
        return True, "npm test passed."

    # Extract a bounded log tail for LLM context.
    stdout_tail = "\n".join(result.stdout.splitlines()[-_TEST_LOG_TAIL_LINES:])
    stderr_tail = "\n".join(result.stderr.splitlines()[-_STDERR_TAIL_LINES:])
    summary = (
        f"npm test FAILED (exit {result.exit_code}).\n"
        f"stdout tail:\n{stdout_tail}\n"
        f"stderr tail:\n{stderr_tail}"
    )
    return False, summary


def _generate_workspace_diff(
    host_repo_root: str,
    sandbox: DockerSandbox,
    candidate_changed_files: List[str],
) -> Tuple[str, List[str]]:
    """
    Generate a unified diff by comparing host baseline to the current sandbox workspace
    for only the specified candidate_changed_files.

    The Docker volume does not contain ``.git``, so ``git diff`` is not available.
    We read files from the sandbox and diff against the host.

    Returns:
        (diff_text_capped, changed_file_paths)
        ``diff_text_capped`` is the diff content, capped at ``_DIFF_CHAR_BUDGET`` chars.
        ``changed_file_paths`` is the full list of repo-relative paths that changed
        (retained even when the diff text itself is truncated).
    """
    if not candidate_changed_files:
        return "(no changed files were provided; diff is empty)", []

    host_root = Path(host_repo_root)
    changed_files: List[str] = []
    diff_parts: List[str] = []

    # Deduplicate files while preserving order.
    seen: Set[str] = set()
    unique_candidates: List[str] = []
    for f in candidate_changed_files:
        if f not in seen:
            seen.add(f)
            unique_candidates.append(f)

    for rel_path in unique_candidates:
        # Normalize path separators.
        rel_path = rel_path.replace("\\", "/")
        parts = Path(rel_path).parts
        if any(p in _DIFF_EXCLUDE_DIRS for p in parts):
            continue
        if Path(rel_path).name in _DIFF_EXCLUDE_NAMES:
            continue
        if Path(rel_path).suffix.lower() in _DIFF_EXCLUDE_SUFFIXES:
            continue

        abs_host = host_root / rel_path
        workspace_content = sandbox.read_file(rel_path)

        try:
            if abs_host.is_file():
                host_content = abs_host.read_text(encoding="utf-8", errors="replace")
            else:
                host_content = None
        except Exception:
            host_content = None

        if workspace_content is None:
            # File deleted in workspace.
            if host_content is not None:
                changed_files.append(rel_path)
                diff_parts.append(f"--- {rel_path} (deleted in workspace)\n")
        elif host_content is None:
            # New file in workspace.
            changed_files.append(rel_path)
            diff_parts.append(f"+++ {rel_path} (new file in workspace)\n")
        elif workspace_content != host_content:
            # File modified.
            changed_files.append(rel_path)
            import difflib
            diff_lines = list(difflib.unified_diff(
                host_content.splitlines(keepends=True),
                workspace_content.splitlines(keepends=True),
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
                lineterm="",
            ))
            diff_parts.append("".join(diff_lines))

    full_diff = "\n".join(diff_parts)
    if not full_diff:
        return (
            "(diff is empty — workspace matches host baseline for all candidate files)",
            changed_files,
        )

    if len(full_diff) > _DIFF_CHAR_BUDGET:
        full_diff = full_diff[:_DIFF_CHAR_BUDGET] + "\n... (diff truncated)"

    return full_diff, changed_files


# ---------------------------------------------------------------------------
# Target identifier collection
# ---------------------------------------------------------------------------


def _collect_target_identifiers(groups: List[VulnerabilityGroup]) -> Set[str]:
    """Collect all CVE/GHSA identifiers from the valid vulnerability groups."""
    identifiers: Set[str] = set()
    for group in groups:
        for cve in group.cve_ids or []:
            if cve:
                identifiers.add(cve.upper().strip())
        for ghsa in group.ghsa_ids or []:
            if ghsa:
                identifiers.add(ghsa.upper().strip())
        # Fallback: individual issue-level identifiers
        for issue in group.issues or []:
            if issue.cve_id:
                identifiers.add(issue.cve_id.upper().strip())
            if issue.ghsa_id:
                identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


# ---------------------------------------------------------------------------
# Path safety helper (used by review tools)
# ---------------------------------------------------------------------------


def _validate_qa_path(file_path: str) -> str:
    """Validate a repo-relative path for QA read-only review tools."""
    candidate = (file_path or "").strip()
    if not candidate:
        raise ValueError("file_path is required.")
    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        raise ValueError(f"Rejected absolute file path '{candidate}'.")
    if ".." in Path(candidate).parts:
        raise ValueError(f"Rejected path traversal in '{candidate}'.")
    return candidate.replace("\\", "/")


# ---------------------------------------------------------------------------
# Execution results cache
# ---------------------------------------------------------------------------


@dataclass
class _QAExecutionResults:
    """Cache for global execution phase results."""

    install: Optional[Tuple[bool, str]] = None          # (ok, summary)
    scan: Optional[Tuple[bool, str, Set[str]]] = None   # (ok, summary, remaining_ids)
    tests: Optional[Tuple[bool, str]] = None            # (ok, summary)


# ---------------------------------------------------------------------------
# Map phase dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupInvestigation:
    """Investigation output for a single vulnerability group (Map phase)."""

    group_id: str
    investigation_text: str
    tool_transcript: str
    errors: List[str] = field(default_factory=list)


@dataclass
class BatchInvestigationArtifact:
    """All Map phase outputs consumed by the Reduce phase."""

    results: _QAExecutionResults
    investigations_by_group: Dict[str, GroupInvestigation]
    holistic_report: str = ""   # filled after reduce phase
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Backcompat dataclasses (kept for legacy path and tests)
# ---------------------------------------------------------------------------


@dataclass
class GroupInvestigationSection:
    """Parsed report block for one vulnerability group (backcompat)."""

    group_id: str
    raw_text: str
    component: str = ""
    strategy: str = ""
    target_identifiers: str = ""
    changed_files: str = ""
    scan_status: str = ""
    remaining_scanner_findings: str = ""
    scan_reasoning: str = ""
    workaround_review: str = ""
    diff_evidence: str = ""
    test_status: str = ""
    attributed_test_failures: str = ""
    causal_reasoning: str = ""
    exonerated_groups: str = ""
    group_summary: str = ""


@dataclass
class ParsedInvestigationReport:
    """Markdown investigative report parsed into shared and per-group sections (backcompat)."""

    raw_report: str
    shared_install_analysis: str
    group_sections: Dict[str, GroupInvestigationSection]
    errors: List[str]
    warnings: List[str]


@dataclass
class GroupEvidencePacket:
    """Group-scoped evidence passed to the Judge phase (backcompat)."""

    group: VulnerabilityGroup
    strategy: str
    fix_plan_status: str
    fix_plan_instruction: str
    action_summaries: List[str]
    shared_install_analysis: str
    install_ok: bool
    install_summary: str
    scan_ok: bool
    scan_summary: str
    remaining_identifiers: List[str]
    tests_ok: bool
    tests_summary: str
    group_block_markdown: str
    section: GroupInvestigationSection


@dataclass
class InvestigationArtifact:
    """Outputs of the Investigator phase consumed by the Judge phase (backcompat)."""

    report_text: str
    parsed_report: ParsedInvestigationReport
    transcript: str
    results: _QAExecutionResults
    errors: List[str]


_REPORT_PREFIX = "# INVESTIGATIVE REPORT"
_GROUP_HEADING_RE = re.compile(r"^### GROUP:\s*(.+?)\s*$", re.MULTILINE)
_BULLET_LABEL_RE = re.compile(r"^- ([^:]+):\s*(.*)$")


def _pipeline_complete(results: _QAExecutionResults) -> bool:
    """Return whether install, scan, and tests have all run at least once."""
    return (
        results.install is not None
        and results.scan is not None
        and results.tests is not None
    )


def _review_ready_error(results: _QAExecutionResults) -> Optional[str]:
    """Return the standard review-tool order error, if any."""
    if _pipeline_complete(results):
        return None
    return (
        "ERROR: Review tools are locked until run_dependency_install, "
        "run_security_scan, and run_unit_tests have all been called in order."
    )


def _resolve_action_summary_group_ids(
    summary: AgentActionSummary,
    known_group_ids: Set[str],
) -> List[str]:
    """Resolve which exact group_ids an AgentActionSummary applies to."""
    raw_group_id = (summary.group_id or "").strip()
    if not raw_group_id:
        return []
    if raw_group_id.startswith("batch:"):
        payload = raw_group_id[len("batch:"):]
        resolved = []
        for part in payload.split(","):
            candidate = part.strip()
            if candidate and candidate in known_group_ids:
                resolved.append(candidate)
        return resolved
    return [raw_group_id] if raw_group_id in known_group_ids else []


def _relevant_action_summaries(
    action_summaries: List[AgentActionSummary],
    group_id: str,
    known_group_ids: Set[str],
) -> List[AgentActionSummary]:
    """Filter action summaries to those explicitly linked to one group."""
    relevant: List[AgentActionSummary] = []
    for summary in action_summaries:
        if group_id in _resolve_action_summary_group_ids(summary, known_group_ids):
            relevant.append(summary)
    return relevant


def _parse_report_bullets(block_text: str) -> Dict[str, str]:
    """Parse markdown '- Label: value' bullets, preserving wrapped lines."""
    fields: Dict[str, str] = {}
    current_label: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_label, current_lines
        if current_label is None:
            return
        fields[current_label] = "\n".join(current_lines).strip()
        current_label = None
        current_lines = []

    for raw_line in block_text.splitlines():
        line = raw_line.rstrip()
        match = _BULLET_LABEL_RE.match(line)
        if match:
            flush()
            current_label = match.group(1).strip()
            current_lines = [match.group(2).strip()]
            continue
        if current_label is not None:
            current_lines.append(line.strip())

    flush()
    return fields


def _group_target_identifiers(group: VulnerabilityGroup) -> Set[str]:
    """Collect normalized scanner identifiers relevant to one group."""
    identifiers: Set[str] = set()
    for cve in group.cve_ids or []:
        if cve:
            identifiers.add(cve.upper().strip())
    for ghsa in group.ghsa_ids or []:
        if ghsa:
            identifiers.add(ghsa.upper().strip())
    for issue in group.issues or []:
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _group_remaining_identifiers(
    group: VulnerabilityGroup,
    remaining_identifiers: Set[str],
) -> List[str]:
    """Return the exact remaining scanner identifiers attributable to one group."""
    return sorted(_group_target_identifiers(group) & remaining_identifiers)


def _group_scan_status(
    scan_result: Optional[Tuple[bool, str, Set[str]]],
    group: VulnerabilityGroup,
) -> str:
    """Classify the scanner outcome for one group from deterministic results."""
    if scan_result is None:
        return "scan_failed"

    scan_ok, _scan_summary, remaining_identifiers = scan_result
    if _group_remaining_identifiers(group, remaining_identifiers):
        return "still_flagged"
    if remaining_identifiers:
        return "cleared"
    if scan_ok:
        return "cleared"
    return "scan_failed"


def _build_fallback_investigation_report(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    candidate_changed_files: List[str],
    results: _QAExecutionResults,
    reason: str,
) -> str:
    """Synthesize a minimal investigative report when the LLM output is malformed."""
    install_ok, install_summary = results.install or (
        False, "run_dependency_install was not called."
    )
    if results.scan is None:
        scan_result: Optional[Tuple[bool, str, Set[str]]] = None
    else:
        scan_result = results.scan
    tests_ok, _ = results.tests or (False, "run_unit_tests was not called.")

    changed_files_text = ", ".join(candidate_changed_files) if candidate_changed_files else "none"
    blocks = [
        _REPORT_PREFIX,
        "## Install Analysis",
        f"- Install Status: {'succeeded' if install_ok else 'failed'}",
        f"- Summary: {reason}",
        "- Suspected Responsible Group(s): unknown",
        f"- Evidence: {install_summary}",
        "",
    ]

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "(unknown)")
        group_identifiers = sorted(_group_target_identifiers(group))
        group_remaining = (
            _group_remaining_identifiers(group, scan_result[2])
            if scan_result is not None
            else []
        )
        scan_status = _group_scan_status(scan_result, group)
        blocks.extend([
            f"### GROUP: {group.group_id}",
            f"- Component: {group.vulnerable_component or '(unknown)'}",
            f"- Strategy: {strategy}",
            f"- Target Identifiers: {', '.join(group_identifiers) if group_identifiers else 'none'}",
            f"- Changed Files: {changed_files_text}",
            "",
            f"- Scan Status: {scan_status}",
            f"- Remaining Scanner Findings: {', '.join(group_remaining) if group_remaining else 'none'}",
            "- Scan Reasoning: Investigation report was synthesized from deterministic QA results.",
            "",
            (
                "- Workaround Review: not applicable"
                if strategy != "code_workaround"
                else "- Workaround Review: not reviewed; fallback report due to malformed investigator output."
            ),
            (
                "- Diff Evidence: not applicable"
                if strategy != "code_workaround"
                else "- Diff Evidence: none reviewed."
            ),
            "",
            f"- Test Status: {'passed' if tests_ok else 'failed'}",
            "- Attributed Test Failures: none",
            "- Causal Reasoning: No trusted investigator prose was available; defer to deterministic QA logs.",
            "- Exonerated Groups: none",
            "",
            "- Group Summary: Fallback summary generated because the investigator output was missing or malformed.",
            "",
        ])
    return "\n".join(blocks).strip()


def _parse_investigation_report(
    report_text: str,
    valid_groups: List[VulnerabilityGroup],
) -> ParsedInvestigationReport:
    """Parse the investigator markdown report into shared and per-group sections (backcompat)."""
    known_group_ids = {group.group_id for group in valid_groups}
    errors: List[str] = []
    warnings: List[str] = []
    normalized = (report_text or "").strip()

    if not normalized.startswith(_REPORT_PREFIX):
        errors.append("Investigation report missing required '# INVESTIGATIVE REPORT' heading.")
        normalized = f"{_REPORT_PREFIX}\n{normalized}".strip()

    matches = list(_GROUP_HEADING_RE.finditer(normalized))
    shared_end = matches[0].start() if matches else len(normalized)
    shared_install_analysis = normalized[len(_REPORT_PREFIX):shared_end].strip()

    sections: Dict[str, GroupInvestigationSection] = {}
    for index, match in enumerate(matches):
        group_id = match.group(1).strip()
        block_start = match.start()
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        raw_block = normalized[block_start:block_end].strip()
        if group_id not in known_group_ids:
            warnings.append(f"Ignoring unknown investigation block heading for '{group_id}'.")
            continue
        bullet_fields = _parse_report_bullets(raw_block)
        sections[group_id] = GroupInvestigationSection(
            group_id=group_id,
            raw_text=raw_block,
            component=bullet_fields.get("Component", ""),
            strategy=bullet_fields.get("Strategy", ""),
            target_identifiers=bullet_fields.get("Target Identifiers", ""),
            changed_files=bullet_fields.get("Changed Files", ""),
            scan_status=bullet_fields.get("Scan Status", ""),
            remaining_scanner_findings=bullet_fields.get("Remaining Scanner Findings", ""),
            scan_reasoning=bullet_fields.get("Scan Reasoning", ""),
            workaround_review=bullet_fields.get("Workaround Review", ""),
            diff_evidence=bullet_fields.get("Diff Evidence", ""),
            test_status=bullet_fields.get("Test Status", ""),
            attributed_test_failures=bullet_fields.get("Attributed Test Failures", ""),
            causal_reasoning=bullet_fields.get("Causal Reasoning", ""),
            exonerated_groups=bullet_fields.get("Exonerated Groups", ""),
            group_summary=bullet_fields.get("Group Summary", ""),
        )

    for group in valid_groups:
        if group.group_id in sections:
            continue
        errors.append(f"Investigation report missing block for group '{group.group_id}'.")
        placeholder = "\n".join([
            f"### GROUP: {group.group_id}",
            "- Scan Analysis: missing",
            "- Test Attribution & Exoneration: missing",
            "- Diff Review: missing",
        ])
        sections[group.group_id] = GroupInvestigationSection(
            group_id=group.group_id,
            raw_text=placeholder,
        )

    return ParsedInvestigationReport(
        raw_report=normalized,
        shared_install_analysis=shared_install_analysis,
        group_sections=sections,
        errors=errors,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Step 0: Global Execution (deterministic Python, no LLM tools)
# ---------------------------------------------------------------------------


def _run_global_execution(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: Set[str],
) -> _QAExecutionResults:
    """
    Run install, security scan, and unit tests exactly once via direct Python calls.

    No LLM tool wrappers are involved — execution is deterministic and sequential.
    Results are stored in a _QAExecutionResults cache for downstream use.
    """
    results = _QAExecutionResults()

    logger.info("qa_critic: [Step 0] running npm install.")
    results.install = _run_install(sandbox)
    install_ok, _ = results.install

    logger.info("qa_critic: [Step 0] running security scan (install_ok=%s).", install_ok)
    results.scan = _run_security_scan(sandbox, workspace_volume, target_identifiers)

    logger.info("qa_critic: [Step 0] running unit tests.")
    results.tests = _run_unit_tests(sandbox)

    return results


# ---------------------------------------------------------------------------
# QA toolbelt (backcompat: execution + review tools together)
# ---------------------------------------------------------------------------


def build_qa_toolbelt(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: Set[str],
    candidate_changed_files: List[str],
    host_repo_root: Optional[str],
) -> Tuple[List, _QAExecutionResults]:
    """
    Build the QA-only toolbelt with one-shot execution tools and read-only review tools.

    This function is retained for backward compatibility.  The new map-reduce pipeline
    uses _run_global_execution (Step 0) for execution and build_qa_review_toolbelt for
    the read-only review toolbelt given to individual investigators.

    Execution tools (run_dependency_install, run_security_scan, run_unit_tests) are
    one-shot guarded: the first call performs the real work and caches the result;
    subsequent calls return the cached result immediately.

    Review tools (generate_workspace_diff, list_changed_files, read_file_context,
    search_codebase_pattern, inspect_ast_symbol, query_qa_logs) are always callable
    and never modify workspace state.

    Returns:
        (tools_list, results_cache)
        ``results_cache`` can be inspected after the agent loop to verify which
        execution tools were actually called and what they returned.
    """
    from src.orchestrator.remedy_tools import (
        _make_inspect_ast_symbol_tool,
        _make_search_codebase_pattern_tool,
    )

    results = _QAExecutionResults()
    search_tool = _make_search_codebase_pattern_tool(sandbox)
    inspect_tool = _make_inspect_ast_symbol_tool(sandbox)

    # ------------------------------------------------------------------
    # Execution tools (one-shot guarded)
    # ------------------------------------------------------------------

    @tool
    def run_dependency_install() -> str:
        """
        Run 'npm install --package-lock=true' inside the workspace.
        Must be called first, before run_security_scan and run_unit_tests.
        Repeated calls return the cached result immediately.
        """
        if results.install is not None:
            _, summary = results.install
            return f"[CACHED — already run] {summary}"
        ok, summary = _run_install(sandbox)
        results.install = (ok, summary)
        return summary

    @tool
    def run_security_scan() -> str:
        """
        Run OWASP Dependency-Check against the workspace and check for remaining
        target CVE/GHSA identifiers.
        Must be called after run_dependency_install.
        Repeated calls return the cached result immediately.
        """
        if results.scan is not None:
            _, summary, _ = results.scan
            return f"[CACHED — already run] {summary}"
        if results.install is None:
            return (
                "ERROR: run_security_scan must be called after "
                "run_dependency_install."
            )
        ok, summary, remaining = _run_security_scan(
            sandbox, workspace_volume, target_identifiers
        )
        results.scan = (ok, summary, remaining)
        return summary

    @tool
    def run_unit_tests() -> str:
        """
        Run 'npm test' inside the workspace.
        Must be called after run_security_scan.
        Repeated calls return the cached result immediately.
        """
        if results.tests is not None:
            _, summary = results.tests
            return f"[CACHED — already run] {summary}"
        if results.scan is None:
            return (
                "ERROR: run_unit_tests must be called after run_security_scan."
            )
        ok, summary = _run_unit_tests(sandbox)
        results.tests = (ok, summary)
        return summary

    # ------------------------------------------------------------------
    # Review tools (read-only, always callable)
    # ------------------------------------------------------------------

    @tool
    def list_changed_files() -> str:
        """
        List the repo-relative file paths that the remedy agents reported as changed.
        Use this before generate_workspace_diff or read_file_context to understand
        the remediation scope.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if not candidate_changed_files:
            return "(no changed files were reported by remedy agents)"
        return "\n".join(f"  - {f}" for f in candidate_changed_files)

    @tool
    def generate_workspace_diff() -> str:
        """
        Generate a unified diff of only the remedy-agent-reported changed files,
        comparing the host baseline to the current sandbox workspace.
        Call this when a workaround/scan paradox needs investigation or when a
        CODE_WORKAROUND group requires diff-based peer review.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if host_repo_root is None:
            return "ERROR: host_repo_root is not available; cannot generate diff."
        diff_text, _ = _generate_workspace_diff(
            host_repo_root, sandbox, candidate_changed_files
        )
        return diff_text

    @tool
    def read_file_context(file_path: str) -> str:
        """
        Read the current content of a workspace file for review.
        Only accepts repo-relative paths; absolute paths and '..' traversal are rejected.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        try:
            rel_path = _validate_qa_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: File '{rel_path}' not found in workspace."
        if len(content) > _FILE_READ_MAX_CHARS:
            content = content[:_FILE_READ_MAX_CHARS] + "\n... (truncated)"
        return content

    @tool
    def query_qa_logs(log_type: str) -> str:
        """
        Return the bounded log output for a specific QA execution phase.
        log_type must be one of: 'install', 'scan', 'tests'.
        Use this instead of re-running a tool when you need a longer excerpt
        of a previously-run command's output.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if log_type == "install":
            if results.install is None:
                return "ERROR: run_dependency_install has not been called yet."
            _, summary = results.install
            return summary[:_LOG_QUERY_MAX_CHARS]
        if log_type == "scan":
            if results.scan is None:
                return "ERROR: run_security_scan has not been called yet."
            _, summary, _ = results.scan
            return summary[:_LOG_QUERY_MAX_CHARS]
        if log_type == "tests":
            if results.tests is None:
                return "ERROR: run_unit_tests has not been called yet."
            _, summary = results.tests
            return summary[:_LOG_QUERY_MAX_CHARS]
        return "ERROR: log_type must be one of: 'install', 'scan', 'tests'."

    @tool
    def search_codebase_pattern(root_dir: str = ".", pattern: str = "") -> str:
        """Search the workspace after the fixed QA pipeline has completed."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(search_tool.invoke({"root_dir": root_dir, "pattern": pattern}))

    @tool
    def inspect_ast_symbol(file_path: str, symbol_name: str) -> str:
        """Inspect a specific symbol after the fixed QA pipeline has completed."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(inspect_tool.invoke({"file_path": file_path, "symbol_name": symbol_name}))

    tools = [
        run_dependency_install,
        run_security_scan,
        run_unit_tests,
        list_changed_files,
        generate_workspace_diff,
        read_file_context,
        search_codebase_pattern,
        inspect_ast_symbol,
        query_qa_logs,
    ]
    return tools, results


# ---------------------------------------------------------------------------
# Map phase: read-only review toolbelt for individual investigators
# ---------------------------------------------------------------------------


def build_qa_review_toolbelt(
    sandbox: DockerSandbox,
    candidate_changed_files: List[str],
    host_repo_root: Optional[str],
    results: _QAExecutionResults,
) -> List:
    """
    Build a read-only review toolbelt for individual group investigators.

    This toolbelt contains NO execution tools.  The results cache is pre-populated
    by _run_global_execution (Step 0) before any investigator runs.

    Tools: list_changed_files, generate_workspace_diff, read_file_context,
           search_codebase_pattern, inspect_ast_symbol, query_qa_logs.
    """
    from src.orchestrator.remedy_tools import (
        _make_inspect_ast_symbol_tool,
        _make_search_codebase_pattern_tool,
    )

    search_tool = _make_search_codebase_pattern_tool(sandbox)
    inspect_tool = _make_inspect_ast_symbol_tool(sandbox)

    @tool
    def list_changed_files() -> str:
        """
        List the repo-relative file paths that the remedy agents reported as changed.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if not candidate_changed_files:
            return "(no changed files were reported by remedy agents)"
        return "\n".join(f"  - {f}" for f in candidate_changed_files)

    @tool
    def generate_workspace_diff() -> str:
        """
        Generate a unified diff of remedy-agent-reported changed files vs. host baseline.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if host_repo_root is None:
            return "ERROR: host_repo_root is not available; cannot generate diff."
        diff_text, _ = _generate_workspace_diff(
            host_repo_root, sandbox, candidate_changed_files
        )
        return diff_text

    @tool
    def read_file_context(file_path: str) -> str:
        """
        Read the current content of a workspace file for review.
        Only accepts repo-relative paths; absolute paths and '..' traversal are rejected.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        try:
            rel_path = _validate_qa_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: File '{rel_path}' not found in workspace."
        if len(content) > _FILE_READ_MAX_CHARS:
            content = content[:_FILE_READ_MAX_CHARS] + "\n... (truncated)"
        return content

    @tool
    def query_qa_logs(log_type: str) -> str:
        """
        Return cached QA log output. log_type: 'install', 'scan', or 'tests'.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if log_type == "install":
            if results.install is None:
                return "ERROR: run_dependency_install has not been called yet."
            _, summary = results.install
            return summary[:_LOG_QUERY_MAX_CHARS]
        if log_type == "scan":
            if results.scan is None:
                return "ERROR: run_security_scan has not been called yet."
            _, summary, _ = results.scan
            return summary[:_LOG_QUERY_MAX_CHARS]
        if log_type == "tests":
            if results.tests is None:
                return "ERROR: run_unit_tests has not been called yet."
            _, summary = results.tests
            return summary[:_LOG_QUERY_MAX_CHARS]
        return "ERROR: log_type must be one of: 'install', 'scan', 'tests'."

    @tool
    def search_codebase_pattern(root_dir: str = ".", pattern: str = "") -> str:
        """Search the workspace for a regex pattern in source files."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(search_tool.invoke({"root_dir": root_dir, "pattern": pattern}))

    @tool
    def inspect_ast_symbol(file_path: str, symbol_name: str) -> str:
        """Inspect a specific named symbol in a workspace source file."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(inspect_tool.invoke({"file_path": file_path, "symbol_name": symbol_name}))

    return [
        list_changed_files,
        generate_workspace_diff,
        read_file_context,
        search_codebase_pattern,
        inspect_ast_symbol,
        query_qa_logs,
    ]


# ---------------------------------------------------------------------------
# Map phase: individual investigator prompt builder
# ---------------------------------------------------------------------------


def _build_individual_investigator_prompt(
    group: VulnerabilityGroup,
    strategy: str,
    results: _QAExecutionResults,
    group_remaining_ids: List[str],
    candidate_changed_files: List[str],
    action_summaries: List[AgentActionSummary],
) -> str:
    """Build a group-scoped system prompt for one individual investigator."""
    fix_plan = group.fix_plan
    fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
    fix_instruction = fix_plan.instruction if fix_plan else "(none)"
    cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
    ghsas = ", ".join(group.ghsa_ids or []) or "(none)"

    install_ok, install_summary = results.install or (False, "not run")
    scan_ok, scan_summary, _ = results.scan or (False, "not run", set())
    tests_ok, tests_summary = results.tests or (False, "not run")

    summaries_text = "\n".join(
        f"  - {s.status.value}: {s.summary}" for s in action_summaries
    ) or "  (none)"

    changed_files_text = (
        "\n".join(f"  - {f}" for f in candidate_changed_files)
        if candidate_changed_files
        else "  (none)"
    )

    remaining_text = (
        ", ".join(group_remaining_ids) if group_remaining_ids else "(none — scanner cleared this group)"
    )

    return f"""You are a QA Investigator Agent assigned to review exactly ONE vulnerability group.

## Your Assigned Group
- Group ID       : {group.group_id}
- Component      : {group.vulnerable_component or '(unknown)'}
- Issue Type     : {group.issue_type.value}
- Routing Strategy: {strategy}
- CVEs           : {cves}
- GHSAs          : {ghsas}
- Fix Plan Status: {fix_plan_status}
- Fix Instruction: {fix_instruction}

## Agent Action Summaries for This Group
{summaries_text}

## Global Execution Results (already completed — do NOT re-run)
- Install: {'SUCCESS' if install_ok else 'FAILED'}
  {install_summary[:1500]}

- Security Scan: {'SUCCESS' if scan_ok else 'FAILED'}
  {scan_summary[:1500]}

- Unit Tests: {'SUCCESS' if tests_ok else 'FAILED'}
  {tests_summary[:2000]}

## This Group's Deterministic Remaining Scanner Identifiers
{remaining_text}

## Files Changed by Remedy Agents (all groups combined)
{changed_files_text}

## Your Task
You are investigating ONLY this group ({group.group_id}). Use the provided review tools
(list_changed_files, generate_workspace_diff, read_file_context, search_codebase_pattern,
inspect_ast_symbol, query_qa_logs) as needed to answer the following questions.

All three execution tools (install, scan, tests) have ALREADY been run globally.
Do NOT attempt to call run_dependency_install, run_security_scan, or run_unit_tests —
they are not available to you.

## Questions to Answer
1. Package/Domain Purpose: What does this package/component do? What domain does it serve?
2. Relevant Global Failures: Which (if any) of the install/scan/test failures are relevant to this group's domain?
3. Plausible Causation: Did this group's remediation plausibly cause the observed install or test failures? Reason deductively.
4. Scanner Findings: Do the deterministic remaining scanner identifiers above indicate this group still has unresolved vulnerabilities?
5. Workaround Path Review (if CODE_WORKAROUND): Does the changed code plausibly block the vulnerable execution path? Inspect the diff or relevant files.
6. Exoneration or Uncertainty: Explicitly state whether this group is exonerated from failures attributed to other groups, or whether there is genuine uncertainty.

## Output Format
Write a free-form Markdown investigation report answering the 6 questions above.
Be specific. Reference exact test names, file names, or scanner identifiers where possible.
End with a one-sentence summary verdict for this group.

Do NOT assign a final pass/fail verdict — that is the Batch Judge's responsibility.
"""


# ---------------------------------------------------------------------------
# Map phase: run individual investigators (one per group)
# ---------------------------------------------------------------------------


def _run_individual_investigations(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    candidate_changed_files: List[str],
    sandbox: DockerSandbox,
    repo_root: Optional[str],
    results: _QAExecutionResults,
) -> Dict[str, GroupInvestigation]:
    """
    Map phase: run one bounded ReAct investigator per vulnerability group.

    Investigators run sequentially to avoid concurrent Docker/workspace access.
    Each investigator receives a group-scoped prompt and a read-only review toolbelt.
    On crash or max-rounds exceeded, a fallback investigation is synthesized from
    deterministic results for that group only.

    Returns:
        Dict[group_id, GroupInvestigation]
    """
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    known_group_ids = {group.group_id for group in valid_groups}
    investigations: Dict[str, GroupInvestigation] = {}

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "version_bump")

        remaining_scan = results.scan[2] if results.scan else set()
        group_remaining_ids = _group_remaining_identifiers(group, remaining_scan)

        relevant_summaries = _relevant_action_summaries(
            action_summaries, group.group_id, known_group_ids
        )

        # Narrow candidate files to those mentioned in action summaries for this group,
        # falling back to the full batch list if none are found.
        narrowed_files: List[str] = []
        for s in relevant_summaries:
            pass  # Action summaries don't carry file lists in current schema; use batch list
        group_candidate_files = candidate_changed_files  # use full batch list

        system_prompt = _build_individual_investigator_prompt(
            group=group,
            strategy=strategy,
            results=results,
            group_remaining_ids=group_remaining_ids,
            candidate_changed_files=group_candidate_files,
            action_summaries=relevant_summaries,
        )

        review_tools = build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=group_candidate_files,
            host_repo_root=repo_root,
            results=results,
        )

        llm = ChatOpenAI(model=model_name, temperature=0)
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Please investigate group '{group.group_id}' now. "
                    "Use review tools as needed, then write your Markdown investigation report."
                )
            ),
        ]

        logger.info("qa_critic: [Map] starting investigator for group '%s'.", group.group_id)
        try:
            loop_result = run_bounded_subagent_loop(
                llm=llm,
                tools=review_tools,
                initial_messages=initial_messages,
                touched_files=set(),
            )
            investigation_text = (loop_result.final_text or "").strip()
            transcript_parts = []
            for event in loop_result.tool_events:
                transcript_parts.append(f"[TOOL: {event.name}]\n{event.content[:1000]}")
            transcript_parts.append(f"[AGENT FINAL]\n{investigation_text}")
            tool_transcript = "\n\n".join(transcript_parts)
            errors = list(loop_result.errors)

            if not investigation_text:
                errors.append(
                    f"qa_critic: investigator for group '{group.group_id}' returned empty output; "
                    "using fallback."
                )
                investigation_text = _build_fallback_investigation_for_group(
                    group=group,
                    strategy=strategy,
                    results=results,
                    group_remaining_ids=group_remaining_ids,
                    reason="Investigator returned empty output.",
                )

            investigations[group.group_id] = GroupInvestigation(
                group_id=group.group_id,
                investigation_text=investigation_text,
                tool_transcript=tool_transcript,
                errors=errors,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"qa_critic: investigator for group '{group.group_id}' crashed: {exc}"
            logger.error(err)
            fallback_text = _build_fallback_investigation_for_group(
                group=group,
                strategy=strategy,
                results=results,
                group_remaining_ids=group_remaining_ids,
                reason=f"Investigator crashed: {exc}",
            )
            investigations[group.group_id] = GroupInvestigation(
                group_id=group.group_id,
                investigation_text=fallback_text,
                tool_transcript="",
                errors=[err],
            )
        logger.info(
            "qa_critic: [Map] investigator for '%s' complete. text_len=%d",
            group.group_id,
            len(investigations[group.group_id].investigation_text),
        )

    return investigations


def _build_fallback_investigation_for_group(
    group: VulnerabilityGroup,
    strategy: str,
    results: _QAExecutionResults,
    group_remaining_ids: List[str],
    reason: str,
) -> str:
    """Synthesize a minimal investigation for a single group when the investigator fails."""
    install_ok, _ = results.install or (False, "not run")
    tests_ok, _ = results.tests or (False, "not run")
    scan_ok = results.scan[0] if results.scan else False

    remaining_text = ", ".join(group_remaining_ids) if group_remaining_ids else "none"
    scan_status = "still_flagged" if group_remaining_ids else ("cleared" if scan_ok else "scan_failed")

    return (
        f"## Fallback Investigation: {group.group_id}\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Component:** {group.vulnerable_component or '(unknown)'}\n"
        f"**Strategy:** {strategy}\n"
        f"**Target Identifiers:** {', '.join(sorted(_group_target_identifiers(group))) or 'none'}\n\n"
        f"**Deterministic Results:**\n"
        f"- Install: {'SUCCESS' if install_ok else 'FAILED'}\n"
        f"- Security Scan: {'SUCCESS' if scan_ok else 'FAILED'}\n"
        f"- Scan Status for this group: {scan_status}\n"
        f"- Remaining Scanner Identifiers: {remaining_text}\n"
        f"- Unit Tests: {'PASSED' if tests_ok else 'FAILED'}\n\n"
        f"**Summary:** Fallback investigation synthesized from deterministic results only. "
        f"No investigator prose was available."
    )


# ---------------------------------------------------------------------------
# Reduce phase: batch judge
# ---------------------------------------------------------------------------


def _build_batch_judge_prompt(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    results: _QAExecutionResults,
    investigations_by_group: Dict[str, GroupInvestigation],
) -> str:
    """Build the single comprehensive prompt for the batch judge."""
    known_group_ids = {group.group_id for group in valid_groups}

    install_ok, install_summary = results.install or (False, "not run")
    scan_ok, scan_summary, remaining_global = results.scan or (False, "not run", set())
    tests_ok, tests_summary = results.tests or (False, "not run")

    # Detect install conflict type for guardrail hint
    install_conflict_hint = ""
    if not install_ok:
        for pattern in _PEER_CONFLICT_PATTERNS:
            if pattern.lower() in install_summary.lower():
                install_conflict_hint = (
                    f"\n⚠️  Install failure contains '{pattern}' — "
                    "VERSION_BUMP failures should be categorized as PEER_CONFLICT."
                )
                break

    group_sections = []
    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "version_bump")
        fix_plan = group.fix_plan
        fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
        fix_instruction = fix_plan.instruction if fix_plan else "(none)"
        cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
        ghsas = ", ".join(group.ghsa_ids or []) or "(none)"
        group_remaining = _group_remaining_identifiers(group, remaining_global)
        remaining_text = ", ".join(group_remaining) if group_remaining else "(none)"

        relevant_summaries = _relevant_action_summaries(
            action_summaries, group.group_id, known_group_ids
        )
        summaries_text = "\n".join(
            f"    - {s.status.value}: {s.summary}" for s in relevant_summaries
        ) or "    (none)"

        investigation = investigations_by_group.get(group.group_id)
        investigation_text = (
            investigation.investigation_text if investigation else "(no investigation available)"
        )

        group_sections.append(
            f"---\n"
            f"### Group: {group.group_id}\n"
            f"- Component      : {group.vulnerable_component or '(unknown)'}\n"
            f"- Strategy       : {strategy}\n"
            f"- CVEs           : {cves}\n"
            f"- GHSAs          : {ghsas}\n"
            f"- Fix Plan Status: {fix_plan_status}\n"
            f"- Fix Instruction: {fix_instruction}\n"
            f"- **Remaining Scanner Identifiers (deterministic):** {remaining_text}\n"
            f"- Agent Action Summaries:\n{summaries_text}\n\n"
            f"**Individual Investigation:**\n{investigation_text}\n"
        )

    all_group_sections = "\n".join(group_sections)

    return f"""You are the Batch Judge in a map-reduce QA evaluation pipeline.

You have received individual investigation reports for {len(valid_groups)} vulnerability group(s).
Your job is to synthesize these into a holistic report and emit exactly one QAEvaluation per group.

## Global Execution Results

### Install
- Success: {install_ok}
- Summary: {install_summary[:2000]}{install_conflict_hint}

### Security Scan
- Success: {scan_ok}
- Summary: {scan_summary[:2000]}

### Unit Tests
- Success: {tests_ok}
- Summary: {tests_summary[:3000]}

## Individual Group Investigations

{all_group_sections}

## Evaluation Rules

1. **VERSION_BUMP passes** only when:
   - Install succeeded, AND
   - This group's Remaining Scanner Identifiers is "(none)", AND
   - Tests pass OR the investigation explicitly attributes test failures to a DIFFERENT group and exonerates this one.

2. **CODE_WORKAROUND may pass** even when scanner still flags vulnerabilities, if:
   - The code review or diff proves the vulnerable execution path is blocked, AND
   - Tests pass OR the investigation exonerates this group from test failures.
   - Trust the workaround code review for CODE_WORKAROUND groups.

3. Use **PEER_CONFLICT** for install failures caused by dependency conflicts (ERESOLVE, EBADENGINE, peer tree).

4. Use **BREAKING_CHANGE** for test regressions or behavior changes caused by this group's remediation.

5. Use **SECURITY_FLAG** for unresolved scanner evidence or flawed workaround logic.

6. **Do not double-attribute** the same test failure to multiple groups unless evidence explicitly supports multiple causes.

7. Resolve contradictions between individual investigations using the deterministic scanner results as the ground truth.

## Output Requirements

Return a BatchQAResult with:
- `holistic_report`: A markdown narrative listing: (a) responsible groups, (b) possibly responsible groups, (c) exonerated groups. Reference specific test names, scanner IDs, or diff evidence.
- `evaluations`: A list of exactly {len(valid_groups)} QAEvaluation objects, one per group.
  - Each evaluation must have: group_id (exact), passed (bool), failure_category (null if passed), retry_feedback (null if passed, specific actionable guidance if failed).

You MUST emit exactly {len(valid_groups)} evaluations, one for each group ID listed above.
"""


def _run_batch_judge(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    results: _QAExecutionResults,
    investigations_by_group: Dict[str, GroupInvestigation],
) -> BatchQAResult:
    """
    Reduce phase: one structured LLM call across all group investigations.

    Uses ChatOpenAI.with_structured_output(BatchQAResult) exactly once.
    On LLM failure, synthesizes a failed BatchQAResult for all groups.
    """
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(BatchQAResult)

    prompt = _build_batch_judge_prompt(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        results=results,
        investigations_by_group=investigations_by_group,
    )

    logger.info("qa_critic: [Reduce] invoking batch judge for %d groups.", len(valid_groups))
    try:
        batch_result: BatchQAResult = llm.invoke(prompt)
        logger.info(
            "qa_critic: [Reduce] batch judge returned %d evaluations.",
            len(batch_result.evaluations),
        )
        return batch_result
    except Exception as exc:  # noqa: BLE001
        logger.error("qa_critic: batch judge LLM failed — %s", exc)
        fallback_evals = [
            QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"Batch QA Judge LLM failed entirely: {exc}. "
                    "Cannot evaluate this group; please retry."
                ),
            )
            for group in valid_groups
        ]
        return BatchQAResult(
            holistic_report=(
                f"## Batch Judge Failure\n\nThe batch judge LLM call failed: {exc}\n\n"
                "All groups marked as failed with SECURITY_FLAG pending retry."
            ),
            evaluations=fallback_evals,
        )


# ---------------------------------------------------------------------------
# Python guardrails
# ---------------------------------------------------------------------------


def _apply_guardrails(
    valid_groups: List[VulnerabilityGroup],
    batch_result: BatchQAResult,
    results: _QAExecutionResults,
    group_strategies: Dict[str, str],
) -> Tuple[Dict[str, QAEvaluation], List[str]]:
    """
    Normalize and validate BatchQAResult evaluations into a Dict[group_id, QAEvaluation].

    Guardrails applied (in order):
    1. Unknown group_ids in evaluations are dropped with an error.
    2. Duplicate evaluations: keep the first, log an error.
    3. Missing groups: synthesize passed=False / SECURITY_FLAG evaluation.
    4. Deterministic scanner guardrail for VERSION_BUMP groups:
       If remaining scanner identifiers exist → force passed=False / SECURITY_FLAG.
       CODE_WORKAROUND groups are exempt (trust the workaround code review).
    5. Install error guardrail: if install failed with ERESOLVE/EBADENGINE/peer text,
       downgrade BREAKING_CHANGE to PEER_CONFLICT for VERSION_BUMP groups.

    Returns:
        (evaluations_dict, error_list)
    """
    known_group_ids = {group.group_id for group in valid_groups}
    errors: List[str] = []

    # Phase 1: deduplicate and filter unknown group_ids
    seen: Set[str] = set()
    normalized: Dict[str, QAEvaluation] = {}
    for evaluation in batch_result.evaluations:
        gid = evaluation.group_id
        if gid not in known_group_ids:
            errors.append(
                f"qa_critic guardrail: batch judge emitted evaluation for unknown group '{gid}'; dropped."
            )
            continue
        if gid in seen:
            errors.append(
                f"qa_critic guardrail: duplicate evaluation for group '{gid}'; keeping first."
            )
            continue
        seen.add(gid)
        normalized[gid] = evaluation

    # Phase 2: fill missing groups
    for group in valid_groups:
        if group.group_id not in normalized:
            errors.append(
                f"qa_critic guardrail: batch judge omitted group '{group.group_id}'; synthesizing failure."
            )
            normalized[group.group_id] = QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Batch QA Judge omitted this group; retry required.",
            )

    # Phase 3: deterministic scanner guardrail
    remaining_global: Set[str] = results.scan[2] if results.scan else set()
    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "version_bump")
        group_remaining = _group_remaining_identifiers(group, remaining_global)
        if not group_remaining:
            continue  # Scanner cleared this group — no guardrail needed.
        if strategy == "code_workaround":
            continue  # Trust the workaround code review — exempt from forced failure.
        # VERSION_BUMP with remaining identifiers → force failed.
        current = normalized[group.group_id]
        if current.passed:
            errors.append(
                f"qa_critic guardrail: group '{group.group_id}' (VERSION_BUMP) has remaining "
                f"scanner identifiers {group_remaining} but judge marked passed=True; "
                "forcing passed=False / SECURITY_FLAG."
            )
            normalized[group.group_id] = QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"Scanner still detects unresolved identifiers: {', '.join(group_remaining)}. "
                    "The version bump did not fully resolve the vulnerability. "
                    "Check that the correct version was applied."
                ),
            )

    # Phase 4: install conflict guardrail — map BREAKING_CHANGE → PEER_CONFLICT
    install_ok = results.install[0] if results.install else True
    if not install_ok:
        install_summary = results.install[1] if results.install else ""
        is_peer_conflict = any(
            p.lower() in install_summary.lower() for p in _PEER_CONFLICT_PATTERNS
        )
        if is_peer_conflict:
            for group in valid_groups:
                strategy = group_strategies.get(group.group_id, "version_bump")
                if strategy != "version_bump":
                    continue
                current = normalized[group.group_id]
                if not current.passed and current.failure_category == FailureCategory.BREAKING_CHANGE:
                    normalized[group.group_id] = QAEvaluation(
                        group_id=group.group_id,
                        passed=False,
                        failure_category=FailureCategory.PEER_CONFLICT,
                        retry_feedback=(
                            (current.retry_feedback or "")
                            + " [Guardrail: install failure contains peer conflict indicators; "
                            "reclassified from BREAKING_CHANGE to PEER_CONFLICT.]"
                        ),
                    )

    return normalized, errors


# ---------------------------------------------------------------------------
# QA agent system prompt (backcompat — used by legacy _run_investigator_phase)
# ---------------------------------------------------------------------------


def _build_qa_system_prompt(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    candidate_changed_files: List[str],
) -> str:
    """Build the system prompt for the bounded QA agent loop (backcompat)."""
    known_group_ids = {group.group_id for group in valid_groups}
    groups_text_parts = []
    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "(unknown)")
        fix_plan = group.fix_plan
        fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
        fix_instruction = fix_plan.instruction if fix_plan else "(none)"
        cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
        ghsas = ", ".join(group.ghsa_ids or []) or "(none)"

        relevant_summaries = _relevant_action_summaries(
            action_summaries,
            group.group_id,
            known_group_ids,
        )
        summaries_text = "\n".join(
            f"    - {s.status.value}: {s.summary}" for s in relevant_summaries
        ) or "    (none)"

        groups_text_parts.append(
            f"  GROUP: {group.group_id}\n"
            f"    Component   : {group.vulnerable_component or '(unknown)'}\n"
            f"    Strategy    : {strategy}\n"
            f"    CVEs        : {cves}\n"
            f"    GHSAs       : {ghsas}\n"
            f"    Fix Status  : {fix_plan_status}\n"
            f"    Instruction : {fix_instruction}\n"
            f"    Agent Summaries:\n{summaries_text}"
        )

    groups_text = "\n\n".join(groups_text_parts)
    changed_files_text = (
        "\n".join(f"  - {f}" for f in candidate_changed_files)
        if candidate_changed_files
        else "  (none)"
    )

    return f"""You are a QA Critic Agent performing a structured security remediation review.

## Vulnerability Groups Under Review
{groups_text}

## Files Changed by Remedy Agents
{changed_files_text}

## Required Execution Sequence
You MUST call the following three tools in this exact order before using any review tools:
1. `run_dependency_install` — installs dependencies and surfaces install errors
2. `run_security_scan` — runs OWASP Dependency-Check and checks for remaining CVEs/GHSAs
3. `run_unit_tests` — runs the full test suite

Each execution tool is one-shot guarded: repeated calls return the cached result.
Do NOT skip or reorder these three steps.

## Review Tools (Conditional — Call Only When Needed)
After running all three execution tools, use review tools only if failures or ambiguous
signals require investigation:
- `list_changed_files` — list the files the remedy agents modified
- `generate_workspace_diff` — diff changed files vs. host baseline (for workaround/scan paradox)
- `read_file_context` — read a specific workspace file
- `search_codebase_pattern` — regex search across workspace source files
- `inspect_ast_symbol` — extract a named function or class from a file
- `query_qa_logs` — retrieve the bounded log output for install, scan, or tests

If all three execution tools pass cleanly (install OK, zero remaining identifiers, tests OK),
you MAY finalize without calling any review tools.

When finished, you MUST output a final markdown report in this exact overall shape:

# INVESTIGATIVE REPORT

## 0. Holistic Batch Analysis (Chain of Thought)
Step 1 - Failure Identification: (List the exact test names or install errors that failed. If none, state 'None').
Step 2 - Package Domain Mapping: (Briefly state the core functionality or domain of EVERY package evaluated in this batch).
Step 3 - Causal Linkage: (Logically connect the failures from Step 1 to the specific package domains from Step 2. e.g., 'Test X is a cryptography test, so it was broken by package Y').
Step 4 - Exoneration: (Explicitly list the packages whose domains have no logical connection to the failures).

## 1. Install Analysis

## Install Analysis
- Install Status: succeeded | failed
- Summary: ...
- Suspected Responsible Group(s): ...
- Evidence: ...

### GROUP: <exact group_id>
- Component: ...
- Strategy: VERSION_BUMP | CODE_WORKAROUND
- Target Identifiers: ...
- Changed Files: ...

- Scan Status: cleared | still_flagged | scan_failed
- Remaining Scanner Findings: ...
- Scan Reasoning: ...

- Workaround Review: ...
- Diff Evidence: ...

- Test Status: passed | failed | not_run
- Attributed Test Failures: ...
- Causal Reasoning: ... (If tests failed, you MUST deduce which package update likely caused it. Use deductive reasoning based on the package's domain. Do not state that failures were not attributed.)
- Exonerated Groups: ... (Explicitly list the group_ids that are unrelated to the test failure, so they are not unfairly penalized.)

- Group Summary: ...

Rules:
- Emit exactly one `### GROUP: <exact group_id>` block for every group in this batch. There are {len(valid_groups)} groups in this batch. You MUST output exactly {len(valid_groups)} `### GROUP: <exact group_id>` blocks. Do NOT stop early. Do NOT consolidate them.
- Use the exact group_id string in each heading.
- Use exact group_id strings in `Exonerated Groups`.
- Use exact CVE/GHSA/package names in `Remaining Scanner Findings`.
- `Workaround Review` and `Diff Evidence` are required for CODE_WORKAROUND groups.
- Do NOT assign final pass/fail verdicts or failure categories. Provide forensic reasoning only.
"""


# ---------------------------------------------------------------------------
# Backcompat: Investigator and Judge helpers (old single-agent path)
# ---------------------------------------------------------------------------


def _build_group_evidence_packet(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    results: _QAExecutionResults,
    parsed_report: ParsedInvestigationReport,
    group: VulnerabilityGroup,
) -> GroupEvidencePacket:
    """Build the Judge packet for one vulnerability group (backcompat)."""
    known_group_ids = {candidate.group_id for candidate in valid_groups}
    strategy = group_strategies.get(group.group_id, "(unknown)")
    fix_plan = group.fix_plan
    fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
    fix_plan_instruction = fix_plan.instruction if fix_plan else "(none)"
    install_ok, install_summary = results.install or (
        False, "run_dependency_install was not called."
    )
    if results.scan is None:
        scan_ok, scan_summary, remaining_identifiers = (
            False, "run_security_scan was not called.", set()
        )
    else:
        scan_ok, scan_summary, remaining_identifiers = results.scan
    tests_ok, tests_summary = results.tests or (False, "run_unit_tests was not called.")
    relevant_summaries = _relevant_action_summaries(
        action_summaries,
        group.group_id,
        known_group_ids,
    )
    section = parsed_report.group_sections[group.group_id]
    return GroupEvidencePacket(
        group=group,
        strategy=strategy,
        fix_plan_status=fix_plan_status,
        fix_plan_instruction=fix_plan_instruction,
        action_summaries=[
            f"{summary.status.value}: {summary.summary}"
            for summary in relevant_summaries
        ],
        shared_install_analysis=parsed_report.shared_install_analysis,
        install_ok=install_ok,
        install_summary=install_summary,
        scan_ok=scan_ok,
        scan_summary=scan_summary,
        remaining_identifiers=_group_remaining_identifiers(group, remaining_identifiers),
        tests_ok=tests_ok,
        tests_summary=tests_summary,
        group_block_markdown=section.raw_text,
        section=section,
    )


def _run_investigator_phase(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    candidate_changed_files: List[str],
    sandbox: DockerSandbox,
    repo_root: Optional[str],
    workspace_volume: str,
    target_identifiers: Set[str],
) -> InvestigationArtifact:
    """Run the bounded QA investigator agent and parse its final report (backcompat)."""
    tools, results = build_qa_toolbelt(
        sandbox=sandbox,
        workspace_volume=workspace_volume,
        target_identifiers=target_identifiers,
        candidate_changed_files=candidate_changed_files,
        host_repo_root=repo_root,
    )

    system_prompt = _build_qa_system_prompt(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        candidate_changed_files=candidate_changed_files,
    )

    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    initial_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                "Begin your QA review now. "
                "Call run_dependency_install, run_security_scan, and run_unit_tests "
                "in order first. Then use review tools only if needed. "
                "Finish by outputting the required '# INVESTIGATIVE REPORT' markdown heading. "
                "Ensure you go through the reasoning steps in Section 0. of the report before referencing this for the group analysis. "
            )
        ),
    ]

    logger.info("qa_critic: starting bounded QA agent loop.")
    loop_result = run_bounded_subagent_loop(
        llm=llm,
        tools=tools,
        initial_messages=initial_messages,
        touched_files=set(),
    )
    logger.info(
        "qa_critic: agent loop complete. final_text_len=%d tool_events=%d",
        len(loop_result.final_text),
        len(loop_result.tool_events),
    )

    transcript_parts = []
    for event in loop_result.tool_events:
        transcript_parts.append(f"[TOOL: {event.name}]\n{event.content[:1000]}")
    transcript_parts.append(f"[AGENT FINAL]\n{loop_result.final_text}")
    agent_transcript = "\n\n".join(transcript_parts)

    errors = list(loop_result.errors)
    report_text = (loop_result.final_text or "").strip()
    if not report_text.startswith(_REPORT_PREFIX):
        errors.append(
            "qa_critic: investigator did not return the required INVESTIGATIVE REPORT format; "
            "synthesizing fallback report."
        )
        report_text = _build_fallback_investigation_report(
            valid_groups=valid_groups,
            group_strategies=group_strategies,
            candidate_changed_files=candidate_changed_files,
            results=results,
            reason="Investigator output was missing or malformed.",
        )

    parsed_report = _parse_investigation_report(report_text, valid_groups)
    errors.extend(parsed_report.errors)
    errors.extend(parsed_report.warnings)
    return InvestigationArtifact(
        report_text=report_text,
        parsed_report=parsed_report,
        transcript=agent_transcript,
        results=results,
        errors=errors,
    )


def _run_judge_phase(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    investigation: InvestigationArtifact,
) -> Dict[str, QAEvaluation]:
    """Run the zero-shot Judge once per vulnerability group (backcompat)."""
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(QAEvaluation)

    evaluations: Dict[str, QAEvaluation] = {}
    for group in valid_groups:
        packet = _build_group_evidence_packet(
            valid_groups=valid_groups,
            group_strategies=group_strategies,
            action_summaries=action_summaries,
            results=investigation.results,
            parsed_report=investigation.parsed_report,
            group=group,
        )
        cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
        ghsas = ", ".join(group.ghsa_ids or []) or "(none)"
        summaries_text = "\n".join(f"- {summary}" for summary in packet.action_summaries) or "- (none)"

        prompt = f"""You are the Judge phase of a QA Critic for one vulnerability group.

## Group
- Group ID: {group.group_id}
- Component: {group.vulnerable_component or '(unknown)'}
- Issue Type: {group.issue_type.value}
- CVEs: {cves}
- GHSAs: {ghsas}
- Routing Strategy: {packet.strategy}
- Fix Plan Status: {packet.fix_plan_status}
- Fix Plan Instruction: {packet.fix_plan_instruction}

## Shared Install Analysis
{packet.shared_install_analysis or '(none)'}

## Group-Specific Investigation Block
{packet.group_block_markdown}

## Deterministic Flags (Hard Rules)
- Global Install Success: {packet.install_ok}
- Global Tests Success: {packet.tests_ok}
- THIS GROUP'S Remaining Scanner Identifiers: {', '.join(packet.remaining_identifiers) if packet.remaining_identifiers else '(none)'}

## Relevant Action Summaries
{summaries_text}

## Parsed Group Fields
- Scan Reasoning: {packet.section.scan_reasoning or '(none)'}
- Workaround Review: {packet.section.workaround_review or '(none)'}
- Diff Evidence: {packet.section.diff_evidence or '(none)'}
- Attributed Test Failures: {packet.section.attributed_test_failures or '(none)'}
- Causal Reasoning: {packet.section.causal_reasoning or '(none)'}
- Exonerated Groups: {packet.section.exonerated_groups or '(none)'}
- Group Summary: {packet.section.group_summary or '(none)'}

## Evaluation Rules
1. VERSION_BUMP passes only when install succeeds, THIS GROUP'S Remaining Scanner Identifiers is "(none)" under Deterministic Flags (Hard Rules), AND (tests pass OR the investigation report explicitly attributes the test failures to a different group / exonerates this group).
2. CODE_WORKAROUND may pass even when the scanner flags that there are still vulnerabilities when the code review or diff proves the vulnerable path is blocked AND (normal tests pass OR the investigation report explicitly exonerates this group from the test failures).
3. Use PEER_CONFLICT for install failures caused by dependency conflicts such as ERESOLVE, EBADENGINE, or peer tree incompatibilities.
4. Use BREAKING_CHANGE for test regressions or behavior changes caused by this group's remediation.
5. Use SECURITY_FLAG for unresolved scanner evidence or flawed workaround logic.
6. Judge only this group. Ignore any missing data about other groups.

Return a QAEvaluation for group_id="{group.group_id}" with:
- passed: true/false
- failure_category: null when passed=true; otherwise PEER_CONFLICT, BREAKING_CHANGE, or SECURITY_FLAG
- retry_feedback: null when passed=true; otherwise specific, actionable guidance for the remediation agent
"""

        try:
            evaluation: QAEvaluation = llm.invoke(prompt)
            evaluations[group.group_id] = evaluation
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "qa_critic: LLM evaluation failed for group %s - %s",
                group.group_id,
                exc,
            )
            evaluations[group.group_id] = QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"QA Critic LLM evaluation raised an exception: {exc}. "
                    "Please retry the remediation."
                ),
            )

    return evaluations


# ---------------------------------------------------------------------------
# Compatibility wrapper and node entry point
# ---------------------------------------------------------------------------


def _extract_group_evaluations(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    results: _QAExecutionResults,
    agent_transcript: str,
) -> Dict[str, QAEvaluation]:
    """Backward-compatible wrapper around the Judge phase."""
    parsed_report = _parse_investigation_report(
        agent_transcript if agent_transcript.strip().startswith(_REPORT_PREFIX)
        else _build_fallback_investigation_report(
            valid_groups=valid_groups,
            group_strategies=group_strategies,
            candidate_changed_files=[],
            results=results,
            reason="Legacy evaluation call did not provide a structured investigation report.",
        ),
        valid_groups,
    )
    artifact = InvestigationArtifact(
        report_text=parsed_report.raw_report,
        parsed_report=parsed_report,
        transcript=agent_transcript,
        results=results,
        errors=parsed_report.errors + parsed_report.warnings,
    )
    return _run_judge_phase(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        investigation=artifact,
    )


@traceable(name="qa_critic")
def run_qa_critic_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node: run the map-reduce QA pipeline and evaluate each group.

    Pipeline:
      Step 0 — Global Execution (deterministic Python, no LLM tools)
      Map     — One bounded ReAct investigator per group (read-only tools)
      Reduce  — One batch judge call (with_structured_output(BatchQAResult))
      Guards  — Python guardrails normalize and validate evaluations
    """
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups") or []
    workspace_volume: Optional[str] = state.get("workspace_volume")
    repo_root: Optional[str] = state.get("repo_root")
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []
    group_strategies: Dict[str, str] = state.get("group_strategies") or {}
    candidate_changed_files: List[str] = state.get("changed_files") or []

    if not valid_groups and not state.get("force_qa"):
        logger.info("qa_critic: no valid groups - skipping QA.")
        return {
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "changed_files": [],
            "qa_investigation_report": "",
        }

    if not workspace_volume:
        err = "qa_critic: workspace_volume is not set; cannot run QA."
        logger.error(err)
        failed_evals = {
            group.group_id: QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: workspace_volume is missing.",
            )
            for group in valid_groups
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": [err],
            "changed_files": [],
            "qa_investigation_report": "",
        }

    target_identifiers = _collect_target_identifiers(valid_groups)
    logger.info(
        "qa_critic: evaluating %d groups with %d target identifiers.",
        len(valid_groups),
        len(target_identifiers),
    )

    errors: List[str] = []
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # ------------------------------------------------------------------
            # Step 0: Global Execution (deterministic Python, exactly once)
            # ------------------------------------------------------------------
            results = _run_global_execution(
                sandbox=sandbox,
                workspace_volume=workspace_volume,
                target_identifiers=target_identifiers,
            )

            # ------------------------------------------------------------------
            # Pipeline completeness guard
            # ------------------------------------------------------------------
            missing_tools: List[str] = []
            if results.install is None:
                missing_tools.append("run_dependency_install")
            if results.scan is None:
                missing_tools.append("run_security_scan")
            if results.tests is None:
                missing_tools.append("run_unit_tests")
            if missing_tools:
                missing_list = ", ".join(missing_tools)
                err = (
                    f"qa_critic: global execution did not complete all required steps; "
                    f"missing: {missing_list}."
                )
                logger.error(err)
                errors.append(err)
                for tool_name in missing_tools:
                    errors.append(
                        f"qa_critic: required tool '{tool_name}' did not produce a result."
                    )
                failed_evals = {
                    group.group_id: QAEvaluation(
                        group_id=group.group_id,
                        passed=False,
                        failure_category=FailureCategory.SECURITY_FLAG,
                        retry_feedback=(
                            f"QA global execution incomplete: {missing_list} did not run. "
                            "Please retry."
                        ),
                    )
                    for group in valid_groups
                }
                return {
                    "qa_evaluations": failed_evals,
                    "eval_status": "failures_detected",
                    "status": "qa_failed",
                    "errors": errors,
                    "changed_files": [],
                    "qa_investigation_report": "",
                }

            # ------------------------------------------------------------------
            # Map: Individual Investigators (one per group, sequential)
            # ------------------------------------------------------------------
            investigations_by_group = _run_individual_investigations(
                valid_groups=valid_groups,
                group_strategies=group_strategies,
                action_summaries=action_summaries,
                candidate_changed_files=candidate_changed_files,
                sandbox=sandbox,
                repo_root=repo_root,
                results=results,
            )


    except RuntimeError as exc:
        err = f"qa_critic: Docker sandbox unavailable - {exc}"
        logger.error(err)
        errors.append(err)
        failed_evals = {
            group.group_id: QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: Docker sandbox could not start.",
            )
            for group in valid_groups
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": errors,
            "changed_files": [],
            "qa_investigation_report": "",
        }

    # Collect errors from map phase
    for investigation in investigations_by_group.values():
        errors.extend(investigation.errors)

    # ------------------------------------------------------------------
    # Reduce: Batch Judge (one LLM call for all groups)
    # ------------------------------------------------------------------
    batch_result = _run_batch_judge(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        results=results,
        investigations_by_group=investigations_by_group,
    )

    # ------------------------------------------------------------------
    # Python Guardrails
    # ------------------------------------------------------------------
    qa_evaluations, guardrail_errors = _apply_guardrails(
        valid_groups=valid_groups,
        batch_result=batch_result,
        results=results,
        group_strategies=group_strategies,
    )
    errors.extend(guardrail_errors)

    all_passed = all(evaluation.passed for evaluation in qa_evaluations.values())
    eval_status = "all_passed" if all_passed else "failures_detected"

    logger.info(
        "qa_critic: eval_status=%s (%d/%d passed).",
        eval_status,
        sum(1 for evaluation in qa_evaluations.values() if evaluation.passed),
        len(qa_evaluations),
    )

    return {
        "qa_evaluations": qa_evaluations,
        "eval_status": eval_status,
        "status": "qa_completed",
        "changed_files": candidate_changed_files,
        "errors": errors,
        "qa_investigation_report": batch_result.holistic_report,
    }
