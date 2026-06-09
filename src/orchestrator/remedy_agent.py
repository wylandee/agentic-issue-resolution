"""
remedy_agent.py — Phase 5 Remedy Agent node for the AppSec Orchestrator.

Public API
----------
run_remedy_agent(state: OrchestratorState) -> Dict[str, Any]
    LangGraph node that reads each ``VulnerabilityGroup`` in the state,
    resolves its target file on the host, constructs a detailed prompt for
    the LLM, and returns a validated list of ``EditRequest`` objects.

Design principles
-----------------
* **Read-only** — this node never mutates files or runs commands.  It only
  reads the current file content from the host and generates EditRequests.
* **No ``REMEDY_LLM_ENABLED`` gate** — unlike the triage agent, this node is
  always expected to call the LLM when invoked.  If the LLM call fails, the
  error is appended to state and the node returns ``status="remedy_failed"``.
* **Self-correction loop** — when ``test_failures`` or ``scan_failures`` are
  present in state, the node treats this as a retry attempt and includes the
  failure context in the prompt.  If ``retry_count >= max_retries``, the node
  returns immediately without invoking the LLM.
* **Strict preflight validation** — each returned ``EditRequest`` is validated:
  ``old_text`` must appear exactly once in the current file; ``file_path``
  must match the resolved target; security checks reject absolute paths,
  directory traversal, files outside repo_root, and non-UTF-8 files.

Environment variables
---------------------
REMEDY_LLM_MODEL  : OpenAI model name (default: ``gpt-4o-mini``)
OPENAI_API_KEY    : Required at runtime
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.contracts.schemas import (
    EditRequest,
    IssueType,
    LocalizedIssue,
    RemedyAgentOutput,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

# Lazy import guard for LangChain — allows the module to be imported even when
# langchain-openai is not installed.  The actual requirement is checked at
# node-call time inside run_remedy_agent().
try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
except ImportError:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Target file resolution
# ---------------------------------------------------------------------------


def _resolve_target_file(
    group: VulnerabilityGroup,
    repo_root: Path,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the repo-relative target file for a vulnerability group.

    For SCA groups the resolution order is:
      1. ``localized_issues[0].manifest_file``
      2. ``group.file_path``
      3. ``group.issues[0].file_path``

    For SAST groups the resolution order is:
      1. ``group.file_path``
      2. ``group.issues[0].file_path``

    Returns
    -------
    (relative_path, error_message)
        On success: ``(str, None)``
        On failure: ``(None, reason_str)``
    """
    candidate: Optional[str] = None

    if group.issue_type == IssueType.SCA:
        # Prefer manifest_file from localized issues
        for li in group.localized_issues:
            if li.manifest_file:
                candidate = li.manifest_file
                break
        if not candidate:
            candidate = group.file_path
        if not candidate and group.issues:
            candidate = group.issues[0].file_path
    else:  # SAST
        candidate = group.file_path
        if not candidate and group.issues:
            candidate = group.issues[0].file_path

    if not candidate:
        return None, f"Group '{group.group_id}': no target file could be resolved."

    # Security checks
    # ``os.path.isabs`` is platform-specific on Windows, so we also treat any
    # explicit leading slash or backslash as absolute for repo-relative paths.
    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        return None, (
            f"Group '{group.group_id}': rejected absolute file path '{candidate}'."
        )

    parts = Path(candidate).parts
    if ".." in parts:
        return None, (
            f"Group '{group.group_id}': rejected path traversal in '{candidate}'."
        )

    abs_target = (repo_root / candidate).resolve()

    # Ensure the resolved path is inside repo_root
    try:
        abs_target.relative_to(repo_root.resolve())
    except ValueError:
        return None, (
            f"Group '{group.group_id}': path '{candidate}' resolves outside repo_root."
        )

    if not abs_target.exists():
        return None, (
            f"Group '{group.group_id}': target file '{candidate}' does not exist in repo."
        )
    if abs_target.is_dir():
        return None, (
            f"Group '{group.group_id}': target path '{candidate}' is a directory."
        )

    return candidate, None


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_group_section(group: VulnerabilityGroup) -> str:
    """Render a structured description of the vulnerability group."""
    cve_list = ", ".join(group.cve_ids) if group.cve_ids else "none"
    versions = ", ".join(group.versions) if group.versions else "unknown"
    sources = ", ".join(s.value for s in group.sources) if group.sources else "unknown"

    # Representative issue details
    rep_issue = next(
        (i for i in group.issues if str(i.id) == str(group.representative_issue_id)),
        group.issues[0] if group.issues else None,
    )
    rep_id_str = str(group.representative_issue_id)
    rep_msg = (rep_issue.message or "N/A") if rep_issue else "N/A"

    # Member issue summary (max 5 to avoid token bloat)
    member_lines = []
    for issue in group.issues[:5]:
        member_lines.append(
            f"  - ID={issue.id} | CVE={issue.cve_id or 'N/A'} | "
            f"severity={issue.severity.value} | file={issue.file_path or 'N/A'}"
        )
    if len(group.issues) > 5:
        member_lines.append(f"  ... and {len(group.issues) - 5} more.")

    parts = [
        f"Group ID      : {group.group_id}",
        f"Issue Type    : {group.issue_type.value}",
        f"Component     : {group.vulnerable_component or 'unknown'}",
        f"File Path     : {group.file_path or 'N/A'}",
        f"CVEs          : {cve_list}",
        f"Versions      : {versions}",
        f"Sources       : {sources}",
        f"Rep. Issue ID : {rep_id_str}",
        f"Rep. Message  : {rep_msg}",
        "Member Issues:",
    ]
    parts.extend(member_lines)
    return "\n".join(parts)


def _build_fix_plan_section(group: VulnerabilityGroup) -> str:
    """Render fix plan details or a note that none is available."""
    fp = group.fix_plan
    if fp is None:
        return (
            "No fix plan is available for this group. You MUST derive the smallest "
            "safe security fix directly from the vulnerability data, CVE descriptions, "
            "and the current file content. Do NOT invent versions; if a safe version "
            "is not determinable from the file content, emit an empty edits list for "
            "this group rather than guessing."
        )
    lines = [
        f"Status         : {fp.status.value}",
        f"Strategy       : {fp.strategy_used}",
        f"Fixed Version  : {fp.fixed_version or 'N/A'}",
        f"Instruction    : {fp.instruction}",
    ]
    if fp.workaround_snippets:
        lines.append("Workaround Snippets:")
        for snippet in fp.workaround_snippets[:3]:
            lines.append(f"  ---\n{snippet}\n  ---")
    return "\n".join(lines)


def _build_target_section(
    group: VulnerabilityGroup,
    rel_path: str,
    repo_root: str,
    file_content: str,
) -> str:
    abs_path = str(Path(repo_root) / rel_path)
    return "\n".join([
        f"Absolute Path  : {abs_path}  (for human context only)",
        f"file_path      : {rel_path}  ← use this EXACTLY in your EditRequest",
        f"repo_root      : {repo_root}  ← use this EXACTLY in your EditRequest",
        "",
        "Current file content:",
        "```",
        file_content,
        "```",
    ])


def _build_feedback_section(
    test_failures: Optional[str],
    scan_failures: Optional[str],
    retry_count: int,
    max_retries: int,
) -> str:
    """Build a clearly visible failure-feedback section for retry prompts."""
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║        ⚠  PREVIOUS ATTEMPT FAILED — SELF-CORRECT  ⚠ ║",
        "╚══════════════════════════════════════════════════════╝",
        f"Retry {retry_count} of {max_retries}.",
        "",
        "Your previous edit(s) did NOT fix the issue.  Study the errors below and",
        "produce a corrected set of EditRequests that addresses the root cause.",
        "",
    ]
    if test_failures:
        lines += ["=== UNIT TEST FAILURES ===", test_failures, ""]
    if scan_failures:
        lines += ["=== ODC SCAN FAILURES ===", scan_failures, ""]
    return "\n".join(lines)


def _build_prompt(
    group: VulnerabilityGroup,
    rel_path: str,
    repo_root: str,
    file_content: str,
    test_failures: Optional[str],
    scan_failures: Optional[str],
    retry_count: int,
    max_retries: int,
) -> str:
    """Assemble the full structured prompt for the Remedy Agent LLM call."""
    has_feedback = bool(test_failures or scan_failures)
    sections = []

    # Role
    sections.append(
        "You are an autonomous application security engineer.  "
        "Your task is to produce EXACT Search/Replace blocks (EditRequests) that fix "
        "the vulnerability described below without introducing regressions.\n"
        "You MUST NOT fabricate code that is not derived from the current file content.\n"
        "You MUST NOT change anything unrelated to the security fix."
    )

    # Failure feedback (retry path only)
    if has_feedback:
        sections.append(
            _build_feedback_section(test_failures, scan_failures, retry_count, max_retries)
        )

    # Vulnerability data
    sections.append("=== VULNERABILITY GROUP ===\n" + _build_group_section(group))

    # Fix plan
    sections.append("=== FIX PLAN ===\n" + _build_fix_plan_section(group))

    # Target file
    sections.append("=== TARGET FILE ===\n" + _build_target_section(
        group, rel_path, repo_root, file_content
    ))

    # Output instructions
    sections.append("\n".join([
        "=== OUTPUT INSTRUCTIONS ===",
        "Return a single RemedyAgentOutput object with an 'edits' list.",
        "Each item in 'edits' is an EditRequest with these fields:",
        "  repo_root  : MUST be exactly the repo_root shown above",
        "  file_path  : MUST be exactly the file_path shown above",
        "  old_text   : ⚠ CRITICAL — copy CHARACTER-FOR-CHARACTER from the file above,",
        "               including ALL whitespace and indentation.  The string must match",
        "               EXACTLY ONCE in the file.  If it matches zero or more than once,",
        "               the edit will be REJECTED.",
        "  new_text   : the corrected replacement text",
        "  dry_run    : false",
        "  rationale  : a concise one-sentence explanation of the fix",
        "",
        "If you cannot produce a safe, correct fix, return an empty 'edits' list rather",
        "than guessing.  An empty list is always safer than a wrong edit.",
    ]))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Output validation
# ---------------------------------------------------------------------------


def _validate_edits(
    raw_edits: List[EditRequest],
    group: VulnerabilityGroup,
    rel_path: str,
    repo_root: str,
    file_content: str,
) -> Tuple[List[EditRequest], List[str]]:
    """
    Validate and normalise LLM-returned ``EditRequest`` objects for one group.

    Checks applied for each edit:
    1. ``file_path`` must match ``rel_path`` for this group.
    2. ``old_text`` must appear exactly once in the current file content.
    3. ``repo_root`` is overwritten to ``state["repo_root"]``.
    4. ``issue_id`` is filled from ``group.representative_issue_id`` if missing.
    5. ``dry_run`` is kept as-is (defaults to False if not set by LLM).

    Returns
    -------
    (valid_edits, error_strings)
    """
    valid: List[EditRequest] = []
    errors: List[str] = []

    for i, edit in enumerate(raw_edits):
        label = f"Group '{group.group_id}' edit[{i}]"

        # file_path must match target
        if edit.file_path != rel_path:
            errors.append(
                f"{label}: file_path mismatch — expected '{rel_path}', "
                f"got '{edit.file_path}'. Skipped."
            )
            continue

        # old_text must match exactly once
        count = file_content.count(edit.old_text)
        if count == 0:
            errors.append(
                f"{label}: old_text not found in '{rel_path}'. Skipped."
            )
            continue
        if count > 1:
            errors.append(
                f"{label}: old_text matches {count} times in '{rel_path}' "
                "(ambiguous anchor). Skipped."
            )
            continue

        # Normalise repo_root and issue_id
        normalised = EditRequest(
            repo_root=repo_root,
            file_path=edit.file_path,
            old_text=edit.old_text,
            new_text=edit.new_text,
            dry_run=edit.dry_run,
            max_deletion_lines=edit.max_deletion_lines,
            issue_id=edit.issue_id if edit.issue_id is not None else group.representative_issue_id,
            rationale=edit.rationale,
        )
        valid.append(normalised)

    return valid, errors


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def run_remedy_agent(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Remedy Agent.

    Reads each ``VulnerabilityGroup`` in ``state["valid_groups"]``, resolves its
    target file, calls the LLM (via LangChain structured output), validates the
    returned edits, and returns an updated state dict.

    Returns one of:
    * ``{"edit_requests": [...], "status": "edits_generated"}``
    * ``{"status": "no_edits_generated", "errors": [...]}``
    * ``{"status": "remedy_failed", "errors": [...]}``
    * ``{"status": "max_retries_exceeded"}``

    The caller (LangGraph) merges the returned dict into the running state.
    """
    repo_root_str: str = state.get("repo_root", "")
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups", [])
    retry_count: int = state.get("retry_count", 0)
    max_retries: int = state.get("max_retries", 3)
    test_failures: Optional[str] = state.get("test_failures")
    scan_failures: Optional[str] = state.get("scan_failures")

    has_feedback = bool(test_failures or scan_failures)

    # -- Guard: max retries exceeded -----------------------------------------
    if has_feedback and retry_count >= max_retries:
        logger.warning(
            "Remedy Agent: retry_count=%d >= max_retries=%d — aborting.",
            retry_count,
            max_retries,
        )
        return {"status": "max_retries_exceeded"}

    # -- Validate repo_root --------------------------------------------------
    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"Remedy Agent: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {"status": "remedy_failed", "errors": [msg]}

    # -- Process groups ------------------------------------------------------
    all_valid_edits: List[EditRequest] = []
    all_errors: List[str] = []
    structured_llm = None  # lazily constructed on first successful group resolution

    for group in valid_groups:
        # Resolve target file
        rel_path, resolve_err = _resolve_target_file(group, repo_root)
        if resolve_err:
            logger.warning("Remedy Agent: %s", resolve_err)
            all_errors.append(resolve_err)
            continue

        # Read file content
        abs_target = repo_root / rel_path  # type: ignore[operator]
        try:
            file_content = abs_target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            msg = (
                f"Group '{group.group_id}': target file '{rel_path}' is not "
                "valid UTF-8 and cannot be processed."
            )
            logger.warning("Remedy Agent: %s", msg)
            all_errors.append(msg)
            continue
        except OSError as exc:
            msg = f"Group '{group.group_id}': could not read '{rel_path}' — {exc}."
            logger.warning("Remedy Agent: %s", msg)
            all_errors.append(msg)
            continue

        # Lazy-construct the LLM on the first group that passes file resolution
        if structured_llm is None:
            if ChatOpenAI is None:
                msg = (
                    "Remedy Agent: 'langchain-openai' is not installed.  "
                    "Run: pip install langchain-openai"
                )
                logger.error(msg)
                return {"status": "remedy_failed", "errors": [msg]}
            model_name = os.environ.get("REMEDY_LLM_MODEL", _DEFAULT_MODEL)
            llm = ChatOpenAI(model=model_name, temperature=0)
            structured_llm = llm.with_structured_output(RemedyAgentOutput)

        # Build prompt
        prompt = _build_prompt(
            group=group,
            rel_path=rel_path,
            repo_root=repo_root_str,
            file_content=file_content,
            test_failures=test_failures,
            scan_failures=scan_failures,
            retry_count=retry_count,
            max_retries=max_retries,
        )

        # Invoke LLM
        try:
            llm_output: RemedyAgentOutput = structured_llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001
            msg = (
                f"Group '{group.group_id}': LLM call failed — {exc}."
            )
            logger.error("Remedy Agent: %s", msg)
            all_errors.append(msg)
            continue

        # Validate returned edits
        valid_edits, validation_errors = _validate_edits(
            raw_edits=llm_output.edits,
            group=group,
            rel_path=rel_path,
            repo_root=repo_root_str,
            file_content=file_content,
        )
        all_valid_edits.extend(valid_edits)
        all_errors.extend(validation_errors)

    # -- Compute retry_count update ------------------------------------------
    new_retry_count = retry_count + 1 if has_feedback else retry_count

    # -- Build return value --------------------------------------------------
    if all_valid_edits:
        result: Dict[str, Any] = {
            "edit_requests": all_valid_edits,
            "status": "edits_generated",
        }
        if new_retry_count != retry_count:
            result["retry_count"] = new_retry_count
        if all_errors:
            result["errors"] = all_errors
        return result

    # No valid edits produced
    if all_errors:
        return {
            "status": "remedy_failed",
            "errors": all_errors,
        }

    return {
        "status": "no_edits_generated",
        "errors": [
            "Remedy Agent: LLM produced no usable edits for any group."
        ],
    }
