"""
qa_critic.py - Agentic QA evaluator node for the Phase 5 orchestrator.

The QA Critic is a bounded ReAct agent that:
  1. Runs npm install, OWASP Dependency-Check, and npm test exactly once via
     one-shot-guarded execution tools.
  2. Optionally calls read-only review tools (diff, file reads, pattern search,
     AST inspection, log queries) when failures or ambiguous signals need
     investigation.
  3. Calls a single-shot structured LLM per group to produce QAEvaluation
     objects after the agent review phase completes.

Heavy QA commands (install, scan, tests) are intentionally *not* exposed as
tools to the update or workaround subagents; they live here only.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langsmith import traceable

from src.contracts.schemas import (
    AgentActionSummary,
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
_ODC_CACHE_VOLUME = "odc-cache"

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
_DIFF_EXCLUDE_NAMES = frozenset({_ODC_REPORT_NAME, "dependency-check-report.html"})


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
# QA runner helpers (deterministic, called by tool wrappers)
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
        return False, summary, set()

    if found_identifiers is None:
        summary = (
            f"FAILURE: Dependency-Check report was not parseable "
            f"(exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        return False, summary, set()

    remaining = {ident.upper().strip() for ident in target_identifiers if ident}
    remaining &= found_identifiers

    if remaining:
        summary = (
            "FAILURE: Dependency-Check still reports the following target "
            f"identifier(s): {', '.join(sorted(remaining))}"
        )
        return False, summary, remaining

    return True, "Dependency-Check found no remaining target vulnerability identifiers.", set()


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
    """Cache for one-shot execution tool results. Populated during the agent loop."""

    install: Optional[Tuple[bool, str]] = None          # (ok, summary)
    scan: Optional[Tuple[bool, str, Set[str]]] = None   # (ok, summary, remaining_ids)
    tests: Optional[Tuple[bool, str]] = None            # (ok, summary)


# ---------------------------------------------------------------------------
# QA toolbelt
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

    tools = [
        run_dependency_install,
        run_security_scan,
        run_unit_tests,
        list_changed_files,
        generate_workspace_diff,
        read_file_context,
        _make_search_codebase_pattern_tool(sandbox),
        _make_inspect_ast_symbol_tool(sandbox),
        query_qa_logs,
    ]
    return tools, results


# ---------------------------------------------------------------------------
# QA agent system prompt
# ---------------------------------------------------------------------------


def _build_qa_system_prompt(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    candidate_changed_files: List[str],
) -> str:
    """Build the system prompt for the bounded QA agent loop."""
    groups_text_parts = []
    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "(unknown)")
        fix_plan = group.fix_plan
        fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
        fix_instruction = fix_plan.instruction if fix_plan else "(none)"
        cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
        ghsas = ", ".join(group.ghsa_ids or []) or "(none)"

        relevant_summaries = [s for s in action_summaries if s.group_id == group.group_id]
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

## Final Evaluation Rules
Produce one verdict per group after completing your review:

1. **VERSION_BUMP groups**: Pass only when:
   - Install succeeds (ERESOLVE/EBADENGINE → PEER_CONFLICT)
   - Scanner reports zero remaining target identifiers (otherwise SECURITY_FLAG)
   - Unit tests pass (regressions → BREAKING_CHANGE)

2. **CODE_WORKAROUND groups**: Pass when:
   - The workspace diff clearly neutralizes the vulnerable code path
   - Normal (non-exploit) tests still pass
   - Scanner may still report findings — suppress as EXPECTED FALSE POSITIVE only when
     the diff proves the path is blocked; otherwise → SECURITY_FLAG
   - Failing exploit-specific tests are acceptable if the diff proves the path is blocked

3. **Failure categories**:
   - PEER_CONFLICT: ERESOLVE, EBADENGINE, or dependency-tree conflict in install
   - BREAKING_CHANGE: regression or API breakage in unit tests
   - SECURITY_FLAG: unresolved scanner evidence or unblocked vulnerable path

When finished with your review, clearly state your final conclusions per group.
The structured evaluation will be extracted from your analysis afterward.
"""


# ---------------------------------------------------------------------------
# Per-group structured evaluation (called after agent loop)
# ---------------------------------------------------------------------------


def _extract_group_evaluations(
    valid_groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    results: _QAExecutionResults,
    agent_transcript: str,
) -> Dict[str, QAEvaluation]:
    """
    Run a single-shot structured LLM call per group to extract a QAEvaluation.

    Accepts the full agent loop transcript (tool events + final text) and
    per-group context, and uses structured output to produce a typed verdict.

    Returns a mapping of group_id -> QAEvaluation.
    """
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(QAEvaluation)

    install_ok, install_summary = results.install or (
        False, "run_dependency_install was not called."
    )
    if results.scan is not None:
        scan_ok, scan_summary, remaining_identifiers = results.scan
    else:
        scan_ok, scan_summary, remaining_identifiers = (
            False, "run_security_scan was not called.", set()
        )
    test_ok, test_summary = results.tests or (False, "run_unit_tests was not called.")

    evaluations: Dict[str, QAEvaluation] = {}

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "(unknown)")
        fix_plan = group.fix_plan
        fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
        fix_plan_instruction = fix_plan.instruction if fix_plan else "(none)"
        cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
        ghsas = ", ".join(group.ghsa_ids or []) or "(none)"

        relevant_summaries = [s for s in action_summaries if s.group_id == group.group_id]
        summaries_text = "\n".join(
            f"  - {s.status.value}: {s.summary}" for s in relevant_summaries
        ) or "  (none)"

        remaining_text = (
            ", ".join(sorted(remaining_identifiers)) if remaining_identifiers else "(none)"
        )

        prompt = f"""You are a QA Critic extracting a structured verdict for one vulnerability group.

## Group: {group.group_id}
- Component: {group.vulnerable_component or '(unknown)'}
- Issue Type: {group.issue_type.value}
- CVEs: {cves}
- GHSAs: {ghsas}
- Fix Plan Status: {fix_plan_status}
- Fix Plan Instruction: {fix_plan_instruction}
- Routing Strategy: {strategy}

## Agent Action Summaries
{summaries_text}

## Execution Results
- Install Success: {install_ok}
- Install Summary: {install_summary[:2000]}
- Scan Success: {scan_ok}
- Scan Summary: {scan_summary[:2000]}
- Remaining Target Identifiers: {remaining_text}
- Tests Success: {test_ok}
- Tests Summary: {test_summary[:3000]}

## QA Agent Review Transcript
{agent_transcript[:6000]}

## Evaluation Rules
1. VERSION_BUMP: Pass only when install succeeds, zero remaining target identifiers,
   and tests pass.
2. CODE_WORKAROUND: Pass when diff proves vulnerable path is blocked and normal tests pass.
   Suppress scanner findings as false positives ONLY when the diff proves blocking.
3. PEER_CONFLICT: ERESOLVE, EBADENGINE, or dependency-tree conflict.
4. BREAKING_CHANGE: regression or API breakage in tests.
5. SECURITY_FLAG: unresolved scanner evidence or unblocked vulnerable path.

Return a QAEvaluation for group_id="{group.group_id}" with:
- passed: true/false
- failure_category: null when passed=true; otherwise PEER_CONFLICT, BREAKING_CHANGE, or SECURITY_FLAG
- retry_feedback: null when passed=true; otherwise specific, actionable guidance for the agent
"""

        try:
            evaluation: QAEvaluation = llm.invoke(prompt)
            evaluations[group.group_id] = evaluation
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "qa_critic: LLM evaluation failed for group %s — %s",
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
# Node entry point
# ---------------------------------------------------------------------------


@traceable(name="qa_critic")
def run_qa_critic_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node: run the agentic QA pipeline and evaluate each group.

    The QA agent loop:
      1. Must call run_dependency_install, run_security_scan, run_unit_tests (in order).
      2. May call read-only review tools when failures or ambiguous signals need
         investigation.
      3. After the loop, per-group QAEvaluation is extracted via structured LLM output.

    Returns updates to OrchestratorState.
    """
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups") or []
    workspace_volume: Optional[str] = state.get("workspace_volume")
    repo_root: Optional[str] = state.get("repo_root")
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []
    group_strategies: Dict[str, str] = state.get("group_strategies") or {}
    candidate_changed_files: List[str] = state.get("changed_files") or []

    if not valid_groups:
        logger.info("qa_critic: no valid groups — skipping QA.")
        return {
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "changed_files": [],
        }

    if not workspace_volume:
        err = "qa_critic: workspace_volume is not set; cannot run QA."
        logger.error(err)
        failed_evals = {
            g.group_id: QAEvaluation(
                group_id=g.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: workspace_volume is missing.",
            )
            for g in valid_groups
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": [err],
            "changed_files": [],
        }

    target_identifiers = _collect_target_identifiers(valid_groups)
    logger.info(
        "qa_critic: evaluating %d groups with %d target identifiers.",
        len(valid_groups),
        len(target_identifiers),
    )

    errors: List[str] = []
    loop_result = None
    results: Optional[_QAExecutionResults] = None

    # ------------------------------------------------------------------
    # Open sandbox and run bounded QA agent loop
    # ------------------------------------------------------------------
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
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
                        "When finished, state your final conclusions per group."
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

            if loop_result.errors:
                errors.extend(loop_result.errors)

    except RuntimeError as exc:
        err = f"qa_critic: Docker sandbox unavailable — {exc}"
        logger.error(err)
        errors.append(err)
        failed_evals = {
            g.group_id: QAEvaluation(
                group_id=g.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="QA infrastructure failure: Docker sandbox could not start.",
            )
            for g in valid_groups
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": errors,
            "changed_files": [],
        }

    # ------------------------------------------------------------------
    # Verify that all required execution tools were called by the agent
    # ------------------------------------------------------------------
    assert results is not None  # guarded by the except above
    missing_tools = []
    if results.install is None:
        missing_tools.append("run_dependency_install")
    if results.scan is None:
        missing_tools.append("run_security_scan")
    if results.tests is None:
        missing_tools.append("run_unit_tests")

    if missing_tools:
        err = (
            f"qa_critic: agent did not call required execution tool(s): "
            f"{', '.join(missing_tools)}. Cannot produce a valid evaluation."
        )
        logger.error(err)
        errors.append(err)
        failed_evals = {
            g.group_id: QAEvaluation(
                group_id=g.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"QA agent failed to run required execution tool(s): "
                    f"{', '.join(missing_tools)}. This is an infrastructure failure. "
                    "Please retry."
                ),
            )
            for g in valid_groups
        }
        return {
            "qa_evaluations": failed_evals,
            "eval_status": "failures_detected",
            "status": "qa_failed",
            "errors": errors,
            "changed_files": candidate_changed_files,
        }

    # ------------------------------------------------------------------
    # Build agent transcript and extract per-group structured evaluations
    # ------------------------------------------------------------------
    assert loop_result is not None
    transcript_parts = []
    for event in loop_result.tool_events:
        transcript_parts.append(f"[TOOL: {event.name}]\n{event.content[:1000]}")
    transcript_parts.append(f"[AGENT FINAL]\n{loop_result.final_text}")
    agent_transcript = "\n\n".join(transcript_parts)

    logger.info(
        "qa_critic: extracting structured evaluations for %d groups.", len(valid_groups)
    )
    qa_evaluations = _extract_group_evaluations(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        results=results,
        agent_transcript=agent_transcript,
    )

    all_passed = all(ev.passed for ev in qa_evaluations.values())
    eval_status = "all_passed" if all_passed else "failures_detected"

    logger.info(
        "qa_critic: eval_status=%s (%d/%d passed).",
        eval_status,
        sum(1 for ev in qa_evaluations.values() if ev.passed),
        len(qa_evaluations),
    )

    return {
        "qa_evaluations": qa_evaluations,
        "eval_status": eval_status,
        "status": "qa_completed",
        "changed_files": candidate_changed_files,
        "errors": errors,
    }
