"""
qa_critic.py - Standalone QA evaluator node for the Phase 5 orchestrator.

The QA Critic:
  1. Opens the shared workspace volume (read-only for heavy tools).
  2. Runs npm install, OWASP Dependency-Check, and npm test exactly once.
  3. Generates a host-vs-workspace diff (no .git in the Docker volume).
  4. Calls a single-shot structured LLM to produce per-group ``QAEvaluation``.

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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.contracts.schemas import (
    AgentActionSummary,
    FailureCategory,
    FixPlanStatus,
    QAEvaluation,
    RoutingStrategy,
    VulnerabilityGroup,
)
from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

from langsmith import traceable

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
# QA runner helpers
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
    host_root = Path(host_repo_root)
    changed_files: List[str] = []
    diff_parts: List[str] = []

    # Deduplicate files while preserving order.
    seen = set()
    unique_candidates = []
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
# LLM critic
# ---------------------------------------------------------------------------


def _build_group_prompt(
    group: VulnerabilityGroup,
    strategy: Optional[str],
    relevant_summaries: List[AgentActionSummary],
    install_ok: bool,
    install_summary: str,
    scan_ok: bool,
    scan_summary: str,
    remaining_identifiers: Set[str],
    test_ok: bool,
    test_summary: str,
    diff_text: str,
    changed_files: List[str],
) -> str:
    """Build a structured prompt for evaluating a single vulnerability group."""
    fix_plan = group.fix_plan
    fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
    fix_plan_instruction = fix_plan.instruction if fix_plan else "(none)"

    cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
    ghsas = ", ".join(group.ghsa_ids) if group.ghsa_ids else "(none)"

    summaries_text = "\n".join(
        f"  - group={s.group_id} status={s.status.value}: {s.summary}"
        for s in relevant_summaries
    ) or "  (none)"

    remaining_text = ", ".join(sorted(remaining_identifiers)) if remaining_identifiers else "(none)"
    changed_files_text = "\n".join(f"  - {f}" for f in changed_files) or "  (none)"

    return f"""You are a QA Critic evaluating whether a security remediation was successful.

## Group: {group.group_id}
- Component: {group.vulnerable_component or '(unknown)'}
- Issue Type: {group.issue_type.value}
- CVEs: {cves}
- GHSAs: {ghsas}
- Fix Plan Status: {fix_plan_status}
- Fix Plan Instruction: {fix_plan_instruction}
- Routing Strategy: {strategy or '(unknown)'}

## Agent Action Summaries
{summaries_text}

## Install Result
- Success: {install_ok}
- Summary: {install_summary[:2000]}

## Security Scan Result
- Success: {scan_ok}
- Summary: {scan_summary[:2000]}
- Remaining Target Identifiers: {remaining_text}

## Unit Test Result
- Success: {test_ok}
- Summary: {test_summary[:3000]}

## Workspace Diff (changed files)
{changed_files_text}

## Workspace Diff (content, capped)
```
{diff_text}
```

## Evaluation Rules

1. **Version bump groups** (strategy=VERSION_BUMP): Pass only when:
   - Install succeeds (ERESOLVE/EBADENGINE → PEER_CONFLICT).
   - Scanner reports zero remaining target identifiers (otherwise SECURITY_FLAG).
   - Unit tests pass (regressions → BREAKING_CHANGE).

2. **Code workaround groups** (strategy=CODE_WORKAROUND): Pass when:
   - The diff clearly neutralizes the vulnerable code path.
   - Normal (non-security-exploit) tests still pass.
   - Scanner may still report findings — this is an EXPECTED FALSE POSITIVE for
     CODE_WORKAROUND; suppress scanner findings if the diff proves the path is
     blocked.
   - If the diff does NOT prove the path is blocked, classify as SECURITY_FLAG.

3. **CTF/exploit tests**: Failing exploit-specific tests can mean success ONLY
   when the diff clearly neutralizes the exploit AND normal tests are not broken.

4. **Failure categories**:
   - PEER_CONFLICT: ERESOLVE, EBADENGINE, or dependency-tree conflict in install.
   - BREAKING_CHANGE: regression/API breakage in tests.
   - SECURITY_FLAG: unresolved scanner evidence or unblocked vulnerable path.

Return a QAEvaluation for group_id="{group.group_id}" with:
- passed: true/false
- failure_category: null when passed=true; otherwise PEER_CONFLICT, BREAKING_CHANGE, or SECURITY_FLAG
- retry_feedback: null when passed=true; otherwise specific, actionable guidance for the agent
"""


def _run_llm_critic(
    groups: List[VulnerabilityGroup],
    group_strategies: Dict[str, str],
    action_summaries: List[AgentActionSummary],
    install_ok: bool,
    install_summary: str,
    scan_ok: bool,
    scan_summary: str,
    remaining_identifiers: Set[str],
    test_ok: bool,
    test_summary: str,
    diff_text: str,
    changed_files: List[str],
) -> Dict[str, QAEvaluation]:
    """
    Run a single-shot structured LLM evaluation for each group.

    Returns a mapping of group_id -> QAEvaluation.
    """
    from langchain_openai import ChatOpenAI

    model_name = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(QAEvaluation)

    evaluations: Dict[str, QAEvaluation] = {}

    for group in groups:
        strategy = group_strategies.get(group.group_id)

        relevant_summaries = [
            s for s in action_summaries if s.group_id == group.group_id
        ]

        prompt = _build_group_prompt(
            group=group,
            strategy=strategy,
            relevant_summaries=relevant_summaries,
            install_ok=install_ok,
            install_summary=install_summary,
            scan_ok=scan_ok,
            scan_summary=scan_summary,
            remaining_identifiers=remaining_identifiers,
            test_ok=test_ok,
            test_summary=test_summary,
            diff_text=diff_text,
            changed_files=changed_files,
        )

        try:
            evaluation: QAEvaluation = llm.invoke(prompt)
            evaluations[group.group_id] = evaluation
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "qa_critic: LLM evaluation failed for group %s — %s",
                group.group_id,
                exc,
            )
            # Synthesise a failure evaluation so the supervisor can route.
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

@traceable(name="qa_critic") # for testing only
def run_qa_critic_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node: run the deterministic QA pipeline and evaluate each group.

    Runs exactly once per orchestrator invocation:
      1. npm install
      2. OWASP Dependency-Check
      3. npm test
      4. workspace diff (host vs sandbox)
    Then calls the LLM critic per group.

    Returns updates to ``OrchestratorState``.
    """
    valid_groups: List[VulnerabilityGroup] = state.get("valid_groups") or []
    workspace_volume: Optional[str] = state.get("workspace_volume")
    repo_root: Optional[str] = state.get("repo_root")
    action_summaries: List[AgentActionSummary] = state.get("action_summaries") or []
    group_strategies: Dict[str, str] = state.get("group_strategies") or {}

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

    # ------------------------------------------------------------------
    # Open sandbox and run QA pipeline (exactly once)
    # ------------------------------------------------------------------
    errors: List[str] = []
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # 1. Install
            logger.info("qa_critic: running npm install.")
            install_ok, install_summary = _run_install(sandbox)
            if not install_ok:
                logger.warning("qa_critic: install failed — %s", install_summary[:200])

            # 2. Security scan
            logger.info("qa_critic: running OWASP Dependency-Check.")
            scan_ok, scan_summary, remaining_identifiers = _run_security_scan(
                sandbox, workspace_volume, target_identifiers
            )
            if not scan_ok:
                logger.warning("qa_critic: scan failed — %s", scan_summary[:200])

            # 3. Unit tests
            logger.info("qa_critic: running npm test.")
            test_ok, test_summary = _run_unit_tests(sandbox)
            if not test_ok:
                logger.warning("qa_critic: tests failed — %s", test_summary[:200])

            # 4. Workspace diff
            logger.info("qa_critic: generating workspace diff.")
            if repo_root:
                diff_text, changed_files = _generate_workspace_diff(
                    repo_root,
                    sandbox,
                    state.get("changed_files") or [],
                )
            else:
                diff_text, changed_files = "", []

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
    # LLM evaluation (one call per group, outside sandbox context)
    # ------------------------------------------------------------------
    logger.info("qa_critic: running LLM critic for %d groups.", len(valid_groups))
    qa_evaluations = _run_llm_critic(
        groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        install_ok=install_ok,
        install_summary=install_summary,
        scan_ok=scan_ok,
        scan_summary=scan_summary,
        remaining_identifiers=remaining_identifiers,
        test_ok=test_ok,
        test_summary=test_summary,
        diff_text=diff_text,
        changed_files=changed_files,
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
        "changed_files": changed_files,
        "errors": errors,
    }
