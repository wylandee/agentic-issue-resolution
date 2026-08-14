"""
qa_critic.py - Agentic QA evaluator node for the Phase 5 orchestrator.

The QA Critic now follows a map-reduce architecture:

  Step 0 â€” Global Execution (deterministic Python):
    run_dependency_install â†’ run_security_scan â†’ run_unit_tests, called exactly
    once via direct Python helpers, with no LLM tools involved.

  Map â€” Individual Investigators:
    One bounded ReAct agent per vulnerability group, given a group-scoped
    prompt and a read-only review toolbelt.  Each agent answers only for its
    assigned group.

  Reduce â€” Batch Judge:
    One ChatOpenAI.with_structured_output(BatchQAResult) call across the
    current group scope, normally one group because Supervisor dispatch is
    per-task, while direct batch callers remain supported.

  Python Guardrails:
    Normalize, validate, and fill missing/duplicate/unknown evaluations
    deterministically.  Enforce scanner and install-error policies.

The node preserves existing per-task QA evaluation semantics while adding
graph-level scan snapshot outputs for vulnerabilities introduced during
remediation.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    BatchQAResult,
    FailureCategory,
    FinalFullScanResult,
    FixPlanStatus,
    NoFixMitigationStage,
    ODCScanEvidence,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    ScanFallbackReason,
    ScanScope,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop
from remediation_engine.orchestration.trajectory_exporter import invoke_with_trajectory
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.settings import AppSettings
from remediation_engine.tools.lockfile_closure import (
    ClosureResolutionError,
    DependencyClosure,
    build_sliced_lockfile_artifacts,
    resolve_dependency_closure,
)

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
_TEST_FAILURE_MAX_ITEMS = 8
_TEST_FAILURE_CONTEXT_LINES = 12
_TEST_FAILURE_EXCERPT_CHARS = 500
_LOCAL_NPM_TEST_RUNNERS = frozenset({"jest", "mocha", "vitest"})

# Exclusion patterns for workspace diff
_DIFF_EXCLUDE_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "dependency-check-data",
        "coverage",
        ".nyc_output",
        ".cache",
    }
)
_DIFF_EXCLUDE_SUFFIXES = frozenset({".map", ".lock"})
_DIFF_EXCLUDE_NAMES = frozenset({_ODC_REPORT_NAME, _ODC_HTML_REPORT_NAME})

# Install error patterns that indicate peer/engine conflicts
_PEER_CONFLICT_PATTERNS = ("ERESOLVE", "EBADENGINE", "peer dep", "peer tree")
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_MOCHA_FAILURE_RE = re.compile(r"^\s*(\d+)\)\s+(.+)$")
# Jest/Vitest prefixes are normally Unicode bullets/crosses.  The mojibake
# alternatives keep parsing resilient when output was decoded with a legacy
# console code page before it reached the evaluator.
_JEST_FAILURE_RE = re.compile(r"^\s*(?:[●✕×•]|â—|âœ•|Ã—)\s+(.+)$")
_TAP_FAILURE_RE = re.compile(r"^\s*not ok\b(?:\s+\d+)?\s*-?\s*(.+)?$")
_SUBTEST_RE = re.compile(r"^\s*# Subtest:\s+(.+)$")
_FAIL_LINE_RE = re.compile(r"^\s*FAIL\b(?:\s+(.+))?$")
_EXCEPTION_RE = re.compile(
    r"(AssertionError|TypeError|ReferenceError|SyntaxError|RangeError|Error):"
)
_STACK_NOISE_RE = re.compile(r"^\s*(at\s+.+|\^\s*|[-]{3,}|\s*operator:\s+.+)$")


# ---------------------------------------------------------------------------
# Legacy ODC helpers (ported from old remedy_tools.py)
# ---------------------------------------------------------------------------


def _read_report_from_workspace(
    sandbox: DockerSandbox,
    relative_dir: str = "",
) -> str | None:
    """Read an ODC JSON report from a workspace-relative report directory."""
    try:
        clean_dir = relative_dir.strip("/\\")
        report_path = f"{clean_dir}/{_ODC_REPORT_NAME}".lstrip("/")
        return sandbox.read_file(report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to read ODC report from workspace â€” %s", exc)
        return None


def _persist_workspace_report_to_host(
    sandbox: DockerSandbox,
    workspace_name: str,
    host_path: Path,
) -> Path | None:
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


def _parse_report_identifiers(report_text: str) -> set[str] | None:
    """Parse CVE/GHSA identifiers from the ODC JSON report text."""
    issues = _parse_report_issues(report_text)
    if issues is None:
        return None

    identifiers: set[str] = set()
    for issue in issues:
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _parse_report_issues(report_text: str) -> list[VulnerabilityIssue] | None:
    """Parse the complete typed vulnerability snapshot from an ODC report."""
    try:
        report = json.loads(report_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to decode ODC report JSON â€” %s", exc)
        return None

    try:
        from remediation_engine.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning("qa_critic: remediation_engine.tools.odc_parser not importable.")
        return None

    try:
        return parse_vulnerabilities(report)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to parse ODC vulnerabilities â€” %s", exc)
        return None


def _odc_command(workspace_volume: str, scan_subdir: str | None = None) -> list[str]:
    """Build a safe ODC Docker command for the full or targeted workspace."""
    scan_path = "/scan"
    if scan_subdir:
        scan_subdir = _validate_qa_path(scan_subdir)
        scan_path = f"/scan/{scan_subdir}"
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
        scan_path,
        "--format",
        "JSON",
        "--format",
        "HTML",
        "--out",
        scan_path,
        "--noupdate",
    ]

    extra_args = os.environ.get("ODC_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def _run_odc(workspace_volume: str) -> subprocess.CompletedProcess[str]:
    """Execute OWASP Dependency-Check in Docker against the shared workspace volume."""
    cmd = _odc_command(workspace_volume)
    logger.info("qa_critic: running ODC in Docker: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_ODC_TIMEOUT_SECONDS,
    )


def _run_targeted_odc(
    workspace_volume: str,
    targeted_subdir: str,
) -> subprocess.CompletedProcess[str]:
    """Execute ODC against a validated workspace-relative targeted directory."""
    cmd = _odc_command(workspace_volume, targeted_subdir)
    logger.info("qa_critic: running targeted ODC in Docker: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_ODC_TIMEOUT_SECONDS,
    )


# ---------------------------------------------------------------------------
# QA runner helpers (deterministic, called by tool wrappers and global execution)
# ---------------------------------------------------------------------------


def _run_install(sandbox: DockerSandbox) -> tuple[bool, str]:
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


@dataclass(frozen=True, eq=False)
class _SecurityScanResult:
    """Complete deterministic security-scan outcome.

    The iterator and index accessors intentionally expose the historical
    three-value ``(ok, summary, remaining)`` projection so existing callers
    and tests can continue to consume the scanner while newer graph code uses
    the full identifier snapshot.
    """

    ok: bool
    summary: str
    remaining_identifiers: set[str]
    found_identifiers: set[str]
    new_identifiers: set[str]
    found_issues: list[VulnerabilityIssue] = field(default_factory=list)

    def _legacy_projection(self) -> tuple[bool, str, set[str]]:
        return self.ok, self.summary, self.remaining_identifiers

    def __iter__(self):
        return iter(self._legacy_projection())

    def __getitem__(self, index: int):
        return self._legacy_projection()[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _SecurityScanResult):
            return (
                self.ok == other.ok
                and self.summary == other.summary
                and self.remaining_identifiers == other.remaining_identifiers
                and self.found_identifiers == other.found_identifiers
                and self.new_identifiers == other.new_identifiers
                and self.found_issues == other.found_issues
            )
        if isinstance(other, tuple):
            return self._legacy_projection() == other
        return NotImplemented


def _run_security_scan(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str] | None = None,
) -> _SecurityScanResult:
    """
    Run OWASP Dependency-Check and classify the complete identifier snapshot.

    Returns:
        A ``_SecurityScanResult`` containing the complete post-remediation
        identifier set, unresolved target identifiers, and newly introduced
        identifiers.  Iteration remains backward-compatible with the legacy
        ``(success, summary_text, remaining_identifiers)`` shape.
        ``success=False`` when Docker is absent, ODC times out, or no
        parseable report is produced.
        ``remaining_identifiers`` is empty on success or on hard failure.
    """
    baseline = {
        identifier.upper().strip()
        for identifier in (
            baseline_identifiers if baseline_identifiers is not None else target_identifiers
        )
        if identifier and identifier.strip()
    }

    if shutil.which("docker") is None:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        logger.warning("qa_critic: %s", msg)
        return _SecurityScanResult(False, msg, set(), set(), set())

    try:
        proc = _run_odc(workspace_volume)
    except FileNotFoundError:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        return _SecurityScanResult(False, msg, set(), set(), set())
    except subprocess.TimeoutExpired:
        msg = f"FAILURE: Dependency-Check timed out after {_ODC_TIMEOUT_SECONDS}s."
        return _SecurityScanResult(False, msg, set(), set(), set())
    except Exception as exc:  # noqa: BLE001
        msg = f"FAILURE: Dependency-Check subprocess error â€” {exc}"
        return _SecurityScanResult(False, msg, set(), set(), set())

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
    found_identifiers = _parse_report_identifiers(report_text) if report_text is not None else None
    found_issues = _parse_report_issues(report_text) if report_text is not None else None
    # Legacy direct callers/tests may replace the identifier-only parser. In
    # that compatibility mode there is no typed issue snapshot to propagate,
    # but the identifier scan can still be evaluated normally.
    if found_issues is None and found_identifiers is not None:
        found_issues = []

    if proc.returncode != 0 and (found_identifiers is None or found_issues is None):
        summary = (
            f"FAILURE: Dependency-Check exited {proc.returncode} and produced "
            "no parseable report.\n"
            f"stdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        summary += report_location_note
        return _SecurityScanResult(False, summary, set(), set(), set())

    if found_identifiers is None or found_issues is None:
        summary = (
            f"FAILURE: Dependency-Check report was not parseable "
            f"(exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        summary += report_location_note
        return _SecurityScanResult(False, summary, set(), set(), set())

    found_identifiers = {
        identifier.upper().strip() for identifier in found_identifiers if identifier
    }
    remaining = {ident.upper().strip() for ident in target_identifiers if ident}
    remaining &= found_identifiers
    new_identifiers = found_identifiers - baseline

    if remaining:
        remaining_text = ", ".join(sorted(remaining))
        summary = (
            "FAILURE: Dependency-Check found unresolved target vulnerabilities. "
            f"Remaining identifiers: {remaining_text}"
        )
        if new_identifiers:
            summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
        summary += report_location_note
        return _SecurityScanResult(
            False,
            summary,
            remaining,
            found_identifiers,
            new_identifiers,
            found_issues,
        )

    summary = "Dependency-Check found no remaining target vulnerability identifiers."
    if new_identifiers:
        summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
    summary += report_location_note
    return _SecurityScanResult(
        True,
        summary,
        set(),
        found_identifiers,
        new_identifiers,
        found_issues,
    )


def _targeted_report_host_path(targeted_subdir: str, suffix: str) -> Path:
    """Return an ignored, unique host path for a targeted ODC report."""
    stamp = f"{time.time_ns()}-{abs(hash(targeted_subdir))}"
    return _ODC_DEBUG_DIR / "targeted" / f"{stamp}-{suffix}"


def _run_targeted_security_scan(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str],
    targeted_subdir: str,
) -> _SecurityScanResult:
    """Run and classify ODC against a synthetic targeted workspace."""
    baseline = {
        identifier.upper().strip()
        for identifier in baseline_identifiers
        if identifier and identifier.strip()
    }
    if shutil.which("docker") is None:
        return _SecurityScanResult(
            False,
            "FAILURE: docker is not available on PATH; Dependency-Check cannot run.",
            set(),
            set(),
            set(),
        )

    try:
        proc = _run_targeted_odc(workspace_volume, _validate_qa_path(targeted_subdir))
    except FileNotFoundError:
        return _SecurityScanResult(
            False,
            "FAILURE: docker is not available on PATH; Dependency-Check cannot run.",
            set(),
            set(),
            set(),
        )
    except subprocess.TimeoutExpired:
        return _SecurityScanResult(
            False,
            f"FAILURE: Dependency-Check timed out after {_ODC_TIMEOUT_SECONDS}s.",
            set(),
            set(),
            set(),
        )
    except Exception as exc:  # noqa: BLE001
        return _SecurityScanResult(
            False,
            f"FAILURE: Dependency-Check subprocess error — {exc}",
            set(),
            set(),
            set(),
        )

    report_dir = _validate_qa_path(targeted_subdir)
    report_json = f"{report_dir}/{_ODC_REPORT_NAME}"
    report_html = f"{report_dir}/{_ODC_HTML_REPORT_NAME}"
    saved_html = _persist_workspace_report_to_host(
        sandbox,
        report_html,
        _targeted_report_host_path(report_dir, _ODC_HTML_REPORT_NAME),
    )
    saved_json = _persist_workspace_report_to_host(
        sandbox,
        report_json,
        _targeted_report_host_path(report_dir, _ODC_REPORT_NAME),
    )
    report_location_note = ""
    if saved_html is not None:
        report_location_note = f"\nHTML report saved to: {saved_html}"
        if saved_json is not None:
            report_location_note += f"\nJSON report saved to: {saved_json}"

    report_text = _read_report_from_workspace(sandbox, report_dir)
    found_identifiers = _parse_report_identifiers(report_text) if report_text is not None else None
    found_issues = _parse_report_issues(report_text) if report_text is not None else None
    if found_issues is None and found_identifiers is not None:
        found_issues = []
    if proc.returncode != 0 and (found_identifiers is None or found_issues is None):
        summary = (
            f"FAILURE: Dependency-Check exited {proc.returncode} and produced no parseable report.\n"
            f"stdout:\n{proc.stdout[:2000]}\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        return _SecurityScanResult(False, summary + report_location_note, set(), set(), set())
    if found_identifiers is None or found_issues is None:
        summary = (
            f"FAILURE: Dependency-Check report was not parseable (exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        return _SecurityScanResult(False, summary + report_location_note, set(), set(), set())
    if proc.returncode != 0:
        summary = (
            f"FAILURE: targeted Dependency-Check exited {proc.returncode}.\n"
            f"stderr:\n{proc.stderr[:2000]}"
        )
        return _SecurityScanResult(False, summary + report_location_note, set(), set(), set())

    found = {identifier.upper().strip() for identifier in found_identifiers if identifier}
    remaining = {
        identifier.upper().strip() for identifier in target_identifiers if identifier
    } & found
    new_identifiers = found - baseline
    if remaining:
        summary = (
            "FAILURE: Dependency-Check found unresolved target vulnerabilities. "
            f"Remaining identifiers: {', '.join(sorted(remaining))}"
        )
        if new_identifiers:
            summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
        return _SecurityScanResult(
            False,
            summary + report_location_note,
            remaining,
            found,
            new_identifiers,
            found_issues,
        )

    summary = "Dependency-Check found no remaining target vulnerability identifiers."
    if new_identifiers:
        summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
    return _SecurityScanResult(
        True,
        summary + report_location_note,
        set(),
        found,
        new_identifiers,
        found_issues,
    )


@dataclass
class _TestFailureBlock:
    """Condensed failure block extracted from raw test output."""

    title: str
    excerpt: str
    source: str
    start_line: int
    end_line: int
    score: int


@dataclass(frozen=True)
class _TestSuitePlan:
    """Detected child test-suite command and runner classification."""

    name: str
    runner: str
    command: str


@dataclass
class _NormalizedTestFailure:
    """One leaf test failure from a structured or runner-aware parser."""

    name: str
    message: str = ""
    failure_type: str = ""
    suite: str = ""


@dataclass
class _NormalizedTestDiagnostic:
    """Runner-level diagnostic that should not inflate failed test counts."""

    message: str
    kind: str = "diagnostic"
    suite: str = ""


@dataclass
class _NormalizedSuiteResult:
    """Normalized result for one child test-suite command."""

    name: str
    runner: str
    command: str
    exit_code: int
    failed_tests: list[_NormalizedTestFailure] = field(default_factory=list)
    diagnostics: list[_NormalizedTestDiagnostic] = field(default_factory=list)
    fallback_summary: str = ""


def _strip_ansi(value: str) -> str:
    """Remove ANSI color/control sequences from test output."""
    return _ANSI_ESCAPE_RE.sub("", value or "")


def _normalize_log_lines(text: str) -> list[str]:
    """Normalize raw log text into plain lines for deterministic parsing."""
    normalized = _strip_ansi(text).replace("\r\n", "\n").replace("\r", "\n")
    return normalized.splitlines()


def _block_matches_failure_header(line: str) -> bool:
    """Return whether a line is a strong failure-block boundary."""
    return bool(
        _MOCHA_FAILURE_RE.match(line)
        or _JEST_FAILURE_RE.match(line)
        or _TAP_FAILURE_RE.match(line)
        or _FAIL_LINE_RE.match(line)
    )


def _score_failure_lines(title: str, excerpt: str, source: str) -> int:
    """Score extracted failure blocks so high-signal items survive caps first."""
    score = 0
    lowered_title = title.lower()
    lowered_excerpt = excerpt.lower()
    if _MOCHA_FAILURE_RE.match(title):
        score += 100
    if _JEST_FAILURE_RE.match(title):
        score += 100
    if _TAP_FAILURE_RE.match(title):
        score += 95
    if "subtest" in lowered_title:
        score += 90
    if _EXCEPTION_RE.search(excerpt):
        score += 70
    if _FAIL_LINE_RE.match(title):
        score += 50
    if "stderr" in source:
        score -= 5
    if "assertionerror" in lowered_excerpt or "typeerror" in lowered_excerpt:
        score += 15
    return score


def _capture_failure_excerpt(
    lines: list[str],
    start_index: int,
    title_line: str,
) -> tuple[str, int]:
    """Capture a bounded high-signal excerpt starting at one failure marker."""
    captured: list[str] = []
    end_index = start_index
    for index in range(start_index, min(len(lines), start_index + _TEST_FAILURE_CONTEXT_LINES)):
        line = lines[index].rstrip()
        if index > start_index and _block_matches_failure_header(line):
            break
        if index > start_index and not line.strip():
            break
        if index > start_index and _STACK_NOISE_RE.match(line):
            continue
        captured.append(line)
        end_index = index

    excerpt = "\n".join(captured).strip()
    if not excerpt:
        excerpt = title_line.strip()
    if len(excerpt) > _TEST_FAILURE_EXCERPT_CHARS:
        excerpt = excerpt[:_TEST_FAILURE_EXCERPT_CHARS].rstrip() + "... (truncated)"
    return excerpt, end_index


def _extract_failure_blocks(text: str, source: str = "stdout") -> list[_TestFailureBlock]:
    """Extract bounded failure blocks from raw test output using grep-like regex scanning."""
    lines = _normalize_log_lines(text)
    blocks: list[_TestFailureBlock] = []
    index = 0
    pending_subtest: str | None = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        title: str | None = None

        if not stripped:
            index += 1
            continue

        subtest_match = _SUBTEST_RE.match(line)
        if subtest_match:
            pending_subtest = subtest_match.group(1).strip()
            index += 1
            continue

        mocha_match = _MOCHA_FAILURE_RE.match(line)
        if mocha_match:
            title = mocha_match.group(2).strip()

        if title is None:
            jest_match = _JEST_FAILURE_RE.match(line)
            if jest_match:
                title = jest_match.group(1).strip()

        if title is None:
            tap_match = _TAP_FAILURE_RE.match(line)
            if tap_match:
                tap_title = (tap_match.group(1) or "").strip()
                title = tap_title or pending_subtest or stripped
                pending_subtest = None

        if title is None:
            fail_match = _FAIL_LINE_RE.match(line)
            if fail_match:
                title = (fail_match.group(1) or "").strip() or stripped

        if title is None and _EXCEPTION_RE.search(line):
            title = pending_subtest or stripped

        if title is None:
            index += 1
            continue

        excerpt, end_index = _capture_failure_excerpt(lines, index, stripped)
        score = _score_failure_lines(stripped, excerpt, source)
        blocks.append(
            _TestFailureBlock(
                title=title,
                excerpt=excerpt,
                source=source,
                start_line=index,
                end_line=end_index,
                score=score,
            )
        )
        index = end_index + 1

    return blocks


def _dedupe_failure_blocks(blocks: list[_TestFailureBlock]) -> list[_TestFailureBlock]:
    """Drop overlapping or repeated failure blocks while keeping the highest-signal copy."""
    deduped: list[_TestFailureBlock] = []
    seen_signatures: set[tuple[str, str]] = set()
    occupied_ranges: list[tuple[int, int, str]] = []

    for block in sorted(blocks, key=lambda item: (-item.score, item.start_line)):
        signature = (
            block.title.strip().lower(),
            block.excerpt.strip().lower(),
        )
        if signature in seen_signatures:
            continue
        overlaps = any(
            block.source == existing_source
            and not (block.end_line < existing_start or block.start_line > existing_end)
            for existing_start, existing_end, existing_source in occupied_ranges
        )
        if overlaps:
            continue
        deduped.append(block)
        seen_signatures.add(signature)
        occupied_ranges.append((block.start_line, block.end_line, block.source))

    return sorted(deduped, key=lambda item: (0 if item.source == "stdout" else 1, item.start_line))


def _fallback_raw_tail(exit_code: int, stdout: str, stderr: str) -> str:
    """Return the legacy bounded raw-tail fallback when no structured failures are detected."""
    stdout_tail = "\n".join(stdout.splitlines()[-_TEST_LOG_TAIL_LINES:])
    stderr_tail = "\n".join(stderr.splitlines()[-_STDERR_TAIL_LINES:])
    return (
        f"npm test FAILED (exit {exit_code}).\n"
        f"stdout tail:\n{stdout_tail}\n"
        f"stderr tail:\n{stderr_tail}"
    )


def _format_failure_summary(exit_code: int, blocks: list[_TestFailureBlock]) -> str:
    """Format extracted test failures into a compact LLM-facing summary."""
    visible_blocks = blocks[:_TEST_FAILURE_MAX_ITEMS]
    lines = [
        f"npm test FAILED (exit {exit_code}).",
        f"Detected Failures: {len(blocks)}",
        "",
        "Failing Tests:",
    ]
    for index, block in enumerate(visible_blocks, start=1):
        source_hint = f" [{block.source}]" if block.source == "stderr" else ""
        lines.append(f"{index}. {block.title}{source_hint}")
        excerpt_lines = block.excerpt.splitlines()
        if excerpt_lines and block.title in excerpt_lines[0]:
            excerpt_body = "\n".join(excerpt_lines[1:]).strip()
        else:
            excerpt_body = block.excerpt
        if excerpt_body:
            lines.append(excerpt_body)
        lines.append("")
    if len(blocks) > len(visible_blocks):
        lines.append(f"... and {len(blocks) - len(visible_blocks)} more failures omitted")

    summary = "\n".join(lines).strip()
    if len(summary) > _LOG_QUERY_MAX_CHARS:
        summary = summary[:_LOG_QUERY_MAX_CHARS].rstrip() + "\n... (summary truncated)"
    return summary


def _summarize_failed_test_output(exit_code: int, stdout: str, stderr: str) -> str:
    """Condense raw npm test output into a bounded failure-focused summary."""
    blocks = _dedupe_failure_blocks(
        [
            *_extract_failure_blocks(stdout, source="stdout"),
            *_extract_failure_blocks(stderr, source="stderr"),
        ]
    )
    if not blocks:
        return _fallback_raw_tail(exit_code, stdout, stderr)
    return _format_failure_summary(exit_code, blocks)


_QA_ERROR_MARKER = re.compile(
    r"(?:error|exception|failed|failure|not\s+a\s+function|undefined|cannot|invalid|missing|required\s+option|not\s+exported|cannot\s+find)",
    re.IGNORECASE,
)
_QA_SOURCE_LOCATION = re.compile(r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:/")


def _normalise_workspace_file_path(path: str) -> str | None:
    """Normalize and constrain a reported path to the QA workspace.

    Args:
        path: A path reported in test output.

    Returns:
        A repository-relative POSIX path, or ``None`` for absolute or
        traversal paths that cannot safely identify a workspace file.
    """
    normalized = (path or "").replace("\\", "/").strip()
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    if not normalized or normalized.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(normalized):
        return None
    if any(part == ".." for part in normalized.split("/")):
        return None
    return normalized


def _validate_qa_source_locations(
    source_locations: list[str],
    sandbox: DockerSandbox,
) -> tuple[list[str], list[str], list[str]]:
    """Keep only source locations that identify files in the QA workspace.

    Args:
        source_locations: Candidate ``path:line[:column]`` values extracted
            from deterministic test output.
        sandbox: Active QA sandbox used to verify file existence.

    Returns:
        A tuple containing valid normalized locations, their distinct file
        paths, and human-readable diagnostics for discarded locations.
    """
    valid_locations: list[str] = []
    valid_files: list[str] = []
    diagnostics: list[str] = []
    for location in source_locations:
        match = _QA_SOURCE_LOCATION.match(location.strip())
        path = _normalise_workspace_file_path(match.group("path")) if match else None
        if match is None:
            reason = "it is not a path:line[:column] value"
        elif path is None:
            reason = "it is not repository-relative"
        else:
            try:
                exists = sandbox.read_file(path) is not None
            except (OSError, RuntimeError) as exc:
                exists = False
                reason = f"the workspace lookup failed: {exc}"
            else:
                reason = "the file does not exist in the QA workspace"
            if exists:
                line_no = match.group("line")
                column_no = match.group("column")
                normalized_location = (
                    f"{path}:{line_no}:{column_no}" if column_no else f"{path}:{line_no}"
                )
                if normalized_location not in valid_locations:
                    valid_locations.append(normalized_location)
                if path not in valid_files:
                    valid_files.append(path)
                continue

        diagnostics.append(f"QA discarded source location '{location}': {reason}.")

    return valid_locations, valid_files, diagnostics


def extract_qa_failure_evidence(
    exit_code: int,
    stdout: str,
    stderr: str,
    attempt_id: str = "",
    task_revision: int = 0,
    *,
    sandbox: DockerSandbox | None = None,
) -> QAFailureEvidence:
    """Extract structured QA evidence from test output.

    Source locations are deterministic only when they can be verified against
    the active QA workspace.  LLM-produced evidence is intentionally handled
    separately by ``_attach_failure_evidence_to_evaluations``.

    Args:
        exit_code: Process exit code for the failed QA command.
        stdout: Captured standard output.
        stderr: Captured standard error.
        attempt_id: Committed remediation attempt identifier.
        task_revision: Committed task revision.
        sandbox: Optional active sandbox used to validate extracted paths.

    Returns:
        Structured deterministic QA failure evidence.
    """
    exact_diagnostics: list[str] = []
    failed_tests: list[str] = []
    source_locations: list[str] = []
    affected_files: list[str] = []

    blocks = _dedupe_failure_blocks(
        [
            *_extract_failure_blocks(stdout, source="stdout"),
            *_extract_failure_blocks(stderr, source="stderr"),
        ]
    )

    for block in blocks:
        title_clean = block.title.rstrip(":").strip() if block.title else ""
        if title_clean and title_clean not in failed_tests:
            failed_tests.append(title_clean)
        if block.excerpt and block.excerpt not in exact_diagnostics:
            exact_diagnostics.append(block.excerpt)

    full_text = f"{stdout}\n{stderr}"
    lines = full_text.splitlines()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "is not a function" in stripped
            or _QA_ERROR_MARKER.search(stripped)
            or re.search(r"\b[A-Za-z_][\w.]*(?:Error|Exception)\s*:", stripped)
        ):
            snippet = stripped[:500] + "... (truncated)" if len(stripped) > 500 else stripped
            if snippet not in exact_diagnostics:
                exact_diagnostics.append(snippet)

        matches = re.finditer(
            r"(?:at\s+.*?\()?([a-zA-Z0-9_\-\./\\]+\.(?:js|ts|jsx|tsx|mjs|cjs)):(\d+)(?::(\d+))?\)?",
            line,
        )
        for match in matches:
            filepath = match.group(1).replace("\\", "/")
            if "node_modules" in filepath:
                continue
            line_no = match.group(2)
            col_no = match.group(3)
            loc = f"{filepath}:{line_no}:{col_no}" if col_no else f"{filepath}:{line_no}"
            if loc not in source_locations:
                source_locations.append(loc)
            if filepath not in affected_files:
                affected_files.append(filepath)

    if sandbox is not None:
        source_locations, affected_files, path_diagnostics = _validate_qa_source_locations(
            source_locations,
            sandbox,
        )
        exact_diagnostics.extend(
            diagnostic for diagnostic in path_diagnostics if diagnostic not in exact_diagnostics
        )

    raw_excerpt = _fallback_raw_tail(exit_code, stdout, stderr)

    return QAFailureEvidence(
        exact_diagnostics=exact_diagnostics[:15],
        failed_tests=failed_tests[:10],
        source_locations=source_locations[:10],
        affected_files=affected_files[:10],
        raw_excerpt=raw_excerpt[:2000],
        attempt_id=attempt_id,
        task_revision=task_revision,
    )


def detect_test_runner(sandbox: DockerSandbox, test_file: str | None = None) -> str:
    """Detect the source test runner for a targeted test file."""
    return _detect_targeted_test_context(sandbox, test_file)[0]


def _normalise_test_path(path: str) -> str:
    """Normalize one repository-relative test path for runner selection."""
    normalized = (path or "").replace("\\", "/").strip().lstrip("/")
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    if normalized.startswith("build/"):
        normalized = normalized[len("build/") :]
    return normalized


def _find_test_package_context(
    sandbox: DockerSandbox,
    test_file: str,
) -> tuple[str, dict[str, Any]]:
    """Find the nearest package.json that owns a targeted test file."""
    normalized = _normalise_test_path(test_file)
    parts = normalized.split("/")
    for index in range(max(len(parts) - 1, 0), -1, -1):
        cwd = "/".join(parts[:index])
        package_json = _read_package_json_for_cwd(sandbox, cwd)
        if package_json is not None:
            return cwd, package_json
    return "", _workspace_json_file(sandbox, "package.json") or {}


def _target_is_under_cwd(test_file: str, cwd: str) -> bool:
    """Return whether a normalized test path belongs to a package directory."""
    normalized = _normalise_test_path(test_file)
    normalized_cwd = cwd.strip().strip("/\\")
    return (
        not normalized_cwd
        or normalized == normalized_cwd
        or normalized.startswith(normalized_cwd + "/")
    )


def _relative_test_path(test_file: str, cwd: str) -> str:
    """Return a test path relative to its owning package directory."""
    normalized = _normalise_test_path(test_file)
    normalized_cwd = cwd.strip().strip("/\\")
    if normalized_cwd and normalized.startswith(normalized_cwd + "/"):
        return normalized[len(normalized_cwd) + 1 :]
    return normalized


def _command_targets_test_file(command: str, test_file: str, cwd: str) -> bool:
    """Check whether a leaf test command covers a targeted path."""
    if not _target_is_under_cwd(test_file, cwd):
        return False

    relative_path = _relative_test_path(test_file, cwd)
    command_text = command.replace('"', "").replace("'", "")
    for token in re.findall(r"[^\s]+", command_text):
        token = token.strip(";,()")
        if not token or token.startswith("-"):
            continue
        if "*" in token or "?" in token:
            prefix = re.split(r"[*?]", token, maxsplit=1)[0].rstrip("/")
            if prefix and relative_path.startswith(prefix):
                return True
        elif token == relative_path or token == test_file:
            return True

    # Commands such as ``ng test`` and ``vitest`` discover files through the
    # package configuration instead of spelling out a glob.
    words = command_text.split()
    return words[:2] == ["ng", "test"] or bool(words and words[0] in {"vitest", "jest"})


def _resolve_targeted_test_command(
    sandbox: DockerSandbox,
    command: str,
    test_file: str,
    *,
    cwd: str,
    package_json: dict[str, Any],
    seen: set[tuple[str, str]] | None = None,
) -> tuple[str, str, str] | None:
    """Resolve a matching npm test command to ``(runner, cwd, leaf command)``.

    Returning the leaf command is important for targeted execution. Appending a
    file to ``npm run test:api -- file`` does not remove a glob already present
    in ``test:api``; it therefore still executes the entire suite. The caller
    replaces the leaf command's test glob with the requested source file.
    """
    seen = seen or set()
    command = (command or "").strip()
    key = (cwd, command)
    if not command or key in seen:
        return None
    seen.add(key)

    cd_match = re.match(r"^\s*cd\s+(?P<child>[^\s&]+)\s+&&\s+(?P<rest>.+)$", command)
    if cd_match:
        child = cd_match.group("child").strip("\"'")
        next_cwd = f"{cwd.rstrip('/')}/{child}" if cwd else child
        child_package = _read_package_json_for_cwd(sandbox, next_cwd) or {}
        if not _target_is_under_cwd(test_file, next_cwd):
            return None
        return _resolve_targeted_test_command(
            sandbox,
            cd_match.group("rest"),
            test_file,
            cwd=next_cwd,
            package_json=child_package,
            seen=seen,
        )

    script_name = _script_name_from_npm_run(command)
    scripts = package_json.get("scripts") if isinstance(package_json, dict) else None
    if script_name and isinstance(scripts, dict) and isinstance(scripts.get(script_name), str):
        resolved = _resolve_targeted_test_command(
            sandbox,
            scripts[script_name],
            test_file,
            cwd=cwd,
            package_json=package_json,
            seen=seen,
        )
        if resolved is not None:
            runner, resolved_cwd, leaf_command = resolved
            return runner, resolved_cwd, leaf_command
        return None

    if not _command_targets_test_file(command, test_file, cwd):
        return None
    runner = _classify_test_command(
        sandbox,
        command,
        cwd=cwd,
        package_json=package_json,
        seen=set(),
    )
    if runner == "npm_text_fallback":
        return None
    return runner, cwd, command


def _detect_targeted_test_context(
    sandbox: DockerSandbox,
    test_file: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(runner, package_cwd, npm_invocation)`` for a targeted test."""
    normalized = _normalise_test_path(test_file or "")
    cwd, package_json = (
        _find_test_package_context(sandbox, normalized)
        if normalized
        else (
            "",
            _workspace_json_file(sandbox, "package.json") or {},
        )
    )
    scripts = package_json.get("scripts") if isinstance(package_json, dict) else None
    test_script = scripts.get("test") if isinstance(scripts, dict) else None

    if normalized and isinstance(test_script, str) and test_script.strip():
        for command in _split_script_chain(test_script):
            resolved = _resolve_targeted_test_command(
                sandbox,
                command,
                normalized,
                cwd=cwd,
                package_json=package_json,
            )
            if resolved is not None:
                return resolved

    if isinstance(test_script, str) and test_script.strip():
        for command in _split_script_chain(test_script):
            runner = _classify_test_command(sandbox, command, cwd=cwd, package_json=package_json)
            if runner != "npm_text_fallback":
                return runner, cwd, "npm test"

    deps: dict[str, Any] = {}
    if isinstance(package_json, dict):
        for key in ("dependencies", "devDependencies", "peerDependencies"):
            if isinstance(package_json.get(key), dict):
                deps.update(package_json[key])

    for dependency, runner in (("mocha", "mocha"), ("jest", "jest"), ("vitest", "vitest")):
        if dependency in deps:
            return runner, cwd, "npm test"
    return "npm_text_fallback", cwd, "npm test"


def _mocha_test_name_variants(test_name: str) -> list[str]:
    """Return likely canonical forms for an LLM-provided Mocha test hint.

    Mocha composes nested suite names with spaces, while agents often render
    the same hierarchy as ``Suite - test``.  Include both forms and the leaf
    test title so the hint remains useful without treating the agent's
    formatting as canonical.

    Args:
        test_name: Human-readable test name or suite/test description.

    Returns:
        Distinct, normalized candidate names in preference order.
    """
    cleaned = re.sub(r"^\s*(?:\d+[.)]\s*)", "", str(test_name or "")).strip()
    cleaned = re.sub(r"\s+\[[^\]]+\]\s*$", "", cleaned).strip()
    variants = [cleaned]
    if " - " in cleaned:
        suite, leaf = cleaned.rsplit(" - ", 1)
        variants.extend((f"{suite} {leaf}", leaf))
    return list(dict.fromkeys(value for value in variants if value))


def _mocha_test_filter(test_name: str) -> str:
    """Build a safe Mocha grep pattern from an LLM-provided test hint."""
    variants = _mocha_test_name_variants(test_name)
    escaped = [re.escape(value).replace(r"\ ", " ") for value in variants]
    return escaped[0] if len(escaped) == 1 else f"(?:{'|'.join(escaped)})"


def build_targeted_test_command(
    runner: str,
    test_file: str,
    test_name: str | None = None,
    npm_invocation: str = "npm test",
    package_cwd: str = "",
) -> str | None:
    """Construct a source-runner command that executes only ``test_file``.

    ``npm_invocation`` is the resolved leaf command from ``package.json``. A
    direct command is used so an existing wildcard cannot continue to select
    the whole suite. Direct npm test-runner commands are prefixed with
    ``npx --no-install`` so they resolve the same workspace-local executable
    that npm scripts expose through ``node_modules/.bin``. The function
    retains the old ``npm test`` append behavior for callers that pass a
    generic npm invocation. ``package_cwd`` is the package working directory
    resolved from the same QA test context and is applied to the command so
    targeted validation uses the same package-local dependency tree.
    """
    safe_file = shlex.quote(test_file)
    safe_name = shlex.quote(test_name) if test_name else None
    safe_mocha_name = shlex.quote(_mocha_test_filter(test_name)) if test_name else None
    invocation = npm_invocation.strip() or "npm test"

    def with_package_cwd(command: str) -> str:
        if package_cwd:
            return f"cd {shlex.quote(package_cwd)} && {command}"
        return command

    if invocation in {"npm test", "npm run test"} or invocation.startswith("npm run "):
        if runner == "mocha":
            args = safe_file
            if safe_mocha_name:
                args += f" --grep {safe_mocha_name}"
            args += " --reporter json"
            return with_package_cwd(f"{invocation} -- {args}")
        if runner == "jest":
            args = safe_file + (f" -t {safe_name}" if safe_name else "")
            return with_package_cwd(f"{invocation} -- {args}")
        if runner == "vitest":
            args = f"--run {safe_file}" + (f" -t {safe_name}" if safe_name else "")
            return with_package_cwd(f"{invocation} -- {args}")
        if runner == "angular_vitest":
            return with_package_cwd(f"{invocation} -- --include {safe_file}")
        if runner == "node_test":
            args = safe_file + (f" --test-name-pattern {safe_name}" if safe_name else "")
            return with_package_cwd(f"{invocation} -- {args}")
        return None

    try:
        tokens = shlex.split(invocation)
    except ValueError:
        tokens = invocation.split()
    if not tokens:
        return None

    if runner == "angular_vitest":
        tokens.extend(["--include", test_file])
    else:
        target_replaced = False
        skip_next = False
        rebuilt: list[str] = []
        for token in tokens:
            if skip_next:
                rebuilt.append(token)
                skip_next = False
                continue
            if token in {"--import", "-r", "--require"}:
                rebuilt.append(token)
                skip_next = True
                continue
            normalized_token = token.replace("\\", "/")
            is_test_path = ("*" in normalized_token or "?" in normalized_token) and (
                "test" in normalized_token or "spec" in normalized_token
            )
            if is_test_path:
                rebuilt.append(test_file)
                target_replaced = True
            else:
                rebuilt.append(token)
        tokens = rebuilt
        if not target_replaced:
            tokens.append(test_file)

        if runner == "vitest" and "--run" not in tokens:
            tokens.insert(1 if tokens else 0, "--run")
        if test_name:
            if runner == "mocha":
                tokens.extend(["--grep", _mocha_test_filter(test_name)])
            elif runner == "jest" or runner == "vitest":
                tokens.extend(["-t", test_name])
            elif runner == "node_test":
                tokens.extend(["--test-name-pattern", test_name])
        if runner == "mocha":
            tokens.extend(["--reporter", "json"])

    # The full QA runner executes npm scripts, which temporarily prepends the
    # package's node_modules/.bin directory to PATH. Targeted validation runs
    # in a fresh shell, so preserve that same workspace dependency context
    # explicitly when the resolved leaf command is a local npm test runner.
    if runner in _LOCAL_NPM_TEST_RUNNERS:
        first_token = tokens[0].replace("\\", "/")
        if first_token == runner:
            tokens = ["npx", "--no-install", *tokens]

    return with_package_cwd(shlex.join(tokens))


def _workspace_json_file(sandbox: DockerSandbox, path: str) -> dict[str, Any] | None:
    """Read one JSON file from the sandbox workspace, returning ``None`` on misses."""
    try:
        content = sandbox.read_file(path)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _split_script_chain(script: str) -> list[str]:
    """Split simple npm ``&&`` chains while preserving each child command."""
    return [part.strip() for part in (script or "").split("&&") if part.strip()]


def _script_name_from_npm_run(command: str) -> str | None:
    """Return the script name from simple ``npm run <name>`` commands."""
    match = re.match(
        r"^\s*npm\s+(?:run|run-script)\s+(?:--silent\s+)?(?P<name>[^\s]+)",
        command,
    )
    return match.group("name") if match else None


def _read_package_json_for_cwd(sandbox: DockerSandbox, cwd: str) -> dict[str, Any] | None:
    """Read package.json for a detected child-suite working directory."""
    normalized_cwd = cwd.strip().strip("/\\")
    package_path = "package.json" if not normalized_cwd else f"{normalized_cwd}/package.json"
    return _workspace_json_file(sandbox, package_path)


def _angular_project_uses_vitest(sandbox: DockerSandbox, cwd: str) -> bool:
    """Best-effort check for Angular's Vitest-backed unit-test builder."""
    package_json = _read_package_json_for_cwd(sandbox, cwd) or {}
    dependency_sections = ("dependencies", "devDependencies", "peerDependencies")
    if any(
        "vitest" in (package_json.get(section) or {})
        for section in dependency_sections
        if isinstance(package_json.get(section), dict)
    ):
        return True

    normalized_cwd = cwd.strip().strip("/\\")
    angular_path = "angular.json" if not normalized_cwd else f"{normalized_cwd}/angular.json"
    angular_json = _workspace_json_file(sandbox, angular_path) or {}
    return "@angular/build:unit-test" in json.dumps(angular_json)


def _classify_test_command(
    sandbox: DockerSandbox,
    command: str,
    *,
    cwd: str = "",
    package_json: dict[str, Any] | None = None,
    seen: set[tuple[str, str]] | None = None,
) -> str:
    """Classify a test command into a supported structured strategy."""
    seen = seen or set()
    command = (command or "").strip()
    lowered = command.lower()
    current_key = (cwd, command)
    if not command or current_key in seen:
        return "npm_text_fallback"
    seen.add(current_key)

    cd_match = re.match(r"^\s*cd\s+(?P<cwd>[^\s&]+)\s+&&\s+(?P<rest>.+)$", command)
    if cd_match:
        child_cwd = cd_match.group("cwd").strip().strip("\"'")
        base_cwd = cwd.rstrip("/\\")
        next_cwd = f"{base_cwd}/{child_cwd}" if base_cwd else child_cwd
        child_package = _read_package_json_for_cwd(sandbox, next_cwd)
        return _classify_test_command(
            sandbox,
            cd_match.group("rest"),
            cwd=next_cwd,
            package_json=child_package,
            seen=seen,
        )

    if "node " in f" {lowered}" and " --test" in f" {lowered}":
        return "node_test"
    if re.search(r"(^|\s)mocha(\s|$)", lowered):
        return "mocha"
    if re.search(r"(^|\s)vitest(\s|$)", lowered):
        return "vitest"
    if re.search(r"(^|\s)ng\s+test(\s|$)", lowered):
        return (
            "angular_vitest" if _angular_project_uses_vitest(sandbox, cwd) else "npm_text_fallback"
        )

    script_name = _script_name_from_npm_run(command)
    scripts = (package_json or {}).get("scripts") if isinstance(package_json, dict) else None
    if script_name and isinstance(scripts, dict) and isinstance(scripts.get(script_name), str):
        return _classify_test_command(
            sandbox,
            scripts[script_name],
            cwd=cwd,
            package_json=package_json,
            seen=seen,
        )

    return "npm_text_fallback"


def _suite_name_from_command(command: str) -> str:
    """Derive a concise display name for one child suite."""
    script_name = _script_name_from_npm_run(command)
    if script_name:
        return script_name.replace("test:", "") or script_name
    cd_match = re.match(r"^\s*cd\s+(?P<cwd>[^\s&]+)\s+&&", command)
    if cd_match:
        return cd_match.group("cwd").strip().strip("\"'")
    return command.split()[0] if command.split() else "npm test"


def _detect_test_suite_plans(sandbox: DockerSandbox) -> list[_TestSuitePlan] | None:
    """Detect npm test child suites from package.json, expanding simple ``&&`` chains."""
    package_json = _workspace_json_file(sandbox, "package.json")
    scripts = package_json.get("scripts") if isinstance(package_json, dict) else None
    test_script = scripts.get("test") if isinstance(scripts, dict) else None
    if not isinstance(test_script, str) or not test_script.strip():
        return None

    commands = _split_script_chain(test_script)
    if not commands:
        return None

    plans: list[_TestSuitePlan] = []
    for command in commands:
        runner = _classify_test_command(sandbox, command, package_json=package_json)
        plans.append(
            _TestSuitePlan(
                name=_suite_name_from_command(command),
                runner=runner,
                command=command,
            )
        )
    return plans


def _structured_command_for_plan(plan: _TestSuitePlan) -> str:
    """Return a JSON-capable command when it is safe; otherwise preserve original."""
    command = plan.command
    lowered = command.lower()
    if plan.runner == "mocha" and "--reporter" not in lowered:
        separator = " -- " if _script_name_from_npm_run(command) else " "
        return f"{command}{separator}--reporter json"
    if plan.runner == "vitest" and "--reporter" not in lowered:
        separator = " -- " if _script_name_from_npm_run(command) else " "
        return f"{command}{separator}--reporter=json"
    return command


def _truncate_summary_text(value: str, limit: int = _TEST_FAILURE_EXCERPT_CHARS) -> str:
    """Bound one message or excerpt for log-query output."""
    value = (value or "").strip()
    if len(value) > limit:
        return value[:limit].rstrip() + "... (truncated)"
    return value


def _diagnostic_from_lines(
    lines: list[str], suite: str, kind: str = "diagnostic"
) -> _NormalizedTestDiagnostic:
    """Build one bounded diagnostic message from raw runner lines."""
    return _NormalizedTestDiagnostic(
        message=_truncate_summary_text("\n".join(line.rstrip() for line in lines if line.strip())),
        kind=kind,
        suite=suite,
    )


def _normalize_node_tap_output(
    stdout: str,
    stderr: str,
    *,
    suite_name: str = "node_test",
) -> tuple[list[_NormalizedTestFailure], list[_NormalizedTestDiagnostic]]:
    """Normalize Node test TAP output without counting parent suites as leaf failures."""
    lines = _normalize_log_lines(f"{stdout or ''}\n{stderr or ''}")
    failures: list[_NormalizedTestFailure] = []
    diagnostics: list[_NormalizedTestDiagnostic] = []
    pending_subtest: str | None = None
    seen_diagnostics: set[str] = set()
    index = 0

    while index < len(lines):
        line = lines[index]
        subtest_match = _SUBTEST_RE.match(line)
        if subtest_match:
            pending_subtest = subtest_match.group(1).strip()
            index += 1
            continue

        tap_match = _TAP_FAILURE_RE.match(line)
        if not tap_match:
            stripped = line.strip()
            if (
                stripped.startswith("# Error:")
                or "generated asynchronous activity after the test ended" in stripped
            ):
                diagnostic_lines = [stripped]
                next_index = index + 1
                while next_index < len(lines) and lines[next_index].strip().startswith("#"):
                    diagnostic_lines.append(lines[next_index].strip())
                    next_index += 1
                diagnostic = _diagnostic_from_lines(
                    diagnostic_lines, suite_name, kind="node_runner"
                )
                if diagnostic.message and diagnostic.message not in seen_diagnostics:
                    diagnostics.append(diagnostic)
                    seen_diagnostics.add(diagnostic.message)
                index = next_index
                continue
            index += 1
            continue

        title = (tap_match.group(1) or "").strip() or pending_subtest or line.strip()
        pending_subtest = None
        block_lines = [line.rstrip()]
        next_index = index + 1
        while next_index < len(lines):
            next_line = lines[next_index]
            if (
                _SUBTEST_RE.match(next_line)
                or re.match(r"^\s*(?:ok|not ok)\b", next_line)
                or next_line.strip().startswith("# Error:")
            ):
                break
            block_lines.append(next_line.rstrip())
            next_index += 1

        block_text = "\n".join(block_lines)
        failure_type_match = re.search(r"failureType:\s*'([^']+)'", block_text)
        failure_type = failure_type_match.group(1) if failure_type_match else ""
        type_match = re.search(r"type:\s*'([^']+)'", block_text)
        test_type = type_match.group(1) if type_match else ""
        message_match = re.search(r"error:\s*'([^']+)'", block_text)
        message = message_match.group(1) if message_match else ""

        if failure_type == "subtestsFailed" or test_type == "suite":
            diagnostic = _diagnostic_from_lines(block_lines, suite_name, kind="node_parent_suite")
            if diagnostic.message and diagnostic.message not in seen_diagnostics:
                diagnostics.append(diagnostic)
                seen_diagnostics.add(diagnostic.message)
        else:
            failures.append(
                _NormalizedTestFailure(
                    name=title,
                    message=_truncate_summary_text(message or block_text),
                    failure_type=failure_type,
                    suite=suite_name,
                )
            )
        index = next_index

    return failures, diagnostics


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from reporter output that may contain extra log lines."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except Exception:  # noqa: BLE001
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(stripped[start : end + 1])
    except Exception:  # noqa: BLE001
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_mocha_json_output(
    stdout: str,
    stderr: str,
    *,
    suite_name: str = "mocha",
) -> tuple[list[_NormalizedTestFailure], list[_NormalizedTestDiagnostic]]:
    """Normalize Mocha JSON reporter output."""
    report = _extract_json_object(stdout) or _extract_json_object(stderr)
    if not report:
        return [], []
    failures: list[_NormalizedTestFailure] = []
    for item in report.get("failures") or []:
        if not isinstance(item, dict):
            continue
        error = item.get("err") if isinstance(item.get("err"), dict) else {}
        full_title = item.get("fullTitle") or item.get("title") or "mocha failure"
        failures.append(
            _NormalizedTestFailure(
                name=str(full_title),
                message=_truncate_summary_text(str(error.get("message") or "")),
                failure_type=str(error.get("name") or ""),
                suite=suite_name,
            )
        )
    return failures, []


def _iter_vitest_failures(node: Any) -> list[dict[str, Any]]:
    """Collect failed assertion nodes from Vitest JSON reporter-like structures."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        status = str(node.get("status") or "").lower()
        if status in {"failed", "fail"} and (node.get("name") or node.get("fullName")):
            found.append(node)
        for value in node.values():
            found.extend(_iter_vitest_failures(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_iter_vitest_failures(value))
    return found


def _normalize_vitest_json_output(
    stdout: str,
    stderr: str,
    *,
    suite_name: str = "vitest",
) -> tuple[list[_NormalizedTestFailure], list[_NormalizedTestDiagnostic]]:
    """Normalize Vitest JSON reporter output when available."""
    report = _extract_json_object(stdout) or _extract_json_object(stderr)
    if not report:
        return [], []
    failures: list[_NormalizedTestFailure] = []
    for item in _iter_vitest_failures(report):
        errors = item.get("errors") if isinstance(item.get("errors"), list) else []
        first_error = errors[0] if errors and isinstance(errors[0], dict) else {}
        failures.append(
            _NormalizedTestFailure(
                name=str(item.get("fullName") or item.get("name") or "vitest failure"),
                message=_truncate_summary_text(str(first_error.get("message") or "")),
                failure_type=str(first_error.get("name") or ""),
                suite=suite_name,
            )
        )
    return failures, []


def _normalize_suite_result(
    plan: _TestSuitePlan,
    result: Any,
    *,
    command: str,
) -> _NormalizedSuiteResult:
    """Normalize one child-suite result into failed tests plus diagnostics."""
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    exit_code = int(getattr(result, "exit_code", 1))
    failures: list[_NormalizedTestFailure] = []
    diagnostics: list[_NormalizedTestDiagnostic] = []

    if plan.runner == "node_test":
        failures, diagnostics = _normalize_node_tap_output(stdout, stderr, suite_name=plan.name)
    elif plan.runner == "mocha":
        failures, diagnostics = _normalize_mocha_json_output(stdout, stderr, suite_name=plan.name)
    elif plan.runner in {"vitest", "angular_vitest"}:
        failures, diagnostics = _normalize_vitest_json_output(stdout, stderr, suite_name=plan.name)

    fallback_summary = ""
    if exit_code != 0 and not failures and not diagnostics:
        fallback_summary = _summarize_failed_test_output(exit_code, stdout, stderr)

    return _NormalizedSuiteResult(
        name=plan.name,
        runner=plan.runner,
        command=command,
        exit_code=exit_code,
        failed_tests=failures,
        diagnostics=diagnostics,
        fallback_summary=fallback_summary,
    )


def _format_normalized_test_summary(suites: list[_NormalizedSuiteResult]) -> str:
    """Format normalized suite results into the bounded query_qa_logs text contract."""
    failed_suites = [suite for suite in suites if suite.exit_code != 0]
    exit_code = failed_suites[-1].exit_code if failed_suites else 0
    failed_tests = [failure for suite in suites for failure in suite.failed_tests]
    diagnostics = [diagnostic for suite in suites for diagnostic in suite.diagnostics]
    lines = [
        f"npm test FAILED (exit {exit_code}).",
        f"Failed Tests: {len(failed_tests)}",
        f"Runner Diagnostics: {len(diagnostics)}",
        "",
        "Suites:",
    ]
    for suite in suites:
        status = "passed" if suite.exit_code == 0 else "failed"
        lines.append(f"- {suite.name}: {status} ({suite.runner}, exit {suite.exit_code})")
        lines.append(f"  command: {suite.command}")

    if failed_tests:
        lines.extend(["", "Failing Tests:"])
        for index, failure in enumerate(failed_tests[:_TEST_FAILURE_MAX_ITEMS], start=1):
            suffix = f" [{failure.suite}]" if failure.suite else ""
            lines.append(f"{index}. {failure.name}{suffix}")
            details = "\n".join(part for part in [failure.failure_type, failure.message] if part)
            if details:
                lines.append(details)
            lines.append("")
        if len(failed_tests) > _TEST_FAILURE_MAX_ITEMS:
            lines.append(
                f"... and {len(failed_tests) - _TEST_FAILURE_MAX_ITEMS} more failed tests omitted"
            )

    if diagnostics:
        lines.extend(["", "Runner Diagnostics:"])
        for index, diagnostic in enumerate(diagnostics[:_TEST_FAILURE_MAX_ITEMS], start=1):
            suffix = f" [{diagnostic.suite}]" if diagnostic.suite else ""
            lines.append(f"{index}. {diagnostic.kind}{suffix}")
            lines.append(diagnostic.message)
            lines.append("")
        if len(diagnostics) > _TEST_FAILURE_MAX_ITEMS:
            lines.append(
                f"... and {len(diagnostics) - _TEST_FAILURE_MAX_ITEMS} more diagnostics omitted"
            )

    for suite in failed_suites:
        if suite.fallback_summary:
            lines.extend(["", suite.fallback_summary])

    summary = "\n".join(lines).strip()
    if len(summary) > _LOG_QUERY_MAX_CHARS:
        summary = summary[:_LOG_QUERY_MAX_CHARS].rstrip() + "\n... (summary truncated)"
    return summary


def _run_detected_test_suites(
    sandbox: DockerSandbox,
    plans: list[_TestSuitePlan],
) -> tuple[bool, str]:
    """Run detected child suites sequentially, preserving npm ``&&`` short-circuiting."""
    suite_results: list[_NormalizedSuiteResult] = []
    for plan in plans:
        command = _structured_command_for_plan(plan)
        result = sandbox.run(command, timeout=_NPM_TEST_TIMEOUT_SECONDS)
        suite_result = _normalize_suite_result(plan, result, command=command)
        suite_results.append(suite_result)
        if suite_result.exit_code != 0:
            return False, _format_normalized_test_summary(suite_results)
    return True, "npm test passed."


def _run_unit_tests(sandbox: DockerSandbox) -> tuple[bool, str]:
    """
    Run workspace unit tests, preferring structured child-suite summaries.

    Returns:
        (success, summary_text)
    """
    plans = _detect_test_suite_plans(sandbox)
    if plans and any(plan.runner != "npm_text_fallback" for plan in plans):
        return _run_detected_test_suites(sandbox, plans)

    result = sandbox.run("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)
    if result.exit_code == 0:
        return True, "npm test passed."

    summary = _summarize_failed_test_output(
        result.exit_code,
        result.stdout,
        result.stderr,
    )
    return False, summary


def _generate_workspace_diff(
    host_repo_root: str,
    sandbox: DockerSandbox,
    candidate_changed_files: list[str],
) -> tuple[str, list[str]]:
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
    changed_files: list[str] = []
    diff_parts: list[str] = []

    # Deduplicate files while preserving order.
    seen: set[str] = set()
    unique_candidates: list[str] = []
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

            diff_lines = list(
                difflib.unified_diff(
                    host_content.splitlines(keepends=True),
                    workspace_content.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
            )
            diff_parts.append("".join(diff_lines))

    full_diff = "\n".join(diff_parts)
    if not full_diff:
        return (
            "(diff is empty â€” workspace matches host baseline for all candidate files)",
            changed_files,
        )

    if len(full_diff) > _DIFF_CHAR_BUDGET:
        full_diff = full_diff[:_DIFF_CHAR_BUDGET] + "\n... (diff truncated)"

    return full_diff, changed_files


# ---------------------------------------------------------------------------
# Target identifier collection
# ---------------------------------------------------------------------------


def _collect_target_identifiers(groups: list[VulnerabilityGroup]) -> set[str]:
    """Collect all CVE/GHSA identifiers from the valid vulnerability groups."""
    identifiers: set[str] = set()
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


def _collect_baseline_identifiers(
    state: OrchestratorState,
    groups: list[VulnerabilityGroup],
) -> set[str]:
    """Resolve the immutable pre-remediation identifier baseline.

    ``initial_orchestrator_state`` normally materializes this field before
    graph execution.  Direct QA callers and older tests may omit it, so use
    the complete initial issue set when present and the current groups only as
    the documented skip-triage fallback.
    """
    if "baseline_scan_identifiers" in state:
        return {
            identifier.upper().strip()
            for identifier in state.get("baseline_scan_identifiers", []) or []
            if identifier and identifier.strip()
        }

    issues = state.get("issues")
    if issues is not None:
        identifiers: set[str] = set()
        for issue in issues:
            if issue.cve_id:
                identifiers.add(issue.cve_id.upper().strip())
            if issue.ghsa_id:
                identifiers.add(issue.ghsa_id.upper().strip())
        return identifiers

    return _collect_target_identifiers(groups)


def _lockfile_paths_for_group(group: VulnerabilityGroup) -> tuple[str, ...]:
    """Return normalized npm lockfile paths associated with an SCA group."""
    candidates: list[str] = []
    candidates.extend(group.file_paths or [])
    if group.file_path:
        candidates.append(group.file_path)
    for issue in group.localized_issues or []:
        if issue.manifest_file:
            candidates.append(issue.manifest_file)

    lockfiles: list[str] = []
    for raw_path in candidates:
        path = str(raw_path or "").strip().split("?", 1)[0].replace("\\", "/")
        if not path or path.startswith("/") or ".." in Path(path).parts:
            continue
        path = path.lstrip("./")
        name = Path(path).name.lower()
        if name in {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}:
            lockfile = path
        elif name == "package.json":
            lockfile = str(Path(path).with_name("package-lock.json"))
        else:
            continue
        if lockfile not in lockfiles:
            lockfiles.append(lockfile)
    return tuple(sorted(lockfiles))


def _build_qa_scan_targets(
    state: OrchestratorState,
    groups: list[VulnerabilityGroup],
) -> list[QAScanTarget] | None:
    """Build task-owned scan targets, or ``None`` for legacy direct QA calls."""
    active_ids = list(state.get("active_target_task_ids") or [])
    if not active_ids:
        # A group-id fallback is retained for older orchestrator callers that
        # predate task-keyed dispatch, but a completely unscoped direct call
        # must keep its historical full-scan behavior.
        active_ids = list(state.get("active_target_group_ids") or [])
    if not active_ids:
        return None

    task_queue: dict[str, RemediationTask] = dict(state.get("task_queue") or {})
    groups_by_id = {group.group_id: group for group in groups}
    targets: list[QAScanTarget] = []
    for active_id in active_ids:
        task = task_queue.get(active_id)
        group_id = task.parent_group_id if task is not None else active_id
        group = groups_by_id.get(group_id)
        if group is None:
            continue
        target_package = (
            task.target_package_name
            if task is not None and task.target_package_name
            else group.vulnerable_component or ""
        ).strip()
        expected_version = task.selected_version if task is not None else None
        if not expected_version and target_package:
            expected_version = (group.dependency_versions or {}).get(target_package)
        if not expected_version and target_package == group.vulnerable_component:
            expected_version = (group.versions or [None])[0]
        ancestry = tuple(name for name in (group.dependency_ancestry or []) if name)
        targets.append(
            QAScanTarget(
                task_id=task.task_id if task is not None else active_id,
                group_id=group.group_id,
                target_package=target_package,
                expected_version=expected_version,
                manifest_paths=_lockfile_paths_for_group(group),
                dependency_ancestry=ancestry,
                target_identifiers=frozenset(_group_target_identifiers(group)),
            )
        )
    return targets


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

    install: tuple[bool, str] | None = None  # (ok, summary)
    scan: Any | None = None  # _SecurityScanResult or legacy tuple
    tests: tuple[bool, str] | None = None  # (ok, summary)
    scan_evidence: ODCScanEvidence | None = None


@dataclass(frozen=True)
class QAScanTarget:
    """Task-owned package target and live lockfile context for QA scanning."""

    task_id: str
    group_id: str
    target_package: str
    expected_version: str | None
    manifest_paths: tuple[str, ...]
    dependency_ancestry: tuple[str, ...]
    target_identifiers: frozenset[str]


def _scan_result_value(scan_result: Any, field: str, default: Any) -> Any:
    """Read a scan field from the new result or a legacy three-tuple."""
    if scan_result is None:
        return default
    if isinstance(scan_result, _SecurityScanResult):
        return getattr(scan_result, field)
    legacy_index = {
        "ok": 0,
        "summary": 1,
        "remaining_identifiers": 2,
    }.get(field)
    if legacy_index is not None:
        try:
            return scan_result[legacy_index]
        except (IndexError, KeyError, TypeError):
            pass
    return default


def _scan_state_projection(
    results: _QAExecutionResults,
    baseline_identifiers: set[str],
    *,
    authoritative: bool = True,
) -> dict[str, Any]:
    """Project the scan cache into serializable graph-state fields."""
    if not authoritative:
        return {
            "baseline_scan_identifiers": sorted(baseline_identifiers),
            "post_remediation_scan_identifiers": [],
            "post_remediation_scan_issues": [],
            "new_vulnerability_identifiers": [],
            "new_vulnerability_status": "not_scanned",
        }

    scan_result = results.scan
    if scan_result is None:
        return {
            "baseline_scan_identifiers": sorted(baseline_identifiers),
            "post_remediation_scan_identifiers": [],
            "post_remediation_scan_issues": [],
            "new_vulnerability_identifiers": [],
            "new_vulnerability_status": "not_scanned",
        }

    found = set(_scan_result_value(scan_result, "found_identifiers", set()) or set())
    new_identifiers = set(_scan_result_value(scan_result, "new_identifiers", set()) or set())
    found_issues = list(_scan_result_value(scan_result, "found_issues", []) or [])
    scan_ok = bool(_scan_result_value(scan_result, "ok", False))
    remaining = set(_scan_result_value(scan_result, "remaining_identifiers", set()) or set())
    if isinstance(scan_result, _SecurityScanResult) and not scan_ok and not found and not remaining:
        status = "scan_failed"
    else:
        status = "detected" if new_identifiers else "none"

    return {
        "baseline_scan_identifiers": sorted(baseline_identifiers),
        "post_remediation_scan_identifiers": sorted(found),
        "post_remediation_scan_issues": found_issues,
        "new_vulnerability_identifiers": sorted(new_identifiers),
        "new_vulnerability_status": status,
    }


def _augment_qa_report_with_scan_findings(
    report: str,
    scan_projection: dict[str, Any],
) -> str:
    """Append deterministic new-finding evidence when the scan found any."""
    new_identifiers = scan_projection.get("new_vulnerability_identifiers", []) or []
    if not new_identifiers:
        return report
    post_identifiers = scan_projection.get("post_remediation_scan_identifiers", []) or []
    section = (
        "## Global Newly Introduced Scanner Findings\n"
        f"- Status: {scan_projection.get('new_vulnerability_status', 'detected')}\n"
        f"- Newly introduced identifiers: {', '.join(new_identifiers)}\n"
        f"- Post-remediation identifier snapshot: {', '.join(post_identifiers) or 'none'}\n"
        "- Attribution: graph-level finding; deferred to the later triage phase."
    )
    return f"{report.rstrip()}\n\n{section}".strip()


# ---------------------------------------------------------------------------
# Map phase dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupInvestigation:
    """Investigation output for a single vulnerability group (Map phase)."""

    group_id: str
    investigation_text: str
    tool_transcript: str
    errors: list[str] = field(default_factory=list)


@dataclass
class BatchInvestigationArtifact:
    """All Map phase outputs consumed by the Reduce phase."""

    results: _QAExecutionResults
    investigations_by_group: dict[str, GroupInvestigation]
    holistic_report: str = ""  # filled after reduce phase
    errors: list[str] = field(default_factory=list)


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
    group_sections: dict[str, GroupInvestigationSection]
    errors: list[str]
    warnings: list[str]


@dataclass
class GroupEvidencePacket:
    """Group-scoped evidence passed to the Judge phase (backcompat)."""

    group: VulnerabilityGroup
    strategy: str
    fix_plan_status: str
    fix_plan_instruction: str
    action_summaries: list[str]
    shared_install_analysis: str
    install_ok: bool
    install_summary: str
    scan_ok: bool
    scan_summary: str
    remaining_identifiers: list[str]
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
    errors: list[str]


_REPORT_PREFIX = "# INVESTIGATIVE REPORT"
_GROUP_HEADING_RE = re.compile(r"^### GROUP:\s*(.+?)\s*$", re.MULTILINE)
_BULLET_LABEL_RE = re.compile(r"^- ([^:]+):\s*(.*)$")


def _pipeline_complete(results: _QAExecutionResults) -> bool:
    """Return whether install, scan, and tests have all run at least once."""
    return results.install is not None and results.scan is not None and results.tests is not None


def _review_ready_error(results: _QAExecutionResults) -> str | None:
    """Return the standard review-tool order error, if any."""
    if _pipeline_complete(results):
        return None
    return (
        "ERROR: Review tools are locked until run_dependency_install, "
        "run_security_scan, and run_unit_tests have all been called in order."
    )


def _resolve_action_summary_group_ids(
    summary: AgentActionSummary,
    known_group_ids: set[str],
) -> list[str]:
    """Resolve which exact group_ids an AgentActionSummary applies to."""
    raw_task_id = (summary.task_id or "").strip()
    if not raw_task_id:
        return []
    if raw_task_id.startswith("batch:"):
        payload = raw_task_id[len("batch:") :]
        resolved = []
        for part in payload.split(","):
            candidate = part.strip()
            if candidate and candidate in known_group_ids:
                resolved.append(candidate)
        return resolved
    return [raw_task_id] if raw_task_id in known_group_ids else []


def _relevant_action_summaries(
    action_summaries: list[AgentActionSummary],
    group_id: str,
    known_group_ids: set[str],
) -> list[AgentActionSummary]:
    """Filter action summaries to those explicitly linked to one group."""
    relevant: list[AgentActionSummary] = []
    for summary in action_summaries:
        if group_id in _resolve_action_summary_group_ids(summary, known_group_ids):
            relevant.append(summary)
    return relevant


def _trim_action_summary_text(summary_text: str, group: VulnerabilityGroup) -> str:
    """Trim batch action summary text to only mention the target group's component."""
    group_id = group.group_id
    match = re.search(r"(updates for |edits for )(.+?)(;|$)", summary_text)
    if match:
        groups_list_str = match.group(2)
        if "," in groups_list_str:
            groups = [g.strip() for g in groups_list_str.split(",")]
            if group_id in groups:
                summary_text = summary_text.replace(groups_list_str, group_id, 1)

    # Collect possible match keywords for this group
    keywords = set()
    if group.vulnerable_component:
        keywords.add(group.vulnerable_component.lower())
    for cve in group.cve_ids or []:
        if cve:
            keywords.add(cve.lower())
    for ghsa in group.ghsa_ids or []:
        if ghsa:
            keywords.add(ghsa.lower())
    for issue in group.issues or []:
        if issue.cve_id:
            keywords.add(issue.cve_id.lower())
        if issue.ghsa_id:
            keywords.add(issue.ghsa_id.lower())

    if not keywords:
        return summary_text

    lines = summary_text.splitlines()
    trimmed_lines = []
    for line in lines:
        stripped = line.strip()
        is_bullet = stripped.startswith(("-", "*", "+")) or re.match(r"^\d+\.", stripped)
        is_action_verb = any(
            stripped.lower().startswith(verb)
            for verb in ["updated", "added", "fixed", "upgraded", "downgraded", "removed"]
        )
        if is_bullet or is_action_verb:
            line_lower = stripped.lower()
            has_kw = False
            for kw in keywords:
                # Custom word boundary matching to handle special characters like @ or - in package names
                pattern = rf"(?:^|[^a-zA-Z0-9_@.-]){re.escape(kw)}(?:$|[^a-zA-Z0-9_.-])"
                if re.search(pattern, line_lower):
                    has_kw = True
                    break
            if has_kw:
                trimmed_lines.append(line)
        else:
            trimmed_lines.append(line)

    return "\n".join(trimmed_lines)


def _parse_report_bullets(block_text: str) -> dict[str, str]:
    """Parse markdown '- Label: value' bullets, preserving wrapped lines."""
    fields: dict[str, str] = {}
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        """Commit the current markdown bullet to the parsed field mapping."""
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


def _group_target_identifiers(group: VulnerabilityGroup) -> set[str]:
    """Collect normalized scanner identifiers relevant to one group."""
    identifiers: set[str] = set()
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
    remaining_identifiers: set[str],
) -> list[str]:
    """Return the exact remaining scanner identifiers attributable to one group."""
    return sorted(_group_target_identifiers(group) & remaining_identifiers)


def _derive_qa_group_strategies(
    valid_groups: list[VulnerabilityGroup],
    configured_strategies: dict[str, Any] | None,
    task_queue: dict[str, Any] | None,
    active_target_task_ids: list[str] | None = None,
) -> dict[str, str]:
    """Derive the QA strategy used by scanner guardrails from task state.

    Args:
        valid_groups: Vulnerability groups being evaluated.
        configured_strategies: Legacy or supervisor-provided group strategies.
        task_queue: Current supervisor task queue, if available.
        active_target_task_ids: Tasks whose attempt produced the QA result.

    Returns:
        A normalized strategy map. ``vulnerable_code_removal`` NO_FIX tasks
        are represented as ``code_workaround`` so remaining SCA identifiers do
        not force a false QA failure. ``package_removal`` tasks remain strict.

    Notes:
        The package-removal stage must still fail when its dependency remains
        in the scan. The exemption applies only after the supervisor has
        committed the same task to vulnerable-code removal.
    """
    effective: dict[str, str] = {}
    for group_id, strategy in dict(configured_strategies or {}).items():
        value = getattr(strategy, "value", strategy)
        effective[str(group_id)] = str(value)

    queue = dict(task_queue or {})
    active_ids = list(active_target_task_ids or [])

    def task_candidates(group_id: str) -> list[Any]:
        ordered: list[Any] = []
        seen: set[str] = set()
        for task_id in [*active_ids, *queue.keys()]:
            if task_id in seen:
                continue
            seen.add(task_id)
            task = queue.get(task_id)
            if task is not None and getattr(task, "parent_group_id", None) == group_id:
                ordered.append(task)
        return ordered

    for group in valid_groups:
        candidates = task_candidates(group.group_id)
        task = next(
            (
                candidate
                for candidate in candidates
                if getattr(candidate, "no_fix_stage", None) is not None
            ),
            None,
        )
        stage = getattr(task, "no_fix_stage", None)
        stage_value = getattr(stage, "value", stage)
        if stage_value == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value:
            effective[group.group_id] = "code_workaround"
        elif stage_value == NoFixMitigationStage.PACKAGE_REMOVAL.value:
            effective[group.group_id] = "no_fix_package_removal"
        elif (
            task is None
            and group.fix_plan is not None
            and group.fix_plan.status == FixPlanStatus.NO_FIX
        ):
            # Legacy QA callers may not have a task queue yet. Treating a
            # NO_FIX group as package-removal strict is safer than granting a
            # code-workaround exemption before Stage 2 is committed.
            effective[group.group_id] = "no_fix_package_removal"

    return effective


def _group_scan_status(
    scan_result: tuple[bool, str, set[str]] | None,
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
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    candidate_changed_files: list[str],
    results: _QAExecutionResults,
    reason: str,
) -> str:
    """Synthesize a minimal investigative report when the LLM output is malformed."""
    install_ok, install_summary = results.install or (
        False,
        "run_dependency_install was not called.",
    )
    if results.scan is None:
        scan_result: tuple[bool, str, set[str]] | None = None
    else:
        scan_result = results.scan
    tests_ok, _ = results.tests or (False, "run_unit_tests was not called.")

    post_scan_identifiers = sorted(
        _scan_result_value(scan_result, "found_identifiers", set()) or set()
    )
    new_identifiers = sorted(_scan_result_value(scan_result, "new_identifiers", set()) or set())

    changed_files_text = ", ".join(candidate_changed_files) if candidate_changed_files else "none"
    blocks = [
        _REPORT_PREFIX,
        "## Install Analysis",
        f"- Install Status: {'succeeded' if install_ok else 'failed'}",
        f"- Summary: {reason}",
        "- Suspected Responsible Group(s): unknown",
        f"- Evidence: {install_summary}",
        f"- Post-remediation Scanner Identifiers: {', '.join(post_scan_identifiers) if post_scan_identifiers else 'none'}",
        f"- Newly Introduced Scanner Identifiers: {', '.join(new_identifiers) if new_identifiers else 'none'}",
        "",
    ]

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "(unknown)")
        group_identifiers = sorted(_group_target_identifiers(group))
        group_remaining = (
            _group_remaining_identifiers(group, scan_result[2]) if scan_result is not None else []
        )
        scan_status = _group_scan_status(scan_result, group)
        blocks.extend(
            [
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
            ]
        )
    return "\n".join(blocks).strip()


def _parse_investigation_report(
    report_text: str,
    valid_groups: list[VulnerabilityGroup],
) -> ParsedInvestigationReport:
    """Parse the investigator markdown report into shared and per-group sections (backcompat)."""
    known_group_ids = {group.group_id for group in valid_groups}
    errors: list[str] = []
    warnings: list[str] = []
    normalized = (report_text or "").strip()

    if not normalized.startswith(_REPORT_PREFIX):
        errors.append("Investigation report missing required '# INVESTIGATIVE REPORT' heading.")
        normalized = f"{_REPORT_PREFIX}\n{normalized}".strip()

    matches = list(_GROUP_HEADING_RE.finditer(normalized))
    shared_end = matches[0].start() if matches else len(normalized)
    shared_install_analysis = normalized[len(_REPORT_PREFIX) : shared_end].strip()

    sections: dict[str, GroupInvestigationSection] = {}
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
        placeholder = "\n".join(
            [
                f"### GROUP: {group.group_id}",
                "- Scan Analysis: missing",
                "- Test Attribution & Exoneration: missing",
                "- Diff Review: missing",
            ]
        )
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


def _targeted_extra_args_conflict() -> bool:
    """Return whether configured ODC arguments override required target paths."""
    extra_args = os.environ.get("ODC_EXTRA_ARGS", "").strip()
    if not extra_args:
        return False
    try:
        tokens = shlex.split(extra_args)
    except ValueError:
        return True
    return any(
        token in {"--scan", "--out"} or token.startswith("--scan=") or token.startswith("--out=")
        for token in tokens
    )


def _closure_fallback_reason(reason: str | None) -> ScanFallbackReason:
    """Map pure resolver diagnostics to the typed QA fallback vocabulary."""
    return {
        "ambiguous_target": ScanFallbackReason.AMBIGUOUS_TARGET,
        "incomplete_closure": ScanFallbackReason.INCOMPLETE_CLOSURE,
        "invalid_lockfile": ScanFallbackReason.INVALID_LOCKFILE,
    }.get(reason or "", ScanFallbackReason.INCOMPLETE_CLOSURE)


def _merge_dependency_closures(
    source_lockfile: str,
    closures: Sequence[DependencyClosure],
) -> DependencyClosure:
    """Union complete closures from one lockfile without losing physical keys."""
    node_map = {node.lockfile_key: node for closure in closures for node in closure.nodes}
    return DependencyClosure(
        source_lockfile=source_lockfile,
        root_keys=tuple(sorted({key for closure in closures for key in closure.root_keys})),
        nodes=tuple(node_map[key] for key in sorted(node_map)),
        includes_optional=any(closure.includes_optional for closure in closures),
        includes_peer=any(closure.includes_peer for closure in closures),
        complete=all(closure.complete for closure in closures),
        lockfile_version=closures[0].lockfile_version,
    )


def _resolve_targeted_closures(
    sandbox: DockerSandbox,
    targets: Sequence[QAScanTarget],
) -> tuple[list[DependencyClosure], ScanFallbackReason | None, str | None]:
    """Read live npm lockfiles and resolve the union needed by active tasks."""
    if not targets:
        return [], ScanFallbackReason.MISSING_LOCKFILE, "No active task scan targets were supplied."

    by_source: dict[str, list[QAScanTarget]] = {}
    for target in targets:
        if not target.manifest_paths:
            return (
                [],
                ScanFallbackReason.MISSING_LOCKFILE,
                (f"Task {target.task_id} has no supported lockfile path."),
            )
        for source_lockfile in target.manifest_paths:
            source_lockfile = _validate_qa_path(source_lockfile)
            if Path(source_lockfile).name.lower() != "package-lock.json":
                return (
                    [],
                    ScanFallbackReason.UNSUPPORTED_PACKAGE_MANAGER,
                    (f"Task {target.task_id} uses unsupported lockfile {source_lockfile}."),
                )
            by_source.setdefault(source_lockfile, []).append(target)

    merged: list[DependencyClosure] = []
    for source_lockfile, source_targets in sorted(by_source.items()):
        raw_lockfile = sandbox.read_file(source_lockfile)
        if raw_lockfile is None:
            return (
                [],
                ScanFallbackReason.MISSING_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} could not be read."),
            )
        try:
            lockfile = json.loads(raw_lockfile)
        except (TypeError, json.JSONDecodeError) as exc:
            return (
                [],
                ScanFallbackReason.INVALID_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} is not valid JSON: {exc}"),
            )
        packages = lockfile.get("packages") if isinstance(lockfile, Mapping) else None
        lockfile_version = (
            lockfile.get("lockfileVersion") if isinstance(lockfile, Mapping) else None
        )
        if not isinstance(packages, Mapping) or not isinstance(lockfile_version, int):
            return (
                [],
                ScanFallbackReason.INVALID_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} lacks a supported packages map."),
            )

        closures: list[DependencyClosure] = []
        for target in source_targets:
            closure = resolve_dependency_closure(
                packages,
                source_lockfile=source_lockfile,
                target_package=target.target_package,
                target_version=target.expected_version,
                dependency_ancestry=target.dependency_ancestry,
                include_optional=True,
                include_peer=True,
                lockfile_version=lockfile_version,
            )
            if not closure.complete:
                return (
                    [],
                    _closure_fallback_reason(closure.fallback_reason),
                    (
                        f"Task {target.task_id} closure failed for {source_lockfile}: "
                        f"{closure.fallback_reason or 'unknown reason'}"
                    ),
                )
            closures.append(closure)
        merged.append(_merge_dependency_closures(source_lockfile, closures))
    return merged, None, None


def _write_targeted_artifacts(
    sandbox: DockerSandbox,
    closures: Sequence[DependencyClosure],
) -> tuple[str, list[str], list[str]]:
    """Write synthetic package roots and return scan path plus closure metadata."""
    targeted_subdir = ".odc-targeted"
    package_names: set[str] = set()
    lockfile_keys: set[str] = set()
    for index, closure in enumerate(closures):
        artifacts = build_sliced_lockfile_artifacts(closure)
        subdir = f"{targeted_subdir}/{index:03d}"
        for filename, content in artifacts.items():
            sandbox.write_file(_validate_qa_path(f"{subdir}/{filename}"), content)
        package_names.update(node.package_name for node in closure.nodes)
        lockfile_keys.update(node.lockfile_key for node in closure.nodes)
    return targeted_subdir, sorted(package_names), sorted(lockfile_keys)


def _cleanup_targeted_artifacts(sandbox: DockerSandbox) -> None:
    """Remove the fixed temporary targeted-scan directory from the workspace."""
    try:
        sandbox.run("rm -rf -- .odc-targeted", timeout=30)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: targeted artifact cleanup failed — %s", exc)


def _scan_evidence(
    *,
    targets: Sequence[QAScanTarget],
    scan_result: Any,
    effective_scope: ScanScope,
    complete: bool,
    fallback_reason: ScanFallbackReason | None = None,
    closures: Sequence[DependencyClosure] = (),
) -> ODCScanEvidence:
    """Build typed, attempt-local ODC evidence from a scan result."""
    return ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=effective_scope,
        authoritative=False,
        covered_task_ids=sorted({target.task_id for target in targets}),
        closure_package_names=sorted(
            {node.package_name for closure in closures for node in closure.nodes}
        ),
        closure_lockfile_keys=sorted(
            {node.lockfile_key for closure in closures for node in closure.nodes}
        ),
        found_identifiers=sorted(
            _scan_result_value(scan_result, "found_identifiers", set()) or set()
        ),
        remaining_target_identifiers=sorted(
            _scan_result_value(scan_result, "remaining_identifiers", set()) or set()
        ),
        complete=complete,
        fallback_reason=fallback_reason,
    )


def _run_global_execution(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str] | None = None,
    scan_targets: Sequence[QAScanTarget] | None = None,
) -> _QAExecutionResults:
    """
    Run install, security scan, and unit tests exactly once via direct Python calls.

    No LLM tool wrappers are involved â€” execution is deterministic and sequential.
    Results are stored in a _QAExecutionResults cache for downstream use.
    """
    results = _QAExecutionResults()

    logger.info("qa_critic: [Step 0] running npm install.")
    results.install = _run_install(sandbox)
    install_ok, _ = results.install

    logger.info("qa_critic: [Step 0] running security scan (install_ok=%s).", install_ok)
    scan_started = time.monotonic()
    if scan_targets is None:
        if baseline_identifiers is None:
            # Preserve the legacy helper call shape for direct callers that do not
            # provide a pre-remediation baseline.
            results.scan = _run_security_scan(sandbox, workspace_volume, target_identifiers)
        else:
            results.scan = _run_security_scan(
                sandbox,
                workspace_volume,
                target_identifiers,
                baseline_identifiers,
            )
    else:
        baseline = baseline_identifiers or target_identifiers
        closures: list[DependencyClosure] = []
        targeted_subdir: str | None = None
        fallback_reason: ScanFallbackReason | None = None
        try:
            if _targeted_extra_args_conflict():
                fallback_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
            else:
                closures, fallback_reason, resolution_detail = _resolve_targeted_closures(
                    sandbox,
                    scan_targets,
                )
                if resolution_detail:
                    logger.info("qa_critic: targeted scan fallback: %s", resolution_detail)
            if fallback_reason is None:
                targeted_subdir = ".odc-targeted"
                _write_targeted_artifacts(sandbox, closures)
                targeted_result = _run_targeted_security_scan(
                    sandbox,
                    workspace_volume,
                    target_identifiers,
                    baseline,
                    targeted_subdir,
                )
                if (
                    not targeted_result.ok
                    and not targeted_result.found_identifiers
                    and not targeted_result.remaining_identifiers
                ):
                    fallback_reason = (
                        ScanFallbackReason.TARGETED_REPORT_UNPARSEABLE
                        if "report" in targeted_result.summary.lower()
                        else ScanFallbackReason.TARGETED_SCAN_FAILED
                    )
                else:
                    results.scan = targeted_result
                    results.scan_evidence = _scan_evidence(
                        targets=scan_targets,
                        scan_result=targeted_result,
                        effective_scope=ScanScope.TARGETED,
                        complete=True,
                        closures=closures,
                    )
        except (ClosureResolutionError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("qa_critic: targeted scan setup failed — %s", exc)
            fallback_reason = ScanFallbackReason.TARGETED_SCAN_FAILED
        finally:
            if targeted_subdir is not None:
                _cleanup_targeted_artifacts(sandbox)

        if fallback_reason is not None:
            if baseline_identifiers is None:
                results.scan = _run_security_scan(sandbox, workspace_volume, target_identifiers)
            else:
                results.scan = _run_security_scan(
                    sandbox,
                    workspace_volume,
                    target_identifiers,
                    baseline,
                )
            results.scan_evidence = _scan_evidence(
                targets=scan_targets,
                scan_result=results.scan,
                effective_scope=ScanScope.FULL,
                complete=False,
                fallback_reason=fallback_reason,
                closures=closures,
            )

    if results.scan_evidence is not None:
        logger.info(
            "qa_critic: scan requested_scope=%s effective_scope=%s tasks=%d closure_packages=%d "
            "fallback_reason=%s duration_seconds=%.3f",
            results.scan_evidence.requested_scope.value,
            results.scan_evidence.effective_scope.value,
            len(results.scan_evidence.covered_task_ids),
            len(results.scan_evidence.closure_package_names),
            results.scan_evidence.fallback_reason.value
            if results.scan_evidence.fallback_reason
            else None,
            time.monotonic() - scan_started,
        )
    else:
        logger.info(
            "qa_critic: scan requested_scope=full effective_scope=full tasks=0 "
            "closure_packages=0 fallback_reason=None duration_seconds=%.3f",
            time.monotonic() - scan_started,
        )

    logger.info("qa_critic: [Step 0] running unit tests.")
    results.tests = _run_unit_tests(sandbox)

    return results


# ---------------------------------------------------------------------------
# QA toolbelt (backcompat: execution + review tools together)
# ---------------------------------------------------------------------------


def build_qa_toolbelt(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    candidate_changed_files: list[str],
    host_repo_root: str | None,
    baseline_identifiers: set[str] | None = None,
) -> tuple[list, _QAExecutionResults]:
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
    from remediation_engine.orchestration.remedy_tools import (
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
            return f"[CACHED â€” already run] {summary}"
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
            return f"[CACHED â€” already run] {summary}"
        if results.install is None:
            return "ERROR: run_security_scan must be called after run_dependency_install."
        if baseline_identifiers is None:
            scan_result = _run_security_scan(
                sandbox,
                workspace_volume,
                target_identifiers,
            )
        else:
            scan_result = _run_security_scan(
                sandbox,
                workspace_volume,
                target_identifiers,
                baseline_identifiers,
            )
        results.scan = scan_result
        return scan_result.summary

    @tool
    def run_unit_tests() -> str:
        """
        Run 'npm test' inside the workspace.
        Must be called after run_security_scan.
        Repeated calls return the cached result immediately.
        """
        if results.tests is not None:
            _, summary = results.tests
            return f"[CACHED â€” already run] {summary}"
        if results.scan is None:
            return "ERROR: run_unit_tests must be called after run_security_scan."
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
        diff_text, _ = _generate_workspace_diff(host_repo_root, sandbox, candidate_changed_files)
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
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Search the workspace after the fixed QA pipeline has completed."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(
            search_tool.invoke(
                {"search_pattern": search_pattern, "target_directory": target_directory}
            )
        )

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
    candidate_changed_files: list[str],
    host_repo_root: str | None,
    results: _QAExecutionResults,
) -> list:
    """
    Build a read-only review toolbelt for individual group investigators.

    This toolbelt contains NO execution tools.  The results cache is pre-populated
    by _run_global_execution (Step 0) before any investigator runs.

    Tools: list_changed_files, generate_workspace_diff, read_file_context,
           search_codebase_pattern, inspect_ast_symbol, query_qa_logs.
    """
    from remediation_engine.orchestration.remedy_tools import (
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
        diff_text, _ = _generate_workspace_diff(host_repo_root, sandbox, candidate_changed_files)
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
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Search the workspace for a regex pattern in source files."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(
            search_tool.invoke(
                {"search_pattern": search_pattern, "target_directory": target_directory}
            )
        )

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
    group_remaining_ids: list[str],
    candidate_changed_files: list[str],
    action_summaries: list[AgentActionSummary],
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
    post_scan_identifiers = sorted(
        _scan_result_value(results.scan, "found_identifiers", set()) or set()
    )
    new_identifiers = sorted(_scan_result_value(results.scan, "new_identifiers", set()) or set())

    summaries_text = (
        "\n".join(f"  - {s.status.value}: {s.summary}" for s in action_summaries) or "  (none)"
    )

    remaining_text = (
        ", ".join(group_remaining_ids)
        if group_remaining_ids
        else "(none â€” scanner cleared this group)"
    )

    trimmed_summaries = []
    for s in action_summaries:
        trimmed_text = _trim_action_summary_text(s.summary, group)
        trimmed_summaries.append(f"  - {s.status.value}: {trimmed_text}")
    summaries_text = "\n".join(trimmed_summaries) or "  (none)"

    return f"""You are a QA Investigator Agent assigned to review exactly ONE vulnerability group.

## Your Assigned Group
- Group ID       : {group.group_id}
- Component      : {group.vulnerable_component or "(unknown)"}
- Issue Type     : {group.issue_type.value}
- Routing Strategy: {strategy}
- CVEs           : {cves}
- GHSAs          : {ghsas}
- Fix Plan Status: {fix_plan_status}
- Fix Instruction: {fix_instruction}

## Agent Action Summaries for This Group
{summaries_text}

## This Group's Deterministic Remaining Scanner Identifiers
{remaining_text}

## Global Post-remediation Scanner Snapshot
All identifiers found after remediation: {", ".join(post_scan_identifiers) if post_scan_identifiers else "(none or unavailable)"}
New identifiers absent from the pre-remediation baseline: {", ".join(new_identifiers) if new_identifiers else "(none)"}
New identifiers are graph-level findings for a later triage phase; do not attribute them to this group unless deterministic evidence explicitly supports that conclusion.

## Your Task
You are investigating ONLY this group ({group.group_id}). Use the provided review tools
(list_changed_files, generate_workspace_diff, read_file_context, search_codebase_pattern,
inspect_ast_symbol, query_qa_logs) as needed to answer the following questions.

All three execution tools (install, scan, tests) have ALREADY been run globally.
Do NOT attempt to call run_dependency_install, run_security_scan, or run_unit_tests â€”
they are not available to you.

## Questions to Answer
1. Package/Domain Purpose: What does this package/component do? What domain does it serve?
2. Relevant Global Failures: Which (if any) of the install/scan/test failures are relevant to this group's domain?
3. Plausible Causation: Did this group's remediation plausibly cause the observed install or test failures? Reason deductively.
4. Scanner Findings: Do the deterministic remaining scanner identifiers above indicate this group still has unresolved vulnerabilities?
5. Global New Findings: Note any newly introduced identifiers, but do not assign ownership to this group without evidence.
6. Workaround Path Review (if CODE_WORKAROUND): Does the changed code plausibly block the vulnerable execution path? Inspect the diff or relevant files.
7. Exoneration or Uncertainty: Explicitly state whether this group is exonerated from failures attributed to other groups, or whether there is genuine uncertainty.

## Output Format
Write a free-form Markdown investigation report answering the 6 questions above.
Be specific. Reference exact test names, file names, or scanner identifiers where possible.
End with a one-sentence summary verdict for this group.

Do NOT assign a final pass/fail verdict â€” that is the Batch Judge's responsibility.
"""


# ---------------------------------------------------------------------------
# Map phase: run individual investigators (one per group)
# ---------------------------------------------------------------------------


def _run_individual_investigations(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    candidate_changed_files: list[str],
    sandbox: DockerSandbox,
    repo_root: str | None,
    results: _QAExecutionResults,
) -> dict[str, GroupInvestigation]:
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

    model_name = AppSettings.from_env().qa_llm_model
    known_group_ids = {group.group_id for group in valid_groups}
    investigations: dict[str, GroupInvestigation] = {}

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "version_bump")

        remaining_scan = results.scan[2] if results.scan else set()
        group_remaining_ids = _group_remaining_identifiers(group, remaining_scan)

        relevant_summaries = _relevant_action_summaries(
            action_summaries, group.group_id, known_group_ids
        )

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
    group_remaining_ids: list[str],
    reason: str,
) -> str:
    """Synthesize a minimal investigation for a single group when the investigator fails."""
    install_ok, _ = results.install or (False, "not run")
    tests_ok, _ = results.tests or (False, "not run")
    scan_ok = results.scan[0] if results.scan else False
    new_identifiers = sorted(_scan_result_value(results.scan, "new_identifiers", set()) or set())

    remaining_text = ", ".join(group_remaining_ids) if group_remaining_ids else "none"
    scan_status = (
        "still_flagged" if group_remaining_ids else ("cleared" if scan_ok else "scan_failed")
    )

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
        f"- Global Newly Introduced Scanner Identifiers: {', '.join(new_identifiers) if new_identifiers else 'none'}\n"
        f"- Unit Tests: {'PASSED' if tests_ok else 'FAILED'}\n\n"
        f"**Summary:** Fallback investigation synthesized from deterministic results only. "
        f"No investigator prose was available."
    )


# ---------------------------------------------------------------------------
# Reduce phase: batch judge
# ---------------------------------------------------------------------------


def _build_batch_judge_prompt(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    results: _QAExecutionResults,
    investigations_by_group: dict[str, GroupInvestigation],
) -> str:
    """Build the single comprehensive prompt for the batch judge."""
    known_group_ids = {group.group_id for group in valid_groups}

    install_ok, install_summary = results.install or (False, "not run")
    scan_ok, scan_summary, remaining_global = results.scan or (False, "not run", set())
    tests_ok, tests_summary = results.tests or (False, "not run")
    post_scan_identifiers = sorted(
        _scan_result_value(results.scan, "found_identifiers", set()) or set()
    )
    new_identifiers = sorted(_scan_result_value(results.scan, "new_identifiers", set()) or set())

    # Detect install conflict type for guardrail hint
    install_conflict_hint = ""
    if not install_ok:
        for pattern in _PEER_CONFLICT_PATTERNS:
            if pattern.lower() in install_summary.lower():
                install_conflict_hint = (
                    f"\nâš ï¸  Install failure contains '{pattern}' â€” "
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
        trimmed_summaries = []
        for s in relevant_summaries:
            trimmed_text = _trim_action_summary_text(s.summary, group)
            trimmed_summaries.append(f"    - {s.status.value}: {trimmed_text}")
        summaries_text = "\n".join(trimmed_summaries) or "    (none)"

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
- Post-remediation identifiers: {", ".join(post_scan_identifiers) if post_scan_identifiers else "(none or unavailable)"}
- Newly introduced identifiers: {", ".join(new_identifiers) if new_identifiers else "(none)"}
- Newly introduced identifiers are global findings for later triage. Do not force them into an existing group's evaluation or retry feedback.

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

6. If a group has BOTH unresolved scanner evidence AND is plausibly causing unit test failures, choose **SECURITY_FLAG** as the single failure_category. Mention the test regression in retry_feedback and the holistic report, but do not label the group BREAKING_CHANGE.

7. **Do not double-attribute** the same test failure to multiple groups unless evidence explicitly supports multiple causes.

8. Resolve contradictions between individual investigations using the deterministic scanner results as the ground truth.
9. Report newly introduced identifiers explicitly in the holistic report, but leave the per-group evaluations scoped to the assigned remediation groups.
10. For failed CODE_WORKAROUND groups, `retry_feedback` MUST include detailed diagnostic feedback:
    - Exact error messages from test execution, runtime logs, or compilation output.
    - Specific failing test names and test files.
    - File and line locations of the failure if available.
    - Specific guidance on syntax errors, type errors, or broken imports introduced by the workaround edit.

## Output Requirements

Return a BatchQAResult with:
- `holistic_report`: A markdown narrative listing: (a) responsible groups, (b) possibly responsible groups, (c) exonerated groups, and (d) any newly introduced global scanner identifiers. Reference specific test names, scanner IDs, or diff evidence.
- `evaluations`: A list of exactly {len(valid_groups)} QAEvaluation objects, one per group.
  - Each evaluation must have: group_id (exact), passed (bool), failure_category (null if passed), retry_feedback (null if passed, specific actionable guidance if failed).

You MUST emit exactly {len(valid_groups)} evaluations, one for each group ID listed above.
"""


def _run_batch_judge(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    results: _QAExecutionResults,
    investigations_by_group: dict[str, GroupInvestigation],
) -> BatchQAResult:
    """
    Reduce phase: one structured LLM call across all group investigations.

    Uses ChatOpenAI.with_structured_output(BatchQAResult) exactly once.
    On LLM failure, synthesizes a failed BatchQAResult for all groups.
    """
    from langchain_openai import ChatOpenAI

    model_name = AppSettings.from_env().qa_llm_model
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(BatchQAResult)

    prompt = _build_batch_judge_prompt(
        valid_groups=valid_groups,
        group_strategies=group_strategies,
        action_summaries=action_summaries,
        results=results,
        investigations_by_group=investigations_by_group,
    )

    logger.info("qa_critic: [Reduce] invoking batch judge for %d group(s).", len(valid_groups))
    try:
        batch_result: BatchQAResult = invoke_with_trajectory(
            "qa_critic.batch_judge",
            lambda: llm.invoke(prompt),
            prompt,
        )
        logger.info(
            "qa_critic: [Reduce] batch judge returned %d evaluation(s).",
            len(batch_result.evaluations),
        )
        return batch_result
    except Exception as exc:  # noqa: BLE001
        logger.error("qa_critic: batch judge LLM failed â€” %s", exc)
        fallback_evals = [
            QAEvaluation(
                task_id=group.group_id,
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
    valid_groups: list[VulnerabilityGroup],
    batch_result: BatchQAResult,
    results: _QAExecutionResults,
    group_strategies: dict[str, str],
) -> tuple[dict[str, QAEvaluation], list[str]]:
    """
    Normalize and validate BatchQAResult evaluations into a Dict[group_id, QAEvaluation].

    Guardrails applied (in order):
    1. Unknown group_ids in evaluations are dropped with an error.
    2. Duplicate evaluations: keep the first, log an error.
    3. Missing groups: synthesize passed=False / SECURITY_FLAG evaluation.
    4. Deterministic scanner guardrail for strict dependency-removal groups:
       If remaining scanner identifiers exist â†’ force passed=False / SECURITY_FLAG.
       CODE_WORKAROUND groups are exempt (trust the workaround code review).
    5. Install error guardrail: if install failed with ERESOLVE/EBADENGINE/peer text,
       downgrade BREAKING_CHANGE to PEER_CONFLICT for VERSION_BUMP groups.

    Returns:
        (evaluations_dict, error_list)
    """
    known_group_ids = {group.group_id for group in valid_groups}
    errors: list[str] = []

    # Phase 1: deduplicate and filter unknown group_ids
    seen: set[str] = set()
    normalized: dict[str, QAEvaluation] = {}
    for evaluation in batch_result.evaluations:
        gid = evaluation.task_id
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
                task_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Batch QA Judge omitted this group; retry required.",
            )

    # Phase 3: deterministic scanner guardrail
    remaining_global: set[str] = results.scan[2] if results.scan else set()
    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "version_bump")
        group_remaining = _group_remaining_identifiers(group, remaining_global)
        if not group_remaining:
            continue  # Scanner cleared this group â€” no guardrail needed.
        if strategy == "code_workaround":
            continue  # Trust the workaround code review â€” exempt from forced failure.
        # Strict dependency-removal strategies with remaining identifiers
        # force failure. Stage-two NO_FIX tasks are normalized to
        # ``code_workaround`` and intentionally skip this branch.
        current = normalized[group.group_id]
        if current.passed or current.failure_category == FailureCategory.BREAKING_CHANGE:
            errors.append(
                f"qa_critic guardrail: group '{group.group_id}' ({strategy}) has remaining "
                f"scanner identifiers {group_remaining} but judge did not prioritize SECURITY_FLAG; "
                "forcing passed=False / SECURITY_FLAG."
            )
            retry_feedback = (
                f"Scanner still detects unresolved identifiers: {', '.join(group_remaining)}. "
                "The dependency-removal stage did not fully remove the vulnerable dependency. "
                "Check that the configured package declaration and resolved dependency graph were updated."
            )
            if (
                current.failure_category == FailureCategory.BREAKING_CHANGE
                and current.retry_feedback
            ):
                retry_feedback = (
                    retry_feedback
                    + " Test regressions may also be present, but SECURITY_FLAG takes precedence until the unresolved scanner findings are cleared. "
                    + current.retry_feedback
                )
            normalized[group.group_id] = QAEvaluation(
                task_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=retry_feedback,
            )

    # Phase 4: install conflict guardrail â€” map BREAKING_CHANGE â†’ PEER_CONFLICT
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
                if (
                    not current.passed
                    and current.failure_category == FailureCategory.BREAKING_CHANGE
                ):
                    normalized[group.group_id] = QAEvaluation(
                        task_id=group.group_id,
                        passed=False,
                        failure_category=FailureCategory.PEER_CONFLICT,
                        retry_feedback=(
                            (current.retry_feedback or "")
                            + " [Guardrail: install failure contains peer conflict indicators; "
                            "reclassified from BREAKING_CHANGE to PEER_CONFLICT.]"
                        ),
                    )

    return normalized, errors


def _attach_failure_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    results: _QAExecutionResults,
    state: OrchestratorState,
    *,
    deterministic_evidence: QAFailureEvidence | None = None,
    sandbox: DockerSandbox | None = None,
) -> dict[str, QAEvaluation]:
    """Attach deterministic QA diagnostics and committed attempt provenance.

    The batch judge is allowed to classify a failure, but it is not the source
    of truth for the failing test output or source paths. Use deterministic
    test evidence and the task's committed attempt envelope so workaround
    retries receive actionable evidence instead of hallucinated locations or
    a blank/stale ``attempt_id``.

    Args:
        evaluations: Batch-judge evaluations keyed by vulnerability group.
        results: Deterministic QA execution results.
        state: Current orchestration state containing committed task envelopes.
        deterministic_evidence: Evidence extracted while the QA sandbox was
            active, when available.
        sandbox: Optional active sandbox for lazy deterministic extraction.

    Returns:
        Evaluations enriched with authoritative evidence and provenance.
    """
    test_evidence = deterministic_evidence
    if test_evidence is None and results.tests and not results.tests[0]:
        test_evidence = extract_qa_failure_evidence(
            1,
            results.tests[1],
            "",
            sandbox=sandbox,
        )

    task_queue = state.get("task_queue", {}) or {}
    tasks_by_group = {
        task.parent_group_id: task
        for task in task_queue.values()
        if getattr(task, "parent_group_id", None)
    }
    enriched: dict[str, QAEvaluation] = {}
    for group_id, evaluation in evaluations.items():
        if evaluation.passed:
            enriched[group_id] = evaluation
            continue

        task = tasks_by_group.get(group_id)
        attempt_id = getattr(task, "current_attempt_id", None) or ""
        task_revision = int(getattr(task, "task_revision", 0) or 0)
        llm_evidence = evaluation.failure_evidence
        evidence = test_evidence
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
                    "attempt_id": attempt_id or evidence.attempt_id,
                    "task_revision": task_revision or evidence.task_revision,
                }
            )
            evaluation = evaluation.model_copy(update={"failure_evidence": evidence})
        enriched[group_id] = evaluation
    return enriched


def _attach_scan_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    evidence: ODCScanEvidence | None,
) -> dict[str, QAEvaluation]:
    """Attach the same attempt-local scan evidence to each QA evaluation."""
    if evidence is None:
        return evaluations
    return {
        group_id: evaluation.model_copy(update={"scan_evidence": evidence})
        for group_id, evaluation in evaluations.items()
    }


# ---------------------------------------------------------------------------
# QA agent system prompt (backcompat â€” used by legacy _run_investigator_phase)
# ---------------------------------------------------------------------------


def _build_qa_system_prompt(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    candidate_changed_files: list[str],
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
        trimmed_summaries = []
        for s in relevant_summaries:
            trimmed_text = _trim_action_summary_text(s.summary, group)
            trimmed_summaries.append(f"    - {s.status.value}: {trimmed_text}")
        summaries_text = "\n".join(trimmed_summaries) or "    (none)"

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
1. `run_dependency_install` â€” installs dependencies and surfaces install errors
2. `run_security_scan` â€” runs OWASP Dependency-Check and checks for remaining CVEs/GHSAs
3. `run_unit_tests` â€” runs the full test suite

Each execution tool is one-shot guarded: repeated calls return the cached result.
Do NOT skip or reorder these three steps.

## Review Tools (Conditional â€” Call Only When Needed)
After running all three execution tools, use review tools only if failures or ambiguous
signals require investigation:
- `list_changed_files` â€” list the files the remedy agents modified
- `generate_workspace_diff` â€” diff changed files vs. host baseline (for workaround/scan paradox)
- `read_file_context` â€” read a specific workspace file
- `search_codebase_pattern` â€” regex search across workspace source files
- `inspect_ast_symbol` â€” extract a named function or class from a file
- `query_qa_logs` â€” retrieve the bounded log output for install, scan, or tests

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
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
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
        False,
        "run_dependency_install was not called.",
    )
    if results.scan is None:
        scan_ok, scan_summary, remaining_identifiers = (
            False,
            "run_security_scan was not called.",
            set(),
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
            f"{summary.status.value}: {summary.summary}" for summary in relevant_summaries
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
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    candidate_changed_files: list[str],
    sandbox: DockerSandbox,
    repo_root: str | None,
    workspace_volume: str,
    target_identifiers: set[str],
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

    model_name = AppSettings.from_env().qa_llm_model
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
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    investigation: InvestigationArtifact,
) -> dict[str, QAEvaluation]:
    """Run the zero-shot Judge once per vulnerability group (backcompat)."""
    from langchain_openai import ChatOpenAI

    model_name = AppSettings.from_env().qa_llm_model
    llm = ChatOpenAI(model=model_name, temperature=0).with_structured_output(QAEvaluation)

    evaluations: dict[str, QAEvaluation] = {}
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
        summaries_text = (
            "\n".join(f"- {summary}" for summary in packet.action_summaries) or "- (none)"
        )

        prompt = f"""You are the Judge phase of a QA Critic for one vulnerability group.

## Group
- Group ID: {group.group_id}
- Component: {group.vulnerable_component or "(unknown)"}
- Issue Type: {group.issue_type.value}
- CVEs: {cves}
- GHSAs: {ghsas}
- Routing Strategy: {packet.strategy}
- Fix Plan Status: {packet.fix_plan_status}
- Fix Plan Instruction: {packet.fix_plan_instruction}

## Shared Install Analysis
{packet.shared_install_analysis or "(none)"}

## Group-Specific Investigation Block
{packet.group_block_markdown}

## Deterministic Flags (Hard Rules)
- Global Install Success: {packet.install_ok}
- Global Tests Success: {packet.tests_ok}
- THIS GROUP'S Remaining Scanner Identifiers: {", ".join(packet.remaining_identifiers) if packet.remaining_identifiers else "(none)"}

## Relevant Action Summaries
{summaries_text}

## Parsed Group Fields
- Scan Reasoning: {packet.section.scan_reasoning or "(none)"}
- Workaround Review: {packet.section.workaround_review or "(none)"}
- Diff Evidence: {packet.section.diff_evidence or "(none)"}
- Attributed Test Failures: {packet.section.attributed_test_failures or "(none)"}
- Causal Reasoning: {packet.section.causal_reasoning or "(none)"}
- Exonerated Groups: {packet.section.exonerated_groups or "(none)"}
- Group Summary: {packet.section.group_summary or "(none)"}

## Evaluation Rules
1. VERSION_BUMP passes only when install succeeds, THIS GROUP'S Remaining Scanner Identifiers is "(none)" under Deterministic Flags (Hard Rules), AND (tests pass OR the investigation report explicitly attributes the test failures to a different group / exonerates this group).
2. CODE_WORKAROUND may pass even when the scanner flags that there are still vulnerabilities when the code review or diff proves the vulnerable path is blocked AND (normal tests pass OR the investigation report explicitly exonerates this group from the test failures).
3. Use PEER_CONFLICT for install failures caused by dependency conflicts such as ERESOLVE, EBADENGINE, or peer tree incompatibilities.
4. Use BREAKING_CHANGE for test regressions or behavior changes caused by this group's remediation.
5. Use SECURITY_FLAG for unresolved scanner evidence or flawed workaround logic.
6. If this group has both unresolved scanner evidence and plausible test regressions, choose SECURITY_FLAG as the single failure_category and mention the regressions in retry_feedback.
7. Judge only this group. Ignore any missing data about other groups.

Return a QAEvaluation for group_id="{group.group_id}" with:
- passed: true/false
- failure_category: null when passed=true; otherwise PEER_CONFLICT, BREAKING_CHANGE, or SECURITY_FLAG
- retry_feedback: null when passed=true; otherwise specific, actionable guidance for the remediation agent
"""

        try:
            evaluation: QAEvaluation = invoke_with_trajectory(
                f"qa_critic.group_judge.{group.group_id}",
                lambda: llm.invoke(prompt),
                prompt,
            )
            evaluations[group.group_id] = evaluation
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "qa_critic: LLM evaluation failed for group %s - %s",
                group.group_id,
                exc,
            )
            evaluations[group.group_id] = QAEvaluation(
                task_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"QA Critic LLM evaluation raised an exception: {exc}. "
                    "Please retry the remediation."
                ),
            )

    return evaluations


def _create_skinny_qa_group(group: VulnerabilityGroup) -> VulnerabilityGroup:
    """Create a skinny copy of a group for QA evaluator agents."""
    return group.model_copy(
        update={
            "issues": [],
            "localized_issues": [],
        }
    )


def _filter_recent_action_summaries(
    action_summaries: list[AgentActionSummary], valid_groups: list[VulnerabilityGroup]
) -> list[AgentActionSummary]:
    """Extract ONLY the most recently appended summary for each group_id."""
    known_group_ids = {g.group_id for g in valid_groups}
    recent_summaries = []
    seen_ids = set()
    for summary in reversed(action_summaries):
        gids = _resolve_action_summary_group_ids(summary, known_group_ids)
        new_gids = set(gids) - seen_ids
        if new_gids:
            recent_summaries.insert(0, summary)
            seen_ids.update(new_gids)
        if seen_ids == known_group_ids:
            break
    return recent_summaries


# ---------------------------------------------------------------------------
# Compatibility wrapper and node entry point
# ---------------------------------------------------------------------------


def _extract_group_evaluations(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    results: _QAExecutionResults,
    agent_transcript: str,
) -> dict[str, QAEvaluation]:
    """Backward-compatible wrapper around the Judge phase."""
    parsed_report = _parse_investigation_report(
        agent_transcript
        if agent_transcript.strip().startswith(_REPORT_PREFIX)
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
def run_qa_critic_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node: run the map-reduce QA pipeline for the supplied group scope.

    Normal Supervisor dispatches supply one task and therefore one parent group;
    direct callers may still provide multiple groups for batch QA.

    Pipeline:
      Step 0 â€” Global Execution (deterministic Python, no LLM tools)
      Map     â€” One bounded ReAct investigator per group (read-only tools)
      Reduce  â€” One batch judge call (with_structured_output(BatchQAResult))
      Guards  â€” Python guardrails normalize and validate evaluations
    """
    valid_groups: list[VulnerabilityGroup] = state.get("valid_groups") or []
    workspace_volume: str | None = state.get("workspace_volume")
    repo_root: str | None = state.get("repo_root")
    action_summaries: list[AgentActionSummary] = state.get("action_summaries") or []
    group_strategies: dict[str, str] = _derive_qa_group_strategies(
        valid_groups,
        state.get("group_strategies") or {},
        state.get("task_queue") or {},
        state.get("active_target_task_ids") or [],
    )
    candidate_changed_files: list[str] = state.get("changed_files") or []
    baseline_identifiers = _collect_baseline_identifiers(state, valid_groups)
    scan_targets = _build_qa_scan_targets(state, valid_groups)
    scan_is_authoritative = scan_targets is None
    unscanned_projection = _scan_state_projection(
        _QAExecutionResults(),
        baseline_identifiers,
        authoritative=scan_is_authoritative,
    )

    if not valid_groups and not state.get("force_qa"):
        logger.info("qa_critic: no valid groups - skipping QA.")
        return {
            "qa_evaluations": {},
            "eval_status": "all_passed",
            "status": "qa_completed",
            "changed_files": [],
            "qa_investigation_report": "",
            **unscanned_projection,
        }

    if not workspace_volume:
        err = "qa_critic: workspace_volume is not set; cannot run QA."
        logger.error(err)
        failed_evals = {
            group.group_id: QAEvaluation(
                task_id=group.group_id,
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
            **unscanned_projection,
        }

    target_identifiers = _collect_target_identifiers(valid_groups)
    logger.info(
        "qa_critic: evaluating %d groups with %d target identifiers.",
        len(valid_groups),
        len(target_identifiers),
    )

    valid_groups = [_create_skinny_qa_group(g) for g in valid_groups]
    action_summaries = _filter_recent_action_summaries(action_summaries, valid_groups)

    errors: list[str] = []
    deterministic_test_evidence: QAFailureEvidence | None = None
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            # ------------------------------------------------------------------
            # Step 0: Global Execution (deterministic Python, exactly once)
            # ------------------------------------------------------------------
            results = _run_global_execution(
                sandbox=sandbox,
                workspace_volume=workspace_volume,
                target_identifiers=target_identifiers,
                baseline_identifiers=baseline_identifiers,
                scan_targets=scan_targets,
            )
            scan_projection = _scan_state_projection(
                results,
                baseline_identifiers,
                authoritative=scan_is_authoritative,
            )

            # ------------------------------------------------------------------
            # Pipeline completeness guard
            # ------------------------------------------------------------------
            missing_tools: list[str] = []
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
                        task_id=group.group_id,
                        passed=False,
                        failure_category=FailureCategory.SECURITY_FLAG,
                        retry_feedback=(
                            f"QA global execution incomplete: {missing_list} did not run. "
                            "Please retry."
                        ),
                    )
                    for group in valid_groups
                }
                failed_evals = _attach_scan_evidence_to_evaluations(
                    failed_evals,
                    results.scan_evidence,
                )
                return {
                    "qa_evaluations": failed_evals,
                    "eval_status": "failures_detected",
                    "status": "qa_failed",
                    "errors": errors,
                    "changed_files": [],
                    "qa_investigation_report": "",
                    "scan_evidence": results.scan_evidence,
                    **scan_projection,
                }

            if results.tests and not results.tests[0]:
                deterministic_test_evidence = extract_qa_failure_evidence(
                    1,
                    results.tests[1],
                    "",
                    sandbox=sandbox,
                )

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
                task_id=group.group_id,
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
            **unscanned_projection,
        }

    # Collect errors from map phase
    for investigation in investigations_by_group.values():
        errors.extend(investigation.errors)

    # ------------------------------------------------------------------
    # Reduce: Batch Judge (one LLM call for the current group scope)
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
    qa_evaluations = _attach_failure_evidence_to_evaluations(
        qa_evaluations,
        results,
        state,
        deterministic_evidence=deterministic_test_evidence,
    )
    qa_evaluations = _attach_scan_evidence_to_evaluations(
        qa_evaluations,
        results.scan_evidence,
    )

    all_passed = all(evaluation.passed for evaluation in qa_evaluations.values())
    eval_status = "all_passed" if all_passed else "failures_detected"

    logger.info(
        "qa_critic: eval_status=%s (%d/%d passed).",
        eval_status,
        sum(1 for evaluation in qa_evaluations.values() if evaluation.passed),
        len(qa_evaluations),
    )

    qa_investigation_report = _augment_qa_report_with_scan_findings(
        batch_result.holistic_report,
        scan_projection,
    )

    return {
        "qa_evaluations": qa_evaluations,
        "eval_status": eval_status,
        "status": "qa_completed",
        "changed_files": candidate_changed_files,
        "errors": errors,
        "qa_investigation_report": qa_investigation_report,
        "scan_evidence": results.scan_evidence,
        **scan_projection,
    }


@traceable(name="final_full_scan")
def run_final_full_scan_node(state: OrchestratorState) -> dict[str, Any]:
    """Run the authoritative full ODC scan immediately before teardown."""
    workspace_volume = state.get("workspace_volume")
    groups: list[VulnerabilityGroup] = list(state.get("valid_groups") or [])
    baseline = _collect_baseline_identifiers(state, groups)
    target_identifiers = _collect_target_identifiers(groups)
    if not workspace_volume:
        error = "final_full_scan: workspace_volume is missing."
        result = FinalFullScanResult(
            completed=False,
            found_identifiers=[],
            remaining_target_identifiers=[],
            new_identifiers=[],
            status="scan_failed",
            triage_required=False,
            error=error,
        )
        return {
            "final_full_scan_result": result,
            "final_full_scan_completed": True,
            "new_vulnerability_status": "scan_failed",
            "triage_required": False,
            "status": "final_scan_failed",
            "errors": [error],
        }

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            scan = _run_security_scan(
                sandbox,
                workspace_volume,
                target_identifiers,
                baseline,
            )
    except RuntimeError as exc:
        error = f"final_full_scan: Docker sandbox unavailable - {exc}"
        result = FinalFullScanResult(
            completed=False,
            found_identifiers=[],
            remaining_target_identifiers=[],
            new_identifiers=[],
            status="scan_failed",
            triage_required=False,
            error=error,
        )
        return {
            "final_full_scan_result": result,
            "final_full_scan_completed": True,
            "new_vulnerability_status": "scan_failed",
            "triage_required": False,
            "status": "final_scan_failed",
            "errors": [error],
        }

    found = sorted(scan.found_identifiers)
    remaining = sorted(scan.remaining_identifiers)
    new_identifiers = sorted(scan.new_identifiers)
    hard_failure = not scan.ok and not scan.found_identifiers and not scan.remaining_identifiers
    status = (
        "scan_failed"
        if hard_failure
        else "detected"
        if new_identifiers
        else "unresolved"
        if remaining
        else "none"
    )
    triage_required = bool(new_identifiers or remaining)
    result = FinalFullScanResult(
        completed=not hard_failure,
        found_identifiers=found,
        remaining_target_identifiers=remaining,
        new_identifiers=new_identifiers,
        found_issues=list(scan.found_issues),
        status=status,
        triage_required=triage_required,
        error=scan.summary if hard_failure else None,
    )
    logger.info(
        "final_full_scan: completed=%s found=%d remaining=%d new=%d triage_required=%s",
        result.completed,
        len(found),
        len(remaining),
        len(new_identifiers),
        triage_required,
    )
    return {
        "final_full_scan_result": result,
        "final_full_scan_completed": True,
        "baseline_scan_identifiers": sorted(baseline),
        "post_remediation_scan_identifiers": found,
        "post_remediation_scan_issues": list(scan.found_issues),
        "new_vulnerability_identifiers": new_identifiers,
        "new_vulnerability_status": status,
        "triage_required": triage_required,
        "status": "final_scan_completed" if not hard_failure else "final_scan_failed",
        "errors": [scan.summary] if hard_failure else [],
    }
