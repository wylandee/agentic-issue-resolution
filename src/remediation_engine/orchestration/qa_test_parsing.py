"""Deterministic test execution, runner detection, and QA evidence parsing."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

from remediation_engine.contracts.schemas import PeerConflictEvidence, QAFailureEvidence
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

from . import _test_normalization
from ._test_normalization import (
    _NormalizedSuiteResult,
    _NormalizedTestDiagnostic,
    _NormalizedTestFailure,
)
from .qa_types import (
    _append_qa_log_records,
    _exception_stream,
    _QAExecutionResults,
    _QALogRecord,
    _subprocess_text,
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


_NPM_INSTALL_TIMEOUT_SECONDS = 900
_NPM_TEST_TIMEOUT_SECONDS = 600
_TEST_LOG_TAIL_LINES = 60
_STDERR_TAIL_LINES = 30
_INSTALL_LOG_TAIL_LINES = 80
_TEST_FAILURE_MAX_ITEMS = 8
_TEST_FAILURE_CONTEXT_LINES = 12
_TEST_FAILURE_EXCERPT_CHARS = 500
_LOCAL_NPM_TEST_RUNNERS = frozenset({"jest", "mocha", "vitest"})
_ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_MOCHA_FAILURE_RE = re.compile(r"^\s*(\d+)\)\s+(.+)$")
_JEST_FAILURE_RE = re.compile(r"^\s*(?:[●✕×•]|â—|âœ•|Ã—)\s+(.+)$")
_TAP_FAILURE_RE = re.compile(r"^\s*not ok\b(?:\s+\d+)?\s*-?\s*(.+)?$")
_SUBTEST_RE = re.compile(r"^\s*# Subtest:\s+(.+)$")
_FAIL_LINE_RE = re.compile(r"^\s*FAIL\b(?:\s+(.+))?$")
_EXCEPTION_RE = re.compile(
    r"(AssertionError|TypeError|ReferenceError|SyntaxError|RangeError|Error):"
)
_STACK_NOISE_RE = re.compile(r"^\s*(at\s+.+|\^\s*|[-]{3,}|\s*operator:\s+.+)$")
_PEER_CONFLICT_PATTERNS = ("ERESOLVE", "EOVERRIDE", "peer dep", "peer tree")
_ENGINE_CONFLICT_PATTERNS = ("EBADENGINE",)
_LOG_QUERY_MAX_CHARS = 6_000


@dataclass(frozen=True)
class _QAInstallOutcome:
    """Structured outcome of the deterministic npm install command."""

    ok: bool
    summary: str
    exit_code: int | None
    error_category: str | None
    raw_stdout: str | None
    raw_stderr: str | None
    log_record: _QALogRecord


@dataclass(frozen=True)
class _QATestExecutionOutcome:
    """Structured outcome of all deterministic test-suite commands."""

    ok: bool
    summary: str
    exit_code: int | None
    failure_count: int | None
    raw_stdout: str | None
    raw_stderr: str | None
    log_records: tuple[_QALogRecord, ...]


def _install_error_category(stdout: str, stderr: str, exit_code: int) -> str:
    """Classify a failed npm install for deterministic retry routing."""
    combined = f"{stdout}\n{stderr}".lower()
    if any(marker.lower() in combined for marker in _ENGINE_CONFLICT_PATTERNS):
        return "ENGINE_CONFLICT"
    if any(marker.lower() in combined for marker in _PEER_CONFLICT_PATTERNS):
        return "PEER_CONFLICT"
    return "INSTALL_FAILURE"


_PEER_FROM_RE = re.compile(
    r"peer(?:Optional)?\s+(?P<peer>@?[A-Za-z0-9._\-/]+)@"
    r"(?:\"(?P<quoted_range>[^\"]+)\"|(?P<unquoted_range>[^\s]+))\s+from\s+"
    r"(?P<requester>@?[A-Za-z0-9._\-/]+)@(?P<requester_version>[^\s]+)",
    re.IGNORECASE,
)
_OVERRIDE_CONFLICT_RE = re.compile(
    r"Override\s+for\s+(?P<peer>@?[A-Za-z0-9._\-/]+)@(?P<range>[^\s]+)\s+"
    r"conflicts\s+with\s+direct\s+dependency",
    re.IGNORECASE,
)
_FOUND_VERSION_RE = re.compile(
    r"Found:\s+(?P<package>@?[A-Za-z0-9._\-/]+)@(?P<version>[^\s]+)",
    re.IGNORECASE,
)


def parse_peer_conflict_evidence(stdout: str, stderr: str) -> list[PeerConflictEvidence]:
    """Parse npm ``ERESOLVE``/``EOVERRIDE`` output into stable evidence."""
    text = _strip_ansi(f"{stdout}\n{stderr}")
    observed_versions = {
        match.group("package"): match.group("version").rstrip(".,")
        for match in _FOUND_VERSION_RE.finditer(text)
    }
    records: dict[tuple[str, str, str, str], PeerConflictEvidence] = {}
    for match in _PEER_FROM_RE.finditer(text):
        peer = match.group("peer").rstrip(".,")
        requester = match.group("requester").rstrip(".,")
        required_range = (
            match.group("quoted_range") or match.group("unquoted_range") or ""
        ).rstrip(".,")
        key = (requester, peer, required_range, observed_versions.get(peer, ""))
        records[key] = PeerConflictEvidence(
            requester_package=requester,
            peer_package=peer,
            required_range=required_range,
            observed_version=observed_versions.get(peer),
            evidence=match.group(0)[:2000],
        )
    for match in _OVERRIDE_CONFLICT_RE.finditer(text):
        peer = match.group("peer").rstrip(".,")
        required_range = match.group("range").rstrip(".,")
        key = ("", peer, required_range, observed_versions.get(peer, ""))
        records[key] = PeerConflictEvidence(
            peer_package=peer,
            required_range=required_range,
            observed_version=observed_versions.get(peer),
            evidence=match.group(0)[:2000],
        )
    if not records and any(
        marker.casefold() in text.casefold() for marker in _PEER_CONFLICT_PATTERNS
    ):
        records[("", "", "", "")] = PeerConflictEvidence(
            evidence="\n".join(text.splitlines()[-20:])[:2000]
        )
    return [records[key] for key in sorted(records)]


def _store_install_outcome(results: _QAExecutionResults, outcome: _QAInstallOutcome) -> None:
    """Store install projections and its private raw evidence."""
    results.install = (outcome.ok, outcome.summary)
    results.install_exit_code = outcome.exit_code
    results.install_error_category = outcome.error_category
    results.install_raw_stdout = outcome.raw_stdout
    results.install_raw_stderr = outcome.raw_stderr
    results.peer_conflicts = parse_peer_conflict_evidence(
        outcome.raw_stdout or "", outcome.raw_stderr or ""
    )
    _append_qa_log_records(results, "install", (outcome.log_record,))


def _store_test_outcome(results: _QAExecutionResults, outcome: _QATestExecutionOutcome) -> None:
    """Store test projections and private suite evidence."""
    results.tests = (outcome.ok, outcome.summary)
    results.test_exit_code = outcome.exit_code
    results.test_failure_count = outcome.failure_count
    results.test_raw_stdout = outcome.raw_stdout
    results.test_raw_stderr = outcome.raw_stderr
    _append_qa_log_records(results, "tests", outcome.log_records)


def _run_install(sandbox: DockerSandbox) -> _QAInstallOutcome:
    """Run npm install and retain bounded and raw deterministic evidence.

    Args:
        sandbox: Active QA sandbox in which npm should run.

    Returns:
        A structured install outcome. Raw streams remain private to the
        invocation and are also represented by one immutable log record.
    """
    error: BaseException | None = None
    try:
        result = sandbox.run(
            "npm install --package-lock=true",
            timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
        )
        exit_code = int(getattr(result, "exit_code", 1))
        stdout = _subprocess_text(getattr(result, "stdout", ""))
        stderr = _subprocess_text(getattr(result, "stderr", ""))
    except Exception as exc:  # noqa: BLE001
        error = exc
        raw_exit_code = getattr(exc, "exit_code", getattr(exc, "returncode", None))
        try:
            exit_code = int(raw_exit_code) if raw_exit_code is not None else None
        except (TypeError, ValueError):
            exit_code = None
        stdout = _exception_stream(exc, "stdout")
        stderr = _exception_stream(exc, "stderr")

    if exit_code == 0:
        summary = "npm install succeeded."
        category = None
    else:
        category = _install_error_category(stdout, stderr, exit_code or 1)
        stdout_tail = "\n".join(stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        stderr_tail = "\n".join(stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        summary = (
            f"npm install FAILED (exit {exit_code if exit_code is not None else 'unknown'}).\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )
        if error is not None:
            summary += f"\nerror: {error}"

    return _QAInstallOutcome(
        ok=exit_code == 0,
        summary=summary,
        exit_code=exit_code,
        error_category=category,
        raw_stdout=stdout,
        raw_stderr=stderr,
        log_record=_QALogRecord(
            phase="install",
            label="npm install",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            error=str(error) if error is not None else None,
        ),
    )


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
    """Return bounded raw output when no structured failure block is detected."""
    stdout_tail = "\n".join(stdout.splitlines()[-_TEST_LOG_TAIL_LINES:])
    stderr_tail = "\n".join(stderr.splitlines()[-_STDERR_TAIL_LINES:])
    return (
        f"npm test FAILED (exit {exit_code}).\n"
        f"stdout tail:\n{stdout_tail}\n"
        f"stderr tail:\n{stderr_tail}"
    )


def _format_failure_summary(exit_code: int, blocks: list[_TestFailureBlock]) -> str:
    """Format parsed test failures into a compact bounded QA summary."""
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
    r"(?:error|exception|failed|failure|fatal|eresolve|eoverride|ebadengine|"
    r"timeout|timed\s+out|diagnostic|not\s+a\s+function|undefined|cannot|"
    r"invalid|missing|required\s+option|not\s+exported|cannot\s+find)",
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
        elif token in (relative_path, test_file):
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
    """Construct a command that executes only the requested test file.

    ``npm_invocation`` is the package script selected from the workspace
    metadata. Direct runner commands use ``npx --no-install`` so they resolve
    the workspace-local executable. ``package_cwd`` keeps the invocation in
    the package directory whose test context selected the command.
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
    return _test_normalization._normalize_mocha_json_output(
        stdout,
        stderr,
        suite_name=suite_name,
        extract_json_object=_extract_json_object,
        truncate_summary_text=_truncate_summary_text,
    )


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
    return _test_normalization._normalize_vitest_json_output(
        stdout,
        stderr,
        suite_name=suite_name,
        extract_json_object=_extract_json_object,
        iter_vitest_failures=_iter_vitest_failures,
        truncate_summary_text=_truncate_summary_text,
    )


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
    status_header = f"npm test FAILED (exit {exit_code})." if failed_suites else "npm test passed."
    lines = [
        status_header,
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
) -> _QATestExecutionOutcome:
    """Run every detected child suite and aggregate raw and parsed evidence."""
    suite_results: list[_NormalizedSuiteResult] = []
    records: list[_QALogRecord] = []
    for plan in plans:
        command = _structured_command_for_plan(plan)
        try:
            result = sandbox.run(command, timeout=_NPM_TEST_TIMEOUT_SECONDS)
            exit_code = int(getattr(result, "exit_code", 1))
            stdout = _subprocess_text(getattr(result, "stdout", ""))
            stderr = _subprocess_text(getattr(result, "stderr", ""))
            suite_result = _normalize_suite_result(plan, result, command=command)
            error = None
        except Exception as exc:  # noqa: BLE001
            exit_code = None
            stdout = _exception_stream(exc, "stdout")
            stderr = _exception_stream(exc, "stderr")
            suite_result = _NormalizedSuiteResult(
                name=plan.name,
                runner=plan.runner,
                command=command,
                exit_code=1,
                failed_tests=[],
                diagnostics=[],
                fallback_summary=f"tests:{plan.name} failed before producing a result: {exc}",
            )
            error = str(exc)
        suite_results.append(suite_result)
        records.append(
            _QALogRecord(
                phase="tests",
                label=f"tests:{plan.name}",
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                error=error,
            )
        )

    failed_suites = [suite for suite in suite_results if suite.exit_code != 0]
    failure_count: int | None = sum(len(suite.failed_tests) for suite in suite_results)
    if any(not suite.failed_tests and suite.exit_code != 0 for suite in failed_suites):
        failure_count = None
    ok = not failed_suites
    exit_code = (
        next(
            (record.exit_code for record in reversed(records) if record.exit_code not in (None, 0)),
            0,
        )
        if ok
        else next(
            (
                record.exit_code
                for record in reversed(records)
                if record.exit_code is not None and record.exit_code != 0
            ),
            None,
        )
    )

    def combined_stream(stream_name: str) -> str:
        values = [getattr(record, stream_name) for record in records]
        if len(values) == 1:
            return values[0]
        return "\n\n".join(
            f"[{record.label} {stream_name}]\n{getattr(record, stream_name)}"
            for record in records
            if getattr(record, stream_name)
        )

    return _QATestExecutionOutcome(
        ok=ok,
        summary=_format_normalized_test_summary(suite_results),
        exit_code=exit_code,
        failure_count=failure_count if not ok or records else 0,
        raw_stdout=combined_stream("stdout"),
        raw_stderr=combined_stream("stderr"),
        log_records=tuple(records),
    )


def _run_unit_tests(sandbox: DockerSandbox) -> _QATestExecutionOutcome:
    """Run workspace tests and retain raw, suite, and normalized evidence."""
    plans = _detect_test_suite_plans(sandbox)
    if plans and any(plan.runner != "npm_text_fallback" for plan in plans):
        return _run_detected_test_suites(sandbox, plans)

    error: BaseException | None = None
    try:
        result = sandbox.run("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)
        exit_code = int(getattr(result, "exit_code", 1))
        stdout = _subprocess_text(getattr(result, "stdout", ""))
        stderr = _subprocess_text(getattr(result, "stderr", ""))
    except Exception as exc:  # noqa: BLE001
        error = exc
        raw_exit_code = getattr(exc, "exit_code", getattr(exc, "returncode", None))
        try:
            exit_code = int(raw_exit_code) if raw_exit_code is not None else None
        except (TypeError, ValueError):
            exit_code = None
        stdout = _exception_stream(exc, "stdout")
        stderr = _exception_stream(exc, "stderr")

    if exit_code == 0:
        summary = "npm test passed."
        failure_count = 0
    else:
        summary = _summarize_failed_test_output(exit_code or 1, stdout, stderr)
        evidence = extract_qa_failure_evidence(exit_code or 1, stdout, stderr)
        failure_count = len(evidence.failed_tests) if evidence.failed_tests else None
        if error is not None:
            summary += f"\nerror: {error}"

    record = _QALogRecord(
        phase="tests",
        label="npm test",
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error=str(error) if error is not None else None,
    )
    return _QATestExecutionOutcome(
        ok=exit_code == 0,
        summary=summary,
        exit_code=exit_code,
        failure_count=failure_count,
        raw_stdout=stdout,
        raw_stderr=stderr,
        log_records=(record,),
    )
