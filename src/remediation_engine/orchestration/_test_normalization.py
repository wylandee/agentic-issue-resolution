"""Structured JSON test-result normalization for the QA parser facade."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_TEST_FAILURE_EXCERPT_CHARS = 500


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


def _truncate_summary_text(value: str, limit: int = _TEST_FAILURE_EXCERPT_CHARS) -> str:
    """Bound one message or excerpt for log-query output."""
    value = (value or "").strip()
    if len(value) > limit:
        return value[:limit].rstrip() + "... (truncated)"
    return value


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
    extract_json_object: Any = _extract_json_object,
    truncate_summary_text: Any = _truncate_summary_text,
) -> tuple[list[_NormalizedTestFailure], list[_NormalizedTestDiagnostic]]:
    """Normalize Mocha JSON reporter output."""
    report = extract_json_object(stdout) or extract_json_object(stderr)
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
                message=truncate_summary_text(str(error.get("message") or "")),
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
    extract_json_object: Any = _extract_json_object,
    iter_vitest_failures: Any = _iter_vitest_failures,
    truncate_summary_text: Any = _truncate_summary_text,
) -> tuple[list[_NormalizedTestFailure], list[_NormalizedTestDiagnostic]]:
    """Normalize Vitest JSON reporter output when available."""
    report = extract_json_object(stdout) or extract_json_object(stderr)
    if not report:
        return [], []
    failures: list[_NormalizedTestFailure] = []
    for item in iter_vitest_failures(report):
        errors = item.get("errors") if isinstance(item.get("errors"), list) else []
        first_error = errors[0] if errors and isinstance(errors[0], dict) else {}
        failures.append(
            _NormalizedTestFailure(
                name=str(item.get("fullName") or item.get("name") or "vitest failure"),
                message=truncate_summary_text(str(first_error.get("message") or "")),
                failure_type=str(first_error.get("name") or ""),
                suite=suite_name,
            )
        )
    return failures, []
