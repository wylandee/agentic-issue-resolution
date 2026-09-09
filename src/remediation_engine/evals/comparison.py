"""Formatting helpers for evaluation run comparison output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _cell(value: Any, default: str = "-") -> str:
    """Convert a value to a single safe table cell."""
    if value is None or value == "":
        return default
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _percent(value: Any) -> str:
    """Format a pass-rate value as a percentage."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "-"


def format_run_label(run: Mapping[str, Any]) -> str:
    """Return a compact, human-readable label for one evaluation run."""
    timestamp = _cell(run.get("timestamp"), "unknown")[:19].replace("T", " ")
    suite = _cell(run.get("suite_name"), "unknown")
    tag = _cell(run.get("tag"), "untagged")
    run_id = _cell(run.get("run_id"), "unknown")
    return f"{timestamp} | {suite} | {tag} ({_percent(run.get('pass_rate'))} pass) | {run_id}"


def _comparison_rows(comparison: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Return changed comparison rows with stable labels and no duplicates."""
    rows: list[tuple[str, Mapping[str, Any]]] = []
    changed_names: set[str] = set()

    for kind, key in (("REGRESSION", "regressions"), ("IMPROVEMENT", "fixes")):
        for row in comparison.get(key, []) or []:
            name = str(row.get("test_name", ""))
            rows.append((kind, row))
            changed_names.add(name)

    for row in comparison.get("comparisons", []) or []:
        name = str(row.get("test_name", ""))
        if name not in changed_names and row.get("status_a") != row.get("status_b"):
            rows.append(("OTHER", row))

    return rows


def _format_change_table(rows: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """Render changed comparison rows as an ASCII table."""
    headers = ("Change", "Suite", "Test", "Baseline", "Candidate", "Score delta")
    table_rows = [
        (
            kind,
            _cell(row.get("suite")),
            _cell(row.get("test_name")),
            _cell(row.get("status_a")),
            _cell(row.get("status_b")),
            _cell(row.get("score_delta")),
        )
        for kind, row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def render_row(row: Sequence[str]) -> str:
        return (
            "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return [render_row(headers), separator, *(render_row(row) for row in table_rows)]


def format_run_comparison(comparison: Mapping[str, Any]) -> str:
    """Format a persisted run comparison for terminal output.

    Args:
        comparison: Result returned by ``EvalDatabase.get_run_comparison``.

    Returns:
        A deterministic human-readable comparison report.
    """
    if comparison.get("error"):
        return f"Evaluation comparison error: {_cell(comparison['error'])}"

    run_a = comparison.get("run_a") or {}
    run_b = comparison.get("run_b") or {}
    pass_rate_delta = comparison.get("pass_rate_delta")
    if pass_rate_delta is None:
        try:
            pass_rate_delta = float(run_b.get("pass_rate", 0.0)) - float(
                run_a.get("pass_rate", 0.0)
            )
        except (TypeError, ValueError):
            pass_rate_delta = 0.0

    regressions = int(comparison.get("total_regressions", 0) or 0)
    improvements = int(comparison.get("total_fixes", 0) or 0)
    rows = _comparison_rows(comparison)
    lines = [
        "Evaluation run comparison",
        f"Baseline:  {format_run_label(run_a)}",
        f"Candidate: {format_run_label(run_b)}",
        f"Pass rate: {_percent(run_a.get('pass_rate'))} -> {_percent(run_b.get('pass_rate'))} "
        f"({float(pass_rate_delta):+.1f} pp)",
        f"Regressions: {regressions} | Improvements: {improvements}",
    ]

    if not rows:
        lines.append("No regressions detected; no status improvements detected.")
        return "\n".join(lines)

    lines.extend(["", "Status changes:", *_format_change_table(rows)])
    return "\n".join(lines)


def format_run_list(runs: Sequence[Mapping[str, Any]]) -> str:
    """Format persisted evaluation runs for the comparison CLI."""
    if not runs:
        return "No evaluation runs recorded."

    headers = ("Timestamp", "Run ID", "Tag", "Suite", "Mode", "Git branch", "Pass rate")
    rows: list[tuple[str, ...]] = []
    for run in runs:
        metadata = run.get("metadata")
        metadata_dict = metadata if isinstance(metadata, Mapping) else {}
        rows.append(
            (
                _cell(run.get("timestamp"), "unknown")[:19].replace("T", " "),
                _cell(run.get("run_id"), "unknown"),
                _cell(run.get("tag"), "untagged"),
                _cell(run.get("suite_name"), "unknown"),
                "live" if run.get("is_live") else "offline",
                _cell(metadata_dict.get("git_branch"), "unknown"),
                _percent(run.get("pass_rate")),
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render_row(row: Sequence[str]) -> str:
        return (
            "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        )

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in rows)])
