"""Tests for deterministic evaluation comparison formatting."""

from __future__ import annotations

from remediation_engine.evals.comparison import format_run_comparison, format_run_list


def _run(run_id: str, tag: str, pass_rate: float) -> dict[str, object]:
    """Build the minimal persisted run shape used by formatter tests."""
    return {
        "run_id": run_id,
        "timestamp": "2026-08-26T12:00:00",
        "tag": tag,
        "suite_name": "tests/evals",
        "pass_rate": pass_rate,
        "metadata": {"git_branch": "feat/eval"},
        "is_live": False,
    }


def test_format_run_comparison_renders_summary_and_changed_rows() -> None:
    """Render pass-rate deltas, regressions, improvements, and escaped names."""
    comparison = {
        "run_a": _run("run-a", "baseline", 100.0),
        "run_b": _run("run-b", "post-change", 50.0),
        "pass_rate_delta": -50.0,
        "total_regressions": 1,
        "total_fixes": 1,
        "regressions": [
            {
                "test_name": "test|regressed\ncase",
                "suite": "triage",
                "status_a": "PASSED",
                "status_b": "FAILED",
                "score_delta": -0.4,
            }
        ],
        "fixes": [
            {
                "test_name": "test_fixed",
                "suite": "report",
                "status_a": "FAILED",
                "status_b": "PASSED",
                "score_delta": 0.3,
            }
        ],
        "comparisons": [],
    }

    output = format_run_comparison(comparison)

    assert "Baseline:" in output
    assert "Candidate:" in output
    assert "(-50.0 pp)" in output
    assert "Regressions: 1 | Improvements: 1" in output
    assert "test\\|regressed case" in output
    assert "IMPROVEMENT" in output


def test_format_run_comparison_reports_no_changes() -> None:
    """Keep an empty comparison concise and explicit."""
    output = format_run_comparison(
        {
            "run_a": _run("run-a", "baseline", 100.0),
            "run_b": _run("run-b", "post-change", 100.0),
            "pass_rate_delta": 0.0,
            "total_regressions": 0,
            "total_fixes": 0,
            "comparisons": [],
        }
    )

    assert "No regressions detected" in output
    assert "Status changes:" not in output


def test_format_run_list_includes_tags_and_git_branch() -> None:
    """Render the fields needed to identify a persisted run from the CLI."""
    output = format_run_list([_run("run-a", "baseline", 100.0)])

    assert "Run ID" in output
    assert "baseline" in output
    assert "feat/eval" in output
    assert "100.0%" in output


def test_format_run_list_handles_empty_database() -> None:
    """Report an empty database without raising."""
    assert format_run_list([]) == "No evaluation runs recorded."
