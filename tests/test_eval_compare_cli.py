"""Tests for the standalone evaluation comparison CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from remediation_engine.evals.db import EvalDatabase
from remediation_engine.evals.models import EvalRunRecord, EvalTestCaseRecord, MetricRecord

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "eval_compare.py"


def _load_cli():
    """Load the script module without requiring scripts to be a package."""
    spec = importlib.util.spec_from_file_location("eval_compare_script", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_run(
    database: EvalDatabase,
    run_id: str,
    timestamp: str,
    tag: str,
    status: str,
) -> None:
    """Persist one small run for CLI tests."""
    database.save_run(
        EvalRunRecord(
            run_id=run_id,
            timestamp=timestamp,
            tag=tag,
            suite_name="tests/evals",
            total_tests=1,
            passed_tests=int(status == "PASSED"),
            failed_tests=int(status == "FAILED"),
            test_cases=[
                EvalTestCaseRecord(
                    test_name="test_shared_case",
                    suite="triage",
                    status=status,
                    metrics=[MetricRecord(metric_name="Accuracy", score=0.5)],
                )
            ],
        )
    )


@pytest.fixture
def comparison_db(tmp_path: Path) -> Path:
    """Create an isolated database containing baseline and candidate runs."""
    db_path = tmp_path / "evals.db"
    database = EvalDatabase(db_path=db_path)
    _save_run(database, "run-a", "2026-08-25T12:00:00", "baseline", "PASSED")
    _save_run(database, "run-b", "2026-08-26T12:00:00", "candidate", "FAILED")
    return db_path


def test_cli_lists_runs_with_tags(comparison_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The list mode exposes tags and identifying run metadata."""
    cli = _load_cli()

    assert cli.main(["--db-path", str(comparison_db), "--list"]) == 0
    output = capsys.readouterr().out
    assert "baseline" in output
    assert "candidate" in output
    assert "run-a" in output
    assert "run-b" in output


def test_cli_compares_tags_and_run_ids(
    comparison_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Explicit tags and exact run IDs both resolve to a comparison."""
    cli = _load_cli()

    assert (
        cli.main(
            [
                "--db-path",
                str(comparison_db),
                "--run-a",
                "baseline",
                "--run-b",
                "candidate",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Regressions: 1" in output
    assert "(-100.0 pp)" in output

    assert (
        cli.main(
            [
                "--db-path",
                str(comparison_db),
                "--run-a",
                "run-a",
                "--run-b",
                "run-b",
            ]
        )
        == 0
    )


def test_cli_latest_compares_newest_two_runs(
    comparison_db: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The latest mode chooses the newest candidate and its predecessor."""
    cli = _load_cli()

    assert cli.main(["--db-path", str(comparison_db), "--latest"]) == 0
    output = capsys.readouterr().out
    assert "Baseline:" in output
    assert "Candidate:" in output
    assert "Regressions: 1" in output


def test_cli_reports_missing_references(
    comparison_db: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Unknown run references return an operational error instead of a traceback."""
    cli = _load_cli()

    assert (
        cli.main(
            [
                "--db-path",
                str(comparison_db),
                "--run-a",
                "missing",
                "--run-b",
                "candidate",
            ]
        )
        == 1
    )
    assert "was not found" in capsys.readouterr().err


def test_cli_rejects_incomplete_comparison_arguments() -> None:
    """Require a candidate reference when explicit baseline mode is selected."""
    cli = _load_cli()

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--run-a", "baseline"])
    assert exc_info.value.code == 2
