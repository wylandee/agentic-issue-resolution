"""Unit tests for the evaluation SQLite persistence layer and UI utilities."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from remediation_engine.evals.db import EvalDatabase
from remediation_engine.evals.models import (
    EvalRunRecord,
    EvalTestCaseRecord,
    MetricRecord,
)
from remediation_engine.evals.runner import create_sample_run


@pytest.fixture
def temp_db(tmp_path: Path) -> EvalDatabase:
    """Provide a clean isolated EvalDatabase instance using a temporary file."""
    db_file = tmp_path / "test_evals.db"
    return EvalDatabase(db_path=db_file)


def test_init_db_creates_tables(temp_db: EvalDatabase) -> None:
    """Verify that all required tables and indexes are created on initialization."""
    with temp_db._get_connection() as conn:
        cursor = conn.cursor()
        tables = [
            row["name"]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        ]
        assert "eval_runs" in tables
        assert "eval_test_cases" in tables
        assert "eval_metrics" in tables


def test_save_and_get_runs(temp_db: EvalDatabase) -> None:
    """Ensure evaluation runs can be saved and retrieved with computed summaries."""
    run = EvalRunRecord(
        run_id="run_001",
        timestamp=datetime.datetime.now().isoformat(),
        suite_name="test_report_eval.py",
        judge_model="gpt-4o",
        is_live=False,
        total_tests=2,
        passed_tests=2,
        failed_tests=0,
        skipped_tests=0,
        duration_seconds=3.5,
        total_cost=0.005,
        metadata={"git_branch": "feat/add-eval-ui"},
        test_cases=[
            EvalTestCaseRecord(
                case_id="case_01",
                test_name="test_report_coverage [case_01]",
                suite="report",
                status="PASSED",
                input_text="Sample input prompt",
                actual_output="Sample actual narrative",
                expected_output="Sample expected narrative",
                context_text="Ground truth context document",
                latency_seconds=1.75,
                cost=0.0025,
                metrics=[
                    MetricRecord(
                        metric_name="Finding Coverage & Accuracy",
                        score=0.95,
                        threshold=0.70,
                        success=True,
                        reason="All findings covered accurately.",
                        evaluation_model="gpt-4o",
                    )
                ],
            ),
            EvalTestCaseRecord(
                case_id="case_02",
                test_name="test_report_negative_constraints [case_02]",
                suite="report",
                status="PASSED",
                input_text="Constraint input prompt",
                actual_output="Clean output without headings",
                expected_output="Clean expected output",
                latency_seconds=1.75,
                cost=0.0025,
                metrics=[
                    MetricRecord(
                        metric_name="Report Constraint Adherence",
                        score=0.90,
                        threshold=0.70,
                        success=True,
                        reason="No headings or prescriptive advice used.",
                        evaluation_model="gpt-4o",
                    )
                ],
            ),
        ],
    )

    saved_id = temp_db.save_run(run)
    assert saved_id == "run_001"

    runs = temp_db.get_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run_001"
    assert runs[0]["pass_rate"] == 100.0
    assert runs[0]["total_tests"] == 2
    assert runs[0]["passed_tests"] == 2
    assert runs[0]["metadata"]["git_branch"] == "feat/add-eval-ui"

    single_run = temp_db.get_run("run_001")
    assert single_run is not None
    assert single_run["suite_name"] == "test_report_eval.py"


def test_get_test_cases_filtering_and_search(temp_db: EvalDatabase) -> None:
    """Verify test cases filtering by status, suite, and full-text search."""
    run = EvalRunRecord(
        run_id="run_002",
        timestamp=datetime.datetime.now().isoformat(),
        suite_name="mixed_evals",
        judge_model="gpt-4o",
        total_tests=3,
        passed_tests=2,
        failed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                case_id="cve_001",
                test_name="test_triage_cve [cve_001]",
                suite="triage",
                status="PASSED",
                input_text="CVE-2023-26159 follow-redirects prototype pollution",
                actual_output="ACTIONABLE with DIRECT_UPDATE",
                metrics=[
                    MetricRecord(
                        metric_name="Triage Accuracy",
                        score=1.0,
                        success=True,
                    )
                ],
            ),
            EvalTestCaseRecord(
                case_id="cve_002",
                test_name="test_triage_cve [cve_002]",
                suite="triage",
                status="FAILED",
                input_text="CVE-2022-0001 lodash vulnerability",
                actual_output="FALSE_POSITIVE",
                expected_output="ACTIONABLE",
                metrics=[
                    MetricRecord(
                        metric_name="Triage Accuracy",
                        score=0.0,
                        success=False,
                        reason="Incorrectly classified actionable issue as false positive.",
                    )
                ],
            ),
            EvalTestCaseRecord(
                case_id="rep_001",
                test_name="test_report_node [rep_001]",
                suite="report",
                status="PASSED",
                input_text="Report deterministic evidence",
                actual_output="Remediation successfully finished",
                metrics=[
                    MetricRecord(
                        metric_name="Finding Coverage",
                        score=0.85,
                        success=True,
                    )
                ],
            ),
        ],
    )
    temp_db.save_run(run)

    # All test cases
    all_cases = temp_db.get_test_cases(run_id="run_002")
    assert len(all_cases) == 3

    # Filter by status: FAILED
    failed_cases = temp_db.get_test_cases(run_id="run_002", status_filter="FAILED")
    assert len(failed_cases) == 1
    assert failed_cases[0]["case_id"] == "cve_002"
    assert len(failed_cases[0]["metrics"]) == 1
    assert not failed_cases[0]["metrics"][0]["success"]

    # Filter by suite: report
    report_cases = temp_db.get_test_cases(run_id="run_002", suite_filter="report")
    assert len(report_cases) == 1
    assert report_cases[0]["case_id"] == "rep_001"

    # Search query
    searched = temp_db.get_test_cases(search_query="follow-redirects")
    assert len(searched) == 1
    assert searched[0]["case_id"] == "cve_001"


def test_metric_trends_aggregation(temp_db: EvalDatabase) -> None:
    """Verify aggregation of metric scores across historical runs."""
    # Run 1
    r1 = EvalRunRecord(
        run_id="r1",
        timestamp="2026-08-25T10:00:00",
        suite_name="report",
        total_tests=1,
        passed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                test_name="t1",
                suite="report",
                metrics=[MetricRecord(metric_name="Faithfulness", score=0.80, success=True)],
            )
        ],
    )
    # Run 2
    r2 = EvalRunRecord(
        run_id="r2",
        timestamp="2026-08-26T10:00:00",
        suite_name="report",
        total_tests=1,
        passed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                test_name="t1",
                suite="report",
                metrics=[MetricRecord(metric_name="Faithfulness", score=0.90, success=True)],
            )
        ],
    )
    temp_db.save_run(r1)
    temp_db.save_run(r2)

    trends = temp_db.get_metric_trends(metric_name="Faithfulness")
    assert len(trends) == 2
    assert trends[0]["avg_score"] == 0.80
    assert trends[1]["avg_score"] == 0.90


def test_run_comparison_detects_regressions_and_fixes(temp_db: EvalDatabase) -> None:
    """Ensure run comparison detects regressions, fixes, and score deltas."""
    # Baseline Run A
    run_a = EvalRunRecord(
        run_id="run_a",
        timestamp="2026-08-25T12:00:00",
        suite_name="suite",
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                test_name="test_1",
                suite="report",
                status="PASSED",
                metrics=[MetricRecord(metric_name="Coverage", score=0.85, success=True)],
            ),
            EvalTestCaseRecord(
                test_name="test_2",
                suite="triage",
                status="FAILED",
                metrics=[MetricRecord(metric_name="Accuracy", score=0.40, success=False)],
            ),
        ],
    )

    # Candidate Run B (test_1 regressed, test_2 fixed)
    run_b = EvalRunRecord(
        run_id="run_b",
        timestamp="2026-08-26T12:00:00",
        suite_name="suite",
        total_tests=2,
        passed_tests=1,
        failed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                test_name="test_1",
                suite="report",
                status="FAILED",
                metrics=[MetricRecord(metric_name="Coverage", score=0.60, success=False)],
            ),
            EvalTestCaseRecord(
                test_name="test_2",
                suite="triage",
                status="PASSED",
                metrics=[MetricRecord(metric_name="Accuracy", score=0.95, success=True)],
            ),
        ],
    )

    temp_db.save_run(run_a)
    temp_db.save_run(run_b)

    comp = temp_db.get_run_comparison("run_a", "run_b")
    assert comp["total_regressions"] == 1
    assert comp["regressions"][0]["test_name"] == "test_1"
    assert comp["total_fixes"] == 1
    assert comp["fixes"][0]["test_name"] == "test_2"


def test_delete_run_cascades(temp_db: EvalDatabase) -> None:
    """Verify that deleting a run removes its child test cases and metrics."""
    run = EvalRunRecord(
        run_id="run_to_delete",
        timestamp=datetime.datetime.now().isoformat(),
        suite_name="temp",
        total_tests=1,
        passed_tests=1,
        test_cases=[
            EvalTestCaseRecord(
                test_name="temp_test",
                status="PASSED",
                metrics=[MetricRecord(metric_name="M1", score=1.0, success=True)],
            )
        ],
    )
    temp_db.save_run(run)
    assert len(temp_db.get_test_cases(run_id="run_to_delete")) == 1

    deleted = temp_db.delete_run("run_to_delete")
    assert deleted
    assert temp_db.get_run("run_to_delete") is None
    assert len(temp_db.get_test_cases(run_id="run_to_delete")) == 0


def test_create_sample_run(temp_db: EvalDatabase) -> None:
    """Ensure sample demo run generator populates valid structured test cases."""
    sample = create_sample_run(db=temp_db)
    assert sample.total_tests == 4
    assert sample.passed_tests >= 3

    runs = temp_db.get_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == sample.run_id

    cases = temp_db.get_test_cases(run_id=sample.run_id)
    assert len(cases) == 4
    for c in cases:
        assert c["input_text"]
        assert c["actual_output"]
        assert len(c["metrics"]) > 0
