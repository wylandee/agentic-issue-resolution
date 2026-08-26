"""SQLite persistence and querying layer for DeepEval evaluation results."""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .models import EvalRunRecord

_DEFAULT_DB_DIR = Path(__file__).resolve().parents[3] / "data" / "evals"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "eval_results.db"


def get_default_db_path() -> Path:
    """Return the resolved database file path from environment or project default."""
    env_path = os.environ.get("EVAL_DB_PATH", "").strip()
    if env_path:
        return Path(env_path).resolve()
    return _DEFAULT_DB_PATH


class EvalDatabase:
    """Thread-safe SQLite evaluation database manager."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialize the evaluation database manager.

        Args:
            db_path: Path to the SQLite database file. Defaults to data/evals/eval_results.db.
        """
        self.db_path = Path(db_path).resolve() if db_path else get_default_db_path()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection with foreign keys and WAL enabled."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        with contextlib.suppress(sqlite3.OperationalError):
            conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        """Initialize the evaluation database tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    suite_name TEXT NOT NULL,
                    judge_model TEXT NOT NULL,
                    is_live INTEGER NOT NULL DEFAULT 0,
                    total_tests INTEGER NOT NULL DEFAULT 0,
                    passed_tests INTEGER NOT NULL DEFAULT 0,
                    failed_tests INTEGER NOT NULL DEFAULT 0,
                    skipped_tests INTEGER NOT NULL DEFAULT 0,
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    total_cost REAL NOT NULL DEFAULT 0.0,
                    metadata_json TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS eval_test_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    case_id TEXT,
                    test_name TEXT NOT NULL,
                    suite TEXT NOT NULL DEFAULT 'general',
                    status TEXT NOT NULL DEFAULT 'PASSED',
                    input_text TEXT DEFAULT '',
                    actual_output TEXT DEFAULT '',
                    expected_output TEXT,
                    context_text TEXT,
                    retrieval_context TEXT,
                    latency_seconds REAL DEFAULT 0.0,
                    cost REAL DEFAULT 0.0,
                    error_message TEXT,
                    additional_metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS eval_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_case_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    threshold REAL NOT NULL DEFAULT 0.70,
                    success INTEGER NOT NULL DEFAULT 1,
                    reason TEXT,
                    evaluation_model TEXT,
                    verbose_logs TEXT,
                    FOREIGN KEY (test_case_id) REFERENCES eval_test_cases(id) ON DELETE CASCADE,
                    FOREIGN KEY (run_id) REFERENCES eval_runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON eval_runs(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_test_cases_run_id ON eval_test_cases(run_id);
                CREATE INDEX IF NOT EXISTS idx_test_cases_suite_status ON eval_test_cases(suite, status);
                CREATE INDEX IF NOT EXISTS idx_metrics_case_id ON eval_metrics(test_case_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_run_metric ON eval_metrics(run_id, metric_name);
                """
            )

    def save_run(self, run: EvalRunRecord) -> str:
        """Persist an evaluation run and all associated test cases and metrics atomically.

        Args:
            run: Complete EvalRunRecord containing test cases and metric outcomes.

        Returns:
            The saved run_id.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert or replace run record
            cursor.execute(
                """
                INSERT INTO eval_runs (
                    run_id, timestamp, suite_name, judge_model, is_live,
                    total_tests, passed_tests, failed_tests, skipped_tests,
                    duration_seconds, total_cost, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    suite_name=excluded.suite_name,
                    judge_model=excluded.judge_model,
                    is_live=excluded.is_live,
                    total_tests=excluded.total_tests,
                    passed_tests=excluded.passed_tests,
                    failed_tests=excluded.failed_tests,
                    skipped_tests=excluded.skipped_tests,
                    duration_seconds=excluded.duration_seconds,
                    total_cost=excluded.total_cost,
                    metadata_json=excluded.metadata_json;
                """,
                (
                    run.run_id,
                    run.timestamp,
                    run.suite_name,
                    run.judge_model,
                    int(run.is_live),
                    run.total_tests,
                    run.passed_tests,
                    run.failed_tests,
                    run.skipped_tests,
                    run.duration_seconds,
                    run.total_cost,
                    json.dumps(run.metadata),
                ),
            )

            # Delete any existing test cases for this run_id to avoid duplication on overwrite
            cursor.execute("DELETE FROM eval_test_cases WHERE run_id = ?", (run.run_id,))

            for tc in run.test_cases:
                cursor.execute(
                    """
                    INSERT INTO eval_test_cases (
                        run_id, case_id, test_name, suite, status,
                        input_text, actual_output, expected_output, context_text,
                        retrieval_context, latency_seconds, cost, error_message,
                        additional_metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.run_id,
                        tc.case_id,
                        tc.test_name,
                        tc.suite,
                        tc.status,
                        tc.input_text or "",
                        tc.actual_output or "",
                        tc.expected_output,
                        tc.context_text,
                        tc.retrieval_context,
                        tc.latency_seconds,
                        tc.cost,
                        tc.error_message,
                        json.dumps(tc.additional_metadata),
                    ),
                )
                test_case_id = cursor.lastrowid

                for metric in tc.metrics:
                    cursor.execute(
                        """
                        INSERT INTO eval_metrics (
                            test_case_id, run_id, metric_name, score, threshold,
                            success, reason, evaluation_model, verbose_logs
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            test_case_id,
                            run.run_id,
                            metric.metric_name,
                            metric.score,
                            metric.threshold,
                            int(metric.success),
                            metric.reason,
                            metric.evaluation_model or run.judge_model,
                            metric.verbose_logs,
                        ),
                    )

            conn.commit()
            return run.run_id

    def get_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve recent evaluation runs with summary calculations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            rows = cursor.execute(
                """
                SELECT
                    id, run_id, timestamp, suite_name, judge_model, is_live,
                    total_tests, passed_tests, failed_tests, skipped_tests,
                    duration_seconds, total_cost, metadata_json
                FROM eval_runs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            runs: list[dict[str, Any]] = []
            for row in rows:
                r = dict(row)
                r["is_live"] = bool(r["is_live"])
                r["metadata"] = json.loads(r.pop("metadata_json") or "{}")
                total = r["total_tests"]
                r["pass_rate"] = round((r["passed_tests"] / total * 100), 1) if total > 0 else 0.0
                runs.append(r)
            return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Retrieve full details for a single evaluation run."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            row = cursor.execute(
                """
                SELECT
                    id, run_id, timestamp, suite_name, judge_model, is_live,
                    total_tests, passed_tests, failed_tests, skipped_tests,
                    duration_seconds, total_cost, metadata_json
                FROM eval_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

            if not row:
                return None

            r = dict(row)
            r["is_live"] = bool(r["is_live"])
            r["metadata"] = json.loads(r.pop("metadata_json") or "{}")
            total = r["total_tests"]
            r["pass_rate"] = round((r["passed_tests"] / total * 100), 1) if total > 0 else 0.0
            return r

    def get_test_cases(
        self,
        run_id: str | None = None,
        status_filter: str | None = None,
        suite_filter: str | None = None,
        search_query: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Retrieve test cases with filters and attached metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT
                    tc.id, tc.run_id, tc.case_id, tc.test_name, tc.suite, tc.status,
                    tc.input_text, tc.actual_output, tc.expected_output, tc.context_text,
                    tc.retrieval_context, tc.latency_seconds, tc.cost, tc.error_message,
                    tc.additional_metadata_json, r.timestamp, r.judge_model
                FROM eval_test_cases tc
                JOIN eval_runs r ON tc.run_id = r.run_id
                WHERE 1=1
            """
            params: list[Any] = []

            if run_id:
                query += " AND tc.run_id = ?"
                params.append(run_id)
            if status_filter and status_filter.upper() != "ALL":
                query += " AND UPPER(tc.status) = ?"
                params.append(status_filter.upper())
            if suite_filter and suite_filter.lower() != "all":
                query += " AND LOWER(tc.suite) = ?"
                params.append(suite_filter.lower())
            if search_query:
                query += """ AND (
                    tc.test_name LIKE ? OR
                    tc.case_id LIKE ? OR
                    tc.input_text LIKE ? OR
                    tc.actual_output LIKE ? OR
                    tc.expected_output LIKE ?
                )"""
                pattern = f"%{search_query}%"
                params.extend([pattern, pattern, pattern, pattern, pattern])

            query += " ORDER BY tc.id ASC LIMIT ?"
            params.append(limit)

            tc_rows = cursor.execute(query, params).fetchall()
            test_cases: list[dict[str, Any]] = []

            for row in tc_rows:
                tc = dict(row)
                tc["additional_metadata"] = json.loads(tc.pop("additional_metadata_json") or "{}")

                # Fetch associated metrics
                metric_rows = cursor.execute(
                    """
                    SELECT id, metric_name, score, threshold, success, reason, evaluation_model, verbose_logs
                    FROM eval_metrics
                    WHERE test_case_id = ?
                    ORDER BY id ASC
                    """,
                    (tc["id"],),
                ).fetchall()

                metrics: list[dict[str, Any]] = []
                for m in metric_rows:
                    md = dict(m)
                    md["success"] = bool(md["success"])
                    metrics.append(md)

                tc["metrics"] = metrics
                test_cases.append(tc)

            return test_cases

    def get_metric_trends(self, metric_name: str | None = None) -> list[dict[str, Any]]:
        """Retrieve aggregated metric performance trends across historical runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = """
                SELECT
                    r.run_id,
                    r.timestamp,
                    r.suite_name,
                    m.metric_name,
                    AVG(m.score) as avg_score,
                    MIN(m.score) as min_score,
                    MAX(m.score) as max_score,
                    AVG(m.threshold) as avg_threshold,
                    COUNT(m.id) as total_evaluations,
                    SUM(m.success) as passed_evaluations
                FROM eval_metrics m
                JOIN eval_runs r ON m.run_id = r.run_id
            """
            params: list[Any] = []
            if metric_name:
                query += " WHERE m.metric_name = ?"
                params.append(metric_name)

            query += """
                GROUP BY r.run_id, m.metric_name
                ORDER BY r.timestamp ASC
            """
            rows = cursor.execute(query, params).fetchall()
            trends: list[dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                total = item["total_evaluations"]
                passed = item["passed_evaluations"] or 0
                item["pass_rate"] = round((passed / total * 100), 1) if total > 0 else 0.0
                item["avg_score"] = (
                    round(item["avg_score"], 4) if item["avg_score"] is not None else 0.0
                )
                trends.append(item)
            return trends

    def get_run_comparison(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        """Compare two evaluation runs for regressions, fixes, and score deltas."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if not run_a or not run_b:
            return {"error": "One or both runs could not be found."}

        cases_a = {c["test_name"]: c for c in self.get_test_cases(run_id=run_id_a)}
        cases_b = {c["test_name"]: c for c in self.get_test_cases(run_id=run_id_b)}

        all_names = sorted(set(cases_a.keys()) | set(cases_b.keys()))
        comparisons: list[dict[str, Any]] = []
        regressions: list[dict[str, Any]] = []
        fixes: list[dict[str, Any]] = []

        for name in all_names:
            ca = cases_a.get(name)
            cb = cases_b.get(name)

            status_a = ca["status"] if ca else "N/A"
            status_b = cb["status"] if cb else "N/A"

            # Primary metric scores if available
            score_a = ca["metrics"][0]["score"] if ca and ca.get("metrics") else None
            score_b = cb["metrics"][0]["score"] if cb and cb.get("metrics") else None
            score_delta = (
                round(score_b - score_a, 4)
                if (score_a is not None and score_b is not None)
                else None
            )

            entry = {
                "test_name": name,
                "suite": (cb or ca)["suite"],
                "status_a": status_a,
                "status_b": status_b,
                "score_a": score_a,
                "score_b": score_b,
                "score_delta": score_delta,
                "case_a": ca,
                "case_b": cb,
            }
            comparisons.append(entry)

            if status_a == "PASSED" and status_b in ("FAILED", "ERROR"):
                regressions.append(entry)
            elif status_a in ("FAILED", "ERROR") and status_b == "PASSED":
                fixes.append(entry)

        pass_rate_a = run_a["pass_rate"]
        pass_rate_b = run_b["pass_rate"]

        return {
            "run_a": run_a,
            "run_b": run_b,
            "pass_rate_delta": round(pass_rate_b - pass_rate_a, 1),
            "total_regressions": len(regressions),
            "total_fixes": len(fixes),
            "regressions": regressions,
            "fixes": fixes,
            "comparisons": comparisons,
        }

    def delete_run(self, run_id: str) -> bool:
        """Delete an evaluation run and all associated test cases and metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM eval_runs WHERE run_id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Clear all evaluation runs, test cases, and metrics."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                DELETE FROM eval_metrics;
                DELETE FROM eval_test_cases;
                DELETE FROM eval_runs;
                """
            )
