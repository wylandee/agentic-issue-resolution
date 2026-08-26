"""Programmatic evaluation suite runner and test execution utilities."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import uuid
from collections.abc import Generator
from pathlib import Path

from .db import EvalDatabase
from .models import EvalRunRecord, EvalTestCaseRecord, MetricRecord

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

SUITE_PATHS: dict[str, list[str]] = {
    "all": ["tests/evals"],
    "subagent": [
        "tests/evals/test_update_subagent_eval.py",
        "tests/evals/test_workaround_subagent_eval.py",
    ],
    "update_subagent": ["tests/evals/test_update_subagent_eval.py"],
    "workaround_subagent": ["tests/evals/test_workaround_subagent_eval.py"],
    "report": ["tests/evals/test_report_eval.py"],
    "triage": ["tests/evals/test_triage_eval.py"],
    "fix_planner": ["tests/evals/test_fix_planner_eval.py"],
    "adapters": ["tests/evals/test_adapters.py"],
}


def run_eval_subprocess(
    suite_key: str = "all",
    is_live: bool = False,
    judge_model: str = "gpt-4o",
    extra_args: list[str] | None = None,
) -> Generator[str, None, int]:
    """Execute pytest evaluation suite as a subprocess, yielding stdout lines in real-time.

    Args:
        suite_key: Target evaluation suite name (e.g. 'all', 'report', 'triage', 'fix_planner').
        is_live: Whether to pass --run-eval-live for live LLM judge evaluations.
        judge_model: Judge model identifier to export in EVAL_JUDGE_MODEL.
        extra_args: Additional pytest CLI arguments.

    Yields:
        Formatted console output lines.

    Returns:
        The process exit code.
    """
    targets = SUITE_PATHS.get(suite_key, ["tests/evals"])
    python_exe = sys.executable

    cmd = [python_exe, "-m", "pytest", "-v", "--strict-markers"]
    if is_live:
        cmd.append("--run-eval-live")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(targets)

    env = os.environ.copy()
    env["EVAL_JUDGE_MODEL"] = judge_model
    if is_live:
        env["EVAL_RUN_LIVE"] = "true"

    yield f"🚀 Launching: {' '.join(cmd)}\n"

    process = subprocess.Popen(
        cmd,
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    if process.stdout:
        yield from iter(process.stdout.readline, "")
        process.stdout.close()

    exit_code = process.wait()
    yield f"\n🏁 Process finished with exit code {exit_code}\n"
    return exit_code


def create_sample_run(
    db: EvalDatabase | None = None,
    suite_name: str = "report",
    judge_model: str = "gpt-4o",
) -> EvalRunRecord:
    """Generate and persist a realistic sample evaluation run for demo and offline verification.

    Args:
        db: EvalDatabase instance to save to. Defaults to project DB if None.
        suite_name: Target suite name.
        judge_model: Judge model name.

    Returns:
        The generated EvalRunRecord.
    """
    database = db or EvalDatabase()
    run_id = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now_iso = datetime.datetime.now().isoformat()

    test_cases: list[EvalTestCaseRecord] = [
        EvalTestCaseRecord(
            case_id="case_01",
            test_name="test_report_coverage_and_faithfulness [case_01]",
            suite="report",
            status="PASSED",
            input_text=(
                "Write a concise executive narrative for a human reader of a software security remediation run.\n"
                "Deterministic evidence:\n"
                '{"status": "completed", "overall_label": "success", "metrics": {"actionable_groups": 3, "groups_fixed": 3}}'
            ),
            actual_output=(
                "The remediation workflow completed successfully. All 3 actionable vulnerability groups "
                "were resolved with validated package updates and passing QA test gates."
            ),
            expected_output=(
                "All 3 actionable vulnerability groups were resolved successfully with passing QA gates."
            ),
            context_text="Remediation status: completed. Overall label: success. Actionable groups: 3, Fixed: 3.",
            latency_seconds=1.84,
            cost=0.0032,
            metrics=[
                MetricRecord(
                    metric_name="Finding Coverage & Accuracy",
                    score=0.92,
                    threshold=0.70,
                    success=True,
                    reason="The narrative faithfully includes all 3 fixed groups and mentions passing QA verification without hallucinations.",
                    evaluation_model=judge_model,
                ),
                MetricRecord(
                    metric_name="Report Constraint Adherence",
                    score=0.98,
                    threshold=0.70,
                    success=True,
                    reason="No invented CVEs, no prescriptive advice, and no Markdown headings were used.",
                    evaluation_model=judge_model,
                ),
            ],
        ),
        EvalTestCaseRecord(
            case_id="case_02",
            test_name="test_report_negative_constraints [case_02]",
            suite="report",
            status="PASSED",
            input_text=(
                "Write a concise executive narrative for a human reader of a software security remediation run.\n"
                "Deterministic evidence:\n"
                '{"status": "partial", "overall_label": "partial_success", "metrics": {"actionable_groups": 4, "groups_fixed": 2, "groups_unresolved": 2}}'
            ),
            actual_output=(
                "The remediation process achieved partial success, successfully fixing 2 actionable vulnerability "
                "groups while 2 groups remain unresolved due to peer dependency conflicts."
            ),
            expected_output=("2 groups fixed and 2 unresolved due to peer dependency conflicts."),
            context_text="Remediation status: partial. Fixed: 2, Unresolved: 2.",
            latency_seconds=2.15,
            cost=0.0041,
            metrics=[
                MetricRecord(
                    metric_name="Report Constraint Adherence",
                    score=0.88,
                    threshold=0.70,
                    success=True,
                    reason="The actual output strictly aligns with the deterministic metrics and accurately notes unresolved groups.",
                    evaluation_model=judge_model,
                ),
            ],
        ),
        EvalTestCaseRecord(
            case_id="case_03",
            test_name="test_triage_cve_classification [triage_001]",
            suite="triage",
            status="PASSED",
            input_text='{"cve_id": "CVE-2023-26159", "package": "follow-redirects", "version": "1.14.7", "manifest": "package.json"}',
            actual_output='{"verdict": "ACTIONABLE", "strategy": "DIRECT_UPDATE", "confidence": 0.95}',
            expected_output='{"verdict": "ACTIONABLE", "strategy": "DIRECT_UPDATE"}',
            context_text="follow-redirects < 1.15.6 is vulnerable to prototype pollution. Direct update available.",
            latency_seconds=0.95,
            cost=0.0018,
            metrics=[
                MetricRecord(
                    metric_name="Triage Verdict Accuracy",
                    score=1.00,
                    threshold=0.70,
                    success=True,
                    reason="Correctly classified as ACTIONABLE with DIRECT_UPDATE strategy.",
                    evaluation_model=judge_model,
                )
            ],
        ),
        EvalTestCaseRecord(
            case_id="case_04",
            test_name="test_fix_planner_strategy_selection [fix_002]",
            suite="fix_planner",
            status="PASSED",
            input_text='{"target": "lodash", "installed": "4.17.20", "advisory": "Prototype pollution in lodash"}',
            actual_output='{"strategy": "UPGRADE_DIRECT", "recommended_version": "4.17.21"}',
            expected_output='{"strategy": "UPGRADE_DIRECT", "recommended_version": "4.17.21"}',
            context_text="Lodash 4.17.21 contains the official patch.",
            latency_seconds=1.12,
            cost=0.0022,
            metrics=[
                MetricRecord(
                    metric_name="Fix Planner Recommendation Accuracy",
                    score=1.00,
                    threshold=0.70,
                    success=True,
                    reason="Selected correct target version 4.17.21.",
                    evaluation_model=judge_model,
                )
            ],
        ),
    ]

    total = len(test_cases)
    passed = sum(1 for tc in test_cases if tc.status == "PASSED")
    failed = total - passed
    duration = sum(tc.latency_seconds for tc in test_cases)
    cost = sum(tc.cost for tc in test_cases)

    run = EvalRunRecord(
        run_id=run_id,
        timestamp=now_iso,
        suite_name=suite_name,
        judge_model=judge_model,
        is_live=False,
        total_tests=total,
        passed_tests=passed,
        failed_tests=failed,
        skipped_tests=0,
        duration_seconds=round(duration, 2),
        total_cost=round(cost, 4),
        metadata={"git_branch": "feat/add-eval-ui", "triggered_by": "sample_generator"},
        test_cases=test_cases,
    )

    database.save_run(run)
    return run
