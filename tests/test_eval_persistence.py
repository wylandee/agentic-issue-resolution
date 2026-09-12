"""Tests for persisting pytest evaluation items alongside DeepEval cases."""

from __future__ import annotations

from types import SimpleNamespace

import remediation_engine.evals.db as eval_db
import tests.evals.conftest as eval_conftest
from remediation_engine.evals.models import EvalTestCaseRecord


def _eval_item(nodeid: str) -> SimpleNamespace:
    """Build a minimal collected item from a dedicated eval file."""
    return SimpleNamespace(
        nodeid=nodeid,
        path=eval_conftest._EVAL_TEST_ROOT / "test_workaround_subagent_eval.py",
        get_closest_marker=lambda marker: None,
    )


def _report(nodeid: str, outcome: str, longrepr: object | None = None) -> SimpleNamespace:
    """Build a minimal pytest call report for persistence tests."""
    return SimpleNamespace(
        nodeid=nodeid,
        when="call",
        outcome=outcome,
        duration=0.25,
        longrepr=longrepr,
    )


def _deep_eval_case(case_id: str, metric_name: str) -> SimpleNamespace:
    """Build a minimal DeepEval case with one metric result."""
    metric = SimpleNamespace(
        name=metric_name,
        score=1.0,
        threshold=0.7,
        success=True,
        reason="matched",
        evaluation_model="test-model",
        verbose_logs=None,
    )
    return SimpleNamespace(
        name=f"{case_id} [Workaround Subagent]",
        additional_metadata={"case_id": case_id},
        metrics_data=[metric],
        input="instruction",
        actual_output="output",
        expected_output="expected",
        context=["context"],
        retrieval_context=None,
        run_duration=0.5,
        evaluation_cost=0.01,
        success=True,
    )


def test_build_records_includes_passed_failed_and_skipped_pytest_items(monkeypatch) -> None:
    """Every selected item is persisted even when it has no DeepEval case."""
    pass_id = "tests/evals/test_workaround_subagent_eval.py::test_tool[case-pass]"
    fail_id = "tests/evals/test_workaround_subagent_eval.py::test_tool[case-fail]"
    skip_id = "tests/evals/test_workaround_subagent_eval.py::test_tool[case-skip]"
    items = [_eval_item(nodeid) for nodeid in (pass_id, fail_id, skip_id)]
    monkeypatch.setattr(
        eval_conftest,
        "_EVAL_TEST_REPORTS",
        {
            pass_id: {"call": _report(pass_id, "passed")},
            fail_id: {"call": _report(fail_id, "failed", "AssertionError: failed")},
            skip_id: {"call": _report(skip_id, "skipped", "not live")},
        },
    )

    records = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=items),
        [_deep_eval_case("case-pass", "Tool Correctness")],
        "test-model",
    )

    assert len(records) == 3
    assert [record.status for record in records] == ["PASSED", "FAILED", "SKIPPED"]
    assert records[0].test_name == pass_id
    assert records[0].metrics[0].metric_name == "Tool Correctness"
    assert records[1].error_message == "AssertionError: failed"
    assert records[2].error_message == "not live"


def test_deep_eval_metrics_match_duplicate_parametrized_items(monkeypatch) -> None:
    """Tool and task metrics for one case attach to their distinct pytest items."""
    case_id = "same-case"
    tool_id = (
        f"tests/evals/test_workaround_subagent_eval.py::test_tool_correctness_deepeval[{case_id}]"
    )
    task_id = (
        f"tests/evals/test_workaround_subagent_eval.py::test_task_completion_deepeval[{case_id}]"
    )
    items = [_eval_item(tool_id), _eval_item(task_id)]
    monkeypatch.setattr(
        eval_conftest,
        "_EVAL_TEST_REPORTS",
        {
            tool_id: {"call": _report(tool_id, "passed")},
            task_id: {"call": _report(task_id, "passed")},
        },
    )

    records = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=items),
        [
            _deep_eval_case(case_id, "Tool Correctness"),
            _deep_eval_case(case_id, "Task Completion"),
        ],
        "test-model",
    )

    assert [record.test_name for record in records] == [tool_id, task_id]
    assert [record.metrics[0].metric_name for record in records] == [
        "Tool Correctness",
        "Task Completion",
    ]


class _OptionParser:
    """Minimal pytest parser double for option-registration tests."""

    def __init__(self) -> None:
        self.options: dict[str, dict[str, object]] = {}

    def addoption(self, *names: str, **kwargs: object) -> None:
        for name in names:
            self.options[name] = kwargs


def test_eval_cli_options_are_registered() -> None:
    """Register the live, tag, and baseline options with the expected defaults."""
    parser = _OptionParser()

    eval_conftest.pytest_addoption(parser)

    assert parser.options["--run-eval-live"]["default"] is False
    assert parser.options["--eval-tag"]["default"] is None
    assert parser.options["--eval-baseline"]["default"] is None


def test_git_metadata_collects_branch_commit_and_dirty_state(monkeypatch) -> None:
    """Collect all three Git fields through the bounded command helper."""
    outputs = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "feat/eval",
        ("rev-parse", "HEAD"): "abc123",
        ("status", "--porcelain", "--untracked-files=all"): " M tests/evals/conftest.py",
    }
    monkeypatch.setattr(
        eval_conftest,
        "_git_command_output",
        lambda arguments: outputs[tuple(arguments)],
    )

    assert eval_conftest._git_metadata() == {
        "git_branch": "feat/eval",
        "git_commit": "abc123",
        "git_dirty": True,
    }


def test_git_metadata_uses_none_when_git_is_unavailable(monkeypatch) -> None:
    """Do not prevent persistence when Git metadata cannot be collected."""
    monkeypatch.setattr(eval_conftest, "_git_command_output", lambda arguments: None)

    assert eval_conftest._git_metadata() == {
        "git_branch": None,
        "git_commit": None,
        "git_dirty": None,
    }


def _session_with_options(options: dict[str, object], reporter: object) -> SimpleNamespace:
    """Build a minimal pytest session for session-finish hook tests."""
    config = SimpleNamespace(
        pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter),
        getoption=lambda name, default=None: options.get(name, default),
    )
    return SimpleNamespace(config=config, items=[])


def test_session_finish_saves_tag_and_resolves_baseline_before_save(monkeypatch) -> None:
    """Persist the normalized tag and compare only after the current run is saved."""
    events: list[object] = []

    class Reporter:
        lines: list[str] = []

        def write_line(self, message: str) -> None:
            self.lines.append(message)

    reporter = Reporter()

    class FakeDatabase:
        db_path = "temporary-evals.db"

        def __init__(self) -> None:
            events.append("init")

        def resolve_run_reference(self, reference: str, suite_name: str):
            events.append(("resolve", reference, suite_name))
            return {
                "run_id": "baseline-run",
                "timestamp": "2026-08-25T12:00:00",
                "tag": "baseline",
                "suite_name": suite_name,
                "pass_rate": 100.0,
            }

        def save_run(self, run):
            events.append(("save", run))
            return run.run_id

        def get_run_comparison(self, run_id_a: str, run_id_b: str):
            events.append(("compare", run_id_a, run_id_b))
            return {
                "run_a": {
                    "run_id": run_id_a,
                    "timestamp": "2026-08-25T12:00:00",
                    "tag": "baseline",
                    "suite_name": "tests/evals",
                    "pass_rate": 100.0,
                },
                "run_b": {
                    "run_id": run_id_b,
                    "timestamp": "2026-08-26T12:00:00",
                    "tag": "post-change",
                    "suite_name": "tests/evals",
                    "pass_rate": 100.0,
                },
                "pass_rate_delta": 0.0,
                "total_regressions": 0,
                "total_fixes": 0,
                "comparisons": [],
            }

    monkeypatch.setattr(eval_db, "EvalDatabase", FakeDatabase)
    monkeypatch.setattr(eval_conftest, "_load_deep_eval_test_cases", lambda: (None, []))
    monkeypatch.setattr(
        eval_conftest,
        "_build_eval_test_case_records",
        lambda session, cases, judge: [
            EvalTestCaseRecord(
                test_name="test_session",
                status="PASSED",
                latency_seconds=0.1,
                cost=0.0,
            )
        ],
    )
    monkeypatch.setattr(
        eval_conftest,
        "_git_metadata",
        lambda: {"git_branch": "feat/eval", "git_commit": "abc123", "git_dirty": False},
    )

    session = _session_with_options(
        {
            "--run-eval-live": False,
            "--eval-tag": "  post-change  ",
            "--eval-baseline": "baseline",
        },
        reporter,
    )
    eval_conftest.pytest_sessionfinish(session, exitstatus=1)

    event_names = [event if isinstance(event, str) else event[0] for event in events]
    assert event_names.index("resolve") < event_names.index("save") < event_names.index("compare")
    saved_run = next(
        event[1] for event in events if isinstance(event, tuple) and event[0] == "save"
    )
    assert saved_run.tag == "post-change"
    assert saved_run.metadata == {
        "git_branch": "feat/eval",
        "git_commit": "abc123",
        "git_dirty": False,
        "pytest_items_recorded": 1,
        "deep_eval_cases_recorded": 0,
        "token_usage_expected_cases": 0,
        "token_usage_observed_cases": 0,
    }
    assert any("Evaluation run comparison" in line for line in reporter.lines)


def test_session_finish_warns_for_missing_baseline_without_raising(monkeypatch) -> None:
    """A missing baseline is diagnostic and must not block run persistence."""
    events: list[str] = []

    class Reporter:
        lines: list[str] = []

        def write_line(self, message: str) -> None:
            self.lines.append(message)

    reporter = Reporter()

    class FakeDatabase:
        db_path = "temporary-evals.db"

        def resolve_run_reference(self, reference: str, suite_name: str):
            events.append("resolve")
            return None

        def save_run(self, run):
            events.append("save")
            return run.run_id

    monkeypatch.setattr(eval_db, "EvalDatabase", FakeDatabase)
    monkeypatch.setattr(eval_conftest, "_load_deep_eval_test_cases", lambda: (None, []))
    monkeypatch.setattr(
        eval_conftest,
        "_build_eval_test_case_records",
        lambda session, cases, judge: [
            EvalTestCaseRecord(
                test_name="test_session",
                status="PASSED",
                latency_seconds=0.1,
                cost=0.0,
            )
        ],
    )
    monkeypatch.setattr(
        eval_conftest,
        "_git_metadata",
        lambda: {"git_branch": None, "git_commit": None, "git_dirty": None},
    )

    session = _session_with_options({"--eval-baseline": "missing"}, reporter)
    eval_conftest.pytest_sessionfinish(session, exitstatus=1)

    assert events == ["resolve", "save"]
    assert any("was not found" in line for line in reporter.lines)
