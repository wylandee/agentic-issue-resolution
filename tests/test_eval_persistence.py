"""Focused tests for eval persistence."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import remediation_engine.evals.db as eval_db
import tests.evals.conftest as eval_conftest
from remediation_engine.evals.comparison import format_run_comparison, format_run_list
from remediation_engine.evals.db import EvalDatabase
from remediation_engine.evals.models import EvalRunRecord, EvalTestCaseRecord, MetricRecord
from tests.evals import test_report_eval

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


def test_duplicate_deepeval_metrics_share_one_pytest_record(monkeypatch) -> None:
    """Metrics emitted by one pytest item stay together in the dashboard."""
    case_id = "same-case"
    live_id = f"tests/evals/test_qa_critic_eval.py::test_qa_critic_live_replay_metrics[{case_id}]"
    offline_id = (
        f"tests/evals/test_qa_critic_eval.py::test_qa_critic_offline_production_replay[{case_id}]"
    )
    items = [_eval_item(offline_id), _eval_item(live_id)]
    monkeypatch.setattr(
        eval_conftest,
        "_EVAL_TEST_REPORTS",
        {
            live_id: {"call": _report(live_id, "passed")},
            offline_id: {"call": _report(offline_id, "passed")},
        },
    )
    deep_eval_cases = [
        _deep_eval_case(case_id, "Tool Correctness"),
        _deep_eval_case(case_id, "Task Completion"),
    ]
    for deep_eval_case in deep_eval_cases:
        deep_eval_case.additional_metadata["replay_source"] = "production_live"

    records = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=items),
        deep_eval_cases,
        "test-model",
    )

    assert [record.test_name for record in records] == [offline_id, live_id]
    assert records[0].metrics == []
    assert [metric.metric_name for metric in records[1].metrics] == [
        "Tool Correctness",
        "Task Completion",
    ]


def test_direct_end_to_end_scores_are_persisted_with_their_pytest_item(monkeypatch) -> None:
    nodeid = "tests/evals/test_end_to_end_eval.py::test_end_to_end_live_selected_run[run-1]"
    monkeypatch.setenv("PYTEST_CURRENT_TEST", f"{nodeid} (call)")
    monkeypatch.setattr(eval_conftest, "_EVAL_DIRECT_RECORDS", {})
    record = EvalTestCaseRecord(
        case_id="run-1",
        test_name="end_to_end[run-1]",
        suite="end_to_end",
        status="PASSED",
    )

    eval_conftest.record_eval_result(record)
    saved = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=[_eval_item(nodeid)]), [], "test-model"
    )

    assert len(saved) == 1
    assert saved[0].suite == "end_to_end"
    assert saved[0].additional_metadata["pytest_nodeid"] == nodeid


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
    assert parser.options["--e2e-trajectory"]["action"] == "append"
    assert parser.options["--e2e-run-repo"]["default"] is None
    assert parser.options["--e2e-input"]["default"] is None


def test_end_to_end_pytest_inputs_are_explicit_and_repeatable(tmp_path) -> None:
    class Config:
        def __init__(self, values):
            self.values = values

        def getoption(self, name, default=None):
            return self.values.get(name, default)

    class Metafunc:
        fixturenames = ["e2e_source"]

        def __init__(self, values):
            self.config = Config(values)
            self.parameters = None

        def parametrize(self, name, values, ids):
            self.parameters = (name, values, ids)

    trajectory_one = tmp_path / "one.md"
    trajectory_two = tmp_path / "two.md"
    metafunc = Metafunc({"--e2e-trajectory": [trajectory_one, trajectory_two]})
    eval_conftest.pytest_generate_tests(metafunc)
    assert metafunc.parameters[0] == "e2e_source"
    assert [item["path"] for item in metafunc.parameters[1]] == [trajectory_one, trajectory_two]

    missing_half = Metafunc({"--e2e-run-repo": tmp_path})
    import pytest

    with pytest.raises(pytest.UsageError, match="must be supplied together"):
        eval_conftest.pytest_generate_tests(missing_half)


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
        option=SimpleNamespace(collectonly=False),
        pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter),
        getoption=lambda name, default=None: options.get(name, default),
    )
    return SimpleNamespace(
        config=config, items=[_eval_item("tests/evals/test_report_eval.py::test_session")]
    )


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


def test_single_selected_item_keeps_all_metrics(monkeypatch):
    nodeid = "tests/evals/test_update_subagent_eval.py::test_live_replay[single]"
    monkeypatch.setattr(
        eval_conftest, "_EVAL_TEST_REPORTS", {nodeid: {"call": _report(nodeid, "passed")}}
    )
    records = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=[_eval_item(nodeid)]),
        [
            _deep_eval_case("single", "Tool Correctness"),
            _deep_eval_case("single", "Task Completion"),
        ],
        "offline-judge",
    )
    assert [metric.metric_name for metric in records[0].metrics] == [
        "Tool Correctness",
        "Task Completion",
    ]


def test_unscored_metric_is_not_persisted_as_a_zero_score(monkeypatch):
    nodeid = "tests/evals/test_report_eval.py::test_report_missing_score"
    monkeypatch.setattr(
        eval_conftest, "_EVAL_TEST_REPORTS", {nodeid: {"call": _report(nodeid, "passed")}}
    )
    case = _deep_eval_case("missing-score", "HallucinationMetric")
    case.metrics_data[0].score = None
    case.metrics_data[0].success = None

    records = eval_conftest._build_eval_test_case_records(
        SimpleNamespace(items=[_eval_item(nodeid)]), [case], "test-model"
    )

    assert records[0].metrics == []
    assert records[0].additional_metadata["unscored_metrics"] == ["HallucinationMetric"]


def test_explicit_identity_wins_over_case_name_heuristics(monkeypatch):
    first = "tests/evals/test_report_eval.py::test_report[one]"
    second = "tests/evals/test_report_eval.py::test_report[two]"
    monkeypatch.setattr(
        eval_conftest,
        "_EVAL_TEST_REPORTS",
        {node: {"call": _report(node, "passed")} for node in (first, second)},
    )
    case = _deep_eval_case("one", "Faithfulness")
    case.additional_metadata["pytest_nodeid"] = second
    assert eval_conftest._match_deep_eval_case_to_nodeid(case, [first, second], set()) == second


def test_session_persistence_failure_preserves_original_failure(monkeypatch):
    monkeypatch.setattr(
        eval_conftest,
        "_persist_eval_session",
        lambda *_: (_ for _ in ()).throw(ImportError("broken import")),
    )
    eval_conftest.pytest_sessionfinish(_session_with_options({}, None), 1)


def test_session_without_evaluation_items_does_not_import_persistence(monkeypatch):
    monkeypatch.setattr(
        eval_conftest,
        "_persist_eval_session",
        lambda *_: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    eval_conftest.pytest_sessionfinish(
        SimpleNamespace(
            config=SimpleNamespace(option=SimpleNamespace(collectonly=False)), items=[]
        ),
        2,
    )


def test_report_metric_case_uses_only_source_evidence(monkeypatch) -> None:
    """All report judges receive the same source-grounded case."""

    class FakeLLMTestCase:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(test_report_eval, "LLMTestCase", FakeLLMTestCase)
    case = test_report_eval._load_golden_cases()[0]
    report = test_report_eval.render_report(case)
    source_evidence = test_report_eval.build_report_context(case)
    metric_case = test_report_eval._build_report_metric_case(
        case,
        report,
        source_evidence,
    )

    assert metric_case.input == source_evidence
    assert metric_case.actual_output == report
    assert getattr(metric_case, "expected_output", None) is None
    assert metric_case.context == [source_evidence]
    assert metric_case.retrieval_context == [source_evidence]
    assert "expected_contract" not in source_evidence
    assert "report_format" not in source_evidence


def test_summarization_metric_receives_case_specific_questions(monkeypatch) -> None:
    """The judge receives factual yes/no questions answerable from both texts."""

    class FakeMetric:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(test_report_eval, "HAS_DEEPEVAL", True)
    monkeypatch.setattr(test_report_eval, "HallucinationMetric", FakeMetric)
    monkeypatch.setattr(test_report_eval, "FaithfulnessMetric", FakeMetric)
    monkeypatch.setattr(test_report_eval, "SummarizationMetric", FakeMetric)
    case = test_report_eval._load_golden_cases()[0]
    metrics = test_report_eval._build_metrics(
        SimpleNamespace(judge_model="test-model", judge_llm=None),
        case,
    )

    questions = metrics[2].kwargs["assessment_questions"]
    assert questions == test_report_eval._report_summarization_questions(case)
    assert questions[0] == "Are there exactly 5 scanner findings across 4 vulnerability groups?"
    assert (
        questions[1]
        == "Are 4 vulnerability groups successfully remediated and 0 vulnerability groups "
        "requiring follow-up?"
    )
    assert any(
        "CVE-2026-30001 -> vulnerable package base64url" in question for question in questions
    )
    assert all("Does the report" not in question for question in questions)
    assert any("5 canonical findings across 4 groups" in question for question in questions)
    assert not any("None" in question for question in questions)


def test_code_detail_files_are_source_only_for_package_removal() -> None:
    """Manifests stay in the main row, while code details list accepted source paths."""
    case = next(
        case
        for case in test_report_eval._load_golden_cases()
        if case["case_id"] == "no_fix_package_removal"
    )
    report = test_report_eval.render_report(case)
    parsed = test_report_eval.parse_report(report)
    table_files = parsed["successful_rows"][0]["Files Changed"]
    source_evidence = test_report_eval.build_report_context(case)

    assert "package.json" in table_files
    assert "package-lock.json" in table_files
    assert parsed["code_detail_file_paths"] == ["routes/b2bOrder.ts"]
    assert "package.json" not in parsed["code_detail_text"]
    assert "package-lock.json" not in parsed["code_detail_text"]
    assert not test_report_eval.validate_report_contract(report, case["expected_contract"])
    assert "package.json" in source_evidence

    tampered_report = report.replace(
        "- **Files changed:** routes/b2bOrder.ts",
        "- **Files changed:** package.json, routes/b2bOrder.ts",
        1,
    )
    violations = test_report_eval.validate_report_contract(
        tampered_report, case["expected_contract"]
    )
    assert any(
        "manifest leaked into Code workaround details: package.json" in item for item in violations
    )


def test_live_report_runs_three_metrics_on_one_case(monkeypatch) -> None:
    """All three report metrics receive the same report and evidence."""

    class FakeLLMTestCase:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    metric_case = object()
    metrics = [object(), object(), object()]
    calls: list[tuple[Any, list[Any]]] = []

    def fake_assert_test(test_case: Any, selected_metrics: list[Any]) -> None:
        calls.append((test_case, selected_metrics))
        raise AssertionError("judge threshold failure")

    monkeypatch.setattr(test_report_eval, "HAS_DEEPEVAL", True)
    monkeypatch.setattr(test_report_eval, "LLMTestCase", FakeLLMTestCase)
    monkeypatch.setattr(
        test_report_eval,
        "render_report",
        lambda case, **_kwargs: "report",
    )
    monkeypatch.setattr(
        test_report_eval,
        "validate_report_contract",
        lambda report, contract, **_kwargs: [],
    )
    monkeypatch.setattr(
        test_report_eval,
        "build_report_context",
        lambda case: ["evidence", "policy"],
    )
    monkeypatch.setattr(test_report_eval, "_build_context", lambda state: object())
    monkeypatch.setattr(
        test_report_eval, "_build_executive_summary_evidence", lambda context: object()
    )
    summary_result = SimpleNamespace(
        used_llm=True,
        status="generated",
        error=None,
        metadata=lambda: {"status": "generated", "model": "test-model"},
    )
    monkeypatch.setattr(
        test_report_eval,
        "generate_executive_summary",
        lambda evidence, *, settings: summary_result,
    )
    monkeypatch.setattr(
        test_report_eval,
        "_build_report_metric_case",
        lambda case, report, context, **_kwargs: metric_case,
    )
    monkeypatch.setattr(
        test_report_eval,
        "_build_metrics",
        lambda settings, case: metrics,
    )
    monkeypatch.setattr(test_report_eval, "assert_test", fake_assert_test)

    case = {
        **test_report_eval._load_golden_cases()[0],
        "case_id": "fixture",
        "expected_contract": {},
    }
    settings = SimpleNamespace(
        is_live=True,
        openai_api_key="test",
        judge_model="test-model",
        judge_llm=None,
        app_settings=object(),
    )
    test_report_eval.TestReportNodeEval().test_report_uses_the_three_requested_metrics(
        case,
        settings,
    )

    assert calls == [(metric_case, metrics)]


def test_report_evidence_documents_build_for_all_golden_cases() -> None:
    """Every curated case produces source facts for judging."""
    for case in test_report_eval._load_golden_cases():
        source_evidence = test_report_eval.build_report_context(case)

        assert f"Run ID: {case['case_id']}." in source_evidence
        assert "There are exactly" in source_evidence
        for group in case["fixture"].get("groups", []):
            if group.get("package"):
                assert group["package"] in source_evidence
