"""Tests for persisting pytest evaluation items alongside DeepEval cases."""

from __future__ import annotations

from types import SimpleNamespace

import tests.evals.conftest as eval_conftest


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
