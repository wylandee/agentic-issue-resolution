"""Compatibility tests for DeepEval case metadata persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.evals.conftest import (
    _case_id_from_test_case,
    _deep_eval_node_score,
    _record_from_deep_eval_case,
    _test_case_metadata,
)
from tests.evals.eval_case_helpers import case_metadata_dict, set_pytest_item_identity


@pytest.mark.parametrize("attribute", ["additional_metadata", "metadata"])
def test_eval_metadata_shapes_survive_case_record_conversion(attribute: str) -> None:
    """Legacy and current DeepEval metadata fields reach persisted records."""
    stored = {
        "case_id": "report-case",
        "executive_summary_generation": {
            "status": "generated",
            "source_fingerprint": "fixture-fingerprint",
        },
    }
    test_case = SimpleNamespace(
        name="report-case [Factual Report Metrics]",
        metrics_data=[],
        **{attribute: stored},
    )

    assert case_metadata_dict(test_case) == stored
    assert _test_case_metadata(test_case) == stored
    assert _case_id_from_test_case(test_case) == "report-case"
    nodeid = "tests/evals/test_report_eval.py::test_report_case[report-case]"
    assert _deep_eval_node_score(test_case, nodeid) >= 10_000

    record = _record_from_deep_eval_case(
        test_case,
        nodeid,
        judge_model="gpt-6-luna",
    )

    assert record.case_id == "report-case"
    assert (
        record.additional_metadata["executive_summary_generation"]
        == stored["executive_summary_generation"]
    )


@pytest.mark.parametrize("attribute", ["additional_metadata", "metadata"])
def test_pytest_identity_is_written_to_supported_metadata_shape(
    attribute: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active pytest identity is attached for either metadata field."""
    original = {"case_id": "report-case"}
    test_case = SimpleNamespace(**{attribute: original})
    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "tests/evals/test_report_eval.py::test_report_case[report-case] (call)",
    )

    set_pytest_item_identity(test_case)

    assert getattr(test_case, attribute)["case_id"] == "report-case"
    assert getattr(test_case, attribute)["pytest_nodeid"] == (
        "tests/evals/test_report_eval.py::test_report_case[report-case]"
    )
