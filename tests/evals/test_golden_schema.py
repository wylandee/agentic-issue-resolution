"""Contract tests for the canonical Phase 2 golden-case envelope."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.evals.golden_schema import (
    GOLDEN_DATASET_NAMES,
    GoldenSchemaError,
    load_golden_dataset,
    validate_golden_dataset,
)

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def test_all_golden_datasets_load_and_validate() -> None:
    """Every maintained dataset satisfies the shared envelope."""
    counts: dict[str, int] = {}
    for name in GOLDEN_DATASET_NAMES:
        cases = load_golden_dataset(GOLDEN_DIR / f"{name}.json", dataset_name=name)
        counts[name] = len(cases)
        assert cases

    assert counts["fix_planner_cases"] == 15


def test_live_replay_datasets_have_structured_replay_inputs() -> None:
    """The four production replay suites expose typed-input payloads."""
    for name in (
        "triage_cases",
        "qa_cases",
        "update_subagent_cases",
        "workaround_subagent_cases",
    ):
        cases = load_golden_dataset(GOLDEN_DIR / f"{name}.json", dataset_name=name)
        assert all(isinstance(case["replay"]["input"], dict) for case in cases)


def test_validator_catches_duplicate_ids_and_bad_envelopes() -> None:
    """Malformed required fields produce actionable diagnostics."""
    cases = [
        {
            "case_id": "duplicate",
            "input": "task",
            "context": ["evidence"],
            "expected_output": "done",
            "expected_tools": [{"name": "read", "args": {}}],
            "eval_type": "triage",
            "replay": {"input": {}},
        },
        {
            "case_id": "duplicate",
            "input": "",
            "context": [1],
            "expected_output": "",
            "expected_tools": [{"name": "", "args": [], "unsupported": True}],
            "eval_type": "triage",
        },
    ]
    violations = validate_golden_dataset(cases, dataset_name="invalid")
    assert any("duplicate case_id" in violation for violation in violations)
    assert any("input must be a non-empty string" in violation for violation in violations)
    assert any("replay object" in violation for violation in violations)
    assert any("unsupported fields" in violation for violation in violations)


def test_loader_rejects_invalid_json(tmp_path: Path) -> None:
    """The public loader fails closed for malformed files."""
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(GoldenSchemaError):
        load_golden_dataset(path)


def test_observations_are_nested_under_offline_fixture() -> None:
    """Historical outputs are not top-level live observation fields."""
    for name in GOLDEN_DATASET_NAMES:
        path = GOLDEN_DIR / f"{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, list)
        for case in payload:
            assert "actual_output" not in case
            assert "tool_calls" not in case
            assert "expected_tool_calls" not in case
