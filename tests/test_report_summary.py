"""Tests for the bounded executive-summary generation boundary."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from remediation_engine.orchestration.report_summary import (
    ExecutiveSummaryDraft,
    ExecutiveSummaryEvidence,
    SpecialRemediation,
    SummaryCount,
    fallback_executive_summary,
    generate_executive_summary,
)
from remediation_engine.settings import AppSettings

_FACTS = (
    "original_counts",
    "special_remediations",
    "remediated_counts",
    "follow_up_counts",
    "new_counts",
    "evidence_limits",
)


def _draft_response(*, duplicate: str | None = None) -> dict[str, Any]:
    """Return natural sentence lead-ins with each typed fact reference."""
    lead_ins = {
        "original_counts": "The run began with",
        "special_remediations": "",
        "remediated_counts": "Final QA accepted remediation for",
        "follow_up_counts": "That leaves",
        "new_counts": "We could not confirm",
        "evidence_limits": "The report also notes",
    }
    return {
        "clauses": [
            {
                "fact": duplicate if fact == "new_counts" and duplicate else fact,
                "lead_in": lead_ins[fact],
            }
            for fact in _FACTS
        ]
    }


def _evidence() -> ExecutiveSummaryEvidence:
    """Return fixture evidence with mixed outcomes and incomplete scan scope."""
    return ExecutiveSummaryEvidence(
        original_findings=SummaryCount(5, source_refs=("issues",)),
        original_groups=SummaryCount(4, source_refs=("initial_valid_groups",)),
        remediated_findings=SummaryCount(0, source_refs=("qa_results_by_attempt",)),
        remediated_groups=SummaryCount(0, source_refs=("task_queue",)),
        follow_up_findings=SummaryCount(5, source_refs=("initial_valid_groups.issues",)),
        follow_up_groups=SummaryCount(3, source_refs=("task_queue",)),
        new_findings=SummaryCount(None, complete=False, scope="incremental dependency inputs"),
        new_groups=SummaryCount(None, complete=False, scope="incremental dependency inputs"),
        special_remediations=(
            SpecialRemediation(
                kind="code workaround",
                package="express-jwt",
                attempt_id="attempt-1",
                outcome="attempted but failed QA",
                details="Source edit was not accepted.",
                source_refs=("group:g1", "attempt:attempt-1"),
            ),
        ),
        scan_scope="Incremental dependency inputs (incomplete)",
        scan_complete=False,
        evidence_limits=("post-scan grouping is incomplete",),
    )


class _StructuredModel:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.schema: type[Any] | None = None
        self.messages: Any = None
        self.calls = 0

    def with_structured_output(self, schema: type[Any]) -> _StructuredModel:
        self.schema = schema
        return self

    def invoke(self, messages: Any) -> Any:
        self.messages = messages
        self.calls += 1
        if isinstance(self.response, list):
            return self.response[min(self.calls - 1, len(self.response) - 1)]
        return self.response


def test_summary_uses_configured_model_and_resolves_factual_slots() -> None:
    """Model-authored framing receives renderer-owned factual values."""
    model = _StructuredModel(_draft_response())
    settings = AppSettings(openai_api_key="test-key", remedy_llm_model="test-model")

    result = generate_executive_summary(
        _evidence(),
        settings=settings,
        model_factory=lambda actual: model if actual is settings else None,
    )

    assert model.schema is ExecutiveSummaryDraft
    assert result.status == "generated"
    assert result.used_llm
    assert result.model == "test-model"
    assert (
        "The run began with **5 original findings** across **4 vulnerability groups**."
        in result.text
    )
    assert "The work included Code workaround for express-jwt" in result.text
    assert (
        "Final QA accepted remediation for **0 findings** across **0 vulnerability groups**."
        in result.text
    )
    assert "at least" not in result.text
    assert "unknown number" not in result.text
    assert "We could not confirm the count of newly detected findings" in result.text
    assert "attempted but failed QA" in result.text
    assert "Evidence limits:" not in result.text
    assert "**5 original findings**" in result.text
    assert "terminal scan coverage is incomplete for incremental dependency inputs" in result.text
    assert "findings outside that scope remain unknown" in result.text
    assert result.text != fallback_executive_summary(_evidence()).text


def test_complete_incremental_new_counts_are_scoped_in_summary() -> None:
    """A complete dependency-input scan is not presented as a full-workspace scan."""
    evidence = replace(
        _evidence(),
        new_findings=SummaryCount(0, scope="incremental_dependency_inputs"),
        new_groups=SummaryCount(0, scope="incremental_dependency_inputs"),
        scan_scope="Incremental dependency inputs (complete)",
        scan_complete=True,
    )
    response = _draft_response()
    response["clauses"][4]["lead_in"] = "The incremental dependency scan recorded"
    model = _StructuredModel(response)

    result = generate_executive_summary(
        evidence,
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert (
        "The incremental dependency scan recorded **0 newly detected findings** across "
        "**0 new vulnerability groups**."
    ) in result.text
    assert "terminal scan coverage is complete for incremental dependency inputs" in result.text


def test_summary_without_credentials_uses_deterministic_fallback() -> None:
    """Missing credentials select the local evidence renderer."""
    called = False

    def model_factory(_settings: AppSettings) -> _StructuredModel:
        nonlocal called
        called = True
        return _StructuredModel({})

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key=""),
        model_factory=model_factory,
    )

    assert result.status == "fallback_no_credentials"
    assert not result.used_llm
    assert not called
    assert result.text == fallback_executive_summary(_evidence()).text


def test_invalid_model_response_falls_back_and_redacts_credentials() -> None:
    """Invalid retries preserve factual fallback and redact secrets."""
    incomplete = _draft_response(duplicate="special_remediations")
    model = _StructuredModel(incomplete)
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-secret-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "fallback_error"
    assert not result.used_llm
    assert "fact-reference mismatch" in (result.error or "")
    assert "missing: new_counts" in (result.error or "")
    assert "repeated: special_remediations" in (result.error or "")
    assert model.calls == 2
    assert "test-secret-key" not in str(result.metadata())
    assert result.text == fallback_executive_summary(_evidence()).text


def test_model_slot_repair_is_bounded_and_keeps_validated_generation() -> None:
    """One repaired schema response may satisfy the required fact references."""
    incomplete = _draft_response(duplicate="special_remediations")
    complete = _draft_response()
    model = _StructuredModel([incomplete, complete])
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert result.used_llm
    assert model.calls == 2
    assert result.text.startswith("The run began with")
    assert "Final QA accepted remediation for" in result.text
    assert result.text != fallback_executive_summary(_evidence()).text


def test_model_repair_also_fixes_category_labels() -> None:
    """A category label and unsupported count in a lead-in get one repair call."""
    invalid = _draft_response()
    invalid["clauses"][0]["lead_in"] = "Baseline: The run began with 5 findings."
    model = _StructuredModel([invalid, _draft_response()])

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.used_llm
    assert model.calls == 2
    assert "Baseline:" not in result.text


def test_model_repair_rejects_unapproved_factual_lead_ins() -> None:
    """The schema rejects factual claims even when they fit the plain-text shape."""
    invalid = _draft_response()
    invalid["clauses"][0]["lead_in"] = "All original findings were present at start"
    model = _StructuredModel([invalid, _draft_response()])

    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    assert result.status == "generated"
    assert model.calls == 2
    assert result.text.startswith("The run began with")


def test_summary_prompt_and_rendering_use_natural_bolded_prose() -> None:
    """The generated lead-ins avoid labels and renderer facts bold numeric figures."""
    model = _StructuredModel(_draft_response())
    result = generate_executive_summary(
        _evidence(),
        settings=AppSettings(openai_api_key="test-key", remedy_llm_model="test-model"),
        model_factory=lambda _settings: model,
    )

    messages = model.messages
    assert "category followed by a colon" in messages[0].content
    assert '"fact": "original_counts"' in messages[1].content
    assert '"new_counts": ["We could not confirm"' in messages[1].content
    assert "The run began with **5 original findings**" in result.text
    assert not any(
        f"{label}:" in result.text.casefold()
        for label in ("baseline", "accepted outcomes", "remaining follow-up", "evidence limits")
    )


def test_summary_fingerprint_changes_when_special_evidence_changes() -> None:
    """Source fingerprints change when resolved remediation evidence changes."""
    original = _evidence()
    changed_remediation = replace(
        original.special_remediations[0],
        package="ws",
        details="A different accepted source edit.",
    )
    changed = replace(original, special_remediations=(changed_remediation,))

    first = fallback_executive_summary(original)
    second = fallback_executive_summary(changed)

    assert first.source_fingerprint != second.source_fingerprint
