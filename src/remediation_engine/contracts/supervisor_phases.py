"""Typed intermediate contracts for deterministic supervisor phases."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.schemas import (
    QAEvaluation,
    RemediationTask,
    StateConsistencyEvent,
    UpdateRetryDiagnostics,
)


class ReconciliationResult(BaseModel):
    """Output of worker and QA result reconciliation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_queue: dict[str, RemediationTask]
    qa_evaluations: dict[str, QAEvaluation] = Field(default_factory=dict)
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics] = Field(default_factory=dict)
    consistency_events: list[StateConsistencyEvent] = Field(default_factory=list)
    auto_constraints: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EligibleActions(BaseModel):
    """Pure projection of the actions currently eligible for routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    non_terminal_tasks: list[str] = Field(default_factory=list)
    qa_ready_task_ids: list[str] = Field(default_factory=list)
    workable_tasks: list[str] = Field(default_factory=list)
    no_fix_workable: list[str] = Field(default_factory=list)
    exhausted_pivots: list[str] = Field(default_factory=list)
    retry_version_bumps: list[str] = Field(default_factory=list)
    new_version_bumps: list[str] = Field(default_factory=list)
    workaround_tasks: list[str] = Field(default_factory=list)
    triage_required: bool = False


class AuditRecord(BaseModel):
    """Deterministic supervisor decision audit record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_code: DecisionCode
    next_node: str
    target_task_ids: list[str] = Field(default_factory=list)
    reasoning: str = ""
    state_revision: int = 0
    consistency_events: list[StateConsistencyEvent] = Field(default_factory=list)


__all__ = ["AuditRecord", "EligibleActions", "ReconciliationResult"]
