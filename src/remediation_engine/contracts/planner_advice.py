"""Typed, advisory planner output contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from remediation_engine.contracts.schemas import SCARemediationStage


class PlannerAdvice(BaseModel):
    """Per-task stage recommendation from the planner LLM.

    The recommendation is advisory.  Python still selects versions and builds
    the worker instruction from registry facts and the task state.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(..., min_length=1)
    requested_stage: SCARemediationStage
    candidate_constraints: list[str] = Field(default_factory=list)
    reasoning: str = ""


class PlannerBatchAdvice(BaseModel):
    """Batch planner output for retryable version-bump tasks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    advice: list[PlannerAdvice] = Field(default_factory=list)


__all__ = ["PlannerAdvice", "PlannerBatchAdvice"]
