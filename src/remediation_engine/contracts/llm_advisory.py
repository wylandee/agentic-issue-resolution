"""Advisory-only LLM output contract for supervisor enrichment."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LLMAdvisory(BaseModel):
    """Human-readable enrichment that cannot change supervisor state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reasoning: str = Field(default="")
    feedback_by_task: dict[str, str] = Field(default_factory=dict)
    new_constraints: list[str] = Field(default_factory=list)


__all__ = ["LLMAdvisory"]
