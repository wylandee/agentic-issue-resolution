"""Validated runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a conventional environment boolean."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    """Application settings with the existing environment names preserved."""

    openai_api_key: str = ""
    remedy_llm_model: str = "gpt-4o-mini"
    triage_llm_enabled: bool = False
    triage_llm_model: str = "gpt-4o-mini"
    supervisor_llm_model: str = "gpt-4o-mini"
    update_llm_model: str = "gpt-4o-mini"
    workaround_llm_model: str = "gpt-4o-mini"
    qa_llm_model: str = "gpt-4o-mini"
    serper_api_key: str = ""
    odc_extra_args: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "AppSec-Remediation-Engine"
    langsmith_endpoint: str = ""
    remediation_trajectory_dir: Path | None = None
    remedy_bypass_workaround_subagent: bool = False
    remedy_disable_post_qa_triage: bool = False
    remedy_disable_retriage: bool = False

    @classmethod
    def from_env(cls) -> AppSettings:
        """Build settings from the process environment without side effects."""
        trajectory = os.environ.get("REMEDIATION_TRAJECTORY_DIR", "").strip()
        default_model = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini").strip()
        if not default_model:
            default_model = "gpt-4o-mini"

        def model_override(name: str) -> str:
            """Return a node-specific model or the legacy remedy fallback."""
            return os.environ.get(name, "").strip() or default_model

        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            remedy_llm_model=default_model,
            triage_llm_enabled=_env_bool("TRIAGE_LLM_ENABLED"),
            triage_llm_model=os.environ.get("TRIAGE_LLM_MODEL", "gpt-4o-mini").strip(),
            supervisor_llm_model=model_override("SUPERVISOR_LLM_MODEL"),
            update_llm_model=model_override("UPDATE_LLM_MODEL"),
            workaround_llm_model=model_override("WORKAROUND_LLM_MODEL"),
            qa_llm_model=model_override("QA_LLM_MODEL"),
            serper_api_key=os.environ.get("SERPER_API_KEY", "").strip(),
            odc_extra_args=os.environ.get("ODC_EXTRA_ARGS", "").strip(),
            langsmith_tracing=_env_bool("LANGSMITH_TRACING"),
            langsmith_api_key=os.environ.get("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.environ.get(
                "LANGSMITH_PROJECT", "AppSec-Remediation-Engine"
            ).strip(),
            langsmith_endpoint=os.environ.get("LANGSMITH_ENDPOINT", "").strip(),
            remediation_trajectory_dir=Path(trajectory) if trajectory else None,
            remedy_bypass_workaround_subagent=_env_bool("REMEDY_BYPASS_WORKAROUND_SUBAGENT"),
            remedy_disable_post_qa_triage=_env_bool("REMEDY_DISABLE_POST_QA_TRIAGE"),
            remedy_disable_retriage=_env_bool("REMEDY_DISABLE_RETRIAGE"),
        )
