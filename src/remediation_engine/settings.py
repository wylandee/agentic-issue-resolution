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


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Parse an environment integer and validate an optional lower bound."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


DEFAULT_REMEDY_RETRIAGE_LIMIT = 3


@dataclass(frozen=True)
class AppSettings:
    """Immutable runtime settings loaded from the supported environment names."""

    openai_api_key: str = ""
    remedy_llm_model: str = "gpt-4o-mini"
    triage_llm_enabled: bool = False
    triage_llm_model: str = "gpt-4o-mini"
    update_llm_model: str = "gpt-4o-mini"
    workaround_llm_model: str = "gpt-4o-mini"
    qa_llm_model: str = "gpt-4o-mini"
    serper_api_key: str = ""
    github_token: str = ""
    odc_extra_args: str = ""
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "AppSec-Remediation-Engine"
    langsmith_endpoint: str = ""
    triage_cache_dir: Path | None = None
    remediation_trajectory_dir: Path | None = None
    remediation_report_dir: Path | None = None
    remedy_bypass_workaround_subagent: bool = False
    remedy_disable_post_qa_triage: bool = False
    # Development-only safety valve. Production remains unlimited by default.
    remedy_retriage_limit_enabled: bool = False
    remedy_retriage_limit: int = DEFAULT_REMEDY_RETRIAGE_LIMIT

    @classmethod
    def from_env(cls) -> AppSettings:
        """Build immutable settings from the process environment.

        This constructor only reads environment variables and constructs a
        value object; it does not create directories, contact services, or
        mutate the host repository.
        """
        trajectory = os.environ.get("REMEDIATION_TRAJECTORY_DIR", "").strip()
        report_dir = os.environ.get("REMEDIATION_REPORT_DIR", "").strip()
        default_model = os.environ.get("REMEDY_LLM_MODEL", "gpt-4o-mini").strip()
        if not default_model:
            default_model = "gpt-4o-mini"

        def model_override(name: str) -> str:
            """Return a node-specific model or the configured default model."""
            return os.environ.get(name, "").strip() or default_model

        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            remedy_llm_model=default_model,
            triage_llm_enabled=_env_bool("TRIAGE_LLM_ENABLED"),
            triage_llm_model=model_override("TRIAGE_LLM_MODEL"),
            update_llm_model=model_override("UPDATE_LLM_MODEL"),
            workaround_llm_model=model_override("WORKAROUND_LLM_MODEL"),
            qa_llm_model=model_override("QA_LLM_MODEL"),
            serper_api_key=os.environ.get("SERPER_API_KEY", "").strip(),
            github_token=os.environ.get("GITHUB_TOKEN", "").strip(),
            odc_extra_args=os.environ.get("ODC_EXTRA_ARGS", "").strip(),
            langsmith_tracing=_env_bool("LANGSMITH_TRACING"),
            langsmith_api_key=os.environ.get("LANGSMITH_API_KEY", "").strip(),
            langsmith_project=os.environ.get(
                "LANGSMITH_PROJECT", "AppSec-Remediation-Engine"
            ).strip(),
            langsmith_endpoint=os.environ.get("LANGSMITH_ENDPOINT", "").strip(),
            triage_cache_dir=(
                Path(os.environ["TRIAGE_CACHE_DIR"].strip())
                if os.environ.get("TRIAGE_CACHE_DIR", "").strip()
                else None
            ),
            remediation_trajectory_dir=Path(trajectory) if trajectory else None,
            remediation_report_dir=Path(report_dir) if report_dir else None,
            remedy_bypass_workaround_subagent=_env_bool("REMEDY_BYPASS_WORKAROUND_SUBAGENT"),
            remedy_disable_post_qa_triage=_env_bool("REMEDY_DISABLE_POST_QA_TRIAGE"),
            remedy_retriage_limit_enabled=_env_bool("REMEDY_RETRIAGE_LIMIT_ENABLED"),
            remedy_retriage_limit=_env_int(
                "REMEDY_RETRIAGE_LIMIT",
                DEFAULT_REMEDY_RETRIAGE_LIMIT,
                minimum=0,
            ),
        )
