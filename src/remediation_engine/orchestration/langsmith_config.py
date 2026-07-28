"""
Phase 5 LangSmith tracing helpers.

This module is intentionally scoped to the Phase 5 orchestrator entrypoint so
Tracing is scoped to the current orchestrator entrypoint.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from remediation_engine.contracts.schemas import VulnerabilityGroup
from remediation_engine.orchestration.subagent_runtime import MAX_SUBAGENT_TOOL_CALL_ROUNDS

log = logging.getLogger(__name__)

_DEFAULT_LANGSMITH_PROJECT = "AppSec-Remediation-Engine"
_PHASE5_RUN_NAME = "phase5_orchestrator"
_PHASE5_TAGS = ["phase-5", "orchestrator", "langgraph"]


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_phase5_tracing_enabled() -> bool:
    """Return whether LangSmith tracing is enabled for Phase 5."""
    return _is_truthy(os.environ.get("LANGSMITH_TRACING"))


def build_phase5_runnable_config(
    repo_root: str,
    valid_groups: list[VulnerabilityGroup],
) -> tuple[dict[str, Any] | None, uuid.UUID | None]:
    """
    Build a RunnableConfig-like dict for the Phase 5 orchestrator.

    Returns ``(None, None)`` when tracing is disabled.
    """
    if not is_phase5_tracing_enabled():
        return None, None

    run_id = uuid.uuid4()
    config: dict[str, Any] = {
        "run_id": run_id,
        "run_name": _PHASE5_RUN_NAME,
        "tags": list(_PHASE5_TAGS),
        "metadata": {
            "repo_name": Path(repo_root).name,
            "repo_root": repo_root,
            "vulnerability_group_count": len(valid_groups),
            "max_tool_call_rounds": MAX_SUBAGENT_TOOL_CALL_ROUNDS,
        },
    }
    return config, run_id


def resolve_phase5_trace_url(run_id: uuid.UUID) -> Optional[str]:
    """
    Resolve a LangSmith trace URL for a completed Phase 5 run.

    Any lookup failure is treated as non-fatal and returns ``None``.
    """
    try:
        wait_for_all_tracers()
        client = Client()
        run = client.read_run(run_id)
        project_name = os.environ.get("LANGSMITH_PROJECT") or _DEFAULT_LANGSMITH_PROJECT
        return str(client.get_run_url(run=run, project_name=project_name))
    except Exception as exc:  # pragma: no cover - defensive logging path
        log.warning("Phase 5 LangSmith URL lookup failed for run_id=%s: %s", run_id, exc)
        return None


