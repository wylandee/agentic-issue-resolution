"""
Phase 5 LangSmith tracing helpers.

This module is intentionally scoped to the Phase 5 orchestrator entrypoint so
Tracing is scoped to the current orchestrator entrypoint.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from remediation_engine.contracts.schemas import VulnerabilityGroup
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.subagent_runtime import MAX_SUBAGENT_TOOL_CALL_ROUNDS

log = logging.getLogger(__name__)

_DEFAULT_LANGSMITH_PROJECT = "AppSec-Remediation-Engine"
_PHASE5_RUN_NAME = "phase5_orchestrator"
_PHASE5_TAGS = ["phase-5", "orchestrator", "langgraph"]


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_phase5_tracing_enabled(settings=None) -> bool:
    """Return whether LangSmith tracing is enabled for Phase 5."""
    return (settings or get_runtime_settings()).langsmith_tracing


def build_phase5_runnable_config(
    repo_root: str,
    valid_groups: list[VulnerabilityGroup],
    settings=None,
    target_packages: list[str] | None = None,
) -> tuple[dict[str, Any] | None, uuid.UUID | None]:
    """
    Build a RunnableConfig-like dict for the Phase 5 orchestrator.

    Returns ``(None, None)`` when tracing is disabled.
    """
    if not is_phase5_tracing_enabled(settings):
        return None, None

    run_id = uuid.uuid4()
    metadata = {
        "repo_name": Path(repo_root).name,
        "repo_root": repo_root,
        "vulnerability_group_count": len(valid_groups),
        "max_tool_call_rounds": MAX_SUBAGENT_TOOL_CALL_ROUNDS,
    }
    if target_packages:
        metadata.update(
            {
                "target_packages": list(target_packages),
                "target_package_scope_enabled": True,
            }
        )
    config: dict[str, Any] = {
        "run_id": run_id,
        "run_name": _PHASE5_RUN_NAME,
        "tags": list(_PHASE5_TAGS),
        "metadata": metadata,
    }
    return config, run_id


def resolve_phase5_trace_url(run_id: uuid.UUID) -> str | None:
    """
    Resolve a LangSmith trace URL for a completed Phase 5 run.

    Any lookup failure is treated as non-fatal and returns ``None``.
    """
    try:
        wait_for_all_tracers()
        client = Client()
        run = client.read_run(run_id)
        project_name = get_runtime_settings().langsmith_project or _DEFAULT_LANGSMITH_PROJECT
        return str(client.get_run_url(run=run, project_name=project_name))
    except Exception as exc:  # pragma: no cover - defensive logging path
        log.warning("Phase 5 LangSmith URL lookup failed for run_id=%s: %s", run_id, exc)
        return None


def mark_phase5_trace_failed(run_id: uuid.UUID | str, error: BaseException | str) -> None:
    """Close an interrupted Phase 5 root run as an error in LangSmith.

    Args:
        run_id: LangSmith root run identifier.
        error: Exception or diagnostic that interrupted the graph.

    Raises:
        Exception: Propagates LangSmith client errors to the caller, which
            should treat this as best-effort cleanup and preserve the original
            orchestration error.
    """
    message = str(error).strip() or type(error).__name__
    settings = get_runtime_settings()
    client_kwargs: dict[str, Any] = {
        # This is an emergency terminal update.  Do not enqueue it behind the
        # normal tracing worker, which may be the work that was interrupted.
        "auto_batch_tracing": False,
        # Keep interruption cleanup bounded so a LangSmith outage cannot keep
        # the CLI process alive after the graph has already stopped.
        "timeout_ms": (2_000, 5_000),
    }
    if settings.langsmith_endpoint:
        client_kwargs["api_url"] = settings.langsmith_endpoint
    if settings.langsmith_api_key:
        client_kwargs["api_key"] = settings.langsmith_api_key
    client = Client(**client_kwargs)
    client.update_run(
        run_id,
        end_time=datetime.now(UTC),
        error=message,
        outputs={"status": "completed_with_errors", "error": message},
    )
