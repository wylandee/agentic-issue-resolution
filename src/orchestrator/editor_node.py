"""
editor_node.py - Phase 5 workspace builder node for the AppSec Orchestrator.

This module keeps the historical filename ``editor_node.py`` for migration
compatibility, but its Phase 5 responsibility is now only to prepare the shared
Docker named volume workspace. File edits are performed later by the Remedy
Agent via native tools, and dependency installation happens in
``workspace_sync_node.py``.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def run_workspace_builder_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node - Workspace Builder.

    Creates a Docker named volume and copies the host repository into it so the
    Remedy Agent can edit files in-place using native tools.
    """
    repo_root_str: str = state.get("repo_root", "")
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = (
            f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid "
            "directory."
        )
        logger.error(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "errors": [msg],
        }

    volume_name = f"agent_workspace_{uuid.uuid4().hex[:8]}"
    logger.info("workspace_builder_node: creating workspace volume %s.", volume_name)

    client = None
    try:
        client = get_docker_client()
        client.volumes.create(name=volume_name)
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: failed to create workspace volume - {exc}"
        logger.exception("workspace_builder_node: workspace volume creation failed.")
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "errors": [msg],
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        with DockerSandbox(repo_root_str, workspace_volume=volume_name):
            logger.info(
                "workspace_builder_node: repository copied into shared volume %s.",
                volume_name,
            )
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: sandbox error - {exc}"
        logger.exception("workspace_builder_node: repository copy failed.")
        return {
            "status": "workspace_build_failed",
            "workspace_volume": volume_name,
            "errors": [msg],
        }

    return {
        "workspace_volume": volume_name,
        "status": "workspace_ready",
    }


def run_editor_node(state: OrchestratorState) -> Dict[str, Any]:
    """Backward-compatible alias for the Phase 5 workspace builder."""
    return run_workspace_builder_node(state)
