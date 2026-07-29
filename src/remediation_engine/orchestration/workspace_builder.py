"""
workspace_builder.py - Shared Docker workspace preparation for remediation workers.

The node creates a named volume and copies the host repository into it. Worker
nodes perform edits and validation inside that shared workspace.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def run_workspace_builder_node(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node - Workspace Builder.

    Creates a Docker named volume and copies the host repository into it so the
    Specialist workers can edit files in-place using native tools.
    """
    repo_root_str: str = state.get("repo_root", "")
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid directory."
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
