"""
workspace_sync_node.py - Phase 5 dependency sync node.

Runs ``npm install --package-lock=true`` inside the shared Docker workspace
after the Remedy Agent has finished editing files.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_NPM_INSTALL_TIMEOUT_SECONDS = 600


def run_workspace_sync_node(state: OrchestratorState) -> Dict[str, Any]:
    """LangGraph node - dependency sync inside the shared workspace."""
    workspace_volume: Optional[str] = state.get("workspace_volume")
    if not workspace_volume:
        msg = "workspace_sync_node: workspace_volume is missing from state."
        logger.error(msg)
        return {
            "status": "dependency_sync_failed",
            "install_failures": msg,
            "errors": [msg],
        }

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            install_result = sandbox.run(
                "npm install --package-lock=true",
                timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
            )
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_sync_node: sandbox error - {exc}"
        logger.exception("workspace_sync_node: unexpected sandbox error.")
        return {
            "status": "dependency_sync_failed",
            "install_failures": msg,
            "errors": [msg],
        }

    if install_result.exit_code != 0:
        failure = (
            f"npm install failed (exit {install_result.exit_code}).\n"
            f"stdout:\n{install_result.stdout}\n"
            f"stderr:\n{install_result.stderr}"
        )
        logger.warning("workspace_sync_node: %s", failure)
        return {
            "status": "dependency_sync_failed",
            "install_failures": failure,
            "errors": [failure],
        }

    logger.info("workspace_sync_node: dependency sync completed successfully.")
    return {
        "status": "dependencies_ready",
        "install_failures": None,
    }
