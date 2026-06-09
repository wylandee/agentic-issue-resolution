"""
tester_node.py — Phase 5 Tester Node for the AppSec Orchestrator.

This node reuses the shared Docker named volume prepared by the editor node and
executes ``npm test`` directly inside that workspace.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)


def run_tester_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Tester.

    Mounts ``workspace_volume`` into a sandbox and runs ``npm test`` without
    reinstalling dependencies.
    """
    workspace_volume: Optional[str] = state.get("workspace_volume")
    if not workspace_volume:
        msg = "tester_node: workspace_volume is missing from state."
        logger.error(msg)
        return {"status": "test_failed", "test_failures": msg, "errors": [msg]}

    logger.info(
        "tester_node: starting sandbox against shared workspace volume %s.",
        workspace_volume,
    )

    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            test_result = sandbox.run("npm test", timeout=600)
            if test_result.exit_code != 0:
                failure = (
                    f"npm test failed (exit {test_result.exit_code}).\n"
                    f"stdout:\n{test_result.stdout}\n"
                    f"stderr:\n{test_result.stderr}"
                )
                logger.warning("tester_node: %s", failure)
                return {"status": "test_failed", "test_failures": failure}
    except Exception as exc:  # noqa: BLE001
        msg = f"tester_node: sandbox error — {exc}"
        logger.exception("tester_node: unexpected sandbox error.")
        return {"status": "test_failed", "test_failures": msg, "errors": [msg]}

    logger.info("tester_node: all tests passed.")
    return {"status": "tested", "test_failures": None}
