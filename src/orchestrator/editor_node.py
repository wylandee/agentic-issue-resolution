"""
editor_node.py — Phase 5 Editor Node for the AppSec Orchestrator.

This node creates a shared Docker named volume, streams the repository into it,
applies Remedy Agent edits inside the sandbox, and installs dependencies so the
same workspace can be reused by the scanner and tester nodes.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import EditRequest
from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def run_editor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Editor.

    Creates a named-volume workspace, applies ``edit_requests`` inside it, and
    runs ``npm install --no-audit --no-fund`` so later nodes can reuse the same
    dependency tree.
    """
    edit_requests: List[EditRequest] = state.get("edit_requests", [])
    repo_root_str: str = state.get("repo_root", "")

    if not edit_requests:
        logger.info("editor_node: no edit_requests — skipping.")
        return {"status": "no_edits"}

    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"editor_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "edit_failed",
            "test_failures": msg,
            "errors": [msg],
            "workspace_volume": None,
        }

    volume_name = f"agent_workspace_{uuid.uuid4().hex[:8]}"
    logger.info("editor_node: creating workspace volume %s.", volume_name)

    client = None
    try:
        client = get_docker_client()
        client.volumes.create(name=volume_name)
    except Exception as exc:  # noqa: BLE001
        msg = f"editor_node: failed to create workspace volume — {exc}"
        logger.exception("editor_node: workspace volume creation failed.")
        return {
            "status": "edit_failed",
            "test_failures": msg,
            "errors": [msg],
            "workspace_volume": None,
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        with DockerSandbox(repo_root_str, workspace_volume=volume_name) as sandbox:
            for edit in edit_requests:
                file_path = edit.file_path
                logger.info("editor_node: reading '%s' from workspace volume.", file_path)
                current = sandbox.read_file(file_path)
                if current is None:
                    error_msg = (
                        f"editor_node: cannot read '{file_path}' from workspace volume — "
                        "file missing or unreadable."
                    )
                    logger.error(error_msg)
                    return {
                        "status": "edit_failed",
                        "test_failures": error_msg,
                        "errors": [error_msg],
                        "workspace_volume": volume_name,
                    }

                count = current.count(edit.old_text)
                if count == 0:
                    error_msg = (
                        f"editor_node: old_text not found in '{file_path}'. "
                        "The file may have already been patched or the anchor is wrong."
                    )
                    logger.error(error_msg)
                    return {
                        "status": "edit_failed",
                        "test_failures": error_msg,
                        "errors": [error_msg],
                        "workspace_volume": volume_name,
                    }
                if count > 1:
                    error_msg = (
                        f"editor_node: old_text matches {count} times in '{file_path}' "
                        "(ambiguous anchor). Edit rejected."
                    )
                    logger.error(error_msg)
                    return {
                        "status": "edit_failed",
                        "test_failures": error_msg,
                        "errors": [error_msg],
                        "workspace_volume": volume_name,
                    }

                patched = current.replace(edit.old_text, edit.new_text, 1)
                logger.info("editor_node: writing patched '%s' to workspace volume.", file_path)
                sandbox.write_file(file_path, patched)

            logger.info("editor_node: running npm install inside shared workspace.")
            npm_result = sandbox.run("npm install --package-lock=true", timeout=600)
            if npm_result.exit_code != 0:
                failure_output = (
                    f"npm install failed (exit {npm_result.exit_code}).\n"
                    f"stdout:\n{npm_result.stdout}\n"
                    f"stderr:\n{npm_result.stderr}"
                )
                logger.error("editor_node: %s", failure_output)
                return {
                    "status": "edit_failed",
                    "test_failures": failure_output,
                    "errors": [failure_output],
                    "workspace_volume": volume_name,
                }

    except Exception as exc:  # noqa: BLE001
        msg = f"editor_node: sandbox error — {exc}"
        logger.exception("editor_node: unexpected sandbox error.")
        return {
            "status": "edit_failed",
            "test_failures": msg,
            "errors": [msg],
            "workspace_volume": volume_name,
        }

    logger.info("editor_node: shared workspace volume ready: %s", volume_name)
    return {
        "workspace_volume": volume_name,
        "status": "edited",
        "test_failures": None,
        "scan_failures": None,
    }
