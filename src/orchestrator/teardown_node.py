"""
teardown_node.py - Final cleanup node for the Phase 5 AppSec Orchestrator.

This node reads changed files back out of the shared Docker named volume to
build a unified diff, then always attempts to delete the volume.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _read_host_text(repo_root: Path, rel_path: str) -> str:
    file_path = repo_root / rel_path
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def _build_diff(rel_path: str, before_text: str, after_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


def run_teardown_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node - Teardown.

    Reads updated changed files from the workspace volume, generates a unified
    diff against the host repository, and always attempts to remove the named
    volume.
    """
    repo_root_str: str = state.get("repo_root", "")
    workspace_volume: Optional[str] = state.get("workspace_volume")
    changed_files: List[str] = sorted(set(state.get("changed_files", [])))

    diff_chunks: List[str] = []
    errors: List[str] = []
    client = None

    try:
        if changed_files and workspace_volume and repo_root_str:
            repo_root = Path(repo_root_str)
            if not repo_root.is_dir():
                msg = f"teardown_node: repo_root '{repo_root_str}' is not a valid directory."
                logger.error(msg)
                errors.append(msg)
            else:
                try:
                    with DockerSandbox(
                        repo_root=None,
                        workspace_volume=workspace_volume,
                    ) as sandbox:
                        for rel_path in changed_files:
                            updated_text = sandbox.read_file(rel_path)
                            if updated_text is None:
                                continue
                            original_text = _read_host_text(repo_root, rel_path)
                            diff_text = _build_diff(rel_path, original_text, updated_text)
                            if diff_text:
                                diff_chunks.append(diff_text)
                except Exception as exc:  # noqa: BLE001
                    msg = f"teardown_node: failed to extract diff from workspace volume - {exc}"
                    logger.exception("teardown_node: diff extraction failed.")
                    errors.append(msg)
    finally:
        if workspace_volume:
            try:
                client = get_docker_client()
                client.volumes.get(workspace_volume).remove(force=True)
                logger.info("teardown_node: removed workspace volume %s.", workspace_volume)
            except Exception as exc:  # noqa: BLE001
                msg = f"teardown_node: failed to remove workspace volume '{workspace_volume}' - {exc}"
                logger.exception("teardown_node: workspace volume cleanup failed.")
                errors.append(msg)
            finally:
                if client is not None:
                    _close_client(client)

    result: Dict[str, Any] = {
        "status": "completed",
        "workspace_volume": None,
        "changed_files": changed_files,
        "diff": "".join(diff_chunks),
    }
    if errors:
        result["errors"] = errors
    return result
