"""
editor_node.py — Phase 5 Editor Node for the AppSec Orchestrator.

Public API
----------
run_editor_node(state: OrchestratorState) -> Dict[str, Any]
    LangGraph node that applies ``EditRequest`` objects produced by the Remedy
    Agent entirely **inside** an ephemeral Docker sandbox.  Modified file
    contents are extracted from the sandbox into ``pending_files`` so the host
    repository is never touched.

Design principles
-----------------
* **Host-immutable** — no ``apply_edit``, no writes to ``repo_root``.
  All mutations happen inside the container via ``DockerSandbox.write_file``.
* **Exact-match anchor** — ``old_text`` must appear exactly once in the current
  sandbox file; zero or ambiguous matches cause the whole edit batch to abort.
* **npm manifest awareness** — if any edit touches a manifest file
  (``package.json``, ``package-lock.json``, etc.) the node runs
  ``npm install --ignore-scripts --package-lock-only`` and captures the updated
  lockfile into ``pending_files``.
* **Atomic pending_files** — the node returns all extracted files in one shot;
  if any extraction fails the error is logged but the node still returns the
  files it could read.

Statuses returned
-----------------
* ``"no_edits"``    — ``edit_requests`` list is empty.
* ``"edited"``      — all edits applied; ``pending_files`` populated.
* ``"edit_failed"`` — a pre-flight check, sandbox error, or npm install failed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import EditRequest
from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

# Manifest filenames that trigger an npm install after editing
_NPM_MANIFESTS = frozenset(
    {"package.json", "package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml"}
)


def _is_npm_manifest(file_path: str) -> bool:
    """Return True if the file basename is a recognised npm manifest."""
    return Path(file_path).name in _NPM_MANIFESTS


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def run_editor_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Editor.

    Applies ``state["edit_requests"]`` inside an ephemeral Docker sandbox and
    populates ``pending_files`` with the resulting file contents.

    Returns a state-update dict; the caller (LangGraph) merges it into the
    running state.
    """
    edit_requests: List[EditRequest] = state.get("edit_requests", [])
    repo_root_str: str = state.get("repo_root", "")

    # ── Guard: nothing to do ────────────────────────────────────────────────
    if not edit_requests:
        logger.info("editor_node: no edit_requests — skipping.")
        return {"status": "no_edits"}

    # ── Guard: repo_root must exist ─────────────────────────────────────────
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"editor_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "edit_failed",
            "test_failures": msg,
            "errors": [msg],
        }

    logger.info(
        "editor_node: starting sandbox from %s to apply %d edit(s).",
        repo_root_str,
        len(edit_requests),
    )

    try:
        with DockerSandbox(repo_root_str) as sandbox:
            modified_paths: List[str] = []
            error_msg: Optional[str] = None

            # ── Apply each EditRequest ──────────────────────────────────────
            for edit in edit_requests:
                file_path = edit.file_path

                # Read current content from sandbox
                logger.info("editor_node: reading '%s' from sandbox.", file_path)
                current = sandbox.read_file(file_path)
                if current is None:
                    error_msg = (
                        f"editor_node: cannot read '{file_path}' from sandbox — "
                        "file missing or unreadable."
                    )
                    logger.error(error_msg)
                    break

                # Exact-match check for old_text
                count = current.count(edit.old_text)
                if count == 0:
                    error_msg = (
                        f"editor_node: old_text not found in '{file_path}'. "
                        "The file may have already been patched or the anchor is wrong."
                    )
                    logger.error(error_msg)
                    break
                if count > 1:
                    error_msg = (
                        f"editor_node: old_text matches {count} times in '{file_path}' "
                        "(ambiguous anchor). Edit rejected."
                    )
                    logger.error(error_msg)
                    break

                # Apply replacement
                new_content = current.replace(edit.old_text, edit.new_text, 1)

                # Write back to sandbox
                logger.info("editor_node: writing patched '%s' to sandbox.", file_path)
                sandbox.write_file(file_path, new_content)
                if file_path not in modified_paths:
                    modified_paths.append(file_path)

            if error_msg:
                return {
                    "status": "edit_failed",
                    "test_failures": error_msg,
                    "errors": [error_msg],
                }

            # ── npm install if any manifest was touched ─────────────────────
            touched_manifest = any(_is_npm_manifest(p) for p in modified_paths)
            if touched_manifest:
                logger.info(
                    "editor_node: npm manifest modified — running "
                    "npm install --ignore-scripts --package-lock-only."
                )
                npm_result = sandbox.run(
                    "npm install --ignore-scripts --package-lock-only",
                    timeout=120,
                )
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
                    }

                # If a lockfile was generated, add it to modified paths
                lockfile = sandbox.read_file("package-lock.json")
                if lockfile is not None and "package-lock.json" not in modified_paths:
                    logger.info("editor_node: package-lock.json updated — will be extracted.")
                    modified_paths.append("package-lock.json")

            # ── Extract modified files into pending_files ───────────────────
            logger.info(
                "editor_node: extracting %d modified file(s) into pending_files.",
                len(modified_paths),
            )
            pending_files: Dict[str, str] = {}
            for path in modified_paths:
                content = sandbox.read_file(path)
                if content is None:
                    logger.warning(
                        "editor_node: could not extract '%s' from sandbox — skipping.",
                        path,
                    )
                    continue
                pending_files[path] = content
                logger.debug("editor_node: extracted %d bytes from '%s'.", len(content), path)

    except Exception as exc:  # noqa: BLE001
        msg = f"editor_node: sandbox error — {exc}"
        logger.exception("editor_node: unexpected sandbox error.")
        return {
            "status": "edit_failed",
            "test_failures": msg,
            "errors": [msg],
        }

    logger.info(
        "editor_node: done — %d file(s) in pending_files.", len(pending_files)
    )
    return {
        "pending_files": pending_files,
        "status": "edited",
        "test_failures": None,
        "scan_failures": None,
    }
