"""
remedy_tools.py - Native workspace tools for the Phase 5 Remedy Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set

from langchain_core.tools import tool

from src.runtime.sandbox_mgr import DockerSandbox


def _validate_workspace_path(file_path: str) -> str:
    candidate = (file_path or "").strip()
    if not candidate:
        raise ValueError("file_path is required.")
    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        raise ValueError(f"Rejected absolute file path '{candidate}'.")
    if ".." in Path(candidate).parts:
        raise ValueError(f"Rejected path traversal in '{candidate}'.")
    return candidate.replace("\\", "/")


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _restore_newlines(text: str, newline_style: str) -> str:
    if newline_style == "\r\n":
        return text.replace("\n", "\r\n")
    return text


def _detect_newline_style(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    return "\n"


def build_agent_tools(sandbox: DockerSandbox, touched_files: Set[str]) -> List:
    """Build the native LangChain tool set for the active workspace sandbox."""

    @tool
    def read_workspace_file(file_path: str) -> str:
        """Read the current UTF-8 text content of a repo file from the workspace."""
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        content = sandbox.read_file(rel_path)
        if content is None:
            return (
                f"ERROR: Could not read '{rel_path}'. Verify the path and call "
                "read_workspace_file again if needed."
            )
        return content

    @tool
    def deterministic_search_replace(
        file_path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """
        Apply an exact one-time search/replace to a workspace file.

        ``old_text`` must match exactly once after newline normalization.
        """
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        current = sandbox.read_file(rel_path)
        if current is None:
            return (
                f"ERROR: Could not read '{rel_path}'. Call read_workspace_file to "
                "verify the current file content."
            )

        newline_style = _detect_newline_style(current)
        current_norm = _normalise_newlines(current)
        old_norm = _normalise_newlines(old_text)
        new_norm = _normalise_newlines(new_text)

        count = current_norm.count(old_norm)
        if count == 0:
            return (
                "ERROR: old_text not found. Call read_workspace_file to verify your "
                "anchor."
            )
        if count > 1:
            return (
                "ERROR: old_text found multiple times. Make anchor more specific."
            )

        updated = current_norm.replace(old_norm, new_norm, 1)
        sandbox.write_file(rel_path, _restore_newlines(updated, newline_style))
        touched_files.add(rel_path)
        return f"SUCCESS: File modified: {rel_path}"

    return [read_workspace_file, deterministic_search_replace]
