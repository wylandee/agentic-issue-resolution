"""
remedy_tools.py - Specialized native workspace tools for Phase 5 subagents.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set

from langchain_core.tools import tool

from src.runtime.sandbox_mgr import DockerSandbox
from src.tools.registry_tools import view_npm_package_versions

logger = logging.getLogger(__name__)

_MANIFEST_SYNC_TIMEOUT_SECONDS = 120
_SYNTAX_CHECK_TIMEOUT_SECONDS = 30

_REPO_MAP_MAX_ENTRIES = 400
_SEARCH_MAX_BYTES = 32_768
_SEARCH_TIMEOUT_SECONDS = 15
_INSPECT_TEXT_MAX_CHARS = 8_000


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


def _workspace_dir_for_manifest(manifest_path: str) -> str:
    parent = Path(manifest_path).parent.as_posix()
    if parent in ("", "."):
        return "/workspace"
    return f"/workspace/{parent}"


def _normalize_manifest_targets(target_manifest_paths: Iterable[str]) -> List[str]:
    """Return stable, validated package.json targets for one update batch."""
    manifest_paths = sorted(
        {
            _validate_workspace_path(path)
            for path in target_manifest_paths
            if path
        }
    )
    invalid = [
        path for path in manifest_paths if Path(path).name != "package.json"
    ]
    if invalid:
        raise ValueError(
            "All target manifest paths must point to package.json files. "
            f"Invalid values: {invalid}"
        )
    return manifest_paths


def _normalize_package_manifest_targets(
    package_manifest_paths: Mapping[str, Iterable[str]],
) -> Dict[str, List[str]]:
    """Return validated package-to-manifest targets for one update batch."""
    normalized: Dict[str, List[str]] = {}
    for package_name, manifest_paths in package_manifest_paths.items():
        package_key = (package_name or "").strip()
        if not package_key:
            raise ValueError("Package manifest target keys must be non-empty.")
        normalized[package_key] = _normalize_manifest_targets(manifest_paths)
    return normalized


def _make_read_repository_map_tool(sandbox: DockerSandbox):
    @tool
    def read_repository_map() -> str:
        """Return a deterministic ASCII tree of every file/directory in the workspace."""
        script = (
            "find /workspace -not -path '*/node_modules/*' "
            "-not -path '*/.git/*' "
            "-not -name '*.map' "
            "| sort"
        )
        result = sandbox.run(script, timeout=10)
        if result.exit_code != 0:
            return f"ERROR: Could not list workspace: {result.stderr.strip()}"

        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return "(workspace is empty)"

        capped = lines[:_REPO_MAP_MAX_ENTRIES]
        truncated = len(lines) > _REPO_MAP_MAX_ENTRIES
        output = "\n".join(capped)
        if truncated:
            output += f"\n... (truncated, {len(lines) - _REPO_MAP_MAX_ENTRIES} more entries)"
        return output

    return read_repository_map


def _make_revert_workspace_file_tool(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
):
    @tool
    def revert_workspace_file(file_path: str) -> str:
        """
        Restore a workspace file to its original host baseline state.
        """
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        baseline_file = host_repo_root / rel_path
        if not baseline_file.is_file():
            return f"ERROR: Baseline file '{rel_path}' does not exist on host."

        try:
            content = baseline_file.read_text(encoding="utf-8")
        except Exception as exc:
            return f"ERROR: Baseline file '{rel_path}' is unreadable: {exc}"

        try:
            sandbox.write_file(rel_path, content)
        except Exception as exc:
            return f"ERROR: Failed to overwrite file in sandbox '{rel_path}': {exc}"

        touched_files.discard(rel_path)
        return f"SUCCESS: Reverted workspace file '{rel_path}' to its baseline state."

    return revert_workspace_file


def _make_modify_npm_dependency_tool(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
):
    allowed_manifest_paths_by_package = {
        package_name: set(manifest_paths)
        for package_name, manifest_paths in _normalize_package_manifest_targets(
            package_manifest_paths
        ).items()
    }

    @tool
    def modify_npm_dependency(
        package_name: str,
        target_version: str,
        dependency_type: str,
        manifest_path: str = "package.json",
    ) -> str:
        """
        Modify dependencies, devDependencies, or overrides in a package.json.
        """
        import re

        safe_pattern = re.compile(r"^[a-zA-Z0-9.\-/@~^*]+$")
        if not safe_pattern.match(package_name):
            return (
                f"ERROR: Invalid package_name '{package_name}'. Only alphanumeric "
                "characters, dots, hyphens, slashes, and @ signs are allowed."
            )
        if not safe_pattern.match(target_version):
            return (
                f"ERROR: Invalid target_version '{target_version}'. Only "
                "alphanumeric characters, dots, hyphens, slashes, @, ~, ^, and * "
                "are allowed."
            )
        if dependency_type not in ("dependencies", "devDependencies", "overrides"):
            return (
                "ERROR: dependency_type must be strictly one of: "
                "'dependencies', 'devDependencies', or 'overrides'."
            )

        try:
            rel_manifest = _validate_workspace_path(manifest_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        if Path(rel_manifest).name != "package.json":
            return "ERROR: manifest_path must point to a package.json file."
        allowed_manifest_paths = allowed_manifest_paths_by_package.get(package_name)
        if not allowed_manifest_paths:
            known_packages = ", ".join(sorted(allowed_manifest_paths_by_package))
            return (
                f"ERROR: package_name '{package_name}' is not an allowed target for "
                f"this batch. Allowed package_name values: {known_packages}."
            )
        if rel_manifest not in allowed_manifest_paths:
            allowed = ", ".join(sorted(allowed_manifest_paths))
            return (
                f"ERROR: manifest_path '{rel_manifest}' is not an allowed target for "
                f"package '{package_name}'. Allowed manifest_path values: {allowed}."
            )

        package_expr = f"{dependency_type}[{package_name}]={target_version}"
        npm_cmd = shlex.join(["npm", "pkg", "set", package_expr])
        workspace_dir = _workspace_dir_for_manifest(rel_manifest)
        if workspace_dir == "/workspace":
            cmd_str = npm_cmd
        else:
            cmd_str = f"cd {shlex.quote(workspace_dir)} && {npm_cmd}"

        logger.info(
            "remedy_tools: modifying npm dependency in sandbox: %s", cmd_str
        )
        result = sandbox.run(cmd_str)
        if result.exit_code == 0:
            touched_files.add(rel_manifest)
            return (
                "SUCCESS: Natively updated "
                f"{dependency_type}.{package_name} to {target_version} in {rel_manifest}."
            )
        return (
            f"FAILURE: Failed to modify npm dependency in {rel_manifest} "
            f"(exit {result.exit_code}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return modify_npm_dependency


def _make_validate_manifest_sync_tool(
    sandbox: DockerSandbox,
    target_manifest_paths: Iterable[str],
):
    manifest_paths = _normalize_manifest_targets(target_manifest_paths)

    @tool
    def validate_manifest_sync() -> str:
        """
        Validate that target package manifests can synchronize without scripts.
        """
        if not manifest_paths:
            return "FAILURE: No target manifest paths were provided for validation."

        for manifest_path in manifest_paths:
            workspace_dir = _workspace_dir_for_manifest(manifest_path)
            cmd = (
                f"cd {shlex.quote(workspace_dir)} && "
                "npm install --package-lock-only --ignore-scripts"
            )
            result = sandbox.run(cmd, timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS)
            if result.exit_code != 0:
                return (
                    f"FAILURE: Manifest sync failed for {manifest_path} "
                    f"(exit {result.exit_code}).\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

        return (
            "SUCCESS: Manifest synchronization succeeded for "
            f"{', '.join(manifest_paths)}."
        )

    return validate_manifest_sync


def _make_deterministic_search_replace_tool(
    sandbox: DockerSandbox,
    touched_files: Set[str],
):
    @tool
    def deterministic_search_replace(
        file_path: str,
        old_text: str,
        new_text: str,
    ) -> str:
        """
        Apply an exact one-time search/replace to a workspace file.
        """
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        current = sandbox.read_file(rel_path)
        if current is None:
            return (
                f"ERROR: Could not read '{rel_path}'. Use inspect_ast_symbol or "
                "search_codebase_pattern to verify the current file content."
            )

        newline_style = _detect_newline_style(current)
        current_norm = _normalise_newlines(current)
        old_norm = _normalise_newlines(old_text)
        new_norm = _normalise_newlines(new_text)

        count = current_norm.count(old_norm)
        if count == 0:
            return (
                "ERROR: old_text not found. Use inspect_ast_symbol or "
                "search_codebase_pattern to verify your anchor."
            )
        if count > 1:
            return "ERROR: old_text found multiple times. Make anchor more specific."

        updated = current_norm.replace(old_norm, new_norm, 1)
        sandbox.write_file(rel_path, _restore_newlines(updated, newline_style))
        touched_files.add(rel_path)
        return f"SUCCESS: File modified: {rel_path}"

    return deterministic_search_replace


def _make_search_codebase_pattern_tool(sandbox: DockerSandbox):
    @tool
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Lexically search for an extended-regex pattern across workspace files."""
        if not search_pattern or not search_pattern.strip():
            return "ERROR: search_pattern is required."

        td = (target_directory or ".").strip()
        if td != ".":
            try:
                _validate_workspace_path(td)
            except ValueError as exc:
                return f"ERROR: {exc}"

        safe_pattern = search_pattern.replace("'", "'\"'\"'")
        search_root = f"/workspace/{td}" if td != "." else "/workspace"
        cmd = (
            f"grep -RInE "
            f"--include='*.js' --include='*.ts' --include='*.jsx' --include='*.tsx' "
            f"--include='*.mjs' --include='*.cjs' --include='*.json' "
            f"--exclude-dir=node_modules --exclude-dir=.git "
            f"-- '{safe_pattern}' '{search_root}'"
        )

        try:
            result = sandbox.run(cmd, timeout=_SEARCH_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: search failed: {exc}"

        if result.exit_code == 1 and not result.stdout.strip():
            return f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

        if result.exit_code not in (0, 1):
            return (
                f"ERROR: grep exited {result.exit_code}.\n"
                f"stderr: {result.stderr.strip()[:500]}"
            )

        output = result.stdout
        if len(output.encode()) > _SEARCH_MAX_BYTES:
            truncated_output = output.encode()[:_SEARCH_MAX_BYTES].decode(errors="replace")
            last_nl = truncated_output.rfind("\n")
            truncated_output = truncated_output[:last_nl] if last_nl != -1 else truncated_output
            output = truncated_output + "\n... (output truncated at 32 KB)"

        return output.strip() or f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

    return search_codebase_pattern


def _make_inspect_ast_symbol_tool(sandbox: DockerSandbox):
    @tool
    def inspect_ast_symbol(
        file_path: str,
        symbol_name: str,
        line_hint: int = 0,
    ) -> str:
        """Extract the full source text of a named function, class, or method from a workspace file."""
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: Could not read '{rel_path}'. Verify the path exists."

        try:
            from src.tools.code_map import (
                find_named_symbol,
                language_for_path,
                parse_source,
            )
        except ImportError:
            return "ERROR: code_map module is unavailable."

        lang = language_for_path(rel_path)
        if lang is None:
            return (
                f"ERROR: No AST parser available for '{rel_path}'. "
                "Only JS/TS files (.js, .jsx, .ts, .tsx, .mjs, .cjs) are supported."
            )

        source_bytes = content.encode("utf-8", errors="replace")
        tree = parse_source(source_bytes, lang)
        if tree is None:
            return "ERROR: tree-sitter is unavailable; cannot parse AST."

        hint = int(line_hint) if line_hint else None
        try:
            result = find_named_symbol(
                tree.root_node,
                symbol_name,
                source_bytes,
                line_hint=hint,
            )
        except ValueError as exc:
            return f"ERROR: {exc}"
        except RuntimeError as exc:
            return f"ERROR: {exc}"

        if result is None:
            return (
                f"NOT FOUND: Symbol '{symbol_name}' not found in '{rel_path}'. "
                "Use search_codebase_pattern to locate the correct file."
            )

        node_text = result["text"]
        if len(node_text) > _INSPECT_TEXT_MAX_CHARS:
            node_text = node_text[:_INSPECT_TEXT_MAX_CHARS] + "\n... (truncated)"

        return (
            f"SYMBOL: {result['symbol_name']}\n"
            f"TYPE  : {result['node_type']}\n"
            f"LINES : {result['start_line']}-{result['end_line']}\n"
            f"FILE  : {rel_path}\n"
            f"---\n"
            f"{node_text}"
        )

    return inspect_ast_symbol


def _make_validate_code_syntax_tool(sandbox: DockerSandbox):
    @tool
    def validate_code_syntax(file_path: str) -> str:
        """Validate syntax for a JS/TS-family source file inside the workspace."""
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        suffix = Path(rel_path).suffix.lower()
        if suffix in {".js", ".mjs", ".cjs"}:
            cmd = f"node -c {shlex.quote(f'/workspace/{rel_path}')}"
        elif suffix in {".ts", ".tsx", ".jsx"}:
            cmd = (
                f"npx --yes esbuild {shlex.quote(rel_path)} --outfile=/dev/null"
            )
        else:
            return (
                f"ERROR: validate_code_syntax does not support '{rel_path}'. "
                "Supported extensions are .js, .mjs, .cjs, .ts, .tsx, and .jsx."
            )

        result = sandbox.run(cmd, timeout=_SYNTAX_CHECK_TIMEOUT_SECONDS)
        if result.exit_code == 0:
            return f"SUCCESS: Syntax validation passed for {rel_path}."
        return (
            f"FAILURE: Syntax validation failed for {rel_path} "
            f"(exit {result.exit_code}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return validate_code_syntax


def build_update_toolbelt(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
    target_manifest_paths: Iterable[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    enable_registry_lookup: bool = False,
) -> List:
    """Build the strict update-only toolbelt."""
    manifest_paths = _normalize_manifest_targets(target_manifest_paths)
    toolbelt = [
        _make_read_repository_map_tool(sandbox),
        _make_modify_npm_dependency_tool(
            sandbox,
            touched_files,
            package_manifest_paths,
        ),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        _make_validate_manifest_sync_tool(sandbox, manifest_paths),
    ]
    if enable_registry_lookup:
        toolbelt.append(view_npm_package_versions)
    return toolbelt


def build_workaround_toolbelt(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
) -> List:
    """Build the strict workaround-only toolbelt."""
    return [
        _make_read_repository_map_tool(sandbox),
        _make_search_codebase_pattern_tool(sandbox),
        _make_inspect_ast_symbol_tool(sandbox),
        _make_deterministic_search_replace_tool(sandbox, touched_files),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        _make_validate_code_syntax_tool(sandbox),
    ]
