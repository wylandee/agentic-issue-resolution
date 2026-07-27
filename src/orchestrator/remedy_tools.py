"""
remedy_tools.py - Specialized native workspace tools for Phase 5 subagents.
"""

from __future__ import annotations

import logging
import os
import requests
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set

from langchain_core.tools import tool

from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_MANIFEST_SYNC_TIMEOUT_SECONDS = 120
_SYNTAX_CHECK_TIMEOUT_SECONDS = 30

_REPO_MAP_MAX_ENTRIES = 400
_READ_FILE_MAX_LINES = 200
_READ_FILE_MAX_BYTES = 16_384
_SEARCH_MAX_BYTES = 32_768
_SEARCH_TIMEOUT_SECONDS = 15
_INSPECT_TEXT_MAX_CHARS = 8_000

_SERPER_SEARCH_URL = "https://google.serper.dev/search"
_SERPER_REQUEST_TIMEOUT = 10
_SERPER_MAX_RESULTS = 3
_SEARCH_WEB_MAX_CALLS = 3

_JINA_READER_URL_PREFIX = "https://r.jina.ai/"
_READ_WEB_PAGE_TIMEOUT = 15
_READ_WEB_PAGE_MAX_CHARS = 16_000


def _validate_workspace_path(file_path: str) -> str:
    candidate = (file_path or "").strip()
    if not candidate:
        raise ValueError("file_path is required.")
    if os.path.isabs(candidate) or candidate.startswith(("/", "\\")):
        raise ValueError(f"Rejected absolute file path '{candidate}'.")
    
    parts = Path(candidate).parts
    if ".." in parts:
        raise ValueError(f"Rejected path traversal in '{candidate}'.")
    if parts and parts[0] in ("build", "dist"):
        raise ValueError(f"Accessing compiled files in '{parts[0]}/' is strictly forbidden. Please modify the original source files instead.")
        
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
            "find . -not -path '*/node_modules/*' "
            "-not -path '*/.git/*' "
            "-not -name '*.map' "
            "| sed 's|^./||' | sort"
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


def _make_read_workspace_file_tool(sandbox: DockerSandbox):
    @tool
    def read_workspace_file(
        file_path: str,
        start_line: int = 1,
        end_line: int = 0,
    ) -> str:
        """Read the contents of a workspace file, optionally scoped to a line range.

        Returns the file content with line numbers prefixed (e.g., '  42: code').
        If end_line is 0 or omitted, reads from start_line to the end of the file
        (capped at 200 lines). Use this tool to inspect file context after
        search_codebase_pattern identifies a relevant file and line number.
        """
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        content = sandbox.read_file(rel_path)
        if content is None:
            return (
                f"ERROR: Could not read '{rel_path}'. "
                "Use search_codebase_pattern or read_repository_map to verify the path."
            )

        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return f"FILE: {rel_path} (empty file, 0 lines)"

        s = max(1, int(start_line))
        if end_line <= 0:
            e = min(s + _READ_FILE_MAX_LINES - 1, total_lines)
        else:
            e = min(int(end_line), total_lines)

        if s > total_lines:
            return f"ERROR: start_line {s} exceeds file length ({total_lines} lines)."

        selected = lines[s - 1 : e]

        width = len(str(e))
        numbered = [f"{str(i).rjust(width)}: {line}" for i, line in enumerate(selected, start=s)]
        output = "\n".join(numbered)

        if len(output.encode("utf-8", errors="replace")) > _READ_FILE_MAX_BYTES:
            output = output.encode("utf-8", errors="replace")[:_READ_FILE_MAX_BYTES].decode(
                errors="replace"
            )
            last_nl = output.rfind("\n")
            if last_nl > 0:
                output = output[:last_nl]
            output += "\n... (output truncated at 16 KB)"

        header = f"FILE: {rel_path} (lines {s}-{e} of {total_lines})"
        return f"{header}\n{output}"

    return read_workspace_file


def _make_revert_workspace_file_tool(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
):
    @tool
    def revert_workspace_file(file_path: str, package_name: Optional[str] = None) -> str:
        """
        Restore a workspace file to its original host baseline state.

        If package_name is provided and the file is package.json, only revert the specified package's version to baseline.
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

        if package_name:
            if not rel_path.endswith("package.json"):
                return "ERROR: package_name can only be specified for package.json files."

            import json
            try:
                baseline_data = json.loads(content)
            except Exception as exc:
                return f"ERROR: Failed to parse baseline package.json: {exc}"

            sandbox_content = sandbox.read_file(rel_path)
            if not sandbox_content:
                return f"ERROR: Sandbox file '{rel_path}' is missing or unreadable."

            try:
                sandbox_data = json.loads(sandbox_content)
            except Exception as exc:
                return f"ERROR: Failed to parse sandbox package.json: {exc}"

            if not isinstance(baseline_data, dict) or not isinstance(sandbox_data, dict):
                return "ERROR: package.json is not a valid JSON object."

            reverted_any = False
            for dep_type in ("dependencies", "devDependencies", "overrides"):
                baseline_deps = baseline_data.get(dep_type)
                sandbox_deps = sandbox_data.get(dep_type)

                if isinstance(baseline_deps, dict) and package_name in baseline_deps:
                    if not isinstance(sandbox_deps, dict):
                        sandbox_data[dep_type] = {}
                        sandbox_deps = sandbox_data[dep_type]
                    sandbox_deps[package_name] = baseline_deps[package_name]
                    reverted_any = True
                else:
                    if isinstance(sandbox_deps, dict) and package_name in sandbox_deps:
                        del sandbox_deps[package_name]
                        reverted_any = True

            if not reverted_any:
                return f"NOTE: Package '{package_name}' was not found/modified in '{rel_path}'."

            try:
                new_content = json.dumps(sandbox_data, indent=2) + "\n"
                sandbox.write_file(rel_path, new_content)
            except Exception as exc:
                return f"ERROR: Failed to write updated package.json to sandbox: {exc}"

            if sandbox_data == baseline_data:
                touched_files.discard(rel_path)

            return f"SUCCESS: Reverted dependency '{package_name}' in '{rel_path}' to its baseline state."

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
    attempted_versions_by_package: Optional[Mapping[str, Set[str]]] = None,
    override_required_packages: Optional[Iterable[str]] = None,
    require_planning_answers: bool = False,
    planning_state: Optional[Dict[str, bool]] = None,
    execution_state: Optional[Dict[str, int | bool]] = None,
):
    allowed_manifest_paths_by_package = {
        package_name: set(manifest_paths)
        for package_name, manifest_paths in _normalize_package_manifest_targets(
            package_manifest_paths
        ).items()
    }
    override_required_package_names = {
        package_name.strip()
        for package_name in (override_required_packages or [])
        if package_name and package_name.strip()
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
        if (
            package_name in override_required_package_names
            and dependency_type != "overrides"
        ):
            return (
                f"ERROR: Package '{package_name}' is constrained to npm overrides for "
                "this task. Retry modify_npm_dependency with dependency_type='overrides'. "
                "No manifest changes were made."
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
            if execution_state is not None:
                execution_state["edits_started"] = True
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
    execution_state: Optional[Dict[str, int | bool]] = None,
):
    manifest_paths = _normalize_manifest_targets(target_manifest_paths)

    @tool
    def validate_manifest_sync() -> str:
        """
        Validate the final target package manifests once without scripts.
        """
        if execution_state is not None:
            calls = int(execution_state.get("validation_calls", 0))
            if calls >= 1:
                return "ERROR: validate_manifest_sync may only be called once per update worker run."
            if not execution_state.get("edits_started", False):
                return "ERROR: validate_manifest_sync is only allowed after manifest edits."
            execution_state["validation_calls"] = calls + 1
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
        search_root = td
        cmd = (
            f"grep -RInE "
            f"--include='*.js' --include='*.ts' --include='*.jsx' --include='*.tsx' "
            f"--include='*.mjs' --include='*.cjs' --include='*.json' "
            f"--exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build --exclude-dir=dist "
            f"-- '{safe_pattern}' '{search_root}' | sed 's|^./||'"
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


def _make_run_typecheck_tool(sandbox: DockerSandbox):
    @tool
    def run_typecheck() -> str:
        """Run TypeScript compilation check to catch type errors and import failures."""
        try:
            result = sandbox.run("npx tsc --noEmit 2>&1 | head -50", timeout=60)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: Typecheck execution failed: {exc}"

        if result.exit_code == 0:
            return "SUCCESS: TypeScript compilation passed cleanly."
        output = result.stdout.strip()
        if len(output) > 2000:
            output = output[:2000] + "\n... (truncated)"
        return f"TYPECHECK ERRORS (exit code {result.exit_code}):\n{output}"

    return run_typecheck


def build_update_toolbelt(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
    target_manifest_paths: Iterable[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    enable_registry_lookup: bool = False,
    attempted_versions_by_package: Optional[Mapping[str, Set[str]]] = None,
    override_required_packages: Optional[Iterable[str]] = None,
    require_planning_answers: bool = False,
    planning_state: Optional[Dict[str, bool]] = None,
    execution_state: Optional[Dict[str, int | bool]] = None,
) -> List:
    """Build the strict update-only toolbelt."""
    manifest_paths = _normalize_manifest_targets(target_manifest_paths)
    toolbelt = [
        _make_read_repository_map_tool(sandbox),
        _make_modify_npm_dependency_tool(
            sandbox,
            touched_files,
            package_manifest_paths,
            attempted_versions_by_package=attempted_versions_by_package,
            override_required_packages=override_required_packages,
            require_planning_answers=require_planning_answers,
            planning_state=planning_state,
            execution_state=execution_state,
        ),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        _make_validate_manifest_sync_tool(sandbox, manifest_paths, execution_state),
    ]
    return toolbelt


def build_workaround_toolbelt(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    host_repo_root: Path,
) -> List:
    """Build the strict workaround-only toolbelt."""
    return [
        _make_record_plan_tool(),
        _make_search_web_tool(),
        _make_read_web_page_tool(),
        _make_read_repository_map_tool(sandbox),
        _make_read_workspace_file_tool(sandbox),
        _make_search_codebase_pattern_tool(sandbox),
        _make_inspect_ast_symbol_tool(sandbox),
        _make_deterministic_search_replace_tool(sandbox, touched_files),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        _make_validate_code_syntax_tool(sandbox),
        _make_run_typecheck_tool(sandbox),
    ]

def _make_record_plan_tool():
    @tool
    def record_plan(investigation_findings: str, exact_code_changes: str) -> str:
        """
        Record your investigation findings and your exact plan for code changes.
        You MUST call this tool and document your exact 'old_text' and 'new_text' BEFORE executing any deterministic_search_replace.
        """
        logger.debug("Workaround subagent findings: %s", investigation_findings)
        logger.debug("Workaround subagent planned changes: %s", exact_code_changes)
        return "Plan recorded successfully. You may proceed with deterministic_search_replace."

    return record_plan


def _make_search_web_tool():
    """Create a web search tool backed by Serper.dev."""
    _calls_remaining = [_SEARCH_WEB_MAX_CALLS]  # mutable container for closure

    @tool
    def search_web(query: str) -> str:
        """
        Search the web using Google for documentation, migration guides,
        changelogs, or breaking changes related to the vulnerability fix.

        Use this tool to research how a library's API changed between versions
        before writing any code changes.

        Limited to 3 calls per session. Returns up to 3 result snippets.
        """
        if _calls_remaining[0] <= 0:
            return f"ERROR: search_web call limit reached (max {_SEARCH_WEB_MAX_CALLS} per session). Use the results you already have."

        api_key = os.environ.get("SERPER_API_KEY", "").strip()
        if not api_key:
            return "ERROR: SERPER_API_KEY is not set. Cannot perform web search."

        _calls_remaining[0] -= 1

        try:
            resp = requests.post(
                _SERPER_SEARCH_URL,
                json={"q": query},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=_SERPER_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            organic = data.get("organic") or []

            results = []
            for item in organic[:_SERPER_MAX_RESULTS]:
                if not isinstance(item, dict):
                    continue
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                results.append(f"**{title}**\n{snippet}\nURL: {link}")

            if not results:
                return "No results found for this query. Try a different search query."

            calls_left = _calls_remaining[0]
            header = f"Found {len(results)} results ({calls_left} searches remaining):\n\n"
            return header + "\n\n---\n\n".join(results)

        except Exception as exc:
            logger.warning("search_web failed: %s", exc)
            return f"ERROR: Web search failed - {exc}. Try again or proceed without web results."

    return search_web


def _make_read_web_page_tool():
    """Create a tool to fetch full markdown content of a web page using Jina Reader."""
    @tool
    def read_web_page(url: str) -> str:
        """
        Fetch the full readable markdown text of a web page given its URL.

        Use this tool after search_web to read complete migration guides, breaking change lists,
        or documentation pages found in search results.
        """
        target_url = (url or "").strip()
        if not target_url:
            return "ERROR: url is required."

        jina_url = f"{_JINA_READER_URL_PREFIX}{target_url}"
        try:
            resp = requests.get(
                jina_url,
                headers={"Accept": "text/plain"},
                timeout=_READ_WEB_PAGE_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.text or ""
            if not text.strip():
                return f"No readable content extracted from {target_url}."

            if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                text = text[:_READ_WEB_PAGE_MAX_CHARS] + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"

            return f"--- Markdown content of {target_url} ---\n\n{text}"

        except Exception as exc:
            logger.warning("read_web_page failed for %s: %s", target_url, exc)
            return f"ERROR: Failed to read web page {target_url} - {exc}"

    return read_web_page
