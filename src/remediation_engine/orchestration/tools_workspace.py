"""Workspace file, repository-map, search, AST inspection, and revert tools."""

from __future__ import annotations

from ._tool_support import (
    _INSPECT_TEXT_MAX_CHARS,
    _READ_FILE_MAX_BYTES,
    _READ_FILE_MAX_LINES,
    _REPO_MAP_MAX_ENTRIES,
    _SEARCH_MAX_BYTES,
    _SEARCH_TIMEOUT_SECONDS,
    Any,
    DockerSandbox,
    Path,
    _validate_workspace_path,
    resolve_repository_path,
    tool,
)


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


def _make_read_workspace_file_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any] | None = None,
):
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

        if plan_state is not None:
            plan_state.setdefault("read_files", set()).add(rel_path)
            plan_state.setdefault("inspected_files", set()).add(rel_path)
            plan_state["local_investigation_complete"] = True
            plan_state.setdefault("evidence_ledger", []).append(f"workspace:{rel_path}")

        content = sandbox.read_file(rel_path)
        if content is None:
            return (
                f"ERROR: Could not read '{rel_path}'. "
                "Use search_codebase_pattern or read_repository_map to verify the path."
            )

        if plan_state is not None and rel_path.startswith("node_modules/"):
            plan_state["has_authoritative_evidence"] = True
            plan_state["evidence_source"] = rel_path

        lines = content.splitlines()
        total_lines = len(lines)

        if total_lines == 0:
            return f"FILE: {rel_path} (empty file, 0 lines)"

        s = max(1, int(start_line))
        if end_line <= 0:
            e = min(s + _READ_FILE_MAX_LINES - 1, total_lines)
        else:
            requested_end = int(end_line)
            if requested_end < s:
                return f"ERROR: end_line {requested_end} precedes start_line {s}."
            e = min(requested_end, s + _READ_FILE_MAX_LINES - 1, total_lines)

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
    touched_files: set[str],
    host_repo_root: Path,
):
    @tool
    def revert_workspace_file(file_path: str, package_name: str | None = None) -> str:
        """
        Restore a workspace file to its original host baseline state.

        If package_name is provided and the file is package.json, only revert the specified package's version to baseline.
        """
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        try:
            baseline_file = resolve_repository_path(host_repo_root, rel_path)
        except ValueError as exc:
            return f"ERROR: Baseline path '{rel_path}' is blocked: {exc}"
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
            for dep_type in (
                "dependencies",
                "devDependencies",
                "peerDependencies",
                "optionalDependencies",
                "overrides",
                "resolutions",
            ):
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

            baseline_pnpm = baseline_data.get("pnpm")
            sandbox_pnpm = sandbox_data.get("pnpm")
            baseline_pnpm_overrides = (
                baseline_pnpm.get("overrides") if isinstance(baseline_pnpm, dict) else None
            )
            sandbox_pnpm_overrides = (
                sandbox_pnpm.get("overrides") if isinstance(sandbox_pnpm, dict) else None
            )
            if isinstance(sandbox_pnpm_overrides, dict) and package_name in sandbox_pnpm_overrides:
                if (
                    isinstance(baseline_pnpm_overrides, dict)
                    and package_name in baseline_pnpm_overrides
                ):
                    sandbox_pnpm_overrides[package_name] = baseline_pnpm_overrides[package_name]
                else:
                    del sandbox_pnpm_overrides[package_name]
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


def _make_search_codebase_pattern_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Lexically search for an extended-regex pattern across workspace source files."""
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
            f"--include='*.mjs' --include='*.cjs' "
            f"--exclude-dir=node_modules --exclude-dir=.git --exclude-dir=build --exclude-dir=dist "
            f"--exclude-dir=data --exclude-dir=reports --exclude-dir=.pytest_cache "
            f"-- '{safe_pattern}' '{search_root}' | sed 's|^./||'"
        )

        try:
            result = sandbox.run(cmd, timeout=_SEARCH_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: search failed: {exc}"

        if result.exit_code == 1 and not result.stdout.strip():
            return f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

        if result.exit_code not in (0, 1):
            return f"ERROR: grep exited {result.exit_code}.\nstderr: {result.stderr.strip()[:500]}"

        output = result.stdout
        if plan_state is not None:
            plan_state["local_investigation_complete"] = True
            plan_state.setdefault("evidence_ledger", []).append(f"code-search:{search_pattern}")

        if len(output.encode()) > _SEARCH_MAX_BYTES:
            truncated_output = output.encode()[:_SEARCH_MAX_BYTES].decode(errors="replace")
            last_nl = truncated_output.rfind("\n")
            truncated_output = truncated_output[:last_nl] if last_nl != -1 else truncated_output
            output = truncated_output + "\n... (output truncated at 32 KB)"

        return output.strip() or f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

    return search_codebase_pattern


def _make_inspect_ast_symbol_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any] | None = None,
):
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
            from remediation_engine.tools.code_map import (
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
        except (ValueError, RuntimeError) as exc:
            return f"ERROR: {exc}"

        if result is None:
            return (
                f"NOT FOUND: No declared function, class, or method named "
                f"'{symbol_name}' was found in '{rel_path}'. Do not retry the "
                "same symbol. Imported identifiers and package names are not "
                "AST symbols. Use search_codebase_pattern to find the relevant "
                "call site, then inspect its enclosing declared symbol, or use "
                "read_workspace_file and document the fallback in record_plan. "
                "For an arrow-function binding, pass the variable name only after "
                "confirming it is declared in this file; deterministic_replace_ast_symbol "
                "accepts either the expression body or a complete enclosing declaration."
            )

        if plan_state is not None:
            plan_state.setdefault("inspected_symbols", set()).add(f"{rel_path}:{symbol_name}")
            plan_state.setdefault("inspected_files", set()).add(rel_path)
            plan_state["local_investigation_complete"] = True
            plan_state.setdefault("evidence_ledger", []).append(f"ast:{rel_path}:{symbol_name}")

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


__all__ = [name for name in globals() if not name.startswith("__")]
