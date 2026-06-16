"""
remedy_tools.py - Native workspace tools for the Phase 5 Remedy Agent.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Set

from langchain_core.tools import tool

from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_NPM_INSTALL_TIMEOUT_SECONDS = 900
_NPM_TEST_TIMEOUT_SECONDS = 600
_ODC_TIMEOUT_SECONDS = 300
_ODC_REPORT_NAME = "dependency-check-report.json"
_ODC_CACHE_VOLUME = "odc-cache"

# Exploration tool limits
_REPO_MAP_MAX_ENTRIES = 400       # max lines in read_repository_map output
_SEARCH_MAX_BYTES = 32_768        # 32 KB output cap on search_codebase_pattern
_SEARCH_TIMEOUT_SECONDS = 15      # per-grep timeout
_INSPECT_TEXT_MAX_CHARS = 8_000   # character cap on inspect_ast_symbol node text


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

def _read_report_from_workspace(sandbox: DockerSandbox) -> Optional[str]:
    try:
        return sandbox.read_file(_ODC_REPORT_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("remedy_tools: failed to read ODC report from workspace - %s", exc)
        return None


def _parse_report_identifiers(report_text: str) -> Optional[Set[str]]:
    try:
        report = json.loads(report_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("remedy_tools: failed to decode ODC report JSON - %s", exc)
        return None

    try:
        from src.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning("remedy_tools: src.tools.odc_parser not importable.")
        return None

    identifiers: Set[str] = set()
    for issue in parse_vulnerabilities(report):
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _run_odc(workspace_volume: str) -> "subprocess.CompletedProcess[str]":
    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        "root",
        "-v",
        f"{workspace_volume}:/scan",
        "-v",
        f"{_ODC_CACHE_VOLUME}:/usr/share/dependency-check/data",
        "owasp/dependency-check:latest",
        "--project",
        "sandbox_scan",
        "--scan",
        "/scan",
        "--format",
        "JSON",
        "--out",
        "/scan",
        "--noupdate",
    ]

    extra_args = os.environ.get("ODC_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    logger.info("remedy_tools: running ODC in Docker: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_ODC_TIMEOUT_SECONDS,
    )


def build_agent_tools(
    sandbox: DockerSandbox,
    touched_files: Set[str],
    target_cves: Set[str],
    host_repo_root: Path,
) -> List:
    """
    Build the native LangChain tool set for the active workspace sandbox.

    ``target_cves`` intentionally keeps its historical name, but the set may
    contain both CVE and GHSA identifiers for scan validation parity.
    """

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
            return "ERROR: old_text found multiple times. Make anchor more specific."

        updated = current_norm.replace(old_norm, new_norm, 1)
        sandbox.write_file(rel_path, _restore_newlines(updated, newline_style))
        touched_files.add(rel_path)
        return f"SUCCESS: File modified: {rel_path}"

    @tool
    def run_dependency_install() -> str:
        """Runs npm install to synchronize node_modules with package.json edits. MUST run after edits and before testing/scanning."""
        result = sandbox.run(
            "npm install --package-lock=true",
            timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
        )
        if result.exit_code == 0:
            return "SUCCESS: npm install completed successfully."
        return (
            f"FAILURE: npm install failed (exit {result.exit_code}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    @tool
    def run_security_scan() -> str:
        """Executes OWASP Dependency-Check on the workspace to verify if target CVEs are resolved."""
        workspace_volume = getattr(sandbox, "_workspace_volume", None)
        if not workspace_volume:
            return "FAILURE: workspace_volume is unavailable for security scanning."
        if shutil.which("docker") is None:
            return "FAILURE: docker is not available on PATH, so Dependency-Check cannot run."

        try:
            proc = _run_odc(workspace_volume)
        except FileNotFoundError:
            return "FAILURE: docker is not available on PATH, so Dependency-Check cannot run."
        except subprocess.TimeoutExpired:
            return f"FAILURE: Dependency-Check timed out after {_ODC_TIMEOUT_SECONDS}s."
        except Exception as exc:  # noqa: BLE001
            return f"FAILURE: Dependency-Check subprocess error - {exc}"

        report_text = _read_report_from_workspace(sandbox)
        found_identifiers = (
            _parse_report_identifiers(report_text) if report_text is not None else None
        )

        if proc.returncode != 0 and found_identifiers is None:
            return (
                f"FAILURE: Dependency-Check exited {proc.returncode} and produced no "
                "parseable report.\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

        if found_identifiers is None:
            return (
                f"FAILURE: Dependency-Check report was not parseable (exit {proc.returncode}).\n"
                f"stderr:\n{proc.stderr}"
            )

        remaining = {identifier.upper().strip() for identifier in target_cves if identifier}
        remaining &= found_identifiers
        if remaining:
            return (
                "FAILURE: Dependency-Check still reports the following target "
                f"vulnerability identifier(s): {', '.join(sorted(remaining))}"
            )

        return "SUCCESS: Dependency-Check found no remaining target vulnerability identifiers."

    @tool
    def revert_workspace_file(file_path: str) -> str:
        """
        Discard all edits made to a workspace file and restore it to its original host baseline state.
        Use this tool to resolve "blind alley" corruption loops (like invalid JSON formatting or syntax errors).
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

        if rel_path in touched_files:
            touched_files.remove(rel_path)

        return f"SUCCESS: Reverted workspace file '{rel_path}' to its baseline state."

    @tool
    def modify_npm_dependency(
        package_name: str,
        target_version: str,
        dependency_type: str,
    ) -> str:
        """
        Natively modify npm dependencies, devDependencies, or overrides in package.json using npm pkg set.
        This ensures perfect JSON syntax and structural validity without manual text search/replace.
        """
        import re
        import shlex

        # Check package_name and target_version for safety (allow alphanumeric, dots, hyphens, slashes, and @)
        safe_pattern = re.compile(r"^[a-zA-Z0-9.\-/@]+$")
        if not safe_pattern.match(package_name):
            return f"ERROR: Invalid package_name '{package_name}'. Only alphanumeric characters, dots, hyphens, slashes, and @ signs are allowed."
        if not safe_pattern.match(target_version):
            return f"ERROR: Invalid target_version '{target_version}'. Only alphanumeric characters, dots, hyphens, slashes, and @ signs are allowed."

        # Verify dependency_type
        if dependency_type not in ("dependencies", "devDependencies", "overrides"):
            return "ERROR: dependency_type must be strictly one of: 'dependencies', 'devDependencies', or 'overrides'."

        # Build and execute the command safely
        args = ["npm", "pkg", "set", f"{dependency_type}[{package_name}]={target_version}"]
        cmd_str = shlex.join(args)

        logger.info("remedy_tools: modifying npm dependency in sandbox: %s", cmd_str)
        result = sandbox.run(cmd_str)

        if result.exit_code == 0:
            touched_files.add("package.json")
            return f"SUCCESS: Natively updated {dependency_type}.{package_name} to {target_version} in package.json."
        else:
            return (
                f"FAILURE: Failed to modify npm dependency (exit {result.exit_code}).\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    @tool
    def run_unit_tests() -> str:
        """Runs npm test inside the workspace to verify functionality."""
        result = sandbox.run("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)
        if result.exit_code == 0:
            return "SUCCESS: npm test passed."
        return (
            f"FAILURE: npm test failed (exit {result.exit_code}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # ------------------------------------------------------------------
    # Read-only codebase exploration tools
    # ------------------------------------------------------------------

    @tool
    def read_repository_map() -> str:
        """Return a deterministic ASCII tree of every file/directory in the workspace.

        Use this as your FIRST action to understand repository structure before
        searching for specific files or symbols. Output is capped at
        400 entries to stay context-sized.
        """
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

    @tool
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Lexical search for an extended-regex pattern across workspace files.

        Args:
            search_pattern: Extended regex (ERE) to search for. Must not be empty.
            target_directory: Repo-relative subdirectory to search within.
                              Defaults to "." (entire workspace). Must not be
                              absolute or traverse outside the workspace.

        Returns matched lines with file:line format, capped at 32 KB. Use this
        to locate which file defines a symbol before calling inspect_ast_symbol,
        or to verify a pattern appears after an edit.

        Prefer inspect_ast_symbol over read_workspace_file for large source files.
        """
        if not search_pattern or not search_pattern.strip():
            return "ERROR: search_pattern is required."

        # Validate target_directory
        td = (target_directory or ".").strip()
        if td != ".":
            try:
                _validate_workspace_path(td)
            except ValueError as exc:
                return f"ERROR: {exc}"

        # Build the grep command inside the sandbox (node:22, has grep with -E)
        # Use single-quoted pattern passed via sh -c to avoid shell expansion issues.
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
            # grep exit 1 = no matches
            return f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

        if result.exit_code not in (0, 1):
            return (
            f"ERROR: grep exited {result.exit_code}.\n"
            f"stderr: {result.stderr.strip()[:500]}"
        )

        output = result.stdout
        if len(output.encode()) > _SEARCH_MAX_BYTES:
            # Truncate to byte budget and mark
            truncated_output = output.encode()[:_SEARCH_MAX_BYTES].decode(errors="replace")
            # Trim to last complete line
            last_nl = truncated_output.rfind("\n")
            truncated_output = truncated_output[:last_nl] if last_nl != -1 else truncated_output
            output = truncated_output + "\n... (output truncated at 32 KB)"

        return output.strip() or f"NO MATCH: Pattern '{search_pattern}' not found in '{td}'."

    @tool
    def inspect_ast_symbol(
        file_path: str,
        symbol_name: str,
        line_hint: int = 0,
    ) -> str:
        """Extract the full source text of a named function, class, or method from a workspace file.

        Prefer this over read_workspace_file when you need to inspect a specific
        symbol in a large source file; it avoids loading the entire file into
        context.

        Args:
            file_path: Repo-relative path to the source file (JS/TS only).
            symbol_name: Exact name of the symbol to inspect (case-sensitive).
            line_hint: Optional 1-indexed line number near the symbol. Used to
                       disambiguate when the name appears multiple times.
                       Pass 0 to omit.

        Returns a structured report with the symbol's line range and source text,
        or an ERROR string if the file cannot be read / parsed / found.
        """
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
            result = find_named_symbol(tree.root_node, symbol_name, source_bytes, line_hint=hint)
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

    return [
        read_workspace_file,
        deterministic_search_replace,
        revert_workspace_file,
        modify_npm_dependency,
        run_dependency_install,
        run_security_scan,
        run_unit_tests,
        read_repository_map,
        search_codebase_pattern,
        inspect_ast_symbol,
    ]
