"""
remedy_tools.py - Specialized native workspace tools for Phase 5 subagents.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import uuid
from base64 import b64decode
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import requests
from langchain_core.tools import tool

from remediation_engine.contracts.schemas import (
    FailureCategory,
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundExecutionPhase,
    WorkaroundPlannedReplacement,
    WorkaroundValidationResult,
    WorkaroundValidationStatus,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_MANIFEST_SYNC_TIMEOUT_SECONDS = 120
_SYNTAX_CHECK_TIMEOUT_SECONDS = 30
_NPM_TEST_TIMEOUT_SECONDS = 180
_LINT_CHECK_TIMEOUT_SECONDS = 60
_RUNTIME_SMOKE_TIMEOUT_SECONDS = 30

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
_GITHUB_API_URL_PREFIX = "https://api.github.com/"
_READ_WEB_PAGE_TIMEOUT = 15
_READ_WEB_PAGE_MAX_CHARS = 16_000

_SOURCE_MODULE_SUFFIXES = frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})
_TEST_DIRECTORY_NAMES = frozenset({"test", "tests", "__tests__"})

# Runtime smoke must prove that the changed module can load.  Application
# entrypoints are deliberately excluded because importing them can start a
# server, connect to a database, or perform other work unrelated to the
# changed behavior.  These markers are intentionally conservative: a module
# is rejected only when its source contains an unmistakable bootstrap call.
_RUNTIME_BOOTSTRAP_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:app|server)\.listen\s*\(", re.IGNORECASE), "starts an application listener"),
    (re.compile(r"\b(?:app|server)\.(?:start|run)\s*\(", re.IGNORECASE), "starts the application"),
    (
        re.compile(r"\bvalidateDependencies(?:Basic)?\s*\(", re.IGNORECASE),
        "runs dependency/bootstrap validation",
    ),
    (
        re.compile(r"\b(?:initialize|init)(?:Database|Db|Server|App)\s*\(", re.IGNORECASE),
        "initializes application infrastructure",
    ),
    (
        re.compile(
            r"\b(?:mongoose|sequelize|prisma)\s*\.?(?:connect|initialize)\s*\(", re.IGNORECASE
        ),
        "connects to application infrastructure",
    ),
)


def _is_authoritative_evidence_source(source: str) -> bool:
    """Return whether an evidence source URL or path is authoritative."""
    candidate = (source or "").strip().lower()
    if not candidate:
        return False
    if any(
        untrusted in candidate
        for untrusted in ("stackoverflow.com", "stackexchange.com", "reddit.com", "snippet-only")
    ):
        return False

    authoritative_markers = (
        "github.com/",
        "github.com/advisories",
        "raw.githubusercontent.com",
        "nvd.nist.gov",
        "cve.org",
        "osv.dev",
        "security.snyk.io",
        "npmjs.com/package/",
        "registry.npmjs.org",
        "expressjs.com",
        "jwt.io",
        "node_modules/",
        "readme",
        "index.d.ts",
        "package.json",
        "official advisory",
        "package repository",
        "package docs",
        "npm registry",
        "installed package",
    )
    return any(marker in candidate for marker in authoritative_markers)


def _is_infrastructure_failure(error_text: str) -> bool:
    """Classify whether a test execution error is strictly an infrastructure failure."""
    text = (error_text or "").strip()
    if not text:
        return False
    lowered = text.lower()

    # A missing bare package is a test-runner/install precondition failure.
    # Relative and absolute paths remain code failures because the workaround
    # may have introduced a bad local import.  The substitution gate records
    # the exact diagnostic and independently checks that the replacement does
    # not import the unavailable package.
    missing_module = re.search(
        r"cannot find module\s+['\"]([^'\"]+)['\"]|module not found[^'\"]*['\"]([^'\"]+)['\"]",
        lowered,
    )
    if missing_module:
        module_name = next((group for group in missing_module.groups() if group), "")
        if module_name and not module_name.startswith((".", "/", "file:")):
            return True

    # Code failure markers MUST NOT qualify as infrastructure-only failures
    code_failure_markers = (
        "assertionerror",
        "assert ",
        "expect(",
        "should equal",
        "expected ",
        "to.equal",
        "to.be",
        "syntaxerror",
        "typeerror",
        "referenceerror",
        "uncaught exception",
        "test failed",
        "tests failed",
        "failing test:",
        "1 failing",
        "2 failing",
    )
    if any(marker in lowered for marker in code_failure_markers):
        return False

    infra_markers = (
        "sqlite3",
        "better-sqlite3",
        "node-gyp",
        "prebuild-install",
        "bindings",
        "native module",
        "compiled binding",
        "dlopen",
        "failed to map segment",
        "cannot open shared object",
        "library not loaded",
        "image not found",
        "no suitable image found",
        "missing native binding",
        "missing binary",
        "command not found: sqlite3",
        "command not found",
        "err_dlopen_failed",
        "cannot find module 'sqlite3'",
        "cannot find module 'better-sqlite3'",
        "command timed out after",
        "timed out after",
    )
    return any(marker in lowered for marker in infra_markers)


def _runtime_smoke_bootstrap_reason(file_path: str, source_content: str) -> str | None:
    """Return why a source module is unsafe for import-only runtime smoke.

    Args:
        file_path: Repository-relative source path being considered.
        source_content: Current source contents for ``file_path``.

    Returns:
        A short reason when the module appears to bootstrap the application;
        otherwise ``None``.
    """
    normalized = _normalise_newlines(source_content)
    for pattern, reason in _RUNTIME_BOOTSTRAP_MARKERS:
        if pattern.search(normalized):
            return reason
    return None


def _select_lightweight_runtime_smoke_target(
    requested_file: str,
    candidate_files: Sequence[str],
    targeted_test_file: str | None,
    sandbox: DockerSandbox,
) -> tuple[str | None, str | None]:
    """Select a safe source module for import-only runtime smoke.

    An explicitly supplied test, compiled artifact, missing file, or invalid
    path is rejected.  If the requested source module is an application
    bootstrap, the selector falls back to the first changed source module
    that can be imported without starting the application.  The caller records
    the selected path in validation state so the evidence reflects what ran.

    Args:
        requested_file: Agent-provided runtime smoke path.
        candidate_files: Current modified source files to consider as a safe
            fallback, in deterministic order.
        targeted_test_file: Targeted test path, if already selected.
        sandbox: Workspace sandbox used to read source contents.

    Returns:
        ``(selected_path, note)`` on success. ``note`` explains a fallback
        selection. On failure, returns ``(None, diagnostic)``.
    """
    normalized_requested, path_error = _runtime_smoke_path_error(
        requested_file,
        targeted_test_file=targeted_test_file,
    )
    if path_error:
        return None, path_error
    assert normalized_requested is not None

    requested_content = sandbox.read_file(normalized_requested)
    if requested_content is None:
        return None, (
            f"Source module '{normalized_requested}' could not be found. "
            "Use read_repository_map or read_workspace_file to resolve a source path."
        )

    requested_reason = _runtime_smoke_bootstrap_reason(
        normalized_requested,
        requested_content,
    )
    if requested_reason is None:
        return normalized_requested, None

    normalized_candidates: list[str] = []
    for raw_path in candidate_files:
        try:
            candidate = _validate_workspace_path(str(raw_path))
        except ValueError:
            continue
        if candidate in normalized_candidates:
            continue
        if candidate in (normalized_requested, targeted_test_file):
            continue
        if Path(candidate).suffix.lower() not in _SOURCE_MODULE_SUFFIXES:
            continue
        if _is_test_file_path(candidate):
            continue
        content = sandbox.read_file(candidate)
        if content is None:
            continue
        if _runtime_smoke_bootstrap_reason(candidate, content) is None:
            normalized_candidates.append(candidate)

    if normalized_candidates:
        selected = normalized_candidates[0]
        return selected, (
            f"Requested runtime smoke target '{normalized_requested}' was not used because it "
            f"{requested_reason}; selected lightweight source module '{selected}' instead."
        )

    return None, (
        f"Runtime smoke target '{normalized_requested}' is unsafe because it {requested_reason}, "
        "and no lightweight changed source module was available as a replacement."
    )


def _is_test_file_path(file_path: str) -> bool:
    """Return whether a repository-relative path identifies a test/spec file."""
    path = Path(file_path.replace("\\", "/"))
    lowered_parts = {part.lower() for part in path.parts[:-1]}
    filename = path.name.lower()
    return bool(
        lowered_parts & _TEST_DIRECTORY_NAMES
        or ".test." in filename
        or ".spec." in filename
        or filename.endswith((".test", ".spec"))
    )


def _runtime_smoke_path_error(
    runtime_file: str,
    targeted_test_file: str | None = None,
) -> tuple[str | None, str | None]:
    """Validate and normalize a lightweight source-module smoke target."""
    try:
        rel_path = _validate_workspace_path(runtime_file)
    except ValueError as exc:
        return None, str(exc)

    if Path(rel_path).suffix.lower() not in _SOURCE_MODULE_SUFFIXES:
        return None, (
            f"Runtime smoke target '{rel_path}' must be a JavaScript/TypeScript source module."
        )
    if _is_test_file_path(rel_path):
        return None, (
            f"Runtime smoke target '{rel_path}' is a test/spec file. "
            "Choose a lightweight source module; runtime smoke and targeted tests must be separate."
        )

    if targeted_test_file:
        try:
            normalized_target = _validate_workspace_path(targeted_test_file)
        except ValueError:
            normalized_target = targeted_test_file.replace("\\", "/").strip().lstrip("/")
        if rel_path == normalized_target:
            return None, (
                f"Runtime smoke target '{rel_path}' is also the targeted test. "
                "Use a lightweight source module for runtime smoke."
            )
    return rel_path, None


def _validate_workspace_path(file_path: str) -> str:
    candidate = (file_path or "").strip().replace("\\", "/")
    if not candidate:
        raise ValueError("file_path is required.")

    if candidate.startswith("/workspace/"):
        candidate = candidate[len("/workspace/") :]
    elif candidate.startswith("workspace/"):
        candidate = candidate[len("workspace/") :]

    if os.path.isabs(candidate) or candidate.startswith("/"):
        raise ValueError(f"Rejected absolute file path '{candidate}'.")

    parts = Path(candidate).parts
    if ".." in parts:
        raise ValueError(f"Rejected path traversal in '{candidate}'.")
    if parts and parts[0] in ("build", "dist"):
        raise ValueError(
            f"Accessing compiled files in '{parts[0]}/' is strictly forbidden. Please modify the original source files instead."
        )

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


def _normalize_manifest_targets(target_manifest_paths: Iterable[str]) -> list[str]:
    """Return stable, validated package.json targets for one update batch."""
    manifest_paths = sorted(
        {_validate_workspace_path(path) for path in target_manifest_paths if path}
    )
    invalid = [path for path in manifest_paths if Path(path).name != "package.json"]
    if invalid:
        raise ValueError(
            f"All target manifest paths must point to package.json files. Invalid values: {invalid}"
        )
    return manifest_paths


def _normalize_package_manifest_targets(
    package_manifest_paths: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    """Return validated package-to-manifest targets for one update batch."""
    normalized: dict[str, list[str]] = {}
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
            phase = plan_state.get("phase")
            last_result = plan_state.get("last_validation_result")
            last_status = getattr(last_result, "overall_status", None) or (
                last_result.get("overall_status") if isinstance(last_result, dict) else None
            )
            if phase == WorkaroundExecutionPhase.VALIDATE.value and last_status not in (
                WorkaroundValidationStatus.INFRA_FAILURE,
                "INFRA_FAILURE",
            ):
                return (
                    "ERROR: [PHASE_VIOLATION] A source edit must be followed by "
                    "validate_workaround before further investigation."
                )
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
    touched_files: set[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    attempted_versions_by_package: Mapping[str, set[str]] | None = None,
    override_required_packages: Iterable[str] | None = None,
    require_planning_answers: bool = False,
    planning_state: dict[str, bool] | None = None,
    execution_state: dict[str, int | bool] | None = None,
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
        if package_name in override_required_package_names and dependency_type != "overrides":
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

        logger.info("remedy_tools: modifying npm dependency in sandbox: %s", cmd_str)
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
    execution_state: dict[str, int | bool] | None = None,
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
                return (
                    "ERROR: validate_manifest_sync may only be called once per update worker run."
                )
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

        return f"SUCCESS: Manifest synchronization succeeded for {', '.join(manifest_paths)}."

    return validate_manifest_sync


def _is_prohibited_target(rel_path: str) -> bool:
    """Check if file is a manifest or test file that workaround subagent must not modify."""
    norm = rel_path.replace("\\", "/").lstrip("/")
    basename = Path(norm).name.lower()

    if basename in (
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pom.xml",
        "build.gradle",
        "requirements.txt",
        "pyproject.toml",
    ):
        return True

    parts = norm.lower().split("/")
    if any(p in ("test", "tests", "__tests__", "spec", "specs") for p in parts):
        return True

    return any(
        basename.endswith(ext)
        for ext in (
            ".test.js",
            ".spec.js",
            ".test.ts",
            ".spec.ts",
            ".test.jsx",
            ".spec.jsx",
            ".test.tsx",
            ".spec.tsx",
            "_test.py",
            "_spec.py",
        )
    )


def _apply_replacements_to_content(
    content: str,
    replacements: list[WorkaroundPlannedReplacement],
) -> tuple[str, str | None]:
    """Apply a list of non-overlapping planned replacements to file content."""
    newline_style = _detect_newline_style(content)
    norm_content = _normalise_newlines(content)

    all_intervals: list[tuple[int, int, str]] = []

    for r in replacements:
        old_norm = _normalise_newlines(r.old_text)
        new_norm = _normalise_newlines(r.new_text)

        if old_norm == new_norm:
            return (
                content,
                "ERROR: [NO_OP_EDIT] Replacement cannot be a no-op (old_text equals new_text).",
            )

        indices: list[int] = []
        start_idx = 0
        while True:
            idx = norm_content.find(old_norm, start_idx)
            if idx == -1:
                break
            indices.append(idx)
            start_idx = idx + max(1, len(old_norm))

        if len(indices) != r.expected_occurrences:
            return (
                content,
                f"ERROR: [OCCURRENCE_MISMATCH] Expected {r.expected_occurrences} occurrence(s) of old_text, but found {len(indices)}.",
            )

        for idx in indices:
            all_intervals.append((idx, idx + len(old_norm), new_norm))

    sorted_intervals = sorted(all_intervals, key=lambda x: x[0])
    for i in range(1, len(sorted_intervals)):
        prev_start, prev_end, _ = sorted_intervals[i - 1]
        curr_start, curr_end, _ = sorted_intervals[i]
        if curr_start < prev_end:
            return (
                content,
                "ERROR: [OVERLAPPING_REPLACEMENTS] Overlapping replacement spans detected.",
            )

    cur = norm_content
    for start, end, replacement_text in sorted(all_intervals, key=lambda x: x[0], reverse=True):
        cur = cur[:start] + replacement_text + cur[end:]

    updated_content = _restore_newlines(cur, newline_style)
    return (updated_content, None)


def _make_deterministic_apply_edit_set_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_apply_edit_set(
        replacements: list[WorkaroundPlannedReplacement],
    ) -> str:
        """Apply one atomic edit set containing exact planned replacements.

        The edit set is written immediately but remains pending until
        ``validate_workaround`` returns ``PASS``. A code-failing validation
        restores the pre-iteration snapshot, so a revised plan must include
        every required change from the failed set again. Infrastructure or
        blocked validation results retain the pending set for recovery; a
        passing validation promotes it to the validated cumulative patch.

        Each item in ``replacements`` must be a flat replacement object. Do not
        nest replacements under a file-level ``path`` or ``replacements`` key.

        Example payload::

            {
                "replacements": [
                    {
                        "file_path": "src/auth.ts",
                        "old_text": "const oldName = 1",
                        "new_text": "const newName = 1",
                        "expected_occurrences": 1
                    }
                ]
            }
        """
        if plan_state is not None:
            if not plan_state.get("recorded", False) or "planned_replacements" not in plan_state:
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits with deterministic_apply_edit_set."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit set is permitted per iteration before validation. Call validate_workaround now."

            if plan_state.get("require_authoritative_evidence") and not plan_state.get(
                "has_authoritative_evidence"
            ):
                return (
                    "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                    "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                    "or the installed package's README/types in node_modules."
                )

        raw_list = replacements
        if isinstance(raw_list, str):
            try:
                raw_list = json.loads(raw_list)
            except Exception:
                return (
                    "ERROR: [INVALID_REPLACEMENTS] replacements must be a valid JSON array or list."
                )

        if not isinstance(raw_list, list) or not raw_list:
            return "ERROR: [INVALID_REPLACEMENTS] replacements must be a non-empty list."

        submitted_planned: list[WorkaroundPlannedReplacement] = []
        for idx, item in enumerate(raw_list):
            try:
                if isinstance(item, WorkaroundPlannedReplacement):
                    r_obj = item
                elif isinstance(item, dict):
                    r_obj = WorkaroundPlannedReplacement(**item)
                else:
                    return f"ERROR: [INVALID_REPLACEMENTS] Item at index {idx} is invalid."
                submitted_planned.append(r_obj)
            except Exception as exc:
                return f"ERROR: [INVALID_REPLACEMENTS] Item at index {idx} failed validation: {exc}"

        recorded_data = plan_state.get("planned_replacements", []) if plan_state else []
        recorded_planned = [WorkaroundPlannedReplacement(**d) for d in recorded_data]

        if [r.model_dump() for r in submitted_planned] != [
            r.model_dump() for r in recorded_planned
        ]:
            return "ERROR: [PLAN_MISMATCH] Submitted replacements do not match the recorded plan's planned_replacements exactly."

        affected_files_set: set[str] = set()
        for r in submitted_planned:
            try:
                rel_path = _validate_workspace_path(r.file_path)
            except ValueError as exc:
                return f"ERROR: {exc}"
            if _is_prohibited_target(rel_path):
                return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."
            affected_files_set.add(rel_path)

        if len(submitted_planned) > 16:
            return f"ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 16 replacements per edit set (got {len(submitted_planned)})."
        if len(affected_files_set) > 8:
            return f"ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 8 affected files per edit set (got {len(affected_files_set)})."
        total_text_bytes = sum(len(r.new_text.encode("utf-8")) for r in submitted_planned)
        if total_text_bytes > 65536:
            return (
                "ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 64 KiB of combined replacement text allowed."
            )

        if plan_state is not None:
            inspected = (
                plan_state.get("inspected_files", set())
                | plan_state.get("read_files", set())
                | plan_state.get("fallback_files", set())
            )
            for rel_path in affected_files_set:
                suffix = Path(rel_path).suffix.lower()
                if (
                    suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
                    and rel_path not in inspected
                ):
                    return (
                        f"ERROR: [MISSING_INSPECTION] AST inspection required before edit on '{rel_path}'. "
                        "Use inspect_ast_symbol or read_workspace_file first."
                    )

        file_snapshots: dict[str, str] = {}
        for rel_path in affected_files_set:
            curr = sandbox.read_file(rel_path)
            if curr is None:
                return f"ERROR: Could not read '{rel_path}'."
            file_snapshots[rel_path] = curr

        replacements_by_file: dict[str, list[WorkaroundPlannedReplacement]] = {}
        for r in submitted_planned:
            norm_p = _validate_workspace_path(r.file_path)
            replacements_by_file.setdefault(norm_p, []).append(r)

        new_file_contents: dict[str, str] = {}
        for rel_path, file_repls in replacements_by_file.items():
            curr = file_snapshots[rel_path]
            updated, err = _apply_replacements_to_content(curr, file_repls)
            if err:
                return f"ERROR: Failed applying replacements to '{rel_path}': {err}"
            new_file_contents[rel_path] = updated

        written_files: list[str] = []
        try:
            for rel_path, new_content in new_file_contents.items():
                sandbox.write_file(rel_path, new_content)
                written_files.append(rel_path)
        except Exception as exc:
            for rel_path, orig in file_snapshots.items():
                sandbox.write_file(rel_path, orig)
            return (
                f"ERROR: [WRITE_FAILURE] Write failed on '{rel_path}': {exc}. All files restored."
            )

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
        for rel_path in new_file_contents:
            syntax_res = syntax_tool.invoke({"file_path": rel_path})
            if "FAILURE" in syntax_res or "ERROR" in syntax_res:
                for p, orig in file_snapshots.items():
                    sandbox.write_file(p, orig)
                return f"ERROR: [SYNTAX_FAILURE] Replacement produced invalid syntax in '{rel_path}'. All files in edit set restored.\n{syntax_res}"

        plan_rev = plan_state.get("plan_revision", 1) if plan_state else 1
        iteration = plan_state.get("iteration", 1) if plan_state else 1
        patch_id = f"patch_r{plan_rev}_i{iteration}_{uuid.uuid4().hex[:8]}"

        edits: list[WorkaroundEdit] = []
        for idx, r in enumerate(submitted_planned):
            norm_p = _validate_workspace_path(r.file_path)
            edits.append(
                WorkaroundEdit(
                    file_path=norm_p,
                    old_text=r.old_text,
                    new_text=r.new_text,
                    symbol_name=r.symbol_name,
                    patch_id=patch_id,
                    replacement_index=idx,
                    expected_occurrences=r.expected_occurrences,
                    edit_index=idx,
                )
            )

        edit_set = WorkaroundEditSet(
            patch_id=patch_id,
            plan_revision=plan_rev,
            iteration=iteration,
            affected_files=sorted(list(new_file_contents.keys())),
            replacements=edits,
        )

        touched_files.update(new_file_contents.keys())
        if plan_state is not None:
            plan_state["pending_edit_set"] = edit_set
            plan_state["pending_snapshots"] = file_snapshots
            plan_state["current_iteration_edit"] = edits[0] if edits else None
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value

        res_dict = {
            "status": "SUCCESS",
            "patch_id": patch_id,
            "affected_files": sorted(list(new_file_contents.keys())),
            "replacement_count": len(edits),
            "phase_transition": "EXECUTE -> VALIDATE",
        }
        return (
            f"SUCCESS: Atomic edit set '{patch_id}' applied successfully and is pending validation. "
            "A CODE_FAILURE will restore the pre-iteration checkpoint; a PASS will commit this edit set.\n"
            f"JSON: {json.dumps(res_dict)}"
        )

    return deterministic_apply_edit_set


def _make_deterministic_search_replace_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_search_replace(
        file_path: str,
        old_text: str,
        new_text: str,
        symbol_name: str | None = None,
    ) -> str:
        """
        Apply an exact one-time search/replace to a workspace file.
        """
        if plan_state is not None:
            if not plan_state.get("recorded", False):
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits with deterministic_search_replace."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit is permitted per iteration before validation. Call validate_workaround now."

        if (
            plan_state is not None
            and plan_state.get("require_authoritative_evidence")
            and not plan_state.get("has_authoritative_evidence")
        ):
            return (
                "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                "or the installed package's README/types in node_modules."
            )

        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        if _is_prohibited_target(rel_path):
            return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

        planned_files = {
            str(path).replace("\\", "/").lstrip("/")
            for path in (plan_state or {}).get("planned_files", [])
        }
        if (
            plan_state is not None
            and plan_state.get("recorded")
            and planned_files
            and rel_path not in planned_files
        ):
            return (
                f"ERROR: [PLAN_VIOLATION] '{rel_path}' is outside the recorded workaround plan. "
                "Re-run record_plan with every causally related source file before editing it; "
                "do not apply an isolated or unrelated fix."
            )

        suffix = Path(rel_path).suffix.lower()
        if suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}:
            inspected_files = plan_state.get("inspected_files", set()) if plan_state else set()
            fallback_files = plan_state.get("fallback_files", set()) if plan_state else set()
            if rel_path not in inspected_files and rel_path not in fallback_files:
                return (
                    f"ERROR: [MISSING_INSPECTION] AST inspection required before first edit on '{rel_path}'. "
                    "Use inspect_ast_symbol for the target symbol, or use read_workspace_file "
                    "and document a no-symbol fallback in record_plan before editing."
                )

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

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
        syntax_res = syntax_tool.invoke({"file_path": rel_path})
        if "FAILURE" in syntax_res or "ERROR" in syntax_res:
            sandbox.write_file(rel_path, current)
            return f"ERROR: Replacement produced invalid syntax in '{rel_path}'. Edit reverted.\n{syntax_res}"

        touched_files.add(rel_path)
        if plan_state is not None:
            edits = plan_state.setdefault("successful_edits", [])
            edit_obj = WorkaroundEdit(
                file_path=rel_path,
                old_text=old_text,
                new_text=new_text,
                symbol_name=symbol_name,
                edit_index=len(edits),
            )
            edits.append(edit_obj)
            plan_state["current_iteration_edit"] = edit_obj
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value
        return f"SUCCESS: File modified: {rel_path}"

    return deterministic_search_replace


def _make_deterministic_replace_ast_symbol_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_replace_ast_symbol(
        file_path: str,
        symbol_name: str,
        replacement: str,
        line_hint: int = 0,
    ) -> str:
        """
        Replace the complete AST body of a declared function, class, or method.
        """
        if plan_state is not None:
            if not plan_state.get("recorded", False):
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit is permitted per iteration before validation. Call validate_workaround now."

        if (
            plan_state is not None
            and plan_state.get("require_authoritative_evidence")
            and not plan_state.get("has_authoritative_evidence")
        ):
            return (
                "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                "or the installed package's README/types in node_modules."
            )

        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        if _is_prohibited_target(rel_path):
            return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

        planned_files = {
            str(path).replace("\\", "/").lstrip("/")
            for path in (plan_state or {}).get("planned_files", [])
        }
        if (
            plan_state is not None
            and plan_state.get("recorded")
            and planned_files
            and rel_path not in planned_files
        ):
            return (
                f"ERROR: [PLAN_VIOLATION] '{rel_path}' is outside the recorded workaround plan. "
                "Re-run record_plan with every causally related source file before editing it; "
                "do not apply an isolated or unrelated fix."
            )

        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: Could not read '{rel_path}'."

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
            return f"ERROR: No AST parser available for '{rel_path}'."

        source_bytes = content.encode("utf-8", errors="replace")
        tree = parse_source(source_bytes, lang)
        if tree is None:
            return "ERROR: tree-sitter unavailable."

        hint = int(line_hint) if line_hint else None
        try:
            res = find_named_symbol(tree.root_node, symbol_name, source_bytes, line_hint=hint)
        except (ValueError, RuntimeError) as exc:
            return f"ERROR: {exc}"

        if res is None:
            return (
                f"NOT FOUND: No declared function, class, or method named '{symbol_name}' found in '{rel_path}'. "
                "Imported identifiers and package names are not AST symbols. "
                "Use search_codebase_pattern to find the call site and inspect its "
                "enclosing declared symbol. For an arrow-function binding, pass the "
                "variable name only after confirming it is declared in this file; "
                "this tool accepts either the expression body or a complete enclosing declaration."
            )

        old_text = res["text"]

        # tree-sitter reports an arrow function's expression as the symbol,
        # but models often correctly reason in terms of the complete exported
        # declaration.  If the replacement is declaration-shaped, expand the
        # replacement scope to the enclosing export/lexical declaration.  This
        # avoids producing invalid text such as ``export const name = const
        # name = ...`` while retaining expression-only replacement semantics.
        replacement_stripped = replacement.strip()
        declaration_shaped = bool(
            re.match(
                r"^(?:export\s+(?:default\s+)?)?(?:const|let|var|function|class)\b",
                replacement_stripped,
            )
        )
        if res.get("node_type") == "arrow_function" and declaration_shaped:
            start_byte = int(res.get("start_byte", -1))
            end_byte = int(res.get("end_byte", -1))
            enclosing_nodes = []
            if start_byte >= 0 and end_byte >= 0:
                stack = [tree.root_node]
                while stack:
                    node = stack.pop()
                    stack.extend(getattr(node, "children", []) or [])
                    if (
                        node.start_byte <= start_byte
                        and node.end_byte >= end_byte
                        and node.type
                        in {
                            "variable_declarator",
                            "lexical_declaration",
                            "variable_declaration",
                            "export_statement",
                        }
                    ):
                        node_text = node.text
                        if isinstance(node_text, bytes):
                            node_text = node_text.decode("utf-8", errors="replace")
                        if symbol_name in str(node_text):
                            enclosing_nodes.append(node)
            if enclosing_nodes:
                replacement_scope = max(
                    enclosing_nodes,
                    key=lambda node: node.end_byte - node.start_byte,
                )
                raw_scope = replacement_scope.text
                old_text = (
                    raw_scope.decode("utf-8", errors="replace")
                    if isinstance(raw_scope, bytes)
                    else str(raw_scope)
                )
            else:
                return (
                    f"ERROR: Replacement for arrow symbol '{symbol_name}' is a complete declaration, "
                    "but its enclosing declaration could not be identified. "
                    "Retry with only the arrow/function expression body."
                )
        if old_text not in content:
            return f"ERROR: Symbol text for '{symbol_name}' could not be cleanly anchored in '{rel_path}'."

        count = content.count(old_text)
        if count > 1:
            return f"ERROR: Symbol text for '{symbol_name}' appears {count} times in '{rel_path}'. Provide a line_hint."

        updated = content.replace(old_text, replacement, 1)

        sandbox.write_file(rel_path, updated)

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
        syntax_res = syntax_tool.invoke({"file_path": rel_path})
        if "FAILURE" in syntax_res or "ERROR" in syntax_res:
            sandbox.write_file(rel_path, content)
            return f"ERROR: Replacement produced invalid syntax in '{rel_path}'. Edit reverted.\n{syntax_res}"

        touched_files.add(rel_path)
        if plan_state is not None:
            plan_state.setdefault("inspected_files", set()).add(rel_path)
            edits = plan_state.setdefault("successful_edits", [])
            edit_obj = WorkaroundEdit(
                file_path=rel_path,
                old_text=old_text,
                new_text=replacement,
                symbol_name=symbol_name,
                edit_index=len(edits),
            )
            edits.append(edit_obj)
            plan_state["current_iteration_edit"] = edit_obj
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value
        return f"SUCCESS: Symbol '{symbol_name}' in '{rel_path}' successfully replaced."

    return deterministic_replace_ast_symbol


def _make_search_codebase_pattern_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Lexically search for an extended-regex pattern across workspace source files."""
        if plan_state is not None:
            phase = plan_state.get("phase")
            last_result = plan_state.get("last_validation_result")
            last_status = getattr(last_result, "overall_status", None) or (
                last_result.get("overall_status") if isinstance(last_result, dict) else None
            )
            if phase == WorkaroundExecutionPhase.VALIDATE.value and last_status not in (
                WorkaroundValidationStatus.INFRA_FAILURE,
                "INFRA_FAILURE",
            ):
                return (
                    "ERROR: [PHASE_VIOLATION] A source edit must be followed by "
                    "validate_workaround before further investigation."
                )
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
        if plan_state is not None:
            phase = plan_state.get("phase")
            last_result = plan_state.get("last_validation_result")
            last_status = getattr(last_result, "overall_status", None) or (
                last_result.get("overall_status") if isinstance(last_result, dict) else None
            )
            if phase == WorkaroundExecutionPhase.VALIDATE.value and last_status not in (
                WorkaroundValidationStatus.INFRA_FAILURE,
                "INFRA_FAILURE",
            ):
                return (
                    "ERROR: [PHASE_VIOLATION] A source edit must be followed by "
                    "validate_workaround before further investigation."
                )
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


def _make_run_targeted_test_tool(
    sandbox: DockerSandbox,
    preferred_test_files: Sequence[str] | None = None,
    plan_state: dict[str, Any] | None = None,
):
    preferred = tuple(
        path.replace("\\", "/").strip().lstrip("/")
        for path in (preferred_test_files or [])
        if isinstance(path, str) and path.strip()
    )

    @tool
    def run_targeted_test(
        test_file: str,
        test_name: str | None = None,
    ) -> str:
        """
        Run a bounded targeted test for fast diagnostic feedback.
        Only accepts repository-relative test file paths.
        """
        if not test_file or not test_file.strip():
            return "ERROR: test_file is required."

        norm_path = test_file.replace("\\", "/").strip().lstrip("/")
        if ".." in norm_path or norm_path.startswith("/") or norm_path.startswith("C:"):
            return (
                f"ERROR: Invalid test file path '{test_file}'. Must be a repository-relative path."
            )
        if norm_path.startswith(("build/", "dist/")):
            return (
                f"ERROR: Compiled test path '{norm_path}' is not supported. "
                "Use the original source test path under test/, tests/, or the source package directory."
            )

        accepted_alt = (plan_state or {}).get("accepted_alternative_test")
        if preferred and norm_path not in preferred and norm_path != accepted_alt:
            preferred_text = ", ".join(preferred)
            return (
                f"ERROR: Targeted test '{norm_path}' is not the QA-recommended target. "
                f"Run one of these source test files instead: {preferred_text}."
            )

        if sandbox.read_file(norm_path) is None:
            return (
                f"BLOCKED: Targeted test path '{norm_path}' could not be verified in the workspace. "
                "Resolve the source test path from read_repository_map before choosing a replacement."
            )

        from remediation_engine.orchestration.qa_critic import (
            _detect_targeted_test_context,
            build_targeted_test_command,
            extract_qa_failure_evidence,
        )

        runner, package_cwd, npm_invocation = _detect_targeted_test_context(sandbox, norm_path)
        if runner == "npm_text_fallback":
            return (
                f"BLOCKED: Cannot run targeted test on '{norm_path}'. "
                "No safe runner-specific target command can be constructed for runner 'npm_text_fallback'."
            )

        relative_test_file = norm_path
        if package_cwd and norm_path.startswith(package_cwd.rstrip("/") + "/"):
            relative_test_file = norm_path[len(package_cwd.rstrip("/")) + 1 :]
        cmd = build_targeted_test_command(
            runner,
            relative_test_file,
            test_name,
            npm_invocation=npm_invocation,
            package_cwd=package_cwd,
        )
        if not cmd:
            return f"BLOCKED: Could not construct targeted test command for runner '{runner}' and file '{norm_path}'."

        try:
            result = sandbox.run(cmd, timeout=_NPM_TEST_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: Targeted test execution was unavailable - {exc}"

        output = (result.stdout + "\n" + result.stderr).strip()
        lowered_output = output.lower()
        if (
            result.exit_code == 124
            or "command timed out after" in lowered_output
            or "sandbox is not running" in lowered_output
        ):
            return (
                f"FAILURE: Targeted test infrastructure failed ({norm_path}).\n"
                f"Diagnostic:\n{output[:1500]}"
            )

        if result.exit_code == 0:
            name_str = f" [{test_name}]" if test_name else ""
            return f"SUCCESS: Targeted test passed ({runner}): {norm_path}{name_str}\n\nstdout:\n{result.stdout[:1000]}"

        evidence = extract_qa_failure_evidence(result.exit_code, result.stdout, result.stderr)
        diag_str = (
            "\n".join(evidence.exact_diagnostics[:5]) or result.stderr[:500] or result.stdout[:500]
        )
        loc_str = "\n".join(evidence.source_locations[:5])
        test_str = test_name or (evidence.failed_tests[0] if evidence.failed_tests else "unknown")

        return (
            f"FAILURE: Targeted test failed ({runner}): {norm_path} (exit {result.exit_code})\n"
            f"Failing Test: {test_str}\n"
            f"Exact Diagnostics:\n{diag_str}\n"
            f"Source Locations:\n{loc_str}\n"
            f"Raw Excerpt:\n{evidence.raw_excerpt[:1000]}"
        )

    return run_targeted_test


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
            cmd = f"npx --yes esbuild {shlex.quote(rel_path)} --outfile=/dev/null"
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


def _revert_current_iteration_edit(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any],
    touched_files: set[str],
) -> None:
    """Revert the unvalidated edit set made in the current iteration."""
    pending_snapshots = plan_state.get("pending_snapshots", {})
    pending_edit_set = plan_state.get("pending_edit_set")

    if pending_snapshots:
        for file_path, orig_content in pending_snapshots.items():
            try:
                sandbox.write_file(file_path, orig_content)
                touched_files.discard(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to revert unvalidated snapshot edit on %s: %s", file_path, exc
                )
    elif pending_edit_set:
        for edit in reversed(pending_edit_set.replacements):
            try:
                curr = sandbox.read_file(edit.file_path)
                if curr and edit.new_text in curr:
                    updated = curr.replace(edit.new_text, edit.old_text, 1)
                    sandbox.write_file(edit.file_path, updated)
                    touched_files.discard(edit.file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to revert unvalidated edit on %s: %s", edit.file_path, exc)
    else:
        last_edit = plan_state.get("current_iteration_edit")
        if last_edit:
            try:
                curr = sandbox.read_file(last_edit.file_path)
                if curr and last_edit.new_text in curr:
                    updated = curr.replace(last_edit.new_text, last_edit.old_text, 1)
                    sandbox.write_file(last_edit.file_path, updated)
                    touched_files.discard(last_edit.file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to revert unvalidated edit on %s: %s", last_edit.file_path, exc
                )

    plan_state["pending_edit_set"] = None
    plan_state["pending_snapshots"] = {}
    plan_state["current_iteration_edit"] = None

    plan_state["phase"] = WorkaroundExecutionPhase.INVESTIGATE.value
    plan_state["iteration"] = int(plan_state.get("iteration", 1)) + 1
    plan_state["successful_edit_count_this_iteration"] = 0
    plan_state["validated_files"] = []
    plan_state["validation_passed"] = False
    plan_state["accepted_alternative_test"] = None


def _make_validate_workaround_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
    preferred_test_files: Sequence[str] | None = None,
):
    """Build one short-circuiting validation gate for source workarounds.

    The gate runs syntax, typecheck, lint, runtime-import, and targeted-test
    checks in that order. It returns immediately after the first failed gate,
    preserving the exact diagnostic for the worker's next reasoning turn.
    Individual check implementations remain private helpers so update workers
    can retain their manifest-specific validation contract, while workaround
    workers receive one atomic validation tool.
    """
    if plan_state is None:
        plan_state = {}

    syntax_tool = _make_validate_code_syntax_tool(sandbox)
    targeted_test_tool = _make_run_targeted_test_tool(sandbox, preferred_test_files, plan_state)

    def _failure(gate: str, detail: str) -> str:
        return f"FAILURE: Workaround validation gate '{gate}' failed.\n{detail[:4000]}"

    def _record_result(result: WorkaroundValidationResult) -> WorkaroundValidationResult:
        """Persist the complete gate result for the supervisor success contract."""
        plan_state["last_validation_result"] = result
        plan_state["validation_passed"] = result.overall_status == WorkaroundValidationStatus.PASS
        plan_state["validated_files"] = list(result.validated_files)
        if result.infrastructure_diagnostics:
            plan_state["last_infrastructure_diagnostics"] = result.infrastructure_diagnostics
            plan_state["infrastructure_failure_details"] = result.infrastructure_diagnostics
        return result

    def _run_typecheck_gate() -> str:
        tsconfig = sandbox.read_file("tsconfig.json")
        if not isinstance(tsconfig, str) or not tsconfig.strip():
            return "SKIPPED: TypeScript gate (tsconfig.json not found)."
        try:
            result = sandbox.run("npx --no-install tsc --noEmit", timeout=60)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: TypeScript gate execution failed: {exc}"
        if result.exit_code == 0:
            return "SUCCESS: TypeScript compilation passed cleanly."
        output = (result.stdout + "\n" + result.stderr).strip()
        lowered = output.lower()
        if (
            result.exit_code == 127
            or "command not found" in lowered
            or "cannot find module" in lowered
            or "npx: not found" in lowered
            or "err_module_not_found" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return f"BLOCKED: TypeScript gate blocked.\nCommand: npx --no-install tsc --noEmit\nDiagnostic:\n{output}"
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: TypeScript compilation failed (exit {result.exit_code}).\n{output}"

    def _run_lint_gate(source_files: Sequence[str]) -> str:
        source_files = [
            path
            for path in source_files
            if Path(path).suffix.lower() in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}
        ]
        if not source_files:
            return "SKIPPED: Lint gate (no lintable source files were modified)."
        try:
            eslint_probe = sandbox.run("test -x node_modules/.bin/eslint", timeout=10)
        except Exception as exc:  # noqa: BLE001
            return f"FAILURE: Lint gate probe failed: {exc}"
        if eslint_probe.exit_code != 0:
            return "SKIPPED: Lint gate (the repository does not provide eslint)."
        command = "npx --no-install eslint --no-error-on-unmatched-pattern " + " ".join(
            shlex.quote(path) for path in source_files
        )
        try:
            result = sandbox.run(command, timeout=_LINT_CHECK_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"FAILURE: Lint gate execution failed: {exc}"
        if result.exit_code == 0:
            return f"SUCCESS: Lint passed for {', '.join(source_files)}."
        output = (result.stdout + "\n" + result.stderr).strip()
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: Lint failed (exit {result.exit_code}).\n{output}"

    def _run_runtime_smoke_gate(runtime_file: str | None) -> str:
        if not runtime_file or not runtime_file.strip():
            return "FAILURE: Runtime smoke gate failed: runtime_smoke_file must be explicitly supplied for workaround validation."
        rel_path, path_error = _runtime_smoke_path_error(runtime_file)
        if path_error:
            return f"ERROR: [INVALID_RUNTIME_SMOKE] {path_error}"
        assert rel_path is not None
        suffix = Path(rel_path).suffix.lower()

        # Resolve the source path before launching Node. A missing path is a
        # target-selection error, not evidence that the code change is broken.
        source_content = sandbox.read_file(rel_path)
        if source_content is None:
            return (
                f"ERROR: [INVALID_RUNTIME_SMOKE] Source module '{rel_path}' could not be found. "
                "Use read_repository_map or read_workspace_file to resolve a source path."
            )

        import json

        import_expression = json.dumps(f"./{rel_path}")
        script = (
            f"import({import_expression}).catch((error) => {{ "
            "console.error(error?.stack || error); process.exitCode = 1; })"
        )
        loader = "--import tsx " if suffix in {".ts", ".tsx", ".jsx"} else ""
        command = f"node {loader}--input-type=module -e {shlex.quote(script)}"
        try:
            result = sandbox.run(command, timeout=_RUNTIME_SMOKE_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: Runtime smoke gate execution failed: {exc}"
        if result.exit_code == 0:
            return f"SUCCESS: Runtime smoke import passed for {rel_path}."
        output = (result.stdout + "\n" + result.stderr).strip()
        lowered = output.lower()
        if (
            result.exit_code == 124
            or "command timed out after" in lowered
            or "sandbox is not running" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return (
                f"BLOCKED: Runtime smoke gate unavailable for {rel_path}.\n"
                f"Command: {command}\nDiagnostic:\n{output}"
            )
        if suffix in {".ts", ".tsx", ".jsx"} and (
            result.exit_code == 127
            or "cannot find module 'tsx'" in lowered
            or "tsx: command not found" in lowered
            or "err_module_not_found" in lowered
            or "npx: not found" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return (
                f"BLOCKED: Runtime smoke gate blocked.\nCommand: {command}\nDiagnostic:\n{output}"
            )
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: Runtime smoke import failed for {rel_path}.\n{output}"

    @tool
    def validate_workaround(
        modified_files: list[str],
        runtime_smoke_file: str | None = None,
        targeted_test_file: str | None = None,
        targeted_test_name: str | None = None,
    ) -> str:
        """Run all workaround validation gates and stop at the first failure.

        ``modified_files`` must include every source file changed by the
        current cumulative patch. A QA-recommended test is selected
        automatically when the caller omits ``targeted_test_file``.
        ``runtime_smoke_file`` must be a lightweight repository source module;
        test/spec files and compiled ``build/`` or ``dist/`` paths are rejected,
        and the smoke module must be distinct from the targeted test.
        """
        plan_state["validation_calls"] = int(plan_state.get("validation_calls", 0)) + 1

        requested = [
            str(path).replace("\\", "/").strip().lstrip("/")
            for path in (modified_files or [])
            if str(path).strip()
        ]
        current = [
            str(path).replace("\\", "/").strip().lstrip("/")
            for path in touched_files
            if str(path).strip()
        ]
        files = list(dict.fromkeys([*requested, *current]))
        if not files:
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax="FAILURE: No modified files were supplied.",
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            return f"FAILURE: Workaround validation gate 'input' failed. No modified files were supplied.\nJSON: {res.model_dump_json()}"

        prohibited = [path for path in files if _is_prohibited_target(path)]
        if prohibited:
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=f"Prohibited files modified: {', '.join(prohibited)}",
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            return f"FAILURE: Workaround validation gate 'input' failed. Prohibited files included.\nJSON: {res.model_dump_json()}"

        # Reject an explicitly wrong QA target before spending time in the
        # static/runtime gates. This keeps target-selection diagnostics useful
        # even when the caller omitted the explicit smoke-file argument.
        if targeted_test_file and preferred_test_files:
            normalized_target = targeted_test_file.replace("\\", "/").strip().lstrip("/")
            if normalized_target.startswith(("build/", "dist/")):
                return (
                    f"ERROR: Compiled test path '{normalized_target}' is not supported. "
                    "Use the original source test path under test/, tests/, or the source package directory."
                )
            accepted_alt = plan_state.get("accepted_alternative_test")
            preferred = {
                path.replace("\\", "/").strip().lstrip("/") for path in preferred_test_files
            }
            if normalized_target not in preferred and normalized_target != accepted_alt:
                return (
                    f"ERROR: Targeted test '{normalized_target}' is not the QA-recommended target. "
                    f"Run one of these source test files instead: {', '.join(sorted(preferred))}."
                )

        smoke_target = runtime_smoke_file
        smoke_selection_note = ""
        if smoke_target is not None and smoke_target.strip():
            normalized_smoke, smoke_selection_message = _select_lightweight_runtime_smoke_target(
                smoke_target,
                files,
                targeted_test_file=targeted_test_file,
                sandbox=sandbox,
            )
            if normalized_smoke is None:
                return f"ERROR: [INVALID_RUNTIME_SMOKE] {smoke_selection_message}"
            smoke_target = normalized_smoke
            smoke_selection_note = smoke_selection_message or ""

        syntax_msg = ""
        for file_path in files:
            current_content = sandbox.read_file(file_path)
            if current_content is None:
                syntax_msg = f"FAILURE: Modified file '{file_path}' could not be re-read."
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                        syntax=syntax_msg,
                        validated_files=[],
                        failure_category=FailureCategory.BREAKING_CHANGE,
                    )
                )
                _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                return f"FAILURE: Workaround validation gate 'syntax' failed.\n{syntax_msg}\nJSON: {res.model_dump_json()}"
            syntax_result = syntax_tool.invoke({"file_path": file_path})
            if str(syntax_result).startswith("BLOCKED:"):
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.BLOCKED,
                        syntax=str(syntax_result),
                        validated_files=[],
                    )
                )
                return f"{syntax_result}\nJSON: {res.model_dump_json()}"
            if not str(syntax_result).startswith("SUCCESS:"):
                syntax_msg = str(syntax_result)
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                        syntax=syntax_msg,
                        validated_files=[],
                        failure_category=FailureCategory.BREAKING_CHANGE,
                    )
                )
                _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                return f"FAILURE: Workaround validation gate 'syntax' failed.\n{syntax_msg}\nJSON: {res.model_dump_json()}"
            syntax_msg = str(syntax_result)

        typecheck_msg = _run_typecheck_gate()
        if typecheck_msg.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    typecheck=typecheck_msg,
                    validated_files=[],
                )
            )
            return f"{typecheck_msg}\nJSON: {res.model_dump_json()}"
        if typecheck_msg.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'typecheck' failed.\n{typecheck_msg}\nJSON: {res.model_dump_json()}"

        lint_msg = _run_lint_gate(files)
        if lint_msg.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    lint=lint_msg,
                    validated_files=[],
                )
            )
            return f"{lint_msg}\nJSON: {res.model_dump_json()}"
        if lint_msg.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    lint=lint_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'lint' failed.\n{lint_msg}\nJSON: {res.model_dump_json()}"

        if smoke_target is None and not plan_state.get("runtime_smoke_required", False) and files:
            smoke_target = next(
                (
                    path
                    for path in files
                    if Path(path).suffix.lower() in _SOURCE_MODULE_SUFFIXES
                    and not _is_test_file_path(path)
                ),
                None,
            )
        runtime_smoke_result = _run_runtime_smoke_gate(smoke_target)
        smoke_msg = runtime_smoke_result
        if smoke_selection_note:
            smoke_msg = f"{smoke_selection_note}\n{smoke_msg}"
        if smoke_target:
            plan_state["runtime_smoke_file"] = smoke_target
        if runtime_smoke_result.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    runtime_smoke=smoke_msg,
                    validated_files=[],
                    infrastructure_diagnostics=smoke_msg,
                )
            )
            return f"{smoke_msg}\nJSON: {res.model_dump_json()}"
        if runtime_smoke_result.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    lint=lint_msg,
                    runtime_smoke=smoke_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'runtime_smoke' failed.\n{smoke_msg}\nJSON: {res.model_dump_json()}"

        accepted_alt = plan_state.get("accepted_alternative_test")
        test_file = accepted_alt or targeted_test_file
        if not test_file and preferred_test_files:
            test_file = next(iter(preferred_test_files), None)

        if not test_file and plan_state.get("targeted_test_required", False):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    lint=lint_msg,
                    runtime_smoke=smoke_msg,
                    targeted_test="FAILURE: A targeted test file is required for this workaround.",
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            return (
                "FAILURE: Workaround validation gate 'targeted_test' failed: "
                "targeted_test_file must be supplied.\n"
                f"JSON: {res.model_dump_json()}"
            )

        test_msg = ""
        alt_used = bool(accepted_alt)
        if test_file:
            test_res_str = targeted_test_tool.invoke(
                {"test_file": test_file, "test_name": targeted_test_name}
            )
            test_msg = str(test_res_str)
            if test_msg.startswith("BLOCKED:"):
                is_infra = _is_infrastructure_failure(test_msg)
                if is_infra:
                    plan_state["latest_failed_targeted_test"] = test_file
                    plan_state["latest_test_failure_infra"] = True
                    plan_state["latest_infra_diagnostics"] = test_msg
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.BLOCKED,
                        syntax=syntax_msg,
                        typecheck=typecheck_msg,
                        lint=lint_msg,
                        runtime_smoke=smoke_msg,
                        targeted_test=test_msg,
                        targeted_test_file=test_file,
                        alternative_used=alt_used,
                        validated_files=[],
                        infrastructure_diagnostics=test_msg,
                    )
                )
                return f"{test_msg}\nJSON: {res.model_dump_json()}"

            if not test_msg.startswith("SUCCESS:"):
                is_infra = _is_infrastructure_failure(test_msg)
                plan_state["latest_failed_targeted_test"] = test_file
                plan_state["latest_test_failure_infra"] = is_infra
                plan_state["latest_infra_diagnostics"] = test_msg

                if is_infra:
                    res = _record_result(
                        WorkaroundValidationResult(
                            overall_status=WorkaroundValidationStatus.INFRA_FAILURE,
                            syntax=syntax_msg,
                            typecheck=typecheck_msg,
                            lint=lint_msg,
                            runtime_smoke=smoke_msg,
                            targeted_test=test_msg,
                            targeted_test_file=test_file,
                            alternative_used=alt_used,
                            validated_files=[],
                            infrastructure_diagnostics=test_msg,
                        )
                    )
                    return f"FAILURE: Workaround validation gate 'targeted_test' failed (INFRASTRUCTURE_FAILURE).\n{test_msg}\nJSON: {res.model_dump_json()}"
                else:
                    res = _record_result(
                        WorkaroundValidationResult(
                            overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                            syntax=syntax_msg,
                            typecheck=typecheck_msg,
                            lint=lint_msg,
                            runtime_smoke=smoke_msg,
                            targeted_test=test_msg,
                            targeted_test_file=test_file,
                            alternative_used=alt_used,
                            validated_files=[],
                            failure_category=FailureCategory.BREAKING_CHANGE,
                        )
                    )
                    _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                    return f"FAILURE: Workaround validation gate 'targeted_test' failed.\n{test_msg}\nJSON: {res.model_dump_json()}"

        res = _record_result(
            WorkaroundValidationResult(
                overall_status=WorkaroundValidationStatus.PASS,
                syntax=syntax_msg,
                typecheck=typecheck_msg,
                lint=lint_msg,
                runtime_smoke=smoke_msg,
                targeted_test=test_msg,
                targeted_test_file=test_file,
                alternative_used=alt_used,
                alternative_test_mapping_details=dict(
                    plan_state.get("original_to_alternative_test_details", {})
                ),
                validated_files=files,
            )
        )
        plan_state["validated_files"] = files
        plan_state["validation_passed"] = True
        plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value

        pending_edit_set = plan_state.get("pending_edit_set")
        if pending_edit_set is not None:
            plan_state.setdefault("successful_edit_sets", []).append(pending_edit_set)
            plan_state.setdefault("successful_edits", []).extend(pending_edit_set.replacements)
            plan_state["pending_edit_set"] = None
            plan_state["pending_snapshots"] = {}
        return (
            "SUCCESS: Workaround validation gate passed. "
            f"Validated files: {', '.join(files)}; "
            f"runtime smoke: {smoke_target}; "
            f"targeted test: {test_file or 'skipped'}. "
            "The pending edit set is now committed as part of the validated cumulative patch.\n"
            f"JSON: {res.model_dump_json()}"
        )

    return validate_workaround


def build_update_toolbelt(
    sandbox: DockerSandbox,
    touched_files: set[str],
    host_repo_root: Path,
    target_manifest_paths: Iterable[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    enable_registry_lookup: bool = False,
    attempted_versions_by_package: Mapping[str, set[str]] | None = None,
    override_required_packages: Iterable[str] | None = None,
    require_planning_answers: bool = False,
    planning_state: dict[str, bool] | None = None,
    execution_state: dict[str, int | bool] | None = None,
) -> list:
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
    touched_files: set[str],
    host_repo_root: Path,
    plan_state: dict[str, Any] | None = None,
    mandatory_search_terms: dict[str, str] | None = None,
    preferred_test_files: Sequence[str] | None = None,
) -> list:
    """Build the strict workaround-only toolbelt."""
    if plan_state is None:
        plan_state = {}

    if plan_state.get("recorded") and not plan_state.get("phase"):
        plan_state["phase"] = WorkaroundExecutionPhase.EXECUTE.value
    else:
        plan_state.setdefault("phase", WorkaroundExecutionPhase.INVESTIGATE.value)

    plan_state.setdefault("iteration", 1)
    plan_state.setdefault("local_investigation_complete", False)
    plan_state.setdefault("plan_revision", 0)
    plan_state.setdefault("successful_edit_count", 0)
    plan_state.setdefault("successful_edit_count_this_iteration", 0)
    plan_state.setdefault("last_validation_result", None)
    plan_state.setdefault("last_infrastructure_diagnostics", None)
    plan_state.setdefault("infrastructure_failure_details", None)
    plan_state.setdefault("validation_calls", 0)
    plan_state.setdefault(
        "original_targeted_test",
        preferred_test_files[0] if preferred_test_files else None,
    )
    plan_state.setdefault("accepted_alternative_test", None)
    plan_state.setdefault("validated_files", [])
    plan_state.setdefault("inspected_symbols", set())
    plan_state.setdefault("inspected_files", set())
    plan_state.setdefault("fallback_files", set())
    plan_state.setdefault("read_files", set())
    plan_state.setdefault("successful_edits", [])
    plan_state.setdefault("web_search_performed", False)
    plan_state.setdefault("recorded", False)
    # The real workaround toolbelt never permits a successful validation with
    # an inferred smoke target or a skipped QA-targeted test.
    plan_state.setdefault("runtime_smoke_required", True)
    plan_state.setdefault("targeted_test_required", True)

    return [
        _make_record_plan_tool(plan_state),
        _make_record_targeted_test_substitution_tool(sandbox, plan_state),
        _make_search_web_tool(mandatory_search_terms=mandatory_search_terms, plan_state=plan_state),
        _make_read_web_page_tool(plan_state),
        _make_read_repository_map_tool(sandbox),
        _make_read_workspace_file_tool(sandbox, plan_state),
        _make_search_codebase_pattern_tool(sandbox, plan_state),
        _make_inspect_ast_symbol_tool(sandbox, plan_state),
        _make_deterministic_apply_edit_set_tool(sandbox, touched_files, plan_state),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        _make_validate_workaround_tool(sandbox, touched_files, plan_state, preferred_test_files),
    ]


def _make_record_plan_tool(plan_state: dict[str, Any]):
    @tool
    def record_plan(
        affected_files: list[str],
        affected_symbols: list[str],
        security_invariant: str,
        causal_hypothesis: str,
        planned_replacements: list[WorkaroundPlannedReplacement],
        evidence_source: str | None = None,
    ) -> str:
        """Record the evidence-backed plan before applying source edits.

        ``planned_replacements`` must be a flat list of exact replacement
        objects. Do not wrap the list in a file-level ``path`` or
        ``replacements`` object. Each replacement must include
        ``file_path``, ``old_text``, and ``new_text``; ``expected_occurrences``
        defaults to 1.

        Example payload::

            {
                "affected_files": ["src/auth.ts"],
                "affected_symbols": ["authorize"],
                "security_invariant": "JWT authorization remains enforced",
                "causal_hypothesis": "The dependency changed its export shape",
                "planned_replacements": [
                    {
                        "file_path": "src/auth.ts",
                        "old_text": "import oldName from 'auth-lib'",
                        "new_text": "import { newName } from 'auth-lib'",
                        "expected_occurrences": 1
                    }
                ],
                "evidence_source": "workspace:src/auth.ts"
            }

        MUST be called after local investigation is complete and BEFORE
        executing any code edits.
        """
        if plan_state.get("inspected_files") or plan_state.get("read_files"):
            plan_state["local_investigation_complete"] = True

        if not plan_state.get("local_investigation_complete", False):
            return "ERROR: [PLAN_REJECTED] Local codebase investigation must complete before recording a plan."

        phase = plan_state.get("phase", WorkaroundExecutionPhase.INVESTIGATE.value)
        if phase == WorkaroundExecutionPhase.VALIDATE.value:
            return (
                "ERROR: [PHASE_VIOLATION] A plan cannot be recorded while awaiting validation. "
                "Call validate_workaround now; a code failure will restart the loop in INVESTIGATE."
            )
        if phase not in {
            WorkaroundExecutionPhase.INVESTIGATE.value,
            WorkaroundExecutionPhase.PLAN.value,
            WorkaroundExecutionPhase.EXECUTE.value,
        }:
            return f"ERROR: [PHASE_VIOLATION] Cannot record a plan in phase '{phase}'."
        if plan_state.get("pending_edit_set") is not None:
            return (
                "ERROR: [PHASE_VIOLATION] The current edit set has not been validated. "
                "Call validate_workaround before recording another plan."
            )

        sec_inv = str(security_invariant or "").strip()
        causal_hyp = str(causal_hypothesis or "").strip()
        ev_source = str(evidence_source or "").strip()

        if not sec_inv:
            return "ERROR: [PLAN_REJECTED] security_invariant cannot be empty."
        if not causal_hyp:
            return "ERROR: [PLAN_REJECTED] causal_hypothesis cannot be empty."
        if not ev_source:
            return "ERROR: [PLAN_REJECTED] evidence_source is required and cannot be empty."

        if plan_state.get("require_authoritative_evidence"):
            evidence_is_authoritative = bool(
                plan_state.get("has_authoritative_evidence")
                or _is_authoritative_evidence_source(ev_source)
            )
            if not evidence_is_authoritative:
                return (
                    "ERROR: [MISSING_EVIDENCE] Authoritative evidence is required before accepting a "
                    "QA_REGRESSION_REPAIR plan. Set evidence_source to an official advisory, package "
                    "repository/docs URL, npm registry metadata, or an installed package README/types path."
                )

        raw_replacements = planned_replacements
        if isinstance(raw_replacements, str):
            try:
                raw_replacements = json.loads(raw_replacements)
            except Exception:
                return "ERROR: [PLAN_REJECTED] planned_replacements must be a valid JSON array or list."

        if not isinstance(raw_replacements, list) or not raw_replacements:
            return "ERROR: [PLAN_REJECTED] planned_replacements must be a non-empty list of replacements."

        replacements: list[WorkaroundPlannedReplacement] = []
        for idx, item in enumerate(raw_replacements):
            try:
                if isinstance(item, WorkaroundPlannedReplacement):
                    r_obj = item
                elif isinstance(item, dict):
                    r_obj = WorkaroundPlannedReplacement(**item)
                else:
                    return f"ERROR: [PLAN_REJECTED] Item at index {idx} in planned_replacements is invalid."
                replacements.append(r_obj)
            except Exception as exc:
                return f"ERROR: [PLAN_REJECTED] Item at index {idx} in planned_replacements failed validation: {exc}"

        if len(replacements) > 16:
            return f"ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 16 replacements (got {len(replacements)})."

        files_list = affected_files if isinstance(affected_files, list) else [str(affected_files)]
        symbols_list = (
            affected_symbols if isinstance(affected_symbols, list) else [str(affected_symbols)]
        )

        declared_files = set()
        for f in files_list:
            f_str = str(f).strip()
            if f_str:
                try:
                    declared_files.add(_validate_workspace_path(f_str))
                except ValueError:
                    declared_files.add(f_str.replace("\\", "/").lstrip("/"))

        if not declared_files:
            return "ERROR: [PLAN_REJECTED] At least one affected file must be specified."

        replacement_files = set()
        seen_specs = set()
        total_text_bytes = 0

        for r in replacements:
            try:
                norm_p = _validate_workspace_path(r.file_path)
            except ValueError as exc:
                return f"ERROR: [PLAN_REJECTED] {exc}"

            if _is_prohibited_target(norm_p):
                return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

            if not r.old_text:
                return f"ERROR: [PLAN_REJECTED] Anchor old_text cannot be empty for replacement on '{norm_p}'."

            if r.old_text == r.new_text:
                return f"ERROR: [PLAN_REJECTED] Replacement cannot be a no-op (old_text equals new_text) on '{norm_p}'."

            if r.expected_occurrences <= 0:
                return f"ERROR: [PLAN_REJECTED] expected_occurrences must be positive for replacement on '{norm_p}'."

            spec_key = (norm_p, r.old_text, r.new_text, r.symbol_name)
            if spec_key in seen_specs:
                return f"ERROR: [PLAN_REJECTED] Duplicate replacement specification found for file '{norm_p}'."
            seen_specs.add(spec_key)

            replacement_files.add(norm_p)
            total_text_bytes += len(r.new_text.encode("utf-8"))

        if len(replacement_files) > 8:
            return f"ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 8 affected files (got {len(replacement_files)})."

        if total_text_bytes > 65536:
            return "ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 64 KiB of combined replacement text."

        if declared_files != replacement_files:
            return (
                f"ERROR: [PLAN_REJECTED] Mismatch between declared affected_files ({sorted(declared_files)}) "
                f"and files in planned_replacements ({sorted(replacement_files)})."
            )

        inspected_files = (
            plan_state.get("inspected_files", set())
            | plan_state.get("read_files", set())
            | plan_state.get("fallback_files", set())
        )
        for f in replacement_files:
            if f not in inspected_files:
                return (
                    f"ERROR: [PLAN_REJECTED] Target file '{f}' has not been inspected. "
                    "Inspect all planned files using read_workspace_file or inspect_ast_symbol before recording a plan."
                )

        plan_state["recorded"] = True
        plan_state["plan_revision"] = int(plan_state.get("plan_revision", 0)) + 1
        plan_state["phase"] = WorkaroundExecutionPhase.EXECUTE.value
        plan_state["successful_edit_count_this_iteration"] = 0

        plan_state["planned_replacements"] = [r.model_dump() for r in replacements]
        plan_state["planned_files"] = sorted(list(replacement_files))

        new_symbols = [str(s).strip() for s in symbols_list if str(s).strip()]
        existing_symbols = list(plan_state.get("planned_symbols", []))
        all_symbols = list(dict.fromkeys(existing_symbols + new_symbols))
        plan_state["planned_symbols"] = all_symbols

        plan_state["security_invariant"] = sec_inv
        plan_state["causal_hypothesis"] = causal_hyp
        plan_state["evidence_source"] = ev_source
        plan_state.setdefault("evidence_ledger", []).append(ev_source)
        if _is_authoritative_evidence_source(ev_source):
            plan_state["has_authoritative_evidence"] = True

        for f in replacement_files:
            plan_state.setdefault("fallback_files", set()).add(f)

        logger.debug(
            "Workaround subagent plan recorded (revision %s): %s",
            plan_state["plan_revision"],
            plan_state,
        )
        resp_json = {
            "status": "SUCCESS",
            "plan_revision": plan_state["plan_revision"],
            "phase_transition": "PLAN -> EXECUTE",
            "evidence_source": ev_source,
            "planned_targets": sorted(list(replacement_files)),
            "planned_replacements": [r.model_dump() for r in replacements],
        }
        return (
            f"SUCCESS: Plan revision {plan_state['plan_revision']} recorded successfully.\n"
            f"Evidence Source: {ev_source}\n"
            f"Phase Transition: PLAN -> EXECUTE\n"
            f"Planned Targets: {', '.join(sorted(replacement_files))}\n"
            f"JSON: {json.dumps(resp_json)}"
        )

    return record_plan


def _make_record_targeted_test_substitution_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any],
):
    @tool
    def record_targeted_test_substitution(
        original_test: str,
        alternative_test: str,
        infrastructure_failure_evidence: str,
        shared_behavior_explanation: str,
        evidence_sources: Any,
        infrastructure_avoidance_explanation: str,
    ) -> str:
        """Register an existing repository alternative test as a substitution for an infrastructure-failed targeted test."""
        if not original_test or not str(original_test).strip():
            return "ERROR: [SUBSTITUTION_REJECTED] original_test is required."
        if not alternative_test or not str(alternative_test).strip():
            return "ERROR: [SUBSTITUTION_REJECTED] alternative_test is required."

        infra_ev = str(infrastructure_failure_evidence or "").strip()
        shared_exp = str(shared_behavior_explanation or "").strip()
        avoid_exp = str(infrastructure_avoidance_explanation or "").strip()

        if not infra_ev:
            return "ERROR: [SUBSTITUTION_REJECTED] infrastructure_failure_evidence cannot be empty."
        if not shared_exp:
            return "ERROR: [SUBSTITUTION_REJECTED] shared_behavior_explanation cannot be empty."
        if not avoid_exp:
            return "ERROR: [SUBSTITUTION_REJECTED] infrastructure_avoidance_explanation cannot be empty."

        ev_sources = (
            evidence_sources if isinstance(evidence_sources, list) else [str(evidence_sources)]
        )
        cleaned_ev_sources = [str(s).strip() for s in ev_sources if str(s).strip()]
        if not cleaned_ev_sources:
            return "ERROR: [SUBSTITUTION_REJECTED] At least one evidence source must be provided."

        phase = plan_state.get("phase")
        if phase and phase != WorkaroundExecutionPhase.VALIDATE.value:
            return (
                "ERROR: [SUBSTITUTION_REJECTED] An alternative targeted test may only be "
                "registered after the targeted validation gate reports an infrastructure failure."
            )

        try:
            norm_orig = _validate_workspace_path(original_test)
            norm_alt = _validate_workspace_path(alternative_test)
        except ValueError as exc:
            return f"ERROR: [SUBSTITUTION_REJECTED] {exc}"

        # 1. Verify original test was the latest failed targeted test
        latest_failed = plan_state.get("latest_failed_targeted_test")
        if not latest_failed or norm_orig != latest_failed:
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] original_test '{norm_orig}' does not match "
                f"the latest failed targeted test '{latest_failed}'."
            )

        # 2. Verify failure was classified as infrastructure-only
        if not plan_state.get("latest_test_failure_infra", False):
            return (
                "ERROR: [SUBSTITUTION_REJECTED] Substitution is allowed only for infrastructure failures. "
                "Assertion failures, syntax failures, type failures, and application runtime errors do not qualify."
            )

        alt_content = sandbox.read_file(norm_alt)
        if alt_content is None:
            return f"ERROR: [SUBSTITUTION_REJECTED] alternative_test file '{norm_alt}' does not exist in workspace."

        # 3. Verify alternative is different from original
        if norm_alt == norm_orig:
            return "ERROR: [SUBSTITUTION_REJECTED] alternative_test must be different from original_test."

        if not _is_test_file_path(norm_alt):
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] alternative_test '{norm_alt}' must be an existing "
                "repository test/spec file. A source module cannot replace the targeted test."
            )

        # 4. Verify alternative test file has been locally inspected
        inspected_files = plan_state.get("inspected_files", set()) | plan_state.get(
            "read_files", set()
        )
        if norm_alt not in inspected_files:
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] alternative_test file '{norm_alt}' has not been inspected. "
                "Read or inspect the test file using read_workspace_file before registering it."
            )

        # 5. Verify alternative directly imports or invokes neither unavailable dep nor failed infra setup
        infra_diag = (plan_state.get("latest_infra_diagnostics") or "").lower()
        alt_lowered = alt_content.lower()

        forbidden_keywords = []
        if "sqlite" in infra_diag:
            forbidden_keywords.extend(["sqlite3", "better-sqlite3"])
        if "native" in infra_diag or "bindings" in infra_diag:
            forbidden_keywords.extend(["node-gyp", "prebuild-install", "bindings"])

        for kw in forbidden_keywords:
            if kw in alt_lowered:
                return (
                    f"ERROR: [SUBSTITUTION_REJECTED] alternative_test '{norm_alt}' directly imports/invokes "
                    f"unavailable infrastructure component '{kw}'."
                )

        # 6. Verify only one alternative per iteration
        curr_iter = int(plan_state.get("iteration", 1))
        if plan_state.get("substitution_registered_for_iteration") == curr_iter:
            return f"ERROR: [SUBSTITUTION_REJECTED] Only one alternative test substitution is permitted for iteration {curr_iter}."

        plan_state["accepted_alternative_test"] = norm_alt
        plan_state["substitution_registered_for_iteration"] = curr_iter
        plan_state.setdefault("original_to_alternative_test_mapping", {})[norm_orig] = norm_alt
        plan_state.setdefault("original_to_alternative_test_evidence", {})[norm_orig] = (
            cleaned_ev_sources
        )
        plan_state.setdefault("original_to_alternative_test_details", {})[norm_orig] = {
            "infrastructure_failure_evidence": infra_ev,
            "shared_behavior_explanation": shared_exp,
            "infrastructure_avoidance_explanation": avoid_exp,
            "evidence_sources": ", ".join(cleaned_ev_sources),
        }

        return (
            f"SUCCESS: Registered alternative targeted test '{norm_alt}' as full replacement for "
            f"'{norm_orig}' in iteration {curr_iter}. Mapping evidence recorded."
        )

    return record_targeted_test_substitution


def _make_search_web_tool(
    mandatory_search_terms: dict[str, str] | None = None,
    plan_state: dict[str, Any] | None = None,
):
    """Create a web search tool backed by Serper.dev."""
    _calls_remaining = [3]

    @tool
    def search_web(query: str) -> str:
        """Search the web for vulnerability fixes, migration guides, or documentation.

        Pass your own targeted query. Do not wrap queries in quotes.
        """
        if plan_state is not None and not plan_state.get("local_investigation_complete", False):
            return (
                "ERROR: [INVESTIGATION_REQUIRED] Local codebase investigation must complete "
                "using search_codebase_pattern, read_workspace_file, or inspect_ast_symbol "
                "before calling search_web."
            )
        if (
            plan_state is not None
            and plan_state.get("phase") == WorkaroundExecutionPhase.VALIDATE.value
        ):
            return (
                "ERROR: [PHASE_VIOLATION] Web research is not a validation action. "
                "Call validate_workaround, or register an evidence-backed alternative test "
                "after an infrastructure-only targeted-test failure."
            )

        if _calls_remaining[0] <= 0:
            return f"ERROR: search_web call limit reached (max {_SEARCH_WEB_MAX_CALLS} per session). Use the results you already have."

        api_key = os.environ.get("SERPER_API_KEY", "").strip()
        if not api_key:
            return "ERROR: SERPER_API_KEY is not set. Cannot perform web search."

        _calls_remaining[0] -= 1

        effective_query = (query or "").replace('"', "").replace("'", "").strip()
        if not effective_query:
            return (
                "ERROR: search_web requires a worker-selected query. Classify the workaround "
                "type first and include the relevant package, advisory, migration, scanner, "
                "or test-regression terms."
            )

        try:
            resp = requests.post(
                _SERPER_SEARCH_URL,
                json={"q": effective_query},
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

            calls_left = _calls_remaining[0]
            if plan_state is not None:
                plan_state["web_search_performed"] = True

            if not results:
                return f"Effective Query: {effective_query}\nNo results found for this query."

            header = f"Effective Query: {effective_query}\nFound {len(results)} results ({calls_left} searches remaining):\n\n"
            return header + "\n\n---\n\n".join(results)

        except Exception as exc:
            logger.warning("search_web failed: %s", exc)
            return f"Effective Query: {effective_query}\nERROR: Web search failed - {exc}."

    return search_web


def _github_api_url(target_url: str) -> str | None:
    """Translate a public GitHub URL into its corresponding GitHub API URL."""
    parsed = urlparse(target_url)
    hostname = (parsed.hostname or "").lower()
    if hostname not in {"github.com", "www.github.com"}:
        return None

    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return None

    if parts[0].lower() == "advisories" and len(parts) >= 2:
        return f"{_GITHUB_API_URL_PREFIX}advisories/{quote(parts[1], safe='')}"
    if len(parts) < 2:
        return None

    owner, repository = parts[0], parts[1]
    if repository.endswith(".git"):
        repository = repository[:-4]
    base = f"{_GITHUB_API_URL_PREFIX}repos/{quote(owner, safe='')}/{quote(repository, safe='')}"
    if len(parts) == 2:
        return base

    resource = parts[2].lower()
    if resource in {"issues", "pulls", "commits"} and len(parts) >= 4:
        return f"{base}/{resource}/{quote(parts[3], safe='')}"
    if resource == "releases" and len(parts) >= 5 and parts[3].lower() == "tag":
        return f"{base}/releases/tags/{quote('/'.join(parts[4:]), safe='')}"
    if resource in {"blob", "raw", "tree"} and len(parts) >= 4:
        ref = quote(parts[3], safe="")
        content_path = quote("/".join(parts[4:]), safe="/")
        endpoint = f"{base}/contents/{content_path}" if content_path else f"{base}/contents"
        return f"{endpoint}?ref={ref}"
    return base


def _github_headers() -> dict[str, str]:
    """Build GitHub API headers without exposing an optional token in logs."""
    headers = {
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _decode_github_response(resp: Any, target_url: str) -> str:
    """Decode raw GitHub content or a JSON API response into readable text."""
    text = getattr(resp, "text", "") or ""
    if text.strip():
        try:
            payload = resp.json()
        except Exception:  # noqa: BLE001
            return text
        if isinstance(payload, dict):
            if payload.get("encoding") == "base64" and payload.get("content"):
                try:
                    return b64decode(str(payload["content"]).replace("\n", "")).decode("utf-8")
                except Exception:  # noqa: BLE001
                    return text
            title = (
                payload.get("title")
                or payload.get("name")
                or payload.get("login")
                or "GitHub Content"
            )
            body = payload.get("body") or payload.get("description") or payload.get("content") or ""
            if title or body:
                return f"Title: {title}\nURL: {target_url}\n\n{body}".strip()
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        if isinstance(payload, list):
            import json

            return json.dumps(payload, indent=2, ensure_ascii=False)
        return text
    return f"No readable content extracted from {target_url}."


def _make_read_web_page_tool(plan_state: dict[str, Any] | None = None):
    """Create a tool to fetch web pages, using GitHub API, raw content, Jina, and npm fallbacks."""

    @tool
    def read_web_page(url: str) -> str:
        """
        Fetch the full readable markdown text of a web page given its URL.

        Use this tool after search_web to read complete migration guides, breaking change lists,
        or documentation pages found in search results.
        """
        if plan_state is not None:
            if not plan_state.get("local_investigation_complete", False):
                return (
                    "ERROR: [INVESTIGATION_REQUIRED] Local codebase investigation must complete "
                    "before calling read_web_page."
                )
            if not plan_state.get("web_search_performed", False):
                return "ERROR: [SEARCH_REQUIRED] You must execute search_web before calling read_web_page."
            if plan_state.get("phase") == WorkaroundExecutionPhase.VALIDATE.value:
                return (
                    "ERROR: [PHASE_VIOLATION] Web research is not a validation action. "
                    "Call validate_workaround instead."
                )

        target_url = (url or "").strip()
        if not target_url:
            return "ERROR: url is required."

        # 1. GitHub API & Raw GitHub fallback
        if "github.com" in target_url and not target_url.startswith(
            "https://raw.githubusercontent.com"
        ):
            github_api_url = _github_api_url(target_url)
            if github_api_url:
                try:
                    resp = requests.get(
                        github_api_url,
                        headers=_github_headers(),
                        timeout=_READ_WEB_PAGE_TIMEOUT,
                    )
                    resp.raise_for_status()
                    text = _decode_github_response(resp, target_url)
                    if text and text.strip():
                        if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                            text = (
                                text[:_READ_WEB_PAGE_MAX_CHARS]
                                + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                            )
                        if plan_state is not None and _is_authoritative_evidence_source(target_url):
                            plan_state["has_authoritative_evidence"] = True
                            plan_state["evidence_source"] = target_url
                        return f"--- Markdown content of {target_url} ---\n\n{text}"
                except Exception as exc:  # noqa: BLE001
                    logger.debug("GitHub API fetch failed for %s: %s", target_url, exc)

            raw_url = (
                target_url.replace("github.com", "raw.githubusercontent.com")
                .replace("/blob/", "/")
                .replace("/tree/", "/")
            )
            try:
                resp = requests.get(raw_url, timeout=_READ_WEB_PAGE_TIMEOUT)
                resp.raise_for_status()
                text = resp.text or ""
                if text and text.strip():
                    if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                        text = (
                            text[:_READ_WEB_PAGE_MAX_CHARS]
                            + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                        )
                    if plan_state is not None and _is_authoritative_evidence_source(target_url):
                        plan_state["has_authoritative_evidence"] = True
                        plan_state["evidence_source"] = target_url
                    return f"--- Markdown content of {target_url} ---\n\n{text}"
            except Exception as exc:  # noqa: BLE001
                logger.debug("Raw GitHub fetch failed for %s: %s", raw_url, exc)

        # 2. Jina Reader fallback
        jina_url = f"{_JINA_READER_URL_PREFIX}{target_url}"
        try:
            resp = requests.get(
                jina_url,
                headers={"Accept": "text/plain"},
                timeout=_READ_WEB_PAGE_TIMEOUT,
            )
            resp.raise_for_status()
            text = resp.text or ""
            if text and text.strip():
                if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                    text = (
                        text[:_READ_WEB_PAGE_MAX_CHARS]
                        + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                    )
                if plan_state is not None and _is_authoritative_evidence_source(target_url):
                    plan_state["has_authoritative_evidence"] = True
                    plan_state["evidence_source"] = target_url
                return f"--- Markdown content of {target_url} ---\n\n{text}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Jina Reader fetch failed for %s: %s", target_url, exc)

        # 3. Direct page fetch fallback
        try:
            resp = requests.get(target_url, timeout=_READ_WEB_PAGE_TIMEOUT)
            resp.raise_for_status()
            text = resp.text or ""
            if text and text.strip():
                if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                    text = (
                        text[:_READ_WEB_PAGE_MAX_CHARS]
                        + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                    )
                if plan_state is not None and _is_authoritative_evidence_source(target_url):
                    plan_state["has_authoritative_evidence"] = True
                    plan_state["evidence_source"] = target_url
                return f"--- Markdown content of {target_url} ---\n\n{text}"
        except Exception as exc:  # noqa: BLE001
            logger.debug("Direct page fetch failed for %s: %s", target_url, exc)

        # 4. npm registry fallback if URL relates to npm package
        if (
            "npmjs.com" in target_url
            or "registry.npmjs.org" in target_url
            or not target_url.startswith("http")
        ):
            pkg_name = (
                target_url.split("package/")[-1].split("/")[0].strip()
                if "package/" in target_url
                else target_url.strip()
            )
            if pkg_name:
                try:
                    resp = requests.get(
                        f"https://registry.npmjs.org/{pkg_name}",
                        timeout=_READ_WEB_PAGE_TIMEOUT,
                    )
                    resp.raise_for_status()
                    text = resp.text or ""
                    if text and text.strip():
                        if len(text) > _READ_WEB_PAGE_MAX_CHARS:
                            text = (
                                text[:_READ_WEB_PAGE_MAX_CHARS]
                                + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
                            )
                        if plan_state is not None and _is_authoritative_evidence_source(target_url):
                            plan_state["has_authoritative_evidence"] = True
                            plan_state["evidence_source"] = target_url
                        return f"--- Markdown content of {target_url} ---\n\n{text}"
                except Exception:  # noqa: BLE001
                    pass

        return f"ERROR: Failed to read web page {target_url} - No readable content extracted from any fallback source."

    return read_web_page
