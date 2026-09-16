"""
remedy_tools.py - Specialized native workspace tools for Phase 5 subagents.
"""

from __future__ import annotations

import json  # noqa: F401
import logging
import re
import shlex  # noqa: F401
import uuid  # noqa: F401
from base64 import b64decode  # noqa: F401
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass  # noqa: F401
from pathlib import Path
from typing import Any  # noqa: F401
from urllib.parse import quote, urlparse  # noqa: F401

import requests  # noqa: F401
from langchain_core.tools import tool  # noqa: F401

from remediation_engine.contracts.schemas import (
    FailureCategory,  # noqa: F401
    NoFixMitigationStage,  # noqa: F401
    WorkaroundEdit,  # noqa: F401
    WorkaroundEditSet,  # noqa: F401
    WorkaroundExecutionPhase,  # noqa: F401
    WorkaroundPlannedReplacement,  # noqa: F401
    WorkaroundValidationResult,  # noqa: F401
    WorkaroundValidationStatus,  # noqa: F401
)  # noqa: F401
from remediation_engine.orchestration.runtime_context import get_runtime_settings  # noqa: F401
from remediation_engine.runtime.path_policy import (
    normalize_workspace_path,
    resolve_repository_path,  # noqa: F401
)  # noqa: F401
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


def _run_readonly(sandbox: DockerSandbox, command: str, *, timeout: int) -> Any:
    """Run an explicitly read-only workspace command when supported.

    Production Docker sandboxes use the revision-aware read-cache path. Test
    doubles and replay sandboxes continue to use their existing ``run`` seam,
    which keeps this optimization backward-compatible for injected adapters.

    Args:
        sandbox: Docker-backed or test-double workspace.
        command: Command that does not modify workspace inputs.
        timeout: Maximum execution time in seconds.

    Returns:
        The sandbox command result.
    """
    if isinstance(sandbox, DockerSandbox):
        return sandbox.run_readonly(command, timeout=timeout)
    return sandbox.run(command, timeout=timeout)


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
    path is rejected. If the requested source module bootstraps the
    application, the selector considers changed source files in deterministic
    order and records which safe module was selected.

    Args:
        requested_file: Agent-provided runtime smoke path.
        candidate_files: Changed source files eligible for deterministic
            selection.
        targeted_test_file: Already selected targeted test path.
        sandbox: Workspace sandbox used to read source contents.

    Returns:
        ``(selected_path, note)`` on success, where ``note`` describes a
        deterministic alternate selection. On failure, returns
        ``(None, diagnostic)``.
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


def _select_targeted_test_file(
    requested_file: str | None,
    preferred_test_files: Sequence[str] | None,
    accepted_alternative_test: str | None,
    sandbox: DockerSandbox,
) -> tuple[str | None, str | None, str | None]:
    """Resolve an agent test selection to one deterministic existing source test.

    Directory paths, source modules, and missing paths are common model-level
    selection mistakes.  When the supervisor/QA context provides a preferred
    test, those mistakes are corrected locally instead of being sent back to
    the model for repeated retries.  An existing, different test file remains
    rejected so the QA target contract is not silently weakened.

    Args:
        requested_file: Agent-provided test path, if any.
        preferred_test_files: QA-recommended repository-relative test paths.
        accepted_alternative_test: Previously approved infrastructure-only
            replacement test, if any.
        sandbox: Workspace sandbox used to verify candidate files.

    Returns:
        A tuple of ``(selected_path, correction_note, error)``.  Exactly one
        of ``correction_note`` or ``error`` may be populated.
    """
    candidate_paths: list[str] = []
    for raw_path in [accepted_alternative_test, *(preferred_test_files or [])]:
        if not raw_path:
            continue
        try:
            candidate = _validate_workspace_path(str(raw_path))
        except ValueError:
            continue
        if (
            candidate in candidate_paths
            or Path(candidate).suffix.lower() not in _SOURCE_MODULE_SUFFIXES
            or not _is_test_file_path(candidate)
            or sandbox.read_file(candidate) is None
        ):
            continue
        candidate_paths.append(candidate)

    if requested_file and requested_file.strip():
        raw_requested = requested_file.replace("\\", "/").strip().lstrip("/")
        if raw_requested.startswith(("build/", "dist/")):
            return (
                None,
                None,
                f"Compiled test path '{raw_requested}' is not supported. "
                "Use the original source test path under test/, tests/, or the source package directory.",
            )
        try:
            normalized_requested = _validate_workspace_path(requested_file)
        except ValueError as exc:
            return None, None, str(exc)

        if normalized_requested.startswith(("build/", "dist/")):
            return (
                None,
                None,
                f"Compiled test path '{normalized_requested}' is not supported. "
                "Use the original source test path under test/, tests/, or the source package directory.",
            )

        if normalized_requested in candidate_paths:
            return normalized_requested, None, None

        requested_content = sandbox.read_file(normalized_requested)
        requested_suffix = Path(normalized_requested).suffix.lower()
        requested_is_source_module = (
            requested_suffix in _SOURCE_MODULE_SUFFIXES
            and not _is_test_file_path(normalized_requested)
        )
        requested_is_directory_hint = not requested_suffix or normalized_requested.endswith("/")
        requested_is_missing = requested_content is None

        if candidate_paths and (
            requested_is_source_module or requested_is_directory_hint or requested_is_missing
        ):
            prefix = normalized_requested.rstrip("/") + "/"
            scoped_candidates = [
                candidate for candidate in candidate_paths if candidate.startswith(prefix)
            ]
            selected = scoped_candidates[0] if scoped_candidates else candidate_paths[0]
            return (
                selected,
                (
                    f"Targeted test path '{normalized_requested}' was not used; "
                    f"selected canonical QA test '{selected}'."
                ),
                None,
            )

        return normalized_requested, None, None

    if candidate_paths:
        selected = candidate_paths[0]
        return selected, f"Selected canonical QA test '{selected}'.", None
    return None, None, None


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
    candidate = str(file_path or "").strip()
    if candidate.replace("\\", "/").startswith("workspace/"):
        candidate = candidate.replace("\\", "/")[len("workspace/") :]
    candidate = normalize_workspace_path(candidate)
    parts = Path(candidate).parts
    if parts and parts[0] in ("build", "dist"):
        raise ValueError(
            f"Accessing compiled files in '{parts[0]}/' is strictly forbidden. Please modify the original source files instead."
        )

    return candidate


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
    """Return stable, validated ``package.json`` targets for one update task."""
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
    """Return validated package-to-manifest targets for update tasks."""
    normalized: dict[str, list[str]] = {}
    for package_name, manifest_paths in package_manifest_paths.items():
        package_key = (package_name or "").strip()
        if not package_key:
            raise ValueError("Package manifest target keys must be non-empty.")
        normalized[package_key] = _normalize_manifest_targets(manifest_paths)
    return normalized


__all__ = [name for name in globals() if not name.startswith("__")]
