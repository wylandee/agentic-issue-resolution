"""Workspace repository access, web research, and read-only inspection tools."""

from __future__ import annotations

import logging
import re
from base64 import b64decode
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

import requests
from langchain_core.tools import tool

from remediation_engine.contracts.schemas import (
    WorkaroundExecutionPhase,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.runtime.path_policy import (
    normalize_workspace_path,
    repository_relative_path,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox import DockerSandbox
from remediation_engine.settings import get_runtime_settings
from remediation_engine.tools.identity import normalize_python_package_name
from remediation_engine.tools.maven_manifest_locator import (
    MavenManifestError,
    find_dependency_in_pom,
    locate_maven_manifests,
    parse_pom_xml,
)

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
_JAVA_SOURCE_SUFFIXES = frozenset({".java"})
_PYTHON_SOURCE_MODULE_SUFFIXES = frozenset({".py"})
_PYTHON_RUNTIME_BOOTSTRAP_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^\s*if\s+__name__\s*==\s*[\"']__main__[\"']\s*:", re.MULTILINE),
        "contains an application entrypoint",
    ),
    (re.compile(r"\b(?:app|application)\.run\s*\("), "starts an application server"),
    (re.compile(r"\buvicorn\.run\s*\("), "starts an application server"),
    (
        re.compile(r"\bmanage\.execute_from_command_line\s*\("),
        "starts a Django management command",
    ),
)
_TEST_DIRECTORY_NAMES = frozenset({"test", "tests", "__tests__", "spec", "specs"})

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
    source_url = candidate if "://" in candidate else f"https://{candidate}"
    try:
        source_host = (urlparse(source_url).hostname or "").lower()
    except ValueError:
        source_host = ""
    if source_host in {
        "pypi.org",
        "www.pypi.org",
        "packaging.python.org",
        "docs.python.org",
    }:
        return True

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


def _runtime_smoke_bootstrap_reason(
    file_path: str,
    source_content: str,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str | None:
    """Return why a source module is unsafe for import-only runtime smoke."""
    normalized = _normalise_newlines(source_content)
    if language == ProjectLanguage.PYTHON:
        if Path(file_path).name == "manage.py":
            return "is a Django management entrypoint"
        if Path(file_path).name in {"setup.py", "conftest.py"}:
            return "is a Python manifest or helper file, not an application source module"
        for pattern, reason in _PYTHON_RUNTIME_BOOTSTRAP_MARKERS:
            if pattern.search(normalized):
                return reason
        return None
    for pattern, reason in _RUNTIME_BOOTSTRAP_MARKERS:
        if pattern.search(normalized):
            return reason
    return None


def _source_suffixes(language: ProjectLanguage) -> frozenset[str]:
    """Return the source suffixes supported by workaround tooling."""
    if language == ProjectLanguage.PYTHON:
        return _PYTHON_SOURCE_MODULE_SUFFIXES
    if language == ProjectLanguage.JAVA:
        return _JAVA_SOURCE_SUFFIXES
    return _SOURCE_MODULE_SUFFIXES


def _test_suffixes(language: ProjectLanguage) -> frozenset[str]:
    """Return source suffixes accepted for targeted test files."""
    return _source_suffixes(language)


def _select_lightweight_runtime_smoke_target(
    requested_file: str,
    candidate_files: Sequence[str],
    targeted_test_file: str | None,
    sandbox: DockerSandbox,
    *,
    source_suffixes: frozenset[str] | None = None,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> tuple[str | None, str | None]:
    """Select a safe source module for import-only runtime smoke."""
    normalized_requested, path_error = _runtime_smoke_path_error(
        requested_file,
        targeted_test_file=targeted_test_file,
        source_suffixes=source_suffixes,
        language=language,
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
        language=language,
    )
    if requested_reason is None:
        return normalized_requested, None

    normalized_candidates: list[str] = []
    for raw_path in candidate_files:
        try:
            candidate = _validate_workspace_path(str(raw_path))
        except ValueError:
            continue
        if candidate in normalized_candidates or candidate in (
            normalized_requested,
            targeted_test_file,
        ):
            continue
        if Path(candidate).suffix.lower() not in (source_suffixes or _source_suffixes(language)):
            continue
        if source_suffixes is not None and Path(candidate).suffix.lower() not in source_suffixes:
            continue
        if _is_test_file_path(candidate, language=language):
            continue
        content = sandbox.read_file(candidate)
        if content is None:
            continue
        if _runtime_smoke_bootstrap_reason(candidate, content, language=language) is None:
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


def _is_test_file_path(
    file_path: str,
    *,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> bool:
    """Return whether a repository-relative path identifies a test/spec file."""
    path = Path(file_path.replace("\\", "/"))
    lowered_parts = {part.lower() for part in path.parts[:-1]}
    filename = path.name.lower()
    is_java_test_class = path.suffix.lower() == ".java" and (
        filename.startswith("test")
        or filename.endswith(("test.java", "tests.java", "testcase.java"))
    )
    is_named_test = bool(
        ".test." in filename or ".spec." in filename or filename.endswith((".test", ".spec"))
    )
    if language == ProjectLanguage.PYTHON:
        is_named_test = (
            is_named_test
            or filename == "conftest.py"
            or filename.startswith("test_")
            or filename.endswith("_test.py")
        )
    return bool(lowered_parts & _TEST_DIRECTORY_NAMES or is_named_test or is_java_test_class)


def _select_targeted_test_file(
    requested_file: str | None,
    preferred_test_files: Sequence[str] | None,
    accepted_alternative_test: str | None,
    sandbox: DockerSandbox,
    *,
    source_suffixes: frozenset[str] | None = None,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
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
            or Path(candidate).suffix.lower() not in (source_suffixes or _test_suffixes(language))
            or not _is_test_file_path(candidate, language=language)
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
        requested_is_source_module = requested_suffix in (
            source_suffixes or _source_suffixes(language)
        ) and not _is_test_file_path(normalized_requested, language=language)
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
    *,
    source_suffixes: frozenset[str] | None = None,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> tuple[str | None, str | None]:
    """Validate and normalize a lightweight source-module smoke target."""
    try:
        rel_path = _validate_workspace_path(runtime_file)
    except ValueError as exc:
        return None, str(exc)

    suffixes = source_suffixes or _source_suffixes(language)
    if Path(rel_path).suffix.lower() not in suffixes:
        accepted = ", ".join(sorted(suffixes))
        return None, (
            f"Runtime smoke target '{rel_path}' must use a supported source suffix: {accepted}."
        )
    if _is_test_file_path(rel_path, language=language):
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


def _normalize_manifest_targets(
    target_manifest_paths: Iterable[str],
    allowed_manifest_names: Iterable[str] = ("package.json",),
) -> list[str]:
    """Return stable, validated paths with one of the explicitly allowed basenames."""
    manifest_paths = sorted(
        {_validate_workspace_path(path) for path in target_manifest_paths if path}
    )
    allowed_names = {str(name).strip() for name in allowed_manifest_names if str(name).strip()}
    invalid = [path for path in manifest_paths if Path(path).name not in allowed_names]
    if invalid:
        raise ValueError(
            f"Target manifest paths must point to {sorted(allowed_names)} files. Invalid values: {invalid}"
        )
    return manifest_paths


def _normalize_package_manifest_targets(
    package_manifest_paths: Mapping[str, Iterable[str]],
    allowed_manifest_names: Iterable[str] = ("package.json",),
) -> dict[str, list[str]]:
    """Return validated package-to-manifest targets for update tasks."""
    normalized: dict[str, list[str]] = {}
    for package_name, manifest_paths in package_manifest_paths.items():
        package_key = (package_name or "").strip()
        if not package_key:
            raise ValueError("Package manifest target keys must be non-empty.")
        normalized[package_key] = _normalize_manifest_targets(
            manifest_paths,
            allowed_manifest_names=allowed_manifest_names,
        )
    return normalized


def _resolve_maven_manifest_scope(
    group: Any,
    repo_root: Path,
    *,
    allow_dependency_management_creation: bool = False,
) -> tuple[list[str], list[str]]:
    """Resolve Maven targets and their in-repository reactor/parent scope.

    Missing exact declarations are accepted only for a Supervisor-authorized
    dependency-management operation. A versionless direct declaration may use
    this operation because Maven can inherit its version from dependency
    management; an explicit direct version remains outside that authorization.
    """
    root = Path(repo_root).resolve()
    coordinate = str(getattr(group, "vulnerable_component", "")).strip()
    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*)*:"
        r"[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?\Z",
        coordinate,
    ):
        return [], [
            f"Group '{getattr(group, 'group_id', '')}': Maven target must be a "
            "canonical group:artifact coordinate."
        ]
    group_id, artifact_id = coordinate.split(":", 1)

    target_types: dict[str, set[str]] = {}
    owner_paths: set[str] = set()
    for localized in getattr(group, "localized_issues", ()) or ():
        issue = getattr(localized, "issue", None)
        is_maven = (
            str(getattr(localized, "package_manager", "") or "").strip().lower() == "maven"
            or str(getattr(issue, "ecosystem", "") or "").strip().lower() == "maven"
            or str(getattr(issue, "purl", "") or "").strip().lower().startswith("pkg:maven/")
        )
        if not is_maven:
            continue
        path = str(getattr(localized, "manifest_file", "") or "")
        path = path.replace("\\", "/").strip()
        dependency_type = str(getattr(localized, "declaration_type", "") or "").strip()
        if path:
            target_types.setdefault(path, set()).add(dependency_type)
        owner = str(getattr(localized, "version_property_file", "") or "")
        owner = owner.replace("\\", "/").strip()
        if owner:
            owner_paths.add(owner)
    if not target_types:
        return [], [
            f"Group '{getattr(group, 'group_id', '')}': no localized Maven POM target is available."
        ]

    errors: list[str] = []
    target_rel_paths: set[str] = set()
    owner_rel_paths: set[str] = set()

    def contained_pom(value: str) -> str:
        normalized = normalize_workspace_path(value, allow_workspace_prefix=False)
        path = resolve_repository_path(root, normalized)
        if path.name != "pom.xml" or path.is_dir() or not path.is_file():
            raise ValueError(f"'{value}' is not an existing in-repository pom.xml")
        rel = repository_relative_path(path, root)
        if not rel:
            raise ValueError(f"'{value}' is outside the repository")
        return rel

    for path, dependency_types in target_types.items():
        try:
            rel = contained_pom(path)
            if not dependency_types or not dependency_types <= {
                "dependencies",
                "dependencyManagement",
            }:
                raise ValueError(
                    "localized Maven target type must be dependencies or dependencyManagement"
                )
            target_rel_paths.add(rel)
        except (OSError, ValueError) as exc:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': rejected Maven target '{path}': {exc}."
            )
    for path in owner_paths:
        try:
            owner_rel_paths.add(contained_pom(path))
        except (OSError, ValueError) as exc:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': rejected Maven "
                f"property owner '{path}': {exc}."
            )
    if errors or not target_rel_paths:
        return [], errors

    manifests: dict[str, Any] = {}
    parse_errors: dict[str, str] = {}
    for path in locate_maven_manifests(root):
        rel = repository_relative_path(path, root)
        if not rel:
            continue
        try:
            manifests[rel] = parse_pom_xml(path)
        except (MavenManifestError, OSError, ValueError) as exc:
            parse_errors[rel] = str(exc)
    for rel in sorted(target_rel_paths | owner_rel_paths):
        if rel not in manifests:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': authorized Maven POM "
                f"'{rel}' could not be parsed: "
                f"{parse_errors.get(rel, 'POM was not discovered')}."
            )
    if errors:
        return [], errors

    for rel, dependency_types in target_types.items():
        try:
            target_rel = contained_pom(rel)
            dependency = find_dependency_in_pom(
                group_id,
                artifact_id,
                manifests[target_rel],
            )
        except (MavenManifestError, OSError, ValueError) as exc:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': ambiguous Maven target '{rel}': {exc}."
            )
            continue
        ensure_managed = allow_dependency_management_creation and dependency_types == {
            "dependencyManagement"
        }
        if dependency is None and ensure_managed:
            continue
        if (
            dependency is not None
            and ensure_managed
            and dependency.dependency_type == "dependencies"
            and dependency.declared_version is None
        ):
            continue
        if dependency is None or dependency.dependency_type not in dependency_types:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': POM '{rel}' "
                "does not contain the exact "
                f"{group_id}:{artifact_id} declaration at its authorized target type."
            )
    if errors:
        return [], errors

    module_children: dict[str, set[str]] = {}
    reverse_edges: dict[str, set[str]] = {}
    parent_of: dict[str, str] = {}
    for rel, manifest in manifests.items():
        if manifest.parent_path is not None:
            parent_rel = repository_relative_path(manifest.parent_path, root)
            if parent_rel:
                parent_of[rel] = parent_rel
                reverse_edges.setdefault(rel, set()).add(parent_rel)
        children: set[str] = set()
        for module in manifest.modules:
            module_path = Path(module)
            if module_path.is_absolute() or "\\" in module:
                parse_errors[rel] = "unsafe Maven module path"
                break
            candidate = manifest.path.parent / module_path
            if candidate.name != "pom.xml":
                candidate = candidate / "pom.xml"
            child_rel = repository_relative_path(candidate, root)
            if not child_rel:
                parse_errors[rel] = "Maven module escapes the repository"
                break
            try:
                resolve_repository_path(root, child_rel)
            except (OSError, ValueError) as exc:
                parse_errors[rel] = str(exc)
                break
            children.add(child_rel)
            reverse_edges.setdefault(child_rel, set()).add(rel)
        if children:
            module_children[rel] = children

    ancestor_roots: set[str] = set()
    for target in target_rel_paths:
        ancestors = {target}
        pending = [target]
        while pending:
            child = pending.pop()
            for parent in reverse_edges.get(child, ()):
                if parent not in ancestors:
                    ancestors.add(parent)
                    pending.append(parent)
        topmost = {item for item in ancestors if not (reverse_edges.get(item, set()) & ancestors)}
        if not topmost:
            topmost = {target}
        ancestor_roots.update(topmost)
    if len(ancestor_roots) != 1:
        return [], [
            f"Group '{getattr(group, 'group_id', '')}': Maven targets do not resolve "
            "to one deterministic in-repository reactor root."
        ]
    reactor_root = next(iter(ancestor_roots))

    authorized: set[str] = set()
    pending = [reactor_root]
    while pending:
        rel = pending.pop()
        if rel in authorized:
            continue
        if rel not in manifests:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': reactor POM "
                f"'{rel}' could not be parsed."
            )
            continue
        if rel in parse_errors:
            errors.append(
                f"Group '{getattr(group, 'group_id', '')}': reactor POM "
                f"'{rel}' has invalid module metadata."
            )
            continue
        authorized.add(rel)
        pending.extend(module_children.get(rel, ()))
        parent = parent_of.get(rel)
        if parent:
            pending.append(parent)
    if not owner_rel_paths <= authorized:
        errors.append(
            f"Group '{getattr(group, 'group_id', '')}': Maven property owner "
            "is outside the authorized reactor/parent chain."
        )
    if not target_rel_paths <= authorized:
        errors.append(
            f"Group '{getattr(group, 'group_id', '')}': Maven target POM "
            "is outside the authorized reactor/parent chain."
        )
    return ([], errors) if errors else (sorted(authorized), [])


def _make_search_web_tool(
    plan_state: dict[str, Any] | None = None,
):
    """Create a web search tool backed by Serper.dev."""
    _calls_remaining = [_SEARCH_WEB_MAX_CALLS]

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

        api_key = get_runtime_settings().serper_api_key
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
    token = get_runtime_settings().github_token
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


def _pypi_project_from_target(target: str) -> tuple[str, str] | None:
    """Resolve a bare Python distribution name or PyPI project URL."""
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        if (parsed.hostname or "").lower() not in {"pypi.org", "www.pypi.org"}:
            return None
        parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2 or parts[0].lower() not in {"project", "pypi"}:
            return None
        raw_name = parts[1]
    else:
        raw_name = target
    normalized = normalize_python_package_name(raw_name.strip())
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", normalized):
        return None
    page_url = f"https://pypi.org/project/{quote(normalized, safe='')}/"
    return normalized, page_url


def _format_pypi_project(payload: Any, package_name: str) -> str | None:
    """Render the useful project metadata from PyPI's JSON API."""
    if not isinstance(payload, dict) or not isinstance(payload.get("info"), dict):
        return None
    info = payload["info"]
    name = info.get("name") or package_name
    lines = [f"# PyPI project: {name}"]
    for label, key in (
        ("Latest version", "version"),
        ("Summary", "summary"),
        ("Requires Python", "requires_python"),
        ("Homepage", "home_page"),
    ):
        value = info.get(key)
        if value:
            lines.append(f"{label}: {value}")
    project_urls = info.get("project_urls")
    if isinstance(project_urls, dict):
        for label, link in sorted(project_urls.items()):
            if link:
                lines.append(f"{label}: {link}")
    return "\n".join(lines)


def _read_pypi_project(
    package_name: str,
    page_url: str,
    evidence_source: str,
    plan_state: dict[str, Any] | None,
) -> str | None:
    """Read PyPI's project JSON first, then bounded reader/direct page fallbacks."""
    json_url = f"https://pypi.org/pypi/{quote(package_name, safe='')}/json"
    try:
        response = requests.get(json_url, timeout=_READ_WEB_PAGE_TIMEOUT)
        response.raise_for_status()
        text = _format_pypi_project(response.json(), package_name)
        if text and text.strip():
            if plan_state is not None and _is_authoritative_evidence_source(evidence_source):
                plan_state["has_authoritative_evidence"] = True
                plan_state["evidence_source"] = evidence_source
            return text
    except Exception as exc:  # noqa: BLE001
        logger.debug("PyPI JSON fetch failed for %s: %s", package_name, exc)

    for fallback_url, headers in (
        (f"{_JINA_READER_URL_PREFIX}{page_url}", {"Accept": "text/plain"}),
        (page_url, None),
    ):
        try:
            kwargs: dict[str, Any] = {"timeout": _READ_WEB_PAGE_TIMEOUT}
            if headers is not None:
                kwargs["headers"] = headers
            response = requests.get(fallback_url, **kwargs)
            response.raise_for_status()
            text = response.text or ""
            if text and text.strip():
                if plan_state is not None and _is_authoritative_evidence_source(evidence_source):
                    plan_state["has_authoritative_evidence"] = True
                    plan_state["evidence_source"] = evidence_source
                return text
        except Exception as exc:  # noqa: BLE001
            logger.debug("PyPI page fetch failed for %s: %s", package_name, exc)
    return None


def _truncate_page_text(text: str) -> str:
    if len(text) > _READ_WEB_PAGE_MAX_CHARS:
        return (
            text[:_READ_WEB_PAGE_MAX_CHARS]
            + f"\n\n[Content truncated at {_READ_WEB_PAGE_MAX_CHARS} characters...]"
        )
    return text


def _make_read_web_page_tool(
    plan_state: dict[str, Any] | None = None,
    *,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
):
    """Create a page reader with language-specific registry metadata fallbacks."""

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
        if language is ProjectLanguage.PYTHON:
            pypi_project = _pypi_project_from_target(target_url)
            if pypi_project is not None:
                package_name, page_url = pypi_project
                evidence_source = target_url if urlparse(target_url).scheme else page_url
                text = _read_pypi_project(
                    package_name,
                    page_url,
                    evidence_source,
                    plan_state,
                )
                if text:
                    return (
                        f"--- Markdown content of {target_url} ---\n\n{_truncate_page_text(text)}"
                    )

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
        if language is ProjectLanguage.NODEJS and (
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


def _make_read_repository_map_tool(sandbox: DockerSandbox):
    @tool
    def read_repository_map(
        query: str = "",
        include_all: bool = False,
        offset: int = 0,
        limit: int = _REPO_MAP_MAX_ENTRIES,
    ) -> str:
        """Return a deterministic, queryable repository path map.

        The default response is the first bounded page for compatibility with
        the original tool.  ``query`` performs a case-insensitive literal
        substring filter over repository-relative paths.  Use ``offset`` and
        ``limit`` for another page, or set ``include_all`` to return every
        matching entry without the default page cap.

        Args:
            query: Optional literal path substring to match.
            include_all: Whether to bypass the page limit for the filtered map.
            offset: Zero-based entry offset after filtering.
            limit: Maximum entries in one page when ``include_all`` is false.

        Returns:
            A sorted newline-delimited path list, with an actionable truncation
            marker when another page is available.
        """
        if offset < 0:
            return "ERROR: offset must be non-negative."
        if limit <= 0:
            return "ERROR: limit must be positive."

        normalized_query = str(query or "").strip().replace("\\", "/")
        script = (
            "find . "
            "-not -path '*/node_modules/*' -not -path '*/.git/*' "
            "-not -path '*/.venv' -not -path '*/.venv/*' "
            "-not -path '*/venv' -not -path '*/venv/*' "
            "-not -path '*/__pycache__' -not -path '*/__pycache__/*' "
            "-not -path '*/.pytest_cache' -not -path '*/.pytest_cache/*' "
            "-not -path '*/.remedy-pipenv' -not -path '*/.remedy-pipenv/*' "
            "-not -name '*.pyc' -not -name '*.map' "
            "| sed 's|^./||' | sort"
        )
        result = _run_readonly(sandbox, script, timeout=10)
        if result.exit_code != 0:
            return f"ERROR: Could not list workspace: {result.stderr.strip()}"

        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        if not lines:
            return "(workspace is empty)"

        if normalized_query:
            query_folded = normalized_query.casefold()
            lines = [line for line in lines if query_folded in line.casefold()]
            if not lines:
                return f"(no repository entries matched query: {normalized_query})"

        if include_all:
            selected = lines[offset:]
            truncated = False
        else:
            selected = lines[offset : offset + limit]
            truncated = offset + len(selected) < len(lines)

        if not selected:
            return f"(no repository entries at offset {offset})"

        output = "\n".join(selected)
        if truncated:
            remaining = len(lines) - offset - len(selected)
            output += (
                f"\n... (truncated, {remaining} more entries; use offset or "
                "include_all=true, optionally with query)"
            )
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
    *,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
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
        excluded_dirs = (
            "--exclude-dir=node_modules --exclude-dir=.git "
            "--exclude-dir=build --exclude-dir=dist --exclude-dir=data "
            "--exclude-dir=reports --exclude-dir=.pytest_cache "
            "--exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__ "
            "--exclude-dir=.remedy-pipenv"
        )
        if language == ProjectLanguage.PYTHON:
            source_includes = (
                "--include='*.py' --exclude='setup.py' --exclude='conftest.py' --exclude='*.pyc' "
            )
        else:
            source_includes = (
                "--include='*.js' --include='*.ts' --include='*.jsx' --include='*.tsx' "
                "--include='*.mjs' --include='*.cjs' "
            )
        cmd = (
            f"grep -RInE {source_includes}{excluded_dirs} "
            f"-- '{safe_pattern}' '{search_root}' | sed 's|^./||'"
        )

        try:
            result = _run_readonly(sandbox, cmd, timeout=_SEARCH_TIMEOUT_SECONDS)
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
            from remediation_engine.tools.source import (
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
                "Supported extensions are JS/TS (.js, .jsx, .ts, .tsx, .mjs, .cjs) and "
                "Python (.py when the Python grammar is available)."
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
