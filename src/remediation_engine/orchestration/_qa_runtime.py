"""Shared runtime state and scan-target helpers for QA execution.

This module owns deterministic execution evidence projections, workspace
diffing, and preparation of scan targets from the Supervisor's task queue and
committed attempt snapshots. Docker-backed workspaces are treated as
ephemeral; helpers read them for evidence and never mutate the host baseline.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import logging
import re
import shlex
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    DependencyEvidenceStatus,
    NoFixMitigationStage,
    ODCScanEvidence,
    QADependencyEvidence,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScanScope,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.lockfile_closure import (
    DependencyClosure,
    build_sliced_lockfile_artifacts,
    resolve_dependency_closure,
)
from remediation_engine.tools.package_identity import normalize_python_package_name

from ._tool_support import _run_readonly
from .qa_odc import _ODC_HTML_REPORT_NAME, _ODC_REPORT_NAME
from .qa_test_parsing import _NPM_INSTALL_TIMEOUT_SECONDS, _workspace_json_file
from .qa_types import (
    QAScanTarget,
    QATaskContext,
    _append_qa_log_records,
    _QAExecutionResults,
    _QALogRecord,
    _QAPackageState,
    _scan_result_value,
    _SecurityScanResult,
    _validate_qa_path,
)

logger = logging.getLogger(__name__)

_DIFF_CHAR_BUDGET = 8_000
_DIFF_EXCLUDE_DIRS = frozenset(
    {
        "node_modules",
        ".git",
        "dependency-check-data",
        "coverage",
        ".nyc_output",
        ".cache",
    }
)
_DIFF_EXCLUDE_SUFFIXES = frozenset({".map", ".lock"})
_DIFF_EXCLUDE_NAMES = frozenset({_ODC_REPORT_NAME, _ODC_HTML_REPORT_NAME})
_QA_ACTION_SUMMARY_MAX_CHARS = 1_200
_BULLET_LABEL_RE = re.compile(r"^- ([^:]+):\s*(.*)$")
_REPORT_PREFIX = "# INVESTIGATIVE REPORT"


def _label_scan_records(
    records: Sequence[_QALogRecord],
    label: str | None,
) -> tuple[_QALogRecord, ...]:
    """Apply a deterministic display label to scan records when requested."""
    if not label:
        return tuple(records)
    return tuple(
        record if record.label == label else replace(record, label=label) for record in records
    )


def _store_scan_outcome(
    results: _QAExecutionResults,
    scan_result: _SecurityScanResult,
    *,
    label: str | None = None,
) -> None:
    records = tuple(scan_result.scan_records)
    if records:
        if label:
            records = _label_scan_records(records, label)
    else:
        records = (
            _QALogRecord(
                phase="scan",
                label=label or "odc:full",
                exit_code=scan_result.exit_code,
                stdout=scan_result.raw_stdout or scan_result.summary,
                stderr=scan_result.raw_stderr or "",
                error=scan_result.diagnostic_log_path,
            ),
        )
    if records != scan_result.scan_records:
        scan_result = replace(scan_result, scan_records=records)
    results.scan = scan_result
    _append_qa_log_records(results, "scan", records)


def _generate_workspace_diff(
    host_repo_root: str,
    sandbox: DockerSandbox,
    candidate_changed_files: list[str],
) -> tuple[str, list[str]]:
    """
    Generate a unified diff by comparing host baseline to the current sandbox workspace
    for only the specified candidate_changed_files.

    The Docker volume does not contain ``.git``, so ``git diff`` is not available.
    We read files from the sandbox and diff against the host.

    Returns:
        (diff_text_capped, changed_file_paths)
        ``diff_text_capped`` is the diff content, capped at ``_DIFF_CHAR_BUDGET`` chars.
        ``changed_file_paths`` is the full list of repo-relative paths that changed
        (retained even when the diff text itself is truncated).
    """
    if not candidate_changed_files:
        return "(no changed files were provided; diff is empty)", []

    host_root = Path(host_repo_root)
    changed_files: list[str] = []
    diff_parts: list[str] = []
    blocked_paths: list[str] = []

    # Deduplicate files while preserving order.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for f in candidate_changed_files:
        if f not in seen:
            seen.add(f)
            unique_candidates.append(f)

    for rel_path in unique_candidates:
        try:
            rel_path = normalize_workspace_path(rel_path, allow_workspace_prefix=False)
            abs_host = resolve_repository_path(host_root, rel_path)
        except WorkspacePathError as exc:
            blocked_paths.append(f"{rel_path!r}: {exc}")
            continue
        parts = Path(rel_path).parts
        if any(p in _DIFF_EXCLUDE_DIRS for p in parts):
            continue
        if Path(rel_path).name in _DIFF_EXCLUDE_NAMES:
            continue
        if Path(rel_path).suffix.lower() in _DIFF_EXCLUDE_SUFFIXES:
            continue

        try:
            workspace_content = sandbox.read_file(rel_path)
        except WorkspacePathError as exc:
            blocked_paths.append(f"{rel_path!r}: {exc}")
            continue

        try:
            if abs_host.is_file():
                host_content = abs_host.read_text(encoding="utf-8", errors="replace")
            else:
                host_content = None
        except Exception:
            host_content = None

        if workspace_content is None:
            # File deleted in workspace.
            if host_content is not None:
                changed_files.append(rel_path)
                diff_parts.append(f"--- {rel_path} (deleted in workspace)\n")
        elif host_content is None:
            # New file in workspace.
            changed_files.append(rel_path)
            diff_parts.append(f"+++ {rel_path} (new file in workspace)\n")
        elif workspace_content != host_content:
            # File modified.
            changed_files.append(rel_path)
            import difflib

            diff_lines = list(
                difflib.unified_diff(
                    host_content.splitlines(keepends=True),
                    workspace_content.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
            )
            diff_parts.append("".join(diff_lines))

    full_diff = "\n".join(diff_parts)
    if not full_diff:
        empty_diff = (
            "(diff is empty â€” workspace matches host baseline for all candidate files)",
            changed_files,
        )
        if blocked_paths:
            return (
                "ERROR: blocked candidate path(s):\n"
                + "\n".join(f"- {path}" for path in blocked_paths)
                + "\n\n"
                + empty_diff[0],
                changed_files,
            )
        return empty_diff

    if len(full_diff) > _DIFF_CHAR_BUDGET:
        full_diff = full_diff[:_DIFF_CHAR_BUDGET] + "\n... (diff truncated)"

    if blocked_paths:
        full_diff = (
            "ERROR: blocked candidate path(s):\n"
            + "\n".join(f"- {path}" for path in blocked_paths)
            + "\n\n"
            + full_diff
        )

    return full_diff, changed_files


def _workspace_remediation_fingerprint(
    host_repo_root: str | None,
    sandbox: DockerSandbox,
    candidate_changed_files: Sequence[str],
) -> str:
    """Return a stable digest of the material workspace change.

    Args:
        host_repo_root: Repository baseline used for the workspace comparison.
        sandbox: Active remediation workspace volume.
        candidate_changed_files: Files reported as candidates by workers.

    Returns:
        A deterministic SHA-256 digest of the actual changed files and their
        current unified diff. An empty or unchanged workspace therefore keeps
        the same digest across final scans.

    Side Effects:
        Reads the candidate files from the sandbox and the host repository.
        The helper does not modify either workspace.
    """
    candidates = [path for path in candidate_changed_files if isinstance(path, str)]
    changed_files: list[str]
    diff_text: str
    if host_repo_root:
        try:
            diff_text, changed_files = _generate_workspace_diff(
                host_repo_root,
                sandbox,
                candidates,
            )
        except Exception as exc:  # noqa: BLE001 - fingerprinting must not abort the scan
            logger.warning("qa_critic: workspace fingerprint failed: %s", exc)
            diff_text = ""
            changed_files = sorted(set(candidates))
    else:
        diff_text = ""
        changed_files = sorted(set(candidates))

    payload = {
        "changed_files": sorted(set(changed_files)),
        "diff": diff_text,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _collect_target_identifiers(groups: list[VulnerabilityGroup]) -> set[str]:
    """Collect all CVE/GHSA identifiers from the valid vulnerability groups."""
    identifiers: set[str] = set()
    for group in groups:
        for cve in group.cve_ids or []:
            if cve:
                identifiers.add(cve.upper().strip())
        for ghsa in group.ghsa_ids or []:
            if ghsa:
                identifiers.add(ghsa.upper().strip())
        # Fallback: individual issue-level identifiers
        for issue in group.issues or []:
            if issue.cve_id:
                identifiers.add(issue.cve_id.upper().strip())
            if issue.ghsa_id:
                identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _collect_baseline_identifiers(
    state: OrchestratorState,
    groups: list[VulnerabilityGroup],
) -> set[str]:
    """Resolve the immutable pre-remediation scanner-identifier baseline.

    The graph records this baseline before remediation begins. When the
    caller supplies the initial typed issue set instead, its CVE/GHSA
    identifiers are used. The current groups are used only when no initial
    issue snapshot is available.
    """
    if "baseline_scan_identifiers" in state:
        return {
            identifier.upper().strip()
            for identifier in state.get("baseline_scan_identifiers", []) or []
            if identifier and identifier.strip()
        }

    issues = state.get("issues")
    if issues is not None:
        identifiers: set[str] = set()
        for issue in issues:
            if issue.cve_id:
                identifiers.add(issue.cve_id.upper().strip())
            if issue.ghsa_id:
                identifiers.add(issue.ghsa_id.upper().strip())
        return identifiers

    return _collect_target_identifiers(groups)


def _group_is_python(group: VulnerabilityGroup) -> bool:
    """Return whether the group's canonical issue or authorization is PyPI-backed."""
    issues = list(getattr(group, "issues", []) or [])
    issues.extend(
        localized.issue
        for localized in (getattr(group, "localized_issues", []) or [])
        if getattr(localized, "issue", None) is not None
    )
    for issue in issues:
        ecosystem = str(getattr(issue, "ecosystem", "") or "").strip().casefold()
        purl_type = str(getattr(issue, "purl", "") or "").partition(":")[2].partition("/")[0]
        if ecosystem in {"pypi", "python"} or purl_type.casefold() == "pypi":
            return True
    for localized in getattr(group, "localized_issues", []) or []:
        if str(getattr(localized, "package_manager", "") or "").casefold() in {"pip", "pipenv"}:
            return True
    candidates = [*(getattr(group, "file_paths", []) or []), getattr(group, "file_path", None)]
    return any(
        Path(str(path or "")).name.casefold()
        in {"pipfile", "pipfile.lock", "pyproject.toml", "setup.cfg"}
        or (
            Path(str(path or "")).name.casefold().startswith("requirements")
            and Path(str(path or "")).name.casefold().endswith(".txt")
        )
        for path in candidates
    )


def _python_manifest_paths_for_group(group: VulnerabilityGroup) -> tuple[str, ...]:
    """Return only localized or scanner-reported Python manifest paths."""
    localized_paths = [
        getattr(issue, "manifest_file", None)
        for issue in (getattr(group, "localized_issues", []) or [])
        if getattr(issue, "manifest_file", None)
    ]
    candidates = localized_paths or [
        *(getattr(group, "file_paths", []) or []),
        getattr(group, "file_path", None),
    ]
    manifests: set[str] = set()
    for raw_path in candidates:
        path = str(raw_path or "").strip().split("?", 1)[0].split("#", 1)[0]
        path = path.replace("\\", "/")
        if not path or path.startswith("/") or ".." in Path(path).parts:
            continue
        path = path.lstrip("./")
        name = Path(path).name.casefold()
        if name == "pipfile.lock":
            path = str(Path(path).with_name("Pipfile"))
            name = "pipfile"
        if name in {"pyproject.toml", "setup.cfg", "pipfile"} or (
            name.startswith("requirements") and name.endswith(".txt")
        ):
            manifests.add(path)
    return tuple(sorted(manifests))


def _lockfile_paths_for_group(group: VulnerabilityGroup) -> tuple[str, ...]:
    """Return normalized package-manager lockfiles associated with an SCA group."""
    if _group_is_python(group):
        return tuple(
            sorted(
                str(Path(manifest).with_name("Pipfile.lock"))
                for manifest in _python_manifest_paths_for_group(group)
                if Path(manifest).name.casefold() == "pipfile"
            )
        )

    candidates: list[str] = []
    candidates.extend(group.file_paths or [])
    if group.file_path:
        candidates.append(group.file_path)
    for issue in group.localized_issues or []:
        if issue.manifest_file:
            candidates.append(issue.manifest_file)

    explicit_lockfiles: set[str] = set()
    manifest_paths: set[str] = set()
    lockfile_names = {"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml"}
    for raw_path in candidates:
        path = str(raw_path or "").strip().split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
        if not path or path.startswith("/") or ".." in Path(path).parts:
            continue
        path = path.lstrip("./")
        name = Path(path).name.lower()
        if name in lockfile_names:
            explicit_lockfiles.add(path)
        elif name == "package.json":
            manifest_paths.add(path)
    if explicit_lockfiles:
        return tuple(sorted(explicit_lockfiles))
    return tuple(
        sorted(str(Path(manifest).with_name("package-lock.json")) for manifest in manifest_paths)
    )


def _build_qa_scan_targets(
    state: OrchestratorState,
    groups: list[VulnerabilityGroup],
) -> list[QAScanTarget] | None:
    """Build scan targets from the active tasks and their committed attempts."""
    active_ids = list(state.get("active_target_task_ids") or [])
    if not active_ids:
        return None

    task_queue: dict[str, RemediationTask] = dict(state.get("task_queue") or {})
    groups_by_id = {group.group_id: group for group in groups}
    targets: list[QAScanTarget] = []
    for active_id in active_ids:
        task = task_queue.get(active_id)
        group_id = task.parent_group_id if task is not None else active_id
        group = groups_by_id.get(group_id)
        if group is None:
            continue
        target_package = (
            task.target_package_name
            if task is not None and task.target_package_name
            else group.vulnerable_component or ""
        ).strip()
        attempt_version, has_attempt_version_evidence = _attempt_version_evidence(state, task)
        expected_version = (
            task.selected_version if task is not None and task.selected_version else attempt_version
        )
        # A workaround task deliberately has no supervisor-selected package
        # version.  Its worker may leave the installed dependency at whatever
        # version the live workspace resolves after install.  Falling back to
        # the group's baseline version here can therefore point the closure
        # resolver at a package instance that no longer exists (for example,
        # an express-jwt workaround task with a live 8.x installation and a
        # baseline group version from the original 0.x finding).  Leaving the
        # version unconstrained lets the resolver select the live instance or
        # fail closed when the live lockfile is genuinely ambiguous.
        is_unversioned_workaround = (
            task is not None
            and task.strategy == RoutingStrategy.CODE_WORKAROUND
            and not task.selected_version
        )
        if (
            not expected_version
            and target_package
            and not is_unversioned_workaround
            and not has_attempt_version_evidence
        ):
            expected_version = (group.dependency_versions or {}).get(target_package)
        if (
            not expected_version
            and target_package == group.vulnerable_component
            and not is_unversioned_workaround
            and not has_attempt_version_evidence
        ):
            expected_version = (group.versions or [None])[0]
        ancestry = tuple(name for name in (group.dependency_ancestry or []) if name)
        targets.append(
            QAScanTarget(
                task_id=task.task_id if task is not None else active_id,
                group_id=group.group_id,
                target_package=target_package,
                expected_version=expected_version,
                manifest_paths=_lockfile_paths_for_group(group),
                dependency_ancestry=ancestry,
                target_identifiers=frozenset(group_target_identifiers(group)),
            )
        )
    return targets


def _attempt_version_evidence(
    state: OrchestratorState,
    task: RemediationTask | None,
) -> tuple[str | None, bool]:
    """Return the exact version proven by the current task attempt, if any.

    The immutable attempt snapshot is authoritative when its identity matches
    the active task. A valid worker result may refine that version when it
    records an allowed effective candidate or one uniquely executed version.

    Args:
        state: Orchestrator state containing attempt snapshots and worker results.
        task: Active task owning the QA scan target, if one was resolved.

    Returns:
        A ``(version, has_version_evidence)`` tuple. The version is populated
        only when exactly one authorized version is proven; the boolean remains
        true when execution evidence exists but is not uniquely resolvable.
    """
    if task is None or task.strategy != RoutingStrategy.VERSION_BUMP:
        return None, False

    attempt_id = task.current_attempt_id
    snapshots = state.get("attempt_snapshots_by_id") or {}
    snapshot = snapshots.get(attempt_id) if attempt_id else None
    snapshot_is_current = (
        snapshot is not None
        and getattr(snapshot, "attempt_id", None) == attempt_id
        and getattr(snapshot, "task_id", None) == task.task_id
        and getattr(snapshot, "task_revision", None) == task.task_revision
    )

    worker_results = state.get("worker_results_by_attempt") or {}
    worker_result = worker_results.get(attempt_id) if attempt_id else None
    worker_is_current = (
        worker_result is not None
        and getattr(worker_result, "attempt_id", None) == attempt_id
        and getattr(worker_result, "task_id", None) == task.task_id
        and getattr(worker_result, "task_revision", None) == task.task_revision
    )
    if worker_result is not None and not worker_is_current and not snapshot_is_current:
        # Neither source has current provenance; do not use a stale result.
        return None, True

    if worker_is_current:
        execution = getattr(worker_result, "execution_diagnostics", None)
        effective_version = _normalise_dependency_version(
            getattr(execution, "effective_target_version", None)
        )
        raw_versions = list(getattr(execution, "executed_versions", []) or [])
        if not raw_versions:
            raw_versions = list(getattr(worker_result, "executed_versions", []) or [])
        versions = list(
            dict.fromkeys(
                normalized
                for version in raw_versions
                if (normalized := _normalise_dependency_version(version)) is not None
            )
        )
        if effective_version is not None:
            versions = [effective_version]
        if versions:
            allowed_versions = {
                normalized
                for version in (
                    getattr(snapshot, "allowed_target_versions", []) if snapshot_is_current else []
                )
                if (normalized := _normalise_dependency_version(version)) is not None
            }
            if allowed_versions and any(version not in allowed_versions for version in versions):
                return None, True
            return (versions[0] if len(versions) == 1 else None), True

    if snapshot_is_current:
        snapshot_version = _normalise_dependency_version(
            getattr(snapshot, "selected_version", None)
        )
        if snapshot_version is not None:
            return snapshot_version, True
        # A current attempt without its selected version is incomplete
        # supervisor provenance and must not fall back to finding metadata.
        return None, True

    if worker_result is not None or snapshot is not None:
        # An untrusted attempt record was present but could not authorize a
        # dependency version for this task.
        return None, True
    return None, False


def _json_dependency_present(package_json: dict[str, Any], package: str) -> bool:
    """Return whether a package is declared in a direct dependency section."""
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        values = package_json.get(section)
        if isinstance(values, dict) and package in values:
            return True
    return False


def _dependency_tree_contains(value: Any, package: str) -> bool:
    """Return whether an npm dependency tree contains a named package."""
    if isinstance(value, dict):
        if value.get("name") == package and ("version" in value or "resolved" in value):
            return True
        dependencies = value.get("dependencies")
        if isinstance(dependencies, dict) and package in dependencies:
            return True
        return any(_dependency_tree_contains(item, package) for item in value.values())
    if isinstance(value, list):
        return any(_dependency_tree_contains(item, package) for item in value)
    return False


def _normalise_dependency_version(value: Any) -> str | None:
    """Return a comparable dependency version without a leading ``v``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lstrip("vV")
    return normalized or None


def _json_pointer(parts: Sequence[str]) -> str:
    """Build a JSON Pointer for compact evidence references."""
    escaped = (part.replace("~", "~0").replace("/", "~1") for part in parts)
    return "#/" + "/".join(escaped)


def _dependency_declarations(
    package_json: Mapping[str, Any],
    package: str,
) -> dict[str, str]:
    """Return direct and override declarations for one package."""
    declarations: dict[str, str] = {}
    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        values = package_json.get(section)
        if isinstance(values, Mapping) and package in values:
            value = values[package]
            if isinstance(value, str) and value.strip():
                declarations[_json_pointer((section, package))] = value.strip()

    def visit_overrides(value: Any, path: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping):
            return
        candidate = value.get(package)
        if isinstance(candidate, str) and candidate.strip():
            declarations[_json_pointer((*path, package))] = candidate.strip()
        for key, child in value.items():
            if isinstance(child, Mapping):
                visit_overrides(child, (*path, str(key)))

    visit_overrides(package_json.get("overrides"), ("overrides",))
    return declarations


def _dependency_tree_versions(value: Any, package: str) -> set[str]:
    """Collect resolved versions for a package from an npm dependency tree."""
    versions: set[str] = set()
    if isinstance(value, Mapping):
        if value.get("name") == package:
            version = _normalise_dependency_version(value.get("version"))
            if version:
                versions.add(version)
        dependencies = value.get("dependencies")
        if isinstance(dependencies, Mapping):
            package_node = dependencies.get(package)
            if isinstance(package_node, Mapping):
                version = _normalise_dependency_version(package_node.get("version"))
                if version:
                    versions.add(version)
            for child in dependencies.values():
                versions.update(_dependency_tree_versions(child, package))
        for key, child in value.items():
            if key == "dependencies":
                continue
            versions.update(_dependency_tree_versions(child, package))
    elif isinstance(value, list):
        for child in value:
            versions.update(_dependency_tree_versions(child, package))
    return versions


def _lockfile_versions(value: Any, package: str) -> set[str]:
    """Collect package versions from npm lockfile ``packages`` entries."""
    if not isinstance(value, Mapping):
        return set()
    versions: set[str] = set()
    packages = value.get("packages")
    if isinstance(packages, Mapping):
        suffix = f"/node_modules/{package}"
        root_key = f"node_modules/{package}"
        for key, node in packages.items():
            normalized_key = str(key).replace("\\", "/")
            if (normalized_key == root_key or normalized_key.endswith(suffix)) and isinstance(
                node, Mapping
            ):
                version = _normalise_dependency_version(node.get("version"))
                if version:
                    versions.add(version)
    versions.update(_dependency_tree_versions(value.get("dependencies"), package))
    return versions


def _manifest_paths_for_group(group: VulnerabilityGroup) -> tuple[str, ...]:
    """Return normalized editable manifests associated with an SCA group."""
    if _group_is_python(group):
        return _python_manifest_paths_for_group(group)
    candidates = [
        *(getattr(group, "file_paths", []) or []),
        getattr(group, "file_path", None),
        *(
            getattr(issue, "manifest_file", None)
            for issue in (getattr(group, "localized_issues", []) or [])
        ),
    ]
    manifests: set[str] = set()
    lockfile_names = {"package-lock.json", "npm-shrinkwrap.json", "yarn.lock", "pnpm-lock.yaml"}
    for raw_path in candidates:
        path = str(raw_path or "").strip().split("?", 1)[0].split("#", 1)[0]
        path = path.replace("\\", "/")
        if not path or path.startswith("/") or ".." in Path(path).parts:
            continue
        path = path.lstrip("./")
        name = Path(path).name.lower()
        if name == "package.json":
            manifests.add(path)
        elif name in lockfile_names:
            manifests.add(str(Path(path).with_name("package.json")))
    return tuple(sorted(manifests))


def _lockfile_paths_for_manifests(manifests: Sequence[str]) -> tuple[str, ...]:
    """Return npm lockfile paths corresponding to package manifests."""
    return tuple(
        sorted({str(Path(manifest).with_name("package-lock.json")) for manifest in manifests})
    )


def _python_package_name(name: str) -> str:
    """Apply PEP 503 normalization to one distribution name."""
    return normalize_python_package_name(str(name or "").strip())


def _python_version(value: Any) -> str | None:
    """Return a canonical PEP 440 version or ``None`` when it is not parseable."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return str(Version(value.strip()))
    except InvalidVersion:
        return None


def _python_requirement(value: str) -> Requirement | None:
    """Parse one static requirement, omitting pip options and comments."""
    text = value.strip()
    if not text or text.startswith(("#", "-")):
        return None
    text = re.split(r"\s+#", text, maxsplit=1)[0].strip()
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _matching_python_declarations(
    sandbox: DockerSandbox,
    manifest: str,
    package: str,
) -> tuple[list[tuple[str, str, Requirement | None]], str | None]:
    """Parse only static Python declarations for a target distribution."""
    content = sandbox.read_file(manifest)
    if not isinstance(content, str):
        return [], f"Manifest '{manifest}' was unavailable."

    target = _python_package_name(package)
    matches: list[tuple[str, str, Requirement | None]] = []
    name = Path(manifest).name.casefold()
    if name.startswith("requirements") and name.endswith(".txt"):
        for line_number, line in enumerate(content.splitlines(), start=1):
            candidate = _python_requirement(line)
            candidate_name = (
                candidate.name
                if candidate is not None
                else (
                    re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line).group(1)
                    if re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
                    else ""
                )
            )
            if _python_package_name(candidate_name) == target:
                matches.append((f"{manifest}#L{line_number}", line.strip(), candidate))
        return matches, None

    def add_requirement(pointer: str, raw: Any) -> None:
        if not isinstance(raw, str):
            return
        requirement = _python_requirement(raw)
        inferred_name = (
            requirement.name
            if requirement is not None
            else re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", raw)
        )
        package_name = (
            inferred_name
            if isinstance(inferred_name, str)
            else inferred_name.group(1)
            if inferred_name
            else ""
        )
        if _python_package_name(package_name) == target:
            matches.append((pointer, raw.strip(), requirement))

    if name == "setup.cfg":
        parser = configparser.RawConfigParser()
        try:
            parser.read_string(content)
        except configparser.Error:
            return [], f"Manifest '{manifest}' could not be parsed."
        if parser.has_option("options", "install_requires"):
            values = parser.get("options", "install_requires", raw=True).splitlines()
            for raw in values:
                add_requirement(f"{manifest}#install_requires", raw)
        if parser.has_section("options.extras_require"):
            for extra, values in parser.items("options.extras_require", raw=True):
                for raw in values.splitlines():
                    add_requirement(f"{manifest}#extras_require/{extra}", raw)
        return matches, None

    try:
        document = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return [], f"Manifest '{manifest}' could not be parsed."
    if not isinstance(document, Mapping):
        return [], f"Manifest '{manifest}' did not contain a TOML table."
    if name == "pyproject.toml":
        project = document.get("project")
        if not isinstance(project, Mapping):
            return [], None
        dependencies = project.get("dependencies")
        if isinstance(dependencies, list):
            for raw in dependencies:
                add_requirement(f"{manifest}#dependencies", raw)
        optional = project.get("optional-dependencies")
        if isinstance(optional, Mapping):
            for extra, values in optional.items():
                if isinstance(values, list):
                    for raw in values:
                        add_requirement(f"{manifest}#optional-dependencies/{extra}", raw)
        return matches, None

    if name == "pipfile":
        for section in ("packages", "dev-packages"):
            packages = document.get(section)
            if not isinstance(packages, Mapping):
                continue
            for declared_name, declaration in packages.items():
                if (
                    not isinstance(declared_name, str)
                    or _python_package_name(declared_name) != target
                ):
                    continue
                version_value = (
                    declaration.get("version") if isinstance(declaration, Mapping) else declaration
                )
                raw = str(version_value or "*")
                requirement = None
                if raw.strip() not in {"", "*"}:
                    specifier = raw.strip()
                    if not specifier.startswith(("==", "!=", "~=", ">=", "<=", ">", "<", "===")):
                        specifier = f"=={specifier}"
                    requirement = _python_requirement(f"{declared_name}{specifier}")
                matches.append((f"{manifest}#{section}/{declared_name}", raw, requirement))
        return matches, None

    return [], f"Manifest '{manifest}' is not a supported Python declaration."


def _python_declaration_accepts(
    requirement: Requirement | None,
    raw_declaration: str,
    expected: str,
) -> bool | None:
    """Check whether a static declaration authorizes the selected version."""
    if raw_declaration.strip() in {"", "*"}:
        return True
    if requirement is None or requirement.marker is not None or requirement.url is not None:
        return None
    if requirement.extras:
        return None
    try:
        return Version(expected) in requirement.specifier
    except (InvalidVersion, ValueError):
        return None


def _read_python_lock_versions(
    sandbox: DockerSandbox,
    lockfiles: Sequence[str],
    package: str,
) -> tuple[set[str], list[str], list[str], list[str]]:
    """Read exact locked versions for one Pipfile target."""
    versions: set[str] = set()
    parsed: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    target = _python_package_name(package)
    for lockfile in lockfiles:
        content = sandbox.read_file(lockfile)
        if not isinstance(content, str):
            missing.append(lockfile)
            continue
        try:
            document = json.loads(content)
        except json.JSONDecodeError:
            errors.append(f"Lockfile '{lockfile}' was not valid JSON.")
            continue
        if not isinstance(document, Mapping):
            errors.append(f"Lockfile '{lockfile}' did not contain a JSON object.")
            continue
        parsed.append(lockfile)
        for category in ("default", "develop"):
            packages = document.get(category)
            if not isinstance(packages, Mapping):
                continue
            for locked_name, metadata in packages.items():
                if not isinstance(locked_name, str) or _python_package_name(locked_name) != target:
                    continue
                raw_version = metadata.get("version") if isinstance(metadata, Mapping) else None
                if isinstance(raw_version, str) and raw_version.startswith("=="):
                    version = _python_version(raw_version[2:])
                    if version:
                        versions.add(version)
    return versions, parsed, missing, errors


def _installed_python_versions(
    sandbox: DockerSandbox,
    package: str,
) -> tuple[set[str], str | None, bool]:
    """Inspect an installed distribution through the persistent project venv."""
    command = f".venv/bin/python -m pip show {shlex.quote(package)}"
    try:
        result = _run_readonly(sandbox, command, timeout=_NPM_INSTALL_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001
        return set(), f"Installed Python distribution inspection failed: {exc}", False

    exit_code = getattr(result, "exit_code", None)
    stdout = getattr(result, "stdout", "")
    stderr = getattr(result, "stderr", "")
    text = "\n".join(
        value
        for value in (
            stdout if isinstance(stdout, str) else "",
            stderr if isinstance(stderr, str) else "",
        )
        if value
    )
    if isinstance(exit_code, int) and exit_code != 0:
        if re.search(r"package\(s\) not found", text, re.IGNORECASE):
            return set(), f"Installed distribution '{package}' is missing from .venv.", True
        return set(), f"pip show could not verify '{package}' (exit code {exit_code}).", False
    name_match = re.search(r"^Name:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    version_match = re.search(r"^Version:\s*(.+?)\s*$", text, re.MULTILINE | re.IGNORECASE)
    if not name_match or not version_match:
        return set(), f"pip show returned incomplete evidence for '{package}'.", False
    if _python_package_name(name_match.group(1)) != _python_package_name(package):
        return set(), f"pip show returned a different distribution than '{package}'.", False
    version = _python_version(version_match.group(1))
    if version is None:
        return set(), f"Installed version for '{package}' was not valid PEP 440.", False
    return {version}, None, True


def _python_versions_match(versions: set[str], expected: str) -> bool:
    """Compare observed version strings using PEP 440 equivalence."""
    if not versions:
        return False
    try:
        expected_version = Version(expected)
        return all(Version(value) == expected_version for value in versions)
    except InvalidVersion:
        return False


def _collect_python_dependency_package_state(
    sandbox: DockerSandbox,
    group: VulnerabilityGroup,
    task: RemediationTask | None,
    manifests: Sequence[str],
    expected_version: str | None,
    *,
    lockfiles: Sequence[str] | None,
    version_evidence_inconclusive: bool,
) -> _QAPackageState:
    """Collect Python manifest, installed distribution, and Pipenv lock evidence."""
    package = _python_package_name(
        getattr(task, "target_package_name", None) or group.vulnerable_component or ""
    )
    expected = (
        None
        if version_evidence_inconclusive
        else _python_version(expected_version or getattr(task, "selected_version", None))
    )
    manifest_paths = tuple(manifests)
    python_lockfiles = tuple(
        lockfiles if lockfiles is not None else _lockfile_paths_for_group(group)
    )
    diagnostics: list[str] = []
    declarations: dict[str, str] = {}
    parsed_manifests: list[str] = []
    missing_declarations: list[str] = []
    declaration_checks: list[bool | None] = []
    for manifest in manifest_paths:
        try:
            entries, error = _matching_python_declarations(sandbox, manifest, package)
        except Exception as exc:  # noqa: BLE001
            entries, error = [], f"Manifest '{manifest}' could not be inspected: {exc}"
        if error:
            diagnostics.append(error)
            continue
        parsed_manifests.append(manifest)
        if not entries:
            missing_declarations.append(manifest)
        for reference, raw, requirement in entries:
            declarations[reference] = raw
            if expected is not None:
                declaration_checks.append(_python_declaration_accepts(requirement, raw, expected))

    if not package:
        diagnostics.append("Target Python distribution name is unavailable.")
    if not manifest_paths:
        diagnostics.append("Authorized Python manifest paths are unavailable.")
    if expected is None:
        diagnostics.append("Supervisor-selected target version is unavailable or invalid.")

    installed_versions: set[str] = set()
    installed_error: str | None = None
    installed_known = False
    if package:
        installed_versions, installed_error, installed_known = _installed_python_versions(
            sandbox, package
        )
        if installed_error:
            diagnostics.append(installed_error)

    pipfile_manifests = [
        manifest for manifest in manifest_paths if Path(manifest).name.casefold() == "pipfile"
    ]
    relevant_lockfiles = python_lockfiles if pipfile_manifests else ()
    locked_versions, parsed_lockfiles, missing_lockfiles, lock_errors = _read_python_lock_versions(
        sandbox, relevant_lockfiles, package
    )
    diagnostics.extend(lock_errors)
    if missing_lockfiles:
        diagnostics.append(
            "Authorized Pipfile.lock was unavailable: " + ", ".join(missing_lockfiles)
        )

    all_manifests_parsed = len(parsed_manifests) == len(manifest_paths)
    all_locks_parsed = len(parsed_lockfiles) == len(relevant_lockfiles)
    manifest_state = (
        "unknown" if not all_manifests_parsed else "present" if declarations else "absent"
    )
    graph_state = "present" if installed_versions else "absent" if installed_known else "unknown"

    if missing_declarations:
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            "Target distribution has no authorized declaration in: "
            + ", ".join(missing_declarations)
        )
    elif declaration_checks and False in declaration_checks:
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append("A Python declaration does not permit the Supervisor-selected version.")
    elif (
        expected is not None
        and installed_known
        and not _python_versions_match(installed_versions, expected)
    ):
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            f"Installed distribution versions {sorted(installed_versions) or ['none']} "
            f"do not exactly match Supervisor-selected version {expected}."
        )
    elif (
        expected is not None
        and relevant_lockfiles
        and all_locks_parsed
        and not _python_versions_match(locked_versions, expected)
    ):
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            f"Pipfile.lock versions {sorted(locked_versions) or ['none']} "
            f"do not exactly match Supervisor-selected version {expected}."
        )
    elif (
        not package
        or not manifest_paths
        or expected is None
        or not all_manifests_parsed
        or not installed_known
        or any(check is None for check in declaration_checks)
        or len(declaration_checks) < len(declarations)
        or not all_locks_parsed
        or bool(lock_errors)
    ):
        status = DependencyEvidenceStatus.INCONCLUSIVE
    else:
        status = DependencyEvidenceStatus.VERIFIED

    evidence = QADependencyEvidence(
        status=status,
        target_package=package,
        expected_version=expected,
        manifest_paths=list(manifest_paths),
        lockfile_paths=list(relevant_lockfiles),
        declarations=declarations,
        resolved_versions=sorted(installed_versions),
        lockfile_versions=sorted(locked_versions),
        evidence_refs=sorted(
            {
                *declarations,
                *(f"{lockfile}#resolved/{package}" for lockfile in parsed_lockfiles),
            }
        ),
        diagnostics=list(dict.fromkeys(diagnostics)),
    )
    return _QAPackageState(
        manifest_state=manifest_state,
        graph_state=graph_state,
        diagnostics=tuple(evidence.diagnostics),
        dependency_evidence=evidence,
    )


def _collect_python_package_presence_state(
    sandbox: DockerSandbox,
    group: VulnerabilityGroup,
    package: str,
    manifests: Sequence[str],
    lockfiles: Sequence[str],
    *,
    expect_present: bool,
) -> _QAPackageState:
    """Collect Python declaration and installed-package presence evidence."""
    package = _python_package_name(package)
    diagnostics: list[str] = []
    declarations: dict[str, str] = {}
    parsed_manifests: list[str] = []
    for manifest in manifests:
        try:
            entries, error = _matching_python_declarations(sandbox, manifest, package)
        except Exception as exc:  # noqa: BLE001
            entries, error = [], f"Manifest '{manifest}' could not be inspected: {exc}"
        if error:
            diagnostics.append(error)
            continue
        parsed_manifests.append(manifest)
        declarations.update({reference: raw for reference, raw, _requirement in entries})

    installed_versions, installed_error, installed_known = _installed_python_versions(
        sandbox,
        package,
    )
    if installed_error:
        diagnostics.append(installed_error)
    locked_versions, parsed_lockfiles, missing_lockfiles, lock_errors = _read_python_lock_versions(
        sandbox, lockfiles, package
    )
    diagnostics.extend(lock_errors)
    if missing_lockfiles:
        diagnostics.append(
            "Authorized Pipfile.lock was unavailable: " + ", ".join(missing_lockfiles)
        )

    manifest_state = (
        "unknown"
        if not manifests or len(parsed_manifests) != len(manifests)
        else "present"
        if declarations
        else "absent"
    )
    locks_parsed = len(parsed_lockfiles) == len(lockfiles) and not lock_errors
    graph_state = (
        "present"
        if installed_versions or locked_versions
        else "unknown"
        if not installed_known or not locks_parsed
        else "absent"
    )
    expected_state = "present" if expect_present else "absent"
    if manifest_state == "unknown" or graph_state == "unknown":
        status = DependencyEvidenceStatus.INCONCLUSIVE
    elif manifest_state == expected_state and graph_state == expected_state:
        status = DependencyEvidenceStatus.VERIFIED
    else:
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            f"Python package state was manifest={manifest_state}, graph={graph_state}; "
            f"expected both to be {expected_state}."
        )

    evidence = QADependencyEvidence(
        status=status,
        target_package=package,
        expected_version=None,
        manifest_paths=list(manifests),
        lockfile_paths=list(lockfiles),
        declarations=declarations,
        resolved_versions=sorted(installed_versions),
        lockfile_versions=sorted(locked_versions),
        evidence_refs=sorted(
            {
                *declarations,
                *(f"{lockfile}#resolved/{package}" for lockfile in parsed_lockfiles),
            }
        ),
        diagnostics=list(dict.fromkeys(diagnostics)),
    )
    return _QAPackageState(
        manifest_state=manifest_state,
        graph_state=graph_state,
        diagnostics=tuple(evidence.diagnostics),
        dependency_evidence=evidence,
    )


def _collect_dependency_package_state(
    sandbox: DockerSandbox,
    group: VulnerabilityGroup,
    task: RemediationTask | None,
    manifests: Sequence[str],
    expected_version: str | None,
    *,
    lockfiles: Sequence[str] | None = None,
    version_evidence_inconclusive: bool = False,
) -> _QAPackageState:
    """Collect compact deterministic dependency evidence for a version task."""
    if _group_is_python(group):
        return _collect_python_dependency_package_state(
            sandbox,
            group,
            task,
            manifests,
            expected_version,
            lockfiles=lockfiles,
            version_evidence_inconclusive=version_evidence_inconclusive,
        )
    package = str(
        getattr(task, "target_package_name", None) or group.vulnerable_component or ""
    ).strip()
    expected = _normalise_dependency_version(
        None
        if version_evidence_inconclusive
        else expected_version or getattr(task, "selected_version", None)
    )
    lockfiles = tuple(lockfiles or _lockfile_paths_for_manifests(manifests))
    diagnostics: list[str] = []
    declarations: dict[str, str] = {}
    parsed_manifest_paths: list[str] = []
    missing_declarations: list[str] = []
    resolved_versions: set[str] = set()
    graph_errors: list[str] = []
    parsed_lockfile_paths: list[str] = []
    lockfile_versions: set[str] = set()
    missing_lockfiles: list[str] = []

    if not package:
        diagnostics.append("Target package name is unavailable.")
    if not manifests:
        diagnostics.append("Authorized package.json manifest paths are unavailable.")
    if expected is None:
        diagnostics.append("Supervisor-selected target version is unavailable.")

    for manifest in manifests:
        payload = _workspace_json_file(sandbox, manifest)
        if payload is None:
            diagnostics.append(f"Manifest '{manifest}' was unavailable or not valid JSON.")
            continue
        parsed_manifest_paths.append(manifest)
        entries = _dependency_declarations(payload, package)
        if not entries:
            missing_declarations.append(manifest)
        for pointer, value in entries.items():
            declarations[f"{manifest}{pointer}"] = value

        cwd_path = Path(manifest).parent
        cwd = "" if str(cwd_path) == "." else cwd_path.as_posix().strip("/")
        prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        command = f"{prefix}npm ls {shlex.quote(package)} --all --json"
        try:
            command_result = _run_readonly(
                sandbox,
                command,
                timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
            )
            exit_code = getattr(command_result, "exit_code", None)
            if isinstance(exit_code, int) and exit_code != 0:
                graph_errors.append(f"npm ls failed for {manifest} with exit code {exit_code}.")
                continue
            raw = getattr(command_result, "stdout", "")
            if not isinstance(raw, str) or not raw.strip():
                graph_errors.append(f"npm ls returned no parseable graph for {manifest}.")
                continue
            tree = json.loads(raw)
            if not isinstance(tree, Mapping):
                graph_errors.append(f"npm ls returned no parseable graph for {manifest}.")
                continue
            resolved_versions.update(_dependency_tree_versions(tree, package))
            resolved_versions.update(_lockfile_versions(tree, package))
        except Exception as exc:  # noqa: BLE001
            graph_errors.append(f"npm graph inspection failed for {manifest}: {exc}")

    for lockfile in lockfiles:
        payload = _workspace_json_file(sandbox, lockfile)
        if payload is None:
            missing_lockfiles.append(lockfile)
            continue
        parsed_lockfile_paths.append(lockfile)
        lockfile_versions.update(_lockfile_versions(payload, package))

    diagnostics.extend(graph_errors)
    if missing_lockfiles:
        diagnostics.append(
            "Authorized lockfile(s) were unavailable: " + ", ".join(missing_lockfiles)
        )
    all_manifests_parsed = len(parsed_manifest_paths) == len(manifests)
    all_lockfiles_parsed = len(parsed_lockfile_paths) == len(lockfiles)
    graph_state = (
        "unknown"
        if not all_manifests_parsed or graph_errors
        else "present"
        if resolved_versions
        else "absent"
    )
    manifest_state = (
        "unknown" if not all_manifests_parsed else "present" if declarations else "absent"
    )

    if missing_declarations:
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            "Target package has no authorized declaration in: " + ", ".join(missing_declarations)
        )
    elif (
        expected is not None
        and all_lockfiles_parsed
        and lockfiles
        and lockfile_versions != {expected}
    ):
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            f"Lockfile versions {sorted(lockfile_versions) or ['none']} "
            f"do not exactly match Supervisor-selected version {expected}."
        )
    elif (
        expected is not None
        and all_manifests_parsed
        and not graph_errors
        and resolved_versions != {expected}
    ):
        status = DependencyEvidenceStatus.MISMATCH
        diagnostics.append(
            f"Resolved dependency versions {sorted(resolved_versions) or ['none']} "
            f"do not exactly match Supervisor-selected version {expected}."
        )
    elif (
        not package
        or not manifests
        or expected is None
        or not all_manifests_parsed
        or not all_lockfiles_parsed
        or bool(graph_errors)
    ):
        status = DependencyEvidenceStatus.INCONCLUSIVE
    else:
        status = DependencyEvidenceStatus.VERIFIED

    evidence = QADependencyEvidence(
        status=status,
        target_package=package,
        expected_version=expected,
        manifest_paths=list(manifests),
        lockfile_paths=list(lockfiles),
        declarations=declarations,
        resolved_versions=sorted(resolved_versions),
        lockfile_versions=sorted(lockfile_versions),
        evidence_refs=sorted(
            {
                *declarations,
                *(f"{lockfile}#resolved/{package}" for lockfile in parsed_lockfile_paths),
            }
        ),
        diagnostics=list(dict.fromkeys(diagnostics)),
    )
    return _QAPackageState(
        manifest_state=manifest_state,
        graph_state=graph_state,
        diagnostics=tuple(evidence.diagnostics),
        dependency_evidence=evidence,
    )


def _collect_group_package_state(
    sandbox: DockerSandbox,
    group: VulnerabilityGroup,
    policy: QAPolicy | None,
    *,
    task: RemediationTask | None = None,
    expected_version: str | None = None,
    version_evidence_inconclusive: bool = False,
) -> _QAPackageState:
    """Collect deterministic dependency state for package-backed QA policies."""
    if policy not in {
        QAPolicy.VERSION_BUMP,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }:
        return _QAPackageState()

    task_package = (
        getattr(task, "target_package_name", None) if policy == QAPolicy.VERSION_BUMP else None
    )
    package = str(task_package or group.vulnerable_component or "").strip()
    manifests = list(_manifest_paths_for_group(group))
    lockfiles = _lockfile_paths_for_group(group)
    managers = {
        (getattr(issue, "package_manager", "") or "").strip().lower()
        for issue in (getattr(group, "localized_issues", []) or [])
        if (getattr(issue, "package_manager", "") or "").strip()
    }

    if policy == QAPolicy.VERSION_BUMP:
        if _group_is_python(group):
            unsupported_managers = managers - {"pip", "pipenv"}
        else:
            unsupported_managers = managers - {"npm"}
        if unsupported_managers:
            manager_text = ", ".join(sorted(managers))
            diagnostic = f"Unsupported or unknown package manager(s): {manager_text}."
            evidence = QADependencyEvidence(
                status=DependencyEvidenceStatus.INCONCLUSIVE,
                target_package=package,
                expected_version=(
                    _python_version(expected_version)
                    if _group_is_python(group)
                    else _normalise_dependency_version(expected_version)
                ),
                manifest_paths=manifests,
                lockfile_paths=list(lockfiles),
                diagnostics=[diagnostic],
            )
            return _QAPackageState(
                manifest_state="unknown",
                graph_state="unknown",
                diagnostics=(diagnostic,),
                dependency_evidence=evidence,
            )
        return _collect_dependency_package_state(
            sandbox,
            group,
            task,
            manifests,
            expected_version,
            lockfiles=lockfiles,
            version_evidence_inconclusive=version_evidence_inconclusive,
        )

    if _group_is_python(group):
        return _collect_python_package_presence_state(
            sandbox,
            group,
            package,
            manifests,
            lockfiles,
            expect_present=policy == QAPolicy.NO_FIX_CODE_REMOVAL,
        )
    if managers != {"npm"}:
        manager_text = ", ".join(sorted(managers)) if managers else "unknown"
        return _QAPackageState(
            manifest_state="unknown",
            graph_state="unknown",
            diagnostics=(f"Unsupported or unknown package manager(s): {manager_text}.",),
        )
    if not package or not manifests:
        return _QAPackageState(
            manifest_state="unknown",
            graph_state="unknown",
            diagnostics=("Package name or authorized manifest paths are unavailable.",),
        )

    parsed_manifests: list[tuple[str, dict[str, Any]]] = []
    for manifest in manifests:
        payload = _workspace_json_file(sandbox, manifest)
        if payload is not None:
            parsed_manifests.append((manifest, payload))
    manifest_state = (
        "unknown"
        if len(parsed_manifests) != len(manifests)
        else (
            "present"
            if any(_json_dependency_present(payload, package) for _, payload in parsed_manifests)
            else "absent"
        )
    )

    graph_states: list[str] = []
    graph_diagnostics: list[str] = []
    for manifest, _payload in parsed_manifests:
        cwd_path = Path(manifest).parent
        cwd = "" if str(cwd_path) == "." else cwd_path.as_posix().strip("/")
        prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        command = f"{prefix}npm ls {shlex.quote(package)} --all --json"
        try:
            command_result = _run_readonly(
                sandbox,
                command,
                timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
            )
            raw = (command_result.stdout or "").strip()
            tree = json.loads(raw) if raw else None
            if isinstance(tree, dict):
                graph_states.append(
                    "present" if _dependency_tree_contains(tree, package) else "absent"
                )
            else:
                graph_states.append("unknown")
                graph_diagnostics.append(f"npm ls returned no parseable graph for {manifest}.")
        except Exception as exc:  # noqa: BLE001
            graph_states.append("unknown")
            graph_diagnostics.append(f"npm graph inspection failed for {manifest}: {exc}")
    graph_state = (
        "unknown"
        if len(parsed_manifests) != len(manifests) or not graph_states or "unknown" in graph_states
        else ("present" if "present" in graph_states else "absent")
    )
    return _QAPackageState(
        manifest_state=manifest_state,
        graph_state=graph_state,
        diagnostics=tuple(graph_diagnostics),
    )


def _scan_state_projection(
    results: _QAExecutionResults,
    baseline_identifiers: set[str],
    *,
    authoritative: bool = True,
) -> dict[str, Any]:
    """Project the scan cache into serializable graph-state fields."""
    if not authoritative:
        return {
            "baseline_scan_identifiers": sorted(baseline_identifiers),
            "post_remediation_scan_identifiers": [],
            "post_remediation_scan_issues": [],
            "new_vulnerability_identifiers": [],
            "new_vulnerability_status": "not_scanned",
        }

    scan_result = results.scan
    if scan_result is None:
        return {
            "baseline_scan_identifiers": sorted(baseline_identifiers),
            "post_remediation_scan_identifiers": [],
            "post_remediation_scan_issues": [],
            "new_vulnerability_identifiers": [],
            "new_vulnerability_status": "not_scanned",
        }

    found = set(_scan_result_value(scan_result, "found_identifiers", set()) or set())
    new_identifiers = set(_scan_result_value(scan_result, "new_identifiers", set()) or set())
    found_issues = list(_scan_result_value(scan_result, "found_issues", []) or [])
    scan_ok = bool(_scan_result_value(scan_result, "ok", False))
    remaining = set(_scan_result_value(scan_result, "remaining_identifiers", set()) or set())
    if isinstance(scan_result, _SecurityScanResult) and not scan_ok and not found and not remaining:
        status = "scan_failed"
    else:
        status = "detected" if new_identifiers else "none"

    return {
        "baseline_scan_identifiers": sorted(baseline_identifiers),
        "post_remediation_scan_identifiers": sorted(found),
        "post_remediation_scan_issues": found_issues,
        "new_vulnerability_identifiers": sorted(new_identifiers),
        "new_vulnerability_status": status,
    }


def _pipeline_complete(results: _QAExecutionResults) -> bool:
    """Return whether install, scan-or-skip, and tests have run at least once."""
    scan_complete = results.scan is not None or results.scan_skipped
    return results.install is not None and scan_complete and results.tests is not None


def _review_ready_error(results: _QAExecutionResults) -> str | None:
    """Return the standard review-tool order error, if any."""
    if _pipeline_complete(results):
        return None
    return (
        "ERROR: Review tools are locked until run_dependency_install, "
        "run_security_scan (or an explicit scan skip), and run_unit_tests have all "
        "been called in order."
    )


def _resolve_action_summary_task_ids(
    summary: AgentActionSummary,
    known_task_ids: set[str],
) -> list[str]:
    """Resolve which exact task IDs an AgentActionSummary applies to."""
    raw_task_id = (summary.task_id or "").strip()
    if not raw_task_id:
        return []
    if raw_task_id.startswith("batch:"):
        payload = raw_task_id[len("batch:") :]
        resolved = []
        for part in payload.split(","):
            candidate = part.strip()
            if candidate and candidate in known_task_ids:
                resolved.append(candidate)
        return resolved
    return [raw_task_id] if raw_task_id in known_task_ids else []


def _relevant_action_summaries(
    action_summaries: list[AgentActionSummary],
    task_id: str,
    known_task_ids: set[str],
) -> list[AgentActionSummary]:
    """Filter action summaries to those explicitly linked to one task."""
    relevant: list[AgentActionSummary] = []
    for summary in action_summaries:
        if task_id in _resolve_action_summary_task_ids(summary, known_task_ids):
            relevant.append(summary)
    return relevant


def _trim_action_summary_text(summary_text: str, group: VulnerabilityGroup) -> str:
    """Trim an action summary to the component owned by the evaluated task."""
    group_id = group.group_id
    match = re.search(r"(updates for |edits for )(.+?)(;|$)", summary_text)
    if match:
        groups_list_str = match.group(2)
        if "," in groups_list_str:
            groups = [g.strip() for g in groups_list_str.split(",")]
            if group_id in groups:
                summary_text = summary_text.replace(groups_list_str, group_id, 1)

    # Collect possible match keywords for this group
    keywords = set()
    if group.vulnerable_component:
        keywords.add(group.vulnerable_component.lower())
    for cve in group.cve_ids or []:
        if cve:
            keywords.add(cve.lower())
    for ghsa in group.ghsa_ids or []:
        if ghsa:
            keywords.add(ghsa.lower())
    for issue in group.issues or []:
        if issue.cve_id:
            keywords.add(issue.cve_id.lower())
        if issue.ghsa_id:
            keywords.add(issue.ghsa_id.lower())

    if not keywords:
        return summary_text

    lines = summary_text.splitlines()
    trimmed_lines = []
    for line in lines:
        stripped = line.strip()
        is_bullet = stripped.startswith(("-", "*", "+")) or re.match(r"^\d+\.", stripped)
        is_action_verb = any(
            stripped.lower().startswith(verb)
            for verb in ["updated", "added", "fixed", "upgraded", "downgraded", "removed"]
        )
        if is_bullet or is_action_verb:
            line_lower = stripped.lower()
            has_kw = False
            for kw in keywords:
                # Custom word boundary matching to handle special characters like @ or - in package names
                pattern = rf"(?:^|[^a-zA-Z0-9_@.-]){re.escape(kw)}(?:$|[^a-zA-Z0-9_.-])"
                if re.search(pattern, line_lower):
                    has_kw = True
                    break
            if has_kw:
                trimmed_lines.append(line)
        else:
            trimmed_lines.append(line)

    return "\n".join(trimmed_lines)


def _bounded_qa_action_summary(summary_text: str, group: VulnerabilityGroup) -> str:
    """Trim one active evaluator action summary to a bounded context budget."""
    summary = _trim_action_summary_text(summary_text, group)
    if len(summary) <= _QA_ACTION_SUMMARY_MAX_CHARS:
        return summary
    marker = "... (summary truncated)"
    return summary[: _QA_ACTION_SUMMARY_MAX_CHARS - len(marker)].rstrip() + marker


def _parse_report_bullets(block_text: str) -> dict[str, str]:
    """Parse markdown '- Label: value' bullets, preserving wrapped lines."""
    fields: dict[str, str] = {}
    current_label: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        """Commit the current markdown bullet to the parsed field mapping."""
        nonlocal current_label, current_lines
        if current_label is None:
            return
        fields[current_label] = "\n".join(current_lines).strip()
        current_label = None
        current_lines = []

    for raw_line in block_text.splitlines():
        line = raw_line.rstrip()
        match = _BULLET_LABEL_RE.match(line)
        if match:
            flush()
            current_label = match.group(1).strip()
            current_lines = [match.group(2).strip()]
            continue
        if current_label is not None:
            current_lines.append(line.strip())

    flush()
    return fields


def group_target_identifiers(group: VulnerabilityGroup) -> set[str]:
    """Collect normalized scanner identifiers relevant to one group."""
    identifiers: set[str] = set()
    for cve in group.cve_ids or []:
        if cve:
            identifiers.add(cve.upper().strip())
    for ghsa in group.ghsa_ids or []:
        if ghsa:
            identifiers.add(ghsa.upper().strip())
    for issue in group.issues or []:
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _build_qa_task_contexts(
    valid_groups: Sequence[VulnerabilityGroup],
    task_queue: Mapping[str, Any],
    active_target_task_ids: Sequence[str],
) -> list[QATaskContext]:
    """Resolve active task IDs to their authoritative vulnerability groups."""
    contexts: list[QATaskContext] = []
    seen: set[str] = set()
    groups_by_id = {group.group_id: group for group in valid_groups}
    for task_id in active_target_task_ids:
        if task_id in seen:
            raise ValueError(f"duplicate active task ID {task_id!r}")
        seen.add(task_id)
        task = task_queue.get(task_id)
        if task is None:
            raise ValueError(f"active task ID {task_id!r} is missing from task_queue")
        if not getattr(task, "current_attempt_id", None):
            raise ValueError(f"active task {task_id!r} has no committed attempt")
        group = groups_by_id.get(getattr(task, "parent_group_id", ""))
        if group is None:
            raise ValueError(
                f"task {task_id!r} references missing parent group "
                f"{getattr(task, 'parent_group_id', '')!r}"
            )
        contexts.append(QATaskContext(task=task, group=group))
    return contexts


def _derive_qa_task_strategies(
    valid_groups: list[VulnerabilityGroup],
    configured_strategies: Mapping[str, Any] | None,
    task_queue: Mapping[str, Any] | None,
    active_target_task_ids: Sequence[str] | None = None,
) -> dict[str, str]:
    """Resolve the Supervisor-selected strategy for each active task.

    Results are keyed by task ID because a parent group may own multiple
    active tasks with different stages or strategies.
    """
    groups_by_id = {group.group_id: group for group in valid_groups}
    queue = dict(task_queue or {})
    configured = dict(configured_strategies or {})
    effective: dict[str, str] = {}
    for task_id in active_target_task_ids or ():
        task = queue.get(task_id)
        if task is None:
            continue
        group = groups_by_id.get(getattr(task, "parent_group_id", ""))
        if group is None:
            continue
        stage = getattr(task, "no_fix_stage", None)
        stage_value = getattr(stage, "value", stage)
        if stage_value == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value:
            effective[task_id] = "code_workaround"
            continue
        if stage_value == NoFixMitigationStage.PACKAGE_REMOVAL.value:
            effective[task_id] = "no_fix_package_removal"
            continue
        raw_strategy = getattr(task, "strategy", None)
        if raw_strategy is None:
            raw_strategy = configured.get(task_id)
        if raw_strategy is not None:
            effective[task_id] = str(getattr(raw_strategy, "value", raw_strategy))
    return effective


def _derive_qa_task_policies(
    valid_groups: list[VulnerabilityGroup],
    task_queue: Mapping[str, Any] | None,
    active_target_task_ids: Sequence[str] | None = None,
    attempt_snapshots_by_id: Mapping[str, Any] | None = None,
) -> dict[str, QAPolicy | None]:
    """Resolve committed QA policy provenance independently for each task."""
    del valid_groups
    queue = dict(task_queue or {})
    snapshots = dict(attempt_snapshots_by_id or {})
    policies: dict[str, QAPolicy | None] = {}
    for task_id in active_target_task_ids or ():
        task = queue.get(task_id)
        if task is None or not getattr(task, "current_attempt_id", None):
            policies[task_id] = None
            continue
        snapshot = snapshots.get(task.current_attempt_id)
        snapshot_policy = (
            snapshot.get("qa_policy")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "qa_policy", None)
            if snapshot is not None
            else None
        )
        task_policy = getattr(task, "qa_policy", None)
        try:
            normalized_snapshot_policy = (
                snapshot_policy
                if isinstance(snapshot_policy, QAPolicy)
                else QAPolicy(str(snapshot_policy))
                if snapshot_policy is not None
                else None
            )
            normalized_task_policy = (
                task_policy
                if isinstance(task_policy, QAPolicy)
                else QAPolicy(str(task_policy))
                if task_policy is not None
                else None
            )
        except ValueError:
            normalized_snapshot_policy = None
            normalized_task_policy = None
        if (
            normalized_snapshot_policy is None
            or normalized_task_policy is None
            or normalized_snapshot_policy != normalized_task_policy
        ):
            logger.error(
                "qa_critic: missing or contradictory QA policy provenance for task %s.",
                task_id,
            )
            policies[task_id] = None
        else:
            policies[task_id] = normalized_snapshot_policy
    return policies


def _group_scan_status(
    scan_result: _SecurityScanResult | None,
    group: VulnerabilityGroup,
    *,
    scan_skipped: bool = False,
) -> str:
    """Classify the scanner outcome for one group from deterministic results."""
    if scan_skipped:
        return "skipped"
    if scan_result is None:
        return "scan_failed"

    scan_ok = bool(_scan_result_value(scan_result, "ok", False))
    remaining_identifiers = set(
        _scan_result_value(scan_result, "remaining_identifiers", set()) or set()
    )
    if group_target_identifiers(group) & remaining_identifiers:
        return "still_flagged"
    if remaining_identifiers or scan_ok:
        return "cleared"
    return "scan_failed"


def _build_fallback_investigation_report(
    valid_groups: list[VulnerabilityGroup],
    group_strategies: dict[str, str],
    candidate_changed_files: list[str],
    results: _QAExecutionResults,
    reason: str,
) -> str:
    """Synthesize a minimal investigative report when the LLM output is malformed."""
    install_ok, install_summary = results.install or (
        False,
        "run_dependency_install was not called.",
    )
    scan_result: _SecurityScanResult | None = results.scan
    tests_ok, _ = results.tests or (False, "run_unit_tests was not called.")

    post_scan_identifiers = sorted(
        _scan_result_value(scan_result, "found_identifiers", set()) or set()
    )
    new_identifiers = sorted(_scan_result_value(scan_result, "new_identifiers", set()) or set())

    changed_files_text = ", ".join(candidate_changed_files) if candidate_changed_files else "none"
    blocks = [
        _REPORT_PREFIX,
        "## Install Analysis",
        f"- Install Status: {'succeeded' if install_ok else 'failed'}",
        f"- Summary: {reason}",
        "- Suspected Responsible Group(s): unknown",
        f"- Evidence: {install_summary}",
        f"- Post-remediation Scanner Identifiers: {', '.join(post_scan_identifiers) if post_scan_identifiers else 'none'}",
        f"- Newly Introduced Scanner Identifiers: {', '.join(new_identifiers) if new_identifiers else 'none'}",
        (
            f"- Scanner Phase: skipped ({results.scan_skip_reason or 'explicit policy'})"
            if results.scan_skipped
            else "- Scanner Phase: executed"
        ),
        "",
    ]

    for group in valid_groups:
        strategy = group_strategies.get(group.group_id, "")
        group_identifiers = sorted(group_target_identifiers(group))
        group_remaining = sorted(
            group_target_identifiers(group)
            & set(_scan_result_value(scan_result, "remaining_identifiers", set()) or set())
        )
        scan_status = _group_scan_status(
            scan_result,
            group,
            scan_skipped=results.scan_skipped,
        )
        blocks.extend(
            [
                f"### GROUP: {group.group_id}",
                f"- Component: {group.vulnerable_component or '(unknown)'}",
                f"- Strategy: {strategy}",
                f"- Target Identifiers: {', '.join(group_identifiers) if group_identifiers else 'none'}",
                f"- Changed Files: {changed_files_text}",
                "",
                f"- Scan Status: {scan_status}",
                f"- Remaining Scanner Findings: {', '.join(group_remaining) if group_remaining else 'none'}",
                "- Scan Reasoning: Investigation report was synthesized from deterministic QA results.",
                "",
                (
                    "- Workaround Review: not applicable"
                    if strategy != "code_workaround"
                    else "- Workaround Review: not reviewed; fallback report due to malformed investigator output."
                ),
                (
                    "- Diff Evidence: not applicable"
                    if strategy != "code_workaround"
                    else "- Diff Evidence: none reviewed."
                ),
                "",
                f"- Test Status: {'passed' if tests_ok else 'failed'}",
                "- Attributed Test Failures: none",
                "- Causal Reasoning: No trusted investigator prose was available; defer to deterministic QA logs.",
                "- Exonerated Groups: none",
                "",
                "- Group Summary: Fallback summary generated because the investigator output was missing or malformed.",
                "",
            ]
        )
    return "\n".join(blocks).strip()


def _targeted_extra_args_conflict() -> bool:
    """Return whether configured ODC arguments override required target paths."""
    extra_args = get_runtime_settings().odc_extra_args
    if not extra_args:
        return False
    try:
        tokens = shlex.split(extra_args)
    except ValueError:
        return True
    return any(
        token in {"--scan", "--out"} or token.startswith("--scan=") or token.startswith("--out=")
        for token in tokens
    )


def _closure_fallback_reason(reason: str | None) -> ScanFallbackReason:
    """Map pure resolver diagnostics to the typed QA fallback vocabulary."""
    return {
        "no_matching_target": ScanFallbackReason.NO_MATCHING_TARGET,
        "multiple_targets": ScanFallbackReason.MULTIPLE_TARGETS,
        "incomplete_closure": ScanFallbackReason.INCOMPLETE_CLOSURE,
        "invalid_lockfile": ScanFallbackReason.INVALID_LOCKFILE,
    }.get(reason or "", ScanFallbackReason.INCOMPLETE_CLOSURE)


def _merge_dependency_closures(
    source_lockfile: str,
    closures: Sequence[DependencyClosure],
) -> DependencyClosure:
    """Union complete closures from one lockfile without losing physical keys."""
    node_map = {node.lockfile_key: node for closure in closures for node in closure.nodes}
    return DependencyClosure(
        source_lockfile=source_lockfile,
        root_keys=tuple(sorted({key for closure in closures for key in closure.root_keys})),
        nodes=tuple(node_map[key] for key in sorted(node_map)),
        includes_optional=any(closure.includes_optional for closure in closures),
        includes_peer=any(closure.includes_peer for closure in closures),
        complete=all(closure.complete for closure in closures),
        lockfile_version=closures[0].lockfile_version,
    )


def _resolve_targeted_closures(
    sandbox: DockerSandbox,
    targets: Sequence[QAScanTarget],
) -> tuple[list[DependencyClosure], ScanFallbackReason | None, str | None]:
    """Read live npm lockfiles and resolve the union needed by active tasks."""
    if not targets:
        return [], ScanFallbackReason.MISSING_LOCKFILE, "No active task scan targets were supplied."

    by_source: dict[str, list[QAScanTarget]] = {}
    for target in targets:
        if not target.manifest_paths:
            return (
                [],
                ScanFallbackReason.MISSING_LOCKFILE,
                (f"Task {target.task_id} has no supported lockfile path."),
            )
        for source_lockfile in target.manifest_paths:
            source_lockfile = _validate_qa_path(source_lockfile)
            if Path(source_lockfile).name.lower() != "package-lock.json":
                return (
                    [],
                    ScanFallbackReason.UNSUPPORTED_PACKAGE_MANAGER,
                    (f"Task {target.task_id} uses unsupported lockfile {source_lockfile}."),
                )
            by_source.setdefault(source_lockfile, []).append(target)

    merged: list[DependencyClosure] = []
    for source_lockfile, source_targets in sorted(by_source.items()):
        raw_lockfile = sandbox.read_file(source_lockfile)
        if raw_lockfile is None:
            return (
                [],
                ScanFallbackReason.MISSING_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} could not be read."),
            )
        try:
            lockfile = json.loads(raw_lockfile)
        except (TypeError, json.JSONDecodeError) as exc:
            return (
                [],
                ScanFallbackReason.INVALID_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} is not valid JSON: {exc}"),
            )
        packages = lockfile.get("packages") if isinstance(lockfile, Mapping) else None
        lockfile_version = (
            lockfile.get("lockfileVersion") if isinstance(lockfile, Mapping) else None
        )
        if not isinstance(packages, Mapping) or not isinstance(lockfile_version, int):
            return (
                [],
                ScanFallbackReason.INVALID_LOCKFILE,
                (f"Live workspace lockfile {source_lockfile} lacks a supported packages map."),
            )

        closures: list[DependencyClosure] = []
        for target in source_targets:
            closure = resolve_dependency_closure(
                packages,
                source_lockfile=source_lockfile,
                target_package=target.target_package,
                target_version=target.expected_version,
                dependency_ancestry=target.dependency_ancestry,
                include_optional=True,
                include_peer=True,
                lockfile_version=lockfile_version,
            )
            if not closure.complete:
                return (
                    [],
                    _closure_fallback_reason(closure.fallback_reason),
                    (
                        f"Task {target.task_id} closure failed for {source_lockfile}: "
                        f"{closure.fallback_reason or 'unknown reason'}"
                    ),
                )
            closures.append(closure)
        merged.append(_merge_dependency_closures(source_lockfile, closures))
    return merged, None, None


def _write_targeted_artifacts(
    sandbox: DockerSandbox,
    closures: Sequence[DependencyClosure],
) -> tuple[str, list[str], list[str]]:
    """Write synthetic package roots and return scan path plus closure metadata."""
    targeted_subdir = ".odc-targeted"
    package_names: set[str] = set()
    lockfile_keys: set[str] = set()
    for index, closure in enumerate(closures):
        artifacts = build_sliced_lockfile_artifacts(closure)
        subdir = f"{targeted_subdir}/{index:03d}"
        for filename, content in artifacts.items():
            sandbox.write_file(_validate_qa_path(f"{subdir}/{filename}"), content)
        package_names.update(node.package_name for node in closure.nodes)
        lockfile_keys.update(node.lockfile_key for node in closure.nodes)
    return targeted_subdir, sorted(package_names), sorted(lockfile_keys)


def _cleanup_targeted_artifacts(sandbox: DockerSandbox) -> None:
    """Remove the fixed temporary targeted-scan directory from the workspace."""
    try:
        sandbox.run("rm -rf -- .odc-targeted", timeout=30)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: targeted artifact cleanup failed — %s", exc)


def _scan_evidence(
    *,
    targets: Sequence[QAScanTarget],
    scan_result: Any,
    effective_scope: ScanScope,
    complete: bool,
    fallback_reason: ScanFallbackReason | None = None,
    closures: Sequence[DependencyClosure] = (),
) -> ODCScanEvidence:
    """Build typed, attempt-local ODC evidence from a scan result."""
    return ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=effective_scope,
        authoritative=False,
        covered_task_ids=sorted({target.task_id for target in targets}),
        closure_package_names=sorted(
            {node.package_name for closure in closures for node in closure.nodes}
        ),
        closure_lockfile_keys=sorted(
            {node.lockfile_key for closure in closures for node in closure.nodes}
        ),
        found_identifiers=sorted(
            _scan_result_value(scan_result, "found_identifiers", set()) or set()
        ),
        remaining_target_identifiers=sorted(
            _scan_result_value(scan_result, "remaining_identifiers", set()) or set()
        ),
        complete=complete,
        fallback_reason=fallback_reason,
    )
