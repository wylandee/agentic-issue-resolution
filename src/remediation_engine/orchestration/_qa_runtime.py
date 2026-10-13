"""Shared runtime state and scan-target helpers for QA execution.

This module owns deterministic execution evidence projections, workspace
diffing, and preparation of scan targets from the Supervisor's task queue and
committed attempt snapshots. Docker-backed workspaces are treated as
ephemeral; helpers read them for evidence and never mutate the host baseline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    DependencyEvidenceStatus,
    NoFixMitigationStage,
    ODCScanEvidence,
    PackageScanOccurrence,
    QADependencyEvidence,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScanScope,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.scan_identity import package_scan_occurrence_key
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.lockfile_closure import (
    DependencyClosure,
    TargetPackageResolution,
    resolve_dependency_closure_for_target,
    resolve_target_packages,
)

from .qa_odc import (
    _ODC_HTML_REPORT_NAME,
    _ODC_REPORT_NAME,
    _normalise_odc_report_path,
)
from .qa_test_parsing import _NPM_INSTALL_TIMEOUT_SECONDS, _workspace_json_file
from .qa_types import (
    QAScanTarget,
    QATaskContext,
    _append_qa_log_records,
    _ODCReportedPackageOccurrence,
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


@dataclass(frozen=True)
class _TargetedPackageResolutionFailure:
    """Task-scoped reason and diagnostic for a target that could not be resolved."""

    reason: ScanFallbackReason
    detail: str


@dataclass(frozen=True)
class _NonNpmInputProject:
    """One changed non-npm ODC input and its staged companion files."""

    source_input_path: str
    staged_input_paths: tuple[str, ...]
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class _TargetedPackageBatch:
    """Successfully resolved packages and isolated failures from a batch lookup."""

    target_packages: tuple[TargetPackageResolution, ...]
    closures_by_task: Mapping[str, tuple[DependencyClosure, ...]]
    covered_task_ids: tuple[str, ...]
    failures_by_task: Mapping[str, _TargetedPackageResolutionFailure]
    inventory_updates: Mapping[str, str]
    inventory_removals: tuple[str, ...]
    covered_input_paths: tuple[str, ...]
    non_npm_input_projects: tuple[_NonNpmInputProject, ...]
    fallback_reason: ScanFallbackReason | None = None
    fallback_detail: str | None = None
    skipped_task_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TargetedArtifactManifest:
    """Artifact identities and original occurrence attribution for one scan."""

    scan_subdir: str
    expected_package_occurrences: tuple[tuple[str, str], ...]
    occurrences_by_project: Mapping[str, tuple[PackageScanOccurrence, ...]]
    task_ids_by_project: Mapping[str, tuple[str, ...]]
    input_paths_by_project: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class _PackageLockChangeBatch:
    """Inventory deltas and exact closures for changed package-lock entries."""

    closures_by_task: Mapping[str, tuple[DependencyClosure, ...]]
    target_packages: tuple[TargetPackageResolution, ...]
    inventory_updates: Mapping[str, str]
    inventory_removals: tuple[str, ...]
    covered_input_paths: tuple[str, ...]
    failures_by_task: Mapping[str, _TargetedPackageResolutionFailure]


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


def _lockfile_paths_for_group(group: VulnerabilityGroup) -> tuple[str, ...]:
    """Return normalized npm lockfile paths associated with an SCA group."""
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


def _package_occurrence_key(
    source_input_path: str,
    lockfile_key: str | None,
    package_name: str,
    version: str,
    occurrence_index: int = 0,
) -> str:
    """Return the canonical identity for one reported package occurrence."""
    return package_scan_occurrence_key(
        source_input_path,
        lockfile_key,
        package_name,
        version,
        occurrence_index,
    )


def _package_lock_inventory(
    lockfile_path: str,
    lockfile: Mapping[str, Any],
) -> dict[str, str]:
    """Fingerprint ODC-relevant metadata for every physical lockfile entry.

    The root package entry is included because changes to its declared edges
    can alter the installed graph even though it is not an ODC package
    occurrence. Callers replace the accepted inventory with this complete
    snapshot so absent keys represent deletions.

    Args:
        lockfile_path: Repository-relative path to the package lockfile.
        lockfile: Decoded npm package-lock document.

    Returns:
        Canonical occurrence keys mapped to SHA-256 metadata fingerprints.

    Raises:
        ValueError: If the path or package map cannot be represented safely.
    """
    source_input_path = str(lockfile_path).strip().replace("\\", "/")
    while source_input_path.startswith("./"):
        source_input_path = source_input_path[2:]
    path_parts = source_input_path.split("/")
    if (
        not source_input_path
        or source_input_path.startswith("/")
        or any(part in {"", ".", ".."} for part in path_parts)
    ):
        raise ValueError(f"invalid package-lock path: {lockfile_path!r}")

    packages = lockfile.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError(f"package-lock {source_input_path!r} has no packages mapping")

    inventory: dict[str, str] = {}
    relevant_edge_fields = (
        "dependencies",
        "optionalDependencies",
        "peerDependencies",
        "peerDependenciesMeta",
    )
    for lockfile_key, raw_metadata in sorted(packages.items(), key=lambda item: str(item[0])):
        if not isinstance(lockfile_key, str) or not isinstance(raw_metadata, Mapping):
            raise ValueError(f"invalid package-lock entry in {source_input_path!r}")
        physical_name = lockfile_key.rsplit("node_modules/", 1)[-1] if lockfile_key else ""
        raw_name = raw_metadata.get("name")
        package_name = str(raw_name) if isinstance(raw_name, str) and raw_name else physical_name
        raw_version = raw_metadata.get("version")
        version = str(raw_version) if raw_version is not None else ""
        odc_metadata: dict[str, Any] = {
            "name": package_name,
            "version": version,
        }
        for field_name in relevant_edge_fields:
            if field_name in raw_metadata:
                odc_metadata[field_name] = raw_metadata[field_name]
        try:
            encoded_metadata = json.dumps(
                odc_metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid ODC metadata for {lockfile_key!r} in {source_input_path!r}"
            ) from exc
        occurrence_key = _package_occurrence_key(
            source_input_path,
            lockfile_key,
            package_name,
            version,
        )
        inventory[occurrence_key] = hashlib.sha256(encoded_metadata).hexdigest()
    return inventory


def _parse_package_occurrence_key(
    occurrence_key: str,
) -> tuple[str, str | None, str, str, int] | None:
    """Parse and validate the canonical package occurrence key."""
    try:
        values = json.loads(occurrence_key)
    except (TypeError, ValueError):
        return None
    if not isinstance(values, list) or len(values) != 5:
        return None
    source_input_path, lockfile_key, package_name, version, occurrence_index = values
    if (
        not isinstance(source_input_path, str)
        or not isinstance(lockfile_key, str)
        or not isinstance(package_name, str)
        or package_name != package_name.casefold()
        or not isinstance(version, str)
        or isinstance(occurrence_index, bool)
        or not isinstance(occurrence_index, int)
        or occurrence_index < 0
    ):
        return None
    try:
        normalized_path = _validate_qa_path(source_input_path)
    except ValueError:
        return None
    if (
        normalized_path != source_input_path
        or _package_occurrence_key(
            source_input_path,
            lockfile_key,
            package_name,
            version,
            occurrence_index,
        )
        != occurrence_key
    ):
        return None
    return source_input_path, lockfile_key, package_name, version, occurrence_index


def _worker_package_lock_paths(
    state: OrchestratorState | None,
    active_task_ids: set[str],
    *,
    package_lock_only: bool = True,
) -> tuple[dict[str, set[str]], list[str]]:
    """Collect changed workspace files from current and retained pivot attempts.

    By default only package-lock paths are returned. Non-npm callers request
    all changed paths and filter them by their supported ODC input types.
    """
    if state is None or not active_task_ids:
        return {}, []

    task_queue = state.get("task_queue", {}) or {}
    worker_results = state.get("worker_results_by_attempt", {}) or {}
    attempt_snapshots = state.get("attempt_snapshots_by_id", {}) or {}
    if not isinstance(task_queue, Mapping) or not isinstance(worker_results, Mapping):
        return {}, ["worker attempt state is not a mapping"]

    active_by_retained_parent: dict[str, set[str]] = {}
    for active_task_id in active_task_ids:
        current_task = task_queue.get(active_task_id)
        parent_task_id = _scan_result_value(current_task, "parent_task_id", None)
        visited: set[str] = set()
        while isinstance(parent_task_id, str) and parent_task_id not in visited:
            visited.add(parent_task_id)
            parent_task = task_queue.get(parent_task_id)
            if parent_task is None:
                break
            status = _scan_result_value(parent_task, "status", None)
            status_value = str(getattr(status, "value", status)).casefold()
            if status_value == "pivoted":
                active_by_retained_parent.setdefault(parent_task_id, set()).add(active_task_id)
            parent_task_id = _scan_result_value(parent_task, "parent_task_id", None)

    def _latest_retained_result(task_id: str) -> tuple[str, Any] | None:
        """Return the newest worker result for a retained pivot parent."""
        candidates: list[tuple[tuple[int, int, int, str], str, Any]] = []
        for attempt_key, result in worker_results.items():
            if _scan_result_value(result, "task_id", None) != task_id:
                continue
            attempt_id = str(_scan_result_value(result, "attempt_id", None) or attempt_key)
            snapshot = (
                attempt_snapshots.get(attempt_id)
                if isinstance(attempt_snapshots, Mapping)
                else None
            )

            def _as_int(value: Any) -> int:
                try:
                    return int(value or 0)
                except (TypeError, ValueError):
                    return 0

            order = (
                _as_int(_scan_result_value(result, "task_revision", 0)),
                _as_int(_scan_result_value(snapshot, "attempt_number", 0)),
                _as_int(_scan_result_value(snapshot, "state_revision", 0)),
                attempt_id,
            )
            candidates.append((order, attempt_id, result))
        if not candidates:
            return None
        _order, attempt_id, result = max(candidates, key=lambda item: item[0])
        return attempt_id, result

    task_attempts: dict[str, tuple[str, Any, set[str]]] = {}
    for task_id in active_task_ids:
        task = task_queue.get(task_id)
        attempt_id = _scan_result_value(task, "current_attempt_id", None)
        if isinstance(attempt_id, str) and attempt_id:
            result = worker_results.get(attempt_id)
            if result is not None and _scan_result_value(result, "task_id", None) == task_id:
                task_attempts[task_id] = (attempt_id, result, {task_id})
    for parent_task_id, descendants in active_by_retained_parent.items():
        parent_task = task_queue.get(parent_task_id)
        attempt_id = _scan_result_value(parent_task, "current_attempt_id", None)
        result = worker_results.get(attempt_id) if isinstance(attempt_id, str) else None
        if result is None:
            latest = _latest_retained_result(parent_task_id)
            if latest is not None:
                attempt_id, result = latest
        if result is not None:
            task_attempts[parent_task_id] = (str(attempt_id), result, descendants)

    paths_by_task: dict[str, set[str]] = {}
    errors: list[str] = []
    for owner_task_id, (_attempt_id, result, affected_tasks) in task_attempts.items():
        changed_files = _scan_result_value(result, "changed_files", []) or []
        if isinstance(changed_files, (str, bytes)) or not isinstance(changed_files, Sequence):
            errors.append(f"worker result for {owner_task_id} has invalid changed_files")
            continue
        for raw_path in changed_files:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            try:
                path = _validate_qa_path(raw_path)
            except ValueError as exc:
                errors.append(f"worker result for {owner_task_id} has an invalid path: {exc}")
                continue
            if package_lock_only and Path(path).name.casefold() != "package-lock.json":
                continue
            paths_by_task.setdefault(path, set()).update(affected_tasks)
    return paths_by_task, errors


def _non_npm_input_companions(
    source_input_path: str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]] | None:
    """Return required companion alternatives and optional companions."""
    basename = Path(source_input_path).name.casefold()
    required = {
        "cargo.lock": (("Cargo.toml",),),
        "go.sum": (("go.mod",),),
        "pipfile.lock": (("Pipfile",),),
        "poetry.lock": (("pyproject.toml",),),
        "uv.lock": (("pyproject.toml",),),
        "pdm.lock": (("pyproject.toml",),),
        "composer.lock": (("composer.json",),),
        "gemfile.lock": (("Gemfile",),),
        "gradle.lockfile": (("build.gradle", "build.gradle.kts"),),
    }
    optional = {
        "cargo.toml": ("Cargo.lock",),
        "go.mod": ("go.sum",),
        "pipfile": ("Pipfile.lock",),
        "pyproject.toml": ("poetry.lock", "uv.lock", "pdm.lock"),
        "composer.json": ("composer.lock",),
        "gemfile": ("Gemfile.lock",),
        "build.gradle": ("gradle.lockfile",),
        "build.gradle.kts": ("gradle.lockfile",),
    }
    if basename in required:
        return required[basename], ()
    if basename in optional:
        return (), optional[basename]
    if basename == "pom.xml" or (basename.startswith("requirements") and basename.endswith(".txt")):
        return (), ()
    if Path(basename).suffix in {".jar", ".war", ".ear", ".aar", ".nupkg", ".whl"}:
        return (), ()
    return None


def _workspace_file_exists(sandbox: DockerSandbox, path: str) -> bool:
    """Check one normalized workspace file without reading potentially binary data."""
    try:
        result = sandbox.run(f"test -f {shlex.quote(path)}", timeout=30)
    except Exception:  # noqa: BLE001 - inability to map an input is inconclusive
        return False
    return result.exit_code == 0


def _collect_non_npm_input_projects(
    sandbox: DockerSandbox,
    state: OrchestratorState | None,
    targets: Sequence[QAScanTarget],
) -> tuple[
    tuple[_NonNpmInputProject, ...],
    dict[str, _TargetedPackageResolutionFailure],
    tuple[str, ...],
]:
    """Map changed non-npm ODC inputs and their local companion files."""
    if state is None:
        return (), {}, ()

    active_task_ids = {target.task_id for target in targets}
    if not active_task_ids:
        active_task_ids = set(state.get("active_target_task_ids", []) or [])
    if not active_task_ids:
        return (), {}, ()
    changed_paths_by_task, path_errors = _worker_package_lock_paths(
        state,
        active_task_ids,
        package_lock_only=False,
    )
    failures: dict[str, _TargetedPackageResolutionFailure] = {}

    def mark_failure(task_ids: Sequence[str] | set[str], detail: str) -> None:
        for task_id in set(task_ids) or active_task_ids:
            previous = failures.get(task_id)
            if previous is None:
                failures[task_id] = _TargetedPackageResolutionFailure(
                    ScanFallbackReason.TARGETED_SCAN_FAILED,
                    detail,
                )
            elif detail not in previous.detail:
                failures[task_id] = _TargetedPackageResolutionFailure(
                    previous.reason,
                    f"{previous.detail}; {detail}",
                )

    for detail in path_errors:
        mark_failure(active_task_ids, detail)

    known_package_lock_paths: set[str] = set()
    accepted_inventory = state.get("accepted_package_inventory_by_lockfile", {}) or {}
    if isinstance(accepted_inventory, Mapping):
        for occurrence_key in accepted_inventory:
            parsed_key = (
                _parse_package_occurrence_key(occurrence_key)
                if isinstance(occurrence_key, str)
                else None
            )
            if parsed_key is not None:
                known_package_lock_paths.add(parsed_key[0])
    for target in targets:
        for raw_path in target.manifest_paths:
            try:
                path = _validate_qa_path(raw_path)
            except ValueError:
                continue
            if Path(path).name.casefold() == "package-lock.json":
                known_package_lock_paths.add(path)

    projects_by_source: dict[str, tuple[tuple[str, ...], set[str]]] = {}
    staged_input_paths: set[str] = set()
    for source_input_path, task_ids in sorted(changed_paths_by_task.items()):
        basename = Path(source_input_path).name.casefold()
        if basename == "package-lock.json":
            continue
        if basename in {"yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json"}:
            mark_failure(
                task_ids,
                f"Changed package-manager input {source_input_path} is not supported by targeted closure scanning.",
            )
            continue
        if basename == "package.json":
            companion_lockfile = (Path(source_input_path).parent / "package-lock.json").as_posix()
            if companion_lockfile in known_package_lock_paths:
                continue
            mark_failure(
                task_ids,
                f"Changed npm manifest {source_input_path} has no inventoried package-lock.json.",
            )
            continue

        companion_spec = _non_npm_input_companions(source_input_path)
        if companion_spec is None:
            if Path(basename).suffix == ".lock":
                mark_failure(
                    task_ids,
                    f"Changed lockfile {source_input_path} cannot be mapped to a supported ODC input.",
                )
            continue
        required_companions, optional_companions = companion_spec
        if not _workspace_file_exists(sandbox, source_input_path):
            mark_failure(
                task_ids,
                f"Changed ODC input {source_input_path} is not readable in the workspace.",
            )
            continue

        staged_paths = {source_input_path}
        missing_required = False
        parent = Path(source_input_path).parent
        for alternatives in required_companions:
            candidates = [(parent / name).as_posix() for name in alternatives]
            companion = next(
                (path for path in candidates if _workspace_file_exists(sandbox, path)),
                None,
            )
            if companion is None:
                mark_failure(
                    task_ids,
                    (
                        f"Changed ODC input {source_input_path} is missing companion "
                        f"manifest/lockfile: {' or '.join(candidates)}."
                    ),
                )
                missing_required = True
                break
            staged_paths.add(companion)
        if missing_required:
            continue

        for name in optional_companions:
            companion = (parent / name).as_posix()
            if _workspace_file_exists(sandbox, companion):
                staged_paths.add(companion)

        existing = projects_by_source.get(source_input_path)
        if existing is not None:
            staged_paths.update(existing[0])
            merged_tasks = set(existing[1]) | set(task_ids)
            projects_by_source[source_input_path] = (tuple(sorted(staged_paths)), merged_tasks)
        else:
            projects_by_source[source_input_path] = (tuple(sorted(staged_paths)), set(task_ids))
        staged_input_paths.update(staged_paths)

    projects = tuple(
        _NonNpmInputProject(
            source_input_path=source_input_path,
            staged_input_paths=paths,
            task_ids=tuple(sorted(task_ids)),
        )
        for source_input_path, (paths, task_ids) in sorted(projects_by_source.items())
    )
    return projects, failures, tuple(sorted(staged_input_paths))


def _changed_package_lock_inputs(
    state: OrchestratorState | None,
    targets: Sequence[QAScanTarget],
    lockfile_loader: Callable[
        [str],
        tuple[Mapping[str, Any] | None, int | None, ScanFallbackReason | None, str | None],
    ],
) -> _PackageLockChangeBatch:
    """Compare every relevant live package-lock inventory with accepted state."""
    if state is None:
        return _PackageLockChangeBatch({}, (), {}, (), (), {})

    active_task_ids = {target.task_id for target in targets}
    if not active_task_ids:
        active_task_ids = set(state.get("active_target_task_ids", []) or [])
    target_task_ids_by_path: dict[str, set[str]] = {}
    target_paths: set[str] = set()
    failures_by_task: dict[str, _TargetedPackageResolutionFailure] = {}

    def mark_failure(
        task_ids: Sequence[str] | set[str],
        reason: ScanFallbackReason,
        detail: str,
    ) -> None:
        """Retain one failure per affected active task without losing diagnostics."""
        affected = set(task_ids) or active_task_ids
        for task_id in affected:
            previous = failures_by_task.get(task_id)
            if previous is None:
                failures_by_task[task_id] = _TargetedPackageResolutionFailure(reason, detail)
            elif detail not in previous.detail:
                failures_by_task[task_id] = _TargetedPackageResolutionFailure(
                    previous.reason,
                    f"{previous.detail}; {detail}",
                )

    for target in targets:
        for raw_path in target.manifest_paths:
            try:
                path = _validate_qa_path(raw_path)
            except ValueError:
                continue
            target_paths.add(path)
            target_task_ids_by_path.setdefault(path, set()).add(target.task_id)

    worker_paths_by_task, worker_path_errors = _worker_package_lock_paths(
        state,
        active_task_ids,
    )
    for detail in worker_path_errors:
        mark_failure(active_task_ids, ScanFallbackReason.TARGETED_SCAN_FAILED, detail)

    accepted_inventory = state.get("accepted_package_inventory_by_lockfile", {}) or {}
    accepted_by_path: dict[str, dict[str, str]] = {}
    if not isinstance(accepted_inventory, Mapping):
        mark_failure(
            active_task_ids,
            ScanFallbackReason.INVALID_LOCKFILE,
            "Accepted package inventory is not a mapping.",
        )
    else:
        for raw_key, fingerprint in accepted_inventory.items():
            parsed_key = (
                _parse_package_occurrence_key(raw_key) if isinstance(raw_key, str) else None
            )
            if parsed_key is None or not isinstance(fingerprint, str):
                mark_failure(
                    active_task_ids,
                    ScanFallbackReason.INVALID_LOCKFILE,
                    f"Accepted package inventory key cannot be mapped: {raw_key!r}.",
                )
                continue
            source_input_path = parsed_key[0]
            accepted_by_path.setdefault(source_input_path, {})[raw_key] = fingerprint

    paths_to_read = set(accepted_by_path) | target_paths | set(worker_paths_by_task)

    def owners_for_path(source_input_path: str) -> set[str]:
        return (
            worker_paths_by_task.get(source_input_path)
            or target_task_ids_by_path.get(source_input_path)
            or active_task_ids
        )

    current_lockfiles: dict[str, tuple[Mapping[str, Any], int]] = {}
    inventory_updates: dict[str, str] = {}
    inventory_removals: set[str] = set()
    changed_keys_by_path: dict[str, list[str]] = {}
    covered_input_paths: set[str] = set()
    for source_input_path in sorted(paths_to_read):
        packages, lockfile_version, reason, detail = lockfile_loader(source_input_path)
        previous_inventory = accepted_by_path.get(source_input_path, {})
        if reason is not None or packages is None or lockfile_version is None:
            if (
                reason == ScanFallbackReason.MISSING_LOCKFILE
                and source_input_path in worker_paths_by_task
                and source_input_path not in target_paths
            ):
                inventory_removals.update(previous_inventory)
                covered_input_paths.add(source_input_path)
                continue
            mark_failure(
                owners_for_path(source_input_path),
                reason or ScanFallbackReason.INVALID_LOCKFILE,
                detail or f"Package-lock input {source_input_path} could not be read.",
            )
            continue

        try:
            current_inventory = _package_lock_inventory(
                source_input_path,
                {"packages": packages},
            )
        except ValueError as exc:
            mark_failure(
                owners_for_path(source_input_path),
                ScanFallbackReason.INVALID_LOCKFILE,
                f"Package-lock input {source_input_path} cannot be inventoried: {exc}",
            )
            continue

        current_lockfiles[source_input_path] = (packages, lockfile_version)
        changed_keys = sorted(
            key
            for key, fingerprint in current_inventory.items()
            if previous_inventory.get(key) != fingerprint
        )
        inventory_updates.update({key: current_inventory[key] for key in changed_keys})
        inventory_removals.update(set(previous_inventory) - set(current_inventory))
        if changed_keys:
            changed_keys_by_path[source_input_path] = changed_keys
        elif set(previous_inventory) - set(current_inventory):
            covered_input_paths.add(source_input_path)

    closures_by_task: dict[str, list[DependencyClosure]] = {}
    changed_resolutions: dict[tuple[str, str], TargetPackageResolution] = {}
    for source_input_path, changed_keys in sorted(changed_keys_by_path.items()):
        packages, lockfile_version = current_lockfiles[source_input_path]
        if lockfile_version not in {2, 3}:
            mark_failure(
                owners_for_path(source_input_path),
                ScanFallbackReason.INVALID_LOCKFILE,
                f"Changed package-lock input {source_input_path} uses unsupported version "
                f"{lockfile_version}.",
            )
            continue

        path_complete = True
        for occurrence_key in changed_keys:
            parsed_key = _parse_package_occurrence_key(occurrence_key)
            if parsed_key is None:
                path_complete = False
                mark_failure(
                    owners_for_path(source_input_path),
                    ScanFallbackReason.INVALID_LOCKFILE,
                    f"Changed package occurrence cannot be mapped: {occurrence_key!r}.",
                )
                continue
            _source_path, lockfile_key, _key_name, version, _occurrence_index = parsed_key
            metadata = packages.get(lockfile_key)
            if not isinstance(metadata, Mapping):
                path_complete = False
                mark_failure(
                    owners_for_path(source_input_path),
                    ScanFallbackReason.INVALID_LOCKFILE,
                    f"Changed package occurrence {occurrence_key!r} is absent from its lockfile.",
                )
                continue

            raw_name = metadata.get("name")
            package_name = (
                str(raw_name)
                if lockfile_key == "" and isinstance(raw_name, str) and raw_name
                else lockfile_key.rsplit("node_modules/", 1)[-1]
            )
            if not package_name or not version:
                path_complete = False
                mark_failure(
                    owners_for_path(source_input_path),
                    ScanFallbackReason.INVALID_LOCKFILE,
                    f"Changed package occurrence {occurrence_key!r} has no package identity.",
                )
                continue

            resolutions, resolution_error = resolve_target_packages(
                packages,
                source_lockfile=source_input_path,
                target_package=package_name,
                target_version=version,
                lockfile_version=lockfile_version,
            )
            resolution = next(
                (
                    item
                    for item in resolutions
                    if item.target_node is not None
                    and item.target_node.lockfile_key == lockfile_key
                ),
                None,
            )
            if resolution_error or resolution is None:
                path_complete = False
                mark_failure(
                    owners_for_path(source_input_path),
                    _target_resolution_fallback_reason(resolution_error),
                    (
                        f"Changed package occurrence {occurrence_key!r} could not be "
                        f"resolved exactly: {resolution_error or 'missing physical key'}."
                    ),
                )
                continue

            closure = resolve_dependency_closure_for_target(packages, resolution)
            if not closure.complete:
                path_complete = False
                mark_failure(
                    owners_for_path(source_input_path),
                    _target_resolution_fallback_reason(closure.fallback_reason),
                    (
                        f"Changed package occurrence {occurrence_key!r} has an incomplete "
                        f"dependency closure: {closure.fallback_reason or 'unknown reason'}."
                    ),
                )
                continue

            changed_resolutions[(source_input_path, lockfile_key)] = resolution
            for task_id in owners_for_path(source_input_path):
                closures_by_task.setdefault(task_id, []).append(closure)

        if path_complete:
            covered_input_paths.add(source_input_path)

    return _PackageLockChangeBatch(
        closures_by_task={
            task_id: tuple(closures) for task_id, closures in closures_by_task.items()
        },
        target_packages=tuple(changed_resolutions[key] for key in sorted(changed_resolutions)),
        inventory_updates=inventory_updates,
        inventory_removals=tuple(sorted(inventory_removals)),
        covered_input_paths=tuple(sorted(covered_input_paths)),
        failures_by_task=failures_by_task,
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
    """Return normalized package.json paths associated with an SCA group."""
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


def _collect_dependency_package_state(
    sandbox: DockerSandbox,
    group: VulnerabilityGroup,
    task: RemediationTask | None,
    manifests: Sequence[str],
    expected_version: str | None,
    *,
    lockfiles: Sequence[str] | None = None,
    version_evidence_inconclusive: bool = False,
    inspect_resolved_graph: bool = True,
    npm_ls_disabled: bool = False,
) -> _QAPackageState:
    """Collect compact deterministic dependency evidence for a version task."""
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
    if not inspect_resolved_graph and npm_ls_disabled:
        diagnostics.append("npm ls is temporarily disabled; installed graph was not inspected.")
    elif not inspect_resolved_graph:
        graph_errors.append("npm ls skipped because npm install failed; resolved graph is unknown.")

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

        if not inspect_resolved_graph:
            continue
        cwd_path = Path(manifest).parent
        cwd = "" if str(cwd_path) == "." else cwd_path.as_posix().strip("/")
        prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        command = f"{prefix}npm ls {shlex.quote(package)} --all --json"
        try:
            command_result = sandbox.run(command, timeout=_NPM_INSTALL_TIMEOUT_SECONDS)
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
        if not all_manifests_parsed or graph_errors or not inspect_resolved_graph
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
        and inspect_resolved_graph
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
    inspect_resolved_graph: bool = True,
    npm_ls_disabled: bool = False,
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
        if managers and managers != {"npm"}:
            manager_text = ", ".join(sorted(managers))
            diagnostic = f"Unsupported or unknown package manager(s): {manager_text}."
            evidence = QADependencyEvidence(
                status=DependencyEvidenceStatus.INCONCLUSIVE,
                target_package=package,
                expected_version=_normalise_dependency_version(expected_version),
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
            inspect_resolved_graph=inspect_resolved_graph,
            npm_ls_disabled=npm_ls_disabled,
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
    if not inspect_resolved_graph and npm_ls_disabled:
        graph_diagnostics.append(
            "npm ls is temporarily disabled; installed graph was not inspected."
        )
    elif not inspect_resolved_graph:
        graph_diagnostics.append(
            "npm ls skipped because npm install failed; resolved graph is unknown."
        )
    for manifest, _payload in parsed_manifests if inspect_resolved_graph else ():
        cwd_path = Path(manifest).parent
        cwd = "" if str(cwd_path) == "." else cwd_path.as_posix().strip("/")
        prefix = f"cd {shlex.quote(cwd)} && " if cwd else ""
        command = f"{prefix}npm ls {shlex.quote(package)} --all --json"
        try:
            command_result = sandbox.run(command, timeout=_NPM_INSTALL_TIMEOUT_SECONDS)
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


def _target_resolution_fallback_reason(reason: str | None) -> ScanFallbackReason:
    """Map target-selection diagnostics to the typed QA fallback vocabulary."""
    return {
        "no_matching_target": ScanFallbackReason.NO_MATCHING_TARGET,
        "multiple_targets": ScanFallbackReason.MULTIPLE_TARGETS,
        "invalid_lockfile": ScanFallbackReason.INVALID_LOCKFILE,
    }.get(reason or "", ScanFallbackReason.TARGETED_SCAN_FAILED)


def _resolve_targeted_packages(
    sandbox: DockerSandbox,
    targets: Sequence[QAScanTarget],
    state: OrchestratorState | None = None,
) -> _TargetedPackageBatch:
    """Resolve active targets and every changed package-lock occurrence.

    Lockfiles are read and parsed once per path. Target closures and inventory
    delta closures are retained by task; unreadable or unrepresentable changed
    inputs remain task-scoped failures.
    """
    active_task_ids = {target.task_id for target in targets} or set(
        (state or {}).get("active_target_task_ids", []) or []
    )
    if not active_task_ids:
        return _TargetedPackageBatch(
            target_packages=(),
            closures_by_task={},
            covered_task_ids=(),
            failures_by_task={},
            inventory_updates={},
            inventory_removals=(),
            covered_input_paths=(),
            fallback_reason=ScanFallbackReason.MISSING_LOCKFILE,
            fallback_detail="No active task scan targets were supplied.",
            non_npm_input_projects=(),
        )

    lockfile_cache: dict[
        str,
        tuple[Mapping[str, Any] | None, int | None, ScanFallbackReason | None, str | None],
    ] = {}

    def load_lockfile(
        source_lockfile: str,
    ) -> tuple[Mapping[str, Any] | None, int | None, ScanFallbackReason | None, str | None]:
        """Read and validate one npm package-lock file, caching both success and failure."""
        if source_lockfile in lockfile_cache:
            return lockfile_cache[source_lockfile]
        try:
            raw_lockfile = sandbox.read_file(source_lockfile)
            if raw_lockfile is None:
                result = (
                    None,
                    None,
                    ScanFallbackReason.MISSING_LOCKFILE,
                    f"Live workspace lockfile {source_lockfile} could not be read.",
                )
            else:
                lockfile = json.loads(raw_lockfile)
                packages = lockfile.get("packages") if isinstance(lockfile, Mapping) else None
                lockfile_version = (
                    lockfile.get("lockfileVersion") if isinstance(lockfile, Mapping) else None
                )
                if not isinstance(packages, Mapping) or not isinstance(lockfile_version, int):
                    result = (
                        None,
                        None,
                        ScanFallbackReason.INVALID_LOCKFILE,
                        f"Live workspace lockfile {source_lockfile} lacks a supported packages map.",
                    )
                else:
                    result = (packages, lockfile_version, None, None)
        except (OSError, RuntimeError) as exc:
            result = (
                None,
                None,
                ScanFallbackReason.MISSING_LOCKFILE,
                f"Live workspace lockfile {source_lockfile} could not be read: {exc}",
            )
        except (TypeError, ValueError) as exc:
            result = (
                None,
                None,
                ScanFallbackReason.INVALID_LOCKFILE,
                f"Live workspace lockfile {source_lockfile} is not valid JSON: {exc}",
            )
        lockfile_cache[source_lockfile] = result
        return result

    change_batch = _changed_package_lock_inputs(state, targets, load_lockfile)

    resolved_packages: list[TargetPackageResolution] = list(change_batch.target_packages)
    resolved_identities = {
        (
            resolution.source_lockfile,
            resolution.target_node.lockfile_key,
            resolution.target_node.version,
        )
        for resolution in resolved_packages
        if resolution.target_node is not None
    }
    closures_by_task: dict[str, list[DependencyClosure]] = {
        task_id: list(closures) for task_id, closures in change_batch.closures_by_task.items()
    }
    covered_task_ids = set(closures_by_task)
    skipped_task_ids: set[str] = set()
    target_absent_input_paths: set[str] = set()
    failures_by_task = dict(change_batch.failures_by_task)

    def retain_failure(task_id: str, failure: _TargetedPackageResolutionFailure) -> None:
        previous = failures_by_task.get(task_id)
        if previous is None:
            failures_by_task[task_id] = failure
        elif failure.detail not in previous.detail:
            failures_by_task[task_id] = _TargetedPackageResolutionFailure(
                previous.reason,
                f"{previous.detail}; {failure.detail}",
            )

    non_npm_input_projects, non_npm_failures, non_npm_covered_paths = (
        _collect_non_npm_input_projects(sandbox, state, targets)
    )
    for task_id, failure in non_npm_failures.items():
        retain_failure(task_id, failure)
    for project in non_npm_input_projects:
        covered_task_ids.update(project.task_ids)
    for target in targets:
        if not target.manifest_paths:
            retain_failure(
                target.task_id,
                _TargetedPackageResolutionFailure(
                    ScanFallbackReason.MISSING_LOCKFILE,
                    f"Task {target.task_id} has no supported lockfile path.",
                ),
            )
            continue

        task_resolutions: list[TargetPackageResolution] = []
        task_closures: list[DependencyClosure] = []
        missing_target_paths: set[str] = set()
        task_failure: _TargetedPackageResolutionFailure | None = None
        for raw_source_lockfile in target.manifest_paths:
            try:
                source_lockfile = _validate_qa_path(raw_source_lockfile)
            except ValueError as exc:
                task_failure = _TargetedPackageResolutionFailure(
                    ScanFallbackReason.MISSING_LOCKFILE,
                    f"Task {target.task_id} has an invalid lockfile path: {exc}",
                )
                break
            if Path(source_lockfile).name.lower() != "package-lock.json":
                task_failure = _TargetedPackageResolutionFailure(
                    ScanFallbackReason.UNSUPPORTED_PACKAGE_MANAGER,
                    f"Task {target.task_id} uses unsupported lockfile {source_lockfile}.",
                )
                break

            packages, lockfile_version, lockfile_reason, lockfile_detail = load_lockfile(
                source_lockfile
            )
            if lockfile_reason is not None or packages is None or lockfile_version is None:
                task_failure = _TargetedPackageResolutionFailure(
                    lockfile_reason or ScanFallbackReason.INVALID_LOCKFILE,
                    lockfile_detail or f"Task {target.task_id} lockfile could not be resolved.",
                )
                break

            resolutions, resolution_error = resolve_target_packages(
                packages,
                source_lockfile=source_lockfile,
                target_package=target.target_package,
                target_version=target.expected_version,
                dependency_ancestry=target.dependency_ancestry,
                lockfile_version=lockfile_version,
            )
            if resolution_error or not resolutions:
                reason = _target_resolution_fallback_reason(resolution_error)
                if (
                    reason == ScanFallbackReason.NO_MATCHING_TARGET
                    and target.target_package.strip()
                ):
                    versionless_resolutions, versionless_error = resolve_target_packages(
                        packages,
                        source_lockfile=source_lockfile,
                        target_package=target.target_package,
                        target_version=None,
                        dependency_ancestry=target.dependency_ancestry,
                        lockfile_version=lockfile_version,
                    )
                    if versionless_error == "no_matching_target" and not versionless_resolutions:
                        missing_target_paths.add(source_lockfile)
                        target_absent_input_paths.add(source_lockfile)
                        continue
                    if versionless_resolutions:
                        versions = sorted(
                            {
                                resolution.target_node.version
                                for resolution in versionless_resolutions
                                if resolution.target_node is not None
                            }
                        )
                        reason = ScanFallbackReason.TARGETED_SCAN_FAILED
                        detail = (
                            f"Task {target.task_id} target package {target.target_package!r} "
                            f"was found at version(s) {', '.join(versions)}, but not at "
                            f"selected version {target.expected_version!r}."
                        )
                    else:
                        detail = (
                            f"Task {target.task_id} target selection failed for "
                            f"{source_lockfile}: {resolution_error or 'unknown reason'}"
                        )
                else:
                    detail = (
                        f"Task {target.task_id} target selection failed for "
                        f"{source_lockfile}: {resolution_error or 'unknown reason'}"
                    )
                task_failure = _TargetedPackageResolutionFailure(reason, detail)
                break

            lockfile_closures: list[DependencyClosure] = []
            for resolution in resolutions:
                closure = resolve_dependency_closure_for_target(packages, resolution)
                if not closure.complete:
                    task_failure = _TargetedPackageResolutionFailure(
                        _target_resolution_fallback_reason(closure.fallback_reason),
                        (
                            f"Task {target.task_id} dependency closure failed for "
                            f"{source_lockfile}:{resolution.target_node.lockfile_key if resolution.target_node else ''}: "
                            f"{closure.fallback_reason or 'incomplete_closure'}"
                        ),
                    )
                    break
                lockfile_closures.append(closure)
            if task_failure is not None:
                break
            task_resolutions.extend(resolutions)
            task_closures.extend(lockfile_closures)

        if task_failure is not None:
            retain_failure(target.task_id, task_failure)
            logger.info(
                "qa_critic: skipping target package for %s: %s",
                target.task_id,
                task_failure.detail,
            )
            continue
        if not task_resolutions:
            if target.task_id in failures_by_task or not missing_target_paths:
                retain_failure(
                    target.task_id,
                    _TargetedPackageResolutionFailure(
                        ScanFallbackReason.NO_MATCHING_TARGET,
                        f"Task {target.task_id} did not resolve to any package-lock entry.",
                    ),
                )
                continue
            covered_task_ids.add(target.task_id)
            has_other_scan_input = bool(closures_by_task.get(target.task_id)) or any(
                target.task_id in project.task_ids for project in non_npm_input_projects
            )
            if not has_other_scan_input:
                skipped_task_ids.add(target.task_id)
            logger.info(
                "qa_critic: target package %s was absent from its live lockfiles; "
                "skipping that package's targeted scan.",
                target.target_package,
            )
            continue

        covered_task_ids.add(target.task_id)
        for resolution in task_resolutions:
            node = resolution.target_node
            identity = (
                resolution.source_lockfile,
                node.lockfile_key if node is not None else "",
                node.version if node is not None else "",
            )
            if identity not in resolved_identities:
                resolved_identities.add(identity)
                resolved_packages.append(resolution)
        closures_by_task.setdefault(target.task_id, []).extend(task_closures)

    for task_id, closures in closures_by_task.items():
        if closures:
            covered_task_ids.add(task_id)
    covered_input_paths = set(change_batch.covered_input_paths)
    covered_input_paths.update(non_npm_covered_paths)
    for closures in closures_by_task.values():
        covered_input_paths.update(closure.source_lockfile for closure in closures)
    covered_input_paths.update(target_absent_input_paths)
    first_failure = next(iter(failures_by_task.values()), None)
    return _TargetedPackageBatch(
        non_npm_input_projects=non_npm_input_projects,
        target_packages=tuple(resolved_packages),
        closures_by_task={
            task_id: tuple(closures) for task_id, closures in closures_by_task.items()
        },
        covered_task_ids=tuple(sorted(covered_task_ids)),
        failures_by_task=failures_by_task,
        inventory_updates=change_batch.inventory_updates,
        inventory_removals=change_batch.inventory_removals,
        covered_input_paths=tuple(sorted(covered_input_paths)),
        fallback_reason=first_failure.reason if first_failure else None,
        fallback_detail=(
            "; ".join(failure.detail for failure in failures_by_task.values())
            if failures_by_task
            else None
        ),
        skipped_task_ids=tuple(sorted(skipped_task_ids)),
    )


def _write_targeted_artifacts(
    sandbox: DockerSandbox,
    closures_by_task: Mapping[str, Sequence[DependencyClosure]],
    non_npm_input_projects: Sequence[_NonNpmInputProject] = (),
) -> _TargetedArtifactManifest:
    """Stage each changed npm lockfile once to bound Node Audit requests."""
    targeted_subdir = ".odc-targeted"
    task_ids_by_lockfile: dict[str, set[str]] = {}
    for task_id, closures in sorted(closures_by_task.items()):
        for closure in closures:
            if not closure.complete or len(closure.root_keys) != 1:
                raise ValueError(closure.fallback_reason or "incomplete_closure")
            source_lockfile = _validate_qa_path(closure.source_lockfile)
            task_ids_by_lockfile.setdefault(source_lockfile, set()).add(task_id)

    if not task_ids_by_lockfile and not non_npm_input_projects:
        raise ValueError("no_target_packages")

    expected_package_occurrences: list[tuple[str, str]] = []
    occurrences_by_project: dict[str, tuple[PackageScanOccurrence, ...]] = {}
    task_ids_by_project: dict[str, tuple[str, ...]] = {}
    input_paths_by_project: dict[str, tuple[str, ...]] = {}
    artifact_index = 0

    for source_lockfile, task_ids in sorted(task_ids_by_lockfile.items()):
        raw_lockfile = sandbox.read_file(source_lockfile)
        if not isinstance(raw_lockfile, str) or not raw_lockfile.strip():
            raise ValueError(f"could not read changed package-lock input {source_lockfile}")
        try:
            lockfile = json.loads(raw_lockfile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid package-lock input {source_lockfile}") from exc
        if not isinstance(lockfile, Mapping):
            raise ValueError(f"invalid package-lock input {source_lockfile}")
        lockfile_version = lockfile.get("lockfileVersion")
        packages = lockfile.get("packages")
        if lockfile_version not in {2, 3} or not isinstance(packages, Mapping):
            raise ValueError(f"unsupported package-lock input {source_lockfile}")

        root_metadata = packages.get("", {})
        if not isinstance(root_metadata, Mapping):
            raise ValueError(f"invalid package-lock root in {source_lockfile}")
        package_json_path = _validate_qa_path(str(Path(source_lockfile).with_name("package.json")))
        package_json_text: str | None = None
        try:
            raw_package_json = sandbox.read_file(package_json_path)
        except (OSError, RuntimeError, ValueError):
            raw_package_json = None
        if isinstance(raw_package_json, str) and raw_package_json.strip():
            try:
                parsed_manifest = json.loads(raw_package_json)
            except (TypeError, ValueError):
                parsed_manifest = None
            if isinstance(parsed_manifest, Mapping) and "packages" not in parsed_manifest:
                package_json_text = raw_package_json
        if package_json_text is None:
            root_name = root_metadata.get("name") or lockfile.get("name")
            root_version = root_metadata.get("version") or lockfile.get("version")
            package_manifest: dict[str, Any] = {
                "name": root_name if isinstance(root_name, str) and root_name else "remedy-scan",
                "version": (
                    str(root_version) if root_version is not None and str(root_version) else "0.0.0"
                ),
                "private": True,
            }
            for field_name in (
                "dependencies",
                "devDependencies",
                "optionalDependencies",
                "peerDependencies",
            ):
                edges = root_metadata.get(field_name)
                if isinstance(edges, Mapping):
                    package_manifest[field_name] = dict(edges)
            package_json_text = json.dumps(package_manifest, indent=2, sort_keys=True) + "\n"

        project_path = f"{targeted_subdir}/{artifact_index:03d}"
        artifact_index += 1
        sandbox.write_file(
            _validate_qa_path(f"{project_path}/package.json"),
            package_json_text,
        )
        sandbox.write_file(
            _validate_qa_path(f"{project_path}/package-lock.json"),
            raw_lockfile,
        )

        occurrences: list[PackageScanOccurrence] = []
        for lockfile_key, metadata in sorted(packages.items(), key=lambda item: str(item[0])):
            if not isinstance(lockfile_key, str) or not isinstance(metadata, Mapping):
                raise ValueError(f"invalid package occurrence in {source_lockfile}")
            if not lockfile_key:
                continue
            raw_name = metadata.get("name")
            package_name = (
                str(raw_name)
                if isinstance(raw_name, str) and raw_name
                else lockfile_key.rsplit("node_modules/", 1)[-1]
            )
            version = str(metadata.get("version") or "").strip()
            if not package_name or not version:
                continue
            occurrences.append(
                PackageScanOccurrence(
                    source_input_path=source_lockfile,
                    package_name=package_name,
                    version=version,
                    lockfile_key=lockfile_key,
                )
            )
            # One lockfile preserves a single Node Audit request per input;
            # companion manifests let ODC identify each physical package.
            package_manifest_path = _validate_qa_path(f"{project_path}/{lockfile_key}/package.json")
            sandbox.write_file(
                package_manifest_path,
                json.dumps(
                    {"name": package_name, "version": version},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            )
        occurrences_by_project[project_path] = tuple(occurrences)
        task_ids_by_project[project_path] = tuple(sorted(task_ids))
        input_paths_by_project[project_path] = (source_lockfile,)
        expected_package_occurrences.extend(
            (occurrence.package_name, occurrence.version) for occurrence in occurrences
        )

    for project in sorted(non_npm_input_projects, key=lambda item: item.source_input_path):
        project_path = f"{targeted_subdir}/{artifact_index:03d}"
        artifact_index += 1
        copied_names: set[str] = set()
        for source_path in project.staged_input_paths:
            validated_source = _validate_qa_path(source_path)
            destination_name = Path(validated_source).name
            if not destination_name or destination_name in copied_names:
                raise ValueError(f"changed input files collide in {project.source_input_path}")
            copied_names.add(destination_name)
            destination_path = _validate_qa_path(f"{project_path}/{destination_name}")
            command = (
                f"mkdir -p -- {shlex.quote(project_path)} && "
                f"cp -- {shlex.quote(validated_source)} {shlex.quote(destination_path)}"
            )
            result = sandbox.run(command, timeout=30)
            if result.exit_code != 0:
                raise ValueError(f"could not stage changed ODC input {source_path}")

        occurrences_by_project[project_path] = ()
        task_ids_by_project[project_path] = project.task_ids
        input_paths_by_project[project_path] = (
            project.source_input_path,
            *(path for path in project.staged_input_paths if path != project.source_input_path),
        )

    return _TargetedArtifactManifest(
        scan_subdir=targeted_subdir,
        expected_package_occurrences=tuple(
            sorted(
                expected_package_occurrences,
                key=lambda item: (item[0].casefold(), item[1], item[0]),
            )
        ),
        occurrences_by_project=occurrences_by_project,
        task_ids_by_project=task_ids_by_project,
        input_paths_by_project=input_paths_by_project,
    )


def _targeted_project_for_report_path(file_path: str | None) -> str | None:
    """Return a targeted project directory from one ODC dependency path."""
    if not isinstance(file_path, str) or not file_path.strip():
        return None
    normalized = _normalise_odc_report_path(file_path)
    if normalized is None:
        return None
    parts = normalized.split("/")
    if len(parts) < 3 or parts[0] != ".odc-targeted":
        return None
    return "/".join(parts[:2])


def _issue_identifiers(issues: Sequence[Any]) -> list[str]:
    """Return both CVE and GHSA identifiers from occurrence-attributed issues."""
    identifiers: set[str] = set()
    for issue in issues:
        for field_name in ("cve_id", "ghsa_id"):
            value = _scan_result_value(issue, field_name, None)
            if isinstance(value, str) and value.strip():
                identifiers.add(value.strip().upper())
    return sorted(identifiers)


def _attribute_reported_occurrences(
    artifact_manifest: _TargetedArtifactManifest,
    scan_result: Any,
) -> tuple[_TargetedArtifactManifest, bool]:
    """Map report package occurrences and findings onto original occurrences."""
    raw_reported = _scan_result_value(scan_result, "reported_package_occurrences", None)
    if (
        raw_reported is None
        or isinstance(raw_reported, (str, bytes))
        or not isinstance(raw_reported, Sequence)
    ):
        return artifact_manifest, False

    mapping_complete = True
    reports_by_project: dict[str, list[_ODCReportedPackageOccurrence]] = {}
    for reported in raw_reported:
        project_path = _targeted_project_for_report_path(
            _scan_result_value(reported, "file_path", None)
        )
        if project_path is None or project_path not in artifact_manifest.input_paths_by_project:
            mapping_complete = False
            continue
        reports_by_project.setdefault(project_path, []).append(reported)

    issues = _scan_result_value(scan_result, "found_issues", []) or []
    issues_by_identity: dict[tuple[str, str, str], list[Any]] = {}
    issues_by_report_path: dict[tuple[str, str, str, str], list[Any]] = {}
    issues_by_project: dict[str, list[Any]] = {}
    unmapped_issues: list[Any] = []
    for issue in issues:
        package_name = _scan_result_value(issue, "package_name", None)
        package_version = _scan_result_value(issue, "package_version", None)
        file_path = _scan_result_value(issue, "file_path", None)
        project_path = _targeted_project_for_report_path(file_path)
        if not isinstance(package_name, str) or not package_name.strip() or project_path is None:
            mapping_complete = False
            unmapped_issues.append(issue)
            continue
        identity = (
            project_path,
            package_name.casefold(),
            str(package_version or ""),
        )
        normalized_path = _normalise_odc_report_path(file_path)
        issues_by_identity.setdefault(identity, []).append(issue)
        issues_by_project.setdefault(project_path, []).append(issue)
        if normalized_path is not None:
            issues_by_report_path.setdefault(
                (*identity, normalized_path),
                [],
            ).append(issue)

    consumed_issue_ids: set[int] = set()

    def matching_issues(
        project_path: str,
        package_name: str,
        version: str,
        report_path: str | None,
        *,
        identity_count: int,
    ) -> list[Any]:
        """Attach merged ODC findings to every matching physical occurrence."""
        nonlocal mapping_complete
        identity = (project_path, package_name.casefold(), version)
        identity_candidates = list(issues_by_identity.get(identity, []))
        if not identity_candidates:
            return []
        if identity_count > 1:
            if not report_path:
                mapping_complete = False
            consumed_issue_ids.update(id(issue) for issue in identity_candidates)
            return identity_candidates

        candidates = issues_by_report_path.get((*identity, report_path), []) if report_path else []
        available = [issue for issue in candidates if id(issue) not in consumed_issue_ids]
        if available:
            consumed_issue_ids.update(id(issue) for issue in available)
            return available
        available = [issue for issue in identity_candidates if id(issue) not in consumed_issue_ids]
        consumed_issue_ids.update(id(issue) for issue in available)
        return available

    occurrences_by_project: dict[str, tuple[PackageScanOccurrence, ...]] = {}
    for project_path, input_paths in artifact_manifest.input_paths_by_project.items():
        templates = artifact_manifest.occurrences_by_project.get(project_path, ())
        project_reports = reports_by_project.get(project_path, [])
        attributed: list[PackageScanOccurrence] = []
        if templates:
            expected_by_identity: dict[tuple[str, str], list[PackageScanOccurrence]] = {}
            for occurrence in templates:
                expected_by_identity.setdefault(
                    (occurrence.package_name.casefold(), occurrence.version),
                    [],
                ).append(occurrence)
            reported_by_identity: dict[
                tuple[str, str],
                list[_ODCReportedPackageOccurrence],
            ] = {}
            for reported in project_reports:
                ecosystem = str(_scan_result_value(reported, "ecosystem", "")).casefold()
                if ecosystem != "npm":
                    mapping_complete = False
                    continue
                identity = (
                    str(_scan_result_value(reported, "package_name", "")).casefold(),
                    str(_scan_result_value(reported, "version", "")),
                )
                reported_by_identity.setdefault(identity, []).append(reported)
            expected_identities = set(expected_by_identity)
            reported_identities = set(reported_by_identity)
            if expected_identities != reported_identities:
                mapping_complete = False
            for identity in sorted(expected_by_identity):
                originals = sorted(
                    expected_by_identity[identity],
                    key=lambda occurrence: occurrence.lockfile_key or "",
                )
                reports_for_identity = sorted(
                    reported_by_identity.get(identity, []),
                    key=lambda reported: str(_scan_result_value(reported, "file_path", "") or ""),
                )
                representative_report = reports_for_identity[0] if reports_for_identity else None
                identity_count = len(originals)
                for occurrence in originals:
                    if representative_report is None:
                        mapping_complete = False
                        occurrence_issues = matching_issues(
                            project_path,
                            occurrence.package_name,
                            occurrence.version,
                            None,
                            identity_count=identity_count,
                        )
                    else:
                        occurrence_issues = matching_issues(
                            project_path,
                            occurrence.package_name,
                            occurrence.version,
                            _scan_result_value(representative_report, "file_path", None),
                            identity_count=identity_count,
                        )
                    attributed.append(
                        occurrence.model_copy(
                            update={
                                "found_identifiers": _issue_identifiers(occurrence_issues),
                                "found_issues": occurrence_issues,
                            }
                        )
                    )
        else:
            if not input_paths:
                if project_reports:
                    mapping_complete = False
                occurrences_by_project[project_path] = ()
                continue
            source_input_path = input_paths[0]
            sorted_reports = sorted(
                project_reports,
                key=lambda reported: (
                    str(_scan_result_value(reported, "file_path", "") or ""),
                    str(_scan_result_value(reported, "package_name", "")).casefold(),
                    str(_scan_result_value(reported, "version", "")),
                ),
            )
            occurrence_indexes: dict[tuple[str, str], int] = {}
            report_counts = Counter(
                (
                    str(_scan_result_value(reported, "package_name", "")).casefold(),
                    str(_scan_result_value(reported, "version", "")),
                )
                for reported in sorted_reports
            )
            for reported in sorted_reports:
                package_name = str(_scan_result_value(reported, "package_name", "") or "")
                version = str(_scan_result_value(reported, "version", "") or "")
                if not package_name or not version:
                    mapping_complete = False
                    continue
                identity = (package_name.casefold(), version)
                occurrence_index = occurrence_indexes.get(identity, 0)
                occurrence_indexes[identity] = occurrence_index + 1
                occurrence_issues = matching_issues(
                    project_path,
                    package_name,
                    version,
                    _scan_result_value(reported, "file_path", None),
                    identity_count=report_counts[identity],
                )
                attributed.append(
                    PackageScanOccurrence(
                        source_input_path=source_input_path,
                        package_name=package_name,
                        version=version,
                        lockfile_key=None,
                        occurrence_index=occurrence_index,
                        found_identifiers=_issue_identifiers(occurrence_issues),
                        found_issues=occurrence_issues,
                    )
                )

        occurrences_by_project[project_path] = tuple(attributed)

    for project_path, input_paths in artifact_manifest.input_paths_by_project.items():
        if artifact_manifest.occurrences_by_project.get(project_path) or not input_paths:
            continue
        fallback_issues: dict[tuple[str, str], list[Any]] = {}
        for issue in issues_by_project.get(project_path, []):
            if id(issue) in consumed_issue_ids:
                continue
            package_name = _scan_result_value(issue, "package_name", None)
            if not isinstance(package_name, str) or not package_name.strip():
                mapping_complete = False
                continue
            version = str(_scan_result_value(issue, "package_version", None) or "")
            fallback_issues.setdefault((package_name, version), []).append(issue)
        source_input_path = input_paths[0]
        existing_occurrences = list(occurrences_by_project.get(project_path, ()))
        for package_name, version in sorted(fallback_issues):
            matching_existing = [
                occurrence
                for occurrence in existing_occurrences
                if occurrence.package_name.casefold() == package_name.casefold()
                and occurrence.version == version
            ]
            occurrence_index = len(matching_existing)
            attached_issues = fallback_issues[(package_name, version)]
            consumed_issue_ids.update(id(issue) for issue in attached_issues)
            existing_occurrences.append(
                PackageScanOccurrence(
                    source_input_path=source_input_path,
                    package_name=package_name,
                    version=version,
                    lockfile_key=None,
                    occurrence_index=occurrence_index,
                    found_identifiers=_issue_identifiers(attached_issues),
                    found_issues=attached_issues,
                )
            )
        occurrences_by_project[project_path] = tuple(existing_occurrences)

    for issue in unmapped_issues:
        if id(issue) not in consumed_issue_ids:
            mapping_complete = False
    for issue in issues:
        if id(issue) in consumed_issue_ids:
            continue
        project_path = _targeted_project_for_report_path(
            _scan_result_value(issue, "file_path", None)
        )
        if (
            project_path is None
            or artifact_manifest.occurrences_by_project.get(project_path)
            or project_path not in artifact_manifest.input_paths_by_project
        ):
            mapping_complete = False
    return (
        replace(artifact_manifest, occurrences_by_project=occurrences_by_project),
        mapping_complete,
    )


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
    target_packages: Sequence[TargetPackageResolution] = (),
    covered_task_ids: Sequence[str] | None = None,
    batch: _TargetedPackageBatch | None = None,
    artifact_manifest: _TargetedArtifactManifest | None = None,
    scan_performed: bool = True,
    scan_skip_reason: str | None = None,
) -> ODCScanEvidence:
    """Build typed, attempt-local ODC evidence from a targeted scan batch."""
    covered_occurrences = (
        [
            occurrence
            for occurrences in artifact_manifest.occurrences_by_project.values()
            for occurrence in occurrences
        ]
        if artifact_manifest is not None
        else []
    )
    closure_package_names = {occurrence.package_name for occurrence in covered_occurrences}
    closure_lockfile_keys = {
        occurrence.lockfile_key
        for occurrence in covered_occurrences
        if occurrence.lockfile_key is not None
    }
    covered_input_paths = set(batch.covered_input_paths) if batch is not None else set()
    covered_input_paths.update(occurrence.source_input_path for occurrence in covered_occurrences)
    if artifact_manifest is not None:
        for project_input_paths in artifact_manifest.input_paths_by_project.values():
            covered_input_paths.update(project_input_paths)
    if artifact_manifest is None:
        closure_package_names.update(
            resolution.target_node.package_name
            for resolution in target_packages
            if resolution.target_node is not None
        )
        closure_lockfile_keys.update(
            resolution.target_node.lockfile_key
            for resolution in target_packages
            if resolution.target_node is not None
        )
    inventory_updates = dict(batch.inventory_updates) if batch is not None else {}
    inventory_removals = list(batch.inventory_removals) if batch is not None else []

    return ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=effective_scope,
        authoritative=False,
        covered_task_ids=sorted(
            set(covered_task_ids)
            if covered_task_ids is not None
            else {target.task_id for target in targets}
        ),
        closure_package_names=sorted(closure_package_names),
        closure_lockfile_keys=sorted(closure_lockfile_keys),
        covered_occurrences=covered_occurrences,
        found_identifiers=sorted(
            _scan_result_value(scan_result, "found_identifiers", set()) or set()
        ),
        new_identifiers=sorted(_scan_result_value(scan_result, "new_identifiers", set()) or set()),
        found_issues=list(_scan_result_value(scan_result, "found_issues", []) or []),
        remaining_target_identifiers=sorted(
            _scan_result_value(scan_result, "remaining_identifiers", set()) or set()
        ),
        covered_input_paths=sorted(covered_input_paths),
        removed_occurrence_keys=inventory_removals,
        inventory_updates=inventory_updates,
        inventory_removals=inventory_removals,
        scan_performed=scan_performed,
        scan_skip_reason=scan_skip_reason,
        complete=complete,
        fallback_reason=fallback_reason,
    )


def _scan_evidence_by_task(
    evidence: ODCScanEvidence | None,
    *,
    targets: Sequence[QAScanTarget],
    baseline_identifiers: set[str],
    batch: _TargetedPackageBatch | None,
    artifact_manifest: _TargetedArtifactManifest | None,
) -> dict[str, ODCScanEvidence]:
    """Project shared scanner results into task-owned occurrence evidence."""
    if evidence is None:
        return {}

    targets_by_task = {target.task_id: target for target in targets}
    task_ids = set(targets_by_task)
    if batch is not None:
        task_ids.update(batch.covered_task_ids)
        task_ids.update(batch.failures_by_task)
    projects_by_task: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    if artifact_manifest is not None:
        for project_path, project_task_ids in artifact_manifest.task_ids_by_project.items():
            for task_id in project_task_ids:
                projects_by_task.setdefault(task_id, []).append(project_path)

    normalized_baseline = {
        identifier.strip().upper()
        for identifier in baseline_identifiers
        if isinstance(identifier, str) and identifier.strip()
    }
    result: dict[str, ODCScanEvidence] = {}
    for task_id in sorted(task_ids):
        task_closures = batch.closures_by_task.get(task_id, ()) if batch is not None else ()
        task_occurrence_keys = {
            _package_occurrence_key(
                closure.source_lockfile,
                node.lockfile_key,
                node.package_name,
                node.version,
            )
            for closure in task_closures
            for node in closure.nodes
        }
        restrict_to_task_closure = bool(task_closures)
        project_paths = sorted(set(projects_by_task.get(task_id, [])))
        target_skipped = batch is not None and task_id in getattr(batch, "skipped_task_ids", ())
        occurrences_by_key: dict[str, PackageScanOccurrence] = {}
        covered_input_paths: set[str] = set()
        for project_path in project_paths:
            if artifact_manifest is None:
                continue
            covered_input_paths.update(
                artifact_manifest.input_paths_by_project.get(project_path, ())
            )
            for occurrence in artifact_manifest.occurrences_by_project.get(project_path, ()):
                occurrence_key = _package_occurrence_key(
                    occurrence.source_input_path,
                    occurrence.lockfile_key,
                    occurrence.package_name,
                    occurrence.version,
                    occurrence.occurrence_index,
                )
                if restrict_to_task_closure and occurrence_key not in task_occurrence_keys:
                    continue
                previous = occurrences_by_key.get(occurrence_key)
                if previous is None:
                    occurrences_by_key[occurrence_key] = occurrence
                    continue
                merged_issues: dict[tuple[str, str, str, str, str, str], Any] = {}
                for issue in (*previous.found_issues, *occurrence.found_issues):
                    issue_identity = (
                        str(_scan_result_value(issue, "cve_id", None) or ""),
                        str(_scan_result_value(issue, "ghsa_id", None) or ""),
                        str(_scan_result_value(issue, "package_name", None) or ""),
                        str(_scan_result_value(issue, "package_version", None) or ""),
                        str(_scan_result_value(issue, "file_path", None) or ""),
                        str(_scan_result_value(issue, "purl", None) or ""),
                    )
                    merged_issues[issue_identity] = issue
                merged_identifiers = set(previous.found_identifiers) | set(
                    occurrence.found_identifiers
                )
                occurrences_by_key[occurrence_key] = previous.model_copy(
                    update={
                        "found_identifiers": sorted(merged_identifiers),
                        "found_issues": list(merged_issues.values()),
                    }
                )
        if (
            not evidence.scan_performed
            and batch is not None
            and task_id in evidence.covered_task_ids
        ):
            covered_input_paths.update(batch.covered_input_paths)
        if target_skipped and task_id in targets_by_task:
            covered_input_paths.update(targets_by_task[task_id].manifest_paths)

        occurrences = list(occurrences_by_key.values())
        found_identifiers = {
            identifier.strip().upper()
            for occurrence in occurrences
            for identifier in occurrence.found_identifiers
            if identifier and identifier.strip()
        }
        target_ids = {
            identifier.strip().upper()
            for identifier in (
                targets_by_task[task_id].target_identifiers if task_id in targets_by_task else ()
            )
            if identifier and identifier.strip()
        }
        found_issues_by_identity: dict[tuple[str, str, str, str, str, str], Any] = {}
        for occurrence in occurrences:
            for issue in occurrence.found_issues:
                issue_identity = (
                    str(_scan_result_value(issue, "cve_id", None) or ""),
                    str(_scan_result_value(issue, "ghsa_id", None) or ""),
                    str(_scan_result_value(issue, "package_name", None) or ""),
                    str(_scan_result_value(issue, "package_version", None) or ""),
                    str(_scan_result_value(issue, "file_path", None) or ""),
                    str(_scan_result_value(issue, "purl", None) or ""),
                )
                found_issues_by_identity[issue_identity] = issue

        task_failure = batch.failures_by_task.get(task_id) if batch is not None else None
        task_complete = (
            evidence.complete
            and task_failure is None
            and task_id in evidence.covered_task_ids
            and (bool(project_paths) or target_skipped or not evidence.scan_performed)
        )
        task_inventory_updates: dict[str, str] = {}
        task_inventory_removals: list[str] = []
        if batch is not None:
            for occurrence_key, fingerprint in batch.inventory_updates.items():
                parts = _parse_package_occurrence_key(occurrence_key)
                if parts is not None and parts[0] in covered_input_paths:
                    task_inventory_updates[occurrence_key] = fingerprint
            for occurrence_key in batch.inventory_removals:
                parts = _parse_package_occurrence_key(occurrence_key)
                if parts is not None and parts[0] in covered_input_paths:
                    task_inventory_removals.append(occurrence_key)

        result[task_id] = evidence.model_copy(
            update={
                "covered_task_ids": [task_id],
                "closure_package_names": sorted(
                    {occurrence.package_name for occurrence in occurrences}
                ),
                "closure_lockfile_keys": sorted(
                    {
                        occurrence.lockfile_key
                        for occurrence in occurrences
                        if occurrence.lockfile_key is not None
                    }
                ),
                "found_identifiers": sorted(found_identifiers),
                "new_identifiers": sorted(found_identifiers - normalized_baseline),
                "found_issues": list(found_issues_by_identity.values()),
                "remaining_target_identifiers": sorted(found_identifiers & target_ids),
                "covered_input_paths": sorted(covered_input_paths),
                "covered_occurrences": occurrences,
                "removed_occurrence_keys": task_inventory_removals,
                "inventory_updates": task_inventory_updates,
                "inventory_removals": task_inventory_removals,
                "scan_performed": evidence.scan_performed and not target_skipped,
                "scan_skip_reason": "no_matching_target" if target_skipped else None,
                "complete": task_complete,
                "fallback_reason": (
                    task_failure.reason
                    if task_failure is not None
                    else evidence.fallback_reason
                    if not task_complete
                    else None
                ),
            }
        )
    return result
