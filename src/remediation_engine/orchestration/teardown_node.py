"""
teardown_node.py - Final cleanup node for the Phase 5 AppSec Orchestrator.

This node reads changed files back out of the shared Docker named volume to
build a unified diff, then always attempts to delete the volume.
"""

from __future__ import annotations

import difflib
import logging
import time
from pathlib import Path
from typing import Any

from remediation_engine.orchestration.state import ChangedFilesProjection, OrchestratorState
from remediation_engine.orchestration.task_utils import (
    effective_group_status,
    task_group_lineage,
    terminal_outcome_issues,
)
from remediation_engine.runtime.docker_client import close_docker_client
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


_WORKSPACE_VOLUME_CLEANUP_ATTEMPTS = 3
_WORKSPACE_VOLUME_CLEANUP_RETRY_SECONDS = 0.25


def _close_client(client) -> None:
    """Compatibility wrapper for the shared Docker client boundary."""
    close_docker_client(client)


def _read_host_text(repo_root: Path, rel_path: str) -> str:
    file_path = resolve_repository_path(repo_root, rel_path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def _build_diff(rel_path: str, before_text: str, after_text: str) -> str:
    """Build a unified diff with standard missing-final-newline markers."""
    rel_path = normalize_workspace_path(rel_path, allow_workspace_prefix=False)
    before_has_no_final_newline = bool(before_text) and not before_text.endswith(("\n", "\r"))
    after_has_no_final_newline = bool(after_text) and not after_text.endswith(("\n", "\r"))

    def diff_lines(text: str) -> list[str]:
        # ``difflib`` joins a final unterminated line to the next diff record.
        # Supplying logical lines with a newline lets us add the marker in the
        # same position used by git and GNU diff.
        return [f"{line}\n" for line in text.splitlines()]

    before_lines = diff_lines(before_text)
    after_lines = diff_lines(after_text)
    rendered: list[str] = []
    for line in difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    ):
        rendered.append(line)
        if line.startswith(("---", "+++", "@@")) or not line:
            continue
        prefix = line[0]
        content = line[1:]
        missing_before = (
            before_has_no_final_newline
            and bool(before_lines)
            and prefix in {"-", " "}
            and content == before_lines[-1]
        )
        missing_after = (
            after_has_no_final_newline
            and bool(after_lines)
            and prefix in {"+", " "}
            and content == after_lines[-1]
        )
        if missing_before or missing_after:
            rendered.append("\\ No newline at end of file\n")
    return "".join(rendered)


def _docker_not_found(exc: BaseException) -> bool:
    """Return whether the Docker resource is already absent."""
    error_name = exc.__class__.__name__.lower()
    detail = str(exc).lower()
    return error_name == "notfound" or "no such container" in detail or "no such volume" in detail


def _docker_volume_in_use(exc: BaseException) -> bool:
    """Return whether Docker rejected volume removal because it is attached."""
    detail = str(exc).lower()
    return "volume is in use" in detail or "409" in detail or "conflict" in detail


def _container_label(container: Any) -> str:
    """Return a stable label for a Docker container in cleanup diagnostics."""
    return str(
        getattr(container, "name", None) or getattr(container, "id", None) or "unknown-container"
    ).lstrip("/")


def _force_remove_attached_container(container: Any) -> str | None:
    """Remove one container attached to the run-owned workspace volume.

    Args:
        container: Docker SDK container object to remove.

    Returns:
        None when the container is gone; otherwise a concise cleanup error.

    Side Effects:
        Stops/kills and force-removes the supplied Docker container.
    """
    label = _container_label(container)
    try:
        container.remove(force=True)
        return None
    except Exception as remove_error:  # noqa: BLE001
        if _docker_not_found(remove_error):
            return None

        # A daemon can report a conflict while the container is transitioning
        # out of the running state. Kill once, then retry removal.
        try:
            container.kill()
        except Exception as kill_error:  # noqa: BLE001
            if _docker_not_found(kill_error):
                return None

        try:
            container.remove(force=True)
            return None
        except Exception as retry_error:  # noqa: BLE001
            if _docker_not_found(retry_error):
                return None
            return f"container '{label}' removal failed: {retry_error}"


def _cleanup_workspace_volume(client: Any, workspace_volume: str) -> tuple[bool, list[str]]:
    """Remove attached containers before deleting a workspace volume.

    Args:
        client: Connected Docker SDK client.
        workspace_volume: Engine-owned named volume to clean up.

    Returns:
        A removed flag and cleanup errors for the final report.

    Side Effects:
        Lists all containers attached to the volume, force-removes them, and
        retries volume removal when the Docker daemon reports a 409 conflict.
    """
    errors: list[str] = []
    volume_error: BaseException | None = None

    for attempt in range(_WORKSPACE_VOLUME_CLEANUP_ATTEMPTS):
        try:
            attached = list(
                client.containers.list(
                    all=True,
                    filters={"volume": workspace_volume},
                )
                or []
            )
        except Exception as exc:  # noqa: BLE001
            attached = []
            errors.append(
                f"teardown_node: failed to list containers attached to '{workspace_volume}' - {exc}"
            )

        for container in attached:
            cleanup_error = _force_remove_attached_container(container)
            if cleanup_error:
                errors.append(f"teardown_node: {cleanup_error}")

        try:
            client.volumes.get(workspace_volume).remove(force=True)
            return True, errors
        except Exception as exc:  # noqa: BLE001
            if _docker_not_found(exc):
                return True, errors
            volume_error = exc
            if not _docker_volume_in_use(exc):
                break
            if attempt < _WORKSPACE_VOLUME_CLEANUP_ATTEMPTS - 1:
                time.sleep(_WORKSPACE_VOLUME_CLEANUP_RETRY_SECONDS)

    if volume_error is not None:
        errors.append(
            f"teardown_node: failed to remove workspace volume '{workspace_volume}' - {volume_error}"
        )
    return False, errors


def _revert_unfixable_packages_in_json(
    original_text: str,
    updated_text: str,
    unfixable_packages: set[str],
    errors: list[str] | None = None,
) -> str:
    import json

    try:
        orig_obj = json.loads(original_text)
        upd_obj = json.loads(updated_text)

        indent = 2
        for line in updated_text.splitlines():
            if line.startswith(" ") or line.startswith("\t"):
                indent = len(line) - len(line.lstrip())
                if "\t" in line:
                    indent = "\t"
                break

        dep_keys = {
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
            "overrides",
            "resolutions",
        }

        for pkg in unfixable_packages:
            for key in dep_keys:
                if key in upd_obj and isinstance(upd_obj[key], dict) and pkg in upd_obj[key]:
                    if (
                        key in orig_obj
                        and isinstance(orig_obj.get(key), dict)
                        and pkg in orig_obj[key]
                    ):
                        upd_obj[key][pkg] = orig_obj[key][pkg]
                    else:
                        del upd_obj[key][pkg]

                    if not upd_obj[key] and (key not in orig_obj or not orig_obj[key]):
                        del upd_obj[key]

            if (
                "pnpm" in upd_obj
                and isinstance(upd_obj["pnpm"], dict)
                and "overrides" in upd_obj["pnpm"]
                and pkg in upd_obj["pnpm"]["overrides"]
            ):
                if (
                    "pnpm" in orig_obj
                    and isinstance(orig_obj.get("pnpm"), dict)
                    and "overrides" in orig_obj["pnpm"]
                    and pkg in orig_obj["pnpm"]["overrides"]
                ):
                    upd_obj["pnpm"]["overrides"][pkg] = orig_obj["pnpm"]["overrides"][pkg]
                else:
                    del upd_obj["pnpm"]["overrides"][pkg]
                if not upd_obj["pnpm"]["overrides"] and (
                    "pnpm" not in orig_obj or "overrides" not in orig_obj.get("pnpm", {})
                ):
                    del upd_obj["pnpm"]["overrides"]

        return json.dumps(upd_obj, indent=indent) + "\n"
    except Exception as exc:
        if errors is not None:
            packages = ", ".join(sorted(unfixable_packages))
            errors.append(
                f"teardown_node: could not safely revert unfixable package entries "
                f"({packages}) in malformed JSON: {exc}"
            )
        # A malformed manifest cannot be edited safely by key-matching regexes;
        # preserve the host baseline instead of risking a repeated-key rewrite.
        return original_text


def run_teardown_node(state: OrchestratorState) -> dict[str, Any]:
    """Run teardown and return a reportable terminal state.

    Args:
        state: Current Phase 5 orchestration state.

    Returns:
        A state update containing the diff, cleanup diagnostics, and the
        workspace volume name when cleanup could not complete.

    Side Effects:
        Reads changed files from the Docker workspace and attempts to remove
        attached containers and the run-owned workspace volume.
    """
    try:
        return _run_teardown_node_impl(state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("teardown_node: unexpected teardown failure.")
        return {
            "status": "completed_with_errors",
            "workspace_volume": state.get("workspace_volume"),
            "changed_files": ChangedFilesProjection(),
            "diff": "",
            "errors": list(state.get("errors", []) or [])
            + [f"teardown_node: unexpected teardown failure - {exc}"],
        }


def _run_teardown_node_impl(state: OrchestratorState) -> dict[str, Any]:
    """
    LangGraph node - Teardown.

    Reads updated changed files from the workspace volume, generates a unified
    diff against the host repository, and always attempts to remove the named
    volume.
    """
    # Teardown is the final state boundary.  Reconcile terminal tasks here as
    # well as in the supervisor so a direct teardown call or a graph path that
    # reaches cleanup after a worker surrender cannot preserve a stale current
    # attempt or retry plan in the final state.
    from remediation_engine.orchestration.supervisor_node import (
        reconcile_phase5_state_before_teardown,
    )

    prior_errors = list(state.get("errors", []) or [])
    barrier_state = reconcile_phase5_state_before_teardown(state)
    state = {
        **state,
        **barrier_state,
        # The barrier returns only newly discovered errors; preserve the
        # reducer-owned errors already accumulated by worker/QA nodes.
        "errors": prior_errors + list(barrier_state.get("errors", []) or []),
    }
    repo_root_str: str = state.get("repo_root", "")
    workspace_volume: str | None = state.get("workspace_volume")
    diff_chunks: list[str] = []
    errors: list[str] = []
    changed_files: list[str] = []
    normalized_changed_files: set[str] = set()
    for raw_path in state.get("changed_files", []) or []:
        if not isinstance(raw_path, str):
            errors.append(
                f"teardown_node: blocked changed-file path {raw_path!r} - path must be a string"
            )
            continue
        try:
            safe_path = normalize_workspace_path(raw_path, allow_workspace_prefix=False)
        except WorkspacePathError as exc:
            errors.append(f"teardown_node: blocked changed-file path '{raw_path}' - {exc}")
            continue
        normalized_changed_files.add(safe_path)
    candidate_changed_files = sorted(normalized_changed_files)
    client = None
    volume_removed = not bool(workspace_volume)

    task_queue = state.get("task_queue", {})
    valid_groups = state.get("valid_groups", [])
    # A group can have a pivoted parent and one or more child tasks. Resolve
    # status through task lineage so a failed child cannot be masked by a
    # historical QA-passed parent.
    from remediation_engine.contracts.schemas import TaskStatus

    passed_files: set[str] = set()
    unfixable_files: set[str] = set()
    unfixable_packages: set[str] = set()

    for g in valid_groups:
        files = set()
        if getattr(g, "file_path", None):
            files.add(g.file_path)
        for p in getattr(g, "file_paths", []):
            files.add(p)
        for li in getattr(g, "localized_issues", []):
            if getattr(li, "manifest_file", None):
                files.add(li.manifest_file)

        group_tasks = task_group_lineage(task_queue, g.group_id)
        group_status = effective_group_status(task_queue, g.group_id)
        if group_status in {TaskStatus.QA_PASSED.value, TaskStatus.MITIGATED.value}:
            passed_files.update(files)
        elif group_status in {
            TaskStatus.UNFIXABLE.value,
            TaskStatus.INCONCLUSIVE.value,
            TaskStatus.NEEDS_RETRY.value,
            TaskStatus.PENDING.value,
            TaskStatus.OPTIMISTICALLY_FIXED.value,
            TaskStatus.PIVOTED.value,
        }:
            unfixable_files.update(files)
            for task in group_tasks:
                target_package = getattr(task, "target_package_name", None)
                if target_package:
                    unfixable_packages.add(target_package)
            if getattr(g, "vulnerable_component", None):
                unfixable_packages.add(g.vulnerable_component)

    worker_results_by_attempt = state.get("worker_results_by_attempt", {})
    for attempt_res in worker_results_by_attempt.values():
        task = task_queue.get(attempt_res.task_id)
        if task is not None and attempt_res.changed_files:
            group_status = effective_group_status(task_queue, task.parent_group_id)
            if group_status in {TaskStatus.QA_PASSED.value, TaskStatus.MITIGATED.value}:
                passed_files.update(attempt_res.changed_files)
            elif group_status in {
                TaskStatus.UNFIXABLE.value,
                TaskStatus.INCONCLUSIVE.value,
                TaskStatus.NEEDS_RETRY.value,
                TaskStatus.PENDING.value,
                TaskStatus.OPTIMISTICALLY_FIXED.value,
                TaskStatus.PIVOTED.value,
            }:
                unfixable_files.update(attempt_res.changed_files)
                target_package = getattr(task, "target_package_name", None)
                if target_package:
                    unfixable_packages.add(target_package)

    files_to_exclude = unfixable_files - passed_files

    try:
        if candidate_changed_files and workspace_volume and repo_root_str:
            repo_root = Path(repo_root_str)
            if not repo_root.is_dir():
                msg = f"teardown_node: repo_root '{repo_root_str}' is not a valid directory."
                logger.error(msg)
                errors.append(msg)
            else:
                try:
                    with DockerSandbox(
                        repo_root=None,
                        workspace_volume=workspace_volume,
                    ) as sandbox:
                        # Attempt archives are internal transaction state and
                        # must never survive into the final volume lifecycle.
                        sandbox.cleanup_workspace_snapshots()
                        for rel_path in candidate_changed_files:
                            if rel_path in files_to_exclude:
                                continue
                            try:
                                rel_path = normalize_workspace_path(
                                    rel_path,
                                    allow_workspace_prefix=False,
                                )
                                updated_text = sandbox.read_file(rel_path)
                            except WorkspacePathError as exc:
                                errors.append(
                                    f"teardown_node: blocked workspace path '{rel_path}' - {exc}"
                                )
                                continue
                            host_path = resolve_repository_path(repo_root, rel_path)
                            host_exists = host_path.is_file()
                            if updated_text is None and not host_exists:
                                continue
                            if updated_text is None:
                                updated_text = ""
                            try:
                                original_text = _read_host_text(repo_root, rel_path)
                            except WorkspacePathError as exc:
                                errors.append(
                                    f"teardown_node: blocked host path '{rel_path}' - {exc}"
                                )
                                continue

                            if rel_path.endswith(".json") and unfixable_packages:
                                updated_text = _revert_unfixable_packages_in_json(
                                    original_text,
                                    updated_text,
                                    unfixable_packages,
                                    errors,
                                )

                            diff_text = _build_diff(rel_path, original_text, updated_text)
                            if diff_text:
                                diff_chunks.append(diff_text)
                                changed_files.append(rel_path)
                except Exception as exc:  # noqa: BLE001
                    msg = f"teardown_node: failed to extract diff from workspace volume - {exc}"
                    logger.exception("teardown_node: diff extraction failed.")
                    errors.append(msg)
    finally:
        if workspace_volume:
            try:
                client = get_docker_client()
                volume_removed, cleanup_errors = _cleanup_workspace_volume(client, workspace_volume)
                errors.extend(cleanup_errors)
                if volume_removed:
                    logger.info("teardown_node: removed workspace volume %s.", workspace_volume)
                else:
                    logger.error(
                        "teardown_node: workspace volume %s remains after cleanup.",
                        workspace_volume,
                    )
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"teardown_node: failed to remove workspace volume '{workspace_volume}' - {exc}"
                )
                logger.exception("teardown_node: workspace volume cleanup failed.")
                errors.append(msg)
            finally:
                if client is not None:
                    try:
                        _close_client(client)
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"teardown_node: Docker client close failed - {exc}")

    outcome_issues = terminal_outcome_issues(
        {
            **state,
            **barrier_state,
            # Teardown clears the volume in its output, but the input value is
            # needed to detect a task queue that reached cleanup without the
            # required authoritative scan.
            "workspace_volume": workspace_volume,
        }
    )
    outcome_errors = [f"teardown_node: {reason}" for reason in outcome_issues]
    errors.extend(error for error in outcome_errors if error not in errors)
    terminal_has_errors = bool(state.get("errors")) or bool(errors) or bool(outcome_issues)
    result: dict[str, Any] = {
        **barrier_state,
        "status": "completed_with_errors" if terminal_has_errors else "completed",
        "workspace_volume": None if volume_removed else workspace_volume,
        "changed_files": ChangedFilesProjection(changed_files),
        "diff": "".join(diff_chunks),
    }
    if errors:
        result["errors"] = errors
    return result
