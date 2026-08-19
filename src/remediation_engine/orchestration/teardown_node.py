"""
teardown_node.py - Final cleanup node for the Phase 5 AppSec Orchestrator.

This node reads changed files back out of the shared Docker named volume to
build a unified diff, then always attempts to delete the volume.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from pathlib import Path
from typing import Any

from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)


_WORKSPACE_VOLUME_CLEANUP_ATTEMPTS = 3
_WORKSPACE_VOLUME_CLEANUP_RETRY_SECONDS = 0.25


def _close_client(client) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _read_host_text(repo_root: Path, rel_path: str) -> str:
    file_path = repo_root / rel_path
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def _build_diff(rel_path: str, before_text: str, after_text: str) -> str:
    return "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
        )
    )


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
    original_text: str, updated_text: str, unfixable_packages: set[str]
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
    except Exception:
        for pkg in unfixable_packages:
            pattern = r'("' + re.escape(pkg) + r'"\s*:\s*"[^"]*")'
            orig_matches = re.findall(pattern, original_text)
            upd_matches = re.findall(pattern, updated_text)

            if len(orig_matches) == 1 and len(upd_matches) == 1:
                updated_text = updated_text.replace(upd_matches[0], orig_matches[0])
            elif len(orig_matches) > 1 and len(orig_matches) == len(upd_matches):
                for orig_m, upd_m in zip(orig_matches, upd_matches, strict=True):
                    updated_text = updated_text.replace(upd_m, orig_m, 1)
        return updated_text


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
            "changed_files": sorted(set(state.get("changed_files", []))),
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
    changed_files: list[str] = sorted(set(state.get("changed_files", [])))

    diff_chunks: list[str] = []
    errors: list[str] = []
    client = None
    volume_removed = not bool(workspace_volume)

    task_queue = state.get("task_queue", {})
    valid_groups = state.get("valid_groups", [])
    # Build a lookup from parent_group_id to task status
    from remediation_engine.contracts.schemas import TaskStatus

    task_by_group: dict = {}
    for task in task_queue.values():
        task_by_group[task.parent_group_id] = task

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

        task = task_by_group.get(g.group_id)
        if task is not None:
            if task.status in {TaskStatus.QA_PASSED, TaskStatus.MITIGATED, TaskStatus.PIVOTED}:
                passed_files.update(files)
            elif task.status == TaskStatus.UNFIXABLE:
                unfixable_files.update(files)
                target_package = getattr(task, "target_package_name", None) or getattr(
                    g, "vulnerable_component", None
                )
                if target_package:
                    unfixable_packages.add(target_package)

    worker_results_by_attempt = state.get("worker_results_by_attempt", {})
    for attempt_res in worker_results_by_attempt.values():
        task = task_queue.get(attempt_res.task_id)
        if task is not None and attempt_res.changed_files:
            if task.status in {TaskStatus.QA_PASSED, TaskStatus.MITIGATED, TaskStatus.PIVOTED}:
                passed_files.update(attempt_res.changed_files)
            elif task.status == TaskStatus.UNFIXABLE:
                unfixable_files.update(attempt_res.changed_files)

    files_to_exclude = unfixable_files - passed_files

    try:
        if changed_files and workspace_volume and repo_root_str:
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
                        for rel_path in changed_files:
                            if rel_path in files_to_exclude:
                                continue
                            updated_text = sandbox.read_file(rel_path)
                            if updated_text is None:
                                continue
                            original_text = _read_host_text(repo_root, rel_path)

                            if rel_path.endswith(".json") and unfixable_packages:
                                updated_text = _revert_unfixable_packages_in_json(
                                    original_text, updated_text, unfixable_packages
                                )

                            diff_text = _build_diff(rel_path, original_text, updated_text)
                            if diff_text:
                                diff_chunks.append(diff_text)
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

    terminal_has_errors = bool(state.get("errors")) or bool(errors)
    result: dict[str, Any] = {
        **barrier_state,
        "status": "completed_with_errors" if terminal_has_errors else "completed",
        "workspace_volume": None if volume_removed else workspace_volume,
        "changed_files": changed_files,
        "diff": "".join(diff_chunks),
    }
    if errors:
        result["errors"] = errors
    return result
