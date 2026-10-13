"""
workspace_builder.py - Shared Docker workspace preparation for remediation workers.

The node creates a named volume, copies the host repository into it, and
materializes every npm package in that shared workspace. Worker nodes perform
edits and validation against the initialized volume.
"""

from __future__ import annotations

import json
import logging
import shlex
import uuid
from pathlib import Path
from typing import Any

from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.docker_client import close_docker_client
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)

_NPM_INSTALL_COMMAND = "npm install --package-lock=true"
_NPM_INSTALL_TIMEOUT_SECONDS = 900
_INSTALL_LOG_TAIL_LINES = 80
_SKIP_PACKAGE_DIR_NAMES = frozenset({".git", "node_modules"})


def _close_client(client) -> None:
    """Compatibility wrapper for the shared Docker client boundary."""
    close_docker_client(client)


def _discover_package_directories(repo_root: Path) -> list[Path]:
    """Return npm package directories in deterministic parent-first order.

    Args:
        repo_root: Host repository root whose package manifests were copied to
            the shared workspace.

    Returns:
        Relative package-directory paths. Directories below ``.git`` and
        ``node_modules`` are excluded because they are not source packages to
        initialize.

    Side effects:
        Reads package manifest paths from ``repo_root``; does not modify files.
    """
    package_directories: set[Path] = set()
    for package_json in repo_root.rglob("package.json"):
        if not package_json.is_file():
            continue
        relative_manifest = package_json.relative_to(repo_root)
        if any(part in _SKIP_PACKAGE_DIR_NAMES for part in relative_manifest.parts):
            continue
        package_directories.add(relative_manifest.parent)

    return sorted(
        package_directories,
        key=lambda path: (len(path.parts), path.as_posix()),
    )


def _collect_source_package_lock_inventory(
    repo_root: Path,
) -> tuple[dict[str, str], list[str]]:
    """Read and fingerprint all source package-lock files before npm installs.

    Args:
        repo_root: Host repository root copied into the Docker workspace.

    Returns:
        A complete occurrence inventory and diagnostics. The inventory is
        empty when any source lockfile cannot be parsed, preventing a partial
        snapshot from being treated as authoritative.
    """
    from remediation_engine.orchestration._qa_runtime import _package_lock_inventory

    lockfiles = sorted(
        (
            path
            for path in repo_root.rglob("package-lock.json")
            if path.is_file()
            and not any(
                part in _SKIP_PACKAGE_DIR_NAMES
                for part in path.relative_to(repo_root).parts
            )
        ),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    inventory: dict[str, str] = {}
    errors: list[str] = []
    for lockfile_path in lockfiles:
        relative_path = lockfile_path.relative_to(repo_root).as_posix()
        try:
            document = json.loads(lockfile_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("package-lock root must be an object")
            inventory.update(_package_lock_inventory(relative_path, document))
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{relative_path}: {exc}")
    return ({} if errors else inventory), errors



def _install_workspace_dependencies(
    sandbox: DockerSandbox,
    package_directories: list[Path],
) -> None:
    """Install all discovered npm packages inside the shared Docker volume.

    Args:
        sandbox: Running Docker sandbox mounted at ``/workspace``.
        package_directories: Relative package-directory paths in parent-first
            order, as returned by :func:`_discover_package_directories`.

    Raises:
        RuntimeError: If an npm install exits non-zero. The exception includes
            bounded stdout and stderr tails for pipeline diagnostics.

    Side effects:
        Runs ``npm install --package-lock=true`` in each package directory
        inside the temporary Docker volume. The host repository is not
        modified.
    """
    for package_directory in package_directories:
        package_label = package_directory.as_posix() or "."
        command = _NPM_INSTALL_COMMAND
        if package_directory != Path("."):
            command = f"cd {shlex.quote(package_label)} && {_NPM_INSTALL_COMMAND}"

        logger.info(
            "workspace_builder_node: installing npm dependencies in %s.",
            package_label,
        )
        result = sandbox.run(command, timeout=_NPM_INSTALL_TIMEOUT_SECONDS)
        if result.exit_code == 0:
            continue

        stdout_tail = "\n".join(result.stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        raise RuntimeError(
            f"npm install failed in {package_label} (exit {result.exit_code}).\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )


def run_workspace_builder_node(state: OrchestratorState) -> dict[str, Any]:
    """Copy the source repository and install npm dependencies in Docker.

    The source package-lock inventory is captured before any npm install can
    rewrite the workspace lockfiles. Invalid source lockfiles are reported as
    incomplete inventory evidence without preventing ordinary setup.
    """
    repo_root_str = state.get("repo_root", "")
    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "errors": [msg],
        }

    source_inventory, inventory_errors = _collect_source_package_lock_inventory(repo_root)
    inventory_diagnostics = [
        f"workspace_builder_node: source package-lock inventory failed: {error}"
        for error in inventory_errors
    ]
    if inventory_errors:
        logger.warning(
            "workspace_builder_node: source package-lock inventory is incomplete: %s",
            "; ".join(inventory_errors),
        )

    volume_name = f"agent_workspace_{uuid.uuid4().hex[:8]}"
    logger.info("workspace_builder_node: creating workspace volume %s.", volume_name)

    client = None
    try:
        client = get_docker_client()
        client.volumes.create(name=volume_name)
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: failed to create workspace volume - {exc}"
        logger.exception("workspace_builder_node: workspace volume creation failed.")
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "accepted_package_inventory_by_lockfile": source_inventory,
            "package_inventory_baseline_complete": not inventory_errors,
            "errors": [msg, *inventory_diagnostics],
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        with DockerSandbox(repo_root, workspace_volume=volume_name) as sandbox:
            logger.info(
                "workspace_builder_node: repository copied into shared volume %s.",
                volume_name,
            )
            package_directories = _discover_package_directories(repo_root)
            if package_directories:
                _install_workspace_dependencies(
                    sandbox=sandbox,
                    package_directories=package_directories,
                )
            else:
                logger.info(
                    "workspace_builder_node: no npm package manifests found; "
                    "skipping dependency installation."
                )
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: sandbox setup failed - {exc}"
        logger.exception("workspace_builder_node: repository copy or dependency setup failed.")
        return {
            "status": "workspace_build_failed",
            "workspace_volume": volume_name,
            "accepted_package_inventory_by_lockfile": source_inventory,
            "package_inventory_baseline_complete": not inventory_errors,
            "errors": [msg, *inventory_diagnostics],
        }

    return {
        "workspace_volume": volume_name,
        "status": "workspace_ready",
        "accepted_package_inventory_by_lockfile": source_inventory,
        "package_inventory_baseline_complete": not inventory_errors,
        "errors": inventory_diagnostics,
    }
