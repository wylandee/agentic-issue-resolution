"""
workspace_builder.py - Shared Docker workspace preparation for remediation workers.

The node creates the run-owned workspace volume, copies the host repository
into it, and initializes the selected language toolchain in-place.
"""

from __future__ import annotations

import logging
import shlex
import uuid
from pathlib import Path
from typing import Any

from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.docker_client import close_docker_client
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)

_NPM_INSTALL_COMMAND = "npm install --package-lock=true"
_INSTALL_TIMEOUT_SECONDS = 900
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


def _install_workspace_dependencies(
    sandbox: DockerSandbox,
    package_directories: list[Path],
) -> None:
    """Install all discovered npm packages inside the shared Docker volume.

    Args:
        sandbox: Running Docker sandbox mounted at ``/workspace``.
        package_directories: Relative package directories in parent-first
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
        result = sandbox.run(command, timeout=_INSTALL_TIMEOUT_SECONDS)
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
    """
    LangGraph node - Workspace Builder.

    Creates the run-owned workspace volume and, for Java, a separate Maven
    repository volume. The selected language's toolchain initializes the
    copied project without mutating the host repository.
    """
    repo_root_str: str = state.get("repo_root", "")
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "maven_cache_volume": None,
            "errors": [msg],
        }

    project_language = state.get("project_language", ProjectLanguage.NODEJS)
    language_config = LANGUAGE_CONFIGS[project_language]
    is_java = project_language == ProjectLanguage.JAVA
    volume_name = f"agent_workspace_{uuid.uuid4().hex[:8]}"
    maven_cache_volume = f"agent_maven_repository_{uuid.uuid4().hex[:8]}" if is_java else None
    logger.info("workspace_builder_node: creating workspace volume %s.", volume_name)

    client = None
    creation_step = "workspace"
    try:
        client = get_docker_client()
        client.volumes.create(name=volume_name)
        if maven_cache_volume is not None:
            creation_step = "Maven cache"
            client.volumes.create(name=maven_cache_volume)
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: failed to create {creation_step} volume - {exc}"
        logger.exception("workspace_builder_node: %s volume creation failed.", creation_step)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": volume_name if creation_step == "Maven cache" else None,
            "maven_cache_volume": None,
            "errors": [msg],
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        sandbox_options: dict[str, Any] = {"workspace_volume": volume_name}
        if is_java:
            sandbox_options.update(
                image=language_config.docker_image,
                maven_repository_volume=maven_cache_volume,
                archive_excluded_dirs=language_config.excluded_dirs,
            )
        with DockerSandbox(repo_root_str, **sandbox_options) as sandbox:
            logger.info(
                "workspace_builder_node: repository copied into shared volume %s.",
                volume_name,
            )
            if is_java:
                command = language_config.install_command
                logger.info("workspace_builder_node: resolving Maven dependencies in reactor root.")
                result = sandbox.run(command, timeout=_INSTALL_TIMEOUT_SECONDS)
                if result.exit_code != 0:
                    stdout_tail = "\n".join(result.stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
                    stderr_tail = "\n".join(result.stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
                    raise RuntimeError(
                        f"Maven dependency resolution failed in . (exit {result.exit_code}).\n"
                        f"stdout tail:\n{stdout_tail}\n"
                        f"stderr tail:\n{stderr_tail}"
                    )
            else:
                package_directories = _discover_package_directories(Path(repo_root_str))
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
            "maven_cache_volume": maven_cache_volume,
            "errors": [msg],
        }

    return {
        "workspace_volume": volume_name,
        "maven_cache_volume": maven_cache_volume,
        "status": "workspace_ready",
    }
