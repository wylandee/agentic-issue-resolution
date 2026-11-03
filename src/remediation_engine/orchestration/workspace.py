"""Workspace creation, dependency setup, full scans, teardown, and cleanup."""

from __future__ import annotations

import ast
import configparser
import difflib
import json
import logging
import shlex
import time
import tomllib
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from remediation_engine.contracts.schemas import (
    FinalFullScanResult,
    ScannerExecutionStatus,
    model_or_dict_value,
)
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage, resolve_project_language
from remediation_engine.orchestration.state import (
    ChangedFilesProjection,
    OrchestratorState,
    effective_group_status,
    task_group_lineage,
    terminal_outcome_issues,
)
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    resolve_repository_path,
)
from remediation_engine.runtime.sandbox import DockerSandbox, close_docker_client, get_docker_client
from remediation_engine.runtime.telemetry import get_active_trajectory_recorder
from remediation_engine.tools.npm import lockfile_key_matches_package
from remediation_engine.tools.python_mutations import restore_python_package_changes
from remediation_engine.tools.qa_scan import _run_security_scan, _SecurityScanResult

logger = logging.getLogger(__name__)

_NPM_INSTALL_COMMAND = "npm install --package-lock=true"
_INSTALL_TIMEOUT_SECONDS = 900
_PYTHON_SETUP_TIMEOUT_SECONDS = 900
_PYTHON_PIP_VERSION = "26.2.1"
_INSTALL_LOG_TAIL_LINES = 80
_SKIP_PACKAGE_DIR_NAMES = frozenset(
    {".git", "node_modules", ".venv", "venv", ".remedy-pipenv", "__pycache__"}
)


class _UnsupportedPythonManagerError(RuntimeError):
    """Raised when Python dependencies require an unsupported environment manager."""


def _close_client(client) -> None:
    """Close the Docker client through the shared runtime boundary."""
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
    from remediation_engine.tools.qa_workspace import _package_lock_inventory

    lockfiles = sorted(
        (
            path
            for path in repo_root.rglob("package-lock.json")
            if path.is_file()
            and not any(
                part in _SKIP_PACKAGE_DIR_NAMES for part in path.relative_to(repo_root).parts
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


def _toml_mapping(path: Path) -> dict[str, Any]:
    """Read a TOML manifest without allowing an unbounded parse diagnostic."""
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RuntimeError(
            f"could not read Python project metadata from {path.name}: {exc}"
        ) from exc
    return value


def _python_requires_constraints(repo_root: Path, *, pipfile: bool) -> list[str]:
    """Collect static Python-version constraints used by the supported manifests."""
    constraints: list[str] = []
    pyproject_path = repo_root / "pyproject.toml"
    setup_cfg_path = repo_root / "setup.cfg"
    setup_py_path = repo_root / "setup.py"

    if pipfile:
        pipfile = _toml_mapping(repo_root / "Pipfile")
        requires = pipfile.get("requires", {})
        if isinstance(requires, dict):
            version = requires.get("python_version")
            full_version = requires.get("python_full_version")
            if version:
                constraints.append(f"=={version}.*")
            if full_version:
                constraints.append(f"=={full_version}")
        return constraints

    if pyproject_path.is_file():
        project = _toml_mapping(pyproject_path).get("project", {})
        if isinstance(project, dict) and project.get("requires-python"):
            constraints.append(str(project["requires-python"]))

    if setup_cfg_path.is_file():
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(setup_cfg_path, encoding="utf-8")
        except (configparser.Error, OSError) as exc:
            raise RuntimeError(f"could not read setup.cfg Python metadata: {exc}") from exc
        if parser.has_option("options", "python_requires"):
            constraints.append(parser.get("options", "python_requires"))

    if setup_py_path.is_file():
        try:
            module = ast.parse(setup_py_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            raise RuntimeError(f"could not inspect setup.py Python metadata: {exc}") from exc
        for node in ast.walk(module):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Name)
                and function.id == "setup"
                or isinstance(function, ast.Attribute)
                and function.attr == "setup"
            ):
                continue
            for keyword in node.keywords:
                if keyword.arg == "python_requires":
                    try:
                        constraints.append(str(ast.literal_eval(keyword.value)))
                    except (ValueError, TypeError):
                        continue
    return constraints


def _supports_python_311(constraint: str) -> bool:
    """Return whether a PEP 440 requirement admits any Python 3.11 patch release."""
    try:
        specifier = SpecifierSet(constraint)
    except InvalidSpecifier as exc:
        raise RuntimeError(f"invalid Python version requirement: {constraint[:160]}") from exc
    return any(
        specifier.contains(Version(f"3.11.{patch}"), prereleases=True) for patch in range(100)
    )


def _has_static_setup_py_metadata(path: Path) -> bool:
    """Recognize setup.py metadata without executing project code on the host."""
    try:
        module = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"could not inspect setup.py build metadata: {exc}") from exc
    metadata_keys = {"name", "version", "packages", "py_modules", "install_requires"}
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Name)
            and function.id == "setup"
            or isinstance(function, ast.Attribute)
            and function.attr == "setup"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg not in metadata_keys:
                continue
            try:
                ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                continue
            return True
    return False


def _python_project_setup_plan(repo_root: Path) -> tuple[str, ...]:
    """Validate the Python 3.11 profile and return container-only setup commands."""
    pipfile_path = repo_root / "Pipfile"
    if pipfile_path.is_file():
        constraints = _python_requires_constraints(repo_root, pipfile=True)
        commands = [
            "rm -rf .venv .remedy-pipenv",
            "python -m venv .remedy-pipenv",
            f".remedy-pipenv/bin/python -m pip install --upgrade pip=={_PYTHON_PIP_VERSION}",
            ".remedy-pipenv/bin/python -m pip install pipenv",
            "PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv --python 3.11",
            f".venv/bin/python -m pip install --upgrade pip=={_PYTHON_PIP_VERSION}",
        ]
        if not (repo_root / "Pipfile.lock").is_file():
            commands.append("PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv install --dev")
        else:
            commands.append("PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv sync --dev")
        commands.append(
            "if ! .venv/bin/python -c 'import pytest' >/dev/null 2>&1; "
            "then .venv/bin/python -m pip install pytest; fi"
        )
    else:
        pyproject_path = repo_root / "pyproject.toml"
        pyproject = _toml_mapping(pyproject_path) if pyproject_path.is_file() else {}
        project = pyproject.get("project", {})
        project = project if isinstance(project, dict) else {}
        tool = pyproject.get("tool", {})
        tool = tool if isinstance(tool, dict) else {}
        poetry = tool.get("poetry", {})
        poetry = poetry if isinstance(poetry, dict) else {}
        poetry_dependencies = poetry.get("dependencies", {})
        poetry_dev_dependencies = poetry.get("dev-dependencies", {})
        poetry_groups = poetry.get("group", {})
        poetry_dependency_tables = [
            value
            for value in (poetry_dependencies, poetry_dev_dependencies)
            if isinstance(value, dict)
        ]
        if isinstance(poetry_groups, dict):
            poetry_dependency_tables.extend(
                group.get("dependencies", {})
                for group in poetry_groups.values()
                if (isinstance(group, dict) and isinstance(group.get("dependencies", {}), dict))
            )
        poetry_only = any(
            name.casefold() != "python"
            for dependencies in poetry_dependency_tables
            for name in dependencies
        )
        uv = tool.get("uv", {})
        uv = uv if isinstance(uv, dict) else {}
        uv_dependency_groups = pyproject.get("dependency-groups", {})
        uv_only_dependencies = bool(uv_dependency_groups) or any(
            key in uv for key in ("dependencies", "dev-dependencies", "dependency-groups")
        )
        if (poetry_only and not project.get("dependencies")) or (
            uv_only_dependencies and not project.get("dependencies")
        ):
            manager = "Poetry" if poetry_only else "uv"
            raise _UnsupportedPythonManagerError(
                f"{manager}-managed Python dependencies are unsupported; "
                "use requirements.txt, static PEP 621/setup.cfg metadata, or Pipenv."
            )

        constraints = _python_requires_constraints(repo_root, pipfile=False)
        commands = [
            "rm -rf .venv .remedy-pipenv",
            "python -m venv .venv",
            f".venv/bin/python -m pip install --upgrade pip=={_PYTHON_PIP_VERSION}",
        ]
        if (repo_root / "requirements.txt").is_file():
            commands.append(".venv/bin/python -m pip install -r requirements.txt")

        config = configparser.ConfigParser(interpolation=None)
        if (repo_root / "setup.cfg").is_file():
            try:
                config.read(repo_root / "setup.cfg", encoding="utf-8")
            except (configparser.Error, OSError) as exc:
                raise RuntimeError(f"could not read setup.cfg build metadata: {exc}") from exc
        setup_cfg_metadata = config.has_section("metadata") and config.has_option(
            "metadata", "name"
        )
        setup_cfg_options = config.has_section("options") and any(
            config.has_option("options", option)
            for option in ("packages", "py_modules", "install_requires")
        )
        pyproject_metadata = bool(
            project.get("name")
            or project.get("dependencies")
            or project.get("optional-dependencies")
        )
        setup_py_metadata = (repo_root / "setup.py").is_file() and _has_static_setup_py_metadata(
            repo_root / "setup.py"
        )
        if pyproject_metadata or setup_cfg_metadata or setup_cfg_options or setup_py_metadata:
            commands.append(".venv/bin/python -m pip install -e .")
        commands.append(
            "if ! .venv/bin/python -c 'import pytest' >/dev/null 2>&1; "
            "then .venv/bin/python -m pip install pytest; fi"
        )

    for constraint in constraints:
        if not _supports_python_311(constraint):
            raise RuntimeError(
                f"project requires Python {constraint[:160]}, incompatible with Python 3.11."
            )
    return tuple(commands)


def _install_python_workspace(sandbox: DockerSandbox, commands: tuple[str, ...]) -> None:
    """Run the bounded Python environment setup exclusively inside the volume."""
    for command in commands:
        logger.info("workspace_builder_node: running Python setup command %s.", command)
        result = sandbox.run(command, timeout=_PYTHON_SETUP_TIMEOUT_SECONDS)
        if result.exit_code == 0:
            continue
        stdout_tail = "\n".join(result.stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        stderr_tail = "\n".join(result.stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
        raise RuntimeError(
            f"Python workspace setup command failed (exit {result.exit_code}): "
            f"{command[:180]}.\nstdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
        )


def run_workspace_builder_node(state: OrchestratorState) -> dict[str, Any]:
    """Copy the source repository and install dependencies for its language.

    The source package-lock inventory is captured before any npm install can
    rewrite the workspace lockfiles. Invalid source lockfiles are reported as
    incomplete inventory evidence without preventing ordinary setup. Python
    metadata is inspected statically on the host; environment setup runs only
    inside the language-specific Docker image.
    """
    repo_root_str = state.get("repo_root", "")
    repo_root = Path(repo_root_str)
    if not repo_root_str or not repo_root.is_dir():
        msg = f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "maven_cache_volume": None,
            "errors": [msg],
        }

    raw_language = state.get("project_language")
    context = state.get("system_context")
    requested_language = (
        str(raw_language)
        if raw_language is not None
        else getattr(context, "primary_language", None)
    )
    try:
        project_language = resolve_project_language(repo_root, requested_language)
    except ValueError as exc:
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "maven_cache_volume": None,
            "errors": [f"workspace_builder_node: unsupported project language: {exc}"],
        }
    language_config = LANGUAGE_CONFIGS[project_language]
    is_java = project_language == ProjectLanguage.JAVA
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

    try:
        python_setup_commands = (
            _python_project_setup_plan(repo_root)
            if project_language == ProjectLanguage.PYTHON
            else ()
        )
    except Exception as exc:  # noqa: BLE001 - unsupported managers fail before volume creation
        msg = f"workspace_builder_node: Python setup is unsupported - {exc}"
        logger.warning(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "accepted_package_inventory_by_lockfile": source_inventory,
            "package_inventory_baseline_complete": not inventory_errors,
            "errors": [msg, *inventory_diagnostics],
        }

    volume_name = f"agent_workspace_{uuid.uuid4().hex[:8]}"
    maven_cache_volume = f"agent_maven_repository_{uuid.uuid4().hex[:8]}" if is_java else None
    logger.info("workspace_builder_node: creating workspace volume %s.", volume_name)

    client = None
    workspace_created = False
    maven_cache_created = False
    creation_step = "workspace"
    try:
        client = get_docker_client()
        client.volumes.create(name=volume_name)
        workspace_created = True
        recorder = get_active_trajectory_recorder()
        if recorder is not None:
            recorder.register_workspace_volume(volume_name)
        if maven_cache_volume is not None:
            creation_step = "Maven cache"
            client.volumes.create(name=maven_cache_volume)
            maven_cache_created = True
            if recorder is not None:
                recorder.register_maven_cache_volume(maven_cache_volume)
    except Exception as exc:  # noqa: BLE001
        msg = f"workspace_builder_node: failed to create {creation_step} volume - {exc}"
        logger.exception("workspace_builder_node: %s volume creation failed.", creation_step)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": volume_name if workspace_created else None,
            "maven_cache_volume": maven_cache_volume if maven_cache_created else None,
            "accepted_package_inventory_by_lockfile": source_inventory,
            "package_inventory_baseline_complete": not inventory_errors,
            "errors": [msg, *inventory_diagnostics],
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        with DockerSandbox(
            repo_root,
            image=language_config.docker_image,
            workspace_volume=volume_name,
            maven_repository_volume=maven_cache_volume,
            archive_excluded_dirs=language_config.excluded_dirs,
        ) as sandbox:
            logger.info(
                "workspace_builder_node: repository copied into shared volume %s.",
                volume_name,
            )
            if is_java:
                result = sandbox.run(
                    language_config.install_command, timeout=_INSTALL_TIMEOUT_SECONDS
                )
                if result.exit_code != 0:
                    stdout_tail = "\n".join(result.stdout.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
                    stderr_tail = "\n".join(result.stderr.splitlines()[-_INSTALL_LOG_TAIL_LINES:])
                    raise RuntimeError(
                        "Maven package build failed in . "
                        f"(exit {result.exit_code}).\nstdout tail:\n{stdout_tail}\n"
                        f"stderr tail:\n{stderr_tail}"
                    )
            elif project_language == ProjectLanguage.PYTHON:
                _install_python_workspace(sandbox, python_setup_commands)
            else:
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
            "maven_cache_volume": maven_cache_volume,
            "accepted_package_inventory_by_lockfile": source_inventory,
            "package_inventory_baseline_complete": not inventory_errors,
            "errors": [msg, *inventory_diagnostics],
        }

    return {
        "workspace_volume": volume_name,
        "maven_cache_volume": maven_cache_volume,
        "status": "workspace_ready",
        "accepted_package_inventory_by_lockfile": source_inventory,
        "package_inventory_baseline_complete": not inventory_errors,
        "errors": inventory_diagnostics,
    }


_WORKSPACE_VOLUME_CLEANUP_ATTEMPTS = 3
_WORKSPACE_VOLUME_CLEANUP_RETRY_SECONDS = 0.25


_attempt_value = model_or_dict_value


def _read_host_text(repo_root: Path, rel_path: str) -> str:
    file_path = resolve_repository_path(repo_root, rel_path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8")


def _build_diff(
    rel_path: str,
    before_text: str,
    after_text: str,
    *,
    before_exists: bool = True,
    after_exists: bool = True,
) -> str:
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
        fromfile=f"a/{rel_path}" if before_exists else "/dev/null",
        tofile=f"b/{rel_path}" if after_exists else "/dev/null",
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


def _cleanup_workspace_volume(
    client: Any,
    workspace_volume: str,
    volume_kind: str = "workspace",
) -> tuple[bool, list[str]]:
    """Remove attached containers before deleting one run-owned volume.

    Args:
        client: Connected Docker SDK client.
        workspace_volume: Engine-owned named volume to clean up.
        volume_kind: Human-readable resource type included in diagnostics.

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
            f"teardown_node: failed to remove {volume_kind} volume '{workspace_volume}' - {volume_error}"
        )
    return False, errors


def _cleanup_run_owned_volume(
    volume_name: str,
    *,
    expected_prefix: str,
    volume_kind: str,
) -> tuple[bool, list[str]]:
    """Remove one validated run-owned volume outside the normal teardown node.

    Args:
        volume_name: Docker volume name recorded by this run.
        expected_prefix: Required engine-owned name prefix for the resource.
        volume_kind: Human-readable type used in cleanup diagnostics.

    Returns:
        Whether Docker confirmed removal and any cleanup diagnostics.

    Side Effects:
        Connects to Docker, removes containers attached to the volume, removes
        the volume, and closes the Docker client.
    """
    if not volume_name.startswith(expected_prefix):
        return False, [
            f"teardown_node: refused emergency cleanup of unexpected {volume_kind} volume "
            f"'{volume_name}'"
        ]
    try:
        client = get_docker_client()
    except Exception as exc:  # noqa: BLE001 - cleanup cannot mask the graph failure
        return False, [
            f"teardown_node: could not connect to Docker for emergency {volume_kind} cleanup "
            f"of '{volume_name}' - {exc}"
        ]

    removed = False
    errors: list[str] = []
    try:
        removed, errors = _cleanup_workspace_volume(client, volume_name, volume_kind)
    except BaseException as exc:  # noqa: BLE001 - client closure also covers interrupts
        errors.append(
            f"teardown_node: emergency {volume_kind} cleanup of '{volume_name}' failed - {exc}"
        )
    finally:
        try:
            _close_client(client)
        except BaseException as exc:  # noqa: BLE001
            errors.append(f"teardown_node: emergency Docker client close failed - {exc}")
    return removed, errors


def _cleanup_run_owned_workspace_volume(workspace_volume: str) -> tuple[bool, list[str]]:
    """Remove a validated run-owned workspace volume after an orchestration error."""
    return _cleanup_run_owned_volume(
        workspace_volume,
        expected_prefix="agent_workspace_",
        volume_kind="workspace",
    )


def _cleanup_run_owned_maven_cache_volume(volume_name: str) -> tuple[bool, list[str]]:
    """Remove a validated run-owned Maven repository volume after an error."""
    return _cleanup_run_owned_volume(
        volume_name,
        expected_prefix="agent_maven_repository_",
        volume_kind="Maven cache",
    )


def _clear_active_workspace_volume(workspace_volume: str | None) -> None:
    """Clear the current run's resource record after verified volume removal."""
    if not workspace_volume:
        return
    from remediation_engine.runtime.telemetry import (
        get_active_trajectory_recorder,
    )

    recorder = get_active_trajectory_recorder()
    if recorder is not None:
        recorder.clear_workspace_volume(workspace_volume)


def _clear_active_maven_cache_volume(volume_name: str | None) -> None:
    """Clear the trajectory's Maven cache record after confirmed removal."""
    if not volume_name:
        return
    from remediation_engine.runtime.telemetry import (
        get_active_trajectory_recorder,
    )

    recorder = get_active_trajectory_recorder()
    if recorder is not None:
        recorder.clear_maven_cache_volume(volume_name)


def _lockfile_dependencies_contain(
    dependencies: Any,
    unfixable_packages: set[str],
    parent_key: str = "",
) -> bool:
    """Return whether a v1/v2 dependency subtree contains an unresolved package."""
    if not isinstance(dependencies, dict):
        return False
    for package_name, entry in dependencies.items():
        physical_key = (
            f"{parent_key}/node_modules/{package_name}"
            if parent_key
            else f"node_modules/{package_name}"
        )
        if any(
            lockfile_key_matches_package(physical_key, unfixable)
            for unfixable in unfixable_packages
        ):
            return True
        nested = entry.get("dependencies") if isinstance(entry, dict) else None
        if _lockfile_dependencies_contain(nested, unfixable_packages, physical_key):
            return True
    return False


def _restore_npm_lockfile_dependencies(
    original_dependencies: dict[str, Any],
    updated_dependencies: dict[str, Any],
    unfixable_packages: set[str],
    parent_key: str = "",
) -> None:
    """Restore matching v1/v2 package entries recursively from host JSON."""
    for package_name in sorted(set(original_dependencies) | set(updated_dependencies)):
        physical_key = (
            f"{parent_key}/node_modules/{package_name}"
            if parent_key
            else f"node_modules/{package_name}"
        )
        original_entry = original_dependencies.get(package_name)
        updated_entry = updated_dependencies.get(package_name)
        if any(
            lockfile_key_matches_package(physical_key, unfixable)
            for unfixable in unfixable_packages
        ):
            if package_name in original_dependencies:
                updated_dependencies[package_name] = deepcopy(original_entry)
            else:
                updated_dependencies.pop(package_name, None)
            continue
        if not isinstance(updated_entry, dict):
            if isinstance(original_entry, dict) and _lockfile_dependencies_contain(
                original_entry.get("dependencies"),
                unfixable_packages,
                physical_key,
            ):
                # Keep the ancestor needed to represent the restored physical
                # nested package path when the candidate removed that ancestor.
                updated_dependencies[package_name] = deepcopy(original_entry)
            continue
        original_nested = (
            original_entry.get("dependencies") if isinstance(original_entry, dict) else None
        )
        updated_nested = updated_entry.get("dependencies")
        if isinstance(original_nested, dict) or isinstance(updated_nested, dict):
            if not isinstance(updated_nested, dict):
                updated_nested = {}
                updated_entry["dependencies"] = updated_nested
            _restore_npm_lockfile_dependencies(
                original_nested if isinstance(original_nested, dict) else {},
                updated_nested,
                unfixable_packages,
                physical_key,
            )
            if not updated_nested and not isinstance(original_nested, dict):
                updated_entry.pop("dependencies", None)


def _restore_npm_lockfile_packages(
    original_packages: dict[str, Any],
    updated_packages: dict[str, Any],
    unfixable_packages: set[str],
) -> None:
    """Restore matching v3 physical entries and nested package descendants."""
    package_keys = {str(key) for key in original_packages} | {str(key) for key in updated_packages}
    for package_name in sorted(unfixable_packages):
        roots = {key for key in package_keys if lockfile_key_matches_package(key, package_name)}
        affected = set(roots)
        root_prefixes = {key.replace("\\", "/").strip("/") + "/node_modules/" for key in roots}
        for key in package_keys:
            normalized = key.replace("\\", "/").strip("/")
            if any(normalized.startswith(prefix) for prefix in root_prefixes):
                affected.add(key)
        for key in sorted(affected):
            if key in original_packages:
                updated_packages[key] = deepcopy(original_packages[key])
            else:
                updated_packages.pop(key, None)

    original_root = original_packages.get("")
    updated_root = updated_packages.get("")
    if not isinstance(original_root, dict) or not isinstance(updated_root, dict):
        return
    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        original_entries = original_root.get(section)
        updated_entries = updated_root.get(section)
        if not isinstance(original_entries, dict) and not isinstance(updated_entries, dict):
            continue
        if not isinstance(updated_entries, dict):
            updated_entries = {}
            updated_root[section] = updated_entries
        for package_name in sorted(unfixable_packages):
            if isinstance(original_entries, dict) and package_name in original_entries:
                updated_entries[package_name] = deepcopy(original_entries[package_name])
            else:
                updated_entries.pop(package_name, None)
        if not updated_entries and not isinstance(original_entries, dict):
            updated_root.pop(section, None)


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

        dependency_sections = (
            "dependencies",
            "devDependencies",
            "peerDependencies",
            "optionalDependencies",
            "overrides",
            "resolutions",
        )
        lockfile_like = any(
            "lockfileVersion" in document or "packages" in document
            for document in (orig_obj, upd_obj)
            if isinstance(document, dict)
        )
        for package_name in sorted(unfixable_packages):
            for key in dependency_sections:
                if lockfile_like and key == "dependencies":
                    continue
                if (
                    key in upd_obj
                    and isinstance(upd_obj[key], dict)
                    and package_name in upd_obj[key]
                ):
                    if (
                        key in orig_obj
                        and isinstance(orig_obj.get(key), dict)
                        and package_name in orig_obj[key]
                    ):
                        upd_obj[key][package_name] = deepcopy(orig_obj[key][package_name])
                    else:
                        del upd_obj[key][package_name]

                    if not upd_obj[key] and (key not in orig_obj or not orig_obj[key]):
                        del upd_obj[key]

            if (
                "pnpm" in upd_obj
                and isinstance(upd_obj["pnpm"], dict)
                and "overrides" in upd_obj["pnpm"]
                and package_name in upd_obj["pnpm"]["overrides"]
            ):
                if (
                    "pnpm" in orig_obj
                    and isinstance(orig_obj.get("pnpm"), dict)
                    and "overrides" in orig_obj["pnpm"]
                    and package_name in orig_obj["pnpm"]["overrides"]
                ):
                    upd_obj["pnpm"]["overrides"][package_name] = deepcopy(
                        orig_obj["pnpm"]["overrides"][package_name]
                    )
                else:
                    del upd_obj["pnpm"]["overrides"][package_name]
                if not upd_obj["pnpm"]["overrides"] and (
                    "pnpm" not in orig_obj or "overrides" not in orig_obj.get("pnpm", {})
                ):
                    del upd_obj["pnpm"]["overrides"]

        original_packages = orig_obj.get("packages")
        updated_packages = upd_obj.get("packages")
        if "packages" in orig_obj and not isinstance(original_packages, dict):
            raise ValueError("host npm lockfile packages field is not an object")
        if "packages" in upd_obj and not isinstance(updated_packages, dict):
            raise ValueError("workspace npm lockfile packages field is not an object")
        if isinstance(updated_packages, dict):
            _restore_npm_lockfile_packages(
                original_packages if isinstance(original_packages, dict) else {},
                updated_packages,
                unfixable_packages,
            )

        lockfile_version = upd_obj.get("lockfileVersion", orig_obj.get("lockfileVersion"))
        if lockfile_version in {1, 2}:
            original_dependencies = orig_obj.get("dependencies")
            updated_dependencies = upd_obj.get("dependencies")
            if "dependencies" in orig_obj and not isinstance(original_dependencies, dict):
                raise ValueError("host npm lockfile dependencies field is not an object")
            if "dependencies" in upd_obj and not isinstance(updated_dependencies, dict):
                raise ValueError("workspace npm lockfile dependencies field is not an object")
            if isinstance(updated_dependencies, dict):
                _restore_npm_lockfile_dependencies(
                    original_dependencies if isinstance(original_dependencies, dict) else {},
                    updated_dependencies,
                    unfixable_packages,
                )

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
    workspace_volume = state.get("workspace_volume")
    maven_cache_volume = state.get("maven_cache_volume")
    try:
        result = _run_teardown_node_impl(state)
        if result.get("workspace_volume") is None:
            _clear_active_workspace_volume(workspace_volume)
        if result.get("maven_cache_volume") is None:
            _clear_active_maven_cache_volume(maven_cache_volume)
        return result
    except BaseException as exc:  # noqa: BLE001 - clean up before propagating interrupts
        workspace_volume = state.get("workspace_volume")
        maven_cache_volume = state.get("maven_cache_volume")
        cleanup_errors: list[str] = []
        volume_removed = not bool(workspace_volume)
        maven_cache_removed = not bool(maven_cache_volume)
        if isinstance(workspace_volume, str) and workspace_volume:
            volume_removed, cleanup_errors = _cleanup_run_owned_workspace_volume(workspace_volume)
            if volume_removed:
                _clear_active_workspace_volume(workspace_volume)
        if isinstance(maven_cache_volume, str) and maven_cache_volume:
            maven_cache_removed, maven_errors = _cleanup_run_owned_maven_cache_volume(
                maven_cache_volume
            )
            cleanup_errors.extend(maven_errors)
            if maven_cache_removed:
                _clear_active_maven_cache_volume(maven_cache_volume)
        if not isinstance(exc, Exception):
            raise
        logger.exception("teardown_node: unexpected teardown failure.")
        return {
            "status": "completed_with_errors",
            "workspace_volume": None if volume_removed else workspace_volume,
            "maven_cache_volume": None if maven_cache_removed else maven_cache_volume,
            "changed_files": ChangedFilesProjection(),
            "diff": "",
            "errors": [f"teardown_node: unexpected teardown failure - {exc}", *cleanup_errors],
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
    from remediation_engine.orchestration.state import (
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
    maven_cache_volume: str | None = state.get("maven_cache_volume")
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
    client = None
    volume_removed = not bool(workspace_volume)
    maven_cache_removed = not bool(maven_cache_volume)

    task_queue = state.get("task_queue", {})
    valid_groups = state.get("valid_groups", [])
    worker_results_by_attempt = state.get("worker_results_by_attempt", {}) or {}
    # ``changed_files`` is an accumulated candidate projection, while worker
    # envelopes are the authoritative per-attempt record of what an accepted
    # worker actually touched. Include both before reading the final volume;
    # otherwise manifest-only update attempts can disappear at teardown when
    # the bridge did not emit a separate changed-file projection.
    for attempt_res in worker_results_by_attempt.values():
        for raw_path in _attempt_value(attempt_res, "changed_files", []) or []:
            if not isinstance(raw_path, str):
                errors.append(
                    "teardown_node: blocked worker changed-file path "
                    f"{raw_path!r} - path must be a string"
                )
                continue
            try:
                normalized_changed_files.add(
                    normalize_workspace_path(raw_path, allow_workspace_prefix=False)
                )
            except WorkspacePathError as exc:
                errors.append(
                    f"teardown_node: blocked worker changed-file path '{raw_path}' - {exc}"
                )
    candidate_changed_files = sorted(normalized_changed_files)

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
        if group_status == TaskStatus.QA_PASSED.value:
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

    for attempt_res in worker_results_by_attempt.values():
        task = task_queue.get(_attempt_value(attempt_res, "task_id"))
        attempt_changed_files = _attempt_value(attempt_res, "changed_files", []) or []
        if task is not None and attempt_changed_files:
            group_status = effective_group_status(task_queue, task.parent_group_id)
            if group_status == TaskStatus.QA_PASSED.value:
                passed_files.update(attempt_changed_files)
            elif group_status in {
                TaskStatus.UNFIXABLE.value,
                TaskStatus.INCONCLUSIVE.value,
                TaskStatus.NEEDS_RETRY.value,
                TaskStatus.PENDING.value,
                TaskStatus.OPTIMISTICALLY_FIXED.value,
                TaskStatus.PIVOTED.value,
            }:
                unfixable_files.update(attempt_changed_files)
                target_package = getattr(task, "target_package_name", None)
                if target_package:
                    unfixable_packages.add(target_package)
    files_to_exclude = unfixable_files - passed_files

    raw_language = state.get("project_language", ProjectLanguage.NODEJS)
    project_language = (
        raw_language
        if isinstance(raw_language, ProjectLanguage)
        else ProjectLanguage(raw_language or ProjectLanguage.NODEJS.value)
    )
    sandbox_image = LANGUAGE_CONFIGS[project_language].docker_image

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
                        image=sandbox_image,
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
                            workspace_file_exists = updated_text is not None
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

                            if (
                                project_language == ProjectLanguage.NODEJS
                                and rel_path.endswith(".json")
                                and unfixable_packages
                            ):
                                updated_text = _revert_unfixable_packages_in_json(
                                    original_text,
                                    updated_text,
                                    unfixable_packages,
                                    errors,
                                )
                            elif project_language == ProjectLanguage.PYTHON and unfixable_packages:
                                companion_manifest_text = None
                                if Path(rel_path).name.casefold() == "pipfile.lock":
                                    companion_path = (Path(rel_path).parent / "Pipfile").as_posix()
                                    original_companion = _read_host_text(repo_root, companion_path)
                                    workspace_companion = sandbox.read_file(companion_path)
                                    if not isinstance(workspace_companion, str):
                                        workspace_companion = original_companion
                                    companion_manifest_text = restore_python_package_changes(
                                        companion_path,
                                        original_companion,
                                        workspace_companion,
                                        unfixable_packages,
                                        errors,
                                    )
                                updated_text = restore_python_package_changes(
                                    rel_path,
                                    original_text,
                                    updated_text,
                                    unfixable_packages,
                                    errors,
                                    companion_manifest_text=companion_manifest_text,
                                )

                            diff_text = _build_diff(
                                rel_path,
                                original_text,
                                updated_text,
                                before_exists=host_exists,
                                after_exists=workspace_file_exists,
                            )
                            if diff_text:
                                diff_chunks.append(diff_text)
                                changed_files.append(rel_path)
                except Exception as exc:  # noqa: BLE001
                    msg = f"teardown_node: failed to extract diff from workspace volume - {exc}"
                    logger.exception("teardown_node: diff extraction failed.")
                    errors.append(msg)
    finally:
        if workspace_volume or maven_cache_volume:
            try:
                client = get_docker_client()
                if workspace_volume:
                    volume_removed, cleanup_errors = _cleanup_workspace_volume(
                        client, workspace_volume
                    )
                    errors.extend(cleanup_errors)
                    if volume_removed:
                        logger.info("teardown_node: removed workspace volume %s.", workspace_volume)
                    else:
                        logger.error(
                            "teardown_node: workspace volume %s remains after cleanup.",
                            workspace_volume,
                        )
                if maven_cache_volume:
                    maven_cache_removed, cleanup_errors = _cleanup_workspace_volume(
                        client, maven_cache_volume, "Maven cache"
                    )
                    errors.extend(cleanup_errors)
                    if maven_cache_removed:
                        logger.info(
                            "teardown_node: removed Maven cache volume %s.", maven_cache_volume
                        )
                    else:
                        logger.error(
                            "teardown_node: Maven cache volume %s remains after cleanup.",
                            maven_cache_volume,
                        )
            except Exception as exc:  # noqa: BLE001
                msg = f"teardown_node: failed to remove run-owned volumes - {exc}"
                logger.exception("teardown_node: run-owned volume cleanup failed.")
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
        "maven_cache_volume": None if maven_cache_removed else maven_cache_volume,
        "changed_files": ChangedFilesProjection(changed_files),
        "diff": "".join(diff_chunks),
    }
    if errors:
        result["errors"] = errors
    return result


def _state_value(value: Any, field: str, default: Any = None) -> Any:
    """Read one field from either a typed model or a state dictionary."""
    if isinstance(value, Mapping):
        return value.get(field, default)
    return getattr(value, field, default)


def run_final_full_scan_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run an optional full-workspace ODC scan on the modified Docker volume.

    The scan is performed only after the Supervisor routes a terminal run to
    this node. It never scans or modifies the host repository. If execution or
    report parsing fails, previously accumulated incremental identifiers are
    preserved and the full-scan result records the failure.
    """
    workspace_volume = state.get("workspace_volume")
    baseline = {
        identifier.strip().upper()
        for identifier in (state.get("baseline_scan_identifiers") or [])
        if isinstance(identifier, str) and identifier.strip()
    }
    if not workspace_volume:
        result = _SecurityScanResult(
            False,
            "FAILURE: final full scan skipped because the workspace volume is unavailable.",
            set(),
            set(),
            set(),
            execution_status=ScannerExecutionStatus.NOT_RUN,
        )
    else:
        project_language = state.get("project_language", ProjectLanguage.NODEJS)
        if not isinstance(project_language, ProjectLanguage):
            try:
                project_language = ProjectLanguage(project_language)
            except (TypeError, ValueError):
                project_language = ProjectLanguage.NODEJS
        try:
            with DockerSandbox(
                repo_root=None,
                workspace_volume=workspace_volume,
                image=LANGUAGE_CONFIGS[project_language].docker_image,
            ) as sandbox:
                if project_language == ProjectLanguage.NODEJS:
                    result = _run_security_scan(
                        sandbox,
                        workspace_volume,
                        baseline,
                        baseline,
                    )
                else:
                    result = _run_security_scan(
                        sandbox,
                        workspace_volume,
                        baseline,
                        baseline,
                        project_language=project_language,
                    )
        except Exception as exc:  # noqa: BLE001 - teardown must still run
            logger.exception("final_full_scan: Dependency-Check execution failed.")
            result = _SecurityScanResult(
                False,
                f"FAILURE: final full scan could not complete: {exc}",
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            )

    completed = result.execution_status == ScannerExecutionStatus.SUCCESS
    prior_result = state.get("final_full_scan_result")
    if completed:
        found_identifiers = sorted(result.found_identifiers)
        new_identifiers = sorted(result.new_identifiers)
        remaining_identifiers = sorted(result.remaining_identifiers)
        found_issues = list(result.found_issues)
    else:
        # Failed final scans must not erase identifiers already accumulated by
        # successful targeted scans.
        found_identifiers = list(state.get("post_remediation_scan_identifiers", []) or [])
        new_identifiers = list(state.get("new_vulnerability_identifiers", []) or [])
        remaining_identifiers = list(
            _state_value(prior_result, "remaining_target_identifiers", []) or []
        )
        found_issues = list(state.get("post_remediation_scan_issues", []) or [])

    status = (
        "scan_failed"
        if not completed
        else "detected"
        if new_identifiers
        else "unresolved"
        if remaining_identifiers
        else "none"
    )
    triage_required = bool(
        completed
        and state.get("post_qa_triage_enabled", True)
        and (new_identifiers or remaining_identifiers)
    )
    error = None if completed else result.summary
    final_result = FinalFullScanResult(
        completed=completed,
        authoritative=completed,
        coverage_scope="full_workspace",
        coverage_complete=completed,
        found_identifiers=found_identifiers,
        remaining_target_identifiers=remaining_identifiers,
        new_identifiers=new_identifiers,
        found_issues=found_issues,
        status=status,
        triage_required=triage_required,
        error=error,
    )
    if error:
        logger.warning("final_full_scan: %s", error)

    return {
        "status": "final_scan_completed" if completed else "final_scan_failed",
        "final_full_scan_result": final_result,
        "post_remediation_scan_identifiers": found_identifiers,
        "post_remediation_scan_issues": found_issues,
        "new_vulnerability_identifiers": new_identifiers,
        "new_vulnerability_status": status,
        "triage_required": triage_required,
        "errors": [f"final_full_scan: {error}"] if error else [],
    }
