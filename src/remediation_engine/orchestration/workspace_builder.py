"""
workspace_builder.py - Shared Docker workspace preparation for remediation workers.

The node creates a named volume, copies the host repository into it, and
materializes every npm package in that shared workspace. Worker nodes perform
edits and validation against the initialized volume.
"""

from __future__ import annotations

import ast
import configparser
import logging
import shlex
import tomllib
import uuid
from pathlib import Path
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.orchestration.state import OrchestratorState
from remediation_engine.runtime.docker_client import close_docker_client
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client

logger = logging.getLogger(__name__)

_NPM_INSTALL_COMMAND = "npm install --package-lock=true"
_NPM_INSTALL_TIMEOUT_SECONDS = 900
_PYTHON_SETUP_TIMEOUT_SECONDS = 900
_INSTALL_LOG_TAIL_LINES = 80
_SKIP_PACKAGE_DIR_NAMES = frozenset(
    {".git", "node_modules", ".venv", "venv", ".remedy-pipenv", "__pycache__"}
)


class _UnsupportedPythonManagerError(RuntimeError):
    """Raised when Python dependencies require an unsupported environment manager."""


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
            ".remedy-pipenv/bin/python -m pip install pipenv",
            "PIPENV_VENV_IN_PROJECT=1 .remedy-pipenv/bin/pipenv --python 3.11",
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
        commands = ["rm -rf .venv .remedy-pipenv", "python -m venv .venv"]
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
    """
    LangGraph node - Workspace Builder.

    Creates a named volume, copies the host repository into it, and installs
    dependencies with the selected project language's runtime before workers
    edit the initialized shared workspace.
    """
    repo_root_str: str = state.get("repo_root", "")
    raw_language = state.get("project_language", ProjectLanguage.NODEJS)
    project_language = (
        raw_language
        if isinstance(raw_language, ProjectLanguage)
        else ProjectLanguage(raw_language or ProjectLanguage.NODEJS.value)
    )
    image = LANGUAGE_CONFIGS[project_language].docker_image
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"workspace_builder_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {
            "status": "workspace_build_failed",
            "workspace_volume": None,
            "errors": [msg],
        }

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
            "errors": [msg],
        }
    finally:
        if client is not None:
            _close_client(client)

    try:
        repo_root = Path(repo_root_str)
        python_setup_commands = (
            _python_project_setup_plan(repo_root)
            if project_language == ProjectLanguage.PYTHON
            else ()
        )
        with DockerSandbox(
            repo_root_str,
            image=image,
            workspace_volume=volume_name,
        ) as sandbox:
            logger.info(
                "workspace_builder_node: repository copied into shared volume %s.",
                volume_name,
            )
            if project_language == ProjectLanguage.PYTHON:
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
            "errors": [msg],
        }

    return {
        "workspace_volume": volume_name,
        "status": "workspace_ready",
    }
