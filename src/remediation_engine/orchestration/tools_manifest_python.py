"""Atomic Python dependency manifest transactions for worker tools."""

from __future__ import annotations

import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from remediation_engine.orchestration._tool_support import (
    _MANIFEST_SYNC_TIMEOUT_SECONDS,
    DockerSandbox,
    NoFixMitigationStage,
    WorkaroundExecutionPhase,
    _validate_workspace_path,
    _workspace_dir_for_manifest,
    json,
    shlex,
    tool,
)
from remediation_engine.tools.package_identity import normalize_python_package_name
from remediation_engine.tools.python_manifest_locator import (
    _pyproject_entries,
    _setup_cfg_entries,
    parse_requirements_txt,
    update_requirements_dependency,
)

from .tools_manifest import (
    _bounded_command_output,
    _capture_package_checkpoint,
    _normalize_python_manifest_targets,
    _PackageCheckpoint,
    _restore_package_checkpoint,
    _tool_error,
)

_SUPPORTED_DECLARATIONS = frozenset(
    {
        "requirements",
        "dependencies",
        "optional-dependencies",
        "install_requires",
        "extras_require",
        "packages",
        "dev-packages",
    }
)
_REQUIREMENTS_FILENAME = re.compile(r"requirements(?:[-_.][^/\\]+)?\.txt$", re.IGNORECASE)
_CANONICAL_PACKAGE_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REQUIREMENT_LINE = re.compile(
    r"^(?P<indent>\s*)(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<separator>\s*)"
    r"(?:(?:===|==|~=|!=|<=|>=|<|>)(?P<operator_space>\s*)[^\s;#]+)?"
    r"(?P<trailing>\s*)$"
)
_HASH_OPTION = re.compile(r"\s+--hash(?:=|\s)", re.IGNORECASE)


def _canonical_package(value: str) -> str | None:
    raw = str(value or "").strip()
    normalized = normalize_python_package_name(raw)
    if not normalized or not _CANONICAL_PACKAGE_NAME.fullmatch(normalized):
        return None
    return normalized if raw == normalized else None


def _canonical_version(value: str) -> str | None:
    raw = str(value or "").strip()
    try:
        normalized = str(Version(raw))
    except InvalidVersion:
        return None
    return normalized if raw == normalized else None


def _normalize_python_package_manifest_targets(
    package_manifest_paths: Mapping[str, Iterable[str]],
) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for raw_package, paths in package_manifest_paths.items():
        package = _canonical_package(str(raw_package or ""))
        if package is None:
            raise ValueError(
                f"Python package manifest target must be a canonical PEP 503 name: {raw_package!r}."
            )
        if package in normalized:
            raise ValueError(f"Duplicate canonical Python package target: {package!r}.")
        normalized[package] = _normalize_python_manifest_targets(paths)
    return normalized


def _approved_python_versions(
    allowed: Mapping[str, Iterable[str]] | None,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for raw_package, versions in (allowed or {}).items():
        package = _canonical_package(str(raw_package or ""))
        if package is None:
            raise ValueError(
                f"Python version allowlist key must be a canonical PEP 503 name: {raw_package!r}."
            )
        result[package] = {
            version
            for raw_version in versions
            if (version := _canonical_version(str(raw_version))) is not None
        }
    return result


def _approved_python_types(
    allowed: Mapping[str, Iterable[str]] | None,
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for raw_package, values in (allowed or {}).items():
        package = _canonical_package(str(raw_package or ""))
        if package is None:
            raise ValueError(
                f"Python declaration allowlist key must be a canonical PEP 503 name: {raw_package!r}."
            )
        result[package] = {str(value).strip() for value in values if str(value).strip()}
    return result


def _read_requirement_text(path: Path, text: str) -> list[Any]:
    """Reuse the repository parser without staging any data in the workspace."""
    with tempfile.TemporaryDirectory(prefix="remedy-python-manifest-") as directory:
        temporary_path = Path(directory) / path.name
        temporary_path.write_text(text, encoding="utf-8", newline="")
        return parse_requirements_txt(temporary_path)


def _with_temporary_manifest(path: Path, text: str, parser: Any) -> Any:
    with tempfile.TemporaryDirectory(prefix="remedy-python-manifest-") as directory:
        temporary_path = Path(directory) / path.name
        temporary_path.write_text(text, encoding="utf-8", newline="")
        return parser(temporary_path)


def _requirement_is_editable(requirement: Requirement, raw_line: str) -> bool:
    return (
        requirement.url is None
        and requirement.marker is None
        and not requirement.extras
        and len(list(requirement.specifier)) <= 1
        and not _HASH_OPTION.search(raw_line)
    )


def _matching_declarations(
    manifest_path: str,
    content: str,
    package_name: str,
) -> list[tuple[str, Requirement, int, str]]:
    path = Path(manifest_path)
    name = path.name.casefold()
    matches: list[tuple[str, Requirement, int, str]] = []
    if _REQUIREMENTS_FILENAME.fullmatch(path.name):
        for entry in _read_requirement_text(path, content):
            if normalize_python_package_name(entry.name) == package_name:
                matches.append(
                    ("requirements", entry.requirement, entry.line_number, entry.raw_line)
                )
        return matches
    if name == "pyproject.toml":
        entries = _with_temporary_manifest(path, content, _pyproject_entries)
        for requirement, declaration_type, line_number, raw_value in entries:
            if normalize_python_package_name(requirement.name) == package_name:
                matches.append((declaration_type, requirement, line_number or 0, raw_value))
        return matches
    if name == "setup.cfg":
        entries = _with_temporary_manifest(path, content, _setup_cfg_entries)
        for requirement, declaration_type, line_number, raw_value in entries:
            if normalize_python_package_name(requirement.name) == package_name:
                matches.append((declaration_type, requirement, line_number, raw_value))
        return matches
    if name == "pipfile":
        try:
            document = tomlkit.parse(content)
        except (ValueError, tomlkit.exceptions.ParseError):
            return []
        for declaration_type in ("packages", "dev-packages"):
            section = document.get(declaration_type)
            if not isinstance(section, Mapping):
                continue
            for raw_name, value in section.items():
                if normalize_python_package_name(str(raw_name)) != package_name:
                    continue
                specifier = value.get("version") if isinstance(value, Mapping) else value
                if not isinstance(specifier, str) or not specifier or specifier == "*":
                    requirement = Requirement(str(raw_name))
                else:
                    try:
                        requirement = Requirement(f"{raw_name}{specifier}")
                    except InvalidRequirement:
                        continue
                matches.append((declaration_type, requirement, 0, str(raw_name)))
        return matches
    return []


def _replace_requirement_line(raw_line: str, package_name: str, version: str) -> str:
    comment_match = re.search(r"\s+#", raw_line)
    declaration = raw_line[: comment_match.start()] if comment_match else raw_line
    comment = raw_line[comment_match.start() :] if comment_match else ""
    match = _REQUIREMENT_LINE.fullmatch(declaration)
    if match is None or normalize_python_package_name(match.group("name")) != package_name:
        raise ValueError("unsupported Python dependency declaration syntax")
    return (
        f"{match.group('indent')}{match.group('name')}{match.group('separator')}"
        f"=={match.group('operator_space') or ''}{version}{match.group('trailing')}"
        f"{comment}"
    )


def _line_at(text: str, line_number: int) -> tuple[list[str], str]:
    lines = text.splitlines(keepends=True)
    if line_number < 1 or line_number > len(lines):
        raise ValueError("could not locate the authorized dependency declaration line")
    return lines, lines[line_number - 1]


def _update_python_manifest(
    manifest_path: str,
    content: str,
    package_name: str,
    target_version: str,
    declaration_type: str,
) -> str:
    name = Path(manifest_path).name.casefold()
    matches = [
        item
        for item in _matching_declarations(manifest_path, content, package_name)
        if item[0] == declaration_type
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one matching direct Python declaration of the approved type"
        )
    _, requirement, line_number, raw_value = matches[0]
    if not _requirement_is_editable(requirement, raw_value):
        raise ValueError("unsupported, conditional, direct-URL, or hash-pinned Python declaration")

    if _REQUIREMENTS_FILENAME.fullmatch(Path(manifest_path).name):
        return _with_temporary_manifest(
            Path(manifest_path),
            content,
            lambda temporary_path: update_requirements_dependency(
                temporary_path,
                package_name,
                target_version,
            ),
        )

    if name == "pyproject.toml":
        document = tomlkit.parse(content)
        project = document.get("project")
        if not isinstance(project, Mapping):
            raise ValueError("pyproject.toml must contain a static [project] declaration")
        collections: list[Any] = []
        if declaration_type == "dependencies":
            collections.append(project.get("dependencies"))
        else:
            optional = project.get("optional-dependencies")
            if isinstance(optional, Mapping):
                collections.extend(optional.values())
        replacement_count = 0
        for values in collections:
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for index, raw_value in enumerate(values):
                if not isinstance(raw_value, str):
                    continue
                try:
                    parsed = Requirement(raw_value)
                except InvalidRequirement:
                    continue
                if normalize_python_package_name(parsed.name) != package_name:
                    continue
                values[index] = f"{parsed.name}=={target_version}"
                replacement_count += 1
        if replacement_count != 1:
            raise ValueError("expected exactly one editable declaration in pyproject.toml")
        return tomlkit.dumps(document)

    if name == "setup.cfg":
        lines, original_line = _line_at(content, line_number)
        assignment = re.fullmatch(
            r"(?P<prefix>\s*[A-Za-z0-9_.-]+\s*=\s*)(?P<value>.*?)(?P<ending>\r?\n)?", original_line
        )
        if assignment:
            raw_value = assignment.group("value")
            replaced = _replace_requirement_line(raw_value, package_name, target_version)
            lines[line_number - 1] = (
                assignment.group("prefix") + replaced + (assignment.group("ending") or "")
            )
        else:
            body = original_line.rstrip("\r\n")
            ending = original_line[len(body) :]
            lines[line_number - 1] = (
                _replace_requirement_line(
                    body,
                    package_name,
                    target_version,
                )
                + ending
            )
        return "".join(lines)

    if name == "pipfile":
        document = tomlkit.parse(content)
        section = document.get(declaration_type)
        if not isinstance(section, Mapping):
            raise ValueError("Pipfile direct declaration section is missing")
        actual_key = next(
            key for key in section if normalize_python_package_name(str(key)) == package_name
        )
        value = section[actual_key]
        if isinstance(value, Mapping):
            if "version" not in value or any(
                key in value for key in ("git", "path", "file", "editable", "markers")
            ):
                raise ValueError("unsupported Pipfile declaration form")
            value["version"] = f"=={target_version}"
        elif isinstance(value, str):
            section[actual_key] = f"=={target_version}"
        else:
            raise ValueError("unsupported Pipfile declaration form")
        return tomlkit.dumps(document)
    raise ValueError("manifest_path is not a supported Python dependency manifest")


def _remove_python_manifest_entry(
    manifest_path: str,
    content: str,
    package_name: str,
) -> str:
    matches = _matching_declarations(manifest_path, content, package_name)
    if len(matches) != 1:
        raise ValueError("expected exactly one directly declared Python dependency to remove")
    declaration_type, _, line_number, raw_value = matches[0]
    path = Path(manifest_path)
    name = path.name.casefold()
    if _REQUIREMENTS_FILENAME.fullmatch(path.name):
        lines = content.splitlines(keepends=True)
        start = line_number - 1
        end = start + 1
        if lines[start].rstrip("\r\n").rstrip().endswith("\\"):
            while end < len(lines):
                continued = lines[end]
                end += 1
                if not continued.rstrip("\r\n").rstrip().endswith("\\"):
                    break
        del lines[start:end]
        return "".join(lines)
    if name == "pyproject.toml":
        document = tomlkit.parse(content)
        project = document.get("project")
        if not isinstance(project, Mapping):
            raise ValueError("pyproject.toml must contain a static [project] declaration")
        if declaration_type == "dependencies":
            collections = [project.get("dependencies")]
        else:
            optional = project.get("optional-dependencies")
            collections = list(optional.values()) if isinstance(optional, Mapping) else []
        removed = 0
        for values in collections:
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                continue
            for index in range(len(values) - 1, -1, -1):
                raw = values[index]
                if not isinstance(raw, str):
                    continue
                try:
                    requirement = Requirement(raw)
                except InvalidRequirement:
                    continue
                if normalize_python_package_name(requirement.name) == package_name:
                    del values[index]
                    removed += 1
        if removed != 1:
            raise ValueError("could not remove exactly one pyproject.toml dependency")
        return tomlkit.dumps(document)
    if name == "setup.cfg":
        lines = content.splitlines(keepends=True)
        line_index = line_number - 1
        line = lines[line_index]
        assignment = re.fullmatch(r"\s*[A-Za-z0-9_.-]+\s*=\s*(.*?)\s*(?:\r?\n)?", line)
        if assignment:
            if normalize_python_package_name(raw_value.split("==", 1)[0].strip()) != package_name:
                raise ValueError("refusing to remove a compound setup.cfg declaration")
            del lines[line_index]
        else:
            del lines[line_index]
        return "".join(lines)
    if name == "pipfile":
        document = tomlkit.parse(content)
        section = document.get(declaration_type)
        if not isinstance(section, Mapping):
            raise ValueError("Pipfile direct declaration section is missing")
        actual_key = next(
            key for key in section if normalize_python_package_name(str(key)) == package_name
        )
        del section[actual_key]
        return tomlkit.dumps(document)
    raise ValueError("manifest_path is not a supported Python dependency manifest")


def _python_validation_command(manifest_path: str) -> str:
    workspace_dir = _workspace_dir_for_manifest(manifest_path)
    if _REQUIREMENTS_FILENAME.fullmatch(Path(manifest_path).name):
        install_args = f"-r {shlex.quote('/workspace/' + manifest_path)}"
    else:
        install_args = "-e ."
    command = f"/workspace/.venv/bin/python -m pip install --dry-run {install_args}"
    return f"cd {shlex.quote(workspace_dir)} && {command}"


def _pipenv_command(manifest_path: str, action: str) -> str:
    workspace_dir = _workspace_dir_for_manifest(manifest_path)
    command = f"/workspace/.remedy-pipenv/bin/pipenv {action}"
    if action == "sync --dev":
        command = f"PIPENV_VENV_IN_PROJECT=1 {command}"
    return f"cd {shlex.quote(workspace_dir)} && {command}"


def _rollback_python_transaction(
    sandbox: DockerSandbox,
    checkpoint: _PackageCheckpoint,
    touched_files: set[str],
    manifest_path: str,
) -> list[str]:
    errors: list[str] = []
    rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
    if rollback_error:
        errors.append(rollback_error)
    saved_lock = checkpoint.files.get(
        _validate_workspace_path((Path(manifest_path).parent / "Pipfile.lock").as_posix())
    )
    if isinstance(saved_lock, str):
        try:
            result = sandbox.run(
                _pipenv_command(manifest_path, "sync --dev"),
                timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Pipenv rollback sync failed: {exc}")
        else:
            if result.exit_code != 0:
                errors.append(
                    "Pipenv rollback sync failed "
                    f"(exit {result.exit_code}): {_bounded_command_output(result.stderr)}"
                )
    return errors


def _changed_checkpoint_files(
    sandbox: DockerSandbox,
    checkpoint: _PackageCheckpoint,
) -> list[str]:
    changed: list[str] = []
    for path, before in checkpoint.files.items():
        after_value = sandbox.read_file(path)
        after = after_value if isinstance(after_value, str) else None
        if after != before:
            changed.append(path)
    return sorted(changed)


def _success_result(message: str, changed_files: list[str]) -> str:
    projection = json.dumps({"changed_files": changed_files}, sort_keys=True)
    return f"SUCCESS: {message} JSON: {projection}"


def _record_transaction_attempt(
    execution_state: dict[str, Any],
    package_name: str,
    signature: str,
) -> str | None:
    attempts_by_package = execution_state.setdefault("manifest_transaction_attempts_by_package", {})
    signatures_by_package = execution_state.setdefault(
        "manifest_transaction_signatures_by_package", {}
    )
    signatures = set(signatures_by_package.get(package_name, []))
    if signature in signatures:
        return _tool_error(
            "RETRY_PARAMETERS_UNCHANGED",
            "Retry the package with a different allowed target_version or declaration type.",
        )
    if int(attempts_by_package.get(package_name, 0)) >= 3:
        return _tool_error(
            "RETRY_LIMIT_REACHED",
            "At most three combined update attempts are allowed for this package.",
        )
    signatures.add(signature)
    signatures_by_package[package_name] = sorted(signatures)
    attempts_by_package[package_name] = int(attempts_by_package.get(package_name, 0)) + 1
    execution_state["manifest_transaction_attempts"] = (
        int(execution_state.get("manifest_transaction_attempts", 0)) + 1
    )
    return None


def _make_modify_and_validate_python_dependency_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    *,
    allowed_target_versions_by_package: Mapping[str, Iterable[str]] | None = None,
    allowed_dependency_types_by_package: Mapping[str, Iterable[str]] | None = None,
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
):
    """Build a worker transaction restricted to committed PyPI targets."""
    manifests_by_package = _normalize_python_package_manifest_targets(package_manifest_paths)
    allowed_versions = _approved_python_versions(allowed_target_versions_by_package)
    allowed_types = _approved_python_types(allowed_dependency_types_by_package)
    state = execution_state if execution_state is not None else {}
    checkpoints = package_checkpoints if package_checkpoints is not None else {}

    @tool
    def modify_and_validate_python_dependency(
        package_name: str,
        target_version: str,
        dependency_type: str,
        manifest_path: str,
    ) -> str:
        """Atomically update one Supervisor-authorized Python dependency."""
        package = _canonical_package(package_name)
        if package is None:
            return _tool_error("INVALID_ARGUMENT", "package_name must be a canonical PEP 503 name.")
        version = _canonical_version(target_version)
        if version is None:
            return _tool_error(
                "INVALID_ARGUMENT", "target_version must be a canonical PEP 440 version."
            )
        dependency = str(dependency_type or "").strip()
        if dependency not in _SUPPORTED_DECLARATIONS:
            return _tool_error(
                "INVALID_ARGUMENT",
                "dependency_type must be one of: "
                + ", ".join(sorted(_SUPPORTED_DECLARATIONS))
                + ".",
            )
        try:
            relative_manifest = _validate_workspace_path(manifest_path)
        except ValueError as exc:
            return _tool_error("INVALID_ARGUMENT", str(exc))
        try:
            supported_targets = _normalize_python_manifest_targets([relative_manifest])
        except ValueError as exc:
            return _tool_error("INVALID_ARGUMENT", str(exc))
        relative_manifest = supported_targets[0]

        approved_manifests = manifests_by_package.get(package)
        if not approved_manifests:
            return _tool_error(
                "TARGET_NOT_ALLOWED", f"package_name {package!r} is not an allowed target."
            )
        if relative_manifest not in approved_manifests:
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"manifest_path {relative_manifest!r} is not allowed for {package!r}.",
            )
        if package not in allowed_versions or version not in allowed_versions[package]:
            candidates = ", ".join(sorted(allowed_versions.get(package, set()))) or "none"
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"target_version {version!r} is not in the Supervisor-approved candidates: {candidates}.",
            )
        if package not in allowed_types or dependency not in allowed_types[package]:
            types = ", ".join(sorted(allowed_types.get(package, set()))) or "none"
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"dependency_type {dependency!r} is not in the Supervisor-approved declaration types: {types}.",
            )
        pending = str(state.get("pending_validation_package", "") or "")
        if pending and pending != package:
            return _tool_error(
                "TARGET_NOT_ALLOWED",
                f"Package {pending!r} has an unfinished update transaction.",
            )
        signature = "|".join((package, version, dependency, relative_manifest))
        attempt_error = _record_transaction_attempt(state, package, signature)
        if attempt_error:
            return attempt_error

        pipfile = Path(relative_manifest).name.casefold() == "pipfile"
        checkpoint: _PackageCheckpoint | None = None
        try:
            if package not in checkpoints:
                checkpoints[package] = _capture_package_checkpoint(
                    sandbox,
                    [relative_manifest],
                    touched_files,
                    package_ecosystem="pypi",
                )
            checkpoint = checkpoints[package]
            before = checkpoint.files.get(relative_manifest)
            if not isinstance(before, str):
                raise ValueError(
                    f"authorized manifest {relative_manifest!r} is missing or unreadable"
                )
            edited = _update_python_manifest(
                relative_manifest,
                before,
                package,
                version,
                dependency,
            )
            sandbox.write_file(relative_manifest, edited)
            state["edits_started"] = True
            state["pending_validation_package"] = package
            state["validation_calls"] = int(state.get("validation_calls", 0)) + 1

            if pipfile:
                command = _pipenv_command(relative_manifest, "lock")
            else:
                command = _python_validation_command(relative_manifest)
            result = sandbox.run(command, timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS)
            if result.exit_code != 0:
                failure_name = "Pipenv lock" if pipfile else "Python dependency validation"
                raise RuntimeError(
                    f"{failure_name} failed (exit {result.exit_code}). "
                    f"stdout: {_bounded_command_output(result.stdout)}; "
                    f"stderr: {_bounded_command_output(result.stderr)}"
                )
            if pipfile:
                result = sandbox.run(
                    _pipenv_command(relative_manifest, "sync --dev"),
                    timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
                )
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"Pipenv sync failed (exit {result.exit_code}). "
                        f"stdout: {_bounded_command_output(result.stdout)}; "
                        f"stderr: {_bounded_command_output(result.stderr)}"
                    )
                lock_path = _validate_workspace_path(
                    (Path(relative_manifest).parent / "Pipfile.lock").as_posix()
                )
                if not isinstance(sandbox.read_file(lock_path), str):
                    raise RuntimeError("Pipenv lock completed without producing Pipfile.lock")

            changed_files = _changed_checkpoint_files(sandbox, checkpoint)
            touched_files.update(changed_files)
            checkpoints.pop(package, None)
            state["pending_validation_package"] = None
            validated = list(state.get("validated_packages", []) or [])
            if package not in validated:
                validated.append(package)
            state["validated_packages"] = validated
            return _success_result(
                f"Updated {dependency}.{package} to {version} in {relative_manifest}; "
                "Python dependency validation succeeded.",
                changed_files,
            )
        except Exception as exc:  # noqa: BLE001
            rollback_errors: list[str] = []
            if checkpoint is not None:
                rollback_errors = (
                    _rollback_python_transaction(
                        sandbox,
                        checkpoint,
                        touched_files,
                        relative_manifest,
                    )
                    if pipfile
                    else [
                        error
                        for error in [
                            _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                        ]
                        if error
                    ]
                )
            checkpoints.pop(package, None)
            state["pending_validation_package"] = None
            failure = _tool_error("PYTHON_TRANSACTION_FAILED", str(exc))
            rollback_note = (
                " Rollback failed: " + " | ".join(rollback_errors)
                if rollback_errors
                else " Rollback: package checkpoint restored."
            )
            return failure + rollback_note

    return modify_and_validate_python_dependency


def _make_remove_no_fix_python_dependency_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any],
    package_name: str,
    manifest_paths: Sequence[str],
    package_manager: str,
):
    """Build the narrow direct-dependency removal used only in Python NO_FIX."""
    configured_package = _canonical_package(package_name)
    manifests = _normalize_python_manifest_targets(manifest_paths)
    manager = str(package_manager or "").strip().casefold()
    checkpoint: _PackageCheckpoint | None = None

    @tool
    def remove_no_fix_python_dependency(
        package_name: str,
        manifest_path: str,
    ) -> str:
        """Remove one authorized direct Python declaration and sync Pipenv if needed."""
        nonlocal checkpoint
        if plan_state.get("no_fix_stage") != NoFixMitigationStage.PACKAGE_REMOVAL.value:
            return (
                "NOT_APPLICABLE: Python package removal is available only during PACKAGE_REMOVAL."
            )
        if not plan_state.get("recorded") or not plan_state.get("package_removal_planned"):
            return (
                "ERROR: [PLAN_VIOLATION] Record an explicit package-removal plan before "
                "calling remove_no_fix_python_dependency."
            )
        requested_package = _canonical_package(package_name)
        if configured_package is None or requested_package != configured_package:
            return (
                "ERROR: [ALLOWLIST] remove_no_fix_python_dependency accepts only the configured "
                f"canonical package '{configured_package or ''}'."
            )
        try:
            relative_manifest = _validate_workspace_path(manifest_path)
            relative_manifest = _normalize_python_manifest_targets([relative_manifest])[0]
        except (ValueError, IndexError) as exc:
            return f"ERROR: [ALLOWLIST] {exc}"
        if relative_manifest not in manifests:
            return (
                "ERROR: [ALLOWLIST] manifest_path is not an authorized Python target. "
                f"Allowed paths: {', '.join(manifests)}."
            )
        pipfile = Path(relative_manifest).name.casefold() == "pipfile"
        expected_manager = "pipenv" if pipfile else "pip"
        if manager != expected_manager:
            return (
                "NOT_APPLICABLE: this manifest requires the "
                f"'{expected_manager}' manager; no manifest or lockfile was changed."
            )
        before_content = sandbox.read_file(relative_manifest)
        if not isinstance(before_content, str):
            return f"NOT_APPLICABLE: manifest '{relative_manifest}' is missing or unreadable."
        try:
            matches = _matching_declarations(relative_manifest, before_content, requested_package)
            if len(matches) != 1:
                return (
                    "NOT_APPLICABLE: the package must have exactly one directly declared entry "
                    "in the authorized Python manifest."
                )
            if checkpoint is None:
                checkpoint = _capture_package_checkpoint(
                    sandbox,
                    [relative_manifest],
                    touched_files,
                    package_ecosystem="pypi",
                )
                attempt_snapshots = plan_state.setdefault("attempt_file_snapshots", {})
                attempt_absent = set(plan_state.setdefault("attempt_absent_paths", []))
                for path, content in checkpoint.files.items():
                    if isinstance(content, str):
                        attempt_snapshots.setdefault(path, content)
                    else:
                        attempt_absent.add(path)
                plan_state["attempt_absent_paths"] = sorted(attempt_absent)
                baseline = plan_state.setdefault("stage_baseline_snapshots", {})
                absent = set(plan_state.setdefault("stage_baseline_absent_paths", []))
                for path, content in checkpoint.files.items():
                    if isinstance(content, str):
                        baseline.setdefault(path, content)
                    else:
                        absent.add(path)
                plan_state["stage_baseline_absent_paths"] = sorted(absent)

            edited = _remove_python_manifest_entry(
                relative_manifest,
                before_content,
                requested_package,
            )
            sandbox.write_file(relative_manifest, edited)
            pipfile = Path(relative_manifest).name.casefold() == "pipfile"
            if pipfile:
                lock_result = sandbox.run(
                    _pipenv_command(relative_manifest, "lock"),
                    timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
                )
                if lock_result.exit_code != 0:
                    raise RuntimeError(
                        f"Pipenv lock failed (exit {lock_result.exit_code}). "
                        f"stdout: {_bounded_command_output(lock_result.stdout)}; "
                        f"stderr: {_bounded_command_output(lock_result.stderr)}"
                    )
                sync_result = sandbox.run(
                    _pipenv_command(relative_manifest, "sync --dev"),
                    timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
                )
                if sync_result.exit_code != 0:
                    raise RuntimeError(
                        f"Pipenv sync failed (exit {sync_result.exit_code}). "
                        f"stdout: {_bounded_command_output(sync_result.stdout)}; "
                        f"stderr: {_bounded_command_output(sync_result.stderr)}"
                    )
                lock_path = _validate_workspace_path(
                    (Path(relative_manifest).parent / "Pipfile.lock").as_posix()
                )
                if not isinstance(sandbox.read_file(lock_path), str):
                    raise RuntimeError("Pipenv lock completed without producing Pipfile.lock")
            else:
                validation_result = sandbox.run(
                    _python_validation_command(relative_manifest),
                    timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
                )
                if validation_result.exit_code != 0:
                    raise RuntimeError(
                        "Python dependency validation failed "
                        f"(exit {validation_result.exit_code}). "
                        f"stdout: {_bounded_command_output(validation_result.stdout)}; "
                        f"stderr: {_bounded_command_output(validation_result.stderr)}"
                    )
            changed_files = _changed_checkpoint_files(sandbox, checkpoint)
            touched_files.update(changed_files)
            plan_state["package_removal_planned"] = True
            plan_state["package_removal_completed"] = True
            plan_state["package_removal_files"] = sorted(
                set(plan_state.get("package_removal_files", [])) | set(changed_files)
            )
            plan_state["no_fix_package_removed"] = True
            source_edit_pending = plan_state.get("pending_edit_set") is not None
            plan_state["phase"] = (
                WorkaroundExecutionPhase.VALIDATE.value
                if not plan_state.get("planned_replacements") or source_edit_pending
                else WorkaroundExecutionPhase.EXECUTE.value
            )
            return _success_result(
                f"Removed the configured direct Python dependency from {relative_manifest}; "
                "Pipenv lock/sync completed when applicable.",
                changed_files,
            )
        except Exception as exc:  # noqa: BLE001
            rollback_errors: list[str] = []
            if checkpoint is not None:
                pipfile = Path(relative_manifest).name.casefold() == "pipfile"
                if pipfile:
                    rollback_errors = _rollback_python_transaction(
                        sandbox,
                        checkpoint,
                        touched_files,
                        relative_manifest,
                    )
                else:
                    rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                    if rollback_error:
                        rollback_errors.append(rollback_error)
            checkpoint = None
            suffix = (
                " Rollback failed: " + " | ".join(rollback_errors)
                if rollback_errors
                else " Rollback: package checkpoint restored."
            )
            return f"FAILURE: Python package removal failed: {exc}.{suffix}"

    return remove_no_fix_python_dependency


__all__ = [
    "_make_modify_and_validate_python_dependency_tool",
    "_make_remove_no_fix_python_dependency_tool",
]
