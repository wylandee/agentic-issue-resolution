"""Atomic Python dependency manifest transactions for worker tools."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
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


def _declaration_is_editable(manifest_path: str, requirement: Requirement, raw_line: str) -> bool:
    """Allow static Pipfile ranges while retaining strict line-file checks."""
    if Path(manifest_path).name.casefold() == "pipfile":
        return (
            requirement.url is None
            and requirement.marker is None
            and not requirement.extras
            and not _HASH_OPTION.search(raw_line)
        )
    return _requirement_is_editable(requirement, raw_line)


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
    if not _declaration_is_editable(manifest_path, requirement, raw_value):
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


def restore_python_package_changes(
    manifest_path: str,
    original_text: str,
    updated_text: str,
    package_names: Iterable[str],
    errors: list[str] | None = None,
    *,
    companion_manifest_text: str | None = None,
) -> str:
    """Restore selected Python declarations from the host baseline.

    The teardown path uses this helper when a shared manifest contains both
    accepted package changes and unresolved package changes. Only declarations
    for ``package_names`` are restored; unrelated successful declarations stay
    in the candidate patch. Pipfile.lock entries are restored by canonical
    package identity as well.

    Args:
        manifest_path: Repository-relative Python manifest or Pipfile.lock path.
        original_text: Exact host baseline content.
        updated_text: Current workspace content.
        package_names: Python distributions whose declarations must be restored.
        errors: Optional diagnostics list for fail-closed parse or ambiguity cases.
        companion_manifest_text: Restored Pipfile content used to refresh the
            Pipfile.lock manifest hash when restoring a lockfile.

    Returns:
        Updated text with only the selected package changes restored. If a
        supported file cannot be safely edited, the host baseline is returned.

    Side Effects:
        None. The helper parses and returns text without reading or writing files.
    """
    targets = {
        normalized
        for package_name in package_names
        if (normalized := normalize_python_package_name(str(package_name or "")))
    }
    if not targets or original_text == updated_text:
        return updated_text

    path = Path(manifest_path)
    name = path.name.casefold()
    try:
        if _REQUIREMENTS_FILENAME.fullmatch(path.name) or name == "setup.cfg":
            return _restore_python_line_declarations(
                manifest_path,
                original_text,
                updated_text,
                targets,
            )
        if name == "pyproject.toml":
            return _restore_pyproject_declarations(
                original_text,
                updated_text,
                targets,
            )
        if name == "pipfile":
            return _restore_pipfile_declarations(original_text, updated_text, targets)
        if name == "pipfile.lock":
            return _restore_pipfile_lock_entries(
                original_text,
                updated_text,
                targets,
                companion_manifest_text,
            )
    except (
        OSError,
        TypeError,
        ValueError,
        tomlkit.exceptions.ParseError,
        json.JSONDecodeError,
    ) as exc:
        if errors is not None:
            errors.append(
                f"could not safely restore Python package entries in {manifest_path}: {exc}"
            )
        return original_text
    return updated_text


def _restore_python_line_declarations(
    manifest_path: str,
    original_text: str,
    updated_text: str,
    package_names: set[str],
) -> str:
    """Restore requirements/setup.cfg declarations while preserving other lines."""
    original_lines = original_text.splitlines(keepends=True)
    updated = updated_text
    original_positions: list[tuple[int, str]] = []
    for package_name in package_names:
        matches = _matching_declarations(manifest_path, original_text, package_name)
        if len(matches) > 1:
            raise ValueError(f"ambiguous baseline declaration for {package_name!r}")
        if matches:
            original_positions.append((matches[0][2], package_name))

    for original_line_number, package_name in sorted(original_positions):
        original_matches = _matching_declarations(manifest_path, original_text, package_name)
        current_matches = _matching_declarations(manifest_path, updated, package_name)
        if len(current_matches) > 1:
            raise ValueError(f"ambiguous workspace declaration for {package_name!r}")
        if not original_matches:
            continue
        baseline_line = original_lines[original_line_number - 1]
        updated_lines = updated.splitlines(keepends=True)
        if current_matches:
            updated_lines[current_matches[0][2] - 1] = baseline_line
        else:
            insert_at = min(original_line_number - 1, len(updated_lines))
            if insert_at < len(updated_lines) and not baseline_line.endswith(("\n", "\r")):
                ending = "\r\n" if "\r\n" in updated else "\n"
                baseline_line += ending
            updated_lines.insert(insert_at, baseline_line)
        updated = "".join(updated_lines)

    # Remove declarations introduced for a package that had no baseline entry.
    for package_name in package_names:
        if any(name == package_name for _, name in original_positions):
            continue
        current_matches = _matching_declarations(manifest_path, updated, package_name)
        if len(current_matches) > 1:
            raise ValueError(f"ambiguous workspace declaration for {package_name!r}")
        if current_matches:
            lines = updated.splitlines(keepends=True)
            del lines[current_matches[0][2] - 1]
            updated = "".join(lines)
    return updated


def _iter_pyproject_dependency_arrays(document: Any) -> list[tuple[tuple[str, str], Any]]:
    project = document.get("project") if isinstance(document, Mapping) else None
    if not isinstance(project, Mapping):
        return []
    result: list[tuple[tuple[str, str], Any]] = []
    dependencies = project.get("dependencies")
    if isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes)):
        result.append((("dependencies", ""), dependencies))
    optional = project.get("optional-dependencies")
    if isinstance(optional, Mapping):
        for extra_name, values in optional.items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                result.append((("optional-dependencies", str(extra_name)), values))
    return result


def _pyproject_requirement_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return normalize_python_package_name(Requirement(value).name)
    except InvalidRequirement:
        return None


def _restore_pyproject_declarations(
    original_text: str,
    updated_text: str,
    package_names: set[str],
) -> str:
    original_document = tomlkit.parse(original_text)
    updated_document = tomlkit.parse(updated_text)
    original_arrays = _iter_pyproject_dependency_arrays(original_document)
    updated_arrays = _iter_pyproject_dependency_arrays(updated_document)
    baseline_entries: list[tuple[tuple[str, str], int, str]] = []
    for section_key, values in original_arrays:
        for index, raw_value in enumerate(values):
            if _pyproject_requirement_name(raw_value) in package_names and isinstance(
                raw_value, str
            ):
                baseline_entries.append((section_key, index, raw_value))

    for _, values in updated_arrays:
        for index in range(len(values) - 1, -1, -1):
            if _pyproject_requirement_name(values[index]) in package_names:
                del values[index]

    project = updated_document.get("project")
    if not isinstance(project, Mapping):
        if baseline_entries:
            raise ValueError("pyproject.toml project table is missing")
        return updated_text
    for section_key, index, raw_value in sorted(
        baseline_entries, key=lambda item: (item[0], item[1])
    ):
        section, extra_name = section_key
        if section == "dependencies":
            values = project.get("dependencies")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                values = tomlkit.array()
                values.multiline(True)
                project["dependencies"] = values
        else:
            optional = project.get("optional-dependencies")
            if not isinstance(optional, Mapping):
                optional = tomlkit.table()
                project["optional-dependencies"] = optional
            values = optional.get(extra_name)
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                values = tomlkit.array()
                values.multiline(True)
                optional[extra_name] = values
        values.insert(min(index, len(values)), raw_value)
    return tomlkit.dumps(updated_document)


def _restore_pipfile_declarations(
    original_text: str,
    updated_text: str,
    package_names: set[str],
) -> str:
    original_document = tomlkit.parse(original_text)
    updated_document = tomlkit.parse(updated_text)
    for section_name in ("packages", "dev-packages"):
        original_section = original_document.get(section_name)
        updated_section = updated_document.get(section_name)
        if not isinstance(original_section, Mapping):
            original_section = {}
        if not isinstance(updated_section, Mapping):
            if not any(
                normalize_python_package_name(str(key)) in package_names for key in original_section
            ):
                continue
            updated_section = tomlkit.table()
            updated_document[section_name] = updated_section
        for key in list(updated_section):
            if normalize_python_package_name(str(key)) in package_names:
                del updated_section[key]
        for key, value in original_section.items():
            if normalize_python_package_name(str(key)) in package_names:
                updated_section[key] = deepcopy(value)
    return tomlkit.dumps(updated_document)


def _restore_pipfile_lock_entries(
    original_text: str,
    updated_text: str,
    package_names: set[str],
    pipfile_text: str | None,
) -> str:
    original_document = json.loads(original_text)
    updated_document = json.loads(updated_text)
    if not isinstance(original_document, Mapping) or not isinstance(updated_document, dict):
        raise ValueError("Pipfile.lock content must be JSON objects")
    for section_name in ("default", "develop"):
        original_section = original_document.get(section_name)
        updated_section = updated_document.get(section_name)
        if not isinstance(original_section, Mapping):
            original_section = {}
        if not isinstance(updated_section, dict):
            if not any(
                normalize_python_package_name(str(key)) in package_names for key in original_section
            ):
                continue
            updated_section = {}
            updated_document[section_name] = updated_section
        for key in list(updated_section):
            if normalize_python_package_name(str(key)) in package_names:
                del updated_section[key]
        for key, value in original_section.items():
            if normalize_python_package_name(str(key)) in package_names:
                updated_section[key] = deepcopy(value)
    if pipfile_text is not None:
        metadata = updated_document.get("_meta")
        if isinstance(metadata, dict):
            lock_hash = metadata.get("hash")
            if isinstance(lock_hash, dict) and isinstance(lock_hash.get("sha256"), str):
                lock_hash["sha256"] = _calculate_pipfile_hash(pipfile_text)
    return json.dumps(updated_document, indent=4) + "\n"


def _calculate_pipfile_hash(pipfile_text: str) -> str:
    """Calculate Pipenv's stable hash for a restored Pipfile document."""
    parsed = tomlkit.parse(pipfile_text)
    reserved_sections = {
        "source",
        "packages",
        "dev-packages",
        "requires",
        "scripts",
        "pipfile",
        "pipenv",
        "default",
        "develop",
    }

    def normalized_section(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("Pipfile dependency categories must be tables")
        return {normalize_python_package_name(str(name)): item for name, item in value.items()}

    hash_data: dict[str, Any] = {
        "_meta": {
            "sources": parsed.get("source", {}),
            "requires": parsed.get("requires", {}),
        },
        "default": normalized_section(parsed.get("packages", {})),
        "develop": normalized_section(parsed.get("dev-packages", {})),
    }
    for category, value in parsed.items():
        if category in reserved_sections:
            continue
        hash_data[str(category)] = normalized_section(value)
    content = json.dumps(hash_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _python_validation_command(manifest_path: str) -> str:
    workspace_dir = _workspace_dir_for_manifest(manifest_path)
    if _REQUIREMENTS_FILENAME.fullmatch(Path(manifest_path).name):
        install_args = f"-r {shlex.quote('/workspace/' + manifest_path)}"
    else:
        install_args = "-e ."
    command = f"/workspace/.venv/bin/python -m pip install --dry-run {install_args}"
    return f"cd {shlex.quote(workspace_dir)} && {command}"


def _python_requirements_validation_command(manifest_paths: Sequence[str]) -> str:
    """Resolve all same-environment requirements files in one pip transaction."""
    normalized = sorted({_normalize_python_manifest_targets([path])[0] for path in manifest_paths})
    if not normalized or any(
        not _REQUIREMENTS_FILENAME.fullmatch(Path(path).name) for path in normalized
    ):
        raise ValueError("combined Python pip validation requires requirements*.txt files")
    parents = {Path(path).parent.as_posix() for path in normalized}
    if len(parents) != 1:
        raise ValueError(
            "requirements files from distinct workspace directories need separate validation"
        )
    workspace_dir = _workspace_dir_for_manifest(normalized[0])
    install_args = " ".join(f"-r {shlex.quote('/workspace/' + path)}" for path in normalized)
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


def apply_multi_package_python_action(
    sandbox: DockerSandbox,
    action: Any,
    touched_files: set[str],
    checkpoint_store: dict[str, _PackageCheckpoint] | None = None,
) -> tuple[bool, str]:
    """Apply one committed Python batch as a single manifest transaction.

    Every manifest and adjacent Pipfile lock is snapshotted before the first
    edit. All authorized declarations are edited before package-manager
    validation begins. A failure restores the union checkpoint and resyncs
    previously locked Pipfile environments before returning control.

    Args:
        sandbox: Running Docker workspace sandbox.
        action: Immutable Supervisor-committed multi-package action.
        touched_files: Mutable changed-file projection for the worker result.
        checkpoint_store: Optional store retaining the successful union
            checkpoint until the caller accepts or rolls back the worker result.

    Returns:
        ``(True, "")`` when every declaration and resolver check succeeds;
        otherwise ``(False, diagnostic)`` after best-effort restoration.
    """
    from .tools_manifest import _capture_package_checkpoint

    mutations = list(getattr(action, "package_mutations", ()) or ())
    if not mutations:
        return False, "Python multi-package action contains no mutations."

    try:
        normalized_mutations: list[tuple[Any, str, str, str]] = []
        manifest_paths: set[str] = set()
        for mutation in mutations:
            if getattr(mutation, "ecosystem", None) != "pypi":
                raise ValueError("Python batch contains a mutation from another ecosystem.")
            package = _canonical_package(str(getattr(mutation, "package_name", "")))
            version = _canonical_version(str(getattr(mutation, "target_version", "")))
            dependency = str(getattr(mutation, "dependency_type", "") or "").strip()
            if package is None:
                raise ValueError("Python batch package names must be canonical PEP 503 names.")
            if version is None:
                raise ValueError("Python batch versions must be canonical PEP 440 versions.")
            if dependency not in _SUPPORTED_DECLARATIONS:
                raise ValueError(f"Unsupported Python declaration type: {dependency!r}.")
            manifest = _normalize_python_manifest_targets(
                [str(getattr(mutation, "manifest_path", ""))]
            )[0]
            normalized_mutations.append((mutation, package, version, manifest))
            manifest_paths.add(manifest)

        checkpoint = _capture_package_checkpoint(
            sandbox,
            sorted(manifest_paths),
            touched_files,
            package_ecosystem="pypi",
        )
        checkpoint_key = "__multi_package_action__"
        if checkpoint_store is not None:
            checkpoint_store[checkpoint_key] = checkpoint

        for manifest in sorted(manifest_paths):
            if Path(manifest).name.casefold() == "pipfile":
                lock_path = _validate_workspace_path(
                    (Path(manifest).parent / "Pipfile.lock").as_posix()
                )
                if not isinstance(checkpoint.files.get(lock_path), str):
                    raise ValueError(
                        f"Pipfile batch requires an existing lockfile for restorable sync: {lock_path}."
                    )

        for mutation, package, version, manifest in sorted(
            normalized_mutations,
            key=lambda item: (item[3], item[1], str(getattr(item[0], "task_id", ""))),
        ):
            dependency = str(mutation.dependency_type)
            current = sandbox.read_file(manifest)
            if not isinstance(current, str):
                raise ValueError(
                    f"Authorized Python manifest is missing or unreadable: {manifest}."
                )
            matches = [
                item
                for item in _matching_declarations(manifest, current, package)
                if item[0] == dependency
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one authorized {dependency} declaration for {package} in {manifest}."
                )
            edited = _update_python_manifest(manifest, current, package, version, dependency)
            sandbox.write_file(manifest, edited)
            touched_files.add(manifest)

        for manifest in sorted(manifest_paths):
            if Path(manifest).name.casefold() != "pipfile":
                continue
            result = sandbox.run(
                _pipenv_command(manifest, "lock"),
                timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Pipenv lock failed for {manifest} (exit {result.exit_code}). "
                    f"stdout: {_bounded_command_output(result.stdout)}; "
                    f"stderr: {_bounded_command_output(result.stderr)}"
                )
            result = sandbox.run(
                _pipenv_command(manifest, "sync --dev"),
                timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"Pipenv sync failed for {manifest} (exit {result.exit_code}). "
                    f"stdout: {_bounded_command_output(result.stdout)}; "
                    f"stderr: {_bounded_command_output(result.stderr)}"
                )
            lock_path = _validate_workspace_path(
                (Path(manifest).parent / "Pipfile.lock").as_posix()
            )
            lock_content = sandbox.read_file(lock_path)
            if not isinstance(lock_content, str):
                raise RuntimeError(f"Pipenv sync did not preserve {lock_path}.")
            lock = json.loads(lock_content)
            if not isinstance(lock, Mapping):
                raise RuntimeError(f"Pipfile lock must be a JSON object: {lock_path}.")
            for _, package, version, mutation_manifest in normalized_mutations:
                if mutation_manifest != manifest:
                    continue
                locked_values = []
                for section_name in ("default", "develop"):
                    section = lock.get(section_name)
                    if not isinstance(section, Mapping):
                        continue
                    for raw_name, record in section.items():
                        if normalize_python_package_name(str(raw_name)) != package:
                            continue
                        raw_locked = (
                            record.get("version") if isinstance(record, Mapping) else record
                        )
                        locked_values.append(str(raw_locked or ""))
                exact_lock = False
                for raw_locked in locked_values:
                    try:
                        locked_version = Version(raw_locked.lstrip("=<>!~ "))
                    except InvalidVersion:
                        continue
                    exact_lock = exact_lock or locked_version == Version(version)
                if not exact_lock:
                    raise RuntimeError(
                        f"Pipfile lock verification failed for {package}: "
                        f"expected {version}, found {locked_values or 'no package entry'}."
                    )

        requirements_by_parent: dict[str, list[str]] = {}
        for manifest in manifest_paths:
            if _REQUIREMENTS_FILENAME.fullmatch(Path(manifest).name):
                requirements_by_parent.setdefault(Path(manifest).parent.as_posix(), []).append(
                    manifest
                )
        for manifests in requirements_by_parent.values():
            result = sandbox.run(
                _python_requirements_validation_command(manifests),
                timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
            )
            if result.exit_code != 0:
                joined_manifests = ", ".join(sorted(manifests))
                raise RuntimeError(
                    f"Combined pip validation failed for {joined_manifests} "
                    f"(exit {result.exit_code}). "
                    f"stdout: {_bounded_command_output(result.stdout)}; "
                    f"stderr: {_bounded_command_output(result.stderr)}"
                )

        for mutation, package, version, manifest in normalized_mutations:
            current = sandbox.read_file(manifest)
            if not isinstance(current, str):
                raise RuntimeError(f"Python manifest disappeared after validation: {manifest}.")
            matches = [
                item
                for item in _matching_declarations(manifest, current, package)
                if item[0] == str(mutation.dependency_type)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Python manifest verification failed for {package} in {manifest}."
                )
            requirement = matches[0][1]
            specifiers = list(requirement.specifier)
            if (
                len(specifiers) != 1
                or specifiers[0].operator != "=="
                or Version(str(specifiers[0].version)) != Version(version)
            ):
                raise RuntimeError(
                    f"Python manifest verification failed for {package} in {manifest}: "
                    f"expected exact version {version}."
                )

        for path, before in checkpoint.files.items():
            after_value = sandbox.read_file(path)
            after = after_value if isinstance(after_value, str) else None
            if after != before:
                touched_files.add(path)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - atomic boundary owns restoration
        checkpoint = locals().get("checkpoint")
        if checkpoint is None:
            return False, f"Python multi-package action was rejected before mutation: {exc}."
        rollback_errors: list[str] = []
        rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
        if rollback_error:
            rollback_errors.append(rollback_error)
        if not rollback_errors:
            for manifest in sorted(locals().get("manifest_paths", ())):
                if Path(manifest).name.casefold() != "pipfile":
                    continue
                lock_path = _validate_workspace_path(
                    (Path(manifest).parent / "Pipfile.lock").as_posix()
                )
                if not isinstance(checkpoint.files.get(lock_path), str):
                    continue
                try:
                    result = sandbox.run(
                        _pipenv_command(manifest, "sync --dev"),
                        timeout=_MANIFEST_SYNC_TIMEOUT_SECONDS,
                    )
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"Pipenv rollback sync failed: {rollback_exc}")
                else:
                    if result.exit_code != 0:
                        rollback_errors.append(
                            "Pipenv rollback sync failed "
                            f"(exit {result.exit_code}): {_bounded_command_output(result.stderr)}"
                        )
        checkpoint_store = checkpoint_store
        if checkpoint_store is not None and not rollback_errors:
            checkpoint_store.pop("__multi_package_action__", None)
        suffix = (
            " Rollback failed: " + " | ".join(rollback_errors)
            if rollback_errors
            else " Rollback: union package checkpoint restored."
        )
        return False, f"Python multi-package action failed: {exc}.{suffix}"


def _make_modify_batch_python_dependencies_tool(
    sandbox: DockerSandbox,
    action: Any,
    touched_files: set[str],
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
) -> Any:
    """Build the single argument-free tool for one committed Python batch."""
    state = execution_state if execution_state is not None else {}

    @tool
    def modify_batch_python_dependencies() -> str:
        """Apply and validate the complete Supervisor-committed Python batch."""
        if state.get("multi_package_action_executed"):
            return _tool_error(
                "MULTI_PACKAGE_ACTION_ALREADY_EXECUTED",
                "The committed cluster action has already been executed; return control to the Supervisor.",
            )
        state["multi_package_action_executed"] = True
        try:
            succeeded, error = apply_multi_package_python_action(
                sandbox,
                action,
                touched_files,
                checkpoint_store=package_checkpoints,
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary returns typed failure
            succeeded, error = False, f"Python batch action raised an unexpected error: {exc}"
        state["multi_package_action_succeeded"] = succeeded
        state["multi_package_action_error"] = error or None
        if succeeded:
            label = getattr(action, "cluster_id", None) or "unclustered"
            return (
                "SUCCESS: Applied the Supervisor-committed Python multi-package action "
                f"for cluster {label}; every declaration and resolver check passed."
            )
        return _tool_error(
            "MULTI_PACKAGE_ACTION_FAILED",
            error or "The committed Python batch failed and was rolled back.",
        )

    return modify_batch_python_dependencies


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
