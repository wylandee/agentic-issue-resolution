"""Safe localization and narrow requirements.txt edits for Python dependencies."""

from __future__ import annotations

import configparser
import json
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit
from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version

from remediation_engine.contracts.schemas import LocalizedIssue, VulnerabilityIssue
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    repository_relative_path,
    resolve_repository_path,
)
from remediation_engine.tools.package_identity import (
    normalize_python_package_name,
    package_name_from_purl,
)

_REQUIREMENTS_FILE = re.compile(r"requirements(?:[-_.][^/\\]+)?\.txt$", re.IGNORECASE)
_EXCLUDED_DIRS = {
    ".git",
    ".pytest_cache",
    ".remedy-pipenv",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}
_HASH_OPTION = re.compile(r"\s+--hash(?:=|\s)", re.IGNORECASE)
_SUPPORTED_DECLARATIONS = {
    "requirements",
    "dependencies",
    "optional-dependencies",
    "install_requires",
    "extras_require",
    "packages",
    "dev-packages",
}


@dataclass(frozen=True)
class RequirementEntry:
    """A parsed requirement with its original source location and line."""

    path: Path
    line_number: int
    name: str
    requirement: Requirement
    raw_line: str


def _normalized_name(name: str) -> str:
    return normalize_python_package_name(name.strip())


def _requirement_from_line(line: str) -> Requirement | None:
    """Parse a single requirement line, ignoring pip options and comments."""
    value = line.strip()
    if not value or value.startswith("#") or value.startswith("-"):
        return None
    if value.endswith("\\"):
        value = value[:-1].rstrip()
    value = re.split(r"\s+#", value, maxsplit=1)[0].strip()
    value = _HASH_OPTION.split(value, maxsplit=1)[0].strip()
    if not value:
        return None
    try:
        return Requirement(value)
    except InvalidRequirement:
        return None


def parse_requirements_txt(path: Path) -> list[RequirementEntry]:
    """Parse standalone requirement entries while preserving original lines.

    Options, comments, blank lines, and unsupported continuation lines are not
    dependency declarations. Hash-pinned requirements remain discoverable, but
    the update helper refuses to rewrite them.
    """
    source = Path(path)
    entries: list[RequirementEntry] = []
    skipping_continuation = False
    for line_number, raw_with_ending in enumerate(
        source.read_text(encoding="utf-8").splitlines(keepends=True), start=1
    ):
        raw_line = raw_with_ending.rstrip("\r\n")
        stripped = raw_line.rstrip()
        if skipping_continuation:
            skipping_continuation = stripped.endswith("\\")
            continue
        requirement = _requirement_from_line(raw_line)
        if requirement is not None:
            entries.append(
                RequirementEntry(
                    path=source,
                    line_number=line_number,
                    name=requirement.name,
                    requirement=requirement,
                    raw_line=raw_line,
                )
            )
        if stripped.endswith("\\"):
            skipping_continuation = True
    return entries


def find_dependency_in_requirements(name: str, path: Path) -> RequirementEntry | None:
    """Find one unambiguous PEP 503-equivalent requirement declaration."""
    target = _normalized_name(name)
    if not target:
        return None
    matches = [
        entry for entry in parse_requirements_txt(path) if _normalized_name(entry.name) == target
    ]
    return matches[0] if len(matches) == 1 else None


def update_requirements_dependency(path: Path, package_name: str, new_version: str) -> str:
    """Pin one simple requirements.txt declaration, preserving line formatting."""
    target = _normalized_name(package_name)
    if not target:
        raise ValueError("package_name must be a non-empty Python distribution name")
    try:
        Version(new_version)
    except InvalidVersion as exc:
        raise ValueError(f"invalid Python package version: {new_version!r}") from exc

    source = Path(path)
    original = source.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    matches = [
        entry for entry in parse_requirements_txt(source) if _normalized_name(entry.name) == target
    ]
    if len(matches) != 1:
        raise ValueError("expected exactly one matching requirements declaration")

    entry = matches[0]
    raw_line = entry.raw_line
    requirement = entry.requirement
    if (
        _HASH_OPTION.search(raw_line)
        or requirement.url is not None
        or requirement.marker is not None
        or requirement.extras
        or len(list(requirement.specifier)) > 1
    ):
        raise ValueError("unsupported or hash-pinned requirements declaration")

    ending = "\r\n" if lines[entry.line_number - 1].endswith("\r\n") else "\n"
    if not lines[entry.line_number - 1].endswith(("\n", "\r")):
        ending = ""
    body = raw_line
    comment_start = re.search(r"\s+#", body)
    comment = body[comment_start.start() :] if comment_start else ""
    declaration = body[: comment_start.start()] if comment_start else body
    match = re.fullmatch(
        r"(?P<indent>\s*)(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
        r"(?P<separator>\s*)(?:(?P<operator>===|==|~=|!=|<=|>=|<|>)"
        r"(?P<operator_space>\s*)(?P<version>[^\s;#]+))?(?P<trailing>\s*)",
        declaration,
    )
    if match is None or _normalized_name(match.group("name")) != target:
        raise ValueError("unsupported requirements declaration syntax")

    replacement = (
        f"{match.group('indent')}{match.group('name')}{match.group('separator')}"
        f"=={match.group('operator_space') or ''}{new_version}{match.group('trailing')}{comment}{ending}"
    )
    lines[entry.line_number - 1] = replacement
    return "".join(lines)


def _read_toml(path: Path) -> Any | None:
    try:
        return tomlkit.parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, tomlkit.exceptions.ParseError):
        return None


def _toml_line(text: str, value: str, cursor: int) -> tuple[int | None, int]:
    index = text.find(value, cursor)
    if index < 0:
        index = text.find(value)
    if index < 0:
        return None, cursor
    return text.count("\n", 0, index) + 1, index + len(value)


def _pyproject_entries(path: Path) -> list[tuple[Requirement, str, int | None, str]]:
    document = _read_toml(path)
    if not isinstance(document, Mapping):
        return []
    project = document.get("project")
    if not isinstance(project, Mapping):
        return []
    text = path.read_text(encoding="utf-8")
    entries: list[tuple[Requirement, str, int | None, str]] = []
    cursor = 0
    fields: list[tuple[str, Any]] = [("dependencies", project.get("dependencies"))]
    optional = project.get("optional-dependencies")
    if isinstance(optional, Mapping):
        fields.extend(("optional-dependencies", value) for value in optional.values())
    for declaration_type, values in fields:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for raw_value in values:
            if not isinstance(raw_value, str):
                continue
            try:
                requirement = Requirement(raw_value)
            except InvalidRequirement:
                continue
            line_number, cursor = _toml_line(text, raw_value, cursor)
            entries.append((requirement, declaration_type, line_number, raw_value))
    return entries


def _setup_cfg_entries(path: Path) -> list[tuple[Requirement, str, int, str]]:
    text = path.read_text(encoding="utf-8")
    parser = configparser.RawConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error:
        return []
    lines = text.splitlines()
    section = ""
    option: str | None = None
    entries: list[tuple[Requirement, str, int, str]] = []

    def add(raw_value: str, kind: str, line_number: int) -> None:
        requirement = _requirement_from_line(raw_value)
        if requirement is not None:
            entries.append((requirement, kind, line_number, raw_value.strip()))

    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^]]+)\]\s*(?:[#;].*)?$", line)
        if header:
            section = header.group(1).strip().casefold()
            option = None
            continue
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        assignment = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*=(?!=)\s*(.*?)\s*$", line)
        if assignment:
            key = assignment.group(1).casefold()
            value = assignment.group(2)
            option = None
            if section == "options" and key == "install_requires":
                option = "install_requires"
            elif section == "options.extras_require":
                option = "extras_require"
            if option and value:
                add(value, option, index + 1)
            continue
        if option and line[:1].isspace():
            add(line.strip(), option, index + 1)
        else:
            option = None
    return entries


def _pipfile_sections(path: Path) -> dict[str, dict[str, Any]]:
    document = _read_toml(path)
    if not isinstance(document, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for section, declaration_type in (("packages", "packages"), ("dev-packages", "dev-packages")):
        values = document.get(section)
        if isinstance(values, Mapping):
            result[declaration_type] = dict(values)
    return result


def _pipfile_line(path: Path, package_name: str, declaration_type: str) -> int | None:
    section_name = "packages" if declaration_type == "packages" else "dev-packages"
    escaped = re.escape(package_name)
    section = ""
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        header = re.match(r"^\s*\[([^]]+)\]", line)
        if header:
            section = header.group(1).strip().casefold()
            continue
        if section == section_name and re.match(
            rf"^\s*(?:['\"]{escaped}['\"]|{escaped})\s*=", line, re.IGNORECASE
        ):
            return line_number
    return None


def _pipfile_lock(lock_path: Path) -> dict[str, dict[str, str | None]]:
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, Mapping):
        return {}
    result: dict[str, dict[str, str | None]] = {}
    for category in ("default", "develop"):
        packages = data.get(category)
        if not isinstance(packages, Mapping):
            continue
        versions: dict[str, str | None] = {}
        for name, metadata in packages.items():
            if not isinstance(name, str):
                continue
            normalized_name = _normalized_name(name)
            if not normalized_name:
                continue
            version: str | None = None
            value = metadata.get("version") if isinstance(metadata, Mapping) else None
            if isinstance(value, str) and value.startswith("=="):
                with suppress(InvalidVersion):
                    version = str(Version(value[2:]))
            versions[normalized_name] = version
        result[category] = versions
    return result


def _lock_category_for_issue(
    issue: VulnerabilityIssue, lock_data: Mapping[str, Mapping[str, str | None]], name: str
) -> str | None:
    raw_path = issue.file_path or str((issue.raw_payload or {}).get("filePath", ""))
    hint = re.search(r"(?:^|[/?!#])(?:default|develop)(?:[/?!#]|$)", raw_path, re.IGNORECASE)
    if hint:
        category = hint.group(0).strip("/?!#").casefold()
        return category if name in lock_data.get(category, {}) else None
    categories = [category for category, packages in lock_data.items() if name in packages]
    return categories[0] if len(categories) == 1 else None


def _snippet(path: Path, line_number: int | None, context: int = 1) -> str | None:
    if line_number is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    start = max(1, line_number - context)
    end = min(len(lines), line_number + context)
    return "\n".join(lines[start - 1 : end])


def _supported_manifest(path: Path) -> bool:
    name = path.name.casefold()
    if _REQUIREMENTS_FILE.fullmatch(name):
        return True
    if name == "pyproject.toml":
        document = _read_toml(path)
        project = document.get("project") if isinstance(document, Mapping) else None
        if not isinstance(project, Mapping):
            return False
        dependencies = project.get("dependencies")
        optional = project.get("optional-dependencies")
        return (
            isinstance(dependencies, Sequence) and not isinstance(dependencies, (str, bytes))
        ) or isinstance(optional, Mapping)
    if name == "setup.cfg":
        try:
            parser = configparser.RawConfigParser()
            parser.read_string(path.read_text(encoding="utf-8"))
        except (OSError, configparser.Error):
            return False
        return parser.has_option("options", "install_requires") or parser.has_section(
            "options.extras_require"
        )
    if name == "pipfile":
        return bool(_pipfile_sections(path))
    return False


def locate_python_manifests(project_dir: Path) -> list[Path]:
    """Return supported static Python manifests beneath a project root."""
    root = Path(project_dir).resolve()
    if not root.is_dir():
        return []
    manifests: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if name.casefold() not in _EXCLUDED_DIRS and not (current_path / name).is_symlink()
        ]
        for filename in filenames:
            candidate = current_path / filename
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                safe_path = resolve_repository_path(root, candidate.relative_to(root))
            except (OSError, WorkspacePathError):
                continue
            if _supported_manifest(safe_path):
                manifests.append(safe_path)
    return sorted(manifests, key=lambda item: item.relative_to(root).as_posix().casefold())


def _issue_name(issue: VulnerabilityIssue) -> str:
    raw_name = issue.package_name
    if not raw_name and issue.purl:
        raw_name = package_name_from_purl(issue.purl)
    return _normalized_name(raw_name or "")


def _is_pypi_issue(issue: VulnerabilityIssue) -> bool:
    ecosystem = (issue.ecosystem or "").strip().casefold()
    purl_type = (issue.purl or "").partition(":")[2].partition("/")[0].casefold()
    return ecosystem in {"pypi", "python"} or purl_type == "pypi"


def _issue_path(issue: VulnerabilityIssue) -> str:
    return str((issue.raw_payload or {}).get("filePath") or issue.file_path or "").strip()


def _start_directory(repo_root: Path, raw_path: str) -> Path | None:
    if not raw_path:
        return repo_root
    candidate_text = raw_path.split("?", 1)[0].split("#", 1)[0].strip().replace("\\", "/")
    if candidate_text == "/scan":
        return repo_root
    if candidate_text.startswith("/scan/"):
        # Dependency-Check appends ``:distribution/version`` to files under its
        # conventional container mount; map the remaining path to the checkout.
        relative = candidate_text[len("/scan/") :].split(":", 1)[0]
    else:
        relative = repository_relative_path(candidate_text, repo_root)
    if candidate_text.startswith("/src/"):
        # Dependency-Check's conventional scan root maps to the repository root.
        relative = candidate_text[len("/src/") :]
    elif candidate_text.startswith("/workspace/"):
        relative = candidate_text[len("/workspace/") :]
    elif relative == "src":
        return repo_root
    elif relative and relative.startswith("src/"):
        # Some ODC paths have already had their leading slash stripped by the
        # VulnerabilityIssue validator. Preserve a real repo-relative src path;
        # otherwise interpret it as the conventional ODC scan-root prefix.
        try:
            source_path = resolve_repository_path(repo_root, relative)
        except (OSError, WorkspacePathError):
            return None
        if not source_path.exists():
            relative = relative[4:]
    if relative is None:
        return None
    try:
        candidate = resolve_repository_path(repo_root, relative)
    except (OSError, WorkspacePathError):
        return None
    if candidate.is_dir():
        return candidate
    if candidate.exists():
        return candidate.parent
    # ODC may report an artifact path that is absent from the checkout. Walk
    # up only within the already validated repository-relative path.
    current = candidate.parent
    while current != repo_root and not current.exists():
        current = current.parent
    return current if current.is_dir() else repo_root


def _manifest_candidates_by_directory(manifests: list[Path]) -> dict[Path, list[Path]]:
    by_directory: dict[Path, list[Path]] = {}
    for path in manifests:
        by_directory.setdefault(path.parent, []).append(path)
    priority = {"pipfile": 0, "pyproject.toml": 1, "setup.cfg": 2}
    for paths in by_directory.values():
        paths.sort(key=lambda path: (priority.get(path.name.casefold(), 3), path.name.casefold()))
    return by_directory


def _find_manifest_match(
    issue: VulnerabilityIssue, repo_root: Path, target_name: str
) -> tuple[Path | None, bool | None, str | None, int | None, str | None, dict[str, str]]:
    manifests = locate_python_manifests(repo_root)
    by_directory = _manifest_candidates_by_directory(manifests)
    start = _start_directory(repo_root, _issue_path(issue))
    if start is None:
        return None, None, None, None, None, {}
    current = start
    while True:
        candidates = by_directory.get(current, [])
        pipfile = next((path for path in candidates if path.name.casefold() == "pipfile"), None)
        # Pipenv is authoritative when it coexists with other declarations.
        if pipfile is not None:
            sections = _pipfile_sections(pipfile)
            declarations = [
                (section, package)
                for section, packages in sections.items()
                for package in packages
                if _normalized_name(package) == target_name
            ]
            if len(declarations) > 1:
                return pipfile, None, None, None, None, {}
            if len(declarations) == 1:
                section, package = declarations[0]
                line_number = _pipfile_line(pipfile, str(package), section)
                lock = pipfile.with_name("Pipfile.lock")
                lock_data = _pipfile_lock(lock) if lock.is_file() else {}
                lock_category = "default" if section == "packages" else "develop"
                version = lock_data.get(lock_category, {}).get(target_name)
                versions = {target_name: version} if version else {}
                return pipfile, True, section, line_number, _snippet(pipfile, line_number), versions

            lock = pipfile.with_name("Pipfile.lock")
            lock_data = _pipfile_lock(lock) if lock.is_file() else {}
            category = _lock_category_for_issue(issue, lock_data, target_name)
            if category is not None:
                version = lock_data.get(category, {}).get(target_name)
                declaration_type = "packages" if category == "default" else "dev-packages"
                versions = {target_name: version} if version else {}
                return pipfile, False, declaration_type, None, None, versions
            return pipfile, None, None, None, None, {}

        for manifest in candidates:
            if _REQUIREMENTS_FILE.fullmatch(manifest.name):
                entries = parse_requirements_txt(manifest)
                matches = [
                    entry for entry in entries if _normalized_name(entry.name) == target_name
                ]
                if len(matches) == 1:
                    entry = matches[0]
                    return (
                        manifest,
                        True,
                        "requirements",
                        entry.line_number,
                        _snippet(manifest, entry.line_number),
                        {},
                    )
                if len(matches) > 1:
                    return manifest, None, None, None, None, {}
            elif manifest.name.casefold() == "pyproject.toml":
                matches = [
                    entry
                    for entry in _pyproject_entries(manifest)
                    if _normalized_name(entry[0].name) == target_name
                ]
                if len(matches) > 1:
                    return manifest, None, None, None, None, {}
                if len(matches) == 1:
                    _, declaration_type, line_number, _ = matches[0]
                    return (
                        manifest,
                        True,
                        declaration_type,
                        line_number,
                        _snippet(manifest, line_number),
                        {},
                    )
            elif manifest.name.casefold() == "setup.cfg":
                matches = [
                    entry
                    for entry in _setup_cfg_entries(manifest)
                    if _normalized_name(entry[0].name) == target_name
                ]
                if len(matches) > 1:
                    return manifest, None, None, None, None, {}
                if len(matches) == 1:
                    _, declaration_type, line_number, _ = matches[0]
                    return (
                        manifest,
                        True,
                        declaration_type,
                        line_number,
                        _snippet(manifest, line_number),
                        {},
                    )
        if candidates:
            # A nearer Python project boundary is authoritative even when it
            # does not declare this package; do not borrow an ancestor pin.
            return candidates[0], None, None, None, None, {}
        parent = current.parent
        if parent == current or not parent.is_relative_to(repo_root):
            break
        current = parent
    return None, None, None, None, None, {}


def locate_from_issue(issue: VulnerabilityIssue, repo_path: Path) -> LocalizedIssue:
    """Localize a PyPI issue to its nearest supported Python declaration."""
    repo_root = Path(repo_path).resolve()
    is_python = _is_pypi_issue(issue)
    target_name = _issue_name(issue) if is_python else ""
    canonical_issue = issue
    if target_name and issue.package_name != target_name:
        canonical_issue = issue.model_copy(update={"package_name": target_name})

    manifest, direct, declaration_type, line_number, snippet, versions = (
        _find_manifest_match(canonical_issue, repo_root, target_name)
        if target_name
        else (None, None, None, None, None, {})
    )
    manifest_relative: str | None = None
    manager: str | None = None
    if manifest is not None:
        try:
            manifest_relative = manifest.resolve().relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            manifest = None
            direct = None
            declaration_type = None
            line_number = None
            snippet = None
            versions = {}
        else:
            manager = "pipenv" if manifest.name.casefold() == "pipfile" else "pip"

    confidence = 0.0
    if manifest is not None and direct is True:
        confidence = 0.95 if line_number is not None else 0.75
    elif manifest is not None and direct is False:
        confidence = 0.60
    return LocalizedIssue(
        issue=canonical_issue,
        manifest_file=manifest_relative,
        is_direct_dependency=direct,
        manifest_line=line_number,
        manifest_snippet=snippet,
        package_manager=manager,
        dependency_ancestry=[],
        dependency_versions=versions,
        declaration_type=declaration_type if declaration_type in _SUPPORTED_DECLARATIONS else None,
        localization_confidence=confidence,
    )
