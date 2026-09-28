"""Safe localization of Maven findings to one in-repository POM."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
import xml.parsers.expat as expat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from xml.sax.saxutils import escape as _xml_escape

from remediation_engine.contracts import LocalizedIssue, VulnerabilityIssue
from remediation_engine.contracts.version_policy import compare_maven_versions
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    repository_relative_path,
    resolve_repository_path,
)
from remediation_engine.tools.package_identity import package_name_from_purl


@dataclass(frozen=True)
class MavenDependencyInfo:
    """One unambiguous Maven dependency declaration."""

    group_id: str
    artifact_id: str
    manifest_file: str
    dependency_type: str
    is_direct: bool
    declared_version: str | None
    property_name: str | None
    property_file: str | None
    line_number: int | None


@dataclass(frozen=True)
class MavenManifest:
    """Parsed project coordinates, reactor relationships, and declarations."""

    path: Path
    group_id: str | None
    artifact_id: str | None
    parent_path: Path | None
    modules: tuple[str, ...]
    dependencies: tuple[MavenDependencyInfo, ...]
    dependency_management: tuple[MavenDependencyInfo, ...]
    properties: Mapping[str, str]


class MavenManifestError(ValueError):
    """Raised when a POM cannot be interpreted without ambiguity."""


_DTD_OR_ENTITY = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)


def _has_dtd_or_entity(payload: bytes) -> bool:
    if _DTD_OR_ENTITY.search(payload):
        return True
    encodings = ["utf-8-sig"]
    if payload.startswith((b"\x00\x00\xfe\xff", b"\xff\xfe\x00\x00")):
        encodings.insert(0, "utf-32")
    elif payload.startswith(b"\x00\x00\x00<"):
        encodings.insert(0, "utf-32-be")
    elif payload.startswith(b"<\x00\x00\x00"):
        encodings.insert(0, "utf-32-le")
    elif payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(0, "utf-16")
    elif payload.startswith(b"\x00<\x00?"):
        encodings.insert(0, "utf-16-be")
    elif payload.startswith(b"<\x00?\x00"):
        encodings.insert(0, "utf-16-le")
    for encoding in encodings:
        try:
            decoded = payload.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", decoded, re.IGNORECASE):
            return True
    return False


_PROPERTY_REFERENCE = re.compile(r"\$\{([^{}]+)\}")
_EXCLUDED_DIRS = {".git", "node_modules", "target"}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    return [child for child in list(element) if _local_name(child.tag) == name]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    matches = _children(element, name)
    return matches[0] if matches else None


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    value = element.text.strip()
    return value or None


def _parse_xml(path: Path) -> tuple[bytes, ET.Element]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise MavenManifestError(f"cannot read POM {path}") from exc
    if _has_dtd_or_entity(payload):
        raise MavenManifestError(f"DTD or entity declaration is forbidden in {path}")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MavenManifestError(f"malformed Maven POM: {path}") from exc
    if _local_name(root.tag) != "project":
        raise MavenManifestError(f"not a Maven project POM: {path}")
    return payload, root


def _raw_properties(root: ET.Element) -> dict[str, str]:
    props = _child(root, "properties")
    if props is None:
        return {}
    result: dict[str, str] = {}
    for element in list(props):
        name = _local_name(element.tag)
        value = _text(element)
        if value is not None:
            result[name] = value
    return result


def _resolve(value: str | None, values: Mapping[str, str]) -> str | None:
    if value is None:
        return None
    result = value.strip()
    for _ in range(20):
        match = _PROPERTY_REFERENCE.search(result)
        if match is None:
            return result
        replacement = values.get(match.group(1))
        if replacement is None:
            return None
        result = result[: match.start()] + replacement + result[match.end() :]
    return None


def _repo_parent_path(root: ET.Element, pom_path: Path, repo_root: Path) -> Path | None:
    parent = _child(root, "parent")
    if parent is None:
        return None
    relative = _child(parent, "relativePath")
    rel_value = _text(relative)
    if relative is not None and rel_value is None:
        return None
    rel_value = rel_value or "../pom.xml"
    candidate = Path(rel_value)
    if candidate.is_absolute():
        raise MavenManifestError("absolute Maven parent relativePath is forbidden")
    candidate_path = (pom_path.parent / candidate).resolve()
    try:
        candidate_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise MavenManifestError("Maven parent POM resolves outside the repository") from exc
    parent_rel = repository_relative_path(candidate_path, repo_root)
    if parent_rel is None:
        raise MavenManifestError("Maven parent POM resolves outside the repository")
    if any(part in _EXCLUDED_DIRS for part in Path(parent_rel).parts):
        raise MavenManifestError("Maven parent POM is in an excluded directory")
    try:
        parent_path = resolve_repository_path(repo_root, parent_rel)
    except WorkspacePathError as exc:
        raise MavenManifestError("Maven parent POM resolves outside the repository") from exc
    if parent_path.name != "pom.xml":
        raise MavenManifestError("Maven parent relativePath must identify a pom.xml")
    return parent_path if parent_path.is_file() else None


def _manifest_path_text(path: Path, repo_root: Path | None) -> str:
    if repo_root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise MavenManifestError("POM is outside the repository") from exc


def _dependency_entries(
    root: ET.Element,
    section: str,
    project_group: str | None,
    properties: Mapping[str, str],
    property_owners: Mapping[str, Path],
    pom_path: Path,
    repo_root: Path | None,
) -> tuple[MavenDependencyInfo, ...]:
    if section == "dependencies":
        containers = _children(root, "dependencies")
    else:
        containers = []
        for management in _children(root, "dependencyManagement"):
            containers.extend(_children(management, "dependencies"))
    result: list[MavenDependencyInfo] = []
    manifest_file = _manifest_path_text(pom_path, repo_root)
    source_lines = pom_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for container in containers:
        for dep in _children(container, "dependency"):
            raw_group = _text(_child(dep, "groupId"))
            raw_artifact = _text(_child(dep, "artifactId"))
            group = _resolve(raw_group or project_group, properties)
            artifact = _resolve(raw_artifact, properties)
            raw_version = _text(_child(dep, "version"))
            if group is None or artifact is None:
                raise MavenManifestError("unresolved Maven dependency groupId/artifactId")
            property_match = _PROPERTY_REFERENCE.fullmatch(raw_version or "")
            property_name = property_match.group(1) if property_match else None
            owner = property_owners.get(property_name) if property_name else None
            line_number = _artifact_line(source_lines, artifact)
            result.append(
                MavenDependencyInfo(
                    group_id=group,
                    artifact_id=artifact,
                    manifest_file=manifest_file,
                    dependency_type=section,
                    is_direct=section == "dependencies",
                    declared_version=raw_version,
                    property_name=property_name,
                    property_file=(
                        _manifest_path_text(owner, repo_root) if owner is not None else None
                    ),
                    line_number=line_number,
                )
            )
    return tuple(result)


def _artifact_line(lines: list[str], artifact_id: str) -> int | None:
    escaped = re.escape(artifact_id)
    pattern = re.compile(
        rf"<(?:(?:[A-Za-z_][\w.-]*):)?artifactId\s*>\s*{escaped}\s*</",
        re.IGNORECASE,
    )
    for number, line in enumerate(lines, start=1):
        if pattern.search(line):
            return number
    return None


def _profile_has_target(
    root: ET.Element,
    group_id: str,
    artifact_id: str,
    properties: Mapping[str, str] | None = None,
    project_group: str | None = None,
) -> bool:
    values = dict(properties or {})
    values.update(
        {
            "project.groupId": project_group or "",
            "pom.groupId": project_group or "",
        }
    )
    for profile in root.iter():
        if _local_name(profile.tag) != "profile":
            continue
        for dep in profile.iter():
            if _local_name(dep.tag) != "dependency":
                continue
            group = _resolve(_text(_child(dep, "groupId")) or project_group, values)
            artifact = _resolve(_text(_child(dep, "artifactId")), values)
            if group == group_id and artifact == artifact_id:
                return True
    return False


def _module_pom_path(pom_path: Path, module: str, repo_root: Path) -> Path:
    module_path = Path(module.strip())
    if not module.strip() or module_path.is_absolute():
        raise MavenManifestError("Maven module path is invalid")
    try:
        base_relative = pom_path.parent.resolve().relative_to(repo_root.resolve())
        module_relative = normalize_workspace_path(
            (base_relative / module_path).as_posix(),
            allow_workspace_prefix=False,
        )
        candidate = resolve_repository_path(repo_root, module_relative)
    except (ValueError, WorkspacePathError) as exc:
        raise MavenManifestError("Maven module path escapes the repository") from exc
    if candidate.name != "pom.xml":
        candidate = candidate / "pom.xml"
    if candidate == pom_path.resolve() or not candidate.is_file():
        raise MavenManifestError("Maven module does not identify an in-repository POM")
    return candidate


def _load_manifest(
    path: Path,
    repo_root: Path,
    cache: dict[Path, tuple[MavenManifest, ET.Element, dict[str, Path], str | None]],
    visiting: set[Path],
) -> tuple[MavenManifest, ET.Element, dict[str, Path], str | None]:
    resolved_path = path.resolve()
    if resolved_path in cache:
        return cache[resolved_path]
    if resolved_path in visiting:
        raise MavenManifestError("cyclic Maven parent POM chain")
    try:
        resolved_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise MavenManifestError("POM is outside the repository") from exc
    visiting.add(resolved_path)
    _payload, xml_root = _parse_xml(resolved_path)
    parent_path = _repo_parent_path(xml_root, resolved_path, repo_root)
    parent_manifest: MavenManifest | None = None
    parent_owners: dict[str, Path] = {}
    if parent_path is not None:
        parent_manifest, _parent_xml, parent_owners, _parent_version = _load_manifest(
            parent_path, repo_root, cache, visiting
        )

    own_properties = _raw_properties(xml_root)
    effective_properties: dict[str, str] = {}
    if parent_manifest is not None:
        effective_properties.update(parent_manifest.properties)
    effective_properties.update(own_properties)
    owners = dict(parent_owners)
    owners.update({name: resolved_path for name in own_properties})

    parent_group = parent_manifest.group_id if parent_manifest else None
    parent_version = _effective_version_from_cache(parent_path, cache) if parent_path else None
    parent_values = {
        **effective_properties,
        "project.parent.groupId": parent_group or "",
        "pom.parent.groupId": parent_group or "",
        "project.parent.artifactId": (parent_manifest.artifact_id if parent_manifest else "") or "",
        "pom.parent.artifactId": (parent_manifest.artifact_id if parent_manifest else "") or "",
        "project.parent.version": parent_version or "",
        "pom.parent.version": parent_version or "",
    }
    if parent_manifest is not None:
        parent_node = _child(xml_root, "parent")
        declared_parent_group = _resolve(
            _text(_child(parent_node, "groupId")) if parent_node is not None else None,
            parent_values,
        )
        declared_parent_artifact = _resolve(
            _text(_child(parent_node, "artifactId")) if parent_node is not None else None,
            parent_values,
        )
        declared_parent_version = _resolve(
            _text(_child(parent_node, "version")) if parent_node is not None else None,
            parent_values,
        )
        if (
            not declared_parent_group
            or not declared_parent_artifact
            or not declared_parent_version
            or parent_version is None
        ):
            raise MavenManifestError("unresolved Maven parent coordinates or version")
        if (
            declared_parent_group != parent_manifest.group_id
            or declared_parent_artifact != parent_manifest.artifact_id
        ):
            raise MavenManifestError("Maven parent coordinates are ambiguous")
        try:
            parent_version_matches = (
                compare_maven_versions(declared_parent_version, parent_version) == 0
            )
        except ValueError as exc:
            raise MavenManifestError("Maven parent version is ambiguous") from exc
        if not parent_version_matches:
            raise MavenManifestError("Maven parent coordinates are ambiguous")
    group_id = _resolve(_text(_child(xml_root, "groupId")) or parent_group, parent_values)
    artifact_id = _resolve(_text(_child(xml_root, "artifactId")), parent_values)
    project_version = _resolve(_text(_child(xml_root, "version")) or parent_version, parent_values)
    builtins = {
        **parent_values,
        "project.groupId": group_id or "",
        "pom.groupId": group_id or "",
        "project.artifactId": artifact_id or "",
        "pom.artifactId": artifact_id or "",
        "project.version": project_version or "",
        "pom.version": project_version or "",
    }
    group_id = _resolve(_text(_child(xml_root, "groupId")) or parent_group, builtins)
    artifact_id = _resolve(_text(_child(xml_root, "artifactId")), builtins)
    project_version = _resolve(_text(_child(xml_root, "version")) or parent_version, builtins)
    module_node = _child(xml_root, "modules")
    modules = tuple(
        value
        for module in (_children(module_node, "module") if module_node is not None else [])
        if (value := _text(module)) is not None
    )
    for module in modules:
        _module_pom_path(resolved_path, module, repo_root)
    dependencies = _dependency_entries(
        xml_root, "dependencies", group_id, builtins, owners, resolved_path, repo_root
    )
    management = _dependency_entries(
        xml_root, "dependencyManagement", group_id, builtins, owners, resolved_path, repo_root
    )
    manifest = MavenManifest(
        path=resolved_path,
        group_id=group_id,
        artifact_id=artifact_id,
        parent_path=parent_path,
        modules=modules,
        dependencies=dependencies,
        dependency_management=management,
        properties=MappingProxyType(dict(effective_properties)),
    )
    result = (manifest, xml_root, owners, project_version)
    cache[resolved_path] = result
    visiting.remove(resolved_path)
    return result


def _effective_version_from_cache(path: Path, cache: Mapping[Path, tuple]) -> str | None:
    item = cache.get(path.resolve())
    return item[3] if item is not None else None


def parse_pom_xml(path: Path) -> MavenManifest:
    """Parse a POM and its safe local parent chain using Maven local names."""
    pom_path = Path(path).resolve()
    boundary = _guess_repository_root(pom_path)
    cache: dict[Path, tuple[MavenManifest, ET.Element, dict[str, Path], str | None]] = {}
    return _load_manifest(pom_path, boundary, cache, set())[0]


def _guess_repository_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / ".git").exists():
            return parent.resolve()
    pom_roots = [
        parent for parent in (path.parent, *path.parents) if (parent / "pom.xml").is_file()
    ]
    return pom_roots[-1].resolve() if pom_roots else path.parent.resolve()


def locate_maven_manifests(project_dir: Path) -> list[Path]:
    """Return contained POM paths, pruning generated/vendor trees but not .mvn."""
    root = Path(project_dir).resolve()
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in _EXCLUDED_DIRS and not (Path(current) / name).is_symlink()
        )
        if "pom.xml" not in files:
            continue
        candidate = Path(current) / "pom.xml"
        if candidate.is_symlink():
            continue
        try:
            resolved = resolve_repository_path(root, candidate.relative_to(root))
        except (OSError, WorkspacePathError):
            continue
        if resolved.is_file():
            paths.append(resolved)
    return sorted(paths)


def _target_entries(pom: MavenManifest, group_id: str, artifact_id: str) -> tuple[list, list]:
    direct = [
        dep
        for dep in pom.dependencies
        if dep.group_id == group_id and dep.artifact_id == artifact_id
    ]
    managed = [
        dep
        for dep in pom.dependency_management
        if dep.group_id == group_id and dep.artifact_id == artifact_id
    ]
    return direct, managed


def find_dependency_in_pom(
    group_id: str,
    artifact_id: str,
    pom: MavenManifest,
) -> MavenDependencyInfo | None:
    """Find one exact, unambiguous direct or dependency-management GAV entry."""
    _payload, root = _parse_xml(pom.path)
    values = dict(pom.properties)
    values.update(
        {
            "project.groupId": pom.group_id or "",
            "pom.groupId": pom.group_id or "",
            "project.artifactId": pom.artifact_id or "",
            "pom.artifactId": pom.artifact_id or "",
        }
    )
    if _profile_has_target(root, group_id, artifact_id, values, pom.group_id):
        raise MavenManifestError("Maven profile declarations are ambiguous")
    direct, managed = _target_entries(pom, group_id, artifact_id)
    for entries in (direct, managed):
        if len(entries) > 1:
            raise MavenManifestError("duplicate Maven group:artifact declaration")

    for container in list(root):
        if _local_name(container.tag) == "dependencies":
            dependencies = [container]
        elif _local_name(container.tag) == "dependencyManagement":
            dependencies = _children(container, "dependencies")
        else:
            dependencies = []
        for deps in dependencies:
            for dep in _children(deps, "dependency"):
                dep_group = _resolve(_text(_child(dep, "groupId")) or pom.group_id, values)
                dep_artifact = _resolve(_text(_child(dep, "artifactId")), values)
                if (
                    dep_group == group_id
                    and dep_artifact == artifact_id
                    and (_text(_child(dep, "classifier")) or _text(_child(dep, "type")))
                ):
                    raise MavenManifestError("classifier/type Maven declaration is ambiguous")
    if direct:
        return direct[0]
    if managed:
        return managed[0]
    return None


def _canonical_gav(issue: VulnerabilityIssue) -> tuple[str, str] | None:
    purl = issue.purl or ""
    if purl.lower().startswith("pkg:maven/"):
        name = package_name_from_purl(purl)
    else:
        name = (issue.package_name or "").strip()
    if not name:
        return None
    parts = name.split(":")
    if (
        len(parts) != 2
        or not all(part and part.strip() == part for part in parts)
        or any(
            "/" in part or "\\" in part or any(char.isspace() for char in part) for part in parts
        )
    ):
        return None
    return parts[0], parts[1]


def _scan_relative_path(issue: VulnerabilityIssue, repo_root: Path) -> str | None:
    raw_payload = issue.raw_payload if isinstance(issue.raw_payload, dict) else {}
    raw = raw_payload.get("filePath")
    if not isinstance(raw, str) or not raw.strip():
        raw = issue.file_path
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip().replace("\\", "/")
    if raw.startswith("/scan/"):
        raw = raw[len("/scan/") :]
    elif raw == "/scan":
        raise MavenManifestError("ODC scan path does not identify a file")
    if raw.startswith("/workspace/"):
        try:
            Path(raw).resolve().relative_to(repo_root.resolve())
        except ValueError as exc:
            raise MavenManifestError("Maven scan path is outside the repository") from exc

    contained = repository_relative_path(raw, repo_root)
    if contained is not None:
        return contained
    if raw.startswith("/") or re.match(r"^[A-Za-z]:[/\\]", raw):
        raise MavenManifestError("Maven scan path is outside the repository")
    try:
        normalized = normalize_workspace_path(raw, allow_workspace_prefix=False)
        resolve_repository_path(repo_root, normalized)
    except WorkspacePathError as exc:
        raise MavenManifestError("unsafe Maven scan path") from exc
    return normalized


def _roots_and_links(
    manifests: Mapping[Path, MavenManifest],
    repo_root: Path,
) -> tuple[set[Path], dict[Path, set[Path]]]:
    parents: dict[Path, set[Path]] = {path: set() for path in manifests}
    for path, pom in manifests.items():
        if pom.parent_path in manifests:
            parents[path].add(pom.parent_path)
        for module in pom.modules:
            module_candidate = _module_pom_path(path, module, repo_root)
            if module_candidate not in manifests:
                raise MavenManifestError("Maven module POM was not discovered safely")
            parents[module_candidate].add(path)
    root_paths = {path for path in manifests if not parents[path]}
    return root_paths, parents


def locate_maven_from_issue(issue: VulnerabilityIssue, repo_path: Path) -> LocalizedIssue:
    """Resolve a Maven finding to one contained POM or fail closed."""
    empty = dict(
        issue=issue,
        manifest_file=None,
        is_direct_dependency=None,
        package_manager="maven",
        localization_confidence=0.0,
    )
    try:
        repo_root = Path(repo_path).resolve()
        if not repo_root.is_dir():
            raise MavenManifestError("repository root is unavailable")
        gav = _canonical_gav(issue)
        if gav is None:
            raise MavenManifestError("Maven finding has no canonical group:artifact identity")
        pom_paths = locate_maven_manifests(repo_root)
        if not pom_paths:
            raise MavenManifestError("repository has no Maven POM")
        cache: dict[Path, tuple[MavenManifest, ET.Element, dict[str, Path], str | None]] = {}
        manifests: dict[Path, MavenManifest] = {}
        for path in pom_paths:
            manifest, _root, _owners, _version = _load_manifest(path, repo_root, cache, set())
            manifests[path.resolve()] = manifest
        root_paths, parents = _roots_and_links(manifests, repo_root)
        if len(root_paths) != 1:
            raise MavenManifestError("repository has missing or multiple Maven project roots")
        root_path = next(iter(root_paths))

        relative = _scan_relative_path(issue, repo_root)
        selected: Path | None = None
        if relative is not None:
            scan_path = resolve_repository_path(repo_root, relative)
            current = scan_path if scan_path.is_dir() else scan_path.parent
            while True:
                candidate = current / "pom.xml"
                if candidate.is_symlink():
                    raise MavenManifestError("symlink POM cannot be authorized")
                if candidate.resolve() in manifests:
                    selected = candidate.resolve()
                    break
                if current == repo_root or repo_root not in current.parents:
                    break
                current = current.parent
        if selected is None:
            selected = root_path
        if selected not in manifests:
            raise MavenManifestError("no authorized Maven POM for scan path")

        # Require the selected POM to belong to the unique root's reactor/parent chain.
        ancestors: set[Path] = set()
        pending = [selected]
        while pending:
            item = pending.pop()
            if item in ancestors:
                continue
            ancestors.add(item)
            pending.extend(parents.get(item, ()))
        if root_path not in ancestors:
            raise MavenManifestError("scan path POM is not in the unique Maven reactor")

        manifest = manifests[selected]
        group_id, artifact_id = gav
        direct, managed = _target_entries(manifest, group_id, artifact_id)
        found = find_dependency_in_pom(group_id, artifact_id, manifest)
        inherited_direct = False
        inherited_management_entry = None
        parent_path = manifest.parent_path
        while parent_path in manifests:
            parent_manifest = manifests[parent_path]
            parent_declarations, parent_management = _target_entries(
                parent_manifest, group_id, artifact_id
            )
            find_dependency_in_pom(group_id, artifact_id, parent_manifest)
            if parent_declarations:
                inherited_direct = True
            if parent_management and inherited_management_entry is None:
                inherited_management_entry = parent_management[0]
            parent_path = parent_manifest.parent_path

        direct_entry = direct[0] if direct else None
        if direct_entry is not None:
            is_direct = True
            if direct_entry.declared_version:
                declaration_type = "dependencies"
                target_entry = direct_entry
            else:
                declaration_type = "dependencyManagement"
                target_entry = managed[0] if managed else inherited_management_entry
        else:
            declaration_type = "dependencyManagement"
            is_direct = inherited_direct
            target_entry = managed[0] if managed else inherited_management_entry or found
        manifest_rel = selected.relative_to(repo_root).as_posix()
        line_number = target_entry.line_number if target_entry else None
        return LocalizedIssue(
            issue=issue,
            manifest_file=manifest_rel,
            is_direct_dependency=is_direct,
            manifest_line=line_number,
            package_manager="maven",
            declaration_type=declaration_type,
            version_property_name=(
                target_entry.property_name if target_entry is not None else None
            ),
            version_property_file=(
                target_entry.property_file if target_entry is not None else None
            ),
            localization_confidence=0.95 if is_direct and target_entry else 0.80,
        )
    except (MavenManifestError, OSError, ValueError, ET.ParseError):
        return LocalizedIssue(**empty)


@dataclass(frozen=True)
class _XmlElementSpan:
    """Byte offsets for one parsed XML element."""

    start: int
    end: int
    text_start: int
    text_end: int


def _parse_xml_source(source: str) -> ET.Element:
    """Parse a UTF-8 POM source, rejecting DTDs and entity declarations."""
    if not isinstance(source, str):
        raise MavenManifestError("POM source must be UTF-8 text")
    if "\ufffd" in source:
        raise MavenManifestError("POM source is not valid UTF-8")
    declaration = re.match(r"\ufeff?\s*<\?xml\b([^?]*)\?>", source[:256], re.IGNORECASE)
    encoding_match = (
        re.search(r"\bencoding\s*=\s*(['\"])([^'\"]+)\1", declaration.group(1), re.IGNORECASE)
        if declaration
        else None
    )
    if encoding_match and encoding_match.group(2).strip().lower().replace("_", "-") not in {
        "utf-8",
        "utf8",
    }:
        raise MavenManifestError("POM source is not valid UTF-8")
    try:
        payload = source.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise MavenManifestError("POM source is not valid UTF-8") from exc
    if _has_dtd_or_entity(payload):
        raise MavenManifestError("DTD or entity declarations are forbidden")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise MavenManifestError("malformed Maven POM XML") from exc
    if _local_name(root.tag) != "project":
        raise MavenManifestError("not a Maven project POM")
    return root


def _tag_end(payload: bytes, start: int) -> int:
    """Return the byte offset just after an XML tag's closing ``>``."""
    quote = 0
    for index in range(start + 1, len(payload)):
        char = payload[index]
        if quote:
            if char == quote:
                quote = 0
        elif char in (ord("'"), ord('"')):
            quote = char
        elif char == ord(">"):
            return index + 1
    raise MavenManifestError("malformed XML tag boundary")


def _xml_element_spans(payload: bytes) -> dict[tuple[tuple[str, int], ...], _XmlElementSpan]:
    """Use Expat byte offsets to map structural element paths to raw spans."""
    parser = expat.ParserCreate(namespace_separator="}")
    stack: list[tuple[tuple[tuple[str, int], ...], int, int, bool]] = []
    sibling_counts: list[dict[str, int]] = []
    spans: dict[tuple[tuple[str, int], ...], _XmlElementSpan] = {}

    def start_element(name: str, _attrs: dict[str, str]) -> None:
        start = parser.CurrentByteIndex
        end = _tag_end(payload, start)
        local = name.rsplit("}", 1)[-1].split(":")[-1]
        counts = sibling_counts[-1] if sibling_counts else {}
        ordinal = counts.get(local, 0) + 1
        counts[local] = ordinal
        path = (stack[-1][0] if stack else ()) + ((local, ordinal),)
        empty = payload[start:end].rstrip().endswith(b"/>")
        stack.append((path, start, end, empty))
        sibling_counts.append({})

    def end_element(_name: str) -> None:
        path, start, text_start, empty = stack.pop()
        sibling_counts.pop()
        if empty:
            close_start = text_start
            end = text_start
        else:
            close_start = parser.CurrentByteIndex
            end = _tag_end(payload, close_start)
        spans[path] = _XmlElementSpan(start, end, text_start, close_start)

    parser.StartElementHandler = start_element
    parser.EndElementHandler = end_element
    try:
        parser.Parse(payload, True)
    except expat.ExpatError as exc:
        raise MavenManifestError("malformed Maven POM XML") from exc
    return spans


def _element_paths(root: ET.Element) -> dict[int, tuple[tuple[str, int], ...]]:
    paths: dict[int, tuple[tuple[str, int], ...]] = {}

    def visit(element: ET.Element, path: tuple[tuple[str, int], ...]) -> None:
        paths[id(element)] = path
        counts: dict[str, int] = {}
        for child in list(element):
            name = _local_name(child.tag)
            ordinal = counts.get(name, 0) + 1
            counts[name] = ordinal
            visit(child, path + ((name, ordinal),))

    root_name = _local_name(root.tag)
    visit(root, ((root_name, 1),))
    return paths


def _span_for(
    source: str,
    root: ET.Element,
    element: ET.Element,
) -> tuple[bytes, _XmlElementSpan]:
    payload = source.encode("utf-8")
    span = _xml_element_spans(payload).get(_element_paths(root)[id(element)])
    if span is None:
        raise MavenManifestError("validated XML element has no Expat byte span")
    return payload, span


def _replace_element_text(source: str, root: ET.Element, element: ET.Element, value: str) -> str:
    payload, span = _span_for(source, root, element)
    if list(element):
        raise MavenManifestError("target XML value contains nested elements")
    raw = payload[span.text_start : span.text_end]
    if b"<" in raw:
        raise MavenManifestError("target XML value contains comments or nested markup")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MavenManifestError("POM source is not valid UTF-8") from exc
    prefix_len = len(text) - len(text.lstrip())
    suffix_len = len(text) - len(text.rstrip())
    prefix = text[:prefix_len]
    suffix = text[len(text) - suffix_len :] if suffix_len else ""
    replacement = (prefix + _xml_escape(value) + suffix).encode("utf-8")
    updated = payload[: span.text_start] + replacement + payload[span.text_end :]
    return updated.decode("utf-8")


def _properties_for_source(root: ET.Element) -> dict[str, str]:
    properties = _child(root, "properties")
    if properties is None:
        return {}
    result: dict[str, str] = {}
    for child in list(properties):
        value = _text(child)
        if value is not None:
            result[_local_name(child.tag)] = value
    return result


def _dependency_matches(
    dependency: ET.Element,
    root: ET.Element,
    group_id: str,
    artifact_id: str,
    properties: Mapping[str, str],
) -> bool:
    project_group = (
        _text(_child(root, "groupId"))
        or properties.get("project.groupId")
        or properties.get("pom.groupId")
    )
    actual_group = _resolve(_text(_child(dependency, "groupId")) or project_group, properties)
    actual_artifact = _resolve(_text(_child(dependency, "artifactId")), properties)
    return actual_group == group_id and actual_artifact == artifact_id


def _section_dependencies(root: ET.Element, dependency_type: str) -> list[ET.Element]:
    if dependency_type == "dependencies":
        return [
            dependency
            for section in _children(root, "dependencies")
            for dependency in _children(section, "dependency")
        ]
    if dependency_type == "dependencyManagement":
        return [
            dependency
            for management in _children(root, "dependencyManagement")
            for section in _children(management, "dependencies")
            for dependency in _children(section, "dependency")
        ]
    raise MavenManifestError("dependency_type must be dependencies or dependencyManagement")


def update_pom_dependency_version(
    source: str,
    group_id: str,
    artifact_id: str,
    target_version: str,
    dependency_type: str = "dependencies",
    properties: Mapping[str, str] | None = None,
) -> str:
    """Replace one exact-GAV version text span without serializing the POM."""
    root = _parse_xml_source(source)
    values = dict(_properties_for_source(root))
    values.update(properties or {})
    if _profile_has_target(
        root,
        group_id,
        artifact_id,
        values,
        _text(_child(root, "groupId")) or values.get("project.groupId"),
    ):
        raise MavenManifestError("Maven profile declaration is ambiguous")
    matches = [
        dependency
        for dependency in _section_dependencies(root, dependency_type)
        if _dependency_matches(dependency, root, group_id, artifact_id, values)
    ]
    if len(matches) != 1:
        raise MavenManifestError("missing or duplicate exact-GAV dependency declaration")
    dependency = matches[0]
    if _text(_child(dependency, "classifier")) or _text(_child(dependency, "type")):
        raise MavenManifestError("classifier/type Maven declaration is ambiguous")
    version = _child(dependency, "version")
    if version is None:
        raise MavenManifestError("dependency declaration has no explicit version text")
    if list(version):
        raise MavenManifestError("dependency version contains nested markup")
    return _replace_element_text(source, root, version, target_version)


def update_pom_property_value(source: str, property_name: str, target_value: str) -> str:
    """Update exactly one project property value using its validated raw span."""
    root = _parse_xml_source(source)
    properties = _child(root, "properties")
    if properties is None:
        raise MavenManifestError(f"property {property_name!r} is unresolved")
    matches = [child for child in list(properties) if _local_name(child.tag) == property_name]
    if len(matches) != 1:
        raise MavenManifestError(f"property {property_name!r} is unresolved or ambiguous")
    return _replace_element_text(source, root, matches[0], target_value)


def _indent_before(payload: bytes, offset: int) -> bytes:
    line_start = payload.rfind(b"\n", 0, offset) + 1
    indent = payload[line_start:offset]
    return indent if not indent.strip() else b""


def _newline(payload: bytes) -> bytes:
    return b"\r\n" if b"\r\n" in payload else b"\n"


def _tag_prefix(source: str, root_span: _XmlElementSpan) -> str:
    payload = source.encode("utf-8")
    opening = payload[root_span.start : root_span.text_start].decode("utf-8")
    match = re.match(r"<([A-Za-z_][\w.-]*(?::[A-Za-z_][\w.-]*)?)", opening)
    if match is None:
        raise MavenManifestError("unsafe POM root element")
    qname = match.group(1)
    return qname.split(":", 1)[0] + ":" if ":" in qname else ""


def _replace_inner_with_child(
    source: str,
    root: ET.Element,
    parent: ET.Element,
    child_xml: str,
    children: Sequence[ET.Element],
) -> str:
    payload, parent_span = _span_for(source, root, parent)
    newline = _newline(payload)
    parent_indent = _indent_before(payload, parent_span.start)
    if children:
        _child_payload, last_span = _span_for(source, root, children[-1])
        child_indent = _indent_before(payload, last_span.start)
        if not child_indent:
            child_indent = parent_indent + b"  "
        tail = payload[last_span.end : parent_span.text_end]
        insertion = last_span.end
        inserted = newline + child_indent + child_xml.encode("utf-8")
        if not tail.startswith(newline):
            inserted += newline + parent_indent
        return (payload[:insertion] + inserted + payload[insertion:]).decode("utf-8")

    if b"<" in payload[parent_span.text_start : parent_span.text_end]:
        raise MavenManifestError("unsafe insertion would replace existing XML markup")
    child_indent = parent_indent + b"  "
    contents = newline + child_indent + child_xml.encode("utf-8") + newline + parent_indent
    updated = payload[: parent_span.text_start] + contents + payload[parent_span.text_end :]
    return updated.decode("utf-8")


_POM_ROOT_ORDER = (
    "modelVersion",
    "parent",
    "groupId",
    "artifactId",
    "version",
    "packaging",
    "name",
    "description",
    "url",
    "inceptionYear",
    "organization",
    "licenses",
    "developers",
    "contributors",
    "mailingLists",
    "prerequisites",
    "modules",
    "scm",
    "issueManagement",
    "ciManagement",
    "properties",
    "dependencyManagement",
    "dependencies",
    "repositories",
    "pluginRepositories",
    "build",
    "reporting",
    "profiles",
)


def _insert_root_dependency_management(
    source: str,
    root: ET.Element,
    management_xml: str,
) -> str:
    payload = source.encode("utf-8")
    spans = _xml_element_spans(payload)
    root_span = spans.get(_element_paths(root)[id(root)])
    if root_span is None:
        raise MavenManifestError("safe dependencyManagement insertion point not found")
    root_children = list(root)
    dm_rank = _POM_ROOT_ORDER.index("dependencyManagement")
    following: ET.Element | None = None
    for child in root_children:
        name = _local_name(child.tag)
        if name not in _POM_ROOT_ORDER:
            raise MavenManifestError("unknown root element order makes insertion unsafe")
        if _POM_ROOT_ORDER.index(name) > dm_rank:
            following = child
            break
    newline = _newline(payload)
    root_indent = _indent_before(payload, root_span.start)
    child_indent = root_indent + b"  "
    if following is not None:
        following_span = spans.get(_element_paths(root)[id(following)])
        if following_span is None:
            raise MavenManifestError("safe dependencyManagement insertion point not found")
        line_start = payload.rfind(b"\n", 0, following_span.start) + 1
        indent = payload[line_start : following_span.start]
        if indent.strip():
            raise MavenManifestError("inline root elements make insertion unsafe")
        insert_at = line_start
        inserted = child_indent + management_xml.encode("utf-8") + newline
    else:
        close_start = root_span.text_end
        line_start = payload.rfind(b"\n", 0, close_start) + 1
        current_indent = payload[line_start:close_start]
        if current_indent.strip():
            raise MavenManifestError("inline root closing tag makes insertion unsafe")
        insert_at = line_start
        inserted = child_indent + management_xml.encode("utf-8") + newline
    return (payload[:insert_at] + inserted + payload[insert_at:]).decode("utf-8")


def add_dependency_management_entry(
    source: str,
    group_id: str,
    artifact_id: str,
    target_version: str,
) -> str:
    """Add one local dependencyManagement override at a safe Maven root location."""
    root = _parse_xml_source(source)
    values = _properties_for_source(root)
    existing = [
        dependency
        for dependency in _section_dependencies(root, "dependencyManagement")
        if _dependency_matches(dependency, root, group_id, artifact_id, values)
    ]
    if existing:
        raise MavenManifestError("duplicate local dependencyManagement target")
    if _profile_target(root, group_id, artifact_id, values, _text(_child(root, "groupId"))):
        raise MavenManifestError("profile declaration makes Maven target ambiguous")
    spans = _xml_element_spans(source.encode("utf-8"))
    paths = _element_paths(root)
    root_span = spans.get(paths[id(root)])
    if root_span is None:
        raise MavenManifestError("safe dependencyManagement insertion point not found")
    prefix = _tag_prefix(source, root_span)
    q = lambda name: f"{prefix}{name}"  # noqa: E731
    dep_xml = (
        f"<{q('dependency')}><{q('groupId')}>{_xml_escape(group_id)}</{q('groupId')}>"
        f"<{q('artifactId')}>{_xml_escape(artifact_id)}</{q('artifactId')}>"
        f"<{q('version')}>{_xml_escape(target_version)}</{q('version')}>"
        f"</{q('dependency')}>"
    )
    management_sections = _children(root, "dependencyManagement")
    if len(management_sections) > 1:
        raise MavenManifestError("duplicate dependencyManagement sections")
    newline = _newline(source.encode("utf-8")).decode("ascii")
    if management_sections:
        management = management_sections[0]
        dependency_containers = _children(management, "dependencies")
        if len(dependency_containers) > 1:
            raise MavenManifestError("duplicate dependencyManagement dependency containers")
        if dependency_containers:
            container = dependency_containers[0]
            children = _children(container, "dependency")
            return _replace_inner_with_child(source, root, container, dep_xml, children)
        dm_span = spans.get(paths[id(management)])
        if dm_span is None:
            raise MavenManifestError("safe dependencyManagement insertion point not found")
        dm_indent = _indent_before(source.encode("utf-8"), dm_span.start).decode("utf-8")
        deps_xml = (
            f"<{q('dependencies')}>{newline}{dm_indent}  {dep_xml}{newline}"
            f"{dm_indent}</{q('dependencies')}>"
        )
        return _replace_inner_with_child(source, root, management, deps_xml, [])

    project_indent = _indent_before(source.encode("utf-8"), root_span.start).decode("utf-8")
    management_xml = (
        f"<{q('dependencyManagement')}>{newline}"
        f"{project_indent}    <{q('dependencies')}>{newline}"
        f"{project_indent}      {dep_xml}{newline}"
        f"{project_indent}    </{q('dependencies')}>{newline}"
        f"{project_indent}  </{q('dependencyManagement')}>"
    )
    return _insert_root_dependency_management(source, root, management_xml)


def _profile_target(
    root: ET.Element,
    group_id: str,
    artifact_id: str,
    properties: Mapping[str, str] | None = None,
    project_group: str | None = None,
) -> bool:
    values = dict(properties or {})
    for profile in root.iter():
        if _local_name(profile.tag) != "profile":
            continue
        for dependency in profile.iter():
            if _local_name(dependency.tag) != "dependency":
                continue
            group = _resolve(_text(_child(dependency, "groupId")) or project_group, values)
            artifact = _resolve(_text(_child(dependency, "artifactId")), values)
            if group == group_id and artifact == artifact_id:
                return True
    return False


def remove_direct_dependency_entry(
    source: str,
    group_id: str,
    artifact_id: str,
    properties: Mapping[str, str] | None = None,
) -> str:
    """Remove one exact unclassified dependency from root ``<dependencies>`` only."""
    root = _parse_xml_source(source)
    values = dict(_properties_for_source(root))
    values.update(properties or {})
    if _profile_target(root, group_id, artifact_id, values, _text(_child(root, "groupId"))):
        raise MavenManifestError("profile declaration makes Maven target ambiguous")
    matches = [
        dependency
        for dependency in _section_dependencies(root, "dependencies")
        if _dependency_matches(dependency, root, group_id, artifact_id, values)
    ]
    if len(matches) != 1:
        raise MavenManifestError("missing or duplicate direct Maven dependency")
    dependency = matches[0]
    if _text(_child(dependency, "classifier")) or _text(_child(dependency, "type")):
        raise MavenManifestError("classifier/type Maven declaration is ambiguous")
    payload, span = _span_for(source, root, dependency)
    start, end = span.start, span.end
    line_start = payload.rfind(b"\n", 0, start) + 1
    if not payload[line_start:start].strip():
        start = line_start
        if payload[end : end + 2] == b"\r\n":
            end += 2
        elif payload[end : end + 1] == b"\n":
            end += 1
    return (payload[:start] + payload[end:]).decode("utf-8")
