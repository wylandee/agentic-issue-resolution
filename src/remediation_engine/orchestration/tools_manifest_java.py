"""Supervisor-authorized atomic Maven POM update and removal transactions.

All manifest bytes are read from and written to ``DockerSandbox``.  XML edits
are delegated to byte-span helpers so the original POM formatting and unrelated
content remain untouched.
"""

from __future__ import annotations

import re
import shlex
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from langchain_core.tools import tool

from remediation_engine.contracts.schemas import (
    MavenTargetOperation,
    NoFixMitigationStage,
    WorkaroundExecutionPhase,
)
from remediation_engine.contracts.version_policy import (
    compare_maven_versions,
    is_stable_maven_version,
)
from remediation_engine.runtime.path_policy import WorkspacePathError, normalize_workspace_path
from remediation_engine.tools.maven_manifest_locator import (
    MavenManifestError,
    add_dependency_management_entry,
    remove_direct_dependency_entry,
    update_pom_dependency_version,
    update_pom_property_value,
)


@dataclass(frozen=True)
class _MavenPackageCheckpoint:
    """Pre-transaction contents of every authorized POM and touched-file state."""

    files: dict[str, str | None]
    touched_files_before: set[str]


@dataclass(frozen=True)
class _PomDocument:
    path: str
    source: str
    root: ET.Element
    group_id: str | None
    artifact_id: str | None
    version: str | None
    parent_path: str | None
    properties: dict[str, str]
    property_owners: dict[str, str]


_GAV_RE = re.compile(
    r"(?P<group>[A-Za-z0-9_][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_][A-Za-z0-9_-]*)*):"
    r"(?P<artifact>[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?)\Z"
)
_XML_PROPERTY = re.compile(r"\$\{([^{}]+)\}")
_GLOBAL_COORDINATE_LOCKS: dict[str, threading.Lock] = {}
_GLOBAL_COORDINATE_LOCKS_GUARD = threading.Lock()


def _coordinate_lock(coordinate: str) -> threading.Lock:
    with _GLOBAL_COORDINATE_LOCKS_GUARD:
        return _GLOBAL_COORDINATE_LOCKS.setdefault(coordinate, threading.Lock())


def _error(code: str, message: str) -> str:
    return f"ERROR_CODE: {code}: {message}"


def _safe_gav(value: str) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    match = _GAV_RE.fullmatch(value)
    return (match.group("group"), match.group("artifact")) if match else None


def _safe_manifest_path(value: str) -> str:
    normalized = normalize_workspace_path(str(value))
    if PurePosixPath(normalized).name != "pom.xml":
        raise ValueError("manifest_path must point to pom.xml")
    return normalized


def _normalize_paths_by_package(
    values: Mapping[str, Iterable[str]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for package_name, paths in values.items():
        coordinate = str(package_name).strip()
        if _safe_gav(coordinate) is None:
            continue
        safe: set[str] = set()
        for path in paths:
            try:
                safe.add(_safe_manifest_path(path))
            except (TypeError, ValueError, WorkspacePathError):
                continue
        if safe:
            result[coordinate] = safe
    return result


def _canonical_version(value: str) -> str | None:
    version = str(value or "").strip()
    if not version or not is_stable_maven_version(version):
        return None
    # Defense in depth: versions never enter the shell command, but restrict
    # replacement text to the same conservative Maven token alphabet.
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._+-]*", version):
        return None
    return version


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _child(element: ET.Element, name: str) -> ET.Element | None:
    matches = _children(element, name)
    return matches[0] if matches else None


def _text(element: ET.Element | None) -> str | None:
    if element is None or element.text is None:
        return None
    text = element.text.strip()
    return text or None


def _resolve_property(
    value: str | None,
    properties: Mapping[str, str],
    resolving: frozenset[str] = frozenset(),
) -> str | None:
    """Resolve property references without confusing repeated use with cycles."""
    if value is None or len(resolving) > 32:
        return None
    text = str(value).strip()
    matches = list(_XML_PROPERTY.finditer(text))
    if not matches:
        return text
    pieces: list[str] = []
    cursor = 0
    for match in matches:
        name = match.group(1)
        if name in resolving or name not in properties:
            return None
        replacement = _resolve_property(
            properties[name],
            properties,
            resolving | {name},
        )
        if replacement is None:
            return None
        pieces.extend((text[cursor : match.start()], replacement))
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _load_documents(
    contents: Mapping[str, str | None],
    allowed_paths: set[str],
) -> dict[str, _PomDocument]:
    """Parse only the authorized in-repository POM snapshot."""
    documents: dict[str, _PomDocument] = {}
    for path in sorted(allowed_paths):
        source = contents.get(path)
        if source is None:
            raise MavenManifestError(f"authorized POM is missing or unreadable: {path}")
        # parse_pom_xml works from disk and would escape the sandbox boundary;
        # parse this already-read source directly with the same XML guarantees.
        from remediation_engine.tools.maven_manifest_locator import _parse_xml_source

        root = _parse_xml_source(source)
        parent = _child(root, "parent")
        relative_node = _child(parent, "relativePath") if parent is not None else None
        relative = _text(relative_node) if relative_node is not None else "../pom.xml"
        if parent is not None and relative_node is not None and relative is None or parent is None:
            parent_path = None
        else:
            if "\\" in str(relative) or PurePosixPath(str(relative)).is_absolute():
                raise MavenManifestError("parent POM path escapes authorized repository")
            base = PurePosixPath(path).parent
            candidate = PurePosixPath(str(base / relative))
            parts: list[str] = []
            for part in candidate.parts:
                if part in ("", "."):
                    continue
                if part == "..":
                    if not parts:
                        raise MavenManifestError("parent POM path escapes authorized repository")
                    parts.pop()
                else:
                    parts.append(part)
            parent_path = "/".join(parts)
            if PurePosixPath(parent_path).name != "pom.xml":
                raise MavenManifestError("parent relativePath must identify pom.xml")
            if parent_path not in allowed_paths:
                # An existing external/in-repository but unauthorized parent is
                # never read. Its properties cannot authorize a safe edit.
                parent_path = None

        own_properties: dict[str, str] = {}
        props_node = _child(root, "properties")
        if props_node is not None:
            for prop in list(props_node):
                value = _text(prop)
                if value is not None:
                    own_properties[_local_name(prop.tag)] = value

        documents[path] = _PomDocument(
            path=path,
            source=source,
            root=root,
            group_id=_text(_child(root, "groupId")),
            artifact_id=_text(_child(root, "artifactId")),
            version=_text(_child(root, "version")),
            parent_path=parent_path,
            properties=own_properties,
            property_owners={name: path for name in own_properties},
        )

    visiting: set[str] = set()
    resolved: dict[str, _PomDocument] = {}

    def load(path: str) -> _PomDocument:
        if path in resolved:
            return resolved[path]
        if path in visiting:
            raise MavenManifestError("cyclic Maven parent POM chain")
        visiting.add(path)
        document = documents[path]
        parent = load(document.parent_path) if document.parent_path in documents else None
        properties = dict(parent.properties) if parent else {}
        owners = dict(parent.property_owners) if parent else {}
        properties.update(document.properties)
        owners.update(document.property_owners)
        parent_node = _child(document.root, "parent")
        declared_parent_group = (
            _text(_child(parent_node, "groupId")) if parent_node is not None else None
        )
        declared_parent_artifact = (
            _text(_child(parent_node, "artifactId")) if parent_node is not None else None
        )
        declared_parent_version = (
            _text(_child(parent_node, "version")) if parent_node is not None else None
        )
        if parent is not None:
            properties.update(
                {
                    "project.parent.groupId": parent.group_id or "",
                    "pom.parent.groupId": parent.group_id or "",
                    "project.parent.artifactId": parent.artifact_id or "",
                    "pom.parent.artifactId": parent.artifact_id or "",
                    "project.parent.version": parent.version or "",
                    "pom.parent.version": parent.version or "",
                }
            )
            expected_parent_group = _resolve_property(declared_parent_group, properties)
            expected_parent_artifact = _resolve_property(declared_parent_artifact, properties)
            expected_parent_version = _resolve_property(declared_parent_version, properties)
            if (
                not expected_parent_group
                or not expected_parent_artifact
                or not expected_parent_version
            ):
                raise MavenManifestError("unresolved Maven parent property coordinates/version")
            if (
                expected_parent_group != parent.group_id
                or expected_parent_artifact != parent.artifact_id
                or parent.version is None
            ):
                raise MavenManifestError("Maven parent POM coordinates are ambiguous")
            try:
                parent_version_matches = (
                    compare_maven_versions(expected_parent_version, parent.version) == 0
                )
            except ValueError as exc:
                raise MavenManifestError("Maven parent POM version is ambiguous") from exc
            if not parent_version_matches:
                raise MavenManifestError("Maven parent POM coordinates are ambiguous")
            parent_group = parent.group_id
            parent_version = parent.version
        else:
            parent_group = _resolve_property(declared_parent_group, properties)
            parent_artifact = _resolve_property(declared_parent_artifact, properties)
            parent_version = _resolve_property(declared_parent_version, properties)
            properties.update(
                {
                    "project.parent.groupId": parent_group or "",
                    "pom.parent.groupId": parent_group or "",
                    "project.parent.artifactId": parent_artifact or "",
                    "pom.parent.artifactId": parent_artifact or "",
                    "project.parent.version": parent_version or "",
                    "pom.parent.version": parent_version or "",
                }
            )
        group_id = _resolve_property(document.group_id or parent_group, properties)
        artifact_id = _resolve_property(document.artifact_id, properties)
        version = _resolve_property(
            document.version or (parent.version if parent else parent_version),
            properties,
        )
        if group_id:
            properties["project.groupId"] = group_id
            properties["pom.groupId"] = group_id
        if artifact_id:
            properties["project.artifactId"] = artifact_id
            properties["pom.artifactId"] = artifact_id
        if version:
            properties["project.version"] = version
            properties["pom.version"] = version
        updated = _PomDocument(
            path=path,
            source=document.source,
            root=document.root,
            group_id=group_id,
            artifact_id=artifact_id,
            version=version,
            parent_path=document.parent_path,
            properties=properties,
            property_owners=owners,
        )
        resolved[path] = updated
        visiting.remove(path)
        return updated

    for path in documents:
        load(path)
    for document in resolved.values():
        _module_child_paths(document)
    return resolved


def _profile_target(
    root: ET.Element,
    group_id: str,
    artifact_id: str,
    properties: Mapping[str, str],
    project_group: str | None,
) -> bool:
    for profile in root.iter():
        if _local_name(profile.tag) != "profile":
            continue
        for dependency in profile.iter():
            if _local_name(dependency.tag) != "dependency":
                continue
            raw_group = _text(_child(dependency, "groupId")) or project_group
            raw_artifact = _text(_child(dependency, "artifactId"))
            if (
                _resolve_property(raw_group, properties) == group_id
                and _resolve_property(raw_artifact, properties) == artifact_id
            ):
                return True
    return False


def _resolve_dependency_coordinate_value(
    value: str | None,
    document: _PomDocument,
) -> str | None:
    resolved = _resolve_property(value, document.properties)
    if resolved is None and value and _XML_PROPERTY.search(value):
        raise MavenManifestError("unresolved Maven dependency coordinate property")
    return resolved


def _exact_target_occurrences(
    document: _PomDocument,
    group_id: str,
    artifact_id: str,
) -> tuple[list[ET.Element], list[ET.Element]]:
    if _profile_target(
        document.root, group_id, artifact_id, document.properties, document.group_id
    ):
        raise MavenManifestError("Maven profile declaration makes target ambiguous")
    direct: list[ET.Element] = []
    managed: list[ET.Element] = []
    for container in _children(document.root, "dependencies"):
        for dep in _children(container, "dependency"):
            dep_group = _resolve_dependency_coordinate_value(
                _text(_child(dep, "groupId")) or document.group_id, document
            )
            dep_artifact = _resolve_dependency_coordinate_value(
                _text(_child(dep, "artifactId")), document
            )
            if dep_group == group_id and dep_artifact == artifact_id:
                if _text(_child(dep, "classifier")) or _text(_child(dep, "type")):
                    raise MavenManifestError("classifier/type declaration makes target ambiguous")
                direct.append(dep)
    for section in _children(document.root, "dependencyManagement"):
        for container in _children(section, "dependencies"):
            for dep in _children(container, "dependency"):
                dep_group = _resolve_dependency_coordinate_value(
                    _text(_child(dep, "groupId")) or document.group_id, document
                )
                dep_artifact = _resolve_dependency_coordinate_value(
                    _text(_child(dep, "artifactId")), document
                )
                if dep_group == group_id and dep_artifact == artifact_id:
                    if _text(_child(dep, "classifier")) or _text(_child(dep, "type")):
                        raise MavenManifestError(
                            "classifier/type declaration makes target ambiguous"
                        )
                    managed.append(dep)
    if len(direct) > 1 or len(managed) > 1:
        raise MavenManifestError("duplicate exact-GAV declaration")
    return direct, managed


def _owner_property(document: _PomDocument, raw_version: str) -> tuple[str, str] | None:
    raw = raw_version.strip()
    match = re.fullmatch(r"\$\{([^{}]+)\}", raw)
    if match is None:
        if "${" in raw or "}" in raw:
            raise MavenManifestError("unresolved Maven version property expression")
        return None
    name = match.group(1)
    owner = document.property_owners.get(name)
    if owner is None or name not in document.properties:
        raise MavenManifestError(f"unresolved Maven version property {name}")
    resolved = _resolve_property(document.properties[name], document.properties)
    if resolved is None:
        raise MavenManifestError(f"unresolved or cyclic Maven version property {name}")
    return name, owner


def _parent_is_dependency_version(root: ET.Element, candidate: ET.Element) -> bool:
    parents: dict[int, ET.Element] = {}
    for parent in root.iter():
        for child in list(parent):
            parents[id(child)] = parent
    dependency = parents.get(id(candidate))
    if dependency is None or _local_name(dependency.tag) != "dependency":
        return False
    sections = parents.get(id(dependency))
    if sections is None or _local_name(sections.tag) != "dependencies":
        return False
    owner = parents.get(id(sections))
    if owner is None:
        return False
    if _local_name(owner.tag) == "dependencyManagement":
        owner = parents.get(id(owner))
    return owner is not None and _local_name(owner.tag) == "project"


def _version_parent_is_exact_gav(
    document: _PomDocument,
    version_element: ET.Element,
    group_id: str,
    artifact_id: str,
) -> bool:
    parents: dict[int, ET.Element] = {}
    for parent in document.root.iter():
        for child in list(parent):
            parents[id(child)] = parent
    dependency = parents.get(id(version_element))
    if dependency is None or not _parent_is_dependency_version(document.root, version_element):
        return False
    raw_group = _text(_child(dependency, "groupId")) or document.group_id
    raw_artifact = _text(_child(dependency, "artifactId"))
    return (
        _resolve_property(raw_group, document.properties) == group_id
        and _resolve_property(raw_artifact, document.properties) == artifact_id
    )


def _module_child_paths(document: _PomDocument) -> list[str]:
    modules = _child(document.root, "modules")
    if modules is None:
        return []
    children: list[str] = []
    for module in _children(modules, "module"):
        raw = _text(module)
        if not raw or "\\" in raw or PurePosixPath(raw).is_absolute():
            raise MavenManifestError("unsafe Maven reactor module path")
        candidate = PurePosixPath(document.path).parent / raw
        parts: list[str] = []
        for part in candidate.parts:
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    raise MavenManifestError("Maven reactor module escapes repository")
                parts.pop()
            else:
                parts.append(part)
        child_path = "/".join(parts)
        if PurePosixPath(child_path).name != "pom.xml":
            child_path = f"{child_path}/pom.xml"
        children.append(child_path)
    return children


def _check_property_isolation(
    documents: Mapping[str, _PomDocument],
    property_name: str,
    owner_path: str,
    group_id: str,
    artifact_id: str,
) -> None:
    """Require every authorized use of an edited property to version this GAV."""
    pending_modules = [owner_path]
    visited_modules: set[str] = set()
    while pending_modules:
        parent_path = pending_modules.pop()
        if parent_path in visited_modules:
            continue
        visited_modules.add(parent_path)
        parent_document = documents.get(parent_path)
        if parent_document is None:
            raise MavenManifestError(
                f"Maven property {property_name} is shared outside the authorized reactor"
            )
        for child_path in _module_child_paths(parent_document):
            if child_path not in documents:
                raise MavenManifestError(
                    f"Maven property {property_name} is shared outside the authorized reactor"
                )
            pending_modules.append(child_path)
    for document in documents.values():
        for element in document.root.iter():
            values = [element.text or "", *element.attrib.values()]
            same_owner_reference = any(
                match.group(1) == property_name
                and document.property_owners.get(property_name) == owner_path
                for value in values
                for match in _XML_PROPERTY.finditer(value)
            )
            if not same_owner_reference:
                continue
            version_reference = _XML_PROPERTY.fullmatch((element.text or "").strip())
            is_target_version = (
                _local_name(element.tag) == "version"
                and _parent_is_dependency_version(document.root, element)
                and _version_parent_is_exact_gav(document, element, group_id, artifact_id)
                and version_reference is not None
                and version_reference.group(1) == property_name
                and document.property_owners.get(property_name) == owner_path
            )
            if not is_target_version:
                raise MavenManifestError(
                    f"Maven property {property_name} is shared outside {group_id}:{artifact_id}"
                )


def _topological_root(
    allowed_paths: set[str],
    documents: Mapping[str, _PomDocument],
    selected_manifest: str,
) -> str:
    """Choose the highest authorized POM in the selected parent/reactor chain."""
    root_path = selected_manifest
    visited: set[str] = set()
    while root_path in documents and root_path not in visited:
        visited.add(root_path)
        parent_path = documents[root_path].parent_path
        if parent_path not in allowed_paths or parent_path not in documents:
            module_parents = [
                candidate
                for candidate, document in documents.items()
                if root_path in _module_child_paths(document)
            ]
            if len(module_parents) > 1:
                raise MavenManifestError("duplicate reactor parent makes Maven root ambiguous")
            parent_path = module_parents[0] if module_parents else None
        if parent_path not in allowed_paths or parent_path not in documents:
            break
        root_path = parent_path
    return root_path


def _transaction_error_from_parse(exc: Exception) -> tuple[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "property" in lowered and ("unresolved" in lowered or "cyclic" in lowered):
        return "POM_PROPERTY_UNRESOLVED", message
    if "shared" in lowered:
        return "POM_PROPERTY_SHARED", message
    if "insertion" in lowered:
        return "POM_INSERTION_UNSAFE", message
    if any(marker in lowered for marker in ("ambiguous", "duplicate", "classifier", "profile")):
        return "POM_TARGET_AMBIGUOUS", message
    if any(
        marker in lowered
        for marker in ("malformed", "dtd", "entity declaration", "not a maven project", "utf-8")
    ):
        return "POM_INVALID_XML", message
    return "EDIT_FAILED", message


def _capture_checkpoint(
    sandbox: Any,
    allowed_paths: Iterable[str],
    touched_files: set[str],
) -> _MavenPackageCheckpoint:
    files = {path: sandbox.read_file(path) for path in sorted(set(allowed_paths))}
    return _MavenPackageCheckpoint(files, set(touched_files))


def _read_authorized_poms(sandbox: Any, allowed_paths: Iterable[str]) -> dict[str, str]:
    contents: dict[str, str] = {}
    for path in sorted(set(allowed_paths)):
        source = sandbox.read_file(path)
        if not isinstance(source, str):
            raise MavenManifestError(f"authorized POM is missing or unreadable after edit: {path}")
        contents[path] = source
    return contents


def _restore_checkpoint(
    sandbox: Any,
    checkpoint: _MavenPackageCheckpoint,
    touched_files: set[str],
) -> str | None:
    errors: list[str] = []
    for path, source in checkpoint.files.items():
        try:
            if source is None:
                result = sandbox.run(f"rm -f -- {shlex.quote(path)}")
                if result.exit_code != 0:
                    errors.append(f"{path}: delete failed (exit {result.exit_code})")
            else:
                sandbox.write_file(path, source)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
    touched_files.clear()
    touched_files.update(checkpoint.touched_files_before)
    return "POM rollback failed: " + " | ".join(errors) if errors else None


def _sync_maven(sandbox: Any, root_pom: str) -> tuple[bool, str]:
    command = shlex.join(["mvn", "-B", "-q", "-f", root_pom, "dependency:resolve"])
    try:
        result = sandbox.run(command, timeout=600)
    except Exception as exc:  # timeout and command exceptions are sync failures
        return False, f"Maven dependency synchronization raised: {exc}"
    if result.exit_code != 0:
        return False, (
            f"Maven dependency synchronization failed (exit {result.exit_code}); "
            f"stdout: {str(result.stdout or '')[:3000]}; stderr: {str(result.stderr or '')[:3000]}"
        )
    return True, ""


def _maven_transaction(
    sandbox: Any,
    touched_files: set[str],
    coordinate: str,
    selected_manifest: str,
    allowed_paths: set[str],
    transform,
) -> str:
    checkpoint: _MavenPackageCheckpoint | None = None
    try:
        checkpoint = _capture_checkpoint(sandbox, allowed_paths, touched_files)
        contents = dict(checkpoint.files)
        documents = _load_documents(contents, allowed_paths)
        changed = transform(contents, documents)
        if not changed:
            return _error("EDIT_FAILED", f"No editable local declaration found for {coordinate}.")
        for path, source in contents.items():
            if source != checkpoint.files[path]:
                sandbox.write_file(path, source)
        post_edit = _read_authorized_poms(sandbox, allowed_paths)
        if post_edit != contents:
            raise MavenManifestError("authorized POM changed unexpectedly during edit")
        post_edit_documents = _load_documents(post_edit, allowed_paths)
        root_pom = _topological_root(allowed_paths, post_edit_documents, selected_manifest)
        succeeded, detail = _sync_maven(sandbox, root_pom)
        if not succeeded:
            rollback = _restore_checkpoint(sandbox, checkpoint, touched_files)
            suffix = f" {rollback}" if rollback else " POM checkpoint restored."
            return _error("MANIFEST_SYNC_FAILED", detail + suffix)
        post_sync = _read_authorized_poms(sandbox, allowed_paths)
        if post_sync != post_edit:
            raise MavenManifestError("Maven synchronization unexpectedly changed an authorized POM")
        _load_documents(post_sync, allowed_paths)
        for path, before in checkpoint.files.items():
            if post_sync[path] != before:
                touched_files.add(path)
        return ""
    except Exception as exc:  # noqa: BLE001
        rollback = (
            _restore_checkpoint(sandbox, checkpoint, touched_files)
            if checkpoint is not None
            else None
        )
        code, message = _transaction_error_from_parse(exc)
        suffix = (
            f" Rollback: {rollback}"
            if rollback
            else " Rollback: authorized POM checkpoint restored."
        )
        return _error(code, message + suffix)


def _make_modify_and_validate_maven_dependency_tool(
    sandbox: Any,
    touched_files: set[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    allowed_target_versions_by_package: Mapping[str, Iterable[str]],
    allowed_dependency_types_by_package: Mapping[str, Iterable[str]],
    execution_state: dict[str, Any] | None = None,
    maven_target_operations_by_package: Mapping[str, Iterable[str]] | None = None,
):
    """Create the sole worker-visible Maven dependency update tool."""
    allowed_paths_by_coordinate = _normalize_paths_by_package(package_manifest_paths)
    approved_versions = {
        str(key).strip(): {
            str(value).strip() for value in ([values] if isinstance(values, str) else values)
        }
        for key, values in allowed_target_versions_by_package.items()
    }
    approved_types = {
        str(key).strip(): {
            str(value).strip() for value in ([values] if isinstance(values, str) else values)
        }
        for key, values in allowed_dependency_types_by_package.items()
    }
    approved_operations = {
        str(key).strip(): {
            str(getattr(value, "value", value)).strip()
            for value in ([values] if isinstance(values, str) else values)
        }
        for key, values in (maven_target_operations_by_package or {}).items()
    }
    state = execution_state if execution_state is not None else {}

    def record_attempt(coordinate: str, signature: str) -> str | None:
        attempts = state.setdefault("manifest_transaction_attempts_by_package", {})
        signatures = state.setdefault("manifest_transaction_signatures_by_package", {})
        current = set(signatures.get(coordinate, []))
        if signature in current:
            return _error(
                "RETRY_PARAMETERS_UNCHANGED",
                "Retry with a different Supervisor-approved target version or dependency type.",
            )
        if int(attempts.get(coordinate, 0)) >= 3:
            return _error(
                "RETRY_LIMIT_REACHED",
                "At most three combined Maven attempts are allowed for this package.",
            )
        current.add(signature)
        signatures[coordinate] = sorted(current)
        attempts[coordinate] = int(attempts.get(coordinate, 0)) + 1
        state["manifest_transaction_attempts"] = (
            int(state.get("manifest_transaction_attempts", 0)) + 1
        )
        return None

    @tool
    def modify_and_validate_maven_dependency(
        package_name: str,
        target_version: str,
        dependency_type: str,
        manifest_path: str,
    ) -> str:
        """Apply the committed Maven operation to the exact approved GAV atomically."""
        coordinate = str(package_name or "").strip()
        gav = _safe_gav(coordinate)
        if gav is None:
            return _error(
                "INVALID_ARGUMENT",
                "package_name must be canonical safe group:artifact.",
            )
        version = _canonical_version(target_version)
        if version is None:
            return _error(
                "INVALID_ARGUMENT",
                "target_version must be a stable safe Maven version.",
            )
        kind = str(dependency_type or "").strip()
        if kind not in {"dependencies", "dependencyManagement"}:
            return _error(
                "INVALID_ARGUMENT",
                "dependency_type must be dependencies or dependencyManagement.",
            )
        try:
            selected_manifest = _safe_manifest_path(manifest_path)
        except (ValueError, WorkspacePathError) as exc:
            return _error("INVALID_ARGUMENT", str(exc))
        allowed_paths = allowed_paths_by_coordinate.get(coordinate)
        if not allowed_paths or selected_manifest not in allowed_paths:
            allowed = ", ".join(sorted(allowed_paths or ())) or "none"
            return _error(
                "TARGET_NOT_ALLOWED",
                f"manifest_path is not authorized; allowed POMs: {allowed}.",
            )
        if coordinate not in approved_versions or version not in approved_versions[coordinate]:
            return _error(
                "TARGET_NOT_ALLOWED",
                "target_version is not Supervisor-approved for this exact GAV.",
            )
        if coordinate not in approved_types or approved_types[coordinate] != {kind}:
            return _error(
                "TARGET_NOT_ALLOWED",
                "dependency_type is not the committed Supervisor target type.",
            )
        operations = approved_operations.get(
            coordinate,
            {MavenTargetOperation.UPDATE_DECLARATION.value},
        )
        if len(operations) != 1:
            return _error(
                "TARGET_NOT_ALLOWED",
                "Maven edit operation is missing or ambiguous in the committed authorization.",
            )
        operation = next(iter(operations))
        if operation not in {
            MavenTargetOperation.UPDATE_DECLARATION.value,
            MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT.value,
        } or (
            operation == MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT.value
            and kind != "dependencyManagement"
        ):
            return _error(
                "TARGET_NOT_ALLOWED",
                "Maven edit operation is not authorized for the committed declaration type.",
            )

        with _coordinate_lock(coordinate):
            pending = str(state.get("pending_validation_package", "") or "")
            if pending and pending != coordinate:
                return _error(
                    "TARGET_NOT_ALLOWED",
                    f"Package {pending!r} has an unfinished Maven transaction.",
                )
            signature = "|".join((coordinate, version, kind, selected_manifest))
            attempt_error = record_attempt(coordinate, signature)
            if attempt_error:
                return attempt_error
            state["pending_validation_package"] = coordinate
            group_id, artifact_id = gav

            def transform(
                contents: dict[str, str | None],
                documents: Mapping[str, _PomDocument],
            ) -> bool:
                direct_entries: dict[str, ET.Element] = {}
                managed_entries: dict[str, ET.Element] = {}
                for path, document in documents.items():
                    direct, managed = _exact_target_occurrences(document, group_id, artifact_id)
                    if direct:
                        direct_entries[path] = direct[0]
                    if managed:
                        managed_entries[path] = managed[0]

                changed = False
                if kind == "dependencies":
                    if not direct_entries:
                        raise MavenManifestError("no exact direct dependency declaration to update")
                    for path, dependency in direct_entries.items():
                        version_node = _child(dependency, "version")
                        if version_node is None:
                            raise MavenManifestError("direct dependency has no explicit version")
                        raw_version = _text(version_node) or ""
                        prop = _owner_property(documents[path], raw_version)
                        if prop is not None:
                            name, owner = prop
                            _check_property_isolation(documents, name, owner, group_id, artifact_id)
                            before = contents[owner]
                            after = update_pom_property_value(before, name, version)
                            if after != before:
                                contents[owner] = after
                                changed = True
                        else:
                            before = contents[path]
                            after = update_pom_dependency_version(
                                before,
                                group_id,
                                artifact_id,
                                version,
                                "dependencies",
                                documents[path].properties,
                            )
                            if after != before:
                                contents[path] = after
                                changed = True
                else:
                    root_path = _topological_root(set(contents), documents, selected_manifest)
                    if any(
                        _text(_child(dependency, "version"))
                        for dependency in direct_entries.values()
                    ):
                        raise MavenManifestError(
                            "direct explicit version cannot be updated through dependencyManagement"
                        )
                    managed_paths = [path for path in sorted(managed_entries)]
                    if managed_paths:
                        for managed_path in managed_paths:
                            raw_version = (
                                _text(_child(managed_entries[managed_path], "version")) or ""
                            )
                            prop = _owner_property(documents[managed_path], raw_version)
                            if prop is not None:
                                name, owner = prop
                                _check_property_isolation(
                                    documents, name, owner, group_id, artifact_id
                                )
                                before = contents[owner]
                                after = update_pom_property_value(before, name, version)
                                changed_path = owner
                            else:
                                before = contents[managed_path]
                                after = update_pom_dependency_version(
                                    before,
                                    group_id,
                                    artifact_id,
                                    version,
                                    "dependencyManagement",
                                    documents[managed_path].properties,
                                )
                                changed_path = managed_path
                            contents[changed_path] = after
                            changed = changed or after != before
                    else:
                        if operation != MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT.value:
                            raise MavenManifestError(
                                "no managed declaration exists and the Supervisor did not "
                                "authorize creating one"
                            )
                        before = contents[root_path]
                        after = add_dependency_management_entry(
                            before, group_id, artifact_id, version
                        )
                        contents[root_path] = after
                        changed = after != before
                return changed

            try:
                error = _maven_transaction(
                    sandbox,
                    touched_files,
                    coordinate,
                    selected_manifest,
                    allowed_paths,
                    transform,
                )
            finally:
                state["pending_validation_package"] = None
            if error:
                return error
            state["edits_started"] = True
            state["validation_calls"] = int(state.get("validation_calls", 0)) + 1
            validated = list(state.get("validated_packages", []) or [])
            if coordinate not in validated:
                validated.append(coordinate)
            state["validated_packages"] = validated
            return (
                f"SUCCESS: Updated {coordinate} to {version} through Maven target {kind}; "
                "synchronized authorized POMs."
            )

    return modify_and_validate_maven_dependency


def _remove_maven_no_fix_dependency_transaction(
    sandbox: Any,
    touched_files: set[str],
    plan_state: dict[str, Any],
    package_name: str,
    manifest_paths: Sequence[str],
    requested_package: str,
    manifest_path: str,
) -> str:
    """Remove one committed Maven direct dependency and synchronize atomically."""
    if plan_state.get("no_fix_stage") != NoFixMitigationStage.PACKAGE_REMOVAL.value:
        return _error("PLAN_VIOLATION", "Maven removal is allowed only in PACKAGE_REMOVAL.")
    if not plan_state.get("recorded") or not plan_state.get("package_removal_planned"):
        return _error(
            "PLAN_VIOLATION",
            "Record a Supervisor-committed package-removal plan first.",
        )
    coordinate = str(package_name or "").strip()
    if str(requested_package or "").strip() != coordinate or _safe_gav(coordinate) is None:
        return _error(
            "TARGET_NOT_ALLOWED",
            "Only the committed exact Maven group:artifact may be removed.",
        )
    try:
        requested_path = _safe_manifest_path(manifest_path)
        allowed_paths = {_safe_manifest_path(path) for path in manifest_paths}
    except (TypeError, ValueError, WorkspacePathError) as exc:
        return _error("INVALID_ARGUMENT", str(exc))
    if requested_path not in allowed_paths:
        return _error("TARGET_NOT_ALLOWED", "manifest_path is not an authorized POM.")

    with _coordinate_lock(coordinate):
        checkpoint: _MavenPackageCheckpoint | None = None
        try:
            checkpoint = _capture_checkpoint(sandbox, allowed_paths, touched_files)
            contents = dict(checkpoint.files)
            documents = _load_documents(contents, allowed_paths)
            group_id, artifact_id = _safe_gav(coordinate)  # type: ignore[misc]
            document = documents[requested_path]
            direct, _managed = _exact_target_occurrences(document, group_id, artifact_id)
            if not direct:
                return _error(
                    "EDIT_FAILED",
                    "The exact GAV has no removable direct <dependencies> entry.",
                )
            after = remove_direct_dependency_entry(
                contents[requested_path],
                group_id,
                artifact_id,
                document.properties,
            )
            if after == contents[requested_path]:
                return _error("EDIT_FAILED", "No direct Maven declaration was removed.")
            contents[requested_path] = after
            for path, source in contents.items():
                if source != checkpoint.files[path]:
                    sandbox.write_file(path, source)
            post_edit = _read_authorized_poms(sandbox, allowed_paths)
            if post_edit != contents:
                raise MavenManifestError("authorized POM changed unexpectedly during edit")
            post_edit_documents = _load_documents(post_edit, allowed_paths)
            root_path = _topological_root(allowed_paths, post_edit_documents, requested_path)
            succeeded, detail = _sync_maven(sandbox, root_path)
            if not succeeded:
                rollback = _restore_checkpoint(sandbox, checkpoint, touched_files)
                suffix = f" {rollback}" if rollback else " POM checkpoint restored."
                return _error("MANIFEST_SYNC_FAILED", detail + suffix)
            post_sync = _read_authorized_poms(sandbox, allowed_paths)
            if post_sync != post_edit:
                raise MavenManifestError(
                    "Maven synchronization unexpectedly changed an authorized POM"
                )
            _load_documents(post_sync, allowed_paths)
            changed: list[str] = []
            for path, before in checkpoint.files.items():
                if post_sync[path] != before:
                    touched_files.add(path)
                    changed.append(path)
            plan_state["package_removal_completed"] = True
            plan_state["no_fix_package_removed"] = True
            plan_state["package_removal_files"] = sorted(
                set(plan_state.get("package_removal_files", [])) | set(changed)
            )
            source_edit_pending = plan_state.get("pending_edit_set") is not None
            plan_state["phase"] = (
                WorkaroundExecutionPhase.VALIDATE.value
                if not plan_state.get("planned_replacements") or source_edit_pending
                else WorkaroundExecutionPhase.EXECUTE.value
            )
            return (
                f"SUCCESS: Removed direct Maven dependency {coordinate} and synchronized "
                f"{', '.join(changed)}."
            )
        except Exception as exc:  # noqa: BLE001
            rollback = (
                _restore_checkpoint(sandbox, checkpoint, touched_files)
                if checkpoint is not None
                else None
            )
            code, message = _transaction_error_from_parse(exc)
            suffix = (
                f" Rollback: {rollback}"
                if rollback
                else " Rollback: authorized POM checkpoint restored."
            )
            return _error(code, message + suffix)


__all__ = [
    "_MavenPackageCheckpoint",
    "_make_modify_and_validate_maven_dependency_tool",
    "_remove_maven_no_fix_dependency_transaction",
]
