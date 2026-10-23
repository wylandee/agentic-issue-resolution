"""
reachability.py - Deterministic SCA package reachability analysis.

Scans application source files for import/require statements and uses the
result to annotate SCA vulnerability groups with a coarse reachability signal.

Decision model
--------------
- Imported in application code -> True
- Not imported, but listed as a direct dependency -> False
- Not imported and not a direct dependency -> None (unknown/transitive)
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from remediation_engine.contracts.schemas import IssueType, VulnerabilityGroup
from remediation_engine.language import ProjectLanguage
from remediation_engine.runtime.path_policy import WorkspacePathError, resolve_repository_path
from remediation_engine.tools.code_map import (
    extract_all_imports,
    language_for_path,
    load_source_bytes,
    parse_source,
)

logger = logging.getLogger(__name__)

_SOURCE_SUFFIXES = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
_EXCLUDED_PARTS = {"node_modules", ".git", "dist", "build"}
_MODULE_SPECIFIER_PATTERNS = (
    re.compile(r"\bfrom\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"\bimport\s*['\"]([^'\"]+)['\"]"),
    re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]"),
)


def _imported_package_names(statement: str) -> set[str]:
    """Return all npm package identities referenced by one import statement."""
    packages: set[str] = set()
    for pattern in _MODULE_SPECIFIER_PATTERNS:
        for match in pattern.finditer(statement):
            module = match.group(1)
            if not module or module.startswith((".", "/", "#", "node:")):
                continue
            segments = module.split("/")
            if segments[0].startswith("@"):
                package_name = "/".join(segments[:2]) if len(segments) > 1 else None
            else:
                package_name = segments[0]
            if package_name:
                packages.add(package_name)
    return packages


def _is_pypi_issue_group(group: VulnerabilityGroup) -> bool:
    """Return whether an SCA group is explicitly identified as PyPI."""
    for issue in group.issues:
        ecosystem = str(issue.ecosystem or "").strip().casefold()
        purl = str(issue.purl or "").strip().casefold()
        if ecosystem in {"pypi", "python"} or purl.startswith("pkg:pypi/"):
            return True
    return False


def _load_direct_dependencies(repo_root: Path) -> set[str]:
    """Best-effort load of direct dependencies from package.json."""
    try:
        package_json = resolve_repository_path(repo_root, "package.json")
    except WorkspacePathError:
        return set()
    if not package_json.is_file():
        return set()

    try:
        raw = package_json.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reachability: failed to parse %s: %s", package_json, exc)
        return set()

    if not isinstance(data, dict):
        return set()

    direct_deps: set[str] = set()
    for key in ("dependencies", "devDependencies"):
        deps = data.get(key)
        if isinstance(deps, dict):
            direct_deps.update(str(name) for name in deps)
    return direct_deps


def _iter_source_files(repo_root: Path) -> Iterable[Path]:
    """Yield application source files while skipping generated/vendor paths."""
    for path in repo_root.rglob("*"):
        if path.is_symlink():
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        if any(part in _EXCLUDED_PARTS for part in path.parts):
            continue
        yield path


def _collect_global_imports(repo_root: Path) -> set[str]:
    """Aggregate all import/require statements across application source files."""
    global_imports: set[str] = set()

    for source_file in _iter_source_files(repo_root):
        try:
            language = language_for_path(str(source_file))
            if language is None:
                continue

            source_bytes = load_source_bytes(source_file)
            if source_bytes is None:
                continue

            tree = parse_source(source_bytes, language)
            if tree is None:
                logger.warning(
                    "Reachability: parse returned no AST for %s; skipping.",
                    source_file,
                )
                continue

            imports = extract_all_imports(tree.root_node, source_bytes)
            global_imports.update(
                package_name
                for statement in imports
                for package_name in _imported_package_names(statement)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reachability: failed to analyze %s: %s",
                source_file,
                exc,
            )

    return global_imports


def analyze_reachability(
    groups: list[VulnerabilityGroup],
    repo_root: str | Path,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> None:
    """Mutate SCA groups in place with a deterministic reachability signal."""
    root = Path(repo_root).resolve()
    if not root.exists():
        logger.warning("Reachability: repo root does not exist: %s", root)
        return

    maven_groups: set[int] = set()
    if project_language == ProjectLanguage.JAVA:
        for group in groups:
            if group.issue_type != IssueType.SCA:
                continue
            if any(
                (issue.ecosystem or "").strip().lower() == "maven"
                or (issue.purl or "").strip().lower().startswith("pkg:maven/")
                for issue in group.issues
            ):
                group.is_reachable = None
                maven_groups.add(id(group))
        if all(group.issue_type != IssueType.SCA or id(group) in maven_groups for group in groups):
            return

    npm_groups: list[VulnerabilityGroup] = []
    for group in groups:
        if group.issue_type != IssueType.SCA or id(group) in maven_groups:
            continue
        if _is_pypi_issue_group(group):
            # JavaScript import names and package.json declarations do not
            # establish whether a PyPI distribution is used by Python code.
            group.is_reachable = None
        else:
            npm_groups.append(group)
    if not npm_groups:
        return

    root = Path(repo_root)
    direct_deps = _load_direct_dependencies(root)
    global_imports = _collect_global_imports(root)

    for group in npm_groups:
        component = (group.vulnerable_component or "").strip()
        if component and component in global_imports:
            group.is_reachable = True
        elif component in direct_deps:
            group.is_reachable = False
        else:
            group.is_reachable = None
