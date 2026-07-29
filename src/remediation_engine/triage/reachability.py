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
from collections.abc import Iterable
from pathlib import Path

from remediation_engine.contracts.schemas import IssueType, VulnerabilityGroup
from remediation_engine.tools.code_map import (
    extract_imports,
    language_for_path,
    load_source_bytes,
    parse_source,
)

logger = logging.getLogger(__name__)

_SOURCE_SUFFIXES = {".js", ".ts", ".jsx", ".tsx"}
_EXCLUDED_PARTS = {"node_modules", ".git", "dist", "build"}


def _load_direct_dependencies(repo_root: Path) -> set[str]:
    """Best-effort load of direct dependencies from package.json."""
    package_json = repo_root / "package.json"
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
            direct_deps.update(str(name) for name in deps.keys())
    return direct_deps


def _iter_source_files(repo_root: Path) -> Iterable[Path]:
    """Yield application source files while skipping generated/vendor paths."""
    for path in repo_root.rglob("*"):
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

            imports = extract_imports(tree.root_node, source_bytes)
            global_imports.update(imp for imp in imports if imp)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Reachability: failed to analyze %s: %s",
                source_file,
                exc,
            )

    return global_imports


def analyze_reachability(groups: list[VulnerabilityGroup], repo_root: str | Path) -> None:
    """Mutate SCA groups in place with a deterministic reachability signal."""
    root = Path(repo_root)
    if not root.exists():
        logger.warning("Reachability: repo root does not exist: %s", root)
        return

    direct_deps = _load_direct_dependencies(root)
    global_imports = _collect_global_imports(root)

    for group in groups:
        if group.issue_type != IssueType.SCA:
            continue

        component = (group.vulnerable_component or "").strip()
        if component and any(component in imported for imported in global_imports):
            group.is_reachable = True
        elif component in direct_deps:
            group.is_reachable = False
        else:
            group.is_reachable = None
