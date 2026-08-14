"""
SCA Manifest Locator â€” finds *where* a vulnerable package is declared.

Separation-of-Concerns refactor
---------------------------------
This module is a **pure locator**.  It answers:

  "In which file, on which line, and with which package manager is this
   dependency declared?"

It does **not**:
  * Call external APIs (OSV, GitHub Advisory, etc.)
  * Suggest fix versions
  * Generate natural-language instructions

Fix planning (safe-version lookup, override block generation, PR body) is the
responsibility of the downstream "Plan Fix" agent which receives a typed
``LocalizedIssue`` from this tool.

Key capabilities preserved
--------------------------
* ``normalize_package_name`` â€” handles tgz/jar/scoped/colon-version/PURL forms.
* ``parse_lockfile_path`` â€” parses ODC lockfile-path ancestry notation.
* ``_find_nearest_manifest`` â€” walks up from the ODC file path to the closest
  ``package.json`` still inside the repo root.
* ``detect_package_manager`` â€” npm / yarn / pnpm heuristic.
* ``_run_package_lock_generation`` â€” generates a lockfile when absent so that
  transitive paths can be resolved.
* Line-number and snippet extraction for direct dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import unquote  # noqa: F401 â€” kept for callers that may use it

from semantic_version import NpmSpec, Version

from remediation_engine.contracts import (
    LocalizedIssue,
    VulnerabilityIssue,
)
from remediation_engine.contracts.schemas import (
    CWEEntry,  # noqa: F401 â€” re-exported for convenience
)

try:
    from packageurl import PackageURL

    _PURL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PURL_AVAILABLE = False


log = logging.getLogger(__name__)

_LOCKFILE_PATH_RE = re.compile(
    r"^(?P<lockfile>.+?(?:package-lock\.json|yarn\.lock|pnpm-lock\.yaml))"
    r"[?#](?P<ancestry>.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Package-name normalisation
# ---------------------------------------------------------------------------


def normalize_package_name(raw_name: str) -> str:
    """Normalise noisy ODC identifiers into a bare package name.

    Handles:
    - ``lodash-4.17.21.tgz`` â†’ ``lodash``
    - ``@tootallnate/once:1.1.2`` â†’ ``@tootallnate/once``
    - ``pkg:npm/%40tootallnate%2Fonce@1.1.2`` â†’ ``@tootallnate/once``
    - ``bench.js`` â†’ ``bench`` (non-npm file â€” kept as-is after ext strip)
    """
    name = (raw_name or "").strip()
    if not name:
        return ""

    # PURL form: pkg:npm/...
    if name.startswith("pkg:"):
        extracted = _package_name_from_purl(name)
        return extracted or ""

    # Strip version suffix introduced by ODC colon notation (@pkg:ver or name:ver)
    # but preserve scoped packages like @scope/pkg
    if name.startswith("@"):
        # e.g. @tootallnate/once:1.1.2
        colon_idx = name.rfind(":")
        slash_idx = name.index("/") if "/" in name else -1
        if colon_idx > slash_idx:
            name = name[:colon_idx]
    else:
        name = name.split(":")[0]

    # Strip parenthetical suffixes  "name (comment)"
    name = re.sub(r"\s*\(.*?\)", "", name)

    # Keep clean scoped packages intact
    if (
        name.startswith("@")
        and "/" in name
        and not re.search(r"\.(tgz|jar|zip)$", name, re.IGNORECASE)
    ):
        return name.strip()

    # Remove archive/js suffixes
    name = re.sub(r"\.(tgz|jar|zip|js)$", "", name, flags=re.IGNORECASE)

    # Remove trailing version segment: -1.2.3, _1.2.3, .1.2.3, -v1.2.3
    name = re.sub(r"[-_.]v?\d+(?:\.\d+)*(?:[-+][A-Za-z0-9._-]+)?$", "", name)

    return name.strip()


def _package_name_from_purl(purl_str: str) -> str | None:
    """Extract decoded package name (with namespace) from a PURL string.

    For npm: packageurl-python puts '@scope/name' directly in purl.name
    with namespace=None, so we just return name.
    For other ecosystems with a namespace, we join with ':'.
    """
    if _PURL_AVAILABLE:
        try:
            purl = PackageURL.from_string(purl_str)
            name = purl.name or ""
            ns = purl.namespace or ""
            if not name:
                return None
            if purl.type == "npm":
                # name already contains '@scope/pkg' for scoped packages
                return name
            if ns:
                return f"{ns}:{name}"
            return name
        except Exception:
            pass
    # Manual fallback
    try:
        segment = purl_str.split("/", 2)[-1]
        return segment.split("@")[0].replace("%40", "@").replace("%2F", "/") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ODC lockfile-path ancestry parsing
# ---------------------------------------------------------------------------


def parse_lockfile_path(file_path: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse ODC's composite file-path notation.

    Returns ``(lockfile_rel, immediate_parent_pkg, leaf_pkg)`` where:
    - ``lockfile_rel`` is the lockfile path portion (e.g. ``/src/package-lock.json``)
    - ``immediate_parent_pkg`` is the direct parent in the ancestry chain
      (``None`` for root-level entries)
    - ``leaf_pkg`` is the package name at the end of the ancestry chain

    Examples
    --------
    ``/src/package-lock.json?/lodash:2.4.2``
        â†’ (``/src/package-lock.json``, None, ``lodash``)

    ``/src/package-lock.json?pdfkit:0.11.0/crypto-js:^3.1.9-1``
        â†’ (``/src/package-lock.json``, ``pdfkit``, ``crypto-js``)

    ``/src/node_modules/express-jwt/node_modules/moment/moment.js``
        â†’ (None, ``express-jwt``, ``moment``)
    """
    parsed = _parse_dependency_segments(file_path)
    if parsed is not None:
        lockfile, segments = parsed
        if not segments:
            return lockfile, None, None
        leaf = segments[-1][0]
        parent = segments[-2][0] if len(segments) >= 2 else None
        return lockfile, parent, leaf

    # Not a lockfile path â€” may be a node_modules file path.
    # Strategy: find all 'node_modules/<immediate-package-name>' occurrences.
    # Each node_modules entry is the directory immediately after 'node_modules/'.
    # e.g. /src/node_modules/express-jwt/node_modules/moment/moment.js
    #       â†’ packages: ['express-jwt', 'moment'], leaf='moment', parent='express-jwt'
    segments = _node_modules_dependency_segments(file_path)
    if segments:
        leaf = segments[-1][0]
        parent = segments[-2][0] if len(segments) >= 2 else None
        return None, parent, leaf

    return None, None, None


def _split_lockfile_ancestry(ancestry: str) -> list[str]:
    """Split lockfile ancestry tokens without breaking scoped package names."""
    segments: list[str] = []
    remaining = ancestry.lstrip("/")
    while remaining:
        if remaining.startswith("@"):
            scope_separator = remaining.find("/")
            if scope_separator < 0:
                segments.append(remaining)
                break
            after_scope = remaining[scope_separator + 1 :]
            next_separator = after_scope.find("/")
            if next_separator < 0:
                segments.append(remaining)
                break
            segments.append(remaining[: scope_separator + 1 + next_separator])
            remaining = after_scope[next_separator + 1 :]
            continue

        separator = remaining.find("/")
        if separator < 0:
            segments.append(remaining)
            break
        segments.append(remaining[:separator])
        remaining = remaining[separator + 1 :]
    return segments


def _package_token(token: str) -> tuple[str, str | None]:
    """Return ``(package_name, resolved_version)`` from a scanner ancestry token."""
    raw = token.strip()
    version: str | None = None
    if raw.startswith("@"):
        colon = raw.rfind(":")
        slash = raw.find("/")
        if colon > slash:
            version = raw[colon + 1 :].strip() or None
    elif ":" in raw:
        raw, version = raw.split(":", 1)
        version = version.strip() or None
    return normalize_package_name(raw), version


def _node_modules_dependency_segments(file_path: str) -> list[tuple[str, str | None]]:
    """Extract nested ``node_modules`` package names from a filesystem path."""
    parts = [part for part in file_path.replace("\\", "/").split("/") if part]
    segments: list[tuple[str, str | None]] = []
    index = 0
    while index < len(parts):
        if parts[index] != "node_modules" or index + 1 >= len(parts):
            index += 1
            continue
        package = parts[index + 1]
        consumed = 1
        if package.startswith("@") and index + 2 < len(parts):
            package = f"{package}/{parts[index + 2]}"
            consumed = 2
        # A package directory may legitimately end in .js (for example
        # fast.js); unlike a file identifier, it must not be normalized as an
        # archive/source filename.
        segments.append((package.strip(), None))
        index += consumed + 1
    return [(name, version) for name, version in segments if name]


def _parse_dependency_segments(
    file_path: str,
) -> tuple[str, list[tuple[str, str | None]]] | None:
    """Parse a lockfile path into its lockfile and ordered dependency chain."""
    match = _LOCKFILE_PATH_RE.match(file_path or "")
    if not match:
        return None
    lockfile = match.group("lockfile")
    tokens = _split_lockfile_ancestry(match.group("ancestry"))
    segments: list[tuple[str, str | None]] = []
    for token in tokens:
        parsed = _package_token(token)
        if parsed[0]:
            segments.append(parsed)
    return lockfile, segments


def parse_dependency_ancestry(file_path: str) -> list[tuple[str, str | None]]:
    """Return the ordered dependency ancestry encoded in a scanner path.

    Args:
        file_path: ODC lockfile ancestry or a nested ``node_modules`` path.

    Returns:
        A list of ``(package_name, resolved_version)`` pairs ordered from the
        outermost package to the vulnerable leaf. Unknown versions are ``None``.
    """
    parsed = _parse_dependency_segments(file_path)
    if parsed is not None:
        return parsed[1]
    return _node_modules_dependency_segments(file_path)


# ---------------------------------------------------------------------------
# Nearest-manifest resolution
# ---------------------------------------------------------------------------


def _find_nearest_manifest(repo_root: Path, odc_file_path: str) -> Path | None:
    """
    Walk up from the directory implied by the ODC file path to find the closest
    ``package.json`` that is still inside ``repo_root``.

    For lockfile-path notation (``/src/package-lock.json?...``), the directory
    of the lockfile is used as the starting point.
    """
    if not odc_file_path.strip():
        return None

    # Strip the lockfile ancestry suffix if present
    raw_path = odc_file_path.split("?")[0].split("#")[0].strip()

    # Try to make it relative to repo_root. ODC often records paths with a
    # leading /src/ that maps to the repo root, so we prefer the stripped form
    # first and fall back to the raw form if needed.
    candidate_abs = Path(raw_path)
    has_root_prefix = raw_path.startswith(("/", "\\"))
    if candidate_abs.is_absolute() or has_root_prefix:
        rel_parts = candidate_abs.parts[1:]  # drop the leading "/" or "\"
    else:
        rel_parts = candidate_abs.parts

    variants = [Path(*rel_parts)] if rel_parts else []
    if rel_parts and rel_parts[0].lower() in {"src", repo_root.name.lower()}:
        stripped = Path(*rel_parts[1:])
        if str(stripped):
            variants.insert(0, stripped)

    start_dir = repo_root.resolve()
    repo_resolved = repo_root.resolve()
    fallback_dir = start_dir
    fallback_depth = -1
    for variant in variants:
        candidate = repo_root / variant
        current = candidate.resolve()
        if current.exists():
            start_dir = current if current.is_dir() else current.parent
            break
        while True:
            try:
                current.relative_to(repo_resolved)
            except ValueError:
                break

            if current.exists():
                resolved_dir = current if current.is_dir() else current.parent
                try:
                    depth = len(resolved_dir.relative_to(repo_resolved).parts)
                except ValueError:
                    depth = -1
                if depth > fallback_depth:
                    fallback_dir = resolved_dir
                    fallback_depth = depth
                break

            if current == repo_resolved:
                break
            current = current.parent
    else:
        if fallback_depth >= 0:
            start_dir = fallback_dir

    # Walk up from start_dir looking for package.json
    current = start_dir
    while True:
        manifest = current / "package.json"
        if manifest.exists():
            return manifest
        if current == repo_resolved or not current.is_relative_to(repo_resolved):
            break
        current = current.parent

    return None


# ---------------------------------------------------------------------------
# Package-manager detection
# ---------------------------------------------------------------------------


class PackageManagerKind:
    """Known JavaScript package-manager lockfile conventions."""

    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"


def detect_package_manager(manifest_dir: Path) -> str:
    """
    Detect the active package manager for a given manifest directory.

    Heuristic priority:
    1. ``pnpm-lock.yaml`` â†’ pnpm
    2. ``yarn.lock`` â†’ yarn
    3. ``package-lock.json`` â†’ npm
    4. Default â†’ npm
    """
    if (manifest_dir / "pnpm-lock.yaml").exists():
        return PackageManagerKind.PNPM
    if (manifest_dir / "yarn.lock").exists():
        return PackageManagerKind.YARN
    return PackageManagerKind.NPM


# ---------------------------------------------------------------------------
# Manifest read helpers
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return {}


def _find_dependency_line(lines: list[str], package_name: str) -> int | None:
    """Find 1-indexed line number of a dependency declaration in package.json text."""
    pattern = re.compile(rf'^\s*"{re.escape(package_name)}"\s*:\s*')
    for idx, line in enumerate(lines, start=1):
        if pattern.search(line):
            return idx
    return None


def _build_snippet(lines: list[str], line_number: int, context: int = 1) -> str:
    start = max(1, line_number - context)
    end = min(len(lines), line_number + context)
    return "\n".join(lines[start - 1 : end])


def _run_package_lock_generation(repo_path: Path) -> None:
    log.info("Running npm install --package-lock-only in %s", repo_path)
    npm_executable = shutil.which("npm") or "npm"
    is_windows = os.name == "nt"
    try:
        subprocess.run(
            [npm_executable, "install", "--package-lock-only"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True,
            shell=is_windows,
        )
        log.info("package-lock.json generation completed")
    except subprocess.CalledProcessError as e:
        log.error("npm install --package-lock-only failed: %s", (e.stderr or "").strip())
    except FileNotFoundError:
        log.error("npm not found on PATH")


def _lockfile_dependency_edges(metadata: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Return dependency edges declared by one npm lockfile package entry.

    The category is retained so closure consumers can distinguish required,
    optional, and peer edges without duplicating lockfile field semantics.
    """
    edges: list[tuple[str, str, str]] = []
    for category in ("dependencies", "optionalDependencies", "peerDependencies"):
        values = metadata.get(category)
        if not isinstance(values, dict):
            continue
        for package_name, requirement in values.items():
            if isinstance(package_name, str) and isinstance(requirement, str):
                edges.append((package_name, requirement, category))
    return edges


def _lockfile_dependency_requirements(metadata: dict[str, Any]) -> dict[str, str]:
    """Return dependency ranges declared by one npm lockfile package entry.

    Args:
        metadata: One entry from the npm lockfile ``packages`` mapping.

    Returns:
        Dependency names mapped to their declared npm ranges.
    """
    return {
        package_name: requirement
        for package_name, requirement, _category in _lockfile_dependency_edges(metadata)
    }


def _lockfile_package_candidates(
    packages: dict[str, Any],
    parent_key: str,
    package_name: str,
    requirement: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Resolve one dependency from an npm lockfile ``packages`` graph.

    npm checks a package-local ``node_modules`` directory and then walks
    outward toward the repository root. Reproducing that lookup lets the
    locator recover logical dependency edges even when the scanner path only
    reflects physical package nesting.

    Args:
        packages: npm lockfile ``packages`` mapping.
        parent_key: Lockfile key for the package declaring the dependency.
        package_name: Dependency name to resolve.
        requirement: npm range declared by the parent.

    Returns:
        Matching ``(package_key, metadata)`` pairs in npm resolution order.
    """
    parent_parts = [part for part in parent_key.split("/") if part]
    scopes: list[list[str]] = [parent_parts + ["node_modules"]]
    for index in range(len(parent_parts) - 1, -1, -1):
        if parent_parts[index] == "node_modules":
            scopes.append(parent_parts[: index + 1])

    candidates: list[tuple[str, dict[str, Any]]] = []
    seen_keys: set[str] = set()
    for scope in scopes:
        key = "/".join([*scope, package_name])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        metadata = packages.get(key)
        if not isinstance(metadata, dict):
            continue
        version_text = str(metadata.get("version") or "").strip()
        if not version_text:
            continue
        try:
            if not NpmSpec(requirement).match(Version(version_text)):
                continue
        except (TypeError, ValueError):
            # Non-semver specs such as aliases and workspace ranges cannot be
            # evaluated safely here. Keep an exact resolved-version match but
            # otherwise leave the scanner path unchanged.
            if requirement.strip() != version_text:
                continue
        candidates.append((key, metadata))
    return candidates


def _expand_npm_lockfile_ancestry(
    lockfile_path: Path,
    ancestry: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Expand compressed ODC ancestry using npm lockfile dependency edges.

    Dependency-Check paths can omit logical package edges when npm hoists or
    nests the resolved package. The registry parent planner needs the logical
    chain because it evaluates each published dependency range in sequence.

    Args:
        lockfile_path: Path to the npm ``package-lock.json``.
        ancestry: Scanner-supplied path from the direct parent to the leaf.

    Returns:
        The reconstructed logical ancestry, or the original path when the
        lockfile cannot provide a deterministic expansion.
    """
    if len(ancestry) < 2 or not lockfile_path.is_file():
        return ancestry

    lockfile = _read_json_file(lockfile_path)
    packages = lockfile.get("packages")
    if not isinstance(packages, dict):
        return ancestry

    parent_name, parent_version = ancestry[0]
    leaf_name, leaf_version = ancestry[-1]
    parent_key = f"node_modules/{parent_name}"
    parent_metadata = packages.get(parent_key)
    if not isinstance(parent_metadata, dict):
        return ancestry
    if parent_version and str(parent_metadata.get("version")) != parent_version:
        return ancestry

    hint_names = [name for name, _ in ancestry]
    hint_versions = {name: version for name, version in ancestry if version}
    queue: list[tuple[str, list[tuple[str, str | None]], int]] = [
        (parent_key, [(parent_name, parent_version)], 1)
    ]
    visited: set[tuple[str, int]] = set()

    while queue:
        current_key, current_path, hint_index = queue.pop(0)
        visit_key = (current_key, hint_index)
        if visit_key in visited:
            continue
        visited.add(visit_key)

        current_metadata = packages.get(current_key)
        if not isinstance(current_metadata, dict):
            continue
        requirements = _lockfile_dependency_requirements(current_metadata)
        for dependency_name in sorted(requirements):
            requirement = requirements[dependency_name]
            for child_key, child_metadata in _lockfile_package_candidates(
                packages,
                current_key,
                dependency_name,
                requirement,
            ):
                child_version = str(child_metadata.get("version") or "") or None
                next_hint_index = hint_index
                if (
                    next_hint_index < len(hint_names)
                    and dependency_name == hint_names[next_hint_index]
                    and (
                        not hint_versions.get(dependency_name)
                        or hint_versions[dependency_name] == child_version
                    )
                ):
                    next_hint_index += 1

                child_path = [*current_path, (dependency_name, child_version)]
                if (
                    dependency_name == leaf_name
                    and (not leaf_version or child_version == leaf_version)
                    and next_hint_index == len(hint_names)
                ):
                    return child_path

                queue.append((child_key, child_path, next_hint_index))

    return ancestry


def expand_dependency_ancestry_from_repository(
    repo_root: Path,
    manifest_file: str | None,
    odc_file_path: str,
    dependency_ancestry: list[str],
    dependency_versions: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Expand a localized dependency ancestry using a repository lockfile.

    This boundary helper also supports pre-triaged groups, whose localized
    records may bypass :func:`locate_from_issue` before entering Phase 5.

    Args:
        repo_root: Absolute repository root containing the manifest.
        manifest_file: Repository-relative or absolute manifest path.
        odc_file_path: Scanner file path containing lockfile ancestry notation.
        dependency_ancestry: Package names supplied by localization or a
            preprocessed group.
        dependency_versions: Resolved versions keyed by package name.

    Returns:
        A possibly expanded ``(ancestry, versions)`` pair. If the lockfile is
        unavailable or cannot provide a deterministic path, the supplied data
        is returned unchanged.
    """
    lockfile_rel, _, _ = parse_lockfile_path(odc_file_path)
    lockfile_name = Path(lockfile_rel).name if lockfile_rel else ""
    if lockfile_name.lower() != "package-lock.json":
        return list(dependency_ancestry), dict(dependency_versions)

    manifest_path = Path(manifest_file) if manifest_file else repo_root / "package.json"
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    supplied_pairs = [(name, dependency_versions.get(name)) for name in dependency_ancestry if name]
    parsed_pairs = parse_dependency_ancestry(odc_file_path)
    if parsed_pairs and [name for name, _ in parsed_pairs] == dependency_ancestry:
        supplied_pairs = parsed_pairs

    expanded = _expand_npm_lockfile_ancestry(
        manifest_path.parent / lockfile_name,
        supplied_pairs,
    )
    return (
        [name for name, _ in expanded],
        {name: version for name, version in expanded if version},
    )


# ---------------------------------------------------------------------------
# Core locator logic
# ---------------------------------------------------------------------------


def _locate_in_manifest(
    manifest_path: Path,
    package_name: str,
    pkg_manager: str,
) -> dict[str, Any]:
    """
    Inspect a ``package.json`` and return localization data for the package.

    Returns a dict with keys:
      manifest_file, package_name, is_direct, line_number, snippet, package_manager.

    No fix instructions or version suggestions are produced here.
    """
    package_json = _read_json_file(manifest_path)
    dependencies = package_json.get("dependencies") or {}
    dev_dependencies = package_json.get("devDependencies") or {}
    peer_dependencies = package_json.get("peerDependencies") or {}
    optional_dependencies = package_json.get("optionalDependencies") or {}
    declaration_maps = (
        ("dependencies", dependencies),
        ("devDependencies", dev_dependencies),
        ("peerDependencies", peer_dependencies),
        ("optionalDependencies", optional_dependencies),
    )
    all_direct = {
        package: (declaration_type, str(spec))
        for declaration_type, declarations in declaration_maps
        for package, spec in declarations.items()
    }

    is_direct = package_name in all_direct

    result: dict[str, Any] = {
        "manifest_file": str(manifest_path),
        "package_name": package_name,
        "is_direct": is_direct,
        "declaration_type": all_direct.get(package_name, (None, None))[0],
        "declared_version": all_direct.get(package_name, (None, None))[1],
        "line_number": None,
        "snippet": None,
        "package_manager": pkg_manager,
        "direct_declarations": all_direct,
    }

    if is_direct:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        line_number = _find_dependency_line(lines, package_name)
        if line_number is not None:
            result["line_number"] = line_number
            result["snippet"] = _build_snippet(lines, line_number)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def locate_dependency(
    repo_path: Path,
    raw_dependency_name: str,
    odc_file_path: str = "",
) -> dict[str, Any]:
    """
    Locate where a vulnerable SCA dependency is declared in the repository.

    Parameters
    ----------
    repo_path:
        Absolute path to the cloned repository root.
    raw_dependency_name:
        Raw name from ODC (e.g. ``lodash-4.17.21.tgz``, ``@scope/pkg:1.2.3``).
    odc_file_path:
        The ``filePath`` field from ODC â€” used to resolve the nearest manifest
        and parse lockfile ancestry.

    Returns
    -------
    Dict with keys:
      status, manifest_file, package_name, is_direct, line_number, snippet,
      package_manager, lockfile_ancestry.

    No fix instructions or fixed-version suggestions are included.
    """
    package_name = normalize_package_name(raw_dependency_name)
    if not package_name:
        return {"status": "error", "message": "Dependency name is empty after normalisation."}

    # 1. Parse lockfile ancestry from ODC path
    lockfile_rel, ancestor_pkg, leaf_pkg = parse_lockfile_path(odc_file_path)
    dependency_ancestry = parse_dependency_ancestry(odc_file_path)
    effective_name = leaf_pkg or package_name

    # 2. Resolve nearest manifest
    manifest_path = _find_nearest_manifest(repo_path, odc_file_path) if odc_file_path else None
    if manifest_path is None:
        # Fall back to repo root
        root_manifest = repo_path / "package.json"
        manifest_path = root_manifest if root_manifest.exists() else None

    if manifest_path is None:
        return {
            "status": "error",
            "message": f"No package.json found in or above the ODC path '{odc_file_path}'.",
            "package_name": effective_name,
        }

    # ODC may report the physical lockfile nesting rather than every logical
    # dependency edge. Expand npm paths from the lockfile when possible so the
    # parent-version planner receives a complete chain. The raw scanner path
    # remains available through ``lockfile_ancestry`` for diagnostics.
    ancestry_names, dependency_versions = expand_dependency_ancestry_from_repository(
        repo_path,
        str(manifest_path),
        odc_file_path,
        [name for name, _ in dependency_ancestry],
        {name: version for name, version in dependency_ancestry if version},
    )

    # 3. Detect package manager
    pkg_manager = detect_package_manager(manifest_path.parent)

    # 4. Locate in manifest (pure localization â€” no fix planning)
    location = _locate_in_manifest(manifest_path, effective_name, pkg_manager)
    direct_declarations = location.pop("direct_declarations", {})
    parent_name: str | None = None
    parent_version: str | None = None
    parent_declaration_type: str | None = None
    parent_manifest_line: int | None = None
    parent_manifest_snippet: str | None = None
    # Walk outward from the vulnerable leaf and select the nearest package that
    # is actually declared in the editable manifest. This is the only parent
    # target that the update worker is allowed to change.
    for candidate_name, candidate_version in reversed(dependency_ancestry[:-1]):
        declaration = direct_declarations.get(candidate_name)
        if declaration is None:
            continue
        parent_name = candidate_name
        parent_version = candidate_version or dependency_versions.get(candidate_name)
        parent_declaration_type = declaration[0]
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        parent_manifest_line = _find_dependency_line(lines, candidate_name)
        if parent_manifest_line is not None:
            parent_manifest_snippet = _build_snippet(lines, parent_manifest_line)
        break

    return {
        "status": "success",
        **location,
        "lockfile_ancestry": {
            "lockfile": lockfile_rel,
            "ancestor_pkg": ancestor_pkg,
            "leaf_pkg": leaf_pkg,
        },
        "dependency_ancestry": ancestry_names,
        "dependency_versions": dependency_versions,
        "parent_package_name": parent_name,
        "parent_package_version": parent_version,
        "parent_declaration_type": parent_declaration_type,
        "parent_manifest_line": parent_manifest_line,
        "parent_manifest_snippet": parent_manifest_snippet,
    }


def locate_from_issue(
    issue: VulnerabilityIssue,
    repo_path: Path,
) -> LocalizedIssue:
    """
    High-level entry point: accept a typed ``VulnerabilityIssue``, run the
    full SCA locator, and return a typed ``LocalizedIssue``.

    No OSV enrichment or fix-instruction generation is performed here.
    """
    raw_name = issue.package_name or ""
    odc_file_path = issue.file_path or ""
    raw_payload = issue.raw_payload or {}
    odc_file_path = odc_file_path or raw_payload.get("filePath", "")

    result = locate_dependency(
        repo_path=repo_path,
        raw_dependency_name=raw_name,
        odc_file_path=odc_file_path,
    )

    is_success = result.get("status") == "success"
    manifest_rel: str | None = None
    if is_success and result.get("manifest_file"):
        try:
            manifest_rel = (
                Path(result["manifest_file"]).resolve().relative_to(repo_path.resolve()).as_posix()
            )
        except (OSError, ValueError):
            # Never leak an absolute path into the graph contract.  A
            # localization outside the scanned repository is unusable by the
            # remediation workers, so leave it unset and let grouping apply
            # its normal no-target fallback.
            manifest_rel = None

    # Confidence heuristic: direct + line found â†’ high; transitive â†’ medium; error â†’ low
    confidence: float
    if not is_success:
        confidence = 0.0
    elif result.get("is_direct") and result.get("line_number"):
        confidence = 0.95
    elif result.get("is_direct"):
        confidence = 0.75
    else:
        confidence = 0.60

    return LocalizedIssue(
        issue=issue,
        manifest_file=manifest_rel,
        is_direct_dependency=result.get("is_direct"),
        manifest_line=result.get("line_number"),
        manifest_snippet=result.get("snippet"),
        package_manager=result.get("package_manager"),
        dependency_ancestry=result.get("dependency_ancestry", []),
        dependency_versions=result.get("dependency_versions", {}),
        declaration_type=result.get("declaration_type"),
        parent_package_name=result.get("parent_package_name"),
        parent_package_version=result.get("parent_package_version"),
        parent_declaration_type=result.get("parent_declaration_type"),
        parent_manifest_line=result.get("parent_manifest_line"),
        parent_manifest_snippet=result.get("parent_manifest_snippet"),
        localization_confidence=confidence,
    )
