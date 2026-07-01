"""
SCA Manifest Locator — finds *where* a vulnerable package is declared.

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
* ``normalize_package_name`` — handles tgz/jar/scoped/colon-version/PURL forms.
* ``parse_lockfile_path`` — parses ODC lockfile-path ancestry notation.
* ``_find_nearest_manifest`` — walks up from the ODC file path to the closest
  ``package.json`` still inside the repo root.
* ``detect_package_manager`` — npm / yarn / pnpm heuristic.
* ``_run_package_lock_generation`` — generates a lockfile when absent so that
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote  # noqa: F401 — kept for callers that may use it

from src.contracts import (
    IssueSource,
    IssueType,
    LocalizedIssue,
    VulnerabilityIssue,
)
from src.contracts.schemas import CWEEntry  # noqa: F401 — re-exported for convenience

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
    - ``lodash-4.17.21.tgz`` → ``lodash``
    - ``@tootallnate/once:1.1.2`` → ``@tootallnate/once``
    - ``pkg:npm/%40tootallnate%2Fonce@1.1.2`` → ``@tootallnate/once``
    - ``bench.js`` → ``bench`` (non-npm file — kept as-is after ext strip)
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
    if name.startswith("@") and "/" in name and not re.search(r"\.(tgz|jar|zip)$", name, re.IGNORECASE):
        return name.strip()

    # Remove archive/js suffixes
    name = re.sub(r"\.(tgz|jar|zip|js)$", "", name, flags=re.IGNORECASE)

    # Remove trailing version segment: -1.2.3, _1.2.3, .1.2.3, -v1.2.3
    name = re.sub(r"[-_.]v?\d+(?:\.\d+)*(?:[-+][A-Za-z0-9._-]+)?$", "", name)

    return name.strip()


def _package_name_from_purl(purl_str: str) -> Optional[str]:
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


def parse_lockfile_path(file_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
        → (``/src/package-lock.json``, None, ``lodash``)

    ``/src/package-lock.json?pdfkit:0.11.0/crypto-js:^3.1.9-1``
        → (``/src/package-lock.json``, ``pdfkit``, ``crypto-js``)

    ``/src/node_modules/express-jwt/node_modules/moment/moment.js``
        → (None, ``express-jwt``, ``moment``)
    """
    m = _LOCKFILE_PATH_RE.match(file_path)
    if m:
        lockfile = m.group("lockfile")
        ancestry = m.group("ancestry").lstrip("/")

        # Split ancestry into package segments.
        # A segment is either:
        #   - a plain token like "cookie:0.4.2"
        #   - a scoped token like "@tootallnate/once:1.1.2"  (starts with @)
        # We must NOT split scoped tokens on the '/' inside them.
        segments: List[str] = []
        remaining = ancestry
        while remaining:
            if remaining.startswith("@"):
                # consume up to next '/' that is NOT part of the scope
                # e.g. "@tootallnate/once:1.1.2/next-pkg:2.0"
                # The scope+name is everything up to ':version' then optional '/'
                # We find the end of this token by locating the first '/' that comes
                # AFTER the mandatory scope separator.
                slash_scope = remaining.index("/")  # separator between @scope and name
                rest_after_scope = remaining[slash_scope + 1:]  # "once:1.1.2/next..."
                next_slash = rest_after_scope.find("/")
                if next_slash == -1:
                    segments.append(remaining)
                    break
                token = remaining[: slash_scope + 1 + next_slash]
                segments.append(token)
                remaining = rest_after_scope[next_slash + 1:]
            else:
                slash_idx = remaining.find("/")
                if slash_idx == -1:
                    segments.append(remaining)
                    break
                segments.append(remaining[:slash_idx])
                remaining = remaining[slash_idx + 1:]

        if not segments:
            return lockfile, None, None

        leaf = normalize_package_name(segments[-1])
        parent_raw = segments[-2] if len(segments) >= 2 else None
        parent = normalize_package_name(parent_raw) if parent_raw else None

        return lockfile, parent, leaf

    # Not a lockfile path — may be a node_modules file path.
    # Strategy: find all 'node_modules/<immediate-package-name>' occurrences.
    # Each node_modules entry is the directory immediately after 'node_modules/'.
    # e.g. /src/node_modules/express-jwt/node_modules/moment/moment.js
    #       → packages: ['express-jwt', 'moment'], leaf='moment', parent='express-jwt'
    nm_match = re.findall(r"node_modules/([^/]+)", file_path)
    if nm_match:
        leaf = nm_match[-1]
        parent = nm_match[-2] if len(nm_match) >= 2 else None
        return None, parent, leaf

    return None, None, None


# ---------------------------------------------------------------------------
# Nearest-manifest resolution
# ---------------------------------------------------------------------------


def _find_nearest_manifest(repo_root: Path, odc_file_path: str) -> Optional[Path]:
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
        candidate = (repo_root / variant)
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
    NPM = "npm"
    YARN = "yarn"
    PNPM = "pnpm"


def detect_package_manager(manifest_dir: Path) -> str:
    """
    Detect the active package manager for a given manifest directory.

    Heuristic priority:
    1. ``pnpm-lock.yaml`` → pnpm
    2. ``yarn.lock`` → yarn
    3. ``package-lock.json`` → npm
    4. Default → npm
    """
    if (manifest_dir / "pnpm-lock.yaml").exists():
        return PackageManagerKind.PNPM
    if (manifest_dir / "yarn.lock").exists():
        return PackageManagerKind.YARN
    return PackageManagerKind.NPM


# ---------------------------------------------------------------------------
# Manifest read helpers
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return {}


def _find_dependency_line(lines: List[str], package_name: str) -> Optional[int]:
    """Find 1-indexed line number of a dependency declaration in package.json text."""
    pattern = re.compile(rf'^\s*"{re.escape(package_name)}"\s*:\s*')
    for idx, line in enumerate(lines, start=1):
        if pattern.search(line):
            return idx
    return None


def _build_snippet(lines: List[str], line_number: int, context: int = 1) -> str:
    start = max(1, line_number - context)
    end = min(len(lines), line_number + context)
    return "\n".join(lines[start - 1: end])


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


# ---------------------------------------------------------------------------
# Core locator logic
# ---------------------------------------------------------------------------


def _locate_in_manifest(
    manifest_path: Path,
    package_name: str,
    pkg_manager: str,
) -> Dict[str, Any]:
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
    all_direct = {**dependencies, **dev_dependencies, **peer_dependencies}

    is_direct = package_name in all_direct

    result: Dict[str, Any] = {
        "manifest_file": str(manifest_path),
        "package_name": package_name,
        "is_direct": is_direct,
        "line_number": None,
        "snippet": None,
        "package_manager": pkg_manager,
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
) -> Dict[str, Any]:
    """
    Locate where a vulnerable SCA dependency is declared in the repository.

    Parameters
    ----------
    repo_path:
        Absolute path to the cloned repository root.
    raw_dependency_name:
        Raw name from ODC (e.g. ``lodash-4.17.21.tgz``, ``@scope/pkg:1.2.3``).
    odc_file_path:
        The ``filePath`` field from ODC — used to resolve the nearest manifest
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

    # 3. Detect package manager
    pkg_manager = detect_package_manager(manifest_path.parent)

    # 4. Locate in manifest (pure localization — no fix planning)
    location = _locate_in_manifest(manifest_path, effective_name, pkg_manager)

    return {
        "status": "success",
        **location,
        "lockfile_ancestry": {
            "lockfile": lockfile_rel,
            "ancestor_pkg": ancestor_pkg,
            "leaf_pkg": leaf_pkg,
        },
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
    manifest_rel: Optional[str] = None
    if is_success and result.get("manifest_file"):
        try:
            manifest_rel = str(Path(result["manifest_file"]).relative_to(repo_path))
        except ValueError:
            manifest_rel = result["manifest_file"]

    # Confidence heuristic: direct + line found → high; transitive → medium; error → low
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
        localization_confidence=confidence,
    )
