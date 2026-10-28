"""Load fixture-backed inputs and prepare isolated live golden runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import threading
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue
from remediation_engine.language import ProjectLanguage
from remediation_engine.settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "tests" / "evals" / "golden" / "end_to_end_cases.json"
DEFAULT_CLONES_ROOT = PROJECT_ROOT / "data" / "clones"
SUPPRESSION_FILENAME = "suppressions.xml"
SUPPRESSION_CONTAINER_PATH = "/scan/suppressions.xml"

_EXPECTED_CASE_COUNT = 10
_REQUIRED_FIELDS = frozenset(
    {"case_id", "input", "context", "expected_output", "expected_tools", "live"}
)
_CLONE_ECOSYSTEMS = {
    "npm": ("juice-shop", ProjectLanguage.NODEJS.value),
    "python": ("pygoat", ProjectLanguage.PYTHON.value),
    "java": ("vulnado", ProjectLanguage.JAVA.value),
}
_PURL_TYPES = {"npm": "npm", "python": "pypi", "java": "maven"}
_CASE_LOCKS: dict[str, threading.Lock] = {}
_CASE_LOCKS_GUARD = threading.Lock()
_XML_NAMESPACE = "https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.4.xsd"


@dataclass(frozen=True)
class EndToEndGolden:
    """One validated golden and its fixture-backed remediation request inputs."""

    case: Mapping[str, Any]
    repo_root: Path
    fixture_path: Path
    issues: tuple[VulnerabilityIssue, ...]
    request_target_packages: tuple[str, ...]
    requested_packages: tuple[str, ...]
    language: ProjectLanguage

    @property
    def case_id(self) -> str:
        """Return the stable golden identifier."""
        return str(self.case["case_id"])

    @property
    def expected_output(self) -> str:
        """Return the judge-facing completion contract."""
        return str(self.case["expected_output"])

    @property
    def live(self) -> Mapping[str, Any]:
        """Return the validated live-execution settings."""
        return self.case["live"]


def _contained_path(root: Path, relative: str, label: str) -> Path:
    """Resolve a dataset path and reject absolute or escaping values."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"{label} must be relative to the repository: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes the repository: {relative}")
    return resolved


@lru_cache(maxsize=8)
def _load_fixture(path: str, mtime_ns: int, size: int) -> tuple[VulnerabilityIssue, ...]:
    """Read canonical baseline issues once for a stable fixture revision."""
    del mtime_ns, size
    fixture = Path(path)
    content = fixture.read_text(encoding="utf-8")
    if not content.strip():
        return ()
    if content.lstrip().startswith("["):
        return tuple(VulnerabilityIssue.model_validate(item) for item in json.loads(content))
    return tuple(
        VulnerabilityIssue.model_validate_json(line)
        for line in content.splitlines()
        if line.strip()
    )


def _validate_case(case: Any, index: int) -> dict[str, Any]:
    """Validate shared golden fields and live execution metadata."""
    label = f"end_to_end_cases[{index}]"
    if not isinstance(case, dict):
        raise ValueError(f"{label} must be an object")
    missing = _REQUIRED_FIELDS - set(case)
    if missing:
        raise ValueError(f"{label} is missing: {', '.join(sorted(missing))}")
    for field in ("case_id", "input", "expected_output"):
        if not isinstance(case[field], str) or not case[field].strip():
            raise ValueError(f"{label}.{field} must be a non-empty string")
    if not isinstance(case["context"], list) or not all(
        isinstance(item, str) for item in case["context"]
    ):
        raise ValueError(f"{label}.context must be a list of strings")
    if case["expected_tools"] != []:
        raise ValueError(f"{label}.expected_tools must be an empty list for whole-run judging")
    live = case["live"]
    if not isinstance(live, dict):
        raise ValueError(f"{label}.live must be an object")
    ecosystem = live.get("ecosystem")
    if ecosystem not in _CLONE_ECOSYSTEMS:
        raise ValueError(f"{label}.live.ecosystem must be npm, python, or java")
    clone_dir, _language = _CLONE_ECOSYSTEMS[ecosystem]
    if live.get("clone_dir") != clone_dir:
        raise ValueError(f"{label}.live.clone_dir must be {clone_dir!r} for {ecosystem}")
    fixture = live.get("fixture_path")
    if not isinstance(fixture, str) or not fixture.strip():
        raise ValueError(f"{label}.live.fixture_path must be a non-empty repository-relative path")
    packages = live.get("target_packages")
    if (
        not isinstance(packages, list)
        or not packages
        or not all(isinstance(item, str) and item.strip() for item in packages)
    ):
        raise ValueError(f"{label}.live.target_packages must be a non-empty string list")
    if len(set(packages)) != len(packages):
        raise ValueError(f"{label}.live.target_packages contains duplicates")
    return case


def load_end_to_end_goldens(
    *,
    case_ids: list[str] | tuple[str, ...] | None = None,
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    clones_root: Path = DEFAULT_CLONES_ROOT,
) -> list[EndToEndGolden]:
    """Load, validate, and resolve selected end-to-end golden cases.

    Args:
        case_ids: Optional stable IDs to select. An omitted or empty list selects all.
        golden_path: Maintained dataset path.
        clones_root: Root containing the existing language-specific source clones.

    Returns:
        Validated cases with selected typed findings and their source repository paths.

    Raises:
        FileNotFoundError: A dataset or baseline fixture is missing.
        ValueError: Dataset shape, targets, selection, or case count is invalid.
    """
    cases = json.loads(golden_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != _EXPECTED_CASE_COUNT:
        raise ValueError(
            f"End-to-end golden dataset must contain exactly {_EXPECTED_CASE_COUNT} cases"
        )
    validated = [_validate_case(case, index) for index, case in enumerate(cases)]
    ids = [case["case_id"] for case in validated]
    if len(ids) != len(set(ids)):
        raise ValueError("End-to-end golden case IDs must be unique")
    requested = list(case_ids or ())
    if "all" in requested and len(requested) != 1:
        raise ValueError("'all' cannot be combined with individual golden case IDs")
    unknown = sorted(set(requested) - {"all", *ids})
    if unknown:
        raise ValueError("Unknown end-to-end golden case ID(s): " + ", ".join(unknown))
    root = golden_path.resolve().parents[3]
    resolved_clones_root = clones_root.expanduser().resolve()
    goldens: list[EndToEndGolden] = []
    for case in validated:
        live = case["live"]
        fixture = _contained_path(root, live["fixture_path"], f"{case['case_id']} fixture")
        if not fixture.is_file():
            raise FileNotFoundError(f"Golden fixture is missing: {fixture}")
        repo = resolved_clones_root / live["clone_dir"]
        cache_key = (str(fixture), fixture.stat().st_mtime_ns, fixture.stat().st_size)
        baseline = _load_fixture(*cache_key)
        targets = {package.casefold() for package in live["target_packages"]}
        fixture_ecosystem = _PURL_TYPES[live["ecosystem"]]
        issues = tuple(
            issue
            for issue in baseline
            if (issue.package_name or "").strip().casefold() in targets
            and (issue.ecosystem or "").strip().casefold() == fixture_ecosystem
        )
        found = {(issue.package_name or "").strip().casefold() for issue in issues}
        missing_targets = sorted(target for target in targets if target not in found)
        if missing_targets:
            raise ValueError(
                f"{case['case_id']} targets are missing from {fixture.name}: "
                + ", ".join(missing_targets)
            )
        canonical_targets = tuple(sorted({(issue.package_name or "").strip() for issue in issues}))
        if not issues:
            raise ValueError(f"{case['case_id']} selects no baseline findings")
        language = ProjectLanguage(_CLONE_ECOSYSTEMS[live["ecosystem"]][1])
        goldens.append(
            EndToEndGolden(
                case=case,
                repo_root=repo,
                fixture_path=fixture,
                issues=issues,
                request_target_packages=canonical_targets,
                requested_packages=tuple(live["target_packages"]),
                language=language,
            )
        )
    if not requested or requested == ["all"]:
        return goldens
    selected_ids = set(requested)
    return [golden for golden in goldens if golden.case_id in selected_ids]


def _purl_component(ecosystem: str, package: str) -> str:
    """Convert a canonical package name into an encoded package-URL component."""
    if ecosystem == "npm":
        if package.startswith("@") and "/" in package:
            scope, name = package[1:].split("/", 1)
            return f"%40{quote(scope, safe='')}%2F{quote(name, safe='')}"
        return quote(package, safe=".-_~")
    if ecosystem == "python":
        return quote(re.sub(r"[-_.]+", "-", package).lower(), safe=".-_~")
    group, artifact = package.split(":", maxsplit=1)
    return f"{quote(group, safe='.-_~')}/{quote(artifact, safe='.-_~')}"


def generate_suppressions_xml(golden: EndToEndGolden) -> str:
    """Create an ODC suppression document that leaves only target PURLs active."""
    ecosystem = str(golden.live["ecosystem"])
    active = "|".join(
        re.escape(_purl_component(ecosystem, package)) for package in golden.requested_packages
    )
    # Scoped npm names occur in ODC as %40scope%2Fname; accept hex case variants.
    package_type = _PURL_TYPES[ecosystem]
    regex = rf"(?i)^pkg:{re.escape(package_type)}/(?!(?:{active})@)[^@]+@.*$"
    root = ET.Element("suppressions", xmlns=_XML_NAMESPACE)
    suppress = ET.SubElement(root, "suppress")
    ET.SubElement(
        suppress, "notes"
    ).text = f"End-to-end golden {golden.case_id}: suppress non-target {ecosystem} packages"
    ET.SubElement(suppress, "packageUrl", {"regex": "true"}).text = regex
    ET.SubElement(suppress, "vulnerabilityName", {"regex": "true"}).text = ".*"
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _with_suppression_argument(extra_args: str) -> str:
    """Replace caller-provided ODC suppression paths while retaining other flags."""
    tokens = shlex.split(extra_args) if extra_args else []
    retained: list[str] = []
    skip_value = False
    for token in tokens:
        if skip_value:
            skip_value = False
            continue
        if token in {"--suppression", "--suppression-file"}:
            skip_value = True
        elif token.startswith("--suppression=") or token.startswith("--suppression-file="):
            continue
        else:
            retained.append(token)
    retained.extend(("--suppression", SUPPRESSION_CONTAINER_PATH))
    return shlex.join(retained)


@contextmanager
def isolated_golden_workspace(golden: EndToEndGolden) -> Iterator[None]:
    """Install a case-specific suppression file and restore the clone afterward.

    A cross-process file lock serializes cases that share the same clone. The
    original file's bytes and mode are restored after engine or judge failures.
    """
    repo = golden.repo_root.resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"Golden repository clone is missing: {repo}")
    suppression_path = repo / SUPPRESSION_FILENAME
    if suppression_path.is_symlink():
        raise ValueError(f"Refusing to replace symlinked suppression file: {suppression_path}")
    lock_root = repo.parents[1] / "evals" / "locks"
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"golden-{hashlib.sha256(str(repo).encode()).hexdigest()}.lock"
    lock_key = str(lock_path)
    with _CASE_LOCKS_GUARD:
        lock = _CASE_LOCKS.setdefault(lock_key, threading.Lock())
    with lock, lock_path.open("a+b") as lock_file:
        _lock_file(lock_file)
        original_bytes = suppression_path.read_bytes() if suppression_path.exists() else None
        original_mode = suppression_path.stat().st_mode if original_bytes is not None else None
        try:
            suppression_path.write_text(generate_suppressions_xml(golden), encoding="utf-8")
            yield
        finally:
            if original_bytes is None:
                suppression_path.unlink(missing_ok=True)
            else:
                suppression_path.write_bytes(original_bytes)
                if original_mode is not None:
                    suppression_path.chmod(original_mode)
            _unlock_file(lock_file)


def _lock_file(file: Any) -> None:
    """Acquire a portable exclusive lock on one byte of a persistent lock file."""
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        if not file.read(1):
            file.seek(0)
            file.write(b"0")
            file.flush()
        deadline = time.monotonic() + 24 * 60 * 60
        while True:
            file.seek(0)
            try:
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out waiting for golden clone lock") from exc
                time.sleep(0.1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX)


def _unlock_file(file: Any) -> None:
    """Release the portable exclusive lock."""
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def golden_settings(settings: AppSettings) -> AppSettings:
    """Return live settings with golden scan suppression and retriage behavior."""
    from dataclasses import replace

    return replace(
        settings,
        remedy_disable_post_qa_triage=True,
        odc_extra_args=_with_suppression_argument(settings.odc_extra_args),
    )


def golden_fixture_identity(golden: EndToEndGolden) -> dict[str, str]:
    """Return a stable repository-relative fixture name and content digest."""
    resolved = golden.fixture_path.resolve()
    try:
        fixture_path = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        fixture_path = str(resolved)
    return {
        "fixture_path": fixture_path,
        "fixture_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def build_golden_request(golden: EndToEndGolden):
    """Build the scoped development request without injecting expected outcomes."""
    from remediation_engine.api import RemediationRequest

    return RemediationRequest(
        repo_root=golden.repo_root,
        issues=list(golden.issues),
        target_packages=list(golden.request_target_packages),
        system_context=SystemContext(
            public_facing=True,
            deployment_os="linux",
            deployment_architecture="containerized",
            environment="development",
            primary_language=golden.language.value,
        ),
    )
