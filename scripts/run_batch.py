"""Run bounded, profile-scoped remediation batches from canonical issue JSONL."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shlex
import shutil
import sys
import tempfile
import uuid
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote, unquote

from remediation_engine import RemediationRequest, RemediationResult, run_remediation
from remediation_engine.contracts.schemas import IssueType, SystemContext, VulnerabilityIssue
from remediation_engine.language import ProjectLanguage
from remediation_engine.settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "evals" / "batch-runs"
SUPPRESSION_CONTAINER_PATH = "/scan/suppressions.xml"
SUPPRESSION_FILENAME = "suppressions.xml"
_SUPPRESSION_NAMESPACE = (
    "https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.4.xsd"
)
_IGNORED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".workspaces",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "htmlcov",
        "node_modules",
        "target",
        "venv",
    }
)


@dataclass(frozen=True, slots=True)
class CodebaseProfile:
    """Baseline and ecosystem rules for one supported repository."""

    name: str
    language: ProjectLanguage
    ecosystems: frozenset[str]
    purl_types: tuple[str, ...]
    default_repo: Path
    default_baseline: Path


def _profile(
    name: str,
    language: ProjectLanguage,
    ecosystems: frozenset[str],
    purl_types: tuple[str, ...],
    repo_name: str,
    example_name: str,
) -> CodebaseProfile:
    """Build a profile rooted in the repository's four retained examples."""
    example = PROJECT_ROOT / "examples" / example_name
    return CodebaseProfile(
        name=name,
        language=language,
        ecosystems=ecosystems,
        purl_types=purl_types,
        default_repo=PROJECT_ROOT / "data" / "clones" / repo_name,
        default_baseline=example / "baseline_issues.jsonl",
    )


PROFILES = {
    "juice_shop": _profile(
        "juice_shop",
        ProjectLanguage.NODEJS,
        frozenset({"npm", "javascript"}),
        ("npm", "javascript"),
        "juice-shop",
        "juice_shop",
    ),
    "nodegoat": _profile(
        "nodegoat",
        ProjectLanguage.NODEJS,
        frozenset({"npm", "javascript"}),
        ("npm", "javascript"),
        "NodeGoat",
        "nodegoat",
    ),
    "vulnado": _profile(
        "vulnado",
        ProjectLanguage.JAVA,
        frozenset({"maven"}),
        ("maven",),
        "vulnado",
        "vulnado",
    ),
    "pygoat": _profile(
        "pygoat",
        ProjectLanguage.PYTHON,
        frozenset({"pypi", "python"}),
        ("pypi",),
        "pygoat",
        "pygoat",
    ),
}


def load_baseline_issues(path: Path) -> list[VulnerabilityIssue]:
    """Load canonical JSONL findings without modifying the source file.

    Args:
        path: Baseline JSONL path.

    Returns:
        Validated findings in file order.

    Raises:
        FileNotFoundError: If the baseline does not exist.
        ValueError: If a non-empty record is invalid.
    """
    content = path.read_text(encoding="utf-8")
    issues: list[VulnerabilityIssue] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            issues.append(VulnerabilityIssue.model_validate_json(line))
        except ValueError as exc:
            raise ValueError(f"Invalid baseline issue at {path}:{line_number}: {exc}") from exc
    return issues


def _purl_type(issue: VulnerabilityIssue) -> str | None:
    """Return a normalized package type from a PURL, when present."""
    purl = issue.purl or ""
    match = re.match(r"^pkg:([^/]+)/", purl.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else None


def _maven_coordinate(issue: VulnerabilityIssue) -> str | None:
    """Return a Maven coordinate only when its PURL confirms both components."""
    purl = issue.purl or ""
    if _purl_type(issue) != "maven":
        return None
    match = re.match(r"^pkg:maven/([^/]+)/([^/@?#]+)@[^/?#]+", purl, flags=re.IGNORECASE)
    if match is None or not issue.package_name:
        return None
    coordinate = f"{unquote(match.group(1))}:{unquote(match.group(2))}"
    return coordinate if issue.package_name.strip() == coordinate else None


def _package_key(profile: CodebaseProfile, package_name: str) -> str:
    """Normalize package spelling for profile-specific selection matching."""
    name = package_name.strip()
    if profile.name == "pygoat":
        return re.sub(r"[-_.]+", "-", name).lower()
    if profile.language == ProjectLanguage.NODEJS:
        return name.casefold()
    return name


def _issue_key(profile: CodebaseProfile, issue: VulnerabilityIssue) -> str | None:
    """Return the actionable package key for a finding in this profile."""
    if issue.issue_type != IssueType.SCA or not issue.package_name:
        return None
    ecosystem = (issue.ecosystem or "").strip().lower()
    purl_type = _purl_type(issue)
    if profile.language == ProjectLanguage.JAVA:
        coordinate = _maven_coordinate(issue)
        return _package_key(profile, coordinate) if coordinate else None
    if ecosystem not in profile.ecosystems and purl_type not in profile.purl_types:
        return None
    if purl_type is not None and purl_type not in profile.purl_types:
        return None
    return _package_key(profile, issue.package_name)


def actionable_packages(
    profile: CodebaseProfile,
    issues: Sequence[VulnerabilityIssue],
) -> dict[str, str]:
    """Map normalized package keys to deterministic baseline spellings."""
    packages: dict[str, str] = {}
    for issue in issues:
        key = _issue_key(profile, issue)
        if key is None or not issue.package_name:
            continue
        name = issue.package_name.strip()
        if key not in packages or name < packages[key]:
            packages[key] = name
    return dict(sorted(packages.items(), key=lambda item: (item[0], item[1])))


def filter_issues(
    profile: CodebaseProfile,
    issues: Sequence[VulnerabilityIssue],
    selected_keys: set[str],
) -> list[VulnerabilityIssue]:
    """Keep profile-actionable findings for the selected package keys."""
    return [
        issue
        for issue in issues
        if (key := _issue_key(profile, issue)) is not None and key in selected_keys
    ]


def sample_package_batches(
    packages: Sequence[str],
    *,
    batch_size: int,
    iterations: int,
    seed: int,
) -> list[list[str]]:
    """Create deterministic, unique package batches with finite bounds.

    Args:
        packages: Canonical package names.
        batch_size: Number of packages in each batch.
        iterations: Number of distinct batches requested.
        seed: Random sampling seed.

    Returns:
        Sorted package names in each seeded, distinct batch.

    Raises:
        ValueError: If sizes are non-positive, the package pool is too small,
            or the request exceeds the finite number of unique batches.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    pool = sorted(set(packages))
    if batch_size > len(pool):
        raise ValueError(
            f"Cannot sample batch of size {batch_size} from {len(pool)} actionable packages"
        )
    combinations = math.comb(len(pool), batch_size)
    if iterations > combinations:
        raise ValueError(
            f"Cannot create {iterations} distinct batches of size {batch_size}; "
            f"only {combinations} combinations exist"
        )
    rng = random.Random(seed)
    sampled: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    while len(sampled) < iterations:
        batch = tuple(sorted(rng.sample(pool, batch_size)))
        if batch not in seen:
            seen.add(batch)
            sampled.append(list(batch))
    return sampled


def _selected_batches(
    profile: CodebaseProfile,
    issues: Sequence[VulnerabilityIssue],
    *,
    packages: Sequence[str] | None,
    iterations: int,
    batch_size: int,
    seed: int,
) -> tuple[list[list[str]], dict[str, str]]:
    """Resolve CLI package spellings before any output or snapshot is written."""
    available = actionable_packages(profile, issues)
    if not available:
        raise ValueError(f"Baseline contains no actionable packages for {profile.name}")
    if packages is not None:
        if not packages or any(not name.strip() for name in packages):
            raise ValueError("--packages requires one or more non-empty names")
        normalized = [_package_key(profile, name) for name in packages]
        if len(set(normalized)) != len(normalized):
            raise ValueError("--packages contains duplicate package names")
        unknown = [
            name for name, key in zip(packages, normalized, strict=True) if key not in available
        ]
        if unknown:
            raise ValueError(
                f"Unknown or unsupported {profile.name} package(s): {', '.join(unknown)}"
            )
        canonical = [available[key] for key in normalized]
        return [canonical], available
    batches = sample_package_batches(
        list(available.values()), batch_size=batch_size, iterations=iterations, seed=seed
    )
    return batches, available


def _jsonl(issues: Sequence[VulnerabilityIssue]) -> str:
    """Serialize canonical issues as JSONL while retaining all model fields."""
    return "".join(issue.model_dump_json(exclude_none=False) + "\n" for issue in issues)


def _encoded_purl_component(profile: CodebaseProfile, package_name: str, purl_type: str) -> str:
    """Encode a selected package name as a Dependency-Check PURL component."""
    if purl_type == "maven":
        group, artifact = package_name.split(":", maxsplit=1)
        return f"{quote(group, safe='.-_~')}/{quote(artifact, safe='.-_~')}"
    if profile.name == "pygoat":
        package_name = re.sub(r"[-_.]+", "-", package_name).lower()
    return quote(package_name, safe=".-_~")


def generate_suppressions_xml(profile: CodebaseProfile, packages: Sequence[str]) -> str:
    """Build scoped ODC suppressions that keep only selected package PURLs active."""
    if not packages:
        raise ValueError("at least one package is required for scoped suppression")
    root = ET.Element("suppressions", xmlns=_SUPPRESSION_NAMESPACE)
    for purl_type in profile.purl_types:
        active = "|".join(
            re.escape(_encoded_purl_component(profile, package, purl_type)) for package in packages
        )
        suppress = ET.SubElement(root, "suppress")
        ET.SubElement(
            suppress, "notes"
        ).text = f"Batch {profile.name}: suppress non-target {purl_type} packages"
        ET.SubElement(
            suppress, "packageUrl", {"regex": "true"}
        ).text = rf"(?i)^pkg:{re.escape(purl_type)}/(?!(?:{active})@)[^@]+@.*$"
        ET.SubElement(suppress, "vulnerabilityName", {"regex": "true"}).text = ".*"
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _with_suppression_argument(extra_args: str) -> str:
    """Replace caller-provided suppression paths with the isolated workspace path."""
    tokens = shlex.split(extra_args) if extra_args else []
    retained: list[str] = []
    skip_value = False
    for token in tokens:
        if skip_value:
            skip_value = False
        elif token in {"--suppression", "--suppression-file"}:
            skip_value = True
        elif token.startswith("--suppression=") or token.startswith("--suppression-file="):
            continue
        else:
            retained.append(token)
    retained.extend(("--suppression", SUPPRESSION_CONTAINER_PATH))
    return shlex.join(retained)


def _snapshot_ignore(_directory: str, names: list[str]) -> set[str]:
    """Exclude secrets and runtime/vendor files from an isolated source snapshot."""
    ignored = {name for name in names if name in _IGNORED_DIRECTORIES or name.endswith(".pyc")}
    ignored.update(
        name
        for name in names
        if name == ".env" or (name.startswith(".env.") and name != ".env.example")
    )
    return ignored


def _run_directory(output_root: Path) -> Path:
    """Create a collision-resistant, run-owned output directory."""
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:10]
    path = output_root / f"run-{run_id}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_result_files(
    batch_dir: Path,
    issues: Sequence[VulnerabilityIssue],
    result: RemediationResult | None,
    *,
    dry_run: bool,
    error: str | None = None,
) -> dict[str, object]:
    """Persist one batch's selected findings, typed result and patch."""
    issues_path = batch_dir / "issues.jsonl"
    result_path = batch_dir / "result.json"
    patch_path = batch_dir / "patch.diff"
    issues_path.write_text(_jsonl(issues), encoding="utf-8")
    if result is None:
        payload: dict[str, object] = {
            "status": "dry_run" if dry_run else "failed",
            "changed_files": [],
            "errors": [error] if error else [],
            "diff": "",
        }
    else:
        payload = result.model_dump(mode="json", exclude={"raw_state"})
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    patch_path.write_text(str(payload.get("diff") or ""), encoding="utf-8")
    errors = payload.get("errors", [])
    return {
        "status": payload.get("status", "failed"),
        "issue_count": len(issues),
        "changed_files": payload.get("changed_files", []),
        "errors": errors if isinstance(errors, list) else [str(errors)],
        "issues_path": issues_path.relative_to(batch_dir.parent).as_posix(),
        "result_path": result_path.relative_to(batch_dir.parent).as_posix(),
        "patch_path": patch_path.relative_to(batch_dir.parent).as_posix(),
    }


def run_batch(
    codebase: str,
    *,
    repo: Path | None = None,
    baseline: Path | None = None,
    packages: Sequence[str] | None = None,
    iterations: int = 1,
    batch_size: int = 1,
    seed: int = 0,
    output_dir: Path | None = None,
    dry_run: bool = False,
    settings: AppSettings | None = None,
) -> dict[str, object]:
    """Run explicit or seeded package batches and persist typed outcomes.

    Args:
        codebase: One of the four supported profile names.
        repo: Optional caller-owned source repository. Live runs use an isolated
            temporary snapshot and never write into this directory.
        baseline: Optional canonical JSONL baseline override.
        packages: Optional explicit package names for exactly one batch.
        iterations: Number of distinct sampled batches when packages are absent.
        batch_size: Packages per sampled batch.
        seed: Deterministic random seed.
        output_dir: Parent directory for a collision-resistant run directory.
        dry_run: Prepare input/result files without checking a clone or services.
        settings: Optional explicit settings dependency for live runs.

    Returns:
        Run summary with per-batch result and artifact paths.

    Raises:
        ValueError: If profile, package selection, sampling, or output safety
            validation fails before creating output files.
        FileNotFoundError: If the selected baseline or live repository is absent.
    """
    try:
        profile = PROFILES[codebase]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported codebase {codebase!r}; choose from {', '.join(PROFILES)}"
        ) from exc
    baseline_path = (baseline or profile.default_baseline).expanduser().resolve()
    issues = load_baseline_issues(baseline_path)
    batches, available = _selected_batches(
        profile,
        issues,
        packages=packages,
        iterations=iterations,
        batch_size=batch_size,
        seed=seed,
    )
    selected_issue_sets = [
        filter_issues(profile, issues, {_package_key(profile, name) for name in batch})
        for batch in batches
    ]
    if any(not selected for selected in selected_issue_sets):
        raise ValueError("each selected batch must match at least one baseline finding")

    repo_path = (repo or profile.default_repo).expanduser().resolve()
    output_root = (output_dir or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    resolved_settings = settings
    if not dry_run:
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Live repository directory does not exist: {repo_path}")
        if output_root == repo_path or output_root.is_relative_to(repo_path):
            raise ValueError("output_dir must be outside the caller-owned repository")
        if resolved_settings is None:
            from dotenv import load_dotenv

            load_dotenv(PROJECT_ROOT / ".env")
            resolved_settings = AppSettings.from_env()
        # Parse this before creating the output directory so malformed ODC
        # arguments cannot leave a partial run behind.
        suppression_args = _with_suppression_argument(resolved_settings.odc_extra_args)
        resolved_settings = replace(resolved_settings, odc_extra_args=suppression_args)

    run_dir = _run_directory(output_root)
    batch_summaries: list[dict[str, object]] = []
    for index, (package_batch, selected_issues) in enumerate(
        zip(batches, selected_issue_sets, strict=True), start=1
    ):
        batch_dir = run_dir / f"batch-{index:03d}"
        batch_dir.mkdir()
        error: str | None = None
        result: RemediationResult | None = None
        if not dry_run:
            try:
                if resolved_settings is None:
                    raise RuntimeError("live settings were not initialized")
                with tempfile.TemporaryDirectory(prefix="remedy-batch-snapshot-") as temp_root:
                    snapshot = Path(temp_root) / "repository"
                    shutil.copytree(repo_path, snapshot, symlinks=True, ignore=_snapshot_ignore)
                    suppression_path = snapshot / SUPPRESSION_FILENAME
                    if suppression_path.is_symlink():
                        raise ValueError(
                            "Refusing to replace a symlinked suppression file in snapshot"
                        )
                    suppression_path.write_text(
                        generate_suppressions_xml(profile, package_batch), encoding="utf-8"
                    )
                    request = RemediationRequest(
                        repo_root=snapshot,
                        issues=selected_issues,
                        target_packages=package_batch,
                        system_context=SystemContext(
                            public_facing=True,
                            deployment_os="linux",
                            deployment_architecture="containerized",
                            environment="development",
                            primary_language=profile.language.value,
                        ),
                    )
                    result = run_remediation(request, settings=resolved_settings)
            except Exception as exc:  # Preserve a failed iteration and continue.
                error = f"{type(exc).__name__}: {exc}"
        batch_summary = _write_result_files(
            batch_dir,
            selected_issues,
            result,
            dry_run=dry_run,
            error=error,
        )
        batch_summary["iteration"] = index
        batch_summary["selected_packages"] = package_batch
        batch_summaries.append(batch_summary)

    summary: dict[str, object] = {
        "codebase": profile.name,
        "language": profile.language.value,
        "baseline_path": str(baseline_path),
        "baseline_issue_count": len(issues),
        "actionable_package_count": len(available),
        "dry_run": dry_run,
        "seed": seed if packages is None else None,
        "requested_iterations": iterations if packages is None else 1,
        "batch_size": batch_size if packages is None else len(batches[0]),
        "batch_count": len(batch_summaries),
        "batches": batch_summaries,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["run_dir"] = str(run_dir)
    return summary


def _positive_int(value: str) -> int:
    """Parse a positive CLI integer."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Build the common batch-runner CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebase", choices=tuple(PROFILES), required=True)
    parser.add_argument("--repo", type=Path, help="Caller-owned repository directory for live runs")
    parser.add_argument("--baseline", type=Path, help="Canonical issue JSONL override")
    parser.add_argument("--packages", nargs="+", help="Explicit package names for one batch")
    parser.add_argument(
        "--iterations", type=_positive_int, help="Number of distinct sampled batches"
    )
    parser.add_argument("--batch-size", type=_positive_int, help="Packages in each sampled batch")
    parser.add_argument("--seed", type=int, help="Deterministic sampling seed (default: 0)")
    parser.add_argument(
        "--output-dir", type=Path, help="Output parent (default: data/evals/batch-runs)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Prepare batches without a clone or services"
    )
    parser.add_argument(
        "--list-packages", action="store_true", help="List actionable baseline packages"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the CLI request, execute it, and return a process status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    profile = PROFILES[args.codebase]
    baseline = args.baseline or profile.default_baseline
    try:
        issues = load_baseline_issues(baseline)
        available = actionable_packages(profile, issues)
        if args.list_packages:
            if (
                args.packages is not None
                or args.iterations is not None
                or args.batch_size is not None
            ):
                parser.error("--list-packages cannot be combined with batch-selection options")
            print("\n".join(available.values()))
            return 0
        if args.packages is not None and any(
            value is not None for value in (args.iterations, args.batch_size, args.seed)
        ):
            parser.error("--packages cannot be combined with --iterations, --batch-size, or --seed")
        summary = run_batch(
            args.codebase,
            repo=args.repo,
            baseline=baseline,
            packages=args.packages,
            iterations=args.iterations or 1,
            batch_size=args.batch_size or 1,
            seed=args.seed if args.seed is not None else 0,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Saved {summary['batch_count']} batches to {summary['run_dir']}")
    batches = summary["batches"]
    if not isinstance(batches, list):
        return 1
    if args.dry_run:
        return 0
    return (
        0
        if all(batch.get("status") == "completed" and not batch.get("errors") for batch in batches)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
