"""Run the remediation engine against sampled PyGoat package batches."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "pygoat"
_DEFAULT_BASELINE = Path(__file__).resolve().parent / "fixtures" / "baseline_issues.jsonl"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "trajectories"
_SUMMARY_FILENAME = "pygoat-batch-runs-summary.json"

logger = logging.getLogger("pygoat_batch_runner")


@dataclass
class IterationSummary:
    """Summary record for an individual PyGoat package-batch run."""

    iteration: int
    selected_packages: list[str]
    issue_count: int
    status: str
    changed_files: list[str]
    errors: list[str]
    issues_path: str
    result_path: str | None = None
    patch_path: str | None = None
    timestamp: str = ""


def load_baseline_issues(path: Path) -> list[VulnerabilityIssue]:
    """Load canonical JSONL or JSON-array issues from a baseline fixture.

    Args:
        path: Path to the read-only baseline issues file.

    Returns:
        Parsed vulnerability issues.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain valid issue data.
    """
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    if content.lstrip().startswith("["):
        payload = json.loads(content)
        return [VulnerabilityIssue.model_validate(item) for item in payload]
    return [
        VulnerabilityIssue.model_validate_json(line)
        for line in content.splitlines()
        if line.strip()
    ]


def get_unique_packages(issues: list[VulnerabilityIssue]) -> list[str]:
    """Return sorted package names represented in the baseline issues.

    Args:
        issues: Baseline vulnerability issues.

    Returns:
        Sorted, de-duplicated, non-empty package names.
    """
    return sorted(
        {
            issue.package_name.strip()
            for issue in issues
            if issue.package_name and issue.package_name.strip()
        }
    )


def sample_package_batches(
    packages: list[str],
    *,
    batch_size: int = 3,
    num_batches: int = 10,
    seed: int | None = None,
) -> list[list[str]]:
    """Sample distinct package batches without replacement between batches.

    Args:
        packages: Pool of package names.
        batch_size: Number of packages in each batch.
        num_batches: Maximum number of distinct batches to produce.
        seed: Optional seed for reproducible sampling.

    Returns:
        Distinct batches, with package names sorted inside each batch. Fewer
        batches are returned when there are not enough unique combinations.

    Raises:
        ValueError: If ``batch_size`` or ``num_batches`` is not positive, or
            if there are too few packages for the requested batch size.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if num_batches < 1:
        raise ValueError("num_batches must be at least 1")

    package_pool = sorted({package.strip() for package in packages if package and package.strip()})
    if len(package_pool) < batch_size:
        raise ValueError(
            f"Cannot sample batch of size {batch_size} from {len(package_pool)} packages"
        )

    target_count = min(num_batches, math.comb(len(package_pool), batch_size))
    rng = random.Random(seed)
    seen_batches: set[tuple[str, ...]] = set()
    batches: list[list[str]] = []
    while len(batches) < target_count:
        sampled = tuple(sorted(rng.sample(package_pool, batch_size)))
        if sampled not in seen_batches:
            seen_batches.add(sampled)
            batches.append(list(sampled))
    return batches


def filter_issues_for_packages(
    issues: list[VulnerabilityIssue],
    selected_packages: set[str],
) -> list[VulnerabilityIssue]:
    """Keep only issues whose package name is selected.

    Args:
        issues: Full baseline issues.
        selected_packages: Package names active in this batch.

    Returns:
        Issues matching one of the selected packages.
    """
    selected = {package.strip() for package in selected_packages if package.strip()}
    return [
        issue for issue in issues if issue.package_name and issue.package_name.strip() in selected
    ]


def write_issues_jsonl(issues: list[VulnerabilityIssue], output_path: Path) -> None:
    """Write a canonical JSONL issue subset for later inspection or replay.

    Args:
        issues: Issues to serialize.
        output_path: Destination JSONL path. Parent directories are created.

    Side effects:
        Creates or replaces ``output_path``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(issue.model_dump(mode="json", exclude_none=False)) for issue in issues]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_batch_iteration(
    *,
    iteration: int,
    selected_packages: list[str],
    baseline_issues: list[VulnerabilityIssue],
    repo_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> IterationSummary:
    """Run one selected package batch and persist its inputs and outputs.

    Args:
        iteration: One-based iteration index.
        selected_packages: Packages selected for this iteration.
        baseline_issues: Full baseline issue set; not modified.
        repo_root: PyGoat checkout used by live remediation runs.
        output_dir: Directory for batch inputs, results, and patches.
        dry_run: If true, write the batch input but skip remediation.

    Returns:
        Summary of the batch result.

    Side effects:
        Writes a per-iteration canonical JSONL input. Live runs also call the
        remediation API and write its result JSON and proposed unified patch.
    """
    filtered_issues = filter_issues_for_packages(baseline_issues, set(selected_packages))
    issues_path = output_dir / f"pygoat-run-{iteration:02d}-issues.jsonl"
    result_path = output_dir / f"pygoat-run-{iteration:02d}-result.json"
    patch_path = output_dir / f"pygoat-run-{iteration:02d}.patch"
    timestamp = datetime.now(UTC).isoformat()
    write_issues_jsonl(filtered_issues, issues_path)

    if dry_run:
        logger.info(
            "[Dry Run] Prepared %d issues for packages %s", len(filtered_issues), selected_packages
        )
        return IterationSummary(
            iteration=iteration,
            selected_packages=selected_packages,
            issue_count=len(filtered_issues),
            status="dry_run",
            changed_files=[],
            errors=[],
            issues_path=str(issues_path),
            result_path=str(result_path),
            patch_path=str(patch_path),
            timestamp=timestamp,
        )

    try:
        request = RemediationRequest(
            repo_root=repo_root,
            issues=filtered_issues,
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="python",
            ),
        )
        result = run_remediation(request)
        result_path.write_text(
            json.dumps(result.model_dump(exclude={"raw_state"}), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        patch_path.write_text(result.diff, encoding="utf-8")

        logger.info(
            "Iteration %02d finished: status=%s changed_files=%s errors=%d",
            iteration,
            result.status,
            result.changed_files,
            len(result.errors),
        )
        return IterationSummary(
            iteration=iteration,
            selected_packages=selected_packages,
            issue_count=len(filtered_issues),
            status=result.status,
            changed_files=result.changed_files,
            errors=result.errors,
            issues_path=str(issues_path),
            result_path=str(result_path),
            patch_path=str(patch_path),
            timestamp=timestamp,
        )
    except Exception as exc:
        logger.exception("Iteration %02d failed with unhandled exception: %s", iteration, exc)
        return IterationSummary(
            iteration=iteration,
            selected_packages=selected_packages,
            issue_count=len(filtered_issues),
            status="failed",
            changed_files=[],
            errors=[str(exc)],
            issues_path=str(issues_path),
            result_path=str(result_path),
            patch_path=str(patch_path),
            timestamp=timestamp,
        )


def run_batch(
    *,
    iterations: int = 10,
    batch_size: int = 3,
    repo_root: Path = _DEFAULT_REPO,
    baseline_path: Path = _DEFAULT_BASELINE,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    seed: int | None = None,
    dry_run: bool = False,
) -> list[IterationSummary]:
    """Run distinct package batches from a read-only baseline fixture.

    Args:
        iterations: Maximum number of distinct package batches to run.
        batch_size: Number of packages selected for each batch.
        repo_root: PyGoat checkout used for live runs.
        baseline_path: Canonical baseline issues file; never modified.
        output_dir: Directory for per-iteration artifacts and summary JSON.
        seed: Optional seed for reproducible package selection.
        dry_run: If true, prepare batch JSONL inputs without running remediation.

    Returns:
        Summaries for all planned iterations.

    Raises:
        FileNotFoundError: If the baseline is missing, or if a live run's
            repository is missing.
        ValueError: If batch arguments are invalid or the baseline has too few
            unique packages for the requested batch size.
    """
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline issues file not found: {baseline_path}")
    if not dry_run and not repo_root.is_dir():
        raise FileNotFoundError(f"Target repository not found: {repo_root}")

    baseline_issues = load_baseline_issues(baseline_path)
    packages = get_unique_packages(baseline_issues)
    logger.info(
        "Loaded %d baseline issues across %d distinct packages",
        len(baseline_issues),
        len(packages),
    )
    batches = sample_package_batches(
        packages,
        batch_size=batch_size,
        num_batches=iterations,
        seed=seed,
    )
    logger.info("Planned %d distinct package batches", len(batches))

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_batch_iteration(
            iteration=index,
            selected_packages=batch,
            baseline_issues=baseline_issues,
            repo_root=repo_root,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        for index, batch in enumerate(batches, start=1)
    ]

    summary_path = output_dir / _SUMMARY_FILENAME
    summary_payload = {
        "run_date": datetime.now(UTC).isoformat(),
        "requested_iterations": iterations,
        "total_iterations": len(summaries),
        "batch_size": batch_size,
        "seed": seed,
        "dry_run": dry_run,
        "results": [asdict(summary) for summary in summaries],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved batch summary to %s", summary_path)

    print("\n" + "=" * 78)
    print(f"{'Iter':<5} {'Status':<15} {'Findings':<10} {'Changed':<8} {'Selected Packages'}")
    print("-" * 78)
    for summary in summaries:
        packages_text = ", ".join(summary.selected_packages)
        print(
            f"{summary.iteration:<5} {summary.status:<15} {summary.issue_count:<10} "
            f"{len(summary.changed_files):<8} {packages_text}"
        )
    print("=" * 78 + "\n")
    return summaries


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for PyGoat batch remediation runs.

    Args:
        argv: Optional command-line arguments, primarily for tests.

    Returns:
        ``0`` when all runs complete without errors, ``1`` if any iteration
        fails, and ``2`` when inputs or batch arguments are invalid.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10, help="Maximum number of batches")
    parser.add_argument("--batch-size", type=int, default=3, help="Packages selected per batch")
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", str(_DEFAULT_REPO))),
        help="Path to PyGoat clone",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_DEFAULT_BASELINE,
        help="Path to canonical baseline issues JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for per-batch inputs, results, patches, and summary",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible package selection"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare per-batch JSONL inputs without running remediation",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:
        summaries = run_batch(
            iterations=args.iterations,
            batch_size=args.batch_size,
            repo_root=args.repo.expanduser().resolve(),
            baseline_path=args.baseline.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            seed=args.seed,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.error("Batch run failed: %s", exc)
        return 2

    succeeded = all(
        summary.status == "dry_run" or (summary.status == "completed" and not summary.errors)
        for summary in summaries
    )
    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
