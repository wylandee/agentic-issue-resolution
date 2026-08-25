"""Run the remediation engine multiple times against NodeGoat with sampled package batches."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "NodeGoat"
_DEFAULT_BASELINE = Path(__file__).resolve().parent / "fixtures" / "baseline_issues.jsonl"
_DEFAULT_SUPPRESSED_ISSUES = (
    Path(__file__).resolve().parent / "fixtures" / "suppressed" / "odc_suppressed_issues.jsonl"
)
_DEFAULT_SUPPRESSIONS_XML = (
    Path(__file__).resolve().parent / "fixtures" / "suppressed" / "suppressions.xml"
)
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "trajectories"

logger = logging.getLogger("nodegoat_batch_runner")


@dataclass
class IterationSummary:
    """Summary record for an individual batch remediation run."""

    iteration: int
    selected_packages: list[str]
    issue_count: int
    status: str
    changed_files: list[str]
    errors: list[str]
    result_path: str | None = None
    patch_path: str | None = None
    timestamp: str = ""


def load_baseline_issues(path: Path) -> list[VulnerabilityIssue]:
    """Load canonical JSONL issues from the baseline fixture (read-only).

    Args:
        path: Path to the baseline JSONL issues file.

    Returns:
        List of parsed VulnerabilityIssue models.
    """
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        return [VulnerabilityIssue.model_validate(item) for item in payload]
    return [
        VulnerabilityIssue.model_validate_json(line) for line in text.splitlines() if line.strip()
    ]


def get_unique_packages(issues: list[VulnerabilityIssue]) -> list[str]:
    """Extract sorted unique package names from a list of vulnerability issues.

    Args:
        issues: List of vulnerability issues.

    Returns:
        Sorted list of distinct package names.
    """
    packages = {
        issue.package_name.strip()
        for issue in issues
        if issue.package_name and issue.package_name.strip()
    }
    return sorted(packages)


def sample_package_batches(
    packages: list[str],
    *,
    batch_size: int = 3,
    num_batches: int = 10,
    seed: int | None = None,
) -> list[list[str]]:
    """Sample distinct, non-repeating batches of packages.

    Args:
        packages: Pool of available package names.
        batch_size: Number of packages per batch.
        num_batches: Number of batches to generate.
        seed: Optional random seed for reproducible sampling.

    Returns:
        List of batches, where each batch is a sorted list of package names.

    Raises:
        ValueError: If batch_size exceeds the total number of packages.
    """
    if len(packages) < batch_size:
        raise ValueError(f"Cannot sample batch of size {batch_size} from {len(packages)} packages")

    rng = random.Random(seed)
    seen_batches: set[tuple[str, ...]] = set()
    batches: list[list[str]] = []

    # Maximum unique combinations possible
    max_combinations = 1
    for i in range(batch_size):
        max_combinations = max_combinations * (len(packages) - i) // (i + 1)
    target_count = min(num_batches, max_combinations)

    while len(batches) < target_count:
        sampled = tuple(sorted(rng.sample(packages, batch_size)))
        if sampled not in seen_batches:
            seen_batches.add(sampled)
            batches.append(list(sampled))

    return batches


def filter_issues_for_packages(
    issues: list[VulnerabilityIssue],
    selected_packages: set[str],
) -> list[VulnerabilityIssue]:
    """Filter issues to those belonging only to selected packages.

    Args:
        issues: All baseline issues.
        selected_packages: Set of active package names.

    Returns:
        Filtered list of issues matching selected packages.
    """
    return [
        issue
        for issue in issues
        if issue.package_name and issue.package_name.strip() in selected_packages
    ]


def write_suppressed_issues(
    issues: list[VulnerabilityIssue],
    output_path: Path,
) -> None:
    """Write filtered issues to the suppressed issues JSONL fixture.

    Args:
        issues: Filtered issues for active packages.
        output_path: Destination path for odc_suppressed_issues.jsonl.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(issue.model_dump(mode="json", exclude_none=False)) for issue in issues]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_package_name_from_url(url_pattern: str) -> str | None:
    """Extract normalized package name from a Dependency-Check packageUrl regex.

    Args:
        url_pattern: Regex pattern such as '^pkg:npm/adm\\-zip@.*$'.

    Returns:
        Unescaped package name, or None if pattern does not match.
    """
    match = re.search(r"pkg:(?:npm|javascript)/([^@]+)@", url_pattern)
    if not match:
        return None
    raw_name = match.group(1)
    return raw_name.replace(r"\-", "-").replace(r"\.", ".")


def extract_ecosystem_from_url(url_pattern: str) -> str:
    """Extract ecosystem from a packageUrl pattern (npm or javascript).

    Args:
        url_pattern: Regex pattern such as '^pkg:npm/adm\\-zip@.*$'.

    Returns:
        Ecosystem string ('npm' or 'javascript').
    """
    match = re.search(r"pkg:(npm|javascript)/", url_pattern)
    return match.group(1) if match else "npm"


def update_suppressions_xml_content(
    template_content: str,
    selected_packages: set[str],
) -> str:
    """Transform suppressions.xml content so only selected packages are unsuppressed.

    Selected packages are commented out in XML comments so Dependency-Check
    will NOT suppress them during scans/retriaging. All other packages are
    uncommented so Dependency-Check suppresses them.

    Args:
        template_content: Original XML string.
        selected_packages: Set of packages to keep active (unsuppressed).

    Returns:
        Updated XML string content.
    """
    block_pattern = re.compile(
        r"[ \t]*(?:<!--\s*)?(<suppress>\s*<notes>.*?</notes>\s*<packageUrl[^>]*>.*?</packageUrl>\s*<vulnerabilityName[^>]*>.*?</vulnerabilityName>\s*</suppress>)(?:\s*-->)?",
        re.DOTALL,
    )

    def _replace_block(match: re.Match[str]) -> str:
        inner_block = match.group(1)
        url_match = re.search(r"<packageUrl[^>]*>(.*?)</packageUrl>", inner_block)
        if not url_match:
            return match.group(0)

        raw_url = url_match.group(1)
        pkg_name = parse_package_name_from_url(raw_url)
        ecosystem = extract_ecosystem_from_url(raw_url)

        if pkg_name and pkg_name in selected_packages:
            # Comment out the suppression so this package IS scanned / retriaged
            return (
                "<!--\n"
                "    <suppress>\n"
                f"        <notes>Keep {pkg_name} as a selected NodeGoat fixture</notes>\n"
                f'        <packageUrl regex="true">{raw_url}</packageUrl>\n'
                '        <vulnerabilityName regex="true">.*</vulnerabilityName>\n'
                "    </suppress>\n"
                "-->"
            )
        # Active suppression for unselected packages
        display_name = pkg_name or "package"
        return (
            "    <suppress>\n"
            f"        <notes>Suppress all vulnerabilities for {display_name} ({ecosystem})</notes>\n"
            f'        <packageUrl regex="true">{raw_url}</packageUrl>\n'
            '        <vulnerabilityName regex="true">.*</vulnerabilityName>\n'
            "    </suppress>"
        )

    updated_xml = block_pattern.sub(_replace_block, template_content)
    return updated_xml


def update_suppressions_xml(
    template_path: Path,
    output_path: Path,
    selected_packages: set[str],
) -> None:
    """Update suppressions.xml on disk for the given selected packages.

    Args:
        template_path: Path to read existing suppressions.xml.
        output_path: Destination path for updated suppressions.xml.
        selected_packages: Set of packages to keep active.
    """
    content = template_path.read_text(encoding="utf-8")
    updated = update_suppressions_xml_content(content, selected_packages)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(updated, encoding="utf-8")


def copy_suppressions_to_repo(
    suppressions_path: Path,
    repo_root: Path,
) -> Path:
    """Copy suppressions.xml to the repository root directory.

    Args:
        suppressions_path: Path to the generated suppressions.xml.
        repo_root: Path to the target repository clone.

    Returns:
        Path to the copied suppressions file in the repository root.
    """
    destination = repo_root / "suppressions.xml"
    shutil.copy2(suppressions_path, destination)
    return destination


def run_batch_iteration(
    *,
    iteration: int,
    selected_packages: list[str],
    baseline_issues: list[VulnerabilityIssue],
    repo_root: Path,
    suppressed_issues_path: Path,
    suppressions_xml_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> IterationSummary:
    """Execute a single batch remediation iteration.

    Args:
        iteration: Iteration index (1-based).
        selected_packages: List of 3 selected packages.
        baseline_issues: Full read-only baseline issues.
        repo_root: Path to NodeGoat clone.
        suppressed_issues_path: Path for odc_suppressed_issues.jsonl.
        suppressions_xml_path: Path for suppressions.xml.
        output_dir: Output directory for result and patch files.
        dry_run: If True, skip invoking the remediation engine.

    Returns:
        IterationSummary with execution outcome details.
    """
    pkg_set = set(selected_packages)
    logger.info(
        "=== Starting Iteration %02d: packages=%s ===",
        iteration,
        selected_packages,
    )

    # 1. Filter issues for selected packages
    filtered_issues = filter_issues_for_packages(baseline_issues, pkg_set)
    logger.info(
        "Filtered %d baseline findings for active packages %s",
        len(filtered_issues),
        selected_packages,
    )

    # 2. Write new odc_suppressed_issues.jsonl fixture
    write_suppressed_issues(filtered_issues, suppressed_issues_path)

    # 3. Update suppressions.xml in fixtures
    update_suppressions_xml(suppressions_xml_path, suppressions_xml_path, pkg_set)

    # 4. Copy suppressions.xml to data/clones/NodeGoat/suppressions.xml
    copy_suppressions_to_repo(suppressions_xml_path, repo_root)

    timestamp = datetime.now(UTC).isoformat()
    result_path = output_dir / f"nodegoat-run-{iteration:02d}-result.json"
    patch_path = output_dir / f"nodegoat-run-{iteration:02d}.patch"

    if dry_run:
        logger.info("[Dry Run] Skipped run_remediation execution.")
        return IterationSummary(
            iteration=iteration,
            selected_packages=selected_packages,
            issue_count=len(filtered_issues),
            status="dry_run",
            changed_files=[],
            errors=[],
            result_path=str(result_path),
            patch_path=str(patch_path),
            timestamp=timestamp,
        )

    # 5. Run full remediation pipeline
    try:
        request = RemediationRequest(
            repo_root=repo_root,
            issues=filtered_issues,
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="javascript/nodejs",
            ),
        )
        result = run_remediation(request)

        # 6. Save iteration result and patch
        output_dir.mkdir(parents=True, exist_ok=True)
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
    suppressed_issues_path: Path = _DEFAULT_SUPPRESSED_ISSUES,
    suppressions_xml_path: Path = _DEFAULT_SUPPRESSIONS_XML,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    seed: int | None = None,
    dry_run: bool = False,
) -> list[IterationSummary]:
    """Execute the full batch workflow across multiple iterations.

    Args:
        iterations: Total number of iterations to run.
        batch_size: Number of packages per iteration.
        repo_root: Path to repository clone.
        baseline_path: Path to baseline issues fixture (read-only).
        suppressed_issues_path: Path to suppressed issues output JSONL.
        suppressions_xml_path: Path to suppressions.xml.
        output_dir: Path to directory for trajectory results and patches.
        seed: Optional random seed for reproducible sampling.
        dry_run: If True, do not invoke engine LLMs or Docker.

    Returns:
        List of IterationSummary objects for all completed iterations.
    """
    if not repo_root.is_dir():
        raise FileNotFoundError(f"Target repository not found: {repo_root}")
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline issues file not found: {baseline_path}")
    if not suppressions_xml_path.is_file():
        raise FileNotFoundError(f"Suppressions XML template not found: {suppressions_xml_path}")

    # Load baseline issues (strictly read-only)
    baseline_issues = load_baseline_issues(baseline_path)
    packages = get_unique_packages(baseline_issues)
    logger.info(
        "Loaded %d baseline issues across %d distinct packages",
        len(baseline_issues),
        len(packages),
    )

    # Generate distinct package batches
    batches = sample_package_batches(
        packages,
        batch_size=batch_size,
        num_batches=iterations,
        seed=seed,
    )
    logger.info("Planned %d distinct package batches", len(batches))

    summaries: list[IterationSummary] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, batch in enumerate(batches, start=1):
        summary = run_batch_iteration(
            iteration=idx,
            selected_packages=batch,
            baseline_issues=baseline_issues,
            repo_root=repo_root,
            suppressed_issues_path=suppressed_issues_path,
            suppressions_xml_path=suppressions_xml_path,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        summaries.append(summary)

    # Write aggregate summary JSON
    summary_path = output_dir / "nodegoat-batch-runs-summary.json"
    summary_payload = {
        "run_date": datetime.now(UTC).isoformat(),
        "total_iterations": len(summaries),
        "dry_run": dry_run,
        "results": [asdict(s) for s in summaries],
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved batch summary to %s", summary_path)

    # Print summary table
    print("\n" + "=" * 78)
    print(f"{'Iter':<5} {'Status':<15} {'Findings':<10} {'Changed':<8} {'Selected Packages'}")
    print("-" * 78)
    for s in summaries:
        pkgs = ", ".join(s.selected_packages)
        print(
            f"{s.iteration:<5} {s.status:<15} {s.issue_count:<10} {len(s.changed_files):<8} {pkgs}"
        )
    print("=" * 78 + "\n")

    return summaries


def main() -> int:
    """CLI entrypoint for batch remediation runner."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations to execute (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of packages per batch (default: 3)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", str(_DEFAULT_REPO))),
        help="Path to NodeGoat clone (default: data/clones/NodeGoat)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=_DEFAULT_BASELINE,
        help="Path to baseline issues fixture (default: examples/NodeGoat/fixtures/baseline_issues.jsonl)",
    )
    parser.add_argument(
        "--suppressed-issues",
        type=Path,
        default=_DEFAULT_SUPPRESSED_ISSUES,
        help="Path to output suppressed issues JSONL",
    )
    parser.add_argument(
        "--suppressions",
        type=Path,
        default=_DEFAULT_SUPPRESSIONS_XML,
        help="Path to suppressions.xml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to save per-iteration results and patches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible package selection",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare batches and fixtures without invoking live engine remediation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    repo_root = args.repo.expanduser().resolve()
    baseline_path = args.baseline.expanduser().resolve()
    suppressed_issues_path = args.suppressed_issues.expanduser().resolve()
    suppressions_xml_path = args.suppressions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    try:
        summaries = run_batch(
            iterations=args.iterations,
            batch_size=args.batch_size,
            repo_root=repo_root,
            baseline_path=baseline_path,
            suppressed_issues_path=suppressed_issues_path,
            suppressions_xml_path=suppressions_xml_path,
            output_dir=output_dir,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        failed = [s for s in summaries if s.status not in ("completed", "dry_run")]
        return 0 if not failed else 1
    except Exception as exc:
        logger.error("Batch run failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
