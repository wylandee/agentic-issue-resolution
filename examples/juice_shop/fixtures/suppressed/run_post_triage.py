"""Run pre-processed triaged groups against a local Juice Shop clone with informative logging."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import VulnerabilityGroup, VulnerabilityIssue


def _baseline_issues_from_groups(
    groups: list[VulnerabilityGroup],
) -> list[VulnerabilityIssue]:
    """Return the de-duplicated issue baseline represented by pre-triaged groups.

    Args:
        groups: Pre-triaged groups that will be supplied to remediation.

    Returns:
        The issue objects represented by ``groups``, preserving fixture order.

    Raises:
        ValueError: If a group has no issue payload to establish its baseline.
    """
    issues_by_id: dict[str, VulnerabilityIssue] = {}
    for group in groups:
        if not group.issues:
            raise ValueError(f"Pre-triaged group {group.group_id} has no baseline issues.")
        for issue in group.issues:
            issues_by_id.setdefault(str(issue.id), issue)
    return list(issues_by_id.values())


def main() -> int:
    """Execute pre-processed remediation fixture with standard logging and output persistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", "data/clones/juice-shop")),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=Path(__file__).parent / "triaged_groups_suppressed.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trajectories/juice-shop-suppressed-result.json"),
    )
    parser.add_argument(
        "--patch-out",
        type=Path,
        default=Path("data/trajectories/juice-shop-suppressed.patch"),
    )
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.repo.is_dir():
        logging.error("Repository does not exist: %s", args.repo)
        return 2

    if not args.groups.is_file():
        logging.error("Suppressed groups fixture does not exist: %s", args.groups)
        return 2

    raw_groups = json.loads(args.groups.read_text(encoding="utf-8"))
    groups = [VulnerabilityGroup.model_validate(item) for item in raw_groups]
    baseline_issues = _baseline_issues_from_groups(groups)

    logging.info(
        "Starting remediation with %d pre-processed vulnerability group(s) and %d baseline issue(s)...",
        len(groups),
        len(baseline_issues),
    )

    result = run_remediation(
        RemediationRequest(
            repo_root=args.repo,
            valid_groups=groups,
            issues=baseline_issues,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(exclude={"raw_state"}), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.patch_out.write_text(result.diff, encoding="utf-8")

    logging.info("status=%s changed_files=%s", result.status, result.changed_files)
    return 0 if result.status == "completed" and not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
