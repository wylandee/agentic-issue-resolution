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

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_GROUPS = Path(__file__).resolve().parent / "triaged_groups_suppressed.json"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-suppressed-result.json"
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-suppressed.patch"


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
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", str(_DEFAULT_REPO))),
    )
    parser.add_argument(
        "--groups",
        type=Path,
        default=_DEFAULT_GROUPS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--patch-out",
        type=Path,
        default=_DEFAULT_PATCH,
    )
    args = parser.parse_args()
    repo_root = args.repo.expanduser().resolve()
    groups_path = args.groups.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    patch_path = args.patch_out.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not repo_root.is_dir():
        logging.error("Repository does not exist: %s", repo_root)
        return 2

    if not groups_path.is_file():
        logging.error("Suppressed groups fixture does not exist: %s", groups_path)
        return 2

    raw_groups = json.loads(groups_path.read_text(encoding="utf-8"))
    groups = [VulnerabilityGroup.model_validate(item) for item in raw_groups]
    baseline_issues = _baseline_issues_from_groups(groups)

    logging.info(
        "Starting remediation with %d pre-processed vulnerability group(s) and %d baseline issue(s)...",
        len(groups),
        len(baseline_issues),
    )

    result = run_remediation(
        RemediationRequest(
            repo_root=repo_root,
            valid_groups=groups,
            issues=baseline_issues,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(exclude={"raw_state"}), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(result.diff, encoding="utf-8")

    logging.info("status=%s changed_files=%s", result.status, result.changed_files)
    return 0 if result.status == "completed" and not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
