"""Run the parent-first transitive dependency fixture against Juice Shop.

This manual scenario loads three pre-triaged transitive SCA groups and runs the
normal post-triage workflow. Each group names a directly declared parent, so
the Supervisor should try parent OSV-minimum, same-major, and latest releases
before it can commit a package-manager override for the vulnerable child.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import VulnerabilityGroup, VulnerabilityIssue

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_GROUPS = Path(__file__).resolve().parent / "triaged_groups_transitive_parent_first.json"


def load_parent_first_groups(path: Path = _DEFAULT_GROUPS) -> list[VulnerabilityGroup]:
    """Load and validate the three transitive parent-first groups.

    Args:
        path: JSON fixture containing the pre-triaged SCA groups.

    Returns:
        Exactly three validated transitive vulnerability groups.

    Raises:
        FileNotFoundError: If the fixture path does not exist.
        ValueError: If the fixture does not contain the expected packages or
            parent metadata.
        json.JSONDecodeError: If the fixture is not valid JSON.
    """
    raw_groups = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_groups, list) or len(raw_groups) != 3:
        raise ValueError("Parent-first fixture must contain exactly three groups.")

    groups = [VulnerabilityGroup.model_validate(item) for item in raw_groups]
    expected = {"@tootallnate/once", "got", "crypto-js"}
    actual = {group.vulnerable_component for group in groups}
    if actual != expected:
        raise ValueError(f"Expected transitive packages {sorted(expected)}, got {sorted(actual)}.")
    if any(not group.parent_package_name for group in groups):
        raise ValueError("Every parent-first fixture group must name a direct parent.")
    if any(group.parent_declaration_type != "dependencies" for group in groups):
        raise ValueError("Every parent-first fixture group must target dependencies.")
    if any(
        not any(not issue.is_direct_dependency for issue in group.localized_issues)
        for group in groups
    ):
        raise ValueError("Every parent-first fixture group must be localized as transitive.")
    return groups


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
    """Execute the parent-first fixture with standard logging and output persistence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            os.environ.get("TEST_REPO_ROOT", _PROJECT_ROOT / "data" / "clones" / "juice-shop")
        ),
    )
    parser.add_argument("--groups", type=Path, default=_DEFAULT_GROUPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/trajectories/juice-shop-transitive-parent-first-result.json"),
    )
    parser.add_argument(
        "--patch-out",
        type=Path,
        default=Path("data/trajectories/juice-shop-transitive-parent-first.patch"),
    )
    args = parser.parse_args()
    repo_root = args.repo.expanduser().resolve()
    groups_path = args.groups.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    patch_path = args.patch_out.expanduser().resolve()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not repo_root.is_dir():
        logger.error("Repository does not exist: %s", repo_root)
        return 2
    if not groups_path.is_file():
        logger.error("Parent-first groups fixture does not exist: %s", groups_path)
        return 2

    groups = load_parent_first_groups(groups_path)
    baseline_issues = _baseline_issues_from_groups(groups)
    logger.info(
        "Starting parent-first remediation with packages: %s and %d baseline issue(s)",
        ", ".join(sorted(group.vulnerable_component or "unknown" for group in groups)),
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
    logger.info("status=%s changed_files=%s", result.status, result.changed_files)
    return 0 if result.status == "completed" and not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
