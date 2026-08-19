"""Run the sanitize-html retriage fixture against a local Juice Shop clone."""

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
_DEFAULT_GROUPS = Path(__file__).resolve().parent / "triaged_groups_retriage.json"
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-retriage-result.json"
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-retriage.patch"


def load_retriage_groups(
    path: Path = _DEFAULT_GROUPS,
) -> list[VulnerabilityGroup]:
    """Load and validate the single sanitize-html retriage group.

    Args:
        path: JSON fixture containing the pre-triaged group.

    Returns:
        The validated sanitize-html vulnerability group.

    Raises:
        FileNotFoundError: If the group fixture does not exist.
        ValueError: If the fixture does not contain exactly the intended group.
        json.JSONDecodeError: If the fixture is not valid JSON.
        pydantic.ValidationError: If the group violates its contract.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Retriage fixture must contain exactly one group.")

    groups = [VulnerabilityGroup.model_validate(item) for item in payload]
    group = groups[0]
    if group.vulnerable_component != "sanitize-html":
        raise ValueError(
            "Retriage fixture must contain sanitize-html; "
            f"found {group.vulnerable_component!r}."
        )
    if len(group.issues) != 7:
        raise ValueError(
            "Retriage sanitize-html group must contain seven baseline issues; "
            f"found {len(group.issues)}."
        )
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
    """Execute the retriage fixture and persist its result and unified patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", _DEFAULT_REPO)),
    )
    parser.add_argument("--groups", type=Path, default=_DEFAULT_GROUPS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--patch-out", type=Path, default=_DEFAULT_PATCH)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.repo.is_dir():
        logger.error("Repository does not exist: %s", args.repo)
        return 2
    if not args.groups.is_file():
        logger.error("Retriage groups fixture does not exist: %s", args.groups)
        return 2

    groups = load_retriage_groups(args.groups.resolve())
    baseline_issues = _baseline_issues_from_groups(groups)
    logger.info(
        "Starting sanitize-html retriage with %d group(s) and %d baseline issue(s).",
        len(groups),
        len(baseline_issues),
    )

    result = run_remediation(
        RemediationRequest(
            repo_root=args.repo.resolve(),
            valid_groups=groups,
            issues=baseline_issues,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result.model_dump(exclude={"raw_state"}), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.patch_out.parent.mkdir(parents=True, exist_ok=True)
    args.patch_out.write_text(result.diff, encoding="utf-8")
    logger.info("status=%s changed_files=%s", result.status, result.changed_files)
    return 0 if result.status == "completed" and not result.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
