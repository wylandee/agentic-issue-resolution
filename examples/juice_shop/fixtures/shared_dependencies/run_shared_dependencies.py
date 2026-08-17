"""Run the overlapping-dependency Juice Shop fixture.

The fixture supplies two pre-triaged update groups: ``express-jwt`` and its
nested ``jsonwebtoken`` dependency. Their lockfile closures overlap, allowing
the targeted QA path to be evaluated for shared-node union and provenance.
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
_DEFAULT_GROUPS = Path(__file__).resolve().parent / "triaged_groups_shared_dependencies.json"
_DEFAULT_ISSUES = Path(__file__).resolve().parent / "baseline_issues_shared_dependencies.jsonl"
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_OUTPUT = (
    _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-shared-dependencies-result.json"
)
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-shared-dependencies.patch"
_EXPECTED_COMPONENTS = {"express-jwt", "jsonwebtoken"}
_EXPECTED_JSONWEBTOKEN_CVES = {
    "CVE-2022-23539",
    "CVE-2022-23540",
    "CVE-2022-23541",
}


def load_shared_dependency_groups(
    path: Path = _DEFAULT_GROUPS,
) -> list[VulnerabilityGroup]:
    """Load and validate the two pre-triaged overlapping-dependency groups.

    Args:
        path: JSON array containing the extracted triaged groups.

    Returns:
        The validated ``express-jwt`` and nested ``jsonwebtoken`` groups.

    Raises:
        FileNotFoundError: If the group fixture does not exist.
        ValueError: If the fixture does not contain the intended two groups.
        json.JSONDecodeError: If the fixture is not valid JSON.
        pydantic.ValidationError: If a group violates its contract.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("Shared-dependencies fixture must contain exactly two groups.")

    groups = [VulnerabilityGroup.model_validate(item) for item in payload]
    components = {group.vulnerable_component for group in groups}
    if components != _EXPECTED_COMPONENTS:
        raise ValueError(
            "Shared-dependencies fixture must contain express-jwt and jsonwebtoken; "
            f"found {sorted(components)}."
        )

    express_group = next(group for group in groups if group.vulnerable_component == "express-jwt")
    jsonwebtoken_group = next(
        group for group in groups if group.vulnerable_component == "jsonwebtoken"
    )
    if express_group.versions != ["0.1.3"]:
        raise ValueError("The express-jwt group must target version 0.1.3.")
    if jsonwebtoken_group.versions != ["0.1.0"]:
        raise ValueError("The jsonwebtoken group must target nested version 0.1.0.")
    if any(
        group.file_path != "package.json" or group.file_paths != ["package.json"]
        for group in groups
    ):
        raise ValueError("Both shared-dependency groups must authorize package.json.")
    if any(
        localized.manifest_file != "package.json"
        for group in groups
        for localized in group.localized_issues
    ):
        raise ValueError("All localized shared-dependency findings must use package.json.")
    if set(jsonwebtoken_group.cve_ids) != _EXPECTED_JSONWEBTOKEN_CVES:
        raise ValueError(
            "The jsonwebtoken group must contain the three extracted update CVEs: "
            f"{sorted(_EXPECTED_JSONWEBTOKEN_CVES)}."
        )
    if any(group.fix_plan is None for group in groups):
        raise ValueError("Both shared-dependency groups must include a triaged fix plan.")
    return groups


def load_shared_dependency_issues(
    path: Path = _DEFAULT_ISSUES,
    groups: list[VulnerabilityGroup] | None = None,
) -> list[VulnerabilityIssue]:
    """Load the canonical issue subset and verify it matches the groups.

    Args:
        path: JSONL issue subset extracted from the canonical baseline.
        groups: Groups whose member issue IDs establish the expected baseline.

    Returns:
        Validated issue records in fixture order.

    Raises:
        FileNotFoundError: If the issue fixture does not exist.
        ValueError: If issue IDs do not exactly match the selected groups.
        pydantic.ValidationError: If an issue violates its contract.
    """
    issues = [
        VulnerabilityIssue.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if groups is None:
        groups = load_shared_dependency_groups()
    expected_ids = [str(issue.id) for group in groups for issue in group.issues]
    actual_ids = [str(issue.id) for issue in issues]
    if actual_ids != expected_ids:
        raise ValueError(
            "Shared-dependencies issue baseline does not match its triaged groups: "
            f"expected {expected_ids}, found {actual_ids}."
        )
    return issues


def main() -> int:
    """Execute the fixture and persist its result and unified patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", _DEFAULT_REPO)),
    )
    parser.add_argument("--groups", type=Path, default=_DEFAULT_GROUPS)
    parser.add_argument("--issues", type=Path, default=_DEFAULT_ISSUES)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--patch-out", type=Path, default=_DEFAULT_PATCH)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.repo.is_dir():
        logger.error("Repository does not exist: %s", args.repo)
        return 2
    if not args.groups.is_file():
        logger.error("Shared-dependencies group fixture does not exist: %s", args.groups)
        return 2
    if not args.issues.is_file():
        logger.error("Shared-dependencies issue fixture does not exist: %s", args.issues)
        return 2

    groups = load_shared_dependency_groups(args.groups.resolve())
    baseline_issues = load_shared_dependency_issues(args.issues.resolve(), groups)
    logger.info(
        "Starting shared-dependencies remediation with %d groups and %d baseline issues.",
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
    return 0 if result.status in {"completed", "completed_with_errors"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
