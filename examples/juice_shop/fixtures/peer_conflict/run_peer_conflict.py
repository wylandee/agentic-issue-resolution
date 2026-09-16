"""Run the Socket.IO peer-conflict Juice Shop fixture."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_ISSUES = Path(__file__).resolve().parent / "baseline_issues_peer_conflict.jsonl"
_DEFAULT_SUPPRESSIONS = Path(__file__).resolve().parent / "suppressions.xml"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-peer-conflict-result.json"
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-peer-conflict.patch"
_TARGET_PACKAGES = frozenset(
    {
        "socket.io",
        "engine.io",
    }
)
_EXPECTED_ISSUE_COUNT = 4


def load_peer_conflict_issues(
    path: Path = _DEFAULT_ISSUES,
) -> list[VulnerabilityIssue]:
    """Load and validate the peer-conflict canonical JSONL fixture.

    Args:
        path: JSONL file containing the selected baseline findings.

    Returns:
        The canonical findings for the target package set in fixture order.

    Raises:
        FileNotFoundError: If the issue fixture does not exist.
        ValueError: If the fixture contains the wrong package set or finding count.
        pydantic.ValidationError: If a JSONL record violates the issue contract.
    """
    issues = [
        VulnerabilityIssue.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    packages = {issue.package_name for issue in issues if issue.package_name}
    if len(issues) != _EXPECTED_ISSUE_COUNT or packages != _TARGET_PACKAGES:
        raise ValueError(
            "Peer-conflict fixture must contain exactly 4 findings for "
            f"{sorted(_TARGET_PACKAGES)}; found {len(issues)} findings for "
            f"{sorted(packages)}."
        )
    return issues


def copy_suppressions_to_repo(
    suppressions_path: Path,
    repo_root: Path,
) -> Path:
    """Copy the fixture suppression rules into the repository root.

    Args:
        suppressions_path: Fixture suppression XML to copy.
        repo_root: Juice Shop clone that will be scanned.

    Returns:
        The repository-local suppression path.

    Side Effects:
        Overwrites the repository-local suppressions.xml file.
    """
    destination = repo_root / "suppressions.xml"
    shutil.copy2(suppressions_path, destination)
    return destination


def main() -> int:
    """Execute the raw-issue fixture and persist the result and patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", str(_DEFAULT_REPO))),
    )
    parser.add_argument("--issues", type=Path, default=_DEFAULT_ISSUES)
    parser.add_argument("--suppressions", type=Path, default=_DEFAULT_SUPPRESSIONS)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--patch-out", type=Path, default=_DEFAULT_PATCH)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    repo_root = args.repo.expanduser().resolve()
    issues_path = args.issues.expanduser().resolve()
    suppressions_path = args.suppressions.expanduser().resolve()
    if not repo_root.is_dir():
        logger.error("Repository does not exist: %s", repo_root)
        return 2
    if not issues_path.is_file():
        logger.error("Issue fixture does not exist: %s", issues_path)
        return 2
    if not suppressions_path.is_file():
        logger.error("Suppression fixture does not exist: %s", suppressions_path)
        return 2

    issues = load_peer_conflict_issues(issues_path)
    copied_suppressions = copy_suppressions_to_repo(suppressions_path, repo_root)
    logger.info(
        "Starting peer_conflict fixture with %d findings; copied suppressions to %s.",
        len(issues),
        copied_suppressions,
    )

    result = run_remediation(
        RemediationRequest(
            repo_root=repo_root,
            issues=issues,
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="javascript/nodejs",
            ),
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
