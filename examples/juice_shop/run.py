"""Run the current task-queue workflow against a local Juice Shop clone."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import VulnerabilityIssue


def _load_issues(path: Path) -> list[VulnerabilityIssue]:
    """Load canonical JSONL issues, including legacy array-shaped fixtures."""
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        payload = json.loads(text)
        return [VulnerabilityIssue.model_validate(item) for item in payload]
    return [
        VulnerabilityIssue.model_validate_json(line) for line in text.splitlines() if line.strip()
    ]


def main() -> int:
    """Execute the example and write a JSON result plus patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", "data/clones/juice-shop")),
    )
    parser.add_argument(
        "--issues", type=Path, default=Path(__file__).parent / "fixtures" / "baseline_issues.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/trajectories/juice-shop-result.json")
    )
    parser.add_argument(
        "--patch-out", type=Path, default=Path("data/trajectories/juice-shop.patch")
    )
    args = parser.parse_args()
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.repo.is_dir():
        logging.error("Repository does not exist: %s", args.repo)
        return 2
    result = run_remediation(
        RemediationRequest(repo_root=args.repo, issues=_load_issues(args.issues))
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
