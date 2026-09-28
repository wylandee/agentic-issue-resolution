"""Run the task-queue workflow against a local PyGoat clone."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import SystemContext, VulnerabilityIssue

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "pygoat"
_DEFAULT_ISSUES = Path(__file__).resolve().parent / "fixtures" / "baseline_issues.jsonl"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "data" / "trajectories" / "pygoat-result.json"
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "pygoat.patch"
logger = logging.getLogger(__name__)


def _load_issues(path: Path) -> list[VulnerabilityIssue]:
    """Load canonical JSONL findings for one Python remediation run."""
    return [
        VulnerabilityIssue.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run remediation and write its result JSON and unified patch.

    Args:
        argv: Optional command-line arguments, primarily for tests.

    Returns:
        ``0`` for a completed run without errors, ``1`` for a run with
        remediation errors, or ``2`` when an input path is unavailable.

    Side effects:
        Reads the supplied host repository and JSONL findings, invokes the
        public remediation API, and writes result/patch files to the requested
        output paths. The engine performs edits only in its isolated workspace.
    """
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(os.environ.get("TEST_REPO_ROOT", str(_DEFAULT_REPO))),
    )
    parser.add_argument("--issues", type=Path, default=_DEFAULT_ISSUES)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--patch-out", type=Path, default=_DEFAULT_PATCH)
    args = parser.parse_args(argv)

    repo_root = args.repo.expanduser().resolve()
    issues_path = args.issues.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    patch_path = args.patch_out.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not repo_root.is_dir():
        logger.error("Repository does not exist: %s", repo_root)
        return 2
    if not issues_path.is_file():
        logger.error("Issues fixture does not exist: %s", issues_path)
        return 2

    result = run_remediation(
        RemediationRequest(
            repo_root=repo_root,
            issues=_load_issues(issues_path),
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="python",
            ),
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
