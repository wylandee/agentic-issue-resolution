"""Run a five-finding fixture focused on deterministic Supervisor routing.

The fixture is pre-triaged so the run exercises the Phase 5 Supervisor directly.
It intentionally covers one NO_FIX task and four version-bump tasks with
different severities. The run is considered operationally complete when the
graph reaches a terminal result; individual remediation failures remain visible
in the persisted result and trajectory.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from remediation_engine import RemediationRequest, run_remediation
from remediation_engine.contracts.schemas import VulnerabilityGroup

_MAX_ISSUES = 5
_EXPECTED_COMPONENTS = {
    "@tootallnate/once",
    "express-jwt",
    "got",
    "notevil",
    "sanitize-html",
}
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "triaged_groups_deterministic.json"
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_OUTPUT = (
    _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-deterministic-routing-result.json"
)
_DEFAULT_PATCH = _PROJECT_ROOT / "data" / "trajectories" / "juice-shop-deterministic-routing.patch"


def load_fixture(path: Path = _DEFAULT_FIXTURE) -> list[VulnerabilityGroup]:
    """Load and validate the five-finding deterministic-routing fixture.

    Args:
        path: JSON array containing pre-triaged vulnerability groups.

    Returns:
        Validated vulnerability groups.

    Raises:
        FileNotFoundError: If the fixture does not exist.
        ValueError: If the fixture is not an array, exceeds five findings, or
            does not contain the intended routing coverage.
        json.JSONDecodeError: If the fixture is not valid JSON.
        pydantic.ValidationError: If a group violates its contract.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Deterministic routing fixture must be a JSON array: {path}")

    groups = [VulnerabilityGroup.model_validate(item) for item in payload]
    issue_count = sum(len(group.issues) for group in groups)
    if issue_count > _MAX_ISSUES:
        raise ValueError(
            f"Deterministic routing fixture contains {issue_count} issues; "
            f"maximum is {_MAX_ISSUES}."
        )
    if issue_count != _MAX_ISSUES:
        raise ValueError(
            f"Deterministic routing fixture must contain exactly {_MAX_ISSUES} issues; "
            f"found {issue_count}."
        )

    components = {group.vulnerable_component for group in groups}
    if components != _EXPECTED_COMPONENTS:
        raise ValueError(
            "Deterministic routing fixture must contain exactly these components: "
            f"{sorted(_EXPECTED_COMPONENTS)}; found {sorted(components)}."
        )
    return groups


def _routing_summary(result: Any) -> dict[str, Any]:
    """Project deterministic-routing evidence from a remediation result."""
    state = result.raw_state or {}
    task_queue = state.get("task_queue", {}) or {}

    def task_status(task: Any) -> str | None:
        """Return a task status from either a model or serialized mapping."""
        status = task.get("status") if isinstance(task, Mapping) else getattr(task, "status", None)
        return getattr(status, "value", status)

    return {
        "issue_count": sum(len(group.issues) for group in state.get("valid_groups", []) or []),
        "final_decision_code": state.get("decision_code"),
        "final_next_routing_step": state.get("next_routing_step"),
        "supervisor_audit": state.get("supervisor_audit"),
        "task_statuses": {
            task_id: task_status(task) for task_id, task in sorted(task_queue.items())
        },
    }


def main() -> int:
    """Run the deterministic-routing fixture and persist its result and patch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_DEFAULT_REPO)
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--patch-out", type=Path, default=_DEFAULT_PATCH)
    args = parser.parse_args()

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.repo.is_dir():
        logging.error("Repository does not exist: %s", args.repo)
        return 2
    if not args.fixture.is_file():
        logging.error("Deterministic routing fixture does not exist: %s", args.fixture)
        return 2

    groups = load_fixture(args.fixture.resolve())
    logging.info(
        "Starting deterministic-routing fixture with %d groups and %d issues.",
        len(groups),
        sum(len(group.issues) for group in groups),
    )

    result = run_remediation(
        RemediationRequest(
            repo_root=args.repo.resolve(),
            valid_groups=groups,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = result.model_dump(exclude={"raw_state"})
    output["routing_summary"] = _routing_summary(result)
    args.output.write_text(
        json.dumps(output, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.patch_out.parent.mkdir(parents=True, exist_ok=True)
    args.patch_out.write_text(result.diff, encoding="utf-8")

    summary = output["routing_summary"]
    logging.info(
        "status=%s decision_code=%s next=%s task_statuses=%s",
        result.status,
        summary["final_decision_code"],
        summary["final_next_routing_step"],
        summary["task_statuses"],
    )
    return 0 if result.status in {"completed", "completed_with_errors"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
