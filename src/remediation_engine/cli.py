"""Command-line interface for the remediation engine."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .api import RemediationRequest, run_remediation, triage_issues
from .contracts.schemas import (
    RemediationTask,
    SystemContext,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from .orchestration.portfolio_orchestrator import (
    apply_portfolio_plan,
    build_portfolio_plan,
    prepare_portfolio_inputs,
)
from .settings import AppSettings
from .tools.odc_parser import parse_vulnerabilities
from .tools.semgrep_parser import load_findings_from_json, normalize_finding

log = logging.getLogger(__name__)


def _load_json_fixture(path: Path) -> Any:
    """Load one local JSON fixture, rejecting JSONL and other stream formats."""
    return json.loads(path.read_text(encoding="utf-8"))


def _fixture_values(payload: Any, *keys: str) -> Any:
    """Extract a named fixture collection from an optional JSON envelope."""
    if isinstance(payload, Mapping):
        for key in keys:
            if key in payload:
                return payload[key]
    return payload


def _load_groups_fixture(path: Path) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask]]:
    """Load pre-triaged groups and any embedded task queue from a JSON fixture."""
    payload = _load_json_fixture(path)
    groups_payload = _fixture_values(payload, "groups", "valid_groups")
    if isinstance(groups_payload, Mapping):
        groups_payload = [groups_payload]
    if not isinstance(groups_payload, list):
        raise ValueError("solve input must contain a JSON list of pre-triaged groups")
    groups = [VulnerabilityGroup.model_validate(item) for item in groups_payload]
    group_ids = [group.group_id for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("solve input must not contain duplicate group IDs")
    embedded_tasks = _fixture_values(payload, "tasks", "task_queue")
    return groups, _parse_task_values(embedded_tasks) if embedded_tasks is not payload else {}


def _parse_task_values(payload: Any) -> dict[str, RemediationTask]:
    """Validate an optional task fixture in list or task-id mapping form."""
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        values = list(payload.items())
    elif isinstance(payload, list):
        values = [(None, item) for item in payload]
    else:
        raise ValueError("task fixture must be a JSON list or task-id mapping")
    queue: dict[str, RemediationTask] = {}
    for key, raw in values:
        task = RemediationTask.model_validate(raw)
        if key is not None and str(key) != task.task_id:
            raise ValueError(f"task fixture key {key!r} does not match task_id {task.task_id!r}")
        if task.task_id in queue:
            raise ValueError(f"task fixture contains duplicate task ID {task.task_id!r}")
        queue[task.task_id] = task
    return dict(sorted(queue.items()))


def _load_solve_inputs(
    groups_path: Path | None,
    tasks_path: Path | None,
) -> tuple[list[VulnerabilityGroup], dict[str, RemediationTask]]:
    """Load explicit group/task fixtures for the offline portfolio command."""
    if groups_path is None:
        raise ValueError("solve requires a pre-triaged groups JSON fixture")
    groups, embedded_tasks = _load_groups_fixture(groups_path)
    if tasks_path is not None:
        task_payload = _fixture_values(_load_json_fixture(tasks_path), "tasks", "task_queue")
        embedded_tasks = _parse_task_values(task_payload)
    return groups, embedded_tasks


def _solve_output(
    plan: Any,
    prepare_diagnostics: Sequence[str],
    apply_diagnostics: Sequence[str],
) -> dict[str, Any]:
    """Project a portfolio plan into stable, human-readable dry-run JSON."""
    solver_plan = getattr(plan, "solver_plan", None)
    selected = getattr(solver_plan, "selected_plan", None)
    status = getattr(getattr(solver_plan, "status", None), "value", None)
    if status is None:
        status = str(getattr(solver_plan, "status", "UNKNOWN")).upper()
    decisions = sorted(
        (
            decision.model_dump(mode="json")
            for decision in (selected.task_decisions if selected else [])
        ),
        key=lambda item: item["task_id"],
    )
    batches = sorted(
        (batch.model_dump(mode="json") for batch in (selected.batches if selected else [])),
        key=lambda item: item["batch_id"],
    )
    phases = sorted(
        (phase.model_dump(mode="json") for phase in (selected.phases if selected else [])),
        key=lambda item: item["phase_number"],
    )
    diagnostics = sorted(
        {
            str(item)
            for item in (
                *prepare_diagnostics,
                *getattr(plan, "diagnostics", ()),
                *getattr(solver_plan, "diagnostics", ()),
                *apply_diagnostics,
            )
            if str(item).strip()
        }
    )
    return {
        "plan_id": plan.plan_id,
        "portfolio_plan_id": plan.portfolio_plan_id,
        "solver_status": status,
        "task_decisions": decisions,
        "batches": batches,
        "phases": phases,
        "diagnostics": diagnostics,
        "digests": {
            "plan_digest": plan.plan_digest,
            "graph_digest": plan.graph_digest,
            "solver_input_digest": plan.solver_input_digest,
            "solver_domain_digest": getattr(solver_plan, "domain_digest", None),
            "repository_fingerprint": plan.repository_fingerprint,
            "solver_repository_digest": getattr(solver_plan, "repository_digest", None),
        },
        "solver_plan": (solver_plan.model_dump(mode="json") if solver_plan is not None else None),
        "portfolio_plan": plan.model_dump(mode="json"),
    }


def _load_issues(path: Path, input_format: str) -> list[VulnerabilityIssue]:
    """Load typed issues from one supported local scanner input format.

    JSONL is the canonical issue interchange format. ODC and Semgrep JSON are
    accepted at the ingestion boundary and normalized to
    ``VulnerabilityIssue`` models.
    """
    if input_format == "auto":
        input_format = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "odc-json"
    if input_format == "jsonl":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [
            VulnerabilityIssue.model_validate_json(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if input_format == "odc-json":
        return parse_vulnerabilities(json.loads(path.read_text(encoding="utf-8")))
    if input_format == "semgrep-json":
        return [
            issue
            for raw in load_findings_from_json(path)
            if (issue := normalize_finding(raw)) is not None
        ]
    raise ValueError(f"Unsupported input format: {input_format}")


def _write_json(path: Path | None, value: object, *, sort_keys: bool = False) -> None:
    """Write indented JSON to a file or stdout."""
    payload = json.dumps(value, indent=2, default=str, sort_keys=sort_keys)
    if path is None:
        print(payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


def _write_jsonl(path: Path | None, values: list[object]) -> None:
    """Write one compact JSON object per line to a file or stdout."""
    payload = "".join(
        json.dumps(value, separators=(",", ":"), default=str) + "\n" for value in values
    )
    if path is None:
        print(payload, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def _copy_report(source: Path, destination: Path) -> None:
    """Copy a canonical Markdown report to a caller-selected path atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser without reading environment state."""
    parser = argparse.ArgumentParser(prog="remedy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="normalize a scanner report")
    ingest.add_argument("input", type=Path)
    ingest.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    ingest.add_argument("--output", type=Path)
    triage = sub.add_parser("triage", help="load canonical findings for triage")
    triage.add_argument("input", type=Path)
    triage.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    triage.add_argument("--repo", type=Path)
    triage.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="run remediation and emit a patch result")
    run.add_argument("input", type=Path)
    run.add_argument("--repo", required=True, type=Path)
    run.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--patch-out", type=Path)
    run.add_argument("--report-out", type=Path)
    solve = sub.add_parser(
        "solve",
        help="build an offline portfolio plan from pre-triaged groups",
        description=(
            "Build and emit a deterministic portfolio plan without workers or "
            "host-repository mutation. Input must be pre-triaged groups JSON."
        ),
    )
    solve.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="JSON fixture containing a list of pre-triaged groups",
    )
    solve.add_argument(
        "--groups",
        "--group-input",
        dest="groups_input",
        type=Path,
        help="groups JSON fixture (alternative to the positional input)",
    )
    solve.add_argument(
        "--tasks",
        "--task-input",
        dest="tasks_input",
        type=Path,
        help="optional JSON fixture containing a task list or task-id mapping",
    )
    solve.add_argument("--repo", required=True, type=Path)
    solve.add_argument("--output", type=Path)
    return parser


def _run_solve(args: argparse.Namespace, settings: AppSettings) -> int:
    """Run the copy-on-write outer portfolio planner for a local fixture."""
    if args.input is not None and args.groups_input is not None:
        raise ValueError("provide groups either positionally or with --groups, not both")
    groups_path = args.groups_input or args.input
    groups, task_queue = _load_solve_inputs(groups_path, args.tasks_input)
    prepared_groups, prepared_queue, prepare_diagnostics = prepare_portfolio_inputs(
        args.repo,
        groups,
        task_queue,
    )
    plan = build_portfolio_plan(
        args.repo,
        prepared_groups,
        prepared_queue,
        settings=settings,
    )
    _committed_groups, _committed_queue, apply_diagnostics = apply_portfolio_plan(
        plan,
        prepared_groups,
        prepared_queue,
    )
    output = _solve_output(plan, prepare_diagnostics, apply_diagnostics)
    _write_json(args.output, output, sort_keys=True)
    status = output["solver_status"]
    if status not in {"OPTIMAL", "FEASIBLE"} or output["diagnostics"]:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return its process exit code.

    The command reads local input, loads environment-backed settings, and
    writes requested JSON, JSONL, patch, or report outputs. Remediation runs
    use an isolated workspace and do not modify the supplied host repository.
    Input or filesystem validation errors are logged and return exit code 2;
    a completed run with remediation errors returns 1, otherwise 0.
    """
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        settings = AppSettings.from_env()
        if args.command == "solve":
            return _run_solve(args, settings)
        issues = _load_issues(args.input, args.format)
        if args.command == "ingest":
            _write_jsonl(args.output, [issue.model_dump(mode="json") for issue in issues])
            return 0
        if args.command == "triage":
            groups = triage_issues(issues, repo_root=args.repo, settings=settings)
            _write_json(args.output, [group.model_dump(mode="json") for group in groups])
            return 0
        request = RemediationRequest(
            repo_root=args.repo,
            issues=issues,
            valid_groups=[],
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="javascript/nodejs",
            ),
        )
        result = run_remediation(request, settings=settings)
        if args.patch_out:
            args.patch_out.parent.mkdir(parents=True, exist_ok=True)
            args.patch_out.write_text(result.diff, encoding="utf-8")
        if args.report_out:
            if not result.report_path:
                raise ValueError("The run did not produce a canonical report to copy.")
            _copy_report(Path(result.report_path), args.report_out)
        _write_json(args.output, result.model_dump(exclude={"raw_state"}))
        return 0 if result.status == "completed" and not result.errors else 1
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2
