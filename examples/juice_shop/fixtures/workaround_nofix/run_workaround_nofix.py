"""Run the NO_FIX vulnerable-code-removal retry through the Supervisor.

This manual example starts with an unchanged Juice Shop clone, commits a
supervisor-owned ``VULNERABLE_CODE_REMOVAL`` retry attempt, materializes npm
dependencies, and follows the normal workaround-worker/Supervisor routing
against the same temporary Docker volume. It is intentionally not part of the
public remediation workflow and must only be run when live Docker/LLM calls are
desired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    FailureCategory,
    NoFixMitigationStage,
    QAEvaluation,
    QAFailureEvidence,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundPhase,
)
from remediation_engine.orchestration.graph import (
    run_qa_critic_from_orchestrator,
    run_workaround_subagent_from_orchestrator,
)
from remediation_engine.orchestration.langsmith_config import (
    is_phase5_tracing_enabled,
    resolve_phase5_trace_url,
)
from remediation_engine.orchestration.state import initial_orchestrator_state
from remediation_engine.orchestration.supervisor_node import run_supervisor_node
from remediation_engine.orchestration.task_utils import (
    build_initial_remediation_task,
    build_no_fix_retry_instruction,
)
from remediation_engine.orchestration.teardown_node import run_teardown_node
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    export_phase5_trajectory,
    use_trajectory_recorder,
)
from remediation_engine.orchestration.workspace_builder import (
    run_workspace_builder_node,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "notevil_workaround_nofix.json"
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "trajectories"

_NOFIX_METADATA = {
    "replay_type": "notevil_stage_two_nofix_remediation",
    "replay_mode": "retry_remediation",
    "workflow_phase": "qa_regression_repair",
    "package_name": "notevil",
    "no_fix_stage": "vulnerable_code_removal",
    "source_group_id": "sca:package.json:notevil:NO_FIX",
}


def load_nofix_fixture(path: Path = _DEFAULT_FIXTURE) -> dict[str, Any]:
    """Load and validate the explicit Stage 2 NO_FIX fixture.

    Args:
        path: JSON fixture containing the NO_FIX group and initial task context.

    Returns:
        The decoded fixture mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the root or required sections are invalid.
        json.JSONDecodeError: If ``path`` is not valid JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"NO_FIX fixture root must be an object: {path}")
    required_sections = {
        "replay",
        "group",
        "task",
        "qa_evidence",
        "workaround_context",
        "workspace_seed",
        "provenance",
    }
    missing = sorted(required_sections - payload.keys())
    if missing:
        raise ValueError(f"NO_FIX fixture is missing sections: {', '.join(missing)}")
    if (
        payload["replay"].get("initial_no_fix_stage")
        != NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value
    ):
        raise ValueError("NO_FIX fixture must start at VULNERABLE_CODE_REMOVAL.")
    if not payload.get("qa_evidence"):
        raise ValueError("Stage-two NO_FIX fixture must include prior QA failure evidence.")
    if payload["group"].get("group_id") != "sca:package.json:notevil:NO_FIX":
        raise ValueError("NO_FIX fixture must contain the notevil NO_FIX group.")
    return payload


def _instruction_digest(instruction: str) -> str:
    """Return the normalized instruction digest used by dispatch guards."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_initial_state(repo_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    """Construct a committed Stage 2 retry state without Docker.

    Args:
        repo_root: Host clone used later by the workspace builder and teardown.
        fixture: Parsed NO_FIX fixture payload.

    Returns:
        An orchestrator state containing one initial NO_FIX task and its
        immutable workaround attempt snapshot.

    Raises:
        pydantic.ValidationError: If the fixture group or generated task
            violates a Phase 5 contract.
        ValueError: If the fixture's task identity disagrees with its group.
    """
    group = VulnerabilityGroup.model_validate(fixture["group"])
    fixture_task = fixture["task"]
    task_id = str(fixture_task["task_id"])
    if fixture_task.get("parent_group_id") != group.group_id:
        raise ValueError("NO_FIX fixture task does not target its fixture group.")

    requested_stage = NoFixMitigationStage(fixture["replay"]["initial_no_fix_stage"])
    if requested_stage != NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
        raise ValueError("This fixture is dedicated to the Stage 2 NO_FIX retry.")

    # Recreate the task through the production factory so the fixture cannot
    # silently bypass the supervisor's deterministic NO_FIX instruction, then
    # apply the same projection that the Supervisor would commit after a failed
    # package-removal attempt.
    task = build_initial_remediation_task(group, task_id)
    if task.no_fix_stage != NoFixMitigationStage.PACKAGE_REMOVAL:
        raise ValueError("Initial notevil task did not resolve to PACKAGE_REMOVAL.")
    if fixture_task.get("no_fix_stage") != requested_stage.value:
        raise ValueError("Fixture task stage is invalid for the Stage 2 retry.")
    if fixture_task.get("status") != TaskStatus.NEEDS_RETRY.value:
        raise ValueError("Stage-two NO_FIX fixture task must start in NEEDS_RETRY.")
    if fixture_task.get("retry_count") != 1:
        raise ValueError("Stage-two NO_FIX fixture task must start with retry_count=1.")

    context_payload = fixture["workaround_context"]
    qa_payload = fixture.get("qa_evidence")
    qa_evidence = QAFailureEvidence.model_validate(qa_payload)
    failure_feedback = (
        context_payload.get("failure_feedback")
        or "The package-removal attempt failed before a complete QA evaluation was available."
    )
    stage_two_task = task.model_copy(
        update={
            "task_revision": 2,
            "no_fix_stage": requested_stage,
            "status": TaskStatus.NEEDS_RETRY,
            "retry_count": 1,
        }
    )
    prior_evaluation = QAEvaluation(
        task_id=task.task_id,
        passed=False,
        failure_category=FailureCategory.SECURITY_FLAG,
        retry_feedback=failure_feedback,
        failure_evidence=qa_evidence,
    )
    stage_two_task = stage_two_task.model_copy(
        update={
            "instruction": build_no_fix_retry_instruction(
                stage_two_task,
                group,
                evaluation=prior_evaluation,
                failure_feedback=failure_feedback,
            )
        }
    )
    workaround_context = WorkaroundContext(
        phase=WorkaroundPhase(context_payload["phase"]),
        vulnerability_mechanism=context_payload["vulnerability_mechanism"],
        qa_evidence=qa_evidence,
        no_fix_stage=requested_stage,
        reset_prior_stage_workspace=bool(context_payload.get("reset_prior_stage_workspace", True)),
    )

    # A real supervisor dispatch increments the revision for the initial
    # attempt, then increments it again when advancing to Stage 2. This state
    # therefore begins at revision 2 with retry_count=1.
    attempt_id = str(fixture["replay"].get("initial_attempt_id", "notevil-nofix-stage-two-attempt"))
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=stage_two_task.task_id,
        state_revision=2,
        task_revision=stage_two_task.task_revision,
        attempt_number=2,
        strategy_stage=stage_two_task.strategy_stage,
        no_fix_stage=stage_two_task.no_fix_stage,
        selected_version=stage_two_task.selected_version,
        instruction=stage_two_task.instruction,
        instruction_digest=_instruction_digest(stage_two_task.instruction),
        dispatch_node="workaround_subagent",
        plan_id=None,
        workaround_context=workaround_context,
    )
    stage_two_task = stage_two_task.model_copy(update={"current_attempt_id": snapshot.attempt_id})

    state = initial_orchestrator_state(
        repo_root=str(repo_root.resolve()),
        valid_groups=[group],
        issues=group.issues,
    )
    state.update(
        {
            "state_revision": snapshot.state_revision,
            "task_queue": {stage_two_task.task_id: stage_two_task},
            "active_target_task_ids": [stage_two_task.task_id],
            "active_target_group_ids": [],
            "feedback_by_task": {stage_two_task.task_id: failure_feedback},
            "feedback_by_group": {},
            "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            "workspace_volume": None,
            "status": "workaround_nofix_stage_two_pending",
        }
    )
    return state


def build_initial_state(repo_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    """Build the Stage 2 NO_FIX retry state for dry verification.

    This helper performs no Docker, network, LLM, or LangSmith work.

    Args:
        repo_root: Host clone used by a later manual invocation.
        fixture: Parsed output of :func:`load_nofix_fixture`.

    Returns:
        A committed vulnerable-code-removal retry dispatch state.
    """
    return _build_initial_state(repo_root, fixture)


def _merge_worker_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    """Merge the direct workaround bridge result into mutable state."""
    if result.get("changed_files"):
        state["changed_files"] = sorted(
            set(state.get("changed_files", [])) | set(result["changed_files"])
        )
    if result.get("action_summaries"):
        state["action_summaries"] = [
            *state.get("action_summaries", []),
            *result["action_summaries"],
        ]
    if result.get("worker_results_by_attempt"):
        state["worker_results_by_attempt"] = {
            **state.get("worker_results_by_attempt", {}),
            **result["worker_results_by_attempt"],
        }
    if result.get("errors"):
        state["errors"] = list(dict.fromkeys([*state.get("errors", []), *result["errors"]]))


def _merge_qa_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    """Merge QA output before teardown reconciles task status and evidence."""
    for key in (
        "status",
        "qa_evaluations",
        "qa_results_by_attempt",
        "eval_status",
        "qa_investigation_report",
        "baseline_scan_identifiers",
        "post_remediation_scan_identifiers",
        "post_remediation_scan_issues",
        "new_vulnerability_identifiers",
        "new_vulnerability_status",
        "triage_required",
    ):
        if key in result:
            state[key] = result[key]
    if result.get("errors"):
        state["errors"] = list(dict.fromkeys([*state.get("errors", []), *result["errors"]]))


def _merge_supervisor_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    """Apply a Supervisor node patch while preserving additive errors."""
    prior_errors = list(state.get("errors", []) or [])
    state.update({key: value for key, value in result.items() if key != "errors"})
    state["errors"] = list(dict.fromkeys([*prior_errors, *(result.get("errors", []) or [])]))


def _jsonable(value: Any) -> Any:
    """Convert contracts, enums, and paths to JSON-serializable values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


@traceable(
    name="Notevil_NO_FIX_Stage_Two_Vulnerable_Code_Removal_With_QA",
    run_type="chain",
    metadata=_NOFIX_METADATA,
    tags=["notevil", "no-fix", "vulnerable-code-removal", "workaround", "qa-critic"],
)
def _execute_nofix(repo_root: str, fixture: dict[str, Any]) -> dict[str, Any]:
    """Execute the Stage 2 NO_FIX worker attempt through Supervisor routing.

    Args:
        repo_root: Host clone used as the immutable workspace source.
        fixture: Explicit Stage 2 NO_FIX retry fixture payload.

    Returns:
        JSON-serializable worker, QA, task-lifecycle, and cleanup results.

    Side effects:
        Creates and removes a temporary Docker volume, installs dependencies,
        invokes the workaround worker/LLM, Supervisor, and QA services as
        routed, and reads the resulting patch. The host clone is never edited.
    """
    repo_path = Path(repo_root).resolve()
    state = _build_initial_state(repo_path, fixture)
    worker_result: dict[str, Any] = {}
    qa_result: dict[str, Any] = {}
    teardown_result: dict[str, Any] = {}
    install_diagnostics: dict[str, Any] = {}
    supervisor_events: list[dict[str, Any]] = []
    workspace_volume: str | None = None
    workaround_dispatches = 0

    try:
        builder_result = run_workspace_builder_node(state)
        state.update(builder_result)
        if builder_result.get("errors"):
            state["errors"] = [*state.get("errors", []), *builder_result["errors"]]
        workspace_volume = state.get("workspace_volume")
        if builder_result.get("status") != "workspace_ready" or not workspace_volume:
            raise RuntimeError("Workspace builder did not return a ready Docker volume.")

        # Workspace Builder materializes every npm package before this first
        # workaround dispatch, keeping validation tools such as tsc available
        # inside the isolated volume without modifying the host clone.
        install_diagnostics = {
            "status": "completed",
            "handled_by": "workspace_builder",
        }

        # This fixture represents the second dispatch. The workspace remains
        # at the unchanged repository baseline so notevil is still installed;
        # package-removal edits are intentionally not replayed.
        workaround_dispatches = 2
        worker_result = run_workaround_subagent_from_orchestrator(state)
        _merge_worker_result(state, worker_result)

        # Follow the same worker -> Supervisor -> next-node handoff as the
        # production graph. A Stage 2 failure should be terminalized by the
        # Supervisor rather than dispatched for a third NO_FIX attempt.
        while True:
            supervisor_result = run_supervisor_node(state)
            _merge_supervisor_result(state, supervisor_result)
            current_task = state.get("task_queue", {}).get("task-nofix-notevil")
            supervisor_events.append(
                {
                    "status": supervisor_result.get("status"),
                    "next_routing_step": supervisor_result.get("next_routing_step"),
                    "active_target_task_ids": supervisor_result.get("active_target_task_ids", []),
                    "task_status": current_task.status.value if current_task else None,
                    "no_fix_stage": (
                        current_task.no_fix_stage.value
                        if current_task and current_task.no_fix_stage
                        else None
                    ),
                    "retry_count": current_task.retry_count if current_task else None,
                }
            )

            next_step = state.get("next_routing_step")
            if next_step == "workaround_subagent":
                if workaround_dispatches >= 2:
                    raise RuntimeError(
                        "Supervisor requested more than the two allowed NO_FIX workaround dispatches."
                    )
                workaround_dispatches += 1
                worker_result = run_workaround_subagent_from_orchestrator(state)
                _merge_worker_result(state, worker_result)
                continue

            if next_step == "qa_critic":
                qa_result = run_qa_critic_from_orchestrator(state)
                _merge_qa_result(state, qa_result)
                continue

            if next_step == "teardown":
                break

            raise RuntimeError(f"Supervisor returned unsupported next step: {next_step!r}")
    except Exception as exc:  # noqa: BLE001
        message = f"Notevil NO_FIX Stage 2 retry failed before completion: {exc}"
        logger.exception(message)
        state["errors"] = [*state.get("errors", []), message]
    finally:
        if state.get("workspace_volume"):
            try:
                teardown_result = run_teardown_node(state)
                state.update(teardown_result)
            except Exception as exc:  # noqa: BLE001
                message = f"Notevil NO_FIX teardown failed: {exc}"
                logger.exception(message)
                state["errors"] = [*state.get("errors", []), message]

    task = state.get("task_queue", {}).get("task-nofix-notevil")
    task_status = task.status.value if task else None
    task_stage = task.no_fix_stage.value if task and task.no_fix_stage else None
    action_summaries = _jsonable(state.get("action_summaries", []))
    worker_status = action_summaries[-1].get("status") if action_summaries else None
    qa_status = qa_result.get("status")
    qa_eval_status = qa_result.get("eval_status")
    cleanup_ok = workspace_volume is None or teardown_result.get("workspace_volume") is None
    if not cleanup_ok:
        result_status = "surrender"
    elif task_status == TaskStatus.QA_PASSED.value:
        result_status = "success"
    elif task_status == TaskStatus.UNFIXABLE.value:
        result_status = "unfixable"
    elif task_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL.value:
        result_status = "needs_retry"
    elif state.get("errors"):
        result_status = "surrender"
    else:
        result_status = "qa_failure"

    return _jsonable(
        {
            "status": result_status,
            "replay": fixture["replay"],
            "provenance": fixture["provenance"],
            "task": {
                "status": task_status,
                "no_fix_stage": task_stage,
                "retry_count": task.retry_count if task else None,
            },
            "workspace_install": install_diagnostics,
            "supervisor_events": supervisor_events,
            "worker": {
                "status": worker_status,
                "action_summaries": action_summaries,
                "worker_results_by_attempt": state.get("worker_results_by_attempt", {}),
                "changed_files": state.get("changed_files", []),
            },
            "qa": {
                **qa_result,
                "passed": qa_status == "qa_completed" and qa_eval_status == "all_passed",
                "status": qa_status,
                "eval_status": qa_eval_status,
            },
            "teardown": {
                "status": teardown_result.get("status"),
                "workspace_volume": teardown_result.get("workspace_volume"),
                "volume_removed": bool(workspace_volume)
                and teardown_result.get("workspace_volume") is None,
            },
            "changed_files": state.get("changed_files", []),
            "diff": state.get("diff", ""),
            "errors": state.get("errors", []),
        }
    )


def _write_outputs(result: dict[str, Any], output_path: Path, patch_path: Path) -> None:
    """Write the JSON report and unified patch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    patch_path.write_text(str(result.get("diff", "")), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse manual NO_FIX command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_DEFAULT_REPO)
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--patch-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run the manually invoked NO_FIX example and return a shell exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    fixture = load_nofix_fixture(args.fixture.resolve())
    build_initial_state(args.repo, fixture)
    output_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        _DEFAULT_OUTPUT_DIR / f"notevil_workaround_nofix_stage2_{output_stamp}.json"
    )
    patch_path = args.patch_output or (
        _DEFAULT_OUTPUT_DIR / f"notevil_workaround_nofix_stage2_{output_stamp}.patch"
    )

    initial_state = build_initial_state(args.repo, fixture)
    trace_id = uuid.uuid4()
    langsmith_enabled = is_phase5_tracing_enabled()
    recorder = TrajectoryRecorder()
    recorder.record_manual(
        name="notevil_workaround_nofix_stage2.root_input",
        run_type="state",
        inputs=initial_state,
    )
    result: dict[str, Any] | None = None
    run_error: BaseException | None = None
    trace_url: str | None = None
    try:
        with use_trajectory_recorder(recorder):
            result = _execute_nofix(
                str(args.repo.resolve()),
                fixture,
                langsmith_extra={"run_id": trace_id},
            )
        if langsmith_enabled:
            trace_url = resolve_phase5_trace_url(trace_id)
            if trace_url:
                result["langsmith_trace_url"] = trace_url
        result["langsmith_run_id"] = str(trace_id)
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        recorder.record_manual(
            name="notevil_workaround_nofix_stage2.root_output",
            run_type="state",
            inputs={"error": str(run_error)} if run_error else None,
            outputs=result if result is not None else {"error": "no result"},
            error=run_error,
        )
        try:
            trajectory_path = export_phase5_trajectory(
                trace_id=trace_id,
                repo_root=str(args.repo.resolve()),
                initial_state=initial_state,
                final_state=result if result is not None else {"error": "no result"},
                recorder=recorder,
                langsmith_enabled=langsmith_enabled,
                langsmith_url=trace_url,
                run_error=run_error,
            )
            if result is not None:
                result["trajectory_path"] = str(trajectory_path)
        except Exception as export_error:  # noqa: BLE001
            logger.warning("NO_FIX trajectory export failed: %s", export_error)
            if result is not None:
                result.setdefault("errors", []).append(f"trajectory export failed: {export_error}")

    assert result is not None
    _write_outputs(result, output_path.resolve(), patch_path.resolve())
    print(f"NO_FIX status: {result['status']}")
    print(f"Result JSON: {output_path.resolve()}")
    print(f"Unified patch: {patch_path.resolve()}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
