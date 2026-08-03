"""Run a private Express-JWT workaround replay followed by QA.

The replay deliberately starts from the repository clone, prepares the
post-update dependency state inside an ephemeral Docker volume, and dispatches
the workaround worker bridge and then the QA Critic against the same volume. It
is intended for manual LangSmith/LLM trials and is not part of the public
remediation workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import traceable

from remediation_engine.contracts.schemas import (
    QAFailureEvidence,
    RemediationTask,
    TaskAttemptSnapshot,
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
from remediation_engine.orchestration.teardown_node import run_teardown_node
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    export_phase5_trajectory,
    use_trajectory_recorder,
)
from remediation_engine.orchestration.workspace_builder import (
    run_workspace_builder_node,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FIXTURE = Path(__file__).resolve().parent / "express_jwt_workaround_replay.json"
_DEFAULT_REPO = _PROJECT_ROOT / "data" / "clones" / "juice-shop"
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "trajectories"

_REPLAY_METADATA = {
    "replay_type": "express_jwt_workaround_only",
    "replay_mode": "workaround_only",
    "workflow_phase": "qa_regression_repair",
    "package_name": "express-jwt",
    "seed_target_version": "8.5.1",
    "source_trace_id": "249c81ed-5f64-42f2-98d8-013fb6ed0723",
    "source_update_attempt_id": "cf0b6bc0-4868-417d-86a4-fecbf3198afe",
}


def load_replay_fixture(path: Path = _DEFAULT_FIXTURE) -> dict[str, Any]:
    """Load and minimally validate the explicit replay fixture.

    Args:
        path: JSON fixture containing the vulnerability group, task instruction,
            and inherited QA evidence.

    Returns:
        The decoded fixture mapping.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the JSON root is not an object or required sections are
            missing.
        json.JSONDecodeError: If *path* is not valid JSON.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Replay fixture root must be an object: {path}")
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
        raise ValueError(f"Replay fixture is missing sections: {', '.join(missing)}")
    return payload


def _instruction_digest(instruction: str) -> str:
    """Return the same normalized instruction digest used by dispatch guards."""
    normalized = " ".join((instruction or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_replay_state(repo_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    """Construct a committed workaround task state without starting Docker.

    Args:
        repo_root: Host clone used only as the source for the workspace builder
            and later diff extraction.
        fixture: Parsed replay fixture.

    Returns:
        An orchestrator state containing the replay group, task, QA evidence,
        and immutable workaround attempt snapshot.

    Raises:
        pydantic.ValidationError: If a fixture section violates a Phase 5
            contract.
        KeyError: If required replay fields are absent.
    """
    group = VulnerabilityGroup.model_validate(fixture["group"])
    qa_payload = fixture["qa_evidence"]
    qa_evidence = QAFailureEvidence.model_validate(
        {key: value for key, value in qa_payload.items() if key != "retry_feedback"}
    )
    context_payload = fixture["workaround_context"]
    workaround_context = WorkaroundContext(
        phase=WorkaroundPhase(context_payload["phase"]),
        vulnerability_mechanism=context_payload["vulnerability_mechanism"],
        qa_evidence=qa_evidence,
    )

    task = RemediationTask.model_validate(fixture["task"])
    attempt_id = str(
        fixture["replay"].get(
            "source_workaround_attempt_id",
            "express-jwt-workaround-replay-attempt",
        )
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        state_revision=3,
        task_revision=task.task_revision,
        attempt_number=1,
        strategy_stage=task.strategy_stage,
        selected_version=task.selected_version,
        instruction=task.instruction,
        instruction_digest=_instruction_digest(task.instruction),
        dispatch_node="workaround_subagent",
        plan_id=None,
        workaround_context=workaround_context,
    )
    task = task.model_copy(update={"current_attempt_id": snapshot.attempt_id})

    retry_feedback = str(qa_payload.get("retry_feedback", "")).strip()
    state = initial_orchestrator_state(
        repo_root=str(repo_root),
        valid_groups=[group],
        issues=group.issues,
    )
    state.update(
        {
            "state_revision": snapshot.state_revision,
            "task_queue": {task.task_id: task},
            "active_target_task_ids": [task.task_id],
            "active_target_group_ids": [],
            "feedback_by_task": {task.task_id: retry_feedback},
            "feedback_by_group": {},
            "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            "workspace_volume": None,
            "status": "workaround_replay_pending",
        }
    )
    return state


def build_replay_state(repo_root: Path, fixture: dict[str, Any]) -> dict[str, Any]:
    """Publicly callable fixture/state path used for local dry verification.

    This helper performs no Docker, network, LLM, or LangSmith work.

    Args:
        repo_root: Host clone that will be used by a later replay invocation.
        fixture: Parsed output of :func:`load_replay_fixture`.

    Returns:
        A committed workaround dispatch state.
    """
    return _build_replay_state(repo_root.resolve(), fixture)


def _seed_post_update_workspace(
    sandbox: DockerSandbox,
    fixture: dict[str, Any],
) -> list[dict[str, Any]]:
    """Apply the target dependency update and lockfile sync inside the volume.

    Args:
        sandbox: Running sandbox connected to the workspace volume.
        fixture: Parsed replay fixture containing the package and commands.

    Returns:
        Structured command and verification diagnostics for the replay result.

    Raises:
        RuntimeError: If a seed command fails, the manifests do not resolve to
            the target version, or the source file changes during seeding.
    """
    replay = fixture["replay"]
    package_name = str(replay["package_name"])
    target_version = str(replay["target_update_version"])
    source_file = str(fixture["workspace_seed"]["source_file_to_leave_unchanged"])
    source_before = sandbox.read_file(source_file)
    if source_before is None:
        raise RuntimeError(f"Seed workspace is missing {source_file}.")

    commands = [
        (
            "set_manifest_version",
            f"npm pkg set dependencies.{package_name}={target_version}",
        ),
        (
            "synchronize_lockfile",
            str(fixture["workspace_seed"]["install_command"]),
        ),
        (
            "materialize_dependencies",
            "npm install --package-lock=true --no-audit --no-fund",
        ),
    ]
    diagnostics: list[dict[str, Any]] = []
    for label, command in commands:
        result = sandbox.run(f"cd /workspace && {command}")
        diagnostics.append(
            {
                "step": label,
                "command": command,
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_seconds": result.duration_seconds,
            }
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Workspace seed step '{label}' failed with exit code "
                f"{result.exit_code}: {result.stderr or result.stdout}"
            )

    package_text = sandbox.read_file("package.json")
    lockfile_text = sandbox.read_file("package-lock.json")
    source_after = sandbox.read_file(source_file)
    if package_text is None or lockfile_text is None:
        raise RuntimeError("Workspace seed did not produce both npm manifest files.")
    if source_after != source_before:
        raise RuntimeError(
            f"Workspace seed unexpectedly changed {source_file}; refusing to dispatch."
        )

    try:
        package_payload = json.loads(package_text)
        lockfile_payload = json.loads(lockfile_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Workspace seed produced invalid npm JSON: {exc}") from exc

    manifest_version = package_payload.get("dependencies", {}).get(package_name)
    lock_version = (
        lockfile_payload.get("packages", {}).get(f"node_modules/{package_name}", {}).get("version")
    )
    if manifest_version != target_version or lock_version != target_version:
        raise RuntimeError(
            f"Workspace seed version mismatch: manifest={manifest_version!r}, "
            f"lockfile={lock_version!r}, target={target_version!r}."
        )

    installed_pkg_text = sandbox.read_file(f"node_modules/{package_name}/package.json")
    installed_version = None
    if installed_pkg_text:
        with suppress(json.JSONDecodeError):
            installed_version = json.loads(installed_pkg_text).get("version")

    if installed_version != target_version:
        raise RuntimeError(
            f"Installed package version mismatch for {package_name}: expected {target_version!r}, "
            f"got {installed_version!r}."
        )

    tsc_check = sandbox.run("cd /workspace && npx --no-install tsc --version")
    tsx_check = sandbox.run("cd /workspace && npx --no-install tsx --version")
    if tsc_check.exit_code != 0 or tsx_check.exit_code != 0:
        raise RuntimeError(
            f"Required tooling verification failed: tsc exit={tsc_check.exit_code}, tsx exit={tsx_check.exit_code}."
        )

    diagnostics.append(
        {
            "step": "verify_seed",
            "package": package_name,
            "manifest_version": manifest_version,
            "lockfile_version": lock_version,
            "installed_version": installed_version,
            "tsc_version": tsc_check.stdout.strip(),
            "tsx_version": tsx_check.stdout.strip(),
            "source_file_unchanged": True,
        }
    )
    return diagnostics


def _merge_worker_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    """Merge the direct workaround bridge result into mutable replay state."""
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
        worker_map = result["worker_results_by_attempt"]
        state["worker_results_by_attempt"] = {
            **state.get("worker_results_by_attempt", {}),
            **worker_map,
        }
        for attempt_res in worker_map.values():
            errs = getattr(attempt_res, "errors", None) or (
                attempt_res.get("errors") if isinstance(attempt_res, dict) else None
            )
            if errs:
                state["errors"] = list(dict.fromkeys([*state.get("errors", []), *errs]))
            diag = getattr(attempt_res, "execution_diagnostics", None) or (
                attempt_res.get("execution_diagnostics") if isinstance(attempt_res, dict) else None
            )
            if diag:
                reason = getattr(diag, "failure_reason", None) or (
                    diag.get("failure_reason") if isinstance(diag, dict) else None
                )
                if reason:
                    state["errors"] = list(dict.fromkeys([*state.get("errors", []), str(reason)]))

    if result.get("errors"):
        state["errors"] = list(dict.fromkeys([*state.get("errors", []), *result["errors"]]))


def _merge_qa_result(state: dict[str, Any], result: dict[str, Any]) -> None:
    """Merge QA output before teardown reconciles task status and extracts a diff.

    The normal graph sends the QA result through the Supervisor before teardown.
    This replay invokes QA directly, so the result must be copied into the
    mutable state explicitly. In particular, ``qa_evaluations`` and
    ``qa_results_by_attempt`` are required by the teardown reconciliation
    barrier to mark the replayed task as QA-passed or unfixable.

    Args:
        state: Mutable replay state to update.
        result: Output from :func:`run_qa_critic_from_orchestrator`.
    """
    for key in (
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


def _jsonable(value: Any) -> Any:
    """Convert contract values and enums into JSON-serializable values."""
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
    name="Express_JWT_Workaround_Replay_With_QA",
    run_type="chain",
    metadata=_REPLAY_METADATA,
    tags=["express-jwt", "workaround", "qa-critic", "qa-regression-repair", "replay"],
)
def _execute_replay(repo_root: str, fixture: dict[str, Any]) -> dict[str, Any]:
    """Execute the Docker-seeded workaround replay and QA validation.

    Args:
        repo_root: Host clone used as the immutable workspace source and diff
            baseline.
        fixture: Explicit replay fixture payload.

    Returns:
        JSON-serializable replay outcome including worker summaries, QA
        evaluations, cleanup status, and the unified patch.

    Side effects:
        Creates and removes a temporary Docker volume, invokes the workaround
        subagent/LLM, and reads the resulting patch from the volume. The host
        clone is never edited.
    """
    repo_path = Path(repo_root).resolve()
    state = _build_replay_state(repo_path, fixture)
    seed_diagnostics: list[dict[str, Any]] = []
    worker_result: dict[str, Any] = {}
    qa_result: dict[str, Any] = {}
    teardown_result: dict[str, Any] = {}

    try:
        builder_result = run_workspace_builder_node(state)
        state.update(builder_result)
        if builder_result.get("errors"):
            state["errors"] = [*state.get("errors", []), *builder_result["errors"]]

        workspace_volume = state.get("workspace_volume")
        if builder_result.get("status") != "workspace_ready" or not workspace_volume:
            raise RuntimeError("Workspace builder did not return a ready Docker volume.")

        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            seed_diagnostics = _seed_post_update_workspace(sandbox, fixture)

        # This is the sole worker dispatch in this script. It receives the
        # committed task snapshot, seeded post-update workspace, and inherited
        # QA regression evidence through the normal workaround bridge.
        worker_result = run_workaround_subagent_from_orchestrator(state)
        _merge_worker_result(state, worker_result)

        # Run QA against the same post-workaround volume before teardown. This
        # mirrors the graph's worker -> QA handoff while retaining the replay's
        # direct bridge invocation. The merged QA envelopes are then consumed
        # by teardown's reconciliation barrier.
        qa_result = run_qa_critic_from_orchestrator(state)
        _merge_qa_result(state, qa_result)
    except Exception as exc:  # noqa: BLE001
        message = f"Express-JWT workaround replay failed before completion: {exc}"
        logger.exception(message)
        state["errors"] = [*state.get("errors", []), message]
    finally:
        workspace_volume = state.get("workspace_volume")
        if workspace_volume:
            try:
                teardown_result = run_teardown_node(state)
                state.update(teardown_result)
            except Exception as exc:  # noqa: BLE001
                message = f"Express-JWT workaround replay teardown failed: {exc}"
                logger.exception(message)
                state["errors"] = [*state.get("errors", []), message]

    action_summaries = _jsonable(state.get("action_summaries", []))
    worker_status = None
    if action_summaries:
        worker_status = action_summaries[-1].get("status")
    qa_status = qa_result.get("status")
    qa_eval_status = qa_result.get("eval_status")
    qa_passed = qa_status == "qa_completed" and qa_eval_status == "all_passed"
    cleanup_ok = not workspace_volume or teardown_result.get("workspace_volume") is None
    replay_status = (
        "success"
        if worker_status == "success" and qa_passed and not state.get("errors") and cleanup_ok
        else "surrender"
    )

    diff_content = state.get("diff", "") if replay_status == "success" else ""

    return _jsonable(
        {
            "status": replay_status,
            "replay": fixture["replay"],
            "provenance": fixture["provenance"],
            "workspace_seed": {
                **fixture["workspace_seed"],
                "diagnostics": seed_diagnostics,
            },
            "worker": {
                "status": worker_status,
                "action_summaries": action_summaries,
                "worker_results_by_attempt": state.get("worker_results_by_attempt", {}),
                "changed_files": state.get("changed_files", [])
                if replay_status == "success"
                else [],
            },
            "qa": {
                **qa_result,
                "passed": qa_passed,
                "status": qa_status,
                "eval_status": qa_eval_status,
            },
            "teardown": {
                "status": teardown_result.get("status"),
                "workspace_volume": teardown_result.get("workspace_volume"),
                "volume_removed": bool(workspace_volume)
                and teardown_result.get("workspace_volume") is None,
            },
            "changed_files": state.get("changed_files", []) if replay_status == "success" else [],
            "diff": diff_content,
            "errors": state.get("errors", []),
        }
    )


def _write_outputs(
    result: dict[str, Any],
    output_path: Path,
    patch_path: Path,
) -> None:
    """Write the replay JSON report and unified patch."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    patch_path.write_text(str(result.get("diff", "")), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse private replay command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=_DEFAULT_REPO)
    parser.add_argument("--fixture", type=Path, default=_DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--patch-output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    """Run the manually invoked replay and return a shell exit code."""
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    fixture = load_replay_fixture(args.fixture.resolve())
    build_replay_state(args.repo, fixture)
    output_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        _DEFAULT_OUTPUT_DIR / f"express_jwt_workaround_replay_{output_stamp}.json"
    )
    patch_path = args.patch_output or (
        _DEFAULT_OUTPUT_DIR / f"express_jwt_workaround_replay_{output_stamp}.patch"
    )

    initial_state = build_replay_state(args.repo, fixture)
    trace_id = uuid.uuid4()
    langsmith_enabled = is_phase5_tracing_enabled()
    recorder = TrajectoryRecorder()
    recorder.record_manual(
        name="express_jwt_workaround_replay.root_input",
        run_type="state",
        inputs=initial_state,
    )
    result: dict[str, Any] | None = None
    run_error: BaseException | None = None
    trace_url: str | None = None
    try:
        with use_trajectory_recorder(recorder):
            result = _execute_replay(
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
            name="express_jwt_workaround_replay.root_output",
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
            logging.getLogger(__name__).warning(
                "workaround replay trajectory export failed: %s", export_error
            )
            if result is not None:
                result.setdefault("errors", []).append(f"trajectory export failed: {export_error}")

    assert result is not None
    _write_outputs(result, output_path.resolve(), patch_path.resolve())
    print(f"Replay status: {result['status']}")
    qa_output = result.get("qa", {})
    print(f"QA status: {qa_output.get('status')} ({qa_output.get('eval_status')})")
    print(f"Result JSON: {output_path.resolve()}")
    print(f"Unified patch: {patch_path.resolve()}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
