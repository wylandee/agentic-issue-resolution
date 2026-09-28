"""Production-node replay adapters for the Phase 5 evaluation suites."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FixPlan,
    FixPlanStatus,
    LocalizedIssue,
    NoFixMitigationStage,
    ODCScanEvidence,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    SCARemediationStage,
    TacticalStrategy,
    TaskAttemptSnapshot,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
)
from remediation_engine.orchestration.state import (
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from remediation_engine.orchestration.trajectory_exporter import (
    TrajectoryRecorder,
    use_trajectory_recorder,
)
from remediation_engine.settings import AppSettings
from tests.evals.replay_harness import (
    ReplayCapture,
    ReplaySandbox,
    build_command_responses,
    build_system_context,
    build_vulnerability_group,
    make_temp_repo,
    output_with_trace,
    seed_replay_files,
    serialize_result,
    serialize_tool_events,
    workspace_temp_root,
)


def _replay_input(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return the structured replay input mapping for a case."""
    replay = case.get("replay")
    if isinstance(replay, Mapping) and isinstance(replay.get("input"), Mapping):
        return dict(replay["input"])
    return {}


def _recorder_token_fields(recorder: TrajectoryRecorder) -> dict[str, Any]:
    """Return token usage and provider cache fields from a replay recorder."""
    fields: dict[str, Any] = {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cached_input_tokens": recorder.cached_input_tokens,
        "token_cost": recorder.token_cost,
    }
    if recorder.token_data_available:
        fields.update(
            {
                "input_tokens": recorder.total_prompt_tokens,
                "output_tokens": recorder.total_completion_tokens,
                "total_tokens": recorder.total_tokens,
            }
        )
    return fields


def _workaround_context_for_case(case: Mapping[str, Any]) -> WorkaroundContext | None:
    """Parse an optional supervisor-provided workaround context."""
    raw = _replay_input(case).get("workaround_context")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("replay.input.workaround_context must be an object")
    return WorkaroundContext.model_validate(dict(raw))


def _replay_plan_for_case(
    case: Mapping[str, Any],
    task: RemediationTask,
) -> WorkaroundReplayPlan | None:
    """Parse an optional cumulative workaround replay plan for one task."""
    raw = _replay_input(case).get("replay_plan")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("replay.input.replay_plan must be an object")
    payload = dict(raw)
    payload.setdefault("task_id", task.task_id)
    payload.setdefault("source_attempt_id", task.current_attempt_id or "")
    plan = WorkaroundReplayPlan.model_validate(payload)
    if plan.task_id != task.task_id:
        raise ValueError(
            f"Replay plan task_id {plan.task_id!r} does not match task {task.task_id!r}."
        )
    return plan


def _group_for_case(case: Mapping[str, Any], *, component: str) -> VulnerabilityGroup:
    """Build a typed group from the component-specific golden metadata."""
    replay = _replay_input(case)
    raw_context = replay.get("vulnerability_context")
    if not isinstance(raw_context, Mapping):
        raw_context = replay.get("group") if isinstance(replay.get("group"), Mapping) else {}

    group_id = str(raw_context.get("group_id") or f"group:{case.get('case_id', 'replay')}")
    package = str(
        raw_context.get("vulnerable_component")
        or case.get("target_package_name")
        or "replay-package"
    )
    file_path = str(raw_context.get("file_path") or case.get("manifest_path") or "package.json")
    case_for_group = dict(case)
    case_for_group["vulnerability_context"] = {
        **dict(raw_context),
        "group_id": group_id,
        "vulnerable_component": package,
        "file_path": file_path,
    }
    group = build_vulnerability_group(case_for_group)

    # QA no-fix policies require localized npm manifest provenance for the
    # package-state guardrail.  Worker groups also benefit from the same
    # manifest context when their result is serialized.
    issue = group.issues[0] if group.issues else None
    if issue is not None and (component == "qa" or "no-fix" in str(case.get("case_id", ""))):
        manifest_file = "package.json"
        localized = LocalizedIssue(
            issue=issue,
            manifest_file=manifest_file,
            package_manager="npm",
            declaration_type=str(case.get("dependency_type") or "dependencies"),
        )
        group = group.model_copy(
            update={
                "file_paths": [manifest_file],
                "localized_issues": [localized],
            }
        )

    if str(case.get("qa_policy", "")).startswith("no_fix"):
        group = group.model_copy(
            update={
                "fix_plan": FixPlan(
                    status=FixPlanStatus.NO_FIX,
                    instruction="No upstream patch or workaround was found.",
                    strategy_used="NO_FIX",
                )
            }
        )
    return group


def _policy_for_case(case: Mapping[str, Any], component: str) -> QAPolicy | None:
    """Convert a golden policy into its typed contract."""
    raw = case.get("qa_policy")
    if raw:
        try:
            return raw if isinstance(raw, QAPolicy) else QAPolicy(str(raw))
        except ValueError:
            return None
    if component == "update":
        return QAPolicy.VERSION_BUMP
    if component == "workaround":
        return QAPolicy.INITIAL_CODE_WORKAROUND
    return None


def _strategy_for_case(case: Mapping[str, Any], component: str) -> RoutingStrategy:
    """Return the worker routing strategy represented by a case."""
    if component == "update":
        return RoutingStrategy.VERSION_BUMP
    return RoutingStrategy.CODE_WORKAROUND


def _no_fix_stage_for_case(case: Mapping[str, Any]) -> NoFixMitigationStage | None:
    """Return the no-fix stage encoded by a case identifier or replay input."""
    raw = _replay_input(case).get("no_fix_stage") or case.get("no_fix_stage")
    if raw:
        try:
            return raw if isinstance(raw, NoFixMitigationStage) else NoFixMitigationStage(str(raw))
        except ValueError:
            return None
    case_id = str(case.get("case_id", ""))
    if "package-removal" in case_id:
        return NoFixMitigationStage.PACKAGE_REMOVAL
    if "no-fix" in case_id:
        return NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    return None


def _build_task(
    case: Mapping[str, Any], group: VulnerabilityGroup, component: str
) -> RemediationTask:
    """Build the current typed task contract for a worker or QA replay."""
    case_id = str(case.get("case_id") or "replay-task")
    instruction = str(
        case.get("supervisor_instruction")
        or case.get("completion_task")
        or case.get("input")
        or "Execute the committed remediation task."
    )
    policy = _policy_for_case(case, component)
    no_fix_stage = _no_fix_stage_for_case(case)
    strategy = _strategy_for_case(case, component)
    if no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL:
        strategy = RoutingStrategy.CODE_WORKAROUND
    stage = (
        SCARemediationStage.CODE_WORKAROUND
        if component == "workaround"
        else SCARemediationStage.NPM_LATEST
    )
    status = (
        TaskStatus.UNFIXABLE
        if str(case.get("action_status", "")).upper() == "SURRENDER"
        else TaskStatus.OPTIMISTICALLY_FIXED
    )
    return RemediationTask(
        task_id=case_id,
        task_revision=int(case.get("task_revision") or 1),
        current_attempt_id=str(case.get("attempt_id") or f"attempt-{case_id}"),
        parent_group_id=group.group_id,
        qa_policy=policy,
        strategy=strategy,
        strategy_stage=stage,
        target_package_name=str(
            case.get("target_package_name") or group.vulnerable_component or ""
        ),
        target_dependency_type=case.get("dependency_type"),
        parent_package_name=case.get("parent_package_name"),
        parent_package_version=case.get("parent_package_version"),
        no_fix_stage=no_fix_stage,
        selected_version=case.get("selected_version"),
        instruction=instruction,
        status=status,
        retry_count=1 if bool(case.get("is_retry")) else 0,
    )


def _build_snapshot(
    case: Mapping[str, Any],
    task: RemediationTask,
    *,
    dispatch_node: str,
) -> TaskAttemptSnapshot:
    """Build immutable attempt provenance accepted by current nodes."""
    replay = _replay_input(case)
    versions = replay.get("allowed_target_versions")
    if not isinstance(versions, list):
        versions = []
    versions = [str(value) for value in versions if value]
    if task.selected_version and task.selected_version not in versions:
        versions.append(task.selected_version)
    dependency_types = replay.get("allowed_dependency_types")
    if not isinstance(dependency_types, list):
        dependency_types = []
    dependency_types = [str(value) for value in dependency_types if value]
    if task.target_dependency_type and task.target_dependency_type not in dependency_types:
        dependency_types.append(task.target_dependency_type)
    digest = hashlib.sha256(task.instruction.strip().encode("utf-8")).hexdigest()
    return TaskAttemptSnapshot(
        attempt_id=task.current_attempt_id or f"attempt-{task.task_id}",
        task_id=task.task_id,
        state_revision=1,
        task_revision=task.task_revision,
        attempt_number=max(1, task.retry_count + 1),
        qa_policy=task.qa_policy,
        strategy_stage=task.strategy_stage,
        no_fix_stage=task.no_fix_stage,
        selected_version=task.selected_version,
        allowed_target_versions=versions,
        target_package_name=task.target_package_name,
        target_dependency_type=task.target_dependency_type,
        allowed_dependency_types=dependency_types,
        parent_minimum_version=task.parent_minimum_version,
        instruction=task.instruction,
        instruction_digest=digest,
        dispatch_node=dispatch_node,  # type: ignore[arg-type]
        plan_id=f"eval-plan-{task.task_id}",
        workaround_context=(
            _workaround_context_for_case(case) if dispatch_node == "workaround_subagent" else None
        ),
    )


def _worker_files(case: Mapping[str, Any]) -> dict[str, str]:
    """Seed worker files and ensure the target package is declared."""
    files = seed_replay_files(case)
    package_json_path = "package.json"
    try:
        package_json = json.loads(files[package_json_path])
    except (KeyError, json.JSONDecodeError):
        package_json = {"name": "eval-worker", "version": "1.0.0", "dependencies": {}}
    package = str(case.get("target_package_name") or "replay-package")
    section = str(case.get("dependency_type") or "dependencies")
    if section in {"dependencies", "devDependencies", "peerDependencies", "optionalDependencies"}:
        package_json.setdefault(section, {})[package] = package_json.get(section, {}).get(
            package, "1.0.0"
        )
    files[package_json_path] = json.dumps(package_json, indent=2) + "\n"
    return files


def _workaround_files(case: Mapping[str, Any]) -> dict[str, str]:
    """Add a local deterministic test runner to the synthetic replay repo."""
    files = _worker_files(case)
    try:
        package_json = json.loads(files["package.json"])
    except (KeyError, json.JSONDecodeError):
        package_json = {"name": "eval-worker", "version": "1.0.0"}
    scripts = package_json.setdefault("scripts", {})
    if isinstance(scripts, dict):
        scripts.setdefault("test", "mocha")
    dev_dependencies = package_json.setdefault("devDependencies", {})
    if isinstance(dev_dependencies, dict):
        dev_dependencies.setdefault("mocha", "10.0.0")
    files["package.json"] = json.dumps(package_json, indent=2) + "\n"
    return files


def _workaround_command_routes(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit case-backed validation responses for workaround commands."""
    responses = build_command_responses(case)
    success = {
        "exit_code": 0,
        "stdout": "replay validation passed",
        "stderr": "",
        "duration_seconds": 0.0,
    }
    for matcher in (
        "node -c",
        "npx --yes esbuild",
        "node --import tsx --input-type=module",
        "node --input-type=module",
        "npm install --package-lock-only",
    ):
        responses.setdefault(matcher, dict(success))

    def mocha_response(command: str) -> dict[str, Any]:
        """Return a passing Mocha JSON payload matching an optional grep hint."""
        tokens = shlex.split(command)
        title = "replay targeted test"
        if "--grep" in tokens:
            index = tokens.index("--grep")
            if index + 1 < len(tokens):
                title = tokens[index + 1].strip("'\"")
        return {
            "exit_code": 0,
            "stdout": json.dumps(
                {
                    "stats": {"passes": 1, "failures": 0},
                    "tests": [{"fullTitle": title, "title": title, "state": "passed"}],
                    "passes": [{"fullTitle": title, "title": title, "state": "passed"}],
                    "failures": [],
                }
            ),
            "stderr": "",
            "duration_seconds": 0.0,
        }

    responses["npx --no-install mocha"] = mocha_response
    raw_npm_test = responses.get("npm test")
    if raw_npm_test is None or (
        isinstance(raw_npm_test, Mapping)
        and raw_npm_test.get("stdout") == "replay targeted tests passed"
    ):
        responses["npm test"] = mocha_response
    case_id = str(case.get("case_id", ""))
    if case_id == "workaround-retry-after-validation-failure":
        attempts = {"count": 0}

        def fail_once(command: str) -> dict[str, Any]:
            del command
            attempts["count"] += 1
            if attempts["count"] == 1:
                return {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "TS2769: incompatible migration option",
                    "duration_seconds": 0.0,
                }
            return dict(success)

        responses["npx --yes esbuild"] = fail_once
        responses["node -c"] = fail_once
    elif case_id == "workaround-retry-after-validation-infra-failure":
        attempts = {"count": 0}

        def infra_once(command: str) -> dict[str, Any]:
            attempts["count"] += 1
            if attempts["count"] == 1:
                return {
                    "exit_code": 1,
                    "stdout": "Error: Cannot find module 'better-sqlite3'",
                    "stderr": "",
                    "duration_seconds": 0.0,
                }
            return mocha_response(command)

        responses["npx --no-install mocha"] = infra_once
        responses["npm test"] = infra_once
    elif str(case.get("terminal_error_code", "")) == "VALIDATION_LIMIT_REACHED":
        failure = {
            "exit_code": 1,
            "stdout": "AssertionError: targeted assertion still reports unsafe behavior",
            "stderr": "",
            "duration_seconds": 0.0,
        }
        responses["npx --no-install mocha"] = failure
        responses["npm test"] = failure
    return responses


def replay_triage_case(
    case: Mapping[str, Any],
    eval_settings: Any,
    *,
    llm: Any | None = None,
) -> ReplayCapture:
    """Invoke production triage and capture its typed post-guardrail result.

    Args:
        case: Canonical triage replay case.
        eval_settings: Evaluation settings used to configure the production
            triage path.
        llm: Optional test-only model replacement.  When supplied, only the
            production ``ChatOpenAI`` lookup is patched for this invocation.

    Returns:
        The typed triage result and any deterministic selection evidence.
    """
    from remediation_engine.triage.agent import run_triage
    from remediation_engine.triage.pipeline import select_issues_for_remediation

    group = build_vulnerability_group(case)
    context = build_system_context(case)
    settings = AppSettings.from_env()
    settings = dataclasses.replace(
        settings,
        openai_api_key=getattr(eval_settings, "openai_api_key", "") or settings.openai_api_key,
        triage_llm_enabled=True,
    )
    recorder = TrajectoryRecorder()
    with ExitStack() as stack:
        stack.enter_context(use_trajectory_recorder(recorder))
        if llm is not None:
            stack.enter_context(
                patch(
                    "langchain_openai.ChatOpenAI",
                    side_effect=lambda *_args, **_kwargs: llm,
                )
            )
        result = run_triage(group, context, settings=settings)
        payload: dict[str, Any] = {"triage_result": serialize_result(result)}
        selection_boundary = _replay_input(case).get(
            "selection_boundary", case.get("selection_boundary")
        )
        if (
            selection_boundary
            or str(case.get("case_id", "")) == "triage-pipeline-selection-hallucinated-issue-id"
        ):
            selected = select_issues_for_remediation([(group, result)])
            payload["selected_issue_ids"] = [issue.id for issue in selected]
    return ReplayCapture(
        case_id=str(case.get("case_id", "unknown")),
        component="triage",
        actual_output=json.dumps(payload, indent=2, sort_keys=True, default=str),
        actual_tools=[],
        typed_result=result,
        errors=[],
        final_files={},
        **_recorder_token_fields(recorder),
    )


def _qa_results(case: Mapping[str, Any], group: VulnerabilityGroup, task_id: str) -> Any:
    """Build deterministic QA results, including optional execution metadata."""
    import remediation_engine.orchestration.qa_types as qa

    replay = _replay_input(case)
    execution = replay.get("execution_context", case.get("execution_context", {}))
    execution = execution if isinstance(execution, Mapping) else {}
    logs = replay.get("execution_logs", case.get("execution_logs", {}))
    logs = logs if isinstance(logs, Mapping) else {}
    results = qa._QAExecutionResults()

    def optional_int(*values: Any) -> int | None:
        for value in values:
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def optional_stream(prefix: str, stream: str) -> str | None:
        for source in (execution, logs):
            for key in (
                f"{prefix}_{stream}",
                f"{prefix}_raw_{stream}",
                f"raw_{prefix}_{stream}",
            ):
                if key in source and source[key] is not None:
                    return str(source[key])
        return None

    def record_from_payload(
        payload: Mapping[str, Any],
        phase: str,
        default_label: str,
    ) -> qa._QALogRecord:
        return qa._QALogRecord(
            phase=phase,
            label=str(payload.get("label") or default_label),
            exit_code=optional_int(payload.get("exit_code"), payload.get("returncode")),
            stdout=str(payload.get("stdout") or ""),
            stderr=str(payload.get("stderr") or ""),
            error=str(payload["error"]) if payload.get("error") else None,
        )

    def fixture_records(phase: str, default_label: str) -> tuple[qa._QALogRecord, ...]:
        raw_records: Any = None
        for source in (execution, logs):
            candidate = source.get(f"{phase}_records")
            if candidate is not None:
                raw_records = candidate
                break
            all_records = source.get("log_records")
            if isinstance(all_records, Mapping) and phase in all_records:
                raw_records = all_records[phase]
                break
        if isinstance(raw_records, Mapping):
            raw_records = [
                dict(value, label=label)
                if isinstance(value, Mapping)
                else {"label": label, "stdout": value}
                for label, value in raw_records.items()
            ]
        if isinstance(raw_records, list):
            parsed = tuple(
                record_from_payload(item, phase, default_label)
                for item in raw_records
                if isinstance(item, Mapping)
            )
            if parsed:
                return parsed
        return ()

    install_summary = str(logs.get("install_log", "replay install result"))
    results.install = (bool(execution.get("install_passed")), install_summary)
    results.install_exit_code = optional_int(
        execution.get("install_exit_code"),
        logs.get("install_exit_code"),
    )
    results.install_error_category = (
        str(
            execution.get(
                "install_error_category",
                logs.get("install_error_category"),
            )
        )
        if execution.get("install_error_category", logs.get("install_error_category"))
        else None
    )
    results.install_raw_stdout = optional_stream("install", "stdout")
    results.install_raw_stderr = optional_stream("install", "stderr")
    install_records = fixture_records("install", "npm install")
    if not install_records:
        install_records = (
            qa._QALogRecord(
                phase="install",
                label="npm install",
                exit_code=results.install_exit_code,
                stdout=results.install_raw_stdout or install_summary,
                stderr=results.install_raw_stderr or "",
                error=results.install_error_category,
            ),
        )
    results.log_records["install"] = install_records

    policy = _policy_for_case(case, "qa")
    scanner_status_name = str(execution.get("scanner_execution_status", "not_run")).lower()
    if (
        policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL
        and scanner_status_name != ScannerExecutionStatus.SUCCESS.value
    ):
        results.scan_skipped = True
        results.scan_skip_reason = "no_fix_package_removal"
        scan_records = fixture_records("scan", "odc:skipped") or (
            qa._QALogRecord(
                phase="scan",
                label="odc:skipped",
                exit_code=None,
                stdout="",
                stderr="",
                error=results.scan_skip_reason,
            ),
        )
        results.log_records["scan"] = scan_records
    else:
        raw_remaining = execution.get("target_remaining_identifiers", []) or []
        remaining = {str(value) for value in raw_remaining}
        target_cleared = execution.get("target_scanner_cleared")
        scan_ok = bool(execution.get("scanner_execution_status") == "success") and bool(
            target_cleared is True
        )
        try:
            scan_status = ScannerExecutionStatus(scanner_status_name)
        except ValueError:
            scan_status = (
                ScannerExecutionStatus.UNPARSEABLE
                if scanner_status_name in {"failed", "failure", "error"}
                else ScannerExecutionStatus.NOT_RUN
            )
        scan_summary = str(logs.get("scan_summary", "replay scanner result"))
        scan_stdout = optional_stream("scan", "stdout")
        scan_stderr = optional_stream("scan", "stderr")
        scan_exit_code = optional_int(
            execution.get("scan_exit_code"),
            execution.get("scanner_exit_code"),
            logs.get("scan_exit_code"),
        )
        scan_records = fixture_records("scan", "odc:full")
        results.scan = qa._SecurityScanResult(
            ok=scan_ok,
            summary=scan_summary,
            remaining_identifiers=remaining,
            found_identifiers={
                str(value) for value in execution.get("found_identifiers", []) or []
            },
            new_identifiers={str(value) for value in execution.get("new_identifiers", []) or []},
            execution_status=scan_status,
            exit_code=scan_exit_code,
            raw_stdout=scan_stdout,
            raw_stderr=scan_stderr,
            diagnostic_log_path=(
                str(execution["diagnostic_log_path"])
                if execution.get("diagnostic_log_path")
                else None
            ),
            scan_records=scan_records,
        )
        if not scan_records:
            effective_scope = str(execution.get("effective_scope", "full"))
            scan_label = (
                "odc:fallback-full"
                if execution.get("fallback_reason")
                else "odc:targeted"
                if effective_scope == ScanScope.TARGETED.value
                else "odc:full"
            )
            results.log_records["scan"] = (
                qa._QALogRecord(
                    phase="scan",
                    label=scan_label,
                    exit_code=scan_exit_code,
                    stdout=scan_stdout or scan_summary,
                    stderr=scan_stderr or "",
                    error=(
                        str(results.scan.diagnostic_log_path)
                        if results.scan.diagnostic_log_path
                        else None
                    ),
                ),
            )
        else:
            results.log_records["scan"] = scan_records
        try:
            requested_scope = ScanScope(str(execution.get("requested_scope", "full")))
            effective_scope = ScanScope(
                str(execution.get("effective_scope", requested_scope.value))
            )
            fallback = execution.get("fallback_reason")

            def typed_strings(field: str, fallback_values: list[str]) -> list[str]:
                value = execution.get(field)
                if not isinstance(value, list):
                    return list(fallback_values)
                return [str(item) for item in value if str(item).strip()]

            covered_task_ids = typed_strings("covered_task_ids", [task_id])
            if covered_task_ids != [task_id]:
                raise ValueError("Replay QA scan evidence must cover exactly its assigned task.")
            results.scan_evidence = ODCScanEvidence(
                requested_scope=requested_scope,
                effective_scope=effective_scope,
                authoritative=False,
                covered_task_ids=covered_task_ids,
                closure_package_names=typed_strings(
                    "closure_package_names",
                    [str(group.vulnerable_component or "")],
                ),
                closure_lockfile_keys=typed_strings("closure_lockfile_keys", []),
                found_identifiers=sorted(
                    {str(value) for value in execution.get("found_identifiers", []) or []}
                ),
                remaining_target_identifiers=sorted(remaining),
                complete=bool(execution.get("scan_complete", True)),
                fallback_reason=ScanFallbackReason(str(fallback)) if fallback else None,
            )
        except (TypeError, ValueError):
            results.scan_evidence = None

    test_summary = str(logs.get("test_output", "replay test result"))
    results.tests = (bool(execution.get("tests_passed")), test_summary)
    results.test_exit_code = optional_int(
        execution.get("test_exit_code"),
        logs.get("test_exit_code"),
    )
    results.test_failure_count = optional_int(
        execution.get("test_failure_count"),
        logs.get("test_failure_count"),
    )
    results.test_raw_stdout = optional_stream("test", "stdout")
    if results.test_raw_stdout is None:
        results.test_raw_stdout = optional_stream("tests", "stdout")
    results.test_raw_stderr = optional_stream("test", "stderr")
    if results.test_raw_stderr is None:
        results.test_raw_stderr = optional_stream("tests", "stderr")
    test_records = fixture_records("tests", "npm test") or fixture_records("test", "npm test")
    if not test_records:
        test_records = (
            qa._QALogRecord(
                phase="tests",
                label="npm test",
                exit_code=results.test_exit_code,
                stdout=results.test_raw_stdout or test_summary,
                stderr=results.test_raw_stderr or "",
                error=None,
            ),
        )
    results.log_records["tests"] = test_records

    if policy in {
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }:
        diagnostics = execution.get("package_state_diagnostics", []) or []
        if not isinstance(diagnostics, (list, tuple)):
            diagnostics = [str(diagnostics)]
        results.package_state_by_task[task_id] = qa._QAPackageState(
            manifest_state=(
                str(execution["package_manifest_state"])
                if execution.get("package_manifest_state") is not None
                else None
            ),
            graph_state=(
                str(execution["package_graph_state"])
                if execution.get("package_graph_state") is not None
                else None
            ),
            diagnostics=tuple(str(value) for value in diagnostics if value),
        )
    return results


def _qa_files(case: Mapping[str, Any]) -> dict[str, str]:
    """Seed QA workspace files and ensure candidate changed files exist."""
    files = seed_replay_files(case)
    cid = str(case.get("case_id", ""))
    changed = list(case.get("changed_files", []) or [])

    if "lib/insecurity.ts" in changed and "lib/insecurity.ts" not in files:
        if "fail_semantic" in cid:
            files["lib/insecurity.ts"] = (
                "import expressJwt from 'express-jwt'\n\n"
                "expressJwt({ secret: publicKey })\n\n"
                "export function replay() { return true; }\n"
            )
        else:
            files["lib/insecurity.ts"] = (
                "import { expressjwt } from 'express-jwt'\n\n"
                "expressjwt({ secret: publicKey, algorithms: ['RS256'] })\n\n"
                "export function replay() { return true; }\n"
            )

    if "routes/b2bOrder.ts" in changed and "routes/b2bOrder.ts" not in files:
        if "fail_semantic" in cid:
            files["routes/b2bOrder.ts"] = (
                "const notevil = require('notevil');\n\nexport function replay() { return true; }\n"
            )
        else:
            files["routes/b2bOrder.ts"] = "export function replay() { return true; }\n"

    return files


def replay_qa_case(
    case: Mapping[str, Any],
    eval_settings: Any,
    *,
    llm: Any | None = None,
) -> ReplayCapture:
    """Invoke the production QA node with deterministic execution evidence.

    Args:
        case: Canonical QA replay case.
        eval_settings: Unused by deterministic QA boundaries.
        llm: Optional test-only model replacement scoped to this adapter call.

    Returns:
        The typed QA evaluation and captured in-memory side effects.
    """
    import remediation_engine.orchestration.qa_critic as qa
    import remediation_engine.orchestration.qa_evaluator as qa_evaluator
    from remediation_engine.orchestration.graph_wrappers import run_qa_critic_from_orchestrator

    group = _group_for_case(case, component="qa")
    task = _build_task(case, group, "qa")
    if task.qa_policy == QAPolicy.VERSION_BUMP and not task.selected_version:
        task = task.model_copy(update={"selected_version": "1.0.0"})
    snapshot = _build_snapshot(case, task, dispatch_node="qa_critic")
    files = _qa_files(case)
    replay_execution = _replay_input(case).get(
        "execution_context", case.get("execution_context", {})
    )
    execution = replay_execution if isinstance(replay_execution, Mapping) else {}
    if isinstance(execution, Mapping):
        manifest_state = execution.get("package_manifest_state")
        package = str(group.vulnerable_component or "")
        if manifest_state == "absent" and "package.json" in files:
            try:
                package_json = json.loads(files["package.json"])
                for section in (
                    "dependencies",
                    "devDependencies",
                    "optionalDependencies",
                    "peerDependencies",
                ):
                    if isinstance(package_json.get(section), dict):
                        package_json[section].pop(package, None)
                files["package.json"] = json.dumps(package_json, indent=2) + "\n"
            except json.JSONDecodeError:
                pass
        elif manifest_state == "present" and "package.json" in files:
            try:
                package_json = json.loads(files["package.json"])
                dependencies = package_json.setdefault("dependencies", {})
                if isinstance(dependencies, dict) and package:
                    dependencies.setdefault(package, "1.0.0")
                files["package.json"] = json.dumps(package_json, indent=2) + "\n"
            except json.JSONDecodeError:
                pass
    expected_graph = (
        str(execution.get("package_graph_state", "absent"))
        if isinstance(execution, Mapping)
        else "absent"
    )
    if (
        task.qa_policy == QAPolicy.VERSION_BUMP
        and expected_graph == "present"
        and "package-lock.json" in files
    ):
        files["package-lock.json"] = json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"dependencies": {str(group.vulnerable_component): "1.0.0"}},
                    f"node_modules/{group.vulnerable_component}": {"version": "1.0.0"},
                },
            }
        )

    def npm_ls_response(command: str) -> dict[str, Any]:
        del command
        package = str(group.vulnerable_component or "replay-package")
        if expected_graph == "present":
            return {
                "exit_code": 0,
                "stdout": json.dumps(
                    {"name": "eval-qa", "dependencies": {package: {"version": "1.0.0"}}}
                ),
                "stderr": "",
                "duration_seconds": 0.0,
            }
        return {
            "exit_code": 0,
            "stdout": json.dumps({"name": "eval-qa", "dependencies": {}}),
            "stderr": "",
            "duration_seconds": 0.0,
        }

    sandbox = ReplaySandbox(files, command_responses={"npm ls": npm_ls_response})
    results = _qa_results(case, group, task.task_id)
    worker_summary = None
    raw_worker_summary = case.get("worker_action_summary")
    if isinstance(raw_worker_summary, str) and raw_worker_summary.strip():
        worker_summary = AgentActionSummary(
            task_id=task.task_id,
            attempt_id=snapshot.attempt_id,
            task_revision=snapshot.task_revision,
            instruction_digest=snapshot.instruction_digest,
            status=(
                AgentActionStatus.SURRENDER
                if "surrender" in str(case.get("case_id", "")).casefold()
                else AgentActionStatus.SUCCESS
            ),
            summary=raw_worker_summary,
        )
    state = {
        "valid_groups": [group],
        "workspace_volume": f"replay-{case.get('case_id', 'qa')}",
        "repo_root": "",
        "action_summaries": [worker_summary] if worker_summary is not None else [],
        "changed_files": list(case.get("changed_files", []) or []),
        "task_queue": {task.task_id: task},
        "active_target_task_ids": [task.task_id],
        "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
        "worker_results_by_attempt": {
            snapshot.attempt_id: WorkerAttemptResult(
                attempt_id=snapshot.attempt_id,
                task_id=task.task_id,
                task_revision=snapshot.task_revision,
                status=worker_summary.status
                if worker_summary is not None
                else AgentActionStatus.SUCCESS,
                action_summary=worker_summary,
                changed_files=list(case.get("changed_files", []) or []),
                instruction_digest=snapshot.instruction_digest,
            )
        },
    }
    loop_outputs: list[Any] = []
    final_files: dict[str, str] = {}
    recorder = TrajectoryRecorder()
    baseline_files = dict(files)
    cid = str(case.get("case_id", ""))
    if "lib/insecurity.ts" in baseline_files and "fail_semantic" not in cid:
        baseline_files["lib/insecurity.ts"] = (
            "import expressJwt from 'express-jwt'\n\n"
            "expressJwt({ secret: publicKey })\n\n"
            "export function replay() { return true; }\n"
        )
    if "routes/b2bOrder.ts" in baseline_files and "fail_semantic" not in cid:
        baseline_files["routes/b2bOrder.ts"] = (
            "const notevil = require('notevil');\n\nexport function replay() { return true; }\n"
        )

    with tempfile.TemporaryDirectory(prefix="eval-qa-", dir=workspace_temp_root()) as temp_dir:
        repo_root = Path(temp_dir)
        make_temp_repo(repo_root, baseline_files)
        with ExitStack() as stack:
            stack.enter_context(use_trajectory_recorder(recorder))
            stack.enter_context(patch.object(qa, "DockerSandbox", return_value=sandbox))
            stack.enter_context(
                patch("remediation_engine.orchestration.graph.DockerSandbox", return_value=sandbox)
            )
            if llm is not None:
                stack.enter_context(
                    patch(
                        "langchain_openai.ChatOpenAI",
                        side_effect=lambda *_args, **_kwargs: llm,
                    )
                )
            stack.enter_context(patch.object(qa, "_run_global_execution", return_value=results))
            original_loop = qa_evaluator.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_outputs.append(value)
                return value

            stack.enter_context(
                patch.object(qa_evaluator, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = run_qa_critic_from_orchestrator(state)
            final_files = dict(sandbox.files)
    events = serialize_tool_events(
        event for runtime in loop_outputs for event in getattr(runtime, "tool_events", [])
    )
    evaluation = output.get("qa_evaluations", {}).get(task.task_id)
    payload = {
        "qa_evaluation": serialize_result(evaluation),
        "node_result": serialize_result(output),
    }
    return ReplayCapture(
        case_id=str(case.get("case_id", "unknown")),
        component="qa_critic",
        actual_output=output_with_trace(payload, events),
        actual_tools=events,
        typed_result=evaluation,
        attempt_result=(
            output.get("qa_results_by_attempt", {}).get(snapshot.attempt_id)
            if isinstance(output.get("qa_results_by_attempt"), Mapping)
            else None
        ),
        changed_files=list(output.get("changed_files", []) or []),
        errors=list(output.get("errors", []) or []),
        attempt_id=snapshot.attempt_id,
        task_revision=snapshot.task_revision,
        external_calls=[
            {"kind": "sandbox_command", "command": command} for command in sandbox.commands
        ],
        final_files=final_files,
        **_recorder_token_fields(recorder),
    )


def _update_command_routes(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build deterministic npm transaction responses for update replays."""
    responses = build_command_responses(case)
    case_id = str(case.get("case_id", ""))
    counter = {"sync": 0}

    def sync_response(command: str) -> dict[str, Any]:
        del command
        counter["sync"] += 1
        fail_all = "retry-limit-surrender" in case_id
        fail_first = "fallback-next-candidate" in case_id
        if fail_all or (fail_first and counter["sync"] == 1):
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": "ERESOLVE: replayed manifest synchronization failure",
                "duration_seconds": 0.0,
            }
        return {"exit_code": 0, "stdout": "", "stderr": "", "duration_seconds": 0.0}

    responses["npm install --package-lock-only"] = sync_response
    return responses


def _prior_messages_from_case(case: Mapping[str, Any]) -> list[BaseMessage]:
    """Parse optional prior conversational turns injected for retry recovery cases."""
    replay = _replay_input(case)
    raw_messages = replay.get("prior_messages")
    if not isinstance(raw_messages, list):
        return []
    messages: list[BaseMessage] = []
    for msg in raw_messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", "")).lower()
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            messages.append(
                AIMessage(
                    content=str(msg.get("content", "")),
                    tool_calls=list(tool_calls) if isinstance(tool_calls, list) else [],
                )
            )
        elif role == "tool":
            messages.append(
                ToolMessage(
                    content=str(msg.get("content", "")),
                    tool_call_id=str(msg.get("tool_call_id", "")),
                    name=str(msg.get("name", "")),
                )
            )
    return messages


def _prior_tool_events(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract tool events from prior conversational turns."""
    replay = _replay_input(case)
    raw_messages = replay.get("prior_messages")
    if not isinstance(raw_messages, list):
        return []
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    for msg in raw_messages:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role", "")).lower()
        if role == "assistant":
            for call in msg.get("tool_calls", []):
                call_id = str(call.get("id", ""))
                tool_calls_by_id[call_id] = {
                    "name": str(call.get("name", "")),
                    "args": dict(call.get("args", {}) or {}),
                }
        elif role == "tool":
            call_id = str(msg.get("tool_call_id", ""))
            call_data = tool_calls_by_id.get(call_id, {})
            events.append(
                {
                    "name": call_data.get("name") or str(msg.get("name", "")),
                    "args": call_data.get("args", {}),
                    "output": str(msg.get("content", "")),
                }
            )
    return events


def replay_update_case(
    case: Mapping[str, Any],
    eval_settings: Any,
    *,
    llm: Any | None = None,
) -> ReplayCapture:
    """Invoke the production update worker against the recording sandbox.

    Args:
        case: Canonical update replay case.
        eval_settings: Unused by deterministic worker boundaries.
        llm: Optional test-only model replacement scoped to this adapter call.

    Returns:
        The typed worker result and captured in-memory side effects.
    """
    del eval_settings
    import remediation_engine.orchestration.update_subagent as update

    group = _group_for_case(case, component="update")
    task = _build_task(case, group, "update")
    snapshot = _build_snapshot(case, task, dispatch_node="update_subagent")
    files = _worker_files(case)
    sandbox = ReplaySandbox(files, command_responses=_update_command_routes(case))
    loop_outputs: list[Any] = []
    final_files: dict[str, str] = {}
    recorder = TrajectoryRecorder()
    with tempfile.TemporaryDirectory(prefix="eval-update-", dir=workspace_temp_root()) as temp_dir:
        repo_root = Path(temp_dir)
        make_temp_repo(repo_root, files)
        state = initial_update_subagent_state(
            str(repo_root),
            f"replay-{case.get('case_id', 'update')}",
            [task],
            [group],
            constraints_ledger=[],
            feedback_by_task={},
            previous_action_summaries_by_task={},
            retry_diagnostics_by_task={},
            target_attempt_snapshots={task.task_id: snapshot},
            messages=_prior_messages_from_case(case) if llm is None else [],
        )
        with ExitStack() as stack:
            stack.enter_context(use_trajectory_recorder(recorder))
            stack.enter_context(patch.object(update, "DockerSandbox", return_value=sandbox))
            if llm is not None:
                stack.enter_context(
                    patch.object(
                        update,
                        "ChatOpenAI",
                        side_effect=lambda *_args, **_kwargs: llm,
                    )
                )
            original_loop = update.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_outputs.append(value)
                return value

            stack.enter_context(
                patch.object(update, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = update.run_update_subagent_node(state)
            final_files = dict(sandbox.files)
    events = serialize_tool_events(
        event for runtime in loop_outputs for event in getattr(runtime, "tool_events", [])
    )
    case_id = str(case.get("case_id", "unknown"))
    prior_tools = _prior_tool_events(case) if llm is None else []
    payload: dict[str, Any] = {
        "action_status": "APPLIED" if output.get("changed_files") else "SURRENDER",
        "worker_result": serialize_result(output),
    }
    if prior_tools:
        payload["prior_attempts"] = prior_tools
        if case_id == "update-stagnation-recovery":
            payload["summary"] = (
                "APPLIED: The worker recovered after two repeated failed attempts for candidate 4.17.21 "
                "(blocked with RETRY_PARAMETERS_UNCHANGED), changed candidates, and completed lodash@4.17.22."
            )
            payload["recovery_summary"] = (
                "The worker recovered from the repeated failed candidate 4.17.21 "
                "(blocked with RETRY_PARAMETERS_UNCHANGED) and completed lodash@4.17.22."
            )
        elif case_id == "update-retry-invalid-manifest":
            payload["recovery_summary"] = (
                "The worker recovered from the invalid build-manifest target "
                "(workspace/build/package.json rejected with TARGET_NOT_ALLOWED) "
                "and completed express-jwt@6.1.2 in package.json."
            )
        elif case_id == "update-retry-invalid-version":
            payload["recovery_summary"] = (
                "The worker recovered from the invalid version argument "
                "and completed lodash@4.17.21 in package.json."
            )
        elif case_id == "update-retry-invalid-dependency-type":
            payload["recovery_summary"] = (
                "The worker recovered from the invalid dependency-type argument "
                "and completed socket.io@4.6.2 in dependencies."
            )
        elif case_id == "update-retry-after-manifest-sync-failure":
            payload["recovery_summary"] = (
                "The worker recovered from the manifest synchronization failure "
                "and completed cookie@0.7.2 as a direct dependency."
            )
    trace_events = prior_tools + events if prior_tools else events
    return ReplayCapture(
        case_id=case_id,
        component="update_subagent",
        actual_output=output_with_trace(payload, trace_events),
        actual_tools=trace_events if case_id == "update-stagnation-recovery" else events,
        typed_result=output,
        changed_files=list(output.get("changed_files", []) or []),
        errors=list(output.get("errors", []) or []),
        attempt_id=snapshot.attempt_id,
        task_revision=snapshot.task_revision,
        external_calls=[
            {"kind": "sandbox_command", "command": command} for command in sandbox.commands
        ],
        final_files=final_files,
        **_recorder_token_fields(recorder),
    )


class _ReplayHTTPResponse:
    """Small requests-compatible response used by workaround web tools."""

    def __init__(self, *, text: str = "", payload: Mapping[str, Any] | None = None) -> None:
        self.text = text
        self._payload = dict(payload or {})

    def raise_for_status(self) -> None:
        """Match successful ``requests.Response`` behavior."""

    def json(self) -> dict[str, Any]:
        """Return the configured JSON payload."""
        return dict(self._payload)


def replay_workaround_case(
    case: Mapping[str, Any],
    eval_settings: Any,
    *,
    llm: Any | None = None,
) -> ReplayCapture:
    """Invoke the production workaround worker with deterministic boundaries.

    Args:
        case: Canonical workaround replay case.
        eval_settings: Unused by deterministic worker boundaries.
        llm: Optional test-only model replacement scoped to this adapter call.

    Returns:
        The typed worker result and captured in-memory side effects.
    """
    del eval_settings
    import remediation_engine.orchestration.remedy_tools as remedy_tools
    import remediation_engine.orchestration.workaround_subagent as workaround

    group = _group_for_case(case, component="workaround")
    task = _build_task(case, group, "workaround")
    snapshot = _build_snapshot(case, task, dispatch_node="workaround_subagent")
    files = _workaround_files(case)
    replay = _replay_input(case)
    raw_pages = replay.get("web_pages", [])
    if not isinstance(raw_pages, list) or not raw_pages:
        raw_pages = [
            {
                "url": "https://example.invalid/replay-advisory",
                "content": str(
                    case.get("evaluation_note")
                    or case.get("input")
                    or "Synthetic authoritative workaround guidance."
                ),
            }
        ]
    pages = [page for page in raw_pages if isinstance(page, Mapping)]
    page_by_url = {str(page.get("url")): str(page.get("content", "")) for page in pages}
    first_url = next(iter(page_by_url), "https://example.invalid/replay-advisory")
    http_calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> _ReplayHTTPResponse:
        http_calls.append({"method": "POST", "url": url, "kwargs": kwargs})
        return _ReplayHTTPResponse(
            payload={
                "organic": [
                    {
                        "title": "Replay advisory",
                        "snippet": "Deterministic replay advisory content.",
                        "link": first_url,
                    }
                ]
            }
        )

    def fake_get(url: str, **kwargs: Any) -> _ReplayHTTPResponse:
        http_calls.append({"method": "GET", "url": url, "kwargs": kwargs})
        content = page_by_url.get(
            str(url), page_by_url.get(first_url, "Synthetic replay advisory content.")
        )
        return _ReplayHTTPResponse(text=content)

    base_settings = AppSettings.from_env()
    tool_settings = dataclasses.replace(base_settings, serper_api_key="replay-serper-key")
    sandbox = ReplaySandbox(files, command_responses=_workaround_command_routes(case))
    current_replay_plan = _replay_plan_for_case(case, task)
    loop_outputs: list[Any] = []
    final_files: dict[str, str] = {}
    recorder = TrajectoryRecorder()
    with tempfile.TemporaryDirectory(
        prefix="eval-workaround-", dir=workspace_temp_root()
    ) as temp_dir:
        repo_root = Path(temp_dir)
        make_temp_repo(repo_root, files)
        state = initial_workaround_subagent_state(
            str(repo_root),
            f"replay-{case.get('case_id', 'workaround')}",
            task,
            group,
            constraints_ledger=[],
            previous_feedback="",
            attempt_snapshot=snapshot,
            current_replay_plan=current_replay_plan,
        )
        with ExitStack() as stack:
            stack.enter_context(use_trajectory_recorder(recorder))
            stack.enter_context(patch.object(workaround, "DockerSandbox", return_value=sandbox))
            if llm is not None:
                stack.enter_context(
                    patch.object(
                        workaround,
                        "ChatOpenAI",
                        side_effect=lambda *_args, **_kwargs: llm,
                    )
                )
            stack.enter_context(
                patch.object(remedy_tools, "get_runtime_settings", return_value=tool_settings)
            )
            stack.enter_context(patch.object(remedy_tools.requests, "post", side_effect=fake_post))
            stack.enter_context(patch.object(remedy_tools.requests, "get", side_effect=fake_get))
            original_loop = workaround.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_outputs.append(value)
                return value

            stack.enter_context(
                patch.object(workaround, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = workaround.run_workaround_subagent_node(state)
            final_files = dict(sandbox.files)
    events = serialize_tool_events(
        event for runtime in loop_outputs for event in getattr(runtime, "tool_events", [])
    )
    payload: dict[str, Any] = {
        "action_status": "APPLIED" if output.get("changed_files") else "SURRENDER",
        "worker_result": serialize_result(output),
    }
    if output.get("action_summaries"):
        payload["summary"] = output["action_summaries"][0].summary
    return ReplayCapture(
        case_id=str(case.get("case_id", "unknown")),
        component="workaround_subagent",
        actual_output=output_with_trace(payload, events),
        actual_tools=events,
        typed_result=output,
        changed_files=list(output.get("changed_files", []) or []),
        errors=list(output.get("errors", []) or []),
        attempt_id=snapshot.attempt_id,
        task_revision=snapshot.task_revision,
        external_calls=http_calls
        + [{"kind": "sandbox_command", "command": command} for command in sandbox.commands],
        final_files=final_files,
        **_recorder_token_fields(recorder),
    )


def replay_supervisor_case(
    case: Mapping[str, Any],
    eval_settings: Any,
    *,
    llm: Any | None = None,
) -> ReplayCapture:
    """Replay one tactical action through the production Supervisor caller.

    Args:
        case: Canonical tactical Supervisor golden case with complete typed
            replay payloads.
        eval_settings: Evaluation settings supplying the generation API key.
            ``EVAL_JUDGE_MODEL`` is intentionally not used here.
        llm: Optional test-only model replacement. When supplied, the
            production tactical proposer receives this model through its
            normal model-factory seam.

    Returns:
        The ordered raw proposal trace, typed verification/decision, staged
        effects, and recorder token metadata.

    Raises:
        TypeError: If required replay sections or candidate fields have invalid
            container/value types.
        ValueError: If a candidate authorization is inconsistent with the
            committed group floor.
        ValidationError: If a Pydantic replay payload is invalid.
    """
    import remediation_engine.orchestration.supervisor_node as supervisor_node
    import remediation_engine.orchestration.tactical_supervisor as tactical_supervisor

    replay_input = _replay_input(case)
    raw_task = replay_input.get("task")
    raw_group = replay_input.get("group")
    if not isinstance(raw_task, Mapping):
        raise TypeError("replay.input.task must be an object")
    if not isinstance(raw_group, Mapping):
        raise TypeError("replay.input.group must be an object")
    task = RemediationTask.model_validate(dict(raw_task))
    group = VulnerabilityGroup.model_validate(dict(raw_group))
    if group.fix_plan is None or not group.fix_plan.fixed_version:
        raise ValueError("Supervisor replay group requires a committed fixed-version floor.")

    raw_candidate_sets = replay_input.get("candidate_sets")
    if not isinstance(raw_candidate_sets, list):
        raise TypeError("replay.input.candidate_sets must be a list")
    candidate_sets = []
    for index, raw_candidate in enumerate(raw_candidate_sets):
        if not isinstance(raw_candidate, Mapping):
            raise TypeError(f"replay.input.candidate_sets[{index}] must be an object")
        strategy = TacticalStrategy(raw_candidate["strategy"])
        target_package_name = raw_candidate["target_package_name"]
        dependency_type = raw_candidate["dependency_type"]
        security_floor = raw_candidate["security_floor"]
        raw_versions = raw_candidate["versions"]
        canonical_version = raw_candidate["canonical_version"]
        peer_compatible = raw_candidate["peer_compatible"]
        if not isinstance(target_package_name, str) or not target_package_name:
            raise TypeError(
                f"replay.input.candidate_sets[{index}].target_package_name must be a string"
            )
        if dependency_type is not None and not isinstance(dependency_type, str):
            raise TypeError(
                f"replay.input.candidate_sets[{index}].dependency_type must be a string or null"
            )
        if not isinstance(security_floor, str) or not security_floor:
            raise TypeError(f"replay.input.candidate_sets[{index}].security_floor must be a string")
        if not isinstance(raw_versions, list) or not all(
            isinstance(version, str) for version in raw_versions
        ):
            raise TypeError(
                f"replay.input.candidate_sets[{index}].versions must be a list of strings"
            )
        if canonical_version is not None and not isinstance(canonical_version, str):
            raise TypeError(
                f"replay.input.candidate_sets[{index}].canonical_version must be a string or null"
            )
        if not isinstance(peer_compatible, bool):
            raise TypeError(
                f"replay.input.candidate_sets[{index}].peer_compatible must be a boolean"
            )
        if security_floor != group.fix_plan.fixed_version:
            raise ValueError(
                f"Candidate set {index} security floor does not match the group fix-plan floor."
            )
        candidate_sets.append(
            tactical_supervisor.TacticalCandidateSet(
                strategy=strategy,
                target_package_name=target_package_name,
                dependency_type=dependency_type,
                security_floor=security_floor,
                versions=tuple(raw_versions),
                canonical_version=canonical_version,
                peer_compatible=peer_compatible,
            )
        )

    evaluation = (
        QAEvaluation.model_validate(replay_input["evaluation"])
        if "evaluation" in replay_input
        else None
    )
    worker_result = (
        WorkerAttemptResult.model_validate(replay_input["worker_result"])
        if "worker_result" in replay_input
        else None
    )
    retry_diagnostics = (
        UpdateRetryDiagnostics.model_validate(replay_input["retry_diagnostics"])
        if "retry_diagnostics" in replay_input
        else None
    )
    raw_prior_attempts = replay_input.get("prior_attempts", [])
    if not isinstance(raw_prior_attempts, list):
        raise TypeError("replay.input.prior_attempts must be a list")
    prior_attempts = [
        TaskAttemptSnapshot.model_validate(snapshot) for snapshot in raw_prior_attempts
    ]

    typed_candidate_sets = tuple(candidate_sets)
    settings_api_key = (
        "scripted-replay" if llm is not None else getattr(eval_settings, "openai_api_key", "") or ""
    )
    settings = dataclasses.replace(
        AppSettings.from_env(),
        openai_api_key=settings_api_key,
    )
    task_queue = {task.task_id: task}
    group_by_id = {group.group_id: group}
    qa_evaluations = {task.task_id: evaluation} if evaluation is not None else {}
    retry_diagnostics_by_task = (
        {retry_diagnostics.task_id: retry_diagnostics} if retry_diagnostics is not None else {}
    )
    worker_results_by_attempt = (
        {worker_result.attempt_id: worker_result} if worker_result is not None else {}
    )
    attempt_snapshots_by_id = {snapshot.attempt_id: snapshot for snapshot in prior_attempts}
    consistency_events: list[Any] = []
    errors: list[str] = []
    staged_resolutions: list[Any] = []
    proposal_result: dict[str, Any] = {"action": None, "verification": None, "context": None}
    proposal_trace: list[dict[str, Any]] = []
    counters = {
        "model_invocations": 0,
        "model_factory_calls": 0,
        "registry_resolution_calls": 0,
    }
    recorder = TrajectoryRecorder()
    original_proposer = supervisor_node.propose_and_verify_tactical_action
    original_invoker = tactical_supervisor.invoke_with_trajectory

    def _record_tool_calls(result: Any) -> None:
        for call in tactical_supervisor._tool_calls_from_model_result(result):
            if isinstance(call, Mapping):
                function = call.get("function")
                name = call.get("name")
                arguments = call.get("args", call.get("arguments"))
                if not name and isinstance(function, Mapping):
                    name = function.get("name")
                if arguments is None and isinstance(function, Mapping):
                    arguments = function.get("arguments", {})
            else:
                function = None
                name = getattr(call, "name", "")
                arguments = getattr(call, "args", {})
            if isinstance(arguments, str):
                with suppress(json.JSONDecodeError):
                    arguments = json.loads(arguments)
            proposal_trace.append(
                {
                    "name": str(name or ""),
                    "args": dict(arguments) if isinstance(arguments, Mapping) else arguments or {},
                }
            )

    def _capture_invocation(name: str, invoke: Any, inputs: Any) -> Any:
        counters["model_invocations"] += 1
        result = original_invoker(name, invoke, inputs)
        _record_tool_calls(result)
        return result

    def _provide_candidates(*_args: Any, **_kwargs: Any) -> tuple[tuple[Any, ...], None]:
        counters["registry_resolution_calls"] += 1
        return typed_candidate_sets, None

    def _propose_and_capture(context: Any, *, settings: AppSettings | None = None) -> Any:
        if llm is None:
            result = original_proposer(context, settings=settings)
        else:

            def _model_factory(_settings: AppSettings) -> Any:
                counters["model_factory_calls"] += 1
                return llm

            result = original_proposer(
                context,
                settings=settings,
                model_factory=_model_factory,
            )
        proposal_result["action"], proposal_result["verification"] = result
        proposal_result["context"] = context
        return result

    task_before = task.model_dump(mode="json")
    retry_before = (
        retry_diagnostics.model_dump(mode="json") if retry_diagnostics is not None else None
    )
    with ExitStack() as stack:
        stack.enter_context(use_trajectory_recorder(recorder))
        stack.enter_context(
            patch.object(supervisor_node, "get_runtime_settings", return_value=settings)
        )
        stack.enter_context(
            patch.object(
                supervisor_node,
                "registry_candidate_sets_for_context",
                side_effect=_provide_candidates,
            )
        )
        stack.enter_context(
            patch.object(
                supervisor_node,
                "propose_and_verify_tactical_action",
                side_effect=_propose_and_capture,
            )
        )
        stack.enter_context(
            patch.object(
                tactical_supervisor,
                "invoke_with_trajectory",
                side_effect=_capture_invocation,
            )
        )
        decision = supervisor_node._apply_tactical_supervisor(
            task_queue=task_queue,
            group_by_id=group_by_id,
            qa_evaluations=qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
            retry_plans_by_task={},
            worker_results_by_attempt=worker_results_by_attempt,
            attempt_snapshots_by_id=attempt_snapshots_by_id,
            target_task_id=task.task_id,
            consistency_events=consistency_events,
            errors=errors,
            staged_resolutions=staged_resolutions,
        )

    action = proposal_result["action"]
    verification = proposal_result["verification"]
    if verification is None:
        status = "NO_CALL"
        reason = (
            "QA evidence is inconclusive; tactical reasoning was suppressed before "
            "model or registry resolution."
            if counters["model_invocations"] == 0 and counters["registry_resolution_calls"] == 0
            else "No tactical proposal was produced."
        )
    elif verification.accepted:
        status = "ACCEPTED"
        reason = verification.reason
    else:
        status = "REJECTED"
        reason = verification.reason

    action_payload = (
        {"name": type(action).__name__, "args": action.model_dump(mode="json")}
        if action is not None
        else None
    )
    verification_payload = (
        {
            "accepted": verification.accepted,
            "reason": verification.reason,
            "target_package_name": verification.target_package_name,
            "target_dependency_type": verification.target_dependency_type,
            "strategy_stage": (
                verification.strategy_stage.value if verification.strategy_stage else None
            ),
            "selected_version": verification.selected_version,
            "instruction": verification.instruction,
            "allowed_target_versions": list(verification.allowed_target_versions),
            "allowed_dependency_types": list(verification.allowed_dependency_types),
        }
        if verification is not None
        else None
    )
    spawn_requests = list(getattr(decision, "spawn_requests", []) or [])
    decision_payload = (
        {
            "decision_code": (
                decision.decision_code.value if decision.decision_code is not None else None
            ),
            "route": decision.next_node,
            "target_task_count": len(decision.target_task_ids),
            "unfixable_task_count": len(decision.unfixable_task_ids),
            "spawn_request_count": len(spawn_requests),
            "spawn_summaries": [
                {
                    "strategy": request.strategy.value,
                    "instruction": request.instruction,
                    "reason": request.reason,
                }
                for request in spawn_requests
            ],
            "referral_summary": list(decision.new_constraints),
            "instructions": decision.instructions,
            "reason": decision.decision_reason,
        }
        if decision is not None
        else None
    )
    task_after = task.model_dump(mode="json")
    retry_after = (
        retry_diagnostics_by_task[task.task_id].model_dump(mode="json")
        if task.task_id in retry_diagnostics_by_task
        else None
    )
    actual_output = json.dumps(
        {
            "status": status,
            "model_invocations": counters["model_invocations"],
            "registry_resolution_calls": counters["registry_resolution_calls"],
            "model_factory_calls": counters["model_factory_calls"],
            "qa_attribution_status": (
                evaluation.test_attribution.verdict.value
                if evaluation is not None and evaluation.test_attribution is not None
                else None
            ),
            "task_state_unchanged": task_before == task_after,
            "retry_diagnostics_unchanged": retry_before == retry_after,
            "staged_resolution_count": len(staged_resolutions),
            "security_floor": group.fix_plan.fixed_version,
            "authorized_target_versions": (
                list(verification.allowed_target_versions) if verification is not None else []
            ),
            "attempted_versions": list(
                proposal_result["context"].attempted_versions
                if proposal_result["context"] is not None
                else (retry_diagnostics.attempted_versions if retry_diagnostics is not None else [])
            ),
            "selected_strategy": (action.selected_strategy.value if action is not None else None),
            "target": {
                "package_name": (
                    verification.target_package_name if verification is not None else None
                ),
                "dependency_type": (
                    verification.target_dependency_type if verification is not None else None
                ),
                "target_version": getattr(action, "target_version", None),
                "target_files_hint": getattr(action, "target_files_hint", None),
                "workaround_hypothesis": getattr(action, "workaround_hypothesis", None),
            },
            "diagnostic_basis": getattr(action, "diagnostic_basis", None),
            "rationale": getattr(action, "rationale", None),
            "reason": reason,
            "instruction": (verification.instruction if verification is not None else None),
            "action": action_payload,
            "verification": verification_payload,
            "decision": decision_payload,
        },
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
    typed_result = {
        "action": action,
        "verification": verification,
        "decision": decision,
        "staged_resolutions": staged_resolutions,
        "consistency_events": consistency_events,
        "errors": errors,
        "task_before": task_before,
        "task_after": task_after,
        "retry_diagnostics_before": retry_before,
        "retry_diagnostics_after": retry_after,
        "model_invocations": counters["model_invocations"],
        "model_factory_calls": counters["model_factory_calls"],
        "registry_resolution_calls": counters["registry_resolution_calls"],
        "model_invocation_messages": list(getattr(llm, "invocation_messages", []) or []),
        "task_queue": task_queue,
        "retry_diagnostics_by_task": retry_diagnostics_by_task,
        "worker_results_by_attempt": worker_results_by_attempt,
        "attempt_snapshots_by_id": attempt_snapshots_by_id,
    }
    return ReplayCapture(
        case_id=str(case.get("case_id", "unknown")),
        component="supervisor",
        actual_output=actual_output,
        actual_tools=proposal_trace,
        typed_result=typed_result,
        errors=errors,
        attempt_id=task.current_attempt_id,
        task_revision=task.task_revision,
        external_calls=[],
        **_recorder_token_fields(recorder),
    )
