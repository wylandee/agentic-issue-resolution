"""Production-node replay adapters for the Phase 2 evaluation suites."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import tempfile
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    LocalizedIssue,
    NoFixMitigationStage,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    SCARemediationStage,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundReplayPlan,
)
from remediation_engine.orchestration.state import (
    initial_update_subagent_state,
    initial_workaround_subagent_state,
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
                    "failures": [],
                }
            ),
            "stderr": "",
            "duration_seconds": 0.0,
        }

    responses.setdefault("npx --no-install mocha", mocha_response)
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
            del command
            attempts["count"] += 1
            if attempts["count"] == 1:
                return {
                    "exit_code": 1,
                    "stdout": "Error: Cannot find module 'better-sqlite3'",
                    "stderr": "",
                    "duration_seconds": 0.0,
                }
            return mocha_response("npx --no-install mocha")

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


def replay_triage_case(case: Mapping[str, Any], eval_settings: Any) -> ReplayCapture:
    """Invoke production triage and capture its typed post-guardrail result."""
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
    )


def _qa_results(case: Mapping[str, Any], group: VulnerabilityGroup) -> Any:
    """Build deterministic QA execution results from the golden evidence."""
    import remediation_engine.orchestration.qa_critic as qa

    replay = _replay_input(case)
    execution = replay.get("execution_context", case.get("execution_context", {}))
    execution = execution if isinstance(execution, Mapping) else {}
    logs = replay.get("execution_logs", case.get("execution_logs", {}))
    logs = logs if isinstance(logs, Mapping) else {}
    results = qa._QAExecutionResults()
    results.install = (
        bool(execution.get("install_passed")),
        str(logs.get("install_log", "replay install result")),
    )
    policy = _policy_for_case(case, "qa")
    if policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL:
        results.scan_skipped = True
        results.scan_skip_reason = "no_fix_package_removal"
    else:
        raw_remaining = execution.get("target_remaining_identifiers", []) or []
        remaining = {str(value) for value in raw_remaining}
        target_cleared = execution.get("target_scanner_cleared")
        scan_ok = bool(execution.get("scanner_execution_status") == "success") and bool(
            target_cleared is True
        )
        status_name = str(execution.get("scanner_execution_status", "not_run")).lower()
        scan_status = (
            ScannerExecutionStatus.SUCCESS
            if status_name == "success"
            else ScannerExecutionStatus.UNPARSEABLE
            if status_name in {"failed", "failure", "error"}
            else ScannerExecutionStatus.NOT_RUN
        )
        results.scan = qa._SecurityScanResult(
            ok=scan_ok,
            summary=str(logs.get("scan_summary", "replay scanner result")),
            remaining_identifiers=remaining,
            found_identifiers={
                str(value) for value in execution.get("found_identifiers", []) or []
            },
            new_identifiers={str(value) for value in execution.get("new_identifiers", []) or []},
            execution_status=scan_status,
        )
        try:
            requested_scope = ScanScope(str(execution.get("requested_scope", "full")))
            effective_scope = ScanScope(
                str(execution.get("effective_scope", requested_scope.value))
            )
            fallback = execution.get("fallback_reason")
            results.scan_evidence = qa.ODCScanEvidence(
                requested_scope=requested_scope,
                effective_scope=effective_scope,
                authoritative=False,
                covered_task_ids=[str(case.get("case_id"))],
                closure_package_names=[str(group.vulnerable_component or "")],
                closure_lockfile_keys=[],
                found_identifiers=sorted(
                    {str(value) for value in execution.get("found_identifiers", []) or []}
                ),
                remaining_target_identifiers=sorted(remaining),
                complete=bool(execution.get("scan_complete", True)),
                fallback_reason=ScanFallbackReason(str(fallback)) if fallback else None,
            )
        except (TypeError, ValueError):
            results.scan_evidence = None
    results.tests = (
        bool(execution.get("tests_passed")),
        str(logs.get("test_output", "replay test result")),
    )
    return results


def replay_qa_case(case: Mapping[str, Any], eval_settings: Any) -> ReplayCapture:
    """Invoke the production QA node with deterministic execution evidence."""
    del eval_settings
    import remediation_engine.orchestration.qa_critic as qa

    group = _group_for_case(case, component="qa")
    task = _build_task(case, group, "qa")
    snapshot = _build_snapshot(case, task, dispatch_node="qa_critic")
    files = seed_replay_files(case)
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
    expected_graph = (
        str(execution.get("package_graph_state", "absent"))
        if isinstance(execution, Mapping)
        else "absent"
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
    results = _qa_results(case, group)
    state = {
        "valid_groups": [group],
        "workspace_volume": f"replay-{case.get('case_id', 'qa')}",
        "repo_root": "",
        "action_summaries": [],
        "group_strategies": {
            group.group_id: (
                "no_fix_package_removal"
                if task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL
                else "code_workaround"
                if task.strategy == RoutingStrategy.CODE_WORKAROUND
                else "version_bump"
            )
        },
        "changed_files": list(case.get("changed_files", []) or []),
        "task_queue": {task.task_id: task},
        "active_target_task_ids": [task.task_id],
        "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
    }
    loop_results: list[Any] = []
    with tempfile.TemporaryDirectory(prefix="eval-qa-", dir=workspace_temp_root()) as temp_dir:
        repo_root = Path(temp_dir)
        make_temp_repo(repo_root, files)
        state["repo_root"] = str(repo_root)
        with ExitStack() as stack:
            stack.enter_context(patch.object(qa, "DockerSandbox", return_value=sandbox))
            stack.enter_context(patch.object(qa, "_run_global_execution", return_value=results))
            original_loop = qa.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_results.append(value)
                return value

            stack.enter_context(
                patch.object(qa, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = qa.run_qa_critic_node(state)
    events = serialize_tool_events(
        event for runtime in loop_results for event in getattr(runtime, "tool_events", [])
    )
    evaluation = output.get("qa_evaluations", {}).get(group.group_id)
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
        changed_files=list(output.get("changed_files", []) or []),
        errors=list(output.get("errors", []) or []),
        attempt_id=snapshot.attempt_id,
        task_revision=snapshot.task_revision,
        external_calls=[
            {"kind": "sandbox_command", "command": command} for command in sandbox.commands
        ],
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


def replay_update_case(case: Mapping[str, Any], eval_settings: Any) -> ReplayCapture:
    """Invoke the production update worker against the recording sandbox."""
    del eval_settings
    import remediation_engine.orchestration.update_subagent as update

    group = _group_for_case(case, component="update")
    task = _build_task(case, group, "update")
    snapshot = _build_snapshot(case, task, dispatch_node="update_subagent")
    files = _worker_files(case)
    sandbox = ReplaySandbox(files, command_responses=_update_command_routes(case))
    loop_results: list[Any] = []
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
            messages=_prior_messages_from_case(case),
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(update, "DockerSandbox", return_value=sandbox))
            original_loop = update.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_results.append(value)
                return value

            stack.enter_context(
                patch.object(update, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = update.run_update_subagent_node(state)
    events = serialize_tool_events(
        event for runtime in loop_results for event in getattr(runtime, "tool_events", [])
    )
    case_id = str(case.get("case_id", "unknown"))
    prior_tools = _prior_tool_events(case)
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


def replay_workaround_case(case: Mapping[str, Any], eval_settings: Any) -> ReplayCapture:
    """Invoke the production workaround worker with deterministic boundaries."""
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
    loop_results: list[Any] = []
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
            stack.enter_context(patch.object(workaround, "DockerSandbox", return_value=sandbox))
            stack.enter_context(
                patch.object(remedy_tools, "get_runtime_settings", return_value=tool_settings)
            )
            stack.enter_context(patch.object(remedy_tools.requests, "post", side_effect=fake_post))
            stack.enter_context(patch.object(remedy_tools.requests, "get", side_effect=fake_get))
            original_loop = workaround.run_bounded_subagent_loop

            def wrapped_loop(*args: Any, **kwargs: Any) -> Any:
                value = original_loop(*args, **kwargs)
                loop_results.append(value)
                return value

            stack.enter_context(
                patch.object(workaround, "run_bounded_subagent_loop", side_effect=wrapped_loop)
            )
            output = workaround.run_workaround_subagent_node(state)
    events = serialize_tool_events(
        event for runtime in loop_results for event in getattr(runtime, "tool_events", [])
    )
    payload = {"worker_result": serialize_result(output), "workspace_files": dict(sandbox.files)}
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
    )
