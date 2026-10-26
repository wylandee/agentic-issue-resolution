"""Normalize complete remediation runs and assess deterministic evidence contracts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from remediation_engine.orchestration.task_utils import terminal_outcome_issues
from remediation_engine.orchestration.trajectory_exporter import to_jsonable

from .trajectories import TrajectoryDocument, extract_tool_calls, parse_trajectory_markdown


class ExecutionEvent(BaseModel):
    """One observed execution, retaining its identity and task ancestry."""

    run_id: str
    parent_run_id: str | None = None
    sequence: int
    name: str
    inputs: Any = None
    outputs: Any = None
    error: str | None = None


class RunEvidence(BaseModel):
    """Versioned evidence shared by recorded and fresh whole-run evaluations."""

    schema_version: int = 1
    source_run_id: str
    source: Literal["recorded", "fresh"] = "recorded"
    trajectory_path: str | None = None
    request: dict[str, Any]
    final_state: dict[str, Any]
    events: list[ExecutionEvent] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    host_unchanged: bool | None = None
    volumes_removed: bool | None = None


class ContractAssessment(BaseModel):
    """Separate invalid evidence/contracts from an honestly incomplete remediation."""

    violations: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    remediation_complete: bool = False


def normalize_run(
    doc: TrajectoryDocument, *, source: Literal["recorded", "fresh"] = "recorded"
) -> RunEvidence:
    """Normalize root states and observed executions without trusting report claims.

    Args:
        doc: Parsed complete trajectory, including initial and final state.
        source: Whether this evaluation launched the source run.

    Returns:
        Redacted evidence retaining all terminal state and observed calls.

    Raises:
        ValueError: Required root state, identity, or terminal status is missing.
    """
    if not doc.trace_id or not doc.initial_state or not doc.final_state:
        raise ValueError("A complete trajectory needs identity and both root states")
    if doc.final_state.get("status") not in {
        "completed",
        "completed_with_errors",
        "failed",
        "no_work",
        "error",
    }:
        raise ValueError("Trajectory has no recognized terminal status")
    calls = extract_tool_calls(doc.spans)
    tool_spans = []
    seen = set()
    for span in sorted(doc.spans, key=lambda item: item.sequence):
        if span.is_tool and span.run_id not in seen:
            seen.add(span.run_id)
            tool_spans.append(span)
    return RunEvidence(
        source_run_id=doc.trace_id,
        source=source,
        trajectory_path=str(doc.raw_path) if doc.raw_path else None,
        request=to_jsonable(doc.initial_state),
        final_state=to_jsonable(doc.final_state),
        events=[
            ExecutionEvent(
                run_id=span.run_id,
                parent_run_id=span.parent_run_id,
                sequence=span.sequence,
                name=call.name,
                inputs=to_jsonable(call.input_parameters),
                outputs=to_jsonable(call.output),
                error=span.error,
            )
            for span, call in zip(tool_spans, calls, strict=True)
        ],
        coverage={
            "recorded_spans": len(doc.spans),
            "declared_spans": doc.span_count,
            "executed_tools": len(calls),
            "parser_warnings": doc.warnings,
            "parser_diagnostics": doc.diagnostics,
            "scan_scope": (doc.final_state.get("final_full_scan_result") or {}).get(
                "coverage_scope"
            ),
            "cleanup": "unknown",
        },
    )


def load_run(path: Path) -> RunEvidence:
    """Read one explicitly selected UTF-8 trajectory; never discover a corpus."""
    return normalize_run(parse_trajectory_markdown(path))


def assess_contracts(evidence: RunEvidence) -> ContractAssessment:
    """Check lineage, accepted evidence, patch support, and fresh-run isolation.

    Args:
        evidence: Normalized complete run. Historical omissions remain unknown.

    Returns:
        Contract violations, evidence gaps, unresolved outcomes, and completion.
        Safe partial completion is an outcome rather than a contract violation.
    """
    state = evidence.final_state
    assessment = ContractAssessment()
    queue = state.get("task_queue")
    if not isinstance(queue, dict):
        assessment.violations.append("Final task_queue is missing or malformed")
        queue = {}
    snapshots = state.get("attempt_snapshots_by_id") or {}
    worker_results = state.get("worker_results_by_attempt") or {}
    qa_results = state.get("qa_results_by_attempt") or {}
    if not isinstance(snapshots, Mapping):
        assessment.violations.append("Attempt snapshot map is malformed")
        snapshots = {}
    if not isinstance(worker_results, Mapping):
        assessment.violations.append("Worker result map is malformed")
        worker_results = {}
    if not isinstance(qa_results, Mapping):
        assessment.violations.append("QA result map is malformed")
        qa_results = {}
    for task_id, task in queue.items():
        if not isinstance(task, Mapping):
            assessment.violations.append(f"Task record is malformed: {task_id}")
            continue
        if task.get("task_id") != task_id:
            assessment.violations.append(f"Task identity mismatch: {task_id}")
        current = task.get("current_attempt_id")
        snapshot = snapshots.get(current, {})
        if current and not snapshot:
            assessment.unknowns.append(f"Committed attempt missing: {task_id}/{current}")
        if snapshot and not isinstance(snapshot, Mapping):
            assessment.violations.append(f"Committed attempt record is malformed: {task_id}")
            snapshot = {}
        if snapshot and snapshot.get("task_id") != task_id:
            assessment.violations.append(f"Committed attempt belongs to another task: {task_id}")
        if snapshot and snapshot.get("attempt_id") not in {None, current}:
            assessment.violations.append(f"Committed attempt identity mismatch: {task_id}")
        if snapshot and snapshot.get("task_revision") != task.get("task_revision"):
            assessment.violations.append(f"Committed attempt revision is stale: {task_id}")
        if task.get("parent_task_id") and task["parent_task_id"] not in queue:
            assessment.violations.append(f"Missing parent task: {task_id}")
        if current:
            worker = worker_results.get(current)
            if worker is None:
                if str(task.get("status", "")).lower() in {"optimistically_fixed", "qa_passed"}:
                    assessment.unknowns.append(
                        f"Current-attempt worker evidence missing: {task_id}"
                    )
            elif not isinstance(worker, Mapping):
                assessment.violations.append(f"Current worker result is malformed: {task_id}")
            elif (
                worker.get("task_id") != task_id
                or worker.get("attempt_id") != current
                or (snapshot and worker.get("task_revision") != snapshot.get("task_revision"))
                or any(
                    snapshot.get(field) is not None and worker.get(field) != snapshot.get(field)
                    for field in (
                        "cluster_id",
                        "dispatch_batch_id",
                        "action_digest",
                        "instruction_digest",
                    )
                )
            ):
                assessment.violations.append(f"Stale or unrelated current worker result: {task_id}")
            if isinstance(worker, Mapping) and str(task.get("status", "")).lower() == "qa_passed":
                if str(worker.get("status", "")).lower() != "success":
                    assessment.violations.append(
                        f"Accepted QA task lacks a successful current worker: {task_id}"
                    )
                authorized = set(snapshot.get("allowed_target_versions") or [])
                executed = set(worker.get("executed_versions") or [])
                diagnostics = worker.get("execution_diagnostics") or {}
                effective = diagnostics.get("effective_target_version")
                if effective:
                    executed.add(effective)
                if authorized and executed - authorized:
                    assessment.violations.append(
                        f"Worker executed a version outside the committed candidate set: {task_id}"
                    )
                if (
                    snapshot
                    and snapshot.get("selected_version")
                    and not authorized
                    and executed
                    and snapshot.get("selected_version") not in executed
                ):
                    assessment.unknowns.append(
                        f"Executed version could not be matched to committed selection: {task_id}"
                    )
                if (
                    snapshot
                    and snapshot.get("dispatch_node") == "update_subagent"
                    and snapshot.get("selected_version")
                    and not executed
                ):
                    assessment.unknowns.append(
                        f"Executed target version evidence missing: {task_id}"
                    )
        if str(task.get("status", "")).lower() == "qa_passed":
            envelope = qa_results.get(current, {})
            if not current or not snapshot:
                assessment.violations.append(
                    f"Accepted QA task lacks its current committed attempt: {task_id}"
                )
            if not isinstance(envelope, Mapping):
                assessment.unknowns.append(f"Current-attempt QA evidence malformed: {task_id}")
                continue
            evaluation = envelope.get("evaluation", {})
            if not isinstance(evaluation, Mapping):
                assessment.unknowns.append(f"Current-attempt QA evaluation malformed: {task_id}")
                continue
            if not current or not envelope:
                assessment.unknowns.append(f"Current-attempt QA evidence missing: {task_id}")
            elif (
                envelope.get("task_id") != task_id
                or envelope.get("attempt_id") != current
                or evaluation.get("task_id") != task_id
                or (snapshot and envelope.get("task_revision") != snapshot.get("task_revision"))
            ):
                assessment.violations.append(f"Stale or unrelated accepted QA result: {task_id}")
            elif not evaluation.get("passed"):
                assessment.violations.append(f"Accepted task lacks passing QA: {task_id}")
            elif envelope.get("qa_policy_source") not in {None, "attempt_snapshot"}:
                assessment.violations.append(
                    f"Accepted QA policy is not sourced from the committed attempt: {task_id}"
                )
            elif envelope.get("qa_policy_source") is None:
                assessment.unknowns.append(
                    f"Current-attempt QA policy provenance missing: {task_id}"
                )
            elif snapshot.get("qa_policy") is None:
                assessment.unknowns.append(f"Committed QA policy evidence missing: {task_id}")
            elif envelope.get("qa_policy") != snapshot.get("qa_policy"):
                assessment.violations.append(
                    f"Accepted QA policy disagrees with the committed attempt: {task_id}"
                )
            elif any(
                snapshot.get(field) is not None and envelope.get(field) != snapshot.get(field)
                for field in ("cluster_id", "dispatch_batch_id", "action_digest")
            ):
                assessment.violations.append(
                    f"Accepted QA result does not match committed batch identity: {task_id}"
                )
    try:
        assessment.unresolved = terminal_outcome_issues(state)
    except (TypeError, ValueError, AttributeError) as exc:
        assessment.violations.append(f"Malformed terminal outcome: {exc}")
    scan = state.get("final_full_scan_result")
    if queue and not scan:
        assessment.unknowns.append("Terminal scan coverage missing")
    if (
        not queue
        and (evidence.request.get("issues") or evidence.request.get("valid_groups"))
        and not state.get("initial_triage_executed")
    ):
        assessment.unknowns.append("No tasks and no completed triage evidence")
    changed = set(state.get("changed_files") or [])
    patch = str(state.get("diff") or "")
    patch_files = set(re.findall(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE))
    patch_files.update(re.findall(r"^--- a/(.+)$", patch, re.MULTILINE))
    if changed != patch_files:
        assessment.violations.append("Changed-file claims disagree with the proposed patch")
    if any(path.startswith("/") or ".." in Path(path).parts for path in changed):
        assessment.violations.append("Patch contains a path outside the repository")
    isolation = state.get("delta_isolation_by_cluster") or {}
    if isinstance(isolation, Mapping):
        for cluster_id, result in isolation.items():
            if isinstance(result, Mapping) and result.get("candidate_restored") is False:
                assessment.violations.append(
                    f"Atomic candidate rollback was not verified for cluster {cluster_id}"
                )
    clean_claim = state.get("status") == "completed"
    if clean_claim and (assessment.unresolved or state.get("errors")):
        assessment.violations.append(
            "Run claims clean completion despite unresolved outcomes or errors"
        )
    report = str(state.get("report_markdown") or "")
    if report and (assessment.unresolved or state.get("errors")):
        for sentence in re.split(r"[.!?\n]+", report):
            normalized = sentence.casefold()
            if re.search(
                r"\b(?:all|every)\b.{0,100}\b(?:finding|vulnerabilit(?:y|ies)|issue|group)s?\b"
                r".{0,80}\b(?:resolved|remediated|fixed|cleared)\b",
                normalized,
            ) and not re.search(
                r"\b(?:not|never|no|unable|cannot|could not|failed to)\b", normalized
            ):
                assessment.violations.append(
                    "Report claims every finding is resolved despite incomplete outcomes"
                )
                break
    if evidence.source == "fresh":
        if evidence.host_unchanged is not True:
            assessment.violations.append("Host repository preservation was not verified")
        if evidence.volumes_removed is not True:
            assessment.violations.append("Docker volume removal was not verified")
    else:
        assessment.unknowns.append(
            "Host preservation and volume removal were not independently verified"
        )
    assessment.remediation_complete = (
        not (assessment.unresolved or assessment.violations or state.get("errors"))
        and state.get("status") in {"completed", "no_work"}
        and not any(
            any(
                phrase in item
                for phrase in (
                    "QA evidence missing",
                    "worker evidence missing",
                    "version could not be matched",
                    "target version evidence missing",
                    "QA policy provenance missing",
                    "committed QA policy evidence missing",
                    "scan coverage missing",
                    "triage evidence",
                )
            )
            for item in assessment.unknowns
        )
    )
    return assessment


def judge_payload(
    evidence: RunEvidence, assessment: ContractAssessment
) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
    """Build compact factual judge inputs while preserving outcomes and failures.

    Returns:
        Goal, factual outcome JSON, compact executed tools, and omission counts.
        Every task outcome and failure remains represented; text and trace details
        have deterministic bounds, with omission counts returned to the caller.
    """
    state = evidence.final_state
    omissions: Counter[str] = Counter()

    def clipped(value: Any, limit: int, label: str) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        omissions[label] += len(text) - limit
        half = limit // 2
        return text[:half] + "\n[detail omitted]\n" + text[-half:]

    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    def pick(
        value: Any, fields: tuple[str, ...], *, bounded: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        source = mapping(value)
        result = {}
        for field in fields:
            if field not in source:
                continue
            item = source[field]
            if field in bounded and isinstance(item, str):
                item = clipped(item, 3000, f"{field}_characters_omitted")
            result[field] = item
        return result

    def project_list(value: Any, fields: tuple[str, ...]) -> list[Any]:
        if not isinstance(value, list):
            return []
        return [pick(item, fields) for item in value]

    def project_map(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
        return {key: pick(item, fields) for key, item in sorted(mapping(value).items())}

    scan_fields = (
        "completed",
        "authoritative",
        "coverage_scope",
        "coverage_complete",
        "found_identifiers",
        "remaining_target_identifiers",
        "new_identifiers",
        "status",
        "triage_required",
        "error",
    )

    def project_scan(value: Any) -> dict[str, Any]:
        scan = pick(value, scan_fields, bounded=("error",))
        scan["found_issues"] = project_list(
            mapping(value).get("found_issues"),
            ("issue_id", "cve_id", "ghsa_id", "package_name", "installed_version", "severity"),
        )
        return scan

    def project_worker(value: Any) -> dict[str, Any]:
        worker = pick(
            value,
            (
                "attempt_id",
                "task_id",
                "task_revision",
                "cluster_id",
                "dispatch_batch_id",
                "action_digest",
                "status",
                "executed_versions",
                "changed_files",
                "instruction_digest",
            ),
        )
        summary = mapping(mapping(value).get("action_summary"))
        if summary:
            worker["action_summary"] = pick(
                summary,
                ("task_id", "attempt_id", "task_revision", "status", "summary"),
                bounded=("summary",),
            )
        diagnostics = mapping(mapping(value).get("execution_diagnostics"))
        worker["execution_diagnostics"] = pick(
            diagnostics,
            (
                "attempted_versions",
                "executed_versions",
                "effective_target_version",
                "effective_dependency_type",
                "validation_calls",
                "validation_input_errors",
                "validation_passed",
                "failure_reason",
                "final_selected_targeted_test",
                "validated_files",
                "infrastructure_failure_details",
            ),
            bounded=("failure_reason", "infrastructure_failure_details"),
        )
        raw_errors = mapping(value).get("errors") or []
        if not isinstance(raw_errors, list):
            raw_errors = [raw_errors]
        worker["errors"] = [
            clipped(error, 3000, "worker_error_characters_omitted") for error in raw_errors
        ]
        return worker

    def project_qa(value: Any) -> dict[str, Any]:
        qa = pick(
            value,
            (
                "attempt_id",
                "task_id",
                "task_revision",
                "cluster_id",
                "dispatch_batch_id",
                "action_digest",
                "qa_policy",
                "qa_policy_source",
                "batch_scan_candidate_retained",
            ),
        )
        evaluation = mapping(mapping(value).get("evaluation"))
        qa["evaluation"] = pick(
            evaluation,
            (
                "task_id",
                "passed",
                "failure_category",
                "retry_feedback",
                "contract_error",
                "contract_error_reason",
                "evidence_inconclusive",
                "deterministic_gates",
                "semantic_security_review",
                "test_attribution",
                "scan_evidence",
            ),
            bounded=("retry_feedback", "contract_error_reason"),
        )
        qa["batch_scan_evidence"] = project_scan(mapping(value).get("batch_scan_evidence"))
        return qa

    group_fields = (
        "group_id",
        "package_name",
        "package",
        "ecosystem",
        "cve_ids",
        "ghsa_ids",
        "severity",
        "priority",
        "status",
        "target_identifiers",
    )
    issue_fields = (
        "issue_id",
        "cve_id",
        "ghsa_id",
        "package_name",
        "ecosystem",
        "installed_version",
        "severity",
        "description",
        "fixed_version",
        "file_path",
    )

    task_fields = (
        "task_id",
        "task_revision",
        "parent_task_id",
        "parent_group_id",
        "is_synthetic",
        "current_attempt_id",
        "status",
        "strategy",
        "strategy_stage",
        "retry_count",
        "attempt_count",
        "target_package_name",
        "target_dependency_type",
        "target_identifiers",
        "remaining_target_identifiers",
        "new_identifiers",
        "selected_version",
        "failure_reason",
        "unresolved_reason",
        "last_error",
        "parent_minimum_version",
    )
    snapshot_fields = (
        "attempt_id",
        "task_id",
        "task_revision",
        "state_revision",
        "attempt_number",
        "cluster_id",
        "dispatch_batch_id",
        "action_digest",
        "strategy",
        "strategy_stage",
        "dispatch_node",
        "selected_version",
        "allowed_target_versions",
        "target_package_name",
        "target_dependency_type",
        "allowed_dependency_types",
        "instruction_digest",
        "plan_id",
        "portfolio_plan_id",
        "target_manifest_paths",
        "maven_target_operation",
        "qa_policy",
    )
    outcome = {
        "status": state.get("status"),
        "initial_valid_groups": project_list(state.get("initial_valid_groups"), group_fields),
        "task_queue": {
            task_id: pick(
                task, task_fields, bounded=("failure_reason", "unresolved_reason", "last_error")
            )
            for task_id, task in sorted(mapping(state.get("task_queue")).items())
        },
        "attempt_snapshots_by_id": {
            attempt_id: pick(snapshot, (*snapshot_fields, "instruction"), bounded=("instruction",))
            for attempt_id, snapshot in sorted(
                mapping(state.get("attempt_snapshots_by_id")).items()
            )
        },
        "worker_results_by_attempt": {
            attempt_id: project_worker(worker)
            for attempt_id, worker in sorted(
                mapping(state.get("worker_results_by_attempt")).items()
            )
        },
        "qa_results_by_attempt": {
            attempt_id: project_qa(qa)
            for attempt_id, qa in sorted(mapping(state.get("qa_results_by_attempt")).items())
        },
        "qa_evaluations": project_map(
            state.get("qa_evaluations"),
            (
                "task_id",
                "attempt_id",
                "passed",
                "failure_category",
                "retry_feedback",
                "contract_error",
                "evidence_inconclusive",
            ),
        ),
        "action_summaries": project_map(
            state.get("action_summaries"),
            ("task_id", "attempt_id", "task_revision", "status", "summary"),
        ),
        "final_full_scan_result": project_scan(state.get("final_full_scan_result")),
        "latest_incremental_scan_by_input": {
            key: project_scan(scan)
            for key, scan in sorted(mapping(state.get("latest_incremental_scan_by_input")).items())
        },
        "new_vulnerability_identifiers": state.get("new_vulnerability_identifiers", []),
        "errors": [
            clipped(error, 3000, "error_characters_omitted")
            for error in (
                state.get("errors")
                if isinstance(state.get("errors"), list)
                else [state.get("errors")]
                if state.get("errors")
                else []
            )
        ],
        "changed_files": state.get("changed_files", []),
        "diff": clipped(state.get("diff", ""), 20000, "diff_characters_omitted"),
        "consistency_events": state.get("consistency_events", []),
        "delta_isolation_by_cluster": {
            cluster_id: pick(
                record,
                (
                    "status",
                    "responsible_task_ids",
                    "tested_subsets",
                    "executions",
                    "diagnostic",
                    "candidate_restored",
                    "dispatch_batch_id",
                    "attempt_ids_by_task",
                ),
                bounded=("diagnostic",),
            )
            for cluster_id, record in sorted(
                mapping(state.get("delta_isolation_by_cluster")).items()
            )
        },
    }
    goal = (
        "Assess completion of the original remediation request. If actionable findings are present, "
        "remediate them with a proposed patch validated by current-attempt QA. If the request has "
        "no actionable findings, an evidence-backed no_work outcome is complete and does not need "
        "a patch or QA. Unresolved findings and failed or unverified validation reduce completion "
        "even when stopping was safe. Respect the supplied scope.\n"
        + json.dumps(
            {
                "issues": project_list(evidence.request.get("issues"), issue_fields),
                "valid_groups": project_list(evidence.request.get("valid_groups"), group_fields),
                "system_context": clipped(
                    evidence.request.get("system_context", ""),
                    4000,
                    "request_context_characters_omitted",
                ),
                "target_packages": evidence.request.get("target_packages", []),
                "project_language": evidence.request.get("project_language"),
            },
            sort_keys=True,
        )
    )
    outcome["assessment"] = assessment.model_dump()
    outcome["coverage"] = evidence.coverage
    rendered_outcome = json.dumps(
        {"evidence": outcome},
        sort_keys=True,
    )
    tools = []
    selected_events = list(evidence.events)
    if len(selected_events) > 256:
        error_indexes = {index for index, event in enumerate(selected_events) if event.error}
        selected_indexes = set(error_indexes)
        endpoint_indexes = set(range(64)) | set(
            range(len(selected_events) - 64, len(selected_events))
        )
        endpoint_indexes.difference_update(selected_indexes)
        remaining_budget = max(0, 256 - len(selected_indexes))
        selected_indexes.update(sorted(endpoint_indexes)[:remaining_budget])
        selected_events = [
            event for index, event in enumerate(selected_events) if index in selected_indexes
        ]
        omissions["execution_events_omitted"] += len(evidence.events) - len(selected_events)
    for event in selected_events:
        input_text = json.dumps(event.inputs, sort_keys=True)
        if len(input_text) > 2000:
            input_text = clipped(input_text, 2000, "tool_input_characters_omitted")
        output = json.dumps(event.outputs, sort_keys=True)
        if len(output) > 2000:
            output = clipped(output, 2000, "tool_output_characters_omitted")
        tools.append(
            {
                "name": event.name,
                "input_parameters": {"serialized": input_text},
                "output": {
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "result": output,
                    "error": clipped(event.error, 3000, "tool_error_characters_omitted")
                    if event.error
                    else None,
                },
            }
        )
    names = Counter(event.name for event in evidence.events)
    return (
        goal,
        rendered_outcome,
        tools,
        {
            **dict(omissions),
            "execution_event_count": len(evidence.events),
            "execution_event_type_counts": dict(sorted(names.items())),
            "execution_events_omitted": omissions.get("execution_events_omitted", 0),
        },
    )
