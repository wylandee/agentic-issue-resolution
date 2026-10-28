"""Normalize complete remediation runs and assess deterministic evidence contracts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
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


def _latest_qa_attempts_by_task(
    qa_results: Mapping[str, Any], snapshots: Mapping[str, Any]
) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Return each task's latest recorded QA result using Supervisor ordering.

    Terminal tasks intentionally clear ``current_attempt_id``. Their immutable
    attempt snapshot and QA/worker results remain the authoritative lineage, so
    whole-run assessment reconstructs the accepted attempt from those records.
    """
    latest: dict[str, tuple[tuple[int, int, int, str], str, Mapping[str, Any]]] = {}

    def integer(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    for attempt_key, result in qa_results.items():
        if not isinstance(result, Mapping):
            continue
        task_id = result.get("task_id")
        attempt_id = result.get("attempt_id") or attempt_key
        if not isinstance(task_id, str) or not isinstance(attempt_id, str):
            continue
        snapshot = snapshots.get(attempt_id)
        if not isinstance(snapshot, Mapping):
            snapshot = {}
        order = (
            integer(result.get("task_revision")),
            integer(snapshot.get("attempt_number")),
            integer(snapshot.get("state_revision")),
            attempt_id,
        )
        previous = latest.get(task_id)
        if previous is None or order >= previous[0]:
            latest[task_id] = (order, attempt_id, result)
    return {
        task_id: (attempt_id, result) for task_id, (_order, attempt_id, result) in latest.items()
    }


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
    latest_qa_by_task = _latest_qa_attempts_by_task(qa_results, snapshots)
    for task_id, task in queue.items():
        if not isinstance(task, Mapping):
            assessment.violations.append(f"Task record is malformed: {task_id}")
            continue
        if task.get("task_id") != task_id:
            assessment.violations.append(f"Task identity mismatch: {task_id}")
        current = task.get("current_attempt_id")
        task_status = str(task.get("status", "")).lower()
        terminal_qa_without_pointer = task_status == "qa_passed" and not current
        attempt_id = current
        terminal_qa_envelope: Mapping[str, Any] | None = None
        if terminal_qa_without_pointer:
            latest_qa = latest_qa_by_task.get(task_id)
            if latest_qa is None:
                assessment.violations.append(
                    f"Accepted QA task lacks matching immutable attempt evidence: {task_id}"
                )
            else:
                attempt_id, terminal_qa_envelope = latest_qa
        snapshot = snapshots.get(attempt_id, {}) if attempt_id else {}
        if attempt_id and not snapshot:
            if terminal_qa_without_pointer:
                assessment.violations.append(
                    f"Accepted QA task lacks its attempt snapshot: {task_id}/{attempt_id}"
                )
            else:
                assessment.unknowns.append(f"Committed attempt missing: {task_id}/{attempt_id}")
        if snapshot and not isinstance(snapshot, Mapping):
            assessment.violations.append(f"Committed attempt record is malformed: {task_id}")
            snapshot = {}
        if snapshot and snapshot.get("task_id") != task_id:
            assessment.violations.append(f"Committed attempt belongs to another task: {task_id}")
        if snapshot and snapshot.get("attempt_id") not in {None, attempt_id}:
            assessment.violations.append(f"Committed attempt identity mismatch: {task_id}")
        if (
            snapshot
            and not terminal_qa_without_pointer
            and snapshot.get("task_revision") != task.get("task_revision")
        ):
            assessment.violations.append(f"Committed attempt revision is stale: {task_id}")
        if task.get("parent_task_id") and task["parent_task_id"] not in queue:
            assessment.violations.append(f"Missing parent task: {task_id}")
        if attempt_id:
            worker = worker_results.get(attempt_id)
            if worker is None:
                if task_status in {"optimistically_fixed", "qa_passed"}:
                    assessment.unknowns.append(
                        f"Current-attempt worker evidence missing: {task_id}"
                    )
            elif not isinstance(worker, Mapping):
                assessment.violations.append(f"Current worker result is malformed: {task_id}")
            elif (
                worker.get("task_id") != task_id
                or worker.get("attempt_id") != attempt_id
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
            if isinstance(worker, Mapping) and task_status == "qa_passed":
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
        if task_status == "qa_passed":
            envelope = terminal_qa_envelope or qa_results.get(attempt_id, {})
            if not attempt_id or not snapshot:
                assessment.violations.append(
                    f"Accepted QA task lacks its committed attempt: {task_id}"
                )
            if not isinstance(envelope, Mapping):
                assessment.unknowns.append(f"Current-attempt QA evidence malformed: {task_id}")
                continue
            evaluation = envelope.get("evaluation", {})
            if not isinstance(evaluation, Mapping):
                assessment.unknowns.append(f"Current-attempt QA evaluation malformed: {task_id}")
                continue
            if not attempt_id or not envelope:
                assessment.unknowns.append(f"Current-attempt QA evidence missing: {task_id}")
            elif (
                envelope.get("task_id") != task_id
                or envelope.get("attempt_id") != attempt_id
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
        Every task's terminal status remains represented. Attempt and trace details
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

    def compact_tree(
        value: Any,
        label: str,
        *,
        depth: int = 0,
        string_limit: int = 800,
        list_limit: int = 12,
        map_limit: int = 40,
    ) -> Any:
        """Bound nested diagnostic data while keeping its shape and omissions visible."""
        if isinstance(value, str):
            return clipped(value, string_limit, f"{label}_characters_omitted")
        if depth >= 6:
            if isinstance(value, (Mapping, list)):
                omissions[f"{label}_depth_limited"] += 1
                return f"[{type(value).__name__} omitted at depth limit]"
            return value
        if isinstance(value, Mapping):
            items = list(value.items())
            if len(items) > map_limit:
                omissions[f"{label}_keys_omitted"] += len(items) - map_limit
                items = items[:map_limit]
            return {
                str(key): compact_tree(
                    item,
                    f"{label}_{key}",
                    depth=depth + 1,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    map_limit=map_limit,
                )
                for key, item in items
            }
        if isinstance(value, list):
            items = value
            if len(items) > list_limit:
                head = list_limit - max(1, list_limit // 4)
                tail = list_limit - head
                omissions[f"{label}_items_omitted"] += len(items) - list_limit
                items = [*items[:head], *items[-tail:]]
            return [
                compact_tree(
                    item,
                    f"{label}_{index}",
                    depth=depth + 1,
                    string_limit=string_limit,
                    list_limit=list_limit,
                    map_limit=map_limit,
                )
                for index, item in enumerate(items)
            ]
        return value

    def project_strings(value: Any, label: str, *, limit: int = 64) -> list[str]:
        if not isinstance(value, list):
            return []
        selected = value[:limit]
        if len(value) > len(selected):
            omissions[f"{label}_items_omitted"] += len(value) - len(selected)
        return [
            clipped(item, 240, f"{label}_characters_omitted")
            for item in selected
            if isinstance(item, str)
        ]

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

    def project_list(
        value: Any,
        fields: tuple[str, ...],
        *,
        limit: int = 64,
        label: str = "projected_list",
    ) -> list[Any]:
        if not isinstance(value, list):
            return []
        selected = value[:limit]
        if len(value) > len(selected):
            omissions[f"{label}_items_omitted"] += len(value) - len(selected)
        return [
            compact_tree(pick(item, fields), label, string_limit=600, list_limit=16)
            for item in selected
        ]

    def project_map(
        value: Any,
        fields: tuple[str, ...],
        *,
        bounded: tuple[str, ...] = (),
        limit: int = 64,
    ) -> dict[str, Any]:
        items = sorted(mapping(value).items())
        if len(items) > limit:
            omissions["projected_map_items_omitted"] += len(items) - limit
            items = items[:limit]
        return {
            key: compact_tree(
                pick(item, fields, bounded=bounded),
                "projected_map_value",
                string_limit=600,
                list_limit=16,
            )
            for key, item in items
        }

    def bounded_map_entries(
        value: Any,
        *,
        limit: int,
        label: str,
        prioritized_keys: Sequence[str] = (),
    ) -> list[tuple[str, Any]]:
        """Keep priority records and the newest insertion-ordered tail of a map."""
        items = list(mapping(value).items())
        if len(items) <= limit:
            return items
        by_key = dict(items)
        selected_keys = list(dict.fromkeys(key for key in prioritized_keys if key in by_key))
        if len(selected_keys) > limit:
            selected_keys = selected_keys[-limit:]
        selected = set(selected_keys)
        for key, _item in reversed(items):
            if len(selected_keys) >= limit:
                break
            if key not in selected:
                selected.add(key)
                selected_keys.append(key)
        omissions[f"{label}_items_omitted"] += len(items) - len(selected_keys)
        order = {key: index for index, (key, _item) in enumerate(items)}
        return [(key, by_key[key]) for key in sorted(selected_keys, key=order.__getitem__)]

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
        source = mapping(value)
        scan = pick(
            source,
            (*scan_fields, "complete", "execution_status", "covered_input_paths"),
            bounded=("error",),
        )
        if "error" in scan:
            scan["error"] = clipped(scan["error"], 600, "scan_error_characters_omitted")
        if "covered_input_paths" in scan:
            paths = scan["covered_input_paths"]
            scan["covered_input_path_count"] = len(paths) if isinstance(paths, list) else 0
            scan["covered_input_paths"] = project_strings(
                paths, "scan_covered_input_paths", limit=12
            )
        for key in ("closure_package_names", "closure_lockfile_keys"):
            values = source.get(key)
            if isinstance(values, list):
                scan[f"{key}_count"] = len(values)
                scan[key] = project_strings(values, f"scan_{key}", limit=16)
        for key in ("found_identifiers", "new_identifiers", "remaining_target_identifiers"):
            if key in scan:
                scan[f"{key}_count"] = len(scan[key]) if isinstance(scan[key], list) else 0
                scan[key] = project_strings(scan[key], f"scan_{key}", limit=64)
        scan["found_issues"] = project_list(
            source.get("found_issues"),
            (
                "issue_id",
                "cve_id",
                "ghsa_id",
                "package_name",
                "installed_version",
                "package_version",
                "severity",
            ),
            limit=8,
            label="scan_found_issues",
        )
        occurrences = source.get("covered_occurrences")
        if isinstance(occurrences, list):
            identifiers: set[str] = set()
            packages: set[str] = set()
            for occurrence in occurrences:
                if not isinstance(occurrence, Mapping):
                    continue
                package_name = occurrence.get("package_name")
                if isinstance(package_name, str) and package_name:
                    packages.add(package_name)
                for identifier in occurrence.get("found_identifiers", []) or []:
                    if isinstance(identifier, str) and identifier:
                        identifiers.add(identifier)
                for issue in occurrence.get("found_issues", []) or []:
                    if not isinstance(issue, Mapping):
                        continue
                    for key in ("cve_id", "ghsa_id"):
                        identifier = issue.get(key)
                        if isinstance(identifier, str) and identifier:
                            identifiers.add(identifier)
            scan["covered_occurrence_count"] = len(occurrences)
            scan["covered_packages"] = sorted(packages)[:24]
            scan["covered_identifiers"] = sorted(identifiers)[:64]
            if len(packages) > 24:
                omissions["scan_covered_packages_omitted"] += len(packages) - 24
            if len(identifiers) > 64:
                omissions["scan_covered_identifiers_omitted"] += len(identifiers) - 64
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
        if "failure_reason" in worker["execution_diagnostics"]:
            worker["execution_diagnostics"]["failure_reason"] = clipped(
                worker["execution_diagnostics"]["failure_reason"],
                1000,
                "worker_failure_reason_characters_omitted",
            )
        if "infrastructure_failure_details" in worker["execution_diagnostics"]:
            worker["execution_diagnostics"]["infrastructure_failure_details"] = clipped(
                worker["execution_diagnostics"]["infrastructure_failure_details"],
                1000,
                "infrastructure_failure_characters_omitted",
            )
        for key in ("attempted_versions", "executed_versions", "validated_files"):
            if key in worker["execution_diagnostics"]:
                worker["execution_diagnostics"][key] = project_strings(
                    worker["execution_diagnostics"][key], f"worker_{key}", limit=16
                )
        raw_errors = mapping(value).get("errors") or []
        if not isinstance(raw_errors, list):
            raw_errors = [raw_errors]
        selected_errors = (
            raw_errors if len(raw_errors) <= 8 else [*raw_errors[:2], *raw_errors[-6:]]
        )
        worker["errors"] = [
            clipped(error, 1000, "worker_error_characters_omitted") for error in selected_errors
        ]
        if len(raw_errors) > 8:
            omissions["worker_errors_omitted"] += len(raw_errors) - 8
        return compact_tree(worker, "worker_result", string_limit=400, list_limit=8, map_limit=16)

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
        projected_evaluation = pick(
            evaluation,
            (
                "task_id",
                "passed",
                "failure_category",
                "retry_feedback",
                "contract_error",
                "contract_error_reason",
                "evidence_inconclusive",
            ),
            bounded=("retry_feedback", "contract_error_reason"),
        )
        for key in ("retry_feedback", "contract_error_reason"):
            if key in projected_evaluation:
                projected_evaluation[key] = clipped(
                    projected_evaluation[key], 1200, f"qa_{key}_characters_omitted"
                )
        for key in ("deterministic_gates", "semantic_security_review", "test_attribution"):
            if key in evaluation:
                projected_evaluation[key] = compact_tree(
                    evaluation[key],
                    f"qa_{key}",
                    string_limit=450,
                    list_limit=8,
                    map_limit=16,
                )
        scan_evidence = evaluation.get("scan_evidence")
        if isinstance(scan_evidence, Mapping):
            projected_evaluation["scan_evidence"] = project_scan(scan_evidence)
        qa["evaluation"] = projected_evaluation
        qa["batch_scan_evidence"] = project_scan(mapping(value).get("batch_scan_evidence"))
        return compact_tree(qa, "qa_result", string_limit=500, list_limit=8, map_limit=16)

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
    raw_snapshots = mapping(state.get("attempt_snapshots_by_id"))
    raw_qa_results = mapping(state.get("qa_results_by_attempt"))
    latest_qa_attempts = _latest_qa_attempts_by_task(raw_qa_results, raw_snapshots)
    latest_attempt_ids = {attempt_id for attempt_id, _result in latest_qa_attempts.values()}
    latest_attempt_ids.update(
        task.get("current_attempt_id")
        for task in mapping(state.get("task_queue")).values()
        if isinstance(task, Mapping) and isinstance(task.get("current_attempt_id"), str)
    )

    task_queue = {}
    for task_id, task in sorted(mapping(state.get("task_queue")).items()):
        task_queue[task_id] = compact_tree(
            pick(task, task_fields, bounded=("failure_reason", "unresolved_reason", "last_error")),
            f"task_{task_id}",
            string_limit=400,
            list_limit=12,
        )
    snapshots = {}
    for attempt_id, snapshot in bounded_map_entries(
        raw_snapshots,
        limit=64,
        label="attempt_snapshots",
        prioritized_keys=sorted(latest_attempt_ids),
    ):
        projected = pick(
            snapshot,
            (*snapshot_fields, "instruction"),
            bounded=("instruction",),
        )
        if isinstance(projected.get("instruction"), str):
            projected["instruction"] = clipped(
                projected["instruction"], 900, "attempt_instruction_characters_omitted"
            )
        snapshots[attempt_id] = compact_tree(
            projected,
            "attempt_snapshot",
            string_limit=400,
            list_limit=8,
            map_limit=16,
        )
    errors = state.get("errors")
    if not isinstance(errors, list):
        errors = [errors] if errors else []
    if len(errors) > 24:
        omissions["run_errors_omitted"] += len(errors) - 24
        errors = [*errors[:8], *errors[-16:]]

    outcome = {
        "status": state.get("status"),
        "initial_valid_groups": project_list(
            state.get("initial_valid_groups"), group_fields, limit=32, label="initial_groups"
        ),
        "task_queue": task_queue,
        "attempt_snapshots_by_id": snapshots,
        "worker_results_by_attempt": {
            attempt_id: project_worker(worker)
            for attempt_id, worker in bounded_map_entries(
                state.get("worker_results_by_attempt"),
                limit=64,
                label="worker_results",
                prioritized_keys=sorted(latest_attempt_ids),
            )
        },
        "qa_results_by_attempt": {
            attempt_id: project_qa(qa)
            for attempt_id, qa in bounded_map_entries(
                raw_qa_results,
                limit=64,
                label="qa_results",
                prioritized_keys=sorted(latest_attempt_ids),
            )
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
            bounded=("retry_feedback",),
        ),
        "action_summaries": project_map(
            state.get("action_summaries"),
            ("task_id", "attempt_id", "task_revision", "status", "summary"),
            bounded=("summary",),
        ),
        "final_full_scan_result": project_scan(state.get("final_full_scan_result")),
        "latest_incremental_scan_by_input": {
            key: project_scan(scan)
            for key, scan in bounded_map_entries(
                state.get("latest_incremental_scan_by_input"),
                limit=24,
                label="incremental_scans",
            )
        },
        "new_vulnerability_identifiers": project_strings(
            state.get("new_vulnerability_identifiers", []),
            "new_vulnerability_identifiers",
            limit=100,
        ),
        "errors": [clipped(error, 800, "error_characters_omitted") for error in errors],
        "changed_files": project_strings(state.get("changed_files", []), "changed_files", limit=64),
        "diff": clipped(state.get("diff", ""), 10000, "diff_characters_omitted"),
        "consistency_events": compact_tree(
            state.get("consistency_events", []), "consistency_events", list_limit=12
        ),
        "delta_isolation_by_cluster": {
            cluster_id: compact_tree(
                pick(
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
                ),
                "delta_isolation",
                string_limit=500,
                list_limit=8,
            )
            for cluster_id, record in bounded_map_entries(
                state.get("delta_isolation_by_cluster"),
                limit=16,
                label="delta_isolation",
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
                "issues": [
                    {
                        **issue,
                        "description": clipped(
                            issue.get("description", ""),
                            500,
                            "request_issue_description_characters_omitted",
                        ),
                    }
                    for issue in project_list(
                        evidence.request.get("issues"),
                        issue_fields,
                        limit=48,
                        label="request_issues",
                    )
                ],
                "valid_groups": project_list(
                    evidence.request.get("valid_groups"),
                    group_fields,
                    limit=48,
                    label="request_groups",
                ),
                "system_context": clipped(
                    evidence.request.get("system_context", ""),
                    1800,
                    "request_context_characters_omitted",
                ),
                "target_packages": project_strings(
                    evidence.request.get("target_packages", []), "request_target_packages", limit=48
                ),
                "project_language": evidence.request.get("project_language"),
            },
            sort_keys=True,
        )
    )
    outcome["assessment"] = compact_tree(
        assessment.model_dump(), "assessment", string_limit=800, list_limit=24
    )
    outcome["coverage"] = compact_tree(
        evidence.coverage, "coverage", string_limit=500, list_limit=16
    )
    tools = []
    max_events = 24
    error_indexes = [index for index, event in enumerate(evidence.events) if event.error]
    error_by_name: dict[str, int] = {}
    for index in error_indexes:
        error_by_name[evidence.events[index].name] = index
    prioritized_errors = list(dict.fromkeys(error_by_name.values()))
    for index in reversed(error_indexes):
        if len(prioritized_errors) >= 16:
            break
        if index not in prioritized_errors:
            prioritized_errors.append(index)
    if len(prioritized_errors) > 16:
        prioritized_errors = prioritized_errors[-16:]
    selected_indexes = set(prioritized_errors)
    for index in (
        *range(min(4, len(evidence.events))),
        *range(max(0, len(evidence.events) - 4), len(evidence.events)),
    ):
        if len(selected_indexes) >= max_events:
            break
        selected_indexes.add(index)
    for index in range(len(evidence.events) - 1, -1, -1):
        if len(selected_indexes) >= max_events:
            break
        selected_indexes.add(index)
    selected_events = [evidence.events[index] for index in sorted(selected_indexes)]
    omissions["execution_events_omitted"] += len(evidence.events) - len(selected_events)
    for event in selected_events:
        input_text = json.dumps(event.inputs, sort_keys=True)
        if len(input_text) > 350:
            input_text = clipped(input_text, 350, "tool_input_characters_omitted")
        output = json.dumps(event.outputs, sort_keys=True)
        if len(output) > 350:
            output = clipped(output, 350, "tool_output_characters_omitted")
        tools.append(
            {
                "name": event.name,
                "input_parameters": {"serialized": input_text},
                "output": {
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "result": output,
                    "error": clipped(event.error, 500, "tool_error_characters_omitted")
                    if event.error
                    else None,
                },
            }
        )
    names = Counter(event.name for event in evidence.events)
    omission_fields = (
        "attempt_snapshots_items_omitted",
        "worker_results_items_omitted",
        "qa_results_items_omitted",
        "incremental_scans_items_omitted",
        "projected_map_items_omitted",
        "run_errors_omitted",
        "execution_events_omitted",
        "tool_input_characters_omitted",
        "tool_output_characters_omitted",
        "tool_error_characters_omitted",
        "diff_characters_omitted",
    )
    outcome["evidence_compaction"] = {
        "note": (
            "Large scan occurrence sets and execution events are summarized for judging. "
            "The deterministic assessment was computed from complete parsed evidence."
        ),
        "omitted_counts": {
            field: omissions[field] for field in omission_fields if omissions[field]
        },
        "execution_events_included": len(selected_events),
        "execution_events_total": len(evidence.events),
    }
    rendered_outcome = json.dumps({"evidence": outcome}, sort_keys=True)
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
