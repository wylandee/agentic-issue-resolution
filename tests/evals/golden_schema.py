"""Shared validation and loading helpers for evaluation golden cases.

The evaluation datasets deliberately contain both a stable completion contract
and component-specific replay data.  This module keeps that boundary explicit:
``expected_tools`` describes the independent contract, while historical
observations belong in ``offline_fixture`` and are never used as live output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import DecisionCode, ScanFallbackReason, ScanScope

REQUIRED_CASE_FIELDS = ("case_id", "input", "context", "expected_output", "expected_tools")
LIVE_REPLAY_EVAL_TYPES = frozenset(
    {"triage", "qa_critic", "update_subagent", "workaround_subagent", "supervisor"}
)
GOLDEN_DATASET_NAMES = (
    "triage_cases",
    "qa_cases",
    "update_subagent_cases",
    "workaround_subagent_cases",
    "supervisor_cases",
    "report_cases",
    "fix_planner_cases",
)


class GoldenSchemaError(ValueError):
    """Raised when a golden dataset violates the shared case contract."""


def _case_label(dataset_name: str, case: dict[str, Any], index: int) -> str:
    """Return a useful location label for a case error."""
    return f"{dataset_name}[{index}] ({case.get('case_id', '<missing case_id>')!r})"


def _validate_supervisor_replay_input(
    replay_input: Mapping[str, Any],
    label: str,
) -> list[str]:
    """Validate the shallow JSON shape of a tactical replay payload."""
    violations: list[str] = []
    candidate_sets = replay_input.get("candidate_sets")
    if not isinstance(candidate_sets, list):
        violations.append(f"{label}.replay.input.candidate_sets must be a list")
    else:
        required_fields = {
            "strategy",
            "target_package_name",
            "dependency_type",
            "security_floor",
            "versions",
            "canonical_version",
            "peer_compatible",
        }
        for index, candidate in enumerate(candidate_sets):
            candidate_label = f"{label}.replay.input.candidate_sets[{index}]"
            if not isinstance(candidate, Mapping):
                violations.append(f"{candidate_label} must be an object")
                continue
            missing = required_fields - set(candidate)
            if missing:
                violations.append(
                    f"{candidate_label} is missing required fields: {', '.join(sorted(missing))}"
                )
            for field in ("strategy", "target_package_name", "security_floor"):
                value = candidate.get(field)
                if not isinstance(value, str) or not value.strip():
                    violations.append(f"{candidate_label}.{field} must be a non-empty string")
            dependency_type = candidate.get("dependency_type")
            if dependency_type is not None and not isinstance(dependency_type, str):
                violations.append(f"{candidate_label}.dependency_type must be a string or null")
            versions = candidate.get("versions")
            if not isinstance(versions, list) or not all(
                isinstance(version, str) for version in versions
            ):
                violations.append(f"{candidate_label}.versions must be a list of strings")
            canonical_version = candidate.get("canonical_version")
            if canonical_version is not None and not isinstance(canonical_version, str):
                violations.append(f"{candidate_label}.canonical_version must be a string or null")
            if not isinstance(candidate.get("peer_compatible"), bool):
                violations.append(f"{candidate_label}.peer_compatible must be a boolean")

    for field in ("evaluation", "worker_result", "retry_diagnostics"):
        if field in replay_input and not isinstance(replay_input[field], Mapping):
            violations.append(f"{label}.replay.input.{field} must be an object")
    prior_attempts = replay_input.get("prior_attempts")
    if "prior_attempts" in replay_input and (
        not isinstance(prior_attempts, list)
        or not all(isinstance(attempt, Mapping) for attempt in prior_attempts)
    ):
        violations.append(f"{label}.replay.input.prior_attempts must be a list of objects")
    return violations


def _validate_supervisor_expected_replay(case: Mapping[str, Any], label: str) -> list[str]:
    """Validate deterministic expectations for one tactical production replay."""
    violations: list[str] = []
    replay = case.get("expected_replay")
    if not isinstance(replay, Mapping):
        return [f"{label}.expected_replay must be an object"]

    allowed_fields = {
        "model_invocations",
        "verification_accepted",
        "decision_code",
        "spawn_request_count",
    }
    unknown_fields = set(replay) - allowed_fields
    if unknown_fields:
        violations.append(
            f"{label}.expected_replay contains unsupported fields: "
            f"{', '.join(sorted(unknown_fields))}"
        )
    for field in allowed_fields:
        if field not in replay:
            violations.append(f"{label}.expected_replay is missing {field!r}")

    model_invocations = replay.get("model_invocations")
    if type(model_invocations) is not int or model_invocations < 0:
        violations.append(
            f"{label}.expected_replay.model_invocations must be a nonnegative integer"
        )
    spawn_request_count = replay.get("spawn_request_count")
    if type(spawn_request_count) is not int or spawn_request_count < 0:
        violations.append(
            f"{label}.expected_replay.spawn_request_count must be a nonnegative integer"
        )
    verification_accepted = replay.get("verification_accepted")
    if verification_accepted is not None and not isinstance(verification_accepted, bool):
        violations.append(
            f"{label}.expected_replay.verification_accepted must be a boolean or null"
        )
    decision_code = replay.get("decision_code")
    decision_values = {decision.value for decision in DecisionCode}
    if decision_code is not None and (
        not isinstance(decision_code, str) or decision_code not in decision_values
    ):
        violations.append(
            f"{label}.expected_replay.decision_code must be a DecisionCode value or null"
        )

    no_call_case = case.get("case_id") == "supervisor-inconclusive-qa-suppressed"
    if no_call_case:
        if model_invocations != 0:
            violations.append(
                f"{label}.expected_replay no-call case must have zero model_invocations"
            )
        if verification_accepted is not None or decision_code is not None:
            violations.append(
                f"{label}.expected_replay no-call case must use null verification and decision"
            )
        if spawn_request_count != 0 or case.get("expected_tools") != []:
            violations.append(
                f"{label}.expected_replay no-call case must have no spawn and no expected tools"
            )
    elif verification_accepted is None or model_invocations == 0:
        violations.append(
            f"{label}.expected_replay null verification and zero calls are reserved for "
            "the inconclusive no-call case"
        )
    elif verification_accepted is True and decision_code is None:
        violations.append(f"{label}.expected_replay accepted verification requires a decision code")
    elif verification_accepted is False and decision_code is not None:
        violations.append(
            f"{label}.expected_replay rejected verification must not have a decision code"
        )
    return violations


def validate_golden_case(
    case: Any,
    *,
    dataset_name: str = "golden",
    index: int = 0,
) -> list[str]:
    """Return contract violations for one canonical golden case.

    Args:
        case: JSON-decoded golden case.
        dataset_name: Dataset label used in diagnostics.
        index: Zero-based case index used in diagnostics.

    Returns:
        Human-readable contract violations. An empty list means the case is
        valid.

    Notes:
        Domain-specific fields remain intentionally open-ended. Only the
        shared envelope and the replay payload required by live-agent suites
        are validated here.
    """
    if not isinstance(case, dict):
        return [f"{dataset_name}[{index}] must be an object"]

    label = _case_label(dataset_name, case, index)
    violations: list[str] = []
    for field in REQUIRED_CASE_FIELDS:
        if field not in case:
            violations.append(f"{label} is missing required field {field!r}")

    if not isinstance(case.get("case_id"), str) or not case.get("case_id", "").strip():
        violations.append(f"{label} case_id must be a non-empty string")
    if not isinstance(case.get("input"), str) or not case.get("input", "").strip():
        violations.append(f"{label} input must be a non-empty string")
    context = case.get("context")
    if not isinstance(context, list) or not all(isinstance(item, str) for item in context):
        violations.append(f"{label} context must be a list of strings")
    if (
        not isinstance(case.get("expected_output"), str)
        or not case.get("expected_output", "").strip()
    ):
        violations.append(f"{label} expected_output must be a non-empty string")

    expected_tools = case.get("expected_tools")
    if not isinstance(expected_tools, list):
        violations.append(f"{label} expected_tools must be a list")
    else:
        for tool_index, tool_call in enumerate(expected_tools):
            tool_label = f"{label}.expected_tools[{tool_index}]"
            if not isinstance(tool_call, dict):
                violations.append(f"{tool_label} must be an object")
                continue
            unknown_fields = set(tool_call) - {"name", "args", "output"}
            if unknown_fields:
                violations.append(
                    f"{tool_label} contains unsupported fields: {', '.join(sorted(unknown_fields))}"
                )
            if not isinstance(tool_call.get("name"), str) or not tool_call.get("name", "").strip():
                violations.append(f"{tool_label}.name must be a non-empty string")
            if "args" not in tool_call:
                violations.append(f"{tool_label}.args is required")
            elif not isinstance(tool_call["args"], dict):
                violations.append(f"{tool_label}.args must be an object")
            if "output" in tool_call and not isinstance(tool_call["output"], str):
                violations.append(f"{tool_label}.output must be a string when present")

    eval_type = case.get("eval_type")
    if eval_type in {"triage", "report", "fix_planner"} and expected_tools not in ([], None):
        violations.append(f"{label} {eval_type} cases must define expected_tools as []")
    if eval_type in LIVE_REPLAY_EVAL_TYPES:
        replay = case.get("replay")
        if not isinstance(replay, dict):
            violations.append(f"{label} live replay cases require a replay object")
        elif not isinstance(replay.get("input"), dict):
            violations.append(f"{label}.replay.input must be an object")
        else:
            replay_input = replay["input"]
            required_replay_fields = {
                "triage": ("system_context",),
                "qa_critic": (
                    "vulnerability_context",
                    "execution_context",
                    "execution_logs",
                    "workspace_files",
                ),
                "update_subagent": (
                    "workspace_files",
                    "allowed_target_versions",
                    "allowed_dependency_types",
                ),
                "workaround_subagent": (
                    "workspace_files",
                    "allowed_target_versions",
                    "allowed_dependency_types",
                ),
                "supervisor": ("task", "group", "candidate_sets"),
            }[eval_type]
            for replay_field in required_replay_fields:
                if replay_field not in replay_input:
                    violations.append(f"{label}.replay.input is missing {replay_field!r}")

            expected_mapping_fields = {
                "triage": ("system_context",),
                "qa_critic": (
                    "vulnerability_context",
                    "execution_context",
                    "execution_logs",
                    "workspace_files",
                ),
                "update_subagent": ("workspace_files",),
                "workaround_subagent": ("workspace_files",),
                "supervisor": ("task", "group"),
            }[eval_type]
            for replay_field in expected_mapping_fields:
                value = replay_input.get(replay_field)
                if not isinstance(value, Mapping):
                    violations.append(f"{label}.replay.input.{replay_field} must be an object")
            if eval_type in {"update_subagent", "workaround_subagent"}:
                for replay_field in ("allowed_target_versions", "allowed_dependency_types"):
                    value = replay_input.get(replay_field)
                    if not isinstance(value, list) or not all(
                        isinstance(item, str) for item in value
                    ):
                        violations.append(
                            f"{label}.replay.input.{replay_field} must be a list of strings"
                        )
            if eval_type == "qa_critic":
                execution_context = replay_input.get("execution_context")
                if isinstance(execution_context, Mapping):
                    for scan_field in (
                        "requested_scope",
                        "effective_scope",
                        "scan_complete",
                        "covered_task_ids",
                        "closure_package_names",
                        "closure_lockfile_keys",
                        "fallback_reason",
                    ):
                        if scan_field not in execution_context:
                            violations.append(
                                f"{label}.replay.input.execution_context is missing "
                                f"typed scan field {scan_field!r}"
                            )
                    for scope_field in ("requested_scope", "effective_scope"):
                        scope = execution_context.get(scope_field)
                        if scope not in {item.value for item in ScanScope}:
                            violations.append(
                                f"{label}.replay.input.execution_context.{scope_field} "
                                "must be 'targeted' or 'full'"
                            )
                    if not isinstance(execution_context.get("scan_complete"), bool):
                        violations.append(
                            f"{label}.replay.input.execution_context.scan_complete "
                            "must be a boolean"
                        )
                    for list_field in (
                        "covered_task_ids",
                        "closure_package_names",
                        "closure_lockfile_keys",
                    ):
                        value = execution_context.get(list_field)
                        if not isinstance(value, list) or not all(
                            isinstance(item, str) for item in value
                        ):
                            violations.append(
                                f"{label}.replay.input.execution_context.{list_field} "
                                "must be a list of strings"
                            )
                    fallback_reason = execution_context.get("fallback_reason")
                    if fallback_reason is not None and (
                        not isinstance(fallback_reason, str)
                        or fallback_reason not in {item.value for item in ScanFallbackReason}
                    ):
                        violations.append(
                            f"{label}.replay.input.execution_context.fallback_reason "
                            "must be a valid scan fallback reason or null"
                        )
            workspace_files = replay_input.get("workspace_files")
            if workspace_files is not None and (
                not isinstance(workspace_files, Mapping)
                or not all(
                    isinstance(path, str) and isinstance(content, str)
                    for path, content in workspace_files.items()
                )
            ):
                violations.append(
                    f"{label}.replay.input.workspace_files must map string paths to string contents"
                )
            if eval_type == "supervisor":
                violations.extend(_validate_supervisor_replay_input(replay_input, label))
        fixture = case.get("offline_fixture")
        if not isinstance(fixture, dict):
            violations.append(f"{label} live replay cases require an offline_fixture object")
        else:
            if "actual_output" not in fixture:
                violations.append(f"{label}.offline_fixture is missing actual_output")
            if "actual_tools" not in fixture:
                violations.append(f"{label}.offline_fixture is missing actual_tools")
            elif not isinstance(fixture["actual_tools"], list):
                violations.append(f"{label}.offline_fixture.actual_tools must be a list")

    if eval_type == "supervisor":
        violations.extend(_validate_supervisor_expected_replay(case, label))
    return violations


def validate_golden_dataset(
    cases: Any,
    *,
    dataset_name: str = "golden",
) -> list[str]:
    """Return all contract violations for a decoded golden dataset.

    Args:
        cases: JSON-decoded list of case objects.
        dataset_name: Dataset label used in diagnostics.

    Returns:
        Human-readable violations, including duplicate case identifiers.
    """
    if not isinstance(cases, list):
        return [f"{dataset_name} must contain a JSON list of cases"]

    violations: list[str] = []
    seen_ids: set[str] = set()
    for index, case in enumerate(cases):
        violations.extend(validate_golden_case(case, dataset_name=dataset_name, index=index))
        if isinstance(case, dict):
            case_id = case.get("case_id")
            if isinstance(case_id, str) and case_id:
                if case_id in seen_ids:
                    violations.append(f"{dataset_name} contains duplicate case_id {case_id!r}")
                seen_ids.add(case_id)
    return violations


def load_golden_dataset(path: Path, *, dataset_name: str | None = None) -> list[dict[str, Any]]:
    """Load and validate a canonical golden JSON file.

    Args:
        path: JSON file path.
        dataset_name: Optional diagnostic name; defaults to the file stem.

    Returns:
        Validated case dictionaries. A missing file returns an empty list so
        optional eval collections remain safe in minimal environments.

    Raises:
        GoldenSchemaError: If JSON is malformed, the top-level shape is not a
            case list, or any case violates the canonical contract.
        OSError: If an existing file cannot be read.
    """
    if not path.exists():
        return []
    name = dataset_name or path.stem
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenSchemaError(f"{name} is not valid JSON: {exc}") from exc

    cases = payload.get("cases") if isinstance(payload, dict) else payload
    violations = validate_golden_dataset(cases, dataset_name=name)
    if violations:
        raise GoldenSchemaError("\n".join(violations))
    return list(cases)


def offline_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """Return the historical offline fixture for a canonical case.

    Args:
        case: Canonical golden case.

    Returns:
        The fixture mapping, or an empty mapping for cases without a legacy
        observed trace.
    """
    value = case.get("offline_fixture", {})
    return value if isinstance(value, dict) else {}
