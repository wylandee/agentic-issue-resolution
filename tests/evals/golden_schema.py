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

REQUIRED_CASE_FIELDS = ("case_id", "input", "context", "expected_output", "expected_tools")
LIVE_REPLAY_EVAL_TYPES = frozenset(
    {"triage", "qa_critic", "update_subagent", "workaround_subagent"}
)
GOLDEN_DATASET_NAMES = (
    "triage_cases",
    "qa_cases",
    "update_subagent_cases",
    "workaround_subagent_cases",
    "report_cases",
    "fix_planner_cases",
)


class GoldenSchemaError(ValueError):
    """Raised when a golden dataset violates the shared case contract."""


def _case_label(dataset_name: str, case: dict[str, Any], index: int) -> str:
    """Return a useful location label for a case error."""
    return f"{dataset_name}[{index}] ({case.get('case_id', '<missing case_id>')!r})"


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
