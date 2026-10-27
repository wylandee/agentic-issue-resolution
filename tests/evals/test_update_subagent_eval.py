"""DeepEval evaluation suite for the update subagent."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.contracts.schemas import AgentActionStatus
from remediation_engine.language import ProjectLanguage
from tests.evals.conftest import EvalSettings
from tests.evals.eval_case_helpers import (
    as_output_text,
    case_metadata,
    collect_metric_assertions,
    context_strings,
    expected_tools,
    make_tool_calls,
    measure_expected,
    observations,
)
from tests.evals.golden_schema import load_golden_dataset
from tests.evals.replay_adapters import replay_update_case
from tests.evals.replay_harness import (
    ReplayCapture,
    ScriptedReplayModel,
    _case_project_language,
    cached_replay,
    format_tool_trace,
)

try:
    from deepeval import assert_test
    from deepeval.metrics import TaskCompletionMetric as DeepEvalTaskCompletionMetric
    from deepeval.metrics import ToolCorrectnessMetric as DeepEvalToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCallParams

    HAS_DEEPEVAL = True
except ImportError:
    from remediation_engine.evals.trajectories import (
        DeepEvalLLMTestCase as LLMTestCase,  # type: ignore[assignment]
    )

    HAS_DEEPEVAL = False
    assert_test = None  # type: ignore[assignment]
    DeepEvalTaskCompletionMetric = None  # type: ignore[assignment,misc]
    DeepEvalToolCorrectnessMetric = None  # type: ignore[assignment,misc]
    ToolCallParams = None  # type: ignore[assignment,misc]


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "update_subagent_cases.json"
_UPDATE_CASES = [
    case
    for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="update_subagent_cases")
    if case.get("eval_type") == "update_subagent"
]
_UPDATE_CASE_IDS = [case["case_id"] for case in _UPDATE_CASES]
_LIVE_CACHE: dict[tuple[str, str], ReplayCapture] = {}


def _make_tool_calls(raw_calls: list[dict[str, Any]]) -> list[Any]:
    """Convert canonical update tool records."""
    return make_tool_calls(raw_calls, component="update")


def _format_tool_trace(raw_calls: list[dict[str, Any]]) -> str:
    """Render captured update tools."""
    return format_tool_trace(raw_calls)


def build_update_test_case(
    case: dict[str, Any],
    observed_output: Any | None = None,
    observed_tools: list[dict[str, Any]] | None = None,
    *,
    replay_source: str | None = None,
    capture: ReplayCapture | None = None,
) -> LLMTestCase:
    """Construct an update test case from explicit observations."""
    if observed_output is None and observed_tools is None:
        final_output, tool_trace, source = observations(case)
    else:
        final_output = as_output_text(observed_output or "")
        tool_trace = list(observed_tools or [])
        source = replay_source or "production_live"
    target_package = str(case.get("target_package_name", ""))
    metadata = case_metadata(
        case,
        component="update_subagent",
        replay_source=source,
        actual_tools=tool_trace,
        capture=capture,
    )
    metadata.update(
        {
            "target_package_name": target_package,
            "selected_version": case.get("selected_version"),
            "dependency_type": case.get("dependency_type"),
            "action_status": case.get("action_status", "APPLIED"),
            "attempt_id": case.get("attempt_id"),
            "task_revision": case.get("task_revision"),
        }
    )
    return LLMTestCase(
        name=f"{case['case_id']} [Update Subagent]",
        input=str(case["input"]),
        actual_output=f"{final_output}\n\n{_format_tool_trace(tool_trace)}",
        expected_output=str(case["expected_output"]),
        context=context_strings(
            case,
            f"Supervisor instruction: {case.get('supervisor_instruction', '')}",
            f"Repository map: {case.get('repository_map', '(workspace map unavailable)')}",
        ),
        tools_called=_make_tool_calls(tool_trace),
        expected_tools=expected_tools(case, component="update"),
        token_cost=capture.token_cost if capture is not None else None,
        additional_metadata=metadata,
    )


def _measure_expected(
    metric: Any,
    test_case: Any,
    *,
    expected_pass: bool,
    case_id: str,
    label: str,
) -> None:
    """Measure and register an expected-positive or expected-negative metric."""
    measure_expected(
        metric,
        test_case,
        expected_pass=expected_pass,
        case_id=case_id,
        label=label,
        register=assert_test,
    )


def _live_capture(case: dict[str, Any], settings: EvalSettings) -> ReplayCapture:
    """Run one real update worker replay per process."""
    return cached_replay(
        _LIVE_CACHE,
        "update_subagent",
        str(case["case_id"]),
        lambda: replay_update_case(case, settings),
    )


@pytest.mark.eval
class TestUpdateSubagentEval:
    """Evaluate update output and worker tool traces."""

    @pytest.mark.parametrize("case", _UPDATE_CASES or [{}], ids=_UPDATE_CASE_IDS or ["no_cases"])
    def test_update_subagent_live_replay_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run the production worker and judge its captured result."""
        if not case:
            pytest.skip("No golden update cases available")
        if not eval_settings.is_live:
            pytest.skip("Live replay requires --run-eval-live.")
        if not HAS_DEEPEVAL:
            pytest.skip("DeepEval is not installed.")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY is required for live evaluations.")

        capture = _live_capture(case, eval_settings)
        test_case = build_update_test_case(
            case,
            capture.actual_output,
            capture.actual_tools,
            replay_source="production_live",
            capture=capture,
        )
        with collect_metric_assertions():
            if DeepEvalToolCorrectnessMetric is not None:
                tool_metric = DeepEvalToolCorrectnessMetric(
                    threshold=1.0,
                    evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
                    should_consider_ordering=True,
                    should_exact_match=True,
                )
                _measure_expected(
                    tool_metric,
                    test_case,
                    expected_pass=bool(case.get("expected_tool_correctness_pass", True)),
                    case_id=str(case["case_id"]),
                    label="update tool correctness",
                )
            if DeepEvalTaskCompletionMetric is not None:
                task_metric = DeepEvalTaskCompletionMetric(
                    threshold=0.70,
                    model=eval_settings.judge_llm or eval_settings.judge_model,
                    async_mode=False,
                )
                _measure_expected(
                    task_metric,
                    test_case,
                    expected_pass=bool(case.get("expected_completion_pass", True)),
                    case_id=str(case["case_id"]),
                    label="update task completion",
                )


def _tool_signature(tool_events: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return tool names and arguments without fixture output text."""
    return [
        (str(event.get("name", "")), dict(event.get("args", {}) or {})) for event in tool_events
    ]


@pytest.mark.parametrize("case", _UPDATE_CASES or [{}], ids=_UPDATE_CASE_IDS or ["no_cases"])
def test_update_subagent_offline_production_replay(
    case: dict[str, Any],
    eval_settings: EvalSettings,
) -> None:
    """Exercise every update case through the real worker and bounded loop."""
    if not case:
        pytest.skip("No golden update cases available")

    fixture = case.get("offline_fixture", {})
    scripted_tools = fixture.get("actual_tools", []) if isinstance(fixture, dict) else []
    assert isinstance(scripted_tools, list)
    model = ScriptedReplayModel.from_tool_trace(scripted_tools, final_text="")
    capture = replay_update_case(case, eval_settings, llm=model)

    assert model.invocation_count == len(scripted_tools) + 1
    assert _tool_signature(capture.actual_tools) == _tool_signature(scripted_tools)
    assert capture.attempt_id
    assert capture.task_revision == int(case.get("task_revision") or 1)
    assert all(call.get("kind") == "sandbox_command" for call in capture.external_calls)

    case_id = str(case["case_id"])
    summary = capture.typed_result["action_summaries"][0]
    if case.get("language") == "java":
        assert "pom.xml" in case["replay"]["input"]["workspace_files"]
        assert "package.json" not in case["replay"]["input"]["workspace_files"]
        assert "package.json" not in capture.final_files
        expected_maven_tool = (
            "modify_batch_maven_dependencies"
            if case_id == "update-maven-atomic-logback"
            else "modify_and_validate_maven_dependency"
        )
        assert expected_maven_tool in model.bound_tool_names
        assert all("registry" not in name.casefold() for name in model.bound_tool_names)
        assert all(
            "registry" not in str(event.get("name", "")).casefold()
            for event in capture.actual_tools
        )
        assert all(
            "registry" not in str(call.get("command", "")).casefold()
            for call in capture.external_calls
        )
        if case_id == "update-maven-sync-rollback":
            assert summary.status != AgentActionStatus.SUCCESS
            assert not capture.changed_files
            for path, original in case["replay"]["input"]["workspace_files"].items():
                assert capture.final_files[path] == original
            assert "MANIFEST_SYNC_FAILED" in (
                capture.actual_output + "\n" + "\n".join(capture.errors)
            )
            return

        assert summary.status == AgentActionStatus.SUCCESS
        assert capture.changed_files
        if case_id == "update-maven-atomic-logback":
            assert len(capture.typed_result["action_summaries"]) == 2
            assert all(
                item.status == AgentActionStatus.SUCCESS
                for item in capture.typed_result["action_summaries"]
            )
            results = capture.typed_result["worker_results_by_attempt"]
            assert set(results) == {"attempt-classic-001", "attempt-core-001"}
            action = case["replay"]["input"]["multi_package_action"]
            assert {result.cluster_id for result in results.values()} == {action["cluster_id"]}
            assert {result.dispatch_batch_id for result in results.values()} == {
                action["dispatch_batch_id"]
            }
            action_digests = {result.action_digest for result in results.values()}
            assert len(action_digests) == 1
            assert None not in action_digests
            snapshots = case["replay"]["input"]["attempt_snapshots"]
            assert {(snapshot["task_id"], snapshot["attempt_id"]) for snapshot in snapshots} == {
                ("update-maven-atomic-logback-classic", "attempt-classic-001"),
                ("update-maven-atomic-logback-core", "attempt-core-001"),
            }
            assert ET.fromstring(capture.final_files["pom.xml"]).findall(
                ".//{*}dependencyManagement/{*}dependencies/{*}dependency"
            )
            assert capture.final_files["pom.xml"].count("<version>1.2.13</version>") == 2
            return

        if case_id == "update-maven-property-owner":
            parent_xml = ET.fromstring(capture.final_files["pom.xml"])
            child_xml = ET.fromstring(capture.final_files["service/pom.xml"])
            assert parent_xml.findtext(".//{*}postgresql.version") == "42.7.11"
            assert parent_xml.findtext(".//{*}unrelated.property") == "leave-me"
            target = child_xml.find(".//{*}dependency")
            assert target is not None
            assert target.findtext("{*}version") == "${postgresql.version}"
            return

        root = ET.fromstring(capture.final_files[case["manifest_path"]])
        gav = str(case["target_package_name"])
        group_id, artifact_id = gav.split(":", 1)
        matches = [
            node
            for node in root.findall(".//{*}dependency")
            if node.findtext("{*}groupId") == group_id
            and node.findtext("{*}artifactId") == artifact_id
        ]
        assert matches
        assert any(node.findtext("{*}version") == case["selected_version"] for node in matches)
        if case_id == "update-maven-target-recovery":
            assert len(capture.actual_tools) == 3
            outputs = [str(event.get("output", "")) for event in capture.actual_tools]
            assert all("TARGET_NOT_ALLOWED" in output for output in outputs[:2])
            assert outputs[2].startswith("SUCCESS:")
        if case_id == "update-maven-managed-insertion":
            assert "<dependencyManagement>" in capture.final_files["pom.xml"]
        return

    if case_id == "update-retry-limit-surrender" or case.get("action_status") == "SURRENDER":
        assert summary.status == AgentActionStatus.SURRENDER
        if case_id == "update-retry-limit-surrender" or case.get("expected_failure_marker"):
            marker = str(case.get("expected_failure_marker", "RETRY_LIMIT_REACHED"))
            assert marker in (capture.actual_output + "\n" + "\n".join(capture.errors))
        if _case_project_language(case) == ProjectLanguage.PYTHON:
            manifest_path = str(case.get("manifest_path", "requirements.txt"))
            assert (
                capture.final_files[manifest_path]
                == case["replay"]["input"]["workspace_files"][manifest_path]
            )
        else:
            baseline = json.loads(str(case["replay"]["input"]["workspace_files"]["package.json"]))
            final_manifest = json.loads(capture.final_files["package.json"])
            assert final_manifest == baseline
        return

    assert summary.status == AgentActionStatus.SUCCESS
    assert capture.changed_files
    package_name = str(case["target_package_name"])
    if _case_project_language(case) == ProjectLanguage.PYTHON:
        manifest_path = str(case.get("manifest_path", "requirements.txt"))
        requirements = capture.final_files[manifest_path].splitlines()
        assert f"{package_name}=={case['selected_version']}" in requirements
        assert "package.json" not in capture.final_files
    else:
        manifest = json.loads(capture.final_files["package.json"])
        dependency_section = str(case["dependency_type"])
        assert manifest[dependency_section][package_name] == case["selected_version"]


def test_maven_target_recovery_separates_live_and_offline_tool_contracts() -> None:
    """Keep adversarial guardrail calls in the scripted trace, not live expectations."""
    case = next(case for case in _UPDATE_CASES if case["case_id"] == "update-maven-target-recovery")
    expected_live_tools = expected_tools(case, component="update")
    scripted_tools = case["offline_fixture"]["actual_tools"]

    assert len(expected_live_tools) == 1
    assert expected_live_tools[0].name == "modify_and_validate_maven_dependency"
    assert expected_live_tools[0].input_parameters == {
        "package_name": "org.postgresql:postgresql",
        "target_version": "42.7.11",
        "dependency_type": "dependencies",
        "manifest_path": "pom.xml",
    }
    assert len(scripted_tools) == 3
    assert scripted_tools[0]["args"]["package_name"] == "org.example:wrong"
    assert scripted_tools[1]["args"]["manifest_path"] == "module/pom.xml"
