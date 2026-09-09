"""DeepEval evaluation of the deterministic Report Node Markdown contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.orchestration.report_node import generate_report
from tests.evals.conftest import EvalSettings

try:
    from deepeval import assert_test
    from deepeval.metrics import (
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        SummarizationMetric,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    FaithfulnessMetric = None  # type: ignore[assignment,misc]
    GEval = None  # type: ignore[assignment,misc]
    HallucinationMetric = None  # type: ignore[assignment,misc]
    SummarizationMetric = None  # type: ignore[assignment,misc]
    LLMTestCase = None  # type: ignore[assignment,misc]
    LLMTestCaseParams = None  # type: ignore[assignment,misc]
    assert_test = None  # type: ignore[assignment]


_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "report_cases.json"
_REPORT_CASES = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
_REPORT_CASE_IDS = [case["case_id"] for case in _REPORT_CASES]

_REPORT_GEVAL_CRITERIA = """
Evaluate the Actual Output as the final Markdown remediation report against the
deterministic evidence in the Context and Expected Output. The report must
preserve the evidence-backed outcome and critical report rules:

1. The summary's successful and follow-up counts must match the effective final
   task statuses. Every successful group must have one successful-remediation
   row, and every outstanding group must have one follow-up action.
2. Follow-up actions must retain the attempted-remediation evidence that is
   present in the context, including package/version attempts and their
   outcomes. Do not promote a worker claim or a diff into a successful fix
   without QA-passed final task evidence.
3. Version-only remediations must report the supported package transition and
   changed manifest files without inventing source-workaround claims. Code
   workaround and package-removal rows must preserve the supported explanation,
   source file paths, and source diff evidence. A strategy pivot must retain
   both the version transition and the later workaround evidence.
4. A transitive finding must keep the vulnerable finding package as the report
   identity while retaining the editable parent/target package as context when
   the evidence distinguishes them.
5. Never invent CVE/GHSA IDs, package names, versions, statuses, metrics,
   changed files, code changes, scan results, or recommendations. Do not turn
   unavailable evidence into a definitive claim.
""".strip()


def _load_golden_cases() -> list[dict[str, Any]]:
    """Load and validate the curated report-evaluation cases."""
    if not isinstance(_REPORT_CASES, list):
        raise TypeError("Report golden data must be a JSON list.")
    if not all(isinstance(case, dict) for case in _REPORT_CASES):
        raise TypeError("Every report golden case must be an object.")
    return _REPORT_CASES


def _group_state(group: dict[str, Any]) -> dict[str, Any]:
    """Convert compact golden group data into a graph-state group."""
    return {
        "group_id": group["id"],
        "vulnerable_component": group["package"],
        "issue_type": "sca",
        "sources": ["odc"],
        "file_path": group.get("file", "package.json"),
        "issues": [
            {
                "package_name": group["package"],
                "package_version": group.get("version", ""),
                "severity": group.get("severity", "high"),
                "source": "odc",
            }
        ],
    }


def _fixture_diff(fixture: dict[str, Any]) -> str:
    """Build a compact unified diff from package and source evidence."""
    blocks: list[str] = []
    for change in fixture.get("package_changes", []):
        path = change["file"]
        name = change["name"]
        old = change.get("old", "")
        new = change.get("new", "")
        if "lock" in path.casefold():
            context = f'     "node_modules/{name}": {{'
            old_line = f'      "version": "{old}"' if old else ""
            new_line = f'      "version": "{new}"' if new else ""
        else:
            section = change.get("section", "dependencies")
            context = f'  "{section}": {{'
            old_line = f'    "{name}": "{old}"' if old else ""
            new_line = f'    "{name}": "{new}"' if new else ""
        lines = [f"--- a/{path}", f"+++ b/{path}", "@@", f" {context}"]
        if old_line:
            lines.append(f"-{old_line}")
        if new_line:
            lines.append(f"+{new_line}")
        blocks.append("\n".join(lines))

    for change in fixture.get("source_changes", []):
        blocks.append(
            "\n".join(
                [
                    f"--- a/{change['file']}",
                    f"+++ b/{change['file']}",
                    "@@",
                    f"-{change['removed']}",
                    f"+{change['added']}",
                ]
            )
        )
    return "\n".join(blocks)


def _build_report_state(case: dict[str, Any]) -> dict[str, Any]:
    """Expand compact golden data into the production report state contract."""
    fixture = case["fixture"]
    groups = [_group_state(group) for group in fixture.get("groups", [])]
    initial_groups = [
        group
        for group, source in zip(groups, fixture.get("groups", []), strict=True)
        if source.get("initial", True)
    ]
    final_groups = [
        group
        for group, source in zip(groups, fixture.get("groups", []), strict=True)
        if source.get("final", False)
    ]

    task_queue: dict[str, dict[str, Any]] = {}
    qa_evaluations: dict[str, dict[str, Any]] = {}
    for task in fixture.get("tasks", []):
        task_id = task["id"]
        record: dict[str, Any] = {
            "task_id": task_id,
            "parent_group_id": task["group"],
            "parent_task_id": task.get("parent_task_id"),
            "strategy": task.get("strategy", "VERSION_BUMP"),
            "status": task["status"],
        }
        for field in (
            "strategy_stage",
            "no_fix_stage",
            "qa_policy",
            "target_package_name",
            "target_dependency_type",
            "parent_package_name",
            "parent_package_version",
            "selected_version",
            "instruction",
        ):
            if field in task:
                record[field] = task[field]
        task_queue[task_id] = record
        if "qa" in task:
            qa_evaluations[task_id] = {"task_id": task_id, "passed": task["qa"]}

    action_summaries: list[dict[str, Any]] = []
    qa_results_by_attempt: dict[str, dict[str, Any]] = {}
    for attempt in fixture.get("attempts", []):
        task = task_queue[attempt["task"]]
        summary: dict[str, Any] = {
            "task_id": attempt["task"],
            "attempt_id": attempt["id"],
            "status": attempt["status"],
            "target_package_name": attempt.get(
                "package",
                task.get("target_package_name", ""),
            ),
            "changed_files": attempt.get("files", []),
        }
        if "selected_version" in attempt:
            summary["selected_version"] = attempt["selected_version"]
        if "instruction" in attempt:
            summary["instruction"] = attempt["instruction"]
        if "summary" in attempt:
            summary["summary"] = attempt["summary"]
        elif "final_note" in attempt:
            summary["summary"] = f"Final note: {attempt['final_note']}"
        else:
            summary["summary"] = "Attempted remediation."
        if "outcome" in attempt:
            summary["final_outcome"] = attempt["outcome"]
        action_summaries.append(summary)
        if "qa" in attempt:
            qa_results_by_attempt[attempt["id"]] = {
                "attempt_id": attempt["id"],
                "task_id": attempt["task"],
                "evaluation": {"passed": attempt["qa"]},
            }

    changed_files = fixture.get("changed_files")
    if changed_files is None:
        changed_files = sorted(
            {
                change["file"]
                for change in fixture.get("package_changes", [])
            }
            | {
                change["file"]
                for change in fixture.get("source_changes", [])
            }
        )
    return {
        "run_id": case["case_id"],
        "repo_root": "data/clones/juice-shop",
        "status": fixture.get("status", "completed"),
        "issues": [
            {"issue_type": "sca", "package_name": package}
            for package in fixture.get("issues", [])
        ],
        "initial_valid_groups": initial_groups,
        "valid_groups": final_groups,
        "task_queue": task_queue,
        "qa_evaluations": qa_evaluations,
        "qa_results_by_attempt": qa_results_by_attempt,
        "action_summaries": action_summaries,
        "diff": _fixture_diff(fixture),
        "changed_files": changed_files,
        "errors": fixture.get("errors", []),
    }


def build_report_prompt(case: dict[str, Any]) -> str:
    """Build the input presented to the report evaluator.

    Args:
        case: Curated report case containing deterministic state and provenance.

    Returns:
        A bounded JSON prompt describing the report evidence and required output.
    """
    evidence = {
        "provenance": case["provenance"],
        "historical_evidence": case.get("historical_evidence", []),
        "report_state": _build_report_state(case),
        "expected_contract": case["expected_contract"],
    }
    return (
        "Review the deterministic remediation evidence and produce the final Markdown report. "
        "Preserve the report structure and critical rules; do not add facts, recalculate "
        "metrics, change final task statuses, or recommend actions.\n\n"
        f"Evidence:\n{json.dumps(evidence, sort_keys=True, default=str)[:30000]}"
    )


def build_report_context(case: dict[str, Any]) -> list[str]:
    """Build structured retrieval context for all four DeepEval metrics.

    Args:
        case: Curated report case containing deterministic state and provenance.

    Returns:
        Evidence documents containing the historical basis and normalized state.
    """
    evidence = {
        "provenance": case["provenance"],
        "historical_evidence": case.get("historical_evidence", []),
        "report_state": _build_report_state(case),
        "expected_contract": case["expected_contract"],
    }
    return [
        "Final task, QA, attempt, and patch evidence:\n"
        + json.dumps(evidence, indent=2, sort_keys=True, default=str),
        case["expected_output"],
    ]


def render_report(case: dict[str, Any]) -> str:
    """Render the production report for one evidence-backed golden case.

    Args:
        case: Curated report case containing a compatible graph state mapping.

    Returns:
        The deterministic Markdown report emitted by generate_report().
    """
    return generate_report(_build_report_state(case))


def validate_report_contract(report: str, contract: dict[str, Any]) -> list[str]:
    """Return fixture-contract violations without adding another eval metric.

    These checks validate that the curated evidence and production renderer are
    wired together before an optional LLM judge runs. They are not persisted as
    a fifth metric and do not replace the four DeepEval metrics.

    Args:
        report: Rendered production Markdown.
        contract: Expected cardinalities and evidence fragments for the case.

    Returns:
        Human-readable contract violations.
    """
    violations: list[str] = []
    summary = contract.get("summary", {})
    if "fixed" in summary:
        expected = f"| Successfully remediated vulnerability groups | {summary['fixed']} |"
        if expected not in report:
            violations.append(f"missing summary fixed count: {expected}")
    if "follow_up" in summary:
        expected = f"| Vulnerability groups requiring follow-up | {summary['follow_up']} |"
        if expected not in report:
            violations.append(f"missing summary follow-up count: {expected}")

    successful = report.split("## 3. Successful Remediations", 1)[-1].split(
        "## 4. References", 1
    )[0]
    follow_up = report.split("## 2. Follow up Actions", 1)[-1].split(
        "## 3. Successful Remediations", 1
    )[0]
    for package in contract.get("successful_packages", []):
        if f"| {package} |" not in successful:
            violations.append(f"missing successful package: {package}")
    for package in contract.get("follow_up_packages", []):
        if f"— {package} (" not in follow_up:
            violations.append(f"missing follow-up package: {package}")

    for fragment in contract.get("required_fragments", []):
        if fragment not in report:
            violations.append(f"missing required fragment: {fragment}")
    for fragment in contract.get("forbidden_fragments", []):
        if fragment in report:
            violations.append(f"found forbidden fragment: {fragment}")

    code_details = report.split("### Code workaround details", 1)[-1]
    for path in contract.get("code_detail_diff_paths", []):
        if f"--- a/{path}" not in code_details:
            violations.append(f"missing code-detail diff path: {path}")
    for path in contract.get("forbidden_code_detail_diff_paths", []):
        if f"--- a/{path}" in code_details:
            violations.append(f"manifest leaked into code-detail diff: {path}")
    return violations


def _build_metrics(eval_settings: EvalSettings) -> list[Any]:
    """Construct exactly the four DeepEval metrics for report evaluation."""
    if not HAS_DEEPEVAL:
        raise RuntimeError("DeepEval is required for live report evaluation.")
    return [
        HallucinationMetric(
            threshold=0.30,
            model=eval_settings.judge_model,
            include_reason=True,
            verbose_mode=True,
        ),
        FaithfulnessMetric(
            threshold=0.85,
            model=eval_settings.judge_model,
            include_reason=True,
            verbose_mode=True,
        ),
        SummarizationMetric(
            threshold=0.80,
            model=eval_settings.judge_model,
            include_reason=True,
            verbose_mode=True,
        ),
        GEval(
            name="Report Constraint Adherence",
            criteria=_REPORT_GEVAL_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.CONTEXT,
            ],
            threshold=0.70,
            model=eval_settings.judge_model,
            verbose_mode=True,
        ),
    ]


@pytest.mark.eval
class TestReportNodeEval:
    """Evaluate the final report against historical, typed evidence."""

    @pytest.mark.parametrize("case", _load_golden_cases(), ids=_REPORT_CASE_IDS)
    def test_report_uses_only_the_four_requested_metrics(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Run Hallucination, Faithfulness, Summarization, and one GEval."""
        report = render_report(case)
        violations = validate_report_contract(report, case["expected_contract"])
        assert not violations, f"Case {case['case_id']} violated its report contract: {violations}"

        if not eval_settings.is_live:
            return
        if not HAS_DEEPEVAL or assert_test is None or LLMTestCase is None:
            pytest.skip("deepeval package is required for live evaluations")
        if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

        context = build_report_context(case)
        test_case = LLMTestCase(
            name=f"{case['case_id']} [Report Metrics]",
            input=build_report_prompt(case),
            actual_output=report,
            expected_output=case["expected_output"],
            context=context,
            retrieval_context=context,
            additional_metadata={
                "case_id": case["case_id"],
                "provenance": case["provenance"],
                "fixture_type": case.get("fixture_type", "historical"),
            },
        )
        assert_test(test_case, _build_metrics(eval_settings))


@pytest.mark.parametrize("case", _load_golden_cases(), ids=_REPORT_CASE_IDS)
def test_report_goldens_use_current_evidence_contract(case: dict[str, Any]) -> None:
    """Ensure the replacement dataset no longer carries narrative-era fields."""
    assert "generated_narrative" not in case
    assert "evidence_payload" not in case
    assert isinstance(case.get("fixture"), dict)
    assert isinstance(case.get("expected_output"), str)
    assert isinstance(case.get("expected_contract"), dict)
