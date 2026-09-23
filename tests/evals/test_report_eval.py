"""DeepEval evaluation of the deterministic Report Node Markdown contract."""

from __future__ import annotations

import json
import os
import re
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

_REPORT_FORMAT_VERSION = "remediation_report_v2"
_REPORT_SECTIONS = (
    "## 1. Summary",
    "## 2. Follow up Actions",
    "## 3. Successful Remediations",
    "## 4. References",
)


def _report_format_evidence() -> dict[str, Any]:
    """Describe the canonical report shape supplied to live judges.

    The report renderer is deterministic Markdown rather than a free-form
    narrative. Keeping this description beside the eval adapter prevents the
    judge prompt from silently reverting to the older narrative contract.
    """
    return {
        "version": _REPORT_FORMAT_VERSION,
        "sections": list(_REPORT_SECTIONS),
        "finding_identity": (
            "Use the scanner CVE/GHSA/finding identifier in the report. The group id is "
            "internal state and is only a fallback when no scanner identifier exists."
        ),
        "code_detail_scope": (
            "Successful-remediation rows may list manifest and source files. Diff blocks "
            "under Code workaround details must contain only the source files belonging "
            "to the accepted workaround; package-removal attempts are the exception when "
            "the contract explicitly permits manifest removal diffs."
        ),
        "evidence_policy": (
            "Counts, statuses, attempts, package transitions, file paths, and references "
            "must come from the typed report state."
        ),
    }


_REPORT_GEVAL_CRITERIA = """
Evaluate the Actual Output as a canonical remediation_report_v2 Markdown
document against the deterministic evidence in the Context and Expected
Output. Do not judge it as a one-paragraph narrative. The report must preserve
the evidence-backed outcome and critical report rules:

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
   source file paths, and source diff evidence. Manifest files may appear in a
   row's Files Changed cell; only the diff blocks under Code workaround details
   are source-scoped. A strategy pivot must retain both the version transition
   and the later workaround evidence.
4. A transitive finding must keep the vulnerable finding package as the report
   identity while retaining the editable parent/target package as context when
   the evidence distinguishes them.
5. The four numbered report sections, scanner finding identifiers, statuses,
   attempts, and References table must follow the current report format.
6. Never invent CVE/GHSA IDs, package names, versions, statuses, metrics,
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
    result: dict[str, Any] = {
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
    for field_name in (
        "cve_ids",
        "ghsa_ids",
        "finding_ids",
        "rule_ids",
        "parent_contexts",
        "representative_issue_id",
    ):
        if field_name in group:
            result[field_name] = group[field_name]
    issue_overrides = group.get("issues")
    if isinstance(issue_overrides, list) and issue_overrides:
        result["issues"] = issue_overrides
    return result


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
            {change["file"] for change in fixture.get("package_changes", [])}
            | {change["file"] for change in fixture.get("source_changes", [])}
        )
    return {
        "run_id": case["case_id"],
        "repo_root": "data/clones/juice-shop",
        "status": fixture.get("status", "completed"),
        "issues": [
            {"issue_type": "sca", "package_name": package} for package in fixture.get("issues", [])
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
        "report_format": _report_format_evidence(),
    }
    return (
        "Review the deterministic remediation evidence against the canonical "
        f"{_REPORT_FORMAT_VERSION} Markdown report contract. Preserve the four report "
        "sections, use scanner finding identifiers, and do not add facts, recalculate "
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
        "report_format": _report_format_evidence(),
    }
    return [
        "Final task, QA, attempt, and patch evidence:\n"
        + json.dumps(evidence, indent=2, sort_keys=True, default=str),
        build_report_expected_output(case),
    ]


def render_report(case: dict[str, Any]) -> str:
    """Render the production report for one evidence-backed golden case.

    Args:
        case: Curated report case containing a compatible graph state mapping.

    Returns:
        The deterministic Markdown report emitted by generate_report().
    """
    return generate_report(_build_report_state(case))


def _split_table_row(line: str) -> list[str]:
    """Split one Markdown table row while preserving escaped pipes."""
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|") and not text.endswith("\\|"):
        text = text[:-1]
    cells = re.split(r"(?<!\\)\|", text)
    return [cell.replace("\\|", "|").replace("\\\\", "\\").strip() for cell in cells]


def _table_rows(section: str, first_header: str) -> list[dict[str, str]]:
    """Parse rows from the first table containing ``first_header``."""
    lines = section.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("|") and first_header in _split_table_row(line)
        ),
        None,
    )
    if header_index is None:
        return []
    headers = _split_table_row(lines[header_index])
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 1 :]:
        if not line.lstrip().startswith("|"):
            if rows:
                break
            continue
        cells = _split_table_row(line)
        if not cells or all(re.fullmatch(r"-+", cell or "") for cell in cells):
            continue
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        rows.append(dict(zip(headers, cells, strict=False)))
    return rows


def _report_sections(report: str) -> dict[str, str]:
    """Return numbered report sections keyed by their exact heading."""
    matches = list(re.finditer(r"(?m)^## \d+\. .+$", report))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(report)
        heading = match.group(0).strip()
        sections[heading] = report[match.end() : end].strip()
    return sections


def _integer_metric(summary_rows: list[dict[str, str]], name: str) -> int | None:
    """Read an integer value from the report summary table."""
    for row in summary_rows:
        if row.get("Metric") != name:
            continue
        match = re.search(r"\d+", row.get("Value", ""))
        return int(match.group(0)) if match else None
    return None


def _parse_follow_up_records(section: str) -> list[dict[str, Any]]:
    """Parse current-format follow-up headings, statuses, and attempts."""
    heading_re = re.compile(
        r"(?ms)^### (?P<finding>.+?) — (?P<package>.+?) \((?P<severity>[^)\n]+)\)\n"
        r"(?P<body>.*?)(?=^### |\Z)"
    )
    attempt_re = re.compile(
        r"(?ms)^\s*\d+\. \*\*Attempt (?P<number>\d+) \((?P<label>.+?)\):\*\*"
        r"(?P<body>.*?)(?=^\s*\d+\. \*\*Attempt |\Z)"
    )
    records: list[dict[str, Any]] = []
    for match in heading_re.finditer(section):
        body = match.group("body")
        status_match = re.search(r"(?m)^\s*- \*\*Status:\*\*\s*(?P<value>.+)$", body)
        recommendation_match = re.search(
            r"(?m)^\s*- \*\*Recommended action:\*\*\s*(?P<value>.+)$", body
        )
        attempts = [
            {
                "number": int(attempt.group("number")),
                "label": attempt.group("label").strip(),
                "body": attempt.group("body").strip(),
            }
            for attempt in attempt_re.finditer(body)
        ]
        records.append(
            {
                "finding": match.group("finding").strip(),
                "package": match.group("package").strip(),
                "severity": match.group("severity").strip(),
                "status": status_match.group("value").strip() if status_match else "",
                "recommended_action": (
                    recommendation_match.group("value").strip() if recommendation_match else ""
                ),
                "attempts": attempts,
                "body": body.strip(),
            }
        )
    return records


def parse_report(report: str) -> dict[str, Any]:
    """Parse the canonical remediation report into an eval-friendly model.

    Args:
        report: Markdown generated by ``generate_report``.

    Returns:
        Structured sections, summary metrics, remediation rows, follow-up
        records, code-detail diff paths, and references.
    """
    sections = _report_sections(report)
    summary = sections.get("## 1. Summary", "")
    follow_up = sections.get("## 2. Follow up Actions", "")
    successful = sections.get("## 3. Successful Remediations", "")
    references = sections.get("## 4. References", "")
    summary_rows = _table_rows(summary, "Metric")
    success_rows = _table_rows(successful, "Finding")
    code_detail_match = re.search(r"(?ms)^### Code workaround details\n(?P<body>.*)$", successful)
    code_details = code_detail_match.group("body") if code_detail_match else ""
    code_diff_paths = sorted(set(re.findall(r"(?m)^--- a/(?P<path>[^\n]+)$", code_details)))
    reference_rows = _table_rows(references, "Artifact")
    return {
        "sections": sections,
        "summary_rows": summary_rows,
        "summary": {
            "total_findings": _integer_metric(summary_rows, "Total findings (CVEs and GHSAs)"),
            "total_groups": _integer_metric(summary_rows, "Total vulnerability groups"),
            "fixed": _integer_metric(summary_rows, "Successfully remediated vulnerability groups"),
            "follow_up": _integer_metric(summary_rows, "Vulnerability groups requiring follow-up"),
        },
        "follow_up": _parse_follow_up_records(follow_up),
        "successful_rows": success_rows,
        "code_detail_diff_paths": code_diff_paths,
        "references": {row.get("Artifact", ""): row.get("Reference", "") for row in reference_rows},
    }


def build_report_expected_output(case: dict[str, Any]) -> str:
    """Build structured expected output for the report judges.

    The renderer emits a complete Markdown document, so a one-sentence
    narrative is not an appropriate expected output. This compact contract
    keeps the judge focused on report facts and format semantics.
    """
    contract = case["expected_contract"]
    fixture = case["fixture"]
    expected = {
        "format": _REPORT_FORMAT_VERSION,
        "summary": {
            **contract.get("summary", {}),
            "total_findings": contract.get("total_findings", len(fixture.get("issues", []))),
        },
        "finding_identifiers": contract.get("finding_identifiers", []),
        "successful_packages": contract.get("successful_packages", []),
        "follow_up_packages": contract.get("follow_up_packages", []),
        "code_detail_diff_paths": contract.get("code_detail_diff_paths", []),
        "forbidden_code_detail_diff_paths": contract.get("forbidden_code_detail_diff_paths", []),
    }
    return json.dumps(expected, sort_keys=True)


def _summary_output_for_eval(parsed: dict[str, Any]) -> str:
    """Return only the summary facts for the summarization metric."""
    return json.dumps(parsed["summary"], sort_keys=True)


def _expected_summary_for_eval(case: dict[str, Any]) -> str:
    """Return the structured summary expected by the summarization metric."""
    contract = case["expected_contract"]
    return json.dumps(
        {
            **contract.get("summary", {}),
            "total_findings": contract.get(
                "total_findings",
                len(case["fixture"].get("issues", [])),
            ),
        },
        sort_keys=True,
    )


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
    parsed = parse_report(report)
    missing_sections = [
        section for section in _REPORT_SECTIONS if section not in parsed["sections"]
    ]
    violations.extend(f"missing report section: {section}" for section in missing_sections)
    if not parsed["references"]:
        violations.append("references table is missing or empty")

    summary = contract.get("summary", {})
    for key in ("fixed", "follow_up", "total_groups"):
        if key not in summary:
            continue
        actual = parsed["summary"].get(key)
        if actual != summary[key]:
            violations.append(f"summary {key} was {actual!r}, expected {summary[key]!r}")
    expected_total_findings = contract.get("total_findings")
    if expected_total_findings is not None:
        actual_total_findings = parsed["summary"].get("total_findings")
        if actual_total_findings != expected_total_findings:
            violations.append(
                "summary total_findings was "
                f"{actual_total_findings!r}, expected {expected_total_findings!r}"
            )

    successful_packages = {row.get("Package / Target", "") for row in parsed["successful_rows"]}
    follow_up_packages = {record["package"] for record in parsed["follow_up"]}
    for package in contract.get("successful_packages", []):
        if package not in successful_packages:
            violations.append(f"missing successful package: {package}")
    for package in contract.get("follow_up_packages", []):
        if package not in follow_up_packages:
            violations.append(f"missing follow-up package: {package}")

    identifiers = {
        identifier
        for row in parsed["successful_rows"]
        for identifier in row.get("Finding", "").split(", ")
    }
    identifiers.update(
        identifier for record in parsed["follow_up"] for identifier in record["finding"].split(", ")
    )
    for identifier in contract.get("finding_identifiers", []):
        if identifier not in identifiers:
            violations.append(f"missing finding identifier: {identifier}")
    if not contract.get("allow_synthetic_finding_ids", False):
        synthetic = sorted(identifier for identifier in identifiers if identifier.startswith("g-"))
        violations.extend(
            f"synthetic finding identifier is not allowed: {identifier}" for identifier in synthetic
        )

    for fragment in contract.get("required_fragments", []):
        if fragment not in report:
            violations.append(f"missing required fragment: {fragment}")
    for fragment in contract.get("forbidden_fragments", []):
        if fragment in report:
            violations.append(f"found forbidden fragment: {fragment}")

    for path in contract.get("code_detail_diff_paths", []):
        if path not in parsed["code_detail_diff_paths"]:
            violations.append(f"missing code-detail diff path: {path}")
    for path in contract.get("forbidden_code_detail_diff_paths", []):
        if path in parsed["code_detail_diff_paths"]:
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


def _build_metric_test_cases(
    case: dict[str, Any],
    report: str,
    context: list[str],
) -> tuple[Any, Any]:
    """Build full-report and summary-only test cases for live metrics."""
    metadata = {
        "case_id": case["case_id"],
        "provenance": case["provenance"],
        "fixture_type": case.get("fixture_type", "historical"),
        "report_format": _REPORT_FORMAT_VERSION,
    }
    full_report_case = LLMTestCase(
        name=f"{case['case_id']} [Report Metrics]",
        input=build_report_prompt(case),
        actual_output=report,
        expected_output=build_report_expected_output(case),
        context=context,
        retrieval_context=context,
        additional_metadata=metadata,
    )
    summary_case = LLMTestCase(
        name=f"{case['case_id']} [Report Summary]",
        input=(
            build_report_prompt(case)
            + "\n\nFor this metric only, evaluate the JSON summary facts rather than the full Markdown document."
        ),
        actual_output=_summary_output_for_eval(parse_report(report)),
        expected_output=_expected_summary_for_eval(case),
        context=context,
        retrieval_context=context,
        additional_metadata=metadata,
    )
    return full_report_case, summary_case


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
        full_report_case, summary_case = _build_metric_test_cases(case, report, context)
        for metric in _build_metrics(eval_settings):
            metric_case = (
                summary_case
                if metric.__class__.__name__.casefold() == "summarizationmetric"
                else full_report_case
            )
            assert_test(metric_case, [metric])


@pytest.mark.parametrize("case", _load_golden_cases(), ids=_REPORT_CASE_IDS)
def test_report_goldens_use_current_evidence_contract(case: dict[str, Any]) -> None:
    """Ensure the dataset carries the current structured report contract."""
    assert "generated_narrative" not in case
    assert "evidence_payload" not in case
    assert isinstance(case.get("fixture"), dict)
    assert isinstance(case.get("expected_output"), str)
    assert isinstance(case.get("expected_contract"), dict)
    contract = case["expected_contract"]
    assert "summary" in contract
    assert "finding_identifiers" in contract
    assert contract.get("allow_synthetic_finding_ids") is not True
    assert json.loads(case["expected_output"]) == json.loads(build_report_expected_output(case))
