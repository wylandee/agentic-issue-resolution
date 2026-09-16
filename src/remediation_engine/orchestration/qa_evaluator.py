"""Structured QA evaluator prompts, tools, and investigations."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool, tool

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    FailureCategory,
    QACriticLLMOutput,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    RoutingStrategy,
    ScannerExecutionStatus,
    ScratchpadScope,
    SecurityReviewVerdict,
    TestAttributionVerdict,
    VulnerabilityGroup,
)
from remediation_engine.orchestration.context_manager import ContextManager
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

from ._qa_runtime import (
    _bounded_qa_action_summary,
    _generate_workspace_diff,
    _relevant_action_summaries,
    _review_ready_error,
    group_target_identifiers,
)
from .qa_policy_engine import _qa_policy_prompt_block
from .qa_types import (
    QATaskContext,
    _QAExecutionResults,
    _QALogRecord,
    _scan_result_value,
    _validate_qa_path,
)
from .task_utils import select_package_fix_plan

logger = logging.getLogger(__name__)


_FILE_READ_MAX_CHARS = 8_000
_LOG_QUERY_MAX_CHARS = 6_000
_QA_ACTION_SUMMARY_MAX_CHARS = 1_200
_STRUCTURED_REVIEW_VERDICT_RE = re.compile(
    r"^\s*[*_#` -]*(?:structured\s+)?"
    r"(?:(?:semantic\s+security\s+)?review\s+)?verdict\s*[:\-]\s*"
    r"[*_#` -]*(PASS|FAIL|INCONCLUSIVE)\b",
    re.IGNORECASE,
)


def _parse_structured_review_verdict(text: str) -> SecurityReviewVerdict | None:
    """Extract an explicit semantic-review verdict marker from evaluator text."""
    for line in (text or "").splitlines():
        match = _STRUCTURED_REVIEW_VERDICT_RE.match(line)
        if match is not None:
            try:
                return SecurityReviewVerdict(match.group(1).lower())
            except ValueError:
                return None
    return None


def _has_successful_review_tool_evidence(name: str, content: str) -> bool:
    """Return whether a source/diff review tool returned usable evidence."""
    if name not in {
        "generate_workspace_diff",
        "read_file_context",
        "search_codebase_pattern",
        "inspect_ast_symbol",
    }:
        return False
    normalized = (content or "").strip().upper()
    return bool(normalized) and not normalized.startswith(
        ("ERROR:", "FAILURE:", "BLOCKED:", "NOT_APPLICABLE:")
    )


@dataclass
class GroupInvestigation:
    """Structured evaluator output and review provenance for one task."""

    group_id: str
    investigation_text: str
    tool_transcript: str
    task_id: str = ""
    errors: list[str] = field(default_factory=list)
    review_tools_used: list[str] = field(default_factory=list)
    source_review_evidence: bool = False
    fallback: bool = False
    structured_review_verdict: SecurityReviewVerdict | None = None
    evaluation: QAEvaluation | None = None


_QA_LOG_DEFAULT_TAIL_LINES = {"install": 80, "scan": 30, "tests": 60}
_QA_LOG_VIEWS = {"summary", "tail", "errors", "full", "filter"}


def _qa_log_records_for_phase(
    results: _QAExecutionResults,
    log_type: str,
) -> tuple[_QALogRecord, ...]:
    """Return captured records for one deterministic QA phase.

    The execution cache normally contains immutable records. The scalar phase
    projections are materialized only when a caller provides a partially
    populated result object.
    """
    records = results.log_records.get(log_type, ())
    if records:
        return records
    if log_type == "install" and results.install is not None:
        return (
            _QALogRecord(
                phase=log_type,
                label="npm install",
                exit_code=results.install_exit_code,
                stdout=results.install_raw_stdout or results.install[1],
                stderr=results.install_raw_stderr or "",
                error=results.install_error_category,
            ),
        )
    if log_type == "scan":
        if results.scan_skipped:
            return (
                _QALogRecord(
                    phase=log_type,
                    label="odc:skipped",
                    exit_code=None,
                    stdout="",
                    stderr="",
                    error=results.scan_skip_reason or "scan skipped",
                ),
            )
        if results.scan is not None:
            return (
                _QALogRecord(
                    phase=log_type,
                    label="odc:full",
                    exit_code=getattr(results.scan, "exit_code", None),
                    stdout=getattr(results.scan, "raw_stdout", None)
                    or _scan_result_value(results.scan, "summary", "scan completed"),
                    stderr=getattr(results.scan, "raw_stderr", None) or "",
                    error=getattr(results.scan, "diagnostic_log_path", None),
                ),
            )
    if log_type == "tests" and results.tests is not None:
        return (
            _QALogRecord(
                phase=log_type,
                label="npm test",
                exit_code=results.test_exit_code,
                stdout=results.test_raw_stdout or results.tests[1],
                stderr=results.test_raw_stderr or "",
                error=None,
            ),
        )
    return ()


def _qa_log_record_text(record: _QALogRecord) -> str:
    """Render one private log record with stream and lifecycle metadata."""
    lines = [
        f"=== {record.label} (exit_code={record.exit_code}) ===",
    ]
    if record.stdout:
        lines.extend([f"[{record.label} stdout]", record.stdout.rstrip()])
    if record.stderr:
        lines.extend([f"[{record.label} stderr]", record.stderr.rstrip()])
    if record.error:
        lines.extend([f"[error] {record.error}"])
    return "\n".join(lines)


def _qa_log_summary(results: _QAExecutionResults, log_type: str) -> str:
    """Return a bounded metadata-rich summary for one QA phase."""
    if log_type == "install":
        if results.install is None:
            return "ERROR: run_dependency_install has not been called yet."
        details = [results.install[1]]
        if results.install_exit_code is not None:
            details.append(f"exit_code={results.install_exit_code}")
        if results.install_error_category:
            details.append(f"error_category={results.install_error_category}")
        return "\n".join(details)
    if log_type == "scan":
        if results.scan_skipped:
            return f"[SKIPPED] {results.scan_skip_reason or 'scan skipped'}"
        if results.scan is None:
            return "ERROR: run_security_scan has not been called yet."
        details = [_scan_result_value(results.scan, "summary", "scan completed")]
        execution_status = getattr(results.scan, "execution_status", None)
        if execution_status:
            details.append(f"execution_status={execution_status}")
        exit_code = getattr(results.scan, "exit_code", None)
        if exit_code is not None:
            details.append(f"exit_code={exit_code}")
        diagnostic_log_path = getattr(results.scan, "diagnostic_log_path", None)
        if diagnostic_log_path:
            details.append(f"diagnostic_log_path={diagnostic_log_path}")
        return "\n".join(details)
    if log_type == "tests":
        if results.tests is None:
            return "ERROR: run_unit_tests has not been called yet."
        details = [results.tests[1]]
        if results.test_exit_code is not None:
            details.append(f"exit_code={results.test_exit_code}")
        if results.test_failure_count is not None:
            details.append(f"failure_count={results.test_failure_count}")
        return "\n".join(details)
    return "ERROR: log_type must be one of: 'install', 'scan', 'tests'."


def _qa_log_stream_text(record: _QALogRecord) -> str:
    """Render only captured stdout/stderr lines for tail and error views."""
    sections: list[str] = []
    if record.stdout:
        sections.append(f"[{record.label} stdout]\n{record.stdout}")
    if record.stderr:
        sections.append(f"[{record.label} stderr]\n{record.stderr}")
    return "\n\n".join(sections)


def _query_qa_logs(
    results: _QAExecutionResults,
    log_type: str,
    *,
    view: str = "summary",
    filter_pattern: str = "",
    tail_lines: int = 0,
) -> str:
    """Query bounded structured QA execution evidence for one phase."""
    from .qa_test_parsing import _QA_ERROR_MARKER

    if log_type not in {"install", "scan", "tests"}:
        return "ERROR: log_type must be one of: 'install', 'scan', 'tests'."
    if view not in _QA_LOG_VIEWS:
        return "ERROR: view must be one of: 'summary', 'tail', 'errors', 'full', 'filter'."
    if not isinstance(tail_lines, int) or isinstance(tail_lines, bool) or tail_lines < 0:
        return "ERROR: tail_lines must be a non-negative integer."
    if view == "filter" and not filter_pattern:
        return "ERROR: filter_pattern is required when view='filter'."
    try:
        filter_re = re.compile(filter_pattern, re.IGNORECASE) if filter_pattern else None
    except re.error as exc:
        return f"ERROR: filter_pattern is not a valid regular expression: {exc}"

    summary = _qa_log_summary(results, log_type)
    if summary.startswith("ERROR:") or summary.startswith("[SKIPPED]") or view == "summary":
        selected = summary
    else:
        records = _qa_log_records_for_phase(results, log_type)
        stream_rendered = "\n\n".join(_qa_log_stream_text(record) for record in records)
        if view == "errors":
            seen_error_lines: set[str] = set()
            error_lines: list[str] = []
            error_rendered = "\n\n".join(
                part
                for part in [
                    stream_rendered,
                    *(
                        f"[{record.label} error]\n{record.error}"
                        for record in records
                        if record.error
                    ),
                ]
                if part
            )
            for line in error_rendered.splitlines():
                if _QA_ERROR_MARKER.search(line) and line not in seen_error_lines:
                    seen_error_lines.add(line)
                    error_lines.append(line)
            selected = "\n".join(error_lines)
            if not selected:
                selected = "(no error lines captured)"
        elif view == "tail":
            count = tail_lines or _QA_LOG_DEFAULT_TAIL_LINES[log_type]
            selected = "\n".join(stream_rendered.splitlines()[-count:])
        else:
            selected = "\n\n".join(_qa_log_record_text(record) for record in records)
    if filter_re is not None:
        selected = "\n".join(line for line in selected.splitlines() if filter_re.search(line))
        if not selected:
            selected = "(no matching log lines)"
    if len(selected) > _LOG_QUERY_MAX_CHARS:
        marker = "\n... (log output truncated)"
        selected = selected[: max(0, _LOG_QUERY_MAX_CHARS - len(marker))].rstrip()
        selected += marker
    return selected


def build_qa_review_toolbelt(
    sandbox: DockerSandbox,
    candidate_changed_files: list[str],
    host_repo_root: str | None,
    results: _QAExecutionResults,
) -> list:
    """Build read-only review tools for one task's QA evaluation.

    Deterministic install, scan, and test execution has already populated
    ``results`` before these tools are exposed. The toolbelt can inspect
    changed files, workspace diffs, source context, AST matches, and bounded
    QA logs, but cannot execute commands or mutate the workspace.
    """
    from remediation_engine.orchestration.tools_workspace import (
        _make_inspect_ast_symbol_tool,
        _make_search_codebase_pattern_tool,
    )

    search_tool = _make_search_codebase_pattern_tool(sandbox)
    inspect_tool = _make_inspect_ast_symbol_tool(sandbox)

    @tool
    def list_changed_files() -> str:
        """
        List the repo-relative file paths that the remedy agents reported as changed.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if not candidate_changed_files:
            return "(no changed files were reported by remedy agents)"
        return "\n".join(f"  - {f}" for f in candidate_changed_files)

    @tool
    def generate_workspace_diff() -> str:
        """
        Generate a unified diff of remedy-agent-reported changed files vs. host baseline.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        if host_repo_root is None:
            return "ERROR: host_repo_root is not available; cannot generate diff."
        diff_text, _ = _generate_workspace_diff(host_repo_root, sandbox, candidate_changed_files)
        return diff_text

    @tool
    def read_file_context(file_path: str) -> str:
        """
        Read the current content of a workspace file for review.
        Only accepts repo-relative paths; absolute paths and '..' traversal are rejected.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        try:
            rel_path = _validate_qa_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"
        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: File '{rel_path}' not found in workspace."
        if len(content) > _FILE_READ_MAX_CHARS:
            content = content[:_FILE_READ_MAX_CHARS] + "\n... (truncated)"
        return content

    @tool
    def query_qa_logs(
        log_type: str,
        view: str = "summary",
        filter_pattern: str = "",
        tail_lines: int = 0,
    ) -> str:
        """
        Query bounded QA evidence.

        Args:
            log_type: QA phase: ``install``, ``scan``, or ``tests``.
            view: ``summary``, ``tail``, ``errors``, ``full``, or ``filter``.
            filter_pattern: Optional case-insensitive regular expression applied
                to the selected view.
            tail_lines: Number of lines for ``tail``; zero uses a phase default.
        """
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return _query_qa_logs(
            results,
            log_type,
            view=view,
            filter_pattern=filter_pattern,
            tail_lines=tail_lines,
        )

    @tool
    def search_codebase_pattern(search_pattern: str, target_directory: str = ".") -> str:
        """Search the workspace for a regex pattern in source files."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(
            search_tool.invoke(
                {"search_pattern": search_pattern, "target_directory": target_directory}
            )
        )

    @tool
    def inspect_ast_symbol(file_path: str, symbol_name: str) -> str:
        """Inspect a specific named symbol in a workspace source file."""
        review_error = _review_ready_error(results)
        if review_error:
            return review_error
        return str(inspect_tool.invoke({"file_path": file_path, "symbol_name": symbol_name}))

    return [
        list_changed_files,
        generate_workspace_diff,
        read_file_context,
        search_codebase_pattern,
        inspect_ast_symbol,
        query_qa_logs,
    ]


def _build_qa_terminal_tool() -> StructuredTool:
    """Build the terminal tool through which the evaluator returns its decision.

    The bounded runtime intercepts this tool call and validates the arguments
    as a QACriticLLMOutput. The function body is never used as an execution
    path; exposing it as a tool gives providers a strict structured schema.
    """

    def emit_qa_evaluation(**_kwargs: Any) -> str:
        """Accept a validated structured QA result from the model."""
        return "STRUCTURED_QA_RESULT_ACCEPTED"

    return StructuredTool.from_function(
        func=emit_qa_evaluation,
        name="emit_qa_evaluation",
        description=(
            "Return the final structured QA evaluation for the assigned task. "
            f"Use exact serialized enum values: failure_category={_QA_FAILURE_CATEGORY_VALUES}; "
            f"semantic_security_review.verdict={_QA_SECURITY_REVIEW_VALUES}; "
            f"test_attribution.verdict={_QA_TEST_ATTRIBUTION_VALUES}. "
            "Do not write a narrative response."
        ),
        args_schema=QACriticLLMOutput,
    )


def _format_deterministic_test_failure_ledger(
    results: _QAExecutionResults,
    failure_evidence: QAFailureEvidence | None = None,
) -> str:
    """Format bounded, Python-owned test evidence without assigning ownership."""
    if results.tests is None:
        return "- Test execution did not produce a result; attribution is unavailable."
    if results.tests[0]:
        return "- No failed tests were reported by deterministic execution."
    if failure_evidence is None:
        return (
            "- Deterministic test execution failed, but no normalized failure records were "
            "available. Use INCONCLUSIVE rather than inferring an owner from the global summary.\n"
            f"- Bounded test summary: {results.tests[1][:3000]}"
        )
    lines = [
        "- Failure records are evidence only; no owner has been assigned by Python.",
        "- Failed tests:",
        *[
            f"  - {value}"
            for value in (failure_evidence.failed_tests or ["(test name unavailable)"])[:10]
        ],
        "- Exact diagnostics:",
        *[
            f"  - {value}"
            for value in (failure_evidence.exact_diagnostics or ["(diagnostic unavailable)"])[:10]
        ],
        "- Source locations observed in deterministic output:",
        *[
            f"  - {value}"
            for value in (failure_evidence.source_locations or ["(source location unavailable)"])[
                :10
            ]
        ],
        "- Affected files observed in deterministic output:",
        *[
            f"  - {value}"
            for value in (failure_evidence.affected_files or ["(affected file unavailable)"])[:10]
        ],
    ]
    return "\n".join(lines)


_QA_FAILURE_CATEGORY_VALUES = ", ".join(f"`{category.value}`" for category in FailureCategory)
_QA_SECURITY_REVIEW_VALUES = ", ".join(f"`{verdict.value}`" for verdict in SecurityReviewVerdict)
_QA_TEST_ATTRIBUTION_VALUES = ", ".join(f"`{verdict.value}`" for verdict in TestAttributionVerdict)
_QA_EVALUATOR_STATIC_PREAMBLE = f"""You are a task-scoped QA evaluator. Review exactly
one vulnerability group using deterministic evidence and read-only workspace tools.
The global install, scanner, and test commands have already run; never execute them
again. Do not infer ownership from a shared failure without exact evidence.

Use only these read-only tools as needed:
list_changed_files, generate_workspace_diff, read_file_context,
search_codebase_pattern, inspect_ast_symbol, query_qa_logs.
Do not edit files, mutate packages, browse the registry, or rerun commands.

Classify the assigned group only. A group passes only when its policy is satisfied,
the vulnerable path is addressed where required, and the evidence supports the
decision. Python owns deterministic dependency, scanner, test, and provenance
evidence. For VERSION_BUMP, do not require raw manifest or lockfile tool output
when the supplied dependency evidence is verified. Use test attribution only when
the evidence supports one of
{_QA_TEST_ATTRIBUTION_VALUES}; use
`{TestAttributionVerdict.INCONCLUSIVE.value}` when exact failed-test evidence and
causal or exonerating source evidence are insufficient.

Completion is terminal-only: call emit_qa_evaluation exactly once and emit no
free-form final answer. Allowed failure_category values are exactly:
{_QA_FAILURE_CATEGORY_VALUES}. Python attaches deterministic gates, dependency
evidence, failure evidence, scan evidence, and contract/provenance fields; do
not fill those fields.
"""


def _qa_status(
    result: Any,
    *,
    skipped: bool = False,
) -> str:
    """Return the compact status token used in the evaluator context."""
    if skipped:
        return "SKIPPED"
    if result is None:
        return "NOT_RUN"
    execution_status = getattr(result, "execution_status", None)
    status_value = getattr(execution_status, "value", execution_status)
    if str(status_value).lower() == "not_run":
        return "NOT_RUN"
    return "PASS" if bool(_scan_result_value(result, "ok", False)) else "FAIL"


def _build_qa_dynamic_context(
    group: VulnerabilityGroup,
    task_id: str,
    strategy: str,
    results: _QAExecutionResults,
    group_remaining_ids: list[str],
    candidate_changed_files: list[str],
    action_summaries: list[AgentActionSummary],
    qa_policy: QAPolicy | None = None,
) -> str:
    """Build task-owned facts appended to the static evaluator prompt."""
    current_strategy = (
        RoutingStrategy(strategy) if strategy in {item.value for item in RoutingStrategy} else None
    )
    selection = select_package_fix_plan(group, current_strategy)
    fix_plan = selection.plan
    fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
    fix_instruction = fix_plan.instruction if fix_plan else "(none)"
    cves = ", ".join(group.cve_ids) if group.cve_ids else "(none)"
    ghsas = ", ".join(group.ghsa_ids or []) or "(none)"
    install_status = _qa_status(results.install)
    scan_status = _qa_status(results.scan, skipped=results.scan_skipped)
    tests_status = _qa_status(results.tests)
    scan_execution_status = getattr(results.scan, "execution_status", None)
    if hasattr(scan_execution_status, "value"):
        scan_execution_status = scan_execution_status.value
    if scan_execution_status is None and results.scan is not None:
        scan_execution_status = (
            ScannerExecutionStatus.SUCCESS.value
            if bool(_scan_result_value(results.scan, "ok", False))
            else ScannerExecutionStatus.UNPARSEABLE.value
        )
    scan_execution_status = scan_execution_status or (
        "SKIPPED" if results.scan_skipped else "NOT_RUN"
    )
    install_exit_code = (
        str(results.install_exit_code) if results.install_exit_code is not None else "unknown"
    )
    install_category = results.install_error_category or "none"
    test_failure_count = (
        str(results.test_failure_count) if results.test_failure_count is not None else "unknown"
    )
    package_state = results.package_state_by_task.get(task_id)
    if package_state is None:
        package_state_text = "manifest=unknown; graph=unknown; diagnostics=none"
    else:
        package_state_text = (
            f"manifest={package_state.manifest_state or 'unknown'}; "
            f"graph={package_state.graph_state or 'unknown'}; "
            f"diagnostics={'; '.join(package_state.diagnostics[:3]) or 'none'}"
        )
    dependency_evidence = package_state.dependency_evidence if package_state is not None else None
    if dependency_evidence is None:
        dependency_evidence_text = "status=unavailable"
    else:
        dependency_payload = {
            # Keep decision-critical fields first so prompt truncation cannot
            # discard the authoritative status and target version.
            "status": getattr(
                dependency_evidence.status,
                "value",
                dependency_evidence.status,
            ),
            "target_package": dependency_evidence.target_package,
            "expected_version": dependency_evidence.expected_version,
            "manifest_paths": dependency_evidence.manifest_paths[:20],
            "lockfile_paths": dependency_evidence.lockfile_paths[:20],
            "declarations": dict(list(dependency_evidence.declarations.items())[:20]),
            "resolved_versions": dependency_evidence.resolved_versions[:20],
            "lockfile_versions": dependency_evidence.lockfile_versions[:20],
            "evidence_refs": dependency_evidence.evidence_refs[:20],
            "diagnostics": dependency_evidence.diagnostics[:3],
        }
        dependency_evidence_text = json.dumps(
            dependency_payload,
            separators=(",", ":"),
        )
    remaining_text = (
        ", ".join(group_remaining_ids)
        if group_remaining_ids
        else "(none - scanner cleared this group)"
    )
    target_identifiers = sorted(group_target_identifiers(group))
    post_scan_identifiers = sorted(
        _scan_result_value(results.scan, "found_identifiers", set()) or set()
    )
    new_identifiers = sorted(_scan_result_value(results.scan, "new_identifiers", set()) or set())
    summaries_text = (
        "\n".join(
            f"  - {summary.status.value}: {_bounded_qa_action_summary(summary.summary, group)}"
            for summary in action_summaries
        )
        or "  (none)"
    )
    changed_files_text = ", ".join(candidate_changed_files) or "(none reported)"
    policy_block = _qa_policy_prompt_block(qa_policy)
    dependency_evidence_text = dependency_evidence_text[:6_000]

    return f"""{policy_block}
## Assigned Task
- Task ID: {task_id}
- Parent Group ID: {group.group_id}
- Component: {group.vulnerable_component or "(unknown)"}
- Issue type: {group.issue_type.value}
- Routing strategy: {strategy}
- CVEs: {cves}
- GHSAs: {ghsas}
- Target scanner identifiers: {", ".join(target_identifiers) or "(none)"}
- Fix-plan status: {fix_plan_status}
- Fix instruction: {fix_instruction}

## Deterministic Status Flags
- Install: {install_status}
- Security scan: {scan_status}
- Unit tests: {tests_status}
- Install exit code: {install_exit_code}
- Install error category: {install_category}
- Scanner execution status: {scan_execution_status}
- Remaining identifiers for this group: {remaining_text}
- Remaining identifier count: {len(group_remaining_ids)}
- Test failure count: {test_failure_count}
- Package state: {package_state_text}
- Dependency evidence (Python-owned and authoritative): {dependency_evidence_text}

## Identifiers and Files
- Changed files reported by the worker: {changed_files_text}
- All post-remediation scanner identifiers: {", ".join(post_scan_identifiers) if post_scan_identifiers else "(none or unavailable)"}
- New baseline-absent scanner identifiers: {", ".join(new_identifiers) if new_identifiers else "(none)"}

New identifiers are graph-level findings for later triage. Do not attribute them
to this group without direct deterministic evidence.

## Action Summaries
{summaries_text}
## Terminal Requirements
Review only task {task_id} (parent group {group.group_id}). Do not emit Markdown,
prose, or a free-form final answer. Finish with exactly one call to
emit_qa_evaluation:
- task_id: exactly "{task_id}"
- passed: true or false
- failure_category: null when passed=true; otherwise one of {_QA_FAILURE_CATEGORY_VALUES}
- retry_feedback: null when passed=true; otherwise concise guidance with exact evidence
- semantic_security_review: include a verdict from {_QA_SECURITY_REVIEW_VALUES},
  reasoning, and concrete evidence_refs when the assigned policy requires semantic review
- test_attribution: include only when shared tests failed; use one of
  {_QA_TEST_ATTRIBUTION_VALUES} with exact evidence

Python owns deterministic_gates, failure_evidence, scan_evidence, and
contract/provenance fields. Do not fill those fields."""


def _build_individual_investigator_prompt(
    group: VulnerabilityGroup,
    task_id: str,
    strategy: str,
    results: _QAExecutionResults,
    group_remaining_ids: list[str],
    candidate_changed_files: list[str],
    action_summaries: list[AgentActionSummary],
    qa_policy: QAPolicy | None = None,
) -> str:
    """Build the lean static-plus-dynamic prompt for one QA evaluator."""
    return (
        _QA_EVALUATOR_STATIC_PREAMBLE
        + "\n\n"
        + _build_qa_dynamic_context(
            group=group,
            task_id=task_id,
            strategy=strategy,
            results=results,
            group_remaining_ids=group_remaining_ids,
            candidate_changed_files=candidate_changed_files,
            action_summaries=action_summaries,
            qa_policy=qa_policy,
        )
    )


def _sanitize_llm_evaluation(
    evaluation: QACriticLLMOutput,
    task_id: str,
) -> QAEvaluation:
    """Keep only fields the LLM is allowed to author for one task."""
    return QAEvaluation(
        task_id=task_id,
        passed=evaluation.passed,
        failure_category=evaluation.failure_category,
        retry_feedback=evaluation.retry_feedback,
        semantic_security_review=evaluation.semantic_security_review,
        test_attribution=evaluation.test_attribution,
    )


def _run_individual_investigations(
    task_contexts: list[QATaskContext],
    task_strategies: dict[str, str],
    action_summaries: list[AgentActionSummary],
    changed_files_by_task: dict[str, list[str]],
    sandbox: DockerSandbox,
    repo_root: str | None,
    results: _QAExecutionResults,
    task_policies: dict[str, QAPolicy | None],
) -> dict[str, GroupInvestigation]:
    """Run one bounded structured evaluator independently for each task."""
    from langchain_openai import ChatOpenAI

    model_name = get_runtime_settings().qa_llm_model
    known_task_ids = {context.task_id for context in task_contexts}
    evaluations: dict[str, GroupInvestigation] = {}

    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        task_changed_files = changed_files_by_task.get(task_id, [])
        strategy = task_strategies.get(task_id, "")
        remaining_scan = _scan_result_value(results.scan, "remaining_identifiers", set())
        group_remaining_ids = sorted(group_target_identifiers(group) & set(remaining_scan or set()))
        qa_policy = task_policies.get(task_id)
        relevant_summaries = _relevant_action_summaries(action_summaries, task_id, known_task_ids)
        dynamic_context = _build_qa_dynamic_context(
            group=group,
            task_id=task_id,
            strategy=strategy,
            results=results,
            group_remaining_ids=group_remaining_ids,
            candidate_changed_files=task_changed_files,
            action_summaries=relevant_summaries,
            qa_policy=qa_policy,
        )
        terminal_tool = _build_qa_terminal_tool()
        review_tools = [
            *build_qa_review_toolbelt(
                sandbox=sandbox,
                candidate_changed_files=task_changed_files,
                host_repo_root=repo_root,
                results=results,
            ),
            terminal_tool,
        ]
        qa_context_manager = ContextManager(
            review_tools,
            compaction_interval=3,
            skip_phase_gating=True,
            scratchpad_scope=ScratchpadScope.QA,
        )
        llm = ChatOpenAI(model=model_name, temperature=0)
        initial_messages = [
            SystemMessage(content=_QA_EVALUATOR_STATIC_PREAMBLE),
            HumanMessage(
                content=(
                    f"{dynamic_context}\n\n"
                    f"Review task '{task_id}' for group '{group.group_id}' with the "
                    "read-only tools, then call emit_qa_evaluation. Do not emit any "
                    "narrative response."
                )
            ),
        ]

        logger.info("qa_critic: starting structured evaluator for task '%s'.", task_id)
        try:
            loop_result = run_bounded_subagent_loop(
                llm=llm,
                tools=review_tools,
                initial_messages=initial_messages,
                touched_files=set(),
                structured_output_model=QACriticLLMOutput,
                structured_output_tool_name=terminal_tool.name,
                context_manager=qa_context_manager,
                skip_phase_gating=True,
            )

            tool_transcript = json.dumps(
                [
                    {
                        "name": event.name,
                        "args": event.args,
                        "content": event.content[:1000],
                    }
                    for event in loop_result.tool_events
                ],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            review_tools_used: list[str] = []
            source_review_evidence = False
            for event in loop_result.tool_events:
                if event.name in {
                    "generate_workspace_diff",
                    "read_file_context",
                    "search_codebase_pattern",
                    "inspect_ast_symbol",
                }:
                    review_tools_used.append(event.name)
                    source_review_evidence = (
                        source_review_evidence
                        or _has_successful_review_tool_evidence(event.name, event.content)
                    )

            errors = list(loop_result.errors)
            structured_output = loop_result.structured_output
            task_id_mismatch = (
                isinstance(structured_output, QACriticLLMOutput)
                and structured_output.task_id != task_id
            )
            fallback = not isinstance(structured_output, QACriticLLMOutput) or task_id_mismatch
            if task_id_mismatch:
                errors.append(
                    f"qa_critic: evaluator returned task_id {structured_output.task_id!r} "
                    f"for assigned task {task_id!r}."
                )
            if fallback:
                reason = (
                    f"QA evaluator for task '{task_id}' did not emit a valid structured QA result."
                )
                errors.append(reason)
                evaluation = QAEvaluation(
                    task_id=task_id,
                    passed=False,
                    contract_error=True,
                    contract_error_reason=reason,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback=(
                        "The QA evaluator did not return its required structured result. "
                        "Retry QA after correcting the evaluator contract."
                    ),
                )
            else:
                evaluation = _sanitize_llm_evaluation(structured_output, task_id)

            evaluations[task_id] = GroupInvestigation(
                group_id=group.group_id,
                task_id=task_id,
                investigation_text="",
                tool_transcript=tool_transcript,
                errors=list(dict.fromkeys(errors)),
                review_tools_used=sorted(set(review_tools_used)),
                source_review_evidence=source_review_evidence,
                fallback=fallback,
                structured_review_verdict=(
                    evaluation.semantic_security_review.verdict
                    if evaluation.semantic_security_review is not None
                    else None
                ),
                evaluation=evaluation,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"qa_critic: evaluator for task '{task_id}' crashed: {exc}"
            logger.error(err)
            evaluation = QAEvaluation(
                task_id=task_id,
                passed=False,
                contract_error=True,
                contract_error_reason=err,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"Structured QA evaluator failed: {exc}. "
                    "Retry QA after correcting the evaluator."
                ),
            )
            evaluations[task_id] = GroupInvestigation(
                group_id=group.group_id,
                task_id=task_id,
                investigation_text="Fallback: structured QA result unavailable.",
                tool_transcript="",
                errors=[err],
                fallback=True,
                evaluation=evaluation,
            )
        logger.info(
            "qa_critic: structured evaluator for task '%s' complete; passed=%s.",
            task_id,
            evaluations[task_id].evaluation.passed
            if evaluations[task_id].evaluation is not None
            else False,
        )

    return evaluations


def _build_fallback_investigation_for_group(
    group: VulnerabilityGroup,
    strategy: str,
    results: _QAExecutionResults,
    group_remaining_ids: list[str],
    reason: str,
) -> str:
    """Synthesize a minimal investigation for a single group when the investigator fails."""
    install_ok, _ = results.install or (False, "not run")
    tests_ok, _ = results.tests or (False, "not run")
    scan_ok = results.scan.ok if results.scan else results.scan_skipped
    new_identifiers = sorted(_scan_result_value(results.scan, "new_identifiers", set()) or set())

    remaining_text = ", ".join(group_remaining_ids) if group_remaining_ids else "none"
    scan_status = (
        "still_flagged"
        if group_remaining_ids
        else ("skipped" if results.scan_skipped else ("cleared" if scan_ok else "scan_failed"))
    )

    return (
        f"## Fallback Investigation: {group.group_id}\n\n"
        f"**Reason:** {reason}\n\n"
        f"**Component:** {group.vulnerable_component or '(unknown)'}\n"
        f"**Strategy:** {strategy}\n"
        f"**Target Identifiers:** {', '.join(sorted(group_target_identifiers(group))) or 'none'}\n\n"
        f"**Deterministic Results:**\n"
        f"- Install: {'SUCCESS' if install_ok else 'FAILED'}\n"
        f"- Security Scan: {'SKIPPED' if results.scan_skipped else ('SUCCESS' if scan_ok else 'FAILED')}\n"
        f"- Scan Status for this group: {scan_status}\n"
        f"- Remaining Scanner Identifiers: {remaining_text}\n"
        f"- Global Newly Introduced Scanner Identifiers: {', '.join(new_identifiers) if new_identifiers else 'none'}\n"
        f"- Unit Tests: {'PASSED' if tests_ok else 'FAILED'}\n\n"
        f"**Summary:** Fallback investigation synthesized from deterministic results only. "
        f"No investigator prose was available."
    )
