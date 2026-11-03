"""Typed QA evaluation, policy, and evidence validation."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool, tool

from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    DependencyEvidenceStatus,
    FailureCategory,
    IssueType,
    ODCScanEvidence,
    QACriticLLMOutput,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    QASemanticSecurityReview,
    QATestAttribution,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScratchpadScope,
    SecurityReviewVerdict,
    TaskAttemptSnapshot,
    TestAttributionVerdict,
    VulnerabilityGroup,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.state import OrchestratorState, select_package_fix_plan
from remediation_engine.runtime.agent_runtime import (
    ContextManager,
    create_chat_model,
    run_bounded_subagent_loop,
)
from remediation_engine.runtime.sandbox import DockerSandbox
from remediation_engine.settings import get_runtime_settings
from remediation_engine.tools.qa_scan import (
    QATaskContext,
    _QAExecutionResults,
    _QALogRecord,
    _QAPackageState,
    _scan_result_value,
    _validate_qa_path,
)
from remediation_engine.tools.qa_workspace import (
    _bounded_qa_action_summary,
    _generate_workspace_diff,
    _relevant_action_summaries,
    _review_ready_error,
    group_target_identifiers,
)

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
                label="dependency installation",
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
                label="unit tests",
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
    from remediation_engine.tools.test_execution import _QA_ERROR_MARKER

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
    from remediation_engine.tools.workspace import (
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
            "For responsible or exonerated test attribution, include non-empty "
            "responsible_group_ids, failed_tests, and reasoning; use exact group IDs "
            "from the supplied task context. Use inconclusive when group ownership is "
            "not supported by positive evidence. "
            "Do not write a narrative response."
        ),
        args_schema=QACriticLLMOutput,
    )


_QA_FAILURE_CATEGORY_VALUES = ", ".join(f"`{category.value}`" for category in FailureCategory)
_QA_SECURITY_REVIEW_VALUES = ", ".join(f"`{verdict.value}`" for verdict in SecurityReviewVerdict)
_QA_TEST_ATTRIBUTION_VALUES = ", ".join(f"`{verdict.value}`" for verdict in TestAttributionVerdict)
_QA_EVALUATOR_STATIC_PREAMBLE = f"""You are a task-scoped QA evaluator. Review exactly
one vulnerability group using deterministic evidence and read-only workspace tools.
The global install, scanner, and test commands have already run; never execute them
again. The dynamic context declares whether this dispatch is singleton scope. In
singleton scope, a failed application test run is owned by the assigned task because
its worker attempt is the only remediation action preceding QA; do not treat the
global command boundary as evidence of multiple owners.

Use only these read-only tools as needed:
list_changed_files, generate_workspace_diff, read_file_context,
search_codebase_pattern, inspect_ast_symbol, query_qa_logs.
Do not edit files, mutate packages, browse the registry, or rerun commands.

Classify the assigned group only. A group passes only when its policy is satisfied,
the vulnerable path is addressed where required, and the evidence supports the
decision. Python owns deterministic dependency, scanner, test, and provenance
evidence. For VERSION_BUMP, do not require raw manifest or lockfile tool output
when the supplied dependency evidence is verified. In multi-task scope, use test
attribution only when the evidence supports one of
{_QA_TEST_ATTRIBUTION_VALUES}. In singleton scope, deterministic Python normalizes
an executed failed test run to the assigned task; use
`{TestAttributionVerdict.INCONCLUSIVE.value}` only when the test result was not run
or the QA contract/evidence itself is invalid.

Completion is terminal-only: call emit_qa_evaluation exactly once and emit no
free-form final answer. Allowed failure_category values are exactly:
{_QA_FAILURE_CATEGORY_VALUES}. Python attaches deterministic gates, dependency
evidence, failure evidence, scan evidence, and contract/provenance fields; do
not fill those fields.
"""


def _build_qa_evaluator_system_prompt(
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Build read-only evaluator instructions for the detected project language."""
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    if project_language == ProjectLanguage.PYTHON:
        return (
            _QA_EVALUATOR_STATIC_PREAMBLE
            + "\n\nCanonical project language: python (Python). "
            + "Interpret pytest failures as Python unit-test evidence."
        )
    return _QA_EVALUATOR_STATIC_PREAMBLE


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
    if isinstance(result, tuple):
        # Install and test results are stored as ``(ok, summary)`` tuples,
        # whereas scan results expose an ``ok`` attribute.  Keep the prompt
        # projection aligned with the typed execution result instead of
        # treating the tuple as a scan object.
        return "PASS" if result and bool(result[0]) else "FAIL"
    execution_status = getattr(result, "execution_status", None)
    status_value = getattr(execution_status, "value", execution_status)
    if str(status_value).lower() == "not_run":
        return "NOT_RUN"
    if isinstance(result, tuple):
        return "PASS" if bool(result and result[0]) else "FAIL"
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
    singleton_scope: bool = False,
    task_scan_evidence: ODCScanEvidence | None = None,
    attempt_snapshot: TaskAttemptSnapshot | None = None,
    effective_target_version: str | None = None,
) -> str:
    """Build task-owned facts appended to the static evaluator prompt.

    Args:
        group: Vulnerability group under review.
        task_id: Active remediation task identifier.
        strategy: Supervisor-selected routing strategy.
        results: Deterministic QA execution results.
        group_remaining_ids: Target identifiers still present for this group.
        candidate_changed_files: Worker-reported files for this task.
        action_summaries: Bounded worker action summaries.
        qa_policy: Supervisor-owned QA policy, when available.
        singleton_scope: Whether exactly one active task owns this QA run.
        task_scan_evidence: Scanner findings projected to this task's dependency closure.
        attempt_snapshot: Current Supervisor-committed authorization for this QA task.
        effective_target_version: Current worker result's effective dependency version,
            when it carries matching attempt provenance.

    Returns:
        The bounded dynamic evaluator context.
    """
    current_strategy = (
        RoutingStrategy(strategy) if strategy in {item.value for item in RoutingStrategy} else None
    )
    if attempt_snapshot is not None:
        fix_plan_status = "supervisor-committed attempt"
        fix_instruction = attempt_snapshot.instruction
        authorization_lines = [
            f"- Committed attempt ID: {attempt_snapshot.attempt_id}",
            f"- Committed task revision: {attempt_snapshot.task_revision}",
            f"- Authorized target package: {attempt_snapshot.target_package_name or '(not specified)'}",
            f"- Authorized dependency type: {attempt_snapshot.target_dependency_type or '(not specified)'}",
            "- Authorized target versions: "
            + (", ".join(attempt_snapshot.allowed_target_versions) or "(none listed)"),
            "- Committed selected version: " + (attempt_snapshot.selected_version or "(none)"),
            "- Parent minimum version: " + (attempt_snapshot.parent_minimum_version or "(none)"),
            "- Authorized dependency types: "
            + (", ".join(attempt_snapshot.allowed_dependency_types) or "(none listed)"),
            "- Worker-reported effective target version: "
            + (effective_target_version or "(not available)"),
            "- The committed instruction and candidate bounds define authorization. "
            "Use the committed selected version unless the instruction explicitly allows a "
            "bounded alternative; do not broaden the candidate set.",
        ]
        authorization_heading = "## Supervisor-Committed Attempt Authorization"
    else:
        selection = select_package_fix_plan(group, current_strategy)
        fix_plan = selection.plan
        fix_plan_status = fix_plan.status.value if fix_plan else "unknown"
        fix_instruction = fix_plan.instruction if fix_plan else "(none)"
        authorization_lines = [
            "- No current committed attempt snapshot was supplied; this is a legacy context.",
            "- Group fix-plan details are advisory only and do not authorize execution.",
        ]
        authorization_heading = "## Legacy Advisory Fix-Plan Context"
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
    post_scan_identifiers = (
        sorted(task_scan_evidence.found_identifiers) if task_scan_evidence else []
    )
    new_identifiers = sorted(task_scan_evidence.new_identifiers) if task_scan_evidence else []
    scan_coverage = (
        "complete"
        if task_scan_evidence is not None and task_scan_evidence.complete
        else "incomplete"
        if task_scan_evidence is not None or results.scan_evidence is not None
        else "unavailable"
    )
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
    if singleton_scope:
        scope_block = f"""## QA Scope and Test Attribution
- Dispatch scope: SINGLETON
- Active remediation task: {task_id}
- The global commands may run once for efficiency, but this is the only worker attempt preceding QA.
- If Unit tests is FAIL and the test result ran, the failed test run belongs to parent group {group.group_id}.
- Do not emit EXONERATED or INCONCLUSIVE merely because the command output is global or the lockfile contains other changes.
"""
    else:
        scope_block = """## QA Scope and Test Attribution
- Dispatch scope: MULTI_TASK_OR_LEGACY
- Global test ownership is not deterministic from scope alone; require exact attribution evidence.
- Use INCONCLUSIVE when failed-test evidence cannot support responsible or exonerated attribution.
"""

    authorization_block = "\n".join([authorization_heading, *authorization_lines])

    return f"""{policy_block}
{scope_block}
{authorization_block}
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
- Task-scoped scanner coverage: {scan_coverage}
- Remaining identifiers for this group: {remaining_text}
- Remaining identifier count: {len(group_remaining_ids)}
- Test failure count: {test_failure_count}
- Package state: {package_state_text}
- Dependency evidence (Python-owned and authoritative): {dependency_evidence_text}

## Deterministic Test Attribution Rule
- Failed test runs in singleton scope are attributed to the active task by Python after evaluation.
- A test result of NOT_RUN, a missing structured QA result, or an explicitly inconclusive evidence contract remains inconclusive.

## Identifiers and Files
- Changed files reported by the worker: {changed_files_text}
- Task-scoped post-remediation scanner identifiers: {", ".join(post_scan_identifiers) if post_scan_identifiers else "(none or unavailable)"}
- Task-scoped baseline-absent scanner identifiers: {", ".join(new_identifiers) if new_identifiers else "(none or unavailable)"}

Repository-wide scanner findings are intentionally excluded from this task context.
Only the task-scoped occurrence evidence above can affect this task's verdict.

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
- test_attribution: when Unit tests is FAIL, use one of {_QA_TEST_ATTRIBUTION_VALUES};
  provide the complete object fields verdict, responsible_group_ids, failed_tests, and reasoning
- for responsible or exonerated, responsible_group_ids must be a non-empty list of exact
  group_id values from the supplied context, failed_tests must contain exact failed test names,
  and reasoning must explain the evidence for that attribution
- use inconclusive with an empty responsible_group_ids list when positive evidence does not
  identify which remediation group owns the failure; do not blame every group in a batch
- in SINGLETON scope Python assigns test ownership by deterministic contract

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
    singleton_scope: bool = False,
    task_scan_evidence: ODCScanEvidence | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    attempt_snapshot: TaskAttemptSnapshot | None = None,
    effective_target_version: str | None = None,
) -> str:
    """Build the lean static-plus-dynamic prompt for one QA evaluator.

    Args:
        group: Vulnerability group under review.
        task_id: Active remediation task identifier.
        strategy: Supervisor-selected routing strategy.
        results: Deterministic QA execution results.
        group_remaining_ids: Target identifiers still present for this group.
        candidate_changed_files: Worker-reported files for this task.
        action_summaries: Bounded worker action summaries.
        qa_policy: Supervisor-owned QA policy, when available.
        singleton_scope: Whether exactly one active task owns this QA run.
        task_scan_evidence: Scanner findings projected to this task's dependency closure.
        project_language: Canonical repository language used to interpret test evidence.
        attempt_snapshot: Current Supervisor-committed authorization, if available.
        effective_target_version: Current worker result's effective target version.

    Returns:
        The complete evaluator prompt.
    """
    return (
        _build_qa_evaluator_system_prompt(project_language)
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
            singleton_scope=singleton_scope,
            task_scan_evidence=task_scan_evidence,
            attempt_snapshot=attempt_snapshot,
            effective_target_version=effective_target_version,
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
    singleton_scope: bool = False,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    attempt_snapshots_by_id: dict[str, TaskAttemptSnapshot] | None = None,
    worker_results_by_attempt: dict[str, Any] | None = None,
) -> dict[str, GroupInvestigation]:
    """Run one bounded structured evaluator independently for each task.

    Args:
        task_contexts: Task/group contexts being evaluated.
        task_strategies: Supervisor-selected strategy by task ID.
        action_summaries: Recent worker action summaries.
        changed_files_by_task: Worker-reported changed files by task ID.
        sandbox: Active read-only QA sandbox.
        repo_root: Optional host repository root for review tools.
        results: Deterministic global QA results.
        task_policies: Supervisor-owned QA policy by task ID.
        singleton_scope: Whether the dispatch contains exactly one active task.
        project_language: Canonical repository language used to interpret test evidence.
        attempt_snapshots_by_id: Supervisor-committed attempts indexed by attempt ID.
        worker_results_by_attempt: Worker execution results indexed by attempt ID.

    Returns:
        Structured investigation records keyed by task ID.
    """
    from langchain_openai import ChatOpenAI

    settings = get_runtime_settings()
    model_name = settings.qa_llm_model
    known_task_ids = {context.task_id for context in task_contexts}
    evaluations: dict[str, GroupInvestigation] = {}

    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        task_changed_files = changed_files_by_task.get(task_id, [])
        strategy = task_strategies.get(task_id, "")
        task_scan_evidence = results.scan_evidence_by_task.get(task_id)
        if task_scan_evidence is None:
            if results.scan_evidence is not None:
                # Aggregate scan identifiers are not attributable to one task.
                # Missing task projection is incomplete evidence, not a reason
                # to fall back to report-wide findings in the evaluator prompt.
                group_remaining_ids = []
            else:
                remaining_scan = _scan_result_value(results.scan, "remaining_identifiers", set())
                group_remaining_ids = sorted(
                    group_target_identifiers(group) & set(remaining_scan or set())
                )
        else:
            group_remaining_ids = sorted(task_scan_evidence.remaining_target_identifiers)
        qa_policy = task_policies.get(task_id)
        relevant_summaries = _relevant_action_summaries(action_summaries, task_id, known_task_ids)
        attempt_snapshot = None
        effective_target_version = None
        attempt_id = context.task.current_attempt_id
        if attempt_id:
            attempt_snapshot = (attempt_snapshots_by_id or {}).get(attempt_id)
            if (
                attempt_snapshot is None
                or attempt_snapshot.attempt_id != attempt_id
                or attempt_snapshot.task_id != task_id
                or attempt_snapshot.task_revision != context.task.task_revision
            ):
                reason = (
                    f"QA for task '{task_id}' cannot use committed authorization: "
                    "the current attempt snapshot is missing or has stale identity."
                )
                evaluations[task_id] = GroupInvestigation(
                    group_id=group.group_id,
                    task_id=task_id,
                    investigation_text="",
                    tool_transcript="",
                    errors=[reason],
                    fallback=True,
                    evaluation=QAEvaluation(
                        task_id=task_id,
                        passed=False,
                        contract_error=True,
                        contract_error_reason=reason,
                        failure_category=FailureCategory.SECURITY_FLAG,
                        retry_feedback=(
                            "The current task attempt has no matching Supervisor snapshot. "
                            "Re-establish current attempt provenance before evaluating QA."
                        ),
                    ),
                )
                continue

            worker_result = (worker_results_by_attempt or {}).get(attempt_id)
            if (
                worker_result is not None
                and getattr(worker_result, "attempt_id", None) == attempt_id
                and getattr(worker_result, "task_id", None) == task_id
                and getattr(worker_result, "task_revision", None) == context.task.task_revision
            ):
                diagnostics = getattr(worker_result, "execution_diagnostics", None)
                effective_target_version = getattr(
                    diagnostics, "effective_target_version", None
                ) or getattr(worker_result, "effective_target_version", None)
                if not effective_target_version:
                    executed_versions = list(
                        getattr(diagnostics, "executed_versions", None)
                        or getattr(worker_result, "executed_versions", None)
                        or []
                    )
                    if len(set(executed_versions)) == 1:
                        effective_target_version = executed_versions[0]
        dynamic_context = _build_qa_dynamic_context(
            group=group,
            task_id=task_id,
            strategy=strategy,
            results=results,
            group_remaining_ids=group_remaining_ids,
            candidate_changed_files=task_changed_files,
            action_summaries=relevant_summaries,
            qa_policy=qa_policy,
            singleton_scope=singleton_scope,
            task_scan_evidence=task_scan_evidence,
            attempt_snapshot=attempt_snapshot,
            effective_target_version=effective_target_version,
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
        llm = create_chat_model(
            model_name,
            settings=settings,
            reasoning_effort=getattr(settings, "qa_llm_reasoning_effort", None),
            chat_openai_class=ChatOpenAI,
        )
        initial_messages = [
            SystemMessage(content=_build_qa_evaluator_system_prompt(project_language)),
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


# Keep the broader package-state rollout opt-in while requiring installed-tree
# verification for version-bump attempts.
_QA_PACKAGE_STATE_CHECKS_ENABLED = False
_QA_VERSION_BUMP_PACKAGE_STATE_CHECKS_ENABLED = True


_PEER_CONFLICT_PATTERNS = ("ERESOLVE", "EOVERRIDE", "peer dep", "peer tree")
_ENGINE_CONFLICT_PATTERNS = ("EBADENGINE",)


def _package_state_checks_enabled(policy: QAPolicy | None) -> bool:
    """Return whether deterministic package-state checks apply to a policy."""
    return _QA_PACKAGE_STATE_CHECKS_ENABLED or (
        _QA_VERSION_BUMP_PACKAGE_STATE_CHECKS_ENABLED and policy == QAPolicy.VERSION_BUMP
    )


@dataclass(frozen=True)
class _QAPolicyPromptSpec:
    """Prompt contract for one deterministic QA policy."""

    scanner_rule: str
    package_rule: str
    tests_rule: str
    semantic_rule: str
    review_focus: str
    prohibited_conclusions: str


_QA_POLICY_PROMPT_SPECS: dict[QAPolicy, _QAPolicyPromptSpec] = {
    QAPolicy.VERSION_BUMP: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared. Treat remaining target identifiers as a blocking security failure.",
        "Python verifies the authorized manifest and resolved dependency state. Use the supplied typed dependency evidence; do not fail because raw manifest or lockfile output is unavailable.",
        "In singleton scope, attribute an executed failed test run to this task. In multi-task scope, review each failed test independently and assign responsibility only with positive evidence tied to a group's changed package or behavior; otherwise use structured exoneration or INCONCLUSIVE attribution.",
        "A code-path semantic review is not required by this policy.",
        "Validate the Python-owned dependency evidence, target scanner clearance, and any test attribution.",
        "In singleton scope, do not exonerate or mark a failed test run inconclusive merely because install/tests ran through global commands. In multi-task scope, do not assign blame because a group was updated in the same batch, and do not exonerate a group without naming exact failed tests and evidence.",
    ),
    QAPolicy.INITIAL_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers are non-blocking for this policy. They are evidence to interpret, not proof that the code workaround failed.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the initial workaround blocks the vulnerable execution path.",
        "Identify the advisory mechanism, affected call sites, protected call sites, and concrete file/symbol evidence showing whether the workaround is effective.",
        "Do not fail solely because the original scanner identifier remains, and do not pass using test logs without source or diff evidence.",
    ),
    QAPolicy.MIGRATION_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared. Treat remaining target identifiers as a blocking security failure.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A required semantic security review is not part of this policy.",
        "Validate target scanner clearance, migration/install behavior, and whether the remediation caused any test or build regression.",
        "Do not treat a remaining target identifier as non-blocking or use LLM attribution to override hard scanner or test gates.",
    ),
    QAPolicy.MITIGATION_CODE_WORKAROUND: _QAPolicyPromptSpec(
        "Target scanner identifiers are non-blocking for this policy. They are evidence to interpret, not proof that the mitigation failed.",
        "No package-state transition is required by this policy.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the mitigation blocks or materially constrains the vulnerable behavior.",
        "Identify the advisory mechanism, affected call sites, protected call sites, and concrete file/symbol evidence showing whether the mitigation is effective.",
        "Do not fail solely because the original scanner identifier remains, and do not pass using test logs without source or diff evidence.",
    ),
    QAPolicy.NO_FIX_PACKAGE_REMOVAL: _QAPolicyPromptSpec(
        "Target scanner identifiers must be cleared after package removal. Treat remaining target identifiers as a blocking security failure.",
        "The vulnerable package must be absent from every authorized direct manifest and from the resolved dependency graph.",
        "The required test suite must pass.",
        "A code-path semantic security review is not required by this policy.",
        "Confirm the authorized package-removal operation, manifest state, resolved graph, scanner clearance, and test results.",
        "Do not pass while the package remains in an authorized manifest or resolved graph, and do not treat residual target findings as non-blocking.",
    ),
    QAPolicy.NO_FIX_CODE_REMOVAL: _QAPolicyPromptSpec(
        "Target scanner identifiers are expected to remain because the vulnerable package is intentionally retained. They are non-blocking for this policy.",
        "The vulnerable package must remain present in the direct manifest and resolved dependency graph. The workaround must not remove or change dependency metadata.",
        "The required test suite must pass.",
        "A source/diff-based semantic security review is required. Determine whether the vulnerable code path was removed while the package remained installed.",
        "Identify the advisory mechanism, affected and protected call sites, and concrete file/symbol evidence showing that the vulnerable execution path is no longer used.",
        "Do not fail solely because the target package or scanner identifiers remain. Do not pass using test logs without source or diff evidence.",
    ),
}
_STRICT_SCANNER_QA_POLICIES = frozenset(
    {
        QAPolicy.VERSION_BUMP,
        QAPolicy.MIGRATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
    }
)
_HARD_TEST_QA_POLICIES = frozenset(
    {
        QAPolicy.INITIAL_CODE_WORKAROUND,
        QAPolicy.MIGRATION_CODE_WORKAROUND,
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }
)
_REQUIRED_SEMANTIC_QA_POLICIES = frozenset(
    {
        QAPolicy.INITIAL_CODE_WORKAROUND,
        QAPolicy.MITIGATION_CODE_WORKAROUND,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }
)


def _qa_policy_prompt_block(policy: QAPolicy | None) -> str:
    """Render Supervisor-owned QA rules for the structured evaluator."""
    if policy is None:
        return (
            "## QA Policy\n"
            "- Policy: unavailable; do not invent a policy-specific pass.\n"
            "- Treat missing policy as an inconclusive contract condition."
        )
    spec = _QA_POLICY_PROMPT_SPECS[policy]
    package_state_checks_enabled = _package_state_checks_enabled(policy)
    package_rule = (
        spec.package_rule
        if package_state_checks_enabled
        else "Package manifest, lockfile, and installed-tree checks are temporarily disabled. "
        "Do not inspect or fail based on dependency-state evidence."
    )
    review_focus = (
        spec.review_focus
        if package_state_checks_enabled
        else "Assess install, scanner, and test evidence under this task's applicable policy."
    )
    prohibited_conclusions = (
        spec.prohibited_conclusions
        if package_state_checks_enabled
        else "Do not claim package manifest, lockfile, or installed-tree checks were performed."
    )
    return "\n".join(
        [
            "## QA Policy",
            f"- Policy: {policy.value}",
            f"- Scanner rule: {spec.scanner_rule}",
            f"- Package rule: {package_rule}",
            f"- Test rule: {spec.tests_rule}",
            f"- Semantic review rule: {spec.semantic_rule}",
            f"- Review focus: {review_focus}",
            f"- Prohibited conclusions: {prohibited_conclusions}",
        ]
    )


def _valid_test_attribution(
    evaluation: QAEvaluation,
    group_id: str,
    known_group_ids: set[str],
) -> bool:
    """Return whether a structured test attribution contains usable evidence."""
    attribution = evaluation.test_attribution
    if attribution is None or not attribution.failed_tests or not attribution.reasoning.strip():
        return False
    if any(identifier not in known_group_ids for identifier in attribution.responsible_group_ids):
        return False
    if attribution.verdict == TestAttributionVerdict.INCONCLUSIVE:
        return True
    if not attribution.responsible_group_ids:
        return False
    if attribution.verdict == TestAttributionVerdict.RESPONSIBLE:
        return group_id in attribution.responsible_group_ids
    if attribution.verdict == TestAttributionVerdict.EXONERATED:
        return group_id not in attribution.responsible_group_ids
    return False


def _evaluate_policy_gates(
    task_contexts: list[QATaskContext],
    results: _QAExecutionResults,
    task_policies: dict[str, QAPolicy | None],
) -> tuple[dict[str, QADeterministicGates], list[str]]:
    """Record deterministic execution evidence and policy gates per task."""
    gates_by_task: dict[str, QADeterministicGates] = {}
    errors: list[str] = []
    install_passed = bool(results.install and results.install[0])
    install_summary = results.install[1] if results.install else "install did not run"
    tests_passed = results.tests[0] if results.tests is not None else None
    remaining_global = set(
        _scan_result_value(results.scan, "remaining_identifiers", set()) or set()
    )
    scanner_status = _scan_result_value(
        results.scan, "execution_status", ScannerExecutionStatus.NOT_RUN
    )
    if not isinstance(scanner_status, ScannerExecutionStatus):
        try:
            scanner_status = ScannerExecutionStatus(str(scanner_status))
        except ValueError:
            scanner_status = ScannerExecutionStatus.UNPARSEABLE
    if scanner_status == ScannerExecutionStatus.NOT_RUN and results.scan is not None:
        scanner_status = (
            ScannerExecutionStatus.SUCCESS
            if bool(_scan_result_value(results.scan, "ok", False))
            else ScannerExecutionStatus.UNPARSEABLE
        )

    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        policy = task_policies.get(task_id)
        task_scan_evidence = results.scan_evidence_by_task.get(task_id)
        scan_evidence_complete = (
            task_scan_evidence.complete
            if task_scan_evidence is not None
            else False
            if results.scan_evidence is not None
            else None
        )
        task_scan_projection_missing = (
            task_scan_evidence is None
            and results.scan_evidence is not None
            and results.scan_evidence.complete
        )
        target_package_absent = (
            group.issue_type == IssueType.SCA
            and results.scan_target_skipped_by_task.get(task_id)
            == ScanFallbackReason.NO_MATCHING_TARGET.value
            and task_id not in results.scan_target_failures_by_task
            and task_scan_evidence is not None
            and task_scan_evidence.complete
            and task_id in task_scan_evidence.covered_task_ids
            and task_scan_evidence.scan_skip_reason == ScanFallbackReason.NO_MATCHING_TARGET.value
        )
        effective_scanner_status = (
            ScannerExecutionStatus.SUCCESS
            if target_package_absent
            or (results.scan_skipped and policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL)
            else scanner_status
        )
        remaining = (
            sorted(group_target_identifiers(group) & remaining_global)
            if target_package_absent
            else sorted(task_scan_evidence.remaining_target_identifiers)
            if task_scan_evidence is not None
            else []
            if results.scan_evidence is not None
            else sorted(group_target_identifiers(group) & remaining_global)
        )
        package_state = results.package_state_by_task.get(task_id, _QAPackageState())
        dependency_evidence = package_state.dependency_evidence
        package_state_checks_enabled = _package_state_checks_enabled(policy)
        target_cleared = (
            False
            if task_scan_projection_missing
            else None
            if scan_evidence_complete is False
            else effective_scanner_status == ScannerExecutionStatus.SUCCESS and not remaining
        )
        diagnostics: list[str] = []
        if not install_passed:
            diagnostics.append(install_summary)
        if effective_scanner_status != ScannerExecutionStatus.SUCCESS:
            diagnostics.append(
                f"Scanner execution status: {effective_scanner_status.value}. "
                f"{_scan_result_value(results.scan, 'summary', 'scanner did not run')}"
            )
        if target_package_absent:
            diagnostics.append(
                "Targeted ODC was skipped because complete lockfile evidence proved "
                "the target package absent."
            )
        elif task_scan_projection_missing:
            diagnostics.append("Task-specific scanner occurrence coverage is incomplete.")
        elif scan_evidence_complete is False:
            diagnostics.append(
                "ODC scan evidence was incomplete; target clearance is inconclusive."
            )
        if package_state_checks_enabled and not target_package_absent:
            diagnostics.extend(package_state.diagnostics)
        if (
            package_state_checks_enabled
            and policy == QAPolicy.VERSION_BUMP
            and not target_package_absent
            and dependency_evidence is None
        ):
            diagnostics.append("Deterministic dependency evidence was not collected.")
        if policy is None:
            diagnostics.append("QA policy provenance is missing or invalid.")
            errors.append(
                f"qa_critic policy evaluator: task '{task_id}' has missing or invalid policy provenance."
            )

        deterministic_pass = policy is not None and install_passed
        if policy in _STRICT_SCANNER_QA_POLICIES:
            deterministic_pass = deterministic_pass and target_cleared
        if policy in _HARD_TEST_QA_POLICIES:
            deterministic_pass = deterministic_pass and tests_passed is True
        if (
            package_state_checks_enabled
            and policy == QAPolicy.VERSION_BUMP
            and not target_package_absent
        ):
            deterministic_pass = (
                deterministic_pass
                and dependency_evidence is not None
                and dependency_evidence.status == DependencyEvidenceStatus.VERIFIED
            )
        if package_state_checks_enabled and not target_package_absent:
            if policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL:
                deterministic_pass = deterministic_pass and package_state.manifest_state == "absent"
                deterministic_pass = deterministic_pass and package_state.graph_state == "absent"
            if policy == QAPolicy.NO_FIX_CODE_REMOVAL:
                deterministic_pass = (
                    deterministic_pass and package_state.manifest_state == "present"
                )
                deterministic_pass = deterministic_pass and package_state.graph_state == "present"
        gates_by_task[task_id] = QADeterministicGates(
            dependency_evidence=dependency_evidence,
            status="pass" if deterministic_pass else "fail",
            install_passed=install_passed,
            scanner_execution_status=effective_scanner_status,
            target_remaining_identifiers=remaining,
            target_scanner_cleared=target_cleared,
            scan_evidence_complete=scan_evidence_complete,
            target_package_absent=target_package_absent,
            tests_passed=tests_passed,
            package_manifest_state=package_state.manifest_state,
            package_graph_state=package_state.graph_state,
            install_error_category=results.install_error_category,
            peer_conflicts=list(results.peer_conflicts),
            diagnostics=diagnostics,
        )
    return gates_by_task, errors


def _semantic_review_is_evidence_backed(
    review: QASemanticSecurityReview | None,
    investigation: GroupInvestigation | None,
) -> bool:
    """Require structured review fields plus actual successful review-tool evidence."""
    return bool(
        review is not None
        and review.verdict == SecurityReviewVerdict.PASS
        and review.reasoning.strip()
        and review.evidence_refs
        and all(reference.strip() for reference in review.evidence_refs)
        and investigation is not None
        and investigation.source_review_evidence
        and not investigation.fallback
        and investigation.structured_review_verdict == SecurityReviewVerdict.PASS
    )


def _qa_evaluation_items(
    source: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
) -> list[QAEvaluation]:
    """Normalize evaluator output into a list of QAEvaluation items."""
    if isinstance(source, Mapping):
        return list(source.values())
    return list(source)


def _install_conflict_is_terminal(
    gates: QADeterministicGates,
    install_error_category: str | None,
) -> bool:
    """Return whether a failed install has a deterministic dependency conflict."""
    if gates.install_passed:
        return False
    if str(install_error_category or "").upper() in {"PEER_CONFLICT", "ENGINE_CONFLICT"}:
        return True
    install_summary = gates.diagnostics[0] if gates.diagnostics else ""
    conflict_markers = (*_PEER_CONFLICT_PATTERNS, *_ENGINE_CONFLICT_PATTERNS)
    return any(marker.casefold() in install_summary.casefold() for marker in conflict_markers)


def _install_conflict_retry_feedback(gates: QADeterministicGates) -> str:
    """Build retry guidance without claiming unavailable post-install validation."""
    install_summary = (
        gates.diagnostics[0] if gates.diagnostics else "Dependency installation failed."
    )
    return (
        f"{install_summary} Dependency installation failed before post-install QA validation; "
        "resolve the dependency conflict before retrying."
    )


def _disabled_package_state_failure(
    evaluation: QAEvaluation,
    policy: QAPolicy | None,
) -> bool:
    """Return whether a security failure cites temporarily disabled package-state checks."""
    if _package_state_checks_enabled(policy) or policy not in {
        QAPolicy.VERSION_BUMP,
        QAPolicy.NO_FIX_PACKAGE_REMOVAL,
        QAPolicy.NO_FIX_CODE_REMOVAL,
    }:
        return False
    if evaluation.failure_category != FailureCategory.SECURITY_FLAG:
        return False
    feedback = (evaluation.retry_feedback or "").casefold()
    package_state_terms = (
        "manifest",
        "lockfile",
        "package.json",
        "package-lock.json",
        "resolved dependency graph",
        "installed graph",
        "dependency evidence",
        "dependency state",
        "dependency version",
        "package version",
        "selected version",
    )
    return any(term in feedback for term in package_state_terms)


def _known_pastebin_test_failure(evaluation: QAEvaluation) -> bool:
    """Recognize only the Juice Shop's external PasteBin test outage.

    Args:
        evaluation: Structured QA failure with test attribution evidence.

    Returns:
        True when every failed test and source location belongs to the known
        PasteBin-backed Internet Resources test. Other failures remain blocking.
    """
    evidence = evaluation.failure_evidence
    if (
        evaluation.failure_category != FailureCategory.BREAKING_CHANGE
        or evidence is None
        or not evidence.failed_tests
        or not evidence.affected_files
        or not evidence.source_locations
    ):
        return False
    references = (*evidence.failed_tests, *evidence.exact_diagnostics)
    if not any("pastebin.com" in value.casefold() for value in references):
        return False
    if any(
        "pastebin" not in test_name.casefold()
        and test_name.strip().casefold() != "required internet resource"
        for test_name in evidence.failed_tests
    ):
        return False
    expected_path = "test/api/internet-resources.test.ts"
    paths = (*evidence.affected_files, *evidence.source_locations)
    return all(
        path.replace("\\", "/").casefold().split(":", 1)[0].endswith(expected_path)
        for path in paths
    )


def _version_bump_llm_failure_is_relevant(
    evaluation: QAEvaluation,
    gates: QADeterministicGates,
    *,
    test_exonerated: bool,
) -> bool:
    """Return whether an LLM failure adds a version-bump policy failure."""
    if evaluation.passed:
        return False
    if evaluation.failure_category == FailureCategory.BREAKING_CHANGE:
        return gates.tests_passed is False and not test_exonerated
    if evaluation.failure_category == FailureCategory.SECURITY_FLAG:
        feedback = (evaluation.retry_feedback or "").casefold()
        dependency_terms = (
            "manifest",
            "lockfile",
            "package.json",
            "package-lock.json",
            "dependency",
        )
        unavailable_terms = (
            "missing",
            "unavailable",
            "could not",
            "cannot",
            "unable",
            "no deterministic",
            "not found",
            "no match",
            "compacted",
            "omitted",
            "not provided",
            "not visible",
            "truncated",
            "not shown",
            "insufficient",
        )
        if any(term in feedback for term in dependency_terms) and any(
            term in feedback for term in unavailable_terms
        ):
            return False
    return True


def _normalize_singleton_test_attribution(
    evaluation: QAEvaluation,
    context: QATaskContext,
    gates: QADeterministicGates,
    *,
    deterministic_test_evidence: QAFailureEvidence | None = None,
) -> QAEvaluation:
    """Assign an executed failed test run to the only active task.

    Singleton Supervisor dispatch makes the causal scope deterministic: the
    current task's committed worker attempt is the only remediation action
    between the workspace snapshot and QA. This normalization prevents the
    evaluator from converting that deterministic fact into an inconclusive or
    cross-task exoneration verdict.

    Args:
        evaluation: LLM-owned evaluation before policy guardrails.
        context: The active task and parent vulnerability group.
        gates: Python-owned deterministic QA gates.
        deterministic_test_evidence: Optional parsed failed-test evidence.

    Returns:
        The evaluation with singleton responsibility recorded when applicable.
    """
    if (
        gates.tests_passed is not False
        or evaluation.contract_error
        or evaluation.evidence_inconclusive
    ):
        return evaluation

    failed_tests: list[str] = []
    if deterministic_test_evidence is not None:
        failed_tests.extend(deterministic_test_evidence.failed_tests[:10])
    if not failed_tests and evaluation.test_attribution is not None:
        failed_tests.extend(evaluation.test_attribution.failed_tests[:10])
    if not failed_tests:
        failed_tests.append("deterministic unit test suite")

    attribution = QATestAttribution(
        verdict=TestAttributionVerdict.RESPONSIBLE,
        responsible_group_ids=[context.group.group_id],
        failed_tests=failed_tests,
        reasoning=(
            "Singleton QA scope: the deterministic test suite ran after this task's "
            "committed worker attempt, so the failed test outcome is attributed to "
            "the active task by orchestration contract."
        ),
    )
    return evaluation.model_copy(update={"test_attribution": attribution})


def _apply_policy_decision(
    task_contexts: list[QATaskContext],
    batch_result: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
    gates_by_task: dict[str, QADeterministicGates],
    task_policies: dict[str, QAPolicy | None],
    investigations_by_task: dict[str, GroupInvestigation] | None = None,
    install_error_category: str | None = None,
    *,
    terminal_install_conflict: bool = True,
    singleton_scope: bool = False,
    deterministic_test_evidence: QAFailureEvidence | None = None,
) -> tuple[dict[str, QAEvaluation], list[str]]:
    """Apply the Supervisor-owned policy matrix to task-keyed evaluations.

    Args:
        task_contexts: Authoritative active task/group contexts.
        batch_result: Structured evaluator results.
        gates_by_task: Python-owned deterministic gates by task ID.
        task_policies: Supervisor-owned QA policies by task ID.
        investigations_by_task: Optional evaluator investigation metadata.
        install_error_category: Deterministic dependency-install error category.
        terminal_install_conflict: Whether dependency conflicts terminate QA.
        singleton_scope: Whether exactly one active task owns this QA run.
        deterministic_test_evidence: Optional parsed failed-test evidence.

    Returns:
        Normalized task evaluations and guardrail diagnostics.
    """
    known_task_ids = {context.task_id for context in task_contexts}
    known_group_ids = {context.group.group_id for context in task_contexts}
    errors: list[str] = []
    normalized: dict[str, QAEvaluation] = {}
    missing_ids: set[str] = set()
    for evaluation in _qa_evaluation_items(batch_result):
        task_id = evaluation.task_id
        if task_id not in known_task_ids:
            errors.append(f"qa_critic policy evaluator: unknown task '{task_id}' dropped.")
        elif task_id in normalized:
            errors.append(
                f"qa_critic policy evaluator: duplicate evaluation for '{task_id}' dropped."
            )
        else:
            normalized[task_id] = evaluation
    for context in task_contexts:
        task_id = context.task_id
        if task_id not in normalized:
            missing_ids.add(task_id)
            errors.append(f"qa_critic policy evaluator: missing evaluation for '{task_id}'.")
            normalized[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                contract_error=True,
                contract_error_reason="Structured QA evaluator omitted this task.",
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Structured QA evaluator omitted this task; retry required.",
            )

    final: dict[str, QAEvaluation] = {}
    for context in task_contexts:
        task_id = context.task_id
        group = context.group
        policy = task_policies.get(task_id)
        current = normalized[task_id]
        gates = gates_by_task[task_id]
        if singleton_scope and len(task_contexts) == 1:
            current = _normalize_singleton_test_attribution(
                current,
                context,
                gates,
                deterministic_test_evidence=deterministic_test_evidence,
            )
        if (
            not current.contract_error
            and policy is not None
            and group.issue_type == IssueType.SCA
            and gates.target_package_absent
            and gates.install_passed
            and gates.scanner_execution_status == ScannerExecutionStatus.SUCCESS
            and gates.target_scanner_cleared is True
            and not gates.target_remaining_identifiers
        ):
            attribution = current.test_attribution
            version_bump_test_exonerated = (
                policy == QAPolicy.VERSION_BUMP
                and gates.tests_passed is False
                and not current.contract_error
                and (
                    _known_pastebin_test_failure(current)
                    or (
                        attribution is not None
                        and attribution.verdict == TestAttributionVerdict.EXONERATED
                        and _valid_test_attribution(current, group.group_id, known_group_ids)
                    )
                )
            )
            tests_satisfied = (
                gates.tests_passed is True
                or version_bump_test_exonerated
                or (policy not in _HARD_TEST_QA_POLICIES and policy != QAPolicy.VERSION_BUMP)
            )
            if tests_satisfied:
                final[task_id] = QAEvaluation(
                    task_id=task_id,
                    passed=True,
                    deterministic_gates=gates,
                    test_attribution=attribution,
                )
                continue
        if current.contract_error:
            reason = (
                current.contract_error_reason.strip()
                or "The structured QA result could not be validated."
            )
            errors.append(f"qa_critic contract error for '{task_id}': {reason}")
            final[task_id] = current.model_copy(
                update={
                    "task_id": task_id,
                    "passed": False,
                    "deterministic_gates": gates,
                    "failure_category": FailureCategory.SECURITY_FLAG,
                    "retry_feedback": (
                        "QA result is inconclusive because the structured QA result could not be validated. "
                        "No remediation retry was consumed. Re-run QA after correcting the judge contract."
                    ),
                }
            )
            continue

        if terminal_install_conflict and _install_conflict_is_terminal(
            gates, install_error_category
        ):
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=FailureCategory.PEER_CONFLICT,
                retry_feedback=_install_conflict_retry_feedback(gates),
                deterministic_gates=gates,
            )
            continue

        failures: list[tuple[FailureCategory, str]] = []
        evaluator_test_exonerated = (
            policy == QAPolicy.VERSION_BUMP
            and gates.tests_passed is False
            and current.failure_category == FailureCategory.BREAKING_CHANGE
            and (
                _known_pastebin_test_failure(current)
                or (
                    current.test_attribution is not None
                    and _valid_test_attribution(current, group.group_id, known_group_ids)
                )
            )
        )
        dependency_evidence_inconclusive = (
            _package_state_checks_enabled(policy)
            and policy == QAPolicy.VERSION_BUMP
            and not gates.target_package_absent
            and (
                gates.dependency_evidence is None
                or gates.dependency_evidence.status == DependencyEvidenceStatus.INCONCLUSIVE
            )
        )
        scan_evidence_inconclusive = (
            policy in _STRICT_SCANNER_QA_POLICIES and gates.scan_evidence_complete is False
        )
        if policy == QAPolicy.VERSION_BUMP:
            if not _disabled_package_state_failure(
                current, policy
            ) and _version_bump_llm_failure_is_relevant(
                current,
                gates,
                test_exonerated=evaluator_test_exonerated,
            ):
                failures.append(
                    (
                        current.failure_category or FailureCategory.SECURITY_FLAG,
                        current.retry_feedback or "The structured QA evaluator failed this task.",
                    )
                )
        elif (
            not current.passed
            and not evaluator_test_exonerated
            and not _disabled_package_state_failure(current, policy)
        ):
            failures.append(
                (
                    current.failure_category or FailureCategory.SECURITY_FLAG,
                    current.retry_feedback or "The structured QA evaluator failed this task.",
                )
            )
        if task_id in missing_ids:
            failures.append(
                (FailureCategory.SECURITY_FLAG, "Structured QA evaluator omitted this task.")
            )
        if not gates.install_passed:
            install_text = " ".join(gates.diagnostics)
            category = (
                FailureCategory.PEER_CONFLICT
                if any(
                    pattern.lower() in install_text.lower() for pattern in _PEER_CONFLICT_PATTERNS
                )
                else FailureCategory.SECURITY_FLAG
            )
            failures.append(
                (
                    category,
                    "Shared dependency installation failed; every task in this QA batch is failed.",
                )
            )
        if policy is None:
            failures.append(
                (FailureCategory.SECURITY_FLAG, "Missing or invalid QA policy provenance.")
            )
        if policy in _STRICT_SCANNER_QA_POLICIES:
            if gates.scanner_execution_status != ScannerExecutionStatus.SUCCESS:
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "Strict scanner policy requires a trustworthy parseable scanner report.",
                    )
                )
            if gates.target_remaining_identifiers:
                if current.passed or current.failure_category == FailureCategory.BREAKING_CHANGE:
                    errors.append(
                        f"qa_critic guardrail: task '{task_id}' has remaining scanner "
                        "identifiers but judge did not prioritize SECURITY_FLAG; forcing "
                        "passed=False / SECURITY_FLAG."
                    )
                failures.append(
                    (
                        FailureCategory.SECURITY_FLAG,
                        "Remaining target scanner identifiers: "
                        + ", ".join(gates.target_remaining_identifiers),
                    )
                )
        if (
            _package_state_checks_enabled(policy)
            and policy == QAPolicy.VERSION_BUMP
            and not gates.target_package_absent
            and gates.dependency_evidence is not None
            and gates.dependency_evidence.status == DependencyEvidenceStatus.MISMATCH
        ):
            failures.append(
                (
                    FailureCategory.SECURITY_FLAG,
                    "Deterministic dependency evidence does not match the Supervisor-selected target.",
                )
            )
        if policy in _HARD_TEST_QA_POLICIES and gates.tests_passed is not True:
            failures.append(
                (FailureCategory.BREAKING_CHANGE, "The required global test suite did not pass.")
            )
        elif policy == QAPolicy.VERSION_BUMP and gates.tests_passed is not True:
            attribution = current.test_attribution
            valid_exoneration = _known_pastebin_test_failure(current) or bool(
                attribution
                and attribution.verdict == TestAttributionVerdict.EXONERATED
                and attribution.responsible_group_ids
                and all(
                    identifier in known_group_ids
                    for identifier in attribution.responsible_group_ids
                )
                and gates.tests_passed is False
                and attribution.failed_tests
                and attribution.reasoning.strip()
                and group.group_id not in attribution.responsible_group_ids
            )
            if not valid_exoneration:
                failures.append(
                    (
                        FailureCategory.BREAKING_CHANGE,
                        "VERSION_BUMP tests failed without valid structured exoneration evidence.",
                    )
                )
        if _package_state_checks_enabled(policy) and not gates.target_package_absent:
            if policy == QAPolicy.NO_FIX_PACKAGE_REMOVAL:
                if gates.package_manifest_state != "absent":
                    failures.append(
                        (
                            FailureCategory.SECURITY_FLAG,
                            "NO_FIX Stage 1 requires the package to be absent from every authorized direct manifest.",
                        )
                    )
                if gates.package_graph_state != "absent":
                    failures.append(
                        (
                            FailureCategory.SECURITY_FLAG,
                            "NO_FIX Stage 1 requires the package to be absent from the resolved dependency graph.",
                        )
                    )
            if policy == QAPolicy.NO_FIX_CODE_REMOVAL:
                if gates.package_manifest_state != "present":
                    failures.append(
                        (
                            FailureCategory.SECURITY_FLAG,
                            "NO_FIX Stage 2 requires the package to remain present in the direct manifest.",
                        )
                    )
                if gates.package_graph_state != "present":
                    failures.append(
                        (
                            FailureCategory.SECURITY_FLAG,
                            "NO_FIX Stage 2 requires the package to remain present in the resolved dependency graph.",
                        )
                    )

        semantic_review = current.semantic_security_review

        if (
            policy in _REQUIRED_SEMANTIC_QA_POLICIES
            and not gates.target_package_absent
            and not _semantic_review_is_evidence_backed(
                semantic_review, (investigations_by_task or {}).get(task_id)
            )
        ):
            if semantic_review is None or semantic_review.verdict == SecurityReviewVerdict.PASS:
                semantic_review = QASemanticSecurityReview(
                    verdict=SecurityReviewVerdict.INCONCLUSIVE,
                    reasoning="Required semantic review could not be verified from investigator evidence and structured review output.",
                )
            failures.append(
                (
                    FailureCategory.SECURITY_FLAG,
                    "Required semantic security review is missing, inconclusive, or lacks source/diff evidence.",
                )
            )

        # An unavailable dependency graph is a QA-only rerun when it is the
        # sole blocker. Preserve actionable install, scanner, test, and
        # semantic failures so the Supervisor can repair the candidate.
        if (dependency_evidence_inconclusive or scan_evidence_inconclusive) and not failures:
            if dependency_evidence_inconclusive and scan_evidence_inconclusive:
                evidence_diagnostic = (
                    "Dependency and ODC scan evidence were unavailable or incomplete."
                )
            elif scan_evidence_inconclusive:
                evidence_diagnostic = "ODC scan evidence was incomplete."
            else:
                evidence_diagnostic = (
                    "Deterministic dependency evidence was unavailable or incomplete."
                )
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback=(
                    f"{evidence_diagnostic} "
                    "Re-run QA evidence collection; no remediation retry should be consumed."
                ),
                failure_evidence=current.failure_evidence,
                deterministic_gates=gates,
                evidence_inconclusive=True,
                test_attribution=current.test_attribution,
            )
            continue

        category = next(
            (
                candidate
                for candidate in (
                    FailureCategory.SECURITY_FLAG,
                    FailureCategory.PEER_CONFLICT,
                    FailureCategory.BREAKING_CHANGE,
                )
                if any(item[0] == candidate for item in failures)
            ),
            None,
        )
        diagnostics = [message for _category, message in failures]
        if current.retry_feedback and not current.passed:
            diagnostics.append(current.retry_feedback)
        if not failures:
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=True,
                deterministic_gates=gates,
                semantic_security_review=semantic_review,
                test_attribution=current.test_attribution,
            )
        else:
            final[task_id] = QAEvaluation(
                task_id=task_id,
                passed=False,
                failure_category=category or FailureCategory.SECURITY_FLAG,
                retry_feedback=" ".join(dict.fromkeys(diagnostics))
                or "QA policy gate failed; retry required.",
                failure_evidence=current.failure_evidence,
                deterministic_gates=gates,
                semantic_security_review=semantic_review,
                test_attribution=current.test_attribution,
            )
    return final, errors


def _apply_guardrails(
    task_contexts: list[QATaskContext],
    batch_result: Mapping[str, QAEvaluation] | Iterable[QAEvaluation],
    results: _QAExecutionResults,
    task_policies: dict[str, QAPolicy | None],
    investigations_by_task: dict[str, GroupInvestigation] | None = None,
    singleton_scope: bool = False,
    deterministic_test_evidence: QAFailureEvidence | None = None,
) -> tuple[dict[str, QAEvaluation], list[str]]:
    """Apply deterministic QA guardrails to task-keyed evaluations.

    Unknown or duplicate task evaluations are rejected, missing task results
    become inconclusive failures, and policy gates enforce scanner,
    package-state, test, and review requirements. When explicitly enabled for
    a singleton dispatch, failed test attribution is normalized to the active
    task before policy decisions are made.

    Args:
        task_contexts: Authoritative active task/group contexts.
        batch_result: Structured evaluator results.
        results: Deterministic global QA results.
        task_policies: Supervisor-owned QA policies by task ID.
        investigations_by_task: Optional evaluator investigation metadata.
        singleton_scope: Whether exactly one active task owns this QA run.
        deterministic_test_evidence: Optional parsed failed-test evidence.

    Returns:
        Normalized evaluations and guardrail diagnostics.
    """
    gates, gate_errors = _evaluate_policy_gates(task_contexts, results, task_policies)
    evaluations, decision_errors = _apply_policy_decision(
        task_contexts,
        batch_result,
        gates,
        task_policies,
        investigations_by_task,
        install_error_category=results.install_error_category,
        singleton_scope=singleton_scope and len(task_contexts) == 1,
        deterministic_test_evidence=deterministic_test_evidence,
    )
    return evaluations, gate_errors + decision_errors


def _extract_deterministic_test_evidence(
    results: _QAExecutionResults,
    *,
    sandbox: DockerSandbox | None = None,
) -> QAFailureEvidence | None:
    """Extract failure evidence from the captured test process streams."""
    if results.tests is None or results.tests[0]:
        return None
    exit_code = results.test_exit_code
    stdout = results.test_raw_stdout
    stderr = results.test_raw_stderr
    if stdout is None:
        stdout = results.tests[1]
    if stderr is None:
        stderr = ""
    from remediation_engine.tools.test_execution import extract_qa_failure_evidence

    return extract_qa_failure_evidence(
        exit_code if exit_code is not None else 1,
        stdout,
        stderr,
        sandbox=sandbox,
    )


def _attach_failure_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    results: _QAExecutionResults,
    state: OrchestratorState,
    *,
    deterministic_evidence: QAFailureEvidence | None = None,
    sandbox: DockerSandbox | None = None,
) -> dict[str, QAEvaluation]:
    """Attach deterministic failure evidence and committed attempt provenance.

    The structured evaluator classifies the outcome, while deterministic test
    parsing supplies failing output and source paths. The task's committed
    attempt envelope supplies the authoritative ``attempt_id`` so retry
    routing receives actionable evidence instead of inferred locations.

    Args:
        evaluations: Structured evaluations keyed by task identifier.
        results: Deterministic QA execution results.
        state: Current orchestration state with task and attempt projections.
        deterministic_evidence: Previously extracted test evidence, if any.
        sandbox: Optional active sandbox for lazy evidence extraction.

    Returns:
        Evaluations enriched with deterministic evidence and attempt identity.
    """
    test_evidence = deterministic_evidence
    if test_evidence is None:
        test_evidence = _extract_deterministic_test_evidence(results, sandbox=sandbox)

    task_queue = state.get("task_queue", {}) or {}
    enriched: dict[str, QAEvaluation] = {}
    for task_id, evaluation in evaluations.items():
        if evaluation.task_id != task_id:
            logger.warning(
                "QA discarded evaluation evidence for key %r because the "
                "evaluation identifies task %r.",
                task_id,
                evaluation.task_id,
            )
            enriched[task_id] = evaluation.model_copy(update={"failure_evidence": None})
            continue
        if evaluation.passed:
            enriched[task_id] = evaluation
            continue
        llm_evidence = evaluation.failure_evidence
        evidence = test_evidence

        task = task_queue.get(task_id)
        attempt_id = getattr(task, "current_attempt_id", None) if task is not None else None
        task_revision = int(getattr(task, "task_revision", 0) or 0) if task is not None else 0
        snapshots = state.get("attempt_snapshots_by_id", {}) or {}
        snapshot = snapshots.get(attempt_id) if attempt_id else None
        snapshot_task_id = (
            snapshot.get("task_id")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "task_id", None)
            if snapshot is not None
            else None
        )
        snapshot_attempt_id = (
            snapshot.get("attempt_id")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "attempt_id", None)
            if snapshot is not None
            else None
        )
        snapshot_revision = (
            snapshot.get("task_revision")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "task_revision", None)
            if snapshot is not None
            else None
        )
        snapshot_matches = False
        if snapshot is not None:
            try:
                snapshot_matches = int(snapshot_revision) == task_revision
            except (TypeError, ValueError):
                snapshot_matches = False
        if (
            task is None
            or not attempt_id
            or snapshot is None
            or snapshot_task_id != task_id
            or snapshot_attempt_id != attempt_id
            or not snapshot_matches
        ):
            if evidence is not None or llm_evidence is not None:
                logger.warning(
                    "QA discarded failure evidence for task %r because its current "
                    "attempt snapshot was missing or contradictory.",
                    task_id,
                )
            enriched[task_id] = evaluation.model_copy(update={"failure_evidence": None})
            continue
        if evidence is not None and llm_evidence is not None and llm_evidence.source_locations:
            discarded_locations = ", ".join(llm_evidence.source_locations[:5])
            if len(llm_evidence.source_locations) > 5:
                discarded_locations += ", ..."
            diagnostic = (
                "QA discarded LLM-supplied source location(s) because deterministic "
                f"test evidence is authoritative: {discarded_locations}."
            )
            logger.warning(diagnostic)
            evidence = evidence.model_copy(
                update={
                    "exact_diagnostics": [
                        *evidence.exact_diagnostics,
                        diagnostic,
                    ][:15]
                }
            )
        elif evidence is None and llm_evidence is not None:
            diagnostic = (
                "QA discarded LLM-supplied source locations because deterministic "
                "test evidence was unavailable."
            )
            logger.warning(diagnostic)
            evidence = llm_evidence.model_copy(
                update={
                    "source_locations": [],
                    "affected_files": [],
                    "exact_diagnostics": [
                        *llm_evidence.exact_diagnostics,
                        diagnostic,
                    ][:15],
                }
            )
        if evidence is not None:
            evidence = evidence.model_copy(
                update={
                    "attempt_id": attempt_id,
                    "task_revision": task_revision,
                }
            )
            evaluation = evaluation.model_copy(update={"failure_evidence": evidence})
        enriched[task_id] = evaluation
    return enriched


def _attach_scan_evidence_to_evaluations(
    evaluations: dict[str, QAEvaluation],
    evidence: ODCScanEvidence | None,
    task_contexts: list[QATaskContext],
    task_failures: Mapping[str, tuple[ScanFallbackReason, str]] | None = None,
    *,
    evidence_by_task: Mapping[str, ODCScanEvidence] | None = None,
) -> dict[str, QAEvaluation]:
    """Attach task-owned occurrence evidence to each matching evaluation.

    When the artifact manifest produced an exact task projection, it is
    authoritative: tasks without an entry do not inherit the batch-wide
    summary. The legacy batch projection remains for callers without a
    manifest, while preserving findings outside the task's original target IDs.
    """
    if evidence_by_task is not None:
        enriched: dict[str, QAEvaluation] = {}
        for task_id, evaluation in evaluations.items():
            if evaluation.task_id != task_id:
                logger.warning(
                    "QA skipped scan evidence for key %r because the evaluation identifies task %r.",
                    task_id,
                    evaluation.task_id,
                )
                enriched[task_id] = evaluation
                continue
            task_evidence = evidence_by_task.get(task_id)
            if task_evidence is None:
                enriched[task_id] = evaluation
                continue
            task_failure = (task_failures or {}).get(task_id)
            task_complete = task_failure is None and task_evidence.complete
            scoped_evidence = task_evidence.model_copy(
                update={
                    "covered_task_ids": [task_id],
                    "complete": task_complete,
                    "fallback_reason": (
                        task_failure[0]
                        if task_failure is not None
                        else task_evidence.fallback_reason
                        if not task_complete
                        else None
                    ),
                }
            )
            enriched[task_id] = evaluation.model_copy(update={"scan_evidence": scoped_evidence})
        return enriched

    if evidence is None:
        return evaluations
    groups_by_task = {context.task_id: context.group for context in task_contexts}
    enriched = {}
    for task_id, evaluation in evaluations.items():
        if evaluation.task_id != task_id:
            logger.warning(
                "QA skipped scan evidence for key %r because the evaluation identifies task %r.",
                task_id,
                evaluation.task_id,
            )
            enriched[task_id] = evaluation
            continue
        group = groups_by_task.get(task_id)
        if group is None:
            enriched[task_id] = evaluation
            continue
        target_ids = group_target_identifiers(group)
        task_failure = (task_failures or {}).get(task_id)
        task_coverage_complete = task_failure is None and (
            evidence.complete or task_id in evidence.covered_task_ids
        )
        scoped_evidence = evidence.model_copy(
            update={
                "covered_task_ids": [task_id],
                "remaining_target_identifiers": sorted(
                    set(evidence.remaining_target_identifiers) & target_ids
                ),
                "complete": task_coverage_complete,
                "fallback_reason": (
                    None
                    if task_coverage_complete
                    else task_failure[0]
                    if task_failure is not None
                    else evidence.fallback_reason
                ),
            }
        )
        enriched[task_id] = evaluation.model_copy(update={"scan_evidence": scoped_evidence})
    return enriched
