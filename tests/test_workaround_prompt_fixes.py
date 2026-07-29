"""Regression tests for workaround prompt context and QA feedback extraction."""

from __future__ import annotations

from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundContext,
    WorkaroundPhase,
)
from remediation_engine.orchestration.remedy_tools import (
    _make_inspect_ast_symbol_tool,
)
from remediation_engine.orchestration.workaround_subagent import (
    _build_workaround_prompt,
    _create_skinny_subagent_group,
    _extract_vulnerability_mechanism,
)


def _task_and_group() -> tuple[RemediationTask, VulnerabilityGroup]:
    """Build a representative SCA task and finding for prompt tests."""
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        cve_id="CVE-2020-15084",
        package_name="express-jwt",
        message=(
            "### Overview\n"
            "Versions before and including 5.3.3 do not enforce the algorithms "
            "entry. Without algorithms, a jwks-rsa secret can lead to "
            "authorization bypass.\n\n"
            "### Am I affected?\nDetails omitted.\n\n"
            "### How to fix that?\nSpecify algorithms."
        ),
    )
    group = VulnerabilityGroup(
        group_id="grp-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2020-15084"],
        representative_issue_id=issue.id,
        issues=[issue],
    )
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the express-jwt workaround.",
    )
    return task, group


def test_qa_query_uses_the_diagnostic_runtime_error() -> None:
    """The targeted search query should contain the actionable QA error."""
    task, group = _task_and_group()
    feedback = (
        "The API tests failed after the update: "
        "`(0 , import_express_jwt.default) is not a function`."
    )

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        previous_feedback=feedback,
    )

    assert "(0 , import_express_jwt.default) is not a function" in prompt
    assert "construct the query yourself" in prompt
    assert "update_mitigates_cve_but_breaks_tests" in prompt
    assert "=== RECOMMENDED INITIAL SEARCH QUERY ===" in prompt
    assert "migration breaking changes compatibility" in prompt


def test_initial_query_uses_advisory_identifier_and_mechanism() -> None:
    """Initial mitigation should start with the advisory and vulnerable mechanism."""
    task, group = _task_and_group()

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
    )

    assert (
        "Initial evidence-based classification to confirm or reject: initial_code_workaround_or_isolation"
        in prompt
    )
    assert "CVE-2020-15084" in prompt
    assert "security advisory" in prompt
    assert "mitigation" in prompt


def test_scanner_failure_query_prioritizes_unresolved_finding() -> None:
    """Scanner failures should search the advisory mechanism before migration guidance."""
    task, group = _task_and_group()
    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        previous_feedback=(
            "Dependency scanner still reports CVE-2020-15084; remaining scanner "
            "findings require a source-level mitigation."
        ),
    )

    assert (
        "Initial evidence-based classification to confirm or reject: update_does_not_resolve_scanner_findings"
        in prompt
    )
    assert "still vulnerable" in prompt
    assert "source-level mitigation" in prompt


def test_qa_query_prefers_structured_diagnostic_over_generic_retry_feedback() -> None:
    """Search extraction must use the exact QA diagnostic when both forms exist."""
    task, group = _task_and_group()
    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        previous_feedback="Generic QA retry guidance with no useful error details.",
        workaround_context=WorkaroundContext(
            phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
            qa_evidence=QAFailureEvidence(
                exact_diagnostics=["express-jwt: algorithms is a required option"],
                attempt_id="attempt-1",
                task_revision=2,
            ),
        ),
    )

    assert "express-jwt: algorithms is a required option" in prompt
    assert "Generic QA retry guidance" not in prompt


def test_skinny_prompt_preserves_vulnerability_mechanism() -> None:
    """The skinny execution group still receives the vulnerability mechanism."""
    task, group = _task_and_group()
    mechanism = _extract_vulnerability_mechanism(group)
    skinny_group = _create_skinny_subagent_group(group)

    assert skinny_group.issues == []
    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=skinny_group,
        vulnerability_mechanism=mechanism,
    )

    assert "Vulnerability Mechanism:" in prompt
    assert "algorithms" in prompt
    assert "authorization bypass" in prompt


def test_ast_lookup_error_explains_declared_symbol_requirement() -> None:
    """A missing AST symbol should tell the worker how to recover."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    sandbox = MagicMock()
    sandbox.read_file.return_value = "import expressJwt from 'express-jwt';"
    tool = _make_inspect_ast_symbol_tool(sandbox)

    with (
        patch(
            "remediation_engine.tools.code_map.language_for_path",
            return_value="typescript",
        ),
        patch(
            "remediation_engine.tools.code_map.parse_source",
            return_value=SimpleNamespace(root_node=object()),
        ),
        patch(
            "remediation_engine.tools.code_map.find_named_symbol",
            return_value=None,
        ),
    ):
        result = tool.invoke({"file_path": "lib/insecurity.ts", "symbol_name": "expressJwt"})

    assert "declared function, class, or method" in result
    assert "Imported identifiers and package names are not AST symbols" in result
    assert "Do not retry the same symbol" in result
    assert "read_workspace_file" in result
