from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import QAFailureEvidence, WorkaroundContext
from remediation_engine.orchestration.workaround_subagent import (
    WorkaroundPhase,
    _build_workaround_prompt,
    _workaround_search_recommendation,
)


def test_compact_initial_mitigation_prompt():
    task = MagicMock(task_id="task-1", selected_version="1.2.3", instruction="Fix vulnerability")
    group = MagicMock(
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2026-1234"],
        ghsa_ids=[],
        fix_plan=MagicMock(
            status=MagicMock(value="no_fix"), fixed_version=None, workaround_snippets=["snippet1"]
        ),
    )
    group.issues = []

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        workaround_context=None,
    )

    assert "WORKFLOW PHASE: INITIAL_MITIGATION" in prompt
    assert "Target Package: express-jwt (version: 1.2.3)" in prompt
    assert "CVE-2026-1234" in prompt
    assert "=== RECOMMENDED INITIAL SEARCH QUERY ===" in prompt
    assert "=== WORKAROUND SNIPPETS ===" in prompt
    # No noisy QA evidence fields
    assert "Source Attempt ID" not in prompt
    assert "Raw Excerpt" not in prompt


def test_compact_qa_regression_repair_prompt():
    task = MagicMock(task_id="task-2", selected_version="8.5.1", instruction="Update package")
    group = MagicMock(
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2026-9999"],
        ghsa_ids=[],
        fix_plan=MagicMock(status=MagicMock(value="in_progress")),
    )
    qa_ev = QAFailureEvidence(
        attempt_id="att-123",
        task_revision=5,
        exact_diagnostics=["TypeError: jwt is not a function"],
        failed_tests=["tests/jwt.test.ts"],
        source_locations=["lib/insecurity.ts:42:10"],
        affected_files=["lib/insecurity.ts"],
        raw_excerpt="A very large raw log excerpt..." * 10,
    )
    context = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        qa_evidence=qa_ev,
    )

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        workaround_context=context,
    )

    assert "WORKFLOW PHASE: QA_REGRESSION_REPAIR" in prompt
    assert "Dependency update is already seeded; do not modify manifests" in prompt
    assert "TypeError: jwt is not a function" in prompt
    assert "lib/insecurity.ts" in prompt
    assert "Targeted Test File: tests/jwt.test.ts" in prompt

    # Verify omission of noisy/redundant fields
    assert "Source Attempt ID" not in prompt
    assert "Source Revision" not in prompt
    assert "Raw Excerpt" not in prompt
    assert "A very large raw log excerpt" not in prompt


def test_search_recommendation_precedence():
    task = MagicMock(selected_version="2.0.0")
    group = MagicMock(
        vulnerable_component="lodash",
        cve_ids=["CVE-2025-1111"],
        ghsa_ids=[],
        fix_plan=MagicMock(status=MagicMock(value="in_progress")),
    )
    group.issues = []

    # 1. QA Phase -> update_mitigates_cve_but_breaks_tests
    context = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        qa_evidence=QAFailureEvidence(exact_diagnostics=["TypeError: fn is not a function"]),
    )
    rec = _workaround_search_recommendation(task, group, context, None)
    assert rec.scenario == "update_mitigates_cve_but_breaks_tests"

    # 2. Scanner failure -> update_does_not_resolve_scanner_findings
    context_scanner = WorkaroundContext(
        phase=WorkaroundPhase.INITIAL_MITIGATION,
        qa_evidence=None,
    )
    rec_scanner = _workaround_search_recommendation(
        task, group, context_scanner, previous_feedback="scanner findings remaining"
    )
    assert rec_scanner.scenario == "update_does_not_resolve_scanner_findings"

    # 3. No fix -> no_update_available_to_resolve_cve
    group_nofix = MagicMock(
        vulnerable_component="lodash",
        cve_ids=["CVE-2025-1111"],
        ghsa_ids=[],
        fix_plan=MagicMock(status=MagicMock(value="no_fix")),
    )
    rec_nofix = _workaround_search_recommendation(task, group_nofix, context_scanner, None)
    assert rec_nofix.scenario == "no_update_available_to_resolve_cve"

    # 4. Otherwise -> initial_code_workaround_or_isolation
    rec_init = _workaround_search_recommendation(task, group, context_scanner, None)
    assert rec_init.scenario == "initial_code_workaround_or_isolation"
