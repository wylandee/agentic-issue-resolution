"""
Unit and integration tests for Workaround Lifecycle and Evidence-Based Alternative Tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    RemediationTask,
    RoutingStrategy,
    VulnerabilityGroup,
    WorkaroundExecutionPhase,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration.remedy_tools import (
    _is_infrastructure_failure,
    _make_deterministic_search_replace_tool,
    _make_read_web_page_tool,
    _make_record_plan_tool,
    _make_record_targeted_test_substitution_tool,
    _make_search_web_tool,
    _make_validate_workaround_tool,
    build_workaround_toolbelt,
)
from remediation_engine.orchestration.supervisor_node import (
    run_supervisor_node,
)
from remediation_engine.orchestration.workaround_subagent import (
    _workaround_attempt_succeeded,
)


def _mock_sandbox(files: dict[str, str] | None = None):
    files = dict(files or {})
    sandbox = MagicMock()

    def read_file(path: str):
        norm = path.replace("\\", "/").lstrip("/")
        return files.get(norm)

    def write_file(path: str, content: str):
        norm = path.replace("\\", "/").lstrip("/")
        files[norm] = content

    def run(cmd: str, timeout: int = 60):
        res = MagicMock()
        res.exit_code = 0
        res.stdout = "SUCCESS"
        res.stderr = ""
        return res

    sandbox.read_file.side_effect = read_file
    sandbox.write_file.side_effect = write_file
    sandbox.run.side_effect = run
    return sandbox, files


def test_web_search_rejected_before_local_investigation():
    plan_state: dict[str, Any] = {}
    search_web = _make_search_web_tool(plan_state=plan_state)
    res = search_web.invoke({"query": "express jwt vulnerability fix"})
    assert "ERROR: [INVESTIGATION_REQUIRED]" in str(res)


def test_read_web_page_rejected_before_web_search():
    plan_state: dict[str, Any] = {"local_investigation_complete": True}
    read_web_page = _make_read_web_page_tool(plan_state=plan_state)
    res = read_web_page.invoke({"url": "https://github.com/advisories/GHSA-1234"})
    assert "ERROR: [SEARCH_REQUIRED]" in str(res)


def test_plan_rejected_without_evidence():
    plan_state: dict[str, Any] = {"local_investigation_complete": True, "inspected_files": {"src/index.js"}}
    record_plan = _make_record_plan_tool(plan_state)
    res = record_plan.invoke({
        "affected_files": ["src/index.js"],
        "affected_symbols": ["handleAuth"],
        "security_invariant": "Always validate token algorithm",
        "causal_hypothesis": "Missing algorithm check",
        "exact_intended_edits": "Add algorithm check",
        "evidence_source": "",
    })
    assert "ERROR: [PLAN_REJECTED]" in str(res)


def test_plan_rejected_for_uninspected_files():
    plan_state: dict[str, Any] = {"local_investigation_complete": True, "inspected_files": {"src/index.js"}}
    record_plan = _make_record_plan_tool(plan_state)
    res = record_plan.invoke({
        "affected_files": ["src/other.js"],
        "affected_symbols": ["handleAuth"],
        "security_invariant": "Always validate token algorithm",
        "causal_hypothesis": "Missing algorithm check",
        "exact_intended_edits": "Add algorithm check",
        "evidence_source": "https://github.com/advisories/GHSA-1234",
    })
    assert "ERROR: [PLAN_REJECTED]" in str(res)
    assert "src/other.js" in str(res)


def test_edits_rejected_before_valid_plan():
    sandbox, _ = _mock_sandbox({"src/index.js": "const x = 1;"})
    plan_state: dict[str, Any] = {"phase": WorkaroundExecutionPhase.INVESTIGATE.value}
    touched = set()
    edit_tool = _make_deterministic_search_replace_tool(sandbox, touched, plan_state)
    res = edit_tool.invoke({"file_path": "src/index.js", "old_text": "const x = 1;", "new_text": "const x = 2;"})
    assert "ERROR: [PLAN_VIOLATION]" in str(res) or "ERROR: [PHASE_VIOLATION]" in str(res)


def test_edits_rejected_after_one_edit_before_validation():
    sandbox, _ = _mock_sandbox({"src/index.js": "const x = 1;"})
    plan_state: dict[str, Any] = {
        "recorded": True,
        "phase": WorkaroundExecutionPhase.EXECUTE.value,
        "planned_files": ["src/index.js"],
        "inspected_files": {"src/index.js"},
        "successful_edit_count_this_iteration": 1,
    }
    touched = {"src/index.js"}
    edit_tool = _make_deterministic_search_replace_tool(sandbox, touched, plan_state)
    res = edit_tool.invoke({"file_path": "src/index.js", "old_text": "const x = 1;", "new_text": "const x = 2;"})
    assert "ERROR: [ITERATION_LIMIT]" in str(res)


def test_plan_and_web_rejected_while_waiting_for_validation():
    plan_state: dict[str, Any] = {
        "local_investigation_complete": True,
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "recorded": True,
    }
    record_plan = _make_record_plan_tool(plan_state)
    plan_res = record_plan.invoke(
        {
            "affected_files": ["src/index.js"],
            "affected_symbols": ["handleAuth"],
            "security_invariant": "Auth remains enforced",
            "causal_hypothesis": "The imported API changed shape",
            "exact_intended_edits": "Update the import and call sites atomically",
            "evidence_source": "workspace:src/index.js",
        }
    )
    assert "ERROR: [PHASE_VIOLATION]" in str(plan_res)

    search_web = _make_search_web_tool(plan_state=plan_state)
    web_res = search_web.invoke({"query": "express-jwt migration"})
    assert "ERROR: [PHASE_VIOLATION]" in str(web_res)


def test_validation_failure_resets_execution_to_investigate():
    sandbox, files = _mock_sandbox({"src/index.js": "const x = 2;"})
    plan_state: dict[str, Any] = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "iteration": 1,
        "current_iteration_edit": MagicMock(file_path="src/index.js", old_text="const x = 1;", new_text="const x = 2;"),
        "successful_edits": [MagicMock(file_path="src/index.js", old_text="const x = 1;", new_text="const x = 2;")],
    }
    sandbox.run.side_effect = lambda cmd, timeout=60: MagicMock(exit_code=1, stdout="", stderr="Syntax error")
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state, preferred_test_files=["test/index.test.js"])
    res = val_tool.invoke({"modified_files": ["src/index.js"], "runtime_smoke_file": "src/index.js"})

    assert "FAILURE" in str(res)
    assert plan_state["phase"] == WorkaroundExecutionPhase.INVESTIGATE.value
    assert plan_state["iteration"] == 2


def test_runtime_smoke_targets_never_implicitly_skipped():
    sandbox, _ = _mock_sandbox({"src/index.js": "const x = 1;"})
    plan_state: dict[str, Any] = {}
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state)
    res = val_tool.invoke({"modified_files": ["src/index.js"], "runtime_smoke_file": ""})
    assert "FAILURE" in str(res)
    assert "runtime_smoke_file must be explicitly supplied" in str(res)


def test_missing_sqlite_native_bindings_classified_as_infra_only():
    err = "Error: Cannot find module 'better-sqlite3' or missing native bindings"
    assert _is_infrastructure_failure(err) is True


def test_assertion_failures_not_classified_as_infra_only():
    err = "AssertionError: expected 1 to equal 2\n    at Context.<anonymous> (test/index.test.js:10)"
    assert _is_infrastructure_failure(err) is False


def test_alternative_test_rejected_without_infra_failure():
    sandbox, _ = _mock_sandbox({"test/alt.test.js": "console.log('test');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": False,
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke({
        "original_test": "test/orig.test.js",
        "alternative_test": "test/alt.test.js",
        "infrastructure_failure_evidence": "Assertion failure",
        "shared_behavior_explanation": "Tests auth invariant",
        "evidence_sources": ["GHSA-1234"],
        "infrastructure_avoidance_explanation": "Avoids native sqlite",
    })
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)


def test_alternative_test_rejected_without_mapping_evidence():
    sandbox, _ = _mock_sandbox({"test/alt.test.js": "console.log('test');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke({
        "original_test": "test/orig.test.js",
        "alternative_test": "test/alt.test.js",
        "infrastructure_failure_evidence": "Missing native bindings",
        "shared_behavior_explanation": "Tests auth invariant",
        "evidence_sources": [],
        "infrastructure_avoidance_explanation": "Avoids native sqlite",
    })
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)


def test_alternative_test_rejected_when_it_imports_unavailable_dependency():
    sandbox, _ = _mock_sandbox({"test/alt.test.js": "const sqlite3 = require('sqlite3');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "latest_infra_diagnostics": "Cannot find module 'sqlite3'",
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke({
        "original_test": "test/orig.test.js",
        "alternative_test": "test/alt.test.js",
        "infrastructure_failure_evidence": "Missing sqlite3 native binding",
        "shared_behavior_explanation": "Tests auth invariant",
        "evidence_sources": ["GHSA-1234"],
        "infrastructure_avoidance_explanation": "Avoids native sqlite",
    })
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)
    assert "directly imports/invokes" in str(res)


def test_alternative_test_accepted_when_criteria_pass():
    sandbox, _ = _mock_sandbox({"test/alt.test.js": "const jwt = require('jsonwebtoken');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "latest_infra_diagnostics": "Cannot find module 'better-sqlite3'",
        "inspected_files": {"test/alt.test.js"},
        "iteration": 1,
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke({
        "original_test": "test/orig.test.js",
        "alternative_test": "test/alt.test.js",
        "infrastructure_failure_evidence": "Missing native sqlite3",
        "shared_behavior_explanation": "Tests auth invariant",
        "evidence_sources": ["GHSA-1234"],
        "infrastructure_avoidance_explanation": "Uses mock DB instead of sqlite3",
    })
    assert "SUCCESS:" in str(res)
    assert plan_state["accepted_alternative_test"] == "test/alt.test.js"
    assert plan_state["original_to_alternative_test_evidence"]["test/orig.test.js"] == [
        "GHSA-1234"
    ]


def test_real_workaround_toolbelt_requires_explicit_smoke_and_targeted_test():
    sandbox, _ = _mock_sandbox({"src/index.js": "module.exports = true;"})
    plan_state: dict[str, Any] = {}
    tools = {
        tool.name: tool
        for tool in build_workaround_toolbelt(
            sandbox,
            {"src/index.js"},
            MagicMock(),
            plan_state=plan_state,
        )
    }
    plan_state["targeted_test_required"] = True

    res = tools["validate_workaround"].invoke({"modified_files": ["src/index.js"]})
    assert "runtime_smoke_file must be explicitly supplied" in str(res)

    res = tools["validate_workaround"].invoke(
        {"modified_files": ["src/index.js"], "runtime_smoke_file": "src/index.js"}
    )
    assert "targeted_test_file must be supplied" in str(res)


def test_alternative_targeted_test_passing_with_static_checks_and_smoke():
    sandbox, _ = _mock_sandbox({
        "package.json": '{"name": "test-app", "scripts": {"test": "vitest"}}',
        "src/index.js": "module.exports = { auth: () => true };",
        "test/alt.test.js": "const { auth } = require('../src/index');",
    })
    plan_state: dict[str, Any] = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "accepted_alternative_test": "test/alt.test.js",
    }
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state, preferred_test_files=["test/orig.test.js"])
    res = val_tool.invoke({
        "modified_files": ["src/index.js"],
        "runtime_smoke_file": "src/index.js",
    })
    assert "SUCCESS:" in str(res)
    assert plan_state["validated_files"] == ["src/index.js"]


def test_alternative_targeted_test_failure_preventing_optimistic_success():
    sandbox, _ = _mock_sandbox({
        "package.json": '{"name": "test-app", "scripts": {"test": "vitest"}}',
        "src/index.js": "const x = 1;",
    })
    plan_state: dict[str, Any] = {
        "accepted_alternative_test": "test/alt.test.js",
    }
    def mock_run(cmd: str, timeout: int = 60):
        res = MagicMock()
        if "test/alt.test.js" in cmd:
            res.exit_code = 1
            res.stdout = "AssertionError: expected false to be true"
            res.stderr = ""
        else:
            res.exit_code = 0
            res.stdout = "OK"
            res.stderr = ""
        return res

    sandbox.run.side_effect = mock_run
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state)
    res = val_tool.invoke({
        "modified_files": ["src/index.js"],
        "runtime_smoke_file": "src/index.js",
        "targeted_test_file": "test/alt.test.js",
    })

    assert "FAILURE" in str(res)
    assert plan_state.get("validation_passed", False) is False


def test_successful_validation_reporting_accurate_validation_calls():
    runtime = MagicMock()
    runtime.changed_files = ["src/index.js"]
    runtime.errors = []

    plan_state = {
        "validation_calls": 2,
        "validated_files": ["src/index.js"],
        "last_validation_result": {"overall_status": "PASS"},
    }

    succeeded = _workaround_attempt_succeeded(
        runtime,
        has_all_validated=True,
        has_recorded_plan=True,
        requires_targeted_test=False,
        targeted_test_passed=True,
        validation_gate_passed=True,
        plan_state=plan_state,
    )
    assert succeeded is True


def test_failed_validation_producing_no_validated_files():
    runtime = MagicMock()
    runtime.changed_files = ["src/index.js"]
    runtime.errors = ["Validation failed"]

    plan_state = {
        "validation_calls": 1,
        "validated_files": [],
        "last_validation_result": {"overall_status": "CODE_FAILURE"},
    }

    succeeded = _workaround_attempt_succeeded(
        runtime,
        has_all_validated=False,
        has_recorded_plan=True,
        requires_targeted_test=False,
        targeted_test_passed=False,
        validation_gate_passed=False,
        plan_state=plan_state,
    )
    assert succeeded is False


def test_supervisor_refuses_optimistic_fixed_for_skipped_or_blocked_gates():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Workaround",
        current_attempt_id="att-1",
    )
    snapshot = MagicMock(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=0,
        selected_version=None,
        instruction_digest="dig-1",
    )
    worker_result = WorkerAttemptResult(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=0,
        status=AgentActionStatus.SUCCESS,
        changed_files=["src/index.js"],
        instruction_digest="dig-1",
        execution_diagnostics=WorkerExecutionDiagnostics(
            validation_calls=0,
            validation_passed=False,
            validated_files=[],
        ),
    )
    group = VulnerabilityGroup(
        group_id="group-1",
        vulnerable_component="express",
        package_manifest_path="package.json",
        issue_type="sca",
        representative_issue_id="00000000-0000-0000-0000-000000000001",
    )

    state = {
        "task_queue": {"task-1": task},
        "valid_groups": [group],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-1": worker_result},
        "active_target_task_ids": ["task-1"],
    }

    updates = run_supervisor_node(state)
    t = updates["task_queue"]["task-1"]
    assert t.status != "optimistically_fixed"


def test_supervisor_accepts_optimistic_fixed_when_alternative_test_fully_replaces_original():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Workaround",
        current_attempt_id="att-1",
    )
    snapshot = MagicMock(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=0,
        selected_version=None,
        instruction_digest="dig-1",
    )
    worker_result = WorkerAttemptResult(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=0,
        status=AgentActionStatus.SUCCESS,
        changed_files=["src/index.js"],
        instruction_digest="dig-1",
        execution_diagnostics=WorkerExecutionDiagnostics(
            validation_calls=1,
            validation_passed=True,
            per_gate_results={"overall_status": "PASS"},
            validated_files=["src/index.js"],
            final_selected_targeted_test="test/alt.test.js",
            original_to_alternative_test_mapping={"test/orig.test.js": "test/alt.test.js"},
        ),
    )
    group = VulnerabilityGroup(
        group_id="group-1",
        vulnerable_component="express",
        package_manifest_path="package.json",
        issue_type="sca",
        representative_issue_id="00000000-0000-0000-0000-000000000001",
    )

    state = {
        "task_queue": {"task-1": task},
        "valid_groups": [group],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-1": worker_result},
        "active_target_task_ids": ["task-1"],
    }

    updates = run_supervisor_node(state)
    t = updates["task_queue"]["task-1"]
    assert t.status == "optimistically_fixed"
