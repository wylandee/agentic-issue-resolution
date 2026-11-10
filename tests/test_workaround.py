"""Focused tests for workaround."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from remediation_engine.agents.qa_evaluation import _attach_failure_evidence_to_evaluations
from remediation_engine.agents.supervisor import (
    _create_attempt_snapshot,
    _qa_failure_evidence_for_workaround_retry,
    run_supervisor_node,
)
from remediation_engine.agents.workaround import (
    _WORKAROUND_STATIC_INSTRUCTIONS,
    _build_workaround_prompt,
    _preferred_targeted_test_files,
    _primary_non_test_source_files,
    _workaround_attempt_succeeded,
    _workaround_search_recommendation,
    run_workaround_subagent_node,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueType,
    PackageFixPlanCandidate,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundExecutionPhase,
    WorkaroundPhase,
    WorkaroundPlannedReplacement,
    WorkaroundReplayPlan,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.runtime.agent_runtime import (
    ToolEvent,
    _infer_changed_files,
    _stagnation_recovery_instruction,
    run_bounded_subagent_loop,
)
from remediation_engine.tools.npm_mutations import _is_prohibited_target
from remediation_engine.tools.qa_scan import _QAExecutionResults
from remediation_engine.tools.test_execution import (
    _detect_targeted_test_context,
    _normalise_test_path,
    build_targeted_test_command,
    extract_qa_failure_evidence,
)
from remediation_engine.tools.validation import (
    _make_record_plan_tool,
    _make_record_targeted_test_substitution_tool,
    _make_validate_workaround_tool,
    _revert_current_iteration_edit,
)
from remediation_engine.tools.workers import (
    _apply_replacements_to_content,
    _make_deterministic_apply_edit_set_tool,
    build_workaround_toolbelt,
)
from remediation_engine.tools.workspace import (
    _is_authoritative_evidence_source,
    _is_infrastructure_failure,
    _make_read_web_page_tool,
    _make_search_web_tool,
)
from tests.helpers import qa_policy_for_task


def _attempt_snapshot(task: RemediationTask) -> TaskAttemptSnapshot:
    return TaskAttemptSnapshot(
        attempt_id=task.current_attempt_id or "attempt-missing",
        task_id=task.task_id,
        task_revision=task.task_revision,
        instruction=task.instruction,
        instruction_digest="test-digest",
        dispatch_node="workaround_subagent",
        allowed_target_versions=[],
        allowed_dependency_types=[],
    )


def test_extract_qa_failure_evidence_exact_diagnostics():
    stdout = """
    1) express-jwt middleware should handle missing algorithms:
       TypeError: (0 , import_express_jwt.default) is not a function
           at Context.<anonymous> (test/jwt.test.js:42:15)
           at processImmediate (node:internal/timers:478:21)
    """
    stderr = "Error: Invalid header at src/index.js:10"

    evidence = extract_qa_failure_evidence(1, stdout, stderr, attempt_id="att-123", task_revision=2)

    assert evidence.attempt_id == "att-123"
    assert evidence.task_revision == 2
    assert "express-jwt middleware should handle missing algorithms" in evidence.failed_tests
    assert any(
        "(0 , import_express_jwt.default) is not a function" in d
        for d in evidence.exact_diagnostics
    )
    assert any("test/jwt.test.js:42:15" in loc for loc in evidence.source_locations)
    assert "test/jwt.test.js" in evidence.affected_files
    assert "src/index.js" in evidence.affected_files


def test_failed_qa_evidence_is_correlated_to_committed_attempt():
    """Deterministic test evidence carries the worker attempt envelope forward."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply workaround",
        current_attempt_id="attempt-7",
        task_revision=4,
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    evaluations = {
        "task-1": QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="Repair the regression.",
        )
    }
    results = _QAExecutionResults(
        tests=(False, "express-jwt: algorithms is a required option at lib/insecurity.ts:54:35")
    )

    enriched = _attach_failure_evidence_to_evaluations(
        evaluations,
        results,
        {
            "task_queue": {task.task_id: task},
            "attempt_snapshots_by_id": {
                task.current_attempt_id: _attempt_snapshot(task),
            },
        },
    )

    evidence = enriched["task-1"].failure_evidence
    assert evidence is not None
    assert evidence.attempt_id == "attempt-7"
    assert evidence.task_revision == 4
    assert any("algorithms is a required option" in d for d in evidence.exact_diagnostics)


def test_failure_evidence_is_dropped_for_task_without_current_snapshot():
    """Do not attach one task's deterministic failure output to another task."""
    task_one = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply workaround one.",
        current_attempt_id="attempt-1",
        task_revision=1,
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    task_two = task_one.model_copy(
        update={
            "task_id": "task-2",
            "current_attempt_id": "attempt-2",
            "instruction": "Apply workaround two.",
        }
    )
    evaluations = {
        task.task_id: QAEvaluation(
            task_id=task.task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="Retry the task.",
        )
        for task in (task_one, task_two)
    }
    results = _QAExecutionResults(tests=(False, "TypeError at src/one.test.ts:4:2"))
    state = {
        "task_queue": {task.task_id: task for task in (task_one, task_two)},
        "attempt_snapshots_by_id": {
            task_one.current_attempt_id: _attempt_snapshot(task_one),
        },
    }

    enriched = _attach_failure_evidence_to_evaluations(evaluations, results, state)

    assert enriched["task-1"].failure_evidence is not None
    assert enriched["task-1"].failure_evidence.attempt_id == "attempt-1"
    assert enriched["task-2"].failure_evidence is None


def test_deterministic_qa_evidence_replaces_llm_source_locations():
    """Batch-judge paths cannot override the paths from QA test output."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply workaround",
        current_attempt_id="attempt-7",
        task_revision=4,
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    evaluations = {
        "task-1": QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Repair the regression.",
            failure_evidence=QAFailureEvidence(
                exact_diagnostics=["npm test FAILED (exit 1)"],
                source_locations=["server/test/sanitize.test.js"],
                affected_files=["package.json"],
            ),
        )
    }
    results = _QAExecutionResults(tests=(False, "TypeError at test/server/insecuritySpec.ts:184:7"))
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: (
        "describe('insecurity', () => {})" if path == "test/server/insecuritySpec.ts" else None
    )

    enriched = _attach_failure_evidence_to_evaluations(
        evaluations,
        results,
        {
            "task_queue": {task.task_id: task},
            "attempt_snapshots_by_id": {
                task.current_attempt_id: _attempt_snapshot(task),
            },
        },
        sandbox=sandbox,
    )

    evidence = enriched["task-1"].failure_evidence
    assert evidence is not None
    assert evidence.source_locations == ["test/server/insecuritySpec.ts:184:7"]
    assert evidence.affected_files == ["test/server/insecuritySpec.ts"]
    assert "server/test/sanitize.test.js" not in evidence.source_locations
    assert any("LLM-supplied source location" in d for d in evidence.exact_diagnostics)


def test_invalid_deterministic_qa_source_locations_are_discarded():
    """QA does not propagate source paths absent from the workspace."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply workaround",
        current_attempt_id="attempt-7",
        task_revision=4,
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    evaluations = {
        "task-1": QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Repair the regression.",
        )
    }
    results = _QAExecutionResults(
        tests=(False, "AssertionError at server/test/sanitize.test.js:184:7")
    )
    sandbox = MagicMock()
    sandbox.read_file.return_value = None

    enriched = _attach_failure_evidence_to_evaluations(
        evaluations,
        results,
        {
            "task_queue": {task.task_id: task},
            "attempt_snapshots_by_id": {
                task.current_attempt_id: _attempt_snapshot(task),
            },
        },
        sandbox=sandbox,
    )

    evidence = enriched["task-1"].failure_evidence
    assert evidence is not None
    assert evidence.source_locations == []
    assert evidence.affected_files == []
    assert any("does not exist in the QA workspace" in d for d in evidence.exact_diagnostics)


def test_supervisor_creates_workaround_context():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Mitigate express-jwt vulnerability",
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    snapshots = {}
    evidence = QAFailureEvidence(
        exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
        failed_tests=["jwt middleware test"],
    )
    ctx = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        vulnerability_mechanism="express-jwt import signature change",
        qa_evidence=evidence,
    )

    updated_task, snapshot = _create_attempt_snapshot(
        task,
        dispatch_node="workaround_subagent",
        snapshots_by_id=snapshots,
        state_revision=1,
        workaround_context=ctx,
    )

    assert snapshot.workaround_context is not None
    assert snapshot.workaround_context.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
    assert snapshot.workaround_context.qa_evidence == evidence


def test_prohibited_targets():
    assert _is_prohibited_target("package.json") is True
    assert _is_prohibited_target("package-lock.json") is True
    assert _is_prohibited_target("test/jwt.test.js") is True
    assert _is_prohibited_target("src/index.spec.ts") is True
    assert _is_prohibited_target("src/service.js") is False


def test_targeted_test_command_construction():
    mock_sandbox = MagicMock()
    mock_sandbox.read_file.return_value = '{"scripts": {"test": "mocha test/*.js"}}'

    runner = _detect_targeted_test_context(mock_sandbox, "test/jwt.test.js")[0]
    assert runner == "mocha"

    cmd = build_targeted_test_command("mocha", "test/jwt.test.js", "missing algorithms")
    assert cmd == "npm test -- test/jwt.test.js --grep 'missing algorithms' --reporter json"

    jest_cmd = build_targeted_test_command("jest", "test/app.test.js", "login test")
    assert jest_cmd == "npm test -- test/app.test.js -t 'login test'"

    vitest_cmd = build_targeted_test_command("vitest", "test/app.test.js", "login test")
    assert vitest_cmd == "npm test -- --run test/app.test.js -t 'login test'"

    node_cmd = build_targeted_test_command("node_test", "test/app.test.js", "login test")
    assert node_cmd == "npm test -- test/app.test.js --test-name-pattern 'login test'"


def test_targeted_test_context_selects_matching_npm_script():
    root_package = {
        "scripts": {
            "test": "npm run test:frontend && npm run test:server && npm run test:api",
            "test:frontend": "cd frontend && npm run test",
            "test:server": "mocha test/server/**/*.ts",
            "test:api": 'node --test "test/api/**/*.test.ts"',
        }
    }
    frontend_package = {
        "scripts": {"test": "ng test"},
        "devDependencies": {"vitest": "^4.0.0"},
    }
    sandbox = MagicMock()

    def read_file(path):
        if path == "package.json":
            return json.dumps(root_package)
        if path == "frontend/package.json":
            return json.dumps(frontend_package)
        if path == "frontend/angular.json":
            return json.dumps(
                {
                    "projects": {
                        "frontend": {"architect": {"test": {"builder": "@angular/build:unit-test"}}}
                    }
                }
            )
        return None

    sandbox.read_file.side_effect = read_file

    assert _detect_targeted_test_context(sandbox, "test/api/2fa.test.ts") == (
        "node_test",
        "",
        'node --test "test/api/**/*.test.ts"',
    )
    assert _detect_targeted_test_context(sandbox, "frontend/src/app/login.spec.ts") == (
        "angular_vitest",
        "frontend",
        "ng test",
    )


def test_targeted_test_paths_keep_workspace_relative_and_reject_escapes():
    assert (
        _normalise_test_path("/workspace/build/test\\api\\auth.test.ts") == "test/api/auth.test.ts"
    )
    assert _normalise_test_path("") == ""
    assert _normalise_test_path("/etc/passwd") == ""
    assert _normalise_test_path("test/../../secret.test.ts") == ""


def test_qa_evidence_is_reused_after_failed_attempt_is_closed():
    evidence = QAFailureEvidence(
        attempt_id="attempt-1",
        task_revision=2,
        exact_diagnostics=["TypeError: middleware is not a function"],
        source_locations=["/workspace/lib/auth.ts:10:4"],
    )
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="Repair the import.",
        failure_evidence=evidence,
    )
    qa_result = QAAttemptResult(
        attempt_id="attempt-1",
        task_id="task-1",
        task_revision=2,
        qa_policy=QAPolicy.INITIAL_CODE_WORKAROUND,
        qa_policy_source="attempt_snapshot",
        evaluation=evaluation,
    )

    resolved = _qa_failure_evidence_for_workaround_retry(
        "task-1",
        {"task-1": evaluation},
        {"attempt-1": qa_result},
    )

    assert resolved == evidence


def test_qa_evidence_is_inherited_by_workaround_child_from_parent_attempt():
    evidence = QAFailureEvidence(
        attempt_id="update-attempt-1",
        task_revision=3,
        exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
        source_locations=["/workspace/lib/insecurity.ts:54:35"],
    )
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="Repair the express-jwt compatibility regression.",
        failure_evidence=evidence,
    )
    qa_result = QAAttemptResult(
        attempt_id="update-attempt-1",
        task_id="task-1",
        task_revision=3,
        qa_policy=QAPolicy.INITIAL_CODE_WORKAROUND,
        qa_policy_source="attempt_snapshot",
        evaluation=evaluation,
    )

    resolved = _qa_failure_evidence_for_workaround_retry(
        "task-2",
        {"task-1": evaluation},
        {"update-attempt-1": qa_result},
        related_task_ids=["task-1"],
    )

    assert resolved == evidence


def test_preferred_targeted_test_files_come_from_qa_locations():
    evidence = QAFailureEvidence(
        source_locations=["/workspace/test/api/2fa.test.ts:2:1673"],
        affected_files=["/workspace/lib/insecurity.ts"],
    )

    assert _preferred_targeted_test_files(evidence) == ["test/api/2fa.test.ts"]


def test_preferred_targeted_test_files_strip_test_title_from_qa_summary():
    evidence = QAFailureEvidence(
        failed_tests=[
            'test/server/b2bOrderSpec.ts: b2bOrder deserializing arbitrary JSON should not solve "rceChallenge"'
        ]
    )

    assert _preferred_targeted_test_files(evidence) == ["test/server/b2bOrderSpec.ts"]


def test_runtime_errors_prevent_workaround_success():
    runtime = type("Runtime", (), {"changed_files": ["lib/auth.ts"], "errors": ["round limit"]})()

    assert not _workaround_attempt_succeeded(
        runtime,
        has_all_validated=True,
        has_recorded_plan=True,
        requires_targeted_test=False,
        targeted_test_passed=True,
    )


def test_stagnation_recovery_instruction():
    inst = _stagnation_recovery_instruction(
        "inspect_ast_symbol",
        {"symbol_name": "expressJwt"},
        "NOT FOUND: No declared function",
    )
    assert "inspect_ast_symbol" in inst
    assert "PROHIBITION" in inst
    assert "imported identifier" in inst


def test_build_workaround_prompt_biphasic():
    task = RemediationTask(
        task_id="t-1",
        parent_group_id="g-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix vulnerability",
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    group = VulnerabilityGroup(
        group_id="g-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2026-1234"],
        representative_issue_id=str(uuid4()),
    )

    ctx_initial = WorkaroundContext(phase=WorkaroundPhase.INITIAL_MITIGATION)
    prompt_initial = _build_workaround_prompt(task, group, workaround_context=ctx_initial)
    assert "WORKFLOW PHASE: INITIAL_MITIGATION" in prompt_initial
    assert "Follow Investigate -> Plan -> Execute -> Validate" in _WORKAROUND_STATIC_INSTRUCTIONS
    assert "CODE_FAILURE rolls back" in _WORKAROUND_STATIC_INSTRUCTIONS
    assert "whole pending set" in _WORKAROUND_STATIC_INSTRUCTIONS
    assert "re-include every required change" in _WORKAROUND_STATIC_INSTRUCTIONS
    assert "INFRA_FAILURE or BLOCKED retains" in _WORKAROUND_STATIC_INSTRUCTIONS
    assert "pending set for recovery" in _WORKAROUND_STATIC_INSTRUCTIONS

    evidence = QAFailureEvidence(
        exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
        failed_tests=["jwt test"],
        attempt_id="att-1",
    )
    ctx_repair = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        qa_evidence=evidence,
    )
    prompt_repair = _build_workaround_prompt(task, group, workaround_context=ctx_repair)
    assert "WORKFLOW PHASE: QA_REGRESSION_REPAIR" in prompt_repair
    assert "=== QA FAILURE EVIDENCE ===" in prompt_repair
    assert "(0, import_express_jwt.default) is not a function" in prompt_repair
    assert (
        "dependency update and replayed edits are already present"
        in _WORKAROUND_STATIC_INSTRUCTIONS
    )


def test_workaround_toolbelt_ast_gate_and_search_enrichment():
    mock_sandbox = MagicMock()
    mock_sandbox.read_file.return_value = "function oldFunc() { return 1; }"
    mock_sandbox.run.return_value.exit_code = 0
    mock_sandbox.run.return_value.stdout = "SUCCESS"

    touched = set()
    plan_state = {
        "recorded": True,
        "phase": WorkaroundExecutionPhase.EXECUTE.value,
        "planned_replacements": [
            {"file_path": "src/index.js", "old_text": "oldFunc", "new_text": "newFunc"}
        ],
        "local_investigation_complete": True,
    }
    tools = build_workaround_toolbelt(
        mock_sandbox,
        touched,
        host_repo_root=MagicMock(),
        plan_state=plan_state,
    )

    tool_dict = {t.name: t for t in tools}

    # AST gate check before inspection should fail for JS files
    dsr = tool_dict["deterministic_apply_edit_set"]
    res_no_ast = dsr.invoke(
        {
            "replacements": [
                {"file_path": "src/index.js", "old_text": "oldFunc", "new_text": "newFunc"}
            ]
        }
    )
    assert "AST inspection required" in res_no_ast

    # The worker owns query construction; the tool preserves it verbatim.
    search_web_tool = tool_dict["search_web"]
    with patch("requests.post") as mock_post:
        mock_post.return_value.raise_for_status = MagicMock()
        mock_post.return_value.json.return_value = {
            "organic": [
                {"title": "Advisory", "snippet": "Migration guide", "link": "http://example.com"}
            ]
        }
        with patch.dict("os.environ", {"SERPER_API_KEY": "fake_key"}):
            web_res = search_web_tool.invoke({"query": "migration guide"})
            assert "Effective Query: migration guide" in web_res


def test_express_jwt_mocked_end_to_end_scenario():
    """Mocked end-to-end regression scenario based on express-jwt trace."""
    task = RemediationTask(
        task_id="task-jwt",
        parent_group_id="group-jwt",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix express-jwt import",
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    group = VulnerabilityGroup(
        group_id="group-jwt",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2020-15084"],
        file_path="routes/auth.js",
        representative_issue_id=str(uuid4()),
    )

    # 1. Initial attempt
    ctx_initial = WorkaroundContext(
        phase=WorkaroundPhase.INITIAL_MITIGATION,
        vulnerability_mechanism="express-jwt v6 migration require syntax change",
    )
    snapshot1 = TaskAttemptSnapshot(
        attempt_id="att-1",
        task_id="task-jwt",
        task_revision=1,
        dispatch_node="workaround_subagent",
        instruction="Fix express-jwt",
        instruction_digest="digest1",
        workaround_context=ctx_initial,
        allowed_target_versions=[],
        allowed_dependency_types=[],
    )

    state1 = {
        "repo_root": ".",
        "workspace_volume": "vol-123",
        "target_task": task,
        "target_group": group,
        "attempt_snapshot": snapshot1,
    }

    mock_llm = MagicMock()
    tool_call_inspect = {
        "name": "read_workspace_file",
        "args": {"file_path": "routes/auth.js"},
        "id": "call-0",
    }
    tool_call_record = {
        "name": "record_plan",
        "args": {
            "affected_files": ["routes/auth.js"],
            "affected_symbols": ["authMiddleware"],
            "security_invariant": "JWT auth must be enforced",
            "causal_hypothesis": "expressJwt import was updated incorrectly",
            "planned_replacements": [
                {
                    "file_path": "routes/auth.js",
                    "old_text": "const expressJwt = require('express-jwt');",
                    "new_text": "const { expressjwt } = require('express-jwt');",
                }
            ],
            "evidence_source": "https://github.com/advisories/GHSA-1234",
        },
        "id": "call-1",
    }
    tool_call_edit = {
        "name": "deterministic_apply_edit_set",
        "args": {
            "replacements": [
                {
                    "file_path": "routes/auth.js",
                    "old_text": "const expressJwt = require('express-jwt');",
                    "new_text": "const { expressjwt } = require('express-jwt');",
                }
            ],
        },
        "id": "call-2",
    }
    tool_call_val = {
        "name": "validate_workaround",
        "args": {
            "modified_files": ["routes/auth.js"],
            "runtime_smoke_file": "routes/auth.js",
            "targeted_test_file": "test/auth.test.js",
            "targeted_test_name": "jwt auth test",
        },
        "id": "call-val",
    }

    mock_resp0 = MagicMock()
    mock_resp0.content = "Inspecting source code..."
    mock_resp0.tool_calls = [tool_call_inspect]

    mock_resp1 = MagicMock()
    mock_resp1.content = "Planning fix..."
    mock_resp1.tool_calls = [tool_call_record]

    mock_resp2 = MagicMock()
    mock_resp2.content = "Replacing symbol..."
    mock_resp2.tool_calls = [tool_call_edit]

    mock_resp_val = MagicMock()
    mock_resp_val.content = "Validating syntax..."
    mock_resp_val.tool_calls = [tool_call_val]

    mock_resp4 = MagicMock()
    mock_resp4.content = "Fix complete."
    mock_resp4.tool_calls = []

    mock_llm.bind_tools.return_value.invoke.side_effect = [
        mock_resp0,
        mock_resp1,
        mock_resp2,
        mock_resp_val,
        mock_resp4,
    ]

    with (
        patch("remediation_engine.agents.workaround.ChatOpenAI", return_value=mock_llm),
        patch("remediation_engine.agents.workaround.DockerSandbox") as mock_sandbox_cls,
        patch("remediation_engine.tools.source.parse_source") as mock_parse,
        patch("remediation_engine.tools.source.find_named_symbol") as mock_find_sym,
    ):
        sandbox_instance = mock_sandbox_cls.return_value.__enter__.return_value
        source_text = "const expressJwt = require('express-jwt');\nfunction authMiddleware() { return expressJwt({ secret: 'secret' }); }"

        def read_file(path):
            if path == "package.json":
                return json.dumps({"scripts": {"test": "node --test test/*.test.js"}})
            if path == "tsconfig.json":
                return None
            return source_text

        sandbox_instance.read_file.side_effect = read_file
        sandbox_instance.run.return_value.exit_code = 0
        sandbox_instance.run.return_value.stdout = "SUCCESS"

        mock_tree = MagicMock()
        mock_parse.return_value = mock_tree
        mock_find_sym.return_value = {
            "symbol_name": "authMiddleware",
            "node_type": "function_declaration",
            "start_line": 1,
            "end_line": 1,
            "text": "function authMiddleware() { return expressJwt({ secret: 'secret' }); }",
        }
        res = run_workaround_subagent_node(state1)

        assert "worker_results_by_attempt" in res
        attempt_res = res["worker_results_by_attempt"]["att-1"]
        assert attempt_res.status == AgentActionStatus.SUCCESS
        assert attempt_res.replay_plan is not None
        assert attempt_res.replay_plan.security_invariants == ["JWT auth must be enforced"]
        messages = mock_llm.bind_tools.return_value.invoke.call_args_list[0].args[0]
        assert "code security specialist" in messages[0].content
        assert "DYNAMIC WORKAROUND CONTEXT" in messages[-1].content
        assert "routes/auth.js" not in _WORKAROUND_STATIC_INSTRUCTIONS


def test_workaround_schemas_and_serialization() -> None:
    """Verify WorkaroundEdit and WorkaroundReplayPlan contracts."""
    edit1 = WorkaroundEdit(
        file_path="lib/insecurity.ts",
        old_text="expressJwt({",
        new_text="expressjwt({",
        edit_index=1,
    )
    assert edit1.file_path == "lib/insecurity.ts"
    assert edit1.edit_index == 1

    edit_set = WorkaroundEditSet(
        patch_id="patch-1",
        affected_files=["lib/insecurity.ts"],
        replacements=[edit1],
    )
    plan = WorkaroundReplayPlan(
        task_id="task-1",
        pre_attempt_snapshots={"lib/insecurity.ts": "const x = 1;"},
        successful_edit_sets=[edit_set],
        source_attempt_id="att-100",
    )
    assert plan.task_id == "task-1"
    assert len(plan.successful_edit_sets) == 1
    assert plan.pre_attempt_snapshots["lib/insecurity.ts"] == "const x = 1;"

    worker_result = WorkerAttemptResult(
        attempt_id="att-100",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="digest-123",
        replay_plan=plan,
    )
    assert worker_result.replay_plan is not None
    assert worker_result.replay_plan.task_id == "task-1"


def test_targeted_search_query_generation_in_prompt() -> None:
    """Verify that QA error context gives the worker evidence for query selection."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="grp-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix express-jwt missing algorithms",
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    group = VulnerabilityGroup(
        group_id="grp-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2020-15084"],
        representative_issue_id=str(uuid4()),
    )

    prompt = _build_workaround_prompt(
        target_task=task,
        target_group=group,
        previous_feedback='TypeError: "algorithms should be set"',
    )

    assert "=== RECOMMENDED INITIAL SEARCH QUERY ===" in prompt
    assert "Scenario: update_mitigates_cve_but_breaks_tests" in prompt
    assert "express-jwt" in prompt
    assert "algorithms should be set" in prompt


def test_supervisor_commits_matching_replay_plan_and_ignores_stale() -> None:
    """Verify Supervisor only commits replay plans for matching active attempt snapshots."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="grp-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Fix express-jwt missing algorithms",
        status=TaskStatus.PENDING,
        current_attempt_id="att-1",
        task_revision=1,
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=1,
        instruction="Fix express-jwt missing algorithms",
        instruction_digest="digest-1",
        dispatch_node="workaround_subagent",
        allowed_target_versions=[],
        allowed_dependency_types=[],
    )
    edit_obj = WorkaroundEdit(
        file_path="file.js",
        old_text="const orig = 1;",
        new_text="const orig = 2;",
        edit_index=1,
    )
    replay_plan = WorkaroundReplayPlan(
        task_id="task-1",
        pre_attempt_snapshots={"file.js": "const orig = 1;"},
        successful_edit_sets=[
            WorkaroundEditSet(
                patch_id="patch-1",
                affected_files=["file.js"],
                replacements=[edit_obj],
            )
        ],
        source_attempt_id="att-1",
    )
    matching_result = WorkerAttemptResult(
        attempt_id="att-1",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="digest-1",
        replay_plan=replay_plan,
        changed_files=["file.js"],
    )

    group = VulnerabilityGroup(
        group_id="grp-1",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        representative_issue_id=str(uuid4()),
    )

    state: dict[str, Any] = {
        "repo_root": "/fake/repo",
        "valid_groups": [group],
        "task_queue": {"task-1": task},
        "active_target_task_ids": ["task-1"],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-1": matching_result},
    }

    out = run_supervisor_node(state)
    assert "workaround_replay_plans_by_task" in out
    assert "task-1" in out["workaround_replay_plans_by_task"]
    committed_plan = out["workaround_replay_plans_by_task"]["task-1"]
    assert len(committed_plan.successful_edit_sets) == 1
    assert committed_plan.successful_edit_sets[0].replacements[0].new_text == "const orig = 2;"

    # Stale result test: mismatch in instruction_digest
    stale_result = WorkerAttemptResult(
        attempt_id="att-2",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        instruction_digest="wrong-digest",
        replay_plan=WorkaroundReplayPlan(task_id="task-1", source_attempt_id="att-2"),
    )
    stale_state: dict[str, Any] = {
        "repo_root": "/fake/repo",
        "valid_groups": [group],
        "task_queue": {"task-1": task},
        "active_target_task_ids": ["task-1"],
        "attempt_snapshots_by_id": {"att-1": snapshot},
        "worker_results_by_attempt": {"att-2": stale_result},
    }
    stale_out = run_supervisor_node(stale_state)
    assert "task-1" not in stale_out.get("workaround_replay_plans_by_task", {})


def test_is_authoritative_evidence_source() -> None:
    """Verify classification of authoritative vs non-authoritative evidence sources."""
    assert (
        _is_authoritative_evidence_source("https://github.com/advisories/GHSA-6g6m-m6h5-w9gf")
        is True
    )
    assert _is_authoritative_evidence_source("https://registry.npmjs.org/express-jwt") is True
    assert _is_authoritative_evidence_source("node_modules/express-jwt/README.md") is True
    assert _is_authoritative_evidence_source("https://stackoverflow.com/questions/12345") is False
    assert (
        _is_authoritative_evidence_source("https://github.com.evil.example/advisories/x") is False
    )
    assert _is_authoritative_evidence_source("official advisory from a package repository") is False
    assert _is_authoritative_evidence_source("search snippet only") is False
    assert _is_authoritative_evidence_source("") is False


def test_blocked_vs_failure_validation_classification() -> None:
    """Verify BLOCKED vs FAILURE returns in workaround validation tool."""
    mock_sandbox = MagicMock()

    # Case 1: missing node_modules / module -> BLOCKED
    def mock_run_no_nm(cmd, **kw):
        if "tsc" in cmd or "node" in cmd:
            return MagicMock(exit_code=1, stdout="", stderr="Cannot find module 'express-jwt'")
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run_no_nm
    mock_sandbox.read_file.return_value = "const x = 1;"
    touched: set[str] = set()
    tool = _make_validate_workaround_tool(
        mock_sandbox,
        touched,
        plan_state={"targeted_test_required": False},
    )
    res = tool.invoke(
        {
            "modified_files": ["lib/insecurity.ts"],
            "runtime_smoke_file": "lib/insecurity.ts",
        }
    )
    assert res.startswith("BLOCKED:")
    assert "Cannot find module" in res

    # Case 2: node_modules exists, typecheck fails with missing command -> BLOCKED
    def mock_run(cmd, **kw):
        if "test -d node_modules" in cmd:
            return MagicMock(exit_code=0, stdout="", stderr="")
        if "tsc" in cmd:
            return MagicMock(exit_code=127, stdout="", stderr="npx: command not found")
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run
    mock_sandbox.read_file.return_value = "const x = 1;"
    res2 = tool.invoke(
        {
            "modified_files": ["lib/insecurity.ts"],
            "runtime_smoke_file": "lib/insecurity.ts",
        }
    )
    assert res2.startswith("BLOCKED:")
    assert "npx: command not found" in res2

    # Case 3: typecheck fails with code type error -> FAILURE
    def mock_run_type_err(cmd, **kw):
        if "test -d node_modules" in cmd:
            return MagicMock(exit_code=0, stdout="", stderr="")
        if "tsc" in cmd:
            return MagicMock(
                exit_code=2,
                stdout="lib/insecurity.ts(5,1): error TS2304: Cannot find name 'foo'.",
                stderr="",
            )
        return MagicMock(exit_code=0, stdout="", stderr="")

    mock_sandbox.run.side_effect = mock_run_type_err
    res3 = tool.invoke(
        {
            "modified_files": ["lib/insecurity.ts"],
            "runtime_smoke_file": "lib/insecurity.ts",
        }
    )
    assert res3.startswith("FAILURE:")
    assert "TS2304" in res3


def test_subagent_runtime_drains_turn_and_stops_on_infrastructure_blocker() -> None:
    """Verify subagent loop drains all tool calls in turn before exiting on BLOCKED."""
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    tool1 = MagicMock()
    tool1.name = "validate_workaround"
    tool1.invoke.return_value = "BLOCKED: Workaround validation gate 'environment' blocked.\nDiagnostic: node_modules missing"

    tool2 = MagicMock()
    tool2.name = "read_repository_map"
    tool2.invoke.return_value = "tree output"

    ai_msg = MagicMock(
        content="",
        tool_calls=[
            {
                "name": "validate_workaround",
                "args": {"modified_files": ["lib/insecurity.ts"]},
                "id": "call_1",
            },
            {"name": "read_repository_map", "args": {}, "id": "call_2"},
        ],
    )
    mock_llm.invoke.return_value = ai_msg

    touched: set[str] = set()
    result = run_bounded_subagent_loop(
        llm=mock_llm,
        tools=[tool1, tool2],
        initial_messages=[],
        touched_files=touched,
    )

    # LLM should only be invoked once
    assert mock_llm.invoke.call_count == 1
    # Both tool calls should be recorded
    assert len(result.tool_events) == 2
    assert result.tool_events[0].name == "validate_workaround"
    assert result.tool_events[1].name == "read_repository_map"
    # Errors should contain INFRASTRUCTURE_BLOCKER
    assert any("INFRASTRUCTURE_BLOCKER:" in err for err in result.errors)


def test_authoritative_evidence_enforcement_in_edit_tools() -> None:
    """Verify edit tools require authoritative evidence when require_authoritative_evidence is set."""
    mock_sandbox = MagicMock()
    mock_sandbox.read_file.return_value = "const x = 1;"
    mock_sandbox.run.return_value = MagicMock(exit_code=0, stdout="", stderr="")
    touched: set[str] = set()
    plan_state: dict[str, Any] = {
        "recorded": True,
        "phase": WorkaroundExecutionPhase.EXECUTE.value,
        "planned_files": ["lib/insecurity.ts"],
        "inspected_files": {"lib/insecurity.ts"},
        "require_authoritative_evidence": True,
        "has_authoritative_evidence": False,
        "evidence_ledger": ["https://github.com/expressjs/express-jwt"],
    }

    replacements = [
        {
            "file_path": "lib/insecurity.ts",
            "old_text": "const x = 1;",
            "new_text": "const x = 2;",
            "expected_occurrences": 1,
        }
    ]
    plan_state["planned_replacements"] = replacements
    edit_tool = _make_deterministic_apply_edit_set_tool(mock_sandbox, touched, plan_state)
    res = edit_tool.invoke({"replacements": replacements})

    assert res.startswith("ERROR:")
    assert "Authoritative evidence required" in res

    # Record plan with authoritative evidence source
    rec_tool = _make_record_plan_tool(plan_state)
    rec_tool.invoke(
        {
            "affected_files": ["lib/insecurity.ts"],
            "affected_symbols": ["x"],
            "security_invariant": "test invariant",
            "causal_hypothesis": "test hypothesis",
            "planned_replacements": [
                {
                    "file_path": "lib/insecurity.ts",
                    "old_text": "const x = 1;",
                    "new_text": "const x = 2;",
                }
            ],
            "evidence_source": (
                "Express JWT docs: https://github.com/expressjs/express-jwt; "
                "related reference: https://github.com/expressjs/express-jwt/issues/72"
            ),
        }
    )

    assert plan_state.get("has_authoritative_evidence") is True
    assert plan_state.get("evidence_source") == "https://github.com/expressjs/express-jwt"

    # Now edit should succeed
    res2 = edit_tool.invoke({"replacements": replacements})
    assert res2.startswith("SUCCESS:")


def test_compact_initial_mitigation_prompt():
    task = MagicMock(task_id="task-1", selected_version="1.2.3", instruction="Fix vulnerability")
    group = MagicMock(
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2026-1234"],
        ghsa_ids=[],
        fix_plan_candidates=[
            PackageFixPlanCandidate(
                issue_id=str(uuid4()),
                plan=FixPlan(
                    status=FixPlanStatus.WORKAROUND_FOUND,
                    workaround_snippets=["snippet1"],
                    instruction="Use the advisory workaround.",
                    strategy_used="test",
                ),
            )
        ],
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


def test_supervisor_task_instruction_is_not_truncated():
    instruction = "Apply the exact supervisor directive: " + ("inspect-this-file " * 120)
    task = MagicMock(task_id="task-long", selected_version="1.2.3", instruction=instruction)
    group = MagicMock(
        vulnerable_component="express-jwt",
        cve_ids=[],
        ghsa_ids=[],
    )
    group.issues = []

    prompt = _build_workaround_prompt(target_task=task, target_group=group)

    assert f"Task Instruction: {instruction.strip()}" in prompt


def test_java_qa_evidence_selects_source_test_and_excludes_it_from_source_files():
    qa_evidence = QAFailureEvidence(
        attempt_id="att-java-1",
        task_revision=1,
        exact_diagnostics=["mvn test FAILED"],
        failed_tests=["src/test/java/org/example/AuthTest.java: rejectsMissingAuth"],
        source_locations=[
            "src/main/java/org/example/Auth.java:27",
            "src/test/java/org/example/AuthTest.java:12",
        ],
        affected_files=["src/main/java/org/example/Auth.java"],
        raw_excerpt="Tests run: 1, Failures: 1, Errors: 0, Skipped: 0",
    )

    assert _preferred_targeted_test_files(qa_evidence) == [
        "src/test/java/org/example/AuthTest.java"
    ]
    assert _primary_non_test_source_files(qa_evidence) == ["src/main/java/org/example/Auth.java"]


def test_compact_qa_regression_repair_prompt():
    task = MagicMock(task_id="task-2", selected_version="8.5.1", instruction="Update package")
    group = MagicMock(
        vulnerable_component="express-jwt",
        cve_ids=["CVE-2026-9999"],
        ghsa_ids=[],
    )
    qa_ev = QAFailureEvidence(
        attempt_id="att-123",
        task_revision=5,
        exact_diagnostics=["npm test FAILED (exit 2)."],
        failed_tests=["tests/jwt.test.ts"],
        source_locations=["lib/insecurity.ts:42:10"],
        affected_files=["lib/insecurity.ts"],
        raw_excerpt=(
            "npm test FAILED (exit 2).\n"
            "Failing Tests:\n"
            "insecurity jwt middleware rejects an invalid token\n"
            "expected 401 to equal 500\n"
            "TypeError: jwt is not a function"
        ),
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
    assert (
        "dependency update and replayed edits are already present"
        in _WORKAROUND_STATIC_INSTRUCTIONS
    )
    assert "Diagnostic:\n" in prompt
    assert "insecurity jwt middleware rejects an invalid token" in prompt
    assert "expected 401 to equal 500" in prompt
    assert "TypeError: jwt is not a function" in prompt
    assert "lib/insecurity.ts" in prompt
    assert "Targeted Test File: tests/jwt.test.ts" in prompt

    # Verify omission of noisy/redundant fields
    assert "Source Attempt ID" not in prompt
    assert "Source Revision" not in prompt
    assert "Raw Excerpt" not in prompt


def test_search_recommendation_precedence():
    task = MagicMock(selected_version="2.0.0")
    group = MagicMock(
        vulnerable_component="lodash",
        cve_ids=["CVE-2025-1111"],
        ghsa_ids=[],
        fix_plan_candidates=[
            PackageFixPlanCandidate(
                issue_id=str(uuid4()),
                plan=FixPlan(
                    status=FixPlanStatus.VERSION_FOUND,
                    fixed_version="2.0.0",
                    instruction="Upgrade lodash to 2.0.0.",
                    strategy_used="test",
                ),
            )
        ],
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
        fix_plan_candidates=[
            PackageFixPlanCandidate(
                issue_id=str(uuid4()),
                plan=FixPlan(
                    status=FixPlanStatus.NO_FIX,
                    instruction="No upstream fix is available.",
                    strategy_used="test",
                ),
            )
        ],
    )
    rec_nofix = _workaround_search_recommendation(task, group_nofix, context_scanner, None)
    assert rec_nofix.scenario == "no_update_available_to_resolve_cve"

    # 4. Otherwise -> initial_code_workaround_or_isolation
    rec_init = _workaround_search_recommendation(task, group, context_scanner, None)
    assert rec_init.scenario == "initial_code_workaround_or_isolation"


def test_search_recommendation_uses_test_details_instead_of_runner_status():
    task = MagicMock(selected_version="2.0.0")
    group = MagicMock(
        vulnerable_component="sanitize-html",
        cve_ids=[],
        ghsa_ids=[],
    )
    group.issues = []
    context = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        qa_evidence=QAFailureEvidence(
            exact_diagnostics=["npm test FAILED (exit 2)."],
            failed_tests=[
                "insecurity sanitizeHtml can be bypassed by exploiting lack of recursive sanitization"
            ],
            raw_excerpt=(
                "npm test FAILED (exit 2).\n"
                "Detected Failures: 1\n\n"
                "Failing Tests:\n"
                "1. insecurity sanitizeHtml can be bypassed by exploiting lack of recursive sanitization\n"
                "expected '<iframe src=javascript:alert(xss...)>' to equal '<iframe src=javascript:alert(xss...)>'"
            ),
        ),
    )

    recommendation = _workaround_search_recommendation(task, group, context, None)

    assert "npm test FAILED" not in recommendation.initial_query
    assert "sanitizeHtml can be bypassed" in recommendation.initial_query
    assert "expected" in recommendation.initial_query


def test_search_recommendation_keeps_api_exception_detail():
    task = MagicMock(selected_version="8.5.1")
    group = MagicMock(
        vulnerable_component="jsonwebtoken",
        cve_ids=[],
        ghsa_ids=[],
    )
    group.issues = []
    context = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        qa_evidence=QAFailureEvidence(
            exact_diagnostics=["npm test FAILED (exit 2)."],
            raw_excerpt="TypeError: jwt.sign is not a function",
        ),
    )

    recommendation = _workaround_search_recommendation(task, group, context, None)

    assert "npm test FAILED" not in recommendation.initial_query
    assert "TypeError: jwt.sign is not a function" in recommendation.initial_query


def _lifecycle_mock_sandbox(files: dict[str, str] | None = None):
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
    plan_state: dict[str, Any] = {
        "local_investigation_complete": True,
        "inspected_files": {"src/index.js"},
    }
    record_plan = _make_record_plan_tool(plan_state)
    res = record_plan.invoke(
        {
            "affected_files": ["src/index.js"],
            "affected_symbols": ["handleAuth"],
            "security_invariant": "Always validate token algorithm",
            "causal_hypothesis": "Missing algorithm check",
            "planned_replacements": [
                {"file_path": "src/index.js", "old_text": "a", "new_text": "b"}
            ],
            "evidence_source": "",
        }
    )
    assert "ERROR: [PLAN_REJECTED]" in str(res)


def test_plan_rejected_for_uninspected_files():
    plan_state: dict[str, Any] = {
        "local_investigation_complete": True,
        "inspected_files": {"src/index.js"},
    }
    record_plan = _make_record_plan_tool(plan_state)
    res = record_plan.invoke(
        {
            "affected_files": ["src/other.js"],
            "affected_symbols": ["handleAuth"],
            "security_invariant": "Always validate token algorithm",
            "causal_hypothesis": "Missing algorithm check",
            "planned_replacements": [
                {"file_path": "src/other.js", "old_text": "a", "new_text": "b"}
            ],
            "evidence_source": "https://github.com/advisories/GHSA-1234",
        }
    )
    assert "ERROR: [PLAN_REJECTED]" in str(res)
    assert "src/other.js" in str(res)


def test_qa_repair_plan_rejects_non_authoritative_evidence_before_acceptance():
    plan_state: dict[str, Any] = {
        "require_authoritative_evidence": True,
        "local_investigation_complete": True,
        "inspected_files": {"src/index.js"},
    }
    record_plan = _make_record_plan_tool(plan_state)

    result = record_plan.invoke(
        {
            "affected_files": ["src/index.js"],
            "affected_symbols": ["handleAuth"],
            "security_invariant": "Authorization remains enforced",
            "causal_hypothesis": "The dependency changed its export shape",
            "planned_replacements": [
                {"file_path": "src/index.js", "old_text": "old", "new_text": "new"}
            ],
            "evidence_source": "workspace:src/index.js",
        }
    )

    assert "ERROR: [MISSING_EVIDENCE]" in str(result)
    assert plan_state.get("recorded", False) is False


def test_qa_repair_plan_rejects_authority_claim_not_recorded_by_a_reader():
    plan_state: dict[str, Any] = {
        "require_authoritative_evidence": True,
        "local_investigation_complete": True,
        "inspected_files": {"src/index.js"},
    }
    record_plan = _make_record_plan_tool(plan_state)

    result = record_plan.invoke(
        {
            "affected_files": ["src/index.js"],
            "affected_symbols": ["handleAuth"],
            "security_invariant": "Authorization remains enforced",
            "causal_hypothesis": "The dependency changed its export shape",
            "planned_replacements": [
                {"file_path": "src/index.js", "old_text": "old", "new_text": "new"}
            ],
            "evidence_source": "https://github.com/advisories/GHSA-1234",
        }
    )

    assert "ERROR: [MISSING_EVIDENCE]" in str(result)
    assert plan_state.get("recorded", False) is False


def test_edits_rejected_before_valid_plan():
    sandbox, _ = _lifecycle_mock_sandbox({"src/index.js": "const x = 1;"})
    replacements = [
        {
            "file_path": "src/index.js",
            "old_text": "const x = 1;",
            "new_text": "const x = 2;",
            "expected_occurrences": 1,
        }
    ]
    plan_state: dict[str, Any] = {
        "phase": WorkaroundExecutionPhase.INVESTIGATE.value,
        "planned_replacements": replacements,
    }
    touched = set()
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, touched, plan_state)
    res = edit_tool.invoke({"replacements": replacements})
    assert "ERROR: [PLAN_VIOLATION]" in str(res) or "ERROR: [PHASE_VIOLATION]" in str(res)


def test_edits_rejected_after_one_edit_before_validation():
    sandbox, _ = _lifecycle_mock_sandbox({"src/index.js": "const x = 1;"})
    replacements = [
        {
            "file_path": "src/index.js",
            "old_text": "const x = 1;",
            "new_text": "const x = 2;",
            "expected_occurrences": 1,
        }
    ]
    plan_state: dict[str, Any] = {
        "recorded": True,
        "phase": WorkaroundExecutionPhase.EXECUTE.value,
        "planned_files": ["src/index.js"],
        "inspected_files": {"src/index.js"},
        "successful_edit_count_this_iteration": 1,
        "planned_replacements": replacements,
    }
    touched = {"src/index.js"}
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, touched, plan_state)
    res = edit_tool.invoke({"replacements": replacements})
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
            "planned_replacements": [
                {"file_path": "src/index.js", "old_text": "a", "new_text": "b"}
            ],
            "evidence_source": "workspace:src/index.js",
        }
    )
    assert "ERROR: [PHASE_VIOLATION]" in str(plan_res)

    search_web = _make_search_web_tool(plan_state=plan_state)
    web_res = search_web.invoke({"query": "express-jwt migration"})
    assert "ERROR: [PHASE_VIOLATION]" in str(web_res)


def test_validation_failure_resets_execution_to_investigate():
    sandbox, files = _lifecycle_mock_sandbox(
        {
            "src/index.js": "const x = 2;",
            "test/index.test.js": "test('index', () => {});",
        }
    )
    plan_state: dict[str, Any] = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "iteration": 1,
        "current_iteration_edit": MagicMock(
            file_path="src/index.js", old_text="const x = 1;", new_text="const x = 2;"
        ),
    }
    sandbox.run.side_effect = lambda cmd, timeout=60: MagicMock(
        exit_code=1, stdout="", stderr="Syntax error"
    )
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(
        sandbox, touched, plan_state, preferred_test_files=["test/index.test.js"]
    )
    res = val_tool.invoke(
        {"modified_files": ["src/index.js"], "runtime_smoke_file": "src/index.js"}
    )

    assert "FAILURE" in str(res)
    assert plan_state["phase"] == WorkaroundExecutionPhase.INVESTIGATE.value
    assert plan_state["iteration"] == 2


def test_runtime_smoke_targets_never_implicitly_skipped():
    sandbox, _ = _lifecycle_mock_sandbox({"src/index.js": "const x = 1;"})
    plan_state: dict[str, Any] = {}
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state)
    res = val_tool.invoke({"modified_files": ["src/index.js"], "runtime_smoke_file": ""})
    assert "FAILURE" in str(res)
    assert "runtime_smoke_file must be explicitly supplied" in str(res)


def test_runtime_smoke_rejects_test_files_and_keeps_target_separate():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "src/index.js": "const x = 1;",
            "test/index.test.js": "it('works', () => {});",
        }
    )
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(
        sandbox,
        {"src/index.js"},
        plan_state,
        preferred_test_files=["test/index.test.js"],
    )

    result = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "test/index.test.js",
            "targeted_test_file": "test/index.test.js",
        }
    )

    assert "ERROR: [INVALID_RUNTIME_SMOKE]" in str(result)
    assert "test/spec file" in str(result)
    assert sandbox.run.call_count == 0


def test_runtime_smoke_timeout_is_blocked_and_recorded_as_infrastructure():
    sandbox, _ = _lifecycle_mock_sandbox({"src/index.js": "const x = 1;"})
    sandbox.run.side_effect = [
        MagicMock(exit_code=0, stdout="", stderr=""),
        MagicMock(exit_code=1, stdout="", stderr=""),
        MagicMock(exit_code=124, stdout="", stderr="Command timed out after 30s."),
    ]
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(sandbox, {"src/index.js"}, plan_state)

    result = val_tool.invoke(
        {"modified_files": ["src/index.js"], "runtime_smoke_file": "src/index.js"}
    )

    assert "BLOCKED: Runtime smoke gate unavailable" in str(result)
    assert '"overall_status":"BLOCKED"' in str(result)
    assert "Command timed out after 30s." in plan_state["infrastructure_failure_details"]


def test_runtime_smoke_replaces_bootstrap_target_with_changed_source_module():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "app.ts": "validateDependenciesBasic(); server.start();",
            "src/auth.ts": "export const authorize = () => true;",
        }
    )
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(sandbox, {"src/auth.ts"}, plan_state)

    result = val_tool.invoke({"modified_files": ["src/auth.ts"], "runtime_smoke_file": "app.ts"})

    assert "SUCCESS:" in str(result)
    assert "selected lightweight source module 'src/auth.ts'" in str(result)
    assert plan_state["runtime_smoke_file"] == "src/auth.ts"
    assert all("app.ts" not in call.args[0] for call in sandbox.run.call_args_list)


def test_runtime_smoke_failure_after_fallback_is_not_hidden_by_selection_note():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "app.ts": "validateDependenciesBasic(); server.start();",
            "src/auth.ts": "export const authorize = () => true;",
        }
    )
    sandbox.run.side_effect = [
        MagicMock(exit_code=0, stdout="", stderr=""),
        MagicMock(exit_code=0, stdout="", stderr=""),
        MagicMock(exit_code=0, stdout="", stderr=""),
        MagicMock(exit_code=1, stdout="", stderr="TypeError: bad import"),
    ]
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(sandbox, {"src/auth.ts"}, plan_state)

    result = val_tool.invoke({"modified_files": ["src/auth.ts"], "runtime_smoke_file": "app.ts"})

    assert "validation gate 'runtime_smoke' failed" in str(result)
    assert plan_state["validation_passed"] is False


def test_runtime_smoke_rejects_bootstrap_when_no_safe_changed_module_exists():
    sandbox, _ = _lifecycle_mock_sandbox({"app.ts": "validateDependenciesBasic(); server.start();"})
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(sandbox, {"app.ts"}, plan_state)

    result = val_tool.invoke({"modified_files": ["app.ts"], "runtime_smoke_file": "app.ts"})

    assert "ERROR: [INVALID_RUNTIME_SMOKE]" in str(result)
    assert "no lightweight changed source module" in str(result)
    assert sandbox.run.call_count == 0


def test_invalid_validation_preflight_does_not_consume_gate_attempts():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "package.json": '{"scripts": {"test": "node --test test/index.test.js"}}',
            "src/index.js": "module.exports = true;",
            "test/index.test.js": "test('works', () => {});",
        }
    )
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(
        sandbox,
        {"src/index.js"},
        plan_state,
        preferred_test_files=["test/index.test.js"],
    )

    invalid_smoke = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "test/index.test.js",
            "targeted_test_file": "test/index.test.js",
        }
    )
    invalid_test = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "src/index.js",
            "targeted_test_file": "test/missing.test.js",
        }
    )

    assert "[INVALID_VALIDATION_INPUT]" in str(invalid_smoke)
    assert str(invalid_test).startswith("SUCCESS: Workaround validation gate passed")
    assert "canonical QA test 'test/index.test.js'" in str(invalid_test)
    assert plan_state["validation_calls"] == 1
    assert plan_state["validation_input_errors"] == 1
    assert sandbox.run.call_count > 0


def test_validation_canonicalizes_directory_and_source_test_selection() -> None:
    """Bad directory/source selections resolve to the known QA test and smoke module."""
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "package.json": '{"scripts": {"test": "node --test test/server/b2bOrderSpec.ts"}}',
            "app.ts": "validateDependenciesBasic(); server.start();",
            "routes/b2bOrder.ts": "export const order = true;",
            "test/server/b2bOrderSpec.ts": "test('b2bOrder', () => {});",
        }
    )
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(
        sandbox,
        {"routes/b2bOrder.ts"},
        plan_state,
        preferred_test_files=["test/server/b2bOrderSpec.ts"],
    )

    result = val_tool.invoke(
        {
            "modified_files": ["routes/b2bOrder.ts"],
            "runtime_smoke_file": "app.ts",
            "targeted_test_file": "routes/b2bOrder.ts",
            "targeted_test_name": "b2bOrder",
        }
    )

    assert str(result).startswith("SUCCESS: Workaround validation gate passed")
    assert "targeted test: test/server/b2bOrderSpec.ts" in str(result)
    assert plan_state["runtime_smoke_file"] == "routes/b2bOrder.ts"
    assert plan_state["validation_calls"] == 1
    assert plan_state["validation_input_errors"] == 0


def test_source_module_cannot_be_used_as_targeted_test():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "routes/order.ts": "export const order = true;",
            "lib/utils.ts": "export const utils = true;",
        }
    )
    plan_state: dict[str, Any] = {"targeted_test_required": True}
    val_tool = _make_validate_workaround_tool(sandbox, {"routes/order.ts"}, plan_state)

    result = val_tool.invoke(
        {
            "modified_files": ["routes/order.ts"],
            "runtime_smoke_file": "lib/utils.ts",
            "targeted_test_file": "routes/order.ts",
        }
    )

    assert "source module, not a test/spec file" in str(result)
    assert plan_state["validation_calls"] == 0
    assert plan_state["validation_input_errors"] == 1
    assert sandbox.run.call_count == 0


def test_valid_validation_increments_only_executed_gate_attempts():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "package.json": '{"scripts": {"test": "node --test test/index.test.js"}}',
            "src/index.js": "module.exports = true;",
            "test/index.test.js": "test('works', () => {});",
        }
    )
    plan_state: dict[str, Any] = {}
    val_tool = _make_validate_workaround_tool(
        sandbox,
        {"src/index.js"},
        plan_state,
        preferred_test_files=["test/index.test.js"],
    )

    result = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "src/index.js",
            "targeted_test_file": "test/index.test.js",
        }
    )

    assert str(result).startswith("SUCCESS:")
    assert plan_state["validation_calls"] == 1
    assert plan_state["validation_input_errors"] == 0


def test_missing_sqlite_native_bindings_classified_as_infra_only():
    err = "Error: Cannot find module 'better-sqlite3' or missing native bindings"
    assert _is_infrastructure_failure(err) is True


def test_missing_bare_dependency_is_classified_as_infra_only():
    assert _is_infrastructure_failure("Error: Cannot find module 'undici'") is True
    assert _is_infrastructure_failure("Error: Cannot find module './auth.js'") is False


def test_assertion_failures_not_classified_as_infra_only():
    err = (
        "AssertionError: expected 1 to equal 2\n    at Context.<anonymous> (test/index.test.js:10)"
    )
    assert _is_infrastructure_failure(err) is False


def test_targeted_test_timeout_is_infrastructure_only():
    assert _is_infrastructure_failure("Command timed out after 180s.") is True


def test_alternative_test_rejected_without_infra_failure():
    sandbox, _ = _lifecycle_mock_sandbox({"test/alt.test.js": "console.log('test');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": False,
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke(
        {
            "original_test": "test/orig.test.js",
            "alternative_test": "test/alt.test.js",
            "infrastructure_failure_evidence": "Assertion failure",
            "shared_behavior_explanation": "Tests auth invariant",
            "evidence_sources": ["GHSA-1234"],
            "infrastructure_avoidance_explanation": "Avoids native sqlite",
        }
    )
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)


def test_alternative_test_rejected_without_mapping_evidence():
    sandbox, _ = _lifecycle_mock_sandbox({"test/alt.test.js": "console.log('test');"})
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke(
        {
            "original_test": "test/orig.test.js",
            "alternative_test": "test/alt.test.js",
            "infrastructure_failure_evidence": "Missing native bindings",
            "shared_behavior_explanation": "Tests auth invariant",
            "evidence_sources": [],
            "infrastructure_avoidance_explanation": "Avoids native sqlite",
        }
    )
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)


def test_alternative_test_rejected_when_it_imports_unavailable_dependency():
    sandbox, _ = _lifecycle_mock_sandbox(
        {"test/alt.test.js": "const sqlite3 = require('sqlite3');"}
    )
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "latest_infra_diagnostics": "Cannot find module 'sqlite3'",
        "inspected_files": {"test/alt.test.js"},
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke(
        {
            "original_test": "test/orig.test.js",
            "alternative_test": "test/alt.test.js",
            "infrastructure_failure_evidence": "Missing sqlite3 native binding",
            "shared_behavior_explanation": "Tests auth invariant",
            "evidence_sources": ["GHSA-1234"],
            "infrastructure_avoidance_explanation": "Avoids native sqlite",
        }
    )
    assert "ERROR: [SUBSTITUTION_REJECTED]" in str(res)
    assert "directly imports/invokes" in str(res)


def test_alternative_test_accepted_when_criteria_pass():
    sandbox, _ = _lifecycle_mock_sandbox(
        {"test/alt.test.js": "const jwt = require('jsonwebtoken');"}
    )
    plan_state: dict[str, Any] = {
        "latest_failed_targeted_test": "test/orig.test.js",
        "latest_test_failure_infra": True,
        "latest_infra_diagnostics": "Cannot find module 'better-sqlite3'",
        "inspected_files": {"test/alt.test.js"},
        "iteration": 1,
    }
    sub_tool = _make_record_targeted_test_substitution_tool(sandbox, plan_state)
    res = sub_tool.invoke(
        {
            "original_test": "test/orig.test.js",
            "alternative_test": "test/alt.test.js",
            "infrastructure_failure_evidence": "Missing native sqlite3",
            "shared_behavior_explanation": "Tests auth invariant",
            "evidence_sources": ["GHSA-1234"],
            "infrastructure_avoidance_explanation": "Uses mock DB instead of sqlite3",
        }
    )
    assert "SUCCESS:" in str(res)
    assert plan_state["accepted_alternative_test"] == "test/alt.test.js"
    assert plan_state["original_to_alternative_test_evidence"]["test/orig.test.js"] == ["GHSA-1234"]
    assert plan_state["original_to_alternative_test_details"]["test/orig.test.js"] == {
        "infrastructure_failure_evidence": "Missing native sqlite3",
        "shared_behavior_explanation": "Tests auth invariant",
        "infrastructure_avoidance_explanation": "Uses mock DB instead of sqlite3",
        "evidence_sources": "GHSA-1234",
    }


def test_real_workaround_toolbelt_requires_explicit_smoke_and_targeted_test():
    sandbox, _ = _lifecycle_mock_sandbox({"src/index.js": "module.exports = true;"})
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
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "package.json": '{"name": "test-app", "scripts": {"test": "vitest"}}',
            "src/index.js": "module.exports = { auth: () => true };",
            "test/alt.test.js": "const { auth } = require('../src/index');",
        }
    )
    plan_state: dict[str, Any] = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "accepted_alternative_test": "test/alt.test.js",
    }
    touched = {"src/index.js"}
    val_tool = _make_validate_workaround_tool(
        sandbox, touched, plan_state, preferred_test_files=["test/orig.test.js"]
    )
    res = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "src/index.js",
        }
    )
    assert "SUCCESS:" in str(res)
    assert plan_state["validated_files"] == ["src/index.js"]


def test_alternative_targeted_test_failure_preventing_optimistic_success():
    sandbox, _ = _lifecycle_mock_sandbox(
        {
            "package.json": '{"name": "test-app", "scripts": {"test": "vitest"}}',
            "src/index.js": "const x = 1;",
            "test/alt.test.js": "it('auth', () => {});",
        }
    )
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
    res = val_tool.invoke(
        {
            "modified_files": ["src/index.js"],
            "runtime_smoke_file": "src/index.js",
            "targeted_test_file": "test/alt.test.js",
        }
    )

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
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
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
        qa_policy=qa_policy_for_task(RoutingStrategy.CODE_WORKAROUND),
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


def _atomic_patch_mock_sandbox(initial_files: dict[str, str] | None = None):
    files = dict(initial_files or {})

    def read_file(path: str) -> str | None:
        norm = path.replace("\\", "/").lstrip("/")
        return files.get(norm)

    def write_file(path: str, content: str) -> None:
        norm = path.replace("\\", "/").lstrip("/")
        files[norm] = content

    def revert_file(path: str) -> None:
        pass

    sandbox = MagicMock()
    sandbox.read_file.side_effect = read_file
    sandbox.write_file.side_effect = write_file
    sandbox.revert_file.side_effect = revert_file
    sandbox.run.return_value = MagicMock(exit_code=0, stdout="SUCCESS", stderr="")
    return sandbox, files


def test_workaround_tools_expose_typed_flat_replacement_schema():
    """Ensure the model receives replacement item fields in both tool schemas."""
    sandbox, _ = _atomic_patch_mock_sandbox()
    plan_tool = _make_record_plan_tool({})
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, set(), {})

    plan_schema = plan_tool.args_schema.model_json_schema()
    edit_schema = edit_tool.args_schema.model_json_schema()

    replacement_def = plan_schema["$defs"]["WorkaroundPlannedReplacement"]
    assert replacement_def["properties"]["file_path"]["type"] == "string"
    assert replacement_def["properties"]["old_text"]["type"] == "string"
    assert replacement_def["properties"]["new_text"]["type"] == "string"
    assert (
        plan_schema["properties"]["planned_replacements"]["items"]["$ref"]
        == "#/$defs/WorkaroundPlannedReplacement"
    )
    assert (
        edit_schema["properties"]["replacements"]["items"]["$ref"]
        == "#/$defs/WorkaroundPlannedReplacement"
    )
    assert "flat list" in plan_tool.description
    assert "flat replacement object" in edit_tool.description
    assert "pending until" in edit_tool.description
    assert "pre-iteration snapshot" in edit_tool.description
    assert '"file_path"' in plan_tool.description
    assert '"old_text"' in edit_tool.description


def test_typed_replacement_list_validation_success():
    """Verify recording a valid plan with WorkaroundPlannedReplacement items."""
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/auth.js"}}
    record_plan = _make_record_plan_tool(plan_state)

    replacements = [
        WorkaroundPlannedReplacement(
            file_path="src/auth.js",
            old_text="const expressJwt = require('express-jwt');",
            new_text="const { expressjwt } = require('express-jwt');",
            expected_occurrences=1,
            symbol_name="expressjwt",
        ).model_dump()
    ]

    res = record_plan.invoke(
        {
            "affected_files": ["src/auth.js"],
            "affected_symbols": ["expressjwt"],
            "security_invariant": "JWT algorithm must be enforced",
            "causal_hypothesis": "API migration in express-jwt v7+",
            "planned_replacements": replacements,
            "evidence_source": "https://github.com/expressjs/express-jwt",
        }
    )

    assert "SUCCESS:" in res
    assert plan_state["recorded"] is True
    assert plan_state["phase"] == WorkaroundExecutionPhase.EXECUTE.value
    assert len(plan_state["planned_replacements"]) == 1


def test_record_plan_rejections():
    """Verify all record_plan rejection conditions."""
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/a.js", "src/b.js"}}
    record_plan = _make_record_plan_tool(plan_state)

    # 1. Missing evidence_source
    res = record_plan.invoke(
        {
            "affected_files": ["src/a.js"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": [{"file_path": "src/a.js", "old_text": "a", "new_text": "b"}],
            "evidence_source": "",
        }
    )
    assert "ERROR: [PLAN_REJECTED] evidence_source is required" in res

    # 2. Uninspected file
    res = record_plan.invoke(
        {
            "affected_files": ["src/uninspected.js"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": [
                {"file_path": "src/uninspected.js", "old_text": "a", "new_text": "b"}
            ],
            "evidence_source": "docs",
        }
    )
    assert "ERROR: [PLAN_REJECTED] Target file 'src/uninspected.js' has not been inspected" in res

    # 3. Prohibited target
    res = record_plan.invoke(
        {
            "affected_files": ["package.json"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": [
                {"file_path": "package.json", "old_text": "a", "new_text": "b"}
            ],
            "evidence_source": "docs",
        }
    )
    assert "ERROR: [PROHIBITED_TARGET]" in res

    # 4. Mismatched affected_files
    res = record_plan.invoke(
        {
            "affected_files": ["src/a.js", "src/b.js"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": [{"file_path": "src/a.js", "old_text": "a", "new_text": "b"}],
            "evidence_source": "docs",
        }
    )
    assert "ERROR: [PLAN_REJECTED] Mismatch between declared affected_files" in res

    # 5. Duplicate replacement specifications
    res = record_plan.invoke(
        {
            "affected_files": ["src/a.js"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": [
                {"file_path": "src/a.js", "old_text": "a", "new_text": "b"},
                {"file_path": "src/a.js", "old_text": "a", "new_text": "b"},
            ],
            "evidence_source": "docs",
        }
    )
    assert "ERROR: [PLAN_REJECTED] Duplicate replacement specification" in res

    # 6. Exceeded replacement count (>16)
    repls_17 = [
        {"file_path": "src/a.js", "old_text": f"old_{i}", "new_text": f"new_{i}"} for i in range(17)
    ]
    res = record_plan.invoke(
        {
            "affected_files": ["src/a.js"],
            "affected_symbols": ["foo"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": repls_17,
            "evidence_source": "docs",
        }
    )
    assert "ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 16 replacements" in res


def test_express_jwt_atomic_migration_scenario():
    """Reproduce express-jwt migration as one atomic edit set containing import and 2 call sites."""
    initial_content = (
        "const expressJwt = require('express-jwt');\n"
        "app.use(expressJwt({ secret: 'secret1', algorithms: ['HS256'] }));\n"
        "router.use(expressJwt({ secret: 'secret2', algorithms: ['HS256'] }));\n"
    )
    sandbox, files = _atomic_patch_mock_sandbox({"src/app.js": initial_content})
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/app.js"}}

    rec_tool = _make_record_plan_tool(plan_state)
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, set(), plan_state)

    planned_repls = [
        {
            "file_path": "src/app.js",
            "old_text": "const expressJwt = require('express-jwt');",
            "new_text": "const { expressjwt } = require('express-jwt');",
            "expected_occurrences": 1,
        },
        {
            "file_path": "src/app.js",
            "old_text": "app.use(expressJwt({ secret: 'secret1', algorithms: ['HS256'] }));",
            "new_text": "app.use(expressjwt({ secret: 'secret1', algorithms: ['HS256'] }));",
            "expected_occurrences": 1,
        },
        {
            "file_path": "src/app.js",
            "old_text": "router.use(expressJwt({ secret: 'secret2', algorithms: ['HS256'] }));",
            "new_text": "router.use(expressjwt({ secret: 'secret2', algorithms: ['HS256'] }));",
            "expected_occurrences": 1,
        },
    ]

    rec_res = rec_tool.invoke(
        {
            "affected_files": ["src/app.js"],
            "affected_symbols": ["expressjwt"],
            "security_invariant": "JWT middleware must validate token signatures",
            "causal_hypothesis": "express-jwt v7 named export migration",
            "planned_replacements": planned_repls,
            "evidence_source": "https://github.com/expressjs/express-jwt",
        }
    )
    assert "SUCCESS:" in rec_res

    edit_res = edit_tool.invoke({"replacements": planned_repls})
    assert "SUCCESS: Atomic edit set" in edit_res

    updated = files["src/app.js"]
    assert "const { expressjwt } = require('express-jwt');" in updated
    assert "app.use(expressjwt({" in updated
    assert "router.use(expressjwt({" in updated
    assert "expressJwt" not in updated


def test_occurrence_mismatch_zero_writes():
    """Verify that occurrence count mismatch aborts with zero writes."""
    initial_content = "const x = 1;\nconst y = 1;\n"
    sandbox, files = _atomic_patch_mock_sandbox({"src/test.js": initial_content})
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/test.js"}}

    rec_tool = _make_record_plan_tool(plan_state)
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, set(), plan_state)

    planned_repls = [
        {
            "file_path": "src/test.js",
            "old_text": "= 1;",
            "new_text": "= 2;",
            "expected_occurrences": 1,  # Actual is 2
        }
    ]

    rec_tool.invoke(
        {
            "affected_files": ["src/test.js"],
            "affected_symbols": ["x"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": planned_repls,
            "evidence_source": "docs",
        }
    )

    res = edit_tool.invoke({"replacements": planned_repls})
    assert "ERROR: [OCCURRENCE_MISMATCH]" in res
    assert files["src/test.js"] == initial_content  # Zero writes made


def test_overlapping_replacement_rejection_zero_writes():
    """Verify overlapping replacement spans are rejected with zero writes."""
    content = "function test() { return 123; }"
    repls = [
        WorkaroundPlannedReplacement(
            file_path="src/a.js",
            old_text="test() { return 123; }",
            new_text="fn()",
            expected_occurrences=1,
        ),
        WorkaroundPlannedReplacement(
            file_path="src/a.js",
            old_text="return 123;",
            new_text="return 456;",
            expected_occurrences=1,
        ),
    ]

    updated, err = _apply_replacements_to_content(content, repls)
    assert err is not None
    assert "OVERLAPPING_REPLACEMENTS" in err
    assert updated == content


def test_multi_file_atomic_patch_and_syntax_failure_rollback():
    """Verify syntax error in 1 file restores all files in the multi-file edit set."""
    sandbox, files = _atomic_patch_mock_sandbox(
        {
            "src/a.js": "const a = 1;\n",
            "src/b.js": "const b = 1;\n",
        }
    )
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/a.js", "src/b.js"}}

    rec_tool = _make_record_plan_tool(plan_state)
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, set(), plan_state)

    planned_repls = [
        {
            "file_path": "src/a.js",
            "old_text": "const a = 1;",
            "new_text": "const a = 2;",
            "expected_occurrences": 1,
        },
        {
            "file_path": "src/b.js",
            "old_text": "const b = 1;",
            "new_text": "const b = invalid syntax {{{",
            "expected_occurrences": 1,
        },
    ]

    rec_tool.invoke(
        {
            "affected_files": ["src/a.js", "src/b.js"],
            "affected_symbols": ["a", "b"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": planned_repls,
            "evidence_source": "docs",
        }
    )

    def run_cmd(cmd: str, **kwargs):
        if "src/b.js" in cmd or "b.js" in cmd:
            return MagicMock(exit_code=1, stdout="SyntaxError: Unexpected token", stderr="")
        return MagicMock(exit_code=0, stdout="", stderr="")

    sandbox.run.side_effect = run_cmd

    res = edit_tool.invoke({"replacements": planned_repls})
    assert "ERROR: [SYNTAX_FAILURE]" in res
    assert plan_state["phase"] == WorkaroundExecutionPhase.INVESTIGATE.value
    assert plan_state["local_investigation_complete"] is False
    assert files["src/a.js"] == "const a = 1;\n"
    assert files["src/b.js"] == "const b = 1;\n"


def test_source_edit_restore_failure_keeps_primary_error_and_blocks_more_edits():
    sandbox, files = _atomic_patch_mock_sandbox(
        {"src/a.js": "const a = 1;\n", "src/b.js": "const b = 1;\n"}
    )
    plan_state = {
        "local_investigation_complete": True,
        "inspected_files": {"src/a.js", "src/b.js"},
    }
    touched: set[str] = set()
    rec_tool = _make_record_plan_tool(plan_state)
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, touched, plan_state)
    planned_replacements = [
        {
            "file_path": "src/a.js",
            "old_text": "const a = 1;",
            "new_text": "const a = 2;",
            "expected_occurrences": 1,
        },
        {
            "file_path": "src/b.js",
            "old_text": "const b = 1;",
            "new_text": "const b = invalid syntax {{{",
            "expected_occurrences": 1,
        },
    ]
    rec_tool.invoke(
        {
            "affected_files": ["src/a.js", "src/b.js"],
            "affected_symbols": ["a", "b"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": planned_replacements,
            "evidence_source": "docs",
        }
    )

    def write_file(path: str, content: str) -> None:
        if path == "src/a.js" and content == "const a = 1;\n":
            raise OSError("restore disk failure")
        files[path] = content

    sandbox.write_file.side_effect = write_file
    sandbox.run.side_effect = lambda command, **_kwargs: MagicMock(
        exit_code=1 if "src/b.js" in command else 0,
        stdout="SyntaxError: Unexpected token" if "src/b.js" in command else "",
        stderr="",
    )

    result = edit_tool.invoke({"replacements": planned_replacements})

    assert "ERROR: [SYNTAX_FAILURE]" in result
    assert "RESTORATION_INCOMPLETE" in result
    assert "restore disk failure" in result
    assert plan_state["restoration_failed"] is True
    assert plan_state["restoration_failure_paths"] == ["src/a.js"]
    assert files["src/a.js"] == "const a = 2;\n"
    assert files["src/b.js"] == "const b = 1;\n"
    assert touched == {"src/a.js"}

    retry = edit_tool.invoke({"replacements": planned_replacements})
    assert "RESTORATION_INCOMPLETE" in retry


def test_code_failure_rollback_requires_fresh_investigation():
    """A code-failing validation restores the edit and clears investigation state."""
    sandbox, files = _atomic_patch_mock_sandbox(
        {
            "src/a.ts": "const a = 2;\n",
            "tsconfig.json": "{}",
        }
    )
    sandbox.run.side_effect = lambda command, **_kwargs: MagicMock(
        exit_code=1 if "--no-install tsc" in command else 0,
        stdout="type error" if "--no-install tsc" in command else "",
        stderr="",
    )
    plan_state = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "local_investigation_complete": True,
        "pending_snapshots": {"src/a.ts": "const a = 1;\n"},
    }
    touched = {"src/a.ts"}
    validate = _make_validate_workaround_tool(sandbox, touched, plan_state, preferred_test_files=[])

    result = validate.invoke(
        {
            "modified_files": ["src/a.ts"],
            "runtime_smoke_file": "src/a.ts",
        }
    )

    assert "typecheck" in result
    assert files["src/a.ts"] == "const a = 1;\n"
    assert plan_state["phase"] == WorkaroundExecutionPhase.INVESTIGATE.value
    assert plan_state["local_investigation_complete"] is False
    assert touched == set()


def test_current_iteration_rollback_preserves_prior_accepted_file_evidence():
    """Restoring a later edit must retain a path changed by an accepted edit."""
    sandbox, files = _atomic_patch_mock_sandbox({"src/a.ts": "const a = 3;\n"})
    plan_state = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "iteration": 3,
        "validated_files": ["src/a.ts"],
        "pending_snapshots": {"src/a.ts": "const a = 2;\n"},
    }
    touched = {"src/a.ts"}

    errors = _revert_current_iteration_edit(sandbox, plan_state, touched)

    assert errors == []
    assert files["src/a.ts"] == "const a = 2;\n"
    assert touched == {"src/a.ts"}
    assert plan_state["iteration"] == 4
    assert plan_state["pending_snapshots"] == {}


def test_failed_current_iteration_rollback_retains_anchor_and_blocks_worker():
    sandbox, files = _atomic_patch_mock_sandbox(
        {"src/a.ts": "const a = 2;\n", "src/b.ts": "const b = 2;\n"}
    )

    def fail_second_restore(path, content):
        if path == "src/b.ts":
            raise OSError("disk write failed")
        files[path] = content

    sandbox.write_file.side_effect = fail_second_restore
    pending_snapshots = {"src/a.ts": "const a = 1;\n", "src/b.ts": "const b = 1;\n"}
    plan_state = {
        "phase": WorkaroundExecutionPhase.VALIDATE.value,
        "iteration": 3,
        "pending_snapshots": pending_snapshots,
    }
    touched = {"src/a.ts", "src/b.ts"}

    errors = _revert_current_iteration_edit(sandbox, plan_state, touched)

    assert errors == ["src/b.ts: disk write failed"]
    assert files["src/a.ts"] == "const a = 1;\n"
    assert files["src/b.ts"] == "const b = 2;\n"
    assert plan_state["pending_snapshots"] == pending_snapshots
    assert plan_state["iteration"] == 3
    assert plan_state["restoration_failed"] is True
    assert touched == {"src/b.ts"}

    validate = _make_validate_workaround_tool(sandbox, touched, plan_state)
    blocked = validate.invoke({"modified_files": ["src/a.ts"]})

    assert "RESTORATION_INCOMPLETE" in blocked
    assert sandbox.run.call_count == 0


def test_validation_outcomes_commit_and_rollback():
    """Verify validation outcomes: CODE_FAILURE rolls back set, INFRA_FAILURE retains set, PASS commits set."""
    sandbox, files = _atomic_patch_mock_sandbox(
        {
            "src/a.js": "const a = 1;\n",
            "package.json": '{"scripts": {"test": "jest"}}',
        }
    )
    plan_state = {"local_investigation_complete": True, "inspected_files": {"src/a.js"}}
    touched = set()

    rec_tool = _make_record_plan_tool(plan_state)
    edit_tool = _make_deterministic_apply_edit_set_tool(sandbox, touched, plan_state)
    val_tool = _make_validate_workaround_tool(sandbox, touched, plan_state, preferred_test_files=[])

    planned_repls = [
        {
            "file_path": "src/a.js",
            "old_text": "const a = 1;",
            "new_text": "const a = 2;",
            "expected_occurrences": 1,
        }
    ]

    rec_tool.invoke(
        {
            "affected_files": ["src/a.js"],
            "affected_symbols": ["a"],
            "security_invariant": "inv",
            "causal_hypothesis": "hyp",
            "planned_replacements": planned_repls,
            "evidence_source": "docs",
        }
    )

    edit_res = edit_tool.invoke({"replacements": planned_repls})
    assert "pending validation" in edit_res
    assert plan_state["pending_edit_set"] is not None

    val_res = val_tool.invoke(
        {
            "modified_files": ["src/a.js"],
            "runtime_smoke_file": "src/a.js",
        }
    )
    assert "SUCCESS: Workaround validation gate passed" in val_res
    assert plan_state["pending_edit_set"] is None
    assert len(plan_state["successful_edit_sets"]) == 1
    assert plan_state["successful_edit_sets"][0].replacements[0].new_text == "const a = 2;"


def test_replay_atomic_grouping_and_mismatch_rejection():
    """Verify replay applies edit sets atomically and aborts on anchor mismatch."""
    edit1 = WorkaroundEdit(
        file_path="src/a.js",
        old_text="const a = 1;",
        new_text="const a = 2;",
        expected_occurrences=1,
        patch_id="patch-1",
        replacement_index=0,
    )
    edit_set = WorkaroundEditSet(
        patch_id="patch-1",
        plan_revision=1,
        iteration=1,
        affected_files=["src/a.js"],
        replacements=[edit1],
    )
    replay_plan = WorkaroundReplayPlan(
        task_id="task-1",
        pre_attempt_snapshots={"src/a.js": "const a = 1;"},
        successful_edit_sets=[edit_set],
    )

    assert len(replay_plan.successful_edit_sets) == 1
    assert replay_plan.successful_edit_sets[0].patch_id == "patch-1"


def test_subagent_runtime_changed_files_inference():
    """Verify subagent_runtime changed-files inference for deterministic_apply_edit_set."""
    event = ToolEvent(
        name="deterministic_apply_edit_set",
        args={"replacements": [{"file_path": "src/a.js"}, {"file_path": "src/b.js"}]},
        content='SUCCESS: Applied.\nJSON: {"status": "SUCCESS", "affected_files": ["src/a.js", "src/b.js"]}',
    )

    inferred = _infer_changed_files(event)
    assert inferred == ["src/a.js", "src/b.js"]
