"""
Unit and integration tests for the Workaround Subagent Refactor.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    FailureCategory,
    IssueType,
    QAAttemptResult,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    TaskAttemptSnapshot,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundExecutionPhase,
    WorkaroundPhase,
)
from remediation_engine.orchestration.qa_critic import (
    _attach_failure_evidence_to_evaluations,
    _detect_targeted_test_context,
    _QAExecutionResults,
    build_targeted_test_command,
    detect_test_runner,
    extract_qa_failure_evidence,
)
from remediation_engine.orchestration.remedy_tools import (
    _is_prohibited_target,
    build_workaround_toolbelt,
)
from remediation_engine.orchestration.subagent_runtime import (
    _stagnation_recovery_instruction,
)
from remediation_engine.orchestration.supervisor_node import (
    _create_attempt_snapshot,
    _qa_failure_evidence_for_workaround_retry,
)
from remediation_engine.orchestration.workaround_subagent import (
    _build_workaround_prompt,
    _preferred_targeted_test_files,
    _workaround_attempt_succeeded,
    run_workaround_subagent_node,
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
    )
    evaluations = {
        "group-1": QAEvaluation(
            task_id="group-1",
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
        {"task_queue": {task.task_id: task}},
    )

    evidence = enriched["group-1"].failure_evidence
    assert evidence is not None
    assert evidence.attempt_id == "attempt-7"
    assert evidence.task_revision == 4
    assert any("algorithms is a required option" in d for d in evidence.exact_diagnostics)


def test_supervisor_creates_workaround_context():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Mitigate express-jwt vulnerability",
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

    runner = detect_test_runner(mock_sandbox, "test/jwt.test.js")
    assert runner == "mocha"

    cmd = build_targeted_test_command("mocha", "test/jwt.test.js", "missing algorithms")
    assert cmd == "npm test -- test/jwt.test.js --grep 'missing algorithms'"

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
        evaluation=evaluation,
    )

    resolved = _qa_failure_evidence_for_workaround_retry(
        "task-1",
        "group-1",
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
        evaluation=evaluation,
    )

    resolved = _qa_failure_evidence_for_workaround_retry(
        "task-2",
        "group-workaround",
        {"task-1": evaluation},
        {"update-attempt-1": qa_result},
        related_task_ids=["task-1"],
        related_group_ids=["group-update"],
    )

    assert resolved == evidence


def test_preferred_targeted_test_files_come_from_qa_locations():
    evidence = QAFailureEvidence(
        source_locations=["/workspace/test/api/2fa.test.ts:2:1673"],
        affected_files=["/workspace/lib/insecurity.ts"],
    )

    assert _preferred_targeted_test_files(evidence) == ["test/api/2fa.test.ts"]


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
    assert "=== OPERATING PRINCIPLES ===" in prompt_initial
    assert "=== EDIT CHECKPOINT CONTRACT ===" in prompt_initial
    assert "CODE_FAILURE rolls back the entire pending edit set" in prompt_initial
    assert "re-include every required change" in prompt_initial
    assert "INFRA_FAILURE or BLOCKED retains the pending edit set" in prompt_initial

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
    assert "Previously validated edits remain" in prompt_repair


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
    terms = {
        "component": "express-jwt",
        "cve": "CVE-2026-1234",
        "qa_diagnostic": "is not a function",
    }

    tools = build_workaround_toolbelt(
        mock_sandbox,
        touched,
        host_repo_root=MagicMock(),
        plan_state=plan_state,
        mandatory_search_terms=terms,
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
        patch(
            "remediation_engine.orchestration.workaround_subagent.ChatOpenAI", return_value=mock_llm
        ),
        patch(
            "remediation_engine.orchestration.workaround_subagent.DockerSandbox"
        ) as mock_sandbox_cls,
        patch("remediation_engine.tools.code_map.parse_source") as mock_parse,
        patch("remediation_engine.tools.code_map.find_named_symbol") as mock_find_sym,
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
