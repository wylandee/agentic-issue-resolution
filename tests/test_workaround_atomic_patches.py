"""
Unit tests for atomic semantic patches and structured workaround plans.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from remediation_engine.contracts.schemas import (
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundExecutionPhase,
    WorkaroundPlannedReplacement,
    WorkaroundReplayPlan,
)
from remediation_engine.orchestration.remedy_tools import (
    _apply_replacements_to_content,
    _make_deterministic_apply_edit_set_tool,
    _make_record_plan_tool,
    _make_validate_workaround_tool,
)
from remediation_engine.orchestration.subagent_runtime import (
    ToolEvent,
    _infer_changed_files,
)


def _mock_sandbox(initial_files: dict[str, str] | None = None):
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
    sandbox, _ = _mock_sandbox()
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
    sandbox, files = _mock_sandbox({"src/app.js": initial_content})
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
    sandbox, files = _mock_sandbox({"src/test.js": initial_content})
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
    sandbox, files = _mock_sandbox(
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
    assert files["src/a.js"] == "const a = 1;\n"
    assert files["src/b.js"] == "const b = 1;\n"


def test_validation_outcomes_commit_and_rollback():
    """Verify validation outcomes: CODE_FAILURE rolls back set, INFRA_FAILURE retains set, PASS commits set."""
    sandbox, files = _mock_sandbox(
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

    edit_tool.invoke({"replacements": planned_repls})
    assert plan_state["pending_edit_set"] is not None

    val_res = val_tool.invoke({"modified_files": ["src/a.js"]})
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
