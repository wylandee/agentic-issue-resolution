"""Focused regression tests for the supervisor-owned NO_FIX state machine."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from remediation_engine.contracts import NoFixMitigationStage
from remediation_engine.contracts.schemas import (
    CommandResult,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueType,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    TaskStatus,
    VulnerabilityGroup,
    WorkaroundContext,
    WorkaroundPhase,
)
from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt
from remediation_engine.orchestration.supervisor_node import (
    _create_attempt_snapshot,
    _deterministic_routing,
    _no_fix_decision_requires_fallback,
    _no_fix_failure_transition,
    _normalize_qa_evaluations_for_tasks,
)
from remediation_engine.orchestration.task_utils import (
    advance_no_fix_stage,
    build_initial_remediation_task,
    build_no_fix_retry_instruction,
)
from remediation_engine.orchestration.workaround_subagent import _build_workaround_prompt


def _group() -> VulnerabilityGroup:
    return VulnerabilityGroup(
        group_id="sca:package.json:notevil:NO_FIX",
        issue_type=IssueType.SCA,
        vulnerable_component="notevil",
        file_path="package.json",
        file_paths=["package.json"],
        cve_ids=["CVE-2021-23771"],
        ghsa_ids=["GHSA-8G4M-CJM2-96WQ"],
        representative_issue_id=str(uuid4()),
        fix_plan=FixPlan(
            status=FixPlanStatus.NO_FIX,
            instruction="No upstream patch or workaround was found. Inform the user.",
            strategy_used="test",
        ),
    )


def _task(stage: NoFixMitigationStage, *, retry_count: int = 0) -> RemediationTask:
    return RemediationTask(
        task_id="task-nofix",
        parent_group_id=_group().group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        strategy_stage="code_workaround",
        no_fix_stage=stage,
        retry_count=retry_count,
        status=TaskStatus.PENDING,
        instruction="stage instruction",
    )


def test_no_fix_contracts_round_trip_and_defaults():
    assert NoFixMitigationStage.PACKAGE_REMOVAL.value == "package_removal"
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
    )
    assert task.no_fix_stage is None
    assert RemediationTask.model_validate_json(task.model_dump_json()).no_fix_stage is None

    context = WorkaroundContext(
        phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
        no_fix_stage=NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        reset_prior_stage_workspace=True,
    )
    restored = WorkaroundContext.model_validate_json(context.model_dump_json())
    assert restored.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    assert restored.reset_prior_stage_workspace is True


def test_initial_and_non_no_fix_task_creation_are_distinct():
    no_fix_task = build_initial_remediation_task(_group(), "task-1")
    assert no_fix_task.strategy == RoutingStrategy.CODE_WORKAROUND
    assert no_fix_task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL
    assert "PACKAGE REMOVAL" in no_fix_task.instruction
    assert "Inform the user" not in no_fix_task.instruction

    version_group = _group().model_copy(
        update={
            "group_id": "version-group",
            "fix_plan": FixPlan(
                status=FixPlanStatus.VERSION_FOUND,
                fixed_version="1.2.3",
                instruction="update it",
                strategy_used="test",
            ),
        }
    )
    workaround_group = _group().model_copy(
        update={
            "group_id": "workaround-group",
            "fix_plan": FixPlan(
                status=FixPlanStatus.WORKAROUND_FOUND,
                workaround_snippets=["safe workaround"],
                instruction="use the workaround",
                strategy_used="test",
            ),
        }
    )
    assert build_initial_remediation_task(version_group, "task-2").no_fix_stage is None
    assert build_initial_remediation_task(workaround_group, "task-3").no_fix_stage is None


def test_no_fix_stage_transitions_are_exact_and_ignore_terminal_retries():
    package = _task(NoFixMitigationStage.PACKAGE_REMOVAL, retry_count=3)
    updates = advance_no_fix_stage(package)
    assert updates == {
        "status": TaskStatus.NEEDS_RETRY,
        "retry_count": 4,
        "no_fix_stage": NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
    }

    stage_two = _task(NoFixMitigationStage.VULNERABLE_CODE_REMOVAL, retry_count=4)
    terminal = advance_no_fix_stage(stage_two)
    assert terminal["status"] == TaskStatus.UNFIXABLE
    assert terminal["retry_count"] == 5
    assert terminal["no_fix_stage"] == NoFixMitigationStage.UNFIXABLE

    already_terminal = _task(NoFixMitigationStage.UNFIXABLE, retry_count=5)
    assert advance_no_fix_stage(already_terminal) == {}


def test_retry_instruction_uses_group_identifiers_and_failure_evidence():
    evaluation = QAEvaluation(
        task_id="task-nofix",
        passed=False,
        failure_category=FailureCategory.SECURITY_FLAG,
        retry_feedback="scanner still finds the package",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["CVE API remains reachable"],
            failed_tests=["test/security.test.js"],
            raw_excerpt="notevil call path",
        ),
    )
    instruction = build_no_fix_retry_instruction(
        _task(NoFixMitigationStage.VULNERABLE_CODE_REMOVAL).model_copy(
            update={"status": TaskStatus.NEEDS_RETRY}
        ),
        _group(),
        evaluation=evaluation,
    )
    assert "notevil" in instruction
    assert "CVE-2021-23771" in instruction
    assert "GHSA-8G4M-CJM2-96WQ" in instruction
    assert "CVE API remains reachable" in instruction
    assert "indirect callers" in instruction
    assert "manifests" in instruction
    assert "Hint:" in instruction


def test_transition_helper_builds_deterministic_stage_two_instruction_and_reset():
    updates, reset = _no_fix_failure_transition(
        _task(NoFixMitigationStage.PACKAGE_REMOVAL), _group()
    )
    assert reset is True
    assert updates["no_fix_stage"] == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
    assert updates["status"] == TaskStatus.NEEDS_RETRY
    assert "VULNERABLE CODE REMOVAL" in updates["instruction"]


def test_routing_and_llm_guardrails_keep_no_fix_on_one_current_stage():
    task = _task(NoFixMitigationStage.PACKAGE_REMOVAL)
    decision = _deterministic_routing(
        {task.task_id: task},
        {_group().group_id: _group()},
        {},
        {},
    )
    assert decision.next_node == "workaround_subagent"
    assert decision.target_task_ids == [task.task_id]

    stage_two = task.model_copy(
        update={
            "status": TaskStatus.NEEDS_RETRY,
            "no_fix_stage": NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        }
    )
    bad = decision.model_copy(
        update={
            "next_node": "update_subagent",
            "target_task_ids": [stage_two.task_id],
        }
    )
    assert _no_fix_decision_requires_fallback(bad, {stage_two.task_id: stage_two})


def test_attempt_snapshot_preserves_no_fix_stage_and_revision():
    task = _task(NoFixMitigationStage.PACKAGE_REMOVAL)
    snapshots = {}
    updated, snapshot = _create_attempt_snapshot(
        task,
        dispatch_node="workaround_subagent",
        snapshots_by_id=snapshots,
        state_revision=10,
    )
    assert updated.task_revision == 1
    assert snapshot.task_revision == updated.task_revision
    assert snapshot.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL


def test_qa_normalization_keeps_failure_evidence():
    evidence = QAFailureEvidence(exact_diagnostics=["diagnostic"])
    evaluation = QAEvaluation(
        task_id="group-id",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="retry",
        failure_evidence=evidence,
    )
    task = _task(NoFixMitigationStage.PACKAGE_REMOVAL)
    normalized = _normalize_qa_evaluations_for_tasks(
        {task.parent_group_id: evaluation},
        {task.task_id: task},
        [task.task_id],
    )
    assert normalized[task.task_id].failure_evidence == evidence


def test_no_fix_prompts_do_not_contradict_package_removal_or_stage_two_rules():
    task = _task(NoFixMitigationStage.PACKAGE_REMOVAL)
    package_prompt = _build_workaround_prompt(
        task,
        _group(),
        workaround_context=WorkaroundContext(
            phase=WorkaroundPhase.INITIAL_MITIGATION,
            no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        ),
    )
    assert "remove_no_fix_dependency" in package_prompt
    assert "Dependency update is already seeded" not in package_prompt
    assert "manual" in package_prompt.lower()

    stage_two_prompt = _build_workaround_prompt(
        task.model_copy(update={"no_fix_stage": NoFixMitigationStage.VULNERABLE_CODE_REMOVAL}),
        _group(),
        workaround_context=WorkaroundContext(
            phase=WorkaroundPhase.QA_REGRESSION_REPAIR,
            no_fix_stage=NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
        ),
    )
    assert "Keep the vulnerable package installed" in stage_two_prompt
    assert "NEVER modify package.json" in stage_two_prompt
    assert "NEVER modify test files" in stage_two_prompt


class _PackageSandbox:
    """Small in-memory sandbox for scoped package-removal tests."""

    def __init__(self, files: dict[str, str]):
        self.files = dict(files)
        self.commands: list[str] = []
        self.sync_result = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=0)

    def read_file(self, path: str):
        return self.files.get(path)

    def write_file(self, path: str, content: str):
        self.files[path] = content

    def run(self, command: str, **_kwargs):
        self.commands.append(command)
        return self.sync_result


def _package_tool_map(sandbox, plan_state):
    tools = build_workaround_toolbelt(
        sandbox,
        set(),
        Path("/tmp/host-repo"),
        plan_state=plan_state,
        no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        no_fix_package_name="notevil",
        no_fix_manifest_paths=["package.json"],
        no_fix_package_manager="npm",
    )
    return {tool.name: tool for tool in tools}


def test_scoped_package_removal_requires_plan_and_changes_only_authorized_manifest():
    sandbox = _PackageSandbox(
        {
            "package.json": json.dumps({"dependencies": {"notevil": "1.0.0"}}),
            "package-lock.json": '{"packages": {}}',
        }
    )
    plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    tools = _package_tool_map(sandbox, plan_state)
    blocked = tools["remove_no_fix_dependency"].invoke(
        {"requested_package": "notevil", "manifest_path": "package.json"}
    )
    assert "PLAN_VIOLATION" in blocked

    plan = tools["record_plan"].invoke(
        {
            "affected_files": ["package.json"],
            "affected_symbols": [],
            "security_invariant": "the vulnerable package is absent",
            "causal_hypothesis": "the direct dependency is unused",
            "planned_replacements": [],
            "evidence_source": "workspace:package.json",
            "package_removal_requested": True,
        }
    )
    assert "SUCCESS" in plan
    result = tools["remove_no_fix_dependency"].invoke(
        {"requested_package": "notevil", "manifest_path": "package.json"}
    )
    assert "SUCCESS" in result
    assert "notevil" not in json.loads(sandbox.files["package.json"]).get("dependencies", {})
    assert any(
        "npm install --package-lock-only --ignore-scripts" in cmd for cmd in sandbox.commands
    )


def test_scoped_package_removal_rolls_back_manifest_and_lockfile_on_sync_failure():
    original_manifest = json.dumps({"dependencies": {"notevil": "1.0.0"}})
    original_lock = '{"packages": {"node_modules/notevil": {}}}'
    sandbox = _PackageSandbox(
        {"package.json": original_manifest, "package-lock.json": original_lock}
    )
    sandbox.sync_result = CommandResult(
        exit_code=1, stdout="", stderr="sync failed", duration_seconds=0
    )
    plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    tools = _package_tool_map(sandbox, plan_state)
    tools["record_plan"].invoke(
        {
            "affected_files": ["package.json"],
            "affected_symbols": [],
            "security_invariant": "absent",
            "causal_hypothesis": "unused",
            "planned_replacements": [],
            "evidence_source": "workspace:package.json",
            "package_removal_requested": True,
        }
    )
    result = tools["remove_no_fix_dependency"].invoke(
        {"requested_package": "notevil", "manifest_path": "package.json"}
    )
    assert "ROLLBACK" in result
    assert sandbox.files["package.json"] == original_manifest
    assert sandbox.files["package-lock.json"] == original_lock


def test_transitive_package_fails_closed_without_lockfile_edit():
    sandbox = _PackageSandbox({"package.json": json.dumps({"dependencies": {"other": "1.0.0"}})})
    plan_state = {"local_investigation_complete": True, "web_search_performed": True}
    tools = _package_tool_map(sandbox, plan_state)
    tools["record_plan"].invoke(
        {
            "affected_files": ["package.json"],
            "affected_symbols": [],
            "security_invariant": "absent",
            "causal_hypothesis": "transitive package cannot be removed safely",
            "planned_replacements": [],
            "evidence_source": "workspace:package.json",
            "package_removal_requested": True,
        }
    )
    result = tools["remove_no_fix_dependency"].invoke(
        {"requested_package": "notevil", "manifest_path": "package.json"}
    )
    assert "NOT_APPLICABLE" in result
    assert sandbox.commands == []
