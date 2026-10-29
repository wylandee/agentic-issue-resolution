"""
Tests for the Phase 5 Supervisor Node.

Tests have been updated to use the task-centric architecture:
- task_queue (Dict[str, RemediationTask]) replaces aggregate status and retry projections
- target_task_ids replaces target_group_ids in SupervisorDecision
- AgentActionSummary.task_id replaces .group_id
- QAEvaluation.task_id replaces .group_id
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from pydantic import ValidationError

from remediation_engine.contracts import DecisionCode
from remediation_engine.contracts.schemas import (
    MAX_MULTI_PACKAGE_ACTION_SIZE,
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IncrementalScanRecord,
    IssueSource,
    IssueType,
    LocalizedIssue,
    MavenTargetOperation,
    MultiPackageAction,
    ODCScanEvidence,
    PackageFixPlanCandidate,
    PackageMutation,
    PackageScanOccurrence,
    PeerConflictEvidence,
    PortfolioPlan,
    QAAttemptResult,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    ScanScope,
    SCARemediationStage,
    Severity,
    SupervisorDecision,
    SupervisorRetryPlan,
    TaskAttemptSnapshot,
    TaskCluster,
    TaskScanRecord,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
    WorkaroundPhase,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.contracts.solver_models import PortfolioReplanRequest
from remediation_engine.contracts.version_policy import RegistryCandidate
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.supervisor_node import (
    MAX_RETRIES,
    _authorize_update_dispatch,
    _build_consistency_event,
    _build_multi_package_action,
    _calculate_cumulative_scan_metrics,
    _create_attempt_snapshot,
    _deterministic_routing,
    _incremental_scan_coverage_complete,
    _materialize_spawn_requests,
    _maven_attempt_target_operation,
    _maven_attempt_target_paths,
    _nonterminal_sca_task_ids,
    _normalize_target_task_ids_for_node,
    _ordered_update_candidates,
    _peer_conflict_escalation,
    _repair_invalid_planner_plans,
    _task_scan_record_from_qa_result,
    _validate_committed_state,
    _validated_delta_replay_task_ids,
    _verified_absent_package_tasks,
    instruction_digest,
    reconcile_phase5_state_before_teardown,
    run_supervisor_node,
    supervisor_router,
)
from remediation_engine.orchestration.task_utils import derive_initial_strategy
from remediation_engine.settings import AppSettings


def _issue() -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        message="Test issue",
        id=str(uuid4()),
    )


def _sca_group(group_id="g1", fix_status=FixPlanStatus.VERSION_FOUND) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="test-pkg",
        package_version="1.0.0",
        purl="pkg:npm/test-pkg@1.0.0",
        cve_id="CVE-2026-1000",
        ecosystem="npm",
        file_path="package-lock.json",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        declaration_type="dependencies",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )
    fix_plan = FixPlan(
        status=fix_status,
        fixed_version="1.2.3" if fix_status == FixPlanStatus.VERSION_FOUND else None,
        workaround_snippets=["test"] if fix_status == FixPlanStatus.WORKAROUND_FOUND else None,
        instruction="test",
        strategy_used="test",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="test-pkg",
        file_path="package.json",
        cve_ids=[],
        versions=[],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
        fix_plan_candidates=[PackageFixPlanCandidate(issue_id=issue.id, plan=fix_plan)],
        fix_plan=fix_plan,
    )


def _sast_group(group_id="g2") -> VulnerabilityGroup:
    fix_plan = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND,
        fixed_version=None,
        workaround_snippets=["disable it"],
        instruction="test",
        strategy_used="test",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SAST,
        vulnerable_component="test-func",
        file_path="src/index.js",
        cve_ids=[],
        versions=[],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=str(uuid4()),
        issues=[_issue()],
        fix_plan=fix_plan,
    )


def _make_task(
    task_id: str,
    group_id: str,
    strategy: RoutingStrategy = RoutingStrategy.VERSION_BUMP,
    status: TaskStatus = TaskStatus.PENDING,
    retry_count: int = 0,
) -> RemediationTask:
    return RemediationTask(
        task_id=task_id,
        parent_group_id=group_id,
        strategy=strategy,
        status=status,
        retry_count=retry_count,
    )


def _base_state(groups, **overrides) -> dict:
    state = {
        "repo_root": "/tmp/repo",
        "valid_groups": groups,
        "task_queue": {},
        "active_target_task_ids": [],
        "qa_evaluations": {},
        "action_summaries": [],
        "constraints_ledger": [],
        "feedback_by_group": {},
        "feedback_by_task": {},
        "eval_status": "",
        "status": "supervisor_entered",
    }
    state.update(overrides)

    # Test fixtures emit the same committed attempt envelope as graph
    # dispatch.  Production no longer accepts uncorrelated summaries.
    task_queue = dict(state.get("task_queue") or {})
    snapshots = dict(state.get("attempt_snapshots_by_id") or {})
    worker_results = dict(state.get("worker_results_by_attempt") or {})
    summaries = list(state.get("action_summaries") or [])
    normalized_summaries: list[AgentActionSummary] = []
    for summary in summaries:
        task = task_queue.get(summary.task_id)
        if task is None or summary.attempt_id is not None:
            normalized_summaries.append(summary)
            continue
        attempt_id = f"test-attempt-{summary.task_id}"
        task_revision = max(1, task.task_revision)
        task = task.model_copy(
            update={
                "current_attempt_id": attempt_id,
                "task_revision": task_revision,
                "instruction": task.instruction or "test worker instruction",
            }
        )
        task_queue[task.task_id] = task
        digest = instruction_digest(task.instruction)
        snapshot = TaskAttemptSnapshot(
            attempt_id=attempt_id,
            task_id=task.task_id,
            state_revision=1,
            task_revision=task_revision,
            strategy_stage=task.strategy_stage,
            qa_policy=task.qa_policy,
            selected_version=task.selected_version,
            instruction=task.instruction,
            instruction_digest=digest,
            dispatch_node=(
                "workaround_subagent"
                if task.strategy == RoutingStrategy.CODE_WORKAROUND
                else "update_subagent"
            ),
        )
        tagged_summary = summary.model_copy(
            update={
                "attempt_id": attempt_id,
                "task_revision": task_revision,
                "instruction_digest": digest,
            }
        )
        normalized_summaries.append(tagged_summary)
        snapshots[attempt_id] = snapshot
        worker_results[attempt_id] = WorkerAttemptResult(
            attempt_id=attempt_id,
            task_id=task.task_id,
            task_revision=task_revision,
            status=summary.status,
            action_summary=tagged_summary,
            changed_files=[],
            instruction_digest=digest,
        )
    state["task_queue"] = task_queue
    state["action_summaries"] = normalized_summaries
    if snapshots:
        state["attempt_snapshots_by_id"] = snapshots
    if worker_results:
        state["worker_results_by_attempt"] = worker_results
    return state


def _keep_portfolio_current(monkeypatch):
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_planner.fetch_registry_candidates",
        lambda *args, **kwargs: [],
        raising=False,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
        lambda *args, **kwargs: False,
    )


def _stub_portfolio_plan_tool(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    portfolio_tool = MagicMock(return_value={"status": "portfolio_ready", "errors": []})
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.run_portfolio_plan_tool",
        portfolio_tool,
    )
    return portfolio_tool


def _portfolio_stub():
    return SimpleNamespace(
        portfolio_plan_id="portfolio-current",
        task_ids=[],
        clusters=[],
        cluster_order=[],
        task_order=[],
        task_to_cluster={},
        diagnostics=[],
    )


# ===========================================================================
# Schema tests
# ===========================================================================


class TestSupervisorDecisionSchema:
    def test_valid_routes_accepted(self):
        decision = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["t1"],
            instructions="test",
            decision_reason="test",
        )
        assert decision.next_node == "update_subagent"
        deferred = SupervisorDecision(
            next_node="supervisor",
            target_task_ids=[],
            instructions="Reconcile a deferred task before selecting another action.",
            decision_reason="Registry verification is temporarily unavailable.",
        )
        assert deferred.next_node == "supervisor"

    def test_workaround_subagent_rejects_zero_or_two_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="workaround_subagent",
                target_task_ids=["t1", "t2"],
                instructions="test",
                decision_reason="test",
            )

    def test_update_subagent_rejects_zero_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )

    def test_update_subagent_retains_reusable_batch_shape(self):
        decision = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["t1", "t2"],
            instructions="test",
            decision_reason="direct batch compatibility",
        )
        assert decision.target_task_ids == ["t1", "t2"]

    def test_update_subagent_rejects_more_than_configured_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=[f"t{i}" for i in range(MAX_MULTI_PACKAGE_ACTION_SIZE + 1)],
                instructions="test",
                decision_reason="test",
            )

    def test_qa_critic_requires_non_empty_targets_and_teardown_rejects_them(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="qa_critic",
                target_task_ids=[],
                instructions="test",
                decision_reason="test",
            )
        decision = SupervisorDecision(
            next_node="qa_critic",
            target_task_ids=["t1"],
            instructions="test",
            decision_reason="test",
        )
        assert decision.target_task_ids == ["t1"]
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="teardown",
                target_task_ids=["t1"],
                instructions="test",
                decision_reason="test",
            )

    def test_overlapping_unfixable_and_targets_rejected(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=["t1"],
                unfixable_task_ids=["t1"],
                instructions="test",
                decision_reason="test",
            )


def test_deferred_consistency_action_is_mapped_to_typed_rejection():
    event = _build_consistency_event(
        error_code="REGISTRY_VERIFICATION_PENDING",
        task_id="task-1",
        expected_attempt_id=None,
        received_attempt_id=None,
        action="deferred",
        details="registry lookup is pending",
    )

    assert event.action == "rejected"


def test_registry_verification_pending_reenters_supervisor_without_forcing_final_scan(
    monkeypatch: pytest.MonkeyPatch,
):
    group = _sca_group("g-registry-pending", FixPlanStatus.VERSION_FOUND)
    task = _make_task("task-registry-pending", group.group_id)
    state = _base_state(
        [group],
        task_queue={task.task_id: task},
        portfolio_plan=_portfolio_stub(),
        portfolio_dirty=False,
        workspace_volume="workspace-volume",
        final_full_scan_enabled=False,
    )
    _keep_portfolio_current(monkeypatch)
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.registry_candidate_sets_for_context",
        lambda _context: ((), "registry temporarily unavailable"),
    )

    result = run_supervisor_node(state)

    assert result["decision_code"] == DecisionCode.REGISTRY_VERIFICATION_PENDING
    assert result["next_routing_step"] == "supervisor"
    assert result["task_queue"][task.task_id].status == TaskStatus.INCONCLUSIVE
    assert result["task_queue"][task.task_id].retry_count == task.retry_count


def test_attempt_snapshot_allowlists_default_for_legacy_payloads():
    """New candidate fields remain readable when historical snapshots omit them."""
    snapshot = TaskAttemptSnapshot(
        task_id="task-1",
        instruction="Update test-pkg.",
        instruction_digest=instruction_digest("Update test-pkg."),
        dispatch_node="update_subagent",
    )

    assert snapshot.allowed_target_versions == []
    assert snapshot.allowed_dependency_types == []


def test_ordered_update_candidates_excludes_attempted_versions_and_types():
    """A committed retry snapshot contains only unattempted Supervisor candidates."""
    task = _make_task("task-1", "g1").model_copy(
        update={
            "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
            "selected_version": "2.0.0",
            "target_package_name": "test-pkg",
            "target_dependency_type": "overrides",
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        strategy_stage=SCARemediationStage.PACKAGE_OVERRIDE,
        attempted_versions=["2.0.0"],
        candidate_versions_considered=["2.0.0", "2.1.0"],
        attempted_dependency_types=["overrides"],
        candidate_dependency_types=["overrides", "resolutions"],
    )

    versions, dependency_types = _ordered_update_candidates(task, diagnostics=diagnostics)

    assert versions == ["2.1.0"]
    assert dependency_types == ["pnpm_overrides", "resolutions"]


def test_create_attempt_snapshot_commits_update_authorizations_and_target_poms():
    """Candidate and Maven target paths are captured in the worker snapshot."""
    task = _make_task("task-1", "g1").model_copy(
        update={
            "instruction": "Update the committed dependency candidate.",
            "selected_version": "1.2.3",
        }
    )
    committed, snapshot = _create_attempt_snapshot(
        task,
        dispatch_node="update_subagent",
        snapshots_by_id={},
        state_revision=3,
        allowed_target_versions=["1.2.3", "1.4.0", "1.2.3"],
        allowed_dependency_types=["dependencies", "devDependencies", "dependencies"],
        target_manifest_paths=["modules\\widget\\pom.xml", "pom.xml", "modules/widget/pom.xml"],
        maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
    )

    assert committed.current_attempt_id == snapshot.attempt_id
    assert snapshot.allowed_target_versions == ["1.2.3", "1.4.0"]
    assert snapshot.allowed_dependency_types == ["dependencies", "devDependencies"]
    assert snapshot.target_manifest_paths == ["modules/widget/pom.xml", "pom.xml"]
    assert snapshot.maven_target_operation == MavenTargetOperation.UPDATE_DECLARATION


def test_attempt_target_poms_include_only_localized_maven_manifests():
    group = SimpleNamespace(
        localized_issues=[
            SimpleNamespace(
                package_manager="maven",
                manifest_file=r"modules\widget\pom.xml",
                version_property_file="pom.xml",
            ),
            SimpleNamespace(
                package_manager="npm",
                manifest_file="package.json",
                version_property_file=None,
            ),
        ]
    )

    assert _maven_attempt_target_paths(group, ProjectLanguage.JAVA) == [
        "modules/widget/pom.xml",
        "pom.xml",
    ]
    assert _maven_attempt_target_paths(group, ProjectLanguage.NODEJS) == []


def test_maven_attempt_operation_is_committed_from_exact_declaration_type():
    group = SimpleNamespace(
        localized_issues=[
            SimpleNamespace(
                package_manager="maven",
                manifest_file="pom.xml",
                version_property_file=None,
            )
        ]
    )
    task = _make_task("task-1", "g1")

    assert (
        _maven_attempt_target_operation(
            task.model_copy(update={"target_dependency_type": "dependencies"}),
            group,
            ProjectLanguage.JAVA,
        )
        == MavenTargetOperation.UPDATE_DECLARATION
    )
    assert (
        _maven_attempt_target_operation(
            task.model_copy(update={"target_dependency_type": "dependencyManagement"}),
            group,
            ProjectLanguage.JAVA,
        )
        == MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT
    )


def test_update_dispatch_authorization_requires_committed_selected_candidate():
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        candidate_versions_considered=["1.2.3", "1.4.0"],
    )
    valid_task = _make_task("task-1", "g1").model_copy(
        update={"selected_version": "1.2.3", "instruction": "Update test-pkg to 1.2.3."}
    )
    authorization = _authorize_update_dispatch(valid_task, diagnostics=diagnostics)

    assert authorization is not None
    assert authorization.selected_version == "1.2.3"
    assert authorization.allowed_target_versions == ("1.2.3", "1.4.0")

    stale_task = valid_task.model_copy(update={"selected_version": "9.9.9"})
    assert _authorize_update_dispatch(stale_task, diagnostics=diagnostics) is None


def test_retriage_solver_allowlist_replaces_stale_retry_candidate_pool():
    """A fresh solver plan can dispatch even when prior retry evidence is stale."""
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 4,
            "portfolio_plan_id": "portfolio-new",
            "target_package_name": "test-pkg",
            "target_dependency_type": "dependencies",
            "selected_version": "2.0.0",
            "allowed_target_versions": ["2.0.0"],
            "allowed_dependency_types": ["dependencies"],
            "instruction": "Update test-pkg to the newly solver-approved version 2.0.0.",
        }
    )
    stale_plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=2,
        selected_version="1.2.3",
        target_package_name="test-pkg",
        target_dependency_type="devDependencies",
        candidate_versions_considered=["1.2.3"],
        candidate_dependency_types=["devDependencies"],
        exact_instruction="Update test-pkg to 1.2.3.",
    )
    prior_attempt_diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        target_package_name="test-pkg",
        target_dependency_type="devDependencies",
        attempted_versions=["1.2.3"],
        candidate_versions_considered=["1.2.3"],
        attempted_dependency_types=["devDependencies"],
        candidate_dependency_types=["devDependencies"],
    )

    authorization = _authorize_update_dispatch(
        task,
        plan=stale_plan,
        diagnostics=prior_attempt_diagnostics,
    )

    assert authorization is not None
    assert authorization.selected_version == "2.0.0"
    assert authorization.allowed_target_versions == ("2.0.0",)
    assert authorization.allowed_dependency_types == ("dependencies",)


def test_create_attempt_snapshot_rejects_candidate_less_update():
    task = _make_task("task-1", "g1").model_copy(
        update={"instruction": "Update only with a committed candidate."}
    )

    with pytest.raises(ValueError, match="non-empty candidate authorization"):
        _create_attempt_snapshot(
            task,
            dispatch_node="update_subagent",
            snapshots_by_id={},
            state_revision=3,
        )


def test_committed_state_rejects_active_empty_update_authorization():
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": "attempt-1",
            "qa_policy": QAPolicy.VERSION_BUMP,
            "selected_version": "1.2.3",
            "instruction": "Update test-pkg to 1.2.3.",
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-1",
        task_id="task-1",
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        selected_version="1.2.3",
        instruction=task.instruction,
        instruction_digest=instruction_digest(task.instruction),
        dispatch_node="update_subagent",
    )

    events, errors = _validate_committed_state(
        {"task-1": task},
        {"attempt-1": snapshot},
        {},
        {},
        ["task-1"],
        "update_subagent",
    )

    assert any(event.error_code == "UPDATE_ATTEMPT_WITHOUT_CANDIDATE" for event in events)
    assert any("candidate authorization" in error for error in errors)


def test_committed_state_refuses_snapshot_revision_rewind():
    instruction = "Update test-pkg to 2.0.0."
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 4,
            "current_attempt_id": "attempt-current",
            "selected_version": "2.0.0",
            "instruction": instruction,
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        task_revision=1,
        selected_version="1.0.0",
        instruction="Update test-pkg to 1.0.0.",
        instruction_digest=instruction_digest("Update test-pkg to 1.0.0."),
        dispatch_node="update_subagent",
    )
    queue = {"task-1": task}

    events, errors = _validate_committed_state(
        queue, {snapshot.attempt_id: snapshot}, {}, {}, [], "teardown"
    )

    assert queue["task-1"].task_revision == 4
    assert queue["task-1"].selected_version == "2.0.0"
    assert queue["task-1"].instruction == instruction
    assert any(event.error_code == "TASK_SNAPSHOT_REVISION_REJECTED" for event in events)
    assert any("snapshot repair was refused" in error for error in errors)


def test_committed_state_rejects_foreign_task_snapshot_identity():
    instruction = "Update test-pkg to 1.0.0."
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": "attempt-current",
            "selected_version": "1.0.0",
            "instruction": instruction,
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-foreign",
        task_revision=1,
        selected_version="9.9.9",
        instruction="Update test-pkg to 9.9.9.",
        instruction_digest=instruction_digest("Update test-pkg to 9.9.9."),
        dispatch_node="update_subagent",
    )
    queue = {"task-1": task}

    events, errors = _validate_committed_state(
        queue, {snapshot.attempt_id: snapshot}, {}, {}, [], "teardown"
    )

    assert queue["task-1"].task_id == "task-1"
    assert queue["task-1"].selected_version == "1.0.0"
    assert queue["task-1"].instruction == instruction
    assert any(event.error_code == "TASK_SNAPSHOT_IDENTITY_REJECTED" for event in events)
    assert any("foreign task or attempt identity" in error for error in errors)


def test_committed_state_repairs_mutated_inputs_for_same_snapshot_revision():
    committed_instruction = "Update test-pkg to 1.0.0."
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": "attempt-current",
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        task_revision=1,
        selected_version="1.0.0",
        instruction=committed_instruction,
        instruction_digest=instruction_digest(committed_instruction),
        dispatch_node="update_subagent",
    )
    queue = {"task-1": task}

    events, errors = _validate_committed_state(
        queue, {snapshot.attempt_id: snapshot}, {}, {}, [], "teardown"
    )

    assert queue["task-1"].task_revision == 1
    assert queue["task-1"].selected_version == "1.0.0"
    assert queue["task-1"].instruction == committed_instruction
    assert any(event.error_code == "TASK_SNAPSHOT_REPAIRED" for event in events)
    assert not errors


# ===========================================================================
# derive_initial_strategy (now in task_utils)
# ===========================================================================


class TestDeriveInitialStrategy:
    def test_version_found_plan_yields_version_bump(self):
        g = _sca_group(fix_status=FixPlanStatus.VERSION_FOUND)
        assert derive_initial_strategy(g) == RoutingStrategy.VERSION_BUMP

    def test_workaround_found_plan_yields_code_workaround(self):
        g = _sca_group(fix_status=FixPlanStatus.WORKAROUND_FOUND)
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND

    def test_no_fix_plan_yields_code_workaround(self):
        g = _sca_group(fix_status=FixPlanStatus.NO_FIX)
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND

    def test_missing_fix_plan_yields_code_workaround(self):
        g = _sast_group()
        g.fix_plan = None
        assert derive_initial_strategy(g) == RoutingStrategy.CODE_WORKAROUND


# ===========================================================================
# supervisor_router
# ===========================================================================


class TestSupervisorRouterFunction:
    def test_valid_next_routing_steps_route_correctly(self):
        active_targets = {"active_target_task_ids": ["task-1"]}
        assert (
            supervisor_router({"next_routing_step": "update_subagent", **active_targets})
            == "update_subagent"
        )
        assert (
            supervisor_router({"next_routing_step": "workaround_subagent", **active_targets})
            == "workaround_subagent"
        )
        assert (
            supervisor_router({"next_routing_step": "qa_critic", **active_targets}) == "qa_critic"
        )
        assert supervisor_router({"next_routing_step": "teardown"}) == "teardown"
        assert supervisor_router({"next_routing_step": "supervisor"}) == "supervisor"
        assert supervisor_router({"next_routing_step": "final_full_scan"}) == "final_full_scan"

    @pytest.mark.parametrize(
        "step",
        ["update_subagent", "workaround_subagent", "qa_critic"],
    )
    def test_worker_routes_without_active_targets_fail_closed(self, step):
        assert (
            supervisor_router(
                {
                    "next_routing_step": step,
                    "task_queue": {"task-1": _make_task("task-1", "g1")},
                }
            )
            == "teardown"
        )

    def test_missing_or_unknown_step_defaults_to_teardown(self):
        assert supervisor_router({}) == "teardown"
        assert supervisor_router({"next_routing_step": "invalid"}) == "teardown"


@pytest.mark.parametrize(
    ("status", "has_portfolio_plan"),
    [
        ("workspace_ready", False),
        ("triage_completed", True),
    ],
)
def test_supervisor_calls_portfolio_tool_for_initial_and_post_triage_entries(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    has_portfolio_plan: bool,
):
    state = _base_state(
        [_sca_group()],
        status=status,
        portfolio_plan=_portfolio_stub() if has_portfolio_plan else None,
        portfolio_dirty=False,
    )
    portfolio_tool = MagicMock(
        return_value={
            "status": "portfolio_ready",
            "next_routing_step": "teardown",
            "errors": [],
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.run_portfolio_plan_tool",
        portfolio_tool,
    )

    result = run_supervisor_node(state)

    portfolio_tool.assert_called_once_with(state)
    assert result["next_routing_step"] == "supervisor"


def test_supervisor_overlays_replan_patch_on_portfolio_tool_input(
    monkeypatch: pytest.MonkeyPatch,
):
    task = _make_task("task-1", "g1")
    request = PortfolioReplanRequest(
        reason="UNFIXABLE_REPLAN",
        source_portfolio_plan_id="prior-plan",
    )
    reconciled_queue = {"task-1": task}
    supervisor_patch = {
        "status": "supervisor_routed",
        "next_routing_step": "portfolio",
        "decision_code": DecisionCode.PORTFOLIO_PLAN_REQUIRED,
        "supervisor_instructions": "Rebuild the portfolio before dispatch.",
        "task_queue": reconciled_queue,
        "portfolio_replan_request": request,
        "portfolio_dirty": True,
        "errors": ["supervisor-error"],
    }
    state = _base_state(
        [_sca_group()],
        errors=["prior-error"],
        task_queue={"old-task": _make_task("old-task", "g1")},
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._run_supervisor_decision_node",
        lambda _state: supervisor_patch,
    )
    portfolio_plan = _portfolio_stub()
    portfolio_tool = MagicMock(
        return_value={
            "status": "portfolio_ready",
            "next_routing_step": "teardown",
            "portfolio_plan": portfolio_plan,
            "errors": ["portfolio-error"],
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.run_portfolio_plan_tool",
        portfolio_tool,
    )

    result = run_supervisor_node(state)

    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["task_queue"] is reconciled_queue
    assert portfolio_input["portfolio_replan_request"] is request
    assert portfolio_input["errors"] == ["prior-error", "supervisor-error"]
    assert result["decision_code"] == DecisionCode.PORTFOLIO_PLAN_REQUIRED
    assert result["supervisor_instructions"] == supervisor_patch["supervisor_instructions"]
    assert result["portfolio_plan"] is portfolio_plan
    assert result["errors"] == ["supervisor-error", "portfolio-error"]
    assert result["next_routing_step"] == "supervisor"


# ===========================================================================
# run_supervisor_node â€” task initialization
# ===========================================================================


class TestRunSupervisorNodeNormalization:
    def test_creates_tasks_for_all_groups(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])
        result = run_supervisor_node(state)
        tasks = result["task_queue"]
        assert len(tasks) == 1
        task = next(iter(tasks.values()))
        assert task.parent_group_id == "g1"
        assert task.strategy == RoutingStrategy.VERSION_BUMP
        assert task.status == TaskStatus.PENDING

    def test_does_not_create_duplicate_tasks_for_existing_groups(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        existing_task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP)
        state = _base_state([g1], task_queue={"task-1": existing_task})
        result = run_supervisor_node(state)
        # Should still have exactly 1 task
        assert len(result["task_queue"]) == 1

    def test_backfills_missing_policy_on_uncommitted_existing_task(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        existing_task = _make_task("task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP)
        state = _base_state([g1], task_queue={"task-1": existing_task})

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].qa_policy == QAPolicy.VERSION_BUMP

    def test_initial_dispatch_commits_policy_to_task_and_attempt_snapshot(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        result = run_supervisor_node(_base_state([g1]))

        task = result["task_queue"]["task-1"]
        assert task.qa_policy == QAPolicy.VERSION_BUMP
        assert task.current_attempt_id is not None

        snapshot = result["attempt_snapshots_by_id"][task.current_attempt_id]
        assert snapshot.qa_policy == QAPolicy.VERSION_BUMP

    def test_candidate_less_update_is_rejected_before_snapshot_creation(self, monkeypatch):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        monkeypatch.setattr(
            "remediation_engine.orchestration.supervisor_planner.fetch_registry_candidates",
            lambda *_args, **_kwargs: [],
        )

        result = run_supervisor_node(_base_state([g1]))

        assert result["active_target_task_ids"] == []
        assert all(
            snapshot.dispatch_node != "update_subagent"
            for snapshot in result.get("attempt_snapshots_by_id", {}).values()
        )
        assert any(
            event.error_code == "UPDATE_DISPATCH_WITHOUT_CANDIDATE"
            for event in result["consistency_events"]
        )


# ===========================================================================
# run_supervisor_node â€” routing decisions
# ===========================================================================


class TestRunSupervisorNodeVersionBump:
    def test_python_requirements_atomic_cluster_builds_pypi_action(self):
        task_specs = [
            ("task-1", "group-certifi", "certifi", "2024.7.4"),
            ("task-2", "group-sqlparse", "sqlparse", "0.4.4"),
        ]
        groups = {}
        tasks = {}
        for task_id, group_id, package_name, selected_version in task_specs:
            issue = _issue().model_copy(
                update={
                    "source": IssueSource.ODC,
                    "issue_type": IssueType.SCA,
                    "ecosystem": "pypi",
                    "package_name": package_name,
                }
            )
            group = _sca_group(group_id).model_copy(
                update={
                    "vulnerable_component": package_name,
                    "file_path": "requirements.txt",
                    "file_paths": ["requirements.txt"],
                    "sources": [IssueSource.ODC],
                    "issues": [issue],
                }
            )
            groups[group_id] = group
            tasks[task_id] = _make_task(task_id, group_id).model_copy(
                update={
                    "target_package_name": package_name,
                    "target_dependency_type": "requirements",
                    "selected_version": selected_version,
                }
            )

        action = _build_multi_package_action(
            "batch-python",
            ["task-1", "task-2"],
            tasks,
            groups,
        )

        assert action is not None
        assert action.cluster_id == "batch-python"
        assert [
            (
                mutation.manifest_path,
                mutation.package_name,
                mutation.target_version,
                mutation.dependency_type,
                mutation.ecosystem,
            )
            for mutation in action.package_mutations
        ] == [
            ("requirements.txt", "certifi", "2024.7.4", "requirements", "pypi"),
            ("requirements.txt", "sqlparse", "0.4.4", "requirements", "pypi"),
        ]

    def test_first_active_update_task_is_the_only_registry_target(self, monkeypatch):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND).model_copy(
            update={"vulnerable_component": "other-pkg"}
        )
        calls: list[tuple[str, str, set[str]]] = []

        def candidates(package_name, security_floor, attempted_versions=None):
            calls.append((package_name, security_floor, set(attempted_versions or set())))
            return [
                RegistryCandidate(
                    version="2.0.0",
                    semver_key=(2, 0, 0),
                    security_floor_met=True,
                    is_stable=True,
                    same_major=True,
                    already_attempted=False,
                )
            ]

        monkeypatch.setattr(
            "remediation_engine.orchestration.supervisor_planner.fetch_registry_candidates",
            candidates,
        )
        monkeypatch.setattr(
            "remediation_engine.orchestration.supervisor_node.get_runtime_settings",
            lambda: AppSettings(openai_api_key=""),
        )

        result = run_supervisor_node(_base_state([g1, g2]))

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert calls == [("test-pkg", "1.2.3", set())]
        assert result["task_queue"]["task-2"].selected_version is None

    def test_version_bump_tasks_dispatch_one_at_a_time(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1, g2])

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_deterministic_routing_caps_update_dispatch_at_one(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        state = _base_state(groups)

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_terminal_tasks_route_through_optional_final_full_scan(self):
        group = _sast_group("g-final-scan")
        task = _make_task(
            "task-final-scan",
            group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.QA_PASSED,
        )
        base_state = _base_state(
            [group],
            task_queue={task.task_id: task},
            workspace_volume="workspace-volume",
            baseline_scan_complete=True,
            package_inventory_baseline_complete=True,
        )

        enabled = run_supervisor_node({**base_state, "final_full_scan_enabled": True})
        assert enabled["next_routing_step"] == "final_full_scan"
        assert enabled["decision_code"].value == "FINAL_FULL_SCAN_REQUIRED"
        assert supervisor_router({**base_state, **enabled}) == "final_full_scan"

        disabled = run_supervisor_node({**base_state, "final_full_scan_enabled": False})
        assert disabled["next_routing_step"] == "teardown"

    def test_supervisor_preserves_full_scan_findings_for_triage(self):
        group = _sast_group("g-full-scan")
        task = _make_task(
            "task-full-scan",
            group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.QA_PASSED,
        )
        full_scan = {
            "completed": True,
            "authoritative": True,
            "coverage_scope": "full_workspace",
            "coverage_complete": True,
            "found_identifiers": ["CVE-2026-70001"],
            "remaining_target_identifiers": [],
            "new_identifiers": ["CVE-2026-70001"],
            "found_issues": [],
            "status": "detected",
            "triage_required": True,
            "error": None,
        }

        result = run_supervisor_node(
            _base_state(
                [group],
                task_queue={task.task_id: task},
                workspace_volume="workspace-volume",
                final_full_scan_enabled=True,
                final_full_scan_result=full_scan,
                status="final_scan_completed",
                triage_required=True,
                post_qa_triage_enabled=True,
            )
        )

        assert result["next_routing_step"] == "triage"
        assert result["final_full_scan_result"] == full_scan
        assert result["new_vulnerability_identifiers"] == ["CVE-2026-70001"]
        assert result["new_vulnerability_status"] == "detected"
        assert result["triage_required"] is True

    def test_deterministic_retry_routing_dispatches_one_task(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            )
            for i in range(3)
        }
        decision = _deterministic_routing(
            {task.task_id: task for task in tasks.values()},
            {group.group_id: group for group in groups},
            {},
            {},
        )

        assert decision.next_node == "update_subagent"
        assert decision.target_task_ids == ["task-1"]

    def test_active_qa_retry_precedes_unrelated_portfolio_task(self):
        failed_group = _sca_group("g1")
        ready_group = _sca_group("g2")
        failed_task = _make_task(
            "task-1",
            failed_group.group_id,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=1,
        )
        ready_task = _make_task("task-2", ready_group.group_id)

        portfolio_plan = _portfolio_stub()
        cluster_id = "cluster-task-2"
        portfolio_plan.task_ids = ["task-2"]
        portfolio_plan.clusters = [
            SimpleNamespace(
                cluster_id=cluster_id,
                task_ids=["task-2"],
                dependencies=[],
                dispatchable=True,
            )
        ]
        portfolio_plan.cluster_order = [cluster_id]
        portfolio_plan.task_order = ["task-2"]
        portfolio_plan.task_to_cluster = {"task-2": cluster_id}

        decision = _deterministic_routing(
            {"task-1": failed_task, "task-2": ready_task},
            {failed_group.group_id: failed_group, ready_group.group_id: ready_group},
            {
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="Repair the compatibility regression.",
                )
            },
            {},
            active_target_task_ids=["task-1"],
            current_status="qa_completed",
            portfolio_plan=portfolio_plan,
        )

        assert decision.decision_code == DecisionCode.RETRY_VERSION_BUMP
        assert decision.target_task_ids == ["task-1"]
        assert "tactical/retry planning" in decision.instructions

    def test_active_qa_retry_does_not_split_atomic_portfolio_cluster(self):
        groups = [_sca_group(f"g{i}") for i in range(1, 4)]
        tasks = {
            "task-1": _make_task(
                "task-1",
                "g1",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            ),
            "task-2": _make_task(
                "task-2",
                "g2",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            ),
            "task-3": _make_task("task-3", "g3"),
        }
        portfolio_plan = _portfolio_stub()
        unrelated_cluster_id = "unrelated-cluster"
        cluster_id = "atomic-cluster"
        portfolio_plan.task_ids = ["task-3", "task-1", "task-2"]
        portfolio_plan.clusters = [
            SimpleNamespace(
                cluster_id=unrelated_cluster_id,
                task_ids=["task-3"],
                dependencies=[],
                dispatchable=True,
            ),
            SimpleNamespace(
                cluster_id=cluster_id,
                task_ids=["task-1", "task-2"],
                dependencies=[],
                dispatchable=True,
            ),
        ]
        portfolio_plan.cluster_order = [unrelated_cluster_id, cluster_id]
        portfolio_plan.task_order = ["task-3", "task-1", "task-2"]
        portfolio_plan.task_to_cluster = {
            "task-1": cluster_id,
            "task-2": cluster_id,
            "task-3": unrelated_cluster_id,
        }

        decision = _deterministic_routing(
            tasks,
            {group.group_id: group for group in groups},
            {
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="Repair the compatibility regression.",
                )
            },
            {},
            active_target_task_ids=["task-1"],
            current_status="qa_completed",
            portfolio_plan=portfolio_plan,
        )

        assert decision.decision_code == DecisionCode.ATOMIC_CLUSTER_DISPATCH
        assert decision.target_task_ids == ["task-1", "task-2"]

    def test_repeated_routing_advances_to_the_next_task_after_completion(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {f"task-{i + 1}": _make_task(f"task-{i + 1}", f"g{i}") for i in range(3)}
        task_queue = {task.task_id: task for task in tasks.values()}
        groups_by_id = {group.group_id: group for group in groups}

        first = _deterministic_routing(task_queue, groups_by_id, {}, {})
        assert first.target_task_ids == ["task-1"]

        task_queue["task-1"] = task_queue["task-1"].model_copy(
            update={"status": TaskStatus.QA_PASSED}
        )
        second = _deterministic_routing(task_queue, groups_by_id, {}, {})
        assert second.target_task_ids == ["task-2"]


class TestRunSupervisorNodeWorkaround:
    def test_code_workaround_deterministic_routes_one_task(self):
        g1 = _sast_group("g1")
        g2 = _sast_group("g2")
        state = _base_state([g1, g2])

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "workaround_subagent"
        assert len(result["active_target_task_ids"]) == 1


class TestRunSupervisorNodeToQA:
    def test_optimistically_fixed_task_routes_to_qa(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert "task-1" in result["active_target_task_ids"]

    def test_active_cluster_action_survives_qa_handoff(self):
        cluster_id = "cluster-active-qa"
        batch_id = "batch-active-qa"
        task_specs = [("task-1", "g1", "pkg-a"), ("task-2", "g2", "pkg-b")]
        groups = [
            _sca_group(group_id).model_copy(update={"vulnerable_component": package_name})
            for _, group_id, package_name in task_specs
        ]
        action = MultiPackageAction(
            cluster_id=cluster_id,
            dispatch_batch_id=batch_id,
            selected_strategy="version_bump",
            package_mutations=[
                PackageMutation(
                    task_id=task_id,
                    package_name=package_name,
                    target_version="1.2.3",
                    dependency_type="dependencies",
                )
                for task_id, _, package_name in task_specs
            ],
            rationale="test atomic QA handoff",
        )
        action_digest = instruction_digest(action.model_dump_json())
        tasks = {}
        snapshots = {}
        worker_results = {}
        for task_id, group_id, package_name in task_specs:
            attempt_id = f"attempt-{task_id}"
            instruction = f"Update {package_name} to 1.2.3."
            task = _make_task(
                task_id,
                group_id,
                status=TaskStatus.OPTIMISTICALLY_FIXED,
            ).model_copy(
                update={
                    "task_revision": 1,
                    "current_attempt_id": attempt_id,
                    "qa_policy": QAPolicy.VERSION_BUMP,
                    "selected_version": "1.2.3",
                    "allowed_target_versions": ["1.2.3"],
                    "target_package_name": package_name,
                    "target_dependency_type": "dependencies",
                    "instruction": instruction,
                }
            )
            snapshot = TaskAttemptSnapshot(
                attempt_id=attempt_id,
                task_id=task_id,
                task_revision=task.task_revision,
                cluster_id=cluster_id,
                dispatch_batch_id=batch_id,
                action_digest=action_digest,
                qa_policy=task.qa_policy,
                selected_version=task.selected_version,
                target_package_name=package_name,
                target_dependency_type="dependencies",
                instruction=instruction,
                instruction_digest=instruction_digest(instruction),
                dispatch_node="update_subagent",
            )
            tasks[task_id] = task
            snapshots[attempt_id] = snapshot
            worker_results[attempt_id] = WorkerAttemptResult(
                attempt_id=attempt_id,
                task_id=task_id,
                task_revision=task.task_revision,
                cluster_id=cluster_id,
                dispatch_batch_id=batch_id,
                action_digest=action_digest,
                status=AgentActionStatus.SUCCESS,
                execution_diagnostics=WorkerExecutionDiagnostics(validation_passed=True),
                instruction_digest=snapshot.instruction_digest,
            )

        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=[task_id for task_id, _, _ in task_specs],
            active_cluster_id=cluster_id,
            active_dispatch_batch_id=batch_id,
            active_multi_package_action=action,
            attempt_snapshots_by_id=snapshots,
            worker_results_by_attempt=worker_results,
            processed_worker_attempt_ids=list(worker_results),
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1", "task-2"]
        assert result["active_cluster_id"] == cluster_id
        assert result["active_dispatch_batch_id"] == batch_id
        assert result["active_multi_package_action"] == action

    def test_all_optimistically_fixed_tasks_route_to_qa_within_limit(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            # No active_target_task_ids â€” all tasks are already optimistically fixed
            active_target_task_ids=[],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1", "task-2"]

    def test_current_task_routes_to_qa_before_more_updates(self):
        # 12 groups: first 10 are optimistically fixed, 2 are pending.
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(12)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.OPTIMISTICALLY_FIXED if i < 10 else TaskStatus.PENDING,
            )
            for i in range(12)
        }
        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=[f"task-{i + 1}" for i in range(10)],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == [f"task-{index}" for index in range(1, 11)]

    def test_mixed_update_results_route_successful_subset_to_qa(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(5)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                status=TaskStatus.NEEDS_RETRY,
                retry_count=0,
            )
            for i in range(5)
        }
        summaries = [
            AgentActionSummary(
                task_id="task-1", status=AgentActionStatus.SUCCESS, summary="updated jsonwebtoken"
            ),
            AgentActionSummary(
                task_id="task-2", status=AgentActionStatus.SUCCESS, summary="updated express-jwt"
            ),
            AgentActionSummary(
                task_id="task-3",
                status=AgentActionStatus.SUCCESS,
                summary="updated @tootallnate/once",
            ),
            AgentActionSummary(
                task_id="task-4",
                status=AgentActionStatus.SURRENDER,
                summary="ws reverted due to dependency conflict",
            ),
            AgentActionSummary(
                task_id="task-5",
                status=AgentActionStatus.SURRENDER,
                summary="elliptic update path exhausted",
            ),
        ]
        state = _base_state(
            groups,
            task_queue=tasks,
            action_summaries=summaries,
            active_target_task_ids=[f"task-{i + 1}" for i in range(5)],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1", "task-2", "task-3"]
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-2"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-3"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-4"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-5"].status == TaskStatus.NEEDS_RETRY

    def test_terminal_tasks_in_active_batch_are_omitted_from_qa_targets(self):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            "task-1": _make_task("task-1", "g0", status=TaskStatus.OPTIMISTICALLY_FIXED),
            "task-2": _make_task("task-2", "g1", status=TaskStatus.QA_PASSED),
            "task-3": _make_task("task-3", "g2", status=TaskStatus.UNFIXABLE),
        }
        state = _base_state(
            groups,
            task_queue=tasks,
            active_target_task_ids=["task-1", "task-2", "task-3"],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]

    @patch("langchain_openai.ChatOpenAI")
    def test_optimistically_fixed_task_routes_to_qa_before_retry_planning(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.NEEDS_RETRY, retry_count=1)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            active_target_task_ids=["task-1", "task-2"],
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]
        mock_chat.assert_not_called()


class TestRunSupervisorNodeToTeardown:
    def test_qa_passed_routes_to_teardown(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )
        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"

    def test_unfixable_routes_to_teardown(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE)
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )
        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "teardown"


# ===========================================================================
# run_supervisor_node â€” QA evaluation updates
# ===========================================================================


class TestRunSupervisorNodeQAUpdates:
    def test_qa_failed_retry_uses_deterministic_planning(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.NEEDS_RETRY)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="retry with better update evidence",
                )
            },
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["task_queue"]["task-1"].selected_version == "2.0.0"
        assert result["task_queue"]["task-1"].strategy_stage == SCARemediationStage.NPM_LATEST
        assert "exact version 2.0.0" in result["task_queue"]["task-1"].instruction

    def test_qa_completed_passed_marks_task_qa_passed(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_no_fix_package_removal_pass_reaches_qa_passed(self):
        group = _sca_group("g1", FixPlanStatus.NO_FIX)
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
        ).model_copy(
            update={
                "qa_policy": QAPolicy.NO_FIX_PACKAGE_REMOVAL,
                "no_fix_stage": "package_removal",
                "instruction": "Remove test-pkg from the dependency manifests.",
            }
        )
        state = _base_state(
            [group],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_qa_terminalization_clears_attempt_projection_without_repair_events(self):
        g1 = _sca_group("g1")
        instruction = "Pin test-pkg through the package manager override."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa-pass",
            task_id="task-1",
            task_revision=1,
            strategy_stage=SCARemediationStage.PACKAGE_OVERRIDE,
            selected_version="1.2.3",
            target_package_name="test-pkg",
            target_dependency_type="overrides",
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="update_subagent",
        )
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": snapshot.attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "target_dependency_type": snapshot.target_dependency_type,
                "instruction": instruction,
            }
        )
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            qa_results_by_attempt={
                snapshot.attempt_id: QAAttemptResult(
                    attempt_id=snapshot.attempt_id,
                    task_id="task-1",
                    task_revision=1,
                    qa_policy=QAPolicy.VERSION_BUMP,
                    qa_policy_source="attempt_snapshot",
                    evaluation=QAEvaluation(task_id="task-1", passed=True),
                )
            },
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.PACKAGE_OVERRIDE,
                    selected_version="1.2.3",
                    target_package_name="test-pkg",
                    target_dependency_type="overrides",
                    attempted_versions=["1.2.3"],
                    attempted_versions_by_target={"test-pkg": ["1.2.3"]},
                    used_overrides=True,
                )
            },
        )

        result = run_supervisor_node(state)

        committed = result["task_queue"]["task-1"]
        assert committed.status == TaskStatus.QA_PASSED
        assert committed.current_attempt_id is None
        assert committed.selected_version is None
        diagnostics = result["retry_diagnostics_by_task"]["task-1"]
        assert diagnostics.selected_version is None
        assert diagnostics.attempted_versions_by_target == {"test-pkg": ["1.2.3"]}
        assert diagnostics.used_overrides is True
        assert not any(
            event.error_code
            in {"TERMINAL_TASK_FIELDS_NORMALIZED", "DIAGNOSTICS_PROJECTION_REPAIRED"}
            for event in result["consistency_events"]
        )

    def test_inconclusive_qa_requeues_same_attempt_without_worker_retry(self):
        """Missing deterministic evidence reruns QA without consuming a retry."""
        group = _sca_group("g1")
        instruction = "Pin test-pkg through the package manager."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa-inconclusive",
            task_id="task-1",
            task_revision=1,
            strategy_stage=SCARemediationStage.NPM_LATEST,
            selected_version="1.2.3",
            target_package_name="test-pkg",
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="update_subagent",
            qa_policy=QAPolicy.VERSION_BUMP,
        )
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": snapshot.attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "instruction": instruction,
                "qa_policy": QAPolicy.VERSION_BUMP,
            }
        )
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="Dependency evidence was unavailable.",
            evidence_inconclusive=True,
        )
        state = _base_state(
            [group],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            qa_results_by_attempt={
                snapshot.attempt_id: QAAttemptResult(
                    attempt_id=snapshot.attempt_id,
                    task_id="task-1",
                    task_revision=1,
                    qa_policy=QAPolicy.VERSION_BUMP,
                    qa_policy_source="attempt_snapshot",
                    evaluation=evaluation,
                )
            },
        )

        result = run_supervisor_node(state)

        requeued = result["task_queue"]["task-1"]
        assert requeued.status == TaskStatus.OPTIMISTICALLY_FIXED
        assert requeued.current_attempt_id == snapshot.attempt_id
        assert requeued.retry_count == 0
        assert result["processed_qa_attempt_ids"] == []
        assert result["next_routing_step"] == "qa_critic"

    def test_contract_error_qa_failed_requeues_same_attempt_without_worker_retry(self):
        """QA contract failures rerun the current attempt, including qa_failed results."""
        group = _sca_group("g1")
        instruction = "Pin test-pkg through the package manager."
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-qa-contract",
            task_id="task-1",
            task_revision=1,
            strategy_stage=SCARemediationStage.NPM_LATEST,
            selected_version="1.2.3",
            target_package_name="test-pkg",
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="update_subagent",
            qa_policy=QAPolicy.VERSION_BUMP,
        )
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": snapshot.attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "instruction": instruction,
                "qa_policy": QAPolicy.VERSION_BUMP,
            }
        )
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            contract_error=True,
            contract_error_reason="Structured evaluator output was missing.",
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="QA contract must be corrected.",
        )
        state = _base_state(
            [group],
            status="qa_failed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            qa_results_by_attempt={
                snapshot.attempt_id: QAAttemptResult(
                    attempt_id=snapshot.attempt_id,
                    task_id="task-1",
                    task_revision=1,
                    qa_policy=QAPolicy.VERSION_BUMP,
                    qa_policy_source="attempt_snapshot",
                    evaluation=evaluation,
                )
            },
        )

        result = run_supervisor_node(state)

        requeued = result["task_queue"]["task-1"]
        assert requeued.status == TaskStatus.OPTIMISTICALLY_FIXED
        assert requeued.current_attempt_id == snapshot.attempt_id
        assert requeued.retry_count == 0
        assert result["processed_qa_attempt_ids"] == []
        assert result["next_routing_step"] == "qa_critic"

    def test_qa_completed_passed_adds_constraint(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["constraints_ledger"] == ["test-pkg: keep resolved version at 1.2.3"]

    def test_qa_completed_passed_workaround_adds_constraint(self):
        g1 = _sast_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
        )
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["constraints_ledger"] == ["test-func: preserve validated security workaround"]

    def test_qa_completed_passed_does_not_duplicate_existing_constraint(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
            constraints_ledger=["test-pkg: keep resolved version at 1.2.3"],
        )
        result = run_supervisor_node(state)
        assert result["constraints_ledger"] == []

    def test_qa_completed_failed_marks_task_needs_retry(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="try again",
                )
            },
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-1"].retry_count == 1

    def test_not_qa_completed_does_not_update_statuses_from_qa_evals(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="supervisor_entered",
            task_queue={"task-1": task},
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )
        result = run_supervisor_node(state)
        # Status should NOT become QA_PASSED because status != "qa_completed"
        # It will route to qa_critic since task is OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED

    def test_failed_task_keyed_qa_eval_replans_active_child_task(self):
        g1 = _sca_group("g1")
        parent = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        child = _make_task("task-2", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={
                "task-2": child,
                "task-1": parent,
            },
            active_target_task_ids=["task-2"],
            qa_evaluations={
                "task-2": QAEvaluation(
                    task_id="task-2",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="retry the active child task",
                )
            },
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-2"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-2"].retry_count == 1
        assert result["task_queue"]["task-2"].selected_version == "2.0.0"
        assert result["task_queue"]["task-2"].strategy_stage == SCARemediationStage.NPM_LATEST
        assert "exact version 2.0.0" in result["task_queue"]["task-2"].instruction
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]

    def test_retry_update_without_revised_instruction_falls_back_to_high_level_retry(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.NEEDS_RETRY,
        )
        task.instruction = 'Update "test-pkg" in package.json to version "1.2.3".'
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY
        assert result["task_queue"]["task-1"].instruction.startswith(
            "Apply the supervisor-selected test-pkg dependency version"
        )
        assert "exact version 2.0.0" in result["task_queue"]["task-1"].instruction
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]

    def test_missing_retry_instruction_replans_before_dispatch(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.NEEDS_RETRY,
        ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_SAME_MAJOR})
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        result = run_supervisor_node(state)

        committed = result["task_queue"]["task-1"]
        assert committed.selected_version == "2.0.0"
        assert committed.strategy_stage == SCARemediationStage.NPM_LATEST
        assert "exact version 2.0.0" in committed.instruction
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert any("replanning deterministically" in error for error in result["errors"])

    def test_pending_update_task_can_seed_instruction_from_revised_instructions(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        task.instruction = ""
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        committed_instruction = result["task_queue"]["task-1"].instruction
        assert committed_instruction.startswith("Apply the supervisor-selected test-pkg")
        assert "test-pkg" in committed_instruction
        assert "2.0.0" in committed_instruction


def test_invalid_latest_repair_pivots_instead_of_reusing_stale_candidate():
    """An exhausted latest stage must not create an update attempt without a version."""
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(
        update={
            "task_revision": 7,
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.17.7",
            "instruction": "Update test-pkg to exact version 2.17.7.",
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id="task-1",
        strategy_stage=SCARemediationStage.NPM_LATEST,
        attempted_versions=["1.2.3", "2.17.7"],
        latest_version_seen="2.17.7",
    )
    invalid_plan = SupervisorRetryPlan(
        task_id="task-1",
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.17.6",
        attempted_versions=["1.2.3", "2.17.7"],
        candidate_versions_considered=["2.17.6", "0.1.0"],
        latest_version_seen="2.17.7",
        action="retry_update",
        exact_instruction="Update test-pkg to exact version 2.17.6.",
    )

    repaired_diagnostics, repaired_plans = _repair_invalid_planner_plans(
        {task.task_id: invalid_plan},
        {task.task_id: diagnostics},
        {task.task_id: task},
        {group.group_id: group},
        violations=["task task-1: npm_latest selected 2.17.6, but registry latest is 2.17.7"],
    )

    assert repaired_plans[task.task_id].action == "pivot_workaround"
    assert repaired_plans[task.task_id].selected_version is None
    assert repaired_diagnostics[task.task_id].exhausted_update_path is True


def test_exhausted_deterministic_guardrail_routes_workaround_child():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        qa_evaluations={
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Repair the express-jwt compatibility regression.",
                failure_evidence=QAFailureEvidence(
                    attempt_id="update-attempt-1",
                    task_revision=1,
                    exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
                    source_locations=["/workspace/lib/insecurity.ts:54:35"],
                ),
            )
        },
        qa_results_by_attempt={
            "update-attempt-1": QAAttemptResult(
                attempt_id="update-attempt-1",
                task_id="task-1",
                task_revision=1,
                qa_policy=QAPolicy.VERSION_BUMP,
                qa_policy_source="attempt_snapshot",
                evaluation=QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="Repair the express-jwt compatibility regression.",
                    failure_evidence=QAFailureEvidence(
                        attempt_id="update-attempt-1",
                        task_revision=1,
                        exact_diagnostics=["(0, import_express_jwt.default) is not a function"],
                        source_locations=["/workspace/lib/insecurity.ts:54:35"],
                    ),
                ),
            )
        },
        retry_diagnostics_by_task={
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_LATEST,
                selected_version="1.2.3",
                attempted_versions=["1.2.3"],
                latest_version_seen="1.2.3",
                exhausted_update_path=True,
            )
        },
        status="supervisor_entered",
    )

    result = run_supervisor_node(state)

    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert result["next_routing_step"] == "workaround_subagent"
    assert len(result["active_target_task_ids"]) == 1
    child_id = result["active_target_task_ids"][0]
    assert result["task_queue"][child_id].strategy == RoutingStrategy.CODE_WORKAROUND
    assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
    child_attempt_id = result["task_queue"][child_id].current_attempt_id
    child_snapshot = result["attempt_snapshots_by_id"][child_attempt_id]
    assert child_snapshot.workaround_context is not None
    assert child_snapshot.workaround_context.phase == WorkaroundPhase.QA_REGRESSION_REPAIR
    assert child_snapshot.workaround_context.qa_evidence is not None
    assert child_snapshot.workaround_context.qa_evidence.source_locations == [
        "/workspace/lib/insecurity.ts:54:35"
    ]


def test_deterministic_latest_selection_skips_attempted_versions():
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        status="supervisor_entered",
        qa_evaluations={
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="the attempted version still fails QA",
            )
        },
        retry_diagnostics_by_task={
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_LATEST,
                attempted_versions=["6.0.0", "6.1.2"],
                latest_version_seen="8.5.1",
            )
        },
    )

    result = run_supervisor_node(state)

    # Once the latest stage has no eligible candidate, the deterministic
    # fallback pivots to the existing workaround lifecycle instead of
    # dispatching another update attempt.
    assert result["retry_diagnostics_by_task"]["task-1"].selected_version is None
    assert result["retry_diagnostics_by_task"]["task-1"].strategy_stage == (
        SCARemediationStage.NPM_LATEST
    )
    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert result["task_queue"]["task-1"].selected_version is None
    assert result["next_routing_step"] == "workaround_subagent"
    child_id = result["active_target_task_ids"][0]
    assert result["task_queue"][child_id].strategy == RoutingStrategy.CODE_WORKAROUND


def test_empty_same_major_stage_advances_or_pivots_without_unversioned_dispatch(monkeypatch):
    group = _sca_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=1,
    ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_SAME_MAJOR})
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        retry_diagnostics_by_task={
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
                attempted_versions=["2.17.7"],
                latest_version_seen="2.17.7",
            )
        },
    )

    result = run_supervisor_node(state)

    committed = result["task_queue"]["task-1"]
    assert committed.strategy_stage == SCARemediationStage.NPM_LATEST
    assert committed.selected_version is None
    assert committed.exhausted_update_path is True
    assert result["next_routing_step"] == "workaround_subagent"
    child_id = result["active_target_task_ids"][0]
    assert result["task_queue"][child_id].strategy == RoutingStrategy.CODE_WORKAROUND
    assert not any("rejected update_subagent retry dispatch" in error for error in result["errors"])


def test_stale_worker_result_is_ignored_when_new_attempt_is_committed():
    group = _sca_group("g1")
    old_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-old",
        task_id="task-1",
        state_revision=1,
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        selected_version="1.0.0",
        instruction="Update test-pkg to 1.0.0.",
        instruction_digest=instruction_digest("Update test-pkg to 1.0.0."),
        dispatch_node="update_subagent",
    )
    new_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-new",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction="Update test-pkg to 2.0.0.",
        instruction_digest=instruction_digest("Update test-pkg to 2.0.0."),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": "attempt-new",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    stale_result = WorkerAttemptResult(
        attempt_id="attempt-old",
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["1.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.0.0"],
            executed_versions=["1.0.0"],
            validation_passed=True,
        ),
        instruction_digest=old_snapshot.instruction_digest,
    )
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={
            old_snapshot.attempt_id: old_snapshot,
            new_snapshot.attempt_id: new_snapshot,
        },
        worker_results_by_attempt={old_snapshot.attempt_id: stale_result},
    )

    result = run_supervisor_node(state)

    committed = result["task_queue"]["task-1"]
    assert committed.current_attempt_id == "attempt-new"
    assert committed.selected_version == "2.0.0"
    assert committed.status == TaskStatus.OPTIMISTICALLY_FIXED
    assert any(event.error_code == "STALE_WORKER_RESULT" for event in result["consistency_events"])


def test_worker_result_under_current_key_must_carry_current_attempt_id():
    group = _sca_group("g1")
    instruction = "Update test-pkg to 2.0.0."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "instruction": instruction,
        }
    )
    foreign_result = WorkerAttemptResult(
        attempt_id="attempt-foreign",
        task_id="task-1",
        task_revision=2,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["2.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["2.0.0"],
            executed_versions=["2.0.0"],
            validation_passed=True,
        ),
        instruction_digest=snapshot.instruction_digest,
    )

    state = _base_state(
        [group],
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        # The map key is current, but the serialized envelope identity is not.
        worker_results_by_attempt={snapshot.attempt_id: foreign_result},
    )
    result = run_supervisor_node(state)

    assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
    assert result.get("retry_diagnostics_by_task", {}).get("task-1") is None
    assert snapshot.attempt_id not in result["processed_worker_attempt_ids"]
    assert any(
        event.error_code == "WORKER_ATTEMPT_MISMATCH"
        and event.expected_attempt_id == snapshot.attempt_id
        and event.received_attempt_id == "attempt-foreign"
        for event in result["consistency_events"]
    )

    valid_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id="task-1",
        task_revision=snapshot.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["2.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["2.0.0"],
            executed_versions=["2.0.0"],
            validation_passed=True,
        ),
        instruction_digest=snapshot.instruction_digest,
    )
    next_state = {**state, **result}
    next_state["active_target_task_ids"] = ["task-1"]
    next_state["worker_results_by_attempt"] = {snapshot.attempt_id: valid_result}

    next_result = run_supervisor_node(next_state)

    assert next_result["retry_diagnostics_by_task"]["task-1"].executed_versions == ["2.0.0"]
    assert snapshot.attempt_id in next_result["processed_worker_attempt_ids"]


def test_worker_result_persists_attempts_by_target_and_override_usage():
    group = _sca_group("g1")
    instruction = "Pin test-pkg to 1.2.3 through the package manager override."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-override",
        task_id="task-1",
        task_revision=1,
        strategy_stage=SCARemediationStage.PACKAGE_OVERRIDE,
        selected_version="1.2.3",
        target_package_name="test-pkg",
        target_dependency_type="overrides",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task("task-1", "g1").model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "target_package_name": snapshot.target_package_name,
            "target_dependency_type": snapshot.target_dependency_type,
            "instruction": instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["1.2.3"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.2.3"],
            executed_versions=["1.2.3"],
        ),
        instruction_digest=snapshot.instruction_digest,
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            worker_results_by_attempt={snapshot.attempt_id: worker_result},
        )
    )

    diagnostics = result["retry_diagnostics_by_task"]["task-1"]
    assert diagnostics.attempted_versions == ["1.2.3"]
    assert diagnostics.executed_versions == ["1.2.3"]
    assert diagnostics.attempted_versions_by_target == {"test-pkg": ["1.2.3"]}
    assert diagnostics.target_package_name == "test-pkg"
    assert diagnostics.target_dependency_type == "overrides"
    assert diagnostics.used_overrides is True
    assert result["next_routing_step"] == "qa_critic"


def test_allowlisted_alternate_update_candidate_is_reconciled_as_effective():
    """A worker may succeed with a different committed version and declaration type."""
    group = _sca_group("g1")
    instruction = "Update test-pkg using an approved candidate."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-alternate",
        task_id="task-1",
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="1.0.0",
        allowed_target_versions=["1.0.0", "1.1.0"],
        target_package_name="test-pkg",
        target_dependency_type="dependencies",
        allowed_dependency_types=["dependencies", "devDependencies"],
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "target_package_name": snapshot.target_package_name,
            "target_dependency_type": snapshot.target_dependency_type,
            "instruction": instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["1.1.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.0.0", "1.1.0"],
            executed_versions=["1.1.0"],
            effective_target_version="1.1.0",
            effective_dependency_type="devDependencies",
            manifest_transaction_attempts=2,
            validation_passed=True,
        ),
        instruction_digest=snapshot.instruction_digest,
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            worker_results_by_attempt={snapshot.attempt_id: worker_result},
        )
    )

    diagnostics = result["retry_diagnostics_by_task"][task.task_id]
    assert diagnostics.effective_target_version == "1.1.0"
    assert diagnostics.effective_dependency_type == "devDependencies"
    assert diagnostics.attempted_versions == ["1.0.0", "1.1.0"]
    assert result["task_queue"][task.task_id].status == TaskStatus.OPTIMISTICALLY_FIXED


@pytest.mark.parametrize(
    ("executed_versions", "effective_dependency_type", "error_code"),
    [
        (["9.9.9"], "dependencies", "EXECUTED_VERSION_MISMATCH"),
        (["1.0.0"], "peerDependencies", "EXECUTED_DEPENDENCY_TYPE_MISMATCH"),
    ],
)
def test_unallowlisted_effective_update_candidate_is_rejected(
    executed_versions, effective_dependency_type, error_code
):
    """Successful-looking worker output cannot escape the committed allowlists."""
    group = _sca_group("g1")
    instruction = "Update test-pkg using the committed candidate."
    snapshot = TaskAttemptSnapshot(
        attempt_id=f"attempt-invalid-{error_code}",
        task_id="task-1",
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="1.0.0",
        allowed_target_versions=["1.0.0"],
        target_package_name="test-pkg",
        target_dependency_type="dependencies",
        allowed_dependency_types=["dependencies"],
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "target_package_name": snapshot.target_package_name,
            "target_dependency_type": snapshot.target_dependency_type,
            "instruction": instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=executed_versions,
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=executed_versions,
            executed_versions=executed_versions,
            effective_target_version=executed_versions[-1],
            effective_dependency_type=effective_dependency_type,
            validation_passed=True,
        ),
        instruction_digest=snapshot.instruction_digest,
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            worker_results_by_attempt={snapshot.attempt_id: worker_result},
        )
    )

    assert any(event.error_code == error_code for event in result["consistency_events"])


@patch("langchain_openai.ChatOpenAI")
def test_successful_no_fix_package_removal_routes_to_qa_without_odc(mock_chat):
    """A removed NO_FIX package still runs QA install/tests; only ODC is skipped."""
    group = _sca_group("g1", FixPlanStatus.NO_FIX)
    instruction = "Remove test-pkg from the dependency manifests."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-no-fix-removal",
        task_id="task-1",
        task_revision=1,
        strategy_stage=SCARemediationStage.CODE_WORKAROUND,
        no_fix_stage="package_removal",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="workaround_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
    ).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "no_fix_stage": snapshot.no_fix_stage,
            "instruction": instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        execution_diagnostics=WorkerExecutionDiagnostics(
            validation_calls=1,
            validation_passed=True,
            per_gate_results={"overall_status": "PASS"},
            validated_files=[],
        ),
        instruction_digest=snapshot.instruction_digest,
    )
    mock_chat.side_effect = ImportError("No LLM needed for deterministic routing")

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            worker_results_by_attempt={snapshot.attempt_id: worker_result},
        )
    )

    committed = result["task_queue"][task.task_id]
    assert committed.status == TaskStatus.OPTIMISTICALLY_FIXED
    assert committed.current_attempt_id == snapshot.attempt_id
    assert result["next_routing_step"] == "qa_critic"
    assert result["active_target_task_ids"] == [task.task_id]


def test_failed_update_attempt_is_closed_before_retry_planner_commit(monkeypatch):
    group = _sca_group("g1")
    old_instruction = "Update test-pkg to 1.0.0."
    old_snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-failed",
        task_id="task-1",
        state_revision=1,
        task_revision=1,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        selected_version="1.0.0",
        instruction=old_instruction,
        instruction_digest=instruction_digest(old_instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": old_snapshot.attempt_id,
            "strategy_stage": old_snapshot.strategy_stage,
            "selected_version": old_snapshot.selected_version,
            "instruction": old_snapshot.instruction,
        }
    )
    worker_result = WorkerAttemptResult(
        attempt_id=old_snapshot.attempt_id,
        task_id="task-1",
        task_revision=1,
        status=AgentActionStatus.SURRENDER,
        executed_versions=["1.0.0"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            attempted_versions=["1.0.0"],
            executed_versions=["1.0.0"],
            validation_passed=False,
        ),
        instruction_digest=old_snapshot.instruction_digest,
    )
    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={old_snapshot.attempt_id: old_snapshot},
            worker_results_by_attempt={old_snapshot.attempt_id: worker_result},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
                    attempted_versions=["1.0.0"],
                    candidate_versions_considered=["2.0.0"],
                )
            },
        )
    )

    committed = result["task_queue"]["task-1"]
    assert committed.current_attempt_id != old_snapshot.attempt_id
    assert committed.selected_version == "2.0.0"
    assert committed.strategy_stage == SCARemediationStage.NPM_LATEST
    assert committed.task_revision > old_snapshot.task_revision
    assert (
        result["attempt_snapshots_by_id"][committed.current_attempt_id].selected_version == "2.0.0"
    )
    assert not any(
        event.error_code == "TASK_SNAPSHOT_REPAIRED" for event in result["consistency_events"]
    )


def test_stale_qa_result_does_not_increment_retry_count():
    group = _sca_group("g1")
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction="Update test-pkg to 2.0.0.",
        instruction_digest=instruction_digest("Update test-pkg to 2.0.0."),
        dispatch_node="update_subagent",
    )
    stale_qa = QAAttemptResult(
        attempt_id="attempt-old",
        task_id="task-1",
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="stale result",
        ),
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        retry_count=1,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": "attempt-current",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    state = _base_state(
        [group],
        status="qa_completed",
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        qa_results_by_attempt={stale_qa.attempt_id: stale_qa},
    )

    result = run_supervisor_node(state)

    assert result["task_queue"]["task-1"].retry_count == 1
    assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
    assert any(event.error_code == "STALE_QA_RESULT" for event in result["consistency_events"])


def test_qa_envelope_attempt_id_must_match_current_attempt_map_key() -> None:
    group = _sca_group("g1")
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction="Update test-pkg to 2.0.0.",
        instruction_digest=instruction_digest("Update test-pkg to 2.0.0."),
        dispatch_node="update_subagent",
    )
    occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="test-pkg",
        version="2.0.0",
        lockfile_key="node_modules/test-pkg",
        found_identifiers=["CVE-2026-10001"],
    )
    prior_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="test-pkg",
        version="1.0.0",
        lockfile_key="node_modules/test-pkg",
    )
    prior_key = '["package-lock.json","node_modules/test-pkg","test-pkg","1.0.0",0]'
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        covered_occurrences=[occurrence],
        inventory_updates={
            '["package-lock.json","node_modules/test-pkg","test-pkg","2.0.0",0]': "fingerprint"
        },
        complete=True,
    )
    mismatched_result = QAAttemptResult(
        attempt_id="attempt-stale",
        task_id="task-1",
        task_revision=2,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id="task-1",
            passed=True,
            scan_evidence=evidence,
        ),
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        retry_count=1,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": "attempt-current",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": "Update test-pkg to 2.0.0.",
        }
    )
    state = _base_state(
        [group],
        status="qa_completed",
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        qa_results_by_attempt={"attempt-current": mismatched_result},
        accepted_package_inventory_by_lockfile={prior_key: "prior-fingerprint"},
        current_scan_observations_by_occurrence={prior_key: prior_occurrence},
    )

    result = run_supervisor_node(state)

    assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
    assert any(event.error_code == "QA_ATTEMPT_MISMATCH" for event in result["consistency_events"])
    assert result["accepted_package_inventory_by_lockfile"] == {prior_key: "prior-fingerprint"}
    assert result["current_scan_observations_by_occurrence"] == {prior_key: prior_occurrence}
    assert result["latest_successful_scan_by_task"] == {}


def test_missing_qa_result_cannot_dispatch_same_attempt_forever():
    group = _sca_group("g1")
    attempt_id = "attempt-missing-qa-result"
    instruction = "Update test-pkg to 2.0.0."
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id="task-1",
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        target_package_name="test-pkg",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "target_package_name": snapshot.target_package_name,
            "instruction": instruction,
            "qa_policy": QAPolicy.VERSION_BUMP,
        }
    )
    initial = _base_state(
        [group],
        status="qa_completed",
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={attempt_id: snapshot},
        qa_results_by_attempt={},
    )

    first = run_supervisor_node(initial)
    assert first["next_routing_step"] == "qa_critic"
    assert first["qa_dispatch_counts_by_attempt"] == {attempt_id: 1}

    second = run_supervisor_node(
        {
            **initial,
            **first,
            "status": "qa_completed",
            "qa_results_by_attempt": {},
        }
    )
    assert second["next_routing_step"] == "qa_critic"
    assert second["qa_dispatch_counts_by_attempt"] == {attempt_id: 2}

    third = run_supervisor_node(
        {
            **initial,
            **second,
            "status": "qa_completed",
            "qa_results_by_attempt": {},
        }
    )

    terminal = third["task_queue"]["task-1"]
    assert terminal.status == TaskStatus.INCONCLUSIVE
    assert terminal.current_attempt_id is None
    assert third["next_routing_step"] != "qa_critic"
    assert third["qa_dispatch_counts_by_attempt"] == {attempt_id: 2}
    assert any(
        event.error_code == "QA_DISPATCH_LIMIT_REACHED" for event in third["consistency_events"]
    )


def test_inconclusive_active_qa_attempt_is_rechecked_before_other_tasks():
    group1 = _sca_group("g1")
    group2 = _sca_group("g2")
    attempt_id = "attempt-active-qa-recheck"
    instruction = "Update test-pkg to 2.0.0."
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id="task-1",
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        target_package_name="test-pkg",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    active_task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "target_package_name": snapshot.target_package_name,
            "instruction": instruction,
            "qa_policy": QAPolicy.VERSION_BUMP,
        }
    )
    unrelated_task = _make_task("task-2", "g2", status=TaskStatus.OPTIMISTICALLY_FIXED)
    state = _base_state(
        [group1, group2],
        status="supervisor_routed",
        task_queue={"task-1": active_task, "task-2": unrelated_task},
        # A failed QA preparation can clear the active-target projection and
        # return no attempt result. The dispatch ledger still identifies the
        # candidate that must be rechecked before other workspace mutations.
        active_target_task_ids=[],
        attempt_snapshots_by_id={attempt_id: snapshot},
        qa_results_by_attempt={},
        qa_dispatch_counts_by_attempt={attempt_id: 1},
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "qa_critic"
    assert result["active_target_task_ids"] == ["task-1"]
    assert result["qa_dispatch_counts_by_attempt"] == {attempt_id: 2}


def test_terminal_task_cannot_retain_retry_plan():
    group = _sca_group("g1")
    task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE).model_copy(
        update={"instruction": "Do not retry this task."}
    )
    state = _base_state(
        [group],
        task_queue={"task-1": task},
        retry_plans_by_task={
            "task-1": SupervisorRetryPlan(
                task_id="task-1",
                selected_version="8.5.1",
                exact_instruction="Retry with 8.5.1.",
            )
        },
    )

    result = run_supervisor_node(state)

    assert "task-1" not in result["retry_plans_by_task"]
    assert any(
        event.error_code == "TERMINAL_TASK_PLAN_CLEARED" for event in result["consistency_events"]
    )


def test_teardown_barrier_detaches_terminal_attempt_and_is_idempotent():
    group = _sca_group("g1")
    instruction = "Apply the workaround for test-pkg."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-terminal",
        task_id="task-1",
        state_revision=4,
        task_revision=4,
        strategy_stage=SCARemediationStage.CODE_WORKAROUND,
        selected_version=None,
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="workaround_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.UNFIXABLE,
    ).model_copy(
        update={
            "task_revision": 4,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
            "instruction": instruction,
        }
    )
    state = _base_state(
        [group],
        state_revision=4,
        task_queue={"task-1": task},
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        retry_plans_by_task={
            "task-1": SupervisorRetryPlan(
                task_id="task-1",
                strategy_stage=SCARemediationStage.CODE_WORKAROUND,
                selected_version=None,
                action="pivot_workaround",
                exact_instruction=instruction,
            )
        },
    )

    first = reconcile_phase5_state_before_teardown(state)

    committed = first["task_queue"]["task-1"]
    assert committed.status == TaskStatus.UNFIXABLE
    assert committed.current_attempt_id is None
    assert committed.selected_version is None
    assert first["retry_plans_by_task"] == {}
    assert any(
        event.error_code == "TERMINAL_TASK_FIELDS_NORMALIZED"
        for event in first["consistency_events"]
    )

    second = reconcile_phase5_state_before_teardown({**state, **first})
    assert second["task_queue"]["task-1"].current_attempt_id is None
    assert second["retry_plans_by_task"] == {}
    assert second["consistency_events"] == []


def test_terminal_pivot_parent_is_not_reseeded_from_initial_osv_plan():
    group = _sca_group("g1")
    pivot_instruction = "Implement a code workaround for the exhausted update path."
    task = _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE).model_copy(
        update={
            "task_revision": 3,
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": None,
            "exhausted_update_path": True,
            "instruction": pivot_instruction,
            "current_attempt_id": None,
        }
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_LATEST,
                    selected_version=None,
                    exhausted_update_path=True,
                )
            },
        )
    )

    committed = result["task_queue"]["task-1"]
    assert committed.selected_version is None
    assert committed.instruction == pivot_instruction
    assert committed.current_attempt_id is None
    assert "task-1" not in result["retry_plans_by_task"]


def test_pivot_detaches_previous_update_attempt_before_child_dispatch():
    group = _sca_group("g1")
    instruction = "Apply the selected dependency version 8.5.1."
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-update",
        task_id="task-1",
        state_revision=2,
        task_revision=2,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="8.5.1",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=2,
    ).model_copy(
        update={
            "task_revision": 2,
            "current_attempt_id": snapshot.attempt_id,
            "strategy_stage": snapshot.strategy_stage,
            "selected_version": snapshot.selected_version,
            "instruction": snapshot.instruction,
            "exhausted_update_path": False,
        }
    )
    result = run_supervisor_node(
        _base_state(
            [group],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=[],
            attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.NPM_LATEST,
                    selected_version="8.5.1",
                    attempted_versions=["6.0.0", "6.1.2", "8.5.1"],
                    latest_version_seen="8.5.1",
                    exhausted_update_path=True,
                )
            },
        )
    )

    parent = result["task_queue"]["task-1"]
    child_id = result["active_target_task_ids"][0]
    assert result["next_routing_step"] == "workaround_subagent"
    assert parent.status == TaskStatus.UNFIXABLE
    assert parent.current_attempt_id is None
    assert parent.selected_version is None
    assert parent.strategy_stage == SCARemediationStage.NPM_LATEST
    assert parent.exhausted_update_path is True
    assert "exact version" not in parent.instruction.lower()
    assert "child" in parent.instruction.lower()
    assert result["retry_diagnostics_by_task"]["task-1"].strategy_stage == (
        SCARemediationStage.NPM_LATEST
    )
    assert result["retry_diagnostics_by_task"]["task-1"].exhausted_update_path is True
    assert result["task_queue"][child_id].current_attempt_id is not None
    assert result["task_queue"][child_id].current_attempt_id in result["attempt_snapshots_by_id"]


# ===========================================================================
# run_supervisor_node â€” action summary updates
# ===========================================================================


class TestRunSupervisorNodeActionSummary:
    @patch("langchain_openai.ChatOpenAI")
    def test_qa_passed_subset_is_removed_before_retry_routing(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.NEEDS_RETRY, retry_count=1)
        state = _base_state(
            [g1, g2],
            status="qa_completed",
            task_queue={"task-1": task1, "task-2": task2},
            active_target_task_ids=["task-1"],
            qa_evaluations={"task-1": QAEvaluation(task_id="task-1", passed=True)},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            revised_instructions={
                "task-2": "Investigate patched releases or override paths for test-pkg."
            },
            instructions="retry remaining task",
            decision_reason="task-1 passed QA; task-2 still needs retry",
        )
        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        mock_chat.assert_not_called()


class TestRunSupervisorNodeActionSummaryUpdates:
    def test_updates_task_to_optimistically_fixed_on_success(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED

    def test_updates_task_to_needs_retry_on_surrender(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="failed",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY

    def test_updates_workaround_task_to_unfixable_on_surrender(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.CODE_WORKAROUND, status=TaskStatus.PENDING
        )
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SURRENDER,
            summary="Workaround subagent bypassed",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE

    def test_handles_multiple_action_summaries(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.PENDING)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        summary1 = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed task-1",
        )
        summary2 = AgentActionSummary(
            task_id="task-2",
            status=AgentActionStatus.SUCCESS,
            summary="fixed task-2",
        )
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            action_summaries=[summary1, summary2],
            active_target_task_ids=["task-1", "task-2"],
        )
        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert result["task_queue"]["task-2"].status == TaskStatus.OPTIMISTICALLY_FIXED

    def test_does_not_overwrite_terminal_statuses_from_action_summary(self):
        g1 = _sca_group("g1")
        task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            action_summaries=[summary],
            active_target_task_ids=["task-1"],
        )
        result = run_supervisor_node(state)
        # QA_PASSED is terminal â€” must not be overwritten
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_does_not_update_non_active_targets_from_action_summary(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.NEEDS_RETRY)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        summary_g1 = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="fixed it",
        )
        # task-1 has a SUCCESS summary but is NOT in active_target_task_ids.
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            action_summaries=[summary_g1],
            active_target_task_ids=["task-2"],
        )
        result = run_supervisor_node(state)
        # task-1 must NOT be overwritten by its historical summary
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY


# ===========================================================================
# run_supervisor_node â€” max retries / unfixable marking
# ===========================================================================


class TestRunSupervisorNodeTargetGuardrails:
    @patch("langchain_openai.ChatOpenAI")
    def test_llm_update_targets_drop_terminal_tasks(self, mock_chat):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        g2 = _sca_group("g2", FixPlanStatus.VERSION_FOUND)
        task1 = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            revised_instructions={
                "task-1": "Do not dispatch terminal tasks.",
                "task-2": 'Update "test-pkg" in package.json to version "1.2.4".',
            },
            instructions="route update batch",
            decision_reason="router included one stale terminal task",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED

    def test_worker_target_normalization_drops_optimistically_fixed_tasks(self):
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.PENDING)

        assert _normalize_target_task_ids_for_node(
            "update_subagent",
            ["task-1", "task-2"],
            {"task-1": task1, "task-2": task2},
        ) == ["task-2"]

        assert _normalize_target_task_ids_for_node(
            "qa_critic",
            ["task-1", "task-2"],
            {"task-1": task1, "task-2": task2},
        ) == ["task-1"]

    def test_qa_normalization_accepts_nine_ready_tasks_under_limit(self):
        task_ids = [f"task-{index}" for index in range(1, 10)]
        task_queue = {
            task_id: _make_task(
                task_id,
                f"group-{index}",
                status=TaskStatus.OPTIMISTICALLY_FIXED,
            )
            for index, task_id in enumerate(task_ids, start=1)
        }

        assert _normalize_target_task_ids_for_node("qa_critic", task_ids, task_queue) == task_ids


class TestRunSupervisorMaxRetries:
    def test_max_retries_marks_task_unfixable_and_removes_from_targets(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES,
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        result = run_supervisor_node(state)
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert "task-1" not in result["active_target_task_ids"]

    @patch(
        "remediation_engine.orchestration.supervisor_node.get_runtime_settings",
        return_value=AppSettings(openai_api_key=""),
    )
    def test_qa_failure_at_retry_cap_pivots_before_terminalization(self, _settings):
        """The final failed update must be replanned before MAX_RETRIES closes it."""
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            status=TaskStatus.OPTIMISTICALLY_FIXED,
            retry_count=MAX_RETRIES - 1,
        ).model_copy(update={"strategy_stage": SCARemediationStage.NPM_LATEST})
        state = _base_state(
            [g1],
            status="qa_completed",
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="The latest candidate still contains the vulnerability.",
                )
            },
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_all_terminal_after_retry_cap_skips_router_and_tears_down(self, mock_chat):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        g3 = _sca_group("g3")
        task1 = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
        task2 = _make_task(
            "task-2",
            "g2",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES - 1,
        )
        task3 = _make_task(
            "task-3",
            "g3",
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES - 1,
        )
        summaries = [
            AgentActionSummary(
                task_id="task-2",
                status=AgentActionStatus.SURRENDER,
                summary="No viable manifest update remains.",
            ),
            AgentActionSummary(
                task_id="task-3",
                status=AgentActionStatus.SURRENDER,
                summary="No viable manifest update remains.",
            ),
        ]
        state = _base_state(
            [g1, g2, g3],
            task_queue={
                "task-1": task1,
                "task-2": task2,
                "task-3": task3,
            },
            action_summaries=summaries,
            active_target_task_ids=["task-2", "task-3"],
            status="supervisor_entered",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "teardown"
        assert result["active_target_task_ids"] == []
        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["task_queue"]["task-2"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-3"].status == TaskStatus.UNFIXABLE
        mock_chat.assert_not_called()


def test_newly_unfixable_sca_transition_replans_before_dispatch(monkeypatch):
    _keep_portfolio_current(monkeypatch)
    groups = [_sca_group("g1"), _sca_group("g2"), _sca_group("g3")]
    task1 = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.NEEDS_RETRY,
        retry_count=MAX_RETRIES,
    )
    task2 = _make_task("task-2", "g2")
    task3 = _make_task("task-3", "g3")
    state = _base_state(
        groups,
        task_queue={"task-3": task3, "task-1": task1, "task-2": task2},
        portfolio_plan=_portfolio_stub(),
    )
    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)

    result = run_supervisor_node(state)
    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["task_queue"] == result["task_queue"]
    assert portfolio_input["portfolio_replan_request"] == result["portfolio_replan_request"]

    request = result["portfolio_replan_request"]
    assert request.reason == "UNFIXABLE_REPLAN"
    assert request.source_portfolio_plan_id == "portfolio-current"
    assert request.peer_conflict_pairs == []
    assert request.forced_singleton_task_ids == []
    assert result["portfolio_dirty"] is True
    assert result["next_routing_step"] == "supervisor"
    assert result["decision_code"].value == "PORTFOLIO_PLAN_REQUIRED"
    assert result["active_target_task_ids"] == []
    assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
    assert result["task_queue"]["task-2"].current_attempt_id is None
    assert result["task_queue"]["task-3"].current_attempt_id is None
    assert result["supervisor_audit"].reasoning == (
        "Newly unfixable SCA task(s) ['task-1']; remaining nonterminal SCA task(s) "
        "['task-2', 'task-3'] require replanning."
    )


def test_existing_unfixable_sca_does_not_trigger_another_replan(monkeypatch):
    _keep_portfolio_current(monkeypatch)
    groups = [_sca_group("g1"), _sca_group("g2")]
    state = _base_state(
        groups,
        task_queue={
            "task-1": _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE),
            "task-2": _make_task("task-2", "g2").model_copy(
                update={"instruction": "Update the package to an approved version."}
            ),
        },
        portfolio_plan=_portfolio_stub(),
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_not_called()

    assert result["portfolio_replan_request"] is None
    assert result["next_routing_step"] == "update_subagent"
    assert result["active_target_task_ids"] == ["task-2"]


def test_pending_unfixable_replan_waits_for_open_attempt_reconciliation(monkeypatch):
    _keep_portfolio_current(monkeypatch)
    groups = [_sca_group("g1"), _sca_group("g2")]
    open_task = _make_task(
        "task-open",
        "g2",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "qa_policy": QAPolicy.VERSION_BUMP,
            "instruction": "Verify the committed package update.",
        }
    )
    snapshots = {}
    open_task, snapshot = _create_attempt_snapshot(
        open_task,
        dispatch_node="update_subagent",
        snapshots_by_id=snapshots,
        state_revision=1,
        allowed_target_versions=["2.0.0"],
        allowed_dependency_types=["dependencies"],
    )
    request = PortfolioReplanRequest(reason="UNFIXABLE_REPLAN")
    state = _base_state(
        groups,
        task_queue={
            "task-1": _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE),
            open_task.task_id: open_task,
        },
        active_target_task_ids=[open_task.task_id],
        attempt_snapshots_by_id={snapshot.attempt_id: snapshot},
        portfolio_replan_request=request,
        portfolio_plan=_portfolio_stub(),
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_not_called()

    assert result["portfolio_replan_request"] == request
    assert result["portfolio_dirty"] is True
    assert result["next_routing_step"] == "qa_critic"
    assert result["active_target_task_ids"] == [open_task.task_id]
    assert result["task_queue"][open_task.task_id].current_attempt_id == snapshot.attempt_id


def test_pending_unfixable_replan_routes_when_attempt_barrier_is_clear(monkeypatch):
    _keep_portfolio_current(monkeypatch)
    groups = [_sca_group("g1"), _sca_group("g2")]
    request = PortfolioReplanRequest(reason="UNFIXABLE_REPLAN")
    state = _base_state(
        groups,
        task_queue={
            "task-1": _make_task("task-1", "g1", status=TaskStatus.UNFIXABLE),
            "task-2": _make_task("task-2", "g2"),
        },
        portfolio_replan_request=request,
        portfolio_plan=_portfolio_stub(),
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["task_queue"] == result["task_queue"]
    assert portfolio_input["portfolio_replan_request"] == request

    assert result["portfolio_replan_request"] == request
    assert result["next_routing_step"] == "supervisor"
    assert result["decision_code"].value == "PORTFOLIO_PLAN_REQUIRED"
    assert result["active_target_task_ids"] == []


def test_unfixable_replan_clears_without_remaining_nonterminal_sca(monkeypatch):
    _keep_portfolio_current(monkeypatch)
    groups = [_sca_group("g1"), _sast_group("g2")]
    request = PortfolioReplanRequest(reason="UNFIXABLE_REPLAN")
    state = _base_state(
        groups,
        task_queue={
            "task-sca": _make_task("task-sca", "g1", status=TaskStatus.UNFIXABLE),
            "task-sast": _make_task(
                "task-sast",
                "g2",
                strategy=RoutingStrategy.CODE_WORKAROUND,
            ).model_copy(update={"instruction": "Apply the committed source workaround."}),
        },
        portfolio_replan_request=request,
        portfolio_dirty=True,
        portfolio_plan=_portfolio_stub(),
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_not_called()

    assert result["portfolio_replan_request"] is None
    assert result["portfolio_dirty"] is False
    assert result["next_routing_step"] != "portfolio"


def test_pending_unfixable_replan_ignores_committed_workaround_child(
    monkeypatch: pytest.MonkeyPatch,
):
    _keep_portfolio_current(monkeypatch)
    group = _sca_group("g1")
    parent = _make_task(
        "task-1",
        group.group_id,
        status=TaskStatus.UNFIXABLE,
    ).model_copy(
        update={
            "exhausted_update_path": True,
            "portfolio_plan_id": "portfolio-current",
            "strategy_stage": SCARemediationStage.PYPI_LATEST,
        }
    )
    child = _make_task(
        "task-2",
        group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
    ).model_copy(
        update={
            "parent_task_id": parent.task_id,
            "strategy_stage": SCARemediationStage.CODE_WORKAROUND,
            "instruction": "Apply the committed code workaround.",
        }
    )
    plan = _portfolio_stub()
    plan.task_ids = [parent.task_id]
    request = PortfolioReplanRequest(
        reason="UNFIXABLE_REPLAN",
        source_portfolio_plan_id=plan.portfolio_plan_id,
    )
    state = _base_state(
        [group],
        task_queue={parent.task_id: parent, child.task_id: child},
        portfolio_plan=plan,
        portfolio_replan_request=request,
        portfolio_dirty=True,
    )
    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)

    result = run_supervisor_node(state)

    portfolio_tool.assert_not_called()
    assert result["portfolio_replan_request"] is None
    assert result["portfolio_dirty"] is False
    assert result["next_routing_step"] == "workaround_subagent"
    assert result["active_target_task_ids"] == [child.task_id]
    attempt_id = result["task_queue"][child.task_id].current_attempt_id
    assert attempt_id is not None
    assert result["attempt_snapshots_by_id"][attempt_id].dispatch_node == "workaround_subagent"


def test_nonterminal_sca_replan_excludes_only_committed_workaround_children():
    group = _sca_group("g1")
    sibling_group = _sca_group("g2")
    parent = _make_task(
        "task-1",
        group.group_id,
        status=TaskStatus.UNFIXABLE,
    ).model_copy(update={"exhausted_update_path": True})
    child = _make_task(
        "task-2",
        group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
    ).model_copy(update={"parent_task_id": parent.task_id})
    sibling = _make_task("task-3", sibling_group.group_id)
    plan = _portfolio_stub()
    plan.task_ids = [parent.task_id, sibling.task_id]
    task_queue = {
        parent.task_id: parent,
        child.task_id: child,
        sibling.task_id: sibling,
    }

    remaining = _nonterminal_sca_task_ids(
        task_queue,
        {group.group_id: group, sibling_group.group_id: sibling_group},
        plan,
    )

    assert remaining == [sibling.task_id]


def test_early_stale_plan_route_preserves_typed_pending_request(monkeypatch):
    request = PortfolioReplanRequest(
        reason="PEER_CONFLICT_ESCALATION",
        peer_conflict_pairs=[("task-1", "task-2")],
        forced_singleton_task_ids=["task-2"],
    )
    state = _base_state(
        [_sca_group("g1")],
        portfolio_replan_request=request,
        portfolio_plan=_portfolio_stub(),
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
        lambda *args, **kwargs: ["committed plan is stale"],
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["portfolio_replan_request"] == request

    assert result["next_routing_step"] == "supervisor"
    assert result["portfolio_replan_request"] == request


# ===========================================================================
# run_supervisor_node â€” strategy pivots
# ===========================================================================


class TestRunSupervisorNodePeerConflict:
    @patch("langchain_openai.ChatOpenAI")
    def test_workaround_spawn_with_missing_target_is_normalized_to_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
        )
        state = _base_state([g1], task_queue={"task-1": task})
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = (
            SupervisorDecision.model_construct(
                next_node="workaround_subagent",
                target_task_ids=[],
                spawn_requests=[
                    TaskSpawnRequest(
                        parent_task_id="task-1",
                        strategy=RoutingStrategy.CODE_WORKAROUND,
                        instruction="Implement a code workaround for the failed update.",
                        reason="the update path is exhausted",
                    )
                ],
                instructions="pivot to workaround",
                decision_reason="pivot after exhausted update path",
            )
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert result["task_queue"]["task-1"].strategy == RoutingStrategy.VERSION_BUMP

    @patch("langchain_openai.ChatOpenAI")
    def test_llm_strategy_pivot_spawns_child_and_routes_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Implement a code workaround for the unresolved peer conflict.",
                    reason="peer conflict requires a workaround child task",
                )
            ],
            instructions="test",
            decision_reason="test",
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert result["task_queue"]["task-1"].strategy == RoutingStrategy.VERSION_BUMP
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY

    @patch("langchain_openai.ChatOpenAI")
    def test_breaking_change_pivot_marks_parent_pivoted_and_routes_child(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="jsonwebtoken v9 broke runtime expectations",
                )
            },
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Implement a compatibility workaround for the new jsonwebtoken API.",
                    reason="breaking change requires validated upgrade plus workaround child",
                )
            ],
            instructions="test",
            decision_reason="spawn a workaround child after the validated version bump caused regressions",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY

    @patch("langchain_openai.ChatOpenAI")
    def test_strategy_pivot_without_child_instruction_fails_closed(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
        )

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.with_structured_output.return_value.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            updated_task_strategies={"task-1": RoutingStrategy.CODE_WORKAROUND},
            instructions="retry with a workaround",
            decision_reason="pivot the task after the retry failed",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert "task-2" not in result["task_queue"]
        assert result["task_queue"]["task-1"].status == TaskStatus.NEEDS_RETRY

    @patch("langchain_openai.ChatOpenAI")
    def legacy_breaking_change_fallback_spawns_child_instead_of_retrying_parent(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="jsonwebtoken v9 broke runtime expectations",
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.PIVOTED
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_exhausted_update_path_fallback_spawns_child_and_terminalizes_parent(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    selected_version=None,
                    latest_version_seen="9.0.3",
                    used_overrides=False,
                    package_abandoned=False,
                    exhausted_update_path=True,
                    failure_reason="latest version already attempted",
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"
        assert result["task_queue"]["task-2"].strategy == RoutingStrategy.CODE_WORKAROUND

    @patch("langchain_openai.ChatOpenAI")
    def test_exhausted_update_at_retry_cap_pivots_before_terminal_cap(self, mock_chat):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=MAX_RETRIES,
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    latest_version_seen="9.0.3",
                    exhausted_update_path=True,
                )
            },
        )

        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"

    def test_deterministic_guardrail_routes_exhausted_update_to_workaround(self):
        g1 = _sca_group("g1")
        task = _make_task(
            "task-1", "g1", strategy=RoutingStrategy.VERSION_BUMP, status=TaskStatus.NEEDS_RETRY
        )
        state = _base_state(
            [g1],
            task_queue={"task-1": task},
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    registry_query_performed=True,
                    attempted_versions=["9.0.3"],
                    candidate_versions_considered=["9.0.3"],
                    latest_version_seen="9.0.3",
                    exhausted_update_path=True,
                )
            },
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["active_target_task_ids"] == ["task-2"]
        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].parent_task_id == "task-1"

    @patch("langchain_openai.ChatOpenAI")
    def test_deterministic_routing_ignores_multi_spawn_advisory(self, mock_chat):
        groups = [_sca_group(f"g{i}", FixPlanStatus.VERSION_FOUND) for i in range(3)]
        tasks = {
            f"task-{i + 1}": _make_task(
                f"task-{i + 1}",
                f"g{i}",
                strategy=RoutingStrategy.VERSION_BUMP,
                status=TaskStatus.NEEDS_RETRY,
                retry_count=1,
            )
            for i in range(3)
        }
        state = _base_state(
            groups,
            task_queue=tasks,
        )

        router_llm = MagicMock()
        structured = MagicMock()
        mock_chat.return_value = router_llm
        router_llm.with_structured_output.return_value = structured
        structured.invoke.return_value = SupervisorDecision(
            next_node="workaround_subagent",
            target_task_ids=["task-1"],
            spawn_requests=[
                TaskSpawnRequest(
                    parent_task_id="task-1",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 1.",
                    reason="pivot 1",
                ),
                TaskSpawnRequest(
                    parent_task_id="task-2",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 2.",
                    reason="pivot 2",
                ),
                TaskSpawnRequest(
                    parent_task_id="task-3",
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction="Work around package 3.",
                    reason="pivot 3",
                ),
            ],
            instructions="spawn three workaround children",
            decision_reason="batch pivot to workaround",
        )

        result = run_supervisor_node(state)

        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-1"]
        assert all(
            result["task_queue"][task_id].status == TaskStatus.NEEDS_RETRY
            for task_id in ("task-1", "task-2", "task-3")
        )


# ===========================================================================
# run_supervisor_node â€” deterministic routing
# ===========================================================================


class TestRunSupervisorDeterministicRouting:
    def test_version_bump_routes_without_a_supervisor_llm(self):
        g1 = _sca_group("g1", FixPlanStatus.VERSION_FOUND)
        state = _base_state([g1])

        result = run_supervisor_node(state)
        # Deterministic routing sends VERSION_FOUND to update_subagent.
        assert result["next_routing_step"] == "update_subagent"
        assert len(result["active_target_task_ids"]) == 1


class TestBugFixes:
    def test_bug1_deterministic_pivot_populates_feedback_by_task(self):
        group_id = "sca:package.json:express"
        g = _sca_group(group_id)
        task = RemediationTask(
            task_id="task-1",
            parent_group_id=group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        task_queue = {"task-1": task}
        group_by_id = {group_id: g}
        qa_evaluations = {
            "task-1": QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.PEER_CONFLICT,
                retry_feedback="Peer dependency conflict with body-parser",
            )
        }
        retry_diagnostics_by_task = {
            "task-1": UpdateRetryDiagnostics(
                task_id="task-1",
                exhausted_update_path=True,
            )
        }

        decision = _deterministic_routing(
            task_queue=task_queue,
            group_by_id=group_by_id,
            qa_evaluations=qa_evaluations,
            retry_diagnostics_by_task=retry_diagnostics_by_task,
        )

        assert decision.next_node == "workaround_subagent"
        assert len(decision.spawn_requests) == 1
        assert (
            decision.feedback_by_task.get("task-1") == "Peer dependency conflict with body-parser"
        )

    def test_peer_conflict_escalation_ignores_satisfied_peer_evidence(self):
        compiler_group_id = "group-compiler"
        compiler_cli_group_id = "group-compiler-cli"
        compiler_group = _sca_group(compiler_group_id).model_copy(
            update={"vulnerable_component": "@angular/compiler", "file_paths": ["package.json"]}
        )
        compiler_cli_group = _sca_group(compiler_cli_group_id).model_copy(
            update={
                "vulnerable_component": "@angular/compiler-cli",
                "file_paths": ["package.json"],
            }
        )
        compiler_task = _make_task("task-compiler", compiler_group_id).model_copy(
            update={"target_package_name": "@angular/compiler"}
        )
        compiler_cli_task = _make_task("task-cli", compiler_cli_group_id).model_copy(
            update={"target_package_name": "@angular/compiler-cli"}
        )
        task_queue = {
            compiler_task.task_id: compiler_task,
            compiler_cli_task.task_id: compiler_cli_task,
        }
        groups = {
            compiler_group.group_id: compiler_group,
            compiler_cli_group.group_id: compiler_cli_group,
        }
        gates = QADeterministicGates(
            status="fail",
            install_passed=False,
            scanner_execution_status="success",
            tests_passed=False,
            peer_conflicts=[
                PeerConflictEvidence(
                    requester_package="@angular/compiler-cli",
                    requester_version="22.2.1",
                    peer_package="@angular/compiler",
                    required_range="22.2.1",
                    observed_version="22.2.1",
                ),
                PeerConflictEvidence(
                    requester_package="@angular/build",
                    requester_version="21.2.12",
                    peer_package="@angular/compiler",
                    required_range="^21.0.0",
                    observed_version="22.2.1",
                ),
            ],
        )
        evaluation = QAEvaluation(
            task_id=compiler_task.task_id,
            passed=False,
            failure_category=FailureCategory.PEER_CONFLICT,
            retry_feedback="npm install ERESOLVE",
            deterministic_gates=gates,
        )

        escalation, unresolved = _peer_conflict_escalation(
            compiler_task.task_id,
            evaluation,
            task_queue,
            groups,
        )

        assert escalation is None
        assert unresolved == [
            "peer escalation for @angular/build -> @angular/compiler has no existing "
            "eligible active package task"
        ]

        actual_conflict = PeerConflictEvidence(
            requester_package="@angular/compiler-cli",
            requester_version="22.2.1",
            peer_package="@angular/compiler",
            required_range="^21.0.0",
            observed_version="22.2.1",
        )
        actual_conflict_evaluation = evaluation.model_copy(
            update={
                "deterministic_gates": gates.model_copy(
                    update={"peer_conflicts": [actual_conflict]}
                )
            }
        )
        actual_escalation, actual_unresolved = _peer_conflict_escalation(
            compiler_task.task_id,
            actual_conflict_evaluation,
            task_queue,
            groups,
        )

        assert actual_escalation is not None
        assert actual_escalation["peer_conflict_pairs"] == [("task-cli", "task-compiler")]
        assert actual_unresolved == []

    def test_bug2_materialize_spawn_requests_replaces_triage_strategy_bucket(self):
        group_id = "sca:package.json:express"
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id=group_id,
            qa_policy=QAPolicy.VERSION_BUMP,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        task_queue = {"task-1": parent_task}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Pivot to workaround",
            reason="Exhausted version bump",
        )
        errors: list[str] = []

        new_tasks, _ = _materialize_spawn_requests(
            spawn_requests=[spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="The update attempt broke the test suite.",
                )
            },
        )

        assert len(new_tasks) == 1
        child_task = next(iter(new_tasks.values()))
        assert child_task.parent_group_id == "sca:package.json:express"
        assert child_task.strategy == RoutingStrategy.CODE_WORKAROUND
        assert len(errors) == 0

    def test_materialize_spawn_requests_reuses_existing_child_for_parent(self):
        """A replayed pivot must not create a second workaround sibling."""
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id="sca:package.json:express",
            qa_policy=QAPolicy.VERSION_BUMP,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        existing_child = RemediationTask(
            task_id="task-2",
            parent_group_id="sca:package.json:express",
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.PENDING,
            instruction="Apply workaround",
        )
        task_queue = {parent_task.task_id: parent_task, existing_child.task_id: existing_child}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Pivot to workaround",
            reason="Exhausted version bump",
        )
        errors: list[str] = []

        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=[spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
        )

        assert new_tasks == {}
        assert child_ids_by_parent == {"task-1": ["task-2"]}
        assert errors == []

    def test_materialize_spawn_requests_deduplicates_requests_in_one_decision(self):
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id="sca:package.json:express",
            qa_policy=QAPolicy.VERSION_BUMP,
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Bump express",
        )
        task_queue = {parent_task.task_id: parent_task}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Pivot to workaround",
            reason="Exhausted version bump",
        )
        errors: list[str] = []

        new_tasks, child_ids_by_parent = _materialize_spawn_requests(
            spawn_requests=[spawn_req, spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="The update attempt broke the test suite.",
                )
            },
        )

        assert len(new_tasks) == 1
        assert len(child_ids_by_parent["task-1"]) == 1
        assert errors == []

    def test_terminal_tasks_route_to_teardown_without_final_scan(self):
        group = _sast_group("g1")
        task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.QA_PASSED,
        )
        state = _base_state(
            [group],
            task_queue={"task-1": task},
            workspace_volume="workspace-volume",
        )

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
        assert result["next_routing_step"] == "teardown"
        assert result["active_target_task_ids"] == []

    @patch("langchain_openai.ChatOpenAI")
    def test_terminal_pivot_child_releases_other_pending_tasks(self, mock_chat):
        """A terminal pivot child must not starve unrelated pending tasks."""
        groups = [_sca_group("g1"), _sca_group("g2")]
        parent_task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
        )
        existing_child = _make_task(
            "task-2",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.UNFIXABLE,
        ).model_copy(update={"parent_task_id": "task-1"})
        pending_task = _make_task(
            "task-3",
            "g2",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.PENDING,
        )
        state = _base_state(
            groups,
            task_queue={
                "task-1": parent_task,
                "task-2": existing_child,
                "task-3": pending_task,
            },
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    exhausted_update_path=True,
                )
            },
        )
        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.UNFIXABLE
        assert result["task_queue"]["task-2"].status == TaskStatus.UNFIXABLE
        assert result["next_routing_step"] == "update_subagent"
        assert result["active_target_task_ids"] == ["task-3"]

    def test_bug3_reject_workaround_to_workaround_spawn(self):
        parent_task = RemediationTask(
            task_id="task-1",
            parent_group_id="g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            instruction="Workaround",
        )
        task_queue = {"task-1": parent_task}
        spawn_req = TaskSpawnRequest(
            parent_task_id="task-1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            instruction="Refactor workaround",
            reason="Workaround failed",
        )
        errors: list[str] = []

        new_tasks, _ = _materialize_spawn_requests(
            spawn_requests=[spawn_req],
            task_queue=task_queue,
            group_by_id={},
            errors=errors,
        )

        assert len(new_tasks) == 0
        assert len(errors) == 1
        assert (
            "CODE_WORKAROUND parent 'task-1' cannot spawn another CODE_WORKAROUND child"
            in errors[0]
        )

    def test_bug3_deterministic_routing_filters_exhausted_workaround_tasks(self):
        task = RemediationTask(
            task_id="task-1",
            parent_group_id="g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=3,  # MAX_RETRIES
            instruction="Workaround",
        )
        task_queue = {"task-1": task}

        decision = _deterministic_routing(
            task_queue=task_queue,
            group_by_id={},
            qa_evaluations={},
            retry_diagnostics_by_task={},
        )

        assert decision.next_node == "teardown"
        assert decision.target_task_ids == []

    @patch("langchain_openai.ChatOpenAI")
    def test_bug4_programmatic_feedback_injection_for_workaround_retries(self, mock_chat_openai):
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = SupervisorDecision(
            target_task_ids=["task-2"],
            next_node="workaround_subagent",
            spawn_requests=[],
            status_updates={},
            unfixable_task_ids=[],
            new_constraints=[],
            feedback_by_task={},
            instructions="Routing workaround",
            decision_reason="Testing",
        )
        mock_llm.with_structured_output.return_value = mock_structured
        mock_chat_openai.return_value = mock_llm

        group_id = "sca:package.json:express"
        g = _sca_group(group_id)
        task = RemediationTask(
            task_id="task-2",
            parent_group_id=group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            strategy_stage=SCARemediationStage.CODE_WORKAROUND,
            status=TaskStatus.NEEDS_RETRY,
            retry_count=1,
            instruction="Workaround",
        )
        task_queue = {"task-2": task}
        qa_evaluations = {
            "task-2": QAEvaluation(
                task_id="task-2",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Real feedback from QA",
            )
        }

        result = run_supervisor_node(
            {
                "repo_root": "/tmp",
                "issues": [],
                "system_context": MagicMock(),
                "constraints_ledger": [],
                "retry_counts": {},
                "group_strategies": {},
                "action_summaries": [],
                "retry_plans_by_task": {},
                "task_queue": task_queue,
                "valid_groups": [g],
                "qa_evaluations": qa_evaluations,
                "active_target_task_ids": ["task-2"],
                "retry_diagnostics_by_task": {},
            }
        )

        assert result["next_routing_step"] == "workaround_subagent"
        assert result["feedback_by_task"]["task-2"] == "Real feedback from QA"


def _atomic_cluster_delta_failure_state(*, delta_case="valid"):
    """Build one attempt-correlated two-task atomic QA failure."""
    group_a = _sca_group("g-atomic-a")
    group_b = _sca_group("g-atomic-b")
    cluster_id = "cluster-atomic"
    dispatch_batch_id = "batch-atomic"
    action_digest = "atomic-action-digest"
    source_plan_id = "portfolio-source"
    tasks = {}
    snapshots = {}
    workers = {}
    qa_results = {}
    for suffix, group, retry_count in (
        ("a", group_a, 1),
        ("b", group_b, 0),
    ):
        task_id = f"task-{suffix}"
        attempt_id = f"attempt-{suffix}"
        instruction = f"Update test-pkg-{suffix} to 1.2.3."
        task = _make_task(
            task_id,
            group.group_id,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
            retry_count=retry_count,
        ).model_copy(
            update={
                "task_revision": 4,
                "current_attempt_id": attempt_id,
                "portfolio_plan_id": source_plan_id,
                "qa_policy": QAPolicy.VERSION_BUMP,
                "selected_version": "1.2.3",
                "allowed_target_versions": ["1.2.3", "1.3.0"],
                "target_package_name": f"test-pkg-{suffix}",
                "target_dependency_type": "dependencies",
                "instruction": instruction,
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id=attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=dispatch_batch_id,
            action_digest=action_digest,
            portfolio_plan_id=source_plan_id,
            qa_policy=task.qa_policy,
            strategy_stage=task.strategy_stage,
            selected_version=task.selected_version,
            allowed_target_versions=list(task.allowed_target_versions),
            target_package_name=task.target_package_name,
            target_dependency_type=task.target_dependency_type,
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="update_subagent",
        )
        evaluation = QAEvaluation(
            task_id=task_id,
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="the atomic candidate failed deterministic tests",
        )
        tasks[task_id] = task
        snapshots[attempt_id] = snapshot
        workers[attempt_id] = WorkerAttemptResult(
            attempt_id=attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=dispatch_batch_id,
            action_digest=action_digest,
            status=AgentActionStatus.SUCCESS,
            executed_versions=[task.selected_version],
            execution_diagnostics=WorkerExecutionDiagnostics(validation_passed=True),
            instruction_digest=snapshot.instruction_digest,
        )
        qa_results[attempt_id] = QAAttemptResult(
            attempt_id=attempt_id,
            task_id=task_id,
            task_revision=task.task_revision,
            cluster_id=cluster_id,
            dispatch_batch_id=dispatch_batch_id,
            action_digest=action_digest,
            qa_policy=QAPolicy.VERSION_BUMP,
            qa_policy_source="attempt_snapshot",
            evaluation=evaluation,
        )

    responsible_task_ids = ["task-b"]
    attempt_ids_by_task = {"task-a": "attempt-a", "task-b": "attempt-b"}
    record = {
        "status": "IDENTIFIED",
        "responsible_task_ids": responsible_task_ids,
        "tested_subsets": [["task-b"]],
        "executions": 1,
        "candidate_restored": True,
        "cluster_id": cluster_id,
        "dispatch_batch_id": dispatch_batch_id,
        "action_digest": action_digest,
        "source_portfolio_plan_id": source_plan_id,
        "attempt_ids_by_task": attempt_ids_by_task,
    }
    escalation = {
        "reason": "DELTA_ISOLATION_ATTRIBUTION",
        "forced_singleton_task_ids": ["task-b"],
        "triggering_attempt_id": "attempt-b",
        "source_portfolio_plan_id": source_plan_id,
    }
    if delta_case == "stale_attempt":
        record["attempt_ids_by_task"] = {
            **attempt_ids_by_task,
            "task-b": "attempt-old",
        }
    elif delta_case == "foreign_forced_task":
        escalation["forced_singleton_task_ids"] = ["task-foreign"]
    elif delta_case == "stale_source":
        record["source_portfolio_plan_id"] = "portfolio-old"
        escalation["source_portfolio_plan_id"] = "portfolio-old"

    state = _base_state(
        [group_a, group_b],
        status="qa_completed",
        task_queue=tasks,
        active_target_task_ids=["task-a", "task-b"],
        active_cluster_id=cluster_id,
        active_dispatch_batch_id=dispatch_batch_id,
        attempt_snapshots_by_id=snapshots,
        worker_results_by_attempt=workers,
        qa_results_by_attempt=qa_results,
        delta_isolation_by_cluster=(
            {cluster_id: record} if delta_case != "ordinary_failure" else {}
        ),
        portfolio_escalation=(escalation if delta_case != "ordinary_failure" else None),
    )
    return state, tasks, snapshots


def test_atomic_cluster_delta_isolation_charges_only_validated_blame(
    monkeypatch: pytest.MonkeyPatch,
):
    state, original_tasks, snapshots = _atomic_cluster_delta_failure_state()
    approved_inputs = {
        task_id: (
            task.selected_version,
            list(task.allowed_target_versions),
            task.target_package_name,
            task.target_dependency_type,
            task.instruction,
        )
        for task_id, task in original_tasks.items()
    }

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    result = run_supervisor_node(state)
    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["task_queue"] == result["task_queue"]
    assert portfolio_input["portfolio_replan_request"] == result["portfolio_replan_request"]

    blamed = result["task_queue"]["task-b"]
    unblamed = result["task_queue"]["task-a"]
    assert blamed.status == TaskStatus.NEEDS_RETRY
    assert blamed.retry_count == original_tasks["task-b"].retry_count + 1
    assert blamed.current_attempt_id is None
    assert blamed.task_revision > original_tasks["task-b"].task_revision
    assert unblamed.status == TaskStatus.PENDING
    assert unblamed.retry_count == original_tasks["task-a"].retry_count
    assert unblamed.current_attempt_id is None
    assert unblamed.task_revision > original_tasks["task-a"].task_revision
    assert (
        unblamed.selected_version,
        unblamed.allowed_target_versions,
        unblamed.target_package_name,
        unblamed.target_dependency_type,
        unblamed.instruction,
    ) == approved_inputs["task-a"]
    assert set(snapshots) == {"attempt-a", "attempt-b"}
    request = result["portfolio_replan_request"]
    assert isinstance(request, PortfolioReplanRequest)
    assert request.reason == "DELTA_ISOLATION_ATTRIBUTION"
    assert request.forced_singleton_task_ids == ["task-b"]
    assert request.triggering_attempt_id == "attempt-b"
    assert request.source_portfolio_plan_id == "portfolio-source"
    assert result["next_routing_step"] == "supervisor"


def test_delta_isolation_requeues_unblamed_peer_for_authoritative_singleton_qa(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _original_tasks, _snapshots = _atomic_cluster_delta_failure_state()
    _keep_portfolio_current(monkeypatch)
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._apply_tactical_supervisor",
        lambda *args, **kwargs: None,
    )

    def rebuild_portfolio(portfolio_input):
        task_queue = {
            task_id: task.model_copy(update={"portfolio_plan_id": "portfolio-replanned"})
            for task_id, task in portfolio_input["task_queue"].items()
        }
        clusters = [
            TaskCluster(cluster_id=f"singleton-{task_id}", task_ids=[task_id], reason="replanned")
            for task_id in ("task-a", "task-b")
        ]
        plan = PortfolioPlan(
            plan_id="plan-replanned",
            portfolio_plan_id="portfolio-replanned",
            repository_fingerprint="repo-fingerprint",
            graph_digest="graph-digest",
            plan_digest="plan-digest",
            task_ids=["task-a", "task-b"],
            clusters=clusters,
            cluster_order=[cluster.cluster_id for cluster in clusters],
            task_order=["task-a", "task-b"],
            task_to_cluster={task_id: f"singleton-{task_id}" for task_id in ("task-a", "task-b")},
            task_revisions={task_id: task.task_revision for task_id, task in task_queue.items()},
            planned_task_revisions={
                task_id: task.task_revision for task_id, task in task_queue.items()
            },
            task_strategies={task_id: task.strategy for task_id, task in task_queue.items()},
        )
        return {
            "status": "portfolio_ready",
            "next_routing_step": "supervisor",
            "portfolio_plan": plan,
            "portfolio_solver_plan": None,
            "portfolio_iteration": 1,
            "portfolio_replan_request": None,
            "portfolio_escalation": None,
            "portfolio_dirty": False,
            "valid_groups": portfolio_input["valid_groups"],
            "task_queue": task_queue,
            "active_target_task_ids": [],
            "active_cluster_id": None,
            "active_dispatch_batch_id": None,
            "active_multi_package_action": None,
            "errors": [],
        }

    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node.run_portfolio_plan_tool",
        rebuild_portfolio,
    )

    replanned = run_supervisor_node(state)
    record = replanned["delta_isolation_by_cluster"]["cluster-atomic"]
    assert record["authoritative_qa_replay_task_ids"] == ["task-a"]
    assert replanned["task_queue"]["task-a"].status == TaskStatus.PENDING
    assert replanned["task_queue"]["task-a"].retry_count == 1
    assert replanned["retry_diagnostics_by_task"]["task-a"].attempted_versions == ["1.2.3"]
    assert _validated_delta_replay_task_ids(
        replanned["task_queue"],
        replanned["delta_isolation_by_cluster"],
        replanned["attempt_snapshots_by_id"],
    ) == {"task-a"}

    singleton_dispatch = run_supervisor_node(replanned)
    assert singleton_dispatch["next_routing_step"] == "update_subagent"
    assert singleton_dispatch["active_target_task_ids"] == ["task-a"]
    peer = singleton_dispatch["task_queue"]["task-a"]
    attempt = peer.current_attempt_id
    assert attempt is not None
    snapshot = singleton_dispatch["attempt_snapshots_by_id"][attempt]
    assert snapshot.dispatch_batch_id is None
    assert snapshot.cluster_id is None
    assert snapshot.selected_version == "1.2.3"
    assert singleton_dispatch["retry_diagnostics_by_task"]["task-a"].attempted_versions == ["1.2.3"]
    assert (
        singleton_dispatch["delta_isolation_by_cluster"]["cluster-atomic"][
            "authoritative_qa_replay_task_ids"
        ]
        == []
    )

    worker_result = WorkerAttemptResult(
        attempt_id=attempt,
        task_id="task-a",
        task_revision=peer.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["1.2.3"],
        execution_diagnostics=WorkerExecutionDiagnostics(
            validation_passed=True,
            effective_target_version="1.2.3",
        ),
        instruction_digest=snapshot.instruction_digest,
    )
    worker_completed = run_supervisor_node(
        {
            **singleton_dispatch,
            "status": "worker_completed",
            "active_target_task_ids": ["task-a"],
            "worker_results_by_attempt": {
                **singleton_dispatch["worker_results_by_attempt"],
                attempt: worker_result,
            },
        }
    )
    assert worker_completed["next_routing_step"] == "qa_critic"
    assert worker_completed["active_target_task_ids"] == ["task-a"]

    passing_evaluation = QAEvaluation(task_id="task-a", passed=True)
    qa_result = QAAttemptResult(
        attempt_id=attempt,
        task_id="task-a",
        task_revision=worker_completed["task_queue"]["task-a"].task_revision,
        qa_policy=snapshot.qa_policy,
        qa_policy_source="attempt_snapshot",
        evaluation=passing_evaluation,
    )
    qa_completed = run_supervisor_node(
        {
            **worker_completed,
            "status": "qa_completed",
            "active_target_task_ids": ["task-a"],
            "qa_evaluations": {"task-a": passing_evaluation},
            "qa_results_by_attempt": {
                **worker_completed["qa_results_by_attempt"],
                attempt: qa_result,
            },
        }
    )
    assert qa_completed["task_queue"]["task-a"].status == TaskStatus.QA_PASSED


def test_atomic_cluster_without_delta_attribution_keeps_all_member_retry_behavior():
    state, original_tasks, _snapshots = _atomic_cluster_delta_failure_state(
        delta_case="ordinary_failure"
    )

    result = run_supervisor_node(state)

    for task_id, task in original_tasks.items():
        committed = result["task_queue"][task_id]
        assert committed.status == TaskStatus.NEEDS_RETRY
        assert committed.retry_count == task.retry_count + 1
        assert committed.current_attempt_id != task.current_attempt_id
        assert committed.task_revision > task.task_revision


@pytest.mark.parametrize(
    "delta_case",
    ["stale_attempt", "foreign_forced_task", "stale_source"],
)
def test_atomic_cluster_stale_delta_provenance_retries_every_member(delta_case):
    state, original_tasks, _snapshots = _atomic_cluster_delta_failure_state(delta_case=delta_case)

    result = run_supervisor_node(state)

    for task_id, task in original_tasks.items():
        committed = result["task_queue"][task_id]
        assert committed.status == TaskStatus.NEEDS_RETRY
        assert committed.retry_count == task.retry_count + 1
        assert committed.current_attempt_id is None
        assert committed.task_revision > task.task_revision
    assert result["portfolio_escalation"]["forced_singleton_task_ids"] == []
    request = result["portfolio_replan_request"]
    assert request.forced_singleton_task_ids == []
    assert request.triggering_attempt_id is None
    assert request.source_portfolio_plan_id is None


def test_supervisor_commits_complete_passing_scan_into_ledgers():
    group = _sca_group("g1")
    attempt_id = "attempt-current"
    task_revision = 2
    instruction = "Update test-pkg to 2.0.0."
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id="task-1",
        state_revision=2,
        task_revision=task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    old_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="test-pkg",
        version="1.0.0",
        lockfile_key="node_modules/test-pkg",
    )
    new_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="test-pkg",
        version="2.0.0",
        lockfile_key="node_modules/test-pkg",
    )
    stale_new_occurrence = new_occurrence.model_copy(
        update={"found_identifiers": ["CVE-2026-10001"]}
    )
    batch_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-10002",
            "package_name": "unrelated-pkg",
            "package_version": "1.0.0",
            "file_path": "package-lock.json",
        }
    )
    batch_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="unrelated-pkg",
        version="1.0.0",
        lockfile_key="node_modules/unrelated-pkg",
        found_identifiers=["CVE-2026-10002"],
        found_issues=[batch_issue],
    )
    old_key = '["package-lock.json","node_modules/test-pkg","test-pkg","1.0.0",0]'
    new_key = '["package-lock.json","node_modules/test-pkg","test-pkg","2.0.0",0]'
    evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        covered_input_paths=["package-lock.json"],
        covered_occurrences=[new_occurrence],
        removed_occurrence_keys=[old_key],
        inventory_updates={new_key: "new-fingerprint"},
        complete=True,
    )
    batch_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-1"],
        found_identifiers=["CVE-2026-10002"],
        new_identifiers=["CVE-2026-10002"],
        found_issues=[batch_issue],
        covered_input_paths=["package-lock.json"],
        covered_occurrences=[new_occurrence, batch_occurrence],
        complete=True,
        scan_performed=True,
    )
    qa_result = QAAttemptResult(
        attempt_id=attempt_id,
        task_id="task-1",
        task_revision=task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id="task-1",
            passed=True,
            scan_evidence=evidence,
        ),
        batch_scan_evidence=batch_evidence,
        batch_scan_candidate_retained=True,
    )
    task = _make_task(
        "task-1",
        "g1",
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": instruction,
        }
    )
    state = _base_state(
        [group],
        status="qa_completed",
        task_queue={"task-1": task},
        active_target_task_ids=["task-1"],
        attempt_snapshots_by_id={attempt_id: snapshot},
        qa_results_by_attempt={attempt_id: qa_result},
        accepted_package_inventory_by_lockfile={
            old_key: "old-fingerprint",
            new_key: "stale-fingerprint",
        },
        current_scan_observations_by_occurrence={
            old_key: old_occurrence,
            new_key: stale_new_occurrence,
        },
        latest_incremental_scan_by_input={
            "package-lock.json": IncrementalScanRecord(
                source_input_path="package-lock.json",
                task_ids=["task-1"],
                attempt_ids=["attempt-old"],
                found_identifiers=["CVE-2026-10001"],
                new_identifiers=["CVE-2026-10001"],
                complete=True,
            )
        },
    )

    result = run_supervisor_node(state)

    assert result["task_queue"]["task-1"].status == TaskStatus.QA_PASSED
    assert result["accepted_package_inventory_by_lockfile"] == {new_key: "new-fingerprint"}
    assert result["current_scan_observations_by_occurrence"] == {new_key: new_occurrence}
    committed_record = result["latest_successful_scan_by_task"]["task-1"]
    assert committed_record.attempt_id == attempt_id
    assert committed_record.task_revision == task_revision
    assert committed_record.removed_occurrence_keys == [old_key]
    assert committed_record.found_identifiers == []
    assert committed_record.found_issues == []
    incremental_record = result["latest_incremental_scan_by_input"]["package-lock.json"]
    assert incremental_record.attempt_ids == [attempt_id]
    assert incremental_record.found_identifiers == ["CVE-2026-10002"]
    assert result["new_vulnerability_identifiers"] == ["CVE-2026-10002"]
    assert result["post_remediation_scan_identifiers"] == ["CVE-2026-10002"]
    assert result["post_remediation_scan_issues"] == [batch_issue]
    scan_summary = result["final_full_scan_result"]
    assert scan_summary.coverage_scope == "incremental_dependency_inputs"
    assert scan_summary.authoritative is False
    assert scan_summary.new_identifiers == ["CVE-2026-10002"]
    assert state["accepted_package_inventory_by_lockfile"] == {
        old_key: "old-fingerprint",
        new_key: "stale-fingerprint",
    }
    assert state["current_scan_observations_by_occurrence"] == {
        old_key: old_occurrence,
        new_key: stale_new_occurrence,
    }


def _workerless_absence_state(
    *,
    include_portfolio_plan: bool = False,
    other_attempt_open: bool = False,
    open_attempt_status: TaskStatus = TaskStatus.OPTIMISTICALLY_FIXED,
) -> tuple[dict, str]:
    """Build a current passing QA result that removes an untouched SCA target."""
    source_group = _sca_group("g-source")
    target_group = _sca_group("g-target").model_copy(
        update={"vulnerable_component": "base64url", "file_path": "package.json"}
    )
    old_key = '["package-lock.json","node_modules/base64url","base64url","0.0.6",0]'
    source_task_id = "task-source"
    source_attempt_id = "attempt-source"
    source_instruction = "Upgrade test-pkg to 2.0.0."
    source_task = _make_task(
        source_task_id,
        source_group.group_id,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    ).model_copy(
        update={
            "task_revision": 1,
            "current_attempt_id": source_attempt_id,
            "qa_policy": QAPolicy.VERSION_BUMP,
            "target_package_name": "test-pkg",
            "selected_version": "2.0.0",
            "instruction": source_instruction,
        }
    )
    target_task_id = "task-absent"
    target_task = _make_task(target_task_id, target_group.group_id).model_copy(
        update={"target_package_name": "base64url"}
    )
    source_snapshot = TaskAttemptSnapshot(
        attempt_id=source_attempt_id,
        task_id=source_task_id,
        state_revision=1,
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy_stage=source_task.strategy_stage,
        selected_version="2.0.0",
        instruction=source_instruction,
        instruction_digest=instruction_digest(source_instruction),
        dispatch_node="update_subagent",
    )
    source_scan_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=[source_task_id],
        covered_input_paths=["package-lock.json"],
        removed_occurrence_keys=[old_key],
        complete=True,
        scan_performed=True,
    )
    source_qa_result = QAAttemptResult(
        attempt_id=source_attempt_id,
        task_id=source_task_id,
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id=source_task_id,
            passed=True,
            scan_evidence=source_scan_evidence,
        ),
    )

    groups = [source_group, target_group]
    task_queue = {
        source_task_id: source_task,
        target_task_id: target_task,
    }
    snapshots = {source_attempt_id: source_snapshot}
    active_target_task_ids = [source_task_id]
    if other_attempt_open:
        open_group = _sca_group("g-open")
        waiting_group = _sca_group("g-waiting").model_copy(
            update={"vulnerable_component": "waiting-package"}
        )
        open_task = _make_task(
            "task-open",
            open_group.group_id,
            status=open_attempt_status,
        ).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": "attempt-open",
                "qa_policy": QAPolicy.VERSION_BUMP,
                "instruction": "Upgrade the open package.",
            }
        )
        waiting_task = _make_task("task-waiting", waiting_group.group_id).model_copy(
            update={"target_package_name": "waiting-package"}
        )
        open_snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-open",
            task_id="task-open",
            state_revision=1,
            task_revision=1,
            qa_policy=QAPolicy.VERSION_BUMP,
            instruction=open_task.instruction,
            instruction_digest=instruction_digest(open_task.instruction),
            dispatch_node="update_subagent",
        )
        groups.extend([open_group, waiting_group])
        task_queue.update(
            {
                "task-open": open_task,
                "task-waiting": waiting_task,
            }
        )
        snapshots["attempt-open"] = open_snapshot
        active_target_task_ids.append("task-open")

    overrides: dict[str, object] = {
        "status": "qa_completed",
        "task_queue": task_queue,
        "active_target_task_ids": active_target_task_ids,
        "attempt_snapshots_by_id": snapshots,
        "qa_results_by_attempt": {source_attempt_id: source_qa_result},
        "accepted_package_inventory_by_lockfile": {old_key: "old-fingerprint"},
    }
    if include_portfolio_plan:
        plan_task_ids = list(task_queue)
        overrides["portfolio_plan"] = SimpleNamespace(
            plan_id="portfolio-current",
            portfolio_plan_id="portfolio-current",
            task_ids=plan_task_ids,
            clusters=[],
            cluster_order=[],
            task_order=plan_task_ids,
            task_to_cluster={},
            diagnostics=[],
        )
    return _base_state(groups, **overrides), old_key


def test_verified_absent_package_helper_fails_closed() -> None:
    group = _sca_group("g-target").model_copy(
        update={"vulnerable_component": "base64url", "file_path": "package.json"}
    )
    task = _make_task("task-target", group.group_id).model_copy(
        update={"target_package_name": "base64url"}
    )
    old_key = '["package-lock.json","node_modules/base64url","base64url","0.0.6",0]'
    accepted_record = TaskScanRecord(
        task_id="task-scan",
        attempt_id="attempt-scan",
        task_revision=1,
        covered_input_paths=["package-lock.json"],
        removed_occurrence_keys=[old_key],
        complete=True,
    )
    task_queue = {task.task_id: task}
    groups = {group.group_id: group}

    assert _verified_absent_package_tasks(
        task_queue,
        groups,
        {"task-scan": accepted_record},
        {},
    ) == {"task-target": "base64url"}
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {"task-scan": accepted_record.model_copy(update={"complete": False})},
            {},
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {"task-scan": accepted_record.model_copy(update={"covered_input_paths": []})},
            {},
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            {
                group.group_id: group.model_copy(
                    update={"file_path": "yarn.lock", "file_paths": ["yarn.lock"]}
                )
            },
            {"task-scan": accepted_record},
            {},
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {"task-scan": accepted_record},
            {"malformed-occurrence-key": "fingerprint"},
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {"task-scan": accepted_record},
            {
                '["package-lock.json","node_modules/nested/node_modules/base64url",'
                '"base64url","0.0.7",0]': "present-fingerprint"
            },
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {"task-scan": accepted_record},
            {
                '["another/package-lock.json","node_modules/base64url","base64url","0.0.7",0]': (
                    "present-fingerprint"
                )
            },
        )
        == {}
    )
    assert (
        _verified_absent_package_tasks(
            task_queue,
            groups,
            {
                "task-scan": accepted_record.model_copy(
                    update={"removed_occurrence_keys": ["not-a-canonical-tombstone"]}
                )
            },
            {},
        )
        == {}
    )

    worker_decision = _deterministic_routing(
        task_queue,
        groups,
        {},
        {},
        portfolio_plan=None,
    )
    assert worker_decision.next_node == "update_subagent"
    assert worker_decision.target_task_ids == ["task-target"]


def test_supervisor_completes_absent_package_without_worker_attempt() -> None:
    state, old_key = _workerless_absence_state()

    with (
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
            return_value=False,
        ),
    ):
        result = run_supervisor_node(state)

    target = result["task_queue"]["task-absent"]
    assert target.status == TaskStatus.QA_PASSED
    assert (
        target.remediation_action
        == "Removed due to another package upgrade; target absent from every covered package-lock input."
    )
    assert target.current_attempt_id is None
    assert "task-absent" not in result["active_target_task_ids"]
    assert result["accepted_package_inventory_by_lockfile"] == {}
    assert result["latest_successful_scan_by_task"]["task-source"].removed_occurrence_keys == [
        old_key
    ]
    assert all(
        snapshot.task_id != "task-absent" for snapshot in result["attempt_snapshots_by_id"].values()
    )
    assert all(
        worker_result.task_id != "task-absent"
        for worker_result in result["worker_results_by_attempt"].values()
    )
    assert all(
        qa_result.task_id != "task-absent" for qa_result in result["qa_results_by_attempt"].values()
    )


def test_verified_absent_portfolio_member_requires_replan_before_workers(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _old_key = _workerless_absence_state(include_portfolio_plan=True)
    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)

    with (
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
            return_value=False,
        ),
    ):
        result = run_supervisor_node(state)
    portfolio_tool.assert_called_once()
    portfolio_input = portfolio_tool.call_args.args[0]
    assert portfolio_input["task_queue"] == result["task_queue"]
    assert portfolio_input["portfolio_replan_request"] == result["portfolio_replan_request"]

    assert result["task_queue"]["task-absent"].status == TaskStatus.QA_PASSED
    assert result["decision_code"] == DecisionCode.PORTFOLIO_PLAN_REQUIRED
    assert result["next_routing_step"] == "supervisor"
    assert result["portfolio_dirty"] is True
    assert result["portfolio_replan_request"].reason == (
        "Verified target-package absence changed portfolio membership."
    )
    assert result["portfolio_replan_request"].source_portfolio_plan_id == "portfolio-current"
    assert "task-absent" not in result["active_target_task_ids"]


def test_verified_absent_portfolio_member_waits_for_open_attempts_then_replans(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _old_key = _workerless_absence_state(
        include_portfolio_plan=True,
        other_attempt_open=True,
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    with (
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
            return_value=False,
        ),
    ):
        result = run_supervisor_node(state)
    portfolio_tool.assert_not_called()

    assert result["task_queue"]["task-absent"].status == TaskStatus.QA_PASSED
    assert result["portfolio_dirty"] is True
    assert result["portfolio_replan_request"].reason == (
        "Verified target-package absence changed portfolio membership."
    )
    assert result["next_routing_step"] == "qa_critic"
    assert result["active_target_task_ids"] == ["task-open"]
    assert result["task_queue"]["task-waiting"].status == TaskStatus.PENDING


def test_verified_absence_does_not_dispatch_remaining_members_while_worker_attempt_is_open(
    monkeypatch: pytest.MonkeyPatch,
):
    state, _old_key = _workerless_absence_state(
        include_portfolio_plan=True,
        other_attempt_open=True,
        open_attempt_status=TaskStatus.PENDING,
    )

    portfolio_tool = _stub_portfolio_plan_tool(monkeypatch)
    with (
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
            return_value=[],
        ),
        patch(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
            return_value=False,
        ),
    ):
        result = run_supervisor_node(state)
    portfolio_tool.assert_not_called()

    assert result["task_queue"]["task-absent"].status == TaskStatus.QA_PASSED
    assert result["portfolio_dirty"] is True
    assert result["next_routing_step"] == "update_subagent"
    assert result["active_target_task_ids"] == ["task-open"]
    assert result["task_queue"]["task-open"].current_attempt_id == "attempt-open"
    assert result["task_queue"]["task-waiting"].status == TaskStatus.PENDING
    assert set(result["attempt_snapshots_by_id"]) == {"attempt-source", "attempt-open"}


def test_retained_regression_candidate_does_not_commit_batch_scan_findings():
    group = _sca_group("g-regression")
    attempt_id = "attempt-regression"
    task_revision = 1
    instruction = "Upgrade test-pkg to 2.0.0."
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id="task-regression",
        state_revision=1,
        task_revision=task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        selected_version="2.0.0",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    task_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-regression"],
        covered_input_paths=["package-lock.json"],
        complete=True,
        scan_performed=True,
    )
    batch_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-10003",
            "package_name": "report-wide-pkg",
            "package_version": "1.0.0",
        }
    )
    batch_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="report-wide-pkg",
        version="1.0.0",
        lockfile_key="node_modules/report-wide-pkg",
        found_identifiers=["CVE-2026-10003"],
        found_issues=[batch_issue],
    )
    qa_result = QAAttemptResult(
        attempt_id=attempt_id,
        task_id="task-regression",
        task_revision=task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id="task-regression",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The candidate failed an API test.",
            scan_evidence=task_evidence,
        ),
        batch_scan_evidence=ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            covered_task_ids=["task-regression"],
            found_identifiers=["CVE-2026-10003"],
            new_identifiers=["CVE-2026-10003"],
            found_issues=[batch_issue],
            covered_input_paths=["package-lock.json"],
            covered_occurrences=[batch_occurrence],
            complete=True,
            scan_performed=True,
        ),
        batch_scan_candidate_retained=True,
    )
    task = _make_task("task-regression", group.group_id).model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
            "strategy_stage": SCARemediationStage.NPM_LATEST,
            "selected_version": "2.0.0",
            "instruction": instruction,
        }
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            status="qa_completed",
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
            attempt_snapshots_by_id={attempt_id: snapshot},
            qa_results_by_attempt={attempt_id: qa_result},
        )
    )

    assert result["task_queue"][task.task_id].status != TaskStatus.QA_PASSED
    assert result["latest_successful_scan_by_task"] == {}
    assert result["new_vulnerability_identifiers"] == []
    assert result["latest_incremental_scan_by_input"] == {}
    assert result["qa_results_by_attempt"][attempt_id].batch_scan_candidate_retained is True


@pytest.mark.parametrize(
    ("passed", "complete"),
    [(False, True), (True, False)],
)
def test_task_scan_record_rejects_failed_or_incomplete_qa(passed, complete):
    qa_result = QAAttemptResult(
        attempt_id="attempt-1",
        task_id="task-1",
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id="task-1",
            passed=passed,
            failure_category=None if passed else FailureCategory.SECURITY_FLAG,
            retry_feedback=None if passed else "The target remains vulnerable.",
            scan_evidence=ODCScanEvidence(
                requested_scope=ScanScope.TARGETED,
                effective_scope=ScanScope.TARGETED,
                covered_task_ids=["task-1"],
                complete=complete,
            ),
        ),
    )

    assert _task_scan_record_from_qa_result(qa_result) is None


@pytest.mark.parametrize("failing_task_id", [None, "task-b"])
def test_atomic_cluster_commits_scan_records_only_when_every_member_passes(failing_task_id):
    state, _tasks, _snapshots = _atomic_cluster_delta_failure_state(delta_case="ordinary_failure")
    batch_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-77777",
            "package_name": "report-wide-pkg",
            "package_version": "1.0.0",
        }
    )
    batch_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=["task-a", "task-b"],
        found_identifiers=["CVE-2026-77777"],
        new_identifiers=["CVE-2026-77777"],
        found_issues=[batch_issue],
        covered_input_paths=["package-lock-a.json", "package-lock-b.json"],
        covered_occurrences=[
            PackageScanOccurrence(
                source_input_path="package-lock-a.json",
                package_name="report-wide-pkg",
                version="1.0.0",
                lockfile_key="node_modules/report-wide-pkg",
                found_identifiers=["CVE-2026-77777"],
                found_issues=[batch_issue],
            )
        ],
        complete=True,
        scan_performed=True,
    )
    prior_occurrence = PackageScanOccurrence(
        source_input_path="package-lock-prior.json",
        package_name="prior-pkg",
        version="1.0.0",
        lockfile_key="node_modules/prior-pkg",
    )
    prior_key = '["package-lock-prior.json","node_modules/prior-pkg","prior-pkg","1.0.0",0]'
    if failing_task_id is not None:
        state["accepted_package_inventory_by_lockfile"] = {prior_key: "prior-fingerprint"}
        state["current_scan_observations_by_occurrence"] = {prior_key: prior_occurrence}
    occurrence_keys = {}
    for suffix in ("a", "b"):
        task_id = f"task-{suffix}"
        attempt_id = f"attempt-{suffix}"
        occurrence = PackageScanOccurrence(
            source_input_path=f"package-lock-{suffix}.json",
            package_name=f"test-pkg-{suffix}",
            version="1.2.3",
            lockfile_key=f"node_modules/test-pkg-{suffix}",
        )
        occurrence_key = (
            f'["package-lock-{suffix}.json","node_modules/test-pkg-{suffix}",'
            f'"test-pkg-{suffix}","1.2.3",0]'
        )
        occurrence_keys[task_id] = occurrence_key
        previous_result = state["qa_results_by_attempt"][attempt_id]
        passed = failing_task_id is None or task_id != failing_task_id
        state["qa_results_by_attempt"][attempt_id] = previous_result.model_copy(
            update={
                "evaluation": QAEvaluation(
                    task_id=task_id,
                    passed=passed,
                    failure_category=None if passed else FailureCategory.BREAKING_CHANGE,
                    retry_feedback=None if passed else "the atomic candidate failed tests",
                    scan_evidence=ODCScanEvidence(
                        requested_scope=ScanScope.TARGETED,
                        effective_scope=ScanScope.TARGETED,
                        covered_task_ids=[task_id],
                        covered_input_paths=[f"package-lock-{suffix}.json"],
                        covered_occurrences=[occurrence],
                        inventory_updates={occurrence_key: f"fingerprint-{suffix}"},
                        complete=True,
                    ),
                ),
                "batch_scan_evidence": batch_evidence if suffix == "a" else None,
                "batch_scan_candidate_retained": (
                    failing_task_id is None if suffix == "a" else False
                ),
            }
        )

    result = run_supervisor_node(state)

    if failing_task_id is None:
        assert set(result["latest_successful_scan_by_task"]) == {"task-a", "task-b"}
        assert set(result["accepted_package_inventory_by_lockfile"]) == set(
            occurrence_keys.values()
        )
        assert all(task.status == TaskStatus.QA_PASSED for task in result["task_queue"].values())
        assert set(result["latest_incremental_scan_by_input"]) == {
            "package-lock-a.json",
            "package-lock-b.json",
        }
        assert result["new_vulnerability_identifiers"] == ["CVE-2026-77777"]
    else:
        assert result["latest_successful_scan_by_task"] == {}
        assert result["accepted_package_inventory_by_lockfile"] == {prior_key: "prior-fingerprint"}
        assert result["current_scan_observations_by_occurrence"] == {prior_key: prior_occurrence}
        assert state["accepted_package_inventory_by_lockfile"] == {prior_key: "prior-fingerprint"}
        assert state["current_scan_observations_by_occurrence"] == {prior_key: prior_occurrence}

        assert result["latest_incremental_scan_by_input"] == {}


def test_cumulative_scan_metrics_use_latest_qa_passed_records_and_baseline():
    baseline_issue = "CVE-2024-10001"
    introduced_a = "CVE-2026-10001"
    introduced_b = "CVE-2026-10002"
    shared_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": introduced_a,
        }
    )
    second_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": introduced_b,
        }
    )
    task_queue = {
        "task-a": _make_task("task-a", "g1", status=TaskStatus.QA_PASSED),
        "task-b": _make_task("task-b", "g2", status=TaskStatus.QA_PASSED),
        "task-pending": _make_task("task-pending", "g3", status=TaskStatus.PENDING),
    }
    scan_records = {
        "task-a": TaskScanRecord(
            task_id="task-a",
            attempt_id="attempt-a",
            task_revision=1,
            found_identifiers=[baseline_issue, introduced_a],
            new_identifiers=[baseline_issue, introduced_a],
            remaining_target_identifiers=[baseline_issue],
            found_issues=[shared_issue],
            complete=True,
        ),
        "task-b": TaskScanRecord(
            task_id="task-b",
            attempt_id="attempt-b",
            task_revision=1,
            found_identifiers=[introduced_a, introduced_b],
            new_identifiers=[introduced_a, introduced_b],
            found_issues=[shared_issue, second_issue],
            complete=True,
        ),
        "task-pending": TaskScanRecord(
            task_id="task-pending",
            attempt_id="attempt-pending",
            task_revision=1,
            found_identifiers=["CVE-2026-19999"],
            new_identifiers=["CVE-2026-19999"],
            complete=True,
        ),
    }

    new_ids, found_ids, remaining_ids, found_issues = _calculate_cumulative_scan_metrics(
        task_queue,
        scan_records,
        [baseline_issue.lower()],
    )

    assert new_ids == [introduced_a, introduced_b]
    assert found_ids == [baseline_issue, introduced_a, introduced_b]
    assert remaining_ids == [baseline_issue]
    assert len(found_issues) == 2
    assert {issue.cve_id for issue in found_issues} == {introduced_a, introduced_b}


def test_report_wide_scan_snapshot_replaces_stale_task_findings():
    old_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-10001",
        }
    )
    current_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-10002",
        }
    )
    task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
    stale_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="stale-pkg",
        version="1.0.0",
        lockfile_key="node_modules/stale-pkg",
        found_identifiers=["CVE-2026-10001"],
        found_issues=[old_issue],
    )
    stale_task_record = TaskScanRecord(
        task_id="task-1",
        attempt_id="attempt-old",
        task_revision=1,
        found_identifiers=["CVE-2026-10001"],
        new_identifiers=["CVE-2026-10001"],
        found_issues=[old_issue],
        covered_input_paths=["package-lock.json"],
        covered_occurrences=[stale_occurrence],
        complete=True,
    )
    current_input_record = IncrementalScanRecord(
        source_input_path="package-lock.json",
        task_ids=["task-1"],
        attempt_ids=["attempt-current"],
        found_identifiers=["CVE-2026-10002"],
        new_identifiers=["CVE-2026-10002"],
        found_issues=[current_issue],
        complete=True,
    )

    new_ids, found_ids, remaining_ids, found_issues = _calculate_cumulative_scan_metrics(
        {"task-1": task},
        {"task-1": stale_task_record},
        [],
        incremental_scan_records_by_input={"package-lock.json": current_input_record},
    )

    assert new_ids == ["CVE-2026-10002"]
    assert found_ids == ["CVE-2026-10002"]
    assert remaining_ids == []
    assert found_issues == [current_issue]


@pytest.mark.parametrize(
    ("triage_enabled", "expected_route"),
    [(True, "triage"), (False, "teardown")],
)
def test_supervisor_projects_incremental_metrics_before_terminal_routing(
    triage_enabled, expected_route
):
    group = _sast_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-55555",
        }
    )
    scan_record = TaskScanRecord(
        task_id="task-1",
        attempt_id="attempt-1",
        task_revision=1,
        found_identifiers=["CVE-2026-55555"],
        new_identifiers=["CVE-2026-55555"],
        found_issues=[issue],
        complete=True,
    )
    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            latest_successful_scan_by_task={"task-1": scan_record},
            baseline_scan_identifiers=["CVE-2024-10001"],
            baseline_scan_complete=True,
            package_inventory_baseline_complete=True,
            post_qa_triage_enabled=triage_enabled,
        )
    )

    assert result["new_vulnerability_identifiers"] == ["CVE-2026-55555"]
    assert result["post_remediation_scan_identifiers"] == ["CVE-2026-55555"]
    assert result["post_remediation_scan_issues"] == [issue]
    assert result["new_vulnerability_status"] == "detected"
    assert result["triage_required"] is triage_enabled
    assert result["next_routing_step"] == expected_route
    summary = result["final_full_scan_result"]
    assert summary.authoritative is False
    assert summary.coverage_scope == "incremental_dependency_inputs"
    assert summary.coverage_complete is True


def test_supervisor_does_not_retriage_a_scan_already_processed_by_post_qa_triage():
    group = _sast_group("g1")
    task = _make_task(
        "task-1",
        "g1",
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-55555",
        }
    )
    scan_record = TaskScanRecord(
        task_id="task-1",
        attempt_id="attempt-1",
        task_revision=1,
        found_identifiers=["CVE-2026-55555"],
        new_identifiers=["CVE-2026-55555"],
        found_issues=[issue],
        complete=True,
    )

    result = run_supervisor_node(
        _base_state(
            [group],
            task_queue={"task-1": task},
            latest_successful_scan_by_task={"task-1": scan_record},
            baseline_scan_identifiers=["CVE-2024-10001"],
            baseline_scan_complete=True,
            package_inventory_baseline_complete=True,
            status="portfolio_ready",
            triage_required=False,
            post_qa_retriage_count=1,
        )
    )

    assert result["new_vulnerability_identifiers"] == ["CVE-2026-55555"]
    assert result["new_vulnerability_status"] == "detected"
    assert result["triage_required"] is False
    assert result["next_routing_step"] == "teardown"


def test_supervisor_defers_incremental_triage_while_remediation_remains():
    completed_group = _sast_group("g-completed")
    pending_group = _sast_group("g-pending")
    completed_task = _make_task(
        "task-completed",
        completed_group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    pending_task = _make_task(
        "task-pending",
        pending_group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.PENDING,
    ).model_copy(update={"instruction": "Apply the selected workaround."})
    scan_record = TaskScanRecord(
        task_id="task-completed",
        attempt_id="attempt-completed",
        task_revision=1,
        found_identifiers=["CVE-2026-55555"],
        new_identifiers=["CVE-2026-55555"],
        complete=True,
    )

    result = run_supervisor_node(
        _base_state(
            [completed_group, pending_group],
            task_queue={
                completed_task.task_id: completed_task,
                pending_task.task_id: pending_task,
            },
            latest_successful_scan_by_task={
                completed_task.task_id: scan_record,
            },
            baseline_scan_identifiers=["CVE-2024-10001"],
        )
    )

    assert result["triage_required"] is True
    assert result["next_routing_step"] == "workaround_subagent"
    assert result["active_target_task_ids"] == ["task-pending"]


def test_supervisor_retriage_limit_suppresses_triage_and_preserves_metrics():
    result = run_supervisor_node(
        _base_state(
            [_sast_group("g1")],
            triage_required=True,
            new_vulnerability_identifiers=["CVE-2026-55555"],
            new_vulnerability_status="detected",
            post_qa_retriage_limit_reached=True,
        )
    )

    assert result["next_routing_step"] == "teardown"
    assert result["triage_required"] is False
    assert result["new_vulnerability_identifiers"] == ["CVE-2026-55555"]
    assert result["new_vulnerability_status"] == "detected"


def test_supervisor_preserves_known_new_ids_when_incremental_coverage_is_incomplete():
    group_incomplete = _sast_group("g-incomplete")
    group_scanned = _sast_group("g-scanned")
    incomplete_task = _make_task(
        "task-incomplete",
        group_incomplete.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    scanned_task = _make_task(
        "task-scanned",
        group_scanned.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    scan_record = TaskScanRecord(
        task_id=scanned_task.task_id,
        attempt_id="attempt-scanned",
        task_revision=1,
        found_identifiers=["CVE-2026-70001"],
        new_identifiers=["CVE-2026-70001"],
        complete=True,
    )
    incomplete_result = QAAttemptResult(
        attempt_id="attempt-incomplete",
        task_id=incomplete_task.task_id,
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id=incomplete_task.task_id,
            passed=True,
            scan_evidence=ODCScanEvidence(
                requested_scope=ScanScope.TARGETED,
                effective_scope=ScanScope.TARGETED,
                covered_task_ids=[incomplete_task.task_id],
                complete=False,
            ),
        ),
    )

    result = run_supervisor_node(
        _base_state(
            [group_incomplete, group_scanned],
            task_queue={
                incomplete_task.task_id: incomplete_task,
                scanned_task.task_id: scanned_task,
            },
            qa_results_by_attempt={
                incomplete_result.attempt_id: incomplete_result,
            },
            latest_successful_scan_by_task={
                scanned_task.task_id: scan_record,
            },
            baseline_scan_identifiers=[],
            baseline_scan_complete=True,
            package_inventory_baseline_complete=True,
        )
    )

    summary = result["final_full_scan_result"]
    assert result["new_vulnerability_identifiers"] == ["CVE-2026-70001"]
    assert result["new_vulnerability_status"] == "detected"
    assert summary.coverage_scope == "incremental_dependency_inputs"
    assert summary.coverage_complete is False
    assert summary.new_identifiers == ["CVE-2026-70001"]


def test_absent_target_skip_preserves_cumulative_ids_from_scanned_tasks():
    skipped_group = _sast_group("g-skipped")
    scanned_group = _sast_group("g-scanned")
    skipped_task = _make_task(
        "task-skipped",
        skipped_group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    scanned_task = _make_task(
        "task-scanned",
        scanned_group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    skipped_result = QAAttemptResult(
        attempt_id="attempt-skipped",
        task_id=skipped_task.task_id,
        task_revision=1,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id=skipped_task.task_id,
            passed=True,
            scan_evidence=ODCScanEvidence(
                requested_scope=ScanScope.TARGETED,
                effective_scope=ScanScope.TARGETED,
                covered_task_ids=[skipped_task.task_id],
                scan_performed=False,
                scan_skip_reason="no_matching_target",
                complete=True,
            ),
        ),
    )
    skipped_record = _task_scan_record_from_qa_result(skipped_result)
    assert skipped_record is not None
    assert _incremental_scan_coverage_complete(
        {
            "baseline_scan_complete": True,
            "package_inventory_baseline_complete": True,
            "qa_results_by_attempt": {skipped_result.attempt_id: skipped_result},
            "attempt_snapshots_by_id": {},
        },
        None,
    )

    scanned_record = TaskScanRecord(
        task_id=scanned_task.task_id,
        attempt_id="attempt-scanned",
        task_revision=1,
        found_identifiers=["CVE-2026-70001"],
        new_identifiers=["CVE-2026-70001"],
        complete=True,
    )
    new_ids, found_ids, remaining_ids, _ = _calculate_cumulative_scan_metrics(
        {
            skipped_task.task_id: skipped_task,
            scanned_task.task_id: scanned_task,
        },
        {
            skipped_task.task_id: skipped_record,
            scanned_task.task_id: scanned_record,
        },
        [],
    )

    assert new_ids == ["CVE-2026-70001"]
    assert found_ids == ["CVE-2026-70001"]
    assert remaining_ids == []


def test_supervisor_commits_setup_scan_findings_and_inventory_once():
    group = _sast_group("g-setup")
    task = _make_task(
        "task-setup",
        group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-70002",
            "package_name": "new-package",
            "package_version": "1.0.0",
        }
    )
    occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="new-package",
        version="1.0.0",
        lockfile_key="node_modules/new-package",
        found_identifiers=["CVE-2026-70002"],
        found_issues=[issue],
    )
    occurrence_key = '["package-lock.json","node_modules/new-package","new-package","1.0.0",0]'
    setup_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_occurrences=[occurrence],
        covered_input_paths=["package-lock.json"],
        closure_package_names=["new-package"],
        found_identifiers=["CVE-2026-70002"],
        new_identifiers=["CVE-2026-70002"],
        found_issues=[issue],
        inventory_updates={occurrence_key: "setup-fingerprint"},
        complete=True,
    )
    state = _base_state(
        [group],
        task_queue={task.task_id: task},
        accepted_package_inventory_by_lockfile={},
        current_scan_observations_by_occurrence={},
        baseline_scan_identifiers=["CVE-2024-70000"],
        baseline_scan_complete=True,
        package_inventory_baseline_complete=True,
        setup_scan_evidence=setup_evidence,
    )

    result = run_supervisor_node(state)

    assert result["setup_scan_committed"] is True
    assert result["accepted_package_inventory_by_lockfile"] == {occurrence_key: "setup-fingerprint"}
    assert result["current_scan_observations_by_occurrence"] == {occurrence_key: occurrence}
    assert result["new_vulnerability_identifiers"] == ["CVE-2026-70002"]
    assert result["post_remediation_scan_issues"] == [issue]
    assert state["accepted_package_inventory_by_lockfile"] == {}
    assert state.get("setup_scan_committed", False) is False


def test_setup_scan_findings_persist_until_accepted_task_scan_replaces_them():
    group = _sast_group("g-setup")
    task = _make_task(
        "task-setup",
        group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        status=TaskStatus.QA_PASSED,
    )
    setup_issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-70003",
            "package_name": "new-package",
            "package_version": "1.0.0",
        }
    )
    setup_occurrence = PackageScanOccurrence(
        source_input_path="package-lock.json",
        package_name="new-package",
        version="1.0.0",
        lockfile_key="node_modules/new-package",
        found_identifiers=["CVE-2026-70003"],
        found_issues=[setup_issue],
    )
    occurrence_key = '["package-lock.json","node_modules/new-package","new-package","1.0.0",0]'
    setup_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_occurrences=[setup_occurrence],
        covered_input_paths=["package-lock.json"],
        closure_package_names=["new-package"],
        found_identifiers=["CVE-2026-70003"],
        new_identifiers=["CVE-2026-70003"],
        found_issues=[setup_issue],
        inventory_updates={occurrence_key: "setup-fingerprint"},
        complete=True,
    )
    first_state = _base_state(
        [group],
        task_queue={task.task_id: task},
        accepted_package_inventory_by_lockfile={},
        current_scan_observations_by_occurrence={},
        baseline_scan_identifiers=["CVE-2024-70000"],
        baseline_scan_complete=True,
        package_inventory_baseline_complete=True,
        post_qa_triage_enabled=False,
        setup_scan_evidence=setup_evidence,
    )

    first_result = run_supervisor_node(first_state)

    assert first_result["new_vulnerability_identifiers"] == ["CVE-2026-70003"]
    assert first_result["setup_scan_evidence"] == setup_evidence
    assert first_result["setup_scan_committed"] is True

    clean_occurrence = setup_occurrence.model_copy(
        update={"found_identifiers": [], "found_issues": []}
    )
    instruction = "Apply the selected package workaround."
    attempt_id = "attempt-clean-setup"
    task_revision = 2
    next_task = first_result["task_queue"][task.task_id].model_copy(
        update={
            "task_revision": task_revision,
            "current_attempt_id": attempt_id,
            "status": TaskStatus.OPTIMISTICALLY_FIXED,
            "instruction": instruction,
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        task_revision=task_revision,
        state_revision=first_result["state_revision"] + 1,
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="workaround_subagent",
    )
    clean_evidence = ODCScanEvidence(
        requested_scope=ScanScope.TARGETED,
        effective_scope=ScanScope.TARGETED,
        covered_task_ids=[task.task_id],
        covered_input_paths=["package-lock.json"],
        covered_occurrences=[clean_occurrence],
        inventory_updates={occurrence_key: "clean-fingerprint"},
        complete=True,
    )
    qa_result = QAAttemptResult(
        attempt_id=attempt_id,
        task_id=task.task_id,
        task_revision=task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        qa_policy_source="attempt_snapshot",
        evaluation=QAEvaluation(
            task_id=task.task_id,
            passed=True,
            scan_evidence=clean_evidence,
        ),
    )
    second_state = {
        **first_state,
        **first_result,
        "status": "qa_completed",
        "task_queue": {task.task_id: next_task},
        "active_target_task_ids": [task.task_id],
        "attempt_snapshots_by_id": {
            **first_result["attempt_snapshots_by_id"],
            attempt_id: snapshot,
        },
        "qa_results_by_attempt": {
            **first_result["qa_results_by_attempt"],
            attempt_id: qa_result,
        },
    }

    second_result = run_supervisor_node(second_state)

    assert second_result["setup_scan_evidence"] is None
    assert second_result["new_vulnerability_identifiers"] == []
    assert second_result["new_vulnerability_status"] == "none"
    assert second_result["latest_successful_scan_by_task"][task.task_id].found_identifiers == []


def test_later_clean_task_scan_clears_prior_incremental_vulnerability():
    task = _make_task("task-1", "g1", status=TaskStatus.QA_PASSED)
    issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "cve_id": "CVE-2026-70020",
        }
    )
    vulnerable_record = TaskScanRecord(
        task_id="task-1",
        attempt_id="attempt-vulnerable",
        task_revision=1,
        found_identifiers=["CVE-2026-70020"],
        new_identifiers=["CVE-2026-70020"],
        found_issues=[issue],
        complete=True,
    )
    clean_record = TaskScanRecord(
        task_id="task-1",
        attempt_id="attempt-clean",
        task_revision=2,
        complete=True,
    )

    first_metrics = _calculate_cumulative_scan_metrics(
        {"task-1": task},
        {"task-1": vulnerable_record},
        [],
    )
    later_metrics = _calculate_cumulative_scan_metrics(
        {"task-1": task},
        {"task-1": clean_record},
        [],
    )

    assert first_metrics[0] == ["CVE-2026-70020"]
    assert first_metrics[3] == [issue]
    assert later_metrics[0] == []
    assert later_metrics[1] == []
    assert later_metrics[3] == []


# Stage-2 behavior retained during the architecture merge.
class TestSupervisorDecisionSchemaStage2Parity:
    def test_update_subagent_rejects_more_than_ten_targets(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                next_node="update_subagent",
                target_task_ids=[f"t{i}" for i in range(11)],
                instructions="test",
                decision_reason="test",
            )


class TestRunSupervisorNodeToQAStage2Parity:
    def test_all_optimistically_fixed_tasks_route_to_qa_one_at_a_time(self):
        g1 = _sca_group("g1")
        g2 = _sca_group("g2")
        task1 = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED)
        task2 = _make_task("task-2", "g2", status=TaskStatus.OPTIMISTICALLY_FIXED)
        state = _base_state(
            [g1, g2],
            task_queue={"task-1": task1, "task-2": task2},
            # No active_target_task_ids â€” all tasks are already optimistically fixed
            active_target_task_ids=[],
        )

        result = run_supervisor_node(state)
        assert result["next_routing_step"] == "qa_critic"
        assert result["active_target_task_ids"] == ["task-1"]


class TestRunSupervisorNodeQAUpdatesStage2Parity:
    @pytest.mark.parametrize(
        ("qa_status", "evaluation_kwargs"),
        [
            pytest.param(
                "qa_completed",
                {
                    "failure_category": FailureCategory.SECURITY_FLAG,
                    "retry_feedback": "Dependency evidence was unavailable.",
                    "evidence_inconclusive": True,
                },
                id="inconclusive-evidence",
            ),
            pytest.param(
                "qa_failed",
                {
                    "failure_category": FailureCategory.SECURITY_FLAG,
                    "retry_feedback": "QA could not establish an outcome.",
                    "contract_error": True,
                    "contract_error_reason": "Structured evaluator output was missing.",
                },
                id="qa-contract-error",
            ),
        ],
    )
    def test_qa_recheck_limit_terminalizes_same_attempt(self, qa_status, evaluation_kwargs):
        group = _sca_group("g1")
        attempt_id = "attempt-qa-bounded"
        instruction = "Pin test-pkg through the package manager."
        snapshot = TaskAttemptSnapshot(
            attempt_id=attempt_id,
            task_id="task-1",
            task_revision=1,
            strategy_stage=SCARemediationStage.NPM_LATEST,
            selected_version="1.2.3",
            target_package_name="test-pkg",
            instruction=instruction,
            instruction_digest=instruction_digest(instruction),
            dispatch_node="update_subagent",
            qa_policy=QAPolicy.VERSION_BUMP,
        )
        task = _make_task("task-1", "g1", status=TaskStatus.OPTIMISTICALLY_FIXED).model_copy(
            update={
                "task_revision": 1,
                "current_attempt_id": attempt_id,
                "strategy_stage": snapshot.strategy_stage,
                "selected_version": snapshot.selected_version,
                "target_package_name": snapshot.target_package_name,
                "instruction": instruction,
                "qa_policy": QAPolicy.VERSION_BUMP,
            }
        )
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            **evaluation_kwargs,
        )

        def attempt_result():
            return QAAttemptResult(
                attempt_id=attempt_id,
                task_id="task-1",
                task_revision=1,
                qa_policy=QAPolicy.VERSION_BUMP,
                qa_policy_source="attempt_snapshot",
                evaluation=evaluation,
            )

        initial = _base_state(
            [group],
            status=qa_status,
            task_queue={"task-1": task},
            active_target_task_ids=["task-1"],
            attempt_snapshots_by_id={attempt_id: snapshot},
            qa_results_by_attempt={attempt_id: attempt_result()},
        )

        first = run_supervisor_node(initial)
        assert first["task_queue"]["task-1"].status == TaskStatus.OPTIMISTICALLY_FIXED
        assert first["qa_rechecks_by_attempt"] == {attempt_id: 1}
        assert first["processed_qa_attempt_ids"] == []

        second = run_supervisor_node(
            {
                **initial,
                **first,
                "status": qa_status,
                "qa_results_by_attempt": {attempt_id: attempt_result()},
            }
        )

        terminal = second["task_queue"]["task-1"]
        assert terminal.status == TaskStatus.INCONCLUSIVE
        assert terminal.current_attempt_id is None
        assert terminal.retry_count == 0
        assert second["processed_qa_attempt_ids"] == [attempt_id]
        assert second["next_routing_step"] != "qa_critic"
        assert any("recheck limit reached" in error for error in second["errors"])


class TestBugFixesStage2Parity:
    @patch("langchain_openai.ChatOpenAI")
    def test_replayed_pivot_with_terminal_child_routes_to_final_scan(self, mock_chat):
        """A completed workaround child must not produce an empty worker route."""
        group = _sca_group("g1")
        parent_task = _make_task(
            "task-1",
            "g1",
            strategy=RoutingStrategy.VERSION_BUMP,
            status=TaskStatus.NEEDS_RETRY,
        )
        existing_child = _make_task(
            "task-2",
            "g1",
            strategy=RoutingStrategy.CODE_WORKAROUND,
            status=TaskStatus.UNFIXABLE,
        ).model_copy(update={"parent_task_id": "task-1"})
        state = _base_state(
            [group],
            task_queue={"task-1": parent_task, "task-2": existing_child},
            qa_evaluations={
                "task-1": QAEvaluation(
                    task_id="task-1",
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="The update introduced a breaking runtime regression.",
                )
            },
            active_target_task_ids=["task-1"],
            retry_diagnostics_by_task={
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    exhausted_update_path=True,
                )
            },
            workspace_volume="workspace-volume",
            final_full_scan_completed=False,
        )
        mock_chat.side_effect = ImportError("No module named langchain_openai")

        result = run_supervisor_node(state)

        assert result["task_queue"]["task-1"].status == TaskStatus.PIVOTED
        assert result["task_queue"]["task-2"].status == TaskStatus.UNFIXABLE
        assert result["next_routing_step"] == "final_full_scan"
        assert result["active_target_task_ids"] == []
        assert not any(
            error.startswith("supervisor: routing fell back to teardown")
            for error in result.get("errors", [])
        )


def test_pypi_dispatch_uses_canonical_attempts_and_immutable_snapshot_pool():
    task = _make_task("task-pypi", "pypi-group").model_copy(
        update={
            "strategy_stage": SCARemediationStage.PYPI_SAME_MAJOR,
            "selected_version": "1.2.0",
            "target_package_name": "requests",
            "target_dependency_type": "requirements",
            "instruction": "Update requests to exact version 1.2.0.",
        }
    )
    plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.PYPI_SAME_MAJOR,
        selected_version="1.2.0",
        target_package_name="requests",
        target_dependency_type="requirements",
        candidate_versions_considered=["1.2.0", "1.10.0"],
        candidate_dependency_types=["requirements"],
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        target_package_name="REQUESTS",
        target_dependency_type="requirements",
        attempted_versions_by_target={"ReQuests": ["1.10.0"]},
    )

    authorization = _authorize_update_dispatch(
        task,
        plan=plan,
        diagnostics=diagnostics,
        ecosystem="pypi",
    )
    assert authorization is not None
    assert authorization.selected_version == "1.2.0"
    assert authorization.allowed_target_versions == ("1.2.0",)
    assert authorization.allowed_dependency_types == ("requirements",)

    _updated_task, snapshot = _create_attempt_snapshot(
        task,
        dispatch_node="update_subagent",
        snapshots_by_id={},
        state_revision=9,
        allowed_target_versions=authorization.allowed_target_versions,
        allowed_dependency_types=authorization.allowed_dependency_types,
    )
    assert snapshot.allowed_target_versions == ["1.2.0"]
    assert snapshot.selected_version == "1.2.0"
    assert snapshot.target_package_name == "requests"
