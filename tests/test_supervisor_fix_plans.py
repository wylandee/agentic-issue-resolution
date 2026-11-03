"""Focused tests for supervisor fix plans."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

from semantic_version import Version

from remediation_engine import cli
from remediation_engine.agents import supervisor as supervisor_node
from remediation_engine.agents import supervisor_reasoning as supervisor_spawn
from remediation_engine.agents.supervisor import (
    MAX_RETRIES,
    _build_high_level_retry_instruction,
    _next_sca_stage,
    _run_deterministic_retry_planner,
    run_supervisor_node,
)
from remediation_engine.agents.supervisor_reasoning import (
    _plan_initial_transitive_task,
    ensure_group_fix_plans,
    ensure_workaround_fix_plan,
    plan_fixes_for_groups,
    prepare_supervisor_fix_plan_state,
)
from remediation_engine.agents.triage import group_issues
from remediation_engine.agents.update import (
    _UPDATE_WORKER_STATIC_INSTRUCTIONS,
    _build_retry_diagnostics,
    _build_update_prompt,
)
from remediation_engine.contracts import (
    QAAttemptResult,
    QAPolicy,
    SCARemediationStage,
    TaskAttemptSnapshot,
    UpdateRetryDiagnostics,
)
from remediation_engine.contracts.schemas import (
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    PackageFixPlanCandidate,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskSpawnRequest,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import SolverStatus
from remediation_engine.contracts.version_policy import select_version
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.state import (
    build_initial_remediation_task,
    instruction_digest,
)
from remediation_engine.planning import control as supervisor_portfolio
from remediation_engine.planning.portfolio import build_portfolio_plan, prepare_portfolio_inputs
from remediation_engine.runtime.agent_runtime import ToolEvent
from remediation_engine.settings import AppSettings
from remediation_engine.tools.advisories import (
    _extract_fixed_from_osv_vuln,
    _query_osv_fixed_version,
    plan_fix,
)
from remediation_engine.tools.registry import (
    _requirement_matches,
    fetch_registry_candidates,
    plan_npm_parent_version,
    select_npm_parent_version,
)
from tests.helpers import group_with_fix_plans


def _issue(cve_id: str = "CVE-2026-1000") -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        package_name="demo-package",
        ecosystem="npm",
        package_version="1.0.0",
        cve_id=cve_id,
        severity=Severity.HIGH,
        file_path="package-lock.json",
    )


def _localized(issue: VulnerabilityIssue) -> LocalizedIssue:
    return LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=True,
        declaration_type="dependencies",
    )


def _plan(
    status: FixPlanStatus,
    *,
    version: str | None = None,
    snippets: list[str] | None = None,
) -> FixPlan:
    return FixPlan(
        status=status,
        fixed_version=version,
        workaround_snippets=snippets,
        instruction="Apply the confirmed remediation evidence.",
        strategy_used="test",
    )


def _group(*issues: VulnerabilityIssue) -> VulnerabilityGroup:
    return group_issues(
        list(issues),
        localized_sca_issues=[_localized(issue) for issue in issues],
    )[0]


def test_grouping_emits_unplanned_sca_observation_groups() -> None:
    group = _group(_issue())
    assert group.fix_plan is None
    assert group.fix_plan_candidates == []
    assert group.localized_issues[0].manifest_file == "package.json"


def test_offline_portfolio_reports_unplanned_sca_without_creating_tasks(tmp_path) -> None:
    group = _group(_issue())

    prepared_groups, queue, diagnostics = prepare_portfolio_inputs(
        tmp_path,
        [group],
        {},
    )
    plan = build_portfolio_plan(tmp_path, prepared_groups, queue)

    assert queue == {}
    assert prepared_groups[0].fix_plan is None
    assert len(diagnostics) == 1
    assert "missing Supervisor fix plans" in diagnostics[0]
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.clusters
    assert all(not cluster.dispatchable for cluster in plan.clusters)
    assert diagnostics[0] in plan.diagnostics


def test_offline_solve_emits_diagnostic_and_exits_nonzero_for_unplanned_sca(
    monkeypatch, tmp_path
) -> None:
    group = _group(_issue())
    output: dict[str, object] = {}
    monkeypatch.setattr(cli, "_load_solve_inputs", lambda *_args: ([group], {}))
    monkeypatch.setattr(
        cli,
        "_write_json",
        lambda _path, payload, **_kwargs: output.update(payload),
    )

    status = cli._run_solve(
        Namespace(
            input=None,
            groups_input=tmp_path / "groups.json",
            tasks_input=None,
            repo=tmp_path,
            output=tmp_path / "result.json",
        ),
        AppSettings(),
    )

    assert status == 1
    assert output["solver_status"] == SolverStatus.UNKNOWN.value
    assert any("missing Supervisor fix plans" in item for item in output["diagnostics"])
    assert output["dispatchable"] is False
    assert output["task_decisions"] == []


def test_planning_keeps_one_candidate_per_finding() -> None:
    issues = [_issue("CVE-2026-1001"), _issue("CVE-2026-1002")]
    group = _group(*issues)
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    planned = ensure_group_fix_plans(group, lookup=lookup)

    assert lookup.call_count == 2
    assert {candidate.issue_id for candidate in planned.fix_plan_candidates} == {
        issue.id for issue in issues
    }
    assert planned.fix_plan is not None
    assert planned.fix_plan.fixed_version == "1.2.0"
    assert group.fix_plan is None


def test_batch_cache_reuses_equivalent_lookup_and_skips_existing_or_out_of_scope_groups() -> None:
    issue = _issue()
    group = _group(issue)
    duplicate_observation = group.model_copy(update={"group_id": "same-input-other-group"})
    planned_group = group.model_copy(
        update={"fix_plan": _plan(FixPlanStatus.NO_FIX), "fix_plan_candidates": []}
    )
    sast_group = VulnerabilityGroup(
        group_id="sast:test",
        issue_type=IssueType.SAST,
        vulnerable_component="rule",
        representative_issue_id=uuid4(),
        issues=[],
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    result = plan_fixes_for_groups(
        [group, duplicate_observation, planned_group, sast_group],
        lookup=lookup,
    )

    assert lookup.call_count == 1
    assert result[0].fix_plan is not None
    assert result[1].fix_plan is not None
    assert result[2] is planned_group
    assert result[3] is sast_group
    again = plan_fixes_for_groups(result, lookup=lookup)
    assert again[0] is result[0]
    assert lookup.call_count == 1


def test_targeted_workaround_lookup_preserves_version_floor_and_uses_skip_osv() -> None:
    issue = _issue()
    group = _group(issue)
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    group = group.model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [
                PackageFixPlanCandidate(issue_id=issue.id, plan=version_plan),
            ],
        }
    )
    lookup = Mock(
        return_value=_plan(
            FixPlanStatus.WORKAROUND_FOUND,
            snippets=["Disable the affected option."],
        )
    )

    enriched = ensure_workaround_fix_plan(group, lookup=lookup)

    assert lookup.call_count == 1
    assert lookup.call_args.kwargs == {"skip_osv": True}
    assert enriched.fix_plan is not None
    assert enriched.fix_plan.fixed_version == "1.2.0"
    assert any(
        candidate.plan.status == FixPlanStatus.VERSION_FOUND
        for candidate in enriched.fix_plan_candidates
    )
    assert any(
        candidate.plan.status == FixPlanStatus.WORKAROUND_FOUND
        for candidate in enriched.fix_plan_candidates
    )


def test_workaround_no_fix_result_does_not_replace_existing_candidates() -> None:
    issue = _issue()
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    group = _group(issue).model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [PackageFixPlanCandidate(issue_id=issue.id, plan=version_plan)],
        }
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.NO_FIX))

    result = ensure_workaround_fix_plan(group, lookup=lookup)

    assert result is group
    assert len(result.fix_plan_candidates) == 1
    assert result.fix_plan.fixed_version == "1.2.0"


def test_summary_only_workaround_evidence_is_reused_without_another_lookup() -> None:
    issue = _issue()
    group = _group(issue).model_copy(
        update={
            "fix_plan": _plan(
                FixPlanStatus.WORKAROUND_FOUND,
                snippets=["Disable the affected option."],
            ),
            "fix_plan_candidates": [],
        }
    )
    lookup = Mock(side_effect=AssertionError("existing workaround should be reused"))

    result = ensure_workaround_fix_plan(group, lookup=lookup)

    assert result is group
    lookup.assert_not_called()


def test_lookup_and_response_validation_errors_use_typed_no_fix_fallback(caplog) -> None:
    group = _group(_issue())

    failed_call = ensure_group_fix_plans(
        group,
        lookup=Mock(side_effect=RuntimeError("service unavailable")),
    )
    invalid_response = ensure_group_fix_plans(
        group,
        lookup=Mock(return_value={"status": "not-a-plan"}),
    )

    assert failed_call.fix_plan is not None
    assert failed_call.fix_plan.status == FixPlanStatus.NO_FIX
    assert failed_call.fix_plan.strategy_used == "NO_FIX"
    assert failed_call.fix_plan_candidates[0].plan.strategy_used == "planner_error"
    assert invalid_response.fix_plan is not None
    assert invalid_response.fix_plan.status == FixPlanStatus.NO_FIX
    assert "Supervisor fix planning failed" in caplog.text


def test_portfolio_entry_plans_groups_before_preparing_solver_inputs(monkeypatch, tmp_path) -> None:
    group = _group(_issue())
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_reasoning.plan_fix",
        lookup,
    )

    def stop_before_solver(repo_root, groups, task_queue, **kwargs):
        observed["group"] = groups[0]
        observed["queue"] = task_queue
        raise RuntimeError("test stopped after portfolio preparation boundary")

    monkeypatch.setattr(supervisor_portfolio, "prepare_portfolio_inputs", stop_before_solver)
    result = supervisor_portfolio.run_portfolio_plan_tool(
        {
            "repo_root": str(tmp_path),
            "valid_groups": [group],
            "task_queue": {},
        }
    )

    assert lookup.call_count == 1
    assert observed["group"].fix_plan.status == FixPlanStatus.VERSION_FOUND
    assert result["valid_groups"][0].fix_plan_candidates
    assert result["task_queue"] == {}


def test_direct_supervisor_entry_plans_before_validating_portfolio(monkeypatch, tmp_path) -> None:
    group = _group(_issue())
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))
    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_reasoning.plan_fix",
        lookup,
    )

    result = supervisor_node._run_supervisor_decision_node(
        {
            "repo_root": str(tmp_path),
            "valid_groups": [group],
            "task_queue": {},
            "status": "supervisor_entered",
        }
    )

    assert result["next_routing_step"] == "portfolio"
    assert result["valid_groups"][0].fix_plan_candidates
    lookup.assert_called_once()


def test_accepted_workaround_child_gets_targeted_evidence_before_selection(monkeypatch) -> None:
    first_issue = _issue("CVE-2026-1001")
    second_issue = _issue("CVE-2026-1002")
    group = _group(first_issue, second_issue)
    version_plan = _plan(FixPlanStatus.VERSION_FOUND, version="1.2.0")
    workaround_plan = _plan(
        FixPlanStatus.WORKAROUND_FOUND,
        snippets=["Disable the affected option."],
    )
    enriched = group.model_copy(
        update={
            "fix_plan": version_plan,
            "fix_plan_candidates": [
                PackageFixPlanCandidate(issue_id=first_issue.id, plan=version_plan),
                PackageFixPlanCandidate(issue_id=second_issue.id, plan=workaround_plan),
            ],
        }
    )
    targeted_lookup = Mock(return_value=enriched)
    monkeypatch.setattr(supervisor_spawn, "ensure_workaround_fix_plan", targeted_lookup)
    parent = RemediationTask(
        task_id="task-parent",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
    )
    request = TaskSpawnRequest(
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the workaround.",
        reason="The update path is exhausted.",
    )
    groups = [group]
    ledger: dict[str, str] = {}

    new_tasks, child_ids = supervisor_spawn._materialize_spawn_requests(
        [request],
        {parent.task_id: parent},
        {group.group_id: group},
        [],
        valid_groups=groups,
        qa_evaluations={
            parent.task_id: QAEvaluation(
                task_id=parent.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="The target remains vulnerable.",
            )
        },
        workaround_lookup_fingerprints=ledger,
    )

    child = new_tasks[child_ids[parent.task_id][0]]
    targeted_lookup.assert_called_once_with(group)
    assert groups[0] is enriched
    assert child.selected_plan_issue_ids == [str(second_issue.id)]
    assert len(ledger) == 1


def test_rejected_or_duplicate_workaround_spawns_do_not_lookup(monkeypatch) -> None:
    issue = _issue()
    group = _group(issue)
    parent = RemediationTask(
        task_id="task-parent",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
    )
    existing_child = RemediationTask(
        task_id="task-child",
        parent_group_id=group.group_id,
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
    )
    request = TaskSpawnRequest(
        parent_task_id=parent.task_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the workaround.",
        reason="Retry the same pivot.",
    )
    targeted_lookup = Mock(side_effect=AssertionError("lookup should not run"))
    monkeypatch.setattr(supervisor_spawn, "ensure_workaround_fix_plan", targeted_lookup)

    new_tasks, child_ids = supervisor_spawn._materialize_spawn_requests(
        [request],
        {parent.task_id: parent, existing_child.task_id: existing_child},
        {group.group_id: group},
        [],
        qa_evaluations={
            parent.task_id: QAEvaluation(
                task_id=parent.task_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="The target remains vulnerable.",
            )
        },
        workaround_lookup_fingerprints={},
    )

    assert new_tasks == {}
    assert child_ids[parent.task_id] == [existing_child.task_id]
    targeted_lookup.assert_not_called()


def test_reopened_task_refresh_preserves_identity_attempt_history_and_revision() -> None:
    issue = _issue()
    group = _group(issue)
    task = build_initial_remediation_task(group, "task-7")
    task = task.model_copy(
        update={
            "task_revision": 4,
            "status": TaskStatus.PENDING,
            "parent_task_id": "task-parent",
            "ancestry_depth": 1,
        }
    )
    lookup = Mock(return_value=_plan(FixPlanStatus.VERSION_FOUND, version="1.2.0"))

    planned_groups, queue, cleared = prepare_supervisor_fix_plan_state(
        [group],
        {task.task_id: task},
        [task.task_id],
        ProjectLanguage.NODEJS,
        lookup=lookup,
    )

    refreshed = queue[task.task_id]
    assert planned_groups[0].fix_plan is not None
    assert refreshed.task_id == task.task_id
    assert refreshed.task_revision == 4
    assert refreshed.parent_task_id == "task-parent"
    assert refreshed.ancestry_depth == 1
    assert refreshed.strategy == RoutingStrategy.VERSION_BUMP
    assert refreshed.selected_version is None
    assert refreshed.selected_plan_issue_ids == [str(issue.id)]
    assert cleared == []
    lookup.assert_called_once()


def test_unmatched_refresh_marker_is_retained_until_a_group_is_available() -> None:
    task = RemediationTask(
        task_id="task-orphaned",
        parent_group_id="sca:manifest:missing",
        strategy=RoutingStrategy.VERSION_BUMP,
    )

    planned_groups, queue, pending = prepare_supervisor_fix_plan_state(
        [],
        {task.task_id: task},
        [task.task_id],
        ProjectLanguage.NODEJS,
    )

    assert planned_groups == []
    assert queue[task.task_id] is not task
    assert pending == [task.task_id]


def _sca_strategy_issue(cve: str, *, package: str = "lodash") -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=package,
        package_version="4.17.20",
        purl=f"pkg:npm/{package}@4.17.20",
        cve_id=cve,
        ecosystem="npm",
        file_path="package.json",
    )


def _sca_strategy_localized(issue: VulnerabilityIssue) -> LocalizedIssue:
    return LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )


def test_registry_parent_requirement_accepts_operator_whitespace_without_policy_change():
    """Operator-space normalization agrees with manifest range handling."""
    candidate = Version("1.2.4")
    assert _requirement_matches("^ 1.2.3", candidate)
    assert _requirement_matches(">= 1.2.3 < 2.0.0", candidate)
    assert _requirement_matches("~ 1.2.3", candidate)

    assert _requirement_matches("*", candidate)
    assert not _requirement_matches(None, candidate)
    assert not _requirement_matches("", candidate)
    assert not _requirement_matches("latest", candidate)
    assert not _requirement_matches("not a range", candidate)
    assert not _requirement_matches("^1.2.3", Version("1.3.0-beta.1"))


def _sca_strategy_plan(
    status: FixPlanStatus,
    *,
    version: str | None = None,
    snippets: list[str] | None = None,
) -> FixPlan:
    return FixPlan(
        status=status,
        fixed_version=version,
        workaround_snippets=snippets,
        instruction="test",
        strategy_used="osv_api",
    )


def test_osv_extracts_the_minimum_fixed_version_for_one_advisory():
    vuln = {
        "affected": [
            {
                "package": {"name": "lodash", "ecosystem": "npm"},
                "ranges": [
                    {
                        "type": "SEMVER",
                        "events": [
                            {"introduced": "0"},
                            {"fixed": "4.17.21"},
                            {"fixed": "4.18.0"},
                        ],
                    }
                ],
            }
        ]
    }

    fixed, snippets = _extract_fixed_from_osv_vuln(vuln, "lodash")

    assert fixed == "4.17.21"
    assert snippets is None


def test_plan_fix_is_osv_only_and_does_not_fall_through_to_npm_or_serper():
    issue = _sca_strategy_issue("CVE-2026-0001")
    localized = _sca_strategy_localized(issue)
    with (
        patch(
            "remediation_engine.tools.advisories._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ) as osv,
        patch("remediation_engine.tools.advisories._serper_search_and_extract") as serper,
    ):
        result = plan_fix(localized)

    osv.assert_called_once_with(issue)
    serper.assert_not_called()
    assert result["fixed_version"] == "4.17.21"
    assert result["strategy_used"] == "osv_api"


def test_osv_query_uses_the_finding_advisory_when_batch_has_multiple_cves():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "results": [
            {
                "vulns": [
                    {
                        "id": "GHSA-target",
                        "aliases": ["CVE-2026-0001"],
                        "affected": [
                            {
                                "package": {"name": "lodash"},
                                "ranges": [
                                    {
                                        "type": "SEMVER",
                                        "events": [{"fixed": "4.17.21"}],
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "id": "GHSA-other",
                        "aliases": ["CVE-2026-0002"],
                        "affected": [
                            {
                                "package": {"name": "lodash"},
                                "ranges": [
                                    {
                                        "type": "SEMVER",
                                        "events": [{"fixed": "9.0.0"}],
                                    }
                                ],
                            }
                        ],
                    },
                ]
            }
        ]
    }
    with patch("remediation_engine.tools.advisories.requests.post", return_value=response):
        fixed, snippets = _query_osv_fixed_version(_sca_strategy_issue("CVE-2026-0001"))

    assert fixed == "4.17.21"
    assert snippets is None


def test_same_package_findings_share_one_package_group_and_retain_candidates():
    update_a = _sca_strategy_issue("CVE-2026-0001")
    update_b = _sca_strategy_issue("CVE-2026-0002")
    workaround = _sca_strategy_issue("CVE-2026-0003")
    no_fix = _sca_strategy_issue("CVE-2026-0004")
    groups = group_with_fix_plans(
        [update_a, update_b, workaround, no_fix],
        plans_by_finding=[
            (
                _sca_strategy_localized(update_a),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            ),
            (
                _sca_strategy_localized(update_b),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.18.0"),
            ),
            (
                _sca_strategy_localized(workaround),
                _sca_strategy_plan(
                    FixPlanStatus.WORKAROUND_FOUND,
                    snippets=["disable the parser"],
                ),
            ),
            (_sca_strategy_localized(no_fix), _sca_strategy_plan(FixPlanStatus.NO_FIX)),
        ],
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.group_id == "sca:package.json:lodash"
    assert group.fix_plan is not None
    assert group.fix_plan.fixed_version == "4.18.0"
    assert len(group.localized_issues) == 4
    assert {candidate.plan.status for candidate in group.fix_plan_candidates} == {
        FixPlanStatus.VERSION_FOUND,
        FixPlanStatus.WORKAROUND_FOUND,
        FixPlanStatus.NO_FIX,
    }


def test_supervisor_typed_registry_candidates_select_same_major_then_latest(monkeypatch):
    registry_data = {
        "versions": {
            "4.17.21": {},
            "4.18.0": {},
            "5.0.0": {},
            "6.1.0-beta.1": {},
        }
    }
    monkeypatch.setattr(
        "remediation_engine.tools.registry._fetch_package_data",
        lambda package: registry_data,
    )

    candidates = fetch_registry_candidates(
        "lodash",
        "4.17.21",
        {"4.18.0"},
    )
    same_major = select_version(
        candidates,
        SCARemediationStage.NPM_SAME_MAJOR,
        {"4.18.0"},
    )
    latest = select_version(
        candidates,
        SCARemediationStage.NPM_LATEST,
        {"4.18.0"},
    )

    assert same_major == "4.17.21"
    assert latest == "5.0.0"
    assert all("beta" not in candidate.version for candidate in candidates)


def test_supervisor_typed_registry_candidates_share_latest_when_same_major_is_latest(
    monkeypatch,
):
    monkeypatch.setattr(
        "remediation_engine.tools.registry._fetch_package_data",
        lambda package: {"versions": {"4.18.0": {}, "4.17.21": {}}},
    )

    candidates = fetch_registry_candidates("lodash", "4.17.21")
    same_major = select_version(candidates, SCARemediationStage.NPM_SAME_MAJOR, set())
    latest = select_version(candidates, SCARemediationStage.NPM_LATEST, set())

    assert same_major == "4.18.0"
    assert latest == "4.18.0"


def test_registry_candidate_surface_contains_only_three_strategic_slots(monkeypatch):
    monkeypatch.setattr(
        "remediation_engine.tools.registry._fetch_package_data",
        lambda package: {
            "versions": {
                "1.0.0": {},
                "1.1.0": {},
                "1.2.0": {},
                "2.0.0": {},
                "3.0.0": {},
            },
            "dist-tags": {"latest": "3.0.0"},
        },
    )

    candidates = fetch_registry_candidates("pkg", "1.0.0")

    assert [candidate.version for candidate in candidates] == ["1.0.0", "1.2.0", "3.0.0"]
    assert [candidate.selection_roles for candidate in candidates] == [
        ("osv_minimum",),
        ("same_major",),
        ("npm_latest",),
    ]


def test_parent_selector_orders_stable_compatible_releases_by_stage():
    data = {
        "versions": {
            "1.0.1": {"dependencies": {"transitive-child": "^2.0.0"}},
            "1.1.0": {"dependencies": {"transitive-child": "^2.0.0"}},
            "2.0.0": {"dependencies": {"transitive-child": "^2.0.0"}},
            "2.1.0-beta.1": {"dependencies": {"transitive-child": "^2.0.0"}},
            "2.2.0": {"dependencies": {"transitive-child": "^1.0.0"}},
        }
    }

    minimum = select_npm_parent_version(
        data,
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="minimum",
    )
    same_major = select_npm_parent_version(
        data,
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="same_major",
    )
    latest = select_npm_parent_version(
        data,
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="latest",
    )

    assert minimum["selected"] == "1.0.1"
    assert same_major["selected"] == "1.1.0"
    assert latest["selected"] == "2.0.0"
    assert "2.1.0-beta.1" not in latest["compatible"]


def test_parent_selector_resolves_multi_hop_dependency_ancestry():
    parent_data = {
        "versions": {
            "1.0.1": {"dependencies": {"intermediate": "~1.0.0"}},
            "1.1.0": {"dependencies": {"intermediate": "^1.1.0"}},
            "2.0.0": {"dependencies": {"intermediate": "^1.1.0"}},
            "2.1.0-beta.1": {"dependencies": {"intermediate": "^1.1.0"}},
        }
    }
    intermediate_data = {
        "versions": {
            "1.0.0": {"dependencies": {"transitive-child": "^1.0.0"}},
            "1.1.0": {"dependencies": {"transitive-child": "^2.0.0"}},
            "1.2.0": {"dependencies": {"transitive-child": "^3.0.0"}},
        }
    }

    minimum = select_npm_parent_version(
        parent_data,
        parent_package_name="direct-parent",
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="minimum",
        dependency_ancestry=["direct-parent", "intermediate", "transitive-child"],
        registry_data_by_package={"intermediate": intermediate_data},
    )
    same_major = select_npm_parent_version(
        parent_data,
        parent_package_name="direct-parent",
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="same_major",
        dependency_ancestry=["direct-parent", "intermediate", "transitive-child"],
        registry_data_by_package={"intermediate": intermediate_data},
    )
    latest = select_npm_parent_version(
        parent_data,
        parent_package_name="direct-parent",
        child_package_name="transitive-child",
        child_fixed_version="2.0.1",
        installed_parent_version="1.0.0",
        selection="latest",
        dependency_ancestry=["direct-parent", "intermediate", "transitive-child"],
        registry_data_by_package={"intermediate": intermediate_data},
    )

    assert minimum["selected"] == "1.1.0"
    assert same_major["selected"] == "1.1.0"
    assert latest["selected"] == "2.0.0"
    assert latest["compatible"] == ["1.1.0", "2.0.0"]
    assert "2.1.0-beta.1" not in latest["compatible"]


def test_parent_planner_fetches_intermediate_registry_metadata(monkeypatch):
    registry_data = {
        "direct-parent": {
            "versions": {
                "1.0.1": {"dependencies": {"intermediate": "^1.1.0"}},
            }
        },
        "intermediate": {
            "versions": {
                "1.1.0": {"dependencies": {"@tootallnate/once": "^2.0.0"}},
            }
        },
    }
    fetched: list[str] = []

    def fetch(package_name: str):
        fetched.append(package_name)
        return registry_data[package_name]

    monkeypatch.setattr("remediation_engine.tools.registry._fetch_package_data", fetch)

    report = plan_npm_parent_version.invoke(
        {
            "parent_package_name": "direct-parent",
            "child_package_name": "@tootallnate/once",
            "child_fixed_version": "2.0.1",
            "installed_parent_version": "1.0.0",
            "selection": "minimum",
            "dependency_ancestry": "direct-parent -> intermediate -> @tootallnate/once",
        }
    )

    assert fetched == ["direct-parent", "intermediate"]
    assert "Selected Version: 1.0.1" in report
    assert "Dependency Ancestry: direct-parent -> intermediate -> @tootallnate/once" in report


def test_transitive_initial_task_targets_parent_before_override():
    issue = _sca_strategy_issue("CVE-2026-0002", package="transitive-child")
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=False,
        dependency_ancestry=["direct-parent", "transitive-child"],
        dependency_versions={"direct-parent": "1.0.0", "transitive-child": "1.0.0"},
        parent_package_name="direct-parent",
        parent_package_version="1.0.0",
        parent_declaration_type="dependencies",
        localization_confidence=1.0,
    )
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (localized, _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="2.0.1"))
        ],
    )[0]

    task = build_initial_remediation_task(group, "task-1")

    assert task.strategy_stage == SCARemediationStage.OSV_MINIMUM
    assert task.target_package_name == "direct-parent"
    assert task.target_dependency_type == "dependencies"
    assert task.selected_version is None
    assert "direct-parent" in task.instruction
    assert "overrides" not in task.instruction


def test_transitive_initial_planner_projects_only_committed_parent_candidate():
    issue = _sca_strategy_issue("CVE-2026-0003", package="transitive-child")
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=False,
        dependency_ancestry=["direct-parent", "transitive-child"],
        dependency_versions={"direct-parent": "1.0.0", "transitive-child": "1.0.0"},
        parent_package_name="direct-parent",
        parent_declaration_type="dependencies",
        localization_confidence=1.0,
    )
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (localized, _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="2.0.1"))
        ],
    )[0]
    task = build_initial_remediation_task(group, "task-1")
    task = task.model_copy(update={"allowed_target_versions": ["1.0.1"]})

    with patch("remediation_engine.tools.registry._fetch_package_data") as fetch_package_data:
        planned = _plan_initial_transitive_task(task, group)

    assert planned.parent_package_version == "1.0.0"
    assert planned.strategy_stage == SCARemediationStage.OSV_MINIMUM
    assert planned.target_package_name == "direct-parent"
    assert planned.selected_version == "1.0.1"
    fetch_package_data.assert_not_called()


def test_transitive_stage_progression_reaches_override_before_workaround():
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST, transitive=True) == (
        SCARemediationStage.PACKAGE_OVERRIDE
    )
    assert _next_sca_stage(SCARemediationStage.PACKAGE_OVERRIDE, transitive=True) == (
        SCARemediationStage.CODE_WORKAROUND
    )


def test_update_worker_prompt_contains_no_planning_or_registry_phase():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="g1",
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        instruction="Update package.json to exact version 4.18.0.",
    )
    issue = _sca_strategy_issue("CVE-2026-0001")
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (
                _sca_strategy_localized(issue),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            )
        ],
    )[0]
    prompt = _build_update_prompt(
        [(task, group, ["package.json"])],
        [],
        {},
        {},
        {},
    )

    assert "execution worker" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
    assert "exact version 4.18.0" in prompt
    assert "Planning Answers" not in prompt
    assert "per package" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
    assert "package identity and manifest" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
    assert "task allowlists" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
    assert "exactly once for the final batch state" not in prompt


def test_update_worker_records_failed_exact_version_attempts_without_selecting_next_version():
    issue = _sca_strategy_issue("CVE-2026-0001")
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (
                _sca_strategy_localized(issue),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            )
        ],
    )[0]
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        status=TaskStatus.NEEDS_RETRY,
    )
    diagnostics = _build_retry_diagnostics(
        [(task, group, ["package.json"])],
        [
            ToolEvent(
                name="modify_and_validate_npm_dependency",
                args={
                    "package_name": group.vulnerable_component,
                    "target_version": "5.0.0",
                    "dependency_type": "dependencies",
                },
                content="ERROR_CODE: MANIFEST_SYNC_FAILED: package install failed",
            )
        ],
        "surrender",
        [],
        False,
        {},
        [],
    )[task.task_id]

    assert diagnostics.attempted_versions == ["5.0.0"]
    assert diagnostics.selected_version is None


def test_java_initial_task_uses_exact_gav_and_only_maven_declaration_targets():
    tasks = []
    for declaration_type in ("dependencies", "dependencyManagement"):
        issue = VulnerabilityIssue(
            source=IssueSource.ODC,
            issue_type=IssueType.SCA,
            severity=Severity.HIGH,
            package_name="org.example:library",
            package_version="1.0.0",
            purl="pkg:maven/org.example/library@1.0.0",
            ecosystem="Maven",
            cve_id="CVE-2026-0002",
            file_path="modules/service/pom.xml",
        )
        localized = LocalizedIssue(
            issue=issue,
            manifest_file="modules/service/pom.xml",
            package_manager="maven",
            is_direct_dependency=declaration_type == "dependencies",
            declaration_type=declaration_type,
            localization_confidence=1.0,
        )
        group = VulnerabilityGroup(
            group_id=f"maven:{declaration_type}:org.example:library",
            issue_type=IssueType.SCA,
            vulnerable_component="org.example:library",
            file_path="modules/service/pom.xml",
            file_paths=["modules/service/pom.xml"],
            parent_package_name="org.example:parent",
            parent_package_version="1.0.0",
            parent_declaration_type="dependencies",
            dependency_ancestry=["org.example:parent", "org.example:library"],
            representative_issue_id=issue.id,
            issues=[issue],
            localized_issues=[localized],
            fix_plan=_sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="2.4.1"),
        )
        tasks.append(
            build_initial_remediation_task(
                group,
                f"task-{declaration_type}",
                project_language=ProjectLanguage.JAVA,
            )
        )

    assert [task.target_package_name for task in tasks] == [
        "org.example:library",
        "org.example:library",
    ]
    assert [task.target_dependency_type for task in tasks] == [
        "dependencies",
        "dependencyManagement",
    ]
    assert all(task.strategy_stage == SCARemediationStage.OSV_MINIMUM for task in tasks)

    assert all(task.parent_package_name is None for task in tasks)


def test_supervisor_stage_progression_and_retry_cap():
    """Retry policy allows three attempts before terminal handling."""
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM) == SCARemediationStage.NPM_SAME_MAJOR
    assert _next_sca_stage(SCARemediationStage.NPM_SAME_MAJOR) == SCARemediationStage.NPM_LATEST
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST) == SCARemediationStage.CODE_WORKAROUND
    assert MAX_RETRIES == 3


def test_qa_failure_selects_next_committed_candidate_and_preserves_security_floor():
    issue = _sca_strategy_issue("CVE-2026-0001")
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (
                _sca_strategy_localized(issue),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            )
        ],
    )[0]
    instruction = "Update lodash to the committed security floor."
    attempt_id = "attempt-task-1"
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        task_revision=1,
        current_attempt_id=attempt_id,
        qa_policy=QAPolicy.VERSION_BUMP,
        target_package_name="lodash",
        selected_version="4.17.21",
        allowed_target_versions=["4.17.21", "4.18.0"],
        instruction=instruction,
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        qa_policy=QAPolicy.VERSION_BUMP,
        strategy_stage=task.strategy_stage,
        selected_version="4.17.21",
        allowed_target_versions=["4.17.21", "4.18.0"],
        target_package_name="lodash",
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
    )
    evaluation = QAEvaluation(
        task_id=task.task_id,
        passed=False,
        failure_category=FailureCategory.SECURITY_FLAG,
        retry_feedback="the OSV version did not clear the finding",
    )
    result = run_supervisor_node(
        {
            "valid_groups": [group],
            "task_queue": {task.task_id: task},
            "status": "qa_completed",
            "active_target_task_ids": [task.task_id],
            "attempt_snapshots_by_id": {attempt_id: snapshot},
            "retry_diagnostics_by_task": {
                task.task_id: UpdateRetryDiagnostics(
                    task_id=task.task_id,
                    strategy_stage=task.strategy_stage,
                    committed_attempt_id=attempt_id,
                    security_floor="4.17.21",
                    target_package_name="lodash",
                    selected_version="4.17.21",
                    attempted_versions=["4.17.21"],
                    attempted_versions_by_target={"lodash": ["4.17.21"]},
                    candidate_versions_considered=["4.17.21", "4.18.0"],
                    latest_version_seen="4.18.0",
                    registry_query_performed=True,
                )
            },
            "qa_evaluations": {task.task_id: evaluation},
            "qa_results_by_attempt": {
                attempt_id: QAAttemptResult(
                    attempt_id=attempt_id,
                    task_id=task.task_id,
                    task_revision=task.task_revision,
                    qa_policy=QAPolicy.VERSION_BUMP,
                    qa_policy_source="attempt_snapshot",
                    evaluation=evaluation,
                )
            },
        }
    )

    updated_task = result["task_queue"][task.task_id]
    diagnostics = result["retry_diagnostics_by_task"][task.task_id]
    assert updated_task.strategy_stage == SCARemediationStage.NPM_SAME_MAJOR
    assert updated_task.status == TaskStatus.NEEDS_RETRY
    assert updated_task.selected_version == "4.18.0"
    assert diagnostics.strategy_stage == SCARemediationStage.NPM_SAME_MAJOR
    assert diagnostics.attempted_versions == ["4.17.21"]
    assert diagnostics.security_floor == "4.17.21"


def test_exhausted_sca_candidates_do_not_reopen_the_attempted_version():
    issue = _sca_strategy_issue("CVE-2026-0004")
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (
                _sca_strategy_localized(issue),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            )
        ],
    )[0]
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_SAME_MAJOR,
        status=TaskStatus.NEEDS_RETRY,
        target_package_name="lodash",
        allowed_target_versions=["4.17.21"],
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        strategy_stage=task.strategy_stage,
        security_floor="4.17.21",
        target_package_name="lodash",
        attempted_versions=["4.17.21"],
        candidate_versions_considered=["4.17.21"],
    )

    updated, plans = _run_deterministic_retry_planner(
        {task.task_id: task},
        {group.group_id: group},
        {task.task_id: diagnostics},
    )

    plan = plans[task.task_id]
    assert plan.selected_version is None
    assert plan.exhausted_update_path
    assert plan.action == "pivot_workaround"
    assert updated[task.task_id].attempted_versions == ["4.17.21"]
    assert updated[task.task_id].security_floor == "4.17.21"


def test_retry_instruction_contains_exact_selected_version_and_preserves_authority():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="g1",
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        status=TaskStatus.NEEDS_RETRY,
    )
    issue = _sca_strategy_issue("CVE-2026-0001")
    group = group_with_fix_plans(
        [issue],
        plans_by_finding=[
            (
                _sca_strategy_localized(issue),
                _sca_strategy_plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"),
            )
        ],
    )[0]
    diagnostics = {
        "task_id": "task-1",
        "strategy_stage": SCARemediationStage.NPM_LATEST,
        "security_floor": "4.17.21",
        "selected_version": "5.0.0",
    }

    instruction = _build_high_level_retry_instruction(
        task, group, None, UpdateRetryDiagnostics(**diagnostics)
    )

    assert "5.0.0" in instruction
    assert task.strategy_stage == SCARemediationStage.NPM_LATEST
    assert "update only lodash" in instruction
    assert "do not query a registry or choose another version" in instruction
