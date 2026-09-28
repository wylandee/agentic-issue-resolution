from __future__ import annotations

from uuid import uuid4

from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TacticalStrategy,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.version_policy import MavenRegistryCandidate
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.supervisor_node import _ordered_update_candidates
from remediation_engine.orchestration.supervisor_planner import _run_deterministic_retry_planner
from remediation_engine.orchestration.supervisor_policy import _next_sca_stage
from remediation_engine.orchestration.tactical_supervisor import (
    allowed_tactical_strategies,
    build_tactical_context,
    registry_candidate_sets_for_context,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.tools.fix_planner import _extract_fixed_from_osv_vuln


def _maven_group(*, direct: bool, declaration_type: str) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="org.example:widget",
        package_version="1.0",
        purl="pkg:maven/org.example/widget@1.0",
        ecosystem="Maven",
        cve_id="CVE-2025-12345",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="module/pom.xml",
        is_direct_dependency=direct,
        package_manager="maven",
        declaration_type=declaration_type,
        localization_confidence=1.0,
    )
    return VulnerabilityGroup(
        group_id=f"maven-{uuid4()}",
        issue_type=IssueType.SCA,
        vulnerable_component="org.example:widget",
        file_path="module/pom.xml",
        file_paths=["module/pom.xml"],
        cve_ids=["CVE-2025-12345"],
        versions=["1.0"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
        parent_package_name=None if direct else "org.example:parent",
        parent_package_version=None if direct else "2.0",
        parent_declaration_type=None if direct else "dependencies",
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="1.1",
            instruction="Use a Supervisor-approved Maven Central candidate.",
            strategy_used="osv_api",
        ),
    )


def test_java_tasks_keep_exact_gav_and_use_localized_maven_edit_target() -> None:
    direct = build_initial_remediation_task(
        _maven_group(direct=True, declaration_type="dependencies"),
        "task-direct",
        project_language=ProjectLanguage.JAVA,
    )
    managed_direct = build_initial_remediation_task(
        _maven_group(direct=True, declaration_type="dependencyManagement"),
        "task-managed",
        project_language=ProjectLanguage.JAVA,
    )
    transitive = build_initial_remediation_task(
        _maven_group(direct=False, declaration_type="dependencyManagement"),
        "task-transitive",
        project_language=ProjectLanguage.JAVA,
    )

    assert direct.target_package_name == "org.example:widget"
    assert direct.target_dependency_type == "dependencies"
    assert managed_direct.target_package_name == "org.example:widget"
    assert managed_direct.target_dependency_type == "dependencyManagement"
    assert transitive.target_package_name == "org.example:widget"
    assert transitive.target_dependency_type == "dependencyManagement"
    assert all(
        task.strategy_stage == SCARemediationStage.OSV_MINIMUM
        for task in (direct, managed_direct, transitive)
    )
    assert all(task.parent_package_name is None for task in (direct, managed_direct, transitive))


def test_java_retry_stage_and_candidate_whitelist_use_maven_stability() -> None:
    task = build_initial_remediation_task(
        _maven_group(direct=True, declaration_type="dependencies"),
        "task-stage",
        project_language=ProjectLanguage.JAVA,
    ).model_copy(update={"selected_version": "1.2"})
    assert (
        _next_sca_stage(
            SCARemediationStage.OSV_MINIMUM,
            project_language=ProjectLanguage.JAVA,
            maven_mode=True,
        )
        == SCARemediationStage.MAVEN_LATEST
    )
    assert (
        _next_sca_stage(
            SCARemediationStage.MAVEN_LATEST,
            project_language=ProjectLanguage.JAVA,
            maven_mode=True,
        )
        == SCARemediationStage.CODE_WORKAROUND
    )
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM) == SCARemediationStage.NPM_SAME_MAJOR

    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        security_floor="1.1",
        target_package_name="org.example:widget",
        target_dependency_type="dependencies",
        selected_version="1.2",
        candidate_versions_considered=["1.2", "2.0", "1.2-SNAPSHOT", "2.0-rc1", "0.9"],
    )
    versions, types = _ordered_update_candidates(
        task,
        diagnostics=diagnostics,
        project_language=ProjectLanguage.JAVA,
    )
    assert versions == ["1.2"]
    assert types == ["dependencies"]


def _maven_candidate(version: str, role: str) -> MavenRegistryCandidate:
    return MavenRegistryCandidate(
        version=version,
        security_floor_met=True,
        is_stable=True,
        already_attempted=False,
        selection_roles=(role,),
    )


def test_osv_maven_floor_uses_ecosystem_events_and_comparable_version_order() -> None:
    vuln = {
        "affected": [
            {
                "package": {"name": "org.example:widget", "ecosystem": "Maven"},
                "ranges": [
                    {"type": "ECOSYSTEM", "events": [{"fixed": "2.0"}, {"fixed": "1.10"}]},
                    {"type": "SEMVER", "events": [{"fixed": "0.9"}]},
                    {"type": "GIT", "database_specific": {"extracted_events": [{"fixed": "0.8"}]}},
                ],
            }
        ]
    }
    fixed, _snippets = _extract_fixed_from_osv_vuln(
        vuln,
        "org.example:widget",
        ecosystem="Maven",
    )
    assert fixed == "1.10"


def test_maven_tactical_seams_target_exact_gav_and_preserve_approved_pool() -> None:
    group = _maven_group(direct=False, declaration_type="dependencyManagement")
    task = build_initial_remediation_task(
        group,
        "task-tactical",
        project_language=ProjectLanguage.JAVA,
    )
    assert TacticalStrategy.PACKAGE_OVERRIDE not in allowed_tactical_strategies(
        task,
        group,
        ProjectLanguage.JAVA,
    )

    def provider(
        package_name: str, floor: str, attempted: set[str]
    ) -> list[MavenRegistryCandidate]:
        assert package_name == "org.example:widget"
        assert floor == "1.1"
        return [
            _maven_candidate("1.1", "maven_minimum"),
            _maven_candidate("2.0", "maven_latest"),
            _maven_candidate("3.0", "maven_latest"),
        ]

    context = build_tactical_context(
        task,
        group,
        project_language=ProjectLanguage.JAVA,
    )
    minimum_sets, error = registry_candidate_sets_for_context(
        context,
        registry_provider=provider,
    )
    assert error is None
    assert len(minimum_sets) == 1
    assert minimum_sets[0].target_package_name == "org.example:widget"
    assert minimum_sets[0].dependency_type == "dependencyManagement"
    assert minimum_sets[0].versions == ("1.1",)
    assert minimum_sets[0].approved_version_pool == ("1.1", "2.0", "3.0")

    latest_task = task.model_copy(update={"strategy_stage": SCARemediationStage.MAVEN_LATEST})
    prior = UpdateRetryDiagnostics(
        task_id=task.task_id,
        strategy_stage=SCARemediationStage.MAVEN_LATEST,
        security_floor="1.1",
        target_package_name="org.example:widget",
        target_dependency_type="dependencyManagement",
        candidate_versions_considered=list(minimum_sets[0].approved_version_pool),
    )
    latest_context = build_tactical_context(
        latest_task,
        group,
        retry_diagnostics=prior,
        project_language=ProjectLanguage.JAVA,
    )
    latest_sets, error = registry_candidate_sets_for_context(
        latest_context,
        registry_provider=provider,
    )
    assert error is None
    assert latest_sets[0].versions == ("3.0",)
    assert latest_sets[0].approved_version_pool == ("1.1", "2.0", "3.0")


def test_maven_retry_planner_advances_roles_and_never_calls_npm_parent(
    monkeypatch,
) -> None:
    group = _maven_group(direct=False, declaration_type="dependencyManagement")
    task = build_initial_remediation_task(
        group,
        "task-retry",
        project_language=ProjectLanguage.JAVA,
    ).model_copy(update={"status": TaskStatus.NEEDS_RETRY})
    calls: list[tuple[str, set[str], ProjectLanguage, bool]] = []

    def fetch_candidates(
        package_name: str,
        security_floor: str,
        attempted_versions: set[str],
        *,
        project_language: ProjectLanguage,
        maven_mode: bool,
    ) -> list[MavenRegistryCandidate]:
        assert package_name == "org.example:widget"
        assert security_floor == "1.1"
        calls.append((package_name, attempted_versions, project_language, maven_mode))
        return [
            _maven_candidate("1.1", "maven_minimum"),
            _maven_candidate("2.0", "maven_latest"),
        ]

    def forbid_npm_parent(*_args, **_kwargs):
        raise AssertionError("Java Maven retries must not call the npm parent planner")

    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_planner._supervisor_fetch_registry_candidates",
        fetch_candidates,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_planner._supervisor_plan_npm_parent_version",
        forbid_npm_parent,
    )

    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        security_floor="1.1",
        attempted_versions=["1.1"],
        target_package_name="org.example:widget",
        target_dependency_type="dependencyManagement",
    )
    _updated, plans = _run_deterministic_retry_planner(
        {task.task_id: task},
        {group.group_id: group},
        {task.task_id: diagnostics},
        project_language=ProjectLanguage.JAVA,
    )
    plan = plans[task.task_id]
    assert plan.strategy_stage == SCARemediationStage.MAVEN_LATEST
    assert plan.selected_version == "2.0"
    assert plan.target_package_name == "org.example:widget"
    assert plan.target_dependency_type == "dependencyManagement"
    assert plan.parent_minimum_version is None
    assert len(calls) >= 2
    assert all(
        language == ProjectLanguage.JAVA and maven_mode
        for _package, _attempted, language, maven_mode in calls
    )

    attempted_everything = diagnostics.model_copy(update={"attempted_versions": ["1.1", "2.0"]})
    _updated, exhausted_plans = _run_deterministic_retry_planner(
        {task.task_id: task},
        {group.group_id: group},
        {task.task_id: attempted_everything},
        project_language=ProjectLanguage.JAVA,
    )
    exhausted = exhausted_plans[task.task_id]
    assert exhausted.strategy_stage == SCARemediationStage.MAVEN_LATEST
    assert exhausted.selected_version is None
    assert exhausted.exhausted_update_path is True
    assert exhausted.action == "pivot_workaround"


def test_node_update_candidate_ordering_keeps_existing_semver_normalization() -> None:
    task = RemediationTask(
        task_id="task-node",
        parent_group_id="node-group",
        instruction="Update the dependency.",
        target_package_name="left-pad",
        target_dependency_type="dependencies",
        strategy=RoutingStrategy.VERSION_BUMP,
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        selected_version="v1.2.4",
        candidate_versions_considered=["v1.2.4", "1.2.3"],
        target_package_name="left-pad",
        target_dependency_type="dependencies",
    )
    versions, types = _ordered_update_candidates(task, diagnostics=diagnostics)
    assert versions == ["1.2.4", "1.2.3"]
    assert types == ["dependencies"]
