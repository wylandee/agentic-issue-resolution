"""Focused coverage for the OSV-first, strategy-aware SCA flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from remediation_engine.contracts import (
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    Severity,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.subagent_runtime import ToolEvent
from remediation_engine.orchestration.supervisor_node import (
    MAX_RETRIES,
    _build_high_level_retry_instruction,
    _next_sca_stage,
    run_supervisor_node,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.orchestration.update_subagent import (
    _build_retry_diagnostics,
    _build_update_prompt,
)
from remediation_engine.tools.fix_planner import (
    _extract_fixed_from_osv_vuln,
    _query_osv_fixed_version,
    plan_fix,
)
from remediation_engine.tools.registry_tools import (
    plan_npm_parent_version,
    plan_npm_version,
    select_npm_parent_version,
)
from remediation_engine.triage.grouper import group_issues


def _issue(cve: str, *, package: str = "lodash") -> VulnerabilityIssue:
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


def _localized(issue: VulnerabilityIssue) -> LocalizedIssue:
    return LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        package_manager="npm",
        is_direct_dependency=True,
        localization_confidence=1.0,
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
    issue = _issue("CVE-2026-0001")
    localized = _localized(issue)
    with (
        patch(
            "remediation_engine.tools.fix_planner._query_osv_fixed_version",
            return_value=("4.17.21", None),
        ) as osv,
        patch("remediation_engine.tools.fix_planner._serper_search_and_extract") as serper,
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
    with patch("remediation_engine.tools.fix_planner.requests.post", return_value=response):
        fixed, snippets = _query_osv_fixed_version(_issue("CVE-2026-0001"))

    assert fixed == "4.17.21"
    assert snippets is None


def test_same_package_findings_are_separated_by_strategy_and_update_floor_is_aggregated():
    update_a = _issue("CVE-2026-0001")
    update_b = _issue("CVE-2026-0002")
    workaround = _issue("CVE-2026-0003")
    no_fix = _issue("CVE-2026-0004")
    groups = group_issues(
        [update_a, update_b, workaround, no_fix],
        sca_issue_plans=[
            (_localized(update_a), _plan(FixPlanStatus.VERSION_FOUND, version="4.17.21")),
            (_localized(update_b), _plan(FixPlanStatus.VERSION_FOUND, version="4.18.0")),
            (
                _localized(workaround),
                _plan(
                    FixPlanStatus.WORKAROUND_FOUND,
                    snippets=["disable the parser"],
                ),
            ),
            (_localized(no_fix), _plan(FixPlanStatus.NO_FIX)),
        ],
    )

    by_strategy = {group.fix_plan.strategy_used: group for group in groups}
    assert set(by_strategy) == {"UPDATE_VERSION", "WORKAROUND", "NO_FIX"}
    assert by_strategy["UPDATE_VERSION"].fix_plan.fixed_version == "4.18.0"
    assert len(by_strategy["UPDATE_VERSION"].localized_issues) == 2
    assert len(by_strategy["WORKAROUND"].localized_issues) == 1
    assert len(by_strategy["NO_FIX"].localized_issues) == 1


def test_supervisor_npm_tool_selects_same_major_then_latest(monkeypatch):
    registry_data = {
        "versions": {
            "4.17.21": {},
            "4.18.0": {},
            "5.0.0": {},
            "6.1.0-beta.1": {},
        }
    }
    monkeypatch.setattr(
        "remediation_engine.tools.registry_tools._fetch_package_data",
        lambda package: registry_data,
    )

    same_major = plan_npm_version.invoke(
        {
            "package_name": "lodash",
            "security_floor": "4.17.21",
            "selection": "same_major",
            "attempted_versions": "4.18.0",
        }
    )
    latest = plan_npm_version.invoke(
        {
            "package_name": "lodash",
            "security_floor": "4.17.21",
            "selection": "latest",
            "attempted_versions": "4.18.0",
        }
    )

    assert "Selected Version: 4.17.21" in same_major
    assert "Selected Version: 5.0.0" in latest
    assert "6.1.0-beta.1" not in latest


def test_supervisor_npm_tool_skips_same_major_when_it_equals_latest(monkeypatch):
    monkeypatch.setattr(
        "remediation_engine.tools.registry_tools._fetch_package_data",
        lambda package: {"versions": {"4.18.0": {}, "4.17.21": {}}},
    )

    same_major = plan_npm_version.invoke(
        {
            "package_name": "lodash",
            "security_floor": "4.17.21",
            "selection": "same_major",
        }
    )
    latest = plan_npm_version.invoke(
        {
            "package_name": "lodash",
            "security_floor": "4.17.21",
            "selection": "latest",
        }
    )

    assert "Selected Version: 4.18.0" in same_major
    assert "Same-Major Stage: SKIPPED" in same_major
    assert "Selected Version: 4.18.0" in latest


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

    monkeypatch.setattr("remediation_engine.tools.registry_tools._fetch_package_data", fetch)

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
    issue = _issue("CVE-2026-0002", package="transitive-child")
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
    group = group_issues(
        [issue],
        sca_issue_plans=[(localized, _plan(FixPlanStatus.VERSION_FOUND, version="2.0.1"))],
    )[0]

    task = build_initial_remediation_task(group, "task-1")

    assert task.strategy_stage == SCARemediationStage.OSV_MINIMUM
    assert task.target_package_name == "direct-parent"
    assert task.target_dependency_type == "dependencies"
    assert task.selected_version is None
    assert "direct-parent" in task.instruction
    assert "overrides" not in task.instruction


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
    issue = _issue("CVE-2026-0001")
    group = group_issues(
        [issue],
        sca_issue_plans=[
            (_localized(issue), _plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"))
        ],
    )[0]
    prompt = _build_update_prompt(
        [(task, group, ["package.json"])],
        [],
        {},
        {},
        {},
    )

    assert "execution worker" in prompt
    assert "exact version 4.18.0" in prompt
    assert "view_npm_package_versions" not in prompt
    assert "Planning Answers" not in prompt
    assert "per-package" in prompt
    assert "with that package_name" in prompt
    assert "exactly once for the final batch state" not in prompt


def test_update_worker_records_failed_exact_version_attempts_without_selecting_next_version():
    issue = _issue("CVE-2026-0001")
    group = group_issues(
        [issue],
        sca_issue_plans=[
            (_localized(issue), _plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"))
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
                name="modify_npm_dependency",
                args={
                    "package_name": group.vulnerable_component,
                    "target_version": "5.0.0",
                    "dependency_type": "dependencies",
                },
                content="FAILURE: package install failed",
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


def test_supervisor_stage_progression_and_retry_cap():
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM) == SCARemediationStage.NPM_SAME_MAJOR
    assert _next_sca_stage(SCARemediationStage.NPM_SAME_MAJOR) == SCARemediationStage.NPM_LATEST
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST) == SCARemediationStage.CODE_WORKAROUND
    assert MAX_RETRIES == 3


def test_qa_failure_advances_task_and_diagnostics_to_next_supervisor_stage():
    issue = _issue("CVE-2026-0001")
    group = group_issues(
        [issue],
        sca_issue_plans=[
            (_localized(issue), _plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"))
        ],
    )[0]
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
    )
    result = run_supervisor_node(
        {
            "valid_groups": [group],
            "task_queue": {task.task_id: task},
            "status": "qa_completed",
            "qa_evaluations": {
                task.task_id: QAEvaluation(
                    task_id=task.task_id,
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="the OSV version did not clear the finding",
                )
            },
        }
    )

    updated_task = result["task_queue"][task.task_id]
    diagnostics = result["retry_diagnostics_by_task"][task.task_id]
    assert updated_task.strategy_stage == SCARemediationStage.NPM_SAME_MAJOR
    assert diagnostics.strategy_stage == SCARemediationStage.NPM_SAME_MAJOR
    assert diagnostics.security_floor == "4.17.21"


def test_retry_instruction_contains_exact_selected_version_and_stage():
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="g1",
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        status=TaskStatus.NEEDS_RETRY,
    )
    issue = _issue("CVE-2026-0001")
    group = group_issues(
        [issue],
        sca_issue_plans=[
            (_localized(issue), _plan(FixPlanStatus.VERSION_FOUND, version="4.17.21"))
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
    assert SCARemediationStage.NPM_LATEST.value in instruction
