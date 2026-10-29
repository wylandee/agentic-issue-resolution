"""Python parent evidence and retry behavior at the shared Supervisor boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from remediation_engine.contracts import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    SCARemediationStage,
    Severity,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import SolverTaskDecision
from remediation_engine.contracts.version_policy import RegistryCandidate
from remediation_engine.orchestration.portfolio_solver import prepare_python_portfolio_problem
from remediation_engine.orchestration.supervisor_node import (
    _portfolio_plan_is_stale,
    _portfolio_plan_violations,
)
from remediation_engine.orchestration.supervisor_planner import (
    _needs_planner,
    _run_deterministic_retry_planner,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from tests.fix_plan_fixtures import group_with_fix_plans


def _group(*, direct: bool = False, include_parent_hint: bool = True) -> VulnerabilityGroup:
    child = "vulnerable-child"
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=child,
        package_version="1.0.0",
        purl=f"pkg:pypi/{child}@1.0.0",
        cve_id="CVE-2026-1000",
        ecosystem="pypi",
        file_path="Pipfile.lock",
    )
    ancestry = [] if direct else [child] if not include_parent_hint else ["direct-parent", child]
    parent_name = None if direct or not include_parent_hint else "direct-parent"
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="Pipfile",
        package_manager="pipenv",
        is_direct_dependency=direct,
        declaration_type="packages",
        dependency_ancestry=ancestry,
        dependency_versions={"direct-parent": "1.0.0"},
        parent_package_name=parent_name,
        parent_package_version="1.0.0" if parent_name else None,
        parent_declaration_type="packages" if parent_name else None,
        localization_confidence=1.0,
    )
    fix_plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="1.2.0",
        instruction="Install the fixed release.",
        strategy_used="osv_api",
    )
    group = group_with_fix_plans([issue], plans_by_finding=[(localized, fix_plan)])[0]
    return group.model_copy(
        update={
            "parent_package_name": parent_name,
            "dependency_ancestry": ancestry,
        }
    )


def _pipfile_fixture(
    tmp_path: Path,
    *,
    declarations: dict[str, str],
    locked_parents: dict[str, str],
) -> None:
    lines = ["[packages]", *(f'{name} = "{value}"' for name, value in declarations.items())]
    (tmp_path / "Pipfile").write_text("\n".join(lines) + "\n", encoding="utf-8")
    locked = {name: {"version": f"=={version}"} for name, version in locked_parents.items()}
    locked["vulnerable-child"] = {"version": "==1.0.0"}
    (tmp_path / "Pipfile.lock").write_text(
        json.dumps({"default": locked, "develop": {}}), encoding="utf-8"
    )


def _candidate(version: str) -> RegistryCandidate:
    return RegistryCandidate(
        version=version,
        ecosystem="pypi",
        semver_key=None,
        security_floor_met=True,
        is_stable=True,
        same_major=True,
        already_attempted=False,
        selection_roles=("same_major",),
    )


@pytest.mark.parametrize("direct", [True, False], ids=["direct", "no-parent-proof"])
def test_unproven_pipfile_target_never_dispatches_to_registry(tmp_path: Path, direct: bool) -> None:
    _pipfile_fixture(
        tmp_path,
        declarations={"vulnerable-child": "==1.0.0"},
        locked_parents={},
    )
    group = _group(direct=direct)
    task = build_initial_remediation_task(group, "task-1")
    calls: list[tuple[str, str, set[str]]] = []

    def registry(package: str, floor: str, attempted: set[str]):
        calls.append((package, floor, attempted))
        return [_candidate("1.2.0")]

    prepared = prepare_python_portfolio_problem(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
        parent_requires_dist_fetcher=lambda *_: ["vulnerable-child>=1.0"],
    )

    assert prepared.targets == []
    assert not prepared.candidate_catalog_complete
    assert calls == []


@pytest.mark.parametrize("proof_shape", ["missing", "ambiguous"])
def test_missing_or_ambiguous_pipfile_parent_proof_fails_closed_before_candidates(
    tmp_path: Path, proof_shape: str
) -> None:
    if proof_shape == "missing":
        declarations = {"direct-parent": ">=1.0,<2"}
        locked_parents = {"direct-parent": "1.0.0"}
        group = _group()

        def requires_dist(_package: str, _version: str):
            return []

    else:
        declarations = {
            "direct-parent": ">=1.0,<2",
            "second-parent": ">=1.0,<2",
        }
        locked_parents = {"direct-parent": "1.0.0", "second-parent": "1.0.0"}
        group = _group(include_parent_hint=False)

        def requires_dist(_package: str, _version: str):
            return ["vulnerable-child>=1.0"]

    _pipfile_fixture(tmp_path, declarations=declarations, locked_parents=locked_parents)
    task = build_initial_remediation_task(group, "task-1")
    registry_called = False

    def registry(*_args):
        nonlocal registry_called
        registry_called = True
        return [_candidate("1.2.0")]

    prepared = prepare_python_portfolio_problem(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
        parent_requires_dist_fetcher=requires_dist,
    )

    assert prepared.targets == []
    assert not prepared.candidate_catalog_complete
    assert not registry_called
    assert any("parent proof" in item or "parent" in item for item in prepared.diagnostics)


def test_python_retry_uses_only_unattempted_pypi_candidates_from_shared_plan() -> None:
    group = _group()
    task = build_initial_remediation_task(group, "pypi-task").model_copy(
        update={
            "status": TaskStatus.NEEDS_RETRY,
            "retry_count": 1,
            "strategy_stage": SCARemediationStage.PYPI_SAME_MAJOR,
            "target_package_name": "direct-parent",
            "parent_package_name": "direct-parent",
            "parent_package_version": "1.0.0",
            "parent_minimum_version": "1.2.0",
            "selected_version": "1.2.0",
            "allowed_target_versions": ["1.2.0", "1.8.0.post1", "3.1.0.post1"],
            "allowed_dependency_types": ["packages"],
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id=task.task_id,
        strategy_stage=SCARemediationStage.PYPI_SAME_MAJOR,
        target_package_name="direct-parent",
        attempted_versions=["1.2.0"],
        executed_versions=["1.2.0"],
        attempted_versions_by_target={"Direct_Parent": ["1.2.0"]},
        candidate_versions_considered=["1.2.0", "1.8.0.post1", "3.1.0.post1"],
    )
    queue = {task.task_id: task}
    group_by_id = {group.group_id: group}

    assert _needs_planner(queue, {}, {task.task_id: diagnostics}, "qa_completed")
    updated, plans = _run_deterministic_retry_planner(
        queue,
        group_by_id,
        {task.task_id: diagnostics},
    )

    plan = plans[task.task_id]
    assert plan.selected_version == "1.8.0.post1"
    assert plan.action == "retry_update"
    assert plan.target_package_name == "direct-parent"
    assert updated[task.task_id].registry_query_performed is False
    assert updated[task.task_id].attempted_versions == ["1.2.0"]


def test_committed_python_portfolio_accepts_language_specific_target_identity(
    tmp_path: Path,
) -> None:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="certifi",
        package_version="2024.5.22",
        purl="pkg:pypi/certifi@2024.5.22",
        cve_id="CVE-2026-1001",
        ecosystem="pypi",
        file_path="requirements.txt",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="requirements.txt",
        package_manager="pip",
        is_direct_dependency=True,
        declaration_type="requirements",
        localization_confidence=1.0,
    )
    fix_plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="2024.7.4",
        instruction="Install the fixed release.",
        strategy_used="odc_cpe_bound",
    )
    group = group_with_fix_plans([issue], plans_by_finding=[(localized, fix_plan)])[0]
    group = group.model_copy(
        update={"file_path": "requirements.txt", "file_paths": ["requirements.txt"]}
    )
    manifest_content = "certifi==2024.5.22\n"
    (tmp_path / "requirements.txt").write_text(manifest_content, encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_content.encode("utf-8")).hexdigest()
    repository_fingerprint = hashlib.sha256(
        json.dumps(
            {"requirements.txt": manifest_digest}, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    task = build_initial_remediation_task(group, "task-1").model_copy(
        update={
            "portfolio_plan_id": "portfolio-python",
            "selected_version": "2024.7.4",
            "allowed_target_versions": ["2024.7.4"],
            "allowed_dependency_types": ["requirements"],
            "target_dependency_type": "requirements",
        }
    )
    decision = SolverTaskDecision(
        task_id=task.task_id,
        selected_strategy="version_bump",
        selected_version="2024.7.4",
        strategy_stage=task.strategy_stage.value,
        target_occurrence_id="python:requirements.txt:certifi",
        target_group_id=group.group_id,
        target_package_name="certifi",
        manifest_path="requirements.txt",
        dependency_type="requirements",
        allowed_dependency_types=["requirements"],
    )
    plan = SimpleNamespace(
        clusters=[],
        portfolio_plan_id="portfolio-python",
        repository_fingerprint=repository_fingerprint,
        solver_plan=SimpleNamespace(
            selected_plan=SimpleNamespace(
                task_decisions=[decision],
                selected_candidate_versions={"python:requirements.txt:certifi": "2024.7.4"},
                batches=[],
            )
        ),
        task_ids=[task.task_id],
        planned_task_revisions={task.task_id: task.task_revision},
        task_strategies={},
    )

    task_queue = {task.task_id: task}
    assert _portfolio_plan_violations(plan, task_queue, [group]) == []
    assert _portfolio_plan_is_stale(plan, task_queue, [group], str(tmp_path)) is False
    (tmp_path / "requirements.txt").write_text("certifi==2024.7.4\n", encoding="utf-8")
    assert _portfolio_plan_is_stale(plan, task_queue, [group], str(tmp_path)) is True

    for field_name, invalid_value in (
        ("target_occurrence_id", "package.json::certifi"),
        ("manifest_path", "other-requirements.txt"),
        ("lockfile_package_key", "node_modules/certifi"),
    ):
        invalid_decision = decision.model_copy(update={field_name: invalid_value})
        invalid_plan = SimpleNamespace(
            clusters=[],
            portfolio_plan_id="portfolio-python",
            repository_fingerprint=repository_fingerprint,
            solver_plan=SimpleNamespace(
                selected_plan=SimpleNamespace(
                    task_decisions=[invalid_decision],
                    selected_candidate_versions={"python:requirements.txt:certifi": "2024.7.4"},
                    batches=[],
                )
            ),
            task_ids=[task.task_id],
            planned_task_revisions={task.task_id: task.task_revision},
            task_strategies={},
        )
        violations = _portfolio_plan_violations(invalid_plan, task_queue, [group])
        assert any(f"solver identity field {field_name} is invalid" in item for item in violations)
