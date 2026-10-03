"""Deterministic package-portfolio planning and delta-isolation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from remediation_engine.cli import _solve_output
from remediation_engine.contracts import (
    MAX_MULTI_PACKAGE_ACTION_SIZE,
    DependencyParentContext,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    MultiPackageAction,
    PackageMutation,
    Severity,
    SupervisorDecision,
    TaskCluster,
    TaskDependency,
    TaskDependencyKind,
    TaskStatus,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionCertificate,
    PackageResolutionStatus,
    SolverRuntimeFingerprint,
)
from remediation_engine.orchestration.graph import _portfolio_certificate_violations
from remediation_engine.orchestration.portfolio_orchestrator import (
    apply_portfolio_plan,
    build_portfolio_plan,
    isolate_delta_failure,
    materialize_synthetic_dependency_tasks,
    prepare_portfolio_inputs,
    rank_suspect_tasks_by_suspicion,
    score_suspect_package,
)
from remediation_engine.orchestration.portfolio_solver import (
    _build_targets_and_findings,
    _candidate_domains,
    _fetch_candidate_packuments,
    _issue_identity,
)
from remediation_engine.orchestration.qa_test_parsing import parse_peer_conflict_evidence
from remediation_engine.orchestration.state import initial_orchestrator_state
from remediation_engine.orchestration.supervisor_node import (
    _deterministic_routing,
    _portfolio_plan_violations,
    run_supervisor_node,
)
from remediation_engine.orchestration.supervisor_routing import _portfolio_cluster_targets
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from remediation_engine.tools.npm_graph import load_npm_graph_snapshot, make_occurrence_id
from remediation_engine.tools.registry_cache import RegistryPackumentCache
from remediation_engine.triage.grouper import group_issues


def _group(
    package_name: str,
    manifest_path: str,
    *,
    cve_id: str | None = None,
    package_version: str = "1.0.0",
    fixed_version: str = "2.0.0",
):
    issue = VulnerabilityIssue(
        source=IssueSource.SYNTHETIC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=package_name,
        package_version=package_version,
        file_path=manifest_path,
        cve_id=(
            cve_id
            or "CVE-2026-"
            + str(
                int(hashlib.sha256(f"{package_name}:{manifest_path}".encode()).hexdigest()[:8], 16)
            )[:6]
        ),
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest_path,
        package_manager="npm",
        declaration_type="dependencies",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )
    plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version=fixed_version,
        instruction=f"Update {package_name}.",
        strategy_used="osv_api",
    )
    return group_issues([issue], sca_issue_plans=[(localized, plan)])[0]


def _write_manifest(root: Path, relative_path: str, payload: dict) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _tasks(*groups):
    return {
        f"task-{index}": build_initial_remediation_task(group, f"task-{index}")
        for index, group in enumerate(groups, start=1)
    }


def _attach_test_resolution_certificate(plan):
    """Attach matching offline evidence for Supervisor contract tests."""
    solver_plan = plan.solver_plan.model_copy(update={"candidate_catalog_complete": True})
    selected = solver_plan.selected_plan
    assert selected is not None
    assignment_digest = hashlib.sha256(
        json.dumps(
            dict(sorted(selected.selected_candidate_versions.items())),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    covered = sorted(
        coverage_id for batch in selected.batches for coverage_id in batch.resolved_coverage_ids
    )
    workaround = sorted(
        coverage_id for batch in selected.batches for coverage_id in batch.workaround_coverage_ids
    )
    assert not any(batch.unresolved_coverage_ids for batch in selected.batches)
    certificate = PackageResolutionCertificate(
        status=PackageResolutionStatus.CERTIFIED,
        portfolio_plan_id=plan.portfolio_plan_id,
        candidate_plan_id=selected.candidate_plan_id,
        solver_input_digest=plan.solver_input_digest,
        repository_fingerprint=plan.repository_fingerprint,
        task_revisions=plan.task_revisions,
        workspace_graph_digest=plan.workspace_graph_digest,
        candidate_catalog_digest=solver_plan.candidate_catalog_digest,
        candidate_assignment_digest=assignment_digest,
        resolved_graph_digest="test-resolved-graph-digest",
        runtime_fingerprint=SolverRuntimeFingerprint(
            node_version="v22.0.0",
            npm_version="10.0.0",
            platform="linux",
            architecture="x64",
        ),
        covered_coverage_ids=covered,
        workaround_coverage_ids=workaround,
    )
    return plan.model_copy(
        update={
            "solver_plan": solver_plan,
            "resolution_certificate": certificate,
        }
    )


def test_shared_cve_across_packages_keeps_finding_records_distinct(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {"@angular/compiler": "1.0.0", "@angular/core": "1.0.0"},
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/@angular/compiler": {"version": "1.0.0"},
                "node_modules/@angular/core": {"version": "1.0.0"},
            },
        },
    )
    shared_cve = "CVE-2026-50557"
    compiler = _group("@angular/compiler", "package.json", cve_id=shared_cve)
    core = _group("@angular/core", "package.json", cve_id=shared_cve)

    plan = build_portfolio_plan(tmp_path, [compiler, core], _tasks(compiler, core))

    assert plan.solver_plan is not None
    assert plan.solver_plan.selected_plan is not None
    coverage_ids = plan.solver_plan.selected_plan.coverage_ids
    assert len(coverage_ids) == 2
    assert all(coverage_id.startswith("coverage:") for coverage_id in coverage_ids)
    assert shared_cve not in coverage_ids


def test_solver_only_cli_output_is_not_dispatchable(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "app", "dependencies": {"lodash": "1.0.0"}},
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app", "dependencies": {"lodash": "1.0.0"}},
                "node_modules/lodash": {"version": "1.0.0"},
            },
        },
    )
    group = _group("lodash", "package.json")
    plan = build_portfolio_plan(tmp_path, [group], _tasks(group))

    output = _solve_output(plan, [], [])

    assert output["solver_status"] == "OPTIMAL"
    assert output["package_resolution_status"] == "NOT_RUN"
    assert output["dispatchable"] is False


def test_one_finding_covers_every_matching_physical_occurrence(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {"express-jwt": "1.0.0", "jsonwebtoken": "1.0.0"},
        },
    )
    nested_key = "node_modules/express-jwt/node_modules/jsonwebtoken"
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/express-jwt": {
                    "version": "1.0.0",
                    "dependencies": {"jsonwebtoken": "1.0.0"},
                },
                "node_modules/jsonwebtoken": {"version": "1.0.0"},
                nested_key: {"version": "1.0.0"},
            },
        },
    )
    group = _group("jsonwebtoken", "package.json", cve_id="CVE-2026-70001")
    plan = build_portfolio_plan(tmp_path, [group], _tasks(group))

    selected = plan.solver_plan.selected_plan
    assert selected is not None
    finding_id = _issue_identity(group.issues[0])
    occurrence_ids = {
        make_occurrence_id("package.json", "jsonwebtoken"),
        make_occurrence_id("package.json", "jsonwebtoken", nested_key),
    }
    expected_coverage_ids = {
        "coverage:"
        + hashlib.sha256((finding_id + "\0" + occurrence_id).encode("utf-8")).hexdigest()
        for occurrence_id in occurrence_ids
    }
    assert set(selected.coverage_ids) == expected_coverage_ids
    assert selected.unresolved_ids == []
    batch = selected.batches[0]
    assert batch.resolved_finding_ids == [finding_id]
    assert batch.resolved_coverage_ids == sorted(expected_coverage_ids)
    assert batch.unresolved_coverage_ids == []


def test_candidate_catalog_keeps_complete_candidate_metadata_without_truncation(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "app", "dependencies": {"foo": "1.0.0"}},
    )
    group = _group("foo", "package.json", fixed_version="2.0.0")
    queue = _tasks(group)
    snapshot = load_npm_graph_snapshot(tmp_path)
    targets, findings, diagnostics = _build_targets_and_findings(snapshot, [group], queue)
    packument = {
        "name": "foo",
        "versions": {
            "1.0.0": {},
            "2.0.0": {
                "dependencies": {"runtime-child": "^1.0.0"},
                "optionalDependencies": {"optional-child": "~2.0.0"},
                "peerDependencies": {"peer-host": ">=3.0.0"},
                "peerDependenciesMeta": {"peer-host": {"optional": True}},
                "engines": {"node": ">=20"},
                "os": ["darwin"],
                "cpu": ["arm64"],
            },
            "2.1.0": {},
            "2.2.0": {},
        },
    }

    domains, complete, catalog_digest = _candidate_domains(
        snapshot,
        targets,
        findings,
        queue,
        [group],
        AppSettings(solver_max_candidates_per_target=2),
        diagnostics,
        packuments={"foo": packument},
    )

    target_domain = domains[targets[0].occurrence_id]
    registry_versions = {
        candidate.version for candidate in target_domain if candidate.source == "registry"
    }
    assert complete is False
    assert registry_versions == {"1.0.0", "2.0.0", "2.1.0", "2.2.0"}
    assert len(target_domain) > 2
    candidate = next(item for item in target_domain if item.version == "2.0.0")
    assert {item.package_name: item.kind for item in candidate.requirements} == {
        "runtime-child": "dependency",
        "optional-child": "optional_dependency",
        "peer-host": "peer",
    }
    assert next(
        item for item in candidate.requirements if item.package_name == "peer-host"
    ).is_optional
    assert candidate.engines == {"node": ">=20"}
    assert candidate.os == ["darwin"]
    assert candidate.cpu == ["arm64"]
    assert len(catalog_digest) == 64


def test_default_candidate_guard_accepts_82_release_catalog(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "app", "dependencies": {"foo": "1.0.0"}},
    )
    group = _group("foo", "package.json", fixed_version="2.0.0")
    queue = _tasks(group)
    snapshot = load_npm_graph_snapshot(tmp_path)
    targets, findings, diagnostics = _build_targets_and_findings(snapshot, [group], queue)
    packument = {
        "name": "foo",
        "versions": {f"2.{minor}.0": {} for minor in range(82)},
    }

    domains, complete, _catalog_digest = _candidate_domains(
        snapshot,
        targets,
        findings,
        queue,
        [group],
        AppSettings(),
        diagnostics,
        packuments={"foo": packument},
    )

    registry_versions = {
        candidate.version
        for candidate in domains[targets[0].occurrence_id]
        if candidate.source == "registry"
    }
    assert AppSettings().solver_max_candidates_per_target == 1000
    assert len(registry_versions) == 82
    assert complete is True
    assert not any("resource guard exceeded" in item for item in diagnostics)


def test_candidate_packuments_are_force_fetched_once_and_cached(tmp_path: Path):
    fetched: list[str] = []

    def fetcher(package_name: str) -> dict[str, object]:
        fetched.append(package_name)
        return {"name": package_name, "versions": {"1.0.0": {}}}

    settings = AppSettings(solver_cache_dir=tmp_path / "registry-cache")
    packuments, complete, catalog_digest, diagnostics = _fetch_candidate_packuments(
        ["z-package", "a-package", "z-package"],
        settings,
        registry_fetcher=fetcher,
    )

    assert fetched == ["a-package", "z-package"]
    assert complete is True
    assert diagnostics == []
    assert len(catalog_digest) == 64
    cache = RegistryPackumentCache(settings.solver_cache_dir)
    assert cache.get("a-package") == packuments["a-package"]
    assert cache.get("z-package") == packuments["z-package"]


def test_finding_identity_namespaces_scanner_and_record_ids():
    scanner = SimpleNamespace(finding_id="scanner-1", id="internal-1")
    internal = SimpleNamespace(finding_id=" ", id="internal-2")
    missing = SimpleNamespace(finding_id=None, id=None)

    assert _issue_identity(scanner) == "scanner:scanner-1"
    assert _issue_identity(internal) == "record:internal-2"
    with pytest.raises(ValueError, match="no stable identity"):
        _issue_identity(missing)


def test_runtime_dependency_order_is_stable(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "root",
            "dependencies": {"lodash": "1.0.0", "app": "1.0.0"},
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "root",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "root"},
                "node_modules/lodash": {"version": "1.0.0"},
                "node_modules/app": {
                    "version": "1.0.0",
                    "dependencies": {"lodash": "1.0.0"},
                },
            },
        },
    )
    lodash = _group("lodash", "package.json")
    app = _group("app", "package.json")
    queue = _tasks(lodash, app)

    first = build_portfolio_plan(tmp_path, [lodash, app], queue)
    second = build_portfolio_plan(tmp_path, [lodash, app], queue)

    assert first.model_dump() == second.model_dump()
    assert first.task_order.index("task-1") < first.task_order.index("task-2")
    assert {task_id for cluster in first.clusters for task_id in cluster.task_ids} == set(queue)


def test_workspace_namespace_packages_form_one_atomic_cluster(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "root",
            "workspaces": ["packages/*"],
            "dependencies": {"@angular/core": "1.0.0", "@angular/common": "1.0.0"},
        },
    )
    _write_manifest(tmp_path, "packages/core/package.json", {"name": "@angular/core"})
    _write_manifest(tmp_path, "packages/common/package.json", {"name": "@angular/common"})
    core = _group("@angular/core", "package.json")
    common = _group("@angular/common", "package.json")
    queue = _tasks(core, common)

    plan = build_portfolio_plan(tmp_path, [core, common], queue)

    assert len(plan.clusters) == 1
    assert set(plan.clusters[0].task_ids) == {"task-1", "task-2"}
    cluster_id, targets = _portfolio_cluster_targets(plan, queue, qa=False)
    assert cluster_id == plan.clusters[0].cluster_id
    assert set(targets) == {"task-1", "task-2"}


def test_oversized_namespace_keeps_finding_tasks_in_first_bounded_cluster(tmp_path: Path):
    package_names = [
        f"@angular/package-{index}" for index in range(MAX_MULTI_PACKAGE_ACTION_SIZE + 1)
    ]
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {package_name: "1.0.0" for package_name in package_names}},
    )
    finding = _group(package_names[0], "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [finding],
        _tasks(finding),
    )

    plan = build_portfolio_plan(tmp_path, groups, queue)

    assert diagnostics == []
    assert max(len(cluster.task_ids) for cluster in plan.clusters) <= MAX_MULTI_PACKAGE_ACTION_SIZE
    first_cluster = next(cluster for cluster in plan.clusters if "task-1" in cluster.task_ids)
    assert len(first_cluster.task_ids) == MAX_MULTI_PACKAGE_ACTION_SIZE
    assert len(first_cluster.task_ids) > 1
    assert any(
        f"partitioned at {MAX_MULTI_PACKAGE_ACTION_SIZE} tasks" in item for item in plan.diagnostics
    )


def test_missing_namespace_dependency_gets_a_coordination_task(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "juice-shop",
            "dependencies": {
                "@angular/common": "^21.2.14",
                "@angular/core": "^21.2.14",
                "@angular/platform-browser": "^21.2.14",
            },
        },
    )
    core = _group("@angular/core", "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        _tasks(core),
    )

    synthetic = [group for group in groups if group.is_synthetic]
    assert diagnostics == []
    assert {group.vulnerable_component for group in synthetic} == {
        "@angular/common",
        "@angular/platform-browser",
    }
    assert all(not group.cve_ids and not group.ghsa_ids for group in synthetic)
    assert all(group.fix_plan and group.fix_plan.fixed_version == "2.0.0" for group in synthetic)
    synthetic_tasks = [task for task in queue.values() if task.is_synthetic]
    assert len(synthetic_tasks) == 2
    assert all(task.selected_version == "2.0.0" for task in synthetic_tasks)

    plan = build_portfolio_plan(tmp_path, groups, queue)

    assert len(plan.clusters) == 1
    assert set(plan.clusters[0].task_ids) == set(queue)


def test_missing_peer_dependency_gets_a_coordination_task_from_lockfile(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "peer-app",
            "dependencies": {
                "hono": "^4.0.0",
                "@hono/node-server": "^1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "peer-app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "peer-app"},
                "node_modules/hono": {
                    "version": "4.0.0",
                    "peerDependencies": {"@hono/node-server": "^1.0.0"},
                },
                "node_modules/@hono/node-server": {"version": "1.0.0"},
            },
        },
    )
    hono = _group("hono", "package.json", package_version="4.0.0")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [hono],
        _tasks(hono),
    )

    assert diagnostics == []
    assert [group.vulnerable_component for group in groups if group.is_synthetic] == [
        "@hono/node-server"
    ]
    plan = build_portfolio_plan(tmp_path, groups, queue)
    assert len(plan.clusters) == 1
    assert len(plan.clusters[0].task_ids) == 2


def test_unflagged_direct_dependency_gets_a_synthetic_task_without_a_seed(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "flagged-package": "^1.0.0",
                "unflagged-package": "1.2.3",
            },
        },
    )
    flagged = _group("flagged-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [flagged],
        _tasks(flagged),
    )

    synthetic = next(group for group in groups if group.vulnerable_component == "unflagged-package")
    synthetic_task = next(task for task in queue.values() if task.is_synthetic)
    assert diagnostics == []
    assert synthetic.group_id == "sca:package.json:unflagged-package"
    assert synthetic.fix_plan is not None
    assert synthetic.fix_plan.fixed_version == "1.2.3"
    assert synthetic_task.selected_version == "1.2.3"
    assert synthetic_task.parent_group_id == synthetic.group_id


def test_scoped_materialization_excludes_unrelated_direct_dependencies(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "flagged-package": "1.0.0",
                "selected-package": "2.0.0",
                "unrelated-package": "3.0.0",
            },
        },
    )
    flagged = _group("flagged-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [flagged],
        _tasks(flagged),
        target_packages=["flagged-package", "selected-package"],
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {
        "flagged-package",
        "selected-package",
    }
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:flagged-package",
        "sca:package.json:selected-package",
    }


def test_prepare_portfolio_inputs_forwards_target_package_scope(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "selected-package": "1.0.0",
                "unrelated-package": "2.0.0",
            },
        },
    )
    selected = _group("selected-package", "package.json")

    groups, queue, diagnostics = prepare_portfolio_inputs(
        tmp_path,
        [selected],
        _tasks(selected),
        target_packages=["selected-package"],
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {"selected-package"}
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:selected-package",
    }


def test_scoped_materialization_keeps_compatible_peer_validation_only(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "selected-package": "1.0.0",
                "required-peer": "1.0.0",
                "unrelated-package": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/selected-package": {
                    "version": "1.0.0",
                    "peerDependencies": {"required-peer": "^1.0.0"},
                },
                "node_modules/required-peer": {"version": "1.0.0"},
                "node_modules/unrelated-package": {"version": "1.0.0"},
            },
        },
    )
    selected = _group("selected-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [selected],
        _tasks(selected),
        target_packages=["selected-package"],
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {"selected-package"}
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:selected-package",
    }


def test_scoped_materialization_includes_peer_when_target_breaks_required_range(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "selected-package": "1.0.0",
                "required-peer": "1.0.0",
                "unrelated-package": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/selected-package": {"version": "1.0.0"},
                "node_modules/required-peer": {
                    "version": "1.0.0",
                    "peerDependencies": {"selected-package": "^1.0.0"},
                },
                "node_modules/unrelated-package": {"version": "1.0.0"},
            },
        },
    )
    selected = _group("selected-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [selected],
        _tasks(selected),
        target_packages=["selected-package"],
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {
        "selected-package",
        "required-peer",
    }
    assert "sca:package.json:unrelated-package" not in {
        task.parent_group_id for task in queue.values()
    }


def test_scoped_materialization_does_not_expand_optional_peer(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "selected-package": "1.0.0",
                "optional-peer": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/selected-package": {
                    "version": "1.0.0",
                    "peerDependencies": {"optional-peer": "^1.0.0"},
                    "peerDependenciesMeta": {"optional-peer": {"optional": True}},
                },
                "node_modules/optional-peer": {"version": "1.0.0"},
            },
        },
    )
    selected = _group("selected-package", "package.json")

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [selected],
        _tasks(selected),
        target_packages=["selected-package"],
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {"selected-package"}
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:selected-package",
    }


def test_scoped_angular_mutation_closure_stays_at_nine_packages(tmp_path: Path):
    angular_packages = [
        "@angular/animations",
        "@angular/common",
        "@angular/compiler",
        "@angular/compiler-cli",
        "@angular/core",
        "@angular/forms",
        "@angular/platform-browser",
        "@angular/platform-browser-dynamic",
        "@angular/router",
        "@angular/build",
        "@angular/cdk",
        "@angular/cli",
        "@angular/material",
        "@angular/language-service",
    ]
    target_packages = {"@angular/common", "@angular/compiler", "@angular/core"}
    manifest_dependencies = {package: "^21.2.14" for package in angular_packages}
    manifest_dependencies.update(
        {
            "rxjs": "7.8.2",
            "zone.js": "0.15.1",
            "vitest": "3.0.0",
            "jsdom": "26.0.0",
            "@types/node": "22.0.0",
        }
    )
    _write_manifest(
        tmp_path,
        "package.json",
        {"name": "app", "dependencies": manifest_dependencies},
    )
    exact_peer = "21.2.14"
    lock_packages = {
        "": {"name": "app"},
        "node_modules/@angular/animations": {
            "version": exact_peer,
            "peerDependencies": {"@angular/core": exact_peer},
        },
        "node_modules/@angular/common": {"version": exact_peer},
        "node_modules/@angular/compiler": {"version": exact_peer},
        "node_modules/@angular/compiler-cli": {
            "version": exact_peer,
            "peerDependencies": {"@angular/compiler": exact_peer},
        },
        "node_modules/@angular/core": {"version": exact_peer},
        "node_modules/@angular/forms": {
            "version": exact_peer,
            "peerDependencies": {
                "@angular/common": exact_peer,
                "@angular/core": exact_peer,
            },
        },
        "node_modules/@angular/platform-browser": {
            "version": exact_peer,
            "peerDependencies": {
                "@angular/common": exact_peer,
                "@angular/core": exact_peer,
            },
        },
        "node_modules/@angular/platform-browser-dynamic": {
            "version": exact_peer,
            "peerDependencies": {
                "@angular/common": exact_peer,
                "@angular/core": exact_peer,
                "@angular/platform-browser": exact_peer,
            },
        },
        "node_modules/@angular/router": {
            "version": exact_peer,
            "peerDependencies": {"@angular/common": exact_peer},
        },
        "node_modules/@angular/build": {
            "version": exact_peer,
            "peerDependencies": {
                "@angular/compiler": ">=21.0.0 <22.0.0",
                "@angular/compiler-cli": ">=21.0.0 <22.0.0",
            },
        },
        "node_modules/@angular/cdk": {
            "version": exact_peer,
            "peerDependencies": {"@angular/common": ">=21.0.0 <22.0.0"},
        },
        "node_modules/@angular/cli": {"version": exact_peer},
        "node_modules/@angular/material": {
            "version": exact_peer,
            "peerDependencies": {
                "@angular/cdk": exact_peer,
                "@angular/common": ">=21.0.0 <22.0.0",
            },
        },
        "node_modules/@angular/language-service": {"version": exact_peer},
        "node_modules/rxjs": {"version": "7.8.2"},
        "node_modules/zone.js": {"version": "0.15.1"},
        "node_modules/vitest": {"version": "3.0.0"},
        "node_modules/jsdom": {"version": "26.0.0"},
        "node_modules/@types/node": {"version": "22.0.0"},
    }
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {"name": "app", "lockfileVersion": 3, "packages": lock_packages},
    )
    groups = [
        _group(package, "package.json", fixed_version="21.2.17")
        for package in sorted(target_packages)
    ]

    prepared_groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        groups,
        _tasks(*groups),
        target_packages=sorted(target_packages),
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in prepared_groups} == {
        "@angular/animations",
        "@angular/common",
        "@angular/compiler",
        "@angular/compiler-cli",
        "@angular/core",
        "@angular/forms",
        "@angular/platform-browser",
        "@angular/platform-browser-dynamic",
        "@angular/router",
    }
    assert {task.target_package_name for task in queue.values() if task.target_package_name} == {
        group.vulnerable_component for group in prepared_groups
    }
    assert {task.selected_version for task in queue.values()} == {"21.2.17"}


def test_all_exact_direct_dependencies_are_materialized_without_finding_groups(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "first-package": "1.0.0",
                "second-package": "2.0.0",
            },
        },
    )

    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [],
        {},
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in groups} == {
        "first-package",
        "second-package",
    }
    assert {task.parent_group_id for task in queue.values()} == {
        "sca:package.json:first-package",
        "sca:package.json:second-package",
    }
    assert all(task.is_synthetic for task in queue.values())


def test_runtime_edges_use_lockfile_package_metadata(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "app-a": "1.0.0",
                "app-b": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "node_modules/app-a": {
                    "version": "1.0.0",
                    "dependencies": {"app-b": "^1.0.0"},
                },
                "node_modules/app-b": {"version": "1.0.0"},
            },
        },
    )
    app_a = _group("app-a", "package.json")
    app_b = _group("app-b", "package.json")
    queue = _tasks(app_a, app_b)

    plan = build_portfolio_plan(tmp_path, [app_a, app_b], queue)

    assert plan.task_order.index("task-2") < plan.task_order.index("task-1")
    assert any(
        dependency.upstream_task_id == "task-2"
        and dependency.downstream_task_id == "task-1"
        and dependency.edge_type == TaskDependencyKind.RUNTIME
        for cluster in plan.clusters
        for dependency in cluster.dependencies
    )


def test_runtime_related_synthetic_package_keeps_its_resolved_version(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "app-a": "1.0.0",
                "app-b": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "node_modules/app-a": {
                    "version": "1.0.0",
                    "dependencies": {"app-b": "1.0.0"},
                },
                "node_modules/app-b": {"version": "1.0.0"},
            },
        },
    )
    app_a = _group("app-a", "package.json")
    groups, queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [app_a],
        _tasks(app_a),
    )

    app_b = next(group for group in groups if group.vulnerable_component == "app-b")
    assert diagnostics == []
    assert app_b.is_synthetic
    assert app_b.fix_plan is not None
    assert app_b.fix_plan.fixed_version == "1.0.0"
    assert queue["task-2"].selected_version == "1.0.0"


def test_synthetic_dependency_materialization_is_idempotent(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    original_queue = _tasks(core)
    first_groups, first_queue, first_diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        original_queue,
    )
    second_groups, second_queue, second_diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        first_groups,
        first_queue,
    )

    assert first_diagnostics == second_diagnostics == []
    assert [group.group_id for group in second_groups] == [group.group_id for group in first_groups]
    assert list(second_queue) == list(first_queue)
    assert [task.model_dump() for task in second_queue.values()] == [
        task.model_dump() for task in first_queue.values()
    ]


def test_synthetic_target_refresh_keeps_group_identity(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    groups, queue, _diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        [core],
        _tasks(core),
    )
    core_task = queue["task-1"].model_copy(update={"selected_version": "3.0.0"})
    refreshed_groups, refreshed_queue, diagnostics = materialize_synthetic_dependency_tasks(
        tmp_path,
        groups,
        {**queue, "task-1": core_task},
    )

    synthetic_group = next(group for group in refreshed_groups if group.is_synthetic)
    synthetic_task = next(task for task in refreshed_queue.values() if task.is_synthetic)
    assert diagnostics == []
    assert synthetic_group.group_id == "sca:package.json:@angular/common"
    assert synthetic_group.fix_plan is not None
    assert synthetic_group.fix_plan.fixed_version == "3.0.0"
    assert synthetic_task.selected_version == "3.0.0"
    assert synthetic_task.task_revision == 1


def test_supervisor_leaves_synthetic_materialization_to_outer_portfolio(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"@angular/core": "^21.2.14", "@angular/common": "^21.2.14"}},
    )
    core = _group("@angular/core", "package.json")
    state = initial_orchestrator_state(str(tmp_path), [core])

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "portfolio"
    synthetic_groups = [group for group in result["valid_groups"] if group.is_synthetic]
    synthetic_tasks = [task for task in result["task_queue"].values() if task.is_synthetic]
    assert synthetic_groups == []
    assert synthetic_tasks == []


def test_multi_kind_angular_edges_materialize_once_per_task_pair(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "juice-shop",
            "dependencies": {
                "@angular/common": "^21.2.14",
                "@angular/compiler": "^21.2.14",
                "@angular/core": "^21.2.14",
            },
        },
    )
    groups = [
        _group("@angular/common", "package.json"),
        _group("@angular/compiler", "package.json"),
        _group("@angular/core", "package.json"),
    ]

    plan = build_portfolio_plan(tmp_path, groups, _tasks(*groups))

    assert len(plan.clusters) == 1
    dependencies = plan.clusters[0].dependencies
    edge_keys = [
        (dependency.upstream_task_id, dependency.downstream_task_id) for dependency in dependencies
    ]
    assert len(edge_keys) == len(set(edge_keys))
    dependency_by_edge = {
        (dependency.upstream_task_id, dependency.downstream_task_id): dependency.edge_type
        for dependency in dependencies
    }
    assert dependency_by_edge[("task-1", "task-2")] == TaskDependencyKind.WORKSPACE


def test_unrelated_package_groups_remain_singletons(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    left = _group("left-package", "package.json")
    right = _group("right-package", "package.json")
    queue = _tasks(left, right)

    plan = build_portfolio_plan(tmp_path, [left, right], queue)

    assert len(plan.clusters) == 2
    assert {tuple(cluster.task_ids) for cluster in plan.clusters} == {
        ("task-1",),
        ("task-2",),
    }


def test_delta_isolation_identifies_a_unique_failing_package():
    result = isolate_delta_failure(
        ["task-b", "task-a"],
        lambda subset: "FAIL" if "task-a" in subset else "PASS",
    )

    assert result.status == "IDENTIFIED"
    assert result.responsible_task_ids == ("task-a",)
    assert result.executions == 3


def test_delta_isolation_preserves_interaction_and_ambiguity():
    interaction = isolate_delta_failure(
        ["a", "b"], lambda subset: "FAIL" if len(subset) == 2 else "PASS"
    )
    ambiguous = isolate_delta_failure(["a", "b"], lambda _subset: "FAIL")

    assert interaction.status == "INTERACTION_FAILURE"
    assert ambiguous.status == "AMBIGUOUS"
    assert interaction.executions <= 3
    assert ambiguous.executions <= 3


def test_bounded_delta_isolation_probes_only_the_ranked_budgeted_singletons():
    probed = []
    result = isolate_delta_failure(
        ["task-d", "task-c", "task-b", "task-a"],
        lambda subset: probed.append(subset) or "PASS",
        ranked_suspect_task_ids=["task-c", "outside", "task-d", "task-b", "task-a"],
        max_canary_probes=2,
    )

    assert result.status == "INCONCLUSIVE"
    assert result.tested_subsets == (("task-c",), ("task-d",))
    assert result.executions == 2
    assert probed == [("task-c",), ("task-d",)]


def test_bounded_delta_isolation_stops_on_first_ranked_failure():
    probed = []
    result = isolate_delta_failure(
        ["task-a", "task-b", "task-c"],
        lambda subset: probed.append(subset) or "FAIL",
        ranked_suspect_task_ids=["task-b", "task-a", "task-c"],
        max_canary_probes=3,
    )

    assert result.status == "IDENTIFIED"
    assert result.responsible_task_ids == ("task-b",)
    assert result.tested_subsets == (("task-b",),)
    assert result.executions == 1
    assert probed == [("task-b",)]


def test_bounded_delta_isolation_counts_inconclusive_probes_against_budget():
    probed = []
    outcomes = iter(("INCONCLUSIVE", "FAIL"))
    result = isolate_delta_failure(
        ["task-a", "task-b"],
        lambda subset: probed.append(subset) or next(outcomes),
        ranked_suspect_task_ids=["task-a", "task-b"],
        max_canary_probes=2,
    )

    assert result.status == "IDENTIFIED"
    assert result.responsible_task_ids == ("task-b",)
    assert result.tested_subsets == (("task-a",), ("task-b",))
    assert result.executions == 2
    assert probed == [("task-a",), ("task-b",)]


def test_bounded_delta_isolation_uses_configured_default_budget():
    probed = []
    result = isolate_delta_failure(
        ["task-a", "task-b", "task-c"],
        lambda subset: probed.append(subset) or "PASS",
        ranked_suspect_task_ids=["task-a", "task-b", "task-c"],
    )

    assert result.status == "INCONCLUSIVE"
    assert result.tested_subsets == (("task-a",), ("task-b",))
    assert result.executions == 2
    assert probed == [("task-a",), ("task-b",)]


@pytest.mark.parametrize("budget", [0, -1])
def test_bounded_delta_isolation_rejects_nonpositive_budgets_without_probing(budget):
    probed = []
    result = isolate_delta_failure(
        ["task-a"],
        lambda subset: probed.append(subset) or "FAIL",
        max_canary_probes=budget,
    )

    assert result.status == "INCONCLUSIVE"
    assert result.executions == 0
    assert result.tested_subsets == ()
    assert probed == []


def test_bounded_delta_isolation_rejects_empty_ranks_and_uses_sorted_budget_only():
    probed = []
    empty_rank = isolate_delta_failure(
        ["task-a", "task-b"],
        lambda subset: probed.append(subset) or "FAIL",
        ranked_suspect_task_ids=[],
    )
    sorted_budget = isolate_delta_failure(
        ["task-b", "task-a"],
        lambda subset: probed.append(subset) or "PASS",
        max_canary_probes=1,
    )

    assert empty_rank.status == "INCONCLUSIVE"
    assert empty_rank.executions == 0
    assert empty_rank.tested_subsets == ()
    assert sorted_budget.tested_subsets == (("task-a",),)
    assert sorted_budget.executions == 1
    assert probed == [("task-a",)]


@pytest.mark.parametrize("task_count", [11, 20, 30])
def test_bounded_delta_isolation_accepts_up_to_contract_task_limit(task_count):
    task_ids = [f"task-{index:02}" for index in range(task_count)]
    probed = []
    result = isolate_delta_failure(
        task_ids,
        lambda subset: probed.append(subset) or "FAIL",
        ranked_suspect_task_ids=[task_ids[-1]],
        max_canary_probes=1,
    )

    assert result.status == "IDENTIFIED"
    assert result.responsible_task_ids == (task_ids[-1],)
    assert result.executions == 1
    assert probed == [(task_ids[-1],)]


def test_bounded_delta_isolation_rejects_above_contract_task_limit_without_probing():
    probed = []
    result = isolate_delta_failure(
        [f"task-{index:02}" for index in range(MAX_MULTI_PACKAGE_ACTION_SIZE + 1)],
        lambda subset: probed.append(subset) or "FAIL",
        max_canary_probes=1,
    )

    assert result.status == "INCONCLUSIVE"
    assert result.executions == 0
    assert result.tested_subsets == ()
    assert probed == []


def test_legacy_delta_isolation_keeps_ten_task_guard_without_new_options():
    probed = []
    result = isolate_delta_failure(
        [f"task-{index:02}" for index in range(11)],
        lambda subset: probed.append(subset) or "FAIL",
    )

    assert result.status == "INCONCLUSIVE"
    assert result.executions == 0
    assert probed == []


def test_active_cluster_with_internal_dependencies_is_dispatchable(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "root",
            "dependencies": {"@nestjs/core": "1.0.0", "@nestjs/common": "1.0.0"},
        },
    )
    first_group = _group("@nestjs/core", "package.json")
    second_group = _group("@nestjs/common", "package.json")
    queue = _tasks(first_group, second_group)
    for task in queue.values():
        task.status = TaskStatus.PENDING

    plan = build_portfolio_plan(tmp_path, [first_group, second_group], queue)
    cluster_id, target_ids = _portfolio_cluster_targets(plan, queue, qa=False)

    assert cluster_id is not None
    assert set(target_ids) == set(queue)


def test_portfolio_order_dispatches_prerequisite_before_atomic_cluster(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "root",
            "dependencies": {
                "zeta-package": "1.0.0",
                "alpha-package": "1.0.0",
                "beta-package": "1.0.0",
            },
        },
    )
    prerequisite_group = _group("zeta-package", "package.json", cve_id="CVE-2026-10001")
    first_atomic_group = _group(
        "alpha-package",
        "package.json",
        cve_id="CVE-2026-10002",
    )
    second_atomic_group = _group(
        "beta-package",
        "package.json",
        cve_id="CVE-2026-10003",
    )
    groups = [prerequisite_group, first_atomic_group, second_atomic_group]
    queue = _tasks(*groups)
    for task in queue.values():
        task.status = TaskStatus.PENDING

    original_plan = build_portfolio_plan(tmp_path, groups, queue)
    clusters_by_task = {
        task_id: next(cluster for cluster in original_plan.clusters if task_id in cluster.task_ids)
        for task_id in queue
    }
    atomic_cluster = TaskCluster(
        cluster_id="atomic-cluster",
        task_ids=["task-2", "task-3"],
        dependencies=[
            TaskDependency(
                upstream_task_id="task-1",
                downstream_task_id=task_id,
                edge_type=TaskDependencyKind.PEER,
            )
            for task_id in ("task-2", "task-3")
        ],
        reason="test package cluster",
        atomic=True,
        dispatchable=True,
    )
    task_to_cluster = dict(original_plan.task_to_cluster)
    task_to_cluster.update(
        {"task-2": atomic_cluster.cluster_id, "task-3": atomic_cluster.cluster_id}
    )
    plan = original_plan.model_copy(
        update={
            "clusters": [clusters_by_task["task-1"], atomic_cluster],
            "cluster_order": [clusters_by_task["task-1"].cluster_id, atomic_cluster.cluster_id],
            "task_order": ["task-1", "task-2", "task-3"],
            "task_to_cluster": task_to_cluster,
        }
    )
    group_by_id = {group.group_id: group for group in groups}

    prerequisite_decision = _deterministic_routing(
        queue,
        group_by_id,
        {},
        {},
        portfolio_plan=plan,
    )

    assert prerequisite_decision.decision_code.value == "NEW_VERSION_BUMP"
    assert prerequisite_decision.target_task_ids == ["task-1"]
    assert prerequisite_decision.cluster_id is None

    completed_queue = {
        task_id: task.model_copy(
            update={"status": TaskStatus.QA_PASSED} if task_id == "task-1" else {}
        )
        for task_id, task in queue.items()
    }
    atomic_decision = _deterministic_routing(
        completed_queue,
        group_by_id,
        {},
        {},
        portfolio_plan=plan,
    )

    assert atomic_decision.decision_code.value == "ATOMIC_CLUSTER_DISPATCH"
    assert atomic_decision.cluster_id == atomic_cluster.cluster_id
    assert set(atomic_decision.target_task_ids) == {"task-2", "task-3"}


def test_supervisor_commits_shared_atomic_provenance_for_cluster(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    first_group = _group("@example/first", "package.json")
    second_group = _group("@example/second", "package.json")
    groups = [first_group, second_group]
    queue = _tasks(*groups)
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "task_queue": queue,
            "active_target_task_ids": [],
            "portfolio_plan": SimpleNamespace(portfolio_plan_id="test-plan"),
            "status": "workspace_ready",
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._deterministic_routing",
        lambda *args, **kwargs: SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            cluster_id="cluster-1",
            instructions="dispatch cluster",
            decision_reason="test atomic dispatch",
        ),
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "update_subagent"
    assert result["active_cluster_id"] == "cluster-1"
    assert result["active_multi_package_action"] is not None
    assert set(result["active_target_task_ids"]) == {"task-1", "task-2"}
    snapshots = [
        result["attempt_snapshots_by_id"][task.current_attempt_id]
        for task in result["task_queue"].values()
        if task.current_attempt_id
    ]
    assert {snapshot.cluster_id for snapshot in snapshots} == {"cluster-1"}
    assert len({snapshot.dispatch_batch_id for snapshot in snapshots}) == 1
    assert len({snapshot.action_digest for snapshot in snapshots}) == 1
    assert {snapshot.portfolio_plan_id for snapshot in snapshots} == {"test-plan"}


def test_invalid_atomic_action_requests_portfolio_replan(tmp_path: Path, monkeypatch):
    _write_manifest(tmp_path, "package.json", {"name": "root"})
    first_group = _group("@example/first", "package.json")
    second_group = _group("@example/second", "package.json")
    groups = [first_group, second_group]
    queue = _tasks(*groups)
    queue["task-1"] = queue["task-1"].model_copy(update={"selected_version": None})
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "task_queue": queue,
            "active_target_task_ids": [],
            "portfolio_plan": SimpleNamespace(portfolio_plan_id="test-plan"),
            "status": "workspace_ready",
        }
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "remediation_engine.orchestration.supervisor_node._deterministic_routing",
        lambda *args, **kwargs: SupervisorDecision(
            next_node="update_subagent",
            target_task_ids=["task-1", "task-2"],
            cluster_id="cluster-1",
            instructions="dispatch cluster",
            decision_reason="test atomic dispatch",
        ),
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "portfolio"
    assert result["active_target_task_ids"] == []
    assert result["active_cluster_id"] is None
    assert result["active_multi_package_action"] is None
    assert result["portfolio_dirty"] is True
    assert any("requesting a portfolio replan" in error for error in result["errors"])


def test_supervisor_preserves_all_cluster_targets_through_commit(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "root",
            "dependencies": {"@nestjs/core": "1.0.0", "@nestjs/common": "1.0.0"},
        },
    )
    first_group = _group("@nestjs/core", "package.json")
    second_group = _group("@nestjs/common", "package.json")
    queue = _tasks(first_group, second_group)
    prepared_groups, prepared_queue, prepare_diagnostics = prepare_portfolio_inputs(
        tmp_path,
        [first_group, second_group],
        queue,
    )
    assert prepare_diagnostics == []
    plan = build_portfolio_plan(tmp_path, prepared_groups, prepared_queue)
    plan = _attach_test_resolution_certificate(plan)
    committed_groups, committed_queue, apply_diagnostics = apply_portfolio_plan(
        plan,
        prepared_groups,
        prepared_queue,
    )
    assert apply_diagnostics == []
    state = initial_orchestrator_state(str(tmp_path), committed_groups)
    state.update(
        {
            "task_queue": committed_queue,
            "portfolio_plan": plan,
            "portfolio_dirty": False,
            "status": "workspace_ready",
        }
    )

    result = run_supervisor_node(state)

    assert result["next_routing_step"] == "update_subagent"
    assert result["active_cluster_id"] == plan.clusters[0].cluster_id
    assert set(result["active_target_task_ids"]) == set(queue)
    assert result["active_multi_package_action"] is not None
    assert {
        mutation.task_id for mutation in result["active_multi_package_action"].package_mutations
    } == set(queue)


def test_certified_plan_continues_after_breaking_change_workaround_pivot(tmp_path: Path):
    from remediation_engine.contracts.schemas import (
        FailureCategory,
        QAAttemptResult,
        QAEvaluation,
        QAPolicy,
        RoutingStrategy,
        SCARemediationStage,
        TaskAttemptSnapshot,
        UpdateRetryDiagnostics,
    )
    from remediation_engine.orchestration.supervisor_planner import instruction_digest

    groups = [
        _group("express-jwt", "package.json", package_version="0.1.3", fixed_version="7.7.8"),
        _group("companion-package", "package.json", fixed_version="2.0.0"),
    ]
    task_queue = _tasks(*groups)
    portfolio_plan_id = "certified-portfolio"
    package_specs = [
        ("task-1", groups[0], "7.7.8"),
        ("task-2", groups[1], "2.0.0"),
    ]
    selected_versions: dict[str, str] = {}
    decisions = []
    batches = []
    clusters = []
    task_revisions: dict[str, int] = {}
    task_strategies = {}
    cluster_order = []
    task_to_cluster = {}

    for task_id, group, version in package_specs:
        package_name = group.vulnerable_component
        occurrence_id = make_occurrence_id("package.json", package_name)
        selected_versions[occurrence_id] = version
        task_revisions[task_id] = 1
        task_strategies[task_id] = RoutingStrategy.VERSION_BUMP
        task_queue[task_id] = task_queue[task_id].model_copy(
            update={
                "task_revision": 1,
                "portfolio_plan_id": portfolio_plan_id,
                "strategy": RoutingStrategy.VERSION_BUMP,
                "strategy_stage": SCARemediationStage.OSV_MINIMUM,
                "status": TaskStatus.PENDING,
                "target_package_name": package_name,
                "target_dependency_type": "dependencies",
                "selected_version": version,
                "allowed_target_versions": [version],
                "allowed_dependency_types": ["dependencies"],
                "instruction": f"Update {package_name} to {version}.",
            }
        )
        decisions.append(
            SimpleNamespace(
                task_id=task_id,
                selected_strategy=RoutingStrategy.VERSION_BUMP,
                target_group_id=group.group_id,
                target_package_name=package_name,
                manifest_path="package.json",
                lockfile_package_key=f"node_modules/{package_name}",
                target_occurrence_id=occurrence_id,
                dependency_type="dependencies",
                strategy_stage=SCARemediationStage.OSV_MINIMUM,
                selected_version=version,
                allowed_alternative_versions=[],
                allowed_dependency_types=["dependencies"],
            )
        )
        batches.append(
            SimpleNamespace(
                resolved_coverage_ids=[],
                workaround_coverage_ids=[],
                unresolved_coverage_ids=[],
            )
        )
        cluster_id = f"cluster-{task_id}"
        cluster_order.append(cluster_id)
        task_to_cluster[task_id] = cluster_id
        clusters.append(
            SimpleNamespace(
                cluster_id=cluster_id,
                task_ids=[task_id],
                dependencies=[],
                dispatchable=True,
                atomic=False,
                reason="singleton solver task",
            )
        )

    candidate_plan_id = "candidate-1"
    assignment_digest = hashlib.sha256(
        json.dumps(
            dict(sorted(selected_versions.items())),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    selected_plan = SimpleNamespace(
        candidate_plan_id=candidate_plan_id,
        selected_candidate_versions=selected_versions,
        task_decisions=decisions,
        batches=batches,
    )
    solver_plan = SimpleNamespace(
        status="OPTIMAL",
        candidate_catalog_complete=True,
        candidate_catalog_digest="candidate-catalog",
        selected_plan=selected_plan,
    )
    plan = SimpleNamespace(
        plan_id=portfolio_plan_id,
        portfolio_plan_id=portfolio_plan_id,
        solver_plan=solver_plan,
        resolution_certificate=SimpleNamespace(
            status=PackageResolutionStatus.CERTIFIED,
            portfolio_plan_id=portfolio_plan_id,
            candidate_plan_id=candidate_plan_id,
            solver_input_digest="solver-input",
            repository_fingerprint="repository",
            workspace_graph_digest="workspace",
            candidate_catalog_digest="candidate-catalog",
            candidate_assignment_digest=assignment_digest,
            task_revisions=task_revisions,
            covered_coverage_ids=[],
            workaround_coverage_ids=[],
            unresolved_coverage_ids=[],
        ),
        solver_input_digest="solver-input",
        repository_fingerprint="repository",
        workspace_graph_digest="workspace",
        task_revisions=task_revisions,
        planned_task_revisions=task_revisions,
        task_ids=[task_id for task_id, _group, _version in package_specs],
        task_strategies=task_strategies,
        clusters=clusters,
        cluster_order=cluster_order,
        task_order=[task_id for task_id, _group, _version in package_specs],
        task_to_cluster=task_to_cluster,
        diagnostics=[],
    )

    stale_stage_tasks = dict(task_queue)
    stale_stage_tasks["task-2"] = stale_stage_tasks["task-2"].model_copy(
        update={"strategy_stage": SCARemediationStage.NPM_LATEST, "selected_version": None}
    )
    assert "task task-2 strategy stage differs from solver decision" in (
        _portfolio_plan_violations(plan, stale_stage_tasks, groups)
    )

    attempted_version = task_queue["task-1"].selected_version
    instruction = task_queue["task-1"].instruction
    attempt_id = "attempt-express-jwt"
    task_queue["task-1"] = task_queue["task-1"].model_copy(
        update={
            "status": TaskStatus.OPTIMISTICALLY_FIXED,
            "task_revision": 2,
            "retry_count": 0,
            "current_attempt_id": attempt_id,
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id=attempt_id,
        task_id="task-1",
        state_revision=1,
        task_revision=2,
        strategy_stage=SCARemediationStage.OSV_MINIMUM,
        qa_policy=QAPolicy.VERSION_BUMP,
        selected_version=attempted_version,
        instruction=instruction,
        instruction_digest=instruction_digest(instruction),
        dispatch_node="update_subagent",
        portfolio_plan_id=portfolio_plan_id,
    )
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The selected API breaks the existing application.",
    )
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "repo_root": None,
            "status": "qa_completed",
            "portfolio_dirty": False,
            "portfolio_replan_request": None,
            "task_queue": task_queue,
            "portfolio_plan": plan,
            "portfolio_solver_plan": solver_plan,
            "active_target_task_ids": ["task-1"],
            "attempt_snapshots_by_id": {attempt_id: snapshot},
            "qa_results_by_attempt": {
                attempt_id: QAAttemptResult(
                    attempt_id=attempt_id,
                    task_id="task-1",
                    task_revision=2,
                    qa_policy=QAPolicy.VERSION_BUMP,
                    qa_policy_source="attempt_snapshot",
                    evaluation=evaluation,
                )
            },
            "retry_diagnostics_by_task": {
                "task-1": UpdateRetryDiagnostics(
                    task_id="task-1",
                    strategy_stage=SCARemediationStage.OSV_MINIMUM,
                    attempted_versions=[attempted_version],
                    attempted_versions_by_target={"express-jwt": [attempted_version]},
                    candidate_versions_considered=[attempted_version],
                    latest_version_seen=attempted_version,
                    target_package_name="express-jwt",
                    target_dependency_type="dependencies",
                    candidate_dependency_types=["dependencies"],
                )
            },
        }
    )

    pivot = run_supervisor_node(state)

    assert pivot["next_routing_step"] == "workaround_subagent"
    assert pivot["portfolio_dirty"] is False
    assert pivot["portfolio_replan_request"] is None
    parent = pivot["task_queue"]["task-1"]
    child = next(task for task in pivot["task_queue"].values() if task.parent_task_id == "task-1")
    assert parent.status == TaskStatus.PIVOTED
    assert parent.portfolio_plan_id == portfolio_plan_id
    assert parent.selected_version is None
    assert child.strategy == RoutingStrategy.CODE_WORKAROUND
    assert pivot["active_target_task_ids"] == [child.task_id]
    assert selected_plan.selected_candidate_versions == selected_versions

    resumed_state = dict(state)
    resumed_state.update(pivot)
    resumed_tasks = dict(pivot["task_queue"])
    resumed_tasks[child.task_id] = child.model_copy(
        update={"status": TaskStatus.QA_PASSED, "current_attempt_id": None}
    )
    resumed_state.update(
        {
            "status": "supervisor_entered",
            "active_target_task_ids": [],
            "task_queue": resumed_tasks,
            "qa_evaluations": {},
        }
    )

    resumed = run_supervisor_node(resumed_state)

    assert resumed["next_routing_step"] == "update_subagent"
    assert resumed["active_target_task_ids"] == ["task-2"]
    assert (
        resumed["task_queue"]["task-2"].selected_version
        == selected_versions[make_occurrence_id("package.json", "companion-package")]
    )
    assert resumed["task_queue"]["task-2"].portfolio_plan_id == portfolio_plan_id
    assert selected_plan.selected_candidate_versions == selected_versions


def test_peer_parser_preserves_quoted_ranges_and_scoped_packages():
    evidence = parse_peer_conflict_evidence(
        "npm error Found: react@18.2.0\n"
        'npm error peer react@"^18.0.0 || ^19.0.0" from @angular/core@17.0.0',
        "",
    )

    assert len(evidence) == 1
    assert evidence[0].peer_package == "react"
    assert evidence[0].requester_package == "@angular/core"
    assert evidence[0].requester_version == "17.0.0"
    assert evidence[0].required_range == "^18.0.0 || ^19.0.0"
    assert evidence[0].observed_version == "18.2.0"


def test_apply_rejects_a_mismatched_physical_lockfile_occurrence(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {"dependencies": {"lodash": "1.0.0"}},
    )
    group = _group("lodash", "package.json")
    queue = _tasks(group)
    plan = build_portfolio_plan(tmp_path, [group], queue)
    selected = plan.solver_plan.selected_plan
    assert selected is not None
    decision = selected.task_decisions[0].model_copy(
        update={"lockfile_package_key": "node_modules/other-package"}
    )
    tampered_selected = selected.model_copy(update={"task_decisions": [decision]})
    tampered_solver = plan.solver_plan.model_copy(update={"selected_plan": tampered_selected})
    tampered_plan = plan.model_copy(update={"solver_plan": tampered_solver})

    with pytest.raises(ValueError, match="lockfile_package_key"):
        apply_portfolio_plan(tampered_plan, [group], queue)


def test_apply_maps_solver_no_fix_to_code_workaround(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"dependencies": {"lodash": "1.0.0"}})
    group = _group("lodash", "package.json")
    queue = _tasks(group)
    task = queue["task-1"]
    decision = SimpleNamespace(
        task_id="task-1",
        selected_strategy="no_fix",
        selected_version=None,
        allowed_alternative_versions=[],
        allowed_dependency_types=[],
        selected_plan_issue_ids=[],
        dependency_type=None,
        strategy_stage="npm_latest",
        exact_instruction=None,
        target_occurrence_id=make_occurrence_id("package.json", "lodash"),
        target_group_id=group.group_id,
        target_package_name="lodash",
        manifest_path="package.json",
        lockfile_package_key="node_modules/lodash",
    )
    plan = SimpleNamespace(
        plan_id="portfolio-no-fix",
        portfolio_plan_id="portfolio-no-fix",
        task_ids=["task-1"],
        task_revisions={"task-1": task.task_revision},
        solver_plan=SimpleNamespace(
            selected_plan=SimpleNamespace(task_decisions=[decision]),
        ),
    )

    _groups, committed, diagnostics = apply_portfolio_plan(plan, [group], queue)

    assert diagnostics == []
    assert committed["task-1"].strategy.value == "code_workaround"


def test_apply_synthesizes_major_migration_instruction_when_missing(tmp_path: Path):
    _write_manifest(tmp_path, "package.json", {"dependencies": {"lodash": "8.5.1"}})
    group = _group("lodash", "package.json")
    queue = _tasks(group)
    task = queue["task-1"]
    decision = SimpleNamespace(
        task_id="task-1",
        selected_strategy="version_bump",
        selected_version="9.0.2",
        allowed_alternative_versions=[],
        allowed_dependency_types=["dependencies"],
        selected_plan_issue_ids=[],
        dependency_type="dependencies",
        strategy_stage="osv_minimum",
        exact_instruction=None,
        requires_source_migration=True,
        installed_version="8.5.1",
        target_occurrence_id=make_occurrence_id("package.json", "lodash"),
        target_group_id=group.group_id,
        target_package_name="lodash",
        manifest_path="package.json",
        lockfile_package_key="node_modules/lodash",
    )
    plan = SimpleNamespace(
        plan_id="portfolio-major-upgrade",
        portfolio_plan_id="portfolio-major-upgrade",
        task_ids=["task-1"],
        task_revisions={"task-1": task.task_revision},
        solver_plan=SimpleNamespace(
            selected_plan=SimpleNamespace(task_decisions=[decision]),
        ),
    )

    _groups, committed, diagnostics = apply_portfolio_plan(plan, [group], queue)

    assert diagnostics == []
    assert "8.5.1" in committed["task-1"].instruction
    assert "9.0.2" in committed["task-1"].instruction
    assert (
        "Migrate all affected production and test code to the selected package API "
        "while preserving behavior; do not change the solver-approved package or version."
        in committed["task-1"].instruction
    )


def test_supervisor_rejects_stale_resolution_certificate():
    assignment_digest = hashlib.sha256(
        json.dumps({}, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    selected = SimpleNamespace(
        candidate_plan_id="candidate-plan",
        selected_candidate_versions={},
        task_decisions=[],
        batches=[],
    )
    solver_plan = SimpleNamespace(
        status="OPTIMAL",
        candidate_catalog_complete=True,
        candidate_catalog_digest="candidate-catalog-digest",
        selected_plan=selected,
    )
    certificate = SimpleNamespace(
        status="CERTIFIED",
        portfolio_plan_id="portfolio-plan",
        candidate_plan_id="candidate-plan",
        solver_input_digest="solver-input-digest",
        repository_fingerprint="repository-fingerprint",
        workspace_graph_digest="stale-workspace-graph",
        candidate_catalog_digest="candidate-catalog-digest",
        candidate_assignment_digest=assignment_digest,
        task_revisions={},
        covered_coverage_ids=[],
        workaround_coverage_ids=[],
        unresolved_coverage_ids=[],
    )
    plan = SimpleNamespace(
        portfolio_plan_id="portfolio-plan",
        plan_id="portfolio-plan",
        solver_plan=solver_plan,
        resolution_certificate=certificate,
        solver_input_digest="solver-input-digest",
        repository_fingerprint="repository-fingerprint",
        workspace_graph_digest="workspace-graph",
        task_revisions={},
        task_ids=[],
        planned_task_revisions={},
        task_strategies={},
    )
    certificate.workspace_graph_digest = "workspace-graph"
    assert _portfolio_plan_violations(plan, {}, []) == []
    assert _portfolio_certificate_violations(plan, {}) == []
    certificate.candidate_plan_id = "stale-candidate-plan"
    assert _portfolio_certificate_violations(plan, {}) == [
        "package-resolution certificate candidate plan ID is stale"
    ]
    assert _portfolio_plan_violations(plan, {}, []) == [
        "package-resolution certificate candidate plan ID is stale"
    ]
    certificate.candidate_plan_id = "candidate-plan"
    certificate.workspace_graph_digest = "stale-workspace-graph"

    violations = _portfolio_plan_violations(plan, {}, [])

    assert violations == [
        "package-resolution certificate workspace_graph_digest differs from committed plan"
    ]


def test_scoped_transitive_findings_become_override_solver_tasks(tmp_path: Path):
    _write_manifest(
        tmp_path,
        "package.json",
        {
            "name": "app",
            "dependencies": {
                "express-jwt": "1.0.0",
                "jsonwebtoken": "1.0.0",
            },
        },
    )
    _write_manifest(
        tmp_path,
        "package-lock.json",
        {
            "name": "app",
            "lockfileVersion": 3,
            "packages": {
                "": {"name": "app"},
                "node_modules/express-jwt": {
                    "version": "1.0.0",
                    "dependencies": {
                        "jsonwebtoken": "1.0.0",
                        "moment": "1.0.0",
                    },
                },
                "node_modules/jsonwebtoken": {
                    "version": "1.0.0",
                    "dependencies": {"jws": "1.0.0"},
                },
                "node_modules/jws": {
                    "version": "1.0.0",
                    "dependencies": {"base64url": "1.0.0"},
                },
                "node_modules/base64url": {"version": "1.0.0"},
                "node_modules/express-jwt/node_modules/moment": {"version": "1.0.0"},
            },
        },
    )

    express_jwt = _group("express-jwt", "package.json")
    jsonwebtoken = _group("jsonwebtoken", "package.json")

    def transitive_group(package_name: str, ancestry: list[str]):
        group = _group(package_name, "package.json")
        localized = [
            item.model_copy(update={"is_direct_dependency": False, "declaration_type": None})
            for item in group.localized_issues
        ]
        return group.model_copy(
            update={
                "dependency_ancestry": ancestry,
                "localized_issues": localized,
                "versions": ["1.0.0"],
            }
        )

    base64url = transitive_group("base64url", ["jws", "base64url"])
    jws = transitive_group("jws", ["jws"])
    moment = transitive_group("moment", ["moment"])
    groups = [express_jwt, jsonwebtoken, base64url, jws, moment]
    target_packages = [
        "base64url",
        "express-jwt",
        "jsonwebtoken",
        "jws",
        "moment",
    ]

    prepared_groups, queue, diagnostics = prepare_portfolio_inputs(
        tmp_path,
        groups,
        _tasks(*groups),
        target_packages=target_packages,
    )

    assert diagnostics == []
    assert {group.vulnerable_component for group in prepared_groups} == set(target_packages)
    assert {task.parent_group_id for task in queue.values()} == {group.group_id for group in groups}
    override_tasks = {
        task.target_package_name: task
        for task in queue.values()
        if task.target_package_name in {"base64url", "jws", "moment"}
    }
    assert set(override_tasks) == {"base64url", "jws", "moment"}
    assert all(task.target_dependency_type == "overrides" for task in override_tasks.values())
    assert all(task.strategy_stage.value == "package_override" for task in override_tasks.values())

    plan = build_portfolio_plan(
        tmp_path,
        prepared_groups,
        queue,
        target_packages=target_packages,
    )
    selected = plan.solver_plan.selected_plan
    assert selected is not None, (
        plan.solver_plan.status,
        plan.solver_plan.diagnostics,
        plan.diagnostics,
    )
    assert set(plan.task_ids) == set(queue)
    snapshot = load_npm_graph_snapshot(tmp_path)
    expected_coverage_ids = {
        "coverage:"
        + hashlib.sha256(
            (_issue_identity(issue) + "\0" + occurrence.occurrence_id).encode("utf-8")
        ).hexdigest()
        for group in prepared_groups
        for issue in group.issues
        if issue.cve_id or issue.ghsa_id or issue.finding_id
        for occurrence in snapshot.occurrences
        if occurrence.manifest_path == (group.file_paths[0] if group.file_paths else "package.json")
        and occurrence.package_name == group.vulnerable_component
        and occurrence.installed_version == issue.package_version
    }
    assert set(selected.coverage_ids) == expected_coverage_ids
    assert selected.unresolved_ids == []

    decisions = {decision.task_id: decision for decision in selected.task_decisions}
    for package_name, task in override_tasks.items():
        decision = decisions[task.task_id]
        assert decision.target_package_name == package_name
        assert decision.dependency_type == "overrides"
        assert decision.lockfile_package_key
        assert decision.target_occurrence_id.endswith(f"::{decision.lockfile_package_key}")

    mutations = {
        mutation.package_name: mutation
        for batch in selected.batches
        for mutation in batch.mutations
    }
    for package_name in {"base64url", "jws", "moment"}:
        assert mutations[package_name].dependency_type == "overrides"
        assert mutations[package_name].target_version == "2.0.0"

    committed_groups, committed_queue, apply_diagnostics = apply_portfolio_plan(
        plan,
        prepared_groups,
        queue,
    )
    assert apply_diagnostics == []
    violations = _portfolio_plan_violations(plan, committed_queue, committed_groups)
    assert violations == ["portfolio plan is missing its package-resolution certificate"]
    assert len(committed_groups) == 5
    for package_name in {"base64url", "jws", "moment"}:
        task = next(
            task for task in committed_queue.values() if task.target_package_name == package_name
        )
        assert task.target_dependency_type == "overrides"
        assert task.selected_version == "2.0.0"
        assert task.portfolio_plan_id == plan.portfolio_plan_id
        assert '"overrides"' in task.instruction


def _suspicion_mutation(task_id: str, package_name: str, target_version: str):
    return PackageMutation(
        task_id=task_id,
        package_name=package_name,
        target_version=target_version,
        dependency_type="dependencies",
    )


def _suspicion_action(*mutations: PackageMutation) -> MultiPackageAction:
    return MultiPackageAction(
        cluster_id="cluster-test" if len(mutations) > 1 else None,
        selected_strategy="version_bump",
        package_mutations=list(mutations),
        rationale="deterministic suspicion scoring test",
    )


@pytest.mark.parametrize(
    ("package_name", "error_logs", "expected_score"),
    [
        ("lodash", "Failed while loading LODASH!", 10),
        ("foo", "foo foo foo", 10),
        ("foo", "xfoo foo-bar foo.bar foo_bar /node_modules/foo@1.2.3", 10),
        ("foo", "xfoo foo-bar foo.bar foo_bar", 0),
        ("name", "failure in @scope/name", 0),
        ("scope", "failure in @scope/name", 0),
        ("@Scope/Name", "failure in /node_modules/@SCOPE/NAME@2.0.0", 10),
    ],
)
def test_suspicion_package_match_requires_a_case_insensitive_complete_token(
    package_name, error_logs, expected_score
):
    assert score_suspect_package(package_name, "1.0.0", "1.0.0", error_logs) == expected_score


@pytest.mark.parametrize(
    ("target_version", "base_version", "expected_score"),
    [
        ("3.0.0", "2.9.9", 3),
        ("1.3.0", "1.2.9", 1),
        ("1.2.4", "1.2.3", 0),
        ("1.2.3", "1.2.3", 0),
        ("1.2.3", "1.3.0", 0),
        ("2.0.0", "3.0.0", 0),
        ("invalid", "1.0.0", 0),
        ("v2.0.0-rc.1", "=1.9.9+build.3", 3),
    ],
)
def test_suspicion_version_points_use_major_then_minor_only(
    target_version, base_version, expected_score
):
    assert score_suspect_package("pkg", target_version, base_version, "") == expected_score


def test_suspicion_ranking_orders_log_and_version_evidence_then_task_id():
    specifications = [
        ("task-error-major", "error-major", "3.0.0", "1.0.0"),
        ("task-error-minor", "error-minor", "1.3.0", "1.2.0"),
        ("task-major", "major-only", "2.0.0", "1.0.0"),
        ("task-minor", "minor-only", "1.3.0", "1.2.0"),
        ("task-patch", "patch-only", "1.2.4", "1.2.3"),
        ("task-tie-z", "tie-z", "1.2.4", "1.2.3"),
        ("task-tie-a", "tie-a", "1.2.4", "1.2.3"),
    ]
    tasks = []
    groups = {}
    mutations = []
    for task_id, package_name, target, base in specifications:
        group = _group(package_name, "package.json", package_version=base, fixed_version=target)
        task = build_initial_remediation_task(group, task_id)
        task.selected_version = "99.0.0"
        tasks.append(task)
        groups[group.group_id] = group
        mutations.append(_suspicion_mutation(task_id, package_name, target))

    ranked = rank_suspect_tasks_by_suspicion(
        tasks,
        _suspicion_action(*mutations),
        "ERROR-MAJOR failed; error-minor failed",
        groups,
    )

    assert [task.task_id for task, _ in ranked] == [
        "task-error-major",
        "task-error-minor",
        "task-major",
        "task-minor",
        "task-patch",
        "task-tie-a",
        "task-tie-z",
    ]
    assert [score for _, score in ranked] == [13, 11, 3, 1, 0, 0, 0]


def test_suspicion_ranking_prefers_present_solver_baselines_over_group_fallback():
    group = _group("pkg", "package.json", package_version="1.9.0", fixed_version="2.1.0")
    task = build_initial_remediation_task(group, "task-pkg")
    task.selected_version = "99.0.0"
    action = _suspicion_action(_suspicion_mutation("task-pkg", "pkg", "2.1.0"))
    groups = {group.group_id: group}

    exact_solver_baseline = rank_suspect_tasks_by_suspicion(
        [task], action, "", groups, {"task-pkg": "2.0.0"}
    )
    unavailable_solver_baseline = rank_suspect_tasks_by_suspicion(
        [task], action, "", groups, {"task-pkg": "unknown"}
    )
    absent_solver_baseline = rank_suspect_tasks_by_suspicion(
        [task], action, "", groups, {"task-pkg": None}
    )
    group_fallback = rank_suspect_tasks_by_suspicion([task], action, "", groups)

    assert exact_solver_baseline[0][1] == 1
    assert unavailable_solver_baseline[0][1] == 0
    assert absent_solver_baseline[0][1] == 0
    assert group_fallback[0][1] == 3


def test_suspicion_ranking_uses_only_exact_unambiguous_group_fallbacks():
    parent_group = _group("child", "package.json", package_version="1.0.0")
    parent_group.dependency_versions = {"parent": "2.0.0"}
    parent_task = build_initial_remediation_task(parent_group, "task-parent")
    parent_task.parent_package_name = "parent"
    parent_task.parent_package_version = "1.9.0"
    parent_action = _suspicion_action(_suspicion_mutation("task-parent", "parent", "2.1.0"))
    parent_score = rank_suspect_tasks_by_suspicion(
        [parent_task],
        parent_action,
        "",
        {parent_group.group_id: parent_group},
    )[0][1]

    dependency_group = _group("child", "package.json", package_version="1.0.0")
    dependency_group.dependency_versions = {"child": "2.0.0", "other": "1.0.0"}
    dependency_task = build_initial_remediation_task(dependency_group, "task-dependency")
    dependency_score = rank_suspect_tasks_by_suspicion(
        [dependency_task],
        _suspicion_action(_suspicion_mutation("task-dependency", "child", "2.1.0")),
        "",
        {dependency_group.group_id: dependency_group},
    )[0][1]

    context_group = _group("child", "package.json", package_version="1.0.0").model_copy(
        update={
            "parent_package_name": None,
            "parent_package_version": None,
            "parent_contexts": [
                DependencyParentContext(package_name="parent", package_version="2.0.0")
            ],
        }
    )
    context_task = build_initial_remediation_task(context_group, "task-context")
    context_task.parent_package_name = None
    context_task.parent_package_version = None
    context_score = rank_suspect_tasks_by_suspicion(
        [context_task],
        _suspicion_action(_suspicion_mutation("task-context", "parent", "2.1.0")),
        "",
        {context_group.group_id: context_group},
    )[0][1]

    unique_group = _group("child", "package.json", package_version="1.0.0")
    unique_task = build_initial_remediation_task(unique_group, "task-unique")
    unique_score = rank_suspect_tasks_by_suspicion(
        [unique_task],
        _suspicion_action(_suspicion_mutation("task-unique", "child", "2.0.0")),
        "",
        {unique_group.group_id: unique_group},
    )[0][1]
    ambiguous_group = unique_group.model_copy(update={"versions": ["1.0.0", "1.1.0"]})
    ambiguous_task = build_initial_remediation_task(ambiguous_group, "task-ambiguous")
    ambiguous_score = rank_suspect_tasks_by_suspicion(
        [ambiguous_task],
        _suspicion_action(_suspicion_mutation("task-ambiguous", "child", "2.0.0")),
        "",
        {ambiguous_group.group_id: ambiguous_group},
    )[0][1]
    missing_group_task = build_initial_remediation_task(unique_group, "task-missing-group")
    missing_group_score = rank_suspect_tasks_by_suspicion(
        [missing_group_task],
        _suspicion_action(_suspicion_mutation("task-missing-group", "child", "2.0.0")),
        "",
    )[0][1]

    assert parent_score == 3
    assert dependency_score == 1
    assert context_score == 1
    assert unique_score == 3
    assert ambiguous_score == 0
    assert missing_group_score == 0


def test_suspicion_ranking_scores_missing_or_duplicate_mutations_as_zero():
    group = _group("pkg", "package.json", package_version="1.0.0", fixed_version="2.0.0")
    task = build_initial_remediation_task(group, "task-pkg")
    missing_mutation = rank_suspect_tasks_by_suspicion(
        [task],
        _suspicion_action(_suspicion_mutation("other-task", "other", "2.0.0")),
        "pkg failed",
        {group.group_id: group},
    )
    duplicate_mutation = rank_suspect_tasks_by_suspicion(
        [task],
        _suspicion_action(
            _suspicion_mutation("task-pkg", "pkg", "2.0.0"),
            _suspicion_mutation("task-pkg", "other", "2.0.0"),
        ),
        "pkg failed",
        {group.group_id: group},
    )

    assert missing_mutation == [(task, 0)]
    assert duplicate_mutation == [(task, 0)]
