"""Deterministic package-manager certification tests without live services."""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Self

import pytest

from remediation_engine.contracts.schemas import (
    CommandResult,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    PackageMutation,
    PeerConflictEvidence,
    RemediationTask,
    Severity,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionStatus,
    PortfolioReplanRequest,
    SolverCandidateCutKind,
    SolverCandidateRejectionReason,
    SolverCandidateRelation,
    SolverFindingRequirement,
    SolverRuntimeFingerprint,
    SolverStatus,
    SolverTarget,
)
from remediation_engine.orchestration.portfolio_certifier import (
    _assert_workspace_matches_host,
    _AssignmentRejected,
    _AssignmentResult,
    _authorized_workaround,
    _candidate_conflict,
    _CertificationUnknown,
    _last_qa_passed_prefix_digest,
    _peer_conflict_literal_ids,
    _peer_conflict_rejection,
    _read_npm_documents,
    _validate_candidate_relations,
    _validate_candidate_runtime,
    _verify_staged_mutations,
    build_certified_portfolio_plan,
)
from remediation_engine.orchestration.portfolio_solver import (
    _build_evidence_domains,
    _build_targets_and_findings,
    _digest,
    _prepare_portfolio_problem,
)
from remediation_engine.orchestration.qa_test_parsing import (
    _install_error_category,
    parse_peer_conflict_evidence,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import solve_portfolio
from remediation_engine.tools.npm_graph import (
    load_npm_graph_snapshot,
    load_npm_graph_snapshot_from_documents,
    make_occurrence_id,
    resolve_lockfile_dependency_package,
)
from remediation_engine.triage.grouper import group_issues


class _FakeSandboxState:
    def __init__(
        self,
        files: dict[str, str],
        *,
        install_timeout: bool = False,
        raise_on_extract: bool = False,
        cleanup_failures: int = 0,
        invalid_archive: bool = False,
        tamper_archive: bool = False,
        malformed_resolved_lockfile: bool = False,
    ) -> None:
        self.files = dict(files)
        self.install_timeout = install_timeout
        self.raise_on_extract = raise_on_extract
        self.cleanup_failures = cleanup_failures
        self.tamper_archive = tamper_archive
        self.invalid_archive = invalid_archive
        self.malformed_resolved_lockfile = malformed_resolved_lockfile
        self.archive: dict[str, str] | None = None
        self.commands: list[str] = []
        self.install_assignments: list[tuple[str, str]] = []
        self.cleanup_count = 0
        self.archive_create_count = 0
        self.extraction_paths: list[str] = []
        self.batch_read_calls = 0


class _FakeSandbox:
    def __init__(self, state: _FakeSandboxState) -> None:
        self.state = state

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def cleanup_workspace_snapshots(self) -> None:
        return None

    def read_file(self, file_path: str) -> str | None:
        path = str(file_path).replace("\\", "/").strip("/")
        if path.startswith("workspace/"):
            path = path[len("workspace/") :]
        return self.state.files.get(path)

    def read_files(self, paths: list[str]) -> dict[str, str] | None:
        self.state.batch_read_calls += 1
        documents: dict[str, str] = {}
        for file_path in paths:
            content = self.read_file(file_path)
            if content is None:
                return None
            path = str(file_path).replace("\\", "/").strip("/")
            if path.startswith("workspace/"):
                path = path[len("workspace/") :]
            documents[path] = content
        return documents

    def write_file(self, file_path: str, content: str) -> None:
        path = str(file_path).replace("\\", "/").strip("/")
        self.state.files[path] = content

    def _result(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> CommandResult:
        return CommandResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=0.001,
        )

    def run(self, command: str, timeout: int = 300) -> CommandResult:
        self.state.commands.append(command)
        if "node --version" in command:
            return self._result(
                stdout='v22.15.0\n10.9.2\n{"platform":"linux","architecture":"x64"}\n'
            )
        if command.startswith("find "):
            parts = shlex.split(command)
            root = parts[1].rstrip("/")
            prefix = "" if root == "/workspace" else root.removeprefix("/workspace/") + "/"
            compiled_roots = {
                name
                for name in ("build", "dist")
                if f"-path {shlex.quote(f'{root}/{name}')} -prune -o" in command
            }
            paths: list[str] = []
            for key in sorted(self.state.files):
                if not key.startswith(prefix):
                    continue
                relative = key[len(prefix) :]
                relative_parts = PurePosixPath(relative).parts
                if relative_parts and relative_parts[0] in compiled_roots:
                    continue
                if any(
                    name
                    in {"node_modules", ".git", ".remedy-attempt-snapshots", ".remedy-plan-cert"}
                    for name in relative_parts
                ):
                    continue
                if PurePosixPath(relative).name in {
                    "package.json",
                    "package-lock.json",
                    "npm-shrinkwrap.json",
                }:
                    paths.append(f"/workspace/{key}")
            return self._result(stdout="\n".join(paths) + ("\n" if paths else ""))
        if command.startswith("mkdir -p -- /workspace/.remedy-plan-cert"):
            return self._result()
        if command.startswith("tar -cf "):
            parts = shlex.split(command)
            self.state.archive = {
                path: content
                for path, content in self.state.files.items()
                if not any(
                    name
                    in {"node_modules", ".git", ".remedy-attempt-snapshots", ".remedy-plan-cert"}
                    for name in PurePosixPath(path).parts
                )
            }
            self.state.archive_create_count += 1
            self.state.archive_path = parts[2]
            return self._result()
        if command.startswith("wc -c < "):
            return self._result(
                stdout=f"{sum(len(content.encode('utf-8')) for content in (self.state.archive or {}).values())}\n"
            )
        if command.startswith("test -s "):
            if self.state.invalid_archive:
                return self._result(1, stderr="synthetic archive validation failure")
            if self.state.tamper_archive and self.state.archive is not None:
                manifest = json.loads(self.state.archive["package.json"])
                manifest["dependencies"]["jsonwebtoken"] = "99.0.0"
                self.state.archive["package.json"] = json.dumps(manifest, sort_keys=True) + "\n"
            return self._result(0 if self.state.archive is not None else 1)
        if command.startswith("tar -xf "):
            if self.state.raise_on_extract:
                raise RuntimeError("synthetic scratch extraction error")
            parts = shlex.split(command)
            scratch_path = parts[parts.index("-C") + 1].removeprefix("/workspace/").rstrip("/")
            self.state.extraction_paths.append(scratch_path)
            if self.state.archive is None:
                return self._result(1, stderr="archive missing")
            for path, content in self.state.archive.items():
                self.state.files[f"{scratch_path}/{path}"] = content
            return self._result()
        if command.startswith("rm -rf -- "):
            self.state.cleanup_count += 1
            if self.state.cleanup_failures > 0:
                self.state.cleanup_failures -= 1
                return self._result(1, stderr="synthetic cleanup failure")
            for path in shlex.split(command)[3:]:
                if path.startswith("/workspace/"):
                    relative = path.removeprefix("/workspace/").rstrip("/")
                    self.state.files = {
                        key: value
                        for key, value in self.state.files.items()
                        if key != relative and not key.startswith(relative + "/")
                    }
                elif path.startswith("/tmp/"):
                    self.state.archive = None
            return self._result()
        if "npm pkg set" in command:
            parts = shlex.split(command)
            workdir = parts[1].removeprefix("/workspace/").rstrip("/")
            manifest_path = f"{workdir}/package.json"
            payload = json.loads(self.state.files[manifest_path])
            expression = parts[-1]
            left, version = expression.split("=", 1)
            section, package_name = left.split("[", 1)
            package_name = package_name.rstrip("]")
            current: Any = payload
            for segment in section.split("."):
                current = current.setdefault(segment, {})
            current[package_name] = version
            self.state.files[manifest_path] = json.dumps(payload, sort_keys=True) + "\n"
            return self._result()
        if "npm install --package-lock-only" in command:
            if self.state.install_timeout:
                return self._result(124, stderr="synthetic npm timeout")
            parts = shlex.split(command)
            workdir = parts[1].removeprefix("/workspace/").rstrip("/")
            prefix = f"{workdir}/" if workdir else ""
            manifest_path = f"{prefix}package.json"
            manifest = json.loads(self.state.files[manifest_path])
            dependencies = manifest.get("dependencies", {})
            express_version = dependencies.get("express-jwt", "6.1.1")
            jwt_version = dependencies.get("jsonwebtoken", "9.0.2")
            self.state.install_assignments.append((express_version, jwt_version))
            jwt_range = "^8.1.0" if express_version == "6.1.1" else "^9.0.0"
            packages: dict[str, dict[str, Any]] = {
                "": {"name": "app", "dependencies": dict(dependencies)}
            }
            if "express-jwt" in dependencies:
                packages["node_modules/express-jwt"] = {
                    "version": express_version,
                    "dependencies": {"jsonwebtoken": jwt_range},
                }
            if "jsonwebtoken" in dependencies:
                packages["node_modules/jsonwebtoken"] = {"version": jwt_version}
            for package_name, version in sorted(dependencies.items()):
                if package_name in {"express-jwt", "jsonwebtoken"}:
                    continue
                packages[f"node_modules/{package_name}"] = {"version": str(version)}
            if "express-jwt" in dependencies and express_version == "6.1.1":
                packages["node_modules/express-jwt/node_modules/jsonwebtoken"] = {
                    "version": "8.5.1"
                }
            lockfile = {"name": "app", "lockfileVersion": 3, "packages": packages}
            lockfile_path = f"{prefix}package-lock.json"
            if self.state.malformed_resolved_lockfile:
                self.state.files[lockfile_path] = "{malformed json\n"
            else:
                self.state.files[lockfile_path] = json.dumps(lockfile, sort_keys=True) + "\n"
            return self._result()
        return self._result(1, stderr=f"unexpected command: {command}")


class _FakeSandboxFactory:
    def __init__(self, state: _FakeSandboxState) -> None:
        self.state = state
        self.instances: list[_FakeSandbox] = []

    def __call__(self, *, repo_root=None, workspace_volume=None) -> _FakeSandbox:
        assert repo_root is None
        assert workspace_volume
        sandbox = _FakeSandbox(self.state)
        self.instances.append(sandbox)
        return sandbox


def _write_fixture(root: Path, relative: str, payload: dict[str, Any]) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    return content


def _jsonwebtoken_group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.SYNTHETIC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-70001",
        package_name="jsonwebtoken",
        package_version="8.5.1",
        file_path="package.json",
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
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="9.0.0",
        instruction="Upgrade jsonwebtoken to the fixed release.",
        strategy_used="osv_api",
    )
    return group_issues([issue], sca_issue_plans=[(localized, fix_plan)])[0]


def _express_jwt_coordination_group() -> VulnerabilityGroup:
    return VulnerabilityGroup(
        group_id="sca:package.json:express-jwt",
        issue_type=IssueType.SCA,
        representative_issue_id="00000000-0000-0000-0000-000000000001",
        vulnerable_component="express-jwt",
        file_path="package.json",
        file_paths=["package.json"],
        is_synthetic=True,
        versions=["6.1.1"],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="6.1.1",
            instruction="Coordinate express-jwt with the selected jsonwebtoken version.",
            strategy_used="synthetic_dependency",
        ),
    )


def _certifier_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, str], list[VulnerabilityGroup], dict[str, RemediationTask]]:
    manifest = {
        "name": "app",
        "dependencies": {"express-jwt": "6.1.1", "jsonwebtoken": "8.5.1"},
    }
    lockfile = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app"},
            "node_modules/express-jwt": {
                "version": "6.1.1",
                "dependencies": {"jsonwebtoken": "^8.1.0"},
            },
            "node_modules/jsonwebtoken": {"version": "8.5.1"},
            "node_modules/express-jwt/node_modules/jsonwebtoken": {"version": "8.5.1"},
        },
    }
    files = {
        "package.json": _write_fixture(tmp_path, "package.json", manifest),
        "package-lock.json": _write_fixture(tmp_path, "package-lock.json", lockfile),
    }
    jwt_group = _jsonwebtoken_group()
    express_group = _express_jwt_coordination_group()
    jwt_task = build_initial_remediation_task(jwt_group, "task-jsonwebtoken")
    express_task = build_initial_remediation_task(express_group, "task-express-jwt")
    groups = [jwt_group, express_group]
    task_queue = {
        jwt_task.task_id: jwt_task,
        express_task.task_id: express_task,
    }
    return tmp_path, files, groups, task_queue


def _registry_packument_fetcher(package_name: str) -> dict[str, Any]:
    if package_name == "express-jwt":
        return {
            "name": package_name,
            "versions": {
                "6.1.1": {"dependencies": {"jsonwebtoken": "^8.1.0"}},
                "8.5.1": {"dependencies": {"jsonwebtoken": "^9.0.0"}},
            },
        }
    if package_name == "jsonwebtoken":
        return {"name": package_name, "versions": {"8.5.1": {}, "9.0.2": {}}}
    raise ValueError(f"unexpected packument request: {package_name}")


def _run_fake_certifier(
    root: Path,
    files: dict[str, str],
    groups: list[VulnerabilityGroup],
    task_queue: dict[str, RemediationTask],
    *,
    state: _FakeSandboxState | None = None,
    registry_fetcher: Any | None = None,
    solver_top_k: int = 1,
    prior_portfolio_plan: Any | None = None,
    portfolio_replan_request: PortfolioReplanRequest | None = None,
):
    state = state or _FakeSandboxState(files)
    factory = _FakeSandboxFactory(state)
    settings = AppSettings(
        solver_certification_timeout_seconds=60,
        solver_timeout_seconds=5,
        solver_top_k=solver_top_k,
        solver_num_search_workers=1,
        solver_max_candidates_per_target=64,
        solver_cache_dir=root / "registry-cache",
    )
    plan = build_certified_portfolio_plan(
        root,
        "workspace-volume-test",
        groups,
        task_queue,
        settings=settings,
        portfolio_replan_request=portfolio_replan_request,
        prior_portfolio_plan=prior_portfolio_plan,
        registry_fetcher=registry_fetcher or _registry_packument_fetcher,
        sandbox_factory=factory,
    )
    return plan, state, factory


def _certified_prior_plan(
    task_queue: dict[str, RemediationTask],
    host_fingerprint: str,
    batches: list[Any],
    phases: list[Any],
    batch_prefix_graph_digests: dict[str, str],
    selected_candidate_versions: dict[str, str] | None = None,
) -> SimpleNamespace:
    task_revisions = {task_id: task.task_revision for task_id, task in sorted(task_queue.items())}
    selected = SimpleNamespace(
        candidate_plan_id="prior-candidate",
        selected_candidate_versions=dict(selected_candidate_versions or {}),
        batches=batches,
        phases=phases,
    )
    solver_plan = SimpleNamespace(
        status=SolverStatus.OPTIMAL,
        candidate_catalog_complete=True,
        candidate_catalog_digest="prior-catalog",
        selected_plan=selected,
    )
    certificate = SimpleNamespace(
        status=PackageResolutionStatus.CERTIFIED,
        portfolio_plan_id="prior-plan",
        solver_input_digest="prior-input",
        repository_fingerprint=host_fingerprint,
        workspace_graph_digest="prior-workspace",
        candidate_catalog_digest="prior-catalog",
        candidate_plan_id="prior-candidate",
        candidate_assignment_digest=_digest(
            dict(sorted(selected.selected_candidate_versions.items()))
        ),
        task_revisions=task_revisions,
        batch_prefix_graph_digests=batch_prefix_graph_digests,
    )
    return SimpleNamespace(
        plan_id="prior-plan",
        portfolio_plan_id="prior-plan",
        solver_input_digest="prior-input",
        repository_fingerprint=host_fingerprint,
        workspace_graph_digest="prior-workspace",
        task_revisions=task_revisions,
        solver_plan=solver_plan,
        resolution_certificate=certificate,
        clusters=[],
        cluster_order=[],
        task_order=[],
        task_to_cluster={},
        diagnostics=[],
    )


def test_last_qa_passed_prefix_obeys_phase_order_and_terminal_skips():
    task_queue = {
        "task-a": RemediationTask(
            task_id="task-a",
            parent_group_id="group-a",
            strategy="version_bump",
            status=TaskStatus.QA_PASSED,
        ),
        "task-b": RemediationTask(
            task_id="task-b",
            parent_group_id="group-b",
            strategy="version_bump",
            status=TaskStatus.UNFIXABLE,
        ),
        "task-c": RemediationTask(
            task_id="task-c",
            parent_group_id="group-c",
            strategy="version_bump",
            status=TaskStatus.QA_PASSED,
        ),
        "task-d": RemediationTask(
            task_id="task-d",
            parent_group_id="group-d",
            strategy="version_bump",
            status=TaskStatus.QA_PASSED,
        ),
    }
    batches = [
        SimpleNamespace(
            batch_id="batch-a",
            task_ids=["task-a"],
            dispatchable=True,
            mutations=[object()],
        ),
        SimpleNamespace(
            batch_id="batch-terminal",
            task_ids=["task-b"],
            dispatchable=False,
            mutations=[],
        ),
        SimpleNamespace(
            batch_id="batch-c",
            task_ids=["task-c"],
            dispatchable=True,
            mutations=[object()],
        ),
        SimpleNamespace(
            batch_id="batch-d",
            task_ids=["task-d"],
            dispatchable=True,
            mutations=[object()],
        ),
    ]
    phases = [
        SimpleNamespace(phase_number=1, batch_ids=["batch-a", "batch-terminal"]),
        SimpleNamespace(phase_number=2, batch_ids=["batch-c"]),
    ]
    prior_plan = _certified_prior_plan(
        task_queue,
        "host-fingerprint",
        batches,
        phases,
        {
            "batch-a": "prefix-a",
            "batch-c": "prefix-c",
            "batch-d": "prefix-d",
        },
    )
    request = PortfolioReplanRequest(
        reason="UNFIXABLE_REPLAN",
        source_portfolio_plan_id="prior-plan",
    )

    assert (
        _last_qa_passed_prefix_digest(prior_plan, task_queue, request, "host-fingerprint")
        == "prefix-d"
    )
    task_queue["task-c"] = task_queue["task-c"].model_copy(update={"status": TaskStatus.PENDING})
    assert (
        _last_qa_passed_prefix_digest(prior_plan, task_queue, request, "host-fingerprint")
        == "prefix-a"
    )
    task_queue["task-c"] = task_queue["task-c"].model_copy(update={"status": TaskStatus.QA_PASSED})
    task_queue["task-b"] = task_queue["task-b"].model_copy(update={"status": TaskStatus.PENDING})
    assert (
        _last_qa_passed_prefix_digest(prior_plan, task_queue, request, "host-fingerprint") is None
    )
    assert (
        _last_qa_passed_prefix_digest(
            prior_plan,
            task_queue,
            request.model_copy(update={"source_portfolio_plan_id": "wrong-plan"}),
            "host-fingerprint",
        )
        is None
    )
    assert (
        _last_qa_passed_prefix_digest(prior_plan, task_queue, request, "different-host-fingerprint")
        is None
    )


def test_workspace_manifest_drift_requires_exact_certified_prefix_digest():
    host_documents = {
        "package.json": json.dumps({"dependencies": {"foo": "1.0.0"}}),
    }
    workspace_documents = {
        "package.json": json.dumps({"dependencies": {"foo": "2.0.0"}}),
    }
    host_snapshot = load_npm_graph_snapshot_from_documents(host_documents)
    workspace_snapshot = load_npm_graph_snapshot_from_documents(workspace_documents)

    with pytest.raises(_CertificationUnknown, match="manifest contents differ"):
        _assert_workspace_matches_host(host_snapshot, workspace_snapshot, workspace_documents)

    _assert_workspace_matches_host(
        host_snapshot,
        workspace_snapshot,
        workspace_documents,
        qa_passed_prefix_digest=workspace_snapshot.repository_fingerprint,
    )
    with pytest.raises(_CertificationUnknown, match="manifest contents differ"):
        _assert_workspace_matches_host(
            host_snapshot,
            workspace_snapshot,
            workspace_documents,
            qa_passed_prefix_digest="unrelated-prefix",
        )

    different_path_documents = {
        "workspace/package.json": json.dumps({"dependencies": {"foo": "2.0.0"}}),
    }
    different_path_snapshot = load_npm_graph_snapshot_from_documents(different_path_documents)
    with pytest.raises(_CertificationUnknown, match="manifest paths differ"):
        _assert_workspace_matches_host(
            host_snapshot,
            different_path_snapshot,
            different_path_documents,
            qa_passed_prefix_digest=different_path_snapshot.repository_fingerprint,
        )


def test_unmatched_certified_prefix_fails_closed_on_live_manifest_drift(tmp_path: Path):
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    task_queue["task-jsonwebtoken"] = task_queue["task-jsonwebtoken"].model_copy(
        update={"status": TaskStatus.QA_PASSED}
    )
    host_fingerprint = load_npm_graph_snapshot(root).repository_fingerprint
    batch = SimpleNamespace(
        batch_id="batch-prior",
        task_ids=["task-jsonwebtoken"],
        dispatchable=True,
        mutations=[object()],
    )
    prior_plan = _certified_prior_plan(
        task_queue,
        host_fingerprint,
        [batch],
        [SimpleNamespace(phase_number=1, batch_ids=["batch-prior"])],
        {"batch-prior": "not-the-live-prefix"},
    )
    request = PortfolioReplanRequest(
        reason="UNFIXABLE_REPLAN",
        source_portfolio_plan_id="prior-plan",
    )
    state = _FakeSandboxState(files)
    manifest = json.loads(state.files["package.json"])
    manifest["dependencies"]["jsonwebtoken"] = "9.0.2"
    state.files["package.json"] = json.dumps(manifest, sort_keys=True) + "\n"
    lockfile = json.loads(state.files["package-lock.json"])
    lockfile["packages"]["node_modules/jsonwebtoken"]["version"] = "9.0.2"
    state.files["package-lock.json"] = json.dumps(lockfile, sort_keys=True) + "\n"

    plan, state, _factory = _run_fake_certifier(
        root,
        files,
        groups,
        task_queue,
        state=state,
        prior_portfolio_plan=prior_plan,
        portfolio_replan_request=request,
    )

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert any(
        "workspace package manifest contents differ from the host" in item
        for item in plan.resolution_certificate.diagnostics
    )
    assert state.install_assignments == []


def test_outer_replan_does_not_reuse_prior_resolver_cuts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    host_fingerprint = load_npm_graph_snapshot(root).repository_fingerprint
    batch = SimpleNamespace(
        batch_id="batch-prior",
        task_ids=["task-jsonwebtoken"],
        dispatchable=True,
        mutations=[object()],
    )
    prior_plan = _certified_prior_plan(
        task_queue,
        host_fingerprint,
        [batch],
        [SimpleNamespace(phase_number=1, batch_ids=["batch-prior"])],
        {"batch-prior": "stale-prefix"},
    )
    prior_plan.resolution_certificate.rejected_assignment_digests = ["stale-assignment"]
    prior_plan.resolution_certificate.rejection_conflicts = ["stale-conflict"]
    request = PortfolioReplanRequest(
        reason="UNFIXABLE_REPLAN",
        source_portfolio_plan_id="prior-plan",
    )
    calls: list[dict[str, Any]] = []
    solve = portfolio_certifier.solve_portfolio

    def record_solver_call(*args: Any, **kwargs: Any):
        calls.append(dict(kwargs))
        return solve(*args, **kwargs)

    monkeypatch.setattr(portfolio_certifier, "solve_portfolio", record_solver_call)

    _run_fake_certifier(
        root,
        files,
        groups,
        task_queue,
        prior_portfolio_plan=prior_plan,
        portfolio_replan_request=request,
    )

    assert calls
    assert calls[0].get("forbidden_assignments", []) == []
    assert calls[0].get("forbidden_conflicts", []) == []


@pytest.mark.parametrize(
    ("terminal_status", "expected"),
    [
        ("qa_passed", True),
        ("unfixable", False),
        ("inconclusive", False),
        ("pivoted", False),
    ],
)
def test_terminal_workaround_authorization_requires_qa_pass(terminal_status, expected):
    target = SolverTarget(
        occurrence_id="package.json::foo",
        task_id="task-1",
        group_id="group-1",
        package_name="foo",
        target_package_name="foo",
        manifest_path="package.json",
        lockfile_package_key="node_modules/foo",
        installed_version="1.0.0",
        strategy="code_workaround",
        eligible_for_atomic_update=False,
        is_terminal=True,
        terminal_status=terminal_status,
    )
    finding = SolverFindingRequirement(
        finding_id="finding-1",
        ghsa_id="GHSA-AAAA-BBBB-CCCC",
        vulnerable_package="foo",
        target_occurrence_id=target.occurrence_id,
        vulnerable_occurrence_id=target.occurrence_id,
        fixed_version="2.0.0",
        workaround_available=True,
        workaround_plan_ids=["fix-plan-1"],
    )
    prepared = SimpleNamespace(targets=[target])
    selected_plan = SimpleNamespace(
        task_decisions=[
            SimpleNamespace(
                task_id=target.task_id,
                selected_strategy="code_workaround",
                selected_plan_issue_ids=["fix-plan-1"],
            )
        ]
    )

    assert _authorized_workaround(finding, prepared, selected_plan) is expected


def test_terminal_failed_batch_is_skipped_and_kept_unresolved(tmp_path: Path):
    root, _files, groups, task_queue = _certifier_fixture(tmp_path)
    task_queue["task-jsonwebtoken"] = task_queue["task-jsonwebtoken"].model_copy(
        update={"status": TaskStatus.UNFIXABLE}
    )

    plan, state, _factory = _run_fake_certifier(root, _files, groups, task_queue)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    selected = plan.solver_plan.selected_plan
    assert selected is not None
    terminal_batch = next(
        batch for batch in selected.batches if "task-jsonwebtoken" in batch.task_ids
    )
    assert terminal_batch.dispatchable is False
    assert terminal_batch.mutations == []
    assert terminal_batch.unresolved_coverage_ids
    assert set(plan.resolution_certificate.unresolved_coverage_ids) == set(
        terminal_batch.unresolved_coverage_ids
    )
    assert plan.resolution_certificate.covered_coverage_ids == []
    assert terminal_batch.batch_id not in plan.resolution_certificate.batch_prefix_graph_digests
    assert len(state.install_assignments) == 1


def test_terminal_only_replan_remains_unknown_without_executable_batches(tmp_path: Path):
    root, _files, groups, task_queue = _certifier_fixture(tmp_path)
    task_queue = {
        task_id: task.model_copy(update={"status": TaskStatus.UNFIXABLE})
        for task_id, task in task_queue.items()
    }

    plan, state, _factory = _run_fake_certifier(root, _files, groups, task_queue)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert any(
        "no executable package batch" in item for item in plan.resolution_certificate.diagnostics
    )
    assert state.install_assignments == []


def test_metadata_discovery_prunes_compiled_outputs():
    state = _FakeSandboxState(
        {
            "package.json": "{}",
            "package-lock.json": "{}",
            "build/package.json": '{"name":"compiled-build"}',
            "dist/package.json": '{"name":"compiled-dist"}',
        }
    )

    documents = _read_npm_documents(
        _FakeSandbox(state),
        "",
        time.monotonic() + 30,
    )

    assert documents == {"package.json": "{}", "package-lock.json": "{}"}
    assert state.batch_read_calls == 1
    discovery = next(command for command in state.commands if command.startswith("find "))
    assert "-path /workspace/build -prune -o" in discovery
    assert "-path /workspace/dist -prune -o" in discovery


def test_document_loader_and_lockfile_lookup_preserve_physical_identity(tmp_path: Path):
    manifest = {"name": "app", "dependencies": {"outer": "1.0.0", "other": "1.0.0"}}
    lockfile = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app"},
            "node_modules/outer": {"version": "1.0.0"},
            "node_modules/other": {"version": "1.0.0"},
            "node_modules/other/node_modules/child": {"version": "2.0.0"},
        },
    }
    manifest_text = _write_fixture(tmp_path, "package.json", manifest)
    lockfile_text = _write_fixture(tmp_path, "package-lock.json", lockfile)

    host_snapshot = load_npm_graph_snapshot(tmp_path)
    document_snapshot = load_npm_graph_snapshot_from_documents(
        {"package.json": manifest_text, "package-lock.json": lockfile_text}
    )

    assert document_snapshot.serialize() == host_snapshot.serialize()
    assert (
        resolve_lockfile_dependency_package(
            document_snapshot,
            make_occurrence_id("package.json", "other"),
            "child",
        ).package_key
        == "node_modules/other/node_modules/child"
    )
    assert (
        resolve_lockfile_dependency_package(
            document_snapshot,
            make_occurrence_id("package.json", "outer"),
            "child",
        )
        is None
    )

    unsafe = load_npm_graph_snapshot_from_documents(
        {"../package.json": manifest_text, "node_modules/inner/package.json": manifest_text}
    )
    assert unsafe.manifests == ()
    assert any(
        "unsafe" in diagnostic or "ignored" in diagnostic for diagnostic in unsafe.diagnostics
    )


@pytest.mark.parametrize(
    ("remove_nested_copy", "expected_reason"),
    [
        (True, SolverCandidateRejectionReason.NEW_VULNERABLE_COPY),
        (False, SolverCandidateRejectionReason.COVERAGE_UNRESOLVED),
    ],
)
def test_exact_rejection_uses_next_optimal_candidate_without_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remove_nested_copy: bool,
    expected_reason: SolverCandidateRejectionReason,
):
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    if remove_nested_copy:
        lockfile_payload = json.loads(files["package-lock.json"])
        lockfile_payload["packages"].pop("node_modules/express-jwt/node_modules/jsonwebtoken")
        files["package-lock.json"] = _write_fixture(root, "package-lock.json", lockfile_payload)
    original_bytes = {name: (root / name).read_bytes() for name in files}
    solver_results: list[Any] = []
    real_solve = portfolio_certifier.solve_portfolio

    def count_solver_calls(*args: Any, **kwargs: Any) -> Any:
        result = real_solve(*args, **kwargs)
        solver_results.append(result)
        return result

    monkeypatch.setattr(portfolio_certifier, "solve_portfolio", count_solver_calls)

    def fetch_with_unmodeled_parent_range(package_name: str) -> dict[str, Any]:
        if package_name == "express-jwt":
            return {
                "name": package_name,
                "versions": {
                    "6.1.1": {"dependencies": {"jsonwebtoken": "npm:jsonwebtoken@^8.1.0"}},
                    "8.5.1": {"dependencies": {"jsonwebtoken": "^9.0.0"}},
                },
            }
        return _registry_packument_fetcher(package_name)

    plan, state, _factory = _run_fake_certifier(
        root,
        files,
        groups,
        task_queue,
        registry_fetcher=fetch_with_unmodeled_parent_range,
        solver_top_k=2,
    )

    assert len(solver_results) == 1
    assert len(solver_results[0].candidate_plans) >= 2
    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.OPTIMAL
    assert plan.resolution_certificate is not None
    certificate = plan.resolution_certificate
    assert certificate.status == PackageResolutionStatus.CERTIFIED
    assert len(certificate.rejected_assignment_digests) == 1
    assert len(certificate.rejection_conflicts) == 1
    assert certificate.rejection_conflicts[0].reason_code == expected_reason
    assert certificate.rejection_conflicts[0].cut_kind == SolverCandidateCutKind.EXACT_ASSIGNMENT
    selected = plan.solver_plan.selected_plan
    assert selected is not None
    assert certificate.candidate_plan_id == selected.candidate_plan_id
    assert certificate.candidate_assignment_digest == _digest(selected.selected_candidate_versions)
    statistics = certificate.certification_statistics
    assert statistics.solver_calls == 1
    assert statistics.candidate_attempts == 2
    assert statistics.strict_install_invocations > 0
    assert statistics.rejection_counts_by_reason == {expected_reason.value: 1}
    assert statistics.unknown_count == 0
    assert statistics.evidence_relations_modeled + statistics.evidence_relations_unmodeled > 0
    assert statistics.metadata_files_read > 0
    assert statistics.metadata_bytes_read > 0
    assert statistics.archive_bytes > 0
    assert statistics.archive_build_time_seconds >= 0
    assert statistics.extraction_time_seconds >= 0
    assert statistics.total_certification_time_seconds >= 0
    assert statistics.package_manager_time_seconds >= 0
    assert statistics.lockfile_read_time_seconds >= 0
    assert not any("npm error" in item.casefold() for item in certificate.diagnostics)
    assert {name: (root / name).read_bytes() for name in files} == original_bytes
    assert state.archive is None
    assert state.archive_create_count == 1
    assert len(state.extraction_paths) == 2
    assert len(set(state.extraction_paths)) == 2


@pytest.mark.parametrize("failure_mode", ["timeout", "exception"])
def test_certifier_cleans_scratch_after_timeout_or_exception(tmp_path: Path, failure_mode: str):
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(
        files,
        install_timeout=failure_mode == "timeout",
        raise_on_extract=failure_mode == "exception",
    )

    plan, state, _factory = _run_fake_certifier(root, files, groups, task_queue, state=state)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert state.cleanup_count >= 1
    assert state.archive_create_count == 1
    assert len(state.extraction_paths) == (1 if failure_mode == "timeout" else 0)
    assert state.archive is None
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


def test_cleanup_failure_invalidates_certification_even_after_retry(tmp_path: Path):
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(files, cleanup_failures=1)

    plan, state, _factory = _run_fake_certifier(root, files, groups, task_queue, state=state)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert state.archive_create_count == 1
    assert any("scratch cleanup failed" in item for item in plan.resolution_certificate.diagnostics)
    assert state.cleanup_count >= 2
    assert state.archive is None
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


def test_missing_workspace_volume_returns_unknown_without_selection(tmp_path: Path):
    root, _files, groups, task_queue = _certifier_fixture(tmp_path)
    factory = _FakeSandboxFactory({})
    plan = build_certified_portfolio_plan(
        root,
        "",
        groups,
        task_queue,
        settings=AppSettings(solver_certification_timeout_seconds=10),
        registry_fetcher=_registry_packument_fetcher,
        sandbox_factory=factory,
    )

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert factory.instances == []


def test_missing_fresh_packument_stops_before_solver_dispatch(tmp_path: Path):
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(files)
    factory = _FakeSandboxFactory(state)
    fetched: list[str] = []

    def unavailable_fetcher(package_name: str) -> dict[str, Any]:
        fetched.append(package_name)
        raise RuntimeError("registry unavailable")

    plan = build_certified_portfolio_plan(
        root,
        "workspace-volume-test",
        groups,
        task_queue,
        settings=AppSettings(
            solver_certification_timeout_seconds=60,
            solver_num_search_workers=1,
        ),
        registry_fetcher=unavailable_fetcher,
        sandbox_factory=factory,
    )

    assert set(fetched) == {"express-jwt", "jsonwebtoken"}
    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert not any("npm install --package-lock-only" in command for command in state.commands)
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


@pytest.mark.integration
@pytest.mark.docker
def test_strict_npm_resolution_certifies_nested_lockfile(tmp_path: Path):
    pytest.importorskip("docker")
    from remediation_engine.runtime.docker_client import get_docker_client
    from remediation_engine.runtime.sandbox_mgr import DockerSandbox

    client = get_docker_client()
    try:
        client.images.get("node:22")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"node:22 image is not available locally: {exc}")

    packages = {
        "name": "app",
        "dependencies": {
            "parent-a": "file:packages/parent-a",
            "parent-b": "file:packages/parent-b",
            "peer-host": "file:packages/peer-host",
        },
    }
    _write_fixture(tmp_path, "package.json", packages)
    _write_fixture(
        tmp_path,
        "packages/parent-a/package.json",
        {
            "name": "parent-a",
            "version": "1.0.0",
            "dependencies": {"child": "file:../child-v1"},
            "peerDependencies": {"peer-host": "^2.0.0"},
        },
    )
    _write_fixture(
        tmp_path,
        "packages/parent-b/package.json",
        {
            "name": "parent-b",
            "version": "1.0.0",
            "dependencies": {"child": "file:../child-v2"},
        },
    )
    _write_fixture(
        tmp_path, "packages/child-v1/package.json", {"name": "child", "version": "1.0.0"}
    )
    _write_fixture(
        tmp_path, "packages/child-v2/package.json", {"name": "child", "version": "2.0.0"}
    )
    _write_fixture(
        tmp_path, "packages/peer-host/package.json", {"name": "peer-host", "version": "1.0.0"}
    )

    with DockerSandbox(repo_root=tmp_path) as sandbox:
        install = sandbox.run(
            "npm install --package-lock-only --ignore-scripts --no-audit --no-fund "
            "--legacy-peer-deps=true",
        )
        assert install.exit_code == 0, install.stderr or install.stdout
        root_manifest = sandbox.read_file("package.json")
        root_lock = sandbox.read_file("package-lock.json")
        assert root_manifest and root_lock
        snapshot = load_npm_graph_snapshot_from_documents(
            {"package.json": root_manifest, "package-lock.json": root_lock}
        )
        child_packages = [
            package for package in snapshot.lockfile_packages if package.package_name == "child"
        ]
        child_keys = {package.package_key for package in child_packages}
        assert len(child_keys) >= 2, child_keys
        assert any("parent-" in key and "/node_modules/child" in key for key in child_keys), (
            child_keys
        )

        strict = sandbox.run(
            "npm install --package-lock-only --ignore-scripts --no-audit --no-fund "
            "--strict-peer-deps --engine-strict=true --legacy-peer-deps=false --force=false",
            timeout=120,
        )
        assert strict.exit_code != 0
        assert (
            _install_error_category(strict.stdout, strict.stderr, strict.exit_code)
            == "PEER_CONFLICT"
        )
        assert parse_peer_conflict_evidence(strict.stdout, strict.stderr)


@pytest.mark.parametrize(
    ("candidate_update", "runtime_update", "reason_code"),
    [
        ({"engines": {"node": ">=24.0.0"}}, {}, "runtime_engine"),
        ({"os": ["darwin"]}, {"platform": "linux"}, "runtime_platform"),
        ({"cpu": ["arm64"]}, {"architecture": "x64"}, "runtime_platform"),
    ],
)
def test_runtime_rejections_learn_only_safe_unary_conflicts(
    tmp_path: Path,
    candidate_update: dict[str, Any],
    runtime_update: dict[str, str],
    reason_code: str,
) -> None:
    root, _files, groups, task_queue = _certifier_fixture(tmp_path)
    snapshot = load_npm_graph_snapshot(root)
    prepared = _prepare_portfolio_problem(
        root,
        groups,
        task_queue,
        npm_snapshot=snapshot,
        repository_fingerprint=snapshot.repository_fingerprint,
        registry_fetcher=_registry_packument_fetcher,
        settings=AppSettings(solver_cache_dir=tmp_path / "registry-cache"),
    )
    source = next(
        target for target in prepared.targets if target.target_package_name == "express-jwt"
    )
    source_candidate = next(
        item for item in prepared.candidate_domains[source.occurrence_id] if item.version == "8.5.1"
    )
    assignment = {
        target.occurrence_id: prepared.candidate_domains[target.occurrence_id][0].version
        for target in prepared.targets
        if target.eligible_for_atomic_update
    }
    assignment[source.occurrence_id] = source_candidate.version
    domains = {key: list(values) for key, values in prepared.candidate_domains.items()}
    domains[source.occurrence_id] = [
        item.model_copy(update=candidate_update) if item.version == "8.5.1" else item
        for item in domains[source.occurrence_id]
    ]
    prepared = replace(prepared, candidate_domains=domains)
    runtime = SolverRuntimeFingerprint(
        node_version="22.15.0",
        npm_version="10.9.2",
        platform=runtime_update.get("platform", "linux"),
        architecture=runtime_update.get("architecture", "x64"),
    )

    with pytest.raises(_AssignmentRejected) as rejected:
        _validate_candidate_runtime(prepared, assignment, {source.occurrence_id}, runtime)

    conflict = rejected.value.conflict
    assert conflict.reason_code.value == reason_code
    assert conflict.cut_kind == SolverCandidateCutKind.UNARY
    assert [(item.variable_id, item.version) for item in conflict.literals] == [
        (source.occurrence_id, "8.5.1")
    ]


def test_peer_requester_version_maps_only_unique_mutable_relation() -> None:
    def target(occurrence_id: str, task_id: str, package: str, version: str) -> SolverTarget:
        return SolverTarget(
            occurrence_id=occurrence_id,
            task_id=task_id,
            group_id=f"group-{task_id}",
            package_name=package,
            target_package_name=package,
            manifest_path="package.json",
            lockfile_package_key=f"node_modules/{package}",
            installed_version=version,
        )

    source = target("source-occurrence", "task-source", "@angular/core", "17.0.0")
    peer = target("peer-occurrence", "task-peer", "react", "18.2.0")
    relation = SolverCandidateRelation(
        source_occurrence_id=source.occurrence_id,
        source_candidate_version="17.0.0",
        package_name="react",
        version_range="^18.0.0",
        kind="peer",
        target_occurrence_id=peer.occurrence_id,
    )
    prepared = SimpleNamespace(
        targets=[source, peer],
        subgraph=SimpleNamespace(candidate_relations=[relation]),
    )
    evidence = PeerConflictEvidence(
        requester_package="@angular/core",
        requester_version="17.0.0",
        peer_package="react",
        required_range="^18.0.0",
        observed_version="18.2.0",
    )
    assignment = {source.occurrence_id: "17.0.0", peer.occurrence_id: "18.2.0"}

    assert set(_peer_conflict_literal_ids(prepared, assignment, [evidence]) or ()) == {
        source.occurrence_id,
        peer.occurrence_id,
    }

    unique_conflict = _peer_conflict_rejection(prepared, assignment, [evidence]).conflict
    assert unique_conflict.cut_kind == SolverCandidateCutKind.PAIR
    assert {literal.variable_id for literal in unique_conflict.literals} == {
        source.occurrence_id,
        peer.occurrence_id,
    }

    ambiguous_source = target("source-occurrence-2", "task-source-2", "@angular/core", "17.0.0")
    ambiguous_relation = relation.model_copy(
        update={"source_occurrence_id": ambiguous_source.occurrence_id}
    )
    ambiguous = SimpleNamespace(
        targets=[source, ambiguous_source, peer],
        subgraph=SimpleNamespace(candidate_relations=[relation, ambiguous_relation]),
    )
    ambiguous_assignment = {
        **assignment,
        ambiguous_source.occurrence_id: "17.0.0",
    }
    assert _peer_conflict_literal_ids(ambiguous, ambiguous_assignment, [evidence]) is None
    ambiguous_conflict = _peer_conflict_rejection(
        ambiguous, ambiguous_assignment, [evidence]
    ).conflict
    assert ambiguous_conflict.cut_kind == SolverCandidateCutKind.EXACT_ASSIGNMENT
    assert {
        literal.variable_id: literal.version for literal in ambiguous_conflict.literals
    } == ambiguous_assignment
    missing_requester_version = PeerConflictEvidence(
        requester_package="@angular/core",
        peer_package="react",
        required_range="^18.0.0",
        observed_version="18.2.0",
    )
    missing_conflict = _peer_conflict_rejection(
        prepared, assignment, [missing_requester_version]
    ).conflict
    assert missing_conflict.cut_kind == SolverCandidateCutKind.EXACT_ASSIGNMENT


def _prepared_jsonwebtoken_witness_problem(
    tmp_path: Path,
    *,
    child_mode: str = "complete",
    max_candidates: int = 64,
):
    """Build one offline problem whose selected release requires an absent child."""
    root = tmp_path
    manifest = {
        "name": "app",
        "dependencies": {"jsonwebtoken": "8.5.1"},
    }
    lockfile = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app"},
            "node_modules/jsonwebtoken": {"version": "8.5.1"},
        },
    }
    _write_fixture(root, "package.json", manifest)
    _write_fixture(root, "package-lock.json", lockfile)
    group = _jsonwebtoken_group()
    task = build_initial_remediation_task(group, "task-jsonwebtoken")
    snapshot = load_npm_graph_snapshot(root)
    requested: list[str] = []

    def fetch(package_name: str) -> dict[str, Any]:
        requested.append(package_name)
        if package_name == "jsonwebtoken":
            return {
                "name": "jsonwebtoken",
                "versions": {
                    "8.5.1": {},
                    "9.0.2": {"dependencies": {"runtime-child": "^2.0.0"}},
                },
            }
        if package_name == "runtime-child":
            if child_mode == "missing":
                raise ValueError("synthetic registry miss")
            if child_mode == "malformed":
                return {"name": package_name, "versions": []}
            if child_mode == "oversize":
                return {
                    "name": package_name,
                    "versions": {f"{index}.0.0": {} for index in range(1, 4)},
                }
            return {
                "name": package_name,
                "versions": {"1.5.0": {}, "2.5.0": {}},
            }
        raise ValueError(f"unexpected package request {package_name!r}")

    settings = AppSettings(
        solver_certification_timeout_seconds=60,
        solver_timeout_seconds=5,
        solver_top_k=1,
        solver_num_search_workers=1,
        solver_max_candidates_per_target=max_candidates,
        solver_max_model_variables=10_000,
    )
    prepared = _prepare_portfolio_problem(
        root,
        [group],
        {task.task_id: task},
        npm_snapshot=snapshot,
        repository_fingerprint=snapshot.repository_fingerprint,
        registry_fetcher=fetch,
        settings=settings,
        runtime_fingerprint=SolverRuntimeFingerprint(
            node_version="22.15.0",
            npm_version="10.9.2",
            platform="linux",
            architecture="x64",
        ),
    )
    return prepared, snapshot, requested, settings


def test_fresh_child_catalog_builds_evidence_without_creating_a_task(tmp_path: Path) -> None:
    prepared, snapshot, requested, settings = _prepared_jsonwebtoken_witness_problem(tmp_path)
    target = prepared.targets[0]
    relation = next(
        item
        for item in prepared.subgraph.candidate_relations
        if item.source_candidate_version == "9.0.2"
    )

    assert prepared.candidate_catalog_complete is True
    assert requested.count("runtime-child") == 1
    assert len(prepared.subgraph.targets) == 1
    assert relation.evidence_variable_id is not None
    assert relation.target_occurrence_id is None
    assert relation.is_modelled is True
    evidence_domain = prepared.subgraph.evidence_domains[0]
    assert evidence_domain.package_name == "runtime-child"
    assert evidence_domain.candidate_versions == ["1.5.0", "2.5.0"]
    assert evidence_domain.manifest_path == target.manifest_path
    assert evidence_domain.workspace_id == target.workspace_id

    result = solve_portfolio(
        prepared.subgraph,
        prepared.candidate_domains,
        settings=settings,
        candidate_catalog_complete=prepared.candidate_catalog_complete,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.selected_plan is not None
    assert set(result.selected_plan.selected_candidate_versions) == {target.occurrence_id}
    assert result.selected_plan.selected_candidate_versions[target.occurrence_id] == "9.0.2"

    with pytest.raises(_AssignmentRejected) as rejected:
        _validate_candidate_relations(
            prepared,
            snapshot,
            {target.occurrence_id: "9.0.2"},
            {target.occurrence_id},
            set(),
        )
    assert rejected.value.conflict.reason_code.value == "required_dependency_missing"
    assert rejected.value.conflict.cut_kind.value == "exact_assignment"


@pytest.mark.parametrize("child_mode", ["missing", "malformed", "oversize"])
def test_unusable_child_catalog_keeps_source_candidate_for_npm_fallback(
    tmp_path: Path, child_mode: str
) -> None:
    prepared, _snapshot, requested, settings = _prepared_jsonwebtoken_witness_problem(
        tmp_path, child_mode=child_mode, max_candidates=2
    )
    target = prepared.targets[0]
    relation = next(
        item
        for item in prepared.subgraph.candidate_relations
        if item.source_candidate_version == "9.0.2"
    )

    assert prepared.candidate_catalog_complete is True
    assert requested.count("runtime-child") == 1
    assert prepared.subgraph.evidence_domains == []
    assert relation.evidence_variable_id is None
    assert relation.is_modelled is False

    result = solve_portfolio(
        prepared.subgraph,
        prepared.candidate_domains,
        settings=settings,
        candidate_catalog_complete=prepared.candidate_catalog_complete,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
    )

    assert result.status == SolverStatus.OPTIMAL
    assert result.selected_plan is not None
    assert result.selected_plan.selected_candidate_versions[target.occurrence_id] == "9.0.2"


def test_evidence_domains_apply_runtime_and_known_vulnerable_child_floors(
    tmp_path: Path,
) -> None:
    prepared, _snapshot, _requested, settings = _prepared_jsonwebtoken_witness_problem(tmp_path)
    runtime_child_packument = {
        "name": "runtime-child",
        "versions": {
            "1.0.0": {"engines": {"node": ">=24"}},
            "2.0.0": {"os": ["darwin"]},
            "3.0.0": {"cpu": ["arm64"]},
            "4.0.0": {
                "engines": {"node": ">=20"},
                "os": ["linux"],
                "cpu": ["x64"],
            },
        },
    }
    vulnerable_child = prepared.findings[0].model_copy(
        update={"vulnerable_package": "runtime-child", "fixed_version": "2.0.0"}
    )
    all_packuments = {**prepared.packuments, "runtime-child": runtime_child_packument}
    args = (
        prepared.npm_snapshot,
        prepared.targets,
        [vulnerable_child],
        prepared.candidate_domains,
        all_packuments,
        settings,
    )

    first, first_digest, first_diagnostics = _build_evidence_domains(
        *args,
        registry_fetcher=None,
        runtime_fingerprint=prepared.runtime_fingerprint,
    )
    second, second_digest, second_diagnostics = _build_evidence_domains(
        *args,
        registry_fetcher=None,
        runtime_fingerprint=prepared.runtime_fingerprint,
    )

    assert first_diagnostics == second_diagnostics == []
    assert len(first) == 1
    assert first[0].candidate_versions == ["4.0.0"]
    assert first[0].variable_id == second[0].variable_id
    assert first_digest == second_digest


def test_new_pair_cut_discards_batch_remainder_and_resolves_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from remediation_engine.orchestration import portfolio_certifier

    root = tmp_path
    manifest = {
        "name": "app",
        "dependencies": {"express-jwt": "6.1.1", "jsonwebtoken": "8.5.1"},
    }
    lockfile = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app"},
            "node_modules/express-jwt": {
                "version": "6.1.1",
                "dependencies": {"jsonwebtoken": "^8.1.0"},
            },
            "node_modules/jsonwebtoken": {"version": "8.5.1"},
        },
    }
    files = {
        "package.json": _write_fixture(root, "package.json", manifest),
        "package-lock.json": _write_fixture(root, "package-lock.json", lockfile),
    }
    express_group = _express_jwt_coordination_group()
    jwt_group = VulnerabilityGroup(
        group_id="sca:package.json:jsonwebtoken",
        issue_type=IssueType.SCA,
        representative_issue_id="00000000-0000-0000-0000-000000000002",
        vulnerable_component="jsonwebtoken",
        file_path="package.json",
        file_paths=["package.json"],
        is_synthetic=True,
        versions=["8.5.1"],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="8.5.1",
            instruction="Coordinate the jsonwebtoken release.",
            strategy_used="synthetic_dependency",
        ),
    )
    express_task = build_initial_remediation_task(express_group, "task-express-jwt")
    jwt_task = build_initial_remediation_task(jwt_group, "task-jsonwebtoken")
    task_queue = {express_task.task_id: express_task, jwt_task.task_id: jwt_task}
    solver_results: list[Any] = []
    certification_assignments: list[dict[str, str]] = []
    real_solve = portfolio_certifier.solve_portfolio

    def count_solves(*args: Any, **kwargs: Any) -> Any:
        result = real_solve(*args, **kwargs)
        solver_results.append(result)
        return result

    def reject_first_pair_then_certify(
        sandbox: Any,
        workspace_volume: str,
        sandbox_factory: Any,
        prepared: Any,
        selected_plan: Any,
        assignment: dict[str, str],
        runtime: SolverRuntimeFingerprint,
        archive_path: str,
        scratch_root: str,
        metrics: dict[str, Any],
        deadline: float,
    ) -> _AssignmentResult:
        certification_assignments.append(dict(assignment))
        if len(certification_assignments) == 1:
            source = next(
                target for target in prepared.targets if target.target_package_name == "express-jwt"
            )
            child = next(
                target
                for target in prepared.targets
                if target.target_package_name == "jsonwebtoken"
            )
            relation = next(
                relation
                for relation in prepared.subgraph.candidate_relations
                if relation.source_occurrence_id == source.occurrence_id
                and relation.source_candidate_version == assignment[source.occurrence_id]
                and relation.target_occurrence_id == child.occurrence_id
                and relation.is_modelled
            )
            conflict = _candidate_conflict(
                assignment,
                SolverCandidateRejectionReason.DEPENDENCY_RANGE,
                cut_kind=SolverCandidateCutKind.PAIR,
                literal_ids=(source.occurrence_id, child.occurrence_id),
                evidence={
                    "kind": "dependency_range",
                    "package_name": relation.package_name,
                    "version_range": relation.version_range,
                },
                summary="resolved dependency pair was rejected",
            )
            return _AssignmentResult(
                status=PackageResolutionStatus.REJECTED,
                snapshot=prepared.npm_snapshot,
                covered_coverage_ids=(),
                workaround_coverage_ids=(),
                unresolved_coverage_ids=(),
                resolved_lockfile_digests={},
                batch_prefix_graph_digests={},
                diagnostics=(conflict.summary,),
                rejection_conflict=conflict,
            )
        return _AssignmentResult(
            status=PackageResolutionStatus.CERTIFIED,
            snapshot=prepared.npm_snapshot,
            covered_coverage_ids=(),
            workaround_coverage_ids=(),
            unresolved_coverage_ids=(),
            resolved_lockfile_digests={
                lockfile.path: lockfile.raw_digest for lockfile in prepared.npm_snapshot.lockfiles
            },
            batch_prefix_graph_digests={},
            diagnostics=(),
        )

    monkeypatch.setattr(portfolio_certifier, "solve_portfolio", count_solves)
    monkeypatch.setattr(portfolio_certifier, "_certify_assignment", reject_first_pair_then_certify)
    plan, state, _factory = _run_fake_certifier(
        root,
        files,
        [express_group, jwt_group],
        task_queue,
        solver_top_k=2,
    )

    assert len(solver_results) == 2
    assert len(solver_results[0].candidate_plans) == 2
    assert len(certification_assignments) == 2
    assert certification_assignments[0] != certification_assignments[1]
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    assert len(plan.resolution_certificate.rejection_conflicts) == 1
    conflict = plan.resolution_certificate.rejection_conflicts[0]
    assert conflict.cut_kind == SolverCandidateCutKind.PAIR
    assert plan.resolution_certificate.candidate_plan_id == (
        plan.solver_plan.selected_plan.candidate_plan_id
    )
    assert state.archive is None


def test_later_unknown_never_certifies_a_top_k_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    solver_results: list[Any] = []
    attempts: list[dict[str, str]] = []
    real_solve = portfolio_certifier.solve_portfolio

    def fetch_with_unmodeled_parent_range(package_name: str) -> dict[str, Any]:
        if package_name == "express-jwt":
            return {
                "name": package_name,
                "versions": {
                    "6.1.1": {"dependencies": {"jsonwebtoken": "npm:jsonwebtoken@^8.1.0"}},
                    "8.5.1": {"dependencies": {"jsonwebtoken": "^9.0.0"}},
                },
            }
        return _registry_packument_fetcher(package_name)

    def count_solves(*args: Any, **kwargs: Any) -> Any:
        result = real_solve(*args, **kwargs)
        solver_results.append(result)
        return result

    def reject_then_unknown(
        sandbox: Any,
        workspace_volume: str,
        sandbox_factory: Any,
        prepared: Any,
        selected_plan: Any,
        assignment: dict[str, str],
        runtime: SolverRuntimeFingerprint,
        archive_path: str,
        scratch_root: str,
        metrics: dict[str, Any],
        deadline: float,
    ) -> _AssignmentResult:
        attempts.append(dict(assignment))
        if len(attempts) == 1:
            conflict = _candidate_conflict(
                assignment,
                SolverCandidateRejectionReason.UNCLASSIFIED,
                evidence={"kind": "deterministic_unlocalized_test_rejection"},
                summary="first optimal candidate rejected",
            )
            return _AssignmentResult(
                status=PackageResolutionStatus.REJECTED,
                snapshot=prepared.npm_snapshot,
                covered_coverage_ids=(),
                workaround_coverage_ids=(),
                unresolved_coverage_ids=(),
                resolved_lockfile_digests={},
                batch_prefix_graph_digests={},
                diagnostics=(conflict.summary,),
                rejection_conflict=conflict,
            )
        return _AssignmentResult(
            status=PackageResolutionStatus.UNKNOWN,
            snapshot=None,
            covered_coverage_ids=(),
            workaround_coverage_ids=(),
            unresolved_coverage_ids=(),
            resolved_lockfile_digests={},
            batch_prefix_graph_digests={},
            diagnostics=("synthetic resolver timeout",),
        )

    monkeypatch.setattr(portfolio_certifier, "solve_portfolio", count_solves)
    monkeypatch.setattr(portfolio_certifier, "_certify_assignment", reject_then_unknown)
    plan, _state, _factory = _run_fake_certifier(
        root,
        files,
        groups,
        task_queue,
        registry_fetcher=fetch_with_unmodeled_parent_range,
        solver_top_k=2,
    )

    assert len(solver_results) == 1
    assert len(attempts) == 2
    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    certificate = plan.resolution_certificate
    assert certificate.status == PackageResolutionStatus.UNKNOWN
    assert len(certificate.rejection_conflicts) == 1
    assert certificate.candidate_plan_id == (solver_results[0].candidate_plans[1].candidate_plan_id)
    assert any("synthetic resolver timeout" in item for item in certificate.diagnostics)
    assert any(
        "last rejected candidate: first optimal candidate rejected" in item
        for item in certificate.diagnostics
    )


def test_deadline_unknown_preserves_rejected_conflict_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    deadline_expired = False
    real_remaining = portfolio_certifier._remaining_seconds

    def remaining(deadline: float) -> float:
        return 0.0 if deadline_expired else real_remaining(deadline)

    def reject_and_expire(
        sandbox: Any,
        workspace_volume: str,
        sandbox_factory: Any,
        prepared: Any,
        selected_plan: Any,
        assignment: dict[str, str],
        runtime: SolverRuntimeFingerprint,
        archive_path: str,
        scratch_root: str,
        metrics: dict[str, Any],
        deadline: float,
    ) -> _AssignmentResult:
        nonlocal deadline_expired
        conflict = _candidate_conflict(
            assignment,
            SolverCandidateRejectionReason.UNCLASSIFIED,
            evidence={"kind": "deterministic_test_rejection"},
            summary="candidate was rejected before deadline",
        )
        deadline_expired = True
        return _AssignmentResult(
            status=PackageResolutionStatus.REJECTED,
            snapshot=prepared.npm_snapshot,
            covered_coverage_ids=(),
            workaround_coverage_ids=(),
            unresolved_coverage_ids=(),
            resolved_lockfile_digests={},
            batch_prefix_graph_digests={},
            diagnostics=(conflict.summary,),
            rejection_conflict=conflict,
        )

    monkeypatch.setattr(portfolio_certifier, "_remaining_seconds", remaining)
    monkeypatch.setattr(portfolio_certifier, "_certify_assignment", reject_and_expire)

    plan, _state, _factory = _run_fake_certifier(root, files, groups, task_queue, solver_top_k=1)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    certificate = plan.resolution_certificate
    assert certificate.status == PackageResolutionStatus.UNKNOWN
    assert len(certificate.rejected_assignment_digests) == 1
    assert len(certificate.rejection_conflicts) == 1
    assert certificate.certification_statistics.solver_calls == 1
    assert certificate.certification_statistics.candidate_attempts == 1
    assert certificate.certification_statistics.unknown_count == 1
    assert certificate.certification_statistics.rejection_counts_by_reason == {"unclassified": 1}
    assert any("certification deadline expired" in item for item in certificate.diagnostics)
    assert any(
        "last rejected candidate: candidate was rejected before deadline" in item
        for item in certificate.diagnostics
    )


def test_exact_rejection_maps_use_full_assignment_no_goods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    solver_inputs: list[dict[str, Any]] = []
    attempts: list[dict[str, str]] = []
    real_solve = portfolio_certifier.solve_portfolio

    def count_solves(*args: Any, **kwargs: Any) -> Any:
        solver_inputs.append(dict(kwargs))
        return real_solve(*args, **kwargs)

    def reject_every_candidate(
        sandbox: Any,
        workspace_volume: str,
        sandbox_factory: Any,
        prepared: Any,
        selected_plan: Any,
        assignment: dict[str, str],
        runtime: SolverRuntimeFingerprint,
        archive_path: str,
        scratch_root: str,
        metrics: dict[str, Any],
        deadline: float,
    ) -> _AssignmentResult:
        attempts.append(dict(assignment))
        conflict = _candidate_conflict(
            assignment,
            SolverCandidateRejectionReason.UNCLASSIFIED,
            evidence={"kind": "synthetic_exact_rejection"},
            summary="synthetic exact candidate rejection",
        )
        return _AssignmentResult(
            status=PackageResolutionStatus.REJECTED,
            snapshot=prepared.npm_snapshot,
            covered_coverage_ids=(),
            workaround_coverage_ids=(),
            unresolved_coverage_ids=(),
            resolved_lockfile_digests={},
            batch_prefix_graph_digests={},
            diagnostics=(conflict.summary,),
            rejection_conflict=conflict,
        )

    monkeypatch.setattr(portfolio_certifier, "solve_portfolio", count_solves)
    monkeypatch.setattr(portfolio_certifier, "_certify_assignment", reject_every_candidate)
    plan, _state, _factory = _run_fake_certifier(root, files, groups, task_queue, solver_top_k=1)

    assert len(attempts) == 1
    assert len(solver_inputs) == 2
    assert solver_inputs[1]["forbidden_assignments"] == [attempts[0]]
    assert solver_inputs[1]["forbidden_conflicts"] == []
    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.INFEASIBLE
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.REJECTED
    assert len(plan.resolution_certificate.rejected_assignment_digests) == 1
    assert len(plan.resolution_certificate.rejection_conflicts) == 1


def test_dependency_range_failure_learns_pair_only_for_exact_mutable_physical_edge(
    tmp_path: Path,
) -> None:
    root = tmp_path
    manifest = {
        "name": "app",
        "dependencies": {"jsonwebtoken": "8.5.1", "runtime-child": "1.5.0"},
    }
    lockfile = {
        "name": "app",
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "app"},
            "node_modules/jsonwebtoken": {"version": "8.5.1"},
            "node_modules/runtime-child": {"version": "1.5.0"},
        },
    }
    _write_fixture(root, "package.json", manifest)
    _write_fixture(root, "package-lock.json", lockfile)

    child_issue = VulnerabilityIssue(
        source=IssueSource.SYNTHETIC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-70002",
        package_name="runtime-child",
        package_version="1.5.0",
        file_path="package.json",
    )
    child_localized = LocalizedIssue(
        issue=child_issue,
        manifest_file="package.json",
        package_manager="npm",
        declaration_type="dependencies",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )
    child_group = group_issues(
        [child_issue],
        sca_issue_plans=[
            (
                child_localized,
                FixPlan(
                    status=FixPlanStatus.VERSION_FOUND,
                    fixed_version="2.0.0",
                    instruction="Upgrade runtime-child to the fixed release.",
                    strategy_used="osv_api",
                ),
            )
        ],
    )[0]
    parent_group = _jsonwebtoken_group()
    parent_task = build_initial_remediation_task(parent_group, "task-jsonwebtoken")
    child_task = build_initial_remediation_task(child_group, "task-runtime-child")
    queue = {parent_task.task_id: parent_task, child_task.task_id: child_task}
    snapshot = load_npm_graph_snapshot(root)

    def fetch(package_name: str) -> dict[str, Any]:
        if package_name == "jsonwebtoken":
            return {
                "name": package_name,
                "versions": {
                    "8.5.1": {},
                    "9.0.2": {"dependencies": {"runtime-child": "^2.0.0"}},
                },
            }
        if package_name == "runtime-child":
            return {"name": package_name, "versions": {"1.5.0": {}, "2.5.0": {}}}
        raise ValueError(f"unexpected package request {package_name!r}")

    settings = AppSettings(
        solver_max_candidates_per_target=64,
        solver_cache_dir=None,
    )
    prepared = _prepare_portfolio_problem(
        root,
        [parent_group, child_group],
        queue,
        npm_snapshot=snapshot,
        repository_fingerprint=snapshot.repository_fingerprint,
        registry_fetcher=fetch,
        settings=settings,
        runtime_fingerprint=SolverRuntimeFingerprint(
            node_version="22.15.0",
            npm_version="10.9.2",
            platform="linux",
            architecture="x64",
        ),
    )
    parent = next(
        target for target in prepared.targets if target.target_package_name == "jsonwebtoken"
    )
    child = next(
        target for target in prepared.targets if target.target_package_name == "runtime-child"
    )
    assignment = {
        parent.occurrence_id: "9.0.2",
        child.occurrence_id: "1.5.0",
    }

    with pytest.raises(_AssignmentRejected) as rejected:
        _validate_candidate_relations(
            prepared,
            snapshot,
            assignment,
            {parent.occurrence_id, child.occurrence_id},
            set(),
        )

    conflict = rejected.value.conflict
    assert conflict.reason_code == SolverCandidateRejectionReason.DEPENDENCY_RANGE
    assert conflict.cut_kind == SolverCandidateCutKind.PAIR
    assert {literal.variable_id for literal in conflict.literals} == {
        parent.occurrence_id,
        child.occurrence_id,
    }


def test_staging_manifest_mismatch_is_unknown_without_conflict_learning() -> None:
    state = _FakeSandboxState({"package.json": json.dumps({"dependencies": {"foo": "1.0.0"}})})
    mutation = PackageMutation(
        task_id="task-foo",
        package_name="foo",
        manifest_path="package.json",
        target_version="2.0.0",
        dependency_type="dependencies",
    )

    with pytest.raises(_CertificationUnknown):
        _verify_staged_mutations(_FakeSandbox(state), [mutation], None)


def test_pruned_override_uses_candidate_start_lockfile_after_prior_batch() -> None:
    from remediation_engine.contracts.solver_models import SolverMutation
    from remediation_engine.orchestration.portfolio_certifier import (
        _pruned_override_mutations,
        _pruning_baseline_checkpoint,
    )
    from remediation_engine.orchestration.tools_manifest import _PackageCheckpoint

    baseline_manifest = json.dumps({"dependencies": {"express-jwt": "0.1.3"}})
    baseline_lockfile = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {
                "node_modules/express-jwt": {"version": "0.1.3"},
                "node_modules/base64url": {"version": "0.0.6"},
            },
        }
    )
    staged_manifest = json.dumps(
        {
            "dependencies": {"express-jwt": "7.7.8"},
            "overrides": {"base64url": "3.0.0"},
        }
    )
    staged_lockfile = json.dumps(
        {
            "lockfileVersion": 3,
            "packages": {"node_modules/express-jwt": {"version": "7.7.8"}},
        }
    )
    scratch_prefix = ".remedy-plan-cert/run/candidate"
    baseline = _pruning_baseline_checkpoint(
        scratch_prefix,
        ["package.json"],
        {"package.json": baseline_manifest, "package-lock.json": baseline_lockfile},
    )
    mutation = PackageMutation(
        task_id="task-base64url",
        package_name="base64url",
        manifest_path=f"{scratch_prefix}/package.json",
        target_version="3.0.0",
        dependency_type="overrides",
    )
    checkpoint = _PackageCheckpoint(
        files={
            mutation.manifest_path: staged_manifest,
            f"{scratch_prefix}/package-lock.json": staged_lockfile,
        },
        touched_files_before=set(),
    )
    sandbox = _FakeSandbox(
        _FakeSandboxState(
            {
                mutation.manifest_path: staged_manifest,
                f"{scratch_prefix}/package-lock.json": staged_lockfile,
            }
        )
    )

    _verify_staged_mutations(
        sandbox,
        [mutation],
        checkpoint,
        pruning_baseline=baseline,
    )
    solver_mutation = SolverMutation(
        task_id="task-base64url",
        occurrence_id="package.json::base64url",
        package_name="base64url",
        manifest_path="package.json",
        target_version="3.0.0",
        dependency_type="overrides",
    )
    assert _pruned_override_mutations(
        SimpleNamespace(lockfile_packages=[]),
        [solver_mutation],
        [mutation],
        baseline,
    ) == {solver_mutation.occurrence_id}


def test_resolver_graph_drift_is_unknown_without_conflict_learning(tmp_path: Path) -> None:
    prepared, snapshot, _requested, _settings = _prepared_jsonwebtoken_witness_problem(tmp_path)
    target = prepared.targets[0]
    drifted_snapshot = replace(
        snapshot,
        occurrences=tuple(
            occurrence
            for occurrence in snapshot.occurrences
            if occurrence.occurrence_id != target.occurrence_id
        ),
    )

    with pytest.raises(_CertificationUnknown):
        _validate_candidate_relations(
            prepared,
            drifted_snapshot,
            {target.occurrence_id: "9.0.2"},
            {target.occurrence_id},
            set(),
        )


def test_runtime_fingerprint_discovery_failure_produces_no_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from remediation_engine.orchestration import portfolio_certifier

    root, files, groups, task_queue = _certifier_fixture(tmp_path)

    def fail_runtime(_sandbox: Any, _deadline: float) -> SolverRuntimeFingerprint:
        raise _CertificationUnknown("runtime fingerprint unavailable")

    monkeypatch.setattr(portfolio_certifier, "_read_runtime_fingerprint", fail_runtime)
    plan, _state, _factory = _run_fake_certifier(root, files, groups, task_queue)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert plan.resolution_certificate.certification_statistics.unknown_count == 1


def test_invalid_shared_archive_returns_unknown_and_cleans_run_artifacts(tmp_path: Path) -> None:
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(files, invalid_archive=True)

    plan, state, _factory = _run_fake_certifier(root, files, groups, task_queue, state=state)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert state.archive_create_count == 1
    assert state.archive is None
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


def test_extracted_workspace_fingerprint_drift_is_unknown(tmp_path: Path) -> None:
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(files, tamper_archive=True)

    plan, state, _factory = _run_fake_certifier(root, files, groups, task_queue, state=state)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert any(
        "extracted workspace fingerprint" in item
        for item in plan.resolution_certificate.diagnostics
    )
    assert state.archive is None
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


def test_malformed_resolved_metadata_is_unknown_without_learning(tmp_path: Path) -> None:
    root, files, groups, task_queue = _certifier_fixture(tmp_path)
    state = _FakeSandboxState(files, malformed_resolved_lockfile=True)

    plan, state, _factory = _run_fake_certifier(root, files, groups, task_queue, state=state)

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.UNKNOWN
    assert plan.solver_plan.selected_plan is None
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.UNKNOWN
    assert plan.resolution_certificate.rejection_conflicts == []
    assert state.archive is None
    assert not any(path.startswith(".remedy-plan-cert/") for path in state.files)


def test_unfixable_replan_certifies_remaining_task_and_preserves_qa_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import MagicMock

    from remediation_engine.orchestration.graph import run_portfolio_node
    from remediation_engine.orchestration.state import initial_orchestrator_state
    from remediation_engine.orchestration.supervisor_node import MAX_RETRIES, run_supervisor_node
    from remediation_engine.orchestration.teardown_node import run_teardown_node

    def make_group(package_name: str, cve_id: str) -> VulnerabilityGroup:
        issue = VulnerabilityIssue(
            source=IssueSource.SYNTHETIC,
            issue_type=IssueType.SCA,
            severity=Severity.HIGH,
            cve_id=cve_id,
            package_name=package_name,
            package_version="1.0.0",
            file_path="package.json",
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
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="2.0.0",
            instruction=f"Update {package_name} to the fixed release.",
            strategy_used="osv_api",
        )
        return group_issues([issue], sca_issue_plans=[(localized, fix_plan)])[0]

    def lockfile_for(dependencies: dict[str, str]) -> dict[str, Any]:
        packages: dict[str, dict[str, Any]] = {
            "": {"name": "app", "dependencies": dict(dependencies)}
        }
        packages.update(
            {
                f"node_modules/{package_name}": {"version": version}
                for package_name, version in dependencies.items()
            }
        )
        return {"name": "app", "lockfileVersion": 3, "packages": packages}

    groups = [
        make_group("foo", "CVE-2026-71001"),
        make_group("bar", "CVE-2026-71002"),
        make_group("baz", "CVE-2026-71003"),
    ]
    host_dependencies = {"foo": "1.0.0", "bar": "1.0.0", "baz": "1.0.0"}
    host_manifest = {"name": "app", "dependencies": host_dependencies}
    host_lockfile = lockfile_for(host_dependencies)
    _write_fixture(tmp_path, "package.json", host_manifest)
    _write_fixture(tmp_path, "package-lock.json", host_lockfile)

    prefix_dependencies = {"foo": "2.0.0", "bar": "1.0.0", "baz": "1.0.0"}
    workspace_files = {
        "package.json": json.dumps(
            {"name": "app", "dependencies": prefix_dependencies},
            sort_keys=True,
        )
        + "\n",
        "package-lock.json": json.dumps(
            lockfile_for(prefix_dependencies),
            sort_keys=True,
        )
        + "\n",
    }
    workspace_state = _FakeSandboxState(workspace_files)
    prefix_snapshot = load_npm_graph_snapshot_from_documents(workspace_files)
    assert not prefix_snapshot.diagnostics
    host_fingerprint = load_npm_graph_snapshot(tmp_path).repository_fingerprint

    task_a = build_initial_remediation_task(groups[0], "task-a").model_copy(
        update={
            "status": TaskStatus.QA_PASSED,
            "selected_version": "2.0.0",
            "allowed_target_versions": ["2.0.0"],
        }
    )
    task_b = build_initial_remediation_task(groups[1], "task-b").model_copy(
        update={
            "status": TaskStatus.NEEDS_RETRY,
            "retry_count": MAX_RETRIES,
        }
    )
    task_c = build_initial_remediation_task(groups[2], "task-c")
    task_queue = {task.task_id: task for task in (task_a, task_b, task_c)}
    targets, _findings, target_diagnostics = _build_targets_and_findings(
        prefix_snapshot, groups, task_queue
    )
    assert {target.package_name for target in targets} == {"foo", "bar", "baz"}
    assert not target_diagnostics, target_diagnostics
    prior_batch = SimpleNamespace(
        batch_id="batch-task-a",
        task_ids=["task-a"],
        dispatchable=True,
        mutations=[object()],
    )
    foo_occurrence = next(
        occurrence.occurrence_id
        for occurrence in prefix_snapshot.occurrences
        if occurrence.package_name == "foo"
    )
    prior_plan = _certified_prior_plan(
        task_queue,
        host_fingerprint,
        [prior_batch],
        [SimpleNamespace(phase_number=1, batch_ids=["batch-task-a"])],
        {"batch-task-a": prefix_snapshot.repository_fingerprint},
        selected_candidate_versions={foo_occurrence: "2.0.0"},
    )
    state = initial_orchestrator_state(str(tmp_path), groups)
    state.update(
        {
            "repo_root": str(tmp_path),
            "valid_groups": groups,
            "task_queue": task_queue,
            "portfolio_plan": prior_plan,
            "portfolio_solver_plan": prior_plan.solver_plan,
            "portfolio_iteration": 1,
            "portfolio_dirty": False,
            "workspace_volume": "workspace-volume-test",
            "status": "supervisor_entered",
        }
    )
    with monkeypatch.context() as supervisor_guards:
        supervisor_guards.setattr(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_violations",
            lambda *args, **kwargs: [],
        )
        supervisor_guards.setattr(
            "remediation_engine.orchestration.supervisor_node._portfolio_plan_is_stale",
            lambda *args, **kwargs: False,
        )
        first_supervisor = run_supervisor_node(state)

    assert first_supervisor["next_routing_step"] == "portfolio"
    assert first_supervisor["decision_code"].value == "PORTFOLIO_PLAN_REQUIRED"
    assert first_supervisor["task_queue"]["task-b"].status == TaskStatus.UNFIXABLE
    request = first_supervisor["portfolio_replan_request"]
    assert request.reason == "UNFIXABLE_REPLAN"
    assert request.source_portfolio_plan_id == "prior-plan"
    state.update(first_supervisor)

    settings = AppSettings(
        solver_certification_timeout_seconds=60,
        solver_timeout_seconds=5,
        solver_top_k=1,
        solver_num_search_workers=1,
        solver_max_candidates_per_target=64,
        solver_cache_dir=tmp_path / "registry-cache",
    )
    real_builder = build_certified_portfolio_plan

    def fetch_packument(package_name: str) -> dict[str, Any]:
        return {
            "name": package_name,
            "versions": {"1.0.0": {}, "2.0.0": {}},
        }

    sandbox_factory = _FakeSandboxFactory(workspace_state)

    def build_with_fake_workspace(*args: Any, **kwargs: Any) -> Any:
        kwargs["settings"] = settings
        kwargs["registry_fetcher"] = fetch_packument
        kwargs["sandbox_factory"] = sandbox_factory
        return real_builder(*args, **kwargs)

    with monkeypatch.context() as portfolio_seams:
        portfolio_seams.setattr(
            "remediation_engine.orchestration.graph.get_runtime_settings",
            lambda: settings,
        )
        portfolio_seams.setattr(
            "remediation_engine.orchestration.graph.build_certified_portfolio_plan",
            build_with_fake_workspace,
        )
        portfolio_result = run_portfolio_node(state)

    assert portfolio_result["status"] == "portfolio_ready", "\n".join(
        portfolio_result.get("errors", [])
    )
    assert portfolio_result["portfolio_replan_request"] is None
    plan = portfolio_result["portfolio_plan"]
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    assert plan.portfolio_plan_id != "prior-plan"
    selected = plan.solver_plan.selected_plan
    assert selected is not None
    batch_by_task = {task_id: batch for batch in selected.batches for task_id in batch.task_ids}
    assert batch_by_task["task-a"].dispatchable is False
    assert batch_by_task["task-a"].mutations == []
    assert batch_by_task["task-b"].dispatchable is False
    assert batch_by_task["task-b"].mutations == []
    assert batch_by_task["task-c"].dispatchable is True
    assert batch_by_task["task-c"].mutations
    coverage_by_task = {
        task_id: set(batch.coverage_finding_ids)
        for batch in selected.batches
        for task_id in batch.task_ids
    }
    certified_coverage = set(plan.resolution_certificate.covered_coverage_ids)
    unresolved_coverage = set(plan.resolution_certificate.unresolved_coverage_ids)
    assert coverage_by_task["task-a"] <= certified_coverage
    assert coverage_by_task["task-b"] <= unresolved_coverage
    assert coverage_by_task["task-c"] <= certified_coverage
    committed_queue = portfolio_result["task_queue"]
    assert committed_queue["task-a"].status == TaskStatus.QA_PASSED
    assert committed_queue["task-b"].status == TaskStatus.UNFIXABLE
    assert committed_queue["task-c"].selected_version == "2.0.0"
    assert committed_queue["task-b"].portfolio_plan_id == plan.portfolio_plan_id
    assert committed_queue["task-b"].task_revision == plan.planned_task_revisions["task-b"]

    next_supervisor = run_supervisor_node({**state, **portfolio_result})
    assert next_supervisor["next_routing_step"] == "update_subagent"
    assert next_supervisor["active_target_task_ids"] == ["task-c"]

    final_queue = dict(next_supervisor["task_queue"])
    final_queue["task-c"] = final_queue["task-c"].model_copy(
        update={"status": TaskStatus.QA_PASSED, "current_attempt_id": None}
    )
    changed_manifest = json.loads(workspace_state.files["package.json"])
    changed_manifest["dependencies"].update({"bar": "9.9.9", "baz": "2.0.0"})
    workspace_state.files["package.json"] = json.dumps(changed_manifest, sort_keys=True) + "\n"
    changed_lockfile = json.loads(workspace_state.files["package-lock.json"])
    changed_lockfile["packages"][""]["dependencies"] = dict(changed_manifest["dependencies"])
    changed_lockfile["packages"]["node_modules/bar"]["version"] = "9.9.9"
    changed_lockfile["packages"]["node_modules/baz"]["version"] = "2.0.0"
    workspace_state.files["package-lock.json"] = json.dumps(changed_lockfile, sort_keys=True) + "\n"
    teardown_state = {**state, **portfolio_result, **next_supervisor}
    teardown_state.update(
        {
            "task_queue": final_queue,
            "changed_files": ["package.json", "package-lock.json"],
            "final_full_scan_completed": True,
        }
    )
    docker_client = MagicMock()
    with monkeypatch.context() as teardown_seams:
        teardown_seams.setattr(
            "remediation_engine.orchestration.teardown_node.DockerSandbox",
            sandbox_factory,
        )
        teardown_seams.setattr(
            "remediation_engine.orchestration.teardown_node.get_docker_client",
            lambda: docker_client,
        )
        teardown_result = run_teardown_node(teardown_state)

    assert teardown_result["task_queue"]["task-b"].status == TaskStatus.UNFIXABLE
    assert teardown_result["task_queue"]["task-c"].status == TaskStatus.QA_PASSED
    assert teardown_result["changed_files"] == ["package-lock.json", "package.json"]
    assert "9.9.9" not in teardown_result["diff"]
    assert '"foo": "2.0.0"' in teardown_result["diff"]
    assert '"baz": "2.0.0"' in teardown_result["diff"]
    assert json.loads((tmp_path / "package.json").read_text(encoding="utf-8")) == host_manifest
    assert json.loads((tmp_path / "package-lock.json").read_text(encoding="utf-8")) == host_lockfile
