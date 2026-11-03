"""Focused tests for portfolio languages."""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Self
from uuid import uuid4

import pytest

from remediation_engine.agents.supervisor import (
    _build_multi_package_action,
    _ordered_update_candidates,
    _portfolio_plan_is_stale,
    _portfolio_plan_violations,
)
from remediation_engine.agents.supervisor_policy import (
    _needs_planner,
    _next_sca_stage,
    _planner_plan_violations,
    _repair_invalid_planner_plans,
    _run_deterministic_retry_planner,
)
from remediation_engine.agents.supervisor_reasoning import (
    allowed_tactical_strategies,
    build_tactical_context,
    registry_candidate_sets_for_context,
)
from remediation_engine.contracts import (
    SCARemediationStage,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    MavenTargetOperation,
    PackageFixPlanCandidate,
    RemediationTask,
    RoutingStrategy,
    Severity,
    SupervisorRetryPlan,
    TacticalStrategy,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionStatus,
    SolverPythonRuntimeFingerprint,
    SolverStatus,
    SolverTaskDecision,
)
from remediation_engine.contracts.version_policy import MavenRegistryCandidate, RegistryCandidate
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.state import build_initial_remediation_task
from remediation_engine.planning.candidates import (
    apply_portfolio_plan,
    build_python_portfolio_plan,
    prepare_python_portfolio_problem,
)
from remediation_engine.planning.certification import build_certified_portfolio_plan
from remediation_engine.planning.control import _portfolio_certificate_violations
from remediation_engine.settings import AppSettings
from remediation_engine.tools.advisories import _extract_fixed_from_osv_vuln
from tests.helpers import group_with_fix_plans, qa_policy_for_task


def _python_group(
    manifest_file: str = "requirements.txt",
    *,
    package_name: str = "certifi",
    package_version: str = "2022.6.15",
    fixed_version: str = "2022.12.7",
    cve_id: str = "CVE-2023-0001",
):
    issue = VulnerabilityIssue(
        source=IssueSource.SYNTHETIC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=package_name,
        package_version=package_version,
        fixed_version=fixed_version,
        ecosystem="pypi",
        cve_id=cve_id,
        file_path=manifest_file,
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest_file,
        package_manager="pip",
        declaration_type="requirements",
        is_direct_dependency=True,
        localization_confidence=1.0,
    )
    fix = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version="2022.12.7",
        instruction="Update certifi",
        strategy_used="fixture",
    )
    return group_with_fix_plans([issue], plans_by_finding=[(localized, fix)])[0]


def _pypi_candidate(version: str) -> RegistryCandidate:
    return RegistryCandidate(
        version=version,
        ecosystem="pypi",
        semver_key=None,
        security_floor_met=True,
        is_stable=True,
        same_major=False,
        already_attempted=False,
        selection_roles=("osv_minimum",),
    )


def test_python_manifest_uses_common_solver_without_lockfile_fabrication(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("certifi==2022.6.15\n", encoding="utf-8")
    group = _python_group()
    task = build_initial_remediation_task(group, "task-1")

    def registry(package: str, floor: str, attempted: set[str]):
        assert package == "certifi"
        assert floor == "2022.12.7"
        assert attempted == set()
        return [_pypi_candidate(version) for version in ("2024.8.30", "2022.12.7", "2023.5.7")]

    prepared, solver, plan = build_python_portfolio_plan(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
    )

    assert prepared.subgraph.valid
    assert prepared.targets[0].lockfile_package_key is None
    assert prepared.subgraph.occurrences[0].lockfile_package_key is None
    assert prepared.findings[0].finding_id.startswith("record:")
    assert solver.status == SolverStatus.OPTIMAL
    assert solver.candidate_catalog_complete
    assert solver.selected_plan is not None
    assert solver.selected_plan.selected_candidate_versions == {
        prepared.targets[0].occurrence_id: "2022.12.7"
    }
    assert plan.task_revisions == {task.task_id: task.task_revision}


def test_python_uses_odc_cpe_floor_from_serialized_scanner_evidence(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("certifi==2022.12.7\n", encoding="utf-8")
    group = _python_group(package_version="2022.12.7", fixed_version="2022.12.7")
    issue = group.issues[0].model_copy(
        update={
            "source": IssueSource.ODC,
            "fixed_version": None,
            "raw_payload": {
                "vulnerability": {
                    "vulnerableSoftware": [
                        {
                            "software": {
                                "id": "cpe:2.3:a:certifi:certifi:*:*:*:*:*:python:*:*",
                                "versionEndExcluding": "2024.7.4",
                            }
                        }
                    ]
                }
            },
        }
    )
    group = group.model_copy(update={"issues": [issue]})
    task = build_initial_remediation_task(group, "task-odc-floor")
    registry_calls: list[tuple[str, str, set[str]]] = []

    def registry(package: str, floor: str, attempted: set[str]):
        registry_calls.append((package, floor, set(attempted)))
        return [_pypi_candidate("2024.7.4"), _pypi_candidate("2024.8.30")]

    prepared, solver, _plan = build_python_portfolio_plan(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
    )

    assert prepared.findings[0].fixed_version == "2024.7.4"
    assert registry_calls == [("certifi", "2024.7.4", set())]
    assert solver.selected_plan is not None
    assert solver.selected_plan.selected_candidate_versions == {
        prepared.targets[0].occurrence_id: "2024.7.4"
    }


def test_shared_python_version_key_preserves_pep440_numeric_order_and_equivalence() -> None:
    from remediation_engine.planning.candidates import (
        _python_numeric_solver_version,
    )

    assert _python_numeric_solver_version("1.10")[1] > _python_numeric_solver_version("1.9.2")[1]
    assert _python_numeric_solver_version("1.0")[1] == _python_numeric_solver_version("1.0.0")[1]
    assert _python_numeric_solver_version("1.0")[0] == "1.0.0"
    assert (
        _python_numeric_solver_version("1.8.0")[1]
        < _python_numeric_solver_version("1.8.0.post1")[1]
    )
    assert (
        _python_numeric_solver_version("1.8.0.post1")[1]
        < _python_numeric_solver_version("1.8.1")[1]
    )


def _pipfile_parent_group():
    group = _python_group("Pipfile")
    localized = group.localized_issues[0].model_copy(
        update={
            "package_manager": "pipenv",
            "is_direct_dependency": False,
            "parent_package_name": "direct-parent",
            "dependency_ancestry": ["direct-parent", "certifi"],
        }
    )
    return group.model_copy(
        update={
            "localized_issues": [localized],
            "parent_package_name": "direct-parent",
            "dependency_ancestry": ["direct-parent", "certifi"],
        }
    )


def _write_pipfile_parent_fixture(tmp_path: Path, declaration: str = ">=1.0,<2") -> None:
    (tmp_path / "Pipfile").write_text(
        f'[packages]\ndirect-parent = "{declaration}"\n', encoding="utf-8"
    )
    (tmp_path / "Pipfile.lock").write_text(
        json.dumps(
            {
                "_meta": {"requires": {"python_version": "3.11"}},
                "default": {
                    "direct-parent": {"version": "==1.0.0"},
                    "certifi": {"version": "==2022.6.15"},
                },
                "develop": {},
            }
        ),
        encoding="utf-8",
    )


def test_pipfile_parent_uses_shared_solver_and_pep440_candidate_gates(tmp_path: Path) -> None:
    from remediation_engine.planning.candidates import _python_numeric_solver_version

    _write_pipfile_parent_fixture(tmp_path)
    group = _pipfile_parent_group()
    task = build_initial_remediation_task(group, "task-1")
    retry = SimpleNamespace(
        attempted_versions=["1.2.0"],
        executed_versions=["1.2.0"],
        attempted_versions_by_target={"Direct_Parent": ["1.2.0"]},
    )
    registry_calls: list[tuple[str, str, set[str]]] = []

    def registry(package: str, floor: str, attempted: set[str]):
        registry_calls.append((package, floor, set(attempted)))
        return [_pypi_candidate(version) for version in ("1.2.0", "1.8.0.post1", "2.0.0")]

    def requires_dist(package: str, version: str):
        assert package == "direct-parent"
        if version == "1.0.0":
            return ["certifi>=2022.0,<2023.0"]
        if version == "2.0.0":
            return ["certifi>=2023.0"]
        return ["certifi>=2022.12.7,<2023.0"]

    prepared, solver, plan = build_python_portfolio_plan(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
        parent_requires_dist_fetcher=requires_dist,
        retry_diagnostics_by_task={task.task_id: retry},
    )

    assert prepared.candidate_catalog_complete
    assert (
        prepared.parent_contexts[prepared.targets[0].occurrence_id]["parent_minimum_version"]
        == "1.8.0.post1"
    )
    assert prepared.targets[0].target_package_name == "direct-parent"
    assert prepared.targets[0].lockfile_package_key is None
    assert prepared.findings[0].vulnerable_package == "certifi"
    assert prepared.findings[0].direct_parent_name == "direct-parent"
    assert prepared.findings[0].direct_parent_minimum_version == "1.8.0.post1"
    assert prepared.findings[0].is_transitive
    assert solver.status == SolverStatus.OPTIMAL
    assert solver.selected_plan is not None
    selected = solver.selected_plan.selected_candidate_versions[prepared.targets[0].occurrence_id]
    assert (
        _python_numeric_solver_version(selected)[1]
        >= _python_numeric_solver_version("1.8.0.post1")[1]
    )
    assert registry_calls == [("direct-parent", "1.0.0", {"1.0.0", "1.2.0"})]


@pytest.mark.parametrize("declaration", ["==1.2.0", ">=2.0,<3.0"])
def test_pipfile_parent_lock_must_satisfy_direct_declaration(
    tmp_path: Path, declaration: str
) -> None:
    _write_pipfile_parent_fixture(tmp_path, declaration=declaration)
    group = _pipfile_parent_group()
    task = build_initial_remediation_task(group, "task-1")
    registry_called = False

    def registry(*_args):
        nonlocal registry_called
        registry_called = True
        return [_pypi_candidate("1.8.0.post1")]

    prepared = prepare_python_portfolio_problem(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
        parent_requires_dist_fetcher=lambda *_: ["certifi>=2022.0"],
    )
    assert not prepared.candidate_catalog_complete
    assert not prepared.targets
    assert not registry_called


def test_missing_python_workspace_skips_pipfile_parent_registry_lookup(tmp_path: Path) -> None:
    _write_pipfile_parent_fixture(tmp_path)
    group = _pipfile_parent_group()
    task = build_initial_remediation_task(group, "task-1")
    registry_called = False

    def registry(*_args):
        nonlocal registry_called
        registry_called = True
        return [_pypi_candidate("1.8.0.post1")]

    with pytest.raises(ValueError, match="workspace volume is missing"):
        build_certified_portfolio_plan(
            tmp_path,
            "",
            [group],
            {task.task_id: task},
            settings=AppSettings(),
            python_registry_fetcher=registry,
            python_parent_requires_dist_fetcher=lambda *_: ["certifi>=2022.0"],
            project_language=ProjectLanguage.PYTHON,
        )
    assert not registry_called


def test_python_parent_target_fails_closed_before_registry_lookup(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("parent-lib==1.0.0\n", encoding="utf-8")
    group = _python_group().model_copy(update={"parent_package_name": "parent-lib"})
    task = build_initial_remediation_task(group, "task-1").model_copy(
        update={"target_package_name": "parent-lib", "parent_package_name": "parent-lib"}
    )
    called = False

    def registry(*_args):
        nonlocal called
        called = True
        return []

    prepared = prepare_python_portfolio_problem(
        tmp_path,
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        registry_fetcher=registry,
    )

    assert not called
    assert not prepared.candidate_catalog_complete
    assert not prepared.targets
    assert any(
        "direct-only" in diagnostic or "transitive" in diagnostic
        for diagnostic in prepared.diagnostics
    )


@pytest.mark.parametrize(
    ("manifest", "content"),
    [
        ("pyproject.toml", '[project]\ndependencies = ["certifi==2022.6.15"]\n'),
        ("Pipfile", '[packages]\ncertifi = "==2022.6.15"\n'),
        ("setup.cfg", "[options]\ninstall_requires =\n    certifi==2022.6.15\n"),
    ],
)
def test_python_non_requirements_manifests_fail_before_registry_or_report(
    tmp_path: Path, manifest: str, content: str
) -> None:
    (tmp_path / manifest).write_text(content, encoding="utf-8")
    group = _python_group(manifest)
    task = build_initial_remediation_task(group, "task-1")
    called = {"registry": False, "sandbox": False}

    def registry(*_args):
        called["registry"] = True
        return [_pypi_candidate("2022.12.7")]

    def sandbox_factory(**_kwargs):
        called["sandbox"] = True
        raise AssertionError("unsupported manifest must fail before pip certification")

    with pytest.raises(ValueError, match=r"requirements\*\.txt only"):
        build_certified_portfolio_plan(
            tmp_path,
            "workspace-volume",
            [group],
            {task.task_id: task},
            settings=AppSettings(),
            python_registry_fetcher=registry,
            sandbox_factory=sandbox_factory,
            project_language=ProjectLanguage.PYTHON,
        )

    assert called == {"registry": False, "sandbox": False}


class _PythonCertifierSandbox:
    """Small pip-report sandbox fake exercising the common certifier boundary."""

    def __init__(
        self,
        requirements: str | None = None,
        *,
        files: dict[str, str] | None = None,
        pipfile_locked_parent: str | None = None,
        pipfile_locked_child: str = "2022.12.7",
        pip_build_fail_versions: set[str] | None = None,
    ) -> None:
        self.files = dict(files or ({"requirements.txt": requirements} if requirements else {}))
        self.archive: dict[str, str] | None = None
        self.commands: list[str] = []
        self.pipfile_locked_parent = pipfile_locked_parent
        self.pipfile_locked_child = pipfile_locked_child
        self.pip_build_fail_versions = set(pip_build_fail_versions or ())

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    @staticmethod
    def _path(path: str) -> str:
        return path.replace("/workspace/", "").lstrip("/")

    def read_file(self, path: str) -> str | None:
        return self.files.get(self._path(path))

    def write_file(self, path: str, content: str) -> None:
        self.files[self._path(path)] = content

    @staticmethod
    def _result(exit_code: int = 0, stdout: str = "", stderr: str = ""):
        return type(
            "CommandResult",
            (),
            {"exit_code": exit_code, "stdout": stdout, "stderr": stderr, "duration_seconds": 0.001},
        )()

    def run(self, command: str, timeout: int = 300):
        del timeout
        self.commands.append(command)
        if command.startswith("python --version"):
            return self._result(
                stdout=(
                    "Python 3.11.10\n"
                    "pip 24.2 from /usr/local/lib/python3.11/site-packages/pip (python 3.11)\n"
                    '{"platform":"linux","architecture":"x86_64"}\n'
                )
            )
        if command.startswith("tar -cf "):
            self.archive = dict(self.files)
            return self._result()
        if command.startswith("test -s "):
            return self._result(0 if self.archive is not None else 1)
        if command.startswith("wc -c < "):
            size = sum(len(value.encode("utf-8")) for value in (self.archive or {}).values())
            return self._result(stdout=f"{size}\n")
        if command.startswith("mkdir -p -- "):
            return self._result()
        if command.startswith("tar -xf "):
            parts = shlex.split(command)
            destination = self._path(parts[parts.index("-C") + 1]).rstrip("/")
            if self.archive is None:
                return self._result(1, stderr="archive missing")
            self.files.update(
                {f"{destination}/{name}": content for name, content in self.archive.items()}
            )
            return self._result()
        if "python -m pip install --dry-run" in command:
            parts = shlex.split(command)
            report_path = self._path(parts[parts.index("--report") + 1])
            requirements_paths = [
                self._path(parts[index + 1])
                for index, part in enumerate(parts[:-1])
                if part == "-r"
            ]
            records = []
            for requirements_path in requirements_paths:
                content = self.files[requirements_path]
                for line in content.splitlines():
                    match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line.strip())
                    if match:
                        records.append(
                            {"metadata": {"name": match.group(1), "version": match.group(2)}}
                        )
            if any(
                record["metadata"]["version"] in self.pip_build_fail_versions for record in records
            ):
                return self._result(
                    1,
                    stderr=(
                        "error: subprocess-exited-with-error\n"
                        "Getting requirements to build wheel did not run successfully.\n"
                        "AttributeError: 'build_ext' object has no attribute 'cython_sources'"
                    ),
                )
            self.files[report_path] = json.dumps({"install": records})
            return self._result()
        if "pipenv lock" in command:
            parts = shlex.split(command)
            cwd = self._path(parts[parts.index("cd") + 1]).strip("/")
            manifest = self.files.get(f"{cwd}/Pipfile")
            match = re.search(
                r'^direct-parent\s*=\s*"(?:==)?([^";]+)"', manifest or "", re.MULTILINE
            )
            if match is None:
                return self._result(1, stderr=f"selected Pipfile parent missing: {manifest!r}")
            locked_parent = self.pipfile_locked_parent or match.group(1)
            self.files[f"{cwd}/Pipfile.lock"] = json.dumps(
                {
                    "default": {
                        "direct-parent": {"version": f"=={locked_parent}"},
                        "certifi": {"version": f"=={self.pipfile_locked_child}"},
                    },
                    "develop": {},
                }
            )
            return self._result()
        if "pipenv sync --dev" in command:
            return self._result()
        if command.startswith("rm -rf -- "):
            return self._result()
        return self._result(1, stderr=f"unexpected command: {command}")


def test_python_uses_common_supervisor_certificate_with_typed_pip_evidence(
    tmp_path: Path,
) -> None:
    requirement = "certifi==2022.6.15\n"
    (tmp_path / "requirements.txt").write_text(requirement, encoding="utf-8")
    group = _python_group()
    task = build_initial_remediation_task(group, "task-1")
    sandbox = _PythonCertifierSandbox(requirement)
    images: list[str | None] = []

    def sandbox_factory(*, repo_root=None, workspace_volume=None, image=None):
        assert repo_root is None
        assert workspace_volume == "workspace-volume"
        images.append(image)
        return sandbox

    def registry(package: str, floor: str, attempted: set[str]):
        assert (package, floor, attempted) == ("certifi", "2022.12.7", set())
        return [_pypi_candidate("2022.12.7"), _pypi_candidate("2024.8.30")]

    plan = build_certified_portfolio_plan(
        tmp_path,
        "workspace-volume",
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        python_registry_fetcher=registry,
        sandbox_factory=sandbox_factory,
        project_language=ProjectLanguage.PYTHON,
    )

    assert images == ["remediation-engine-python:3.11"]
    assert plan.solver_plan is not None and plan.solver_plan.status == SolverStatus.OPTIMAL
    assert plan.resolution_certificate is not None
    certificate = plan.resolution_certificate
    assert certificate.status == PackageResolutionStatus.CERTIFIED
    assert isinstance(certificate.runtime_fingerprint, SolverPythonRuntimeFingerprint)
    assert certificate.runtime_fingerprint.python_version == "3.11.10"
    assert certificate.task_revisions == plan.task_revisions == {task.task_id: task.task_revision}
    assert _portfolio_certificate_violations(plan, {task.task_id: task}) == []
    committed_groups, committed_queue, diagnostics = apply_portfolio_plan(
        plan, [group], {task.task_id: task}
    )
    assert committed_groups[0].group_id == group.group_id
    assert diagnostics == []
    assert committed_queue[task.task_id].selected_version == "2022.12.7"
    assert committed_queue[task.task_id].allowed_target_versions == ["2022.12.7"]
    assert committed_queue[task.task_id].portfolio_plan_id == plan.portfolio_plan_id
    assert committed_queue[task.task_id].task_revision == task.task_revision + 1
    pip_command = next(
        command for command in sandbox.commands if "pip install --dry-run" in command
    )
    pip_arguments = shlex.split(pip_command)
    assert "-r" in pip_arguments
    requirements_argument = pip_arguments[pip_arguments.index("-r") + 1]
    assert requirements_argument.endswith("/requirements.txt")
    assert "npm" not in " ".join(sandbox.commands)


def test_python_certifier_retries_after_candidate_build_failure(tmp_path: Path) -> None:
    requirement = "PyYAML==5.1\n"
    (tmp_path / "requirements.txt").write_text(requirement, encoding="utf-8")
    group = _python_group(
        package_name="pyyaml",
        package_version="5.1",
        fixed_version="5.4.0",
        cve_id="CVE-2020-14343",
    )
    task = build_initial_remediation_task(group, "task-pyyaml")
    sandbox = _PythonCertifierSandbox(
        requirement,
        pip_build_fail_versions={"5.4.0"},
    )

    def registry(package: str, floor: str, attempted: set[str]):
        assert (package, floor, attempted) == ("pyyaml", "5.4.0", set())
        return [_pypi_candidate("5.4.0"), _pypi_candidate("6.0.0")]

    plan = build_certified_portfolio_plan(
        tmp_path,
        "workspace-volume",
        [group],
        {task.task_id: task},
        settings=AppSettings(solver_top_k=2),
        python_registry_fetcher=registry,
        sandbox_factory=lambda **_kwargs: sandbox,
        project_language=ProjectLanguage.PYTHON,
    )

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.OPTIMAL
    assert plan.solver_plan.selected_plan is not None
    assert list(plan.solver_plan.selected_plan.selected_candidate_versions.values()) == ["6.0.0"]
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    assert plan.resolution_certificate.certification_statistics.candidate_attempts == 2
    assert plan.resolution_certificate.certification_statistics.rejection_counts_by_reason == {
        "build_toolchain": 1
    }
    assert sum("python -m pip install --dry-run" in command for command in sandbox.commands) == 2


def test_python_certifier_resolves_same_environment_requirements_files_together(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements-base.txt").write_text("certifi==2022.6.15\n", encoding="utf-8")
    (tmp_path / "requirements-dev.txt").write_text("urllib3==1.25.11\n", encoding="utf-8")
    groups = [
        _python_group("requirements-base.txt"),
        _python_group(
            "requirements-dev.txt",
            package_name="urllib3",
            package_version="1.25.11",
            fixed_version="2.0.0",
            cve_id="CVE-2026-0002",
        ),
    ]
    tasks = {}
    for index, group in enumerate(groups, start=1):
        task = build_initial_remediation_task(group, f"task-{index}")
        tasks[task.task_id] = task
    sandbox = _PythonCertifierSandbox(
        files={
            "requirements-base.txt": "certifi==2022.6.15\n",
            "requirements-dev.txt": "urllib3==1.25.11\n",
        }
    )
    registry_calls: list[tuple[str, str, set[str]]] = []

    def registry(package: str, floor: str, attempted: set[str]):
        registry_calls.append((package, floor, set(attempted)))
        return [_pypi_candidate(floor)]

    plan = build_certified_portfolio_plan(
        tmp_path,
        "workspace-volume",
        groups,
        tasks,
        settings=AppSettings(),
        python_registry_fetcher=registry,
        sandbox_factory=lambda **_kwargs: sandbox,
        project_language=ProjectLanguage.PYTHON,
    )

    assert plan.solver_plan is not None and plan.solver_plan.status == SolverStatus.OPTIMAL
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    assert sorted(package for package, _, _ in registry_calls) == ["certifi", "urllib3"]
    pip_commands = [
        command for command in sandbox.commands if "python -m pip install --dry-run" in command
    ]
    assert len(pip_commands) == 1
    arguments = shlex.split(pip_commands[0])
    requirements_arguments = [
        arguments[index + 1] for index, argument in enumerate(arguments[:-1]) if argument == "-r"
    ]
    assert len(requirements_arguments) == 2
    assert {Path(path).name for path in requirements_arguments} == {
        "requirements-base.txt",
        "requirements-dev.txt",
    }
    assert plan.resolution_certificate.certification_statistics.strict_install_invocations == 1


def test_pipfile_parent_is_certified_and_committed_through_common_plan(tmp_path: Path) -> None:
    _write_pipfile_parent_fixture(tmp_path)
    group = _pipfile_parent_group()
    task = build_initial_remediation_task(group, "task-pipfile")
    pipfile = (tmp_path / "Pipfile").read_text(encoding="utf-8")
    lockfile = (tmp_path / "Pipfile.lock").read_text(encoding="utf-8")
    sandbox = _PythonCertifierSandbox(files={"Pipfile": pipfile, "Pipfile.lock": lockfile})

    def registry(package: str, floor: str, attempted: set[str]):
        assert (package, floor, attempted) == ("direct-parent", "1.0.0", {"1.0.0"})
        return [_pypi_candidate("1.8.0.post1")]

    def requires_dist(package: str, version: str):
        assert package == "direct-parent"
        if version == "1.0.0":
            return ["certifi>=2022.0,<2023.0"]
        return ["certifi>=2022.12.7,<2023.0"]

    plan = build_certified_portfolio_plan(
        tmp_path,
        "workspace-volume",
        [group],
        {task.task_id: task},
        settings=AppSettings(),
        python_registry_fetcher=registry,
        python_parent_requires_dist_fetcher=requires_dist,
        sandbox_factory=lambda **_kwargs: sandbox,
        project_language=ProjectLanguage.PYTHON,
    )

    assert plan.solver_plan is not None
    assert plan.solver_plan.status == SolverStatus.OPTIMAL, plan.diagnostics
    assert plan.resolution_certificate is not None
    assert plan.resolution_certificate.status == PackageResolutionStatus.CERTIFIED
    assert plan.solver_plan.selected_plan is not None
    assert list(plan.solver_plan.selected_plan.selected_candidate_versions.values()) == [
        "1.8.0.post1"
    ]
    assert any("pipenv lock" in command for command in sandbox.commands)
    assert any("pipenv sync --dev" in command for command in sandbox.commands)
    assert not any("npm" in command for command in sandbox.commands)
    assert _portfolio_certificate_violations(plan, {task.task_id: task}) == []

    _groups, committed, diagnostics = apply_portfolio_plan(plan, [group], {task.task_id: task})
    committed_task = committed[task.task_id]
    assert diagnostics == []
    assert committed_task.target_package_name == "direct-parent"
    assert committed_task.parent_package_name == "direct-parent"
    assert committed_task.parent_minimum_version == "1.8.0.post1"
    assert committed_task.selected_version == "1.8.0.post1"
    assert committed_task.allowed_target_versions == ["1.8.0.post1"]


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


def _stage_parity_python_group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="certifi",
        package_version="1.0.0",
        purl="pkg:pypi/certifi@1.0.0",
        ecosystem="pypi",
    )
    return VulnerabilityGroup(
        group_id="sca:Pipfile:certifi",
        issue_type=IssueType.SCA,
        vulnerable_component="certifi",
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _python_task(group: VulnerabilityGroup) -> RemediationTask:
    return RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        status=TaskStatus.NEEDS_RETRY,
        qa_policy=qa_policy_for_task(RoutingStrategy.VERSION_BUMP),
    )


def test_python_pivot_must_use_pypi_latest_stage() -> None:
    group = _stage_parity_python_group()
    task = _python_task(group)
    plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        action="pivot_workaround",
        exact_instruction="Implement a code workaround.",
    )

    violations = _planner_plan_violations(
        {task.task_id: plan},
        {task.task_id: task},
        {},
        {group.group_id: group},
    )

    assert any("not valid for pypi" in item for item in violations)
    assert any("must be committed at pypi_latest" in item for item in violations)


def test_python_exhaustion_repair_pivots_at_pypi_latest() -> None:
    group = _stage_parity_python_group()
    task = _python_task(group)
    plan = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        selected_version="1.2.0",
        latest_version_seen="1.2.0",
        attempted_versions=["1.2.0"],
        action="retry_update",
        exact_instruction="Update certifi to 1.2.0.",
    )

    diagnostics, repaired = _repair_invalid_planner_plans(
        {task.task_id: plan},
        {},
        {task.task_id: task},
        {group.group_id: group},
        violations=[f"task {task.task_id}: selected version 1.2.0 was already attempted"],
    )

    assert repaired[task.task_id].action == "pivot_workaround"
    assert repaired[task.task_id].strategy_stage == SCARemediationStage.PYPI_LATEST
    assert repaired[task.task_id].selected_version is None
    assert diagnostics[task.task_id].strategy_stage == SCARemediationStage.PYPI_LATEST


def _maven_group(
    *,
    direct: bool,
    declaration_type: str,
    coordinate: str = "org.example:widget",
    manifest_path: str = "module/pom.xml",
) -> VulnerabilityGroup:
    group_id, artifact_id = coordinate.split(":", 1)
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=coordinate,
        package_version="1.0",
        purl=f"pkg:maven/{group_id}/{artifact_id}@1.0",
        ecosystem="Maven",
        cve_id="CVE-2025-12345",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest_path,
        is_direct_dependency=direct,
        package_manager="maven",
        declaration_type=declaration_type,
        localization_confidence=1.0,
    )
    return VulnerabilityGroup(
        group_id=f"maven-{uuid4()}",
        issue_type=IssueType.SCA,
        vulnerable_component=coordinate,
        file_path=manifest_path,
        file_paths=[manifest_path],
        cve_ids=["CVE-2025-12345"],
        versions=["1.0"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
        parent_package_name=None if direct else "org.example:parent",
        parent_package_version=None if direct else "2.0",
        parent_declaration_type=None if direct else "dependencies",
        fix_plan_candidates=[
            PackageFixPlanCandidate(
                issue_id=issue.id,
                plan=FixPlan(
                    status=FixPlanStatus.VERSION_FOUND,
                    fixed_version="1.1",
                    instruction="Use a Supervisor-approved Maven Central candidate.",
                    strategy_used="osv_api",
                ),
            )
        ],
    )


def test_java_atomic_action_commits_full_native_maven_target_scope(tmp_path: Path) -> None:
    widget_group = _maven_group(
        direct=True,
        declaration_type="dependencies",
        manifest_path="module-a/pom.xml",
    )
    library_group = _maven_group(
        direct=True,
        declaration_type="dependencies",
        coordinate="org.example:library",
        manifest_path="module-b/pom.xml",
    )
    groups = {group.group_id: group for group in (widget_group, library_group)}
    for group in groups.values():
        localized = group.localized_issues[0]
        pom_path = tmp_path / localized.manifest_file
        pom_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = group.vulnerable_component.split(":", 1)[1]
        pom_path.write_text(
            '<project xmlns="http://maven.apache.org/POM/4.0.0">'
            "<modelVersion>4.0.0</modelVersion><groupId>com.acme</groupId>"
            f"<artifactId>{artifact}-module</artifactId><version>1.0.0</version>"
            "<dependencies><dependency><groupId>org.example</groupId>"
            f"<artifactId>{artifact}</artifactId><version>1.0.0</version>"
            "</dependency></dependencies></project>",
            encoding="utf-8",
        )
    tasks = {
        "task-widget": build_initial_remediation_task(
            widget_group,
            "task-widget",
            project_language=ProjectLanguage.JAVA,
        ).model_copy(update={"selected_version": "1.2.0"}),
        "task-library": build_initial_remediation_task(
            library_group,
            "task-library",
            project_language=ProjectLanguage.JAVA,
        ).model_copy(update={"selected_version": "1.2.0"}),
    }

    action = _build_multi_package_action(
        "cluster-java",
        tasks,
        tasks,
        groups,
        project_language=ProjectLanguage.JAVA,
        repo_root=str(tmp_path),
    )

    assert action is not None
    assert [mutation.ecosystem for mutation in action.package_mutations] == ["maven", "maven"]
    assert [mutation.target_manifest_paths for mutation in action.package_mutations] == [
        ["module-a/pom.xml"],
        ["module-b/pom.xml"],
    ]
    assert all(
        mutation.maven_target_operation == MavenTargetOperation.UPDATE_DECLARATION
        for mutation in action.package_mutations
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


def test_maven_retry_plan_validation_preserves_native_version_identity_and_pivot_stage() -> None:
    group = _maven_group(direct=True, declaration_type="dependencies")
    task = build_initial_remediation_task(
        group,
        "task-plan-validation",
        project_language=ProjectLanguage.JAVA,
    )
    retry = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.MAVEN_LATEST,
        version_scheme="maven",
        selected_version="v1.0",
        attempted_versions=["1.0"],
        candidate_versions_considered=["v1.0"],
        exact_instruction="Apply the authorized Maven version update.",
    )
    group_by_id = {group.group_id: group}
    assert (
        _planner_plan_violations(
            {task.task_id: retry},
            {task.task_id: task},
            {},
            group_by_id,
            project_language=ProjectLanguage.JAVA,
        )
        == []
    )

    pivot = SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=SCARemediationStage.MAVEN_LATEST,
        version_scheme="maven",
        exhausted_update_path=True,
        action="pivot_workaround",
        exact_instruction="Use the committed workaround path.",
    )
    assert (
        _planner_plan_violations(
            {task.task_id: pivot},
            {task.task_id: task},
            {},
            group_by_id,
            project_language=ProjectLanguage.JAVA,
        )
        == []
    )


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
        "remediation_engine.agents.supervisor_policy._supervisor_fetch_registry_candidates",
        fetch_candidates,
    )
    monkeypatch.setattr(
        "remediation_engine.agents.supervisor_policy._supervisor_plan_npm_parent_version",
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
        qa_policy=qa_policy_for_task(RoutingStrategy.VERSION_BUMP),
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
