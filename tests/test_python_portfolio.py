"""Common Supervisor portfolio coverage for directly declared PyPI packages."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.contracts.solver_models import (
    PackageResolutionStatus,
    SolverPythonRuntimeFingerprint,
    SolverStatus,
)
from remediation_engine.contracts.version_policy import RegistryCandidate
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.portfolio_certifier import (
    build_certified_portfolio_plan,
)
from remediation_engine.orchestration.portfolio_solver import (
    apply_portfolio_plan,
    build_python_portfolio_plan,
    prepare_python_portfolio_problem,
)
from remediation_engine.orchestration.supervisor_portfolio import (
    _portfolio_certificate_violations,
)
from remediation_engine.orchestration.task_utils import build_initial_remediation_task
from remediation_engine.settings import AppSettings
from remediation_engine.triage.grouper import group_issues


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
    return group_issues([issue], sca_issue_plans=[(localized, fix)])[0]


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
    from remediation_engine.orchestration.portfolio_solver import (
        _python_numeric_solver_version,
    )

    assert _python_numeric_solver_version("1.10")[1] > _python_numeric_solver_version("1.9.2")[1]
    assert _python_numeric_solver_version("1.0")[1] == _python_numeric_solver_version("1.0.0")[1]
    assert _python_numeric_solver_version("1.0")[0] == "1.0.0"
    assert _python_numeric_solver_version("1.8.0")[1] < _python_numeric_solver_version(
        "1.8.0.post1"
    )[1]
    assert _python_numeric_solver_version("1.8.0.post1")[1] < _python_numeric_solver_version(
        "1.8.1"
    )[1]


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
    from remediation_engine.orchestration.portfolio_solver import _python_numeric_solver_version

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
        return [
            _pypi_candidate(version)
            for version in ("1.2.0", "1.8.0.post1", "2.0.0")
        ]

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
    assert prepared.parent_contexts[prepared.targets[0].occurrence_id]["parent_minimum_version"] == "1.8.0.post1"
    assert prepared.targets[0].target_package_name == "direct-parent"
    assert prepared.targets[0].lockfile_package_key is None
    assert prepared.findings[0].vulnerable_package == "certifi"
    assert prepared.findings[0].direct_parent_name == "direct-parent"
    assert prepared.findings[0].direct_parent_minimum_version == "1.8.0.post1"
    assert prepared.findings[0].is_transitive
    assert solver.status == SolverStatus.OPTIMAL
    assert solver.selected_plan is not None
    selected = solver.selected_plan.selected_candidate_versions[prepared.targets[0].occurrence_id]
    assert _python_numeric_solver_version(selected)[1] >= _python_numeric_solver_version("1.8.0.post1")[1]
    assert registry_calls == [
        ("direct-parent", "1.0.0", {"1.0.0", "1.2.0"})
    ]


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
                    'Python 3.11.10\n'
                    'pip 24.2 from /usr/local/lib/python3.11/site-packages/pip (python 3.11)\n'
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
            if any(record["metadata"]["version"] in self.pip_build_fail_versions for record in records):
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
    pip_command = next(command for command in sandbox.commands if "pip install --dry-run" in command)
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
    (tmp_path / "requirements-base.txt").write_text(
        "certifi==2022.6.15\n", encoding="utf-8"
    )
    (tmp_path / "requirements-dev.txt").write_text(
        "urllib3==1.25.11\n", encoding="utf-8"
    )
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
        arguments[index + 1]
        for index, argument in enumerate(arguments[:-1])
        if argument == "-r"
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
    sandbox = _PythonCertifierSandbox(
        files={"Pipfile": pipfile, "Pipfile.lock": lockfile}
    )

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

    _groups, committed, diagnostics = apply_portfolio_plan(
        plan, [group], {task.task_id: task}
    )
    committed_task = committed[task.task_id]
    assert diagnostics == []
    assert committed_task.target_package_name == "direct-parent"
    assert committed_task.parent_package_name == "direct-parent"
    assert committed_task.parent_minimum_version == "1.8.0.post1"
    assert committed_task.selected_version == "1.8.0.post1"
    assert committed_task.allowed_target_versions == ["1.8.0.post1"]
