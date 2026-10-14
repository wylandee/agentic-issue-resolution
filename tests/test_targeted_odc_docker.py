"""Opt-in Docker integration coverage for generated targeted ODC artifacts."""

from __future__ import annotations

import json
import os
import shutil
from uuid import uuid4

import pytest

from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.qa_odc import _run_targeted_odc
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client
from remediation_engine.tools.lockfile_closure import (
    build_sliced_lockfile_artifacts,
    resolve_dependency_closure,
)
from remediation_engine.tools.odc_parser import parse_python_packages


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("REMEDY_RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="opt-in live Docker validation; set REMEDY_RUN_DOCKER_TESTS=1",
)
def test_generated_targeted_lockfile_produces_parseable_odc_report() -> None:
    """Verify the synthetic package-lock shape against the real ODC container."""
    docker = pytest.importorskip("docker")
    del docker
    closure = resolve_dependency_closure(
        {
            "": {},
            "node_modules/a": {"version": "1.0.0"},
        },
        target_package="a",
        target_version="1.0.0",
    )
    artifacts = build_sliced_lockfile_artifacts(closure)
    client = get_docker_client()
    volume_name = f"remedy-targeted-test-{uuid4().hex[:12]}"
    client.volumes.create(name=volume_name)
    try:
        with DockerSandbox(repo_root=None, workspace_volume=volume_name) as sandbox:
            for filename, content in artifacts.items():
                sandbox.write_file(f".odc-targeted/000/{filename}", content)
            result = _run_targeted_odc(volume_name, ".odc-targeted")
            assert result.returncode == 0, result.stderr
            report = sandbox.read_file(".odc-targeted/dependency-check-report.json")
            assert report is not None
    finally:
        try:
            client.volumes.get(volume_name).remove(force=True)
        finally:
            client.close()


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("REMEDY_RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="opt-in live Docker validation; set REMEDY_RUN_DOCKER_TESTS=1",
)
def test_generated_python_requirements_produce_target_package_inventory() -> None:
    """Verify ODC recognizes a synthetic pinned requirements.txt artifact."""
    docker = pytest.importorskip("docker")
    del docker
    client = get_docker_client()
    volume_name = f"remedy-targeted-python-test-{uuid4().hex[:12]}"
    client.volumes.create(name=volume_name)
    try:
        with DockerSandbox(repo_root=None, workspace_volume=volume_name) as sandbox:
            sandbox.write_file(
                ".odc-targeted/000/requirements.txt",
                "certifi==2024.7.4\n",
            )
            result = _run_targeted_odc(
                volume_name,
                ".odc-targeted",
                ProjectLanguage.PYTHON,
            )
            assert result.returncode == 0, result.stderr
            report = sandbox.read_file(".odc-targeted/dependency-check-report.json")
            assert report is not None
            assert ("certifi", "2024.7.4") in parse_python_packages(json.loads(report))
    finally:
        try:
            client.volumes.get(volume_name).remove(force=True)
        finally:
            client.close()
