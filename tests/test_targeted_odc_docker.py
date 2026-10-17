"""Opt-in Docker integration coverage for generated targeted ODC artifacts."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from uuid import uuid4

import pytest

from remediation_engine.orchestration._qa_runtime import _write_targeted_artifacts
from remediation_engine.orchestration.qa_odc import (
    _parse_report_package_occurrences,
    _run_targeted_odc,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox, get_docker_client
from remediation_engine.tools.lockfile_closure import resolve_dependency_closure


@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("REMEDY_RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="opt-in live Docker validation; set REMEDY_RUN_DOCKER_TESTS=1",
)
def test_generated_targeted_lockfile_reports_all_physical_occurrences() -> None:
    """Verify one-lockfile artifacts retain package occurrences in real ODC."""
    docker = pytest.importorskip("docker")
    del docker
    packages = {
        "": {
            "name": "remedy-targeted-scan",
            "version": "1.0.0",
            "dependencies": {"jws": "3.2.3"},
        },
        "node_modules/jws": {
            "version": "3.2.3",
            "dependencies": {"jwa": "^1.4.2"},
        },
        "node_modules/jwa": {
            "version": "1.4.2",
            "dependencies": {
                "buffer-equal-constant-time": "^1.0.1",
                "ecdsa-sig-formatter": "1.0.11",
            },
        },
        "node_modules/buffer-equal-constant-time": {"version": "1.0.1"},
        "node_modules/ecdsa-sig-formatter": {
            "version": "1.0.11",
            "dependencies": {"safe-buffer": "^5.0.1"},
        },
        "node_modules/safe-buffer": {"version": "5.2.1"},
    }
    lockfile = {
        "name": "remediation-engine-targeted-scan",
        "version": "0.0.0",
        "lockfileVersion": 3,
        "requires": True,
        "packages": packages,
    }
    package_json = {
        "name": "remediation-engine-targeted-scan",
        "version": "0.0.0",
        "private": True,
        "dependencies": {"jws": "3.2.3"},
    }
    closure = resolve_dependency_closure(
        packages,
        source_lockfile="package-lock.json",
        target_package="jws",
        target_version="3.2.3",
        lockfile_version=3,
    )
    assert closure.complete

    client = get_docker_client()
    volume_name = f"remedy-targeted-test-{uuid4().hex[:12]}"
    client.volumes.create(name=volume_name)
    try:
        with DockerSandbox(repo_root=None, workspace_volume=volume_name) as sandbox:
            sandbox.write_file("package-lock.json", json.dumps(lockfile))
            sandbox.write_file("package.json", json.dumps(package_json))
            manifest = _write_targeted_artifacts(sandbox, {"task-1": (closure,)})

            assert manifest.expected_package_occurrences == (
                ("buffer-equal-constant-time", "1.0.1"),
                ("ecdsa-sig-formatter", "1.0.11"),
                ("jwa", "1.4.2"),
                ("jws", "3.2.3"),
                ("safe-buffer", "5.2.1"),
            )
            assert set(manifest.input_paths_by_project) == {".odc-targeted/000"}
            result = _run_targeted_odc(volume_name, manifest.scan_subdir)
            assert result.returncode == 0, result.stderr

            report_text = sandbox.read_file(".odc-targeted/dependency-check-report.json")
            assert report_text is not None
            occurrences = _parse_report_package_occurrences(report_text)
            assert occurrences is not None
            reported = Counter(
                (item.package_name.casefold(), item.version)
                for item in occurrences
                if item.ecosystem.casefold() == "npm"
            )
            expected = Counter(
                (name.casefold(), version)
                for name, version in manifest.expected_package_occurrences
            )
            assert not expected - reported
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
def test_trace_esbuild_target_scans_after_batch_staging() -> None:
    """Verify batch-staged esbuild 0.28.1 artifacts are readable by real ODC."""
    docker = pytest.importorskip("docker")
    del docker
    target_version = "0.28.1"
    packages = {
        "": {
            "name": "juice-shop",
            "version": "1.0.0",
            "dependencies": {"esbuild": target_version},
        },
        "node_modules/esbuild": {"version": target_version},
    }
    lockfile = {
        "name": "juice-shop",
        "version": "1.0.0",
        "lockfileVersion": 3,
        "requires": True,
        "packages": packages,
    }
    package_json = {
        "name": "juice-shop",
        "version": "1.0.0",
        "private": True,
        "dependencies": {"esbuild": target_version},
    }
    closure = resolve_dependency_closure(
        packages,
        source_lockfile="package-lock.json",
        target_package="esbuild",
        target_version=target_version,
        lockfile_version=3,
    )
    assert closure.complete

    client = get_docker_client()
    volume_name = f"remedy-esbuild-targeted-test-{uuid4().hex[:12]}"
    client.volumes.create(name=volume_name)
    try:
        with DockerSandbox(repo_root=None, workspace_volume=volume_name) as sandbox:
            sandbox.write_files(
                {
                    "package-lock.json": json.dumps(lockfile),
                    "package.json": json.dumps(package_json),
                }
            )
            manifest = _write_targeted_artifacts(sandbox, {"task-3": (closure,)})

            assert manifest.expected_package_occurrences == (("esbuild", target_version),)
            result = _run_targeted_odc(volume_name, manifest.scan_subdir)
            assert result.returncode == 0, result.stderr

            report_text = sandbox.read_file(".odc-targeted/dependency-check-report.json")
            assert report_text is not None
            occurrences = _parse_report_package_occurrences(report_text)
            assert occurrences is not None
            assert any(
                occurrence.package_name.casefold() == "esbuild"
                and occurrence.version == target_version
                and occurrence.ecosystem.casefold() == "npm"
                for occurrence in occurrences
            )
    finally:
        try:
            client.volumes.get(volume_name).remove(force=True)
        finally:
            client.close()
