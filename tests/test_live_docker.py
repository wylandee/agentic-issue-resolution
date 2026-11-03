"""Opt-in Docker integration coverage for generated targeted ODC artifacts."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from uuid import uuid4

import pytest

from remediation_engine.language import ProjectLanguage
from remediation_engine.runtime.sandbox import DockerSandbox, get_docker_client
from remediation_engine.tools.npm import (
    build_sliced_lockfile_artifacts,
    load_npm_graph_snapshot_from_documents,
    resolve_dependency_closure,
)
from remediation_engine.tools.odc_parser import parse_python_packages
from remediation_engine.tools.qa_scan import _parse_report_package_occurrences, _run_targeted_odc
from remediation_engine.tools.qa_workspace import _write_targeted_artifacts
from remediation_engine.tools.test_execution import (
    _install_error_category,
    parse_peer_conflict_evidence,
)


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


def _write_json_fixture(root: Path, relative: str, payload: dict[str, object]) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")
    return content


@pytest.mark.integration
@pytest.mark.docker
@pytest.mark.skipif(
    os.environ.get("REMEDY_RUN_DOCKER_TESTS") != "1" or shutil.which("docker") is None,
    reason="opt-in live Docker validation; set REMEDY_RUN_DOCKER_TESTS=1",
)
def test_strict_npm_resolution_certifies_nested_lockfile(tmp_path: Path):
    pytest.importorskip("docker")
    from remediation_engine.runtime.sandbox import DockerSandbox, get_docker_client

    try:
        client = get_docker_client()
    except RuntimeError as exc:
        pytest.skip(f"Docker daemon is not available: {exc}")
    try:
        client.images.get("node:22")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"node:22 image is not available locally: {exc}")
    finally:
        client.close()

    packages = {
        "name": "app",
        "dependencies": {
            "parent-a": "file:packages/parent-a",
            "parent-b": "file:packages/parent-b",
            "peer-host": "file:packages/peer-host",
        },
    }
    _write_json_fixture(tmp_path, "package.json", packages)
    _write_json_fixture(
        tmp_path,
        "packages/parent-a/package.json",
        {
            "name": "parent-a",
            "version": "1.0.0",
            "dependencies": {"child": "file:../child-v1"},
            "peerDependencies": {"peer-host": "^2.0.0"},
        },
    )
    _write_json_fixture(
        tmp_path,
        "packages/parent-b/package.json",
        {
            "name": "parent-b",
            "version": "1.0.0",
            "dependencies": {"child": "file:../child-v2"},
        },
    )
    _write_json_fixture(
        tmp_path, "packages/child-v1/package.json", {"name": "child", "version": "1.0.0"}
    )
    _write_json_fixture(
        tmp_path, "packages/child-v2/package.json", {"name": "child", "version": "2.0.0"}
    )
    _write_json_fixture(
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
