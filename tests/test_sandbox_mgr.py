"""
tests/test_sandbox_mgr.py â€” Unit tests for remediation_engine.runtime.sandbox_mgr.

All Docker SDK interactions are mocked; no real Docker daemon is required.
"""

from __future__ import annotations

import io
import tarfile
import types
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts import CommandResult
from remediation_engine.runtime.sandbox_mgr import (
    DockerSandbox,
    _make_tar_archive,
    get_docker_client,
)


class _ImageNotFound(Exception):
    pass


class _APIError(Exception):
    pass


class _NotFound(Exception):
    pass


def _docker_modules(image_found: bool = True):
    docker_mod = MagicMock()
    docker_errors = types.SimpleNamespace(
        ImageNotFound=_ImageNotFound,
        APIError=_APIError,
        NotFound=_NotFound,
    )
    docker_mod.errors = docker_errors

    client = MagicMock()
    docker_mod.from_env.return_value = client
    client.ping.return_value = True

    if image_found:
        client.images.get.return_value = MagicMock()
    else:
        client.images.get.side_effect = _ImageNotFound("missing")

    container = MagicMock()
    container.exec_run.return_value = (0, (b"stdout-data", b"stderr-data"))
    container.put_archive.return_value = True
    client.containers.run.return_value = container

    return docker_mod, docker_errors, client, container


class TestCommandResult:
    def test_json_round_trip(self):
        result = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=1.0)
        restored = CommandResult.model_validate_json(result.model_dump_json())
        assert restored.exit_code == 0
        assert restored.stdout == "ok"

    def test_duration_non_negative(self):
        with pytest.raises(Exception):
            CommandResult(exit_code=0, duration_seconds=-1.0)


class TestDockerClientHelper:
    def test_get_docker_client_connects_and_pings(self):
        docker_mod, docker_errors, client, _container = _docker_modules()
        with patch.dict(
            "sys.modules",
            {"docker": docker_mod, "docker.errors": docker_errors},
        ):
            connected = get_docker_client()

        assert connected is client
        client.ping.assert_called_once()

    def test_get_docker_client_raises_on_connection_error(self):
        docker_mod, docker_errors, client, _container = _docker_modules()
        client.ping.side_effect = Exception("Connection refused")
        with (
            patch.dict(
                "sys.modules",
                {"docker": docker_mod, "docker.errors": docker_errors},
            ),
            pytest.raises(RuntimeError, match="Docker daemon unreachable"),
        ):
            get_docker_client()


class TestTarArchiveHelper:
    def test_archive_contains_repo_files(self, tmp_path):
        (tmp_path / "index.js").write_text("console.log('hi')", encoding="utf-8")
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")

        archive = _make_tar_archive(tmp_path)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tf:
            names = tf.getnames()

        assert "index.js" in names
        assert "package.json" in names

    def test_archive_ignores_git_and_node_modules(self, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("[core]", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "leftpad.js").write_text(
            "module.exports = {}", encoding="utf-8"
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "index.js").write_text("console.log('ok')", encoding="utf-8")

        archive = _make_tar_archive(tmp_path)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tf:
            names = tf.getnames()

        assert "src/index.js" in names
        assert not any(name.startswith(".git") for name in names)
        assert not any(name.startswith("node_modules") for name in names)


class TestSandboxLifecycle:
    def test_start_without_volume_streams_repo(self, tmp_path):
        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        docker_mod, docker_errors, client, container = _docker_modules()

        with patch.dict(
            "sys.modules",
            {"docker": docker_mod, "docker.errors": docker_errors},
        ):
            sandbox = DockerSandbox(tmp_path)
            sandbox.start()
            sandbox.teardown()

        assert container.put_archive.called
        args, _kwargs = container.put_archive.call_args
        assert args[0] == "/workspace"

    def test_start_with_workspace_volume_mounts_named_volume(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()

        with patch.dict(
            "sys.modules",
            {"docker": docker_mod, "docker.errors": docker_errors},
        ):
            sandbox = DockerSandbox(tmp_path, workspace_volume="agent_workspace_deadbeef")
            sandbox.start()
            sandbox.teardown()

        _args, kwargs = client.containers.run.call_args
        assert kwargs["volumes"] == {
            "agent_workspace_deadbeef": {"bind": "/workspace", "mode": "rw"}
        }

    def test_repo_root_none_skips_archive_copy(self):
        docker_mod, docker_errors, client, container = _docker_modules()

        with patch.dict(
            "sys.modules",
            {"docker": docker_mod, "docker.errors": docker_errors},
        ):
            sandbox = DockerSandbox(
                repo_root=None,
                workspace_volume="agent_workspace_deadbeef",
            )
            sandbox.start()
            sandbox.teardown()

        container.put_archive.assert_not_called()

    def test_missing_image_triggers_pull(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules(image_found=False)

        with patch.dict(
            "sys.modules",
            {"docker": docker_mod, "docker.errors": docker_errors},
        ):
            sandbox = DockerSandbox(tmp_path)
            sandbox.start()
            sandbox.teardown()

        client.images.pull.assert_called_once()


class TestSandboxRun:
    def _started_sandbox(self, tmp_path, client, container):
        sandbox = DockerSandbox(tmp_path)
        sandbox._client = client
        sandbox._container = container
        sandbox._alive = True
        return sandbox

    def test_run_wraps_command_in_sh_lc(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()
        sandbox = self._started_sandbox(tmp_path, client, container)

        result = sandbox.run("npm test")

        args, kwargs = container.exec_run.call_args
        assert args[0] == ["/bin/sh", "-lc", "npm test"]
        assert kwargs["workdir"] == "/workspace"
        assert result.exit_code == 0

    def test_run_returns_timeout_result(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()
        container.exec_run.side_effect = Exception("request timed out")
        sandbox = self._started_sandbox(tmp_path, client, container)

        result = sandbox.run("sleep 9999", timeout=1)

        assert result.exit_code == 124
        assert sandbox._alive is False

    def test_run_on_dead_sandbox_returns_error(self, tmp_path):
        sandbox = DockerSandbox(tmp_path)
        result = sandbox.run("echo hi")
        assert result.exit_code == 1
        assert "not running" in result.stderr

    def test_traced_methods_remain_callable_when_sandbox_is_not_running(self, tmp_path):
        sandbox = DockerSandbox(tmp_path)

        assert sandbox.run("echo hi").stderr == "Sandbox is not running."
        assert sandbox.read_file("package.json") is None
        with pytest.raises(RuntimeError, match="Sandbox is not running."):
            sandbox.write_file("package.json", "{}")


def _tar_chunks(filename: str, content: str) -> list[bytes]:
    buf = io.BytesIO()
    encoded = content.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=filename)
        info.size = len(encoded)
        tf.addfile(info, io.BytesIO(encoded))
    return [buf.getvalue()]


class TestSandboxFileIO:
    def _started_sandbox(self, tmp_path, client, container):
        sandbox = DockerSandbox(tmp_path)
        sandbox._client = client
        sandbox._container = container
        sandbox._alive = True
        return sandbox

    def test_write_file_puts_archive_into_parent_directory(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()
        sandbox = self._started_sandbox(tmp_path, client, container)

        sandbox.write_file("config/overrides.json", "{}")

        args, _kwargs = container.put_archive.call_args
        assert args[0] == "/workspace/config"

    def test_read_file_uses_workspace_path(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()
        container.get_archive.return_value = (_tar_chunks("package.json", "{}"), {})
        sandbox = self._started_sandbox(tmp_path, client, container)

        result = sandbox.read_file("package.json")

        args, _kwargs = container.get_archive.call_args
        assert args[0] == "/workspace/package.json"
        assert result == "{}"

    def test_read_file_returns_none_when_missing(self, tmp_path):
        docker_mod, docker_errors, client, container = _docker_modules()
        container.get_archive.side_effect = Exception("missing")
        sandbox = self._started_sandbox(tmp_path, client, container)

        assert sandbox.read_file("missing.json") is None
