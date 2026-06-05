"""
tests/test_sandbox_mgr.py — Unit tests for src/runtime/sandbox_mgr.py.

All tests mock the Docker SDK so no real Docker daemon is required.

Covers:
- CommandResult contract: validation, JSON round-trip, export from src.contracts
- Sandbox lifecycle: daemon ping, image pull fallback, unique container names,
  no bind mounts, archive copy into /workspace, context-manager teardown on
  exceptions, idempotent teardown
- Command execution: stdout/stderr decoding, duration tracking, /workspace
  working directory, non-zero exit handling, timeout (exit_code=124), and
  sandbox kill on timeout
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.contracts import CommandResult
from src.runtime.sandbox_mgr import DockerSandbox, _make_tar_archive


# ---------------------------------------------------------------------------
# CommandResult contract
# ---------------------------------------------------------------------------


class TestCommandResult:
    def test_basic_construction(self):
        r = CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=1.5)
        assert r.exit_code == 0
        assert r.stdout == "ok"
        assert r.stderr == ""
        assert r.duration_seconds == 1.5

    def test_defaults(self):
        r = CommandResult(exit_code=1, duration_seconds=0.0)
        assert r.stdout == ""
        assert r.stderr == ""

    def test_json_round_trip(self):
        r = CommandResult(exit_code=0, stdout="hello", stderr="warn", duration_seconds=2.0)
        restored = CommandResult.model_validate_json(r.model_dump_json())
        assert restored.exit_code == 0
        assert restored.stdout == "hello"
        assert restored.stderr == "warn"
        assert restored.duration_seconds == 2.0

    def test_exported_from_contracts(self):
        """CommandResult must be importable from src.contracts."""
        from src.contracts import CommandResult as CR  # noqa: F401
        assert CR is CommandResult

    def test_duration_non_negative(self):
        with pytest.raises(Exception):
            CommandResult(exit_code=0, duration_seconds=-1.0)

    def test_frozen(self):
        r = CommandResult(exit_code=0, duration_seconds=0.5)
        with pytest.raises(Exception):
            r.exit_code = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_docker(image_found: bool = True):
    """Return a mock docker.from_env() client with a pre-configured container."""
    docker_mod = MagicMock()
    client = MagicMock()
    docker_mod.from_env.return_value = client

    # ping succeeds
    client.ping.return_value = True

    # image lookup
    if image_found:
        client.images.get.return_value = MagicMock()
    else:
        import docker.errors
        client.images.get.side_effect = docker.errors.ImageNotFound("nope")
        client.images.pull.return_value = MagicMock()

    # container
    container = MagicMock()
    client.containers.run.return_value = container
    container.exec_run.return_value = (0, (b"stdout-data", b"stderr-data"))
    container.put_archive.return_value = True

    return docker_mod, client, container


# ---------------------------------------------------------------------------
# Sandbox lifecycle
# ---------------------------------------------------------------------------


class TestSandboxLifecycle:
    def test_unique_container_names(self):
        sb1 = DockerSandbox("/tmp/repo")
        sb2 = DockerSandbox("/tmp/repo")
        assert sb1._container_name != sb2._container_name
        assert sb1._container_name.startswith("sandbox-")

    def test_ping_called_on_start(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            client.ping.assert_called_once()
            sb.teardown()

    def test_daemon_unreachable_raises(self, tmp_path):
        docker_mod = MagicMock()
        client = MagicMock()
        docker_mod.from_env.return_value = client
        client.ping.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            with pytest.raises(RuntimeError, match="Docker daemon unreachable"):
                sb.start()

    def test_image_found_locally_no_pull(self, tmp_path):
        docker_mod, client, container = _make_mock_docker(image_found=True)
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            client.images.pull.assert_not_called()
            sb.teardown()

    def test_image_not_found_pulls(self, tmp_path):
        import docker.errors as _de
        docker_mod, client, container = _make_mock_docker(image_found=True)
        # Override image.get to raise the real ImageNotFound (inherits from BaseException)
        client.images.get.side_effect = _de.ImageNotFound("nope")
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            # Patch the errors attribute on the mock to point to the real errors module
            docker_mod.errors = _de
            sb = DockerSandbox(tmp_path)
            sb.start()
            client.images.pull.assert_called_once()
            sb.teardown()

    def test_no_bind_mounts_in_container_run(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            _, kwargs = client.containers.run.call_args
            assert "volumes" not in kwargs
            assert "binds" not in kwargs
            assert kwargs.get("network_mode") == "bridge"
            assert not kwargs.get("privileged", False)
            sb.teardown()

    def test_repo_copied_via_put_archive(self, tmp_path):
        (tmp_path / "package.json").write_text('{"name":"test"}')
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            # put_archive must have been called with /workspace as the target
            assert container.put_archive.called
            args, _ = container.put_archive.call_args
            assert args[0] == "/workspace"
            sb.teardown()

    def test_context_manager_starts_and_tears_down(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            with DockerSandbox(tmp_path) as sb:
                assert sb._alive is True
            assert sb._container is None
            assert sb._alive is False

    def test_context_manager_tears_down_on_exception(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            with pytest.raises(ValueError):
                with DockerSandbox(tmp_path) as sb:
                    raise ValueError("boom")
            assert sb._container is None

    def test_teardown_idempotent(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            sb.teardown()
            sb.teardown()  # second call must not raise
            sb.teardown()  # third call must not raise

    def test_teardown_tolerates_already_removed(self, tmp_path):
        import types
        docker_mod, client, container = _make_mock_docker()
        # Simulate container already gone
        err_mod = types.SimpleNamespace(
            APIError=Exception,
            NotFound=Exception,
            ImageNotFound=Exception,
        )
        container.kill.side_effect = Exception("no such container")
        container.remove.side_effect = Exception("no such container")
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": err_mod}):
            sb = DockerSandbox(tmp_path)
            sb.start()
            sb.teardown()  # must not raise


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


class TestCommandExecution:
    def _started_sandbox(self, tmp_path, docker_mod, client, container):
        sb = DockerSandbox(tmp_path)
        sb._client = client
        sb._container = container
        sb._alive = True
        return sb

    def test_stdout_stderr_decoded(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (b"hello stdout", b"hello stderr"))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            result = sb.run("echo hi")
            assert result.stdout == "hello stdout"
            assert result.stderr == "hello stderr"
            assert result.exit_code == 0

    def test_working_directory_is_workspace(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (b"", b""))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            sb.run("pwd")
            _, kwargs = container.exec_run.call_args
            assert kwargs.get("workdir") == "/workspace"

    def test_command_wrapped_in_sh_lc(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (b"", b""))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            sb.run("npm install")
            positional_args, _ = container.exec_run.call_args
            cmd = positional_args[0]
            assert cmd == ["/bin/sh", "-lc", "npm install"]

    def test_nonzero_exit_code_returned(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (1, (b"", b"some error"))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            result = sb.run("false")
            assert result.exit_code == 1
            assert result.stderr == "some error"

    def test_duration_tracked(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (b"", b""))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            result = sb.run("true")
            assert result.duration_seconds >= 0.0

    def test_timeout_returns_exit_code_124(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.side_effect = Exception("request timed out")
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            result = sb.run("sleep 9999", timeout=1)
            assert result.exit_code == 124

    def test_timeout_kills_container(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.side_effect = Exception("request timed out")
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            sb.run("sleep 9999", timeout=1)
            assert sb._alive is False
            assert sb._container is None

    def test_run_on_dead_sandbox_returns_error(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        sb = DockerSandbox(tmp_path)
        # Never started — _alive is False
        result = sb.run("echo hi")
        assert result.exit_code == 1
        assert "not running" in result.stderr

    def test_none_stdout_handled(self, tmp_path):
        """exec_run may return None for stdout or stderr if stream is empty."""
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (None, None))
        with patch.dict("sys.modules", {"docker": docker_mod, "docker.errors": docker_mod.errors}):
            sb = self._started_sandbox(tmp_path, docker_mod, client, container)
            result = sb.run("true")
            assert result.stdout == ""
            assert result.stderr == ""


# ---------------------------------------------------------------------------
# Tar archive helper
# ---------------------------------------------------------------------------


class TestMakeTarArchive:
    def test_archive_contains_repo_files(self, tmp_path):
        (tmp_path / "index.js").write_text("console.log('hi')")
        (tmp_path / "package.json").write_text("{}")

        archive = _make_tar_archive(tmp_path)
        buf = io.BytesIO(archive)
        with tarfile.open(fileobj=buf, mode="r") as tf:
            names = tf.getnames()
        assert any("index.js" in n for n in names)
        assert any("package.json" in n for n in names)

    def test_archive_non_empty(self, tmp_path):
        (tmp_path / "README.md").write_text("hello")
        archive = _make_tar_archive(tmp_path)
        assert len(archive) > 0


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def _started_sandbox(self, tmp_path, client, container):
        sb = DockerSandbox(tmp_path)
        sb._client = client
        sb._container = container
        sb._alive = True
        return sb

    def test_write_calls_put_archive(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("package.json", '{"name":"test"}')
        assert container.put_archive.called

    def test_write_targets_parent_directory(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("config/overrides.json", "{}")
        args, _ = container.put_archive.call_args
        # First argument to put_archive is the destination directory
        assert args[0] == "/workspace/config"

    def test_write_archive_contains_filename(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("package.json", '{"name":"test"}')
        args, _ = container.put_archive.call_args
        raw_tar = args[1]
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r") as tf:
            names = tf.getnames()
        assert "package.json" in names

    def test_write_archive_content_matches(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        content = '{"name":"juice-shop","version":"1.0.0"}'
        sb.write_file("package.json", content)
        args, _ = container.put_archive.call_args
        raw_tar = args[1]
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r") as tf:
            fobj = tf.extractfile(tf.getmembers()[0])
            assert fobj.read().decode("utf-8") == content

    def test_write_safe_for_special_characters(self, tmp_path):
        """Content with shell-special chars must round-trip without corruption."""
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        special = 'echo "hello $USER"; rm -rf /; it\'s a trap'
        sb.write_file("evil.sh", special)
        args, _ = container.put_archive.call_args
        raw_tar = args[1]
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r") as tf:
            fobj = tf.extractfile(tf.getmembers()[0])
            assert fobj.read().decode("utf-8") == special

    def test_write_strips_leading_slash(self, tmp_path):
        """file_path with a leading '/' must still map under /workspace."""
        docker_mod, client, container = _make_mock_docker()
        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("/package.json", "{}")
        args, _ = container.put_archive.call_args
        assert args[0] == "/workspace"

    def test_write_raises_when_dead(self, tmp_path):
        sb = DockerSandbox(tmp_path)  # never started
        with pytest.raises(RuntimeError, match="not running"):
            sb.write_file("package.json", "{}")

    def test_write_creates_parent_directory(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.exec_run.return_value = (0, (b"", b""))
        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("deep/nested/file.json", "{}")
        # exec_run should have been called with a mkdir -p command
        mkdir_calls = [
            str(c) for c in container.exec_run.call_args_list
            if "mkdir" in str(c)
        ]
        assert len(mkdir_calls) >= 1


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def _make_tar_bytes(filename: str, content: str) -> list[bytes]:
    """Return a list of byte chunks that simulate a Docker get_archive stream."""
    buf = io.BytesIO()
    encoded = content.encode("utf-8")
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo(name=filename)
        info.size = len(encoded)
        tf.addfile(info, io.BytesIO(encoded))
    return [buf.getvalue()]


class TestReadFile:
    def _started_sandbox(self, tmp_path, client, container):
        sb = DockerSandbox(tmp_path)
        sb._client = client
        sb._container = container
        sb._alive = True
        return sb

    def test_read_returns_file_content(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        content = '{"lockfileVersion":2}'
        container.get_archive.return_value = (
            _make_tar_bytes("package-lock.json", content),
            {"size": len(content)},
        )
        sb = self._started_sandbox(tmp_path, client, container)
        result = sb.read_file("package-lock.json")
        assert result == content

    def test_read_calls_get_archive_with_abs_path(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.get_archive.return_value = (
            _make_tar_bytes("package-lock.json", "{}"),
            {},
        )
        sb = self._started_sandbox(tmp_path, client, container)
        sb.read_file("package-lock.json")
        args, _ = container.get_archive.call_args
        assert args[0] == "/workspace/package-lock.json"

    def test_read_strips_leading_slash(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.get_archive.return_value = (
            _make_tar_bytes("package-lock.json", "{}"),
            {},
        )
        sb = self._started_sandbox(tmp_path, client, container)
        sb.read_file("/package-lock.json")
        args, _ = container.get_archive.call_args
        assert args[0] == "/workspace/package-lock.json"

    def test_read_returns_none_when_file_missing(self, tmp_path):
        docker_mod, client, container = _make_mock_docker()
        container.get_archive.side_effect = Exception("no such file or directory")
        sb = self._started_sandbox(tmp_path, client, container)
        result = sb.read_file("nonexistent.json")
        assert result is None

    def test_read_returns_none_when_dead(self, tmp_path):
        sb = DockerSandbox(tmp_path)  # never started
        result = sb.read_file("package-lock.json")
        assert result is None

    def test_read_handles_multipart_stream(self, tmp_path):
        """get_archive may yield chunks; all must be concatenated correctly."""
        docker_mod, client, container = _make_mock_docker()
        content = "multipart content"
        full_tar = _make_tar_bytes("file.txt", content)[0]
        # Split the tar bytes into multiple chunks to simulate streaming
        mid = len(full_tar) // 2
        container.get_archive.return_value = (
            [full_tar[:mid], full_tar[mid:]],
            {},
        )
        sb = self._started_sandbox(tmp_path, client, container)
        result = sb.read_file("file.txt")
        assert result == content

    def test_read_roundtrip_with_write(self, tmp_path):
        """write_file then read_file must return the original content."""
        docker_mod, client, container = _make_mock_docker()
        written_content = '{"name":"juice-shop","overrides":{"lodash":"4.17.21"}}'

        # Capture what write_file puts into put_archive, then feed it to
        # get_archive so we can verify the full round-trip in memory.
        captured_tar: list[bytes] = []

        def capture_put_archive(path, data):
            captured_tar.append(data)

        container.put_archive.side_effect = capture_put_archive

        def mock_get_archive(path):
            return ([captured_tar[0]], {"size": len(captured_tar[0])})

        container.get_archive.side_effect = mock_get_archive

        sb = self._started_sandbox(tmp_path, client, container)
        sb.write_file("package.json", written_content)
        result = sb.read_file("package.json")
        assert result == written_content
