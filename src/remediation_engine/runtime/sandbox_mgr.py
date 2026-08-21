"""
sandbox_mgr.py â€” Ephemeral Docker sandbox for safe command execution.

Phase 5 of the AppSec Remediation Engine. Provides ``DockerSandbox``, a
context-manager that:

1. Starts an ephemeral container (default: ``node:22``).
2. Optionally mounts a Docker named volume at ``/workspace``.
3. Optionally copies a host repository into ``/workspace`` via ``put_archive``.
4. Executes arbitrary shell commands inside the container with ``/bin/sh -lc``,
   capturing stdout / stderr and respecting a configurable timeout.
5. Always tears down the container on exit, even on exceptions.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shlex
import tarfile
import time
import uuid
from pathlib import Path

from langsmith import traceable

from remediation_engine.contracts.schemas import CommandResult
from remediation_engine.runtime.docker_client import (
    close_docker_client,
    get_docker_client,
)
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
)

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "node:22"
_DEFAULT_TIMEOUT_SECONDS = 300
_SKIP_DIR_NAMES = frozenset({".git", "node_modules"})
_WORKSPACE_SNAPSHOT_DIR = ".remedy-attempt-snapshots"
_WORKSPACE_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS = 900
_WORKSPACE_SNAPSHOT_VALIDATION_TIMEOUT_SECONDS = 60


def _workspace_snapshot_archive(snapshot_id: str) -> str:
    """Return the validated archive path for a workspace snapshot."""
    if not _WORKSPACE_SNAPSHOT_ID_RE.fullmatch(snapshot_id):
        raise ValueError(
            "workspace snapshot IDs must contain only letters, numbers, '.', '_' or '-'."
        )
    return f"/workspace/{_WORKSPACE_SNAPSHOT_DIR}/{snapshot_id}/workspace.tar.gz"


def _make_tar_archive(repo_root: Path) -> bytes:
    """
    Create an in-memory tar archive of *repo_root* without dereferencing
    symlinks.

    ``.git`` and ``node_modules`` directories are skipped to avoid streaming
    large or host-specific data into the Docker daemon.
    """
    repo_root = repo_root.resolve()
    buf = io.BytesIO()

    with tarfile.open(fileobj=buf, mode="w", dereference=False) as tf:
        for dirpath, dirnames, filenames in os.walk(repo_root, topdown=True, followlinks=False):
            dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in _SKIP_DIR_NAMES)

            current_dir = Path(dirpath)
            rel_dir = current_dir.relative_to(repo_root)
            if rel_dir != Path("."):
                dir_info = tf.gettarinfo(
                    str(current_dir),
                    arcname=rel_dir.as_posix(),
                )
                tf.addfile(dir_info)

            for filename in sorted(filenames):
                abs_path = current_dir / filename
                rel_path = abs_path.relative_to(repo_root).as_posix()
                file_info = tf.gettarinfo(str(abs_path), arcname=rel_path)
                if file_info.isfile():
                    with abs_path.open("rb") as fh:
                        tf.addfile(file_info, fh)
                else:
                    tf.addfile(file_info)

    return buf.getvalue()


class DockerSandbox:
    """
    Ephemeral Docker sandbox for running commands inside ``/workspace``.

    Parameters
    ----------
    repo_root:
        Optional host repository path to tar-stream into the container.
        When ``None``, the sandbox starts with only the mounted workspace volume.
    image:
        Docker image to use. Defaults to ``node:22``.
    workspace_volume:
        Optional Docker named volume to mount at ``/workspace``.
    """

    def __init__(
        self,
        repo_root: str | Path | None,
        image: str = _DEFAULT_IMAGE,
        workspace_volume: str | None = None,
    ) -> None:
        """Initialize an unopened sandbox configuration."""
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._image = image
        self._workspace_volume = workspace_volume
        self._container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        self._container = None
        self._client = None
        self._alive = False

    @traceable(run_type="tool", name="docker_sandbox.start")
    def start(self) -> None:
        """Start the sandbox container and prepare ``/workspace``."""
        try:
            import docker.errors as docker_errors  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'docker' package is required for DockerSandbox. "
                "Install it with: pip install 'docker>=7.0.0'"
            ) from exc

        client = get_docker_client()
        self._client = client

        try:
            try:
                client.images.get(self._image)
                logger.debug("DockerSandbox: image %r found locally.", self._image)
            except docker_errors.ImageNotFound:
                logger.info("DockerSandbox: image %r not found locally - pulling.", self._image)
                client.images.pull(self._image)
                logger.info("DockerSandbox: pull complete for %r.", self._image)

            run_kwargs = {
                "name": self._container_name,
                "command": "sh -c 'while true; do sleep 3600; done'",
                "detach": True,
                "stdin_open": False,
                "tty": False,
                "network_mode": "bridge",
            }
            if self._workspace_volume:
                run_kwargs["volumes"] = {
                    self._workspace_volume: {"bind": "/workspace", "mode": "rw"}
                }

            logger.info(
                "DockerSandbox: starting container %r from image %r.",
                self._container_name,
                self._image,
            )
            self._container = client.containers.run(self._image, **run_kwargs)

            exit_code, output = self._container.exec_run("mkdir -p /workspace")
            if exit_code != 0:
                detail = (
                    output.decode("utf-8", errors="replace")
                    if isinstance(output, bytes)
                    else output
                )
                raise RuntimeError(
                    "DockerSandbox: failed to initialize /workspace "
                    f"(exit {exit_code}): {detail or 'no diagnostic output'}"
                )

            if self._repo_root is not None:
                logger.info("DockerSandbox: copying %s into container /workspace.", self._repo_root)
                archive = _make_tar_archive(self._repo_root)
                self._container.put_archive("/workspace", archive)
                logger.info("DockerSandbox: repository copied; sandbox ready.")
            else:
                logger.info("DockerSandbox: no repo_root provided; skipping archive copy.")

            self._alive = True
        except Exception:
            # Container creation is transactional: a failed image pull,
            # workspace preparation, or archive upload must not leak a
            # partially started container or its Docker client.
            self.teardown()
            raise

    @traceable(run_type="tool", name="docker_sandbox.teardown")
    def teardown(self) -> None:
        """
        Stop and remove the container. Idempotent and safe to call repeatedly.
        """
        container = self._container
        client = self._client
        try:
            if container is not None:
                try:
                    container.kill()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("DockerSandbox: container kill skipped: %s", exc)
                try:
                    container.remove(force=True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("DockerSandbox: container removal skipped: %s", exc)
                logger.info("DockerSandbox: container %r removed.", self._container_name)
        finally:
            self._container = None
            self._alive = False
            self._client = None
            try:
                close_docker_client(client)
            except Exception as exc:  # noqa: BLE001
                logger.warning("DockerSandbox: Docker client close failed: %s", exc)

    def __enter__(self) -> DockerSandbox:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()
        return None

    def restart(self) -> None:
        """Restart the container while preserving the named workspace volume.

        A timed-out command tears down its container but deliberately leaves
        the named volume intact. This method starts a fresh container against
        that volume without copying the host repository over the existing
        workspace. It is only valid for volume-backed sandboxes.

        Raises:
            RuntimeError: If no named workspace volume was configured.
        """
        if not self._workspace_volume:
            raise RuntimeError("Sandbox restart requires a named workspace volume.")
        self.teardown()
        original_repo_root = self._repo_root
        self._repo_root = None
        try:
            self.start()
        finally:
            self._repo_root = original_repo_root

    def _assert_container_path(self, absolute_path: str) -> None:
        """Reject a container path whose existing symlinks leave ``/workspace``."""
        if self._container is None:
            raise RuntimeError("Sandbox is not running.")
        quoted_path = shlex.quote(absolute_path)
        command = (
            f"resolved=$(realpath -m -- {quoted_path}) || exit 1; "
            'case "$resolved" in /workspace|/workspace/*) exit 0 ;; *) exit 1 ;; esac'
        )
        result = self._container.exec_run(["/bin/sh", "-lc", command])
        exit_code = result[0] if isinstance(result, tuple) else getattr(result, "exit_code", 1)
        if exit_code != 0:
            raise WorkspacePathError(f"container path resolves outside /workspace: {absolute_path}")

    @traceable(run_type="tool", name="docker_sandbox.run")
    def run(
        self,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CommandResult:
        """Execute *command* inside ``/workspace`` using ``/bin/sh -lc``."""
        if not self._alive or self._container is None:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr="Sandbox is not running.",
                duration_seconds=0.0,
            )

        logger.debug("DockerSandbox: running %r (timeout=%ds).", command, timeout)
        start = time.monotonic()

        import threading

        result_holder = []
        exception_holder = []

        def target():
            """Execute the container command on the worker thread."""
            try:
                result = self._container.exec_run(
                    ["/bin/sh", "-lc", command],
                    workdir="/workspace",
                    demux=True,
                    stderr=True,
                    stdout=True,
                    stream=False,
                    socket=False,
                )
                result_holder.append(result)
            except Exception as exc:  # noqa: BLE001
                exception_holder.append(exc)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        duration = time.monotonic() - start

        if thread.is_alive():
            logger.error(
                "DockerSandbox: command %r timed out after %ds â€” killing container.",
                command,
                timeout,
            )
            self.teardown()
            return CommandResult(
                exit_code=124,
                stdout="",
                stderr=f"Command timed out after {timeout}s.",
                duration_seconds=round(duration, 3),
            )

        if exception_holder:
            exc = exception_holder[0]
            err_msg = str(exc)
            if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                logger.error(
                    "DockerSandbox: command %r timed out after %ds â€” killing container.",
                    command,
                    timeout,
                )
                self.teardown()
                return CommandResult(
                    exit_code=124,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s.",
                    duration_seconds=round(duration, 3),
                )

            logger.error("DockerSandbox: unexpected error running %r â€” %s", command, exc)
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=err_msg,
                duration_seconds=round(duration, 3),
            )

        if not result_holder:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr="No execution result returned from sandbox.",
                duration_seconds=round(duration, 3),
            )

        exit_code, (raw_stdout, raw_stderr) = result_holder[0]
        stdout = (raw_stdout or b"").decode("utf-8", errors="replace")
        stderr = (raw_stderr or b"").decode("utf-8", errors="replace")

        if exit_code != 0:
            logger.warning("DockerSandbox: command %r exited %d.", command, exit_code)

        return CommandResult(
            exit_code=exit_code if exit_code is not None else 1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 3),
        )

    @traceable(run_type="tool", name="docker_sandbox.write_file")
    def write_file(self, file_path: str, content: str) -> None:
        """Write *content* to *file_path* inside ``/workspace``."""
        if not self._alive or self._container is None:
            raise RuntimeError("Sandbox is not running.")

        relative_path = normalize_workspace_path(file_path)
        abs_path = f"/workspace/{relative_path}"
        parent_dir = abs_path.rsplit("/", 1)[0]
        filename = abs_path.rsplit("/", 1)[1]

        self._assert_container_path(parent_dir)
        self._container.exec_run(f"mkdir -p -- {shlex.quote(parent_dir)}")

        encoded = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            tf.addfile(info, io.BytesIO(encoded))
        buf.seek(0)

        self._container.put_archive(parent_dir, buf.read())
        logger.debug("DockerSandbox: wrote %d bytes to %s.", len(encoded), abs_path)

    @traceable(run_type="tool", name="docker_sandbox.read_file")
    def read_file(self, file_path: str) -> str | None:
        """Read *file_path* from ``/workspace`` and return decoded text."""
        if not self._alive or self._container is None:
            logger.warning("DockerSandbox.read_file: sandbox is not running.")
            return None

        relative_path = normalize_workspace_path(file_path)
        abs_path = f"/workspace/{relative_path}"

        try:
            self._assert_container_path(abs_path)
            stream, _stat = self._container.get_archive(abs_path)
        except WorkspacePathError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("DockerSandbox.read_file: could not retrieve %s â€” %s", abs_path, exc)
            return None

        raw = b"".join(chunk for chunk in stream)

        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
                member = tf.getmembers()[0]
                extracted = tf.extractfile(member)
                if extracted is None:
                    logger.warning("DockerSandbox.read_file: %s is not a regular file.", abs_path)
                    return None
                return extracted.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DockerSandbox.read_file: failed to extract %s â€” %s", abs_path, exc)
            return None

    @traceable(run_type="tool", name="docker_sandbox.create_workspace_snapshot")
    def create_workspace_snapshot(self, snapshot_id: str) -> None:
        """Create an attempt-local archive of the mounted workspace.

        The archive lives inside the named Docker volume so it follows the
        workspace across the short-lived containers used by worker and QA
        nodes.  The snapshot directory is excluded from its own archive.

        Parameters
        ----------
        snapshot_id:
            Opaque, shell-safe identifier for the attempt or batch.

        Raises
        ------
        RuntimeError
            If the sandbox is not running or the archive command fails.
        ValueError
            If ``snapshot_id`` contains unsafe path characters.
        """
        archive_path = _workspace_snapshot_archive(snapshot_id)
        snapshot_dir = archive_path.rsplit("/", 1)[0]
        if not self._alive:
            raise RuntimeError("Sandbox is not running.")

        quoted_dir = shlex.quote(snapshot_dir)
        quoted_archive = shlex.quote(archive_path)
        command = (
            f"rm -rf -- {quoted_dir} && "
            f"mkdir -p -- {quoted_dir} && "
            f"tar -czf {quoted_archive} "
            f"--exclude=./{_WORKSPACE_SNAPSHOT_DIR} -C /workspace ."
        )
        result = self.run(command, timeout=_WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS)
        if result.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot creation failed for {snapshot_id}: "
                f"{result.stderr or result.stdout or 'unknown error'}"
            )
        validation = self.run(
            f"test -s {quoted_archive} && tar -tzf {quoted_archive} >/dev/null",
            timeout=_WORKSPACE_SNAPSHOT_VALIDATION_TIMEOUT_SECONDS,
        )
        if validation.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot creation produced an invalid archive for {snapshot_id}: "
                f"{validation.stderr or validation.stdout or 'archive is missing or unreadable'}"
            )
        logger.info("DockerSandbox: created workspace snapshot %s.", snapshot_id)

    @traceable(run_type="tool", name="docker_sandbox.restore_workspace_snapshot")
    def restore_workspace_snapshot(self, snapshot_id: str) -> None:
        """Restore the mounted workspace to a prior attempt snapshot.

        Restoration removes every current workspace entry except the private
        snapshot store, then extracts the archived candidate.  Dependencies
        and generated files are included in the archive, so the next worker
        sees the exact pre-attempt workspace rather than a stale install.

        Parameters
        ----------
        snapshot_id:
            Identifier previously passed to :meth:`create_workspace_snapshot`.

        Raises
        ------
        RuntimeError
            If the sandbox is not running or the archive cannot be restored.
        ValueError
            If ``snapshot_id`` contains unsafe path characters.
        """
        archive_path = _workspace_snapshot_archive(snapshot_id)
        if not self._alive:
            raise RuntimeError("Sandbox is not running.")

        quoted_archive = shlex.quote(archive_path)
        validation = self.run(
            f"test -s {quoted_archive} && tar -tzf {quoted_archive} >/dev/null",
            timeout=_WORKSPACE_SNAPSHOT_VALIDATION_TIMEOUT_SECONDS,
        )
        if validation.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot restore failed for {snapshot_id} before workspace cleanup: "
                f"{validation.stderr or validation.stdout or 'snapshot archive is missing or unreadable'}"
            )

        cleanup = self.run(
            "find /workspace -mindepth 1 -maxdepth 1 "
            f"! -name {shlex.quote(_WORKSPACE_SNAPSHOT_DIR)} "
            "-exec rm -rf -- {} +",
            timeout=_WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS,
        )
        if cleanup.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot restore failed for {snapshot_id} while clearing the workspace: "
                f"{cleanup.stderr or cleanup.stdout or 'unknown error'}"
            )

        extraction = self.run(
            f"tar -xzf {quoted_archive} -C /workspace",
            timeout=_WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS,
        )
        if extraction.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot restore failed for {snapshot_id} while extracting the archive: "
                f"{extraction.stderr or extraction.stdout or 'unknown error'}"
            )
        logger.info("DockerSandbox: restored workspace snapshot %s.", snapshot_id)

    @traceable(run_type="tool", name="docker_sandbox.remove_workspace_snapshot")
    def remove_workspace_snapshot(self, snapshot_id: str) -> None:
        """Delete one attempt snapshot from the mounted workspace.

        Missing snapshots are treated as success, making cleanup safe in
        worker error, QA error, and teardown ``finally`` paths.
        """
        archive_path = _workspace_snapshot_archive(snapshot_id)
        if not self._alive:
            raise RuntimeError("Sandbox is not running.")

        snapshot_dir = archive_path.rsplit("/", 1)[0]
        result = self.run(f"rm -rf -- {shlex.quote(snapshot_dir)}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"Workspace snapshot cleanup failed for {snapshot_id}: "
                f"{result.stderr or result.stdout or 'unknown error'}"
            )
        logger.debug("DockerSandbox: removed workspace snapshot %s.", snapshot_id)

    @traceable(run_type="tool", name="docker_sandbox.cleanup_workspace_snapshots")
    def cleanup_workspace_snapshots(self) -> None:
        """Remove all private attempt snapshots from the mounted workspace."""
        if not self._alive:
            raise RuntimeError("Sandbox is not running.")

        result = self.run(f"rm -rf -- {shlex.quote('/workspace/' + _WORKSPACE_SNAPSHOT_DIR)}")
        if result.exit_code != 0:
            raise RuntimeError(
                "Workspace snapshot directory cleanup failed: "
                f"{result.stderr or result.stdout or 'unknown error'}"
            )
        logger.debug("DockerSandbox: removed all workspace attempt snapshots.")
