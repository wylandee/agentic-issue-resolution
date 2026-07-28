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
import tarfile
import time
import uuid
from pathlib import Path
from typing import Optional

from langsmith import traceable

from remediation_engine.contracts.schemas import CommandResult

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "node:22"
_DEFAULT_TIMEOUT_SECONDS = 300
_SKIP_DIR_NAMES = frozenset({".git", "node_modules"})


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
            dirnames[:] = sorted(
                dirname for dirname in dirnames if dirname not in _SKIP_DIR_NAMES
            )

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


def get_docker_client():
    """Return a connected Docker SDK client or raise a clear error."""
    try:
        import docker  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'docker' package is required for DockerSandbox. "
            "Install it with: pip install 'docker>=7.0.0'"
        ) from exc

    logger.info("DockerSandbox: connecting to Docker daemon.")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        logger.error("DockerSandbox: Docker daemon unreachable â€” %s", exc)
        raise RuntimeError(f"Docker daemon unreachable: {exc}") from exc

    return client


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
        workspace_volume: Optional[str] = None,
    ) -> None:
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
            client.images.get(self._image)
            logger.debug("DockerSandbox: image %r found locally.", self._image)
        except docker_errors.ImageNotFound:
            logger.info(
                "DockerSandbox: image %r not found locally â€” pulling.", self._image
            )
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

        exit_code, _ = self._container.exec_run("mkdir -p /workspace")
        if exit_code != 0:
            logger.warning(
                "DockerSandbox: non-zero exit creating /workspace (%d).", exit_code
            )

        if self._repo_root is not None:
            logger.info(
                "DockerSandbox: copying %s into container /workspace.", self._repo_root
            )
            archive = _make_tar_archive(self._repo_root)
            self._container.put_archive("/workspace", archive)
            logger.info("DockerSandbox: repository copied; sandbox ready.")
        else:
            logger.info("DockerSandbox: no repo_root provided; skipping archive copy.")

        self._alive = True

    @traceable(run_type="tool", name="docker_sandbox.teardown")
    def teardown(self) -> None:
        """
        Stop and remove the container. Idempotent and safe to call repeatedly.
        """
        if self._container is None:
            return

        try:
            import docker.errors as docker_errors  # type: ignore[import]
        except ImportError:  # pragma: no cover
            return

        try:
            try:
                self._container.kill()
            except (docker_errors.APIError, docker_errors.NotFound):
                pass
            try:
                self._container.remove(force=True)
            except (docker_errors.APIError, docker_errors.NotFound):
                pass
            logger.info("DockerSandbox: container %r removed.", self._container_name)
        except Exception as exc:
            logger.warning(
                "DockerSandbox: teardown error for %r â€” %s",
                self._container_name,
                exc,
            )
        finally:
            self._container = None
            self._alive = False

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()
        return None

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

        abs_path = f"/workspace/{file_path.lstrip('/')}"
        parent_dir = abs_path.rsplit("/", 1)[0]
        filename = abs_path.rsplit("/", 1)[1]

        self._container.exec_run(f"mkdir -p {parent_dir}")

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
    def read_file(self, file_path: str) -> Optional[str]:
        """Read *file_path* from ``/workspace`` and return decoded text."""
        if not self._alive or self._container is None:
            logger.warning("DockerSandbox.read_file: sandbox is not running.")
            return None

        abs_path = f"/workspace/{file_path.lstrip('/')}"

        try:
            stream, _stat = self._container.get_archive(abs_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DockerSandbox.read_file: could not retrieve %s â€” %s", abs_path, exc
            )
            return None

        raw = b"".join(chunk for chunk in stream)

        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
                member = tf.getmembers()[0]
                extracted = tf.extractfile(member)
                if extracted is None:
                    logger.warning(
                        "DockerSandbox.read_file: %s is not a regular file.", abs_path
                    )
                    return None
                return extracted.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DockerSandbox.read_file: failed to extract %s â€” %s", abs_path, exc
            )
            return None


