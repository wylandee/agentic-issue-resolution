"""
sandbox_mgr.py — Ephemeral Docker sandbox for safe command execution.

Phase 3 of the AppSec Remediation Engine.  Provides ``DockerSandbox``, a
context-manager that:

1. Starts an ephemeral container (default: ``node:20-alpine``).
2. Copies the target repository into ``/workspace`` via a tar archive, with
   no bind-mounts or privileged flags.
3. Executes arbitrary shell commands inside the container with ``/bin/sh -lc``,
   capturing stdout / stderr and respecting a configurable timeout.
4. Always tears down the container on exit, even on exceptions.

Safety design
-------------
* No bind-mounts of the host repo — the repo is tar-streamed into the
  container via ``put_archive``, preventing the container from writing back.
* No Docker socket passed into the container.
* No privileged mode.
* Symlinks are not dereferenced when creating the tar archive.
* Network is left enabled by default so ``npm install`` can reach registries.

Usage
-----
::

    from src.runtime.sandbox_mgr import DockerSandbox

    with DockerSandbox("/path/to/repo") as sb:
        result = sb.run("npm install --prefer-offline", timeout=120)
        print(result.exit_code, result.stdout)
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

from src.contracts.schemas import CommandResult

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "node:20-alpine"
_DEFAULT_TIMEOUT_SECONDS = 300


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_tar_archive(repo_root: Path) -> bytes:
    """
    Create an in-memory tar archive of *repo_root* without dereferencing
    symlinks.  The archive root corresponds to the repo root so that
    ``put_archive`` places files at ``/workspace/<file>``.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", dereference=False) as tf:
        tf.add(str(repo_root), arcname=".", recursive=True)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class DockerSandbox:
    """
    Ephemeral Docker sandbox for running package-manager commands safely.

    Parameters
    ----------
    repo_root:
        Absolute path to the repository whose contents will be copied into
        the container's ``/workspace`` directory.
    image:
        Docker image to use.  Defaults to ``node:20-alpine``.

    Context manager
    ---------------
    ``__enter__`` calls ``start()``; ``__exit__`` always calls ``teardown()``.
    """

    def __init__(
        self,
        repo_root: str | Path,
        image: str = _DEFAULT_IMAGE,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._image = image
        self._container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        self._container = None
        self._client = None
        self._alive = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Connect to the Docker daemon, pull the image if missing, start the
        container, and copy the repository into ``/workspace``.

        Raises
        ------
        RuntimeError
            If the Docker daemon is unreachable.
        """
        try:
            import docker  # type: ignore[import]
            import docker.errors  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'docker' package is required for DockerSandbox.  "
                "Install it with: pip install 'docker>=7.0.0'"
            ) from exc

        logger.info("DockerSandbox: connecting to Docker daemon.")
        try:
            client = docker.from_env()
            client.ping()
        except Exception as exc:
            logger.error("DockerSandbox: Docker daemon unreachable — %s", exc)
            raise RuntimeError(f"Docker daemon unreachable: {exc}") from exc

        self._client = client

        # Pull image only if not already present locally.
        # Import errors from the same module object so that mocked versions
        # resolve correctly without the mock needing to subclass BaseException.
        _image_not_found_exc = docker.errors.ImageNotFound
        try:
            client.images.get(self._image)
            logger.debug("DockerSandbox: image %r found locally.", self._image)
        except _image_not_found_exc:
            logger.info(
                "DockerSandbox: image %r not found locally — pulling.", self._image
            )
            client.images.pull(self._image)
            logger.info("DockerSandbox: pull complete for %r.", self._image)

        # Start a detached long-running container — no bind mounts, no
        # privileged mode, no host workspace access.
        logger.info(
            "DockerSandbox: starting container %r from image %r.",
            self._container_name,
            self._image,
        )
        self._container = client.containers.run(
            self._image,
            name=self._container_name,
            command="sh -c 'while true; do sleep 3600; done'",
            detach=True,
            stdin_open=False,
            tty=False,
            network_mode="bridge",  # network enabled; no host access
        )

        # Create /workspace inside the container
        exit_code, _ = self._container.exec_run("mkdir -p /workspace")
        if exit_code != 0:
            logger.warning(
                "DockerSandbox: non-zero exit creating /workspace (%d).", exit_code
            )

        # Copy the repo into /workspace via a tar stream — no bind mount
        logger.info(
            "DockerSandbox: copying %s into container /workspace.", self._repo_root
        )
        archive = _make_tar_archive(self._repo_root)
        self._container.put_archive("/workspace", archive)
        logger.info("DockerSandbox: repository copied; sandbox ready.")
        self._alive = True

    def teardown(self) -> None:
        """
        Stop and remove the container.  Idempotent — safe to call multiple
        times or when the container was never started.
        """
        if self._container is None:
            return

        try:
            import docker.errors  # type: ignore[import]
        except ImportError:
            return

        try:
            import docker.errors as _de  # type: ignore[import]
            try:
                self._container.kill()
            except (_de.APIError, _de.NotFound):
                pass
            try:
                self._container.remove(force=True)
            except (_de.APIError, _de.NotFound):
                pass
            logger.info(
                "DockerSandbox: container %r removed.", self._container_name
            )
        except Exception as exc:
            logger.warning(
                "DockerSandbox: teardown error for %r — %s",
                self._container_name,
                exc,
            )
        finally:
            self._container = None
            self._alive = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.teardown()
        # Do not suppress exceptions
        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CommandResult:
        """
        Execute *command* inside ``/workspace`` using ``/bin/sh -lc``.

        Parameters
        ----------
        command:
            Shell command string to run.
        timeout:
            Maximum wall-clock seconds to wait.  On timeout, the container is
            killed, ``exit_code=124`` is returned, and the sandbox is marked as
            no longer reusable.

        Returns
        -------
        CommandResult
            Always returns a ``CommandResult`` — never raises on command
            failure or timeout.
        """
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
            try:
                res = self._container.exec_run(
                    ["/bin/sh", "-lc", command],
                    workdir="/workspace",
                    demux=True,
                    stderr=True,
                    stdout=True,
                    stream=False,
                    socket=False,
                )
                result_holder.append(res)
            except Exception as e:
                exception_holder.append(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()

        thread.join(timeout=timeout)
        duration = time.monotonic() - start

        if thread.is_alive():
            logger.error(
                "DockerSandbox: command %r timed out after %ds — killing container.",
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

            # Distinguish timeout from unexpected errors
            if "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                logger.error(
                    "DockerSandbox: command %r timed out after %ds — killing container.",
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

            logger.error(
                "DockerSandbox: unexpected error running %r — %s", command, exc
            )
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
            logger.warning(
                "DockerSandbox: command %r exited %d.", command, exit_code
            )

        return CommandResult(
            exit_code=exit_code if exit_code is not None else 1,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=round(duration, 3),
        )

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def write_file(self, file_path: str, content: str) -> None:
        """
        Write *content* to *file_path* inside the container workspace.

        *file_path* is interpreted relative to ``/workspace``.  Parent
        directories are created automatically.

        The content is encoded as UTF-8 and transferred via an in-memory tar
        archive using ``put_archive``, which is safe for all content including
        strings containing shell special characters or newlines.

        Parameters
        ----------
        file_path:
            Path relative to ``/workspace``, e.g. ``"package.json"`` or
            ``"config/overrides.json"``.
        content:
            String content to write.

        Raises
        ------
        RuntimeError
            If the sandbox is not running.
        """
        if not self._alive or self._container is None:
            raise RuntimeError("Sandbox is not running.")

        abs_path = f"/workspace/{file_path.lstrip('/')}"
        parent_dir = abs_path.rsplit("/", 1)[0]
        filename = abs_path.rsplit("/", 1)[1]

        # Ensure the parent directory exists inside the container
        self._container.exec_run(f"mkdir -p {parent_dir}")

        # Build an in-memory tar containing just the one file
        encoded = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            info = tarfile.TarInfo(name=filename)
            info.size = len(encoded)
            tf.addfile(info, io.BytesIO(encoded))
        buf.seek(0)

        self._container.put_archive(parent_dir, buf.read())
        logger.debug("DockerSandbox: wrote %d bytes to %s.", len(encoded), abs_path)

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read a file from the container workspace and return it as a string.

        *file_path* is interpreted relative to ``/workspace``.  The file is
        transferred via ``get_archive`` (a tar stream), extracted in-memory,
        decoded as UTF-8, and returned.

        Parameters
        ----------
        file_path:
            Path relative to ``/workspace``, e.g. ``"package-lock.json"``.

        Returns
        -------
        str
            Decoded file contents.
        None
            If the file does not exist or the sandbox is not running.
        """
        if not self._alive or self._container is None:
            logger.warning("DockerSandbox.read_file: sandbox is not running.")
            return None

        abs_path = f"/workspace/{file_path.lstrip('/')}"

        try:
            stream, _stat = self._container.get_archive(abs_path)
        except Exception as exc:
            logger.warning(
                "DockerSandbox.read_file: could not retrieve %s — %s", abs_path, exc
            )
            return None

        # Reassemble the streamed chunks into a single bytes buffer
        raw = b"".join(chunk for chunk in stream)

        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
                # The archive contains exactly one member: the requested file
                member = tf.getmembers()[0]
                fobj = tf.extractfile(member)
                if fobj is None:
                    logger.warning(
                        "DockerSandbox.read_file: %s is not a regular file.", abs_path
                    )
                    return None
                return fobj.read().decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning(
                "DockerSandbox.read_file: failed to extract %s — %s", abs_path, exc
            )
            return None
