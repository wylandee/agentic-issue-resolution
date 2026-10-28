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

import hashlib
import io
import logging
import os
import re
import shlex
import tarfile
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from threading import RLock

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
_PINNED_NODE_NPM_VERSION = "11.19.1"
_DEFAULT_TIMEOUT_SECONDS = 300
_SKIP_DIR_NAMES = frozenset({".git", "node_modules"})
_WORKSPACE_SNAPSHOT_DIR = ".remedy-attempt-snapshots"
_WORKSPACE_SNAPSHOT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS = 900
_WORKSPACE_SNAPSHOT_VALIDATION_TIMEOUT_SECONDS = 60
_READ_CACHE_MAX_ENTRIES = 512
_READ_CACHE_MAX_BYTES = 8 * 1024 * 1024


class WorkspaceReadCache:
    """Bounded, revision-aware cache for workspace file reads.

    The cache is deliberately separate from Docker I/O.  A cache hit returns
    before the traced container reader is entered, while the revision key
    prevents content from surviving an explicit workspace mutation or
    snapshot restore.  A cache instance may be shared by nested QA/workaround
    helpers or by short-lived sandbox wrappers for the same attempt.

    Args:
        max_entries: Maximum number of normalized file entries to retain.
        max_bytes: Maximum UTF-8 payload size retained by the cache.

    Raises:
        ValueError: If either bound is not positive.
    """

    def __init__(
        self,
        *,
        max_entries: int = _READ_CACHE_MAX_ENTRIES,
        max_bytes: int = _READ_CACHE_MAX_BYTES,
    ) -> None:
        """Initialize an empty cache with bounded entry and byte limits.

        Args:
            max_entries: Maximum number of file entries to retain.
            max_bytes: Maximum UTF-8 payload size to retain.
        """
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str, int, str], str | None] = OrderedDict()
        self._cached_bytes = 0
        self._epoch = 0
        self._requests = 0
        self._hits = 0
        self._misses = 0
        self._docker_reads = 0
        self._oversized_reads = 0
        self._invalidations = 0
        self._last_invalidation_reason = "initial"
        self._lock = RLock()

    @property
    def epoch(self) -> int:
        """Return the current workspace revision used for cache keys."""
        with self._lock:
            return self._epoch

    def invalidate(self, reason: str) -> None:
        """Discard all entries and advance the workspace revision.

        Args:
            reason: Bounded operational reason for diagnostics and metrics.
        """
        with self._lock:
            self._entries.clear()
            self._cached_bytes = 0
            self._epoch += 1
            self._invalidations += 1
            self._last_invalidation_reason = str(reason or "unknown")[:200]

    def get_or_load(
        self,
        scope: str,
        relative_path: str,
        loader: Callable[[], str | None],
    ) -> str | None:
        """Return a cached file or load it exactly once for the current epoch.

        Args:
            scope: Stable workspace/volume scope for the cache key.
            relative_path: Already normalized workspace-relative path.
            loader: Callable that performs one real Docker read on a miss.

        Returns:
            The file contents, or ``None`` when the loader cannot read it.
        """
        with self._lock:
            self._requests += 1
            key = (str(scope), self._epoch, relative_path)
            if key in self._entries:
                self._hits += 1
                self._entries.move_to_end(key)
                return self._entries[key]

            self._misses += 1
            # Keep the load under the lock so concurrent tool calls for the
            # same file cannot create a burst of duplicate Docker reads.
            self._docker_reads += 1
            content = loader()
            content_size = len(content.encode("utf-8")) if isinstance(content, str) else 0
            if content_size <= self._max_bytes:
                while self._entries and (
                    len(self._entries) >= self._max_entries
                    or self._cached_bytes + content_size > self._max_bytes
                ):
                    _old_key, old_value = self._entries.popitem(last=False)
                    if isinstance(old_value, str):
                        self._cached_bytes -= len(old_value.encode("utf-8"))
                self._entries[key] = content
                self._cached_bytes += content_size
            else:
                self._oversized_reads += 1
            return content

    def metrics(self) -> dict[str, int | str]:
        """Return bounded read-cache counters without exposing file contents."""
        with self._lock:
            return {
                "epoch": self._epoch,
                "requests": self._requests,
                "hits": self._hits,
                "misses": self._misses,
                "docker_reads": self._docker_reads,
                "cached_entries": len(self._entries),
                "cached_bytes": self._cached_bytes,
                "oversized_reads": self._oversized_reads,
                "invalidations": self._invalidations,
                "last_invalidation_reason": self._last_invalidation_reason,
            }


_DEFAULT_READ_FILES_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_READ_FILES = 10_000
_MAX_CONTAINER_PATH_CHECK_COMMAND_BYTES = 32_768


class _ReadFilesResult(dict[str, str]):
    """Document mapping with private bounded data for the trace callback."""

    __slots__ = ("total_bytes", "duration_seconds", "digest")

    def __init__(self) -> None:
        super().__init__()
        self.total_bytes = 0
        self.duration_seconds = 0.0
        self.digest = ""


def _trace_read_files_inputs(inputs: dict[str, object]) -> dict[str, int]:
    """Keep batched read inputs free of workspace paths."""
    paths = inputs.get("paths", ())
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        return {"file_count": 0}
    return {"file_count": min(len(paths), _MAX_READ_FILES + 1)}


def _trace_read_files_outputs(output: object) -> dict[str, object]:
    """Emit bounded read measurements without document contents or paths."""
    if not isinstance(output, _ReadFilesResult):
        return {"file_count": 0, "total_bytes": 0, "duration_seconds": 0.0, "digest": ""}
    return {
        "file_count": len(output),
        "total_bytes": output.total_bytes,
        "duration_seconds": round(output.duration_seconds, 3),
        "digest": output.digest,
    }


def _trace_write_files_inputs(inputs: dict[str, object]) -> dict[str, int]:
    """Keep batched write traces free of workspace paths and file contents."""
    files = inputs.get("files")
    if not isinstance(files, Mapping):
        return {"file_count": 0}
    return {"file_count": len(files)}


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
        read_cache: WorkspaceReadCache | None = None,
    ) -> None:
        """Initialize an unopened sandbox configuration.

        Args:
            repo_root: Optional host repository to copy into the workspace.
            image: Docker image used for the sandbox container.
            workspace_volume: Optional named Docker volume mounted at
                ``/workspace``.
            read_cache: Optional attempt-scoped cache shared by nested
                workspace readers. A private cache is created when omitted.
        """
        self._repo_root = Path(repo_root).resolve() if repo_root is not None else None
        self._image = image
        self._workspace_volume = workspace_volume
        self._container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        self._container = None
        self._client = None
        self._alive = False
        # Read-heavy QA and workaround flows often request the same file from
        # several tools in one turn. Cache hits are handled before the traced
        # I/O helper so they do not create one LangSmith span per duplicate
        # read. The scope is part of the key when a cache is shared across
        # short-lived wrappers for one remediation attempt.
        self._read_cache = read_cache or WorkspaceReadCache()
        self._read_scope = (
            f"volume:{workspace_volume}"
            if workspace_volume
            else f"repo:{self._repo_root or '<unbound>'}"
        )

    @traceable(run_type="tool", name="docker_sandbox.start")
    def start(self) -> None:
        """Start the sandbox container and prepare ``/workspace``.

        Node images have npm pinned before workspace setup so nested install
        scripts use the same resolver version as top-level npm commands. This
        may contact the npm registry when the image has a different npm version.
        """
        self.invalidate_read_cache("sandbox_start")
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

            if self._image.startswith("node:"):
                # npm 10's Arborist crashes while resolving some optional peer
                # sets (including the Vitest graph); pin npm for both the
                # initial install and any nested postinstall npm commands.
                npm_pin_command = (
                    f'if [ "$(npm --version 2>/dev/null)" '
                    f'!= "{_PINNED_NODE_NPM_VERSION}" ]; then '
                    f"npm install --global npm@{_PINNED_NODE_NPM_VERSION}; fi && "
                    f'test "$(npm --version)" = "{_PINNED_NODE_NPM_VERSION}"'
                )
                exit_code, output = self._container.exec_run(["/bin/sh", "-lc", npm_pin_command])
                if exit_code != 0:
                    detail = (
                        output.decode("utf-8", errors="replace")
                        if isinstance(output, bytes)
                        else str(output or "")
                    )
                    raise RuntimeError(
                        "DockerSandbox: failed to pin npm to "
                        f"{_PINNED_NODE_NPM_VERSION} (exit {exit_code}): "
                        f"{detail or 'no diagnostic output'}"
                    )

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
                copied = self._container.put_archive("/workspace", archive)
                if not copied:
                    raise RuntimeError(
                        "DockerSandbox: Docker did not confirm copying the repository "
                        "archive into /workspace."
                    )
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
            self.invalidate_read_cache("sandbox_teardown")
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
        self._assert_container_paths((absolute_path,))

    def _assert_container_paths(self, absolute_paths: Sequence[str]) -> None:
        """Reject paths whose existing symlinks leave ``/workspace``."""
        if self._container is None:
            raise RuntimeError("Sandbox is not running.")

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_size = 0
        for absolute_path in absolute_paths:
            quoted_path = shlex.quote(absolute_path)
            path_size = len(quoted_path.encode("utf-8")) + 1
            if current_batch and current_size + path_size > _MAX_CONTAINER_PATH_CHECK_COMMAND_BYTES:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append(quoted_path)
            current_size += path_size
        if current_batch:
            batches.append(current_batch)

        for batch in batches:
            quoted_paths = " ".join(batch)
            command = (
                f"for path in {quoted_paths}; do "
                'resolved=$(realpath -m -- "$path") || exit 1; '
                'case "$resolved" in /workspace|/workspace/*) : ;; *) exit 1 ;; esac; '
                "done"
            )
            result = self._container.exec_run(["/bin/sh", "-lc", command])
            exit_code = result[0] if isinstance(result, tuple) else getattr(result, "exit_code", 1)
            if exit_code != 0:
                raise WorkspacePathError("one or more container paths resolve outside /workspace")

    @traceable(run_type="tool", name="docker_sandbox.run")
    def run(
        self,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
        *,
        mutates_workspace: bool = True,
    ) -> CommandResult:
        """Execute *command* inside ``/workspace`` using ``/bin/sh -lc``.

        Args:
            command: Shell command to execute inside the workspace.
            timeout: Maximum execution time in seconds.
            mutates_workspace: Whether the caller knows the command can
                change workspace contents. Mutating commands advance the
                read-cache epoch; read-only commands retain cached evidence.

        Returns:
            The typed command result.
        """
        if mutates_workspace:
            self.invalidate_read_cache("sandbox_command")
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

    def run_readonly(
        self,
        command: str,
        timeout: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> CommandResult:
        """Execute a declared read-only command without evicting file evidence.

        Args:
            command: Shell command that does not modify workspace inputs.
            timeout: Maximum execution time in seconds.

        Returns:
            The typed command result.

        Note:
            Callers must use :meth:`run` for commands whose mutation behavior
            is unknown. This method is an explicit opt-in safety boundary.
        """
        return self.run(command, timeout=timeout, mutates_workspace=False)

    def invalidate_read_cache(self, reason: str = "workspace_mutation") -> None:
        """Invalidate cached workspace reads after an external mutation.

        Args:
            reason: Bounded operational reason retained in cache metrics.
        """
        self._read_cache.invalidate(reason)

    def read_cache_metrics(self) -> dict[str, int | str]:
        """Return read-cache counters for diagnostics and trace summaries."""
        return self._read_cache.metrics()

    @traceable(run_type="tool", name="docker_sandbox.write_file")
    def write_file(self, file_path: str, content: str) -> None:
        """Write *content* to *file_path* inside ``/workspace``."""
        if not self._alive or self._container is None:
            raise RuntimeError("Sandbox is not running.")

        self.invalidate_read_cache("write_file")
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

    @traceable(
        run_type="tool",
        name="docker_sandbox.write_files",
        process_inputs=_trace_write_files_inputs,
    )
    def write_files(self, files: Mapping[str, str]) -> None:
        """Write multiple workspace files with one Docker archive upload.

        Args:
            files: Mapping of workspace-relative paths to UTF-8 text contents.

        Raises:
            RuntimeError: If the sandbox is not running.
            TypeError: If any file content is not text.
            WorkspacePathError: If paths are unsafe, collide after normalization,
                or escape the workspace through parent symlinks.
        """
        if not self._alive or self._container is None:
            raise RuntimeError("Sandbox is not running.")
        normalized_files: dict[str, str] = {}
        parent_dirs: set[str] = set()
        for file_path, content in files.items():
            relative_path = normalize_workspace_path(file_path)
            if not isinstance(content, str):
                raise TypeError("workspace file contents must be text.")
            if relative_path in normalized_files:
                raise WorkspacePathError(f"duplicate normalized workspace path: {relative_path!r}")
            normalized_files[relative_path] = content
            parent = PurePosixPath(relative_path).parent
            while parent != PurePosixPath("."):
                parent_dirs.add(parent.as_posix())
                parent = parent.parent

        if not normalized_files:
            return

        conflicting_paths = set(normalized_files).intersection(parent_dirs)
        if conflicting_paths:
            raise WorkspacePathError(
                f"workspace path is both a file and a directory: {min(conflicting_paths)!r}"
            )

        ordered_directories = sorted(parent_dirs, key=lambda path: (path.count("/"), path))
        parent_paths = tuple(f"/workspace/{path}" for path in ordered_directories) or (
            "/workspace",
        )
        self._assert_container_paths(parent_paths)

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for directory in ordered_directories:
                directory_info = tarfile.TarInfo(name=f"{directory}/")
                directory_info.type = tarfile.DIRTYPE
                directory_info.mode = 0o755
                directory_info.mtime = 0
                archive.addfile(directory_info)
            for relative_path, content in normalized_files.items():
                encoded = content.encode("utf-8")
                file_info = tarfile.TarInfo(name=relative_path)
                file_info.size = len(encoded)
                archive.addfile(file_info, io.BytesIO(encoded))

        self.invalidate_read_cache("write_files")
        self._container.put_archive("/workspace", buffer.getvalue())
        logger.debug(
            "DockerSandbox: wrote %d files in one archive.",
            len(normalized_files),
        )

    def _read_file_bytes_untraced(
        self,
        relative_path: str,
        *,
        containment_checked: bool = False,
        regular_only: bool = False,
    ) -> bytes | None:
        """Read one normalized file without creating a trace span."""
        if not self._alive or self._container is None:
            logger.warning("DockerSandbox.read_file: sandbox is not running.")
            return None
        abs_path = f"/workspace/{relative_path}"
        try:
            if not containment_checked:
                self._assert_container_path(abs_path)
            stream, _stat = self._container.get_archive(abs_path)
        except WorkspacePathError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("DockerSandbox.read_file: could not retrieve %s — %s", abs_path, exc)
            return None
        raw = b"".join(chunk for chunk in stream)
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r") as tf:
                members = tf.getmembers()
                member = members[0] if members else None
                if member is None or (regular_only and not member.isfile()):
                    logger.warning("DockerSandbox.read_file: %s is not a regular file.", abs_path)
                    return None
                extracted = tf.extractfile(member)
                if extracted is None:
                    logger.warning("DockerSandbox.read_file: %s is not a regular file.", abs_path)
                    return None
                return extracted.read()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DockerSandbox.read_file: failed to extract %s — %s", abs_path, exc)
            return None

    def read_file(self, file_path: str) -> str | None:
        """Read one workspace file through the bounded, revision-aware cache."""
        relative_path = normalize_workspace_path(file_path)
        return self._read_cache.get_or_load(
            self._read_scope,
            relative_path,
            lambda: self._read_file_from_container(relative_path),
        )

    @traceable(run_type="tool", name="docker_sandbox.read_file")
    def _read_file_from_container(self, relative_path: str) -> str | None:
        """Read one normalized workspace path after a cache miss."""
        content = self._read_file_bytes_untraced(relative_path)
        return content.decode("utf-8", errors="replace") if content is not None else None

    @traceable(
        run_type="tool",
        name="docker_sandbox.read_files",
        process_inputs=_trace_read_files_inputs,
        process_outputs=_trace_read_files_outputs,
    )
    def read_files(
        self,
        paths: Sequence[str],
        *,
        max_total_bytes: int = _DEFAULT_READ_FILES_MAX_TOTAL_BYTES,
    ) -> dict[str, str] | None:
        """Read a bounded set of regular workspace files in one trace span.

        Args:
            paths: Workspace-relative paths to read. Duplicate normalized paths
                are read once.
            max_total_bytes: Total raw file-byte cap, never a truncation limit.
                Tests may lower the cap but cannot raise the hard limit.

        Returns:
            A complete path-to-text mapping, or ``None`` if any path is missing
            or is not a regular file. No partial mapping is returned.

        Raises:
            WorkspacePathError: If a path is unsafe or resolves outside the workspace.
            ValueError: If the path count or byte cap is invalid or exceeded.
        """
        started = time.monotonic()
        if not self._alive or self._container is None:
            logger.warning("DockerSandbox.read_files: sandbox is not running.")
            return None
        if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
            raise ValueError("paths must be a sequence of workspace-relative paths.")
        if (
            isinstance(max_total_bytes, bool)
            or not isinstance(max_total_bytes, int)
            or max_total_bytes < 0
            or max_total_bytes > _DEFAULT_READ_FILES_MAX_TOTAL_BYTES
        ):
            raise ValueError("max_total_bytes must be within the hard batch-read limit.")
        if len(paths) > _MAX_READ_FILES:
            raise ValueError(f"batched workspace read exceeds {_MAX_READ_FILES} files.")
        if any(len(str(path)) > 4096 for path in paths):
            raise ValueError("batched workspace read contains an overlong path.")
        try:
            normalized_paths = sorted({normalize_workspace_path(path) for path in paths})
        except WorkspacePathError:
            raise WorkspacePathError("batched workspace read contains an unsafe path") from None
        if len(normalized_paths) > _MAX_READ_FILES:
            raise ValueError(f"batched workspace read exceeds {_MAX_READ_FILES} files.")
        try:
            for relative_path in normalized_paths:
                self._assert_container_path(f"/workspace/{relative_path}")
        except WorkspacePathError:
            raise WorkspacePathError(
                "batched workspace read path resolves outside the workspace"
            ) from None

        documents = _ReadFilesResult()
        digest = hashlib.sha256()
        total_bytes = 0
        for relative_path in normalized_paths:
            content_bytes = self._read_file_bytes_untraced(
                relative_path, containment_checked=True, regular_only=True
            )
            if content_bytes is None:
                return None
            total_bytes += len(content_bytes)
            if total_bytes > max_total_bytes:
                raise ValueError("batched workspace read exceeds its total-byte limit.")
            documents[relative_path] = content_bytes.decode("utf-8", errors="replace")
            digest.update(relative_path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(content_bytes)
            digest.update(b"\0")
        documents.total_bytes = total_bytes
        documents.duration_seconds = max(0.0, time.monotonic() - started)
        documents.digest = digest.hexdigest()[:24]
        return documents

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
