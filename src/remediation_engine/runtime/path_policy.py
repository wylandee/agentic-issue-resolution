"""Repository-relative path validation for host and sandbox I/O.

All paths that originate in scanner output, model instructions, persisted
state, or tool arguments must pass through this module before they are used as
filesystem paths.  Keeping the policy in one place prevents individual
callers from making subtly different decisions about traversal and symlinks.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath


class WorkspacePathError(ValueError):
    """Raised when a path is not safely relative to the workspace."""


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[/\\]|[/\\]{2})")


def normalize_workspace_path(
    file_path: str | Path,
    *,
    allow_workspace_prefix: bool = True,
) -> str:
    """Return a normalized, traversal-free POSIX workspace-relative path.

    Args:
        file_path: User-, scanner-, or model-supplied path.
        allow_workspace_prefix: Whether ``/workspace/`` may be accepted as a
            container path and converted to a relative path.

    Returns:
        A non-empty POSIX path that contains no ``..`` component.

    Raises:
        WorkspacePathError: If the path is absolute, empty, or contains a
            traversal component.
    """
    raw = str(file_path).strip().replace("\\", "/")
    if not raw:
        raise WorkspacePathError("workspace path must not be empty")

    if allow_workspace_prefix:
        if raw == "/workspace":
            raise WorkspacePathError("workspace root is not a file path")
        if raw.startswith("/workspace/"):
            raw = raw[len("/workspace/") :]

    if raw.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(raw):
        raise WorkspacePathError(f"absolute workspace path is not allowed: {file_path!r}")

    parts = PurePosixPath(raw).parts
    if not parts or any(part == ".." for part in parts):
        raise WorkspacePathError(f"workspace traversal path is not allowed: {file_path!r}")

    normalized = PurePosixPath(raw).as_posix()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise WorkspacePathError(f"workspace path must identify a file: {file_path!r}")
    return normalized


def resolve_repository_path(repo_root: str | Path, file_path: str | Path) -> Path:
    """Resolve a repository-relative path while enforcing symlink containment.

    Args:
        repo_root: Host repository directory.
        file_path: Relative path inside that repository.

    Returns:
        The resolved host path.

    Raises:
        WorkspacePathError: If the path is not relative or resolves outside
            the repository, including through a symlink.
    """
    root = Path(repo_root).resolve()
    relative = normalize_workspace_path(file_path, allow_workspace_prefix=False)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspacePathError(f"path resolves outside repository: {file_path!r}") from exc
    return resolved


def repository_relative_path(value: str | Path | None, repo_root: str | Path) -> str | None:
    """Normalize a scanner path to a safe repository-relative POSIX path.

    Absolute paths are accepted only when they resolve inside ``repo_root``.
    Relative paths are resolved against the repository so symlink escapes are
    rejected as well.  Invalid or outside paths return ``None`` so callers can
    explicitly discard them at an ingestion boundary.

    Args:
        value: Candidate scanner or persisted path.
        repo_root: Host repository directory.

    Returns:
        A safe relative path, or ``None`` for an empty/outside path.
    """
    if value is None:
        return None
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None

    # A Windows drive-qualified path is absolute even when this service is
    # running on POSIX.  ``Path('C:/...')`` would otherwise be interpreted as
    # a harmless relative name on POSIX and could bypass the cross-platform
    # scanner-path policy.
    # A Windows absolute path is valid input when this service is running on
    # Windows; the containment check below still rejects paths outside the
    # repository. On POSIX, reject it because it cannot be resolved safely
    # against the host repository's native path semantics.
    if _WINDOWS_ABSOLUTE_PATH.match(raw) and os.name != "nt":
        return None

    if raw.startswith("/workspace/"):
        try:
            relative = normalize_workspace_path(raw)
            return (
                resolve_repository_path(repo_root, relative)
                .relative_to(Path(repo_root).resolve())
                .as_posix()
            )
        except (OSError, WorkspacePathError):
            return None

    root_input = Path(repo_root)
    root = root_input.resolve()

    # Preserve callers that provide both the repository and the scanner path
    # relative to the current working directory.
    try:
        relative = Path(raw).relative_to(root_input).as_posix()
        candidate = (
            root / normalize_workspace_path(relative, allow_workspace_prefix=False)
        ).resolve()
        return candidate.relative_to(root).as_posix()
    except (OSError, ValueError, WorkspacePathError):
        pass

    # Some legacy Pydantic path normalizers stripped the leading slash from
    # absolute scanner paths before they reached this boundary.  Recover that
    # representation only when its remaining text exactly matches this root.
    raw_without_leading_slash = raw.lstrip("/")
    root_without_leading_slash = root.as_posix().lstrip("/")
    if raw_without_leading_slash.casefold().startswith(root_without_leading_slash.casefold() + "/"):
        relative = raw_without_leading_slash[len(root_without_leading_slash) + 1 :]
        try:
            return (
                (root / normalize_workspace_path(relative, allow_workspace_prefix=False))
                .resolve()
                .relative_to(root)
                .as_posix()
            )
        except (OSError, ValueError, WorkspacePathError):
            return None

    try:
        if raw.startswith("/") or _WINDOWS_ABSOLUTE_PATH.match(raw):
            candidate = Path(raw)
        else:
            relative = normalize_workspace_path(raw, allow_workspace_prefix=False)
            candidate = root / relative
        return candidate.resolve().relative_to(root).as_posix()
    except (OSError, ValueError, WorkspacePathError):
        return None
