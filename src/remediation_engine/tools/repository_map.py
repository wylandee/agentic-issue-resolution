"""Deterministic repository-map construction for specialist prompts."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPOSITORY_MAP_MAX_ENTRIES = 400
_EXCLUDED_DIRECTORY_NAMES = frozenset({".git", "node_modules"})


def build_repository_map(
    repo_root: str | Path,
    max_entries: int = DEFAULT_REPOSITORY_MAP_MAX_ENTRIES,
) -> str:
    """Return a deterministic, bounded list of repository entries.

    Args:
        repo_root: Host repository directory to inspect.
        max_entries: Maximum number of relative entries to include before
            appending a truncation marker.

    Returns:
        A sorted newline-delimited list of repository-relative POSIX paths,
        ``"(workspace is empty)"`` for an empty repository, or a bounded
        ``ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE`` message when the root
        cannot be inspected.

    Raises:
        ValueError: If ``max_entries`` is not positive.
    """
    if max_entries <= 0:
        raise ValueError("max_entries must be positive")

    try:
        root = Path(repo_root).resolve()
        if not root.is_dir():
            return "ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: repository root is not a directory"
    except (OSError, RuntimeError) as exc:
        return f"ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: {str(exc)[:500]}"

    entries: set[str] = set()
    errors: list[str] = []

    def on_walk_error(error: OSError) -> None:
        """Collect a bounded filesystem error without aborting the walk."""
        errors.append(str(error))

    try:
        for current_dir, dirnames, filenames in os.walk(
            root,
            topdown=True,
            onerror=on_walk_error,
            followlinks=False,
        ):
            current_path = Path(current_dir)
            dirnames[:] = sorted(
                dirname
                for dirname in dirnames
                if dirname not in _EXCLUDED_DIRECTORY_NAMES and not dirname.endswith(".map")
            )

            relative_dir = current_path.relative_to(root)
            if relative_dir != Path("."):
                entries.add(relative_dir.as_posix())

            for filename in filenames:
                if filename.endswith(".map"):
                    continue
                relative_file = (current_path / filename).relative_to(root)
                entries.add(relative_file.as_posix())
    except OSError as exc:
        errors.append(str(exc))

    if errors:
        detail = "; ".join(errors)[:500]
        return f"ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE: {detail}"

    ordered_entries = sorted(entries)
    if not ordered_entries:
        return "(workspace is empty)"

    capped = ordered_entries[:max_entries]
    output = "\n".join(capped)
    if len(ordered_entries) > max_entries:
        output += f"\n... (truncated, {len(ordered_entries) - max_entries} more entries)"
    return output
