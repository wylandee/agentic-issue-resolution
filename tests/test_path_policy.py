"""Regression tests for shared repository and sandbox path containment."""

from __future__ import annotations

from pathlib import Path

import pytest

from remediation_engine.orchestration.state import merge_changed_files_reducer
from remediation_engine.runtime.path_policy import (
    WorkspacePathError,
    normalize_workspace_path,
    repository_relative_path,
    resolve_repository_path,
)


@pytest.mark.parametrize(
    "value",
    [
        "../outside.txt",
        "nested/../../outside.txt",
        "/tmp/outside.txt",
        "C:/outside.txt",
        r"\\server\share\outside.txt",
        "/workspace/../outside.txt",
    ],
)
def test_workspace_paths_reject_absolute_and_traversal_forms(value: str) -> None:
    """All model/scanner path spellings fail before file I/O."""
    with pytest.raises(WorkspacePathError):
        normalize_workspace_path(value)


def test_workspace_prefix_is_normalized_but_root_is_not_a_file() -> None:
    assert normalize_workspace_path("/workspace/src/app.ts") == "src/app.ts"
    with pytest.raises(WorkspacePathError):
        normalize_workspace_path("/workspace")


def test_resolve_repository_path_rejects_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspacePathError):
        resolve_repository_path(tmp_path, "linked/secret.txt")


def test_repository_relative_path_accepts_nested_and_workspace_prefixed_files(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "src" / "app.ts"
    nested.parent.mkdir()
    nested.write_text("export {}", encoding="utf-8")

    assert repository_relative_path("src/app.ts", tmp_path) == "src/app.ts"
    assert repository_relative_path(str(nested), tmp_path) == "src/app.ts"
    assert repository_relative_path("/workspace/src/app.ts", tmp_path) == "src/app.ts"
    assert repository_relative_path("../outside.ts", tmp_path) is None
    assert repository_relative_path(r"C:\outside.ts", tmp_path) is None


def test_workspace_prefixed_path_rejects_host_symlink_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-workspace"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    assert repository_relative_path("/workspace/linked/secret.txt", tmp_path) is None


def test_changed_file_reducer_rejects_traversal_instead_of_dropping_it() -> None:
    with pytest.raises(WorkspacePathError, match="rejected path"):
        merge_changed_files_reducer([], ["../outside.ts"])


def test_changed_file_reducer_rejects_non_string_paths() -> None:
    with pytest.raises(WorkspacePathError, match="non-string path"):
        merge_changed_files_reducer([], [None])  # type: ignore[list-item]
