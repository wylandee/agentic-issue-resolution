"""Tests for deterministic repository-map prompt context."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from remediation_engine.tools.repository_map import build_repository_map


def test_repository_map_is_sorted_and_excludes_generated_or_dependency_paths(tmp_path):
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "b.py").write_text("b", encoding="utf-8")
    (source_dir / "a.py").write_text("a", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
    (tmp_path / "bundle.js.map").write_text("ignored", encoding="utf-8")
    (tmp_path / "generated.map").mkdir()

    assert build_repository_map(tmp_path) == "\n".join(
        ["a.txt", "src", "src/a.py", "src/b.py", "z.txt"]
    )


def test_repository_map_preserves_empty_and_truncation_behavior(tmp_path):
    assert build_repository_map(tmp_path) == "(workspace is empty)"

    for name in ("a", "b", "c", "d"):
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")

    assert build_repository_map(tmp_path, max_entries=2) == (
        "a.txt\nb.txt\n... (truncated, 2 more entries)"
    )


def test_repository_map_rejects_non_positive_limit(tmp_path):
    with pytest.raises(ValueError, match="max_entries"):
        build_repository_map(tmp_path, max_entries=0)


def test_repository_map_returns_stable_error_when_read_fails(tmp_path):
    with patch(
        "remediation_engine.tools.repository_map.os.walk",
        side_effect=PermissionError("permission denied"),
    ):
        result = build_repository_map(tmp_path)

    assert result.startswith("ERROR_CODE: REPOSITORY_MAP_UNAVAILABLE:")
    assert "permission denied" in result
