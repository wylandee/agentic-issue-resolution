"""Regression tests for the maintained Juice Shop fixture runners."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from remediation_engine.contracts.schemas import VulnerabilityGroup

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner(name: str, path: Path) -> ModuleType:
    """Load an example runner as an isolated test module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load fixture runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    """Minimal API result used to stop fixture tests before external execution."""

    status = "completed"
    errors: list[str] = []
    changed_files: list[str] = []
    diff = ""
    raw_state: dict[str, object] = {}

    def model_dump(self, **_: object) -> dict[str, object]:
        """Return a JSON-serializable result projection."""
        return {"status": self.status, "changed_files": [], "diff": "", "errors": []}


@pytest.mark.parametrize(
    ("name", "runner_path", "fixture_path", "loader_name"),
    [
        (
            "parent_first",
            "examples/juice_shop/fixtures/transitive_parent_first/run_parent_first.py",
            "examples/juice_shop/fixtures/transitive_parent_first/triaged_groups_transitive_parent_first.json",
            "load_parent_first_groups",
        ),
        (
            "deterministic_routing",
            "examples/juice_shop/fixtures/deterministic_routing/run_deterministic_routing.py",
            "examples/juice_shop/fixtures/deterministic_routing/triaged_groups_deterministic.json",
            "load_fixture",
        ),
        (
            "shared_dependencies",
            "examples/juice_shop/fixtures/shared_dependencies/run_shared_dependencies.py",
            "examples/juice_shop/fixtures/shared_dependencies/triaged_groups_shared_dependencies.json",
            "load_shared_dependency_groups",
        ),
        (
            "suppressed",
            "examples/juice_shop/fixtures/suppressed/run_post_triage.py",
            "examples/juice_shop/fixtures/suppressed/triaged_groups_suppressed.json",
            None,
        ),
    ],
)
def test_pretriaged_runner_passes_group_issues_as_baseline(
    name: str,
    runner_path: str,
    fixture_path: str,
    loader_name: str | None,
    tmp_path: Path,
) -> None:
    """Pre-triaged runners must give final scanning the findings they process."""
    module = _load_runner(name, _PROJECT_ROOT / runner_path)
    fixture = _PROJECT_ROOT / fixture_path

    if loader_name is not None:
        groups = getattr(module, loader_name)(fixture)
    else:
        groups = [
            VulnerabilityGroup.model_validate(item)
            for item in json.loads(fixture.read_text(encoding="utf-8"))
        ]

    expected_issue_ids = list(
        dict.fromkeys(str(issue.id) for group in groups for issue in group.issues)
    )
    fake_result = _FakeResult()
    output_path = tmp_path / f"{name}-result.json"
    patch_path = tmp_path / f"{name}.patch"
    arguments = [
        "--repo",
        str(tmp_path),
        "--groups",
        str(fixture),
        "--output",
        str(output_path),
        "--patch-out",
        str(patch_path),
    ]
    if name == "deterministic_routing":
        arguments = [
            "--repo",
            str(tmp_path),
            "--fixture",
            str(fixture),
            "--output",
            str(output_path),
            "--patch-out",
            str(patch_path),
        ]

    with (
        patch.object(module, "run_remediation", return_value=fake_result) as run,
        patch("sys.argv", [str(_PROJECT_ROOT / runner_path), *arguments]),
    ):
        assert module.main() == 0

    request = run.call_args.args[0]
    actual_issue_ids = [str(issue.id) for issue in request.issues]
    assert actual_issue_ids == expected_issue_ids
    assert {group.group_id for group in request.valid_groups} == {
        group.group_id for group in groups
    }


def test_pretriaged_runner_rejects_group_without_baseline_issue() -> None:
    """A malformed pre-triaged group cannot silently erase the scan baseline."""
    module = _load_runner(
        "parent_first_validation",
        _PROJECT_ROOT / "examples/juice_shop/fixtures/transitive_parent_first/run_parent_first.py",
    )
    groups = module.load_parent_first_groups()
    group = groups[0].model_copy(update={"issues": []})

    with pytest.raises(ValueError, match="has no baseline issues"):
        module._baseline_issues_from_groups([group])
