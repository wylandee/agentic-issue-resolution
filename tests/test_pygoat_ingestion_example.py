"""Offline fixture tests for the PyGoat Python remediation example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

from remediation_engine.api import RemediationRequest
from remediation_engine.cli import _load_issues
from remediation_engine.cli import main as cli_main
from remediation_engine.contracts.schemas import IssueSource, IssueType

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE_ROOT = _PROJECT_ROOT / "examples" / "pygoat"
_SYNTHETIC_REPORT = _EXAMPLE_ROOT / "fixtures" / "synthetic-pypi-dependency-check-report.json"


def _load_runner() -> ModuleType:
    """Load the example runner without importing it as an installed package."""
    path = _EXAMPLE_ROOT / "run.py"
    spec = importlib.util.spec_from_file_location("pygoat_example_runner", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load PyGoat runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResult:
    """Minimal typed-result projection returned by the patched public API."""

    status = "completed"
    changed_files = ["requirements.txt"]
    diff = "--- a/requirements.txt\n+++ b/requirements.txt\n"
    errors: list[str] = []
    raw_state: dict[str, object] = {}

    def model_dump(self, **_: object) -> dict[str, object]:
        """Return the JSON fields consumed by the runner."""
        return {
            "status": self.status,
            "changed_files": self.changed_files,
            "diff": self.diff,
            "errors": self.errors,
        }


def test_synthetic_python_report_roundtrips_and_runner_emits_patch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Exercise fixture ingestion and the runner contract without live services."""
    monkeypatch.setattr("remediation_engine.cli.load_dotenv", lambda: None)
    issues_path = tmp_path / "baseline_issues.jsonl"
    assert (
        cli_main(
            [
                "ingest",
                str(_SYNTHETIC_REPORT),
                "--format",
                "odc-json",
                "--output",
                str(issues_path),
            ]
        )
        == 0
    )

    lines = issues_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    parsed_issues = _load_issues(issues_path, "jsonl")
    assert len(parsed_issues) == 1
    assert parsed_issues[0].source is IssueSource.ODC
    assert parsed_issues[0].issue_type is IssueType.SCA
    assert parsed_issues[0].ecosystem == "pypi"
    assert parsed_issues[0].purl == "pkg:pypi/django@2.2.9"

    runner = _load_runner()
    repo_root = tmp_path / "pygoat"
    repo_root.mkdir()
    result_path = tmp_path / "pygoat-result.json"
    patch_path = tmp_path / "pygoat.patch"
    fake_result = _FakeResult()
    run = MagicMock(return_value=fake_result)
    monkeypatch.setattr(runner, "load_dotenv", lambda: None)
    monkeypatch.setattr(runner, "run_remediation", run)

    assert runner._DEFAULT_REPO == _PROJECT_ROOT / "data" / "clones" / "pygoat"
    assert runner._DEFAULT_ISSUES.name == "baseline_issues.jsonl"
    assert (
        runner.main(
            [
                "--repo",
                str(repo_root),
                "--issues",
                str(issues_path),
                "--output",
                str(result_path),
                "--patch-out",
                str(patch_path),
            ]
        )
        == 0
    )

    request = run.call_args.args[0]
    assert isinstance(request, RemediationRequest)
    assert request.repo_root == repo_root.resolve()
    assert request.system_context is not None
    assert request.system_context.primary_language == "python"
    assert len(request.issues) == 1
    assert request.issues[0].ecosystem == "pypi"
    assert json.loads(result_path.read_text(encoding="utf-8")) == fake_result.model_dump()
    assert patch_path.read_text(encoding="utf-8") == fake_result.diff
    assert list(repo_root.iterdir()) == []
