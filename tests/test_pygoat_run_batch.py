"""Offline tests for the PyGoat batch remediation runner."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from examples.pygoat.run_batch import (
    filter_issues_for_packages,
    get_unique_packages,
    load_baseline_issues,
    run_batch,
    sample_package_batches,
    write_issues_jsonl,
)

from remediation_engine.api import RemediationRequest, RemediationResult
from remediation_engine.contracts.schemas import VulnerabilityIssue


def _issue(index: int, package_name: str) -> VulnerabilityIssue:
    """Build a deterministic PyPI issue for batch runner tests."""
    return VulnerabilityIssue(
        id=f"00000000-0000-0000-0000-{index:012d}",
        source="odc",
        issue_type="sca",
        package_name=package_name,
        package_version="1.0.0",
        purl=f"pkg:pypi/{package_name}@1.0.0",
        ecosystem="pypi",
        message=f"Test issue for {package_name}",
        cve_id=f"CVE-2025-{index:04d}",
    )


def _write_baseline(path: Path, issues: list[VulnerabilityIssue]) -> None:
    """Write deterministic canonical JSONL for a test baseline."""
    path.write_text(
        "\n".join(issue.model_dump_json(exclude_none=False) for issue in issues) + "\n",
        encoding="utf-8",
    )


def test_load_baseline_issues_and_package_helpers(tmp_path: Path) -> None:
    """Load JSONL and JSON arrays, then derive and filter package batches."""
    issues = [_issue(1, "django"), _issue(2, "pyyaml"), _issue(3, "django")]
    jsonl_path = tmp_path / "baseline.jsonl"
    _write_baseline(jsonl_path, issues)
    assert load_baseline_issues(jsonl_path) == issues
    assert get_unique_packages(issues) == ["django", "pyyaml"]
    assert [issue.package_name for issue in filter_issues_for_packages(issues, {"pyyaml"})] == [
        "pyyaml"
    ]

    array_path = tmp_path / "baseline.json"
    array_path.write_text(
        json.dumps([issue.model_dump(mode="json", exclude_none=False) for issue in issues]),
        encoding="utf-8",
    )
    assert load_baseline_issues(array_path) == issues


def test_sample_package_batches_are_seeded_unique_and_validated() -> None:
    """Produce repeatable distinct combinations and reject invalid sizes."""
    packages = ["django", "pyyaml", "certifi", "pyjwt"]
    batches = sample_package_batches(packages, batch_size=2, num_batches=5, seed=7)

    assert batches == sample_package_batches(packages, batch_size=2, num_batches=5, seed=7)
    assert len(batches) == 5
    assert len({tuple(batch) for batch in batches}) == 5
    assert all(len(batch) == 2 and batch == sorted(batch) for batch in batches)

    with pytest.raises(ValueError, match="Cannot sample batch"):
        sample_package_batches(["django"], batch_size=2, num_batches=1)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        sample_package_batches(["django"], batch_size=0, num_batches=1)


def test_write_issues_jsonl_serializes_canonical_models(tmp_path: Path) -> None:
    """Persist selected issues as newline-delimited canonical JSON."""
    output_path = tmp_path / "batches" / "issues.jsonl"
    issues = [_issue(1, "django"), _issue(2, "pyyaml")]

    write_issues_jsonl(issues, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["package_name"] for line in lines] == ["django", "pyyaml"]
    assert load_baseline_issues(output_path) == issues


def test_run_batch_dry_run_writes_per_batch_inputs_without_repo_or_engine(
    tmp_path: Path,
) -> None:
    """Prepare batches and summary without a checkout or live service calls."""
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_issues = [
        _issue(1, "django"),
        _issue(2, "pyyaml"),
        _issue(3, "certifi"),
        _issue(4, "pyjwt"),
        _issue(5, "django"),
    ]
    _write_baseline(baseline_path, baseline_issues)
    output_dir = tmp_path / "trajectories"
    missing_repo = tmp_path / "missing-pygoat"

    with patch("examples.pygoat.run_batch.run_remediation") as run_remediation:
        summaries = run_batch(
            iterations=3,
            batch_size=2,
            repo_root=missing_repo,
            baseline_path=baseline_path,
            output_dir=output_dir,
            seed=123,
            dry_run=True,
        )

    run_remediation.assert_not_called()
    assert len(summaries) == 3
    assert len({tuple(summary.selected_packages) for summary in summaries}) == 3
    for summary in summaries:
        assert summary.status == "dry_run"
        assert summary.issue_count > 0
        assert Path(summary.issues_path).is_file()
        batch_issues = load_baseline_issues(Path(summary.issues_path))
        assert {issue.package_name for issue in batch_issues} == set(summary.selected_packages)
        assert not Path(summary.result_path).exists()
        assert not Path(summary.patch_path).exists()

    summary_path = output_dir / "pygoat-batch-runs-summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["total_iterations"] == 3
    assert payload["dry_run"] is True
    assert len(payload["results"]) == 3
    assert not missing_repo.exists()


def test_run_batch_calls_python_remediation_and_writes_result_artifacts(tmp_path: Path) -> None:
    """Pass each filtered batch through the Python remediation contract."""
    repo_root = tmp_path / "pygoat"
    repo_root.mkdir()
    baseline_path = tmp_path / "baseline.jsonl"
    baseline_issues = [
        _issue(1, "django"),
        _issue(2, "pyyaml"),
        _issue(3, "certifi"),
        _issue(4, "pyjwt"),
    ]
    _write_baseline(baseline_path, baseline_issues)
    output_dir = tmp_path / "trajectories"
    result = RemediationResult(
        run_id="test-run-id",
        status="completed",
        diff="--- a/requirements.txt\n+++ b/requirements.txt\n",
        changed_files=["requirements.txt"],
        task_count=1,
        fixed_task_count=1,
        failed_task_count=0,
        errors=[],
        raw_state={},
    )

    with patch("examples.pygoat.run_batch.run_remediation", return_value=result) as run_remediation:
        summaries = run_batch(
            iterations=2,
            batch_size=2,
            repo_root=repo_root,
            baseline_path=baseline_path,
            output_dir=output_dir,
            seed=5,
        )

    assert run_remediation.call_count == 2
    assert len(summaries) == 2
    for call, summary in zip(run_remediation.call_args_list, summaries, strict=True):
        request = call.args[0]
        assert isinstance(request, RemediationRequest)
        assert request.repo_root == repo_root
        assert request.system_context is not None
        assert request.system_context.primary_language == "python"
        assert {issue.package_name for issue in request.issues} == set(summary.selected_packages)
        assert summary.status == "completed"
        assert summary.changed_files == ["requirements.txt"]
        assert (
            json.loads(Path(summary.result_path).read_text(encoding="utf-8"))["status"]
            == "completed"
        )
        assert Path(summary.patch_path).read_text(encoding="utf-8") == result.diff
