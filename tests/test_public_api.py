"""Contract tests for the supported Python and CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from remediation_engine.api import RemediationRequest, RemediationResult, run_remediation
from remediation_engine.cli import _load_issues, build_parser, main
from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    LocalizedIssue,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.state import normalize_group_paths


def _issue() -> VulnerabilityIssue:
    """Build a minimal canonical SAST finding for API tests."""
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.MEDIUM,
        rule_id="javascript.test",
        file_path="src/app.js",
        message="Unsafe test sink.",
    )


def test_request_result_models_are_stable(tmp_path: Path) -> None:
    """The public models accept paths and expose only patch-oriented fields."""
    request = RemediationRequest(repo_root=tmp_path, issues=[_issue()])
    result = RemediationResult(status="completed", diff="", raw_state={"internal": True})

    assert request.repo_root == tmp_path
    assert result.model_dump() == {
        "status": "completed",
        "changed_files": [],
        "diff": "",
        "errors": [],
        "trajectory_path": None,
    }


def test_run_remediation_projects_orchestrator_state(tmp_path: Path) -> None:
    """The API converts graph state into a typed result without host edits."""
    request = RemediationRequest(repo_root=tmp_path, valid_groups=[])
    state = {
        "status": "completed",
        "changed_files": ["src/app.js"],
        "diff": "--- a/src/app.js\n+++ b/src/app.js\n",
        "errors": [],
        "trajectory_path": "data/trajectories/run.md",
    }
    with (
        patch("remediation_engine.api.triage_issues", return_value=[]),
        patch("remediation_engine.api.run_orchestrator", return_value=state),
    ):
        result = run_remediation(request)

    assert result.status == "completed"
    assert result.changed_files == ["src/app.js"]
    assert result.trajectory_path == "data/trajectories/run.md"


def test_run_remediation_leaves_initial_triage_to_graph(tmp_path: Path) -> None:
    """An API run passes issues to the graph without a hidden pre-triage call."""
    request = RemediationRequest(repo_root=tmp_path, issues=[_issue()])
    state = {"status": "completed", "errors": []}
    with (
        patch("remediation_engine.api.triage_issues") as mock_triage,
        patch("remediation_engine.api.run_orchestrator", return_value=state) as mock_orchestrator,
    ):
        run_remediation(request)

    mock_triage.assert_not_called()
    assert mock_orchestrator.call_args.kwargs["valid_groups"] == []
    assert mock_orchestrator.call_args.kwargs["issues"] == request.issues


def test_run_remediation_reports_completed_with_errors(tmp_path: Path) -> None:
    """A teardown completion with worker errors is not reported as success."""
    request = RemediationRequest(repo_root=tmp_path)
    state = {"status": "completed", "errors": ["worker surrendered"]}
    with patch("remediation_engine.api.run_orchestrator", return_value=state):
        result = run_remediation(request)

    assert result.status == "completed_with_errors"
    assert result.errors == ["worker surrendered"]


def test_cli_auto_loads_canonical_jsonl(tmp_path: Path) -> None:
    """The CLI auto format recognizes JSONL as the canonical issue format."""
    path = tmp_path / "issues.jsonl"
    path.write_text(json.dumps(_issue().model_dump(mode="json")) + "\n", encoding="utf-8")

    issues = _load_issues(path, "auto")

    assert len(issues) == 1
    assert issues[0].rule_id == "javascript.test"


def test_cli_loads_legacy_json_array_with_jsonl_format(tmp_path: Path) -> None:
    """Legacy pretty-printed arrays remain readable as canonical findings."""
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps([_issue().model_dump(mode="json")], indent=2), encoding="utf-8")

    issues = _load_issues(path, "jsonl")

    assert len(issues) == 1
    assert issues[0].rule_id == "javascript.test"


def test_cli_ingest_writes_one_json_object_per_line(tmp_path: Path) -> None:
    """The ingest command emits canonical JSONL rather than a JSON array."""
    source = tmp_path / "issues.jsonl"
    output = tmp_path / "normalized.jsonl"
    source.write_text(json.dumps(_issue().model_dump(mode="json")) + "\n", encoding="utf-8")

    with patch("remediation_engine.cli.load_dotenv"):
        assert main(["ingest", str(source), "--format", "jsonl", "--output", str(output)]) == 0
    lines = [line for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    assert not output.read_text(encoding="utf-8").lstrip().startswith("[")
    assert json.loads(lines[0])["rule_id"] == "javascript.test"


def test_graph_group_paths_are_repo_relative(tmp_path: Path) -> None:
    """Cached groups with absolute manifest paths are safe for workers."""
    issue = _issue().model_copy(
        update={
            "source": IssueSource.ODC,
            "issue_type": IssueType.SCA,
            "package_name": "lodash",
        }
    )
    absolute_manifest = str(tmp_path / "package.json")
    group = VulnerabilityGroup(
        group_id=f"sca:{absolute_manifest}:lodash:UPDATE_VERSION",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path=absolute_manifest,
        file_paths=[absolute_manifest],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[LocalizedIssue(issue=issue, manifest_file=absolute_manifest)],
    )

    normalized = normalize_group_paths([group], str(tmp_path))[0]

    assert normalized.file_path == "package.json"
    assert normalized.file_paths == ["package.json"]
    assert normalized.localized_issues[0].manifest_file == "package.json"
    assert str(tmp_path).replace("\\", "/") not in normalized.group_id


def test_cli_exposes_ingest_triage_and_run_commands() -> None:
    """The CLI parser keeps the three supported operational commands."""
    parser = build_parser()

    for command in ("ingest", "triage", "run"):
        args = parser.parse_args(
            [command, "issues.jsonl", "--repo", "."]
            if command == "run"
            else [command, "issues.jsonl"]
        )
        assert args.command == command
