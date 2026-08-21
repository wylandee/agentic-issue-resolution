"""Contract tests for the supported Python and CLI entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

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
        "report_path": None,
    }


@pytest.mark.parametrize("repo_root", [Path("relative/repo"), Path("missing-repo")])
def test_request_rejects_invalid_repository_paths(repo_root: Path) -> None:
    """The public API fails before orchestration for unsafe repository roots."""
    with pytest.raises(ValueError, match="repo_root"):
        RemediationRequest(repo_root=repo_root)


def test_request_rejects_file_as_repository_root(tmp_path: Path) -> None:
    """A file cannot be used as the host repository workspace."""
    file_path = tmp_path / "not-a-repository"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="existing directory"):
        RemediationRequest(repo_root=file_path)


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


def test_run_remediation_rejects_success_status_when_final_task_failed(tmp_path: Path) -> None:
    """The public result cannot claim success over an unfixable task."""
    request = RemediationRequest(repo_root=tmp_path)
    state = {
        "status": "completed",
        "task_queue": {
            "task-1": {
                "task_id": "task-1",
                "status": "unfixable",
            }
        },
        "errors": [],
    }
    with patch("remediation_engine.api.run_orchestrator", return_value=state):
        result = run_remediation(request)

    assert result.status == "completed_with_errors"
    assert "task task-1 ended in unfixable" in result.errors[0]


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


def test_preprocessed_groups_expand_compressed_npm_ancestry(tmp_path: Path) -> None:
    """Cached groups receive complete ancestry before Supervisor planning."""
    package_json = {
        "name": "fixture-app",
        "version": "1.0.0",
        "dependencies": {"sqlite3": "^5.1.7"},
    }
    (tmp_path / "package.json").write_text(json.dumps(package_json), encoding="utf-8")
    (tmp_path / "package-lock.json").write_text(
        json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"dependencies": {"sqlite3": "^5.1.7"}},
                    "node_modules/sqlite3": {
                        "version": "5.1.7",
                        "optionalDependencies": {"node-gyp": "8.x"},
                    },
                    "node_modules/sqlite3/node_modules/node-gyp": {
                        "version": "8.4.1",
                        "dependencies": {"make-fetch-happen": "^9.1.0"},
                    },
                    "node_modules/sqlite3/node_modules/make-fetch-happen": {
                        "version": "9.1.0",
                        "dependencies": {"http-proxy-agent": "^4.0.1"},
                    },
                    "node_modules/sqlite3/node_modules/http-proxy-agent": {
                        "version": "4.0.1",
                        "dependencies": {"@tootallnate/once": "1"},
                    },
                    "node_modules/@tootallnate/once": {"version": "1.1.2"},
                },
            }
        ),
        encoding="utf-8",
    )
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name="@tootallnate/once",
        package_version="1.1.2",
        file_path="/src/package-lock.json?sqlite3:5.1.7/http-proxy-agent:4.0.1/@tootallnate/once:1.1.2",
    )
    compressed_ancestry = ["sqlite3", "http-proxy-agent", "@tootallnate/once"]
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        is_direct_dependency=False,
        dependency_ancestry=compressed_ancestry,
        dependency_versions={
            "sqlite3": "5.1.7",
            "http-proxy-agent": "4.0.1",
            "@tootallnate/once": "1.1.2",
        },
        parent_package_name="sqlite3",
        parent_package_version="5.1.7",
        parent_declaration_type="dependencies",
    )
    group = VulnerabilityGroup(
        group_id="sca:package.json:@tootallnate/once:UPDATE_VERSION",
        issue_type=IssueType.SCA,
        vulnerable_component="@tootallnate/once",
        file_path="package.json",
        dependency_ancestry=compressed_ancestry,
        dependency_versions=localized.dependency_versions,
        parent_package_name="sqlite3",
        parent_package_version="5.1.7",
        parent_declaration_type="dependencies",
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
    )

    normalized = normalize_group_paths([group], str(tmp_path))[0]

    expected_ancestry = [
        "sqlite3",
        "node-gyp",
        "make-fetch-happen",
        "http-proxy-agent",
        "@tootallnate/once",
    ]
    assert normalized.dependency_ancestry == expected_ancestry
    assert normalized.localized_issues[0].dependency_ancestry == expected_ancestry
    assert normalized.parent_package_name == "sqlite3"


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
