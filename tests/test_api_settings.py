"""Focused tests for api settings."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from remediation_engine.agents.qa_evaluation import _qa_log_records_for_phase
from remediation_engine.agents.workaround import (
    _build_workaround_system_prompt,
    _default_no_fix_package_manager,
)
from remediation_engine.api import (
    RemediationRequest,
    RemediationResult,
    run_remediation,
    triage_issues,
)
from remediation_engine.cli import _load_issues, build_parser, main
from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
    LocalizedIssue,
    RemediationTask,
    RoutingStrategy,
    Severity,
    SystemContext,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import (
    LANGUAGE_CONFIGS,
    ProjectLanguage,
    detect_project_language,
    resolve_project_language,
)
from remediation_engine.orchestration.graph import triage_node
from remediation_engine.orchestration.state import (
    initial_orchestrator_state,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
    normalize_group_paths,
)
from remediation_engine.settings import (
    DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET,
    AppSettings,
    get_bound_runtime_settings,
    get_runtime_settings,
)
from remediation_engine.tools.qa_scan import _QAExecutionResults


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


def test_target_package_scope_is_development_only(tmp_path: Path) -> None:
    """Package scoping is normalized and rejected for production requests."""
    request = RemediationRequest(
        repo_root=tmp_path,
        target_packages=[" @angular/core", "@angular/core", "@angular/common "],
        system_context=SystemContext(environment="development"),
    )

    assert request.target_packages == ["@angular/common", "@angular/core"]

    with pytest.raises(ValueError, match="development-only"):
        RemediationRequest(
            repo_root=tmp_path,
            target_packages=["@angular/core"],
            system_context=SystemContext(environment="production"),
        )


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


def test_triage_issues_binds_injected_settings_and_leaves_fix_planning_to_supervisor(
    monkeypatch,
) -> None:
    """Standalone triage localizes findings and does not run advisory planning."""
    observed_keys: list[str] = []
    settings = AppSettings(openai_api_key="injected-test-key")
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-12345",
        package_name="fixture-package",
        package_version="1.0.0",
        file_path="package.json",
    )

    def observe_enrichment(_cve_ids, *, settings=None):
        observed_keys.append(get_runtime_settings().openai_api_key)
        assert settings is not None
        return {}

    planner_mock = Mock()
    monkeypatch.setattr("remediation_engine.tools.advisories.plan_fix", planner_mock)
    monkeypatch.setattr("remediation_engine.agents.triage._run_enrichment", observe_enrichment)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    groups = triage_issues([issue], settings=settings)

    assert groups
    assert observed_keys == ["injected-test-key"]
    assert groups[0].fix_plan is None
    assert groups[0].fix_plan_candidates == []
    planner_mock.assert_not_called()
    assert get_bound_runtime_settings() is None


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


def test_cli_rejects_json_array_with_jsonl_format(tmp_path: Path) -> None:
    """JSON arrays are rejected when jsonl format is specified."""
    path = tmp_path / "array.jsonl"
    path.write_text(json.dumps([_issue().model_dump(mode="json")], indent=2), encoding="utf-8")

    with pytest.raises(ValueError):
        _load_issues(path, "jsonl")


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
        group_id=f"sca:{absolute_manifest}:lodash",
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
        group_id="sca:package.json:@tootallnate/once",
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


def test_run_remediation_preserves_context_when_filling_language(tmp_path: Path) -> None:
    """A missing language is filled without replacing caller context."""
    (tmp_path / "requirements.txt").touch()
    context = SystemContext(
        public_facing=False,
        deployment_os="linux",
        deployment_architecture="service",
        environment="staging",
        data_sensitivity="high",
        tags={"team": "platform"},
    )
    request = RemediationRequest(repo_root=tmp_path, system_context=context)
    with patch(
        "remediation_engine.api.run_orchestrator",
        return_value={"status": "completed", "errors": []},
    ) as orchestrator:
        run_remediation(request)

    received = orchestrator.call_args.kwargs["system_context"]
    assert received is not context
    assert received.primary_language == ProjectLanguage.PYTHON.value
    assert received.environment == "staging"
    assert received.public_facing is False
    assert received.deployment_architecture == "service"
    assert received.data_sensitivity == "high"
    assert received.tags == {"team": "platform"}
    assert context.primary_language is None
    assert orchestrator.call_args.kwargs["project_language"] is ProjectLanguage.PYTHON


def test_triage_issues_preserves_context_when_filling_language(tmp_path: Path) -> None:
    """Triage derives execution language without discarding caller metadata."""
    (tmp_path / "Pipfile").touch()
    context = SystemContext(environment="staging", tags={"owner": "security"})
    with patch("remediation_engine.api.run_triage_pipeline", return_value=[]) as pipeline:
        triage_issues([], repo_root=tmp_path, system_context=context)

    received = pipeline.call_args.args[1]
    assert received is not context
    assert received.primary_language == ProjectLanguage.PYTHON.value
    assert received.environment == "staging"
    assert received.tags == {"owner": "security"}
    assert context.primary_language is None


def test_context_free_raw_issue_triage_keeps_graph_skip(tmp_path: Path) -> None:
    """Raw findings without a system context still skip graph triage."""
    state = initial_orchestrator_state(
        str(tmp_path),
        [],
        issues=[_issue()],
        system_context=None,
    )
    with patch("remediation_engine.orchestration.graph.run_triage_pipeline") as pipeline:
        result = triage_node(state)

    pipeline.assert_not_called()
    assert state["project_language"] is ProjectLanguage.NODEJS
    assert result["status"] == "triage_skipped"
    assert result["initial_triage_status"] == "skipped_missing_input"


_LLM_REASONING_EFFORT_ENV_NAMES = (
    "REMEDY_LLM_REASONING_EFFORT",
    "SUPERVISOR_LLM_REASONING_EFFORT",
    "TRIAGE_LLM_REASONING_EFFORT",
    "UPDATE_LLM_REASONING_EFFORT",
    "WORKAROUND_LLM_REASONING_EFFORT",
    "QA_LLM_REASONING_EFFORT",
)


def _clear_llm_reasoning_effort_environment(monkeypatch):
    for env_name in _LLM_REASONING_EFFORT_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)


def test_node_models_can_be_configured_independently(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "remedy-default")
    monkeypatch.setenv("SUPERVISOR_LLM_MODEL", "supervisor-model")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "triage-model")
    monkeypatch.setenv("UPDATE_LLM_MODEL", "update-model")
    monkeypatch.setenv("WORKAROUND_LLM_MODEL", "workaround-model")
    monkeypatch.setenv("QA_LLM_MODEL", "qa-model")

    settings = AppSettings.from_env()

    assert settings.triage_llm_model == "triage-model"
    assert settings.supervisor_llm_model == "supervisor-model"
    assert settings.update_llm_model == "update-model"
    assert settings.workaround_llm_model == "workaround-model"
    assert settings.qa_llm_model == "qa-model"


@pytest.mark.parametrize(
    ("env_name", "field_name"),
    [
        ("REMEDY_LLM_REASONING_EFFORT", "remedy_llm_reasoning_effort"),
        ("SUPERVISOR_LLM_REASONING_EFFORT", "supervisor_llm_reasoning_effort"),
        ("TRIAGE_LLM_REASONING_EFFORT", "triage_llm_reasoning_effort"),
        ("UPDATE_LLM_REASONING_EFFORT", "update_llm_reasoning_effort"),
        ("WORKAROUND_LLM_REASONING_EFFORT", "workaround_llm_reasoning_effort"),
        ("QA_LLM_REASONING_EFFORT", "qa_llm_reasoning_effort"),
    ],
)
@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "xhigh", "max"])
def test_luna_reasoning_efforts_are_configurable_per_agent(
    monkeypatch, env_name, field_name, effort
):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    for model_env in (
        "TRIAGE_LLM_MODEL",
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
    ):
        monkeypatch.delenv(model_env, raising=False)
    monkeypatch.setenv(env_name, effort)

    assert getattr(AppSettings.from_env(), field_name) == effort


def test_luna_reasoning_effort_defaults_to_provider_default(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")

    assert AppSettings.from_env().remedy_llm_reasoning_effort is None


def test_supervisor_model_and_effort_inherit_default_remedy_settings(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", "medium")
    monkeypatch.delenv("SUPERVISOR_LLM_MODEL", raising=False)

    settings = AppSettings.from_env()

    assert settings.supervisor_llm_model is None
    assert settings.supervisor_llm_reasoning_effort is None
    assert settings.remedy_llm_model == "gpt-6-luna"
    assert settings.remedy_llm_reasoning_effort == "medium"


def test_supervisor_effort_requires_luna_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SUPERVISOR_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("SUPERVISOR_LLM_REASONING_EFFORT", "xhigh")

    with pytest.raises(ValueError, match="SUPERVISOR_LLM_REASONING_EFFORT only applies"):
        AppSettings.from_env()


def test_luna_reasoning_effort_is_case_normalized(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", " HIGH ")

    assert AppSettings.from_env().remedy_llm_reasoning_effort == "high"


def test_luna_reasoning_effort_rejects_unknown_values(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", "minimal")

    with pytest.raises(ValueError, match="REMEDY_LLM_REASONING_EFFORT must be one of"):
        AppSettings.from_env()


def test_luna_reasoning_effort_requires_gpt_6_luna(monkeypatch):
    _clear_llm_reasoning_effort_environment(monkeypatch)
    monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("REMEDY_LLM_REASONING_EFFORT", "low")

    with pytest.raises(ValueError, match="only applies when its model is gpt-6-luna"):
        AppSettings.from_env()


def test_remediation_node_models_fall_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    for name in (
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = AppSettings.from_env()

    assert settings.remedy_llm_model == "shared-remedy-model"
    assert settings.update_llm_model == "shared-remedy-model"
    assert settings.workaround_llm_model == "shared-remedy-model"
    assert settings.qa_llm_model == "shared-remedy-model"


def test_empty_triage_model_falls_back_to_legacy_remedy_model(monkeypatch):
    monkeypatch.setenv("REMEDY_LLM_MODEL", "shared-remedy-model")
    monkeypatch.setenv("TRIAGE_LLM_MODEL", "")

    assert AppSettings.from_env().triage_llm_model == "shared-remedy-model"


def test_report_settings_are_explicit(monkeypatch, tmp_path):
    """Report persistence directory is configurable via environment."""
    monkeypatch.setenv("REMEDIATION_REPORT_DIR", str(tmp_path))

    settings = AppSettings.from_env()

    assert settings.remediation_report_dir == tmp_path


def test_retriage_limit_is_disabled_by_default_and_toggleable(monkeypatch):
    monkeypatch.delenv("REMEDY_RETRIAGE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("REMEDY_RETRIAGE_LIMIT", raising=False)
    assert AppSettings.from_env().remedy_retriage_limit_enabled is False
    assert AppSettings.from_env().remedy_retriage_limit == 3

    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT", "7")
    assert AppSettings.from_env().remedy_retriage_limit_enabled is True
    assert AppSettings.from_env().remedy_retriage_limit == 7


def test_final_full_scan_is_disabled_by_default_and_toggleable(monkeypatch):
    variable = "REMEDY_FINAL_FULL_SCAN_ENABLED"
    monkeypatch.delenv(variable, raising=False)
    assert AppSettings().remedy_final_full_scan_enabled is False
    assert AppSettings.from_env().remedy_final_full_scan_enabled is False

    monkeypatch.setenv(variable, "true")
    assert AppSettings.from_env().remedy_final_full_scan_enabled is True


def test_retriage_limit_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("REMEDY_RETRIAGE_LIMIT", "not-a-number")

    with pytest.raises(ValueError, match="REMEDY_RETRIAGE_LIMIT must be an integer"):
        AppSettings.from_env()


def test_solver_certification_timeout_has_positive_configurable_default(monkeypatch):
    monkeypatch.delenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", raising=False)
    assert AppSettings.from_env().solver_certification_timeout_seconds == 900

    monkeypatch.setenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", "37")
    assert AppSettings.from_env().solver_certification_timeout_seconds == 37

    monkeypatch.setenv("REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="must be at least 1"):
        AppSettings.from_env()


def test_candidate_limit_default_and_override(monkeypatch):
    variable = "REMEDY_SOLVER_MAX_CANDIDATES_PER_TARGET"
    monkeypatch.delenv(variable, raising=False)

    assert (
        AppSettings().solver_max_candidates_per_target == DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET
    )
    assert (
        AppSettings.from_env().solver_max_candidates_per_target
        == DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET
    )

    monkeypatch.setenv(variable, "96")
    assert AppSettings.from_env().solver_max_candidates_per_target == 96


def test_delta_canary_probe_budget_default_and_override(monkeypatch):
    variable = "REMEDY_MAX_DELTA_CANARY_PROBES"
    monkeypatch.delenv(variable, raising=False)

    assert AppSettings().max_delta_canary_probes == 2
    assert AppSettings.from_env().max_delta_canary_probes == 2

    monkeypatch.setenv(variable, "5")
    assert AppSettings.from_env().max_delta_canary_probes == 5


@pytest.mark.parametrize(
    ("value", "message"),
    [("0", "must be at least 1"), ("not-an-integer", "must be an integer")],
)
def test_delta_canary_probe_budget_requires_positive_integer(monkeypatch, value, message):
    monkeypatch.setenv("REMEDY_MAX_DELTA_CANARY_PROBES", value)

    with pytest.raises(ValueError, match=message):
        AppSettings.from_env()


def test_python_profile_uses_reusable_image_with_native_build_dependencies() -> None:
    image = LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image
    assert image == "remediation-engine-python:3.11"

    dockerfile = Path(__file__).resolve().parents[1] / "docker" / "python-workspace.Dockerfile"
    contents = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim-bookworm" in contents
    assert all(package in contents for package in ("build-essential", "libpq-dev", "python3-dev"))


def _issues_file(tmp_path: Path) -> Path:
    """Create an empty canonical JSONL issue file."""
    path = tmp_path / "issues.jsonl"
    path.write_text("", encoding="utf-8")
    return path


def test_requirements_root_detects_python(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("requests==2.31.0\n", encoding="utf-8")

    assert detect_project_language(tmp_path) is ProjectLanguage.PYTHON


def test_package_json_wins_in_a_mixed_root(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS
    assert resolve_project_language(tmp_path, "  PyThOn3 ") is ProjectLanguage.PYTHON


def test_detection_uses_root_files_only(tmp_path: Path) -> None:
    nested = tmp_path / "service"
    nested.mkdir()
    (nested / "requirements.txt").touch()

    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS
    assert detect_project_language(nested) is ProjectLanguage.PYTHON


def test_unknown_context_language_falls_back_to_detection(tmp_path: Path) -> None:
    (tmp_path / "Pipfile").touch()

    assert resolve_project_language(tmp_path, "Kotlin") is ProjectLanguage.PYTHON


@pytest.mark.parametrize("command", ["triage", "run"])
def test_cli_exposes_language_option_only_for_execution_commands(
    command: str,
) -> None:
    parser = build_parser()
    args = parser.parse_args([command, "issues.jsonl", "--repo", ".", "--language", "python"])

    assert args.language == "python"
    ingest_args = parser.parse_args(["ingest", "issues.jsonl"])
    assert not hasattr(ingest_args, "language")


def test_cli_triage_auto_detects_and_explicitly_overrides_language(tmp_path: Path) -> None:
    issue_path = _issues_file(tmp_path)
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "requirements.txt").touch()
    with (
        patch("remediation_engine.cli.AppSettings.from_env", return_value=object()),
        patch("remediation_engine.cli.triage_issues", return_value=[]) as triage,
    ):
        assert main(["triage", str(issue_path), "--repo", str(tmp_path)]) == 0
        auto_context = triage.call_args.kwargs["system_context"]
        assert auto_context.primary_language == ProjectLanguage.NODEJS.value

        assert (
            main(
                [
                    "triage",
                    str(issue_path),
                    "--repo",
                    str(tmp_path),
                    "--language",
                    "python",
                ]
            )
            == 0
        )
        explicit_context = triage.call_args.kwargs["system_context"]
        assert explicit_context.primary_language == ProjectLanguage.PYTHON.value


def test_cli_run_auto_detects_and_explicitly_overrides_language(tmp_path: Path) -> None:
    issue_path = _issues_file(tmp_path)
    (tmp_path / "requirements.txt").touch()
    result = RemediationResult(status="completed")
    with (
        patch("remediation_engine.cli.AppSettings.from_env", return_value=object()),
        patch("remediation_engine.cli.run_remediation", return_value=result) as run,
    ):
        assert main(["run", str(issue_path), "--repo", str(tmp_path)]) == 0
        auto_context = run.call_args.args[0].system_context
        assert auto_context.primary_language == ProjectLanguage.PYTHON.value

        (tmp_path / "package.json").write_text("{}", encoding="utf-8")
        assert (
            main(
                [
                    "run",
                    str(issue_path),
                    "--repo",
                    str(tmp_path),
                    "--language",
                    "python",
                ]
            )
            == 0
        )
        explicit_context = run.call_args.args[0].system_context
        assert explicit_context.primary_language == ProjectLanguage.PYTHON.value


def test_language_context_builder_keeps_operational_defaults() -> None:
    context = SystemContext(
        public_facing=True,
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language=ProjectLanguage.PYTHON.value,
    )

    assert context.public_facing is True
    assert context.deployment_os == "linux"
    assert context.deployment_architecture == "containerized"
    assert context.environment == "production"


def test_language_configs_preserve_node_contract_and_register_maven_contract() -> None:
    node = LANGUAGE_CONFIGS[ProjectLanguage.NODEJS]
    assert node.docker_image == "node:22"
    assert node.manifest_names == ("package.json",)
    assert node.source_suffixes == frozenset({".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"})
    assert node.install_command == "npm install --package-lock=true"
    assert node.compile_command is None
    assert node.test_command == "npm test"
    assert node.excluded_dirs == frozenset()
    assert "**/tests/**" in node.test_include_patterns
    assert "**/*.test.ts" in node.test_include_patterns

    java = LANGUAGE_CONFIGS[ProjectLanguage.JAVA]
    assert java.docker_image == "maven:3.9-eclipse-temurin-17"
    assert java.manifest_names == ("pom.xml",)
    assert java.source_suffixes == frozenset({".java"})
    assert java.install_command == "mvn -B -q -DskipTests clean package"
    assert java.compile_command == "mvn -B -q -DskipTests compile"
    assert java.test_command == "mvn -B test -Dsurefire.useFile=false"
    assert java.test_include_patterns == (
        "**/Test*.java",
        "**/*Test.java",
        "**/*Tests.java",
        "**/*TestCase.java",
    )
    assert java.excluded_dirs == frozenset({"target"})


def test_detection_is_root_only_and_preserves_node_precedence(tmp_path: Path) -> None:
    nested = tmp_path / "module"
    nested.mkdir()
    (nested / "pom.xml").write_text("<project />", encoding="utf-8")
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS

    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS

    (tmp_path / "package.json").unlink()
    assert detect_project_language(tmp_path) is ProjectLanguage.JAVA


def test_detection_defaults_to_node_without_manifests(tmp_path: Path) -> None:
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS
    assert resolve_project_language(tmp_path) is ProjectLanguage.NODEJS


def test_explicit_node_language_overrides_a_root_maven_manifest(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    assert resolve_project_language(tmp_path, "nodejs") is ProjectLanguage.NODEJS
    assert resolve_project_language(tmp_path, "javascript/nodejs") is ProjectLanguage.NODEJS


def test_explicit_java_language_is_resolved(tmp_path: Path) -> None:
    assert resolve_project_language(tmp_path, "java") is ProjectLanguage.JAVA


@pytest.mark.parametrize("language", ["kotlin", "", "maven"])
def test_unknown_explicit_language_falls_back_to_root_detection(
    tmp_path: Path, language: str
) -> None:
    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    assert resolve_project_language(tmp_path, language) is ProjectLanguage.JAVA


def test_run_api_detects_language_when_context_is_missing(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    request = RemediationRequest(repo_root=tmp_path)
    with patch(
        "remediation_engine.api.run_orchestrator",
        return_value={"status": "completed", "errors": []},
    ) as orchestrator:
        run_remediation(request, settings=Mock())

    context = orchestrator.call_args.kwargs["system_context"]
    assert context.primary_language == ProjectLanguage.JAVA.value
    assert context.public_facing is True
    assert context.deployment_os == "linux"
    assert context.deployment_architecture == "containerized"
    assert context.environment == "production"


def test_api_preserves_explicit_context_fields_while_resolving_missing_language(
    tmp_path: Path,
) -> None:
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    supplied = SystemContext(
        repo_url="https://example.invalid/project",
        environment="staging",
        primary_language=None,
        tags={"owner": "security"},
    )
    request = RemediationRequest(repo_root=tmp_path, system_context=supplied)
    with patch(
        "remediation_engine.api.run_orchestrator",
        return_value={"status": "completed", "errors": []},
    ) as orchestrator:
        run_remediation(request, settings=Mock())

    resolved = orchestrator.call_args.kwargs["system_context"]
    assert resolved.primary_language == ProjectLanguage.NODEJS.value
    assert resolved.repo_url == supplied.repo_url
    assert resolved.environment == supplied.environment
    assert resolved.tags == supplied.tags
    assert supplied.primary_language is None


def test_api_keeps_explicit_java_and_python_contexts(tmp_path: Path) -> None:
    for language in ("java", "python"):
        request = RemediationRequest(
            repo_root=tmp_path,
            system_context=SystemContext(primary_language=language, environment="dev"),
        )
        with patch(
            "remediation_engine.api.run_orchestrator",
            return_value={"status": "completed", "errors": []},
        ) as orchestrator:
            run_remediation(request, settings=Mock())
        assert orchestrator.call_args.kwargs["system_context"].primary_language == language


def test_triage_api_detects_from_repo_and_defaults_to_node_without_repo(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
    with patch("remediation_engine.api.run_triage_pipeline", return_value=[]) as pipeline:
        triage_issues([], repo_root=tmp_path, settings=Mock())
    assert pipeline.call_args.args[1].primary_language == ProjectLanguage.JAVA.value

    with patch("remediation_engine.api.run_triage_pipeline", return_value=[]) as pipeline:
        triage_issues([], settings=Mock())
    assert pipeline.call_args.args[1].primary_language == ProjectLanguage.NODEJS.value


def test_cli_language_option_is_limited_to_triage_and_run() -> None:
    parser = build_parser()
    assert parser.parse_args(["triage", "issues.jsonl", "--language", "java"]).language == "java"
    assert (
        parser.parse_args(["run", "issues.jsonl", "--repo", ".", "--language", "nodejs"]).language
        == "nodejs"
    )
    assert parser.parse_args(["triage", "issues.jsonl"]).language == "auto"
    assert parser.parse_args(["run", "issues.jsonl", "--repo", "."]).language == "auto"

    assert (
        parser.parse_args(["triage", "issues.jsonl", "--language", "python"]).language == "python"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["ingest", "issues.jsonl", "--language", "java"])


def test_cli_passes_language_choice_as_context_for_triage_and_run(tmp_path: Path) -> None:
    issues = tmp_path / "issues.jsonl"
    issue = VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.MEDIUM,
        rule_id="javascript.test",
        file_path="src/app.js",
        message="Test finding.",
    )
    issues.write_text(json.dumps(issue.model_dump(mode="json")) + "\n", encoding="utf-8")
    with (
        patch("remediation_engine.cli.load_dotenv"),
        patch("remediation_engine.cli.AppSettings.from_env", return_value=Mock()),
        patch("remediation_engine.cli.triage_issues", return_value=[]) as triage,
    ):
        assert main(["triage", str(issues), "--language", "java"]) == 0
    triage_context = triage.call_args.kwargs["system_context"]
    assert triage_context.primary_language == "java"
    assert triage_context.environment == "production"

    with (
        patch("remediation_engine.cli.load_dotenv"),
        patch("remediation_engine.cli.AppSettings.from_env", return_value=Mock()),
        patch(
            "remediation_engine.cli.run_remediation",
            return_value=RemediationResult(status="completed"),
        ) as run,
    ):
        assert main(["run", str(issues), "--repo", str(tmp_path), "--language", "java"]) == 0
    run_context = run.call_args.args[0].system_context
    assert run_context.primary_language == "java"
    assert run_context.environment == "production"


def test_orchestrator_and_direct_subagent_states_have_node_defaults(tmp_path: Path) -> None:
    state = initial_orchestrator_state(str(tmp_path), [], system_context=SystemContext())
    assert state["project_language"] is ProjectLanguage.NODEJS
    java_state = initial_orchestrator_state(
        str(tmp_path), [], system_context=SystemContext(primary_language="java")
    )
    assert java_state["project_language"] is ProjectLanguage.JAVA

    update = initial_update_subagent_state(
        repo_root=str(tmp_path),
        workspace_volume="workspace",
        target_tasks=[],
        target_groups=[],
    )
    assert update["project_language"] is ProjectLanguage.NODEJS

    group = VulnerabilityGroup(
        group_id="g",
        issue_type=IssueType.SAST,
        representative_issue_id=uuid4(),
    )
    task = RemediationTask(
        task_id="task-1",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
    )
    workaround = initial_workaround_subagent_state(
        repo_root=str(tmp_path),
        workspace_volume="workspace",
        target_task=task,
        target_group=group,
    )
    assert workaround["project_language"] is ProjectLanguage.NODEJS
    explicit_update = initial_update_subagent_state(
        repo_root=str(tmp_path),
        workspace_volume="workspace",
        target_tasks=[],
        target_groups=[],
        project_language=ProjectLanguage.JAVA,
    )
    assert explicit_update["project_language"] is ProjectLanguage.JAVA
    explicit_workaround = initial_workaround_subagent_state(
        repo_root=str(tmp_path),
        workspace_volume="workspace",
        target_task=task,
        target_group=group,
        project_language=ProjectLanguage.JAVA,
    )
    assert explicit_workaround["project_language"] is ProjectLanguage.JAVA


def test_python_no_fix_package_manager_is_inferred_from_supported_manifests() -> None:
    assert (
        _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["requirements-dev.txt"]) == "pip"
    )
    assert _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["Pipfile"]) == "pipenv"
    assert (
        _default_no_fix_package_manager(ProjectLanguage.PYTHON, ["requirements.txt", "Pipfile"])
        == ""
    )
    assert _default_no_fix_package_manager(ProjectLanguage.NODEJS, []) == "npm"


def test_qa_fallback_log_labels_do_not_assume_npm() -> None:
    results = _QAExecutionResults(
        install=(False, "pip install failed"),
        tests=(False, "pytest failed"),
    )

    install = _qa_log_records_for_phase(results, "install")
    tests = _qa_log_records_for_phase(results, "tests")

    assert install[0].label == "dependency installation"
    assert tests[0].label == "unit tests"


def test_python_workaround_prompt_reuses_the_shared_safety_contract() -> None:
    prompt = " ".join(_build_workaround_system_prompt(ProjectLanguage.PYTHON).split())

    assert "read_repository_map is paged by default" in prompt
    assert "Follow Investigate -> Plan -> Execute -> Validate" in prompt
    assert "CODE_FAILURE rolls back the whole pending set" in prompt
    assert "validate_python_workaround" in prompt
    assert "remove_no_fix_python_dependency" in prompt
    assert "validate_workaround" not in prompt
    assert "remove_no_fix_dependency" not in prompt
    assert "pytest" in prompt
