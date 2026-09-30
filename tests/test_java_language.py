"""Offline contract tests for run-level project-language routing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from remediation_engine.api import (
    RemediationRequest,
    RemediationResult,
    run_remediation,
    triage_issues,
)
from remediation_engine.cli import build_parser, main
from remediation_engine.contracts.schemas import (
    IssueSource,
    IssueType,
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
from remediation_engine.orchestration.state import (
    initial_orchestrator_state,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)


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
    assert java.install_command == "mvn -B -q -DskipTests package"
    assert java.compile_command == "mvn -B -q -DskipTests compile"
    assert java.test_command == "mvn -B test -Dsurefire.useFile=false"
    assert java.test_include_patterns == (
        "**/Test*.java",
        "**/*Test.java",
        "**/*Tests.java",
        "**/*TestCase.java",
    )
    assert java.excluded_dirs == frozenset({"target"})


def test_detection_is_root_only_and_prioritizes_root_pom(tmp_path: Path) -> None:
    nested = tmp_path / "module"
    nested.mkdir()
    (nested / "pom.xml").write_text("<project />", encoding="utf-8")
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert detect_project_language(tmp_path) is ProjectLanguage.NODEJS

    (tmp_path / "pom.xml").write_text("<project />", encoding="utf-8")
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


@pytest.mark.parametrize("language", ["python", "kotlin", "", "maven"])
def test_unregistered_explicit_language_fails_closed(tmp_path: Path, language: str) -> None:
    with pytest.raises(ValueError, match="Unsupported project language"):
        resolve_project_language(tmp_path, language)


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


def test_api_keeps_explicit_java_context_and_rejects_unknown_language(tmp_path: Path) -> None:
    java_context = SystemContext(primary_language="java", environment="dev")
    request = RemediationRequest(repo_root=tmp_path, system_context=java_context)
    with patch(
        "remediation_engine.api.run_orchestrator",
        return_value={"status": "completed", "errors": []},
    ) as orchestrator:
        run_remediation(request, settings=Mock())
    assert orchestrator.call_args.kwargs["system_context"].primary_language == "java"

    invalid = RemediationRequest(
        repo_root=tmp_path,
        system_context=SystemContext(primary_language="python"),
    )
    with pytest.raises(ValueError, match="Unsupported project language"):
        run_remediation(invalid, settings=Mock())


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

    with pytest.raises(SystemExit):
        parser.parse_args(["triage", "issues.jsonl", "--language", "python"])
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
