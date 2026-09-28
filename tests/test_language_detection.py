"""Project-language detection and CLI routing contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from remediation_engine.api import RemediationResult
from remediation_engine.cli import build_parser, main
from remediation_engine.contracts.schemas import SystemContext
from remediation_engine.language import (
    ProjectLanguage,
    detect_project_language,
    resolve_project_language,
)


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
