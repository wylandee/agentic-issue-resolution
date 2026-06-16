from __future__ import annotations

from uuid import UUID, uuid4
from unittest.mock import MagicMock, patch

from src.contracts.schemas import IssueSource, IssueType, Severity, VulnerabilityGroup, VulnerabilityIssue
from src.orchestrator.langsmith_config import build_phase5_runnable_config, resolve_phase5_trace_url
from src.orchestrator.remedy_agent import _MAX_TOOL_CALL_ROUNDS


def _group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        id=str(uuid4()),
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )
    return VulnerabilityGroup(
        group_id=str(uuid4()),
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path="package.json",
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[issue.source],
        representative_issue_id=issue.id,
        issues=[issue],
    )


class TestPhase5LangsmithConfig:
    def test_build_phase5_runnable_config_returns_none_when_tracing_disabled(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "false")

        config, run_id = build_phase5_runnable_config("D:/repos/juice-shop", [_group()])

        assert config is None
        assert run_id is None

    def test_build_phase5_runnable_config_includes_expected_metadata(self, monkeypatch):
        monkeypatch.setenv("LANGSMITH_TRACING", "true")

        groups = [_group(), _group()]
        config, run_id = build_phase5_runnable_config("D:/repos/juice-shop", groups)

        assert config is not None
        assert isinstance(run_id, UUID)
        assert config["run_id"] == run_id
        assert config["run_name"] == "phase5_orchestrator"
        assert config["tags"] == ["phase-5", "orchestrator", "langgraph"]
        assert config["metadata"] == {
            "repo_name": "juice-shop",
            "repo_root": "D:/repos/juice-shop",
            "vulnerability_group_count": 2,
            "max_tool_call_rounds": _MAX_TOOL_CALL_ROUNDS,
        }


class TestPhase5TraceUrlResolution:
    @patch("src.orchestrator.langsmith_config.wait_for_all_tracers")
    @patch("src.orchestrator.langsmith_config.Client")
    def test_resolve_phase5_trace_url_returns_url(self, mock_client_cls, mock_wait, monkeypatch):
        monkeypatch.setenv("LANGSMITH_PROJECT", "AppSec-Remediation-Engine")
        run_id = uuid4()
        run = MagicMock()
        client = mock_client_cls.return_value
        client.read_run.return_value = run
        client.get_run_url.return_value = "https://smith.langchain.com/o/test/projects/p/runs/r"

        result = resolve_phase5_trace_url(run_id)

        assert result == "https://smith.langchain.com/o/test/projects/p/runs/r"
        mock_wait.assert_called_once_with()
        client.read_run.assert_called_once_with(run_id)
        client.get_run_url.assert_called_once_with(
            run=run,
            project_name="AppSec-Remediation-Engine",
        )

    @patch("src.orchestrator.langsmith_config.wait_for_all_tracers")
    @patch("src.orchestrator.langsmith_config.Client")
    def test_resolve_phase5_trace_url_returns_none_when_lookup_fails(
        self,
        mock_client_cls,
        mock_wait,
    ):
        run_id = uuid4()
        client = mock_client_cls.return_value
        client.read_run.side_effect = RuntimeError("boom")

        result = resolve_phase5_trace_url(run_id)

        assert result is None
        mock_wait.assert_called_once_with()

    @patch("src.orchestrator.langsmith_config.wait_for_all_tracers")
    @patch("src.orchestrator.langsmith_config.Client")
    def test_resolve_phase5_trace_url_returns_none_when_url_build_fails(
        self,
        mock_client_cls,
        mock_wait,
    ):
        run_id = uuid4()
        run = MagicMock()
        client = mock_client_cls.return_value
        client.read_run.return_value = run
        client.get_run_url.side_effect = RuntimeError("boom")

        result = resolve_phase5_trace_url(run_id)

        assert result is None
        mock_wait.assert_called_once_with()
