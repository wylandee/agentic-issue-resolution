"""
tests/test_remedy_agent.py - Unit tests for the Phase 5 Remedy Agent tool loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.remedy_agent import run_remedy_agent
from src.orchestrator.state import initial_orchestrator_state


def _sca_group(*, file_path: str | None = None, manifest_file: str | None = None) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path=file_path,
    )
    localized_issues = []
    if manifest_file:
        from src.contracts.schemas import LocalizedIssue

        localized_issues.append(
            LocalizedIssue(
                issue=issue,
                manifest_file=manifest_file,
                localization_confidence=0.9,
            )
        )
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path=file_path,
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=localized_issues,
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="4.17.21",
            instruction="Upgrade lodash to 4.17.21",
            strategy_used="osv_api",
        ),
    )


def _mock_llm_with_responses(*responses):
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock()
    llm.bind_tools.return_value = bound
    return llm, bound


def _sandbox_mock():
    sandbox = MagicMock()
    sandbox.__enter__ = MagicMock(return_value=sandbox)
    sandbox.__exit__ = MagicMock(return_value=None)
    return sandbox


class TestRemedyAgentGuards:
    def test_missing_workspace_volume_returns_failed(self, tmp_path):
        group = _sca_group(manifest_file="package.json")
        (tmp_path / "package.json").write_text('"lodash": "^4.17.15"\n', encoding="utf-8")
        result = run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))
        assert result["status"] == "remedy_failed"
        assert any("workspace_volume" in err for err in result["errors"])

    def test_absolute_path_rejected(self, tmp_path):
        group = _sca_group(file_path=None, manifest_file=None)
        group.issues[0].file_path = "/etc/passwd"
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        with patch("src.orchestrator.remedy_agent.ChatOpenAI"):
            result = run_remedy_agent(state)

        assert result["status"] == "remedy_failed"
        assert any("absolute" in err.lower() for err in result["errors"])

    def test_retry_limit_uses_install_failures_feedback(self, tmp_path):
        (tmp_path / "package.json").write_text('"lodash": "^4.17.15"\n', encoding="utf-8")
        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["install_failures"] = "npm install failed"
        state["retry_count"] = 3
        state["max_retries"] = 3

        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cls:
            result = run_remedy_agent(state)

        mock_cls.assert_not_called()
        assert result["status"] == "max_retries_exceeded"


class TestRemedyAgentToolLoop:
    def test_successful_tool_loop_modifies_workspace_and_returns_new_messages(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies":{"lodash":"^4.17.15"}}\n', encoding="utf-8")
        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["messages"] = [HumanMessage(content="old conversation")]

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="reading",
                tool_calls=[
                    {
                        "name": "read_workspace_file",
                        "args": {"file_path": "package.json"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="editing",
                tool_calls=[
                    {
                        "name": "deterministic_search_replace",
                        "args": {
                            "file_path": "package.json",
                            "old_text": '"lodash":"^4.17.15"',
                            "new_text": '"lodash":"^4.17.21"',
                        },
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )

        sandbox = _sandbox_mock()
        sandbox.read_file.side_effect = [
            '{"dependencies":{"lodash":"^4.17.15"}}\n',
            '{"dependencies":{"lodash":"^4.17.15"}}\n',
        ]

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_remedy_agent(state)

        assert bound.invoke.call_count == 3
        sandbox.write_file.assert_called_once()
        assert result["status"] == "edits_completed"
        assert result["changed_files"] == ["package.json"]
        assert len(result["messages"]) == 7

    def test_no_tool_calls_returns_no_changes_made(self, tmp_path):
        (tmp_path / "package.json").write_text('"lodash": "^4.17.15"\n', encoding="utf-8")
        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        llm, _bound = _mock_llm_with_responses(AIMessage(content="No safe changes required."))
        sandbox = _sandbox_mock()

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_remedy_agent(state)

        assert result["status"] == "no_changes_made"
        assert result["changed_files"] == []

    def test_retry_feedback_increments_retry_count(self, tmp_path):
        (tmp_path / "package.json").write_text('"lodash": "^4.17.15"\n', encoding="utf-8")
        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"
        state["install_failures"] = "npm install failed"
        state["retry_count"] = 1

        llm, _bound = _mock_llm_with_responses(AIMessage(content="done"))
        sandbox = _sandbox_mock()

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ):
            result = run_remedy_agent(state)

        assert result["status"] == "no_changes_made"
        assert result["retry_count"] == 2
