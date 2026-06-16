"""
tests/test_remedy_agent.py - Unit tests for the Phase 5 Remedy Agent tool loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.contracts.schemas import (
    CommandResult,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.remedy_agent import _build_prompt, run_remedy_agent
from src.orchestrator.state import initial_orchestrator_state


def _sca_group(
    *,
    file_path: str | None = None,
    manifest_file: str | None = None,
    ghsa_id: str | None = None,
) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        ghsa_id=ghsa_id,
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
        ghsa_ids=[ghsa_id] if ghsa_id else [],
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
    sandbox._workspace_volume = "agent_workspace_deadbeef"
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


class TestRemedyAgentToolLoop:
    def test_successful_tool_loop_runs_validation_sequence(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies":{"lodash":"^4.17.15"}}\n',
            encoding="utf-8",
        )
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
            AIMessage(
                content="installing",
                tool_calls=[
                    {
                        "name": "run_dependency_install",
                        "args": {},
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="scanning",
                tool_calls=[
                    {
                        "name": "run_security_scan",
                        "args": {},
                        "id": "call-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="testing",
                tool_calls=[
                    {
                        "name": "run_unit_tests",
                        "args": {},
                        "id": "call-5",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )

        sandbox = _sandbox_mock()

        def _read_file(path: str) -> str:
            if path == "package.json":
                return '{"dependencies":{"lodash":"^4.17.15"}}\n'
            if path == "dependency-check-report.json":
                return json.dumps({"dependencies": []})
            raise AssertionError(f"unexpected path: {path}")

        sandbox.read_file.side_effect = _read_file
        sandbox.run.side_effect = [
            CommandResult(exit_code=0, stdout="installed", stderr="", duration_seconds=1.0),
            CommandResult(exit_code=0, stdout="passing", stderr="", duration_seconds=2.0),
        ]

        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.remedy_tools.shutil.which",
            return_value="docker",
        ), patch(
            "src.orchestrator.remedy_tools.subprocess.run",
            return_value=proc,
        ):
            result = run_remedy_agent(state)

        assert bound.invoke.call_count == 6
        sandbox.write_file.assert_called_once()
        sandbox.run.assert_any_call("npm install --package-lock=true", timeout=600)
        sandbox.run.assert_any_call("npm test", timeout=600)
        assert result["status"] == "edits_completed"
        assert result["changed_files"] == ["package.json"]
        assert len(result["messages"]) == 13

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

    def test_prompt_and_tool_factory_include_sequence_and_target_identifiers(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies":{"lodash":"^4.17.15"}}\n',
            encoding="utf-8",
        )
        group = _sca_group(
            manifest_file="package.json",
            ghsa_id="GHSA-VPQ2-C234-7XJ6",
        )
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        llm, bound = _mock_llm_with_responses(AIMessage(content="done"))
        sandbox = _sandbox_mock()
        fake_tool = MagicMock()
        fake_tool.name = "read_workspace_file"
        fake_tool.invoke.return_value = "unused"

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.remedy_agent.build_agent_tools",
            return_value=[fake_tool],
        ) as mock_build_tools:
            result = run_remedy_agent(state)

        prompt_message = bound.invoke.call_args_list[0].args[0][-1]
        prompt_text = prompt_message.content
        assert "REQUIRED STEP-BY-STEP SEQUENCE" in prompt_text
        assert "run_dependency_install" in prompt_text
        assert "run_security_scan" in prompt_text
        assert "run_unit_tests" in prompt_text
        assert "Never introduce trailing commas in package.json." in prompt_text
        assert "exactly 1 groups to fix" in prompt_text
        assert "modify_npm_dependency" in prompt_text
        assert "deterministic_search_replace" in prompt_text
        # Verify the codebase exploration funnel section is present
        assert "CODEBASE EXPLORATION FUNNEL" in prompt_text
        assert "read_repository_map" in prompt_text
        assert "search_codebase_pattern" in prompt_text
        assert "inspect_ast_symbol" in prompt_text
        # Step numbers updated
        assert "1. Map" in prompt_text
        assert "2. Inspect" in prompt_text

        build_args = mock_build_tools.call_args[0]
        assert build_args[2] == {"CVE-2021-44228", "GHSA-VPQ2-C234-7XJ6"}
        assert build_args[3] == Path(str(tmp_path))
        assert result["status"] == "no_changes_made"

    def test_circuit_breaker_stops_when_sandbox_is_not_running(self, tmp_path):
        (tmp_path / "package.json").write_text(
            '{"dependencies":{"lodash":"^4.17.15"}}\n',
            encoding="utf-8",
        )
        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["workspace_volume"] = "agent_workspace_deadbeef"

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="installing",
                tool_calls=[
                    {
                        "name": "run_dependency_install",
                        "args": {},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="this should never be reached"),
        )
        sandbox = _sandbox_mock()
        fake_tool = MagicMock()
        fake_tool.name = "run_dependency_install"
        fake_tool.invoke.return_value = (
            "FAILURE: Sandbox is not running, so npm install cannot continue."
        )

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.remedy_agent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.remedy_agent.build_agent_tools",
            return_value=[fake_tool],
        ):
            result = run_remedy_agent(state)

        assert bound.invoke.call_count == 1
        assert result["status"] == "remedy_failed"
        assert any("Sandbox is not running," in err for err in result["errors"])


class TestRemedyAgentPrompt:
    def test_target_files_are_deduplicated_in_prompt(self):
        group_a = _sca_group(manifest_file="package.json", ghsa_id="GHSA-VPQ2-C234-7XJ6")
        group_b = _sca_group(manifest_file="package.json", ghsa_id="GHSA-RVG8-PWQ2-XJ7Q")
        group_b.group_id = "sca:package.json:lodash-second"
        group_b.ghsa_ids = ["GHSA-RVG8-PWQ2-XJ7Q"]
        group_b.issues[0].ghsa_id = "GHSA-RVG8-PWQ2-XJ7Q"

        prompt = _build_prompt(
            resolved_groups=[(group_a, "package.json"), (group_b, "package.json")],
            repo_root="D:/repo",
        )

        target_files_section = prompt.split("Allowed target files:\n", 1)[1].split(
            "\n\n=== VULNERABILITY GROUP ===",
            1,
        )[0]
        assert target_files_section.count("- package.json") == 1
