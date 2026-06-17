"""
tests/test_remedy_agent.py - Unit tests for Phase 5 specialist subagent wrappers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.contracts.schemas import (
    AgentActionStatus,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.state import (
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from src.orchestrator.update_subagent import run_update_subagent_node
from src.orchestrator.workaround_subagent import run_workaround_subagent_node


def _sca_group(group_id: str = "sca:package.json:lodash", manifest_file: str = "package.json") -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path=manifest_file,
    )
    from src.contracts.schemas import LocalizedIssue

    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path=manifest_file,
        file_paths=[manifest_file],
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[
            LocalizedIssue(
                issue=issue,
                manifest_file=manifest_file,
                localization_confidence=0.9,
            )
        ],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="4.17.21",
            instruction="Upgrade lodash to 4.17.21",
            strategy_used="osv_api",
        ),
    )


def _sast_group(group_id: str = "sast:routes/login.ts:javascript.xss") -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        rule_id="javascript.xss",
        file_path="routes/login.ts",
        message="Unsafe HTML rendering.",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SAST,
        vulnerable_component="javascript.xss",
        file_path="routes/login.ts",
        sources=[IssueSource.SEMGREP],
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.WORKAROUND_FOUND,
            workaround_snippets=["Escape the user input before rendering."],
            instruction="Apply an output-escaping workaround.",
            strategy_used="serper",
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


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


class TestUpdateSubagentWrapper:
    def test_success_requires_validation_after_manifest_edit(self):
        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:lodash", "frontend/package.json")
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group_a, group_b],
            ["lodash must remain >= 4.17.21"],
            {group_a.group_id: "Retry with an override if needed."},
        )

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="updating",
                tool_calls=[
                    {
                        "name": "modify_npm_dependency",
                        "args": {
                            "package_name": "lodash",
                            "target_version": "4.17.21",
                            "dependency_type": "dependencies",
                            "manifest_path": "package.json",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="validating",
                tool_calls=[
                    {
                        "name": "validate_manifest_sync",
                        "args": {},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        tool_validate = MagicMock()
        tool_validate.name = "validate_manifest_sync"
        tool_validate.invoke.return_value = "SUCCESS: Manifest synchronization succeeded for package.json, frontend/package.json."

        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            side_effect=[
                (["package.json"], []),
                (["frontend/package.json"], []),
            ],
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_edit, tool_validate],
        ):
            result = run_update_subagent_node(state)

        assert bound.invoke.call_count == 3
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert "messages" not in result
        assert "package.json" in result["changed_files"]

    def test_no_validation_success_becomes_surrender(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
        )

        llm, _bound = _mock_llm_with_responses(AIMessage(content="done"))
        sandbox = _sandbox_mock()
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."

        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_edit],
        ):
            result = run_update_subagent_node(state)

        assert result["action_summary"].status == AgentActionStatus.SURRENDER


class TestWorkaroundSubagentWrapper:
    def test_success_requires_validation_after_code_edit(self):
        group = _sast_group()
        repo_root = _repo_root()
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            group,
            ["express must remain >= 4.22.1"],
            previous_feedback="Fix the broken regex from the previous attempt.",
        )

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="editing",
                tool_calls=[
                    {
                        "name": "deterministic_search_replace",
                        "args": {
                            "file_path": "routes/login.ts",
                            "old_text": "unsafeRender(input)",
                            "new_text": "safeRender(input)",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="validating",
                tool_calls=[
                    {
                        "name": "validate_code_syntax",
                        "args": {"file_path": "routes/login.ts"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        tool_edit = MagicMock()
        tool_edit.name = "deterministic_search_replace"
        tool_edit.invoke.return_value = "SUCCESS: File modified: routes/login.ts"
        tool_validate = MagicMock()
        tool_validate.name = "validate_code_syntax"
        tool_validate.invoke.return_value = "SUCCESS: Syntax validation passed for routes/login.ts."

        with patch("src.orchestrator.workaround_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.workaround_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.workaround_subagent.build_workaround_toolbelt",
            return_value=[tool_edit, tool_validate],
        ):
            result = run_workaround_subagent_node(state)

        assert bound.invoke.call_count == 3
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert result["changed_files"] == ["routes/login.ts"]
        assert "messages" not in result

    def test_circuit_breaker_surfaces_as_surrender_with_errors(self):
        group = _sast_group()
        repo_root = _repo_root()
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            group,
            [],
        )

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="validating",
                tool_calls=[
                    {
                        "name": "validate_code_syntax",
                        "args": {"file_path": "routes/login.ts"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        )
        sandbox = _sandbox_mock()
        tool_validate = MagicMock()
        tool_validate.name = "validate_code_syntax"
        tool_validate.invoke.return_value = (
            "FAILURE: Sandbox is not running, so syntax validation cannot continue."
        )

        with patch("src.orchestrator.workaround_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.workaround_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.workaround_subagent.build_workaround_toolbelt",
            return_value=[tool_validate],
        ):
            result = run_workaround_subagent_node(state)

        assert bound.invoke.call_count == 1
        assert result["action_summary"].status == AgentActionStatus.SURRENDER
        assert any("Sandbox is not running," in err for err in result["errors"])
