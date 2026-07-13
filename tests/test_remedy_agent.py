"""
tests/test_remedy_agent.py - Unit tests for Phase 5 specialist subagent wrappers.
"""

from __future__ import annotations

import os
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
    _derive_legacy_task_from_group,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from src.orchestrator.update_subagent import (
    _build_retry_diagnostics,
    _build_update_prompt,
    _parse_registry_report_versions,
    run_update_subagent_node,
)
from src.orchestrator.workaround_subagent import _build_workaround_prompt, run_workaround_subagent_node
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
    def test_update_prompt_prioritizes_task_instruction_and_retry_context(self):
        group = _sca_group()
        task = _derive_legacy_task_from_group(group)
        task.instruction = 'Add or update "overrides": {"lodash": "4.17.22"} in package.json.'
        prompt = _build_update_prompt(
            [(task, group, ["package.json"])],
            ["lodash must remain >= 4.17.21"],
            {},
            {},
        )

        assert "## Task Context" in prompt
        assert "Supervisor's Revised Instruction" in prompt
        assert "Why The Previous Attempt Failed" not in prompt
        assert "QA Feedback:" not in prompt
        assert "Previous Worker Outcome:" not in prompt
        assert "4.17.22" in prompt
        assert "authoritative directive" in prompt
        assert "First-pass mode:" in prompt
        assert "First-pass planning questions:" not in prompt
        assert "Planning Answers" not in prompt
        assert "Reasoning Summary" not in prompt

    def test_update_prompt_shows_distinct_exact_instructions_for_multi_target_retry(self):
        group_a = _sca_group("sca:package.json:jsonwebtoken", "package.json")
        group_b = _sca_group("sca:frontend/package.json:ws", "frontend/package.json")
        task_a = _derive_legacy_task_from_group(group_a)
        task_b = _derive_legacy_task_from_group(group_b)
        task_a.task_id = "task-1"
        task_b.task_id = "task-2"
        task_a.retry_count = 1
        task_b.retry_count = 1
        task_a.instruction = 'Update "jsonwebtoken" in package.json to version "9.0.0".'
        task_b.instruction = 'Add or update "overrides": {"ws": "8.20.1"} in package.json.'
        prompt = _build_update_prompt(
            [
                (task_a, group_a, ["package.json"]),
                (task_b, group_b, ["frontend/package.json"]),
            ],
            [],
            {
                "task-1": "Retry exact version bump from planner.",
                "task-2": "Retry with npm overrides instead of a direct dependency edit.",
            },
            {
                "task-1": "Previous attempt hit an ERESOLVE conflict.",
                "task-2": "Previous attempt validated the wrong manifest path.",
            },
        )

        assert "Smart Planning & Rescue" in prompt
        assert 'Supervisor\'s Revised Instruction: Update "jsonwebtoken" in package.json to version "9.0.0".' in prompt
        assert 'Supervisor\'s Revised Instruction: Add or update "overrides": {"ws": "8.20.1"} in package.json.' in prompt
        assert "QA Feedback: Retry exact version bump from planner." in prompt
        assert "QA Feedback: Retry with npm overrides instead of a direct dependency edit." in prompt
        assert "Previous Worker Outcome: Previous attempt hit an ERESOLVE conflict." in prompt
        assert "view_npm_package_versions" in prompt
        assert "## Task Context" in prompt
        assert "## Prior Retry Diagnostics" in prompt
        assert "## Shared Planning Questions (for reference)" in prompt
        assert "What version or override path did the last fix attempt use?" in prompt
        assert "What is the latest version currently available according to npm?" in prompt
        assert "If the latest available version was already attempted and no untried candidates remain" in prompt
        assert "For each target below, the Task Instruction remains authoritative." in prompt
        assert "Planning Answers (for Task" in prompt

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
        assert len(result["action_summaries"]) == 2

        summary_by_task = {
            summary.task_id: summary.summary
            for summary in result["action_summaries"]
        }
        assert "frontend/package.json" not in summary_by_task[group_a.group_id]
        assert "package.json" in summary_by_task[group_a.group_id]
        assert "Final note:" not in summary_by_task[group_a.group_id]
        assert "frontend/package.json" in summary_by_task[group_b.group_id]

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

    def test_retry_success_without_registry_lookup_becomes_surrender(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
            previous_action_summaries_by_task={group.group_id: "Previous version bump failed validation."},
        )
        state["target_tasks"][0].retry_count = 1

        llm, _bound = _mock_llm_with_responses(
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
        tool_validate.invoke.return_value = "SUCCESS: Manifest synchronization succeeded for package.json."

        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_edit, tool_validate],
        ):
            result = run_update_subagent_node(state)

        assert result["action_summary"].status == AgentActionStatus.SURRENDER
        assert any("retry batch reported success without calling view_npm_package_versions" in err for err in result["errors"])

    def test_retry_diagnostics_capture_planning_answers(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
        )
        state["target_tasks"][0].retry_count = 1

        llm, _bound = _mock_llm_with_responses(
            AIMessage(
                content="registry lookup",
                tool_calls=[
                    {
                        "name": "view_npm_package_versions",
                        "args": {"package_name": "lodash"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
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
                        "id": "call-2",
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
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        tool_lookup = MagicMock()
        tool_lookup.name = "view_npm_package_versions"
        tool_lookup.invoke.return_value = (
            "# NPM Registry Report: lodash\n"
            "## dist-tags\n"
            "  latest: 4.17.21\n"
            "\n"
            "## Last 1 Published Versions (newest first)\n"
            "  4.17.21  (2024-01-01)\n"
        )
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        tool_validate = MagicMock()
        tool_validate.name = "validate_manifest_sync"
        tool_validate.invoke.return_value = "SUCCESS: Manifest synchronization succeeded for package.json."

        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_lookup, tool_edit, tool_validate],
        ):
            result = run_update_subagent_node(state)

        diagnostics = result["retry_diagnostics_by_task"][group.group_id]
        assert diagnostics.planning_answers["1_last_fix_attempt"] == "4.17.21"
        assert diagnostics.planning_answers["3_latest_version_available"] == "4.17.21"
        assert "4.17.21" in diagnostics.planning_answers["8_next_candidate"]

    def test_retry_diagnostics_capture_reasoning_summary(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
        )
        state["target_tasks"][0].retry_count = 1

        llm, _bound = _mock_llm_with_responses(
            AIMessage(
                content="registry lookup",
                tool_calls=[
                    {
                        "name": "view_npm_package_versions",
                        "args": {"package_name": "lodash"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
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
                        "id": "call-2",
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
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content=(
                    "Reasoning Summary\n"
                    "- Latest candidate was 4.17.21.\n"
                    "- No safer override was needed.\n"
                    "- Validation passed after the bump."
                )
            ),
        )
        sandbox = _sandbox_mock()
        tool_lookup = MagicMock()
        tool_lookup.name = "view_npm_package_versions"
        tool_lookup.invoke.return_value = (
            "# NPM Registry Report: lodash\n"
            "## dist-tags\n"
            "  latest: 4.17.21\n"
            "\n"
            "## Last 1 Published Versions (newest first)\n"
            "  4.17.21  (2024-01-01)\n"
        )
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        tool_validate = MagicMock()
        tool_validate.name = "validate_manifest_sync"
        tool_validate.invoke.return_value = "SUCCESS: Manifest synchronization succeeded for package.json."

        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_lookup, tool_edit, tool_validate],
        ):
            result = run_update_subagent_node(state)

        diagnostics = result["retry_diagnostics_by_task"][group.group_id]
        assert diagnostics.reasoning_summary.startswith("Reasoning Summary")
        assert "Latest candidate was 4.17.21" in diagnostics.reasoning_summary

    def test_parse_registry_report_versions_places_latest_first(self):
        report = (
            "## Dist-Tags\n"
            "  latest: 4.17.21\n\n"
            "## Last 3 Published Versions (newest first)\n"
            "  4.17.21  (2024-01-01)\n"
            "  4.17.20  (2023-01-01)\n"
            "  4.17.19  (2022-01-01)\n"
        )
        candidates, latest = _parse_registry_report_versions(report)
        assert latest == "4.17.21"
        assert candidates[0] == "4.17.21"

    def test_update_prompt_instructs_latest_first_rule(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
            retry_count=1,
        )
        prompt = _build_update_prompt([(task, group, ["package.json"])], [], {}, {})
        assert "LATEST-FIRST RULE:" in prompt
        assert "attempt the LATEST version found on the npm registry" in prompt

    def test_retry_diagnostics_prioritizes_latest_version_seen(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
        )
        tool_event = MagicMock()
        tool_event.name = "view_npm_package_versions"
        tool_event.args = {"package_name": "lodash"}
        tool_event.content = (
            "## Dist-Tags\n  latest: 4.17.21\n\n"
            "## Last 3 Published Versions (newest first)\n"
            "  4.17.21  (2024-01-01)\n  4.17.20  (2023-01-01)\n"
        )
        diagnostics = _build_retry_diagnostics(
            [(task, group, ["package.json"])],
            [tool_event],
            "Failed attempt",
            ["error"],
            False,
            {},
            [],
        )
        planning_answers = diagnostics[group.group_id].planning_answers
        assert planning_answers["8_next_candidate"] == "4.17.21"

    def test_update_subagent_sends_planning_answers_system_message(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
        )
        state = initial_update_subagent_state(
            repo_root=str(_repo_root()),
            workspace_volume="vol-123",
            target_tasks=[task],
            target_groups=[group],
            constraints_ledger=[],
        )
        llm = MagicMock()
        sandbox = _sandbox_mock()
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        
        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[tool_edit],
        ), patch(
            "src.orchestrator.update_subagent.run_bounded_subagent_loop",
        ) as mock_loop:
            mock_loop.return_value = MagicMock(changed_files=["package.json"], tool_events=[], final_text="done", errors=[])
            run_update_subagent_node(state)
            
            assert mock_loop.call_count == 1
            initial_msgs = mock_loop.call_args[0][2]
            assert "You MUST output your planning answers under the '## Planning Answers' section" in initial_msgs[0].content

    def test_update_prompt_enforces_package_scoped_revert_and_filters_attempted_latest(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        from src.orchestrator.update_subagent import UpdateRetryDiagnostics
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
            retry_count=1,
        )
        diagnostics = UpdateRetryDiagnostics(
            task_id=group.group_id,
            attempted_versions=["4.17.21"],
            latest_version_seen="4.17.21",
            candidate_versions_considered=["4.17.21", "4.17.20"],
            selected_version="4.17.21",
        )
        prompt = _build_update_prompt(
            [(task, group, ["package.json"])],
            [],
            {group.group_id: "SECURITY_FLAG: vulnerability remains"},
            {},
            {group.group_id: diagnostics},
        )
        assert "package_name='failing_package'" in prompt
        assert "If the latest version on the NPM registry was already attempted and still failed QA due to an unresolved vulnerability (`SECURITY_FLAG`)" in prompt
        assert "4.17.20" in prompt  # Next untried candidate is recommended instead of already attempted 4.17.21

    def test_reverted_package_update_status_becomes_surrender(self):
        from src.contracts.schemas import AgentActionStatus, RemediationTask, RoutingStrategy
        from src.orchestrator.subagent_runtime import ToolEvent
        from src.orchestrator.update_subagent import _build_action_summaries
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
        )
        tool_events = [
            ToolEvent(
                name="revert_workspace_file",
                args={"file_path": "package.json", "package_name": "lodash"},
                content="SUCCESS: Reverted package.json",
            )
        ]
        summaries = _build_action_summaries(
            [(task, group, ["package.json"])],
            ["package.json"],
            "reverted due to conflict",
            succeeded=True,
            tool_events=tool_events,
        )
        assert summaries[0].status == AgentActionStatus.SURRENDER
        assert "Stopped without a validated manifest update" in summaries[0].summary

    def test_update_subagent_passes_attempted_versions_and_planning_state_to_toolbelt(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
        )
        state["target_tasks"][0].retry_count = 1
        from src.orchestrator.update_subagent import UpdateRetryDiagnostics
        state["retry_diagnostics_by_task"] = {
            group.group_id: UpdateRetryDiagnostics(
                task_id=group.group_id,
                attempted_versions=["4.17.21"],
            )
        }

        llm = MagicMock()
        sandbox = _sandbox_mock()
        with patch("src.orchestrator.update_subagent.ChatOpenAI", return_value=llm), patch(
            "src.orchestrator.update_subagent.DockerSandbox",
            return_value=sandbox,
        ), patch(
            "src.orchestrator.update_subagent._resolve_manifest_targets",
            return_value=(["package.json"], []),
        ), patch(
            "src.orchestrator.update_subagent.build_update_toolbelt",
            return_value=[],
        ) as mock_toolbelt, patch(
            "src.orchestrator.update_subagent.run_bounded_subagent_loop",
        ) as mock_loop:
            mock_loop.return_value = MagicMock(changed_files=[], tool_events=[], final_text="done", errors=[])
            run_update_subagent_node(state)

            assert mock_toolbelt.call_count == 1
            kwargs = mock_toolbelt.call_args.kwargs
            assert kwargs["attempted_versions_by_package"] == {"lodash": {"4.17.21"}}
            assert kwargs["require_planning_answers"] is True

    def test_retry_diagnostics_marks_exhausted_update_path_when_latest_attempted_without_peer_conflict(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        from src.orchestrator.subagent_runtime import ToolEvent
        from src.orchestrator.update_subagent import _build_retry_diagnostics
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
            retry_count=1,
        )
        report = (
            "## Dist-Tags\n"
            "  latest: 4.17.21\n\n"
            "## Last 3 Published Versions (newest first)\n"
            "  4.17.21\n  4.17.20\n  4.17.19\n"
        )
        tool_events = [
            ToolEvent(
                name="view_npm_package_versions",
                args={"package_name": "lodash"},
                content=report,
            ),
            ToolEvent(
                name="modify_npm_dependency",
                args={"package_name": "lodash", "target_version": "4.17.21"},
                content="SUCCESS",
            ),
        ]
        diags = _build_retry_diagnostics(
            [(task, group, ["package.json"])],
            tool_events=tool_events,
            final_text="reverted",
            errors=[],
            succeeded=False,
            prior_diagnostics_by_task={},
            constraints_ledger=[],
        )
        assert diags[group.group_id].exhausted_update_path is True

    def test_build_action_summaries_marks_surrender_for_unmodified_package_even_when_batch_succeeded(self):
        from src.contracts.schemas import AgentActionStatus
        from src.orchestrator.subagent_runtime import ToolEvent
        from src.orchestrator.update_subagent import _build_action_summaries
        group1 = _sca_group()
        group2 = _sca_group()
        group2.group_id = "sca:package.json:ws:UPDATE_VERSION"
        group2.vulnerable_component = "ws"
        tool_events = [
            ToolEvent(
                name="modify_npm_dependency",
                args={"package_name": "lodash", "target_version": "4.17.21"},
                content="SUCCESS",
            )
        ]
        summaries = _build_action_summaries(
            [
                (MagicMock(task_id=group1.group_id), group1, ["package.json"]),
                (MagicMock(task_id=group2.group_id), group2, ["package.json"]),
            ],
            changed_files=["package.json"],
            final_text="Updated lodash, skipped ws",
            succeeded=True,
            tool_events=tool_events,
        )
        assert summaries[0].status == AgentActionStatus.SUCCESS
        assert summaries[1].status == AgentActionStatus.SURRENDER

    def test_retry_diagnostics_marks_exhausted_update_path_when_reasoning_summary_states_exhausted(self):
        from src.contracts.schemas import RemediationTask, RoutingStrategy
        from src.orchestrator.update_subagent import _build_retry_diagnostics
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            instruction="Bump version",
            retry_count=1,
        )
        diags = _build_retry_diagnostics(
            [(task, group, ["package.json"])],
            tool_events=[],
            final_text="## Reasoning Summary\nThe update path is exhausted because latest version was already attempted.",
            errors=[],
            succeeded=False,
            prior_diagnostics_by_task={},
            constraints_ledger=[],
        )
        assert diags[group.group_id].exhausted_update_path is True

    def test_retry_diagnostics_marks_exhausted_update_path_when_reverted_on_retry_or_latest_attempted(self):
        from src.orchestrator.subagent_runtime import ToolEvent
        from src.orchestrator.update_subagent import _build_retry_diagnostics
        group = _sca_group()
        task = MagicMock(task_id=group.group_id, retry_count=1)
        tool_events = [
            ToolEvent(
                name="revert_workspace_file",
                args={"package_name": "lodash"},
                content="SUCCESS: reverted lodash",
            )
        ]
        diags = _build_retry_diagnostics(
            [(task, group, ["package.json"])],
            tool_events=tool_events,
            final_text="Reverted lodash because latest version attempted",
            errors=[],
            succeeded=False,
            prior_diagnostics_by_task={},
            constraints_ledger=[],
        )
        assert diags[group.group_id].exhausted_update_path is True


class TestWorkaroundSubagentWrapper:
    def test_workaround_prompt_includes_snippets(self):
        group = _sast_group()
        prompt = _build_workaround_prompt(
            group,
            ["express must remain >= 4.22.1"],
            previous_feedback="Keep the change narrow.",
        )

        assert "Workaround snippets:" in prompt
        assert "Escape the user input before rendering." in prompt
        assert "Keep the change narrow." in prompt

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

        with patch.dict(os.environ, {"REMEDY_BYPASS_WORKAROUND_SUBAGENT": "false"}), patch(
            "src.orchestrator.workaround_subagent.ChatOpenAI", return_value=llm
        ), patch(
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

        with patch.dict(os.environ, {"REMEDY_BYPASS_WORKAROUND_SUBAGENT": "false"}), patch(
            "src.orchestrator.workaround_subagent.ChatOpenAI", return_value=llm
        ), patch(
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

    def test_workaround_subagent_bypass_mode(self):
        group = _sast_group()
        repo_root = _repo_root()
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            group,
            [],
        )
        result = run_workaround_subagent_node(state)
        assert result["action_summary"].status == AgentActionStatus.SURRENDER
        assert "Workaround subagent bypassed" in result["action_summary"].summary
