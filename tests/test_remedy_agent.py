"""
tests/test_remedy_agent.py - Unit tests for Phase 5 specialist subagent wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.graph import post_qa_triage_node
from remediation_engine.orchestration.state import (
    _derive_legacy_task_from_group,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from remediation_engine.orchestration.update_subagent import (
    _build_update_prompt,
    run_update_subagent_node,
)
from remediation_engine.orchestration.workaround_subagent import (
    _build_workaround_prompt,
    run_workaround_subagent_node,
)


def _sca_group(
    group_id: str = "sca:package.json:lodash", manifest_file: str = "package.json"
) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path=manifest_file,
    )
    from remediation_engine.contracts.schemas import LocalizedIssue

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

        assert "## Task " in prompt
        assert "Exact supervisor instruction:" in prompt
        assert "Supervisor's Revised Instruction" not in prompt
        assert "Why The Previous Attempt Failed" not in prompt
        assert "QA feedback:" in prompt
        assert "Previous outcome:" in prompt
        assert "4.17.22" in prompt
        assert "task instruction is authoritative" in prompt
        assert "execution worker" in prompt
        assert "First-pass mode:" not in prompt
        assert "First-pass planning questions:" not in prompt
        assert "Planning Answers" not in prompt
        assert "Reasoning Summary" not in prompt

    def test_mixed_first_pass_and_retry_batch_is_rejected_before_execution(self):
        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [group_a, group_b],
            [],
        )
        state["target_tasks"][0] = state["target_tasks"][0].model_copy(
            update={"retry_count": 1, "status": TaskStatus.NEEDS_RETRY}
        )

        with (
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                return_value=(["package.json"], []),
            ),
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI") as mock_chat,
        ):
            result = run_update_subagent_node(state)

        assert "mixed first-pass and retry" in result["errors"][-1]
        assert mock_chat.call_count == 0

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

        assert "execution worker" in prompt
        assert (
            'Exact supervisor instruction: Update "jsonwebtoken" in package.json to version "9.0.0".'
            in prompt
        )
        assert (
            'Exact supervisor instruction: Add or update "overrides": {"ws": "8.20.1"} in package.json.'
            in prompt
        )
        assert "QA feedback: Retry exact version bump from planner." in prompt
        assert (
            "QA feedback: Retry with npm overrides instead of a direct dependency edit." in prompt
        )
        assert "Previous outcome: Previous attempt hit an ERESOLVE conflict." in prompt
        assert "view_npm_package_versions" not in prompt
        assert "## Task task-1" in prompt
        assert "## Task task-2" in prompt
        assert "Planning Answers" not in prompt

    def test_success_requires_validation_after_manifest_edit(self):
        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        group_b.vulnerable_component = "axios"
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
                        "args": {"package_name": "lodash"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="updating second package",
                tool_calls=[
                    {
                        "name": "modify_npm_dependency",
                        "args": {
                            "package_name": "axios",
                            "target_version": "1.7.4",
                            "dependency_type": "dependencies",
                            "manifest_path": "frontend/package.json",
                        },
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="validating second package",
                tool_calls=[
                    {
                        "name": "validate_manifest_sync",
                        "args": {"package_name": "axios"},
                        "id": "call-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = (
            "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        )
        tool_validate = MagicMock()
        tool_validate.name = "validate_manifest_sync"
        tool_validate.invoke.return_value = (
            "SUCCESS: Manifest synchronization succeeded for package.json, frontend/package.json."
        )

        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                side_effect=[
                    (["package.json"], []),
                    (["frontend/package.json"], []),
                ],
            ),
            patch(
                "remediation_engine.orchestration.update_subagent.build_update_toolbelt",
                return_value=[tool_edit, tool_validate],
            ),
        ):
            result = run_update_subagent_node(state)

        assert bound.invoke.call_count == 5
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert "messages" not in result
        assert "package.json" in result["changed_files"]
        assert len(result["action_summaries"]) == 2

        summary_by_task = {
            summary.task_id: summary.summary for summary in result["action_summaries"]
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
        tool_edit.invoke.return_value = (
            "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        )

        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                return_value=(["package.json"], []),
            ),
            patch(
                "remediation_engine.orchestration.update_subagent.build_update_toolbelt",
                return_value=[tool_edit],
            ),
        ):
            result = run_update_subagent_node(state)

        assert result["action_summary"].status == AgentActionStatus.SURRENDER

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
        tool_edit = MagicMock()
        tool_edit.name = "modify_npm_dependency"
        tool_edit.invoke.return_value = (
            "SUCCESS: Natively updated dependencies.lodash to 4.17.21 in package.json."
        )
        tool_validate = MagicMock()
        tool_validate.name = "validate_manifest_sync"
        tool_validate.invoke.return_value = (
            "SUCCESS: Manifest synchronization succeeded for package.json."
        )

        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                return_value=(["package.json"], []),
            ),
            patch(
                "remediation_engine.orchestration.update_subagent.build_update_toolbelt",
                return_value=[tool_edit, tool_validate],
            ),
        ):
            result = run_update_subagent_node(state)

        diagnostics = result["retry_diagnostics_by_task"][group.group_id]
        assert diagnostics.reasoning_summary.startswith("Reasoning Summary")
        assert "Latest candidate was 4.17.21" in diagnostics.reasoning_summary

    def test_reverted_package_update_status_becomes_surrender(self):
        from remediation_engine.contracts.schemas import (
            AgentActionStatus,
            RemediationTask,
            RoutingStrategy,
        )
        from remediation_engine.orchestration.subagent_runtime import ToolEvent
        from remediation_engine.orchestration.update_subagent import _build_action_summaries

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

    def test_update_subagent_passes_override_required_packages_to_toolbelt(self):
        group = _sca_group()
        repo_root = _repo_root()
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [group],
            [],
            feedback_by_task={
                group.group_id: "Retry with npm overrides instead of a direct dependency edit.",
            },
        )
        state["target_tasks"][0].retry_count = 1
        state["target_tasks"][
            0
        ].instruction = 'Add or update "overrides": {"lodash": "4.17.22"} in package.json.'
        from remediation_engine.orchestration.update_subagent import UpdateRetryDiagnostics

        state["retry_diagnostics_by_task"] = {
            group.group_id: UpdateRetryDiagnostics(
                task_id=group.group_id,
                used_overrides=True,
            )
        }

        llm = MagicMock()
        sandbox = _sandbox_mock()
        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                return_value=(["package.json"], []),
            ),
            patch(
                "remediation_engine.orchestration.update_subagent.build_update_toolbelt",
                return_value=[],
            ) as mock_toolbelt,
            patch(
                "remediation_engine.orchestration.update_subagent.run_bounded_subagent_loop",
            ) as mock_loop,
        ):
            mock_loop.return_value = MagicMock(
                changed_files=[], tool_events=[], final_text="done", errors=[]
            )
            run_update_subagent_node(state)

            assert mock_toolbelt.call_count == 1
            kwargs = mock_toolbelt.call_args.kwargs
            assert kwargs["override_required_packages"] == {"lodash"}

    def test_build_action_summaries_marks_surrender_for_unmodified_package_even_when_batch_succeeded(
        self,
    ):
        from remediation_engine.contracts.schemas import AgentActionStatus
        from remediation_engine.orchestration.subagent_runtime import ToolEvent
        from remediation_engine.orchestration.update_subagent import _build_action_summaries

        group1 = _sca_group()
        group2 = _sca_group()
        group2.group_id = "sca:package.json:ws:UPDATE_VERSION"
        group2.vulnerable_component = "ws"
        tool_events = [
            ToolEvent(
                name="modify_npm_dependency",
                args={"package_name": "lodash", "target_version": "4.17.21"},
                content="SUCCESS",
            ),
            ToolEvent(
                name="validate_manifest_sync",
                args={},
                content="SUCCESS: Manifest synchronization succeeded for package.json.",
            ),
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

    def test_build_action_summaries_keeps_validated_package_successful_even_when_batch_fails(self):
        from remediation_engine.contracts.schemas import AgentActionStatus
        from remediation_engine.orchestration.subagent_runtime import ToolEvent
        from remediation_engine.orchestration.update_subagent import _build_action_summaries

        group1 = _sca_group()
        group2 = _sca_group()
        group2.group_id = "sca:package.json:ws:UPDATE_VERSION"
        group2.vulnerable_component = "ws"

        tool_events = [
            ToolEvent(
                name="modify_npm_dependency",
                args={
                    "package_name": "lodash",
                    "target_version": "4.17.21",
                    "manifest_path": "package.json",
                },
                content="SUCCESS",
            ),
            ToolEvent(
                name="validate_manifest_sync",
                args={},
                content="SUCCESS: Manifest synchronization succeeded for package.json.",
            ),
            ToolEvent(
                name="modify_npm_dependency",
                args={
                    "package_name": "ws",
                    "target_version": "8.20.1",
                    "manifest_path": "package.json",
                },
                content="SUCCESS",
            ),
            ToolEvent(
                name="revert_workspace_file",
                args={"file_path": "package.json", "package_name": "ws"},
                content="SUCCESS: Reverted package.json",
            ),
        ]

        summaries = _build_action_summaries(
            [
                (MagicMock(task_id=group1.group_id), group1, ["package.json"]),
                (MagicMock(task_id=group2.group_id), group2, ["package.json"]),
            ],
            changed_files=["package.json"],
            final_text="Updated lodash, reverted ws after dependency conflict",
            succeeded=False,
            tool_events=tool_events,
        )

        assert summaries[0].status == AgentActionStatus.SUCCESS
        assert summaries[1].status == AgentActionStatus.SURRENDER

    def test_workaround_prompt_includes_snippets(self):
        group = _sast_group()
        prompt = _build_workaround_prompt(
            group,
            ["express must remain >= 4.22.1"],
            previous_feedback="Keep the change narrow.",
        )

        assert "WORKAROUND SNIPPETS" in prompt
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
                content="searching",
                tool_calls=[
                    {
                        "name": "search_codebase_pattern",
                        "args": {"search_pattern": "unsafeRender"},
                        "id": "call-0",
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
                        "name": "validate_workaround",
                        "args": {
                            "modified_files": ["routes/login.ts"],
                            "runtime_smoke_file": "routes/login.ts",
                        },
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        tool_search = MagicMock()
        tool_search.name = "search_codebase_pattern"
        tool_search.invoke.return_value = "routes/login.ts:10: unsafeRender(input)"
        tool_edit = MagicMock()
        tool_edit.name = "deterministic_search_replace"
        tool_edit.invoke.return_value = "SUCCESS: File modified: routes/login.ts"
        tool_validate = MagicMock()
        tool_validate.name = "validate_workaround"
        tool_validate.invoke.return_value = (
            "SUCCESS: Workaround validation gate passed. Validated files: routes/login.ts.\n"
            'JSON: {"overall_status":"PASS","validated_files":["routes/login.ts"]}'
        )

        with (
            patch.dict(os.environ, {"REMEDY_BYPASS_WORKAROUND_SUBAGENT": "false"}),
            patch(
                "remediation_engine.orchestration.workaround_subagent.ChatOpenAI", return_value=llm
            ),
            patch(
                "remediation_engine.orchestration.workaround_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.workaround_subagent.build_workaround_toolbelt",
                return_value=[tool_search, tool_edit, tool_validate],
            ),
        ):
            result = run_workaround_subagent_node(state)

        assert bound.invoke.call_count == 4
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
                        "name": "validate_workaround",
                        "args": {"modified_files": ["routes/login.ts"]},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        )
        sandbox = _sandbox_mock()
        tool_validate = MagicMock()
        tool_validate.name = "validate_workaround"
        tool_validate.invoke.return_value = (
            "FAILURE: Workaround validation gate 'runtime_smoke' failed. Sandbox is not running, "
            "so validation cannot continue."
        )

        with (
            patch.dict(os.environ, {"REMEDY_BYPASS_WORKAROUND_SUBAGENT": "false"}),
            patch(
                "remediation_engine.orchestration.workaround_subagent.ChatOpenAI", return_value=llm
            ),
            patch(
                "remediation_engine.orchestration.workaround_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.workaround_subagent.build_workaround_toolbelt",
                return_value=[tool_validate],
            ),
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
        with patch.dict(os.environ, {"REMEDY_BYPASS_WORKAROUND_SUBAGENT": "true"}):
            result = run_workaround_subagent_node(state)
        assert result["action_summary"].status == AgentActionStatus.SURRENDER
        assert "Workaround subagent bypassed" in result["action_summary"].summary

    def test_post_qa_triage_env_var_disable(self):
        state = {"triage_required": True}
        with patch.dict(os.environ, {"REMEDY_DISABLE_POST_QA_TRIAGE": "true"}):
            res = post_qa_triage_node(state)
        assert res["status"] == "triage_skipped"

        with patch.dict(os.environ, {"REMEDY_DISABLE_RETRIAGE": "1"}):
            res = post_qa_triage_node(state)
        assert res["status"] == "triage_skipped"
