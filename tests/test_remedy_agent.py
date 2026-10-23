"""
tests/test_remedy_agent.py - Unit tests for Phase 5 specialist subagent wrappers.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from langchain_core.messages import AIMessage

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    MavenTargetOperation,
    MultiPackageAction,
    NoFixMitigationStage,
    PackageMutation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TacticalStrategy,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.graph import post_qa_triage_node
from remediation_engine.orchestration.state import (
    initial_update_subagent_state,
    initial_workaround_subagent_state,
)
from remediation_engine.orchestration.update_subagent import (
    _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS,
    _UPDATE_WORKER_STATIC_INSTRUCTIONS,
    _build_update_prompt,
    _changed_files_by_task,
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


def _maven_group(
    coordinate: str = "org.example:library",
    manifest_path: str = "modules/service/pom.xml",
) -> VulnerabilityGroup:
    group_id, artifact_id = coordinate.split(":", 1)
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2026-0002",
        package_name=coordinate,
        package_version="1.0.0",
        purl=f"pkg:maven/{group_id}/{artifact_id}@1.0.0",
        ecosystem="Maven",
        file_path=manifest_path,
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file=manifest_path,
        package_manager="maven",
        is_direct_dependency=True,
        declaration_type="dependencies",
        version_property_name="library.version",
        version_property_file="pom.xml",
        localization_confidence=1.0,
    )
    return VulnerabilityGroup(
        group_id=f"sca:{manifest_path}:{coordinate}",
        issue_type=IssueType.SCA,
        vulnerable_component=coordinate,
        file_path=manifest_path,
        file_paths=[manifest_path],
        cve_ids=["CVE-2026-0002"],
        versions=["1.0.0"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="2.4.1",
            instruction="Update org.example:library to the selected version.",
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


def _task_for_group(group: VulnerabilityGroup, **overrides) -> RemediationTask:
    strategy = (
        RoutingStrategy.VERSION_BUMP
        if group.issue_type == IssueType.SCA
        and (group.fix_plan is None or group.fix_plan.status == FixPlanStatus.VERSION_FOUND)
        else RoutingStrategy.CODE_WORKAROUND
    )
    qa_policy = (
        QAPolicy.NO_FIX_PACKAGE_REMOVAL
        if group.fix_plan is not None and group.fix_plan.status == FixPlanStatus.NO_FIX
        else (
            QAPolicy.VERSION_BUMP
            if strategy == RoutingStrategy.VERSION_BUMP
            else QAPolicy.INITIAL_CODE_WORKAROUND
        )
    )
    kwargs = {
        "task_id": group.group_id,
        "parent_group_id": group.group_id,
        "strategy": strategy,
        "qa_policy": qa_policy,
        "instruction": getattr(group.fix_plan, "instruction", "") if group.fix_plan else "",
    }
    kwargs.update(overrides)
    return RemediationTask(**kwargs)


class TestUpdateSubagentWrapper:
    def test_update_prompt_prioritizes_task_instruction_and_retry_context(self):
        group = _sca_group()
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            qa_policy=QAPolicy.VERSION_BUMP,
            instruction='Add or update "overrides": {"lodash": "4.17.22"} in package.json.',
        )
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
        assert (
            "Execute only the Supervisor's task instruction" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
        )
        assert "First-pass mode:" not in prompt
        assert "First-pass planning questions:" not in prompt
        assert "Planning Answers" not in prompt
        assert "Reasoning Summary" not in prompt
        assert "Deterministic repository map:" in prompt
        assert "read_repository_map" not in prompt
        assert "revert_workspace_file" not in prompt
        assert "validate_manifest_sync" not in prompt

    def test_major_upgrade_prompt_commits_source_migration_and_version(self):
        group = _sca_group().model_copy(update={"vulnerable_component": "jsonwebtoken"})
        directive = (
            "Migrate all affected production and test code to the selected package API "
            "while preserving behavior; do not change the solver-approved package or version."
        )
        task = _task_for_group(
            group,
            target_package_name="jsonwebtoken",
            target_dependency_type="dependencies",
            selected_version="9.0.2",
            instruction=(
                "Upgrade package jsonwebtoken from installed version 8.5.1 "
                f"to selected version 9.0.2.\n{directive}"
            ),
        )

        prompt = _build_update_prompt(
            [(task, group, ["package.json"])],
            [],
            {},
            {},
        )

        assert "8.5.1" in prompt
        assert "9.0.2" in prompt
        assert directive in prompt
        assert "Major-version source/test migration: required." in prompt
        assert "source and test migration to the committed package API is authorized" in (
            _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS
        )
        assert "Do not reselect or change any package name or version" in (
            _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS
        )

    def test_mixed_first_pass_and_retry_batch_is_rejected_before_execution(self):
        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        task_a = _task_for_group(group_a)
        task_b = _task_for_group(group_b)
        state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [task_a, task_b],
            [group_a, group_b],
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

    def test_multi_task_changed_file_projection_keeps_shared_lockfile(self):
        group_a = _sca_group("sca:frontend/package.json:lodash", "frontend/package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        task_a = _task_for_group(group_a)
        task_b = _task_for_group(group_b)

        projected = _changed_files_by_task(
            [
                (task_a, group_a, ["frontend/package.json"]),
                (task_b, group_b, ["frontend/package.json"]),
            ],
            ["frontend/package-lock.json", "frontend/package.json"],
        )

        expected = ["frontend/package-lock.json", "frontend/package.json"]
        assert projected == {task_a.task_id: expected, task_b.task_id: expected}

    def test_update_prompt_shows_distinct_exact_instructions_for_multi_target_retry(self):
        group_a = _sca_group("sca:package.json:jsonwebtoken", "package.json")
        group_b = _sca_group("sca:frontend/package.json:ws", "frontend/package.json")
        task_a = RemediationTask(
            task_id="task-1",
            parent_group_id=group_a.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            qa_policy=QAPolicy.VERSION_BUMP,
            retry_count=1,
            instruction='Update "jsonwebtoken" in package.json to version "9.0.0".',
        )
        task_b = RemediationTask(
            task_id="task-2",
            parent_group_id=group_b.group_id,
            strategy=RoutingStrategy.VERSION_BUMP,
            qa_policy=QAPolicy.VERSION_BUMP,
            retry_count=1,
            instruction='Add or update "overrides": {"ws": "8.20.1"} in package.json.',
        )
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
        assert "execution worker" in _UPDATE_WORKER_STATIC_INSTRUCTIONS
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
        assert "## Task task-1" in prompt
        assert "## Task task-2" in prompt
        assert "Planning Answers" not in prompt

    def test_success_requires_combined_manifest_transaction(self):
        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        group_b.vulnerable_component = "axios"
        repo_root = _repo_root()
        task_a = _task_for_group(group_a)
        task_b = _task_for_group(group_b)
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [task_a, task_b],
            [group_a, group_b],
            constraints_ledger=["lodash must remain >= 4.17.21"],
            feedback_by_task={group_a.group_id: "Retry with an override if needed."},
        )

        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="updating",
                tool_calls=[
                    {
                        "name": "modify_and_validate_npm_dependency",
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
                content="updating second package",
                tool_calls=[
                    {
                        "name": "modify_and_validate_npm_dependency",
                        "args": {
                            "package_name": "axios",
                            "target_version": "1.7.4",
                            "dependency_type": "dependencies",
                            "manifest_path": "frontend/package.json",
                        },
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        combined_tool = MagicMock()
        combined_tool.name = "modify_and_validate_npm_dependency"
        combined_tool.invoke.side_effect = [
            "SUCCESS: Natively updated and synchronized dependencies.lodash to 4.17.21 in package.json.",
            "SUCCESS: Natively updated and synchronized dependencies.axios to 1.7.4 in frontend/package.json.",
        ]

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
                return_value=[combined_tool],
            ),
        ):
            result = run_update_subagent_node(state)

        assert bound.invoke.call_count == 3
        messages = bound.invoke.call_args_list[0].args[0]
        assert "dependency-manifest execution worker" in messages[0].content
        assert "DYNAMIC TASK CONTEXT" in messages[-1].content
        assert "package.json" in messages[-1].content
        assert "package.json" not in _UPDATE_WORKER_STATIC_INSTRUCTIONS
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

    def test_multi_package_cluster_uses_llm_committed_action_tool(self):
        from remediation_engine.orchestration.supervisor_planner import instruction_digest

        group_a = _sca_group("sca:package.json:lodash", "package.json")
        group_b = _sca_group("sca:frontend/package.json:axios", "frontend/package.json")
        group_b.vulnerable_component = "axios"
        task_a = _task_for_group(group_a).model_copy(
            update={
                "selected_version": "4.17.21",
                "target_package_name": "lodash",
                "target_dependency_type": "dependencies",
                "current_attempt_id": "attempt-1",
            }
        )
        task_b = _task_for_group(group_b).model_copy(
            update={
                "selected_version": "1.7.4",
                "target_package_name": "axios",
                "target_dependency_type": "dependencies",
                "current_attempt_id": "attempt-2",
            }
        )
        action = MultiPackageAction(
            cluster_id="cluster-1",
            dispatch_batch_id="batch-1",
            selected_strategy=TacticalStrategy.VERSION_BUMP,
            package_mutations=[
                PackageMutation(
                    task_id=task_a.task_id,
                    package_name="lodash",
                    manifest_path="package.json",
                    target_version="4.17.21",
                    dependency_type="dependencies",
                ),
                PackageMutation(
                    task_id=task_b.task_id,
                    package_name="axios",
                    manifest_path="frontend/package.json",
                    target_version="1.7.4",
                    dependency_type="dependencies",
                ),
            ],
            rationale="Keep the package cluster atomic.",
        )
        action_digest = instruction_digest(action.model_dump_json())
        snapshots = {
            task_a.task_id: TaskAttemptSnapshot(
                attempt_id="attempt-1",
                task_id=task_a.task_id,
                cluster_id="cluster-1",
                dispatch_batch_id="batch-1",
                action_digest=action_digest,
                manifest_path="package.json",
                selected_version="4.17.21",
                allowed_target_versions=["4.17.21"],
                target_package_name="lodash",
                target_dependency_type="dependencies",
                allowed_dependency_types=["dependencies"],
                instruction=task_a.instruction,
                instruction_digest=instruction_digest(task_a.instruction),
                dispatch_node="update_subagent",
            ),
            task_b.task_id: TaskAttemptSnapshot(
                attempt_id="attempt-2",
                task_id=task_b.task_id,
                cluster_id="cluster-1",
                dispatch_batch_id="batch-1",
                action_digest=action_digest,
                manifest_path="frontend/package.json",
                selected_version="1.7.4",
                allowed_target_versions=["1.7.4"],
                target_package_name="axios",
                target_dependency_type="dependencies",
                allowed_dependency_types=["dependencies"],
                instruction=task_b.instruction,
                instruction_digest=instruction_digest(task_b.instruction),
                dispatch_node="update_subagent",
            ),
        }
        state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [task_a, task_b],
            [group_a, group_b],
            target_attempt_snapshots=snapshots,
            active_cluster_id="cluster-1",
            dispatch_batch_id="batch-1",
            multi_package_action=action,
        )
        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="execute the committed action",
                tool_calls=[
                    {
                        "name": "modify_batch_npm_dependencies",
                        "args": {},
                        "id": "cluster-call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="cluster action completed"),
        )
        sandbox = _sandbox_mock()

        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ),
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                side_effect=[(["package.json"], []), (["frontend/package.json"], [])],
            ),
            patch(
                "remediation_engine.orchestration.tools_manifest.apply_multi_package_action",
                return_value=(True, ""),
            ) as apply_action,
        ):
            result = run_update_subagent_node(state)

        assert bound.invoke.call_count == 2
        bound_tools = llm.bind_tools.call_args.args[0]
        assert [tool.name for tool in bound_tools] == ["modify_batch_npm_dependencies"]
        first_prompt = bound.invoke.call_args_list[0].args[0]
        assert "MULTI-PACKAGE CLUSTER EXECUTION" in first_prompt[-1].content
        assert "modify_batch_npm_dependencies" in first_prompt[-1].content
        assert '"package_name":"lodash"' in first_prompt[-1].content
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert len(result["worker_results_by_attempt"]) == 2
        apply_action.assert_called_once()
        assert apply_action.call_args.args[1] == action

    def test_maven_multi_package_worker_uses_native_tool_prompt_and_cache(self):
        from remediation_engine.contracts.schemas import TaskAttemptSnapshot
        from remediation_engine.language import LANGUAGE_CONFIGS
        from remediation_engine.orchestration.supervisor_planner import instruction_digest

        group_a = _maven_group()
        group_b = _maven_group(
            coordinate="org.example:widget",
            manifest_path="modules/widget/pom.xml",
        )
        task_a = _task_for_group(
            group_a,
            task_id="task-library",
            selected_version="2.4.1",
            target_package_name="org.example:library",
            target_dependency_type="dependencies",
            current_attempt_id="attempt-library",
        )
        task_b = _task_for_group(
            group_b,
            task_id="task-widget",
            selected_version="3.1.0",
            target_package_name="org.example:widget",
            target_dependency_type="dependencies",
            current_attempt_id="attempt-widget",
        )
        reactor_poms = ["pom.xml", "modules/service/pom.xml", "modules/widget/pom.xml"]
        action = MultiPackageAction(
            cluster_id="cluster-java",
            dispatch_batch_id="batch-java",
            selected_strategy=TacticalStrategy.VERSION_BUMP,
            package_mutations=[
                PackageMutation(
                    task_id=task_a.task_id,
                    package_name="org.example:library",
                    ecosystem="maven",
                    manifest_path="modules/service/pom.xml",
                    target_manifest_paths=reactor_poms,
                    maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
                    target_version="2.4.1",
                    dependency_type="dependencies",
                ),
                PackageMutation(
                    task_id=task_b.task_id,
                    package_name="org.example:widget",
                    ecosystem="maven",
                    manifest_path="modules/widget/pom.xml",
                    target_manifest_paths=reactor_poms,
                    maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
                    target_version="3.1.0",
                    dependency_type="dependencies",
                ),
            ],
            rationale="Update the committed Maven reactor cluster atomically.",
        )
        action_digest = instruction_digest(action.model_dump_json())
        snapshots = {
            task.task_id: TaskAttemptSnapshot(
                attempt_id=task.current_attempt_id,
                task_id=task.task_id,
                cluster_id="cluster-java",
                dispatch_batch_id="batch-java",
                action_digest=action_digest,
                selected_version=task.selected_version,
                allowed_target_versions=[task.selected_version],
                target_package_name=task.target_package_name,
                target_dependency_type=task.target_dependency_type,
                maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
                target_manifest_paths=reactor_poms,
                allowed_dependency_types=["dependencies"],
                instruction=task.instruction,
                instruction_digest=instruction_digest(task.instruction),
                dispatch_node="update_subagent",
            )
            for task in (task_a, task_b)
        }
        state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [task_a.model_copy(update={"selected_version": "2.5.0"}), task_b],
            [group_a, group_b],
            target_attempt_snapshots=snapshots,
            active_cluster_id="cluster-java",
            dispatch_batch_id="batch-java",
            multi_package_action=action,
            project_language=ProjectLanguage.JAVA,
            maven_cache_volume="agent_maven_repository_deadbeef",
        )
        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="execute the committed Maven action",
                tool_calls=[
                    {
                        "name": "modify_batch_maven_dependencies",
                        "args": {},
                        "id": "maven-cluster-call",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Maven cluster action completed"),
        )
        sandbox = _sandbox_mock()

        with (
            patch("remediation_engine.orchestration.update_subagent.ChatOpenAI", return_value=llm),
            patch(
                "remediation_engine.orchestration.update_subagent.DockerSandbox",
                return_value=sandbox,
            ) as sandbox_factory,
            patch(
                "remediation_engine.orchestration.update_subagent._resolve_manifest_targets",
                side_effect=[(reactor_poms, []), (reactor_poms, [])],
            ),
            patch(
                "remediation_engine.orchestration.update_subagent.build_repository_map",
                return_value="Maven repository map",
            ),
            patch(
                "remediation_engine.orchestration.tools_manifest_java.apply_multi_maven_package_action",
                return_value=(True, ""),
            ) as apply_action,
        ):
            result = run_update_subagent_node(state)

        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert [tool.name for tool in llm.bind_tools.call_args.args[0]] == [
            "modify_batch_maven_dependencies"
        ]
        assert (
            "modify_batch_maven_dependencies" in bound.invoke.call_args_list[0].args[0][0].content
        )
        prompt = "\n".join(
            str(message.content) for message in bound.invoke.call_args_list[0].args[0]
        )
        assert "2.4.1" in prompt
        assert "2.5.0" not in prompt
        assert "npm" not in bound.invoke.call_args_list[0].args[0][0].content.lower()
        assert "authorized and required" in bound.invoke.call_args_list[0].args[0][0].content
        assert sandbox_factory.call_args.kwargs == {
            "repo_root": None,
            "workspace_volume": "agent_workspace_deadbeef",
            "image": LANGUAGE_CONFIGS[ProjectLanguage.JAVA].docker_image,
            "maven_repository_volume": "agent_maven_repository_deadbeef",
        }
        apply_action.assert_called_once()
        assert apply_action.call_args.args[1] == action

    def test_no_validation_success_becomes_surrender(self):
        group = _sca_group()
        repo_root = _repo_root()
        task = _task_for_group(group)
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [task],
            [group],
        )

        llm, _bound = _mock_llm_with_responses(AIMessage(content="done"))
        sandbox = _sandbox_mock()
        combined_tool = MagicMock()
        combined_tool.name = "modify_and_validate_npm_dependency"
        combined_tool.invoke.return_value = "SUCCESS: Natively updated and synchronized dependencies.lodash to 4.17.21 in package.json."

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
                return_value=[combined_tool],
            ),
        ):
            result = run_update_subagent_node(state)

        assert result["action_summary"].status == AgentActionStatus.SURRENDER

    def test_retry_diagnostics_capture_reasoning_summary(self):
        group = _sca_group()
        repo_root = _repo_root()
        task = _task_for_group(group, retry_count=1)
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [task],
            [group],
        )

        llm, _bound = _mock_llm_with_responses(
            AIMessage(
                content="updating",
                tool_calls=[
                    {
                        "name": "modify_and_validate_npm_dependency",
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
                content=(
                    "Reasoning Summary\n"
                    "- Latest candidate was 4.17.21.\n"
                    "- No safer override was needed.\n"
                    "- Validation passed after the bump."
                )
            ),
        )
        sandbox = _sandbox_mock()
        combined_tool = MagicMock()
        combined_tool.name = "modify_and_validate_npm_dependency"
        combined_tool.invoke.return_value = "SUCCESS: Natively updated and synchronized dependencies.lodash to 4.17.21 in package.json."

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
                return_value=[combined_tool],
            ),
        ):
            result = run_update_subagent_node(state)

        diagnostics = result["retry_diagnostics_by_task"][group.group_id]
        assert diagnostics.reasoning_summary.startswith("Reasoning Summary")
        assert "Latest candidate was 4.17.21" in diagnostics.reasoning_summary

    def test_reverted_package_update_status_becomes_surrender(self):
        from remediation_engine.contracts.schemas import (
            AgentActionStatus,
            QAPolicy,
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
            qa_policy=QAPolicy.VERSION_BUMP,
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
        task = _task_for_group(
            group,
            retry_count=1,
            instruction='Add or update "overrides": {"lodash": "4.17.22"} in package.json.',
        )
        state = initial_update_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            [task],
            [group],
            feedback_by_task={
                group.group_id: "Retry with npm overrides instead of a direct dependency edit.",
            },
        )
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
        group2.group_id = "sca:package.json:ws"
        group2.vulnerable_component = "ws"
        tool_events = [
            ToolEvent(
                name="modify_and_validate_npm_dependency",
                args={
                    "package_name": "lodash",
                    "target_version": "4.17.21",
                    "dependency_type": "dependencies",
                    "manifest_path": "package.json",
                },
                content="SUCCESS: Updated and synchronized lodash.",
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
        group2.group_id = "sca:package.json:ws"
        group2.vulnerable_component = "ws"

        tool_events = [
            ToolEvent(
                name="modify_and_validate_npm_dependency",
                args={
                    "package_name": "lodash",
                    "target_version": "4.17.21",
                    "dependency_type": "dependencies",
                    "manifest_path": "package.json",
                },
                content="SUCCESS: Updated and synchronized lodash.",
            ),
            ToolEvent(
                name="modify_and_validate_npm_dependency",
                args={
                    "package_name": "ws",
                    "target_version": "8.20.1",
                    "dependency_type": "dependencies",
                    "manifest_path": "package.json",
                },
                content="ERROR_CODE: MANIFEST_SYNC_FAILED: rolled back ws.",
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
        task = _task_for_group(group)
        prompt = _build_workaround_prompt(
            task,
            group,
            ["express must remain >= 4.22.1"],
            previous_feedback="Keep the change narrow.",
        )

        assert "WORKAROUND SNIPPETS" in prompt
        assert "Escape the user input before rendering." in prompt
        assert "Keep the change narrow." in prompt

    def test_maven_no_fix_prompt_names_authorized_poms_without_lockfile_language(self):
        group = _maven_group()
        group.fix_plan = FixPlan(
            status=FixPlanStatus.NO_FIX,
            instruction="Remove the committed Maven dependency if it is direct.",
            strategy_used="test",
        )
        task = RemediationTask(
            task_id=group.group_id,
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            qa_policy=QAPolicy.NO_FIX_PACKAGE_REMOVAL,
            no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
            instruction="Remove the exact Maven GAV through the authorized transaction.",
        )

        prompt = _build_workaround_prompt(
            task,
            group,
            project_language=ProjectLanguage.JAVA,
            authorized_manifest_paths=["pom.xml", "modules/service/pom.xml"],
        )

        assert "remove_no_fix_dependency" in prompt
        assert "pom.xml" in prompt
        assert "org.example:library" in prompt
        assert "lockfile" not in prompt.casefold()

    def test_success_requires_validation_after_code_edit(self):
        group = _sast_group()
        repo_root = _repo_root()
        task = _task_for_group(group)
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            task,
            group,
            constraints_ledger=["express must remain >= 4.22.1"],
            previous_feedback="Fix the broken regex from the previous attempt.",
        )

        replacement = {
            "file_path": "routes/login.ts",
            "old_text": "unsafeRender(input)",
            "new_text": "safeRender(input)",
            "expected_occurrences": 1,
        }
        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="map",
                tool_calls=[
                    {
                        "name": "read_repository_map",
                        "args": {},
                        "id": "call-0",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="read",
                tool_calls=[
                    {
                        "name": "read_workspace_file",
                        "args": {"file_path": "routes/login.ts"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="searching",
                tool_calls=[
                    {
                        "name": "search_codebase_pattern",
                        "args": {"search_pattern": "unsafeRender"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="planning",
                tool_calls=[
                    {
                        "name": "record_plan",
                        "args": {
                            "affected_files": ["routes/login.ts"],
                            "affected_symbols": ["login"],
                            "security_invariant": "User-controlled HTML is escaped.",
                            "causal_hypothesis": "The unsafe renderer receives raw login input.",
                            "planned_replacements": [replacement],
                            "evidence_source": "workspace:routes/login.ts",
                        },
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="editing",
                tool_calls=[
                    {
                        "name": "deterministic_apply_edit_set",
                        "args": {"replacements": [replacement]},
                        "id": "call-4",
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
                        "id": "call-5",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        plan_state_ref: dict[str, dict] = {}

        def _state() -> dict:
            return plan_state_ref["state"]

        def _tool(name: str, result: str, mutate=None) -> MagicMock:
            tool = MagicMock()
            tool.name = name

            def invoke(args):
                if mutate is not None:
                    mutate(args)
                return result

            tool.invoke.side_effect = invoke
            return tool

        def _mark_investigated(_args):
            _state()["local_investigation_complete"] = True
            _state().setdefault("read_files", set()).add("routes/login.ts")

        def _record(_args):
            _state().update(
                {
                    "recorded": True,
                    "phase": "EXECUTE",
                    "planned_replacements": [replacement],
                    "plan_revision": 1,
                }
            )

        def _edit(_args):
            _state().update(
                {
                    "phase": "VALIDATE",
                    "successful_edit_count_this_iteration": 1,
                }
            )

        def _validate(_args):
            _state().update(
                {
                    "phase": "VALIDATE",
                    "validated_files": ["routes/login.ts"],
                    "validation_passed": True,
                    "last_validation_result": {
                        "overall_status": "PASS",
                        "validated_files": ["routes/login.ts"],
                    },
                }
            )
            return (
                "SUCCESS: Workaround validation gate passed. Validated files: routes/login.ts.\n"
                'JSON: {"overall_status":"PASS","validated_files":["routes/login.ts"]}'
            )

        tool_map = [
            _tool("record_plan", "SUCCESS: Plan recorded.", _record),
            _tool("read_repository_map", "SUCCESS: Repository map read."),
            _tool("read_workspace_file", "SUCCESS: routes/login.ts", _mark_investigated),
            _tool(
                "search_codebase_pattern",
                "routes/login.ts:10: unsafeRender(input)",
                _mark_investigated,
            ),
            _tool(
                "deterministic_apply_edit_set",
                'SUCCESS: Atomic edit set applied.\nJSON: {"affected_files":["routes/login.ts"]}',
                _edit,
            ),
            _tool(
                "validate_workaround",
                "SUCCESS: Workaround validation gate passed. Validated files: routes/login.ts.\n"
                'JSON: {"overall_status":"PASS","validated_files":["routes/login.ts"]}',
                _validate,
            ),
        ]

        def _build_toolbelt(*_args, **_kwargs):
            plan_state_ref["state"] = _kwargs["plan_state"]
            return tool_map

        from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop

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
                side_effect=_build_toolbelt,
            ),
            patch(
                "remediation_engine.orchestration.workaround_subagent.run_bounded_subagent_loop",
                wraps=run_bounded_subagent_loop,
            ) as loop,
        ):
            result = run_workaround_subagent_node(state)

        # The successful validation gate is terminal; the old extra textual
        # model round must not be requested.
        assert bound.invoke.call_count == 6
        assert loop.call_args.kwargs["context_manager"] is not None
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert result["changed_files"] == ["routes/login.ts"]
        assert "messages" not in result

    def test_circuit_breaker_surfaces_as_surrender_with_errors(self):
        group = _sast_group()
        repo_root = _repo_root()
        task = _task_for_group(group)
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            task,
            group,
        )

        replacement = {
            "file_path": "routes/login.ts",
            "old_text": "unsafeRender(input)",
            "new_text": "safeRender(input)",
            "expected_occurrences": 1,
        }
        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="map",
                tool_calls=[
                    {
                        "name": "read_repository_map",
                        "args": {},
                        "id": "call-0",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="read",
                tool_calls=[
                    {
                        "name": "read_workspace_file",
                        "args": {"file_path": "routes/login.ts"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="search",
                tool_calls=[
                    {
                        "name": "search_codebase_pattern",
                        "args": {"search_pattern": "unsafeRender"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="plan",
                tool_calls=[
                    {
                        "name": "record_plan",
                        "args": {
                            "affected_files": ["routes/login.ts"],
                            "affected_symbols": ["login"],
                            "security_invariant": "User-controlled HTML is escaped.",
                            "causal_hypothesis": "The unsafe renderer receives raw login input.",
                            "planned_replacements": [replacement],
                            "evidence_source": "workspace:routes/login.ts",
                        },
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="edit",
                tool_calls=[
                    {
                        "name": "deterministic_apply_edit_set",
                        "args": {"replacements": [replacement]},
                        "id": "call-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="validating",
                tool_calls=[
                    {
                        "name": "validate_workaround",
                        "args": {"modified_files": ["routes/login.ts"]},
                        "id": "call-5",
                        "type": "tool_call",
                    }
                ],
            ),
        )
        sandbox = _sandbox_mock()
        plan_state_ref: dict[str, dict] = {}

        def _state() -> dict:
            return plan_state_ref["state"]

        def _tool(name: str, result: str, mutate=None) -> MagicMock:
            tool = MagicMock()
            tool.name = name

            def invoke(args):
                if mutate is not None:
                    mutate(args)
                return result

            tool.invoke.side_effect = invoke
            return tool

        def _investigate(_args):
            _state()["local_investigation_complete"] = True
            _state().setdefault("read_files", set()).add("routes/login.ts")

        def _record(_args):
            _state().update(
                {
                    "recorded": True,
                    "phase": "EXECUTE",
                    "planned_replacements": [replacement],
                }
            )

        def _edit(_args):
            _state()["phase"] = "VALIDATE"

        tools = [
            _tool("read_repository_map", "SUCCESS: Repository map read."),
            _tool("read_workspace_file", "SUCCESS: routes/login.ts", _investigate),
            _tool("search_codebase_pattern", "routes/login.ts:10: unsafeRender", _investigate),
            _tool("record_plan", "SUCCESS: Plan recorded.", _record),
            _tool("deterministic_apply_edit_set", "SUCCESS: Edit applied.", _edit),
            _tool(
                "validate_workaround",
                "BLOCKED: Sandbox is not running, so validation cannot continue.",
            ),
        ]

        def _build_toolbelt(*_args, **_kwargs):
            plan_state_ref["state"] = _kwargs["plan_state"]
            return tools

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
                side_effect=_build_toolbelt,
            ),
        ):
            result = run_workaround_subagent_node(state)

        assert bound.invoke.call_count == 6
        assert result["action_summary"].status == AgentActionStatus.SURRENDER
        assert any("Sandbox is not running," in err for err in result["errors"])

    def test_workaround_subagent_bypass_mode(self):
        group = _sast_group()
        repo_root = _repo_root()
        task = _task_for_group(group)
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            task,
            group,
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

# Stage-2 behavior retained during the architecture merge.
from remediation_engine.language import ProjectLanguage

from remediation_engine.orchestration.update_subagent import _build_update_system_prompt

from remediation_engine.orchestration.workaround_subagent import _WORKAROUND_STATIC_INSTRUCTIONS

from remediation_engine.orchestration.workaround_subagent import _build_workaround_system_prompt

from remediation_engine.orchestration.update_subagent import (
    _MULTI_PACKAGE_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS,
    _build_multi_package_update_prompt,
)


class TestUpdateWorkerPromptLanguageParity:
    @pytest.mark.parametrize(
        ("language", "ecosystem", "package", "manifest", "expected_tool", "wrong_tool"),
        [
            (
                ProjectLanguage.NODEJS,
                "npm",
                "lodash",
                "package.json",
                "modify_and_validate_npm_dependency",
                "modify_and_validate_python_dependency",
            ),
            (
                ProjectLanguage.PYTHON,
                "pypi",
                "certifi",
                "requirements.txt",
                "modify_and_validate_python_dependency",
                "modify_and_validate_npm_dependency",
            ),
        ],
    )
    def test_singleton_prompts_share_authorization_policy_and_expose_matching_tool(
        self, language, ecosystem, package, manifest, expected_tool, wrong_tool
    ):
        from remediation_engine.orchestration.update_subagent import _build_update_system_prompt

        system_prompt = " ".join(_build_update_system_prompt(language).split())
        assert "If the exact Supervisor instruction requires a major-version migration" in (
            system_prompt
        )
        assert "authorized and required to update affected production and test code" in (
            system_prompt
        )
        assert "Otherwise, do not edit source-code files" in system_prompt
        assert expected_tool in system_prompt
        assert wrong_tool not in system_prompt

    @pytest.mark.parametrize(
        ("language", "ecosystem", "package", "manifest", "expected_tool", "wrong_tool"),
        [
            (
                ProjectLanguage.NODEJS,
                "npm",
                "lodash",
                "package.json",
                "modify_batch_npm_dependencies",
                "modify_batch_python_dependencies",
            ),
            (
                ProjectLanguage.PYTHON,
                "pypi",
                "certifi",
                "requirements.txt",
                "modify_batch_python_dependencies",
                "modify_batch_npm_dependencies",
            ),
        ],
    )
    def test_batch_prompts_share_authorization_policy_and_expose_matching_tool(
        self, language, ecosystem, package, manifest, expected_tool, wrong_tool
    ):
        from remediation_engine.orchestration.update_subagent import (
            _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS,
        )

        action = MultiPackageAction(
            cluster_id="cluster-prompt-test",
            dispatch_batch_id="batch-prompt-test",
            selected_strategy=TacticalStrategy.VERSION_BUMP,
            package_mutations=[
                PackageMutation(
                    task_id="task-prompt-test",
                    package_name=package,
                    manifest_path=manifest,
                    target_version="1.2.3",
                    ecosystem=ecosystem,
                    dependency_type="requirements" if ecosystem == "pypi" else "dependencies",
                )
            ],
            rationale="Check language-specific package tool selection.",
        )
        dynamic_prompt = _build_multi_package_update_prompt(
            [],
            action,
            [],
            {},
            {},
            project_language=language,
        )
        static_prompt = (
            _MULTI_PACKAGE_PYTHON_UPDATE_WORKER_STATIC_INSTRUCTIONS
            if language == ProjectLanguage.PYTHON
            else _MULTI_PACKAGE_UPDATE_WORKER_STATIC_INSTRUCTIONS
        )
        normalized_static = " ".join(static_prompt.split())
        normalized_dynamic = " ".join(dynamic_prompt.split())
        assert "major-version migration" in normalized_static
        assert "source and test migration to the committed package API is authorized and required" in (
            normalized_static
        )
        assert "Do not reselect or change any package name or version" in normalized_static
        assert f"Invoke {expected_tool} exactly once" in normalized_dynamic
        assert wrong_tool not in normalized_static + normalized_dynamic

class TestUpdateSubagentWrapperStage2Parity:
    def test_language_matched_worker_prompts_limit_python_authority(self):
        group = _sast_group()
        task = _task_for_group(group)
        update_state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [task],
            [group],
            project_language=ProjectLanguage.PYTHON,
        )
        workaround_state = initial_workaround_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            task,
            group,
            project_language=ProjectLanguage.PYTHON,
        )

        assert _build_update_system_prompt() == _UPDATE_WORKER_STATIC_INSTRUCTIONS
        assert _build_workaround_system_prompt() == _WORKAROUND_STATIC_INSTRUCTIONS

        update_prompt = _build_update_system_prompt(update_state["project_language"])
        assert "Python dependency-manifest" in update_prompt
        assert "requirements.txt" in update_prompt
        assert "pyproject.toml" in update_prompt
        assert "setup.cfg" in update_prompt
        assert "Pipfile" in update_prompt
        assert "setup.py is install-only" in update_prompt
        assert "only modify_and_validate_python_dependency" in update_prompt
        assert "Do\nnot search PyPI" in update_prompt
        assert "get_pypi_versions" not in update_prompt
        assert "select_python_safe_version" not in update_prompt
        assert "modify_and_validate_npm_dependency" not in update_prompt

        group = _sca_group()
        task = _task_for_group(group).model_copy(
            update={
                "selected_version": "4.17.21",
                "target_dependency_type": "requirements",
            }
        )
        dynamic_update_prompt = _build_update_prompt(
            [(task, group, ["requirements.txt"])],
            [],
            {},
            {},
            allowed_target_versions_by_task={
                task.task_id: ["4.17.21", "4.17.22"],
            },
            allowed_dependency_types_by_task={
                task.task_id: ["requirements", "optional-dependencies"],
            },
            project_language=ProjectLanguage.PYTHON,
        )
        assert "Canonical project language: python (Python)." in dynamic_update_prompt
        assert "Committed target version: 4.17.21" in dynamic_update_prompt
        assert "Committed dependency type: requirements" in dynamic_update_prompt
        assert "4.17.22" not in dynamic_update_prompt
        assert "optional-dependencies" not in dynamic_update_prompt
        assert "Allowed target versions:" not in dynamic_update_prompt

        workaround_prompt = _build_workaround_system_prompt(workaround_state["project_language"])
        dynamic_prompt = _build_workaround_prompt(
            task,
            group,
            project_language=workaround_state["project_language"],
        )
        for expected in (
            "Python code security",
            "`.py`",
            "pip or Pipenv",
            "pytest",
            "validate_python_syntax",
            "run_targeted_python_test",
            "validate_python_workaround",
        ):
            assert expected in workaround_prompt
        assert "Canonical project language: python" in dynamic_prompt
        assert "modify_and_validate_npm_dependency" not in workaround_prompt

    def test_subagent_state_helpers_default_to_nodejs(self):
        group = _sast_group()
        task = _task_for_group(group)

        update_state = initial_update_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            [task],
            [group],
        )
        workaround_state = initial_workaround_subagent_state(
            _repo_root(),
            "agent_workspace_deadbeef",
            task,
            group,
        )

        assert update_state["project_language"] is ProjectLanguage.NODEJS
        assert workaround_state["project_language"] is ProjectLanguage.NODEJS

    def test_success_requires_validation_after_code_edit_python_stage2(self):
        group = _sast_group()
        repo_root = _repo_root()
        task = _task_for_group(group)
        state = initial_workaround_subagent_state(
            repo_root,
            "agent_workspace_deadbeef",
            task,
            group,
            constraints_ledger=["express must remain >= 4.22.1"],
            previous_feedback="Fix the broken regex from the previous attempt.",
            project_language=ProjectLanguage.PYTHON,
        )

        replacement = {
            "file_path": "routes/login.ts",
            "old_text": "unsafeRender(input)",
            "new_text": "safeRender(input)",
            "expected_occurrences": 1,
        }
        llm, bound = _mock_llm_with_responses(
            AIMessage(
                content="map",
                tool_calls=[
                    {
                        "name": "read_repository_map",
                        "args": {},
                        "id": "call-0",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="read",
                tool_calls=[
                    {
                        "name": "read_workspace_file",
                        "args": {"file_path": "routes/login.ts"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="searching",
                tool_calls=[
                    {
                        "name": "search_codebase_pattern",
                        "args": {"search_pattern": "unsafeRender"},
                        "id": "call-2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="planning",
                tool_calls=[
                    {
                        "name": "record_plan",
                        "args": {
                            "affected_files": ["routes/login.ts"],
                            "affected_symbols": ["login"],
                            "security_invariant": "User-controlled HTML is escaped.",
                            "causal_hypothesis": "The unsafe renderer receives raw login input.",
                            "planned_replacements": [replacement],
                            "evidence_source": "workspace:routes/login.ts",
                        },
                        "id": "call-3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="editing",
                tool_calls=[
                    {
                        "name": "deterministic_apply_edit_set",
                        "args": {"replacements": [replacement]},
                        "id": "call-4",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="validating",
                tool_calls=[
                    {
                        "name": "validate_python_workaround",
                        "args": {
                            "modified_files": ["routes/login.ts"],
                            "runtime_smoke_file": "routes/login.ts",
                        },
                        "id": "call-5",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="done"),
        )
        sandbox = _sandbox_mock()
        plan_state_ref: dict[str, dict] = {}

        def _state() -> dict:
            return plan_state_ref["state"]

        def _tool(name: str, result: str, mutate=None) -> MagicMock:
            tool = MagicMock()
            tool.name = name

            def invoke(args):
                if mutate is not None:
                    mutate(args)
                return result

            tool.invoke.side_effect = invoke
            return tool

        def _mark_investigated(_args):
            _state()["local_investigation_complete"] = True
            _state().setdefault("read_files", set()).add("routes/login.ts")

        def _record(_args):
            _state().update(
                {
                    "recorded": True,
                    "phase": "EXECUTE",
                    "planned_replacements": [replacement],
                    "plan_revision": 1,
                }
            )

        def _edit(_args):
            _state().update(
                {
                    "phase": "VALIDATE",
                    "successful_edit_count_this_iteration": 1,
                }
            )

        def _validate(_args):
            _state().update(
                {
                    "phase": "VALIDATE",
                    "validated_files": ["routes/login.ts"],
                    "validation_passed": True,
                    "last_validation_result": {
                        "overall_status": "PASS",
                        "validated_files": ["routes/login.ts"],
                    },
                }
            )
            return (
                "SUCCESS: Workaround validation gate passed. Validated files: routes/login.ts.\n"
                'JSON: {"overall_status":"PASS","validated_files":["routes/login.ts"]}'
            )

        tool_map = [
            _tool("record_plan", "SUCCESS: Plan recorded.", _record),
            _tool("read_repository_map", "SUCCESS: Repository map read."),
            _tool("read_workspace_file", "SUCCESS: routes/login.ts", _mark_investigated),
            _tool(
                "search_codebase_pattern",
                "routes/login.ts:10: unsafeRender(input)",
                _mark_investigated,
            ),
            _tool(
                "deterministic_apply_edit_set",
                'SUCCESS: Atomic edit set applied.\nJSON: {"affected_files":["routes/login.ts"]}',
                _edit,
            ),
            _tool(
                "validate_python_workaround",
                "SUCCESS: Workaround validation gate passed. Validated files: routes/login.ts.\n"
                'JSON: {"overall_status":"PASS","validated_files":["routes/login.ts"]}',
                _validate,
            ),
        ]

        def _build_toolbelt(*_args, **_kwargs):
            plan_state_ref["state"] = _kwargs["plan_state"]
            return tool_map

        from remediation_engine.orchestration.subagent_runtime import run_bounded_subagent_loop

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
                side_effect=_build_toolbelt,
            ) as toolbelt_builder,
            patch(
                "remediation_engine.orchestration.workaround_subagent.run_bounded_subagent_loop",
                wraps=run_bounded_subagent_loop,
            ) as loop,
        ):
            result = run_workaround_subagent_node(state)

        # The successful validation gate is terminal; the old extra textual
        # model round must not be requested.
        assert bound.invoke.call_count == 6
        assert toolbelt_builder.call_args.kwargs["language"] is ProjectLanguage.PYTHON
        assert loop.call_args.kwargs["context_manager"] is not None
        assert result["action_summary"].status == AgentActionStatus.SUCCESS
        assert result["changed_files"] == ["routes/login.ts"]
        assert "messages" not in result
