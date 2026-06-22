from __future__ import annotations

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FailureCategory,
    GroupRemediationStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    QAEvaluation,
    RoutingStrategy,
    Severity,
    SupervisorDecision,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.state import (
    RemediationState,
    initial_orchestrator_state,
    initial_update_subagent_state,
    initial_workaround_subagent_state,
    merge_dict_reducer,
)


def _group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        file_path="package.json",
    )
    localized = LocalizedIssue(
        issue=issue,
        manifest_file="package.json",
        localization_confidence=0.9,
    )
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path="package.json",
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        localized_issues=[localized],
    )


class TestMergeDictReducer:
    def test_preserves_unmentioned_keys_and_overrides_explicit_keys(self):
        left = {"group-a": 1, "group-b": 2}
        right = {"group-b": 3, "group-c": 4}

        merged = merge_dict_reducer(left, right)

        assert merged == {"group-a": 1, "group-b": 3, "group-c": 4}
        assert left == {"group-a": 1, "group-b": 2}
        assert right == {"group-b": 3, "group-c": 4}

    def test_handles_none_inputs(self):
        assert merge_dict_reducer(None, None) == {}
        assert merge_dict_reducer({"group-a": 1}, None) == {"group-a": 1}
        assert merge_dict_reducer(None, {"group-a": 1}) == {"group-a": 1}


class TestInitialStateHelpers:
    def test_initial_orchestrator_state_initializes_phase1_fields(self, tmp_path):
        group = _group()

        state = initial_orchestrator_state(str(tmp_path), [group])

        assert state["repo_root"] == str(tmp_path)
        assert state["valid_groups"] == [group]
        assert state["constraints_ledger"] == []
        assert state["retry_counts"] == {}
        assert state["group_strategies"] == {}
        assert state["group_statuses"] == {}
        assert state["qa_evaluations"] == {}
        assert state["action_summaries"] == []
        assert state["changed_files"] == []
        assert state["errors"] == []
        # Supervisor routing defaults
        assert state["next_routing_step"] == ""
        assert state["active_target_group_ids"] == []
        assert state["feedback_by_group"] == {}
        assert state["supervisor_instructions"] == ""
        assert state["eval_status"] == ""
        assert "messages" not in state

    def test_initial_update_subagent_state_initializes_ephemeral_fields(self, tmp_path):
        group = _group()
        constraints = ["lodash must remain >= 4.17.21"]

        state = initial_update_subagent_state(
            str(tmp_path),
            "agent_workspace_deadbeef",
            [group],
            constraints,
            feedback_by_group={
                group.group_id: "Retry with an override instead of a direct bump."
            },
        )

        assert state["repo_root"] == str(tmp_path)
        assert state["workspace_volume"] == "agent_workspace_deadbeef"
        assert state["target_groups"] == [group]
        assert state["constraints_ledger"] == constraints
        assert state["constraints_ledger"] is not constraints
        assert state["feedback_by_group"] == {
            group.group_id: "Retry with an override instead of a direct bump."
        }
        assert state["messages"] == []
        assert state["changed_files"] == []
        assert state["errors"] == []

    def test_initial_workaround_subagent_state_initializes_ephemeral_fields(self, tmp_path):
        group = _group()
        constraints = ["lodash must remain >= 4.17.21"]

        state = initial_workaround_subagent_state(
            str(tmp_path),
            "agent_workspace_deadbeef",
            group,
            constraints,
            previous_feedback="Fix the syntax error before returning.",
        )

        assert state["repo_root"] == str(tmp_path)
        assert state["workspace_volume"] == "agent_workspace_deadbeef"
        assert state["target_group"] == group
        assert state["constraints_ledger"] == constraints
        assert state["constraints_ledger"] is not constraints
        assert state["previous_feedback"] == "Fix the syntax error before returning."
        assert state["messages"] == []
        assert state["changed_files"] == []
        assert state["errors"] == []


class TestLegacyCompatibilityGuard:
    def test_remediation_state_shape_remains_legacy(self):
        state: RemediationState = {
            "repo_root": "/workspace/repo",
            "issue": VulnerabilityIssue(
                source=IssueSource.SEMGREP,
                issue_type=IssueType.SAST,
                severity=Severity.HIGH,
                rule_id="javascript.xss",
                file_path="routes/login.ts",
            ),
            "status": "pending",
            "dry_run": True,
            "errors": [],
        }

        assert "issue" in state
        assert "dry_run" in state
        assert "errors" in state


class TestStateContractExamples:
    def test_phase1_contract_examples_fit_master_state_maps(self):
        summary = AgentActionSummary(
            group_id="group-1",
            status=AgentActionStatus.SUCCESS,
            summary="Updated lodash via an overrides entry in package.json.",
        )
        evaluation = QAEvaluation(
            group_id="group-1",
            passed=False,
            failure_category=FailureCategory.PEER_CONFLICT,
            retry_feedback="Retry with a compatible manifest-wide batch update.",
        )

        retry_counts = merge_dict_reducer(None, {"group-1": 1})
        strategies = merge_dict_reducer(None, {"group-1": RoutingStrategy.VERSION_BUMP})
        qa_evaluations = merge_dict_reducer(None, {"group-1": evaluation})

        assert retry_counts["group-1"] == 1
        assert strategies["group-1"] == RoutingStrategy.VERSION_BUMP
        assert qa_evaluations["group-1"] == evaluation
        assert summary.status == AgentActionStatus.SUCCESS
