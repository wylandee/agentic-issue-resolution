"""Focused Phase 2 tactical Supervisor contract and prompt tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from remediation_engine.contracts.schemas import (
    CodeWorkaroundSupervisorAction,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QADeterministicGates,
    QAEvaluation,
    QAFailureEvidence,
    RemediationTask,
    RoutingStrategy,
    ScannerExecutionStatus,
    Severity,
    TacticalStrategy,
    TacticalSupervisorAction,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.tactical_supervisor import (
    _SUPERVISOR_ACTION_CONTRACTS,
    _SUPERVISOR_STATIC_INSTRUCTIONS,
    _model_action,
    build_supervisor_messages,
    build_tactical_context,
    propose_and_verify_tactical_action,
    registry_candidate_sets_for_context,
    verify_tactical_action,
)
from remediation_engine.settings import AppSettings


def _group() -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        message="dependency vulnerability",
        id=str(uuid4()),
    )
    return VulnerabilityGroup(
        group_id="group-1",
        issue_type=IssueType.SCA,
        vulnerable_component="test-pkg",
        file_path="package.json",
        cve_ids=["CVE-2026-0001"],
        versions=["1.0.0"],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=str(uuid4()),
        issues=[issue],
        fix_plan=FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="1.2.3",
            instruction="Update the dependency.",
            strategy_used="test",
        ),
    )


def _task() -> RemediationTask:
    return RemediationTask(
        task_id="task-1",
        parent_group_id="group-1",
        strategy=RoutingStrategy.VERSION_BUMP,
        instruction="Initial dependency update.",
    )


def test_supervisor_prompt_keeps_static_and_dynamic_messages_separate() -> None:
    context = build_tactical_context(
        _task(),
        _group(),
        candidate_versions=["1.2.3", "1.3.0"],
        candidate_dependency_types=["dependencies"],
    )

    messages = build_supervisor_messages(context)

    assert len(messages) == 2
    assert messages[0].content == _SUPERVISOR_STATIC_INSTRUCTIONS
    assert "task-1" not in messages[0].content
    assert "Target package: test-pkg" in messages[1].content
    normalized_static_prompt = " ".join(messages[0].content.split())
    assert "current committed strategy as context, never as a restriction" in (
        normalized_static_prompt
    )
    assert (
        "candidate whitelist constrains versions and package targets, not whether CODE_WORKAROUND may be selected"
        in normalized_static_prompt
    )
    assert "Current committed strategy is context, not a constraint." in messages[1].content
    assert "leading alternative" in messages[1].content
    assert messages[1].content.index("## Task") < messages[1].content.index(
        "## Registry-Verified Candidates"
    )


def test_verified_version_produces_authoritative_targeted_instruction() -> None:
    context = build_tactical_context(
        _task(),
        _group(),
        candidate_versions=["1.2.3"],
        candidate_dependency_types=["dependencies"],
    )
    action = TacticalSupervisorAction(
        diagnostic_basis="Verified candidate meets the committed security floor.",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        target_version="1.2.3",
        rationale="The verified fixed version addresses the finding.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is True
    assert verification.instruction is not None
    assert "AUTHORIZED TARGET: test-pkg" in verification.instruction
    assert "EXACT VERSION OR HYPOTHESIS: 1.2.3" in verification.instruction
    assert "PROHIBITED OPERATIONS" in verification.instruction


def test_unverified_version_is_rejected_before_dispatch() -> None:
    context = build_tactical_context(
        _task(),
        _group(),
        candidate_versions=["1.2.3"],
    )
    action = TacticalSupervisorAction(
        diagnostic_basis="The proposed version is absent from the verified candidate set.",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        target_version="9.9.9",
        rationale="Try the newest version.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is False
    assert "candidate" in verification.reason


def test_tactical_model_is_disabled_without_an_api_key() -> None:
    context = build_tactical_context(_task(), _group(), candidate_versions=["1.2.3"])

    action, verification = propose_and_verify_tactical_action(
        context,
        settings=AppSettings(openai_api_key=""),
    )

    assert action is None
    assert verification is None


def test_tactical_model_uses_one_of_four_tools_not_a_root_union() -> None:
    """Provider requests use concrete action tools instead of root ``oneOf``."""
    context = build_tactical_context(_task(), _group(), candidate_versions=["1.2.3"])
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "VersionBumpSupervisorAction",
                "args": {
                    "diagnostic_basis": "The verified candidate meets the security floor.",
                    "selected_strategy": "version_bump",
                    "target_version": "1.2.3",
                    "rationale": "The lowest verified candidate is sufficient.",
                },
                "id": "tactical-1",
            }
        ],
    )
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.orchestration.tactical_supervisor.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action = _model_action(
            context,
            AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is not None
    assert action.selected_strategy == TacticalStrategy.VERSION_BUMP
    llm.with_structured_output.assert_not_called()
    contracts = llm.bind_tools.call_args.args[0]
    assert [contract.__name__ for contract in contracts] == [
        "VersionBumpSupervisorAction",
        "PackageOverrideSupervisorAction",
        "CodeWorkaroundSupervisorAction",
        "PortfolioEscalationSupervisorAction",
    ]
    assert llm.bind_tools.call_args.kwargs == {
        "tool_choice": "required",
        "strict": False,
        "parallel_tool_calls": False,
    }


def test_tactical_model_rejects_free_text_without_a_tool_call() -> None:
    """A free-form response remains unavailable and triggers fallback."""
    context = build_tactical_context(_task(), _group(), candidate_versions=["1.2.3"])
    bound_llm = MagicMock()
    bound_llm.invoke.return_value = AIMessage(content="Use the latest version.")
    llm = MagicMock()
    llm.bind_tools.return_value = bound_llm

    with patch(
        "remediation_engine.orchestration.tactical_supervisor.invoke_with_trajectory",
        side_effect=lambda _name, invoke, _inputs: invoke(),
    ):
        action = _model_action(
            context,
            AppSettings(openai_api_key="test-key"),
            model_factory=lambda _settings: llm,
        )

    assert action is None


def test_provider_action_tools_have_flat_required_schemas() -> None:
    """Each provider tool is independently strict without a root ``oneOf``."""
    for contract in _SUPERVISOR_ACTION_CONTRACTS:
        schema = convert_to_openai_tool(contract, strict=False)["function"]["parameters"]
        assert "oneOf" not in schema
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(contract.model_fields)


def test_supervisor_prompt_uses_gate_statuses_and_bounded_failure_summary() -> None:
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The targeted API regression needs a compatibility workaround.",
        failure_evidence=QAFailureEvidence(
            exact_diagnostics=["TypeError: jwt.verify is not a function"],
            failed_tests=["auth middleware rejects a valid token"],
            source_locations=["/workspace/src/auth.ts:12:4"],
            raw_excerpt="raw-secret-log-that-must-not-be-replayed",
        ),
        deterministic_gates=QADeterministicGates(
            status="fail",
            install_passed=True,
            scanner_execution_status=ScannerExecutionStatus.SUCCESS,
            target_scanner_cleared=True,
            tests_passed=False,
            package_manifest_state="match",
            package_graph_state="match",
        ),
    )
    context = build_tactical_context(_task(), _group(), evaluation=evaluation)

    dynamic = build_supervisor_messages(context)[1].content

    assert "Overall gate: FAIL" in dynamic
    assert "Install gate: PASS" in dynamic
    assert "Target scanner gate: PASS" in dynamic
    assert "Unit-test gate: FAIL" in dynamic
    assert "Manifest gate: PASS" in dynamic
    assert "Dependency-graph gate: PASS" in dynamic
    assert "TypeError: jwt.verify is not a function" in dynamic
    assert "raw-secret-log-that-must-not-be-replayed" not in dynamic
    assert dynamic.index("## QA Deterministic Gates") < dynamic.index("## QA Failure Evidence")
    assert dynamic.index("## QA Failure Evidence") < dynamic.index("## Worker Diagnostics")
    assert "raw_excerpt" not in dynamic


def test_evidence_and_model_file_paths_share_workspace_normalization() -> None:
    evaluation = QAEvaluation(
        task_id="task-1",
        passed=False,
        failure_category=FailureCategory.BREAKING_CHANGE,
        retry_feedback="The targeted API regression needs a compatibility workaround.",
        failure_evidence=QAFailureEvidence(
            source_locations=["file:///workspace/src\\auth.ts:12:4"],
        ),
    )
    context = build_tactical_context(_task(), _group(), evaluation=evaluation)
    action = CodeWorkaroundSupervisorAction(
        diagnostic_basis="The source location identifies the failed API call.",
        selected_strategy=TacticalStrategy.CODE_WORKAROUND,
        workaround_hypothesis="Guard the incompatible JWT API call.",
        target_files_hint=["src/auth.ts:12:4"],
        rationale="A source guard directly addresses the breaking API evidence.",
    )

    verification = verify_tactical_action(context, action)

    assert verification.accepted is True
    assert action.target_files_hint == ["src/auth.ts"]


def test_parent_minimum_is_not_used_as_a_security_floor() -> None:
    task = _task().model_copy(update={"parent_minimum_version": "9.9.9"})
    calls: list[str] = []

    def provider(package_name: str, security_floor: str, attempted_versions: set[str]):
        calls.append(security_floor)
        from remediation_engine.contracts.version_policy import RegistryCandidate

        return [
            RegistryCandidate(
                version="1.2.3",
                semver_key=(1, 2, 3),
                security_floor_met=True,
                is_stable=True,
                same_major=True,
                already_attempted=False,
            )
        ]

    context = build_tactical_context(task, _group())
    candidate_sets, error = registry_candidate_sets_for_context(
        context,
        registry_provider=provider,
    )

    assert error is None
    assert calls == ["1.2.3"]
    assert candidate_sets[0].versions == ("1.2.3",)
