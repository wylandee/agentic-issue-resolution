"""Focused tests for QA evaluation, investigation, and policy guardrails."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    DependencyEvidenceStatus,
    FailureCategory,
    FixPlanStatus,
    NoFixMitigationStage,
    QACriticLLMOutput,
    QADependencyEvidence,
    QAEvaluation,
    QAPolicy,
    QASemanticSecurityReview,
    RemediationTask,
    RoutingStrategy,
    ScratchpadScope,
    SecurityReviewVerdict,
    TaskStatus,
    TestAttributionVerdict,
)
from remediation_engine.orchestration._qa_runtime import (
    _derive_qa_task_strategies,
    _group_scan_status,
)
from remediation_engine.orchestration.qa_critic import _run_global_execution
from remediation_engine.orchestration.qa_evaluator import (
    GroupInvestigation,
    _build_individual_investigator_prompt,
    _build_qa_dynamic_context,
    _build_qa_terminal_tool,
    _run_individual_investigations,
)
from remediation_engine.orchestration.qa_policy_engine import _apply_guardrails
from remediation_engine.orchestration.qa_types import (
    QATaskContext,
    _QAExecutionResults,
    _QAPackageState,
    _SecurityScanResult,
)
from tests.test_qa_critic_support import (
    _install_outcome,
    _make_fully_populated_results,
    _make_group,
    _scan_outcome,
    _test_outcome,
)


def _task_context(
    group,
    task_id: str | None = None,
    *,
    strategy: RoutingStrategy = RoutingStrategy.VERSION_BUMP,
    qa_policy=None,
    no_fix_stage=None,
) -> QATaskContext:
    """Build one task-owned QA context for policy/evaluator tests."""
    resolved_task_id = task_id or f"{group.group_id}-task"
    return QATaskContext(
        task=RemediationTask(
            task_id=resolved_task_id,
            parent_group_id=group.group_id,
            strategy=strategy,
            qa_policy=qa_policy,
            no_fix_stage=no_fix_stage,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
            instruction="Test remediation instruction.",
        ),
        group=group,
    )


class TestGroupScanAttribution:
    def test_group_scan_status_is_cleared_when_other_group_is_still_flagged(self):
        group_a = _make_group(group_id="group-a", cve_ids=["CVE-2021-1111"], ghsa_ids=[])
        group_b = _make_group(group_id="group-b", cve_ids=["CVE-2021-2222"], ghsa_ids=[])
        scan_result = _SecurityScanResult(
            ok=False,
            summary="target vulnerabilities remain",
            remaining_identifiers={"CVE-2021-1111"},
            found_identifiers={"CVE-2021-1111"},
            new_identifiers=set(),
        )

        assert _group_scan_status(scan_result, group_a) == "still_flagged"
        assert _group_scan_status(scan_result, group_b) == "cleared"

    def test_investigator_prompt_includes_global_new_findings(self):
        group = _make_group(group_id="group-a")
        results = _make_fully_populated_results(ok=True)
        results.scan = _SecurityScanResult(
            ok=True,
            summary="scan ok",
            remaining_identifiers=set(),
            found_identifiers={"CVE-2025-10001"},
            new_identifiers={"CVE-2025-10001"},
        )

        investigator_prompt = _build_individual_investigator_prompt(
            task_id="task-a",
            group=group,
            strategy="version_bump",
            results=results,
            group_remaining_ids=[],
            candidate_changed_files=[],
            action_summaries=[],
        )
        assert "CVE-2025-10001" in investigator_prompt


def test_dynamic_context_retains_compaction_proof_dependency_evidence():
    """The evaluator receives typed dependency facts instead of raw file output."""
    group = _make_group(group_id="group-a")
    results = _make_fully_populated_results(ok=True)
    results.package_state_by_task["task-a"] = _QAPackageState(
        manifest_state="present",
        graph_state="present",
        dependency_evidence=QADependencyEvidence(
            status=DependencyEvidenceStatus.VERIFIED,
            target_package="lodash",
            expected_version="4.17.21",
            manifest_paths=["package.json"],
            lockfile_paths=["package-lock.json"],
            declarations={"package.json#/dependencies/lodash": "4.17.21"},
            resolved_versions=["4.17.21"],
            evidence_refs=[
                "package.json#/dependencies/lodash",
                "package-lock.json#resolved/lodash",
            ],
        ),
    )

    context = _build_qa_dynamic_context(
        group=group,
        task_id="task-a",
        strategy="version_bump",
        results=results,
        group_remaining_ids=[],
        candidate_changed_files=[],
        action_summaries=[],
        qa_policy=QAPolicy.VERSION_BUMP,
    )

    assert '"status":"verified"' in context
    assert '"expected_version":"4.17.21"' in context
    assert "package-lock.json#resolved/lodash" in context


def test_dynamic_context_renders_install_and_test_tuple_statuses():
    """Render tuple-backed install and test outcomes from their boolean field."""
    group = _make_group(group_id="group-a")

    passing_context = _build_qa_dynamic_context(
        group=group,
        task_id="task-a",
        strategy="version_bump",
        results=_make_fully_populated_results(ok=True),
        group_remaining_ids=[],
        candidate_changed_files=[],
        action_summaries=[],
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    assert "- Install: PASS" in passing_context
    assert "- Unit tests: PASS" in passing_context

    failing_context = _build_qa_dynamic_context(
        group=group,
        task_id="task-a",
        strategy="version_bump",
        results=_make_fully_populated_results(ok=False),
        group_remaining_ids=[],
        candidate_changed_files=[],
        action_summaries=[],
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    assert "- Install: FAIL" in failing_context
    assert "- Unit tests: FAIL" in failing_context


class TestRunGlobalExecution:
    def test_calls_install_scan_and_tests_once(self):
        sandbox = MagicMock()
        target_ids = {"CVE-2021-0001"}
        with (
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_install",
                return_value=_install_outcome(True, "ok"),
            ) as mi,
            patch(
                "remediation_engine.orchestration.qa_odc._run_security_scan",
                return_value=_scan_outcome(True, "ok"),
            ) as ms,
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
                return_value=_test_outcome(True, "ok"),
            ) as mt,
        ):
            results = _run_global_execution(sandbox, "vol", target_ids)
        mi.assert_called_once_with(sandbox)
        ms.assert_called_once_with(sandbox, "vol", target_ids)
        mt.assert_called_once_with(sandbox)
        assert results.install == (True, "ok")
        assert results.scan.ok is True
        assert results.scan.summary == "ok"
        assert results.scan.remaining_identifiers == set()
        assert results.tests == (True, "ok")

    def test_all_three_results_populated(self):
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_install",
                return_value=_install_outcome(False, "fail", exit_code=1),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._run_security_scan",
                return_value=_scan_outcome(False, "fail"),
            ),
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
                return_value=_test_outcome(False, "fail", exit_code=1, failure_count=None),
            ),
        ):
            results = _run_global_execution(sandbox, "vol", set())
        assert results.install is not None
        assert results.scan is not None
        assert results.tests is not None

    def test_passes_baseline_to_security_scan(self):
        sandbox = MagicMock()
        target_ids = {"CVE-2021-0001"}
        baseline_ids = {"CVE-2021-0001", "CVE-2020-0001"}
        with (
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_install",
                return_value=_install_outcome(True, "ok"),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._run_security_scan",
                return_value=_SecurityScanResult(True, "ok", set(), set(), set()),
            ) as scan,
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
                return_value=_test_outcome(True, "ok"),
            ),
        ):
            _run_global_execution(sandbox, "vol", target_ids, baseline_ids)

        scan.assert_called_once_with(sandbox, "vol", target_ids, baseline_ids)

    def test_tests_run_even_when_install_fails(self):
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_install",
                return_value=_install_outcome(False, "FAILED", exit_code=1),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._run_security_scan",
                return_value=_scan_outcome(False, "fail"),
            ),
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
                return_value=_test_outcome(True, "passed."),
            ) as mt,
        ):
            results = _run_global_execution(sandbox, "vol", set())
        mt.assert_called_once()
        assert results.tests == (True, "passed.")

    def test_skip_scan_still_runs_install_and_tests(self):
        """NO_FIX package removal skips only ODC, not the rest of QA."""
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_install",
                return_value=_install_outcome(True, "install ok"),
            ) as install,
            patch(
                "remediation_engine.orchestration.qa_odc._run_security_scan",
            ) as scan,
            patch(
                "remediation_engine.orchestration.qa_test_parsing._run_unit_tests",
                return_value=_test_outcome(True, "tests ok"),
            ) as tests,
        ):
            results = _run_global_execution(
                sandbox,
                "vol",
                {"CVE-2021-0001"},
                skip_scan=True,
                scan_skip_reason="no_fix_package_removal",
            )

        install.assert_called_once_with(sandbox)
        scan.assert_not_called()
        tests.assert_called_once_with(sandbox)
        assert results.scan is None
        assert results.scan_skipped is True
        assert results.scan_skip_reason == "no_fix_package_removal"
        assert results.tests == (True, "tests ok")

    def test_skip_scan_marks_scan_skipped_in_results(self):
        sandbox = MagicMock()
        results = _run_global_execution(
            sandbox=sandbox,
            workspace_volume="vol",
            target_identifiers=set(),
            skip_scan=True,
            scan_skip_reason="no_fix_package_removal",
        )
        assert results.scan_skipped is True
        assert results.scan_skip_reason == "no_fix_package_removal"
        assert results.tests is not None


class TestBuildIndividualInvestigatorPrompt:
    def _prompt(self, group=None, remaining=None):
        if group is None:
            group = _make_group()
        results = _make_fully_populated_results(ok=True)
        return _build_individual_investigator_prompt(
            task_id=f"{group.group_id}-task",
            group=group,
            strategy="version_bump",
            results=results,
            group_remaining_ids=remaining or [],
            candidate_changed_files=["package.json"],
            action_summaries=[],
        )

    def test_contains_group_id(self):
        g = _make_group(group_id="my-group")
        assert "my-group" in self._prompt(group=g)

    def test_contains_cve_ids(self):
        g = _make_group(cve_ids=["CVE-2021-9999"], ghsa_ids=[])
        p = self._prompt(group=g, remaining=["CVE-2021-9999"])
        assert "CVE-2021-9999" in p
        assert "package.json" in p

    def test_instructs_not_to_call_execution_tools(self):
        prompt = self._prompt(
            group=_make_group(),
            remaining=["CVE-2021-23337"],
        )
        lower = prompt.lower()
        assert "never execute" in lower
        assert "free-form" in lower
        assert "emit_qa_evaluation" in lower
        assert "use responsible, exonerated, or inconclusive" not in lower

        description = _build_qa_terminal_tool().description
        for enum_type in (FailureCategory, SecurityReviewVerdict, TestAttributionVerdict):
            for member in enum_type:
                assert member.value in description

    def test_prompt_uses_status_metadata_without_raw_log_bodies(self):
        results = _QAExecutionResults(
            install=(False, "install failed"),
            tests=None,
        )
        results.install_exit_code = 1
        results.install_error_category = "PEER_CONFLICT"
        results.install_raw_stdout = "RAW INSTALL STDOUT " * 100
        results.install_raw_stderr = "RAW INSTALL STDERR " * 100
        prompt = _build_individual_investigator_prompt(
            task_id="task-1",
            group=_make_group(),
            strategy="version_bump",
            results=results,
            group_remaining_ids=[],
            candidate_changed_files=["package.json"],
            action_summaries=[],
        )

        assert "- Install: FAIL" in prompt
        assert "- Security scan: NOT_RUN" in prompt
        assert "- Unit tests: NOT_RUN" in prompt
        assert "- Install exit code: 1" in prompt
        assert "- Install error category: PEER_CONFLICT" in prompt
        assert "RAW INSTALL STDOUT" not in prompt
        assert "RAW INSTALL STDERR" not in prompt

    def test_action_summaries_are_bounded(self):
        summary = AgentActionSummary(
            task_id="task-1",
            status=AgentActionStatus.SUCCESS,
            summary="large action summary " * 500,
        )
        prompt = self._prompt()
        bounded_prompt = _build_individual_investigator_prompt(
            task_id="task-1",
            group=_make_group(),
            strategy="version_bump",
            results=_make_fully_populated_results(ok=True),
            group_remaining_ids=[],
            candidate_changed_files=["package.json"],
            action_summaries=[summary],
        )

        assert "summary truncated" in bounded_prompt
        assert len(bounded_prompt) < len(prompt) + 2_000


class TestRunIndividualInvestigations:
    def _lr(self, text="", errors=None, evaluation=None):
        from remediation_engine.orchestration.subagent_runtime import SubagentRuntimeResult

        return SubagentRuntimeResult(
            final_text=text,
            tool_events=[],
            changed_files=[],
            errors=errors or [],
            structured_output=evaluation,
        )

    def test_one_investigator_per_task(self):
        g1, g2 = _make_group(group_id="g1"), _make_group(group_id="g2")
        contexts = [_task_context(g1, "task-1"), _task_context(g2, "task-2")]
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_evaluator.run_bounded_subagent_loop",
                side_effect=[
                    self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
                    self._lr(evaluation=QACriticLLMOutput(task_id="task-2", passed=True)),
                ],
            ) as ml,
            patch(
                "remediation_engine.orchestration.qa_evaluator.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                task_contexts=contexts,
                task_strategies={"task-1": "version_bump", "task-2": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": [], "task-2": []},
                sandbox=MagicMock(),
                repo_root="/tmp",
                results=results,
                task_policies={"task-1": None, "task-2": None},
            )
        assert ml.call_count == 2
        assert all(
            call.kwargs["structured_output_model"] is QACriticLLMOutput
            for call in ml.call_args_list
        )
        assert all(call.kwargs["skip_phase_gating"] is True for call in ml.call_args_list)
        assert all(
            call.kwargs["context_manager"].compaction_interval == 3
            and call.kwargs["context_manager"].scratchpad_scope == ScratchpadScope.QA
            and call.kwargs["context_manager"].skip_phase_gating is True
            for call in ml.call_args_list
        )
        assert "task-1" in invs and "task-2" in invs
        assert invs["task-1"].evaluation is not None
        assert invs["task-1"].evaluation.passed is True
        assert invs["task-2"].evaluation is not None
        assert invs["task-2"].evaluation.passed is True

    def test_crash_produces_fallback(self):
        group = _make_group(group_id="g1")
        context = _task_context(group, "task-1")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_evaluator.run_bounded_subagent_loop",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "remediation_engine.orchestration.qa_evaluator.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                task_contexts=[context],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
                task_policies={"task-1": None},
            )
        assert invs["task-1"].errors
        assert "Fallback" in invs["task-1"].investigation_text
        assert invs["task-1"].evaluation is not None
        assert invs["task-1"].evaluation.contract_error is True
        assert invs["task-1"].evaluation.contract_error_reason

    def test_empty_output_triggers_fallback(self):
        group = _make_group(group_id="g1")
        context = _task_context(group, "task-1")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_evaluator.run_bounded_subagent_loop",
                return_value=self._lr(text=""),
            ),
            patch(
                "remediation_engine.orchestration.qa_evaluator.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                task_contexts=[context],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
                task_policies={"task-1": None},
            )
        assert invs["task-1"].errors
        assert invs["task-1"].investigation_text == ""
        assert invs["task-1"].fallback is True
        assert invs["task-1"].evaluation is not None
        assert invs["task-1"].evaluation.contract_error is True
        assert invs["task-1"].evaluation.contract_error_reason


class TestApplyGuardrails:
    def _context(
        self,
        group,
        task_id="task-1",
        policy=QAPolicy.VERSION_BUMP,
        strategy=RoutingStrategy.VERSION_BUMP,
        no_fix_stage=None,
    ):
        return _task_context(
            group,
            task_id,
            strategy=strategy,
            qa_policy=policy,
            no_fix_stage=no_fix_stage,
        )

    def _res(
        self,
        task_id="task-1",
        install_ok=True,
        scan_ok=True,
        remaining=None,
        install_summary="ok",
        package_state=None,
    ):
        r = _QAExecutionResults()
        r.install = (install_ok, install_summary)
        r.scan = _SecurityScanResult(
            ok=scan_ok,
            summary="scan ok" if scan_ok else "scan failed",
            remaining_identifiers=set(remaining or ()),
            found_identifiers=set(remaining or ()),
            new_identifiers=set(),
        )
        r.tests = (True, "ok")
        if package_state is not None:
            r.package_state_by_task[task_id] = package_state
        elif task_id == "task-1":
            r.package_state_by_task[task_id] = _QAPackageState(
                manifest_state="present",
                graph_state="present",
                dependency_evidence=QADependencyEvidence(
                    status=DependencyEvidenceStatus.VERIFIED,
                    target_package="lodash",
                    expected_version="4.17.21",
                    manifest_paths=["package.json"],
                    lockfile_paths=["package-lock.json"],
                    declarations={"package.json#/dependencies/lodash": "4.17.21"},
                    resolved_versions=["4.17.21"],
                    evidence_refs=[
                        "package.json#/dependencies/lodash",
                        "package-lock.json#resolved/lodash",
                    ],
                ),
            )
        return r

    def _apply(self, contexts, evaluations, results, investigations=None):
        return _apply_guardrails(
            task_contexts=contexts,
            batch_result=evaluations,
            results=results,
            task_policies={context.task_id: context.task.qa_policy for context in contexts},
            investigations_by_task=investigations,
        )

    def test_valid_passes_through(self):
        context = self._context(_make_group())
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        evals, errors = self._apply([context], evaluations, self._res())
        assert evals["task-1"].passed is True and not errors

    def test_no_fix_stage_two_derives_code_workaround_strategy(self):
        group = _make_group(
            group_id="sca:package.json:notevil",
            cve_ids=["CVE-2021-23771"],
            ghsa_ids=[],
            fix_plan_status=FixPlanStatus.NO_FIX,
        )
        task = RemediationTask(
            task_id="task-nofix",
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            no_fix_stage=NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
        )

        strategies = _derive_qa_task_strategies(
            [group],
            configured_strategies={},
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
        )

        assert strategies["task-nofix"] == "code_workaround"

    def test_no_fix_stage_one_remains_strict(self):
        group = _make_group(
            group_id="sca:package.json:notevil",
            cve_ids=["CVE-2021-23771"],
            ghsa_ids=[],
            fix_plan_status=FixPlanStatus.NO_FIX,
        )
        context = self._context(
            group,
            task_id="task-nofix",
            no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
            strategy=RoutingStrategy.CODE_WORKAROUND,
        )
        evaluations = [QAEvaluation(task_id="task-nofix", passed=True)]
        results = self._res(
            task_id="task-nofix",
            scan_ok=False,
            remaining={"CVE-2021-23771"},
            package_state=_QAPackageState(manifest_state="absent", graph_state="absent"),
        )
        strategies = _derive_qa_task_strategies(
            [group],
            {},
            {context.task_id: context.task},
            [context.task_id],
        )
        evals, _ = self._apply([context], evaluations, results)

        assert strategies["task-nofix"] == "no_fix_package_removal"
        assert evals["task-nofix"].passed is False
        assert evals["task-nofix"].failure_category == FailureCategory.SECURITY_FLAG

    def test_unknown_task_id_dropped(self):
        context = self._context(_make_group())
        evaluations = [
            QAEvaluation(task_id="task-1", passed=True),
            QAEvaluation(
                task_id="ghost",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="x",
            ),
        ]
        evals, errors = self._apply([context], evaluations, self._res())
        assert "ghost" not in evals and any("ghost" in e for e in errors)

    def test_duplicate_keeps_first(self):
        context = self._context(_make_group())
        evaluations = [
            QAEvaluation(task_id="task-1", passed=True),
            QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="second",
            ),
        ]
        evals, errors = self._apply([context], evaluations, self._res())
        assert evals["task-1"].passed is True
        assert any("duplicate" in e.lower() for e in errors)

    def test_missing_task_synthesized(self):
        g1, g2 = _make_group(group_id="g1"), _make_group(group_id="g2")
        contexts = [self._context(g1, "task-1"), self._context(g2, "task-2")]
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        evals, errors = self._apply(
            contexts,
            evaluations,
            self._res(),
        )
        assert evals["task-2"].passed is False
        assert evals["task-2"].failure_category == FailureCategory.SECURITY_FLAG
        assert any("task-2" in error for error in errors)

    def test_version_bump_remaining_forces_fail(self):
        group = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        context = self._context(group)
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        evals, _ = self._apply(
            [context],
            evaluations,
            self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
        )
        assert evals["task-1"].passed is False
        assert evals["task-1"].failure_category == FailureCategory.SECURITY_FLAG

    def test_code_workaround_remaining_can_pass_with_review_evidence(self):
        group = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        context = self._context(
            group,
            policy=QAPolicy.MITIGATION_CODE_WORKAROUND,
            strategy=RoutingStrategy.CODE_WORKAROUND,
        )
        review = QASemanticSecurityReview(
            verdict=SecurityReviewVerdict.PASS,
            reasoning="The protected call site is guarded.",
            evidence_refs=["src/index.js:10"],
        )
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=True,
            semantic_security_review=review,
        )
        investigation = GroupInvestigation(
            group_id=group.group_id,
            task_id="task-1",
            investigation_text="source review",
            tool_transcript="read_file_context",
            source_review_evidence=True,
            structured_review_verdict=SecurityReviewVerdict.PASS,
        )
        evals, errors = self._apply(
            [context],
            [evaluation],
            self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
            {"task-1": investigation},
        )
        assert evals["task-1"].passed is True
        assert not errors

    def test_eresolve_remaps_to_peer_conflict(self):
        context = self._context(_make_group())
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="x",
        )
        evals, _ = self._apply(
            [context],
            [evaluation],
            self._res(install_ok=False, install_summary="ERESOLVE conflict"),
        )
        assert evals["task-1"].failure_category == FailureCategory.PEER_CONFLICT

    def test_remaining_scanner_reclassifies_breaking_change_to_security_flag(self):
        group = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        context = self._context(group)
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="JWT unit tests failed after the version bump.",
        )
        evals, errors = self._apply(
            [context],
            [evaluation],
            self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
        )
        assert evals["task-1"].failure_category == FailureCategory.SECURITY_FLAG
        assert "JWT unit tests failed" in (evals["task-1"].retry_feedback or "")
        assert any("did not prioritize SECURITY_FLAG" in err for err in errors)
