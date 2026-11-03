"""Focused tests for qa critic."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from remediation_engine.agents.qa import _run_global_execution, run_qa_critic_node
from remediation_engine.agents.qa_evaluation import (
    GroupInvestigation,
    _apply_guardrails,
    _build_individual_investigator_prompt,
    _build_qa_dynamic_context,
    _build_qa_terminal_tool,
    _run_individual_investigations,
    build_qa_review_toolbelt,
)
from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    DependencyEvidenceStatus,
    FailureCategory,
    FixPlanStatus,
    NoFixMitigationStage,
    ODCScanEvidence,
    QACriticLLMOutput,
    QADependencyEvidence,
    QAEvaluation,
    QAFailureEvidence,
    QAPolicy,
    QASemanticSecurityReview,
    QATestAttribution,
    RemediationTask,
    RoutingStrategy,
    ScanFallbackReason,
    ScannerExecutionStatus,
    ScanScope,
    ScratchpadScope,
    SecurityReviewVerdict,
    TaskAttemptSnapshot,
    TaskStatus,
    TestAttributionVerdict,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.language import LANGUAGE_CONFIGS, ProjectLanguage
from remediation_engine.tools.qa_scan import (
    QATaskContext,
    _QAExecutionResults,
    _QAPackageState,
    _SecurityScanResult,
)
from remediation_engine.tools.qa_workspace import _derive_qa_task_strategies, _group_scan_status
from tests.helpers import (
    _install_outcome,
    _make_fully_populated_results,
    _make_group,
    _make_minimal_state,
    _scan_outcome,
    _test_outcome,
)


def _add_verified_package_evidence(
    results: _QAExecutionResults,
    task_groups: dict[str, object],
) -> None:
    """Supply matching installed-tree evidence when global QA is mocked."""
    for task_id, group in task_groups.items():
        if task_id in results.package_state_by_task:
            continue
        version = getattr(getattr(group, "fix_plan", None), "fixed_version", None)
        package = str(getattr(group, "vulnerable_component", "") or "").strip()
        if not package or not version:
            continue
        manifest_paths = list(getattr(group, "file_paths", ()) or ())
        if not manifest_paths and getattr(group, "file_path", None):
            manifest_paths = [group.file_path]
        lockfile_paths = [
            f"{path.rsplit('/', 1)[0] + '/' if '/' in path else ''}package-lock.json"
            for path in manifest_paths
            if path.endswith("package.json")
        ]
        results.package_state_by_task[task_id] = _QAPackageState(
            manifest_state="present",
            graph_state="present",
            dependency_evidence=QADependencyEvidence(
                status=DependencyEvidenceStatus.VERIFIED,
                target_package=package,
                expected_version=version,
                manifest_paths=manifest_paths,
                lockfile_paths=lockfile_paths,
                declarations={f"{path}#/{package}": version for path in manifest_paths},
                resolved_versions=[version],
                lockfile_versions=[version],
                evidence_refs=[*manifest_paths, *lockfile_paths],
            ),
        )


def _commit_tasks(state, tasks):
    """Add task-keyed current-attempt snapshots to a test state."""
    committed_tasks = {}
    snapshots = {}
    groups_by_id = {group.group_id: group for group in state.get("valid_groups", [])}
    for task in tasks:
        attempt_id = task.current_attempt_id or f"{task.task_id}-attempt"
        group = groups_by_id.get(task.parent_group_id)
        selected_version = task.selected_version
        if (
            selected_version is None
            and task.qa_policy == QAPolicy.VERSION_BUMP
            and group is not None
            and group.fix_plan is not None
        ):
            selected_version = group.fix_plan.fixed_version
        committed_task = task.model_copy(
            update={
                "current_attempt_id": attempt_id,
                "selected_version": selected_version,
                "target_package_name": task.target_package_name
                or (group.vulnerable_component if group is not None else None),
            }
        )
        committed_tasks[task.task_id] = committed_task
        snapshots[attempt_id] = TaskAttemptSnapshot(
            attempt_id=attempt_id,
            task_id=task.task_id,
            task_revision=committed_task.task_revision,
            qa_policy=committed_task.qa_policy,
            selected_version=committed_task.selected_version,
            instruction=committed_task.instruction or "QA test instruction.",
            instruction_digest=f"digest-{task.task_id}",
            dispatch_node="qa_critic",
        )
    state["task_queue"] = committed_tasks
    state["active_target_task_ids"] = list(committed_tasks)
    state["attempt_snapshots_by_id"] = snapshots
    state["worker_results_by_attempt"] = {
        task.current_attempt_id: {"changed_files": []} for task in committed_tasks.values()
    }
    return state


class TestRunQACriticNode:
    """Tests for the full node entry point (agent loop mocked)."""

    def _patch_node(self, results=None, evaluations=None, group=None):
        """Patch deterministic execution and the task-owned evaluator seam."""
        if group is None:
            group = _make_group()
        if results is None:
            results = _make_fully_populated_results(ok=True)
        if evaluations is None:
            evaluations = [QAEvaluation(task_id="task-1", passed=True)]

        evaluations_by_task = (
            evaluations
            if isinstance(evaluations, dict)
            else {evaluation.task_id: evaluation for evaluation in evaluations}
        )
        _add_verified_package_evidence(
            results,
            {task_id: group for task_id in evaluations_by_task},
        )
        investigations = {
            task_id: GroupInvestigation(
                group_id=group.group_id,
                task_id=task_id,
                investigation_text="",
                tool_transcript="",
                errors=[],
                evaluation=evaluation,
            )
            for task_id, evaluation in evaluations_by_task.items()
        }
        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)
        manifest_payload = json.dumps({"dependencies": {"lodash": "4.17.21"}})
        lockfile_payload = json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"dependencies": {"lodash": "4.17.21"}},
                    "node_modules/lodash": {"version": "4.17.21"},
                },
            }
        )
        mock_sandbox.read_file.side_effect = lambda path: (
            lockfile_payload if str(path).endswith("package-lock.json") else manifest_payload
        )
        mock_sandbox.run.return_value.stdout = json.dumps(
            {"name": "workspace", "dependencies": {"lodash": {"version": "4.17.21"}}}
        )
        return {
            "sandbox": patch(
                "remediation_engine.agents.qa.DockerSandbox",
                return_value=mock_sandbox,
            ),
            "global_exec": patch(
                "remediation_engine.agents.qa._run_global_execution",
                return_value=results,
            ),
            "investigators": patch(
                "remediation_engine.agents.qa._qa_evaluator_module._run_individual_investigations",
                return_value=investigations,
            ),
        }

    def test_all_passed_returns_all_passed_eval_status(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        patches = self._patch_node(group=group, evaluations=evaluations)

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        assert result["eval_status"] == "all_passed"
        assert result["status"] == "qa_completed"
        assert result["qa_evaluations"]["task-1"].passed is True
        report = json.loads(result["qa_investigation_report"])
        assert report["evaluations"]["task-1"]["passed"] is True

    def test_java_qa_uses_maven_image_and_run_owned_cache(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        state["project_language"] = ProjectLanguage.JAVA
        state["maven_cache_volume"] = "agent_maven_repository_test"
        patches = self._patch_node(group=group)

        with (
            patches["sandbox"] as sandbox_type,
            patches["global_exec"],
            patches["investigators"],
        ):
            result = run_qa_critic_node(state)

        sandbox_type.assert_called_once_with(
            repo_root=None,
            workspace_volume="test-vol",
            image="maven:3.9-eclipse-temurin-17",
            maven_repository_volume="agent_maven_repository_test",
            archive_excluded_dirs=frozenset({"target"}),
        )
        assert result["status"] == "qa_completed"

    def test_python_project_language_reaches_global_execution(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        state["project_language"] = ProjectLanguage.PYTHON
        patches = self._patch_node(group=group)

        with (
            patches["sandbox"] as sandbox_factory,
            patches["global_exec"] as global_execution,
            patches["investigators"],
        ):
            run_qa_critic_node(state)

        assert global_execution.call_args.kwargs["project_language"] == ProjectLanguage.PYTHON
        assert (
            sandbox_factory.call_args.kwargs["image"]
            == LANGUAGE_CONFIGS[ProjectLanguage.PYTHON].docker_image
        )

    def test_package_removal_node_skips_only_security_scan(self):
        group = _make_group(fix_plan_status=FixPlanStatus.NO_FIX)
        task = RemediationTask(
            task_id="task-1",
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            qa_policy=QAPolicy.NO_FIX_PACKAGE_REMOVAL,
            no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
            instruction="Remove the vulnerable package.",
        )
        state = _commit_tasks(_make_minimal_state(groups=[group]), [task])
        results = _make_fully_populated_results(ok=True)
        results.scan = None
        results.scan_skipped = True
        results.scan_skip_reason = "no_fix_package_removal"
        results.package_state_by_task["task-1"] = _QAPackageState(
            manifest_state="absent",
            graph_state="absent",
        )
        patches = self._patch_node(group=group, results=results)

        with (
            patches["sandbox"],
            patches["global_exec"] as global_exec,
            patches["investigators"],
            patch(
                "remediation_engine.agents.qa._collect_group_package_state",
                return_value=_QAPackageState(
                    manifest_state="absent",
                    graph_state="absent",
                ),
            ),
        ):
            result = run_qa_critic_node(state)

        call = global_exec.call_args.kwargs
        assert call["skip_scan"] is True
        assert call["scan_skip_reason"] == "no_fix_package_removal"
        assert result["new_vulnerability_status"] == "not_scanned"
        assert result["qa_evaluations"]["task-1"].passed is True

    def test_no_matching_target_is_skipped_but_not_passed(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        results = _make_fully_populated_results(ok=True)
        results.scan_target_failures_by_task = {
            "task-1": (
                ScanFallbackReason.NO_MATCHING_TARGET,
                "Task task-1 target selection failed for package-lock.json: no_matching_target",
            )
        }
        patches = self._patch_node(group=group, results=results)

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        evaluation = result["qa_evaluations"]["task-1"]
        gates = evaluation.deterministic_gates
        assert result["eval_status"] == "failures_detected"
        assert evaluation.passed is False
        assert evaluation.evidence_inconclusive is True
        assert evaluation.failure_category == FailureCategory.SECURITY_FLAG
        assert gates is not None
        assert gates.status == "fail"
        assert gates.scanner_execution_status == ScannerExecutionStatus.SKIPPED
        assert gates.target_scanner_cleared is None
        assert any("scan skipped" in item for item in gates.diagnostics)
        assert "not evidence" in evaluation.retry_feedback.lower()
        assert not any("no_matching_target" in error for error in result["errors"])

    def test_targeted_scan_failure_preserves_actionable_test_failure(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        results = _make_fully_populated_results(ok=True)
        results.tests = (False, "npm test FAILED")
        results.test_raw_stdout = "not ok 1 - api authentication test\n"
        results.test_exit_code = 1
        results.scan_target_failures_by_task = {
            "task-1": (
                ScanFallbackReason.TARGETED_SCAN_FAILED,
                "unresolved npm alias in lockfile closure",
            )
        }
        failure_evidence = QAFailureEvidence(failed_tests=["api authentication test"])
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The version update broke the API tests.",
            failure_evidence=failure_evidence,
        )
        patches = self._patch_node(group=group, results=results, evaluations=[evaluation])

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        final = result["qa_evaluations"]["task-1"]
        assert final.passed is False
        assert final.evidence_inconclusive is False
        assert final.failure_category == FailureCategory.BREAKING_CHANGE
        assert final.failure_evidence is not None
        assert final.failure_evidence.failed_tests == ["api authentication test"]
        assert final.deterministic_gates is not None
        assert final.deterministic_gates.tests_passed is False
        assert final.deterministic_gates.scanner_execution_status == ScannerExecutionStatus.NOT_RUN
        assert "coverage is also incomplete" in final.retry_feedback

    def test_incomplete_scan_keeps_singleton_test_failure_actionable(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        results = _make_fully_populated_results(ok=True)
        results.tests = (False, "npm test FAILED (exit 1)")
        results.test_exit_code = 1
        results.test_raw_stdout = "not ok 1 - API auth test\n"
        results.scan_evidence = ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            complete=False,
            fallback_reason=ScanFallbackReason.TARGETED_SCAN_FAILED,
        )
        results.scan_target_failures_by_task = {
            "task-1": (
                ScanFallbackReason.TARGETED_SCAN_FAILED,
                "unresolved npm alias in lockfile closure",
            )
        }
        patches = self._patch_node(
            group=group,
            results=results,
            evaluations=[QAEvaluation(task_id="task-1", passed=True)],
        )

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        evaluation = result["qa_evaluations"]["task-1"]
        assert evaluation.passed is False
        assert evaluation.evidence_inconclusive is False
        assert evaluation.failure_category == FailureCategory.BREAKING_CHANGE
        assert evaluation.failure_evidence is not None
        assert evaluation.deterministic_gates is not None
        assert evaluation.deterministic_gates.tests_passed is False
        assert evaluation.test_attribution is not None
        assert evaluation.test_attribution.verdict.value == "responsible"
        assert result["scan_evidence"].complete is False

    def test_same_parent_group_tasks_receive_isolated_task_evaluations(self):
        group = _make_group()
        tasks = [
            RemediationTask(
                task_id="task-one",
                parent_group_id=group.group_id,
                strategy=RoutingStrategy.VERSION_BUMP,
                qa_policy=QAPolicy.VERSION_BUMP,
                status=TaskStatus.OPTIMISTICALLY_FIXED,
                instruction="Update the dependency.",
            ),
            RemediationTask(
                task_id="task-two",
                parent_group_id=group.group_id,
                strategy=RoutingStrategy.VERSION_BUMP,
                qa_policy=QAPolicy.VERSION_BUMP,
                status=TaskStatus.OPTIMISTICALLY_FIXED,
                instruction="Update the dependency in the second task.",
            ),
        ]
        state = _commit_tasks(_make_minimal_state(groups=[group]), tasks)
        evaluations = {
            "task-one": QAEvaluation(task_id="task-one", passed=True),
            "task-two": QAEvaluation(task_id="task-two", passed=True),
        }
        patches = self._patch_node(group=group, evaluations=evaluations)

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        assert set(result["qa_evaluations"]) == {"task-one", "task-two"}
        assert {
            task_id: evaluation.task_id for task_id, evaluation in result["qa_evaluations"].items()
        } == {"task-one": "task-one", "task-two": "task-two"}

    def test_new_identifiers_are_reported_without_task_failure(self):
        group = _make_group(cve_ids=["CVE-2021-23337"], ghsa_ids=[])
        state = _make_minimal_state(groups=[group])
        state["baseline_scan_identifiers"] = ["CVE-2021-23337"]
        results = _make_fully_populated_results(ok=True)
        results.scan = _SecurityScanResult(
            ok=True,
            summary="Dependency-Check found a new identifier.",
            remaining_identifiers=set(),
            found_identifiers={"CVE-2025-10001"},
            new_identifiers={"CVE-2025-10001"},
        )
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        patches = self._patch_node(
            group=group,
            results=results,
            evaluations=evaluations,
        )

        with (
            patches["sandbox"],
            patches["global_exec"],
            patches["investigators"],
            patch(
                "remediation_engine.agents.qa._build_qa_scan_targets",
                return_value=None,
            ),
        ):
            result = run_qa_critic_node(state)

        assert result["eval_status"] == "all_passed"
        assert result["qa_evaluations"]["task-1"].passed is True
        assert result["post_remediation_scan_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_status"] == "detected"
        assert "CVE-2025-10001" in result["qa_investigation_report"]

    def test_failures_detected_when_any_task_fails(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evaluations = [
            QAEvaluation(
                task_id="task-1",
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="CVE still present.",
            )
        ]
        patches = self._patch_node(group=group, evaluations=evaluations)

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        assert result["eval_status"] == "failures_detected"
        assert result["status"] == "qa_completed"
        assert result["qa_evaluations"]["task-1"].passed is False

    def test_missing_workspace_volume_returns_qa_failed(self):
        state = _make_minimal_state(workspace_volume=None)
        result = run_qa_critic_node(state)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"
        assert result["errors"]
        assert result["qa_evaluations"]["task-1"].contract_error is True
        assert result["qa_evaluations"]["task-1"].contract_error_reason

    def test_no_valid_groups_returns_all_passed_with_no_evals(self):
        state = _make_minimal_state(groups=[])
        result = run_qa_critic_node(state)

        assert result["status"] == "qa_completed"
        assert result["eval_status"] == "all_passed"
        assert result["qa_evaluations"] == {}

    def test_docker_unavailable_returns_qa_failed(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(side_effect=RuntimeError("Docker daemon unreachable"))
        mock_sandbox.__exit__ = MagicMock(return_value=None)

        with patch(
            "remediation_engine.agents.qa.DockerSandbox",
            return_value=mock_sandbox,
        ):
            result = run_qa_critic_node(state)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"
        assert result["qa_evaluations"]["task-1"].contract_error is True
        assert result["qa_evaluations"]["task-1"].contract_error_reason

    def test_changed_files_propagated_from_state(self):
        group = _make_group()
        state = _make_minimal_state(
            groups=[group],
            changed_files=["package.json", "src/app.ts"],
        )
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        patches = self._patch_node(group=group, evaluations=evaluations)

        with patches["sandbox"], patches["global_exec"], patches["investigators"]:
            result = run_qa_critic_node(state)

        assert "package.json" in result["changed_files"]
        assert "src/app.ts" in result["changed_files"]

    def test_loop_errors_propagated_to_result(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        investigations = {
            "task-1": GroupInvestigation(
                group_id=group.group_id,
                task_id="task-1",
                investigation_text="partial",
                tool_transcript="",
                errors=["Subagent exceeded max rounds."],
            )
        }
        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)
        results = _make_fully_populated_results(ok=True)

        with (
            patch(
                "remediation_engine.agents.qa.DockerSandbox",
                return_value=mock_sandbox,
            ),
            patch(
                "remediation_engine.agents.qa._run_global_execution",
                return_value=results,
            ),
            patch(
                "remediation_engine.agents.qa._qa_evaluator_module._run_individual_investigations",
                return_value=investigations,
            ),
        ):
            result = run_qa_critic_node(state)

        assert any("max rounds" in e.lower() or "exceeded" in e.lower() for e in result["errors"])


class TestQAMissingExecutionTools:
    """Verify qa_failed is returned when the agent skips a required tool."""

    def _run_with_partial_results(self, results: _QAExecutionResults):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)
        investigations = {
            "task-1": GroupInvestigation(
                group_id=group.group_id,
                task_id="task-1",
                investigation_text="partial",
                tool_transcript="",
                errors=[],
            )
        }
        with (
            patch(
                "remediation_engine.agents.qa.DockerSandbox",
                return_value=mock_sandbox,
            ),
            patch(
                "remediation_engine.agents.qa._run_global_execution",
                return_value=results,
            ),
            patch(
                "remediation_engine.agents.qa._qa_evaluator_module._run_individual_investigations",
                return_value=investigations,
            ),
        ):
            return run_qa_critic_node(state), group

    def test_missing_install_returns_qa_failed(self):
        results = _QAExecutionResults(
            scan=_SecurityScanResult(True, "ok", set(), set(), set()),
            tests=(True, "ok"),
        )
        result, _ = self._run_with_partial_results(results)
        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"
        assert result["qa_evaluations"]["task-1"].passed is False
        assert result["qa_evaluations"]["task-1"].contract_error is True
        assert result["qa_evaluations"]["task-1"].contract_error_reason
        assert "run_dependency_install" in " ".join(result["errors"])

    def test_missing_scan_returns_qa_failed(self):
        results = _QAExecutionResults(install=(True, "ok"), tests=(True, "ok"))
        result, _ = self._run_with_partial_results(results)
        assert result["status"] == "qa_failed"
        assert "run_security_scan" in " ".join(result["errors"])
        assert result["qa_evaluations"]["task-1"].contract_error is True
        assert result["qa_evaluations"]["task-1"].contract_error_reason

    def test_missing_tests_returns_qa_failed(self):
        results = _QAExecutionResults(
            install=(True, "ok"),
            scan=_SecurityScanResult(True, "ok", set(), set(), set()),
        )
        result, _ = self._run_with_partial_results(results)
        assert result["status"] == "qa_failed"
        assert result["qa_evaluations"]["task-1"].contract_error is True
        assert result["qa_evaluations"]["task-1"].contract_error_reason
        assert "run_unit_tests" in " ".join(result["errors"])

    def test_all_tools_missing_lists_all_in_error(self):
        result, _ = self._run_with_partial_results(_QAExecutionResults())
        assert result["status"] == "qa_failed"
        error_text = " ".join(result["errors"])
        assert "run_dependency_install" in error_text
        assert "run_security_scan" in error_text
        assert "run_unit_tests" in error_text


class TestRunQACriticNodeMapReduce:
    def _run(self, groups, global_results=None, investigations=None, evaluations=None):
        if global_results is None:
            global_results = _make_fully_populated_results(ok=True)
        task_ids = [f"task-{index}" for index, _ in enumerate(groups, start=1)]
        _add_verified_package_evidence(
            global_results,
            dict(zip(task_ids, groups, strict=True)),
        )
        if evaluations is None:
            evaluations = [QAEvaluation(task_id=task_id, passed=True) for task_id in task_ids]
        evaluations_by_task = (
            evaluations
            if isinstance(evaluations, dict)
            else {evaluation.task_id: evaluation for evaluation in evaluations}
        )
        if investigations is None:
            investigations = {
                task_id: GroupInvestigation(
                    group_id=group.group_id,
                    task_id=task_id,
                    investigation_text="",
                    tool_transcript="",
                    evaluation=evaluations_by_task.get(task_id),
                )
                for task_id, group in zip(task_ids, groups, strict=True)
            }
        else:
            for task_id, investigation in investigations.items():
                if investigation.evaluation is None:
                    investigation.evaluation = evaluations_by_task.get(task_id)

        mock_sb = MagicMock()
        mock_sb.__enter__ = MagicMock(return_value=mock_sb)
        mock_sb.__exit__ = MagicMock(return_value=None)
        manifest_payload = json.dumps({"dependencies": {"lodash": "4.17.21"}})
        lockfile_payload = json.dumps(
            {
                "lockfileVersion": 3,
                "packages": {
                    "": {"dependencies": {"lodash": "4.17.21"}},
                    "node_modules/lodash": {"version": "4.17.21"},
                },
            }
        )
        mock_sb.read_file.side_effect = lambda path: (
            lockfile_payload if str(path).endswith("package-lock.json") else manifest_payload
        )
        mock_sb.run.return_value.stdout = json.dumps(
            {"name": "workspace", "dependencies": {"lodash": {"version": "4.17.21"}}}
        )
        state = _make_minimal_state(groups=groups)
        with (
            patch("remediation_engine.agents.qa.DockerSandbox", return_value=mock_sb),
            patch(
                "remediation_engine.agents.qa._run_global_execution",
                return_value=global_results,
            ) as mg,
            patch(
                "remediation_engine.agents.qa._qa_evaluator_module._run_individual_investigations",
                return_value=investigations,
            ) as mm,
        ):
            result = run_qa_critic_node(state)
        return result, mg, mm

    def test_global_execution_called_once(self):
        _, mg, _ = self._run([_make_group()])
        mg.assert_called_once()

    def test_individual_investigations_called_once(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {
            "task-1": GroupInvestigation("g1", "ok", "", task_id="task-1"),
            "task-2": GroupInvestigation("g2", "ok", "", task_id="task-2"),
        }
        evals = [
            QAEvaluation(task_id="task-1", passed=True),
            QAEvaluation(task_id="task-2", passed=True),
        ]
        _, _, mm = self._run([g1, g2], investigations=invs, evaluations=evals)
        mm.assert_called_once()

    def test_investigation_report_in_output(self):
        g = _make_group()
        evals = [QAEvaluation(task_id="task-1", passed=True)]
        result, _, _ = self._run([g], evaluations=evals)
        report = json.loads(result["qa_investigation_report"])
        assert report["evaluations"]["task-1"]["passed"] is True

    def test_all_passed_status(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {
            "task-1": GroupInvestigation("g1", "ok", "", task_id="task-1"),
            "task-2": GroupInvestigation("g2", "ok", "", task_id="task-2"),
        }
        evals = [
            QAEvaluation(task_id="task-1", passed=True),
            QAEvaluation(task_id="task-2", passed=True),
        ]
        result, _, _ = self._run([g1, g2], investigations=invs, evaluations=evals)
        assert result["eval_status"] == "all_passed"
        assert result["status"] == "qa_completed"
        assert len(result["qa_evaluations"]) == 2

    def test_guardrails_fill_missing_eval(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {
            "task-1": GroupInvestigation("g1", "ok", "", task_id="task-1"),
            "task-2": GroupInvestigation("g2", "ok", "", task_id="task-2"),
        }
        evals = [QAEvaluation(task_id="task-1", passed=True)]
        result, _, _ = self._run([g1, g2], investigations=invs, evaluations=evals)
        assert "task-2" in result["qa_evaluations"]
        assert result["qa_evaluations"]["task-2"].passed is False
        assert result["qa_evaluations"]["task-2"].contract_error is True
        assert result["qa_evaluations"]["task-2"].contract_error_reason

    def test_map_errors_appear_in_output(self):
        g = _make_group()
        invs = {
            "task-1": GroupInvestigation(
                g.group_id,
                "fallback",
                "",
                task_id="task-1",
                errors=["investigator timed out"],
            )
        }
        result, _, _ = self._run([g], investigations=invs)
        assert any("investigator timed out" in e for e in result.get("errors", []))


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

    def test_investigator_prompt_excludes_unrelated_global_findings(self):
        group = _make_group(group_id="group-a")
        results = _make_fully_populated_results(ok=True)
        results.scan = _SecurityScanResult(
            ok=True,
            summary="scan ok",
            remaining_identifiers=set(),
            found_identifiers={"CVE-2025-10001"},
            new_identifiers={"CVE-2025-10001"},
        )
        results.scan_evidence_by_task["task-a"] = ODCScanEvidence(
            requested_scope="targeted",
            effective_scope="targeted",
            covered_task_ids=["task-a"],
            found_identifiers=["CVE-2025-20002"],
            new_identifiers=["CVE-2025-20002"],
            complete=True,
        )

        investigator_prompt = _build_individual_investigator_prompt(
            task_id="task-a",
            group=group,
            strategy="version_bump",
            results=results,
            group_remaining_ids=[],
            candidate_changed_files=[],
            action_summaries=[],
            task_scan_evidence=results.scan_evidence_by_task["task-a"],
        )
        assert "CVE-2025-20002" in investigator_prompt
        assert "CVE-2025-10001" not in investigator_prompt
        assert "Repository-wide scanner findings are intentionally excluded" in investigator_prompt


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


def test_dynamic_context_uses_committed_instruction_and_allows_authorized_alternative():
    """QA sees the sealed attempt rule rather than stale group advice."""
    group = _make_group(group_id="group-a")
    advisory = group.fix_plan.model_copy(
        update={
            "instruction": "Update @angular/platform-browser to 21.2.14.",
            "fixed_version": "21.2.14",
        }
    )
    group = group.model_copy(
        update={
            "vulnerable_component": "@angular/platform-browser",
            "issues": [
                issue.model_copy(update={"package_name": "@angular/platform-browser"})
                for issue in group.issues
            ],
            "fix_plan": advisory,
        }
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-2",
        task_id="task-a",
        task_revision=2,
        selected_version="21.2.17",
        allowed_target_versions=["21.2.17", "21.2.18"],
        target_package_name="@angular/platform-browser",
        target_dependency_type="dependencies",
        allowed_dependency_types=["dependencies"],
        instruction=(
            'Update "@angular/platform-browser" using the committed candidate rule: '
            "choose the lowest verified version from 21.2.17 or 21.2.18."
        ),
        instruction_digest="attempt-digest",
        dispatch_node="update_subagent",
    )

    context = _build_qa_dynamic_context(
        group=group,
        task_id="task-a",
        strategy="version_bump",
        results=_make_fully_populated_results(ok=True),
        group_remaining_ids=[],
        candidate_changed_files=["package.json", "package-lock.json"],
        action_summaries=[],
        qa_policy=QAPolicy.VERSION_BUMP,
        attempt_snapshot=snapshot,
        effective_target_version="21.2.18",
    )

    assert snapshot.instruction in context
    assert "21.2.14" not in context
    assert "21.2.17, 21.2.18" in context
    assert "Worker-reported effective target version: 21.2.18" in context
    assert "Committed selected version: 21.2.17" in context
    assert "Upgrade lodash to 4.17.21." not in context


class TestRunGlobalExecution:
    def test_calls_install_scan_and_tests_once(self):
        sandbox = MagicMock()
        target_ids = {"CVE-2021-0001"}
        with (
            patch(
                "remediation_engine.tools.test_execution._run_install",
                return_value=_install_outcome(True, "ok"),
            ) as mi,
            patch(
                "remediation_engine.tools.qa_scan._run_security_scan",
                return_value=_scan_outcome(True, "ok"),
            ) as ms,
            patch(
                "remediation_engine.tools.test_execution._run_unit_tests",
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
                "remediation_engine.tools.test_execution._run_install",
                return_value=_install_outcome(False, "fail", exit_code=1),
            ),
            patch(
                "remediation_engine.tools.qa_scan._run_security_scan",
                return_value=_scan_outcome(False, "fail"),
            ),
            patch(
                "remediation_engine.tools.test_execution._run_unit_tests",
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
                "remediation_engine.tools.test_execution._run_install",
                return_value=_install_outcome(True, "ok"),
            ),
            patch(
                "remediation_engine.tools.qa_scan._run_security_scan",
                return_value=_SecurityScanResult(True, "ok", set(), set(), set()),
            ) as scan,
            patch(
                "remediation_engine.tools.test_execution._run_unit_tests",
                return_value=_test_outcome(True, "ok"),
            ),
        ):
            _run_global_execution(sandbox, "vol", target_ids, baseline_ids)

        scan.assert_called_once_with(sandbox, "vol", target_ids, baseline_ids)

    def test_tests_run_even_when_install_fails(self):
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.tools.test_execution._run_install",
                return_value=_install_outcome(False, "FAILED", exit_code=1),
            ),
            patch(
                "remediation_engine.tools.qa_scan._run_security_scan",
                return_value=_scan_outcome(False, "fail"),
            ),
            patch(
                "remediation_engine.tools.test_execution._run_unit_tests",
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
                "remediation_engine.tools.test_execution._run_install",
                return_value=_install_outcome(True, "install ok"),
            ) as install,
            patch(
                "remediation_engine.tools.qa_scan._run_security_scan",
            ) as scan,
            patch(
                "remediation_engine.tools.test_execution._run_unit_tests",
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

    def test_python_language_and_pytest_evidence_are_explicit(self):
        prompt = _build_individual_investigator_prompt(
            task_id="task-python",
            group=_make_group(),
            strategy="code_workaround",
            results=_make_fully_populated_results(ok=False),
            group_remaining_ids=[],
            candidate_changed_files=["src/app.py"],
            action_summaries=[],
            project_language=ProjectLanguage.PYTHON,
        )

        assert "Canonical project language: python (Python)." in prompt
        assert "Interpret pytest failures as Python unit-test evidence." in prompt

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
        assert "responsible_group_ids must be a non-empty list" in lower
        assert "exact failed test names" in lower
        assert "do not blame every group in a batch" in lower

        description = _build_qa_terminal_tool().description
        for enum_type in (FailureCategory, SecurityReviewVerdict, TestAttributionVerdict):
            for member in enum_type:
                assert member.value in description
        assert "responsible_group_ids" in description
        assert "exact group IDs" in description

        attribution_schema = QATestAttribution.model_json_schema()
        assert (
            "Exact group_id values"
            in attribution_schema["properties"]["responsible_group_ids"]["description"]
        )
        assert (
            "Exact failing test names"
            in attribution_schema["properties"]["failed_tests"]["description"]
        )

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

    def test_prompt_reports_tuple_execution_statuses_and_singleton_scope(self):
        results = _make_fully_populated_results(ok=True)
        prompt = _build_individual_investigator_prompt(
            task_id="task-1",
            group=_make_group(),
            strategy="version_bump",
            results=results,
            group_remaining_ids=[],
            candidate_changed_files=[],
            action_summaries=[],
            singleton_scope=True,
        )

        assert "- Install: PASS" in prompt
        assert "- Unit tests: PASS" in prompt
        assert "- Dispatch scope: SINGLETON" in prompt
        assert "attribute" in prompt.lower()

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


def test_stale_or_missing_attempt_snapshot_is_rejected_before_qa_model_invocation():
    """Missing and wrong-revision snapshots cannot authorize the current task."""
    task = RemediationTask(
        task_id="task-1",
        parent_group_id="g1",
        strategy=RoutingStrategy.VERSION_BUMP,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        instruction="Current committed instruction.",
        current_attempt_id="attempt-current",
        task_revision=2,
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-current",
        task_id="task-1",
        task_revision=1,
        instruction="Stale committed instruction.",
        instruction_digest="stale-digest",
        dispatch_node="update_subagent",
    )

    for snapshots in ({"attempt-current": snapshot}, {}):
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop"
            ) as investigator,
        ):
            investigations = _run_individual_investigations(
                task_contexts=[QATaskContext(task=task, group=_make_group(group_id="g1"))],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=_make_fully_populated_results(ok=True),
                task_policies={"task-1": QAPolicy.VERSION_BUMP},
                attempt_snapshots_by_id=snapshots,
            )

        investigator.assert_not_called()
        evaluation = investigations["task-1"].evaluation
        assert evaluation is not None
        assert evaluation.contract_error is True
        assert "stale identity" in evaluation.contract_error_reason


def test_qa_evaluator_toolbelt_is_read_only():
    tools = build_qa_review_toolbelt(
        sandbox=MagicMock(),
        candidate_changed_files=["src/app.py"],
        host_repo_root="/tmp",
        results=_make_fully_populated_results(ok=True),
    )
    names = [tool.name for tool in tools]

    assert names == [
        "list_changed_files",
        "generate_workspace_diff",
        "read_file_context",
        "search_codebase_pattern",
        "inspect_ast_symbol",
        "query_qa_logs",
    ]
    assert not any(
        any(operation in name for operation in ("edit", "modify", "remove", "install", "run"))
        for name in names
    )


class TestRunIndividualInvestigations:
    def _lr(self, text="", errors=None, evaluation=None):
        from remediation_engine.runtime.agent_runtime import SubagentRuntimeResult

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
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                side_effect=[
                    self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
                    self._lr(evaluation=QACriticLLMOutput(task_id="task-2", passed=True)),
                ],
            ) as ml,
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
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
                project_language=ProjectLanguage.PYTHON,
            )
        assert all(
            "Canonical project language: python (Python)."
            in call.kwargs["initial_messages"][0].content
            for call in ml.call_args_list
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

    def test_committed_snapshot_and_matching_worker_version_reach_prompt(self):
        """The critic passes valid authorization and effective version to QA."""
        group = _make_group(group_id="g1")
        advisory = group.fix_plan.model_copy(
            update={
                "instruction": "Update @angular/platform-browser to 21.2.14.",
                "fixed_version": "21.2.14",
            }
        )
        group = group.model_copy(
            update={
                "vulnerable_component": "@angular/platform-browser",
                "issues": [
                    issue.model_copy(update={"package_name": "@angular/platform-browser"})
                    for issue in group.issues
                ],
                "fix_plan": advisory,
            }
        )
        task = _task_context(group, "task-1").task.model_copy(
            update={
                "current_attempt_id": "attempt-current",
                "task_revision": 2,
            }
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-current",
            task_id="task-1",
            task_revision=2,
            selected_version="21.2.17",
            allowed_target_versions=["21.2.17", "21.2.18"],
            target_package_name="@angular/platform-browser",
            instruction=(
                'Update "@angular/platform-browser" using the committed candidate rule: '
                "choose the lowest verified version from 21.2.17 or 21.2.18."
            ),
            instruction_digest="attempt-digest",
            dispatch_node="update_subagent",
        )
        worker_result = WorkerAttemptResult(
            attempt_id="attempt-current",
            task_id="task-1",
            task_revision=2,
            status=AgentActionStatus.SUCCESS,
            executed_versions=["21.2.18"],
            instruction_digest="attempt-digest",
            execution_diagnostics=WorkerExecutionDiagnostics(
                effective_target_version="21.2.18",
                executed_versions=["21.2.18"],
            ),
        )
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                return_value=self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
            ) as investigator,
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            _run_individual_investigations(
                task_contexts=[QATaskContext(task=task, group=group)],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": ["package.json", "package-lock.json"]},
                sandbox=MagicMock(),
                repo_root=None,
                results=_make_fully_populated_results(ok=True),
                task_policies={"task-1": QAPolicy.VERSION_BUMP},
                attempt_snapshots_by_id={"attempt-current": snapshot},
                worker_results_by_attempt={"attempt-current": worker_result},
            )

        prompt = investigator.call_args.kwargs["initial_messages"][1].content
        assert snapshot.instruction in prompt
        assert "21.2.14" not in prompt
        assert "Authorized target versions: 21.2.17, 21.2.18" in prompt
        assert "Worker-reported effective target version: 21.2.18" in prompt

    def test_worker_result_with_stale_identity_does_not_supply_effective_version(self):
        """Worker version evidence is ignored when its revision is stale."""
        task = _task_context(_make_group(group_id="g1"), "task-1").task.model_copy(
            update={"current_attempt_id": "attempt-current", "task_revision": 2}
        )
        snapshot = TaskAttemptSnapshot(
            attempt_id="attempt-current",
            task_id="task-1",
            task_revision=2,
            selected_version="1.2.0",
            allowed_target_versions=["1.2.0", "1.3.0"],
            instruction="Use 1.2.0 or the explicitly authorized compatible alternative 1.3.0.",
            instruction_digest="attempt-digest",
            dispatch_node="update_subagent",
        )
        stale_result = WorkerAttemptResult(
            attempt_id="attempt-current",
            task_id="task-1",
            task_revision=1,
            status=AgentActionStatus.SUCCESS,
            executed_versions=["1.3.0"],
            instruction_digest="attempt-digest",
            execution_diagnostics=WorkerExecutionDiagnostics(effective_target_version="1.3.0"),
        )
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                return_value=self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
            ) as investigator,
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            _run_individual_investigations(
                task_contexts=[QATaskContext(task=task, group=_make_group(group_id="g1"))],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=_make_fully_populated_results(ok=True),
                task_policies={"task-1": QAPolicy.VERSION_BUMP},
                attempt_snapshots_by_id={"attempt-current": snapshot},
                worker_results_by_attempt={"attempt-current": stale_result},
            )

        prompt = investigator.call_args.kwargs["initial_messages"][1].content
        assert "Worker-reported effective target version: (not available)" in prompt

    def test_investigator_receives_only_its_task_scanner_projection(self):
        group = _make_group(group_id="g1")
        context = _task_context(group, "task-1")
        results = _make_fully_populated_results(ok=True)
        results.scan = _SecurityScanResult(
            ok=True,
            summary="scan ok",
            remaining_identifiers=set(),
            found_identifiers={"CVE-2025-10001"},
            new_identifiers={"CVE-2025-10001"},
        )
        results.scan_evidence_by_task["task-1"] = ODCScanEvidence(
            requested_scope="targeted",
            effective_scope="targeted",
            covered_task_ids=["task-1"],
            found_identifiers=["CVE-2025-20002"],
            new_identifiers=["CVE-2025-20002"],
            complete=True,
        )
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                return_value=self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
            ) as investigator,
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            _run_individual_investigations(
                task_contexts=[context],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
                task_policies={"task-1": QAPolicy.VERSION_BUMP},
            )

        prompt = investigator.call_args.kwargs["initial_messages"][1].content
        assert "CVE-2025-20002" in prompt
        assert "CVE-2025-10001" not in prompt

    def test_missing_task_projection_does_not_fall_back_to_report_wide_scan(self):
        group = _make_group(group_id="g1", cve_ids=["CVE-2025-10001"])
        context = _task_context(group, "task-1")
        results = _make_fully_populated_results(ok=True)
        results.scan = _SecurityScanResult(
            ok=True,
            summary="scan ok",
            remaining_identifiers={"CVE-2025-10001"},
            found_identifiers={"CVE-2025-10001"},
            new_identifiers=set(),
        )
        results.scan_evidence = ODCScanEvidence(
            requested_scope="targeted",
            effective_scope="targeted",
            covered_task_ids=["task-1"],
            found_identifiers=["CVE-2025-10001"],
            remaining_target_identifiers=["CVE-2025-10001"],
            complete=True,
        )
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                return_value=self._lr(evaluation=QACriticLLMOutput(task_id="task-1", passed=True)),
            ) as investigator,
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            _run_individual_investigations(
                task_contexts=[context],
                task_strategies={"task-1": "version_bump"},
                action_summaries=[],
                changed_files_by_task={"task-1": []},
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
                task_policies={"task-1": QAPolicy.VERSION_BUMP},
            )

        prompt = investigator.call_args.kwargs["initial_messages"][1].content
        assert "Task-scoped scanner coverage: incomplete" in prompt
        assert "- Remaining identifier count: 0" in prompt

    def test_crash_produces_fallback(self):
        group = _make_group(group_id="g1")
        context = _task_context(group, "task-1")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
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
                "remediation_engine.agents.qa_evaluation.run_bounded_subagent_loop",
                return_value=self._lr(text=""),
            ),
            patch(
                "remediation_engine.agents.qa_evaluation.build_qa_review_toolbelt",
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

    def _apply(
        self,
        contexts,
        evaluations,
        results,
        investigations=None,
        *,
        singleton_scope=False,
    ):
        return _apply_guardrails(
            task_contexts=contexts,
            batch_result=evaluations,
            results=results,
            task_policies={context.task_id: context.task.qa_policy for context in contexts},
            investigations_by_task=investigations,
            singleton_scope=singleton_scope,
        )

    def test_valid_passes_through(self):
        context = self._context(_make_group())
        evaluations = [QAEvaluation(task_id="task-1", passed=True)]
        evals, errors = self._apply([context], evaluations, self._res())
        assert evals["task-1"].passed is True and not errors

    def test_singleton_failed_tests_are_attributed_to_active_task(self):
        group = _make_group()
        context = self._context(group)
        results = self._res()
        results.tests = (False, "npm test FAILED")
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="The test owner was inconclusive.",
            test_attribution=QATestAttribution(
                verdict=TestAttributionVerdict.INCONCLUSIVE,
            ),
        )

        evals, errors = self._apply(
            [context],
            [evaluation],
            results,
            singleton_scope=True,
        )

        attribution = evals["task-1"].test_attribution
        assert not errors
        assert attribution is not None
        assert attribution.verdict == TestAttributionVerdict.RESPONSIBLE
        assert attribution.responsible_group_ids == [group.group_id]
        assert attribution.failed_tests == ["deterministic unit test suite"]

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
