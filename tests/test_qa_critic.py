"""QA critic node integration tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from remediation_engine.contracts.schemas import (
    FailureCategory,
    FixPlanStatus,
    NoFixMitigationStage,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    TaskAttemptSnapshot,
    TaskStatus,
)
from remediation_engine.orchestration.qa_critic import run_qa_critic_node
from remediation_engine.orchestration.qa_evaluator import GroupInvestigation
from remediation_engine.orchestration.qa_types import (
    _QAExecutionResults,
    _QAPackageState,
    _SecurityScanResult,
)
from tests.test_qa_critic_support import (
    _make_fully_populated_results,
    _make_group,
    _make_minimal_state,
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
                "remediation_engine.orchestration.qa_critic.DockerSandbox",
                return_value=mock_sandbox,
            ),
            "global_exec": patch(
                "remediation_engine.orchestration.qa_critic._run_global_execution",
                return_value=results,
            ),
            "investigators": patch(
                "remediation_engine.orchestration.qa_critic._qa_evaluator_module._run_individual_investigations",
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
                "remediation_engine.orchestration.qa_critic._collect_group_package_state",
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
                "remediation_engine.orchestration.qa_critic._build_qa_scan_targets",
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
            "remediation_engine.orchestration.qa_critic.DockerSandbox",
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
                "remediation_engine.orchestration.qa_critic.DockerSandbox",
                return_value=mock_sandbox,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_global_execution",
                return_value=results,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._qa_evaluator_module._run_individual_investigations",
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
                "remediation_engine.orchestration.qa_critic.DockerSandbox",
                return_value=mock_sandbox,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_global_execution",
                return_value=results,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._qa_evaluator_module._run_individual_investigations",
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
            patch("remediation_engine.orchestration.qa_critic.DockerSandbox", return_value=mock_sb),
            patch(
                "remediation_engine.orchestration.qa_critic._run_global_execution",
                return_value=global_results,
            ) as mg,
            patch(
                "remediation_engine.orchestration.qa_critic._qa_evaluator_module._run_individual_investigations",
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
