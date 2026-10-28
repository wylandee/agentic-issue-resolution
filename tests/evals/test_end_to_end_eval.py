"""Offline contracts for the complete-system evaluation boundary."""

from __future__ import annotations

import pytest

from remediation_engine.api import RemediationRequest, RemediationResult
from remediation_engine.evals import end_to_end as e2e
from remediation_engine.evals.db import EvalDatabase
from remediation_engine.evals.run_evidence import (
    RunEvidence,
    assess_contracts,
    judge_payload,
    load_run,
    normalize_run,
)
from remediation_engine.evals.trajectories import parse_trajectory_dict
from remediation_engine.settings import AppSettings
from tests.evals.conftest import EvalSettings, record_eval_result


@pytest.fixture
def evidence():
    return RunEvidence(
        source_run_id="source-1",
        request={"issues": [{"cve_id": "CVE-test"}]},
        final_state={
            "status": "completed",
            "task_queue": {
                "task-1": {
                    "task_id": "task-1",
                    "current_attempt_id": "attempt-1",
                    "task_revision": 1,
                    "status": "qa_passed",
                    "strategy": "version_bump",
                }
            },
            "attempt_snapshots_by_id": {
                "attempt-1": {
                    "task_id": "task-1",
                    "task_revision": 1,
                    "qa_policy": {"policy": "standard"},
                }
            },
            "worker_results_by_attempt": {
                "attempt-1": {
                    "task_id": "task-1",
                    "attempt_id": "attempt-1",
                    "task_revision": 1,
                    "status": "success",
                    "executed_versions": [],
                }
            },
            "qa_results_by_attempt": {
                "attempt-1": {
                    "task_id": "task-1",
                    "attempt_id": "attempt-1",
                    "task_revision": 1,
                    "qa_policy_source": "attempt_snapshot",
                    "qa_policy": {"policy": "standard"},
                    "evaluation": {"task_id": "task-1", "passed": True},
                }
            },
            "final_full_scan_result": {
                "coverage_scope": "incremental_dependency_inputs",
                "coverage_complete": True,
            },
            "changed_files": [],
            "diff": "",
            "report_markdown": "One task passed current-attempt QA.",
        },
    )


class Metric:
    def __init__(self, name, score=1.0, error=None):
        self.__name__ = name
        self.score = score
        self.threshold = 0.7
        self.error = error
        self.reason = "measured"
        self.evaluation_model = "offline-judge"
        self.evaluation_cost = 0.01
        self.calls = []

    def measure(self, case):
        self.calls.append(case)
        if isinstance(self.error, Exception):
            raise self.error
        return self.score

    def is_successful(self):
        return self.score >= self.threshold


def metrics(scores=(1.0, 1.0)):
    return [
        Metric(name, score)
        for name, score in zip(
            ("Task Completion", "System Safety and Contract Adherence"),
            scores,
            strict=True,
        )
    ]


def test_complete_run_uses_original_goal_and_independent_evidence(evidence):
    selected = metrics()
    record = e2e.evaluate_run(evidence, metrics_factory=lambda _: selected)
    assert record.status == "PASSED"
    assert record.additional_metadata["assessment"]["remediation_complete"]
    assert "CVE-test" in selected[0].calls[0].input
    assert "qa_results_by_attempt" in selected[0].calls[0].actual_output
    assert selected[0].calls[0].tools_called == []
    assert record.additional_metadata["source_run_id"] == "source-1"
    assert len(record.metrics) == 2


def test_whole_run_metrics_use_structured_safety_scoring(monkeypatch):
    import deepeval.metrics

    class Metric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.threshold = kwargs["threshold"]
            self.name = kwargs.get("name")

    monkeypatch.setattr(deepeval.metrics, "TaskCompletionMetric", Metric)
    metrics = e2e.build_metrics("gpt-4o")
    assert [metric.threshold for metric in metrics] == [0.70, 0.70]
    assert metrics[0].kwargs["model"] == "gpt-4o"
    assert metrics[1].name == "System Safety and Contract Adherence"
    assert metrics[1].evaluation_model == "gpt-4o"


@pytest.mark.eval
def test_end_to_end_live_selected_run(
    e2e_source: dict[str, object] | None,
    eval_settings: EvalSettings,
    request: pytest.FixtureRequest,
) -> None:
    """Score only explicit complete runs in the live suite and persist their metrics."""
    if e2e_source is None:
        pytest.skip("Select a complete run with --e2e-trajectory or explicit fresh-run inputs")
    if not eval_settings.is_live:
        pytest.skip("Whole-run judging requires --run-eval-live")
    if not eval_settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live end-to-end evaluation")

    from remediation_engine.evals.models import EvalTestCaseRecord

    try:
        if e2e_source["source"] == "recorded":
            evidence = load_run(e2e_source["path"])
        else:
            from remediation_engine.cli import _load_issues

            repo = e2e_source["repo"].resolve()
            issues = _load_issues(e2e_source["input"], e2e_source["format"])
            remediation_request = RemediationRequest(repo_root=repo, issues=issues)
            evidence = e2e.run_fresh(
                remediation_request,
                settings=AppSettings.from_env(),
            )
        record = e2e.evaluate_run(
            evidence,
            judge_model=eval_settings.judge_model,
            judge_llm=eval_settings.judge_llm,
            strict=bool(request.config.getoption("--e2e-strict")),
            secrets=(eval_settings.openai_api_key,),
        )
    except Exception as exc:
        source = str(e2e_source.get("path") or e2e_source.get("repo"))
        record = EvalTestCaseRecord(
            test_name=f"end_to_end[{source}]",
            suite="end_to_end",
            status="ERROR",
            error_message=str(exc),
            additional_metadata={"source": e2e_source["source"]},
        )

    record_eval_result(record)
    if record.status in {"FAILED", "ERROR"}:
        pytest.fail(record.error_message or f"End-to-end evaluation status: {record.status}")


@pytest.mark.parametrize("strict,status", [(False, "PASSED"), (True, "FAILED")])
def test_thresholds_are_diagnostic_unless_strict(evidence, strict, status):
    selected = metrics((0.2, 0.3))
    record = e2e.evaluate_run(evidence, strict=strict, metrics_factory=lambda _: selected)
    assert record.status == status
    assert len(record.metrics) == 2
    assert all(len(metric.calls) == 1 for metric in selected)


@pytest.mark.parametrize("status", ["unfixable", "inconclusive", "needs_retry"])
def test_safe_partial_run_is_not_a_contract_failure(evidence, status):
    evidence.final_state["status"] = "completed_with_errors"
    evidence.final_state["task_queue"]["task-1"]["status"] = status
    record = e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics((0.2, 1)))
    assert record.status == "PASSED"
    assert not record.additional_metadata["assessment"]["remediation_complete"]
    assert record.additional_metadata["assessment"]["unresolved"]


def test_safe_worker_surrender_is_not_reported_as_success(evidence):
    evidence.final_state["status"] = "completed_with_errors"
    evidence.final_state["task_queue"]["task-1"]["status"] = "unfixable"
    evidence.final_state["worker_results_by_attempt"]["attempt-1"]["status"] = "surrender"
    record = e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics())
    assert record.status == "PASSED"
    assert record.additional_metadata["assessment"]["remediation_complete"] is False
    assert not record.additional_metadata["assessment"]["violations"]


def test_empty_request_can_complete_without_a_patch(evidence):
    evidence.request["issues"] = []
    evidence.request["valid_groups"] = []
    evidence.final_state.update(
        status="no_work",
        task_queue={},
        changed_files=[],
        diff="",
        report_markdown="No actionable findings were present, so no remediation patch was needed.",
    )
    goal, _, _, _ = judge_payload(evidence, assess_contracts(evidence))
    assert "no actionable findings" in goal.casefold()
    assert "does not need a patch or QA" in goal
    record = e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics())
    assert record.additional_metadata["assessment"]["remediation_complete"] is True


def test_workaround_with_remaining_identifiers_is_accepted_by_policy(evidence):
    evidence.final_state["task_queue"]["task-1"].update(
        strategy="code_workaround", qa_policy="no_fix_code_removal"
    )
    evidence.final_state["qa_results_by_attempt"]["attempt-1"]["evaluation"].update(
        passed=True, remaining_vulnerabilities=["CVE-test"]
    )
    assert assess_contracts(evidence).remediation_complete


@pytest.mark.parametrize("mutation", ["stale", "failed_qa", "patch", "cleanup", "false_success"])
def test_contract_failures_cannot_be_overruled_by_judge(evidence, mutation):
    if mutation == "stale":
        evidence.final_state["qa_results_by_attempt"]["attempt-1"]["task_revision"] = 0
    elif mutation == "failed_qa":
        evidence.final_state["qa_results_by_attempt"]["attempt-1"]["evaluation"]["passed"] = False
    elif mutation == "patch":
        evidence.final_state["changed_files"] = ["invented.py"]
    elif mutation == "cleanup":
        evidence.source = "fresh"
        evidence.host_unchanged = True
        evidence.volumes_removed = False
    else:
        evidence.final_state["task_queue"]["task-1"]["status"] = "unfixable"
    assert e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics()).status == "FAILED"


@pytest.mark.parametrize(
    "mutation,expected_fragment",
    [
        ("stale_worker", "Stale or unrelated current worker"),
        ("atomic_rollback", "Atomic candidate rollback was not verified"),
        ("unsupported_report", "Report claims every finding is resolved"),
    ],
)
def test_safety_contracts_catch_stale_workers_rollback_and_unsupported_claims(
    evidence, mutation, expected_fragment
):
    if mutation == "stale_worker":
        evidence.final_state["worker_results_by_attempt"]["attempt-1"]["task_revision"] = 0
    elif mutation == "atomic_rollback":
        evidence.final_state["delta_isolation_by_cluster"] = {
            "cluster-1": {"candidate_restored": False, "status": "INCONCLUSIVE"}
        }
    else:
        evidence.final_state["status"] = "completed_with_errors"
        evidence.final_state["task_queue"]["task-1"]["status"] = "unfixable"
        evidence.final_state["report_markdown"] = "All findings were successfully resolved."

    assessment = assess_contracts(evidence)
    assert any(expected_fragment in violation for violation in assessment.violations)
    assert not assessment.remediation_complete


def test_worker_execution_outside_committed_candidates_is_a_contract_failure(evidence):
    snapshot = evidence.final_state["attempt_snapshots_by_id"]["attempt-1"]
    snapshot.update(allowed_target_versions=["2.0"], selected_version="2.0")
    evidence.final_state["worker_results_by_attempt"]["attempt-1"]["executed_versions"] = ["9.0"]
    assessment = assess_contracts(evidence)
    assert any("outside the committed candidate set" in item for item in assessment.violations)


@pytest.mark.parametrize(
    "scan",
    [
        None,
        {"coverage_scope": "incremental_dependency_inputs", "coverage_complete": False},
        {
            "coverage_scope": "incremental_dependency_inputs",
            "coverage_complete": True,
            "new_identifiers": ["CVE-new"],
        },
    ],
)
def test_unverified_scans_and_new_findings_prevent_completion(evidence, scan):
    evidence.final_state["status"] = "completed_with_errors"
    evidence.final_state["final_full_scan_result"] = scan
    assert not assess_contracts(evidence).remediation_complete


def test_no_work_run_requires_no_invented_tool_sequence():
    evidence = normalize_run(
        parse_trajectory_dict(
            {
                "trace_id": "no-work",
                "initial_state": {"issues": []},
                "final_state": {
                    "status": "completed",
                    "task_queue": {},
                    "diff": "",
                    "changed_files": [],
                    "initial_triage_executed": True,
                    "report_markdown": "No actionable findings.",
                },
            }
        )
    )
    assert assess_contracts(evidence).remediation_complete
    assert e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics()).status == "PASSED"


def test_judge_error_is_not_a_negative_measurement_and_other_metrics_run(evidence):
    selected = metrics()
    selected[0].error = RuntimeError("provider failed")
    result = e2e.evaluate_run(evidence, metrics_factory=lambda _: selected)
    assert result.status == "ERROR"
    assert len(result.metrics) == 1
    assert result.metrics[0].metric_name == "System Safety and Contract Adherence"
    assert all(metric.calls for metric in selected)


def test_missing_report_is_an_error(evidence):
    evidence.final_state.pop("report_markdown")
    assert e2e.evaluate_run(evidence, metrics_factory=lambda _: metrics()).status == "ERROR"


def test_nested_and_exact_credentials_are_redacted_before_judging_and_storage(evidence):
    secret = "nonstandard-configured-credential"
    evidence.final_state["errors"] = [secret, {"api_key": "nested-secret"}]
    evidence.final_state["status"] = "completed_with_errors"
    selected = metrics()
    result = e2e.evaluate_run(evidence, metrics_factory=lambda _: selected, secrets=[secret])
    assert secret not in result.model_dump_json()
    assert "nested-secret" not in selected[0].calls[0].actual_output


def test_compaction_keeps_all_terminal_failures(evidence):
    from remediation_engine.evals.run_evidence import ExecutionEvent

    evidence.events = [
        ExecutionEvent(run_id="tool", sequence=1, name="test", outputs="x" * 5000, error="failure")
    ]
    _, outcome, tools, omitted = judge_payload(evidence, assess_contracts(evidence))
    assert "task-1" in outcome and "attempt-1" in outcome
    assert omitted["tool_output_characters_omitted"] > 0
    assert tools[0]["output"]["error"] == "failure"


def test_incomplete_trajectory_is_rejected(tmp_path):
    path = tmp_path / "partial.md"
    path.write_text("# Incomplete trajectory")
    with pytest.raises(ValueError, match="complete trajectory"):
        load_run(path)


def test_cli_requires_explicit_inputs():
    with pytest.raises(SystemExit):
        e2e.main(["score"])


def test_cli_scores_explicit_cohort_and_persists_judge_errors(evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(e2e, "load_run", lambda _: evidence)
    original = e2e.evaluate_run
    monkeypatch.setattr(
        e2e,
        "evaluate_run",
        lambda ev, **kw: original(ev, metrics_factory=lambda _: metrics(), **kw),
    )
    path = tmp_path / "eval.db"
    assert (
        e2e.main(
            ["score", "--trajectory", "one.md", "--trajectory", "two.md", "--db-path", str(path)]
        )
        == 0
    )
    database = EvalDatabase(path)
    run = database.get_runs()[0]
    assert run["suite_name"] == "end_to_end" and run["total_tests"] == 2
    assert len(database.get_test_cases(run["run_id"])[0]["metrics"]) == 2
    monkeypatch.setattr(
        e2e, "load_run", lambda _: (_ for _ in ()).throw(ValueError("bad evidence"))
    )
    assert e2e.main(["score", "--trajectory", "bad.md", "--db-path", str(path)]) == 2


@pytest.mark.parametrize("strict,expected_exit", [(False, 0), (True, 1)])
def test_cli_exit_policy_separates_diagnostic_and_strict_thresholds(
    evidence, monkeypatch, tmp_path, strict, expected_exit, capsys
):
    monkeypatch.setattr(e2e, "load_run", lambda _: evidence)
    monkeypatch.setattr(e2e, "build_metrics", lambda _, **kwargs: metrics((0.2, 0.3)))
    args = ["score", "--trajectory", "selected.md", "--db-path", str(tmp_path / "eval.db")]
    if strict:
        args.append("--strict")
    assert e2e.main(args) == expected_exit
    assert "Task Completion: 0.20" in capsys.readouterr().out


def test_cli_returns_error_for_judge_failure(evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(e2e, "load_run", lambda _: evidence)
    selected = metrics()
    selected[0].error = RuntimeError("provider failure")
    monkeypatch.setattr(e2e, "build_metrics", lambda _: selected)
    assert (
        e2e.main(["score", "--trajectory", "selected.md", "--db-path", str(tmp_path / "eval.db")])
        == 2
    )


def test_fresh_path_calls_api_once_and_checks_isolation(evidence, monkeypatch, tmp_path):
    state = evidence.final_state
    path = tmp_path / "trajectory.md"
    doc = parse_trajectory_dict(
        {
            "trace_id": "fresh",
            "initial_state": {"issues": []},
            "final_state": state,
            "spans": [
                {
                    "run_id": "builder",
                    "name": "workspace_builder",
                    "run_type": "chain",
                    "outputs": {"workspace_volume": "temporary-volume"},
                }
            ],
        }
    )
    calls = []
    monkeypatch.setattr(e2e, "repository_fingerprint", lambda _: "same")
    monkeypatch.setattr(e2e, "parse_trajectory_markdown", lambda _: doc)
    monkeypatch.setattr(
        e2e,
        "run_remediation",
        lambda request, settings: (
            calls.append(request)
            or RemediationResult(status="completed", raw_state=state, trajectory_path=str(path))
        ),
    )
    monkeypatch.setattr(e2e, "verify_volume_removal", lambda names: names == {"temporary-volume"})
    result = e2e.run_fresh(RemediationRequest(repo_root=tmp_path), settings=AppSettings())
    assert len(calls) == 1
    assert result.host_unchanged is True and result.volumes_removed is True
    assert result.source == "fresh"


def test_docker_volume_check_is_read_only_and_reports_retained_volumes(monkeypatch):
    import docker

    class Volumes:
        def __init__(self, *, present):
            self.present = present

        def get(self, _name):
            if self.present:
                return object()
            raise docker.errors.NotFound("removed")

    class Client:
        def __init__(self, *, present):
            self.volumes = Volumes(present=present)
            self.closed = False

        def close(self):
            self.closed = True

    for present, expected in ((False, True), (True, False)):
        client = Client(present=present)
        monkeypatch.setattr(docker, "from_env", lambda client=client: client)
        assert e2e.verify_volume_removal({"recorded-volume"}) is expected
        assert client.closed


def test_repository_fingerprint_detects_dirty_and_untracked_edits(tmp_path):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    path = tmp_path / "source.py"
    path.write_text("before")
    before = e2e.repository_fingerprint(tmp_path)
    path.write_text("after")
    assert e2e.repository_fingerprint(tmp_path) != before
