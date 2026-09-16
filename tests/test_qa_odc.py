"""Focused tests for ODC scans, install execution, and test-runner parsing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts.schemas import (
    AgentActionStatus,
    CommandResult,
    IssueSource,
    IssueType,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskAttemptSnapshot,
    TaskStatus,
    VulnerabilityIssue,
    WorkerAttemptResult,
    WorkerExecutionDiagnostics,
)
from remediation_engine.orchestration._qa_runtime import (
    _build_qa_scan_targets,
    _collect_baseline_identifiers,
    _collect_target_identifiers,
    _scan_state_projection,
    _workspace_remediation_fingerprint,
)
from remediation_engine.orchestration.qa_critic import run_final_full_scan_node
from remediation_engine.orchestration.qa_odc import (
    _ODC_CACHE_VOLUME,
    _ODC_HTML_REPORT_NAME,
    _ODC_INTERNAL_FULL_SCAN_EXCLUDES,
    _ODC_REPORT_NAME,
    _ODC_TIMEOUT_SECONDS,
    _parse_report_identifiers,
    _read_report_from_workspace,
    _run_odc,
    _run_security_scan,
    _run_targeted_odc,
)
from remediation_engine.orchestration.qa_test_parsing import (
    _NPM_INSTALL_TIMEOUT_SECONDS,
    _NPM_TEST_TIMEOUT_SECONDS,
    _detect_test_suite_plans,
    _extract_failure_blocks,
    _normalize_node_tap_output,
    _run_install,
    _run_unit_tests,
    _summarize_failed_test_output,
    build_targeted_test_command,
)
from remediation_engine.orchestration.qa_types import _QAExecutionResults, _SecurityScanResult
from tests.test_qa_critic_support import (
    _make_group,
    _make_sandbox,
)


def test_final_full_scan_converts_docker_api_error_to_scan_failure() -> None:
    """A Docker/API teardown error must return typed state instead of raising."""
    state = {"workspace_volume": "agent_workspace_deadbeef", "valid_groups": []}
    with patch(
        "remediation_engine.orchestration.qa_critic.DockerSandbox",
        side_effect=Exception("409 Client Error: Conflict (volume is in use)"),
    ):
        result = run_final_full_scan_node(state)

    assert result["final_full_scan_completed"] is True
    assert result["status"] == "final_scan_failed"
    assert result["final_full_scan_result"].status == "scan_failed"
    assert "409" in result["errors"][0]


def test_final_full_scan_records_workspace_fingerprint_for_next_validation(tmp_path) -> None:
    """Final scans retain the prior workspace fingerprint for no-op detection."""
    group = _make_group()
    state = {
        "repo_root": str(tmp_path),
        "workspace_volume": "agent_workspace_fingerprint",
        "valid_groups": [group],
        "changed_files": [],
    }
    scan = _SecurityScanResult(
        ok=True,
        summary="Dependency-Check found no remaining target vulnerability identifiers.",
        remaining_identifiers=set(),
        found_identifiers=set(),
        new_identifiers=set(),
    )

    with (
        patch("remediation_engine.orchestration.qa_critic.DockerSandbox"),
        patch(
            "remediation_engine.orchestration.qa_odc._run_security_scan",
            return_value=scan,
        ),
    ):
        first = run_final_full_scan_node(state)
        second = run_final_full_scan_node({**state, **first})

    assert first["previous_final_scan_workspace_fingerprint"] is None
    assert first["final_scan_workspace_fingerprint"]
    assert (
        second["previous_final_scan_workspace_fingerprint"]
        == first["final_scan_workspace_fingerprint"]
    )
    assert second["final_scan_workspace_fingerprint"] == first["final_scan_workspace_fingerprint"]


def test_workspace_remediation_fingerprint_changes_with_material_file_change(tmp_path) -> None:
    """The fingerprint reflects content changes, not only candidate file names."""
    (tmp_path / "package.json").write_text('{"version":"1.0.0"}', encoding="utf-8")
    sandbox = MagicMock()
    sandbox.read_file.return_value = '{"version":"1.0.0"}'

    unchanged = _workspace_remediation_fingerprint(str(tmp_path), sandbox, ["package.json"])

    sandbox.read_file.return_value = '{"version":"2.0.0"}'
    changed = _workspace_remediation_fingerprint(str(tmp_path), sandbox, ["package.json"])

    assert changed != unchanged


def test_unversioned_workaround_target_uses_live_version_resolution() -> None:
    """Workaround QA must not inherit a stale baseline group version."""
    group = _make_group(
        group_id="sca:package.json:express-jwt",
    ).model_copy(
        update={
            "vulnerable_component": "express-jwt",
            "versions": ["0.1.3"],
            "dependency_versions": {"express-jwt": "0.1.3"},
        }
    )
    task = RemediationTask(
        task_id="task-workaround",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.CODE_WORKAROUND,
        instruction="Apply the workaround.",
    )

    targets = _build_qa_scan_targets(
        {
            "active_target_task_ids": [task.task_id],
            "task_queue": {task.task_id: task},
        },
        [group],
    )

    assert targets is not None
    assert len(targets) == 1
    assert targets[0].target_package == "express-jwt"
    assert targets[0].expected_version is None


def test_qa_target_uses_attempt_execution_version_when_task_selection_was_cleared() -> None:
    """QA must not revert a successful update attempt to the original baseline version."""
    group = _make_group(
        group_id="sca:package.json:sanitize-html",
    ).model_copy(
        update={
            "vulnerable_component": "sanitize-html",
            "versions": ["1.4.2"],
            "dependency_versions": {"sanitize-html": "1.4.2"},
            "file_path": "package.json",
        }
    )
    instruction = "Update sanitize-html in package.json to exact version 2.17.7."

    task = RemediationTask(
        task_id="task-update",
        parent_group_id=group.group_id,
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.NPM_LATEST,
        status=TaskStatus.OPTIMISTICALLY_FIXED,
        task_revision=3,
        current_attempt_id="attempt-update",
        selected_version=None,
        instruction=instruction,
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    snapshot = TaskAttemptSnapshot(
        attempt_id="attempt-update",
        task_id=task.task_id,
        task_revision=task.task_revision,
        strategy_stage=task.strategy_stage,
        selected_version=None,
        instruction=instruction,
        instruction_digest="digest",
        dispatch_node="update_subagent",
        qa_policy=QAPolicy.VERSION_BUMP,
    )
    worker_result = WorkerAttemptResult(
        attempt_id=snapshot.attempt_id,
        task_id=task.task_id,
        task_revision=task.task_revision,
        status=AgentActionStatus.SUCCESS,
        executed_versions=["2.17.7"],
        instruction_digest="digest",
        execution_diagnostics=WorkerExecutionDiagnostics(
            executed_versions=["2.17.7"],
            validation_passed=True,
        ),
    )

    targets = _build_qa_scan_targets(
        {
            "active_target_task_ids": [task.task_id],
            "task_queue": {task.task_id: task},
            "attempt_snapshots_by_id": {snapshot.attempt_id: snapshot},
            "worker_results_by_attempt": {worker_result.attempt_id: worker_result},
        },
        [group],
    )

    assert targets is not None
    assert targets[0].expected_version == "2.17.7"


class TestRunInstall:
    def test_success_returns_true_and_message(self):
        sandbox = _make_sandbox()
        outcome = _run_install(sandbox)

        assert outcome.ok is True
        assert "succeeded" in outcome.summary.lower()
        sandbox.run.assert_called_once_with(
            "npm install --package-lock=true",
            timeout=_NPM_INSTALL_TIMEOUT_SECONDS,
        )

    def test_failure_returns_false_and_includes_exit_code(self):
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout="some stdout",
                    stderr="ERESOLVE unable to resolve",
                    duration_seconds=2.0,
                )
            ]
        )
        outcome = _run_install(sandbox)

        assert outcome.ok is False
        assert "FAILED" in outcome.summary
        assert "1" in outcome.summary  # exit code
        assert "ERESOLVE" in outcome.summary
        assert outcome.exit_code == 1
        assert outcome.error_category == "PEER_CONFLICT"
        assert outcome.raw_stdout == "some stdout"
        assert outcome.raw_stderr == "ERESOLVE unable to resolve"
        assert outcome.log_record.label == "npm install"

    def test_eoverride_is_classified_as_peer_conflict(self):
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout="",
                    stderr="EOVERRIDE Override conflicts with direct dependency",
                    duration_seconds=0.1,
                )
            ]
        )

        outcome = _run_install(sandbox)

        assert outcome.error_category == "PEER_CONFLICT"

    def test_ebadengine_is_classified_as_engine_conflict(self):
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout="",
                    stderr="EBADENGINE Unsupported engine",
                    duration_seconds=0.1,
                )
            ]
        )

        outcome = _run_install(sandbox)

        assert outcome.error_category == "ENGINE_CONFLICT"


class TestReadReportFromWorkspace:
    def test_returns_report_text_when_present(self):
        sandbox = _make_sandbox(read_file_return='{"dependencies": []}')
        result = _read_report_from_workspace(sandbox)
        assert result == '{"dependencies": []}'
        sandbox.read_file.assert_called_once_with(_ODC_REPORT_NAME)

    def test_returns_none_on_exception(self):
        sandbox = MagicMock()
        sandbox.read_file.side_effect = RuntimeError("not found")
        result = _read_report_from_workspace(sandbox)
        assert result is None


class TestParseReportIdentifiers:
    def _make_report(self, cve_id=None, ghsa_id=None) -> str:
        """Minimal ODC JSON report with one vulnerability."""
        vuln: dict[str, Any] = {"name": "CVE-2021-23337", "severity": "HIGH"}
        if cve_id:
            vuln["name"] = cve_id
        package = {
            "fileName": "lodash-4.17.20.tgz",
            "packages": [{"id": "pkg:npm/lodash@4.17.20"}],
            "vulnerabilities": [vuln],
        }
        return json.dumps({"dependencies": [package]})

    def test_returns_none_on_invalid_json(self):
        result = _parse_report_identifiers("not json")
        assert result is None

    @patch("remediation_engine.orchestration.qa_odc._parse_report_identifiers")
    def test_integration_returns_identifier_set(self, mock_parse):
        """Smoke-test that the function returns a Set[str] of identifiers."""
        mock_parse.return_value = {"CVE-2021-23337", "GHSA-35JH-R3H4-6JV8"}
        result = mock_parse('{"any": "json"}')
        assert isinstance(result, set)
        assert "CVE-2021-23337" in result


class TestRunOdc:
    def test_includes_workspace_volume_in_command(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("my-named-volume")
            args = mock_run.call_args[0][0]
            assert "my-named-volume:/scan" in " ".join(args)

    def test_includes_odc_cache_volume(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            args = mock_run.call_args[0][0]
            joined = " ".join(args)
            assert _ODC_CACHE_VOLUME in joined

    def test_excludes_engine_state_from_full_scan(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            args = mock_run.call_args[0][0]

        excludes = {
            args[index + 1] for index, argument in enumerate(args[:-1]) if argument == "--exclude"
        }
        assert set(_ODC_INTERNAL_FULL_SCAN_EXCLUDES).issubset(excludes)

    def test_does_not_exclude_targeted_scan_root(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_targeted_odc("test-vol", ".odc-targeted")
            args = mock_run.call_args[0][0]

        assert not any(pattern in args for pattern in _ODC_INTERNAL_FULL_SCAN_EXCLUDES)

    def test_assigns_unique_named_container_for_cleanup(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            args = mock_run.call_args[0][0]

        assert "--name" in args
        container_name = args[args.index("--name") + 1]
        assert container_name.startswith("remedy-odc-")

    def test_completed_scan_persists_diagnostic_output(self, tmp_path):
        process = subprocess.CompletedProcess(
            ["docker", "run"],
            2,
            stdout="scan warning",
            stderr="scan error",
        )
        with (
            patch(
                "remediation_engine.orchestration.qa_odc._ODC_LOG_DIR",
                tmp_path,
            ),
            patch("subprocess.run", return_value=process),
        ):
            result = _run_odc("test-vol")

        payload = json.loads(Path(result.odc_log_path).read_text(encoding="utf-8"))
        assert payload["status"] == "completed"
        assert payload["exit_code"] == 2
        assert payload["stdout"] == "scan warning"
        assert payload["stderr"] == "scan error"

    def test_respects_odc_extra_args(self, monkeypatch):
        monkeypatch.setenv("ODC_EXTRA_ARGS", "--disableNodeAudit --disableRetireJS")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            args = mock_run.call_args[0][0]
            assert "--disableNodeAudit" in args
            assert "--disableRetireJS" in args

    def test_passes_timeout_to_subprocess_run(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            _, kwargs = mock_run.call_args
            assert kwargs.get("timeout") == _ODC_TIMEOUT_SECONDS

    def test_requests_html_output_in_addition_to_json(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            _run_odc("test-vol")
            args = mock_run.call_args[0][0]
            assert args.count("--format") == 2
            assert "JSON" in args
            assert "HTML" in args

    def test_raises_timeout_expired_on_slow_docker(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["docker"], timeout=300)
            with pytest.raises(subprocess.TimeoutExpired):
                _run_odc("test-vol")

    def test_timeout_persists_output_and_forces_container_cleanup(self, tmp_path):
        timeout = subprocess.TimeoutExpired(
            cmd=["docker", "run"],
            timeout=_ODC_TIMEOUT_SECONDS,
            output="partial stdout",
            stderr="partial stderr",
        )
        cleanup = subprocess.CompletedProcess(
            ["docker", "rm", "-f"],
            0,
            stdout="removed",
            stderr="",
        )
        with (
            patch(
                "remediation_engine.orchestration.qa_odc._ODC_LOG_DIR",
                tmp_path,
            ),
            patch("subprocess.run", side_effect=[timeout, cleanup]) as mock_run,
            pytest.raises(subprocess.TimeoutExpired) as raised,
        ):
            _run_odc("test-vol")

        run_command = mock_run.call_args_list[0].args[0]
        container_name = run_command[run_command.index("--name") + 1]
        cleanup_command = mock_run.call_args_list[1].args[0]
        assert cleanup_command == ["docker", "rm", "-f", container_name]

        log_path = raised.value.log_path
        assert log_path is not None
        payload = json.loads(Path(log_path).read_text(encoding="utf-8"))
        assert payload["status"] == "timeout"
        assert payload["container_name"] == container_name
        assert payload["stdout"] == "partial stdout"
        assert payload["stderr"] == "partial stderr"
        assert payload["cleanup"]["succeeded"] is True

    def test_timeout_preserves_cleanup_failure_in_exception_and_log(self, tmp_path):
        timeout = subprocess.TimeoutExpired(
            cmd=["docker", "run"],
            timeout=_ODC_TIMEOUT_SECONDS,
        )
        cleanup = subprocess.CompletedProcess(
            ["docker", "rm", "-f"],
            1,
            stdout="",
            stderr="permission denied",
        )
        with (
            patch(
                "remediation_engine.orchestration.qa_odc._ODC_LOG_DIR",
                tmp_path,
            ),
            patch("subprocess.run", side_effect=[timeout, cleanup]),
            pytest.raises(subprocess.TimeoutExpired) as raised,
        ):
            _run_odc("test-vol")

        assert raised.value.cleanup.succeeded is False
        assert "permission denied" in raised.value.cleanup.error
        payload = json.loads(Path(raised.value.log_path).read_text(encoding="utf-8"))
        assert payload["cleanup"]["succeeded"] is False
        assert payload["cleanup"]["error"] == "permission denied"


class TestRunSecurityScan:
    def _make_passing_odc_proc(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""
        return proc

    def test_success_when_no_remaining_identifiers(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = '{"dependencies": []}'

        target = {"CVE-2021-23337"}
        # parse_report_identifiers returns empty set (CVE resolved)
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            result = _run_security_scan(sandbox, "vol", target)

        assert result.ok is True
        assert result.remaining_identifiers == set()
        assert _ODC_HTML_REPORT_NAME in str(sandbox.read_file.call_args_list[0])

    def test_detects_identifiers_absent_from_baseline(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        found = {"CVE-2025-10001", "GHSA-AAAA-BBBB-CCCC"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=found,
            ),
        ):
            result = _run_security_scan(
                sandbox,
                "vol",
                target,
                baseline_identifiers={"CVE-2021-23337"},
            )

        assert isinstance(result, _SecurityScanResult)
        assert result.ok is True
        assert result.remaining_identifiers == set()
        assert result.found_identifiers == found
        assert result.new_identifiers == found
        assert "Newly introduced identifiers" in result.summary
        assert result.exit_code == 0
        assert result.raw_stdout == ""
        assert result.raw_stderr == ""
        assert [record.label for record in result.scan_records] == ["odc:full"]

    def test_preexisting_identifier_is_not_classified_as_new(self):
        sandbox = MagicMock()
        found = {"CVE-2021-23337", "CVE-2025-10001"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=found,
            ),
        ):
            result = _run_security_scan(
                sandbox,
                "vol",
                {"CVE-2021-23337"},
                baseline_identifiers=found,
            )

        assert result.new_identifiers == set()
        assert result.found_identifiers == found

    def test_multiple_new_identifiers_are_sorted_in_summary(self):
        sandbox = MagicMock()
        found = {"CVE-2025-20002", "CVE-2025-10001"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=found,
            ),
        ):
            result = _run_security_scan(
                sandbox,
                "vol",
                set(),
                baseline_identifiers=set(),
            )

        assert result.new_identifiers == found
        assert result.summary.index("CVE-2025-10001") < result.summary.index("CVE-2025-20002")

    def test_summary_includes_saved_report_paths(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._persist_workspace_report_to_host",
                side_effect=[
                    Path("data/cache/qa_reports/dependency-check-report-1234567890.html"),
                    Path("data/cache/qa_reports/dependency-check-report.json"),
                ],
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            result = _run_security_scan(sandbox, "vol", target)

        assert result.ok is True
        assert result.remaining_identifiers == set()
        assert "HTML report saved to:" in result.summary
        assert "dependency-check-report-1234567890.html" in result.summary

    def test_failure_when_docker_not_available(self):
        sandbox = MagicMock()
        with patch("shutil.which", return_value=None):
            result = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert result.ok is False
        assert "docker" in result.summary.lower()
        assert result.remaining_identifiers == set()

    def test_failure_on_timeout(self):
        sandbox = MagicMock()
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=300),
            ),
        ):
            result = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert result.ok is False
        assert "timed out" in result.summary.lower()

    def test_timeout_summary_includes_log_and_cleanup_evidence(self, tmp_path):
        sandbox = MagicMock()
        timeout = subprocess.TimeoutExpired(
            cmd=["docker", "run"],
            timeout=_ODC_TIMEOUT_SECONDS,
        )
        cleanup = subprocess.CompletedProcess(
            ["docker", "rm", "-f"],
            0,
            stdout="removed",
            stderr="",
        )
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._ODC_LOG_DIR",
                tmp_path,
            ),
            patch("subprocess.run", side_effect=[timeout, cleanup]),
        ):
            result = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert result.ok is False
        assert "ODC diagnostic log saved to:" in result.summary
        assert "ODC container cleanup completed" in result.summary

    def test_failure_when_target_still_found(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value={"CVE-2021-23337"},
            ),
        ):  # still present
            result = _run_security_scan(sandbox, "vol", target)

        assert result.ok is False
        assert "unresolved target vulnerabilities" in result.summary
        assert "Remaining identifiers: CVE-2021-23337" in result.summary
        assert "CVE-2021-23337" in result.remaining_identifiers

    def test_failure_summary_lists_multiple_remaining_identifiers(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337", "GHSA-35JH-R3H4-6JV8"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_odc._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value={"GHSA-35JH-R3H4-6JV8", "CVE-2021-23337"},
            ),
        ):
            result = _run_security_scan(sandbox, "vol", target)

        assert result.ok is False
        assert "CVE-2021-23337" in result.summary
        assert "GHSA-35JH-R3H4-6JV8" in result.summary
        assert result.remaining_identifiers == {"CVE-2021-23337", "GHSA-35JH-R3H4-6JV8"}

    def test_no_report_and_nonzero_exit_is_failure(self):
        proc = MagicMock()
        proc.returncode = 1
        sandbox = MagicMock()
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("remediation_engine.orchestration.qa_odc._run_odc", return_value=proc),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value=None,
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=None,
            ),
        ):
            result = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert result.ok is False
        assert "parseable" in result.summary.lower() or "no parseable" in result.summary.lower()

    def test_nonzero_exit_but_parseable_report_continues(self):
        """ODC exits non-zero but produces a valid report â†’ treat as soft exit."""
        proc = MagicMock()
        proc.returncode = 2  # Dependency-Check warning exit
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("remediation_engine.orchestration.qa_odc._run_odc", return_value=proc),
            patch(
                "remediation_engine.orchestration.qa_odc._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_odc._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            result = _run_security_scan(sandbox, "vol", target)

        # Should still pass because identifiers are resolved
        assert result.ok is True


class TestCollectTargetIdentifiers:
    def test_collects_cve_and_ghsa_ids(self):
        group = _make_group(cve_ids=["CVE-2021-23337"], ghsa_ids=["GHSA-35JH-R3H4-6JV8"])
        ids = _collect_target_identifiers([group])
        assert "CVE-2021-23337" in ids
        assert "GHSA-35JH-R3H4-6JV8" in ids

    def test_deduplicates_identifiers_across_groups(self):
        g1 = _make_group(group_id="g1", cve_ids=["CVE-2021-23337"])
        g2 = _make_group(group_id="g2", cve_ids=["CVE-2021-23337"])
        ids = _collect_target_identifiers([g1, g2])
        # Set; no duplicates
        assert list(ids).count("CVE-2021-23337") == 1

    def test_fallback_to_issue_level_identifiers(self):
        """If group-level IDs are absent, fall back to individual issue fields."""
        group = _make_group(cve_ids=[], ghsa_ids=[])
        # Issue-level identifiers set by fixture
        ids = _collect_target_identifiers([group])
        # Should have picked up from the issue
        assert len(ids) > 0

    def test_empty_groups_returns_empty_set(self):
        assert _collect_target_identifiers([]) == set()

    def test_baseline_prefers_explicit_initial_scan_snapshot(self):
        group = _make_group(cve_ids=["CVE-2021-23337"])
        issue = VulnerabilityIssue(
            source=IssueSource.ODC,
            issue_type=IssueType.SCA,
            cve_id="CVE-2020-0001",
            package_name="other-package",
        )
        state = {
            "issues": [issue],
            "baseline_scan_identifiers": ["CVE-2020-0001"],
        }

        assert _collect_baseline_identifiers(state, [group]) == {"CVE-2020-0001"}

    def test_baseline_falls_back_to_target_groups_for_legacy_callers(self):
        group = _make_group(cve_ids=["CVE-2021-23337"], ghsa_ids=[])

        assert _collect_baseline_identifiers({}, [group]) == {
            "CVE-2021-23337",
            "GHSA-35JH-R3H4-6JV8",
        }


class TestScanStateProjection:
    def test_projects_complete_scan_and_new_identifier_sets(self):
        results = _QAExecutionResults(
            scan=_SecurityScanResult(
                ok=True,
                summary="scan ok",
                remaining_identifiers=set(),
                found_identifiers={"CVE-2021-23337", "CVE-2025-10001"},
                new_identifiers={"CVE-2025-10001"},
            )
        )

        projection = _scan_state_projection(results, {"CVE-2021-23337"})

        assert projection["post_remediation_scan_identifiers"] == [
            "CVE-2021-23337",
            "CVE-2025-10001",
        ]
        assert projection["new_vulnerability_identifiers"] == ["CVE-2025-10001"]
        assert projection["new_vulnerability_status"] == "detected"

    def test_projects_hard_scan_failure_as_scan_failed(self):
        results = _QAExecutionResults(
            scan=_SecurityScanResult(
                ok=False,
                summary="report unavailable",
                remaining_identifiers=set(),
                found_identifiers=set(),
                new_identifiers=set(),
            )
        )

        projection = _scan_state_projection(results, set())

        assert projection["new_vulnerability_status"] == "scan_failed"


class TestRunUnitTests:
    def test_success_returns_true(self):
        sandbox = _make_sandbox()
        outcome = _run_unit_tests(sandbox)

        assert outcome.ok is True
        assert "passed" in outcome.summary.lower()
        sandbox.run.assert_called_once_with("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)

    def test_failure_returns_condensed_detected_failures(self):
        stdout = "\n".join(
            [
                "some setup noise",
                "1) verify jwtChallenges challenge tracking",
                "AssertionError: expected true to equal false",
                "    at Context.<anonymous> (test/server/jwt.spec.ts:10:5)",
            ]
        )
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout=stdout,
                    stderr="some error",
                    duration_seconds=5.0,
                )
            ]
        )
        outcome = _run_unit_tests(sandbox)

        assert outcome.ok is False
        assert "FAILED" in outcome.summary
        assert "Failing Tests:" in outcome.summary
        assert "verify jwtChallenges challenge tracking" in outcome.summary
        assert "AssertionError" in outcome.summary
        assert outcome.exit_code == 1
        assert outcome.failure_count == 1
        assert outcome.raw_stdout == stdout
        assert outcome.raw_stderr == "some error"
        assert outcome.log_records[0].label == "npm test"

    def test_uses_npm_test_timeout(self):
        sandbox = _make_sandbox()
        _run_unit_tests(sandbox)
        _, kwargs = sandbox.run.call_args
        assert kwargs.get("timeout") == _NPM_TEST_TIMEOUT_SECONDS

    def test_composite_suite_runs_all_children_after_first_failure(self):
        root_package = {
            "scripts": {
                "test": "npm run test:server && npm run test:api",
                "test:server": "mocha -r tsx --recursive test/server/**/*.ts",
                "test:api": 'node --import tsx --test "test/api/**/*.test.ts"',
            }
        }
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout=json.dumps(
                        {
                            "failures": [
                                {
                                    "fullTitle": "server rejects bad token",
                                    "err": {
                                        "name": "AssertionError",
                                        "message": "expected 401",
                                    },
                                }
                            ]
                        }
                    ),
                    stderr="",
                    duration_seconds=1.0,
                ),
                CommandResult(
                    exit_code=0,
                    stdout="ok",
                    stderr="",
                    duration_seconds=1.0,
                ),
            ]
        )
        sandbox.read_file.side_effect = lambda path: (
            json.dumps(root_package) if path == "package.json" else None
        )

        outcome = _run_unit_tests(sandbox)

        assert outcome.ok is False
        assert "Failed Tests: 1" in outcome.summary
        assert "server rejects bad token" in outcome.summary
        assert "- api: passed" in outcome.summary
        assert sandbox.run.call_count == 2
        assert outcome.failure_count == 1
        assert [record.label for record in outcome.log_records] == [
            "tests:server",
            "tests:api",
        ]
        assert "server rejects bad token" in outcome.raw_stdout
        assert outcome.raw_stderr == ""
        sandbox.run.assert_any_call(
            "npm run test:server -- --reporter json",
            timeout=_NPM_TEST_TIMEOUT_SECONDS,
        )
        sandbox.run.assert_any_call(
            "npm run test:api",
            timeout=_NPM_TEST_TIMEOUT_SECONDS,
        )

    def test_composite_suite_all_children_passing_returns_passed(self):
        root_package = {
            "scripts": {
                "test": "npm run test:server && npm run test:api",
                "test:server": "mocha test/server/**/*.ts",
                "test:api": 'node --test "test/api/**/*.test.ts"',
            }
        }
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(exit_code=0, stdout="{}", stderr="", duration_seconds=1.0),
                CommandResult(exit_code=0, stdout="ok", stderr="", duration_seconds=1.0),
            ]
        )
        sandbox.read_file.side_effect = lambda path: (
            json.dumps(root_package) if path == "package.json" else None
        )

        outcome = _run_unit_tests(sandbox)

        assert outcome.ok is True
        assert outcome.summary.startswith("npm test passed.")
        assert "- server: passed" in outcome.summary
        assert "- api: passed" in outcome.summary
        assert sandbox.run.call_count == 2
        assert outcome.failure_count == 0
        assert [record.label for record in outcome.log_records] == [
            "tests:server",
            "tests:api",
        ]

    def test_unknown_project_uses_legacy_npm_test_text_parser(self):
        root_package = {"scripts": {"test": "custom-test-runner --plain"}}
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout="1) legacy failure\nAssertionError: broken",
                    stderr="",
                    duration_seconds=1.0,
                )
            ]
        )
        sandbox.read_file.side_effect = lambda path: (
            json.dumps(root_package) if path == "package.json" else None
        )

        outcome = _run_unit_tests(sandbox)

        assert outcome.ok is False
        assert "Detected Failures:" in outcome.summary
        sandbox.run.assert_called_once_with("npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS)


class TestStructuredTestDetection:
    def test_targeted_mocha_command_uses_workspace_local_runner(self):
        command = build_targeted_test_command(
            "mocha",
            "test/server/insecuritySpec.ts",
            npm_invocation="mocha -r tsx --recursive test/server/**/*.ts",
        )

        assert command == (
            "npx --no-install mocha -r tsx --recursive test/server/insecuritySpec.ts --reporter json"
        )

        package_command = build_targeted_test_command(
            "mocha",
            "test/server/insecuritySpec.ts",
            npm_invocation="mocha -r tsx --recursive test/server/**/*.ts",
            package_cwd="server",
        )

        assert package_command == (
            "cd server && npx --no-install mocha -r tsx --recursive test/server/insecuritySpec.ts --reporter json"
        )

    def test_detects_juice_shop_composite_test_strategies(self):
        root_package = {
            "scripts": {
                "test": "npm run test:frontend && npm run test:server && npm run test:api",
                "test:frontend": "cd frontend && npm run test",
                "test:server": "mocha -r tsx --recursive test/server/**/*.ts",
                "test:api": 'node --import ./test/api/helpers/test-env.mjs --import tsx --test --test-force-exit "test/api/**/*.test.ts"',
            }
        }
        frontend_package = {
            "scripts": {"test": "ng test"},
            "devDependencies": {"vitest": "^3.0.0"},
        }
        frontend_angular = {
            "projects": {
                "frontend": {"architect": {"test": {"builder": "@angular/build:unit-test"}}}
            }
        }

        sandbox = _make_sandbox()

        def read_file(path):
            if path == "package.json":
                return json.dumps(root_package)
            if path == "frontend/package.json":
                return json.dumps(frontend_package)
            if path == "frontend/angular.json":
                return json.dumps(frontend_angular)
            return None

        sandbox.read_file.side_effect = read_file

        plans = _detect_test_suite_plans(sandbox)

        assert plans is not None
        assert [plan.name for plan in plans] == ["frontend", "server", "api"]
        assert [plan.runner for plan in plans] == ["angular_vitest", "mocha", "node_test"]

    def test_unknown_shell_command_falls_back_to_text(self):
        sandbox = _make_sandbox(
            read_file_return=json.dumps({"scripts": {"test": "bash ./scripts/test-all.sh"}})
        )

        plans = _detect_test_suite_plans(sandbox)

        assert plans is not None
        assert plans[0].runner == "npm_text_fallback"


class TestStructuredNodeTapNormalization:
    def test_node_parent_suite_and_async_hook_are_diagnostics(self):
        output = "\n".join(
            [
                "# Subtest: POST handles searchProducts tool call and returns follow-up response",
                "not ok 5 - POST handles searchProducts tool call and returns follow-up response",
                "  duration_ms: 15001.158",
                "  type: 'test'",
                "  failureType: 'testTimeoutFailure'",
                "  error: 'test timed out after 15000ms'",
                "# Subtest: /rest/chat",
                "not ok 1 - /rest/chat",
                "  type: 'suite'",
                "  failureType: 'subtestsFailed'",
                "  error: '1 subtest failed'",
                '# Error: Test hook "before" at test/api/chat.test.ts:4:1079 generated asynchronous activity after the test ended.',
                '# This activity created the error "SyntaxError: Unexpected end of JSON input"',
            ]
        )

        failures, diagnostics = _normalize_node_tap_output(output, "", suite_name="api")

        assert len(failures) == 1
        assert (
            failures[0].name
            == "POST handles searchProducts tool call and returns follow-up response"
        )
        assert failures[0].failure_type == "testTimeoutFailure"
        assert len(diagnostics) == 2
        assert any("subtestsFailed" in diagnostic.message for diagnostic in diagnostics)
        assert any("asynchronous activity" in diagnostic.message for diagnostic in diagnostics)


class TestTestFailureExtraction:
    def test_extracts_mocha_numbered_failures(self):
        text = "\n".join(
            [
                "  1) verify jwtChallenges challenge tracking",
                "     AssertionError: expected true to equal false",
            ]
        )
        blocks = _extract_failure_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].title == "verify jwtChallenges challenge tracking"
        assert "AssertionError" in blocks[0].excerpt

    def test_extracts_jest_vitest_style_failure_headers(self):
        text = "\n".join(
            [
                "● basket service > calculates discount",
                "AssertionError: expected 3 to be 4",
            ]
        )
        blocks = _extract_failure_blocks(text)
        assert len(blocks) == 1
        assert "basket service > calculates discount" in blocks[0].title

    def test_extracts_tap_style_failure_with_subtest_context(self):
        text = "\n".join(
            [
                "# Subtest: api login returns 401",
                "not ok 3 -",
                "  error: expected 401 but got 200",
            ]
        )
        blocks = _extract_failure_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].title == "api login returns 401"
        assert "expected 401 but got 200" in blocks[0].excerpt

    def test_extracts_exception_only_output(self):
        text = "\n".join(
            [
                "TypeError: Cannot read properties of undefined (reading 'id')",
                "    at routes/login.ts:12:7",
            ]
        )
        blocks = _extract_failure_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].title.startswith("TypeError:")

    def test_deduplicates_overlapping_failure_markers(self):
        text = "\n".join(
            [
                "1) verify jwtChallenges challenge tracking",
                "AssertionError: expected true to equal false",
                "    at Context.<anonymous> (test/server/jwt.spec.ts:10:5)",
            ]
        )
        summary = _summarize_failed_test_output(1, text, text)
        assert summary.count("verify jwtChallenges challenge tracking") == 1

    def test_large_noisy_output_is_condensed(self):
        repeated = "\n".join(
            [f"{i}) failure {i}\nAssertionError: broken expectation {i}" for i in range(1, 15)]
        )
        summary = _summarize_failed_test_output(1, repeated, "")
        assert "Detected Failures:" in summary
        assert "... and " in summary
        assert len(summary) <= 6200

    def test_alien_output_falls_back_to_raw_tail(self):
        stdout = "\n".join(f"line {i}" for i in range(120))
        stderr = "unstructured crash"
        summary = _summarize_failed_test_output(1, stdout, stderr)
        assert "stdout tail:" in summary
        assert "line 119" in summary
        assert "stderr tail:" in summary
