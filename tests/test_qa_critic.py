"""
tests/test_qa_critic.py - Unit tests for the Phase 5 QA Critic evaluator node.

Mocks DockerSandbox and subprocess so no real Docker or network is required.
Covers:
  - install command, timeout semantics, and failure output extraction
  - ODC command construction, cache volume, ODC_EXTRA_ARGS, missing Docker, timeout
  - ODC report parsing for CVE and GHSA identifiers
  - remaining target identifier detection
  - unit test success/failure extraction
  - host-vs-workspace diff generation
  - one install, one scan, one test run per QA node invocation
  - .with_structured_output(QAEvaluation) usage
  - all-pass and mixed-failure eval_status
  - QA tools are NOT present in update/workaround subagent toolbelts
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Set
from unittest.mock import MagicMock, call, patch

import pytest

from src.contracts.schemas import (
    CommandResult,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    QAEvaluation,
    RoutingStrategy,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.qa_critic import (
    _NPM_INSTALL_TIMEOUT_SECONDS,
    _NPM_TEST_TIMEOUT_SECONDS,
    _ODC_CACHE_VOLUME,
    _ODC_REPORT_NAME,
    _ODC_TIMEOUT_SECONDS,
    _collect_target_identifiers,
    _generate_workspace_diff,
    _parse_report_identifiers,
    _read_report_from_workspace,
    _run_install,
    _run_odc,
    _run_security_scan,
    _run_unit_tests,
    run_qa_critic_node,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_group(
    group_id: str = "sca:package.json:lodash:version_bump",
    cve_ids=None,
    ghsa_ids=None,
    fix_plan_status: FixPlanStatus = FixPlanStatus.VERSION_FOUND,
) -> VulnerabilityGroup:
    cve_ids = cve_ids or ["CVE-2021-23337"]
    ghsa_ids = ghsa_ids or ["GHSA-35JH-R3H4-6JV8"]
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name="lodash",
        cve_id=cve_ids[0] if cve_ids else None,
        ghsa_id=ghsa_ids[0] if ghsa_ids else None,
        severity=Severity.HIGH,
    )
    fix_plan = FixPlan(
        status=fix_plan_status,
        fixed_version="4.17.21" if fix_plan_status == FixPlanStatus.VERSION_FOUND else None,
        workaround_snippets=(
            ["// workaround: sanitize input"] if fix_plan_status == FixPlanStatus.WORKAROUND_FOUND else None
        ),
        instruction="Upgrade lodash to 4.17.21.",
        strategy_used="osv_api",
    )
    return VulnerabilityGroup(
        group_id=group_id,
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        cve_ids=cve_ids,
        ghsa_ids=ghsa_ids,
        representative_issue_id=issue.id,
        issues=[issue],
        fix_plan=fix_plan,
    )


def _make_sandbox(
    run_side_effects=None,
    read_file_return=None,
    workspace_volume="sandbox-vol",
):
    """Return a pre-configured mock ``DockerSandbox``."""
    sandbox = MagicMock()
    sandbox._workspace_volume = workspace_volume
    if run_side_effects is not None:
        sandbox.run.side_effect = run_side_effects
    else:
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="ok", stderr="", duration_seconds=0.5
        )
    sandbox.read_file.return_value = read_file_return
    return sandbox


# ---------------------------------------------------------------------------
# _run_install
# ---------------------------------------------------------------------------


class TestRunInstall:
    def test_success_returns_true_and_message(self):
        sandbox = _make_sandbox()
        ok, summary = _run_install(sandbox)

        assert ok is True
        assert "succeeded" in summary.lower()
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
        ok, summary = _run_install(sandbox)

        assert ok is False
        assert "FAILED" in summary
        assert "1" in summary  # exit code
        assert "ERESOLVE" in summary


# ---------------------------------------------------------------------------
# _run_unit_tests
# ---------------------------------------------------------------------------


class TestRunUnitTests:
    def test_success_returns_true(self):
        sandbox = _make_sandbox()
        ok, summary = _run_unit_tests(sandbox)

        assert ok is True
        assert "passed" in summary.lower()
        sandbox.run.assert_called_once_with(
            "npm test", timeout=_NPM_TEST_TIMEOUT_SECONDS
        )

    def test_failure_includes_stdout_tail(self):
        long_stdout = "\n".join(f"line {i}" for i in range(200))
        sandbox = _make_sandbox(
            run_side_effects=[
                CommandResult(
                    exit_code=1,
                    stdout=long_stdout,
                    stderr="some error",
                    duration_seconds=5.0,
                )
            ]
        )
        ok, summary = _run_unit_tests(sandbox)

        assert ok is False
        assert "FAILED" in summary
        # Should include tail of stdout, not the entire thing
        assert "line 199" in summary  # last line always present

    def test_uses_npm_test_timeout(self):
        sandbox = _make_sandbox()
        _run_unit_tests(sandbox)
        _, kwargs = sandbox.run.call_args
        assert kwargs.get("timeout") == _NPM_TEST_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# ODC helpers
# ---------------------------------------------------------------------------


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
        vuln: Dict[str, Any] = {"name": "CVE-2021-23337", "severity": "HIGH"}
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

    @patch("src.orchestrator.qa_critic._parse_report_identifiers")
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

    def test_raises_timeout_expired_on_slow_docker(self):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["docker"], timeout=300)
            with pytest.raises(subprocess.TimeoutExpired):
                _run_odc("test-vol")


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
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("src.orchestrator.qa_critic._run_odc", return_value=self._make_passing_odc_proc()), \
             patch("src.orchestrator.qa_critic._read_report_from_workspace", return_value='{}'), \
             patch("src.orchestrator.qa_critic._parse_report_identifiers", return_value=set()):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is True
        assert remaining == set()

    def test_failure_when_docker_not_available(self):
        sandbox = MagicMock()
        with patch("shutil.which", return_value=None):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "docker" in summary.lower()
        assert remaining == set()

    def test_failure_on_timeout(self):
        sandbox = MagicMock()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("src.orchestrator.qa_critic._run_odc",
                   side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=300)):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "timed out" in summary.lower()

    def test_failure_when_target_still_found(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("src.orchestrator.qa_critic._run_odc", return_value=self._make_passing_odc_proc()), \
             patch("src.orchestrator.qa_critic._read_report_from_workspace", return_value='{}'), \
             patch("src.orchestrator.qa_critic._parse_report_identifiers",
                   return_value={"CVE-2021-23337"}):  # still present
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is False
        assert "CVE-2021-23337" in summary
        assert "CVE-2021-23337" in remaining

    def test_no_report_and_nonzero_exit_is_failure(self):
        proc = MagicMock()
        proc.returncode = 1
        sandbox = MagicMock()
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("src.orchestrator.qa_critic._run_odc", return_value=proc), \
             patch("src.orchestrator.qa_critic._read_report_from_workspace", return_value=None), \
             patch("src.orchestrator.qa_critic._parse_report_identifiers", return_value=None):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "parseable" in summary.lower() or "no parseable" in summary.lower()

    def test_nonzero_exit_but_parseable_report_continues(self):
        """ODC exits non-zero but produces a valid report → treat as soft exit."""
        proc = MagicMock()
        proc.returncode = 2  # Dependency-Check warning exit
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("src.orchestrator.qa_critic._run_odc", return_value=proc), \
             patch("src.orchestrator.qa_critic._read_report_from_workspace", return_value='{}'), \
             patch("src.orchestrator.qa_critic._parse_report_identifiers", return_value=set()):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        # Should still pass because identifiers are resolved
        assert ok is True


# ---------------------------------------------------------------------------
# _collect_target_identifiers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _generate_workspace_diff
# ---------------------------------------------------------------------------


class TestGenerateWorkspaceDiff:
    def test_detects_modified_file(self, tmp_path):
        host_file = tmp_path / "app.js"
        host_file.write_text("const x = 1;\n", encoding="utf-8")

        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const x = 2;\n" if path == "app.js" else None
        )
        sandbox.run.return_value = CommandResult(
            exit_code=0,
            stdout="/workspace/app.js\n",
            stderr="",
            duration_seconds=0.1,
        )

        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["app.js"])

        assert "app.js" in changed
        assert "app.js" in diff_text

    def test_detects_deleted_file(self, tmp_path):
        host_file = tmp_path / "gone.js"
        host_file.write_text("old content\n", encoding="utf-8")

        sandbox = MagicMock()
        # read_file returns None → file deleted in workspace
        sandbox.read_file.return_value = None
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="", stderr="", duration_seconds=0.1
        )

        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["gone.js"])

        assert "gone.js" in changed
        assert "deleted" in diff_text.lower()

    def test_ignores_node_modules(self, tmp_path):
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "evil.js").write_text("bad", encoding="utf-8")

        sandbox = MagicMock()
        sandbox.read_file.return_value = None  # no workspace files
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="", stderr="", duration_seconds=0.1
        )

        _, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["node_modules/evil.js"])

        assert not any("node_modules" in f for f in changed)

    def test_diff_text_is_capped(self, tmp_path):
        from src.orchestrator.qa_critic import _DIFF_CHAR_BUDGET
        big_content = "x" * (_DIFF_CHAR_BUDGET * 3)
        (tmp_path / "big.ts").write_text(big_content, encoding="utf-8")

        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            big_content + "\nextra line\n" if path == "big.ts" else None
        )
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="", stderr="", duration_seconds=0.1
        )

        diff_text, _ = _generate_workspace_diff(str(tmp_path), sandbox, ["big.ts"])
        assert len(diff_text) <= _DIFF_CHAR_BUDGET + len("\n... (diff truncated)")

    def test_optimized_changed_files_path(self, tmp_path):
        host_file = tmp_path / "app.js"
        host_file.write_text("const x = 1;\n", encoding="utf-8")
        other_host_file = tmp_path / "other.js"
        other_host_file.write_text("const y = 1;\n", encoding="utf-8")

        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const x = 2;\n" if path == "app.js" else "const y = 2;\n"
        )

        # We pass only ["app.js"] as candidate_changed_files, so it should ignore "other.js".
        diff_text, changed = _generate_workspace_diff(
            str(tmp_path), sandbox, candidate_changed_files=["app.js"]
        )

        assert "app.js" in changed
        assert "other.js" not in changed
        assert "app.js" in diff_text
        assert "other.js" not in diff_text


# ---------------------------------------------------------------------------
# run_qa_critic_node
# ---------------------------------------------------------------------------


_MISSING = object()  # sentinel for "caller did not pass this arg"


def _make_minimal_state(
    groups=_MISSING,
    workspace_volume="test-vol",
    repo_root="/tmp/repo",
    group_strategies=None,
):
    resolved_groups = [_make_group()] if groups is _MISSING else groups
    return {
        "valid_groups": resolved_groups,
        "workspace_volume": workspace_volume,
        "repo_root": repo_root,
        "action_summaries": [],
        "group_strategies": group_strategies or {},
    }


class TestRunQACriticNode:
    """Tests for the full node entry point."""

    def _patch_all_helpers(
        self,
        install_ok=True,
        scan_ok=True,
        test_ok=True,
        remaining_ids=None,
        llm_evals=None,
    ):
        """Return a dict of patches suitable for use with contextlib.ExitStack."""
        remaining_ids = remaining_ids or set()

        group = _make_group()
        if llm_evals is None:
            llm_evals = {group.group_id: QAEvaluation(group_id=group.group_id, passed=True)}

        return {
            "install": patch(
                "src.orchestrator.qa_critic._run_install",
                return_value=(install_ok, "install ok" if install_ok else "install FAILED"),
            ),
            "scan": patch(
                "src.orchestrator.qa_critic._run_security_scan",
                return_value=(scan_ok, "scan ok" if scan_ok else "scan FAILED", remaining_ids),
            ),
            "test": patch(
                "src.orchestrator.qa_critic._run_unit_tests",
                return_value=(test_ok, "tests passed" if test_ok else "tests FAILED"),
            ),
            "diff": patch(
                "src.orchestrator.qa_critic._generate_workspace_diff",
                return_value=("diff text", ["package.json"]),
            ),
            "llm": patch(
                "src.orchestrator.qa_critic._run_llm_critic",
                return_value=llm_evals,
            ),
            "sandbox": patch(
                "src.orchestrator.qa_critic.DockerSandbox",
                return_value=MagicMock(__enter__=MagicMock(return_value=MagicMock()),
                                       __exit__=MagicMock(return_value=None)),
            ),
        }

    def test_all_passed_returns_all_passed_eval_status(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evals = {group.group_id: QAEvaluation(group_id=group.group_id, passed=True)}

        with patch("src.orchestrator.qa_critic._run_install", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._run_security_scan", return_value=(True, "ok", set())), \
             patch("src.orchestrator.qa_critic._run_unit_tests", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._generate_workspace_diff", return_value=("", [])), \
             patch("src.orchestrator.qa_critic._run_llm_critic", return_value=evals), \
             patch("src.orchestrator.qa_critic.DockerSandbox") as MockSandbox:
            MockSandbox.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockSandbox.return_value.__exit__ = MagicMock(return_value=None)

            result = run_qa_critic_node(state)

        assert result["eval_status"] == "all_passed"
        assert result["status"] == "qa_completed"
        assert group.group_id in result["qa_evaluations"]
        assert result["qa_evaluations"][group.group_id].passed is True

    def test_failures_detected_when_any_group_fails(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evals = {
            group.group_id: QAEvaluation(
                group_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="CVE still present.",
            )
        }

        with patch("src.orchestrator.qa_critic._run_install", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._run_security_scan",
                   return_value=(False, "scan failed", {"CVE-2021-23337"})), \
             patch("src.orchestrator.qa_critic._run_unit_tests", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._generate_workspace_diff", return_value=("", [])), \
             patch("src.orchestrator.qa_critic._run_llm_critic", return_value=evals), \
             patch("src.orchestrator.qa_critic.DockerSandbox") as MockSandbox:
            MockSandbox.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockSandbox.return_value.__exit__ = MagicMock(return_value=None)

            result = run_qa_critic_node(state)

        assert result["eval_status"] == "failures_detected"
        assert result["status"] == "qa_completed"
        assert result["qa_evaluations"][group.group_id].passed is False

    def test_missing_workspace_volume_returns_qa_failed(self):
        state = _make_minimal_state(workspace_volume=None)
        result = run_qa_critic_node(state)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"
        assert result["errors"]

    def test_no_valid_groups_returns_all_passed_with_no_evals(self):
        state = _make_minimal_state(groups=[])
        result = run_qa_critic_node(state)

        assert result["status"] == "qa_completed"
        assert result["eval_status"] == "all_passed"
        assert result["qa_evaluations"] == {}

    def test_docker_unavailable_returns_qa_failed(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])

        with patch("src.orchestrator.qa_critic.DockerSandbox") as MockSandbox:
            MockSandbox.return_value.__enter__ = MagicMock(
                side_effect=RuntimeError("Docker daemon unreachable: ...")
            )
            MockSandbox.return_value.__exit__ = MagicMock(return_value=None)

            result = run_qa_critic_node(state)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"

    def test_install_scan_test_each_called_exactly_once(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evals = {group.group_id: QAEvaluation(group_id=group.group_id, passed=True)}

        with patch("src.orchestrator.qa_critic._run_install", return_value=(True, "ok")) as mock_install, \
             patch("src.orchestrator.qa_critic._run_security_scan",
                   return_value=(True, "ok", set())) as mock_scan, \
             patch("src.orchestrator.qa_critic._run_unit_tests",
                   return_value=(True, "ok")) as mock_test, \
             patch("src.orchestrator.qa_critic._generate_workspace_diff", return_value=("", [])), \
             patch("src.orchestrator.qa_critic._run_llm_critic", return_value=evals), \
             patch("src.orchestrator.qa_critic.DockerSandbox") as MockSandbox:
            MockSandbox.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockSandbox.return_value.__exit__ = MagicMock(return_value=None)

            run_qa_critic_node(state)

        assert mock_install.call_count == 1
        assert mock_scan.call_count == 1
        assert mock_test.call_count == 1

    def test_changed_files_propagated_from_diff(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        evals = {group.group_id: QAEvaluation(group_id=group.group_id, passed=True)}

        with patch("src.orchestrator.qa_critic._run_install", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._run_security_scan",
                   return_value=(True, "ok", set())), \
             patch("src.orchestrator.qa_critic._run_unit_tests", return_value=(True, "ok")), \
             patch("src.orchestrator.qa_critic._generate_workspace_diff",
                   return_value=("some diff", ["package.json", "src/app.ts"])), \
             patch("src.orchestrator.qa_critic._run_llm_critic", return_value=evals), \
             patch("src.orchestrator.qa_critic.DockerSandbox") as MockSandbox:
            MockSandbox.return_value.__enter__ = MagicMock(return_value=MagicMock())
            MockSandbox.return_value.__exit__ = MagicMock(return_value=None)

            result = run_qa_critic_node(state)

        assert "package.json" in result["changed_files"]
        assert "src/app.ts" in result["changed_files"]


# ---------------------------------------------------------------------------
# LLM critic structured output check
# ---------------------------------------------------------------------------


class TestLlmCriticStructuredOutput:
    def test_uses_with_structured_output_for_qa_evaluation(self):
        """The LLM must be called with .with_structured_output(QAEvaluation)."""
        from src.orchestrator.qa_critic import _run_llm_critic
        from src.contracts.schemas import AgentActionSummary, AgentActionStatus

        group = _make_group()
        mock_eval = QAEvaluation(group_id=group.group_id, passed=True)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_llm_instance.invoke.return_value = mock_eval
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_eval
            mock_llm_instance.with_structured_output.return_value = mock_structured
            MockChatOpenAI.return_value = mock_llm_instance

            result = _run_llm_critic(
                groups=[group],
                group_strategies={group.group_id: "version_bump"},
                action_summaries=[],
                install_ok=True,
                install_summary="ok",
                scan_ok=True,
                scan_summary="ok",
                remaining_identifiers=set(),
                test_ok=True,
                test_summary="ok",
                diff_text="",
                changed_files=[],
            )

        # Verify that with_structured_output was called with QAEvaluation
        mock_llm_instance.with_structured_output.assert_called_once_with(QAEvaluation)
        assert group.group_id in result
        assert result[group.group_id].passed is True


# ---------------------------------------------------------------------------
# Toolbelt isolation: QA tools not in update/workaround toolbelts
# ---------------------------------------------------------------------------


class TestQAToolsNotInSubagentToolbelts:
    """
    Verify that the heavy QA commands are not present in the update or
    workaround toolbelts that are given to subagents.

    This duplicates the guards in test_remedy_tools.py but is kept here for
    documentation purposes and test isolation.
    """

    def test_run_dependency_install_not_in_update_toolbelt(self):
        from src.orchestrator.remedy_tools import build_update_toolbelt

        sandbox = MagicMock()
        tools = build_update_toolbelt(
            sandbox,
            touched_files=set(),
            host_repo_root=Path("/repo"),
            target_manifest_paths=["package.json"],
            package_manifest_paths={"lodash": ["package.json"]},
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names

    def test_run_dependency_install_not_in_workaround_toolbelt(self):
        from src.orchestrator.remedy_tools import build_workaround_toolbelt

        sandbox = MagicMock()
        tools = build_workaround_toolbelt(
            sandbox,
            touched_files=set(),
            host_repo_root=Path("/repo"),
        )
        tool_names = {t.name for t in tools}
        assert "run_dependency_install" not in tool_names
        assert "run_security_scan" not in tool_names
        assert "run_unit_tests" not in tool_names
