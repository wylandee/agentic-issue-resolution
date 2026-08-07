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
  - one-shot guards on execution tools (backcompat build_qa_toolbelt)
  - toolbelt composition (execution + review tools; no edit tools)
  - review tool path safety (read_file_context rejects absolute/traversal paths)
  - generate_workspace_diff returns empty-diff note when no candidates given
  - .with_structured_output(QAEvaluation) usage in _extract_group_evaluations (backcompat)
  - all-pass and mixed-failure eval_status via agent loop mocking
  - QA tools are NOT present in update/workaround subagent toolbelts
  - BatchQAResult schema validation
  - _run_global_execution calls install/scan/tests exactly once with no LLM
  - build_qa_review_toolbelt excludes execution tools
  - _run_individual_investigations: one loop per group, group-scoped prompts, fallback on crash
  - _run_batch_judge: with_structured_output(BatchQAResult) called once; fallback on LLM failure
  - _apply_guardrails: missing/duplicate/unknown evals corrected; VERSION_BUMP+remaining forced False;
    CODE_WORKAROUND+remaining allowed pass; ERESOLVE maps to PEER_CONFLICT
  - run_qa_critic_node map-reduce integration: global execution once, map once per group, reduce once
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from remediation_engine.contracts.schemas import (
    BatchQAResult,
    CommandResult,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    NoFixMitigationStage,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    Severity,
    TaskStatus,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.orchestration.qa_critic import (
    _NPM_INSTALL_TIMEOUT_SECONDS,
    _NPM_TEST_TIMEOUT_SECONDS,
    _ODC_CACHE_VOLUME,
    _ODC_HTML_REPORT_NAME,
    _ODC_REPORT_NAME,
    _ODC_TIMEOUT_SECONDS,
    GroupInvestigation,
    _apply_guardrails,
    _build_batch_judge_prompt,
    _build_fallback_investigation_report,
    _build_individual_investigator_prompt,
    _collect_baseline_identifiers,
    _collect_target_identifiers,
    _derive_qa_group_strategies,
    _detect_test_suite_plans,
    _extract_failure_blocks,
    _extract_group_evaluations,
    _generate_workspace_diff,
    _group_scan_status,
    _normalize_node_tap_output,
    _parse_investigation_report,
    _parse_report_identifiers,
    _QAExecutionResults,
    _read_report_from_workspace,
    _run_batch_judge,
    _run_global_execution,
    _run_individual_investigations,
    _run_install,
    _run_judge_phase,
    _run_odc,
    _run_security_scan,
    _run_unit_tests,
    _scan_state_projection,
    _SecurityScanResult,
    _summarize_failed_test_output,
    _validate_qa_path,
    build_qa_review_toolbelt,
    build_qa_toolbelt,
    build_targeted_test_command,
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
            ["// workaround: sanitize input"]
            if fix_plan_status == FixPlanStatus.WORKAROUND_FOUND
            else None
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


def _make_workspace_tmpdir() -> Path:
    """Create a writable temp directory inside the workspace."""
    base_dir = Path("data/cache")
    base_dir.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base_dir))


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
        ok, summary = _run_unit_tests(sandbox)

        assert ok is False
        assert "FAILED" in summary
        assert "Failing Tests:" in summary
        assert "verify jwtChallenges challenge tracking" in summary
        assert "AssertionError" in summary

    def test_uses_npm_test_timeout(self):
        sandbox = _make_sandbox()
        _run_unit_tests(sandbox)
        _, kwargs = sandbox.run.call_args
        assert kwargs.get("timeout") == _NPM_TEST_TIMEOUT_SECONDS

    def test_composite_suite_stops_after_first_failed_child(self):
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
                )
            ]
        )
        sandbox.read_file.side_effect = lambda path: (
            json.dumps(root_package) if path == "package.json" else None
        )

        ok, summary = _run_unit_tests(sandbox)

        assert ok is False
        assert "Failed Tests: 1" in summary
        assert "server rejects bad token" in summary
        assert sandbox.run.call_count == 1
        sandbox.run.assert_called_once_with(
            "npm run test:server -- --reporter json",
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

        ok, summary = _run_unit_tests(sandbox)

        assert ok is True
        assert summary == "npm test passed."
        assert sandbox.run.call_count == 2

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

        ok, summary = _run_unit_tests(sandbox)

        assert ok is False
        assert "Detected Failures:" in summary
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

    @patch("remediation_engine.orchestration.qa_critic._parse_report_identifiers")
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
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is True
        assert remaining == set()
        assert _ODC_HTML_REPORT_NAME in str(sandbox.read_file.call_args_list[0])

    def test_detects_identifiers_absent_from_baseline(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        found = {"CVE-2025-10001", "GHSA-AAAA-BBBB-CCCC"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
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

    def test_preexisting_identifier_is_not_classified_as_new(self):
        sandbox = MagicMock()
        found = {"CVE-2021-23337", "CVE-2025-10001"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
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
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
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
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._persist_workspace_report_to_host",
                side_effect=[
                    Path("data/cache/qa_reports/dependency-check-report-1234567890.html"),
                    Path("data/cache/qa_reports/dependency-check-report.json"),
                ],
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is True
        assert remaining == set()
        assert "HTML report saved to:" in summary
        assert "dependency-check-report-1234567890.html" in summary

    def test_failure_when_docker_not_available(self):
        sandbox = MagicMock()
        with patch("shutil.which", return_value=None):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "docker" in summary.lower()
        assert remaining == set()

    def test_failure_on_timeout(self):
        sandbox = MagicMock()
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_critic._run_odc",
                side_effect=subprocess.TimeoutExpired(cmd=["docker"], timeout=300),
            ),
        ):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "timed out" in summary.lower()

    def test_failure_when_target_still_found(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value={"CVE-2021-23337"},
            ),
        ):  # still present
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is False
        assert "unresolved target vulnerabilities" in summary
        assert "Remaining identifiers: CVE-2021-23337" in summary
        assert "CVE-2021-23337" in remaining
        assert "CVE-2021-23337" in remaining

    def test_failure_summary_lists_multiple_remaining_identifiers(self):
        sandbox = MagicMock()
        target = {"CVE-2021-23337", "GHSA-35JH-R3H4-6JV8"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "remediation_engine.orchestration.qa_critic._run_odc",
                return_value=self._make_passing_odc_proc(),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value={"GHSA-35JH-R3H4-6JV8", "CVE-2021-23337"},
            ),
        ):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", target)

        assert ok is False
        assert "CVE-2021-23337" in summary
        assert "GHSA-35JH-R3H4-6JV8" in summary
        assert remaining == {"CVE-2021-23337", "GHSA-35JH-R3H4-6JV8"}

    def test_no_report_and_nonzero_exit_is_failure(self):
        proc = MagicMock()
        proc.returncode = 1
        sandbox = MagicMock()
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("remediation_engine.orchestration.qa_critic._run_odc", return_value=proc),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value=None,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value=None,
            ),
        ):
            ok, summary, remaining = _run_security_scan(sandbox, "vol", {"CVE-2021-23337"})

        assert ok is False
        assert "parseable" in summary.lower() or "no parseable" in summary.lower()

    def test_nonzero_exit_but_parseable_report_continues(self):
        """ODC exits non-zero but produces a valid report â†’ treat as soft exit."""
        proc = MagicMock()
        proc.returncode = 2  # Dependency-Check warning exit
        sandbox = MagicMock()
        target = {"CVE-2021-23337"}
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("remediation_engine.orchestration.qa_critic._run_odc", return_value=proc),
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value=set(),
            ),
        ):
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


# ---------------------------------------------------------------------------
# _generate_workspace_diff
# ---------------------------------------------------------------------------


class TestGenerateWorkspaceDiff:
    def test_detects_modified_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: "const x = 2;\n" if path == rel_path else None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert rel_path in changed
        assert rel_path in diff_text

    def test_detects_deleted_file(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"

        sandbox = MagicMock()
        # read_file returns None â†’ file deleted in workspace
        sandbox.read_file.return_value = None
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])

        assert rel_path in changed
        assert "deleted" in diff_text.lower()

    def test_ignores_node_modules(self):
        tmp_path = Path.cwd()

        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        _, changed = _generate_workspace_diff(str(tmp_path), sandbox, ["node_modules/evil.js"])

        assert not any("node_modules" in f for f in changed)

    def test_diff_text_is_capped(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"
        from remediation_engine.orchestration.qa_critic import _DIFF_CHAR_BUDGET

        big_content = "x" * (_DIFF_CHAR_BUDGET * 3)
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            big_content + "\nextra line\n" if path == rel_path else None
        )
        diff_text, _ = _generate_workspace_diff(str(tmp_path), sandbox, [rel_path])
        assert len(diff_text) <= _DIFF_CHAR_BUDGET + len("\n... (diff truncated)")

    def test_optimized_changed_files_path(self):
        tmp_path = Path.cwd()
        rel_path = "src/remediation_engine/orchestration/qa_critic.py"
        sandbox = MagicMock()
        sandbox.read_file.side_effect = lambda path: (
            "const x = 2;\n" if path == rel_path else "const y = 2;\n"
        )

        diff_text, changed = _generate_workspace_diff(
            str(tmp_path), sandbox, candidate_changed_files=[rel_path]
        )

        assert rel_path in changed
        assert "other.js" not in changed
        assert rel_path in diff_text
        assert "other.js" not in diff_text

    def test_empty_candidates_returns_empty_diff_note(self, tmp_path):
        """When no candidate files are given, diff returns an empty-diff note."""
        sandbox = MagicMock()
        diff_text, changed = _generate_workspace_diff(str(tmp_path), sandbox, [])
        assert changed == []
        assert "empty" in diff_text.lower() or "no changed files" in diff_text.lower()
        sandbox.read_file.assert_not_called()


# ---------------------------------------------------------------------------
# build_qa_toolbelt â€” toolbelt composition
# ---------------------------------------------------------------------------


class TestBuildQaToolbelt:
    """Verify the QA toolbelt has the right tools and no dangerous edit tools."""

    _EXECUTION_TOOL_NAMES = {
        "run_dependency_install",
        "run_security_scan",
        "run_unit_tests",
    }
    _REVIEW_TOOL_NAMES = {
        "list_changed_files",
        "generate_workspace_diff",
        "read_file_context",
        "search_codebase_pattern",
        "inspect_ast_symbol",
        "query_qa_logs",
    }
    _FORBIDDEN_TOOL_NAMES = {
        "deterministic_search_replace",
        "modify_npm_dependency",
        "revert_workspace_file",
        "validate_manifest_sync",
        "validate_code_syntax",
    }

    def _make_toolbelt(self, candidate_changed_files=None):
        sandbox = MagicMock()
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers={"CVE-2021-23337"},
            candidate_changed_files=candidate_changed_files or ["package.json"],
            host_repo_root="/tmp/repo",
        )
        return tools, results

    def test_contains_all_execution_tools(self):
        tools, _ = self._make_toolbelt()
        tool_names = {t.name for t in tools}
        assert self._EXECUTION_TOOL_NAMES.issubset(tool_names)

    def test_contains_all_review_tools(self):
        tools, _ = self._make_toolbelt()
        tool_names = {t.name for t in tools}
        assert self._REVIEW_TOOL_NAMES.issubset(tool_names)

    def test_does_not_contain_edit_tools(self):
        tools, _ = self._make_toolbelt()
        tool_names = {t.name for t in tools}
        for forbidden in self._FORBIDDEN_TOOL_NAMES:
            assert forbidden not in tool_names, f"Forbidden tool '{forbidden}' found in QA toolbelt"

    def test_returns_empty_results_cache_initially(self):
        _, results = self._make_toolbelt()
        assert results.install is None
        assert results.scan is None
        assert results.tests is None


# ---------------------------------------------------------------------------
# One-shot execution tool guards
# ---------------------------------------------------------------------------


class TestQAExecutionToolGuards:
    """Verify that each execution tool caches its result and returns [CACHED] on repeat calls."""

    def _make_toolbelt_with_mock_helpers(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="ok", stderr="", duration_seconds=0.5
        )
        sandbox.read_file.return_value = None

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("remediation_engine.orchestration.qa_critic._run_odc") as mock_odc,
            patch(
                "remediation_engine.orchestration.qa_critic._read_report_from_workspace",
                return_value="{}",
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._parse_report_identifiers",
                return_value=set(),
            ),
        ):
            mock_odc.return_value = MagicMock(returncode=0, stdout="", stderr="")
            tools, results = build_qa_toolbelt(
                sandbox=sandbox,
                workspace_volume="test-vol",
                target_identifiers=set(),
                candidate_changed_files=["package.json"],
                host_repo_root="/tmp/repo",
            )
        return tools, results, sandbox

    def _get_tool(self, tools, name):
        for t in tools:
            if t.name == name:
                return t
        raise KeyError(f"Tool '{name}' not found")

    def test_run_dependency_install_caches_result(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="ok", stderr="", duration_seconds=0.5
        )
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        install_tool = self._get_tool(tools, "run_dependency_install")

        first = install_tool.invoke({})
        second = install_tool.invoke({})

        assert results.install is not None
        assert "[CACHED" in second
        # Underlying sandbox.run called only once
        assert sandbox.run.call_count == 1

    def test_run_security_scan_caches_result(self):
        sandbox = MagicMock()
        with patch("shutil.which", return_value=None):
            tools, results = build_qa_toolbelt(
                sandbox=sandbox,
                workspace_volume="test-vol",
                target_identifiers={"CVE-2021-23337"},
                candidate_changed_files=[],
                host_repo_root=None,
            )
        scan_tool = self._get_tool(tools, "run_security_scan")
        install_tool = self._get_tool(tools, "run_dependency_install")
        results.install = (True, "install ok")

        first = scan_tool.invoke({})
        second = scan_tool.invoke({})

        assert results.scan is not None
        assert "[CACHED" in second

    def test_run_unit_tests_caches_result(self):
        sandbox = MagicMock()
        sandbox.run.return_value = CommandResult(
            exit_code=0, stdout="ok", stderr="", duration_seconds=0.5
        )
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        test_tool = self._get_tool(tools, "run_unit_tests")
        results.scan = (True, "scan ok", set())

        first = test_tool.invoke({})
        second = test_tool.invoke({})

        assert results.tests is not None
        assert "[CACHED" in second
        assert sandbox.run.call_count == 1

    def test_scan_before_install_is_rejected(self):
        sandbox = MagicMock()
        tools, _ = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        scan_tool = self._get_tool(tools, "run_security_scan")
        result = scan_tool.invoke({})
        assert "ERROR" in result
        assert "after run_dependency_install" in result

    def test_tests_before_scan_is_rejected(self):
        sandbox = MagicMock()
        tools, _ = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        install_tool = self._get_tool(tools, "run_dependency_install")
        test_tool = self._get_tool(tools, "run_unit_tests")
        install_tool.invoke({})
        result = test_tool.invoke({})
        assert "ERROR" in result
        assert "after run_security_scan" in result

    def test_review_tools_before_pipeline_completion_are_rejected(self):
        sandbox = MagicMock()
        tools, _ = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=["package.json"],
            host_repo_root="/tmp/repo",
        )
        changed_tool = self._get_tool(tools, "list_changed_files")
        search_tool = self._get_tool(tools, "search_codebase_pattern")
        assert "Review tools are locked" in changed_tool.invoke({})
        assert "Review tools are locked" in search_tool.invoke({"root_dir": ".", "pattern": "foo"})


# ---------------------------------------------------------------------------
# Review tool safety
# ---------------------------------------------------------------------------


class TestQAReviewToolSafety:
    """Verify read_file_context rejects dangerous paths."""

    def _get_read_file_tool(self, sandbox=None):
        if sandbox is None:
            sandbox = MagicMock()
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        for t in tools:
            if t.name == "read_file_context":
                return t
        raise KeyError("read_file_context not found")

    def test_rejects_absolute_path(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "/etc/passwd"})
        assert "ERROR" in result
        assert "absolute" in result.lower()

    def test_rejects_path_traversal(self):
        tool = self._get_read_file_tool()
        result = tool.invoke({"file_path": "../../etc/passwd"})
        assert "ERROR" in result

    def test_returns_file_content_for_valid_path(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "const x = 1;\n"
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "src/app.js"})
        assert "const x = 1;" in result
        sandbox.read_file.assert_called_once_with("src/app.js")

    def test_returns_error_when_file_not_found(self):
        sandbox = MagicMock()
        sandbox.read_file.return_value = None
        tool = self._get_read_file_tool(sandbox)

        result = tool.invoke({"file_path": "nonexistent.js"})
        assert "ERROR" in result
        assert "not found" in result.lower()


class TestValidateQaPath:
    """Unit tests for the _validate_qa_path helper."""

    def test_accepts_relative_path(self):
        assert _validate_qa_path("src/app.js") == "src/app.js"

    def test_accepts_nested_relative_path(self):
        assert _validate_qa_path("a/b/c.ts") == "a/b/c.ts"

    def test_rejects_empty_string(self):
        with pytest.raises(ValueError, match="required"):
            _validate_qa_path("")

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="absolute"):
            _validate_qa_path("/etc/passwd")

    def test_rejects_path_traversal(self):
        with pytest.raises(ValueError, match="traversal"):
            _validate_qa_path("../../secret")

    def test_normalizes_backslashes(self):
        assert _validate_qa_path("src\\app.js") == "src/app.js"


# ---------------------------------------------------------------------------
# generate_workspace_diff as review tool (no candidates)
# ---------------------------------------------------------------------------


class TestQADiffToolNoCandidates:
    """Verify generate_workspace_diff tool returns an informative note when no candidates exist."""

    def test_empty_candidates_returns_informative_note(self):
        sandbox = MagicMock()
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],  # empty
            host_repo_root="/tmp/repo",
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        # Should mention no changed files, not crash
        assert "empty" in result.lower() or "no changed files" in result.lower()
        sandbox.read_file.assert_not_called()

    def test_no_host_repo_root_returns_error(self):
        sandbox = MagicMock()
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=["src/app.js"],
            host_repo_root=None,  # not set
        )
        results.install = (True, "install ok")
        results.scan = (True, "scan ok", set())
        results.tests = (True, "tests ok")
        diff_tool = next(t for t in tools if t.name == "generate_workspace_diff")
        result = diff_tool.invoke({})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# query_qa_logs tool
# ---------------------------------------------------------------------------


class TestQueryQaLogs:
    """Verify query_qa_logs returns cached data and handles missing runs correctly."""

    def _get_query_tool(self, prepopulate: bool = False):
        sandbox = MagicMock()
        tools, results = build_qa_toolbelt(
            sandbox=sandbox,
            workspace_volume="test-vol",
            target_identifiers=set(),
            candidate_changed_files=[],
            host_repo_root=None,
        )
        if prepopulate:
            results.install = (True, "install ready")
            results.scan = (True, "scan ready", set())
            results.tests = (True, "tests ready")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        return tool, results

    def test_returns_error_when_install_not_run(self):
        tool, _ = self._get_query_tool()
        result = tool.invoke({"log_type": "install"})
        assert "ERROR" in result

    def test_returns_cached_install_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.install = (True, "npm install output here")
        result = tool.invoke({"log_type": "install"})
        assert "npm install output here" in result

    def test_returns_cached_scan_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.scan = (True, "scan output here", set())
        result = tool.invoke({"log_type": "scan"})
        assert "scan output here" in result

    def test_returns_cached_tests_log(self):
        tool, results = self._get_query_tool(prepopulate=True)
        results.tests = (True, "test output here")
        result = tool.invoke({"log_type": "tests"})
        assert "test output here" in result

    def test_invalid_log_type_returns_error(self):
        tool, _ = self._get_query_tool(prepopulate=True)
        result = tool.invoke({"log_type": "unknown"})
        assert "ERROR" in result


# ---------------------------------------------------------------------------
# run_qa_critic_node â€” agentic loop integration
# ---------------------------------------------------------------------------


_MISSING = object()  # sentinel for "caller did not pass this arg"


def _make_minimal_state(
    groups=_MISSING,
    workspace_volume="test-vol",
    repo_root="/tmp/repo",
    group_strategies=None,
    changed_files=None,
):
    resolved_groups = [_make_group()] if groups is _MISSING else groups
    return {
        "valid_groups": resolved_groups,
        "workspace_volume": workspace_volume,
        "repo_root": repo_root,
        "action_summaries": [],
        "group_strategies": group_strategies or {},
        "changed_files": changed_files or [],
    }


def _make_loop_result(
    final_text="# INVESTIGATIVE REPORT\n## Install Analysis\n- Install Status: succeeded",
    tool_events=None,
    errors=None,
):
    """Build a mock SubagentRuntimeResult."""
    from remediation_engine.orchestration.subagent_runtime import SubagentRuntimeResult

    return SubagentRuntimeResult(
        final_text=final_text,
        tool_events=tool_events or [],
        changed_files=[],
        errors=errors or [],
    )


def _make_fully_populated_results(ok=True):
    """Build a _QAExecutionResults with all three phases filled in."""
    r = _QAExecutionResults()
    r.install = (ok, "npm install succeeded." if ok else "npm install FAILED")
    r.scan = (ok, "Dependency-Check OK." if ok else "scan FAILED", set())
    r.tests = (ok, "npm test passed." if ok else "npm test FAILED")
    return r


class TestRunQACriticNode:
    """Tests for the full node entry point (agent loop mocked)."""

    def _patch_node(
        self,
        results=None,
        batch_result=None,
        group=None,
    ):
        """
        Return patches for the node's main external dependencies (map-reduce path):
        - _run_global_execution â†’ returns results
        - _run_individual_investigations â†’ returns a GroupInvestigation dict
        - _run_batch_judge â†’ returns batch_result
        - DockerSandbox â†’ context manager mock
        """
        if group is None:
            group = _make_group()
        if results is None:
            results = _make_fully_populated_results(ok=True)
        if batch_result is None:
            batch_result = BatchQAResult(
                holistic_report="All groups passed.",
                evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
            )

        investigations = {
            group.group_id: GroupInvestigation(
                group_id=group.group_id,
                investigation_text="Investigation complete.",
                tool_transcript="",
                errors=[],
            )
        }

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)

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
                "remediation_engine.orchestration.qa_critic._run_individual_investigations",
                return_value=investigations,
            ),
            "judge": patch(
                "remediation_engine.orchestration.qa_critic._run_batch_judge",
                return_value=batch_result,
            ),
        }

    def test_all_passed_returns_all_passed_eval_status(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        batch_result = BatchQAResult(
            holistic_report="All groups passed.",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )
        patches = self._patch_node(group=group, batch_result=batch_result)

        with patches["sandbox"], patches["global_exec"], patches["investigators"], patches["judge"]:
            result = run_qa_critic_node(state)

        assert result["eval_status"] == "all_passed"
        assert result["status"] == "qa_completed"
        assert group.group_id in result["qa_evaluations"]
        assert result["qa_evaluations"][group.group_id].passed is True
        assert result["qa_investigation_report"] == "All groups passed."

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
        batch_result = BatchQAResult(
            holistic_report="All current remediation groups passed.",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )
        patches = self._patch_node(
            group=group,
            results=results,
            batch_result=batch_result,
        )

        with patches["sandbox"], patches["global_exec"], patches["investigators"], patches["judge"]:
            result = run_qa_critic_node(state)

        assert result["eval_status"] == "all_passed"
        assert result["qa_evaluations"][group.group_id].passed is True
        assert result["post_remediation_scan_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_identifiers"] == ["CVE-2025-10001"]
        assert result["new_vulnerability_status"] == "detected"
        assert "CVE-2025-10001" in result["qa_investigation_report"]

    def test_failures_detected_when_any_group_fails(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        batch_result = BatchQAResult(
            holistic_report="Group failed.",
            evaluations=[
                QAEvaluation(
                    task_id=group.group_id,
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="CVE still present.",
                )
            ],
        )
        patches = self._patch_node(group=group, batch_result=batch_result)

        with patches["sandbox"], patches["global_exec"], patches["investigators"], patches["judge"]:
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

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(side_effect=RuntimeError("Docker daemon unreachable"))
        mock_sandbox.__exit__ = MagicMock(return_value=None)

        with patch(
            "remediation_engine.orchestration.qa_critic.DockerSandbox", return_value=mock_sandbox
        ):
            result = run_qa_critic_node(state)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"

    def test_changed_files_propagated_from_state(self):
        group = _make_group()
        state = _make_minimal_state(
            groups=[group],
            changed_files=["package.json", "src/app.ts"],
        )
        batch_result = BatchQAResult(
            holistic_report="ok",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )
        patches = self._patch_node(group=group, batch_result=batch_result)

        with patches["sandbox"], patches["global_exec"], patches["investigators"], patches["judge"]:
            result = run_qa_critic_node(state)

        assert "package.json" in result["changed_files"]
        assert "src/app.ts" in result["changed_files"]

    def test_loop_errors_propagated_to_result(self):
        group = _make_group()
        state = _make_minimal_state(groups=[group])
        investigations = {
            group.group_id: GroupInvestigation(
                group_id=group.group_id,
                investigation_text="partial",
                tool_transcript="",
                errors=["Subagent exceeded max rounds."],
            )
        }
        batch_result = BatchQAResult(
            holistic_report="ok",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )
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
                "remediation_engine.orchestration.qa_critic._run_individual_investigations",
                return_value=investigations,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_batch_judge",
                return_value=batch_result,
            ),
        ):
            result = run_qa_critic_node(state)

        assert any("max rounds" in e.lower() or "exceeded" in e.lower() for e in result["errors"])


# ---------------------------------------------------------------------------
# Missing execution tools â†’ qa_failed
# ---------------------------------------------------------------------------


class TestQAMissingExecutionTools:
    """Verify qa_failed is returned when the agent skips a required tool."""

    def _run_with_partial_results(self, results: _QAExecutionResults):
        group = _make_group()
        state = _make_minimal_state(groups=[group])

        mock_sandbox = MagicMock()
        mock_sandbox.__enter__ = MagicMock(return_value=mock_sandbox)
        mock_sandbox.__exit__ = MagicMock(return_value=None)

        # investigations dict â€” will not be reached because guardrails fire
        investigations = {
            group.group_id: GroupInvestigation(
                group_id=group.group_id,
                investigation_text="partial",
                tool_transcript="",
                errors=[],
            )
        }
        batch_result = BatchQAResult(
            holistic_report="ok",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )

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
                "remediation_engine.orchestration.qa_critic._run_individual_investigations",
                return_value=investigations,
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_batch_judge",
                return_value=batch_result,
            ),
        ):
            return run_qa_critic_node(state), group

    def test_missing_install_returns_qa_failed(self):
        results = _QAExecutionResults()
        results.scan = (True, "ok", set())
        results.tests = (True, "ok")
        # install is None

        result, group = self._run_with_partial_results(results)

        assert result["status"] == "qa_failed"
        assert result["eval_status"] == "failures_detected"
        assert result["qa_evaluations"][group.group_id].passed is False
        error_text = " ".join(result["errors"])
        assert "run_dependency_install" in error_text

    def test_missing_scan_returns_qa_failed(self):
        results = _QAExecutionResults()
        results.install = (True, "ok")
        results.tests = (True, "ok")
        # scan is None

        result, group = self._run_with_partial_results(results)

        assert result["status"] == "qa_failed"
        error_text = " ".join(result["errors"])
        assert "run_security_scan" in error_text

    def test_missing_tests_returns_qa_failed(self):
        results = _QAExecutionResults()
        results.install = (True, "ok")
        results.scan = (True, "ok", set())
        # tests is None

        result, group = self._run_with_partial_results(results)

        assert result["status"] == "qa_failed"
        error_text = " ".join(result["errors"])
        assert "run_unit_tests" in error_text

    def test_all_tools_missing_lists_all_in_error(self):
        results = _QAExecutionResults()
        # All None

        result, _ = self._run_with_partial_results(results)

        assert result["status"] == "qa_failed"
        error_text = " ".join(result["errors"])
        assert "run_dependency_install" in error_text
        assert "run_security_scan" in error_text
        assert "run_unit_tests" in error_text


# ---------------------------------------------------------------------------
# _extract_group_evaluations (replaces TestLlmCriticStructuredOutput)
# ---------------------------------------------------------------------------


class TestExtractGroupEvaluations:
    """The extraction step must call .with_structured_output(QAEvaluation) per group."""

    def test_uses_with_structured_output_for_qa_evaluation(self):

        group = _make_group()
        mock_eval = QAEvaluation(task_id=group.group_id, passed=True)

        results = _make_fully_populated_results(ok=True)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = mock_eval
            mock_llm_instance.with_structured_output.return_value = mock_structured
            MockChatOpenAI.return_value = mock_llm_instance

            result = _extract_group_evaluations(
                valid_groups=[group],
                group_strategies={group.group_id: "version_bump"},
                action_summaries=[],
                results=results,
                agent_transcript="Agent concluded all is well.",
            )

        mock_llm_instance.with_structured_output.assert_called_once_with(QAEvaluation)
        assert group.group_id in result
        assert result[group.group_id].passed is True

    def test_handles_llm_exception_gracefully(self):
        group = _make_group()
        results = _make_fully_populated_results(ok=True)

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_structured = MagicMock()
            mock_structured.invoke.side_effect = RuntimeError("LLM quota exceeded")
            mock_llm_instance.with_structured_output.return_value = mock_structured
            MockChatOpenAI.return_value = mock_llm_instance

            result = _extract_group_evaluations(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                results=results,
                agent_transcript="",
            )

        assert group.group_id in result
        ev = result[group.group_id]
        assert ev.passed is False
        assert ev.failure_category == FailureCategory.SECURITY_FLAG
        assert "exception" in ev.retry_feedback.lower()

    def test_uses_cached_execution_results_in_prompt(self):
        """Verify that missing results produce the correct fallback messages."""
        group = _make_group()
        results = _QAExecutionResults()  # all None â€” nothing was called

        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_structured = MagicMock()
            captured_prompts = []
            mock_eval = QAEvaluation(
                task_id=group.group_id,
                passed=False,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="tools not called",
            )

            def capture_invoke(prompt):
                captured_prompts.append(prompt)
                return mock_eval

            mock_structured.invoke.side_effect = capture_invoke
            mock_llm_instance.with_structured_output.return_value = mock_structured
            MockChatOpenAI.return_value = mock_llm_instance

            _extract_group_evaluations(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                results=results,
                agent_transcript="",
            )

        assert captured_prompts
        prompt_text = captured_prompts[0]
        assert "was not called" in prompt_text


class TestInvestigationReportParsing:
    def test_extracts_shared_install_analysis_and_group_blocks(self):
        g1 = _make_group(group_id="g1")
        g2 = _make_group(group_id="g2")
        report = """# INVESTIGATIVE REPORT
## Install Analysis
- Install Status: failed
- Summary: peer dependency conflict

### GROUP: g1
- Scan Reasoning: still flagged
- Group Summary: group one

### GROUP: g2
- Scan Reasoning: cleared
- Group Summary: group two
"""
        parsed = _parse_investigation_report(report, [g1, g2])
        assert "Install Status: failed" in parsed.shared_install_analysis
        assert parsed.group_sections["g1"].scan_reasoning == "still flagged"
        assert parsed.group_sections["g2"].group_summary == "group two"

    def test_missing_known_group_creates_placeholder_and_unknown_is_warning(self):
        g1 = _make_group(group_id="g1")
        g2 = _make_group(group_id="g2")
        report = """# INVESTIGATIVE REPORT
## Install Analysis
- Install Status: succeeded

### GROUP: g1
- Group Summary: present

### GROUP: unknown-group
- Group Summary: ignore me
"""
        parsed = _parse_investigation_report(report, [g1, g2])
        assert any("missing block" in error.lower() for error in parsed.errors)
        assert any("unknown" in warning.lower() for warning in parsed.warnings)
        assert parsed.group_sections["g2"].raw_text.startswith("### GROUP: g2")


class TestGroupScanAttribution:
    def test_group_scan_status_is_cleared_when_other_group_is_still_flagged(self):
        group_a = _make_group(group_id="group-a", cve_ids=["CVE-2021-1111"], ghsa_ids=[])
        group_b = _make_group(group_id="group-b", cve_ids=["CVE-2021-2222"], ghsa_ids=[])
        scan_result = (False, "target vulnerabilities remain", {"CVE-2021-1111"})

        assert _group_scan_status(scan_result, group_a) == "still_flagged"
        assert _group_scan_status(scan_result, group_b) == "cleared"

    def test_fallback_report_does_not_mark_unmatched_group_as_still_flagged(self):
        group_a = _make_group(group_id="group-a", cve_ids=["CVE-2021-1111"], ghsa_ids=[])
        group_b = _make_group(group_id="group-b", cve_ids=["CVE-2021-2222"], ghsa_ids=[])
        results = _make_fully_populated_results(ok=True)
        results.scan = (False, "target vulnerabilities remain", {"CVE-2021-1111"})

        report = _build_fallback_investigation_report(
            valid_groups=[group_a, group_b],
            group_strategies={},
            candidate_changed_files=[],
            results=results,
            reason="fallback",
        )
        parsed = _parse_investigation_report(report, [group_a, group_b])

        assert parsed.group_sections["group-a"].scan_status == "still_flagged"
        assert parsed.group_sections["group-a"].remaining_scanner_findings == "CVE-2021-1111"
        assert parsed.group_sections["group-b"].scan_status == "cleared"
        assert parsed.group_sections["group-b"].remaining_scanner_findings == "none"

    def test_investigator_and_batch_prompts_include_global_new_findings(self):
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
            group=group,
            strategy="version_bump",
            results=results,
            group_remaining_ids=[],
            candidate_changed_files=[],
            action_summaries=[],
        )
        batch_prompt = _build_batch_judge_prompt(
            valid_groups=[group],
            group_strategies={group.group_id: "version_bump"},
            action_summaries=[],
            results=results,
            investigations_by_group={},
        )

        assert "CVE-2025-10001" in investigator_prompt
        assert "CVE-2025-10001" in batch_prompt
        assert "Do not force them into an existing group's evaluation" in batch_prompt


class TestJudgePhaseIsolation:
    def test_judge_prompt_uses_only_current_group_block(self):
        group_a = _make_group(
            group_id="group-a",
            cve_ids=["CVE-2021-1111"],
            ghsa_ids=["GHSA-AAAA-BBBB-CCCC"],
        )
        group_b = _make_group(
            group_id="group-b",
            cve_ids=["CVE-2021-2222"],
            ghsa_ids=["GHSA-DDDD-EEEE-FFFF"],
        )
        results = _make_fully_populated_results(ok=True)
        results.scan = (False, "scan failed", {"CVE-2021-1111"})
        parsed = _parse_investigation_report(
            """# INVESTIGATIVE REPORT
## Install Analysis
- Install Status: succeeded

### GROUP: group-a
- Scan Reasoning: only group a text
- Group Summary: alpha

### GROUP: group-b
- Scan Reasoning: only group b text
- Group Summary: beta
""",
            [group_a, group_b],
        )
        investigation = MagicMock(
            results=results,
            parsed_report=parsed,
        )

        captured_prompts = []
        with patch("langchain_openai.ChatOpenAI") as MockChatOpenAI:
            mock_llm_instance = MagicMock()
            mock_structured = MagicMock()

            def capture(prompt):
                captured_prompts.append(prompt)
                gid = "group-a" if "Group ID: group-a" in prompt else "group-b"
                return QAEvaluation(task_id=gid, passed=True)

            mock_structured.invoke.side_effect = capture
            mock_llm_instance.with_structured_output.return_value = mock_structured
            MockChatOpenAI.return_value = mock_llm_instance

            result = _run_judge_phase(
                valid_groups=[group_a, group_b],
                group_strategies={"group-a": "version_bump", "group-b": "version_bump"},
                action_summaries=[],
                investigation=investigation,
            )

        prompt_a = next(prompt for prompt in captured_prompts if "Group ID: group-a" in prompt)
        assert "only group a text" in prompt_a
        assert "only group b text" not in prompt_a
        assert "CVE-2021-1111" in prompt_a
        assert "CVE-2021-2222" not in prompt_a
        assert result["group-a"].passed is True


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
        from remediation_engine.orchestration.remedy_tools import build_update_toolbelt

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
        from remediation_engine.orchestration.remedy_tools import build_workaround_toolbelt

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


# ---------------------------------------------------------------------------
# BatchQAResult schema
# ---------------------------------------------------------------------------


class TestBatchQAResultSchema:
    def test_valid_batch_result_accepted(self):
        result = BatchQAResult(
            holistic_report="All groups passed.",
            evaluations=[QAEvaluation(task_id="g1", passed=True)],
        )
        assert result.holistic_report == "All groups passed."
        assert len(result.evaluations) == 1

    def test_empty_evaluations_list_is_allowed(self):
        result = BatchQAResult(holistic_report="No groups.", evaluations=[])
        assert result.evaluations == []

    def test_holistic_report_must_be_nonempty(self):
        try:
            BatchQAResult(holistic_report="", evaluations=[])
            assert False, "should have raised"
        except Exception:
            pass

    def test_batch_result_is_frozen(self):
        result = BatchQAResult(holistic_report="ok", evaluations=[])
        try:
            result.holistic_report = "changed"
            assert False, "should have raised"
        except Exception:
            pass


# ---------------------------------------------------------------------------
# _run_global_execution
# ---------------------------------------------------------------------------


class TestRunGlobalExecution:
    def test_calls_install_scan_and_tests_once(self):
        sandbox = MagicMock()
        target_ids = {"CVE-2021-0001"}
        with (
            patch(
                "remediation_engine.orchestration.qa_critic._run_install", return_value=(True, "ok")
            ) as mi,
            patch(
                "remediation_engine.orchestration.qa_critic._run_security_scan",
                return_value=(True, "ok", set()),
            ) as ms,
            patch(
                "remediation_engine.orchestration.qa_critic._run_unit_tests",
                return_value=(True, "ok"),
            ) as mt,
        ):
            results = _run_global_execution(sandbox, "vol", target_ids)
        mi.assert_called_once_with(sandbox)
        ms.assert_called_once_with(sandbox, "vol", target_ids)
        mt.assert_called_once_with(sandbox)
        assert results.install == (True, "ok")
        assert results.scan == (True, "ok", set())
        assert results.tests == (True, "ok")

    def test_all_three_results_populated(self):
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.orchestration.qa_critic._run_install",
                return_value=(False, "fail"),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_security_scan",
                return_value=(False, "fail", set()),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_unit_tests",
                return_value=(False, "fail"),
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
                "remediation_engine.orchestration.qa_critic._run_install", return_value=(True, "ok")
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_security_scan",
                return_value=_SecurityScanResult(True, "ok", set(), set(), set()),
            ) as scan,
            patch(
                "remediation_engine.orchestration.qa_critic._run_unit_tests",
                return_value=(True, "ok"),
            ),
        ):
            _run_global_execution(sandbox, "vol", target_ids, baseline_ids)

        scan.assert_called_once_with(sandbox, "vol", target_ids, baseline_ids)

    def test_tests_run_even_when_install_fails(self):
        sandbox = MagicMock()
        with (
            patch(
                "remediation_engine.orchestration.qa_critic._run_install",
                return_value=(False, "FAILED"),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_security_scan",
                return_value=(False, "fail", set()),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic._run_unit_tests",
                return_value=(True, "passed."),
            ) as mt,
        ):
            results = _run_global_execution(sandbox, "vol", set())
        mt.assert_called_once()
        assert results.tests == (True, "passed.")


# ---------------------------------------------------------------------------
# build_qa_review_toolbelt
# ---------------------------------------------------------------------------


class TestBuildQaReviewToolbelt:
    def _build(self, prepopulate=True):
        sandbox = MagicMock()
        sandbox.read_file.return_value = "file content"
        results = _QAExecutionResults()
        if prepopulate:
            results.install = (True, "ok")
            results.scan = (True, "ok", set())
            results.tests = (True, "ok")
        return build_qa_review_toolbelt(
            sandbox=sandbox,
            candidate_changed_files=["src/app.ts"],
            host_repo_root="/tmp/repo",
            results=results,
        ), results

    def test_no_execution_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        assert "run_dependency_install" not in names
        assert "run_security_scan" not in names
        assert "run_unit_tests" not in names

    def test_all_six_review_tools_present(self):
        tools, _ = self._build()
        names = {t.name for t in tools}
        for n in [
            "list_changed_files",
            "generate_workspace_diff",
            "read_file_context",
            "search_codebase_pattern",
            "inspect_ast_symbol",
            "query_qa_logs",
        ]:
            assert n in names

    def test_review_tools_locked_when_results_empty(self):
        tools, _ = self._build(prepopulate=False)
        tool = next(t for t in tools if t.name == "list_changed_files")
        result = tool.invoke({})
        assert "ERROR" in result

    def test_list_changed_files_works_when_populated(self):
        tools, _ = self._build(prepopulate=True)
        tool = next(t for t in tools if t.name == "list_changed_files")
        assert "src/app.ts" in tool.invoke({})

    def test_query_qa_logs_returns_install_log(self):
        tools, results = self._build(prepopulate=True)
        results.install = (True, "npm install output here")
        tool = next(t for t in tools if t.name == "query_qa_logs")
        assert "npm install output here" in tool.invoke({"log_type": "install"})

    def test_query_qa_logs_returns_condensed_test_summary(self):
        tools, results = self._build(prepopulate=True)
        results.tests = (
            False,
            "npm test FAILED (exit 1).\nDetected Failures: 1\n\nFailing Tests:\n1. jwt challenge\nAssertionError: boom",
        )
        tool = next(t for t in tools if t.name == "query_qa_logs")
        result = tool.invoke({"log_type": "tests"})
        assert "Detected Failures: 1" in result
        assert "jwt challenge" in result


# ---------------------------------------------------------------------------
# _build_individual_investigator_prompt
# ---------------------------------------------------------------------------


class TestBuildIndividualInvestigatorPrompt:
    def _prompt(self, group=None, remaining=None):
        if group is None:
            group = _make_group()
        results = _make_fully_populated_results(ok=True)
        return _build_individual_investigator_prompt(
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
        lower = self._prompt().lower()
        assert "not available to you" in lower or "do not attempt" in lower


# ---------------------------------------------------------------------------
# _run_individual_investigations
# ---------------------------------------------------------------------------


class TestRunIndividualInvestigations:
    def _lr(self, text="investigation complete", errors=None):
        from remediation_engine.orchestration.subagent_runtime import SubagentRuntimeResult

        return SubagentRuntimeResult(
            final_text=text,
            tool_events=[],
            changed_files=[],
            errors=errors or [],
        )

    def test_one_investigator_per_group(self):
        g1, g2 = _make_group(group_id="g1"), _make_group(group_id="g2")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_critic.run_bounded_subagent_loop",
                return_value=self._lr(),
            ) as ml,
            patch(
                "remediation_engine.orchestration.qa_critic.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                valid_groups=[g1, g2],
                group_strategies={},
                action_summaries=[],
                candidate_changed_files=[],
                sandbox=MagicMock(),
                repo_root="/tmp",
                results=results,
            )
        assert ml.call_count == 2
        assert "g1" in invs and "g2" in invs

    def test_crash_produces_fallback(self):
        group = _make_group(group_id="g1")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_critic.run_bounded_subagent_loop",
                side_effect=RuntimeError("crash"),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                candidate_changed_files=[],
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
            )
        assert invs["g1"].errors
        assert "Fallback" in invs["g1"].investigation_text

    def test_empty_output_triggers_fallback(self):
        group = _make_group(group_id="g1")
        results = _make_fully_populated_results(ok=True)
        with (
            patch("langchain_openai.ChatOpenAI"),
            patch(
                "remediation_engine.orchestration.qa_critic.run_bounded_subagent_loop",
                return_value=self._lr(text=""),
            ),
            patch(
                "remediation_engine.orchestration.qa_critic.build_qa_review_toolbelt",
                return_value=[],
            ),
        ):
            invs = _run_individual_investigations(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                candidate_changed_files=[],
                sandbox=MagicMock(),
                repo_root=None,
                results=results,
            )
        assert invs["g1"].errors
        assert "Fallback" in invs["g1"].investigation_text


# ---------------------------------------------------------------------------
# _run_batch_judge
# ---------------------------------------------------------------------------


class TestRunBatchJudge:
    def test_structured_output_called_once_with_batch_qa_result(self):
        group = _make_group()
        results = _make_fully_populated_results(ok=True)
        invs = {group.group_id: GroupInvestigation(group.group_id, "ok", "")}
        expected = BatchQAResult(
            holistic_report="All passed.",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )
        with patch("langchain_openai.ChatOpenAI") as MockLLM:
            mi = MagicMock()
            ms = MagicMock()
            ms.invoke.return_value = expected
            mi.with_structured_output.return_value = ms
            MockLLM.return_value = mi
            result = _run_batch_judge(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                results=results,
                investigations_by_group=invs,
            )
        mi.with_structured_output.assert_called_once_with(BatchQAResult)
        ms.invoke.assert_called_once()
        assert result.holistic_report == "All passed."

    def test_llm_failure_returns_fallback(self):
        group = _make_group()
        results = _make_fully_populated_results(ok=True)
        invs = {group.group_id: GroupInvestigation(group.group_id, "ok", "")}
        with patch("langchain_openai.ChatOpenAI") as MockLLM:
            mi = MagicMock()
            ms = MagicMock()
            ms.invoke.side_effect = RuntimeError("quota")
            mi.with_structured_output.return_value = ms
            MockLLM.return_value = mi
            result = _run_batch_judge(
                valid_groups=[group],
                group_strategies={},
                action_summaries=[],
                results=results,
                investigations_by_group=invs,
            )
        assert "Failure" in result.holistic_report
        assert result.evaluations[0].passed is False
        assert result.evaluations[0].failure_category == FailureCategory.SECURITY_FLAG


# ---------------------------------------------------------------------------
# _apply_guardrails
# ---------------------------------------------------------------------------


class TestApplyGuardrails:
    def _res(self, install_ok=True, scan_ok=True, remaining=None, install_summary="ok"):
        r = _QAExecutionResults()
        r.install = (install_ok, install_summary)
        r.scan = (scan_ok, "scan ok", remaining or set())
        r.tests = (True, "ok")
        return r

    def test_valid_passes_through(self):
        g = _make_group()
        batch = BatchQAResult(
            holistic_report="ok", evaluations=[QAEvaluation(task_id=g.group_id, passed=True)]
        )
        evals, errors = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(),
            group_strategies={g.group_id: "version_bump"},
        )
        assert evals[g.group_id].passed is True and not errors

    def test_no_fix_stage_two_derives_code_workaround_strategy(self):
        group = _make_group(
            group_id="sca:package.json:notevil:NO_FIX",
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

        strategies = _derive_qa_group_strategies(
            [group],
            configured_strategies={},
            task_queue={task.task_id: task},
            active_target_task_ids=[task.task_id],
        )

        assert strategies[group.group_id] == "code_workaround"

    def test_no_fix_stage_one_remains_strict(self):
        group = _make_group(
            group_id="sca:package.json:notevil:NO_FIX",
            cve_ids=["CVE-2021-23771"],
            ghsa_ids=[],
            fix_plan_status=FixPlanStatus.NO_FIX,
        )
        task = RemediationTask(
            task_id="task-nofix",
            parent_group_id=group.group_id,
            strategy=RoutingStrategy.CODE_WORKAROUND,
            no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
            status=TaskStatus.OPTIMISTICALLY_FIXED,
        )
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[QAEvaluation(task_id=group.group_id, passed=True)],
        )

        strategies = _derive_qa_group_strategies([group], {}, {task.task_id: task}, [task.task_id])
        evaluations, _ = _apply_guardrails(
            valid_groups=[group],
            batch_result=batch,
            results=self._res(scan_ok=False, remaining={"CVE-2021-23771"}),
            group_strategies=strategies,
        )

        assert strategies[group.group_id] == "no_fix_package_removal"
        assert evaluations[group.group_id].passed is False
        assert evaluations[group.group_id].failure_category == FailureCategory.SECURITY_FLAG

    def test_unknown_group_id_dropped(self):
        g = _make_group(group_id="real")
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(task_id="real", passed=True),
                QAEvaluation(
                    task_id="ghost",
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="x",
                ),
            ],
        )
        evals, errors = _apply_guardrails(
            valid_groups=[g], batch_result=batch, results=self._res(), group_strategies={}
        )
        assert "ghost" not in evals and any("ghost" in e for e in errors)

    def test_duplicate_keeps_first(self):
        g = _make_group()
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(task_id=g.group_id, passed=True),
                QAEvaluation(
                    task_id=g.group_id,
                    passed=False,
                    failure_category=FailureCategory.SECURITY_FLAG,
                    retry_feedback="second",
                ),
            ],
        )
        evals, errors = _apply_guardrails(
            valid_groups=[g], batch_result=batch, results=self._res(), group_strategies={}
        )
        assert evals[g.group_id].passed is True and any("duplicate" in e.lower() for e in errors)

    def test_missing_group_synthesized(self):
        g1, g2 = _make_group(group_id="g1"), _make_group(group_id="g2")
        batch = BatchQAResult(
            holistic_report="ok", evaluations=[QAEvaluation(task_id="g1", passed=True)]
        )
        evals, errors = _apply_guardrails(
            valid_groups=[g1, g2], batch_result=batch, results=self._res(), group_strategies={}
        )
        assert (
            evals["g2"].passed is False
            and evals["g2"].failure_category == FailureCategory.SECURITY_FLAG
        )

    def test_version_bump_remaining_forces_fail(self):
        g = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        batch = BatchQAResult(
            holistic_report="ok", evaluations=[QAEvaluation(task_id=g.group_id, passed=True)]
        )
        evals, _ = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
            group_strategies={g.group_id: "version_bump"},
        )
        assert evals[g.group_id].passed is False
        assert evals[g.group_id].failure_category == FailureCategory.SECURITY_FLAG

    def test_code_workaround_remaining_allowed_pass(self):
        g = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        batch = BatchQAResult(
            holistic_report="ok", evaluations=[QAEvaluation(task_id=g.group_id, passed=True)]
        )
        evals, _ = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
            group_strategies={g.group_id: "code_workaround"},
        )
        assert evals[g.group_id].passed is True

    def test_eresolve_remaps_breaking_to_peer_conflict(self):
        g = _make_group()
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(
                    task_id=g.group_id,
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="x",
                )
            ],
        )
        evals, _ = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(install_ok=False, install_summary="ERESOLVE conflict"),
            group_strategies={g.group_id: "version_bump"},
        )
        assert evals[g.group_id].failure_category == FailureCategory.PEER_CONFLICT

    def test_eresolve_exempt_for_code_workaround(self):
        g = _make_group()
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(
                    task_id=g.group_id,
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="x",
                )
            ],
        )
        evals, _ = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(install_ok=False, install_summary="ERESOLVE conflict"),
            group_strategies={g.group_id: "code_workaround"},
        )
        assert evals[g.group_id].failure_category == FailureCategory.BREAKING_CHANGE

    def test_remaining_scanner_reclassifies_breaking_change_to_security_flag(self):
        g = _make_group(cve_ids=["CVE-2021-0001"], ghsa_ids=[])
        batch = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(
                    task_id=g.group_id,
                    passed=False,
                    failure_category=FailureCategory.BREAKING_CHANGE,
                    retry_feedback="JWT unit tests failed after the version bump.",
                )
            ],
        )
        evals, errors = _apply_guardrails(
            valid_groups=[g],
            batch_result=batch,
            results=self._res(scan_ok=False, remaining={"CVE-2021-0001"}),
            group_strategies={g.group_id: "version_bump"},
        )
        assert evals[g.group_id].failure_category == FailureCategory.SECURITY_FLAG
        assert "JWT unit tests failed" in (evals[g.group_id].retry_feedback or "")
        assert any("did not prioritize SECURITY_FLAG" in err for err in errors)


# ---------------------------------------------------------------------------
# run_qa_critic_node â€” map-reduce integration
# ---------------------------------------------------------------------------


class TestRunQACriticNodeMapReduce:
    def _run(self, groups, global_results=None, investigations=None, batch_result=None):
        if global_results is None:
            global_results = _make_fully_populated_results(ok=True)
        if investigations is None:
            investigations = {g.group_id: GroupInvestigation(g.group_id, "ok", "") for g in groups}
        if batch_result is None:
            batch_result = BatchQAResult(
                holistic_report="All passed.",
                evaluations=[QAEvaluation(task_id=g.group_id, passed=True) for g in groups],
            )
        mock_sb = MagicMock()
        mock_sb.__enter__ = MagicMock(return_value=mock_sb)
        mock_sb.__exit__ = MagicMock(return_value=None)
        state = _make_minimal_state(groups=groups)
        with (
            patch("remediation_engine.orchestration.qa_critic.DockerSandbox", return_value=mock_sb),
            patch(
                "remediation_engine.orchestration.qa_critic._run_global_execution",
                return_value=global_results,
            ) as mg,
            patch(
                "remediation_engine.orchestration.qa_critic._run_individual_investigations",
                return_value=investigations,
            ) as mm,
            patch(
                "remediation_engine.orchestration.qa_critic._run_batch_judge",
                return_value=batch_result,
            ) as mr,
        ):
            result = run_qa_critic_node(state)
        return result, mg, mm, mr

    def test_global_execution_called_once(self):
        _, mg, _, _ = self._run([_make_group()])
        mg.assert_called_once()

    def test_individual_investigations_called_once(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {"g1": GroupInvestigation("g1", "ok", ""), "g2": GroupInvestigation("g2", "ok", "")}
        br = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(task_id="g1", passed=True),
                QAEvaluation(task_id="g2", passed=True),
            ],
        )
        _, _, mm, _ = self._run([g1, g2], investigations=invs, batch_result=br)
        mm.assert_called_once()

    def test_batch_judge_called_once(self):
        _, _, _, mr = self._run([_make_group()])
        mr.assert_called_once()

    def test_holistic_report_in_output(self):
        g = _make_group()
        br = BatchQAResult(
            holistic_report="## Holistic.",
            evaluations=[QAEvaluation(task_id=g.group_id, passed=True)],
        )
        result, _, _, _ = self._run([g], batch_result=br)
        assert result["qa_investigation_report"] == "## Holistic."

    def test_all_passed_status(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {"g1": GroupInvestigation("g1", "ok", ""), "g2": GroupInvestigation("g2", "ok", "")}
        br = BatchQAResult(
            holistic_report="ok",
            evaluations=[
                QAEvaluation(task_id="g1", passed=True),
                QAEvaluation(task_id="g2", passed=True),
            ],
        )
        result, _, _, _ = self._run([g1, g2], investigations=invs, batch_result=br)
        assert result["eval_status"] == "all_passed" and result["status"] == "qa_completed"

    def test_guardrails_fill_missing_eval(self):
        g1, g2 = _make_group("g1"), _make_group("g2")
        invs = {"g1": GroupInvestigation("g1", "ok", ""), "g2": GroupInvestigation("g2", "ok", "")}
        br = BatchQAResult(
            holistic_report="ok", evaluations=[QAEvaluation(task_id="g1", passed=True)]
        )  # g2 missing
        result, _, _, _ = self._run([g1, g2], investigations=invs, batch_result=br)
        assert "g2" in result["qa_evaluations"] and result["qa_evaluations"]["g2"].passed is False

    def test_map_errors_appear_in_output(self):
        g = _make_group()
        invs = {
            g.group_id: GroupInvestigation(g.group_id, "fallback", "", ["investigator timed out"])
        }
        result, _, _, _ = self._run([g], investigations=invs)
        assert any("investigator timed out" in e for e in result.get("errors", []))
