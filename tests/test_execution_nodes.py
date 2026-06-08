"""
tests/test_execution_nodes.py — Unit tests for the Phase 5 execution nodes.

All Docker sandbox calls and subprocess.run calls are mocked; no real Docker
daemon or ODC installation is required.

Coverage
--------
State:
  * pending_files == {} in initial_orchestrator_state

Editor node:
  * no edit_requests → status="no_edits"
  * invalid repo_root → status="edit_failed"
  * successful single edit → pending_files populated, status="edited"
  * missing file in sandbox → status="edit_failed"
  * old_text not found (zero matches) → status="edit_failed"
  * old_text ambiguous (multiple matches) → status="edit_failed"
  * npm manifest edited → npm install triggered; lockfile extracted
  * npm install failure → status="edit_failed", test_failures set
  * sandbox exception → status="edit_failed"

Scanner node:
  * no pending_files → status="scanned", scan_failures=None
  * ODC not on PATH → status="scanned", scan_failures=None (fallback)
  * ODC nonzero exit + no report → status="scan_failed"
  * ODC success, CVE still present → status="scan_failed"
  * ODC success, CVE resolved → status="scanned", scan_failures=None
  * ODC nonzero exit but report exists → still parsed (report-first)
  * no target CVEs in state → always pass

Tester node:
  * no pending_files → status="tested", test_failures=None
  * invalid repo_root → status="test_failed"
  * injects all pending_files before install/test
  * lockfile present in pending_files → uses npm ci
  * lockfile absent → uses npm install
  * install failure → status="test_failed", test_failures set
  * test failure → status="test_failed", test_failures set
  * success → status="tested", test_failures=None
  * sandbox exception → status="test_failed"
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from src.contracts.schemas import (
    EditRequest,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.state import DEFAULT_MAX_RETRIES, initial_orchestrator_state
from src.orchestrator.editor_node import run_editor_node
from src.orchestrator.scanner_node import run_scanner_node
from src.orchestrator.tester_node import run_tester_node


# ===========================================================================
# Shared factories
# ===========================================================================


def _edit(tmp_path: Path, file_path: str, old_text: str, new_text: str) -> EditRequest:
    return EditRequest(
        repo_root=str(tmp_path),
        file_path=file_path,
        old_text=old_text,
        new_text=new_text,
        dry_run=False,
    )


def _sca_group(cve_id: str = "CVE-2021-44228") -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id=cve_id,
        package_name="lodash",
        package_version="4.17.15",
    )
    return VulnerabilityGroup(
        group_id=f"sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        cve_ids=[cve_id],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _make_sandbox_mock(
    read_side_effect=None,
    read_return=None,
    run_exit_code: int = 0,
    run_stdout: str = "",
    run_stderr: str = "",
):
    """Return a fully configured DockerSandbox mock."""
    from src.contracts.schemas import CommandResult

    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=None)

    if read_side_effect is not None:
        mock.read_file.side_effect = read_side_effect
    elif read_return is not None:
        mock.read_file.return_value = read_return
    else:
        mock.read_file.return_value = '"lodash": "^4.17.15"'

    mock.run.return_value = CommandResult(
        exit_code=run_exit_code,
        stdout=run_stdout,
        stderr=run_stderr,
        duration_seconds=0.1,
    )
    mock.write_file.return_value = None
    return mock


# ===========================================================================
# 1. State defaults
# ===========================================================================


class TestStateDefaults:
    def test_pending_files_initialized_to_empty_dict(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        assert "pending_files" in state
        assert state["pending_files"] == {}

    def test_all_expected_keys_present(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        for key in (
            "repo_root", "valid_groups", "retry_count", "max_retries",
            "edit_requests", "pending_files", "test_failures", "scan_failures",
            "status", "errors",
        ):
            assert key in state, f"Key '{key}' missing from initial_orchestrator_state"


# ===========================================================================
# 2. Editor Node
# ===========================================================================


class TestEditorNodeGuards:
    def test_no_edits_returns_no_edits_status(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_editor_node(state)
        assert result["status"] == "no_edits"

    def test_empty_edit_requests_returns_no_edits(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = []
        result = run_editor_node(state)
        assert result["status"] == "no_edits"

    def test_invalid_repo_root_returns_edit_failed(self, tmp_path):
        state = initial_orchestrator_state("/nonexistent/xyz", [])
        state["edit_requests"] = [_edit(tmp_path, "package.json", "a", "b")]
        result = run_editor_node(state)
        assert result["status"] == "edit_failed"
        assert "test_failures" in result
        assert result["test_failures"]

    def test_empty_repo_root_returns_edit_failed(self, tmp_path):
        state = initial_orchestrator_state("", [])
        state["edit_requests"] = [_edit(tmp_path, "package.json", "a", "b")]
        result = run_editor_node(state)
        assert result["status"] == "edit_failed"


class TestEditorNodeHappyPath:
    def test_successful_edit_populates_pending_files(self, tmp_path):
        edit = _edit(tmp_path, "routes/login.ts", "const x = 1;", "const x = 2;")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock()
        # Sequence: read for edit, then extract
        mock_sb.read_file.side_effect = [
            "const x = 1;",   # read before patching
            "const x = 2;",   # extract after patching
        ]

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edited"
        assert "pending_files" in result
        assert "routes/login.ts" in result["pending_files"]
        assert result["test_failures"] is None
        assert result["scan_failures"] is None

    def test_write_file_called_with_new_content(self, tmp_path):
        old = "const x = 1;"
        new = "const x = 2;"
        edit = _edit(tmp_path, "routes/login.ts", old, new)
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock()
        mock_sb.read_file.side_effect = [old, new]

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            run_editor_node(state)

        write_calls = mock_sb.write_file.call_args_list
        assert len(write_calls) == 1
        _path_arg, content_arg = write_calls[0][0]
        assert new in content_arg

    def test_multiple_edits_all_extracted(self, tmp_path):
        edit1 = _edit(tmp_path, "package.json", '"lodash": "^4.17.15"', '"lodash": "^4.17.21"')
        edit2 = _edit(tmp_path, "routes/login.ts", "const x = 1;", "const x = 2;")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit1, edit2]

        from src.contracts.schemas import CommandResult

        mock_sb = _make_sandbox_mock()
        # Sequence for two edits where edit1 is package.json (triggers npm install):
        #   1. read package.json for edit
        #   2. read login.ts for edit
        #   3. npm install (via mock_sb.run)
        #   4. read package-lock.json (check if generated) → None (not generated)
        #   5. extract package.json
        #   6. extract login.ts
        mock_sb.read_file.side_effect = [
            '"lodash": "^4.17.15"',   # 1. read package.json
            "const x = 1;",           # 2. read login.ts
            None,                     # 3. read package-lock.json (not generated)
            '"lodash": "^4.17.21"',   # 4. extract package.json
            "const x = 2;",           # 5. extract login.ts
        ]
        mock_sb.run.return_value = CommandResult(
            exit_code=0, stdout="", stderr="", duration_seconds=0.5
        )

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edited"
        assert "package.json" in result["pending_files"]
        assert "routes/login.ts" in result["pending_files"]

    def test_clears_test_and_scan_failures_on_success(self, tmp_path):
        edit = _edit(tmp_path, "routes/login.ts", "const x = 1;", "const x = 2;")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]
        state["test_failures"] = "old failure"
        state["scan_failures"] = "old scan failure"

        mock_sb = _make_sandbox_mock()
        mock_sb.read_file.side_effect = ["const x = 1;", "const x = 2;"]

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["test_failures"] is None
        assert result["scan_failures"] is None


class TestEditorNodeFailures:
    def test_missing_file_in_sandbox_returns_edit_failed(self, tmp_path):
        edit = _edit(tmp_path, "missing.json", "old", "new")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock(read_return=None)

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "test_failures" in result
        assert result["test_failures"]

    def test_old_text_not_found_returns_edit_failed(self, tmp_path):
        edit = _edit(tmp_path, "package.json", "TEXT_NOT_IN_FILE", "new")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock(read_return='"lodash": "^4.17.15"')

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "not found" in result["test_failures"].lower()

    def test_old_text_ambiguous_returns_edit_failed(self, tmp_path):
        edit = _edit(tmp_path, "package.json", "dup", "new")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock(read_return="dup dup dup")  # matches 3×

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "ambiguous" in result["test_failures"].lower()

    def test_sandbox_exception_returns_edit_failed(self, tmp_path):
        edit = _edit(tmp_path, "package.json", "a", "b")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = MagicMock()
        mock_sb.__enter__ = MagicMock(side_effect=RuntimeError("Docker daemon down"))
        mock_sb.__exit__ = MagicMock(return_value=None)

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "Docker daemon down" in result["test_failures"]


class TestEditorNodeNpmManifest:
    def test_editing_package_json_triggers_npm_install(self, tmp_path):
        edit = _edit(tmp_path, "package.json", '"lodash": "^4.17.15"', '"lodash": "^4.17.21"')
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        from src.contracts.schemas import CommandResult

        mock_sb = _make_sandbox_mock()
        mock_sb.read_file.side_effect = [
            '"lodash": "^4.17.15"',   # read for edit
            None,                      # read package-lock.json (not generated)
            '"lodash": "^4.17.21"',   # extract package.json
        ]
        mock_sb.run.return_value = CommandResult(
            exit_code=0, stdout="npm ok", stderr="", duration_seconds=0.5
        )

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        mock_sb.run.assert_called_once()
        run_cmd = mock_sb.run.call_args[0][0]
        assert "npm install" in run_cmd
        assert "--ignore-scripts" in run_cmd
        assert "--package-lock-only" in run_cmd
        assert result["status"] == "edited"

    def test_npm_install_failure_returns_edit_failed(self, tmp_path):
        edit = _edit(tmp_path, "package.json", '"lodash": "^4.17.15"', '"lodash": "^4.17.21"')
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        from src.contracts.schemas import CommandResult

        mock_sb = _make_sandbox_mock()
        mock_sb.read_file.return_value = '"lodash": "^4.17.15"'
        mock_sb.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="Cannot resolve dependency",
            duration_seconds=0.5,
        )

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edit_failed"
        assert "Cannot resolve dependency" in result["test_failures"]

    def test_lockfile_added_to_pending_files_after_npm_install(self, tmp_path):
        edit = _edit(tmp_path, "package.json", '"lodash": "^4.17.15"', '"lodash": "^4.17.21"')
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        from src.contracts.schemas import CommandResult

        lock_content = '{"lockfileVersion": 2}'
        mock_sb = _make_sandbox_mock()
        # read_file call order:
        #   1. read package.json for edit
        #   2. read package-lock.json (exists after npm install)
        #   3. extract package.json
        #   4. extract package-lock.json
        mock_sb.read_file.side_effect = [
            '"lodash": "^4.17.15"',
            lock_content,
            '"lodash": "^4.17.21"',
            lock_content,
        ]
        mock_sb.run.return_value = CommandResult(
            exit_code=0, stdout="", stderr="", duration_seconds=0.5
        )

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        assert result["status"] == "edited"
        assert "package.json" in result["pending_files"]
        assert "package-lock.json" in result["pending_files"]
        assert result["pending_files"]["package-lock.json"] == lock_content

    def test_non_manifest_edit_does_not_trigger_npm_install(self, tmp_path):
        edit = _edit(tmp_path, "routes/login.ts", "const x = 1;", "const x = 2;")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["edit_requests"] = [edit]

        mock_sb = _make_sandbox_mock()
        mock_sb.read_file.side_effect = ["const x = 1;", "const x = 2;"]

        with patch("src.orchestrator.editor_node.DockerSandbox", return_value=mock_sb):
            result = run_editor_node(state)

        mock_sb.run.assert_not_called()
        assert result["status"] == "edited"


# ===========================================================================
# 3. Scanner Node
# ===========================================================================


class TestScannerNodeGuards:
    def test_no_pending_files_returns_scanned(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_scanner_node(state)
        assert result["status"] == "scanned"
        assert result["scan_failures"] is None

    def test_odc_not_on_path_returns_scanned(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["pending_files"] = {"package.json": '{"dependencies": {"lodash": "4.17.21"}}'}

        with patch("shutil.which", return_value=None):
            result = run_scanner_node(state)

        assert result["status"] == "scanned"
        assert result["scan_failures"] is None


class TestScannerNodeODCFailures:
    def test_odc_nonzero_no_report_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["pending_files"] = {"package.json": "{}"}

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 1
        proc.stdout = "some output"
        proc.stderr = "ODC crashed"

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=proc):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert result["scan_failures"]

    def test_odc_nonzero_with_report_still_parsed(self, tmp_path):
        """If ODC exits non-zero but the report exists, we still parse it."""
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["pending_files"] = {"package.json": "{}"}

        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 1
        proc.stdout = ""
        proc.stderr = "warning: ..."

        # Report with NO vulnerabilities → CVE resolved
        clean_report = {"dependencies": []}

        def fake_run(cmd, **kwargs):
            # Write the report file into the host tmpdir mapped from docker mount
            out_dir = None
            for part in cmd:
                if part.endswith(":/scan"):
                    out_dir = part.split(":/scan")[0]
                    break
            if out_dir:
                Path(out_dir, "dependency-check-report.json").write_text(
                    json.dumps(clean_report), encoding="utf-8"
                )
            return proc

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=fake_run):
            result = run_scanner_node(state)

        # CVE-2021-44228 not found in report → scanned
        assert result["status"] == "scanned"
        assert result["scan_failures"] is None

    def test_odc_timeout_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        state["pending_files"] = {"package.json": "{}"}

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("odc", 300)):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "timed out" in result["scan_failures"].lower()


class TestScannerNodeCVEComparison:
    def _make_odc_report(self, cve_ids: List[str]) -> dict:
        """Build a minimal ODC report containing the given CVE IDs."""
        vulns = [
            {
                "name": cve,
                "severity": "HIGH",
                "description": f"Test vuln {cve}",
                "cvssv3": {"baseScore": 7.5, "baseSeverity": "HIGH"},
            }
            for cve in cve_ids
        ]
        return {
            "dependencies": [
                {
                    "fileName": "lodash-4.17.15.tgz",
                    "filePath": "/tmp/lodash.tgz",
                    "packages": [{"id": "pkg:npm/lodash@4.17.15"}],
                    "vulnerabilities": vulns,
                }
            ]
        }

    def _fake_run_factory(self, report_data: dict):
        """Return a side_effect function that writes *report_data* to the host tmpdir mapped from docker mount."""
        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = ""
        proc.stderr = ""

        def fake_run(cmd, **kwargs):
            out_dir = None
            for part in cmd:
                if part.endswith(":/scan"):
                    out_dir = part.split(":/scan")[0]
                    break
            if out_dir:
                Path(out_dir, "dependency-check-report.json").write_text(
                    json.dumps(report_data), encoding="utf-8"
                )
            return proc

        return fake_run

    def test_cve_still_present_returns_scan_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group("CVE-2021-44228")])
        state["pending_files"] = {"package.json": "{}"}

        report = self._make_odc_report(["CVE-2021-44228"])

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=self._fake_run_factory(report)):
            result = run_scanner_node(state)

        assert result["status"] == "scan_failed"
        assert "CVE-2021-44228" in result["scan_failures"]

    def test_cve_resolved_returns_scanned(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [_sca_group("CVE-2021-44228")])
        state["pending_files"] = {"package.json": "{}"}

        # Report contains a different CVE — target one is resolved
        report = self._make_odc_report(["CVE-2020-99999"])

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=self._fake_run_factory(report)):
            result = run_scanner_node(state)

        assert result["status"] == "scanned"
        assert result["scan_failures"] is None

    def test_no_target_cves_always_passes(self, tmp_path):
        """When valid_groups has no CVE IDs, any scan result passes."""
        group_no_cves = VulnerabilityGroup(
            group_id="sca:package.json:lodash",
            issue_type=IssueType.SCA,
            vulnerable_component="lodash",
            cve_ids=[],  # no CVEs
            versions=["4.17.15"],
            sources=[IssueSource.ODC],
            representative_issue_id=VulnerabilityIssue(
                source=IssueSource.ODC,
                issue_type=IssueType.SCA,
                severity=Severity.HIGH,
            ).id,
            issues=[],
        )
        state = initial_orchestrator_state(str(tmp_path), [group_no_cves])
        state["pending_files"] = {"package.json": "{}"}

        # Report has many CVEs — but none are targeted
        report = self._make_odc_report(["CVE-2020-00001", "CVE-2020-00002"])

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=self._fake_run_factory(report)):
            result = run_scanner_node(state)

        assert result["status"] == "scanned"

    def test_manifest_files_copied_from_host(self, tmp_path):
        """Unchanged host manifests are copied into the scan workspace."""
        # Write a real package-lock.json to host repo
        (tmp_path / "package-lock.json").write_text('{"lockfileVersion": 2}', encoding="utf-8")

        state = initial_orchestrator_state(str(tmp_path), [_sca_group()])
        # Only package.json is in pending_files; package-lock.json is on host
        state["pending_files"] = {"package.json": '{"dependencies": {"lodash": "4.17.21"}}'}

        report = {"dependencies": []}  # clean — CVE resolved

        def fake_run(cmd, **kwargs):
            out_dir = None
            for part in cmd:
                if part.endswith(":/scan"):
                    out_dir = part.split(":/scan")[0]
                    break
            if out_dir:
                Path(out_dir, "dependency-check-report.json").write_text(
                    json.dumps(report), encoding="utf-8"
                )
            proc = MagicMock(spec=subprocess.CompletedProcess)
            proc.returncode = 0
            proc.stdout = ""
            proc.stderr = ""
            return proc

        copied_srcs = []

        def fake_copy2(src, dst):
            copied_srcs.append(src)

        with patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", side_effect=fake_run), \
             patch("src.orchestrator.scanner_node.shutil.copy2", side_effect=fake_copy2):
            run_scanner_node(state)

        # package-lock.json should have been copied from the host repo
        assert any("package-lock.json" in src for src in copied_srcs), (
            f"Expected package-lock.json to be copied from host; got copies: {copied_srcs}"
        )


# ===========================================================================
# 4. Tester Node
# ===========================================================================


class TestTesterNodeGuards:
    def test_no_pending_files_returns_tested(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        result = run_tester_node(state)
        assert result["status"] == "tested"
        assert result["test_failures"] is None

    def test_invalid_repo_root_returns_test_failed(self):
        state = initial_orchestrator_state("/nonexistent/xyz", [])
        state["pending_files"] = {"package.json": "{}"}
        result = run_tester_node(state)
        assert result["status"] == "test_failed"
        assert result["test_failures"]


class TestTesterNodeInjection:
    def test_all_pending_files_injected(self, tmp_path):
        pending = {
            "package.json": '{"name":"test"}',
            "routes/login.ts": "const x = 2;",
        }
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = pending

        from src.contracts.schemas import CommandResult
        mock_sb = _make_sandbox_mock(run_exit_code=0)

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        written_paths = {c[0][0] for c in mock_sb.write_file.call_args_list}
        assert "package.json" in written_paths
        assert "routes/login.ts" in written_paths

    def test_files_injected_before_install(self, tmp_path):
        """write_file must be called before the first sandbox.run call."""
        pending = {"package.json": "{}"}
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = pending

        call_log = []

        from src.contracts.schemas import CommandResult

        mock_sb = MagicMock()
        mock_sb.__enter__ = MagicMock(return_value=mock_sb)
        mock_sb.__exit__ = MagicMock(return_value=None)
        mock_sb.write_file.side_effect = lambda path, content: call_log.append(("write", path))
        mock_sb.run.side_effect = lambda cmd, **kwargs: (
            call_log.append(("run", cmd)),
            CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.1),
        )[1]

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        # All writes must precede all runs
        write_indices = [i for i, (kind, _) in enumerate(call_log) if kind == "write"]
        run_indices = [i for i, (kind, _) in enumerate(call_log) if kind == "run"]
        assert write_indices, "No write calls recorded"
        assert run_indices, "No run calls recorded"
        assert max(write_indices) < min(run_indices), (
            "write_file must be called before any sandbox.run"
        )


class TestTesterNodeLockfileDetection:
    def test_lockfile_in_pending_files_uses_npm_ci(self, tmp_path):
        pending = {
            "package.json": "{}",
            "package-lock.json": '{"lockfileVersion":2}',
        }
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = pending

        from src.contracts.schemas import CommandResult
        mock_sb = _make_sandbox_mock(run_exit_code=0)

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        install_cmd = mock_sb.run.call_args_list[0][0][0]
        assert "npm ci" in install_cmd
        assert "--ignore-scripts" in install_cmd

    def test_lockfile_on_host_uses_npm_ci(self, tmp_path):
        (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult
        mock_sb = _make_sandbox_mock(run_exit_code=0)

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        install_cmd = mock_sb.run.call_args_list[0][0][0]
        assert "npm ci" in install_cmd

    def test_no_lockfile_uses_npm_install(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult
        mock_sb = _make_sandbox_mock(run_exit_code=0)

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        install_cmd = mock_sb.run.call_args_list[0][0][0]
        assert "npm install" in install_cmd
        assert "npm ci" not in install_cmd


class TestTesterNodeOutcomes:
    def test_install_failure_returns_test_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult

        mock_sb = _make_sandbox_mock()
        mock_sb.run.return_value = CommandResult(
            exit_code=1,
            stdout="",
            stderr="npm ERR! peer dependency conflict",
            duration_seconds=0.5,
        )

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            result = run_tester_node(state)

        assert result["status"] == "test_failed"
        assert "peer dependency conflict" in result["test_failures"]

    def test_test_failure_returns_test_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult

        install_ok = CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=0.5)
        test_fail = CommandResult(
            exit_code=1,
            stdout="1 failing\n  AssertionError: expected 200 to equal 201",
            stderr="",
            duration_seconds=2.0,
        )
        mock_sb = _make_sandbox_mock()
        mock_sb.run.side_effect = [install_ok, test_fail]

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            result = run_tester_node(state)

        assert result["status"] == "test_failed"
        assert "AssertionError" in result["test_failures"]

    def test_success_returns_tested(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult

        install_ok = CommandResult(exit_code=0, stdout="added 100 packages", stderr="", duration_seconds=5.0)
        test_ok = CommandResult(exit_code=0, stdout="passing (42)", stderr="", duration_seconds=10.0)
        mock_sb = _make_sandbox_mock()
        mock_sb.run.side_effect = [install_ok, test_ok]

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            result = run_tester_node(state)

        assert result["status"] == "tested"
        assert result["test_failures"] is None

    def test_sandbox_exception_returns_test_failed(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        mock_sb = MagicMock()
        mock_sb.__enter__ = MagicMock(side_effect=RuntimeError("daemon unreachable"))
        mock_sb.__exit__ = MagicMock(return_value=None)

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            result = run_tester_node(state)

        assert result["status"] == "test_failed"
        assert "daemon unreachable" in result["test_failures"]

    def test_npm_test_called_after_successful_install(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult

        install_ok = CommandResult(exit_code=0, stdout="", stderr="", duration_seconds=1.0)
        test_ok = CommandResult(exit_code=0, stdout="passing", stderr="", duration_seconds=2.0)
        mock_sb = _make_sandbox_mock()
        mock_sb.run.side_effect = [install_ok, test_ok]

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            run_tester_node(state)

        run_cmds = [c[0][0] for c in mock_sb.run.call_args_list]
        assert len(run_cmds) == 2
        assert any("npm test" in cmd for cmd in run_cmds)

    def test_npm_test_not_called_if_install_fails(self, tmp_path):
        """If install fails, npm test should NOT be invoked."""
        state = initial_orchestrator_state(str(tmp_path), [])
        state["pending_files"] = {"package.json": "{}"}

        from src.contracts.schemas import CommandResult

        install_fail = CommandResult(exit_code=1, stdout="", stderr="fail", duration_seconds=1.0)
        mock_sb = _make_sandbox_mock()
        mock_sb.run.return_value = install_fail

        with patch("src.orchestrator.tester_node.DockerSandbox", return_value=mock_sb):
            result = run_tester_node(state)

        assert mock_sb.run.call_count == 1  # only install was called
        assert result["status"] == "test_failed"
