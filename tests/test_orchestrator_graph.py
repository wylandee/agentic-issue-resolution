"""
Tests for the Phase 4.1 LangGraph remediation orchestrator.

Structure
---------
TestEditRequestBuilder     — unit tests for edit_request_builder.build_edit_request
TestLocatorNode            — unit tests for graph.locator_node (mocked locators)
TestPlannerNode            — unit tests for graph.planner_node (mocked fix_planner)
TestEditorNode             — unit tests for graph.editor_node (mocked apply_edit)
TestGraphSCADryRun         — integration: SCA direct dependency dry-run through full graph
TestGraphSCANoFix          — integration: SCA no-fix stops before editor
TestGraphSAST              — integration: SAST stops after locator (before planner)
TestGraphFailure           — integration: failure paths (missing file, ambiguous anchor)
"""
from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.contracts.schemas import (
    ASTNodeType,
    EditRequest,
    EditResult,
    EditStatus,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LineRange,
    LocalizedIssue,
    Severity,
    VulnerabilityIssue,
)
from src.orchestrator.edit_request_builder import (
    _find_version_line,
    build_edit_request,
)
from src.orchestrator.graph import (
    _route_after_edit_request_builder,
    _route_after_locator,
    _route_after_planner,
    build_remediation_graph,
    edit_request_builder_node,
    editor_node,
    locator_node,
    planner_node,
    run_remediation,
)
from src.orchestrator.state import RemediationState

# ---------------------------------------------------------------------------
# Shared helpers / factories
# ---------------------------------------------------------------------------


def _sca_issue(
    *,
    package_name: str = "lodash",
    package_version: str = "4.17.20",
    file_path: Optional[str] = "package.json",
    cve_id: Optional[str] = "CVE-2021-23337",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        file_path=file_path,
        package_name=package_name,
        package_version=package_version,
        cve_id=cve_id,
    )


def _sast_issue(*, file_path: str = "src/app.js", line_start: int = 5) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        file_path=file_path,
        line_range=LineRange(start=line_start, end=line_start),
        message="Detected XSS vulnerability",
    )


def _localized_sca(
    issue: VulnerabilityIssue,
    *,
    manifest_file: str = "package.json",
    manifest_snippet: Optional[str] = None,
    is_direct: bool = True,
    manifest_line: int = 3,
    confidence: float = 0.95,
) -> LocalizedIssue:
    snippet = manifest_snippet or f'  "lodash": "^{issue.package_version}"'
    return LocalizedIssue(
        issue=issue,
        manifest_file=manifest_file,
        is_direct_dependency=is_direct,
        manifest_line=manifest_line,
        manifest_snippet=snippet,
        package_manager="npm",
        localization_confidence=confidence,
    )


def _version_found_plan(fixed_version: str = "4.17.21") -> FixPlan:
    return FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        fixed_version=fixed_version,
        workaround_snippets=None,
        instruction=f"Upgrade to {fixed_version}",
        strategy_used="osv_api",
    )


def _no_fix_plan() -> FixPlan:
    return FixPlan(
        status=FixPlanStatus.NO_FIX,
        fixed_version=None,
        workaround_snippets=None,
        instruction="No fix available",
        strategy_used="none",
    )


def _base_state(issue: VulnerabilityIssue, tmp_path: Path) -> RemediationState:
    return {
        "issue": issue,
        "repo_root": str(tmp_path),
        "status": "pending",
        "dry_run": True,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# TestEditRequestBuilder
# ---------------------------------------------------------------------------


class TestEditRequestBuilder:
    def test_find_version_line_json_style(self):
        snippet = '  "lodash": "^4.17.20",\n  "express": "^4.18.0"'
        old, new = _find_version_line(snippet, "lodash", "4.17.20", "4.17.21")
        assert old is not None
        assert "4.17.20" in old
        assert "4.17.21" in new

    def test_find_version_line_no_prefix(self):
        snippet = '  "lodash": "4.17.20"'
        old, new = _find_version_line(snippet, "lodash", "4.17.20", "4.17.21")
        assert old is not None
        assert new == '  "lodash": "4.17.21"'

    def test_find_version_line_scoped_package(self):
        snippet = '  "@angular/core": "^12.0.0"'
        old, new = _find_version_line(snippet, "@angular/core", "12.0.0", "12.2.17")
        assert old is not None
        assert "12.2.17" in new

    def test_find_version_line_requirements_txt(self):
        snippet = "requests==2.26.0\nflask>=2.0.0"
        old, new = _find_version_line(snippet, "requests", "2.26.0", "2.28.0")
        assert old is not None
        assert "2.28.0" in new

    def test_find_version_line_not_found(self):
        snippet = '  "express": "^4.18.0"'
        old, new = _find_version_line(snippet, "lodash", "4.17.20", "4.17.21")
        assert old is None
        assert new is None

    def test_build_edit_request_success(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue, manifest_snippet='  "lodash": "^4.17.20"')
        plan = _version_found_plan("4.17.21")
        req, err = build_edit_request(localized, plan, str(tmp_path), dry_run=True)
        assert err is None
        assert req is not None
        assert req.old_text == '  "lodash": "^4.17.20"'
        assert "4.17.21" in req.new_text
        assert req.dry_run is True
        assert req.file_path == "package.json"
        assert req.issue_id == issue.id

    def test_build_edit_request_no_manifest_file(self, tmp_path: Path):
        issue = _sca_issue()
        localized = LocalizedIssue(issue=issue, localization_confidence=0.95)
        plan = _version_found_plan()
        req, err = build_edit_request(localized, plan, str(tmp_path))
        assert req is None
        assert "manifest_file" in err

    def test_build_edit_request_no_package_version(self, tmp_path: Path):
        issue = _sca_issue(package_version=None)
        localized = _localized_sca(issue)
        plan = _version_found_plan()
        req, err = build_edit_request(localized, plan, str(tmp_path))
        assert req is None
        assert "package_version" in err

    def test_build_edit_request_no_snippet(self, tmp_path: Path):
        issue = _sca_issue()
        localized = LocalizedIssue(
            issue=issue,
            manifest_file="package.json",
            localization_confidence=0.95,
            manifest_snippet=None,
        )
        plan = _version_found_plan()
        req, err = build_edit_request(localized, plan, str(tmp_path))
        assert req is None
        assert "manifest_snippet" in err

    def test_build_edit_request_version_not_in_snippet(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue, manifest_snippet='  "express": "^4.18.0"')
        plan = _version_found_plan()
        req, err = build_edit_request(localized, plan, str(tmp_path))
        assert req is None
        assert err is not None

    def test_build_edit_request_rationale_contains_cve(self, tmp_path: Path):
        issue = _sca_issue(cve_id="CVE-2021-23337")
        localized = _localized_sca(issue, manifest_snippet='  "lodash": "4.17.20"')
        plan = _version_found_plan()
        req, err = build_edit_request(localized, plan, str(tmp_path))
        assert req is not None
        assert "CVE-2021-23337" in req.rationale


# ---------------------------------------------------------------------------
# TestLocatorNode
# ---------------------------------------------------------------------------


class TestLocatorNode:
    def test_sca_locator_called(self, tmp_path: Path):
        issue = _sca_issue()
        state = _base_state(issue, tmp_path)
        localized = _localized_sca(issue)

        # The locators are lazily imported inside locator_node; patch at the source module
        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized):
            result = locator_node(state)

        assert result["status"] == "located"
        assert result["localized_issue"] is localized


    def test_sca_zero_confidence_sets_failed(self, tmp_path: Path):
        issue = _sca_issue()
        state = _base_state(issue, tmp_path)
        low_conf = LocalizedIssue(issue=issue, localization_confidence=0.0)

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=low_conf):
            result = locator_node(state)

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_sast_locator_called(self, tmp_path: Path):
        issue = _sast_issue()
        state = _base_state(issue, tmp_path)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.js").write_text("function f() { eval(x); }\n")

        result = locator_node(state)
        # Should succeed (file exists) regardless of confidence level
        assert result.get("localized_issue") is not None
        assert result.get("status") in ("located", "failed")

    def test_sca_located_sets_correct_status(self, tmp_path: Path):
        issue = _sca_issue()
        state = _base_state(issue, tmp_path)
        localized = _localized_sca(issue)

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized):
            result = locator_node(state)

        assert result["status"] == "located"
        assert result["localized_issue"] is localized


# ---------------------------------------------------------------------------
# TestPlannerNode
# ---------------------------------------------------------------------------


class TestPlannerNode:
    def test_sast_skips_planning(self, tmp_path: Path):
        issue = _sast_issue()
        state: RemediationState = {
            **_base_state(issue, tmp_path),
            "localized_issue": LocalizedIssue(issue=issue, localization_confidence=0.6),
            "status": "located",
        }
        result = planner_node(state)
        assert result["status"] == "localized_needs_remedy_agent"
        # fix_plan should NOT be set for SAST
        assert "fix_plan" not in result

    def test_sca_version_found_routes_to_builder(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue)
        state: RemediationState = {
            **_base_state(issue, tmp_path),
            "localized_issue": localized,
            "status": "located",
        }
        plan_dict = {
            "status": "version_found",
            "fixed_version": "4.17.21",
            "workaround_snippets": None,
            "instruction": "Upgrade to 4.17.21",
            "strategy_used": "osv_api",
        }
        with patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = planner_node(state)

        assert result["status"] == "planned_version_found"
        assert result["fix_plan"].fixed_version == "4.17.21"

    def test_sca_no_fix_stops_at_planner(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue)
        state: RemediationState = {
            **_base_state(issue, tmp_path),
            "localized_issue": localized,
            "status": "located",
        }
        plan_dict = {
            "status": "no_fix",
            "fixed_version": None,
            "workaround_snippets": None,
            "instruction": "No fix available",
            "strategy_used": "none",
        }
        with patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = planner_node(state)

        assert result["status"] == "planned_no_auto_edit"

    def test_sca_workaround_stops_at_planner(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue)
        state: RemediationState = {
            **_base_state(issue, tmp_path),
            "localized_issue": localized,
            "status": "located",
        }
        plan_dict = {
            "status": "workaround_found",
            "fixed_version": None,
            "workaround_snippets": ["pin to 4.17.21"],
            "instruction": "Apply workaround",
            "strategy_used": "serper",
        }
        with patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = planner_node(state)

        assert result["status"] == "planned_workaround_found"


# ---------------------------------------------------------------------------
# TestEditorNode
# ---------------------------------------------------------------------------


class TestEditorNode:
    def _make_edit_request(self, tmp_path: Path, *, dry_run: bool = True) -> EditRequest:
        f = tmp_path / "package.json"
        f.write_text('{\n  "lodash": "4.17.20"\n}\n')
        return EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='  "lodash": "4.17.20"',
            new_text='  "lodash": "4.17.21"',
            dry_run=dry_run,
        )

    def test_dry_run_sets_status_dry_run(self, tmp_path: Path):
        req = self._make_edit_request(tmp_path, dry_run=True)
        state: RemediationState = {
            "issue": _sca_issue(),
            "repo_root": str(tmp_path),
            "status": "edit_request_ready",
            "dry_run": True,
            "errors": [],
            "edit_request": req,
        }
        result = editor_node(state)
        assert result["status"] == "dry_run"
        assert result["edit_result"].status == EditStatus.DRY_RUN
        assert result["edit_result"].unified_diff is not None

    def test_applied_sets_status_edited(self, tmp_path: Path):
        req = self._make_edit_request(tmp_path, dry_run=False)
        state: RemediationState = {
            "issue": _sca_issue(),
            "repo_root": str(tmp_path),
            "status": "edit_request_ready",
            "dry_run": False,
            "errors": [],
            "edit_request": req,
        }
        result = editor_node(state)
        assert result["status"] == "edited"
        assert result["edit_result"].status == EditStatus.APPLIED
        # Verify file was changed
        content = (tmp_path / "package.json").read_text()
        assert "4.17.21" in content

    def test_ambiguous_anchor_fails(self, tmp_path: Path):
        f = tmp_path / "package.json"
        f.write_text(
            '  "lodash": "4.17.20"\n  "lodash": "4.17.20"\n'
        )
        req = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='  "lodash": "4.17.20"',
            new_text='  "lodash": "4.17.21"',
            dry_run=True,
        )
        state: RemediationState = {
            "issue": _sca_issue(),
            "repo_root": str(tmp_path),
            "status": "edit_request_ready",
            "dry_run": True,
            "errors": [],
            "edit_request": req,
        }
        result = editor_node(state)
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_no_edit_request_skips(self, tmp_path: Path):
        state: RemediationState = {
            "issue": _sca_issue(),
            "repo_root": str(tmp_path),
            "status": "planned_no_auto_edit",
            "dry_run": True,
            "errors": [],
            "edit_request": None,
        }
        result = editor_node(state)
        # Returns empty dict — does not alter status
        assert result == {}


# ---------------------------------------------------------------------------
# TestGraphSCADryRun — full graph integration
# ---------------------------------------------------------------------------


class TestGraphSCADryRun:
    """SCA direct dependency bump: mock locator + planner, use real editor (dry-run)."""

    def test_sca_direct_dep_dry_run(self, tmp_path: Path):
        """Full path through locator→planner→edit_request_builder→editor (dry_run=True)."""
        # Set up a real package.json
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text(
            json.dumps(
                {"dependencies": {"lodash": "4.17.20"}}, indent=2
            )
        )
        # The snippet that the locator would return matches the exact file line
        snippet = '    "lodash": "4.17.20"'

        issue = _sca_issue()
        localized = _localized_sca(
            issue,
            manifest_file="package.json",
            manifest_snippet=snippet,
            confidence=0.95,
        )
        plan_dict = {
            "status": "version_found",
            "fixed_version": "4.17.21",
            "workaround_snippets": None,
            "instruction": "Upgrade lodash to 4.17.21",
            "strategy_used": "osv_api",
        }

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized), \
             patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = run_remediation(issue, str(tmp_path), dry_run=True)

        assert result["status"] == "dry_run"
        assert result.get("edit_result") is not None
        assert result["edit_result"].status == EditStatus.DRY_RUN
        assert result["edit_result"].unified_diff is not None
        # File should NOT have been changed (dry_run=True)
        assert "4.17.20" in pkg_json.read_text()

    def test_sca_direct_dep_applied(self, tmp_path: Path):
        """Full path with dry_run=False — file should be updated on disk."""
        pkg_json = tmp_path / "package.json"
        pkg_json.write_text('{\n  "dependencies": {\n    "lodash": "4.17.20"\n  }\n}\n')
        snippet = '    "lodash": "4.17.20"'

        issue = _sca_issue()
        localized = _localized_sca(issue, manifest_snippet=snippet, confidence=0.95)
        plan_dict = {
            "status": "version_found",
            "fixed_version": "4.17.21",
            "workaround_snippets": None,
            "instruction": "Upgrade",
            "strategy_used": "local_regex",
        }

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized), \
             patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = run_remediation(issue, str(tmp_path), dry_run=False)

        assert result["status"] == "edited"
        assert "4.17.21" in pkg_json.read_text()


# ---------------------------------------------------------------------------
# TestGraphSCANoFix
# ---------------------------------------------------------------------------


class TestGraphSCANoFix:
    def test_no_fix_stops_before_editor(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue)
        plan_dict = {
            "status": "no_fix",
            "fixed_version": None,
            "workaround_snippets": None,
            "instruction": "No fix found",
            "strategy_used": "none",
        }

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized), \
             patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = run_remediation(issue, str(tmp_path))

        assert result["status"] == "planned_no_auto_edit"
        # editor should never have run
        assert result.get("edit_result") is None
        assert result.get("edit_request") is None

    def test_workaround_stops_before_editor(self, tmp_path: Path):
        issue = _sca_issue()
        localized = _localized_sca(issue)
        plan_dict = {
            "status": "workaround_found",
            "fixed_version": None,
            "workaround_snippets": ["pin manually"],
            "instruction": "Apply workaround",
            "strategy_used": "serper",
        }

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized), \
             patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = run_remediation(issue, str(tmp_path))

        assert result["status"] == "planned_workaround_found"
        assert result.get("edit_result") is None


# ---------------------------------------------------------------------------
# TestGraphSAST
# ---------------------------------------------------------------------------


class TestGraphSAST:
    def test_sast_stops_after_locator(self, tmp_path: Path):
        """SAST flows: locator → planner (which immediately returns needs_remedy_agent)."""
        issue = _sast_issue()
        (tmp_path / "src").mkdir()
        js_file = tmp_path / "src" / "app.js"
        js_file.write_text(
            textwrap.dedent("""\
                const express = require('express');
                function handleReq(req, res) {
                    const id = req.params.id;
                    eval(id);
                }
            """)
        )
        result = run_remediation(issue, str(tmp_path), dry_run=True)

        # Localized issue should be populated
        assert result.get("localized_issue") is not None
        # Status should be the SAST-specific exit status
        assert result["status"] == "localized_needs_remedy_agent"
        # No planner or editor output for SAST
        assert result.get("fix_plan") is None
        assert result.get("edit_result") is None

    def test_sast_localized_issue_has_symbol(self, tmp_path: Path):
        issue = _sast_issue(file_path="src/app.js", line_start=4)
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.js").write_text(
            "function handleReq(req, res) {\n"
            "    const id = req.params.id;\n"
            "    eval(id);\n"
            "    res.send(id);\n"
            "}\n"
        )
        result = run_remediation(issue, str(tmp_path))
        li = result.get("localized_issue")
        assert li is not None
        # enclosing_symbol should be populated (SAST locator with tree-sitter)
        assert li.enclosing_symbol == "handleReq"


# ---------------------------------------------------------------------------
# TestGraphFailure
# ---------------------------------------------------------------------------


class TestGraphFailure:
    def test_missing_repo_file_produces_failed(self, tmp_path: Path):
        """SCA issue where the manifest file does not exist → locator returns 0.0 confidence."""
        issue = _sca_issue()
        # No files in tmp_path
        with patch("src.tools.manifest_locator.locate_from_issue") as mock_loc:
            mock_loc.return_value = LocalizedIssue(
                issue=issue, localization_confidence=0.0
            )
            result = run_remediation(issue, str(tmp_path))

        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_ambiguous_edit_anchor_produces_failed(self, tmp_path: Path):
        """Build a file with duplicate lines so apply_edit rejects with AMBIGUOUS."""
        # Package.json with duplicate version line to trigger ambiguity
        pkg_json = tmp_path / "package.json"
        dupe_line = '    "lodash": "4.17.20"'
        pkg_json.write_text(f"{dupe_line}\n{dupe_line}\n")

        issue = _sca_issue()
        localized = _localized_sca(
            issue,
            manifest_snippet=dupe_line,
            confidence=0.95,
        )
        plan_dict = {
            "status": "version_found",
            "fixed_version": "4.17.21",
            "workaround_snippets": None,
            "instruction": "Upgrade",
            "strategy_used": "osv_api",
        }

        with patch("src.tools.manifest_locator.locate_from_issue", return_value=localized), \
             patch("src.tools.fix_planner.plan_fix", return_value=plan_dict):
            result = run_remediation(issue, str(tmp_path), dry_run=True)

        assert result["status"] == "failed"
        assert result.get("edit_result") is not None
        assert result["edit_result"].status in (EditStatus.REJECTED, EditStatus.ERROR)

    def test_errors_accumulate_across_nodes(self, tmp_path: Path):
        """Verify the Annotated[list, operator.add] reducer works end-to-end."""
        issue = _sca_issue()
        with patch("src.tools.manifest_locator.locate_from_issue") as mock_loc:
            mock_loc.return_value = LocalizedIssue(
                issue=issue, localization_confidence=0.0
            )
            result = run_remediation(issue, str(tmp_path))

        assert isinstance(result["errors"], list)
        # At least one error was appended by the locator
        assert len(result["errors"]) >= 1

    def test_run_remediation_always_returns_state(self, tmp_path: Path):
        """run_remediation never raises — always returns a RemediationState."""
        issue = _sca_issue()
        with patch("src.tools.manifest_locator.locate_from_issue") as mock_loc:
            mock_loc.return_value = LocalizedIssue(issue=issue, localization_confidence=0.0)
            result = run_remediation(issue, str(tmp_path))

        assert "status" in result
        assert "errors" in result
        assert "issue" in result


# ---------------------------------------------------------------------------
# TestRoutingFunctions
# ---------------------------------------------------------------------------


class TestRoutingFunctions:
    def test_route_after_locator_failed(self):
        from langgraph.graph import END

        state = {"status": "failed"}
        assert _route_after_locator(state) == END

    def test_route_after_locator_located(self):
        state = {"status": "located"}
        assert _route_after_locator(state) == "planner"

    def test_route_after_planner_version_found(self):
        state = {"status": "planned_version_found"}
        assert _route_after_planner(state) == "edit_request_builder"

    def test_route_after_planner_no_fix(self):
        from langgraph.graph import END

        state = {"status": "planned_no_auto_edit"}
        assert _route_after_planner(state) == END

    def test_route_after_planner_sast(self):
        from langgraph.graph import END

        state = {"status": "localized_needs_remedy_agent"}
        assert _route_after_planner(state) == END

    def test_route_after_edit_request_builder_ready(self):
        state = {"status": "edit_request_ready"}
        assert _route_after_edit_request_builder(state) == "editor"

    def test_route_after_edit_request_builder_manual(self):
        from langgraph.graph import END

        state = {"status": "planned_manual_edit_required"}
        assert _route_after_edit_request_builder(state) == END
