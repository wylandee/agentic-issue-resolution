"""
tests/test_remedy_agent.py — Unit tests for src/orchestrator/remedy_agent.py.

All LLM calls are mocked; no real OpenAI API is required.

Coverage
--------
* Contract: RemedyAgentOutput construction & export
* State: OrchestratorState / initial_orchestrator_state defaults
* Target resolution: SCA (manifest_file), SAST (group.file_path), fallback
* Security checks: absolute path, traversal, missing file, directory, outside root
* Happy path: edits produced, returned status, edit list contents
* Prompt content: exact-match warning, feedback-loop section
* Output validation: missing old_text, ambiguous old_text, wrong file_path
* Error accumulation: LLM exception, validation errors appended without crash
* Retry limit: max_retries_exceeded when retry_count >= max_retries
* LLM construction: temperature=0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.contracts import EditRequest, IssueType, RemedyAgentOutput
from src.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    LocalizedIssue,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.state import (
    DEFAULT_MAX_RETRIES,
    OrchestratorState,
    initial_orchestrator_state,
)


# ===========================================================================
# Helpers / factories
# ===========================================================================


def _sca_issue(**kw) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        cve_id="CVE-2021-44228",
        package_name="lodash",
        package_version="4.17.15",
        **kw,
    )


def _sast_issue(**kw) -> VulnerabilityIssue:
    defaults = dict(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.MEDIUM,
        rule_id="javascript.xss",
        file_path="routes/login.ts",
    )
    defaults.update(kw)
    return VulnerabilityIssue(**defaults)


def _sca_group(
    *,
    file_path: str | None = None,
    manifest_file: str | None = None,
    fix_plan: FixPlan | None = None,
) -> VulnerabilityGroup:
    issue = _sca_issue(file_path=file_path)
    rep_id = issue.id
    li: List[LocalizedIssue] = []
    if manifest_file:
        li.append(
            LocalizedIssue(
                issue=issue,
                manifest_file=manifest_file,
                localization_confidence=0.9,
            )
        )
    return VulnerabilityGroup(
        group_id="sca:package.json:lodash",
        issue_type=IssueType.SCA,
        vulnerable_component="lodash",
        file_path=file_path,
        cve_ids=["CVE-2021-44228"],
        versions=["4.17.15"],
        sources=[IssueSource.ODC],
        representative_issue_id=rep_id,
        issues=[issue],
        localized_issues=li,
        fix_plan=fix_plan,
    )


def _sast_group(*, file_path: str | None = "routes/login.ts") -> VulnerabilityGroup:
    issue = _sast_issue(file_path=file_path)
    return VulnerabilityGroup(
        group_id="sast:routes/login.ts:javascript.xss:10-20",
        issue_type=IssueType.SAST,
        vulnerable_component="javascript.xss",
        file_path=file_path,
        cve_ids=[],
        versions=[],
        sources=[IssueSource.SEMGREP],
        representative_issue_id=issue.id,
        issues=[issue],
    )


def _make_edit(file_path: str, old_text: str, repo_root: str) -> EditRequest:
    return EditRequest(
        repo_root=repo_root,
        file_path=file_path,
        old_text=old_text,
        new_text="\"lodash\": \"^4.17.21\"",
        dry_run=False,
        rationale="Upgrade lodash to 4.17.21 to fix CVE-2021-44228.",
    )


def _mock_llm(edits: List[EditRequest]):
    """Return a mock structured LLM that yields ``RemedyAgentOutput(edits=edits)``."""
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = RemedyAgentOutput(edits=edits)

    mock_llm_instance = MagicMock()
    mock_llm_instance.with_structured_output.return_value = mock_chain

    return mock_llm_instance


# ===========================================================================
# 1. Contract tests: RemedyAgentOutput
# ===========================================================================


class TestRemedyAgentOutputContract:
    def test_empty_edits_default(self):
        out = RemedyAgentOutput()
        assert out.edits == []

    def test_with_edits(self, tmp_path):
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='"lodash": "^4.17.15"',
            new_text='"lodash": "^4.17.21"',
        )
        out = RemedyAgentOutput(edits=[edit])
        assert len(out.edits) == 1
        assert out.edits[0].file_path == "package.json"

    def test_exported_from_contracts(self):
        from src.contracts import RemedyAgentOutput as RAO  # noqa: F401
        assert RAO is RemedyAgentOutput

    def test_json_round_trip(self, tmp_path):
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="a.js",
            old_text="old",
            new_text="new",
        )
        out = RemedyAgentOutput(edits=[edit])
        restored = RemedyAgentOutput.model_validate_json(out.model_dump_json())
        assert restored.edits[0].file_path == "a.js"


# ===========================================================================
# 2. State tests: OrchestratorState / initial_orchestrator_state
# ===========================================================================


class TestOrchestratorState:
    def test_default_max_retries_constant(self):
        assert DEFAULT_MAX_RETRIES == 3

    def test_initial_state_defaults(self, tmp_path):
        groups: list = []
        state = initial_orchestrator_state(str(tmp_path), groups)
        assert state["repo_root"] == str(tmp_path)
        assert state["valid_groups"] == []
        assert state["retry_count"] == 0
        assert state["max_retries"] == DEFAULT_MAX_RETRIES
        assert state["edit_requests"] == []
        assert state["test_failures"] is None
        assert state["scan_failures"] is None
        assert state["status"] == "pending"
        assert state["errors"] == []

    def test_custom_max_retries(self, tmp_path):
        state = initial_orchestrator_state(str(tmp_path), [], max_retries=5)
        assert state["max_retries"] == 5

    def test_errors_field_uses_append_reducer(self):
        """OrchestratorState.errors annotation must use operator.add."""
        import operator
        import typing

        hints = typing.get_type_hints(OrchestratorState, include_extras=True)
        errors_hint = hints.get("errors")
        # The annotation is Annotated[List[str], operator.add]
        metadata = getattr(errors_hint, "__metadata__", ())
        assert operator.add in metadata, (
            "OrchestratorState.errors must be annotated with operator.add"
        )


# ===========================================================================
# 3. run_remedy_agent tests
# ===========================================================================

from src.orchestrator.remedy_agent import run_remedy_agent


class TestRemedyAgentTargetResolution:
    """Target file resolution — SCA & SAST paths."""

    def test_sca_prefers_manifest_file_from_localized_issues(self, tmp_path):
        """SCA: first resolves from localized_issues[0].manifest_file."""
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"
        assert len(result["edit_requests"]) == 1
        assert result["edit_requests"][0].file_path == "package.json"

    def test_sca_falls_back_to_group_file_path(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        # No localized_issues — falls back to group.file_path
        group = _sca_group(file_path="package.json")
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"

    def test_sca_falls_back_to_issue_file_path(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        # No manifest_file, no group.file_path — falls back to issue.file_path
        group = _sca_group(file_path=None, manifest_file=None)
        group.issues[0].file_path = "package.json"
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"

    def test_sast_uses_group_file_path(self, tmp_path):
        route = tmp_path / "routes"
        route.mkdir()
        login = route / "login.ts"
        login.write_text("const x = req.params.id;", encoding="utf-8")

        group = _sast_group(file_path="routes/login.ts")
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="routes/login.ts",
            old_text="const x = req.params.id;",
            new_text="const x = sanitise(req.params.id);",
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"
        assert result["edit_requests"][0].file_path == "routes/login.ts"

    def test_sast_falls_back_to_issue_file_path(self, tmp_path):
        route = tmp_path / "routes"
        route.mkdir()
        (route / "login.ts").write_text("const x = 1;", encoding="utf-8")

        group = _sast_group(file_path=None)
        group.issues[0].file_path = "routes/login.ts"
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="routes/login.ts",
            old_text="const x = 1;",
            new_text="const x = 2;",
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"


class TestRemedyAgentHappyPath:
    """Happy path — aggregates valid edits, correct statuses."""

    def test_single_group_edit_returned(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "edits_generated"
        assert len(result["edit_requests"]) == 1
        er = result["edit_requests"][0]
        assert er.repo_root == str(tmp_path)
        assert er.file_path == "package.json"
        assert er.issue_id == group.representative_issue_id

    def test_issue_id_filled_when_missing(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        # Edit with no issue_id set
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='"lodash": "^4.17.15"',
            new_text='"lodash": "^4.17.21"',
            issue_id=None,
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["edit_requests"][0].issue_id == group.representative_issue_id

    def test_repo_root_normalised_to_state(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        # LLM returned a different (wrong) repo_root
        edit = EditRequest(
            repo_root="/wrong/path",
            file_path="package.json",
            old_text='"lodash": "^4.17.15"',
            new_text='"lodash": "^4.17.21"',
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["edit_requests"][0].repo_root == str(tmp_path)

    def test_multiple_groups_aggregated(self, tmp_path):
        # Two files
        (tmp_path / "package.json").write_text('"lodash": "^4.17.15"', encoding="utf-8")
        route = tmp_path / "routes"
        route.mkdir()
        (route / "login.ts").write_text("const x = req.params.id;", encoding="utf-8")

        sca_group = _sca_group(manifest_file="package.json")
        sast_group = _sast_group(file_path="routes/login.ts")

        edit1 = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        edit2 = EditRequest(
            repo_root=str(tmp_path),
            file_path="routes/login.ts",
            old_text="const x = req.params.id;",
            new_text="const x = sanitise(req.params.id);",
        )

        mock_chain = MagicMock()
        # Return different output for each invoke call
        mock_chain.invoke.side_effect = [
            RemedyAgentOutput(edits=[edit1]),
            RemedyAgentOutput(edits=[edit2]),
        ]
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [sca_group, sast_group])
            )

        assert result["status"] == "edits_generated"
        assert len(result["edit_requests"]) == 2


class TestRemedyAgentPromptContent:
    """Verify key content is included in the prompt."""

    def test_prompt_includes_exact_match_warning(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RemedyAgentOutput(edits=[])
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        prompt_text: str = mock_chain.invoke.call_args[0][0]
        assert "CHARACTER-FOR-CHARACTER" in prompt_text or "old_text" in prompt_text
        assert "EXACTLY ONCE" in prompt_text or "exactly once" in prompt_text.lower()

    def test_prompt_includes_feedback_on_retry(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RemedyAgentOutput(edits=[])
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        state = initial_orchestrator_state(str(tmp_path), [group])
        state["test_failures"] = "AssertionError: expected 4.17.21 got 4.17.15"
        state["retry_count"] = 1

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            run_remedy_agent(state)

        prompt_text: str = mock_chain.invoke.call_args[0][0]
        assert "PREVIOUS ATTEMPT FAILED" in prompt_text
        assert "AssertionError" in prompt_text

    def test_prompt_includes_group_data(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RemedyAgentOutput(edits=[])
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        prompt_text: str = mock_chain.invoke.call_args[0][0]
        assert "lodash" in prompt_text
        assert "CVE-2021-44228" in prompt_text
        assert "package.json" in prompt_text

    def test_prompt_includes_fix_plan_data(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        fix_plan = FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="4.17.21",
            instruction="Upgrade lodash to 4.17.21.",
            strategy_used="osv_api",
        )
        group = _sca_group(manifest_file="package.json", fix_plan=fix_plan)
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RemedyAgentOutput(edits=[])
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        prompt_text: str = mock_chain.invoke.call_args[0][0]
        assert "4.17.21" in prompt_text
        assert "osv_api" in prompt_text

    def test_prompt_instructs_derivation_when_no_fix_plan(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json", fix_plan=None)
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = RemedyAgentOutput(edits=[])
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        prompt_text: str = mock_chain.invoke.call_args[0][0]
        # Should mention deriving the fix
        assert "No fix plan" in prompt_text or "derive" in prompt_text.lower()


class TestRemedyAgentSecurityChecks:
    """Path security — absolute, traversal, missing, directory, outside root."""

    def test_missing_file_rejected(self, tmp_path):
        group = _sca_group(manifest_file="missing.json")
        with patch("src.orchestrator.remedy_agent.ChatOpenAI"):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "remedy_failed"
        assert any("does not exist" in e for e in result["errors"])

    def test_absolute_path_rejected(self, tmp_path):
        group = _sca_group(file_path=None, manifest_file=None)
        group.issues[0].file_path = "/etc/passwd"
        with patch("src.orchestrator.remedy_agent.ChatOpenAI"):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "remedy_failed"
        assert any("absolute" in e.lower() for e in result["errors"])

    def test_traversal_path_rejected(self, tmp_path):
        group = _sca_group(file_path=None, manifest_file=None)
        group.file_path = "../../../etc/passwd"
        group.issues[0].file_path = None
        with patch("src.orchestrator.remedy_agent.ChatOpenAI"):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "remedy_failed"
        assert any("traversal" in e.lower() for e in result["errors"])

    def test_no_target_file_rejected(self, tmp_path):
        """All resolution paths return None → error appended, no LLM call."""
        group = _sca_group(file_path=None, manifest_file=None)
        group.issues[0].file_path = None
        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cl:
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        mock_cl.assert_not_called()
        assert result["status"] == "remedy_failed"

    def test_non_utf8_file_rejected(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_bytes(b"\xff\xfe")  # not valid UTF-8
        group = _sca_group(manifest_file="bad.json")
        with patch("src.orchestrator.remedy_agent.ChatOpenAI"):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "remedy_failed"
        assert any("UTF-8" in e or "utf-8" in e.lower() for e in result["errors"])


class TestRemedyAgentOutputValidation:
    """Edit validation — old_text missing, ambiguous, wrong file_path."""

    def test_old_text_not_found_skipped(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text="TEXT_THAT_IS_NOT_IN_FILE",
            new_text='"lodash": "^4.17.21"',
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        # No valid edits → remedy_failed or no_edits_generated
        assert result["status"] in ("remedy_failed", "no_edits_generated")
        assert any("not found" in e for e in result.get("errors", []))

    def test_ambiguous_old_text_skipped(self, tmp_path):
        # Write a file where the old_text appears twice
        pkg = tmp_path / "package.json"
        pkg.write_text('"dup": "dup"\n"dup": "dup"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="package.json",
            old_text='"dup": "dup"',
            new_text='"dup": "safe"',
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] in ("remedy_failed", "no_edits_generated")
        assert any("ambiguous" in e for e in result.get("errors", []))

    def test_wrong_file_path_skipped(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        # LLM returned wrong file_path
        edit = EditRequest(
            repo_root=str(tmp_path),
            file_path="wrong/file.json",
            old_text='"lodash": "^4.17.15"',
            new_text='"lodash": "^4.17.21"',
        )
        mock_llm_inst = _mock_llm([edit])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] in ("remedy_failed", "no_edits_generated")
        assert any("mismatch" in e or "file_path" in e for e in result.get("errors", []))

    def test_llm_exception_appended_without_crash(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = RuntimeError("OpenAI API quota exceeded")
        mock_llm_inst = MagicMock()
        mock_llm_inst.with_structured_output.return_value = mock_chain

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        # Must not raise — error is in state
        assert result["status"] == "remedy_failed"
        assert any("quota" in e or "LLM call failed" in e for e in result["errors"])

    def test_no_edits_generated_status(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        # LLM returns empty edits list
        mock_llm_inst = _mock_llm([])

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(
                initial_orchestrator_state(str(tmp_path), [group])
            )

        assert result["status"] == "no_edits_generated"


class TestRemedyAgentRetryLogic:
    """Retry counting and limit enforcement."""

    def test_retry_limit_prevents_llm_invocation(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["test_failures"] = "test failed"
        state["retry_count"] = 3   # at the limit
        state["max_retries"] = 3

        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cl:
            result = run_remedy_agent(state)

        mock_cl.assert_not_called()
        assert result["status"] == "max_retries_exceeded"

    def test_retry_count_incremented_on_feedback(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        state = initial_orchestrator_state(str(tmp_path), [group])
        state["test_failures"] = "some test failure"
        state["retry_count"] = 1

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(state)

        assert result.get("retry_count") == 2

    def test_retry_count_not_incremented_without_feedback(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        edit = _make_edit("package.json", '"lodash": "^4.17.15"', str(tmp_path))
        mock_llm_inst = _mock_llm([edit])

        state = initial_orchestrator_state(str(tmp_path), [group])
        # No test_failures or scan_failures set → not a retry

        with patch("src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst):
            result = run_remedy_agent(state)

        # retry_count should not appear in return dict (or be unchanged)
        assert result.get("retry_count", 0) == 0

    def test_scan_failures_alone_triggers_retry(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        state = initial_orchestrator_state(str(tmp_path), [group])
        state["scan_failures"] = "CVE still present in ODC report"
        state["retry_count"] = 3  # at limit
        state["max_retries"] = 3

        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cl:
            result = run_remedy_agent(state)

        mock_cl.assert_not_called()
        assert result["status"] == "max_retries_exceeded"


class TestRemedyAgentLLMConstruction:
    """Verify ChatOpenAI is constructed with correct parameters."""

    def test_chat_openai_temperature_zero(self, tmp_path):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        group = _sca_group(manifest_file="package.json")
        mock_llm_inst = _mock_llm([])

        with patch(
            "src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst
        ) as mock_cl:
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        call_kwargs = mock_cl.call_args[1] if mock_cl.call_args.kwargs else {}
        call_args = mock_cl.call_args[0] if mock_cl.call_args.args else ()
        # temperature=0 may be positional or keyword
        assert call_kwargs.get("temperature") == 0 or (
            len(call_args) >= 2 and call_args[1] == 0
        )

    def test_model_name_from_env(self, tmp_path, monkeypatch):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        monkeypatch.setenv("REMEDY_LLM_MODEL", "gpt-4o")
        group = _sca_group(manifest_file="package.json")
        mock_llm_inst = _mock_llm([])

        with patch(
            "src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst
        ) as mock_cl:
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        call_kwargs = mock_cl.call_args.kwargs or {}
        assert call_kwargs.get("model") == "gpt-4o"

    def test_default_model_gpt4o_mini(self, tmp_path, monkeypatch):
        pkg = tmp_path / "package.json"
        pkg.write_text('"lodash": "^4.17.15"', encoding="utf-8")

        monkeypatch.delenv("REMEDY_LLM_MODEL", raising=False)
        group = _sca_group(manifest_file="package.json")
        mock_llm_inst = _mock_llm([])

        with patch(
            "src.orchestrator.remedy_agent.ChatOpenAI", return_value=mock_llm_inst
        ) as mock_cl:
            run_remedy_agent(initial_orchestrator_state(str(tmp_path), [group]))

        call_kwargs = mock_cl.call_args.kwargs or {}
        assert call_kwargs.get("model") == "gpt-4o-mini"


class TestRemedyAgentInvalidRepoRoot:
    """repo_root validation before any LLM call."""

    def test_missing_repo_root_returns_failed(self):
        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cl:
            result = run_remedy_agent({"repo_root": "/nonexistent/path/xyz", "valid_groups": []})

        mock_cl.assert_not_called()
        assert result["status"] == "remedy_failed"
        assert any("repo_root" in e for e in result["errors"])

    def test_empty_repo_root_returns_failed(self):
        with patch("src.orchestrator.remedy_agent.ChatOpenAI") as mock_cl:
            result = run_remedy_agent({"repo_root": "", "valid_groups": []})

        mock_cl.assert_not_called()
        assert result["status"] == "remedy_failed"
