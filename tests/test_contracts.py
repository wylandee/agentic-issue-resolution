"""
Tests for remediation_engine.contracts.schemas.

Coverage goals
--------------
* All enum coercion and happy-path construction.
* Validator rejection paths (path traversal, bad CVE format, line-range inversion,
  negative token counts, etc.).
* JSON round-trip fidelity for every model.
* Property helpers: LineRange.line_count, PatchAttempt.all_validations_passed,
  TrajectoryEvent.total_tokens.
* Realistic SAST + SCA end-to-end object graphs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from remediation_engine.contracts import (
    AgentActionStatus,
    AgentActionSummary,
    ASTNodeType,
    DecisionCode,
    EditRequest,
    EditResult,
    EditStatus,
    FailureCategory,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LocalizedIssue,
    ODCScanEvidence,
    PatchAttempt,
    QAEvaluation,
    RoutingStrategy,
    ScanScope,
    Severity,
    TrajectoryEvent,
    TrajectoryEventKind,
    ValidationResult,
    ValidationStatus,
    VulnerabilityIssue,
)
from remediation_engine.contracts.schemas import (
    CWEEntry,
    FailingTest,
    LineRange,
    SupervisorDecision,
)

# ===========================================================================
# Helpers
# ===========================================================================


def _sast_issue(**overrides) -> VulnerabilityIssue:
    """Factory for a minimal valid SAST VulnerabilityIssue."""
    kwargs = dict(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=Severity.HIGH,
        rule_id="javascript.express.security.audit.sqli.sequelize-injection-express",
        file_path="routes/userProfile.ts",
        line_range=LineRange(start=61, end=63),
        message="Possible SQL injection via unsanitized user input.",
        repo_url="https://github.com/juice-shop/juice-shop",
        base_ref="v15.0.0",
    )
    kwargs.update(overrides)
    return VulnerabilityIssue(**kwargs)


def _sca_issue(**overrides) -> VulnerabilityIssue:
    """Factory for a minimal valid SCA VulnerabilityIssue."""
    kwargs = dict(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.CRITICAL,
        cve_id="CVE-2021-44228",
        package_name="log4j-core",
        package_version="2.14.1",
        fixed_version="2.17.1",
        ecosystem="maven",
        message="Log4Shell RCE vulnerability.",
    )
    kwargs.update(overrides)
    return VulnerabilityIssue(**kwargs)


# ===========================================================================
# LineRange
# ===========================================================================


class TestLineRange:
    def test_valid_single_line(self):
        lr = LineRange(start=5, end=5)
        assert lr.line_count == 1

    def test_valid_multi_line(self):
        lr = LineRange(start=10, end=20)
        assert lr.line_count == 11

    def test_end_before_start_raises(self):
        with pytest.raises(ValidationError, match="end.*>=.*start"):
            LineRange(start=10, end=5)

    def test_zero_start_raises(self):
        with pytest.raises(ValidationError):
            LineRange(start=0, end=1)

    def test_frozen(self):
        lr = LineRange(start=1, end=3)
        with pytest.raises(Exception):
            lr.start = 2  # type: ignore[misc]

    def test_json_round_trip(self):
        lr = LineRange(start=3, end=7)
        reloaded = LineRange.model_validate_json(lr.model_dump_json())
        assert reloaded == lr


# ===========================================================================
# CWEEntry
# ===========================================================================


class TestCWEEntry:
    def test_valid(self):
        cwe = CWEEntry(id="CWE-79", name="XSS")
        assert cwe.id == "CWE-79"

    def test_invalid_pattern(self):
        with pytest.raises(ValidationError):
            CWEEntry(id="79")  # missing "CWE-" prefix

    def test_name_optional(self):
        cwe = CWEEntry(id="CWE-89")
        assert cwe.name is None


# ===========================================================================
# Severity coercion
# ===========================================================================


class TestSeverityCoercion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("high", Severity.HIGH),
            ("HIGH", Severity.HIGH),
            ("critical", Severity.CRITICAL),
            ("medium", Severity.MEDIUM),
            ("low", Severity.LOW),
            ("info", Severity.INFO),
            ("garbage", Severity.UNKNOWN),
            ("", Severity.UNKNOWN),
            (None, Severity.UNKNOWN),
        ],
    )
    def test_coerce(self, raw, expected):
        issue = VulnerabilityIssue(
            source=IssueSource.SEMGREP,
            issue_type=IssueType.SAST,
            severity=raw,
        )
        assert issue.severity == expected


# ===========================================================================
# Phase 5 refactor enums and handoff models
# ===========================================================================


class TestPhase5RefactorEnums:
    def test_failure_category_values(self):
        assert FailureCategory.SECURITY_FLAG.value == "security_flag"

    def test_supervisor_decision_accepts_optional_decision_code(self):
        decision = SupervisorDecision(
            next_node="teardown",
            instructions="done",
            decision_reason="all tasks terminal",
        )
        assert decision.decision_code is None
        coded = decision.model_copy(update={"decision_code": DecisionCode.NO_ACTIONABLE_TASKS})
        assert coded.decision_code == DecisionCode.NO_ACTIONABLE_TASKS
        assert FailureCategory.PEER_CONFLICT.value == "peer_conflict"
        assert FailureCategory.BREAKING_CHANGE.value == "breaking_change"

    def test_routing_strategy_values(self):
        assert RoutingStrategy.VERSION_BUMP.value == "version_bump"
        assert RoutingStrategy.CODE_WORKAROUND.value == "code_workaround"

    def test_agent_action_status_values(self):
        assert AgentActionStatus.SUCCESS.value == "success"
        assert AgentActionStatus.SURRENDER.value == "surrender"


class TestQAEvaluation:
    def test_passed_evaluation_accepts_no_failure_metadata(self):
        evaluation = QAEvaluation(task_id="group-1", passed=True)
        assert evaluation.failure_category is None
        assert evaluation.retry_feedback is None

    def test_passed_evaluation_rejects_failure_metadata(self):
        with pytest.raises(ValidationError, match="passed=True"):
            QAEvaluation(
                task_id="group-1",
                passed=True,
                failure_category=FailureCategory.SECURITY_FLAG,
                retry_feedback="Try a different edit.",
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"task_id": "group-1", "passed": False, "retry_feedback": "Try again."},
            {
                "task_id": "group-1",
                "passed": False,
                "failure_category": FailureCategory.PEER_CONFLICT,
            },
            {
                "task_id": "group-1",
                "passed": False,
                "failure_category": FailureCategory.PEER_CONFLICT,
                "retry_feedback": "   ",
            },
        ],
    )
    def test_failed_evaluation_requires_category_and_feedback(self, kwargs):
        with pytest.raises(ValidationError, match="passed=False"):
            QAEvaluation(**kwargs)

    def test_failed_evaluation_accepts_category_and_feedback(self):
        evaluation = QAEvaluation(
            task_id="group-1",
            passed=False,
            failure_category=FailureCategory.BREAKING_CHANGE,
            retry_feedback="Update the call sites to match the new API.",
        )
        assert evaluation.failure_category == FailureCategory.BREAKING_CHANGE

    def test_json_round_trip(self):
        evaluation = QAEvaluation(
            task_id="group-1",
            passed=False,
            failure_category=FailureCategory.SECURITY_FLAG,
            retry_feedback="The scanner still reports the vulnerable identifier.",
        )
        reloaded = QAEvaluation.model_validate_json(evaluation.model_dump_json())
        assert reloaded == evaluation

    def test_optional_scan_evidence_round_trips_for_passed_evaluation(self):
        evidence = ODCScanEvidence(
            requested_scope=ScanScope.TARGETED,
            effective_scope=ScanScope.TARGETED,
            covered_task_ids=["task-1"],
            closure_package_names=["express"],
            complete=True,
        )
        evaluation = QAEvaluation(
            task_id="task-1",
            passed=True,
            scan_evidence=evidence,
        )

        reloaded = QAEvaluation.model_validate_json(evaluation.model_dump_json())
        assert reloaded.scan_evidence == evidence


class TestAgentActionSummary:
    def test_rejects_empty_summary(self):
        with pytest.raises(ValidationError, match="summary"):
            AgentActionSummary(
                task_id="group-1",
                status=AgentActionStatus.SUCCESS,
                summary="   ",
            )

    def test_json_round_trip(self):
        summary = AgentActionSummary(
            task_id="group-1",
            status=AgentActionStatus.SURRENDER,
            summary="Unable to resolve the peer dependency conflict safely.",
        )
        reloaded = AgentActionSummary.model_validate_json(summary.model_dump_json())
        assert reloaded == summary


# ===========================================================================
# VulnerabilityIssue
# ===========================================================================


class TestVulnerabilityIssue:
    def test_sast_defaults(self):
        issue = _sast_issue()
        assert issue.id is not None
        assert issue.source == IssueSource.SEMGREP
        assert issue.issue_type == IssueType.SAST
        assert issue.severity == Severity.HIGH
        assert issue.file_path == "routes/userProfile.ts"
        assert issue.line_range.start == 61  # type: ignore[union-attr]

    def test_sca_defaults(self):
        issue = _sca_issue()
        assert issue.cve_id == "CVE-2021-44228"
        assert issue.package_name == "log4j-core"
        assert issue.ecosystem == "maven"

    def test_ghsa_normalised_to_upper(self):
        issue = _sca_issue(ghsa_id="ghsa-vpq2-c234-7xj6")
        assert issue.ghsa_id == "GHSA-VPQ2-C234-7XJ6"

    def test_ghsa_backfilled_from_rule_id(self):
        issue = _sca_issue(cve_id=None, ghsa_id=None, rule_id="GHSA-vpq2-c234-7xj6")
        assert issue.ghsa_id == "GHSA-VPQ2-C234-7XJ6"
        assert issue.rule_id == "GHSA-vpq2-c234-7xj6"

    def test_invalid_ghsa_format(self):
        with pytest.raises(ValidationError):
            _sca_issue(ghsa_id="GHSA-123")

    def test_cve_normalised_to_upper(self):
        issue = _sca_issue(cve_id="cve-2021-44228")
        assert issue.cve_id == "CVE-2021-44228"

    def test_invalid_cve_format(self):
        with pytest.raises(ValidationError):
            _sca_issue(cve_id="CVE-21-12345")  # year must be 4 digits

    def test_file_path_leading_slash_stripped(self):
        issue = _sast_issue(file_path="/routes/userProfile.ts")
        assert issue.file_path == "routes/userProfile.ts"

    def test_empty_file_path_becomes_none(self):
        issue = _sast_issue(file_path="   ")
        assert issue.file_path is None

    def test_auto_uuid(self):
        a = _sast_issue()
        b = _sast_issue()
        assert a.id != b.id

    def test_cwe_list(self):
        issue = _sast_issue(cwe=[CWEEntry(id="CWE-89", name="SQL Injection")])
        assert len(issue.cwe) == 1
        assert issue.cwe[0].id == "CWE-89"

    def test_owasp_list(self):
        issue = _sast_issue(owasp=["A03:2021"])
        assert "A03:2021" in issue.owasp

    def test_json_round_trip(self):
        issue = _sast_issue(
            cwe=[CWEEntry(id="CWE-79")],
            owasp=["A03:2021"],
            raw_payload={"rule": "xss", "extra": {"metadata": {"confidence": "HIGH"}}},
        )
        reloaded = VulnerabilityIssue.model_validate_json(issue.model_dump_json())
        assert reloaded.id == issue.id
        assert reloaded.cwe[0].id == "CWE-79"
        assert reloaded.raw_payload["rule"] == "xss"  # type: ignore[index]

    def test_ingested_at_is_utc(self):
        issue = _sast_issue()
        assert issue.ingested_at.tzinfo is not None

    def test_minimal_issue_no_location(self):
        """An SCA finding need not have file_path / line_range."""
        issue = VulnerabilityIssue(
            source=IssueSource.ODC,
            issue_type=IssueType.SCA,
            severity=Severity.HIGH,
            package_name="lodash",
        )
        assert issue.file_path is None
        assert issue.line_range is None


# ===========================================================================
# LocalizedIssue
# ===========================================================================


class TestLocalizedIssue:
    def test_sast_localization(self):
        issue = _sast_issue()
        loc = LocalizedIssue(
            issue=issue,
            enclosing_symbol="updateUserProfile",
            enclosing_node_type=ASTNodeType.FUNCTION,
            sink_expression="sequelize.query(query)",
            imports=["import { sequelize } from '../models'"],
            data_flow_hints=["taint source: req.body.username"],
            snippet="  const query = `SELECT * FROM Users WHERE username = '${req.body.username}'`\n  sequelize.query(query)",
            localization_confidence=0.92,
        )
        assert loc.enclosing_symbol == "updateUserProfile"
        assert loc.localization_confidence == pytest.approx(0.92)
        assert loc.issue.rule_id == issue.rule_id

    def test_sca_localization(self):
        issue = _sca_issue()
        loc = LocalizedIssue(
            issue=issue,
            manifest_file="package.json",
            is_direct_dependency=True,
            manifest_line=42,
            manifest_snippet='    "log4j-core": "2.14.1"',
            fix_instruction="Bump log4j-core to 2.17.1 in package.json line 42.",
            localization_confidence=0.85,
        )
        assert loc.manifest_file == "package.json"
        assert loc.is_direct_dependency is True
        assert loc.manifest_line == 42

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            LocalizedIssue(
                issue=_sast_issue(),
                localization_confidence=1.5,  # > 1.0 is invalid
            )

    def test_json_round_trip(self):
        loc = LocalizedIssue(
            issue=_sast_issue(),
            enclosing_symbol="foo",
            localization_confidence=0.7,
        )
        reloaded = LocalizedIssue.model_validate_json(loc.model_dump_json())
        assert reloaded.issue.id == loc.issue.id
        assert reloaded.localization_confidence == pytest.approx(0.7)

    def test_default_node_type(self):
        loc = LocalizedIssue(issue=_sast_issue())
        assert loc.enclosing_node_type == ASTNodeType.UNKNOWN


# ===========================================================================
# EditRequest / EditResult
# ===========================================================================


class TestEditRequest:
    def _req(self, **overrides) -> EditRequest:
        kwargs = dict(
            repo_root="/workspace/juice-shop",
            file_path="routes/userProfile.ts",
            old_text="  sequelize.query(unsafeQuery)",
            new_text="  sequelize.query(unsafeQuery, { type: QueryTypes.SELECT })",
            issue_id=uuid4(),
            rationale="Parameterise the query to prevent SQL injection.",
        )
        kwargs.update(overrides)
        return EditRequest(**kwargs)

    def test_valid(self):
        req = self._req()
        assert req.dry_run is False
        assert req.max_deletion_lines == 200

    def test_path_traversal_rejected(self):
        with pytest.raises(ValidationError, match="Path traversal"):
            self._req(file_path="../../etc/passwd")

    def test_empty_old_text_rejected(self):
        with pytest.raises(ValidationError):
            self._req(old_text="")

    def test_dry_run_flag(self):
        req = self._req(dry_run=True)
        assert req.dry_run is True

    def test_frozen(self):
        req = self._req()
        with pytest.raises(Exception):
            req.dry_run = True  # type: ignore[misc]

    def test_json_round_trip(self):
        req = self._req()
        reloaded = EditRequest.model_validate_json(req.model_dump_json())
        assert reloaded.file_path == req.file_path
        assert reloaded.old_text == req.old_text


class TestEditResult:
    def _applied_result(self) -> EditResult:
        req = EditRequest(
            repo_root="/workspace/juice-shop",
            file_path="routes/userProfile.ts",
            old_text="old",
            new_text="new",
        )
        return EditResult(
            request=req,
            status=EditStatus.APPLIED,
            unified_diff="--- a/routes/userProfile.ts\n+++ b/routes/userProfile.ts\n@@ -61,1 +61,1 @@\n-old\n+new",
            lines_added=1,
            lines_removed=1,
            applied_at=datetime.now(UTC),
        )

    def test_applied(self):
        result = self._applied_result()
        assert result.status == EditStatus.APPLIED
        assert result.lines_added == 1
        assert "---" in result.unified_diff  # type: ignore[operator]

    def test_rejected_no_diff(self):
        req = EditRequest(
            repo_root="/workspace/juice-shop",
            file_path="routes/userProfile.ts",
            old_text="no such text",
            new_text="something",
        )
        result = EditResult(
            request=req,
            status=EditStatus.REJECTED,
            rejection_reason="No match found for old_text in routes/userProfile.ts",
        )
        assert result.unified_diff is None
        assert "No match" in result.rejection_reason  # type: ignore[operator]

    def test_json_round_trip(self):
        result = self._applied_result()
        reloaded = EditResult.model_validate_json(result.model_dump_json())
        assert reloaded.status == EditStatus.APPLIED
        assert reloaded.lines_added == result.lines_added


# ===========================================================================
# ValidationResult
# ===========================================================================


class TestValidationResult:
    def test_passed(self):
        vr = ValidationResult(
            phase="unit",
            status=ValidationStatus.PASSED,
            exit_code=0,
            command=["npm", "test"],
            duration_seconds=12.4,
        )
        assert vr.status == ValidationStatus.PASSED
        assert vr.exit_code == 0

    def test_failed_with_structured_tests(self):
        vr = ValidationResult(
            phase="unit",
            status=ValidationStatus.FAILED,
            exit_code=1,
            command=["npm", "test"],
            failing_tests=[
                FailingTest(
                    name="routes/userProfile.spec.ts > should sanitize query",
                    file="routes/userProfile.spec.ts",
                    line=88,
                    message="Expected 200 but got 500",
                ),
            ],
            stderr_tail="Error: Cannot find module 'sanitize-html'",
            dependency_conflict_hints=["sanitize-html is not installed"],
        )
        assert len(vr.failing_tests) == 1
        assert vr.failing_tests[0].name.startswith("routes/")

    def test_timeout_status(self):
        vr = ValidationResult(
            phase="security_scan",
            status=ValidationStatus.TIMEOUT,
            command=["semgrep", "--config=auto", "."],
        )
        assert vr.status == ValidationStatus.TIMEOUT

    def test_json_round_trip(self):
        vr = ValidationResult(
            phase="install",
            status=ValidationStatus.PASSED,
            exit_code=0,
            command=["npm", "ci"],
        )
        reloaded = ValidationResult.model_validate_json(vr.model_dump_json())
        assert reloaded.phase == "install"
        assert reloaded.status == ValidationStatus.PASSED


# ===========================================================================
# PatchAttempt
# ===========================================================================


class TestPatchAttempt:
    def _make_attempt(self, validation_statuses: list[ValidationStatus]) -> PatchAttempt:
        issue_id = uuid4()
        edits: list[EditResult] = []
        validations = [
            ValidationResult(
                phase=f"phase_{i}",
                status=s,
                command=["npm", "test"],
            )
            for i, s in enumerate(validation_statuses)
        ]
        return PatchAttempt(
            issue_id=issue_id,
            attempt_number=1,
            edits=edits,
            validations=validations,
            succeeded=all(s == ValidationStatus.PASSED for s in validation_statuses),
        )

    def test_all_passed(self):
        attempt = self._make_attempt([ValidationStatus.PASSED, ValidationStatus.PASSED])
        assert attempt.all_validations_passed is True
        assert attempt.succeeded is True

    def test_one_failed(self):
        attempt = self._make_attempt([ValidationStatus.PASSED, ValidationStatus.FAILED])
        assert attempt.all_validations_passed is False

    def test_no_validations(self):
        attempt = PatchAttempt(issue_id=uuid4(), attempt_number=1)
        assert attempt.all_validations_passed is False

    def test_attempt_number_gte_1(self):
        with pytest.raises(ValidationError):
            PatchAttempt(issue_id=uuid4(), attempt_number=0)

    def test_json_round_trip(self):
        attempt = self._make_attempt([ValidationStatus.PASSED])
        reloaded = PatchAttempt.model_validate_json(attempt.model_dump_json())
        assert reloaded.id == attempt.id
        assert reloaded.validations[0].status == ValidationStatus.PASSED


# ===========================================================================
# TrajectoryEvent
# ===========================================================================


class TestTrajectoryEvent:
    def test_ingest_event(self):
        event = TrajectoryEvent(
            kind=TrajectoryEventKind.INGEST,
            agent="semgrep_ingestor",
            summary="Fetched 342 SAST findings from Semgrep API.",
            input_tokens=0,
            output_tokens=0,
        )
        assert event.kind == TrajectoryEventKind.INGEST
        assert event.total_tokens == 0

    def test_apply_edit_event_with_tokens(self):
        issue_id = uuid4()
        patch_id = uuid4()
        event = TrajectoryEvent(
            issue_id=issue_id,
            patch_attempt_id=patch_id,
            kind=TrajectoryEventKind.APPLY_EDIT,
            agent="remedy_agent",
            summary="Applied search/replace edit to routes/userProfile.ts.",
            detail={"lines_added": 2, "lines_removed": 1},
            input_tokens=512,
            output_tokens=128,
            duration_seconds=0.8,
        )
        assert event.total_tokens == 640
        assert event.detail["lines_added"] == 2  # type: ignore[index]

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            TrajectoryEvent(
                kind=TrajectoryEventKind.VALIDATE,
                summary="Ran tests.",
                input_tokens=-1,
                output_tokens=0,
            )

    def test_frozen(self):
        event = TrajectoryEvent(
            kind=TrajectoryEventKind.INGEST,
            summary="test",
        )
        with pytest.raises(Exception):
            event.summary = "changed"  # type: ignore[misc]

    def test_json_round_trip(self):
        event = TrajectoryEvent(
            kind=TrajectoryEventKind.DELIVER,
            agent="gitops_agent",
            summary="Opened PR #42.",
            detail={"pr_url": "https://github.com/juice-shop/juice-shop/pull/42"},
            input_tokens=200,
            output_tokens=50,
        )
        reloaded = TrajectoryEvent.model_validate_json(event.model_dump_json())
        assert reloaded.id == event.id
        assert reloaded.detail["pr_url"].endswith("/42")  # type: ignore[index]

    def test_occurred_at_is_utc(self):
        event = TrajectoryEvent(
            kind=TrajectoryEventKind.ABORT,
            summary="Max retries exceeded.",
        )
        assert event.occurred_at.tzinfo is not None


# ===========================================================================
# End-to-end object graph: SAST fix trajectory
# ===========================================================================


class TestEndToEndSASTTrajectory:
    """Simulate the full ingestâ†’localizeâ†’applyâ†’validateâ†’deliver pipeline."""

    def test_full_trajectory(self):
        # 1. Ingest
        issue = _sast_issue(
            finding_id="sem-abc123",
            cwe=[CWEEntry(id="CWE-89", name="SQL Injection")],
            owasp=["A03:2021"],
            raw_payload={"rule": "sequelize-injection", "confidence": "HIGH"},
            base_ref="abc123def456",
        )
        ingest_event = TrajectoryEvent(
            issue_id=issue.id,
            kind=TrajectoryEventKind.INGEST,
            agent="semgrep_ingestor",
            summary=f"Ingested finding {issue.finding_id}.",
            input_tokens=50,
            output_tokens=10,
        )

        # 2. Localize
        loc = LocalizedIssue(
            issue=issue,
            enclosing_symbol="updateUserProfile",
            enclosing_node_type=ASTNodeType.FUNCTION,
            sink_expression="sequelize.query(query)",
            imports=["import { sequelize } from '../models'"],
            snippet="const query = `SELECT * FROM Users WHERE id='${req.body.id}'`\nsequelize.query(query)",
            localization_confidence=0.95,
            fix_instruction="Replace raw string interpolation with parameterised query.",
        )
        localize_event = TrajectoryEvent(
            issue_id=issue.id,
            kind=TrajectoryEventKind.LOCALIZE,
            agent="triage_agent",
            summary="Localized to updateUserProfile in routes/userProfile.ts:61.",
            input_tokens=300,
            output_tokens=120,
            detail={"confidence": loc.localization_confidence},
        )

        # 3. Apply edit
        edit_req = EditRequest(
            repo_root="/workspace/juice-shop",
            file_path="routes/userProfile.ts",
            old_text="const query = `SELECT * FROM Users WHERE id='${req.body.id}'`\nsequelize.query(query)",
            new_text="sequelize.query('SELECT * FROM Users WHERE id=?', { replacements: [req.body.id] })",
            issue_id=issue.id,
            rationale="Use parameterised query to prevent SQL injection.",
        )
        edit_result = EditResult(
            request=edit_req,
            status=EditStatus.APPLIED,
            unified_diff="--- a/routes/userProfile.ts\n+++ b/routes/userProfile.ts",
            lines_added=1,
            lines_removed=2,
            applied_at=datetime.now(UTC),
        )
        apply_event = TrajectoryEvent(
            issue_id=issue.id,
            kind=TrajectoryEventKind.APPLY_EDIT,
            agent="remedy_agent",
            summary="Parameterised Sequelize query in routes/userProfile.ts.",
            input_tokens=800,
            output_tokens=200,
        )

        # 4. Validate
        val = ValidationResult(
            phase="unit",
            status=ValidationStatus.PASSED,
            exit_code=0,
            command=["npm", "test"],
            duration_seconds=28.1,
        )
        patch = PatchAttempt(
            issue_id=issue.id,
            attempt_number=1,
            edits=[edit_result],
            validations=[val],
            succeeded=True,
        )
        validate_event = TrajectoryEvent(
            issue_id=issue.id,
            patch_attempt_id=patch.id,
            kind=TrajectoryEventKind.VALIDATE,
            agent="remedy_agent",
            summary="All validation phases passed on attempt 1.",
            input_tokens=100,
            output_tokens=30,
        )

        # 5. Deliver
        deliver_event = TrajectoryEvent(
            issue_id=issue.id,
            kind=TrajectoryEventKind.DELIVER,
            agent="gitops_agent",
            summary="Opened PR fix/sast-sequelize-injection-abc123.",
            detail={"branch": "fix/sast-sequelize-injection-abc123", "pr_number": 7},
            input_tokens=200,
            output_tokens=60,
        )

        trajectory = [ingest_event, localize_event, apply_event, validate_event, deliver_event]

        # Assertions
        assert patch.all_validations_passed
        assert patch.succeeded is True
        assert sum(e.total_tokens for e in trajectory) > 0

        total_input = sum(e.input_tokens for e in trajectory)
        total_output = sum(e.output_tokens for e in trajectory)
        assert total_input == 50 + 300 + 800 + 100 + 200
        assert total_output == 10 + 120 + 200 + 30 + 60

        # JSON serialise and restore every object
        issue2 = VulnerabilityIssue.model_validate_json(issue.model_dump_json())
        assert issue2.cwe[0].id == "CWE-89"

        loc2 = LocalizedIssue.model_validate_json(loc.model_dump_json())
        assert loc2.localization_confidence == pytest.approx(0.95)

        patch2 = PatchAttempt.model_validate_json(patch.model_dump_json())
        assert patch2.all_validations_passed

        for event in trajectory:
            e2 = TrajectoryEvent.model_validate_json(event.model_dump_json())
            assert e2.id == event.id


# ===========================================================================
# JSONL serialisation helper
# ===========================================================================


class TestJSONLSerialisation:
    """Verify objects can be written/read as JSONL lines."""

    def test_vulnerability_issue_jsonl(self):
        issues = [_sast_issue(), _sca_issue()]
        lines = [i.model_dump_json() for i in issues]
        assert all(json.loads(line) for line in lines)  # valid JSON per line

        restored = [VulnerabilityIssue.model_validate_json(line) for line in lines]
        assert restored[0].issue_type == IssueType.SAST
        assert restored[1].issue_type == IssueType.SCA

    def test_trajectory_event_jsonl(self):
        events = [
            TrajectoryEvent(kind=TrajectoryEventKind.INGEST, summary="step 1"),
            TrajectoryEvent(kind=TrajectoryEventKind.LOCALIZE, summary="step 2"),
            TrajectoryEvent(kind=TrajectoryEventKind.DELIVER, summary="step 3"),
        ]
        lines = [e.model_dump_json() for e in events]
        restored = [TrajectoryEvent.model_validate_json(line) for line in lines]
        assert [e.kind for e in restored] == [
            TrajectoryEventKind.INGEST,
            TrajectoryEventKind.LOCALIZE,
            TrajectoryEventKind.DELIVER,
        ]


# ===========================================================================
# FixPlan Invariants
# ===========================================================================


class TestFixPlan:
    def test_version_found_success(self):
        fp = FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version="1.2.3",
            instruction="Update package to 1.2.3",
            strategy_used="osv_api",
        )
        assert fp.status == FixPlanStatus.VERSION_FOUND
        assert fp.fixed_version == "1.2.3"
        assert fp.workaround_snippets is None

    def test_version_found_missing_version_raises(self):
        with pytest.raises(
            ValidationError, match="status='version_found' requires a non-empty fixed_version"
        ):
            FixPlan(
                status=FixPlanStatus.VERSION_FOUND,
                fixed_version=None,
                instruction="Update package",
                strategy_used="osv_api",
            )

    def test_version_found_with_snippets_raises(self):
        with pytest.raises(
            ValidationError, match="status='version_found' must have workaround_snippets=None"
        ):
            FixPlan(
                status=FixPlanStatus.VERSION_FOUND,
                fixed_version="1.2.3",
                workaround_snippets=["snippet"],
                instruction="Update package",
                strategy_used="osv_api",
            )

    def test_workaround_found_success(self):
        fp = FixPlan(
            status=FixPlanStatus.WORKAROUND_FOUND,
            workaround_snippets=["Use safe methods"],
            instruction="Apply workaround",
            strategy_used="serper",
        )
        assert fp.status == FixPlanStatus.WORKAROUND_FOUND
        assert fp.workaround_snippets == ["Use safe methods"]
        assert fp.fixed_version is None

    def test_workaround_found_missing_snippets_raises(self):
        with pytest.raises(
            ValidationError,
            match="status='workaround_found' requires a non-empty workaround_snippets list",
        ):
            FixPlan(
                status=FixPlanStatus.WORKAROUND_FOUND,
                workaround_snippets=None,
                instruction="Apply workaround",
                strategy_used="serper",
            )

    def test_workaround_found_with_version_raises(self):
        with pytest.raises(
            ValidationError, match="status='workaround_found' must have fixed_version=None"
        ):
            FixPlan(
                status=FixPlanStatus.WORKAROUND_FOUND,
                fixed_version="1.2.3",
                workaround_snippets=["snippet"],
                instruction="Apply workaround",
                strategy_used="serper",
            )

    def test_no_fix_success(self):
        fp = FixPlan(
            status=FixPlanStatus.NO_FIX,
            instruction="No fix available",
            strategy_used="none",
        )
        assert fp.status == FixPlanStatus.NO_FIX
        assert fp.fixed_version is None
        assert fp.workaround_snippets is None

    def test_no_fix_with_version_raises(self):
        with pytest.raises(ValidationError, match="status='no_fix' must have fixed_version=None"):
            FixPlan(
                status=FixPlanStatus.NO_FIX,
                fixed_version="1.2.3",
                instruction="No fix available",
                strategy_used="none",
            )

    def test_no_fix_with_snippets_raises(self):
        with pytest.raises(
            ValidationError, match="status='no_fix' must have workaround_snippets=None"
        ):
            FixPlan(
                status=FixPlanStatus.NO_FIX,
                workaround_snippets=["snippet"],
                instruction="No fix available",
                strategy_used="none",
            )
