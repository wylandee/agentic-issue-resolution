"""Focused tests for triage."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import requests

from remediation_engine.agents.triage import (
    TriagePipelineError,
    TriageSelectionError,
    _build_triage_prompt,
    _llm_triage,
    _localize_sca_issues,
    analyze_reachability,
    group_issues,
    run_triage,
    run_triage_pipeline,
    select_issues_for_remediation,
)
from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    LineRange,
    LocalizedIssue,
    Severity,
    SystemContext,
    TriageResult,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.settings import AppSettings
from remediation_engine.tools.advisories import _fetch_epss_chunk, enrich_cves
from tests.helpers import group_with_fix_plans, selected_group_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _issue(
    *,
    severity: Severity = Severity.MEDIUM,
    file_path: str | None = "src/app.js",
    issue_type: IssueType = IssueType.SCA,
    package_name: str = "lodash",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        severity=severity,
        file_path=file_path,
        package_name=package_name,
    )


def _group(
    issue: VulnerabilityIssue,
    *,
    enrichment: CVEEnrichment | None = None,
) -> VulnerabilityGroup:
    return VulnerabilityGroup(
        group_id="sca:src/app.js:lodash",
        issue_type=issue.issue_type,
        vulnerable_component="lodash",
        file_path=issue.file_path,
        cve_ids=["CVE-2021-23337"] if issue.issue_type == IssueType.SCA else [],
        representative_issue_id=issue.id,
        issues=[issue],
        enrichment=enrichment,
    )


def _kev_enrichment(cve: str = "CVE-2021-23337") -> CVEEnrichment:
    return CVEEnrichment(
        cve_id=cve,
        epss=0.9,
        epss_percentile=0.99,
        in_kev=True,
        kev_date_added="2022-01-01",
        enrichment_source="epss+kev",
    )


def _epss_enrichment(epss: float = 0.7, cve: str = "CVE-2021-23337") -> CVEEnrichment:
    return CVEEnrichment(
        cve_id=cve,
        epss=epss,
        epss_percentile=0.85,
        in_kev=False,
        enrichment_source="epss",
    )


def _context(
    environment: str = "production",
    *,
    public_facing: bool | None = None,
    data_sensitivity: str | None = None,
) -> SystemContext:
    return SystemContext(
        environment=environment,
        public_facing=public_facing,
        data_sensitivity=data_sensitivity,
    )


# ---------------------------------------------------------------------------
# Deterministic path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_llm_triage_by_default(monkeypatch):
    monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")


class TestDeterministicTriage:
    def test_basic_valid_result(self):
        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue)
        result = run_triage(group, _context())

        assert isinstance(result, TriageResult)
        assert result.is_valid is True
        assert result.triage_method == "deterministic"
        assert result.group_id == group.group_id
        assert result.recommended_issue_id == issue.id
        assert result.original_severity == Severity.MEDIUM
        assert result.is_unreachable_code is False
        assert result.validity_confidence_score == 1.0
        assert result.priority_confidence_score == 1.0

    def test_no_evidence_means_valid(self):
        """Unknown evidence must never automatically become a false positive."""
        issue = _issue(severity=Severity.UNKNOWN)
        group = _group(issue, enrichment=None)
        result = run_triage(group, _context())
        assert result.is_valid is True
        assert result.false_positive_reason is None

    def test_unknown_file_location_remains_actionable(self):
        """Missing location is uncertainty, not positive development-only evidence."""
        issue = _issue(file_path=None)
        group = _group(issue)

        result = run_triage(group, _context(environment="test"))

        assert result.is_valid is True

    def test_kev_clamps_to_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_kev_enrichment())
        result = run_triage(group, _context())
        assert result.revised_priority == Severity.CRITICAL
        assert result.is_valid is True  # KEV overrides any FP decision

    def test_kev_in_staging_falls_through_to_epss_rule(self):
        """In staging, KEV alone does not force CRITICAL; EPSS 0.9 â‰¥ 0.36 clamps to HIGH."""
        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue, enrichment=_kev_enrichment())  # epss=0.9, in_kev=True
        result = run_triage(group, _context(environment="staging"))
        # Not production â†’ KEV rule skipped; EPSS 0.9 â‰¥ 0.36 rule fires â†’ at least HIGH
        assert result.revised_priority in (Severity.HIGH, Severity.CRITICAL)

    def test_high_epss_clamps_to_at_least_high(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context())
        assert result.revised_priority in (Severity.HIGH, Severity.CRITICAL)

    def test_high_epss_public_facing_forces_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context(public_facing=True))
        assert result.revised_priority == Severity.CRITICAL

    def test_high_epss_high_sensitivity_forces_critical(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.6))
        result = run_triage(group, _context(data_sensitivity="HIGH"))
        assert result.revised_priority == Severity.CRITICAL

    def test_low_epss_does_not_clamp(self):
        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.1))
        result = run_triage(group, _context())
        # No guardrail should raise LOW above LOW in this case
        assert result.revised_priority in (Severity.LOW, Severity.MEDIUM)

    def test_internal_low_epss_high_severity_downgrades_to_medium(self):
        issue = _issue(severity=Severity.HIGH)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.005))
        result = run_triage(group, _context(public_facing=False))
        assert result.revised_priority == Severity.MEDIUM

    def test_original_critical_severity_preserved(self):
        issue = _issue(severity=Severity.CRITICAL)
        group = _group(issue)
        result = run_triage(group, _context())
        assert result.original_severity == Severity.CRITICAL
        assert result.revised_priority == Severity.CRITICAL

    def test_priority_reasoning_is_non_empty(self):
        group = _group(_issue())
        result = run_triage(group, _context())
        assert len(result.priority_reasoning) > 0

    def test_unreachable_code_is_flagged_but_not_marked_false_positive(self):
        issue = _issue(severity=Severity.HIGH)
        group = _group(issue)
        group.is_reachable = False

        result = run_triage(group, _context())

        assert result.is_valid is True
        assert result.false_positive_reason is None
        assert result.is_unreachable_code is True
        assert (
            "Reachability analysis shows the package is not imported" in result.priority_reasoning
        )


# ---------------------------------------------------------------------------
# Dev-scope false positive
# ---------------------------------------------------------------------------


class TestDevScopeFalsePositive:
    def test_dev_env_test_paths_marks_false_positive(self):
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue)
        result = run_triage(group, _context(environment="test"))
        assert result.is_valid is False
        assert result.false_positive_reason is not None

    def test_dev_env_but_prod_path_stays_valid(self):
        """Dev environment alone is not enough â€” the file paths must also be dev/test."""
        issue = _issue(file_path="src/utils.js")  # Not a dev path
        group = _group(issue)
        result = run_triage(group, _context(environment="dev"))
        assert result.is_valid is True

    def test_prod_env_test_path_stays_valid(self):
        """Test path in prod environment is not a false positive."""
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue)
        result = run_triage(group, _context(environment="production"))
        assert result.is_valid is True

    def test_kev_does_not_override_dev_false_positive_outside_production(self):
        issue = _issue(file_path="tests/unit/lodash_test.js")
        group = _group(issue, enrichment=_kev_enrichment())
        result = run_triage(group, _context(environment="test"))
        assert result.is_valid is False
        assert result.false_positive_reason is not None


# ---------------------------------------------------------------------------
# LLM path guardrails
# ---------------------------------------------------------------------------


class TestLLMGuardrails:
    def _make_llm_result(
        self,
        group: VulnerabilityGroup,
        priority: Severity,
    ) -> TriageResult:
        return TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=True,
            original_severity=group.issues[0].severity if group.issues else Severity.UNKNOWN,
            revised_priority=priority,
            is_unreachable_code=False,
            priority_reasoning="LLM says LOW.",
            validity_confidence_score=0.6,
            priority_confidence_score=0.5,
            recommended_issue_id=group.representative_issue_id,
            triage_method="llm",
        )

    def test_llm_under_ranks_kev_guardrail_clamps_to_critical(self, monkeypatch):
        """If LLM returns LOW for a KEV issue, guardrail must clamp to CRITICAL."""
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.LOW)
        group = _group(issue, enrichment=_kev_enrichment())

        llm_result = self._make_llm_result(group, Severity.LOW)

        with patch("remediation_engine.agents.triage._llm_triage", return_value=llm_result):
            result = run_triage(group, _context())

        assert result.revised_priority == Severity.CRITICAL
        assert "[Guardrail]" in result.priority_reasoning
        assert result.priority_confidence_score == 1.0

    def test_llm_no_priority_override_preserves_priority_confidence(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.MEDIUM)
        group = _group(issue, enrichment=_epss_enrichment(epss=0.1))
        llm_result = self._make_llm_result(group, Severity.MEDIUM)

        with patch("remediation_engine.agents.triage._llm_triage", return_value=llm_result):
            result = run_triage(group, _context())

        assert result.revised_priority == Severity.MEDIUM
        assert result.original_severity == Severity.MEDIUM
        assert result.priority_confidence_score == 0.5

    def test_llm_failure_falls_back_to_deterministic(self, monkeypatch):
        """When _llm_triage returns None, deterministic path must be used."""
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "true")

        issue = _issue(severity=Severity.HIGH)
        group = _group(issue)

        with patch("remediation_engine.agents.triage._llm_triage", return_value=None):
            result = run_triage(group, _context())

        assert result.triage_method == "deterministic"

    def test_triage_method_deterministic_when_llm_disabled(self, monkeypatch):
        monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")
        group = _group(_issue())
        result = run_triage(group, _context())
        assert result.triage_method == "deterministic"

    def test_llm_constructor_receives_injected_key_without_environment(self, monkeypatch):
        """The triage client uses bound credentials even when env lookup is empty."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        group = _group(_issue())
        settings = AppSettings(openai_api_key="injected-test-key")
        llm_result = self._make_llm_result(group, Severity.MEDIUM)
        constructor_kwargs = {}

        class FakeStructuredLLM:
            def invoke(self, _prompt):
                return llm_result

        class FakeLLM:
            def with_structured_output(self, _model):
                return FakeStructuredLLM()

        def fake_chat_openai(**kwargs):
            constructor_kwargs.update(kwargs)
            return FakeLLM()

        monkeypatch.setattr("langchain_openai.ChatOpenAI", fake_chat_openai)

        result = _llm_triage(group, _context(), settings=settings)

        assert result is not None
        assert constructor_kwargs["api_key"] == "injected-test-key"


class TestTriagePrompt:
    def test_prompt_includes_reachability_analysis_true(self):
        issue = _issue()
        group = _group(issue)
        group.is_reachable = True

        prompt = _build_triage_prompt(group, _context())

        assert "=== DATA ===" in prompt
        assert (
            "Reachability Analysis: TRUE (Package is explicitly imported in application code)"
            in prompt
        )
        assert "CONFIDENCE SCORING RUBRIC:" in prompt

    def test_prompt_rule_d_uses_explicit_false_reachability(self):
        issue = _issue()
        group = _group(issue)
        group.is_reachable = False

        prompt = _build_triage_prompt(group, _context())

        assert (
            "Reachability Analysis: FALSE (Package is a direct dependency but is NEVER imported in the application source code)"
            in prompt
        )
        assert (
            "Rule D: If Reachability Analysis is explicitly FALSE, the package is unreachable in this application. Set is_unreachable_code=True, but do not treat that fact alone as a false positive."
            in prompt
        )
        assert (
            "Always return original_severity, revised_priority, is_unreachable_code, validity_confidence_score, and priority_confidence_score."
            in prompt
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx() -> SystemContext:
    return SystemContext(environment="production")


def _sca(
    *,
    package_name: str = "lodash",
    file_path: str = "package.json",
    cve_id: str | None = "CVE-2021-23337",
    severity: Severity = Severity.HIGH,
    source: IssueSource = IssueSource.SEMGREP,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SCA,
        severity=severity,
        package_name=package_name,
        cve_id=cve_id,
        file_path=file_path,
    )


def _sast(
    *,
    file_path: str = "src/app.js",
    rule_id: str = "javascript.xss",
    line_start: int = 10,
    line_end: int = 15,
    severity: Severity = Severity.MEDIUM,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=IssueType.SAST,
        severity=severity,
        file_path=file_path,
        rule_id=rule_id,
        line_range=LineRange(start=line_start, end=line_end),
    )


def _empty_enrichment_map(cve_ids: list[str]):
    """Return safe-default CVEEnrichment for all requested CVEs."""
    return {
        cve: CVEEnrichment(cve_id=cve, epss=0.0, in_kev=False, enrichment_source="none")
        for cve in cve_ids
    }


def _localized_sca_issues(issues: list[VulnerabilityIssue]):
    return [
        LocalizedIssue(
            issue=issue,
            manifest_file=issue.file_path,
            is_direct_dependency=True,
            package_manager="npm",
            localization_confidence=0.95,
        )
        for issue in issues
        if issue.issue_type == IssueType.SCA
    ]


# ---------------------------------------------------------------------------
# run_triage_pipeline
# ---------------------------------------------------------------------------


class TestRunTriagePipeline:
    def test_empty_input_returns_empty(self):
        with (
            patch("remediation_engine.agents.triage.enrich_cves", return_value={}),
            patch("remediation_engine.agents.triage._localize_sca_issues", return_value=[]),
        ):
            result = run_triage_pipeline([], _ctx())
        assert result == []

    def test_sca_issue_produces_one_pair(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]
        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
        ):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 1
        group, triage = result[0]
        assert isinstance(group, VulnerabilityGroup)
        assert isinstance(triage, TriageResult)

    def test_sast_issue_produces_one_pair(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sast()]
        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch("remediation_engine.agents.triage._localize_sca_issues", return_value=[]),
        ):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 1

    def test_mixed_input_correct_group_count(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [
            _sca(package_name="lodash", file_path="package.json"),
            _sca(package_name="express", file_path="backend/package.json"),
            _sast(file_path="src/login.js", rule_id="sqli", line_start=5, line_end=5),
        ]
        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
        ):
            result = run_triage_pipeline(issues, _ctx())

        assert len(result) == 3

    def test_all_results_are_tuples_of_correct_types(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca(), _sast()]
        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
        ):
            result = run_triage_pipeline(issues, _ctx())

        for pair in result:
            assert len(pair) == 2
            group, triage = pair
            assert isinstance(group, VulnerabilityGroup)
            assert isinstance(triage, TriageResult)

    def test_cve_enrichment_attached_to_group(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        kev_enrichment = CVEEnrichment(
            cve_id="CVE-2021-23337",
            epss=0.8,
            in_kev=True,
            enrichment_source="epss+kev",
        )
        issues = [_sca(cve_id="CVE-2021-23337")]
        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves",
                return_value={"CVE-2021-23337": kev_enrichment},
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
        ):
            result = run_triage_pipeline(issues, _ctx())

        group, triage = result[0]
        assert group.enrichment is not None
        assert group.enrichment.in_kev is True
        assert triage.revised_priority == Severity.CRITICAL

    def test_reachability_runs_when_repo_root_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]

        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
            patch("remediation_engine.agents.triage.analyze_reachability") as mock_reachability,
        ):
            run_triage_pipeline(issues, _ctx(), repo_root=str(tmp_path))

        mock_reachability.assert_called_once()

    def test_reachability_skipped_when_repo_root_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]
        missing_root = tmp_path / "missing-repo"

        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=_localized_sca_issues(issues),
            ),
            patch("remediation_engine.agents.triage.analyze_reachability") as mock_reachability,
        ):
            run_triage_pipeline(issues, _ctx(), repo_root=str(missing_root))

        mock_reachability.assert_not_called()

    def test_pipeline_localizes_sca_without_fix_planning(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [_sca()]
        planned_pairs = _localized_sca_issues(issues)

        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves", side_effect=_empty_enrichment_map
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=planned_pairs,
            ) as mock_prepare,
            patch("remediation_engine.tools.advisories.plan_fix") as mock_plan,
        ):
            result = run_triage_pipeline(issues, _ctx(), repo_root=str(tmp_path))

        mock_prepare.assert_called_once()
        mock_plan.assert_not_called()
        assert result[0][0].fix_plan_candidates == []
        assert result[0][0].localized_issues

    def test_group_triage_failure_is_explicit_instead_of_dropping_group(
        self, tmp_path, monkeypatch
    ):
        """A failed group is reported with its ID rather than omitted from results."""
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        issues = [
            _sca(package_name="lodash", file_path="package.json"),
            _sca(package_name="express", file_path="backend/package.json"),
        ]
        planned_pairs = _localized_sca_issues(issues)
        seen_group_ids: list[str] = []

        def triage_one(group, _context):
            seen_group_ids.append(group.group_id)
            if len(seen_group_ids) == 2:
                raise RuntimeError("synthetic triage failure")
            return TriageResult(
                group_id=group.group_id,
                is_valid=True,
                revised_priority=Severity.HIGH,
                priority_reasoning="test",
                validity_confidence_score=1.0,
                priority_confidence_score=1.0,
                recommended_issue_id=group.issues[0].id,
                triage_method="deterministic",
            )

        with (
            patch(
                "remediation_engine.agents.triage.enrich_cves",
                side_effect=_empty_enrichment_map,
            ),
            patch(
                "remediation_engine.agents.triage._localize_sca_issues",
                return_value=planned_pairs,
            ),
            patch("remediation_engine.agents.triage.run_triage", side_effect=triage_one),
            pytest.raises(TriagePipelineError) as raised,
        ):
            run_triage_pipeline(issues, _ctx())

        assert raised.value.failed_group_ids == (seen_group_ids[1],)
        assert "synthetic triage failure" in str(raised.value)

    def test_valid_result_without_an_issue_is_explicit(self):
        group = VulnerabilityGroup(
            group_id="sast:missing",
            issue_type=IssueType.SAST,
            representative_issue_id=uuid4(),
            issues=[],
        )
        triage = TriageResult(
            group_id=group.group_id,
            is_valid=True,
            revised_priority=Severity.HIGH,
            priority_reasoning="test",
            validity_confidence_score=1.0,
            priority_confidence_score=1.0,
            recommended_issue_id="00000000-0000-0000-0000-000000000001",
            triage_method="deterministic",
        )

        with pytest.raises(TriageSelectionError, match="sast:missing"):
            select_issues_for_remediation([(group, triage)])

    def test_localize_sca_issues_calls_locator_without_planner(self, tmp_path):
        issue = _sca()
        localized = LocalizedIssue(
            issue=issue,
            manifest_file="package.json",
            is_direct_dependency=True,
            package_manager="npm",
            localization_confidence=0.95,
        )
        with (
            patch(
                "remediation_engine.tools.manifest_locator.locate_from_issue",
                return_value=localized,
            ) as mock_locate,
            patch("remediation_engine.tools.advisories.plan_fix") as mock_plan,
        ):
            localized_issues = _localize_sca_issues([issue], str(tmp_path))

        mock_locate.assert_called_once()
        mock_plan.assert_not_called()
        assert localized_issues == [localized]

    @pytest.mark.parametrize(
        ("updates", "package_name"),
        [
            ({"ecosystem": "python"}, "requests"),
            ({"ecosystem": "pypi"}, "requests"),
            ({"purl": "pkg:pypi/zope.interface"}, "zope-interface"),
        ],
    )
    def test_pypi_findings_use_only_python_localizer_and_canonical_identity(
        self, tmp_path, updates, package_name
    ):
        issue = _sca(package_name=updates.get("package_name", "Requests")).model_copy(
            update=updates
        )
        if "purl" in updates:
            issue = issue.model_copy(update={"package_name": None})

        def locate_python(canonical_issue, _repo_path):
            return LocalizedIssue(
                issue=canonical_issue,
                manifest_file="requirements.txt",
                is_direct_dependency=True,
                declaration_type="requirements",
                package_manager="pip",
                localization_confidence=0.95,
            )

        with (
            patch(
                "remediation_engine.tools.python_manifest_locator.locate_from_issue",
                side_effect=locate_python,
            ) as python_locate,
            patch("remediation_engine.tools.manifest_locator.locate_from_issue") as npm_locate,
            patch("remediation_engine.tools.advisories.plan_fix") as mock_plan,
        ):
            localized_issues = _localize_sca_issues([issue], str(tmp_path))

        python_locate.assert_called_once()
        npm_locate.assert_not_called()
        mock_plan.assert_not_called()
        localized = python_locate.call_args.args[0]
        assert localized.ecosystem == "pypi"
        assert localized.package_name == package_name
        assert localized_issues[0].issue.ecosystem == "pypi"
        assert localized_issues[0].issue.package_name == package_name
        assert localized_issues[0].manifest_file == "requirements.txt"


# ---------------------------------------------------------------------------
# select_issues_for_remediation
# ---------------------------------------------------------------------------


class TestSelectIssuesForRemediation:
    def _make_pair(
        self,
        issue: VulnerabilityIssue,
        *,
        is_valid: bool = True,
        priority: Severity = Severity.HIGH,
        fp_reason: str | None = None,
    ):
        group = VulnerabilityGroup(
            group_id=f"sca:{issue.file_path}:{issue.package_name}",
            issue_type=issue.issue_type,
            representative_issue_id=issue.id,
            issues=[issue],
        )
        triage = TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=is_valid,
            false_positive_reason=fp_reason if not is_valid else None,
            original_severity=issue.severity,
            revised_priority=priority,
            is_unreachable_code=False,
            priority_reasoning="test",
            validity_confidence_score=1.0,
            priority_confidence_score=1.0,
            recommended_issue_id=issue.id,
            triage_method="deterministic",
        )
        return group, triage

    def test_invalid_groups_excluded(self):
        valid_issue = _sca(package_name="lodash", file_path="package.json")
        invalid_issue = _sca(package_name="mocha", file_path="package.json")

        pairs = [
            self._make_pair(valid_issue, is_valid=True, priority=Severity.HIGH),
            self._make_pair(
                invalid_issue,
                is_valid=False,
                fp_reason="Test-only dependency.",
                priority=Severity.LOW,
            ),
        ]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 1
        assert selected[0].package_name == "lodash"

    def test_selected_issues_are_vulnerability_issues(self):
        issue = _sca()
        pairs = [self._make_pair(issue)]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 1
        assert isinstance(selected[0], VulnerabilityIssue)

    def test_priority_sort_critical_before_high_before_medium(self):
        critical_issue = _sca(
            package_name="pkg_c", file_path="c/package.json", severity=Severity.CRITICAL
        )
        high_issue = _sca(package_name="pkg_h", file_path="h/package.json", severity=Severity.HIGH)
        medium_issue = _sca(
            package_name="pkg_m", file_path="m/package.json", severity=Severity.MEDIUM
        )

        # Provide in wrong order intentionally
        pairs = [
            self._make_pair(medium_issue, priority=Severity.MEDIUM),
            self._make_pair(critical_issue, priority=Severity.CRITICAL),
            self._make_pair(high_issue, priority=Severity.HIGH),
        ]
        selected = select_issues_for_remediation(pairs)
        assert len(selected) == 3
        assert selected[0].package_name == "pkg_c"
        assert selected[1].package_name == "pkg_h"
        assert selected[2].package_name == "pkg_m"

    def test_empty_input_returns_empty(self):
        assert select_issues_for_remediation([]) == []

    def test_all_invalid_returns_empty(self):
        issues = [_sca(package_name=f"pkg{i}", file_path=f"p{i}/pkg.json") for i in range(3)]
        pairs = [
            self._make_pair(
                i,
                is_valid=False,
                fp_reason="dev-only",
                priority=Severity.LOW,
            )
            for i in issues
        ]
        selected = select_issues_for_remediation(pairs)
        assert selected == []

    def test_selected_issue_ids_match_recommended(self):
        issue = _sca()
        group = VulnerabilityGroup(
            group_id="sca:package.json:lodash",
            issue_type=IssueType.SCA,
            representative_issue_id=issue.id,
            issues=[issue],
        )
        triage = TriageResult(
            chain_of_thought="test reasoning",
            group_id=group.group_id,
            is_valid=True,
            original_severity=issue.severity,
            revised_priority=Severity.HIGH,
            is_unreachable_code=False,
            priority_reasoning="test",
            validity_confidence_score=1.0,
            priority_confidence_score=1.0,
            recommended_issue_id=issue.id,
            triage_method="deterministic",
        )
        selected = select_issues_for_remediation([(group, triage)])
        assert selected[0].id == issue.id


def test_pipeline_threads_explicit_java_language_to_localization_and_reachability(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    issues = [_sca()]
    context = SystemContext(environment="production", primary_language="java")
    plans = _localized_sca_issues(issues)
    with (
        patch(
            "remediation_engine.agents.triage.enrich_cves",
            side_effect=_empty_enrichment_map,
        ),
        patch(
            "remediation_engine.agents.triage._localize_sca_issues",
            return_value=plans,
        ) as prepare,
        patch("remediation_engine.agents.triage.analyze_reachability") as reachability,
    ):
        run_triage_pipeline(issues, context, repo_root=str(tmp_path))

    assert prepare.call_args.args[2] is ProjectLanguage.JAVA
    assert reachability.call_args.kwargs["project_language"] is ProjectLanguage.JAVA


def test_pipeline_without_primary_language_preserves_node_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    issues = [_sca()]
    with (
        patch(
            "remediation_engine.agents.triage.enrich_cves",
            side_effect=_empty_enrichment_map,
        ),
        patch(
            "remediation_engine.agents.triage._localize_sca_issues",
            return_value=_localized_sca_issues(issues),
        ) as prepare,
    ):
        run_triage_pipeline(issues, _ctx())

    assert prepare.call_args.args[2] is ProjectLanguage.NODEJS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FAKE_KEV_DATA = {
    "vulnerabilities": [
        {"cveID": "CVE-2021-44228", "dateAdded": "2021-12-10"},
        {"cveID": "CVE-2021-23337", "dateAdded": "2022-01-05"},
    ]
}

FAKE_EPSS_DATA = {
    "data": [
        {"cve": "CVE-2021-44228", "epss": "0.974", "percentile": "0.999"},
        {"cve": "CVE-2021-23337", "epss": "0.123", "percentile": "0.450"},
    ]
}


def _fake_response(json_data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# EPSS tests
# ---------------------------------------------------------------------------


class TestEPSSEnrichment:
    def test_invalid_scores_fall_back_without_discarding_valid_entries(self, monkeypatch):
        monkeypatch.setattr(
            "remediation_engine.tools.advisories.requests.get",
            lambda *args, **kwargs: _fake_response(
                {
                    "data": [
                        {"cve": "CVE-2026-0001", "epss": "1.2", "percentile": "0.5"},
                        {"cve": "CVE-2026-0002", "epss": "nan", "percentile": "0.5"},
                        {"cve": "CVE-2026-0004", "epss": "0.5"},
                        {"cve": "CVE-2026-0003", "epss": "0.2", "percentile": "0.9"},
                        None,
                    ]
                }
            ),
        )

        result = _fetch_epss_chunk(["CVE-2026-0001", "CVE-2026-0002", "CVE-2026-0003"])

        assert result == {"CVE-2026-0003": (0.2, 0.9)}

    def test_epss_200_parses_score_and_percentile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),  # KEV download
                _fake_response(FAKE_EPSS_DATA),  # EPSS query
            ]
            result = enrich_cves(["CVE-2021-44228"])

        assert "CVE-2021-44228" in result
        enrichment = result["CVE-2021-44228"]
        assert enrichment.epss == pytest.approx(0.974, rel=1e-3)
        assert enrichment.epss_percentile == pytest.approx(0.999, rel=1e-3)

    def test_epss_multiple_cves_parsed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228", "CVE-2021-23337"])

        assert result["CVE-2021-44228"].epss == pytest.approx(0.974, rel=1e-3)
        assert result["CVE-2021-23337"].epss == pytest.approx(0.123, rel=1e-3)

    def test_epss_timeout_returns_safe_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            # KEV succeeds; EPSS times out
            def side_effect(url, **kwargs):
                if "first.org" in url:
                    raise requests.exceptions.Timeout("timeout")
                return _fake_response(FAKE_KEV_DATA)

            mock_get.side_effect = side_effect
            result = enrich_cves(["CVE-2021-44228"])

        assert "CVE-2021-44228" in result
        enrichment = result["CVE-2021-44228"]
        assert enrichment.epss == 0.0
        assert enrichment.epss_percentile == 0.0
        # Should not raise

    def test_epss_network_error_returns_safe_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:

            def side_effect(url, **kwargs):
                if "first.org" in url:
                    raise requests.exceptions.ConnectionError("refused")
                return _fake_response(FAKE_KEV_DATA)

            mock_get.side_effect = side_effect
            # Must not raise
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].epss == 0.0


# ---------------------------------------------------------------------------
# KEV cache tests
# ---------------------------------------------------------------------------


class TestKEVEnrichment:
    def test_kev_fresh_download_marks_in_kev(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),  # KEV download
                _fake_response(FAKE_EPSS_DATA),  # EPSS
            ]
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].in_kev is True
        assert result["CVE-2021-44228"].kev_date_added == "2021-12-10"

    def test_kev_cache_hit_skips_http(self, tmp_path, monkeypatch):
        """If cache is fresh, no HTTP call should be made to the KEV URL."""
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        # Pre-populate the cache
        cache_path = tmp_path / "kev_cache.json"
        cache_path.write_text(json.dumps(FAKE_KEV_DATA), encoding="utf-8")
        # Make cache appear fresh (mtime = now)
        assert cache_path.exists()

        call_urls: list[str] = []

        def mock_get(url, **kwargs):
            call_urls.append(url)
            return _fake_response(FAKE_EPSS_DATA)

        with patch("remediation_engine.tools.advisories.requests.get", side_effect=mock_get):
            result = enrich_cves(["CVE-2021-44228"])

        # Only the EPSS URL should have been called (KEV came from cache)
        assert all("first.org" in u for u in call_urls), (
            f"KEV URL was called unexpectedly: {call_urls}"
        )
        assert result["CVE-2021-44228"].in_kev is True

    def test_cve_not_in_kev(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            kev_without_cve = {"vulnerabilities": []}
            mock_get.side_effect = [
                _fake_response(kev_without_cve),
                _fake_response({"data": []}),
            ]
            result = enrich_cves(["CVE-2099-99999"])

        assert result["CVE-2099-99999"].in_kev is False

    def test_kev_download_failure_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:

            def side_effect(url, **kwargs):
                if "cisa.gov" in url:
                    raise requests.exceptions.ConnectionError("cisa down")
                return _fake_response({"data": []})

            mock_get.side_effect = side_effect
            # Must not raise
            result = enrich_cves(["CVE-2021-44228"])

        assert result["CVE-2021-44228"].in_kev is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_cve_list_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
        with patch("remediation_engine.tools.advisories.requests.get"):
            result = enrich_cves([])
        assert result == {}

    def test_duplicate_cve_ids_deduplicated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228", "CVE-2021-44228", "CVE-2021-44228"])

        assert len(result) == 1
        assert "CVE-2021-44228" in result

    def test_enrichment_source_field_correct(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

        with patch("remediation_engine.tools.advisories.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(FAKE_KEV_DATA),
                _fake_response(FAKE_EPSS_DATA),
            ]
            result = enrich_cves(["CVE-2021-44228"])

        # Both EPSS (>0) and KEV (in_kev=True) contributed
        assert result["CVE-2021-44228"].enrichment_source in ("epss+kev", "kev+epss")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grouper_sca(
    *,
    package_name: str = "lodash",
    file_path: str = "package.json",
    cve_id: str | None = "CVE-2021-23337",
    ghsa_id: str | None = None,
    source: IssueSource = IssueSource.SEMGREP,
    severity: Severity = Severity.HIGH,
    fixed_version: str | None = None,
    package_version: str | None = "4.17.20",
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SCA,
        severity=severity,
        package_name=package_name,
        cve_id=cve_id,
        ghsa_id=ghsa_id,
        file_path=file_path,
        fixed_version=fixed_version,
        package_version=package_version,
    )


def _grouper_sast(
    *,
    file_path: str = "src/app.js",
    rule_id: str = "javascript.xss",
    line_start: int = 10,
    line_end: int = 15,
    source: IssueSource = IssueSource.SEMGREP,
    severity: Severity = Severity.MEDIUM,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=source,
        issue_type=IssueType.SAST,
        severity=severity,
        file_path=file_path,
        rule_id=rule_id,
        line_range=LineRange(start=line_start, end=line_end),
    )


def _localized(
    issue: VulnerabilityIssue,
    *,
    manifest_file: str | None = None,
    is_direct_dependency: bool | None = True,
    package_manager: str | None = "npm",
    parent_package_name: str | None = None,
    parent_package_version: str | None = None,
    parent_declaration_type: str | None = None,
) -> LocalizedIssue:
    return LocalizedIssue(
        issue=issue,
        manifest_file=manifest_file or issue.file_path,
        is_direct_dependency=is_direct_dependency,
        package_manager=package_manager,
        parent_package_name=parent_package_name,
        parent_package_version=parent_package_version,
        parent_declaration_type=parent_declaration_type,
        localization_confidence=0.95,
    )


def _plan(
    *,
    status: FixPlanStatus,
    fixed_version: str | None = None,
    workaround_snippets: list[str] | None = None,
    strategy_used: str = "osv_api",
) -> FixPlan:
    return FixPlan(
        status=status,
        fixed_version=fixed_version,
        workaround_snippets=workaround_snippets,
        instruction="test plan",
        strategy_used=strategy_used,
    )

    # ---------------------------------------------------------------------------
    # SCA grouping
    # ---------------------------------------------------------------------------


class TestSCAGrouping:
    def test_transitive_same_child_with_different_parents_retains_all_contexts(self):
        issues = [
            _grouper_sca(cve_id="CVE-2026-1001"),
            _grouper_sca(cve_id="CVE-2026-1002"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (
                    _localized(
                        issues[0],
                        is_direct_dependency=False,
                        parent_package_name="parent-a",
                        parent_package_version="1.0.0",
                        parent_declaration_type="dependencies",
                    ),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="1.0.1"),
                ),
                (
                    _localized(
                        issues[1],
                        is_direct_dependency=False,
                        parent_package_name="parent-b",
                        parent_package_version="2.0.0",
                        parent_declaration_type="dependencies",
                    ),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="1.0.1"),
                ),
            ],
        )

        assert len(groups) == 1
        group = groups[0]
        assert {context.package_name for context in group.parent_contexts} == {
            "parent-a",
            "parent-b",
        }
        assert {localized.parent_package_name for localized in group.localized_issues} == {
            "parent-a",
            "parent-b",
        }

    def test_generic_and_parent_aware_paths_share_one_remediation_group(self):
        generic = _grouper_sca(cve_id="CVE-2026-1003")
        parent_aware = _grouper_sca(cve_id="CVE-2026-1004")
        groups = group_with_fix_plans(
            [generic, parent_aware],
            plans_by_finding=[
                (
                    _localized(generic, is_direct_dependency=False),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
                ),
                (
                    _localized(
                        parent_aware,
                        is_direct_dependency=False,
                        parent_package_name="sanitize-html",
                        parent_package_version="1.4.2",
                        parent_declaration_type="dependencies",
                    ),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
                ),
            ],
        )

        assert len(groups) == 1
        assert groups[0].parent_package_name == "sanitize-html"
        assert groups[0].parent_package_version == "1.4.2"
        assert len(groups[0].parent_contexts) == 1

    def test_same_package_multiple_cves_become_one_group(self):
        issues = [
            _grouper_sca(cve_id="CVE-2021-23337"),
            _grouper_sca(cve_id="CVE-2020-28500"),
        ]
        pairs = [
            (
                _localized(issues[0]),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
            ),
            (
                _localized(issues[1]),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.22"),
            ),
        ]
        groups = group_with_fix_plans(issues, plans_by_finding=pairs)
        assert len(groups) == 1
        g = groups[0]
        assert set(g.cve_ids) == {"CVE-2021-23337", "CVE-2020-28500"}
        assert g.issue_type == IssueType.SCA
        selected = selected_group_plan(g)
        assert selected is not None
        assert selected.fixed_version == "4.17.22"
        assert selected.strategy_used == "UPDATE_VERSION"

    def test_duplicate_cve_same_component_is_deduped(self):
        """Same CVE appearing twice (e.g. from the same scanner run) â†’ one CVE ID."""
        issues = [
            _grouper_sca(cve_id="CVE-2021-23337"),
            _grouper_sca(cve_id="CVE-2021-23337"),  # Exact duplicate
        ]
        pairs = [
            (
                _localized(issues[0]),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
            ),
            (
                _localized(issues[1]),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
            ),
        ]
        groups = group_with_fix_plans(issues, plans_by_finding=pairs)
        assert len(groups) == 1
        assert groups[0].cve_ids == ["CVE-2021-23337"]

    def test_cross_tool_same_cve_component_file_merges(self):
        """Semgrep SCA + ODC finding the same CVE on the same package/file â†’ merge."""
        semgrep_issue = _grouper_sca(source=IssueSource.SEMGREP, cve_id="CVE-2021-44228")
        odc_issue = _grouper_sca(source=IssueSource.ODC, cve_id="CVE-2021-44228")
        groups = group_with_fix_plans(
            [semgrep_issue, odc_issue],
            plans_by_finding=[
                (
                    _localized(semgrep_issue),
                    _plan(
                        status=FixPlanStatus.WORKAROUND_FOUND, workaround_snippets=["Workaround A"]
                    ),
                ),
                (
                    _localized(odc_issue),
                    _plan(
                        status=FixPlanStatus.WORKAROUND_FOUND, workaround_snippets=["Workaround B"]
                    ),
                ),
            ],
        )
        assert len(groups) == 1
        g = groups[0]
        assert IssueSource.SEMGREP in g.sources
        assert IssueSource.ODC in g.sources
        assert g.sources == sorted(g.sources, key=lambda source: source.value)
        assert len(g.issues) == 2
        selected = selected_group_plan(g)
        assert selected is not None
        assert selected.strategy_used == "WORKAROUND"
        assert selected.workaround_snippets == ["Workaround A", "Workaround B"]

    def test_different_packages_different_files_make_separate_groups(self):
        issues = [
            _grouper_sca(package_name="lodash", file_path="package.json"),
            _grouper_sca(package_name="express", file_path="backend/package.json"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (_localized(issues[0]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
                (_localized(issues[1]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
            ],
        )
        assert len(groups) == 2

    def test_scoped_npm_purl_fallback_preserves_scope_in_group_identity(self):
        issue = _grouper_sca(package_name=None, package_version="1.2.3").model_copy(
            update={"purl": "pkg:npm/@scope/pkg@1.2.3"}
        )
        plan = FixPlan(
            status=FixPlanStatus.NO_FIX,
            instruction="No fix",
            strategy_used="NO_FIX",
        )

        groups = group_with_fix_plans([issue], plans_by_finding=[(_localized(issue), plan)])

        assert len(groups) == 1
        assert groups[0].vulnerable_component == "@scope/pkg"
        assert "@scope/pkg" in groups[0].group_id

    def test_pypi_package_names_share_pep503_group_identity(self):
        issues = [
            _grouper_sca(package_name="zope.interface", file_path="requirements.txt").model_copy(
                update={"ecosystem": "pypi", "purl": "pkg:pypi/zope.interface"}
            ),
            _grouper_sca(package_name="ZOPE_interface", file_path="requirements.txt").model_copy(
                update={"ecosystem": "python", "purl": "pkg:pypi/zope_interface"}
            ),
        ]
        pairs = [
            (
                _localized(issue, manifest_file="requirements.txt", package_manager="pip"),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="1.0"),
            )
            for issue in issues
        ]

        groups = group_with_fix_plans(issues, plans_by_finding=pairs)

        assert len(groups) == 1
        assert groups[0].vulnerable_component == "zope-interface"
        assert "zope-interface" in groups[0].group_id

    def test_non_pypi_package_identity_remains_unchanged(self):
        issues = [
            _grouper_sca(package_name="zope.interface", file_path="package.json"),
            _grouper_sca(package_name="zope-interface", file_path="package.json"),
        ]
        pairs = [
            (_localized(issue), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none"))
            for issue in issues
        ]

        assert len(group_with_fix_plans(issues, plans_by_finding=pairs)) == 2

    def test_group_tracks_all_manifest_paths(self):
        issues = [
            _grouper_sca(package_name="lodash", file_path="/src/package-lock.json?/lodash:4.17.20"),
            _grouper_sca(
                package_name="lodash", file_path="/frontend/package-lock.json?/lodash:4.17.20"
            ),
        ]
        pairs = [
            (
                _localized(issues[0], manifest_file="package.json"),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
            ),
            (
                _localized(issues[1], manifest_file="frontend/package.json"),
                _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
            ),
        ]

        groups = group_with_fix_plans(issues, plans_by_finding=pairs)

        assert len(groups) == 2
        assert groups[0].file_paths == ["frontend/package.json"] or groups[0].file_paths == [
            "package.json"
        ]
        all_group_paths = sorted(path for group in groups for path in group.file_paths)
        assert all_group_paths == ["frontend/package.json", "package.json"]

    def test_representative_prefers_issue_with_fixed_version(self):
        no_fix = _grouper_sca(fixed_version=None)
        with_fix = _grouper_sca(fixed_version="4.17.21")
        groups = group_issues([no_fix, with_fix])
        assert len(groups) == 1
        assert groups[0].representative_issue_id == with_fix.id

    def test_versions_deduplicated(self):
        issues = [
            _grouper_sca(package_version="4.17.20", cve_id="CVE-2021-23337"),
            _grouper_sca(package_version="4.17.20", cve_id="CVE-2020-28500"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (_localized(issues[0]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
                (_localized(issues[1]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
            ],
        )
        assert groups[0].versions == ["4.17.20"]

    def test_no_cve_sca_still_groups(self):
        issues = [_grouper_sca(cve_id=None), _grouper_sca(cve_id=None)]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (_localized(issues[0]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
                (_localized(issues[1]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
            ],
        )
        # Same component/file â†’ one group; no CVE IDs expected
        assert len(groups) == 1
        assert groups[0].cve_ids == []

    def test_ghsa_ids_are_deduplicated_at_group_level(self):
        issues = [
            _grouper_sca(cve_id=None, ghsa_id="GHSA-VPQ2-C234-7XJ6"),
            _grouper_sca(cve_id=None, ghsa_id="ghsa-vpq2-c234-7xj6"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (_localized(issues[0]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
                (_localized(issues[1]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
            ],
        )
        assert len(groups) == 1
        assert groups[0].cve_ids == []
        assert groups[0].ghsa_ids == ["GHSA-VPQ2-C234-7XJ6"]

    def test_same_component_different_fix_strategies_split_groups(self):
        issues = [
            _grouper_sca(cve_id="CVE-2021-23337"),
            _grouper_sca(cve_id="CVE-2020-28500"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (
                    _localized(issues[0]),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.17.21"),
                ),
                (
                    _localized(issues[1]),
                    _plan(
                        status=FixPlanStatus.WORKAROUND_FOUND,
                        workaround_snippets=["Mitigation: disable feature"],
                    ),
                ),
            ],
        )
        assert len(groups) == 1
        group = groups[0]
        assert group.group_id == "sca:package.json:lodash"
        assert {candidate.plan.status for candidate in group.fix_plan_candidates} == {
            FixPlanStatus.VERSION_FOUND,
            FixPlanStatus.WORKAROUND_FOUND,
        }

    def test_highest_fixed_version_handles_partial_and_v_prefix(self):
        issues = [
            _grouper_sca(cve_id="CVE-2021-23337"),
            _grouper_sca(cve_id="CVE-2020-28500"),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (
                    _localized(issues[0]),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="v1.2"),
                ),
                (
                    _localized(issues[1]),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="1.2.5"),
                ),
            ],
        )
        assert len(groups) == 1
        assert selected_group_plan(groups[0]).fixed_version == "1.2.5"
        assert len(groups[0].localized_issues) == 2

    def test_highest_fixed_version_ignores_build_metadata(self):
        issue = _grouper_sca()
        groups = group_with_fix_plans(
            [issue],
            plans_by_finding=[
                (
                    _localized(issue),
                    _plan(
                        status=FixPlanStatus.VERSION_FOUND,
                        fixed_version="1.2.3+build.1",
                    ),
                )
            ],
        )

        assert selected_group_plan(groups[0]).fixed_version == "1.2.3+build.1"

        # ---------------------------------------------------------------------------
        # SAST grouping
        # ---------------------------------------------------------------------------


class TestSASTGrouping:
    def test_sast_issue_becomes_singleton_group(self):
        issues = [_grouper_sast()]
        groups = group_issues(issues)
        assert len(groups) == 1
        g = groups[0]
        assert g.issue_type == IssueType.SAST
        assert g.group_id.startswith("sast:")
        assert len(g.issues) == 1
        assert g.file_paths == ["src/app.js"]

    def test_different_sast_locations_make_separate_groups(self):
        issues = [
            _grouper_sast(file_path="src/a.js", rule_id="xss", line_start=10, line_end=10),
            _grouper_sast(file_path="src/a.js", rule_id="xss", line_start=50, line_end=55),
            _grouper_sast(file_path="src/b.js", rule_id="xss", line_start=10, line_end=10),
        ]
        groups = group_issues(issues)
        assert len(groups) == 3

    def test_sast_group_has_empty_cve_ids(self):
        groups = group_issues([_grouper_sast()])
        assert groups[0].cve_ids == []

    def test_sast_group_captures_rule_id_as_component(self):
        groups = group_issues([_grouper_sast(rule_id="javascript.sqli")])
        assert groups[0].vulnerable_component == "javascript.sqli"

        # ---------------------------------------------------------------------------
        # Mixed SAST + SCA
        # ---------------------------------------------------------------------------


class TestMixedGrouping:
    def test_mixed_list_correct_group_count(self):
        issues = [
            _grouper_sca(package_name="lodash", file_path="package.json"),
            _grouper_sca(package_name="express", file_path="backend/package.json"),
            _grouper_sast(file_path="src/login.js", rule_id="xss", line_start=5, line_end=5),
        ]
        groups = group_with_fix_plans(
            issues,
            plans_by_finding=[
                (_localized(issues[0]), _plan(status=FixPlanStatus.NO_FIX, strategy_used="none")),
                (
                    _localized(issues[1]),
                    _plan(status=FixPlanStatus.VERSION_FOUND, fixed_version="4.18.3"),
                ),
            ],
        )
        assert len(groups) == 3

    def test_mixed_list_correct_types(self):
        issues = [
            _grouper_sca(),
            _grouper_sast(),
        ]
        groups = group_issues(issues)
        types = {g.issue_type for g in groups}
        assert IssueType.SCA in types
        assert IssueType.SAST in types

    def test_empty_input_returns_empty_list(self):
        assert group_issues([]) == []


def _sca_issue(
    package_name: str,
    *,
    ecosystem: str | None = None,
    purl: str | None = None,
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name=package_name,
        ecosystem=ecosystem,
        purl=purl,
    )


def _sca_group(
    package_name: str,
    *,
    ecosystem: str | None = None,
    purl: str | None = None,
) -> VulnerabilityGroup:
    issue = _sca_issue(package_name, ecosystem=ecosystem, purl=purl)
    return VulnerabilityGroup(
        group_id=f"sca:package.json:{package_name}",
        issue_type=IssueType.SCA,
        vulnerable_component=package_name,
        representative_issue_id=issue.id,
        issues=[issue],
    )


def test_analyze_reachability_distinguishes_direct_and_transitive(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"lodash":"1.0.0","express":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("use_lodash", encoding="utf-8")
    (tmp_path / "node_modules" / "vendor").mkdir(parents=True)
    (tmp_path / "node_modules" / "vendor" / "index.js").write_text(
        "use_express",
        encoding="utf-8",
    )

    monkeypatch.setattr("remediation_engine.agents.triage.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "remediation_engine.agents.triage.load_source_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage.parse_source",
        lambda source_bytes, language: SimpleNamespace(root_node=source_bytes.decode("utf-8")),
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage.extract_all_imports",
        lambda root_node, source_bytes: (
            ['import lodash from "lodash";']
            if "use_lodash" in root_node
            else ['import express from "express";']
            if "use_express" in root_node
            else []
        ),
    )

    groups = [
        _sca_group("lodash"),
        _sca_group("express"),
        _sca_group("minimist"),
    ]

    analyze_reachability(groups, tmp_path)

    assert groups[0].is_reachable is True
    assert groups[1].is_reachable is False
    assert groups[2].is_reachable is None


def test_analyze_reachability_continues_after_parse_failure(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"lodash":"1.0.0","express":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.js").write_text("broken", encoding="utf-8")
    (tmp_path / "src" / "ok.js").write_text("ok", encoding="utf-8")

    monkeypatch.setattr("remediation_engine.agents.triage.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "remediation_engine.agents.triage.load_source_bytes",
        lambda path: path.read_bytes(),
    )

    def _parse(source_bytes, language):
        marker = source_bytes.decode("utf-8")
        if marker == "broken":
            raise RuntimeError("parse failed")
        return SimpleNamespace(root_node=marker)

    monkeypatch.setattr("remediation_engine.agents.triage.parse_source", _parse)
    monkeypatch.setattr(
        "remediation_engine.agents.triage.extract_all_imports",
        lambda root_node, source_bytes: (
            ['import lodash from "lodash";'] if root_node == "ok" else []
        ),
    )

    groups = [_sca_group("lodash"), _sca_group("express")]
    analyze_reachability(groups, tmp_path)

    assert groups[0].is_reachable is True
    assert groups[1].is_reachable is False


def test_analyze_reachability_skips_symlinked_source_outside_repository(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"lodash":"1.0.0"}}',
        encoding="utf-8",
    )
    outside = tmp_path.parent / f"{tmp_path.name}-reachability-outside"
    outside.mkdir()
    (outside / "secret.js").write_text('import lodash from "lodash";', encoding="utf-8")
    (tmp_path / "linked.js").symlink_to(outside / "secret.js")

    groups = [_sca_group("lodash")]
    analyze_reachability(groups, tmp_path)

    assert groups[0].is_reachable is False


def test_reachability_matches_package_identity_not_substrings(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"got":"1.0.0","@scope/pkg":"1.0.0","right":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("imports", encoding="utf-8")
    monkeypatch.setattr("remediation_engine.agents.triage.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "remediation_engine.agents.triage.load_source_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage.parse_source",
        lambda source_bytes, language: SimpleNamespace(root_node=source_bytes.decode()),
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage.extract_all_imports",
        lambda *_: [
            'import forgot from "forgot";',
            'import nested from "@scope/pkg/subpath";',
            'const left = require("left"), right = require("right");',
        ],
    )

    got, scoped, second_require = (
        _sca_group("got"),
        _sca_group("@scope/pkg"),
        _sca_group("right"),
    )
    analyze_reachability([got, scoped, second_require], tmp_path)

    assert got.is_reachable is False
    assert scoped.is_reachable is True
    assert second_require.is_reachable is True


def test_reachability_scans_mjs_cjs_and_imports_after_twentieth(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"module-24":"1.0.0","module-cjs":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "app.mjs").write_text("module", encoding="utf-8")
    (tmp_path / "worker.cjs").write_text("commonjs", encoding="utf-8")
    monkeypatch.setattr("remediation_engine.agents.triage.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "remediation_engine.agents.triage.load_source_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage.parse_source",
        lambda source_bytes, language: SimpleNamespace(root_node=source_bytes.decode()),
    )

    def imports(root_node, _source_bytes):
        if root_node == "module":
            return [f'import mod{i} from "mod{i}";' for i in range(24)] + [
                'import target from "module-24";'
            ]
        return ['const target = require("module-cjs");']

    monkeypatch.setattr("remediation_engine.agents.triage.extract_all_imports", imports)
    groups = [_sca_group("module-24"), _sca_group("module-cjs")]

    analyze_reachability(groups, tmp_path)

    assert [group.is_reachable for group in groups] == [True, True]


def test_reachability_rejects_external_root_manifest_symlink(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "external-package.json"
    outside.write_text('{"dependencies":{"lodash":"1.0.0"}}', encoding="utf-8")
    (repo / "package.json").symlink_to(outside)
    group = _sca_group("lodash")

    analyze_reachability([group], repo)

    assert group.is_reachable is None


# Stage-2 behavior retained during the architecture merge.
def test_analyze_reachability_leaves_pypi_findings_unknown(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"requests":"2.0.0","flask":"2.0.0"}}',
        encoding="utf-8",
    )
    groups = [
        _sca_group("requests", ecosystem="pypi"),
        _sca_group("flask", purl="pkg:pypi/flask@2.0.0"),
        _sca_group("lodash", ecosystem="python"),
    ]

    def _unexpected_js_analysis(*_args, **_kwargs):
        raise AssertionError("PyPI findings must not use JavaScript reachability evidence")

    monkeypatch.setattr(
        "remediation_engine.agents.triage._load_direct_dependencies",
        _unexpected_js_analysis,
    )
    monkeypatch.setattr(
        "remediation_engine.agents.triage._collect_global_imports",
        _unexpected_js_analysis,
    )

    analyze_reachability(groups, tmp_path)

    assert [group.is_reachable for group in groups] == [None, None, None]
