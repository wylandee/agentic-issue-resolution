"""Phase 2: DeepEval and structural evaluation for Fix Planner web extraction."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from remediation_engine.tools.fix_planner import SerperLLMResult
from tests.evals.conftest import EvalSettings

try:
    from deepeval import assert_test
    from deepeval.metrics import FaithfulnessMetric, GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    LLMTestCase = None  # type: ignore[assignment,misc]
    LLMTestCaseParams = None  # type: ignore[assignment,misc]
    GEval = None  # type: ignore[assignment,misc]
    FaithfulnessMetric = None  # type: ignore[assignment,misc]
    assert_test = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Golden Dataset Loader for Pytest Parametrization
# ---------------------------------------------------------------------------

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "triage_cases.json"


def _load_fix_planner_cases() -> list[dict[str, Any]]:
    if not _GOLDEN_FILE.exists():
        return []
    try:
        data = json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [c for c in data if c.get("eval_type") == "fix_planner"]
        return []
    except Exception:
        return []


_FP_CASES = _load_fix_planner_cases()
_FP_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_FP_CASES)]

_SEMVER_RE = re.compile(
    r"^\bv?(\d+)\.(\d+)(?:\.(\d+))?(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b$"
)


# ---------------------------------------------------------------------------
# Prompt & Context Formatting Helpers (Matching Production fix_planner.py)
# ---------------------------------------------------------------------------


def build_fix_planner_production_prompt(
    package_name: str,
    vuln_id: str,
    page_dumps: str,
) -> str:
    """Build the exact prompt template used by fix_planner._llm_extract_remediation."""
    return (
        f"You are analyzing web search results for a security advisory about {package_name} ({vuln_id}).\n\n"
        "Extract the safest remediation strategy:\n"
        "- If an official patched version is explicitly mentioned, return strategy 'VERSION_BUMP' "
        "and the fixed_version.\n"
        "- If no patched version exists or the package is deprecated/unmaintained, but a code snippet, "
        "monkey-patch, or configuration workaround is described, return strategy 'CODE_WORKAROUND' "
        "and the workaround_snippets.\n"
        "- If no fix or workaround is found, return strategy 'NO_FIX'.\n\n"
        f"Web page content:\n{page_dumps[:16000]}"
    )


# ---------------------------------------------------------------------------
# Structural & Deterministic Validation Helpers
# ---------------------------------------------------------------------------


def validate_serper_schema(result_dict: dict[str, Any]) -> list[str]:
    """Validate SerperLLMResult contract invariants."""
    violations: list[str] = []
    try:
        SerperLLMResult(**result_dict)
    except Exception as exc:
        violations.append(f"SerperLLMResult contract violation: {exc}")

    strategy = result_dict.get("strategy")
    fixed_ver = result_dict.get("fixed_version")
    snippets = result_dict.get("workaround_snippets")

    valid_strategies = {"VERSION_BUMP", "CODE_WORKAROUND", "NO_FIX"}
    if strategy not in valid_strategies:
        violations.append(f"Unknown strategy '{strategy}'")

    if strategy == "VERSION_BUMP":
        if not fixed_ver or not str(fixed_ver).strip():
            violations.append("VERSION_BUMP strategy requires non-empty fixed_version")
        elif not _SEMVER_RE.match(str(fixed_ver).strip()):
            violations.append(f"fixed_version '{fixed_ver}' is not valid semver")

    elif strategy == "CODE_WORKAROUND":
        if not snippets or not isinstance(snippets, list):
            violations.append(
                "CODE_WORKAROUND strategy requires non-empty workaround_snippets list"
            )
        elif not any(str(s).strip() for s in snippets):
            violations.append("workaround_snippets contains only blank strings")

    elif strategy == "NO_FIX":
        if fixed_ver is not None and str(fixed_ver).strip():
            violations.append(f"NO_FIX strategy must not contain fixed_version, got '{fixed_ver}'")
        if snippets:
            violations.append(
                f"NO_FIX strategy must not contain workaround_snippets, got {snippets}"
            )

    return violations


def validate_version_in_source(fixed_version: str | None, source_content: str) -> bool:
    """Return True if the fixed_version appears as a substring in source_content."""
    if not fixed_version:
        return True
    # Strip optional leading 'v'
    clean_ver = fixed_version.lstrip("v")
    return clean_ver in source_content or fixed_version in source_content


def validate_snippet_in_source(snippets: list[str] | None, source_content: str) -> list[str]:
    """Return list of snippets whose key terms are missing from source_content."""
    missing: list[str] = []
    if not snippets:
        return missing

    for snippet in snippets:
        # Extract alphanumeric words of length >= 4
        words = [w for w in re.findall(r"[A-Za-z0-9_]+", snippet) if len(w) >= 4]
        if words:
            matching = sum(1 for w in words if w in source_content)
            if matching / len(words) < 0.5:
                missing.append(snippet[:50] + "...")
        elif snippet.strip() not in source_content:
            missing.append(snippet[:50] + "...")

    return missing


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.eval
class TestFixPlannerEval:
    """Evaluation test suite for Fix Planner web extraction results."""

    @pytest.mark.parametrize(
        "case",
        _FP_CASES or [{}],
        ids=_FP_CASE_IDS or ["no_cases"],
    )
    def test_version_extraction_from_advisory(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Correctly extracts strategy and patched version from advisory pages."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_extraction_result", {})
        source_content = case.get("source_page_content", "")
        expected_strat = case.get("expected_strategy")
        expected_ver = case.get("expected_fixed_version")

        # 1. Deterministic schema & value assertions
        schema_violations = validate_serper_schema(result)
        assert not schema_violations, f"Case '{case_id}' failed schema: {schema_violations}"

        assert result.get("strategy") == expected_strat, (
            f"Case '{case_id}' strategy mismatch: expected {expected_strat}, got {result.get('strategy')}"
        )
        if expected_ver is not None:
            assert result.get("fixed_version") == expected_ver, (
                f"Case '{case_id}' version mismatch: expected {expected_ver}, got {result.get('fixed_version')}"
            )

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_fix_planner_production_prompt(
                "target-package", "CVE-2024-XXXX", source_content
            )
            actual_output = json.dumps(result, indent=2)
            expected_output = json.dumps(
                {"strategy": expected_strat, "fixed_version": expected_ver},
                indent=2,
            )

            test_case = LLMTestCase(
                name=f"{case_id} [Fix Extraction Accuracy]",
                input=prompt,
                actual_output=actual_output,
                expected_output=expected_output,
            )

            extraction_geval = GEval(
                name="Fix Extraction Accuracy",
                criteria=(
                    "Given web page content about a vulnerable package, evaluate whether "
                    "the extracted strategy (VERSION_BUMP, CODE_WORKAROUND, NO_FIX) and "
                    "fixed_version are correct and directly supported by the text."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [extraction_geval])

    @pytest.mark.parametrize(
        "case",
        _FP_CASES or [{}],
        ids=_FP_CASE_IDS or ["no_cases"],
    )
    def test_workaround_extraction_from_issues(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Correctly extracts code workaround snippets from issue discussions."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_extraction_result", {})
        expected_strat = case.get("expected_strategy")

        if expected_strat != "CODE_WORKAROUND":
            pytest.skip(f"Case '{case_id}' is strategy '{expected_strat}', not CODE_WORKAROUND")

        # 1. Deterministic assertions
        schema_violations = validate_serper_schema(result)
        assert not schema_violations, f"Case '{case_id}' failed schema: {schema_violations}"

        snippets = result.get("workaround_snippets")
        assert snippets and len(snippets) > 0, (
            f"Case '{case_id}' must provide non-empty workaround_snippets for CODE_WORKAROUND"
        )

        # 2. Live evaluation with DeepEval GEval
        if eval_settings.is_live:
            if not HAS_DEEPEVAL or assert_test is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            source_content = case.get("source_page_content", "")
            prompt = build_fix_planner_production_prompt(
                "target-package", "CVE-2024-XXXX", source_content
            )
            test_case = LLMTestCase(
                name=f"{case_id} [Workaround Extraction Quality]",
                input=prompt,
                actual_output=json.dumps(result, indent=2),
            )

            workaround_geval = GEval(
                name="Workaround Extraction Quality",
                criteria=(
                    "Evaluate whether the extracted code workaround snippets are actionable, "
                    "syntactically valid, safe, and directly mitigate the vulnerability described in the source."
                ),
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                ],
                threshold=0.70,
                model=eval_settings.judge_model,
                verbose_mode=True,
            )

            assert_test(test_case, [workaround_geval])

    @pytest.mark.parametrize(
        "case",
        _FP_CASES or [{}],
        ids=_FP_CASE_IDS or ["no_cases"],
    )
    def test_no_hallucinated_versions(
        self,
        case: dict[str, Any],
        eval_settings: EvalSettings,
    ) -> None:
        """Extracted fixed version and snippets exist in the source web page content."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")
        result = case.get("llm_extraction_result", {})
        source_content = case.get("source_page_content", "")
        fixed_ver = result.get("fixed_version")
        snippets = result.get("workaround_snippets")

        # 1. Deterministic version substring check
        version_in_source = validate_version_in_source(fixed_ver, source_content)
        assert version_in_source, (
            f"Case '{case_id}' extracted version '{fixed_ver}' which is not in source page content!"
        )

        missing_snippets = validate_snippet_in_source(snippets, source_content)
        assert not missing_snippets, (
            f"Case '{case_id}' extracted snippets not found in source: {missing_snippets}"
        )

        # 2. Live evaluation with DeepEval FaithfulnessMetric
        if eval_settings.is_live and result.get("strategy") != "NO_FIX":
            if not HAS_DEEPEVAL or assert_test is None or FaithfulnessMetric is None:
                pytest.skip("deepeval package is required for live evaluations")
            if not eval_settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
                pytest.skip("OPENAI_API_KEY environment variable is required for live evaluations")

            prompt = build_fix_planner_production_prompt(
                "target-package", "CVE-2024-XXXX", source_content
            )
            test_case = LLMTestCase(
                name=f"{case_id} [Extraction Faithfulness]",
                input=prompt,
                actual_output=json.dumps(result, indent=2),
                context=[source_content],
                retrieval_context=[source_content],
            )

            faithfulness = FaithfulnessMetric(
                threshold=0.85,
                model=eval_settings.judge_model,
                include_reason=True,
                verbose_mode=True,
            )

            assert_test(test_case, [faithfulness])


# ---------------------------------------------------------------------------
# Unit tests for the evaluation helpers themselves (adversarial validation)
# ---------------------------------------------------------------------------


def test_fix_planner_evaluator_catches_schema_violations() -> None:
    """Ensure validate_serper_schema catches missing versions, bad semver, and illegal fields."""
    # VERSION_BUMP with missing fixed_version
    bad_bump = {"strategy": "VERSION_BUMP", "fixed_version": None}
    assert any("fixed_version" in v for v in validate_serper_schema(bad_bump))

    # VERSION_BUMP with invalid semver
    bad_semver = {"strategy": "VERSION_BUMP", "fixed_version": "not-a-version"}
    assert any("semver" in v for v in validate_serper_schema(bad_semver))

    # CODE_WORKAROUND with empty list
    bad_workaround = {"strategy": "CODE_WORKAROUND", "workaround_snippets": []}
    assert any("workaround_snippets" in v for v in validate_serper_schema(bad_workaround))

    # NO_FIX with a fixed_version present
    bad_no_fix = {"strategy": "NO_FIX", "fixed_version": "1.0.0"}
    assert any("NO_FIX" in v for v in validate_serper_schema(bad_no_fix))


def test_fix_planner_evaluator_catches_hallucinated_version() -> None:
    """Ensure validate_version_in_source catches hallucinated versions."""
    source_content = "Advisory states vulnerability fixed in version 2.4.1."
    assert validate_version_in_source("2.4.1", source_content)
    assert not validate_version_in_source("9.9.9", source_content)
