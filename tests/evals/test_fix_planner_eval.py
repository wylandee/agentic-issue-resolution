"""Phase 2: DeepEval and structural evaluation for Fix Planner web extraction."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from packaging.version import InvalidVersion, Version

import remediation_engine.tools.fix_planner as fix_planner
from remediation_engine.settings import AppSettings
from remediation_engine.tools.fix_planner import SerperLLMResult
from tests.evals.conftest import EvalSettings
from tests.evals.golden_schema import load_golden_dataset, offline_fixture

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

_GOLDEN_FILE = Path(__file__).resolve().parent / "golden" / "fix_planner_cases.json"


def _load_fix_planner_cases() -> list[dict[str, Any]]:
    """Load the validated canonical Fix Planner dataset."""
    return [
        case
        for case in load_golden_dataset(_GOLDEN_FILE, dataset_name="fix_planner_cases")
        if case.get("eval_type") == "fix_planner"
    ]


_FP_CASES = _load_fix_planner_cases()
_FP_CASE_IDS = [c.get("case_id", f"case_{i}") for i, c in enumerate(_FP_CASES)]
_LIVE_FP_RESULTS: dict[tuple[str, str], dict[str, Any]] = {}

_SEMVER_RE = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


# ---------------------------------------------------------------------------
# Prompt & Context Formatting Helpers (Matching Production fix_planner.py)
# ---------------------------------------------------------------------------


def build_fix_planner_production_prompt(
    package_name: str,
    vuln_id: str,
    page_dumps: str | list[dict[str, str]],
) -> str:
    """Build the same page-framed prompt as production ``plan_fix``."""
    if isinstance(page_dumps, list):
        pages_text = "\n\n".join(
            f"--- Page {index}: {page.get('url', '')} ---\n{page.get('content', '')}"
            for index, page in enumerate(page_dumps, 1)
        )
    else:
        pages_text = page_dumps
    return (
        f"You are a security engineer analyzing web page findings for vulnerable package '{package_name}' "
        f"(vulnerability ID: '{vuln_id}').\n\n"
        "Below are fetched web page contents from security advisories, release notes, or issue threads:\n\n"
        f"{pages_text}\n\n"
        "Analyze the content carefully and extract actionable remediation info:\n"
        "1. If the page provides a clear patched version to bump to, set strategy='VERSION_BUMP' and fixed_version='X.Y.Z'.\n"
        "2. If the page provides a code workaround or mitigation steps, set strategy='CODE_WORKAROUND' and workaround_snippets=[...].\n"
        "3. If no actionable fixed version or code workaround is found, set strategy='NO_FIX'.\n"
    )


def _case_source_content(case: dict[str, Any]) -> str:
    """Render all structured pages with production-style URL framing."""
    pages = case.get("pages", [])
    if isinstance(pages, list) and pages:
        return "\n\n".join(
            f"--- Page {index}: {page.get('url', '')} ---\n{page.get('content', '')}"
            for index, page in enumerate(pages, 1)
            if isinstance(page, dict)
        )
    return str(case.get("source_page_content", ""))


def _case_result(case: dict[str, Any]) -> dict[str, Any]:
    """Return the historical structured result only for offline evaluation."""
    value = offline_fixture(case).get("actual_output", {})
    return value if isinstance(value, dict) else {}


def _faithfulness_projection(result: dict[str, Any]) -> dict[str, Any]:
    """Keep faithfulness scoring on extracted source-backed evidence.

    Deterministic checks validate strategy and schema separately. Null
    extraction fields mean no value was extracted; they do not claim that the
    source proves the value is absent, so they are omitted from the judge input.
    """
    return {
        field: result[field]
        for field in ("fixed_version", "workaround_snippets")
        if result.get(field) not in (None, [])
    }


def _judge_extraction_output(result: dict[str, Any]) -> dict[str, Any]:
    """Keep non-contract reasoning text out of structured extraction judgments."""
    return {
        field: result.get(field) for field in ("strategy", "fixed_version", "workaround_snippets")
    }


def _case_prompt_inputs(case: dict[str, Any]) -> tuple[str, str, list[dict[str, str]]]:
    """Return the package, identifier, and structured pages for production framing."""
    replay_input = case.get("replay", {}).get("input", {})
    replay_input = replay_input if isinstance(replay_input, dict) else {}
    pages = case.get("pages", [])
    return (
        str(replay_input.get("package_name", "target-package")),
        str(replay_input.get("vuln_id", "CVE-2024-XXXX")),
        pages if isinstance(pages, list) else [],
    )


def _live_case_result(
    case: dict[str, Any],
    settings: EvalSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run production structured extraction once for a live golden case.

    Args:
        case: Curated page-backed Fix Planner case.
        settings: Live judge settings, including the API credential.
        monkeypatch: Pytest patch fixture for binding explicit extraction settings.

    Returns:
        The structured result produced by the production page-extraction helper.

    Raises:
        AssertionError: If the production extractor returns no structured result.
    """
    package_name, vuln_id, pages = _case_prompt_inputs(case)
    model_settings = replace(
        AppSettings.from_env(),
        openai_api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY", ""),
    )
    cache_key = (str(case.get("case_id", "unknown")), model_settings.triage_llm_model)
    if cache_key not in _LIVE_FP_RESULTS:
        monkeypatch.setattr(fix_planner, "get_runtime_settings", lambda: model_settings)
        result = fix_planner._llm_extract_remediation(pages, package_name, vuln_id)
        if result is None:
            raise AssertionError(f"Production Fix Planner extraction failed for {cache_key[0]!r}.")
        _LIVE_FP_RESULTS[cache_key] = result
    return _LIVE_FP_RESULTS[cache_key]


def _result_for_evaluation(
    case: dict[str, Any],
    settings: EvalSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Use production output in live mode and fixture observations offline."""
    has_live_credentials = bool(settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    if settings.is_live and HAS_DEEPEVAL and assert_test is not None and has_live_credentials:
        return _live_case_result(case, settings, monkeypatch)
    return _case_result(case)


# ---------------------------------------------------------------------------
# Structural & Deterministic Validation Helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_PACKAGE_NAMES = frozenset({"", "target-package", "target_package", "target package"})


def _case_package_name(case: dict[str, Any]) -> str | None:
    """Return a concrete target package for package-aware evidence checks."""
    replay = case.get("replay", {})
    replay_input = replay.get("input", {}) if isinstance(replay, dict) else {}
    package = replay_input.get("package_name") if isinstance(replay_input, dict) else None
    if not isinstance(package, str):
        return None
    package = package.strip().strip(chr(96))
    return None if package.casefold() in _PLACEHOLDER_PACKAGE_NAMES else package


def validate_serper_schema(
    result_dict: dict[str, Any], *, project_language: str | None = None
) -> list[str]:
    """Validate SerperLLMResult contract and strategy-specific field invariants."""
    violations: list[str] = []
    allowed_fields = {"strategy", "fixed_version", "workaround_snippets", "reasoning"}
    unknown_fields = set(result_dict) - allowed_fields
    if unknown_fields:
        violations.append(
            "SerperLLMResult contains unsupported fields: " + ", ".join(sorted(unknown_fields))
        )
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
        if not isinstance(fixed_ver, str) or not fixed_ver.strip():
            violations.append("VERSION_BUMP strategy requires non-empty fixed_version")
        elif str(project_language or "").casefold() in {"python", "python3"}:
            try:
                Version(fixed_ver.strip())
            except InvalidVersion:
                violations.append(f"fixed_version '{fixed_ver}' is not valid PEP 440")
        elif not _SEMVER_RE.fullmatch(fixed_ver.strip()):
            violations.append(f"fixed_version '{fixed_ver}' is not valid semver")
        if snippets not in (None, []):
            violations.append("VERSION_BUMP strategy must not contain workaround_snippets")

    elif strategy == "CODE_WORKAROUND":
        if fixed_ver is not None and (not isinstance(fixed_ver, str) or fixed_ver.strip()):
            violations.append("CODE_WORKAROUND strategy must not contain fixed_version")
        if not isinstance(snippets, list) or not snippets:
            violations.append(
                "CODE_WORKAROUND strategy requires non-empty workaround_snippets list"
            )
        elif not all(isinstance(snippet, str) for snippet in snippets):
            violations.append("workaround_snippets must contain only strings")
        elif not any(snippet.strip() for snippet in snippets):
            violations.append("workaround_snippets contains only blank strings")

    elif strategy == "NO_FIX":
        if fixed_ver is not None and (not isinstance(fixed_ver, str) or fixed_ver.strip()):
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
        if not isinstance(snippet, str):
            missing.append(str(snippet)[:50] + "...")
            continue
        words = [w.casefold() for w in re.findall(r"[A-Za-z0-9_]+", snippet) if len(w) >= 4]
        source_lower = source_content.casefold()
        if words:
            matching = sum(1 for word in words if word in source_lower)
            if matching / len(words) < 0.5:
                missing.append(snippet[:50] + "...")
        elif snippet.strip().casefold() not in source_lower:
            missing.append(snippet[:50] + "...")

    return missing


def validate_version_supported(
    fixed_version: str | None,
    source_content: str,
    package_name: str | None = None,
) -> bool:
    """Require semver and nearby package-specific security evidence.

    Args:
        fixed_version: Candidate version extracted from the structured result.
        source_content: URL-framed advisory text.
        package_name: Optional concrete target package. When supplied, the
            package name must appear near the security evidence for the version.

    Returns:
        True only when the version is valid and an appropriate source window
        connects it to a security remediation statement.
    """
    if not fixed_version:
        return True
    candidate = str(fixed_version).strip()
    if not _SEMVER_RE.fullmatch(candidate):
        return False
    return _version_has_package_security_evidence(candidate, source_content, package_name)


def validate_python_version_supported(
    fixed_version: str | None,
    source_content: str,
    package_name: str | None = None,
) -> bool:
    """Validate a PyPI version with PEP 440 and the same advisory evidence rule."""
    if not fixed_version:
        return True
    try:
        Version(str(fixed_version).strip())
    except InvalidVersion:
        return False
    return _version_has_package_security_evidence(fixed_version, source_content, package_name)


def _version_has_package_security_evidence(
    fixed_version: str,
    source_content: str,
    package_name: str | None,
) -> bool:
    """Check that a version's nearby security claim refers to the target package."""
    clean_version = str(fixed_version).strip().lstrip("vV")
    version_pattern = re.compile(
        rf"(?<![0-9A-Za-z-]){re.escape(clean_version)}(?![0-9A-Za-z-])",
        re.IGNORECASE,
    )
    package = package_name.strip().strip(chr(96)) if isinstance(package_name, str) else ""
    require_package = bool(package) and package.casefold() not in _PLACEHOLDER_PACKAGE_NAMES
    security_markers = ("fix", "fixed", "patched", "upgrade", "update", "remediat", "security")
    package_pattern = re.compile(re.escape(package), re.IGNORECASE)
    competing_package_version = re.compile(
        r"(?<![A-Za-z0-9@])([A-Za-z0-9@][A-Za-z0-9_./@-]*-[A-Za-z0-9_./@-]*)"
        r"\s+(?:(?:release|version)\s+)?v?\d+\.\d+\.\d+",
        re.IGNORECASE,
    )
    backtick_package = re.compile(
        chr(96) + r"([^" + chr(96) + r"]+)" + chr(96),
        re.IGNORECASE,
    )
    for match in version_pattern.finditer(source_content):
        window_start = max(0, match.start() - 240)
        window = source_content[window_start : match.end() + 240]
        lowered = window.casefold()
        if not any(marker in lowered for marker in security_markers):
            continue
        if require_package:
            package_matches = list(
                package_pattern.finditer(source_content, window_start, match.start())
            )
            if not package_matches:
                continue
            last_package = package_matches[-1]
            region = source_content[last_package.end() : match.end()]
            competing = next(
                (
                    candidate
                    for candidate in competing_package_version.finditer(region)
                    if candidate.group(1).casefold() != package.casefold()
                ),
                None,
            )
            if competing is not None:
                continue
            has_competing_backtick = any(
                candidate.group(1).strip().casefold() != package.casefold()
                and re.search(r"[-/@]", candidate.group(1).strip())
                and not candidate.group(1).strip().upper().startswith(("CVE-", "GHSA-"))
                for candidate in backtick_package.finditer(region)
            )
            if has_competing_backtick:
                continue
        return True
    return False


def validate_case_evidence(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    """Validate strategy-specific evidence against the relevant page set."""
    source = _case_source_content(case)
    replay = case.get("replay", {})
    replay_input = replay.get("input", {}) if isinstance(replay, dict) else {}
    project_language = (
        replay_input.get("project_language", case.get("project_language"))
        if isinstance(replay_input, dict)
        else case.get("project_language")
    )
    violations = validate_serper_schema(result, project_language=str(project_language or ""))
    if result.get("strategy") == "VERSION_BUMP":
        replay_input = case.get("replay", {}).get("input", {})
        project_language = (
            replay_input.get("project_language", case.get("project_language"))
            if isinstance(replay_input, dict)
            else case.get("project_language")
        )
        validator = (
            validate_python_version_supported
            if str(project_language or "").casefold() in {"python", "python3"}
            else validate_version_supported
        )
        if not validator(
            result.get("fixed_version"),
            source,
            package_name=_case_package_name(case),
        ):
            violations.append("fixed_version is not valid or security-supported by the pages")
    if result.get("strategy") == "CODE_WORKAROUND":
        violations.extend(
            f"workaround snippet is not meaningfully supported: {snippet}"
            for snippet in validate_snippet_in_source(result.get("workaround_snippets"), source)
        )
    if result.get("strategy") == "NO_FIX" and (
        result.get("fixed_version") or result.get("workaround_snippets")
    ):
        violations.append("NO_FIX must not contain a fixed version or workaround snippets")
    return violations


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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Correctly extracts strategy and patched version from advisory pages."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")
        result = _result_for_evaluation(case, eval_settings, monkeypatch)
        expected_strat = case.get("expected_strategy")
        expected_ver = case.get("expected_fixed_version")

        # 1. Deterministic schema & value assertions
        replay_input = case.get("replay", {}).get("input", {})
        project_language = (
            replay_input.get("project_language", case.get("project_language"))
            if isinstance(replay_input, dict)
            else case.get("project_language")
        )
        schema_violations = validate_serper_schema(
            result, project_language=str(project_language or "")
        )
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

            package_name, vuln_id, pages = _case_prompt_inputs(case)
            prompt = build_fix_planner_production_prompt(package_name, vuln_id, pages)
            actual_output = json.dumps(_judge_extraction_output(result), indent=2)
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Correctly extracts code workaround snippets from issue discussions."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")

        expected_strat = case.get("expected_strategy")

        if expected_strat != "CODE_WORKAROUND":
            pytest.skip(f"Case '{case_id}' is strategy '{expected_strat}', not CODE_WORKAROUND")
        result = _result_for_evaluation(case, eval_settings, monkeypatch)

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

            package_name, vuln_id, pages = _case_prompt_inputs(case)
            prompt = build_fix_planner_production_prompt(package_name, vuln_id, pages)
            test_case = LLMTestCase(
                name=f"{case_id} [Workaround Extraction Quality]",
                input=prompt,
                actual_output=json.dumps(_judge_extraction_output(result), indent=2),
                expected_output=str(case.get("expected_output", "")),
            )

            workaround_geval = GEval(
                name="Workaround Extraction Quality",
                criteria=(
                    "Evaluate extraction fidelity only: whether each workaround accurately "
                    "captures actionable guidance supported by the supplied page and matches "
                    "the expected extraction. Do not infer that an extracted proposal was "
                    "accepted or validated, and do not judge the proposal's implementation "
                    "success; those are separate remediation outcomes."
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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Extracted fixed version and snippets exist in the source web page content."""
        if not case:
            pytest.skip("No golden fix planner cases available")

        case_id = case.get("case_id", "unknown")
        result = _result_for_evaluation(case, eval_settings, monkeypatch)
        source_content = _case_source_content(case)
        fixed_ver = result.get("fixed_version")
        snippets = result.get("workaround_snippets")

        # 1. Deterministic version substring check
        replay_input = case.get("replay", {}).get("input", {})
        project_language = (
            replay_input.get("project_language", case.get("project_language"))
            if isinstance(replay_input, dict)
            else case.get("project_language")
        )
        version_validator = (
            validate_python_version_supported
            if str(project_language or "").casefold() in {"python", "python3"}
            else validate_version_supported
        )
        version_in_source = version_validator(
            fixed_ver, source_content, package_name=_case_package_name(case)
        )
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

            package_name, vuln_id, pages = _case_prompt_inputs(case)
            prompt = build_fix_planner_production_prompt(package_name, vuln_id, pages)
            faithfulness_context = [source_content]
            if str(project_language or "").casefold() in {"python", "python3"}:
                faithfulness_context.append(
                    "Interpret dotted numeric strings such as 2024.7.4 as PEP 440 package "
                    "versions, not calendar dates."
                )
            if case_id == "python-fix-planner-pyyaml-historical":
                faithfulness_context.append(
                    "The note that 6.0.3 was not the first upstream fixed release only rules "
                    "out an earliest-fix claim; it does not contradict the recorded 6.0.3 "
                    "remediation target and passing task-scoped QA."
                )
            test_case = LLMTestCase(
                name=f"{case_id} [Extraction Faithfulness]",
                input=prompt,
                actual_output=json.dumps(
                    _judge_extraction_output(result)
                    if str(project_language or "").casefold() in {"python", "python3"}
                    else _faithfulness_projection(result),
                    indent=2,
                ),
                context=faithfulness_context,
                retrieval_context=faithfulness_context,
            )

            faithfulness = FaithfulnessMetric(
                threshold=0.85,
                model=eval_settings.judge_model,
                include_reason=True,
                verbose_mode=True,
            )

            assert_test(test_case, [faithfulness])


def test_fix_planner_dataset_has_realistic_structured_pages() -> None:
    """Keep full web captures and compact, explicitly trajectory-backed evidence."""
    assert len(_FP_CASES) == 18
    assert all(case.get("expected_tools") == [] for case in _FP_CASES)
    word_counts: list[int] = []
    for case in _FP_CASES:
        pages = case.get("pages")
        assert isinstance(pages, list) and pages
        assert all(set(page) >= {"url", "content"} for page in pages)
        case_counts = [len(str(page["content"]).split()) for page in pages]
        if case.get("trajectory_evidence_compact"):
            assert case.get("historical_evidence")
            assert min(case_counts) >= 30
        else:
            word_counts.extend(case_counts)
    assert min(word_counts) >= 500
    assert sum(count > 2_000 for count in word_counts) >= 3
    multi_package_cases = [
        case
        for case in _FP_CASES
        if any(
            re.search(r"multi[- ]package|disclosure covers .* and", str(page["content"]), re.I)
            for page in case["pages"]
        )
    ]
    assert len(multi_package_cases) >= 2


def test_fix_planner_case_evidence_is_deterministic() -> None:
    """All offline structured outputs satisfy strategy and source evidence rules."""
    for case in _FP_CASES:
        result = _case_result(case)
        violations = validate_case_evidence(case, result)
        assert not violations, f"Case {case['case_id']} failed evidence checks: {violations}"


def test_fix_planner_faithfulness_projection_omits_null_and_unscored_fields() -> None:
    """Faithfulness judges only extracted versions and actionable snippets."""
    projected = _faithfulness_projection(
        {
            "strategy": "VERSION_BUMP",
            "fixed_version": "1.2.3",
            "workaround_snippets": None,
            "reasoning": "The advisory identifies a patched release.",
        }
    )
    assert projected == {"fixed_version": "1.2.3"}


def test_pyyaml_golden_expected_version_matches_its_page_evidence() -> None:
    """The historical PyYAML page, output, and target must describe one version."""
    case = next(
        case for case in _FP_CASES if case["case_id"] == "python-fix-planner-pyyaml-historical"
    )
    expected_output = json.loads(case["expected_output"])
    expected_version = case["expected_fixed_version"]

    assert expected_output["fixed_version"] == expected_version
    assert _case_result(case)["fixed_version"] == expected_version
    assert expected_version in case["source_page_content"]
    assert expected_version in case["replay"]["input"]["pages"][0]["content"]
    assert all(expected_version in page["content"] for page in case["pages"])


def test_live_fix_planner_case_uses_production_extractor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live case results must come from the production extractor, not golden observations."""
    case = next(case for case in _FP_CASES if case["case_id"] == "python-fix-planner-workaround")
    page_content = case["pages"][0]["content"]
    assert "failed validation" not in page_content.casefold()
    assert page_content == case["source_page_content"]
    assert page_content == case["replay"]["input"]["pages"][0]["content"]
    assert any(
        "did not pass validation" in evidence["observation"].casefold()
        for evidence in case["historical_evidence"]
    )
    production_result = {
        "strategy": "NO_FIX",
        "fixed_version": None,
        "workaround_snippets": [],
    }
    observed_inputs: list[tuple[list[dict[str, str]], str, str]] = []

    def fake_extract(
        pages: list[dict[str, str]],
        package_name: str,
        vuln_id: str,
    ) -> dict[str, Any]:
        observed_inputs.append((pages, package_name, vuln_id))
        return production_result

    monkeypatch.setattr(fix_planner, "_llm_extract_remediation", fake_extract)
    settings = EvalSettings(
        judge_model="test-model",
        is_live=True,
        openai_api_key="test-key",
        trajectory_dir=Path("."),
        golden_dir=_GOLDEN_FILE.parent,
    )
    _LIVE_FP_RESULTS.clear()
    try:
        result = _live_case_result(case, settings, monkeypatch)
    finally:
        _LIVE_FP_RESULTS.clear()

    assert result == production_result
    assert result != _case_result(case)
    assert observed_inputs == [
        (case["pages"], "pyjwt", "CVE-2026-32597"),
    ]
    assert (
        _judge_extraction_output({**production_result, "reasoning": "non-contract explanation"})
        == production_result
    )


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

    package_source = (
        "The target-lib advisory fixes the issue in version 1.2.3. "
        "A neighboring helper-lib release 9.9.9 fixes a different issue."
    )
    assert validate_version_supported("1.2.3", package_source, package_name="target-lib")
    assert not validate_version_supported("9.9.9", package_source, package_name="target-lib")

    python_source = (
        "target-lib security fix in version 2.0rc1. "
        "The helper-lib version 9.9.9 security fix addresses another issue."
    )
    assert validate_python_version_supported("2.0rc1", python_source, package_name="target-lib")
    assert not validate_python_version_supported("9.9.9", python_source, package_name="target-lib")
    assert not validate_python_version_supported(
        "not-a-version", python_source, package_name="target-lib"
    )
