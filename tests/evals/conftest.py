"""Shared pytest configuration, fixtures, and TrajectoryLoader for DeepEval evaluations."""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.evals.adapters import (
    TrajectoryDocument,
    TrajectorySpan,
    parse_trajectory_markdown,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TRAJECTORY_DIR = _PROJECT_ROOT / "data" / "trajectories"
_DEFAULT_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Load .env values if present in repository root
_DOTENV_PATH = _PROJECT_ROOT / ".env"
_DOTENV_VARS: dict[str, str] = {}
if _DOTENV_PATH.exists():
    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(_DOTENV_PATH, override=False)
        _DOTENV_VARS = {
            str(k): str(v) for k, v in dotenv_values(_DOTENV_PATH).items() if v is not None
        }
    except Exception:
        _DOTENV_VARS = {}

# Preserve API key and judge model at import time before root conftest isolation strips them
_INITIAL_OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY", "").strip() or _DOTENV_VARS.get("OPENAI_API_KEY", "").strip()
)
_INITIAL_JUDGE_MODEL = (
    os.environ.get("EVAL_JUDGE_MODEL", "").strip()
    or _DOTENV_VARS.get("EVAL_JUDGE_MODEL", "").strip()
    or _DOTENV_VARS.get("REMEDY_LLM_MODEL", "").strip()
    or "gpt-4o"
)


# ---------------------------------------------------------------------------
# Command-Line Option & Marker Registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register evaluation-specific CLI flags."""
    parser.addoption(
        "--run-eval-live",
        action="store_true",
        default=False,
        help="Run DeepEval metrics with live LLM judge (requires OPENAI_API_KEY).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the eval marker if not already present."""
    config.addinivalue_line(
        "markers",
        "eval: DeepEval LLM evaluation tests (requires OPENAI_API_KEY when running live)",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist all DeepEval evaluation test results to the SQLite evaluation database."""
    try:
        from deepeval.test_run import global_test_run_manager

        from remediation_engine.evals.db import EvalDatabase
        from remediation_engine.evals.models import (
            EvalRunRecord,
            EvalTestCaseRecord,
            MetricRecord,
        )
    except ImportError:
        return

    test_run = global_test_run_manager.get_test_run()
    if not test_run or not getattr(test_run, "test_cases", None):
        return

    run_id = (
        getattr(test_run, "identifier", None)
        or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    )
    now_iso = datetime.datetime.now().isoformat()
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or _INITIAL_JUDGE_MODEL or "gpt-4o"
    is_live = bool(session.config.getoption("--run-eval-live", default=False))

    test_case_records: list[EvalTestCaseRecord] = []
    for tc in test_run.test_cases:
        name_lower = (tc.name or "").lower()
        if "report" in name_lower:
            suite = "report"
        elif "triage" in name_lower:
            suite = "triage"
        elif "fix" in name_lower or "planner" in name_lower:
            suite = "fix_planner"
        elif "critic" in name_lower or "qa" in name_lower:
            suite = "qa_critic"
        elif "subagent" in name_lower or "update" in name_lower or "workaround" in name_lower:
            suite = "subagent"
        elif "budget" in name_lower or "sla" in name_lower or "business" in name_lower:
            suite = "business_rules"
        else:
            suite = "general"

        metrics: list[MetricRecord] = []
        for m in getattr(tc, "metrics_data", None) or []:
            metrics.append(
                MetricRecord(
                    metric_name=m.name,
                    score=float(m.score) if m.score is not None else 0.0,
                    threshold=float(m.threshold) if m.threshold is not None else 0.70,
                    success=bool(m.success),
                    reason=getattr(m, "reason", None),
                    evaluation_model=getattr(m, "evaluation_model", None) or judge_model,
                    verbose_logs=getattr(m, "verbose_logs", None),
                )
            )

        status = "PASSED" if getattr(tc, "success", True) else "FAILED"

        raw_context = getattr(tc, "context", None)
        if isinstance(raw_context, list):
            context_str = "\n---\n".join(str(c) for c in raw_context)
        else:
            context_str = str(raw_context) if raw_context else None

        raw_retrieval = getattr(tc, "retrieval_context", None)
        if isinstance(raw_retrieval, list):
            retrieval_str = "\n---\n".join(str(r) for r in raw_retrieval)
        else:
            retrieval_str = str(raw_retrieval) if raw_retrieval else None

        test_case_records.append(
            EvalTestCaseRecord(
                case_id=getattr(tc, "case_id", None) or tc.name,
                test_name=tc.name or "unnamed_test",
                suite=suite,
                status=status,
                input_text=tc.input or "",
                actual_output=tc.actual_output or "",
                expected_output=getattr(tc, "expected_output", None),
                context_text=context_str,
                retrieval_context=retrieval_str,
                latency_seconds=float(getattr(tc, "run_duration", 0.0) or 0.0),
                cost=float(getattr(tc, "evaluation_cost", 0.0) or 0.0),
                additional_metadata=getattr(tc, "additional_metadata", {}) or {},
                metrics=metrics,
            )
        )

    if not test_case_records:
        return

    total_tests = len(test_case_records)
    passed_tests = sum(1 for r in test_case_records if r.status == "PASSED")
    failed_tests = total_tests - passed_tests
    duration = float(getattr(test_run, "run_duration", 0.0) or 0.0)
    cost = float(getattr(test_run, "evaluation_cost", 0.0) or 0.0)

    run_record = EvalRunRecord(
        run_id=run_id,
        timestamp=now_iso,
        suite_name=getattr(test_run, "test_file", None) or "tests/evals",
        judge_model=judge_model,
        is_live=is_live,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        skipped_tests=0,
        duration_seconds=duration,
        total_cost=cost,
        metadata={"git_branch": "feat/add-subagent-evals"},
        test_cases=test_case_records,
    )

    try:
        db = EvalDatabase()
        db.save_run(run_record)
        logger.info("Persisted %d evaluation test cases to SQLite at %s", total_tests, db.db_path)
    except Exception as exc:
        logger.warning("Failed to auto-persist evaluation run to SQLite: %s", exc)


# ---------------------------------------------------------------------------
# Evaluation Settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSettings:
    """Configuration settings for DeepEval evaluation runs."""

    judge_model: str
    is_live: bool
    openai_api_key: str
    trajectory_dir: Path
    golden_dir: Path


@pytest.fixture(autouse=True)
def preserve_eval_credentials(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore OPENAI_API_KEY and judge settings when running live evals despite root conftest isolation."""
    is_live = bool(request.config.getoption("--run-eval-live", default=False))
    if is_live:
        key = os.environ.get("OPENAI_API_KEY", "").strip() or _INITIAL_OPENAI_API_KEY
        if key:
            monkeypatch.setenv("OPENAI_API_KEY", key)
        judge = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or _INITIAL_JUDGE_MODEL
        if judge:
            monkeypatch.setenv("EVAL_JUDGE_MODEL", judge)
        monkeypatch.setenv("EVAL_RUN_LIVE", "true")


@pytest.fixture
def eval_settings(request: pytest.FixtureRequest) -> EvalSettings:
    """Provide configuration settings for DeepEval evaluation runs.

    Reads ``EVAL_JUDGE_MODEL`` (defaults to .env value or 'gpt-4o') and respects ``--run-eval-live``.
    """
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or _INITIAL_JUDGE_MODEL or "gpt-4o"
    is_live = bool(request.config.getoption("--run-eval-live", default=False))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or _INITIAL_OPENAI_API_KEY

    trajectory_dir_env = os.environ.get("REMEDIATION_TRAJECTORY_DIR", "").strip()
    trajectory_dir = Path(trajectory_dir_env) if trajectory_dir_env else _DEFAULT_TRAJECTORY_DIR

    return EvalSettings(
        judge_model=judge_model,
        is_live=is_live,
        openai_api_key=api_key,
        trajectory_dir=trajectory_dir,
        golden_dir=_DEFAULT_GOLDEN_DIR,
    )


# ---------------------------------------------------------------------------
# Trajectory Loader
# ---------------------------------------------------------------------------


class TrajectoryLoader:
    """Cached loader and query interface for Phase 5 trajectory markdown files."""

    def __init__(self, trajectory_dir: Path | None = None) -> None:
        """Initialize the loader with a target trajectory directory."""
        self.trajectory_dir = trajectory_dir or _DEFAULT_TRAJECTORY_DIR
        self._cache: dict[Path, TrajectoryDocument] = {}
        self._indexed_by_trace: dict[str, TrajectoryDocument] = {}

    def get_trajectory_paths(self) -> list[Path]:
        """Return all trajectory markdown files discovered in the trajectory directory."""
        if not self.trajectory_dir.exists():
            return []
        return sorted(self.trajectory_dir.glob("*.md"))

    def load_by_path(self, path: Path | str) -> TrajectoryDocument:
        """Load and cache a trajectory file from path."""
        p = Path(path).resolve()
        if p in self._cache:
            return self._cache[p]
        doc = parse_trajectory_markdown(p)
        self._cache[p] = doc
        if doc.trace_id:
            self._indexed_by_trace[doc.trace_id] = doc
        return doc

    def load_by_trace_id(self, trace_id: str) -> TrajectoryDocument | None:
        """Load a trajectory matching a specific trace ID."""
        if trace_id in self._indexed_by_trace:
            return self._indexed_by_trace[trace_id]
        for path in self.get_trajectory_paths():
            if trace_id in path.name:
                return self.load_by_path(path)
        # Search all
        for path in self.get_trajectory_paths():
            doc = self.load_by_path(path)
            if doc.trace_id == trace_id:
                return doc
        return None

    def load_all(self) -> list[TrajectoryDocument]:
        """Load all discovered trajectory files."""
        return [self.load_by_path(p) for p in self.get_trajectory_paths()]

    def get_sample_trajectories(self, count: int = 5) -> list[TrajectoryDocument]:
        """Return a small sample of parsed trajectory documents."""
        paths = self.get_trajectory_paths()[:count]
        return [self.load_by_path(p) for p in paths]

    def find_agent_spans(self, agent_name: str) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
        """Find all (doc, span) pairs across all trajectories for a given agent."""
        results: list[tuple[TrajectoryDocument, TrajectorySpan]] = []
        for path in self.get_trajectory_paths():
            doc = self.load_by_path(path)
            matched_spans = doc.spans_for_agent(agent_name)
            for s in matched_spans:
                results.append((doc, s))
        return results


@pytest.fixture(scope="session")
def trajectory_loader() -> TrajectoryLoader:
    """Session-scoped TrajectoryLoader instance."""
    return TrajectoryLoader()


@pytest.fixture
def sample_trajectory_docs(trajectory_loader: TrajectoryLoader) -> list[TrajectoryDocument]:
    """Provide a sample of parsed trajectory documents."""
    return trajectory_loader.get_sample_trajectories(count=5)


@pytest.fixture
def triage_trajectories(
    trajectory_loader: TrajectoryLoader,
) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
    """Provide all discovered triage agent (doc, span) pairs."""
    return trajectory_loader.find_agent_spans("triage")


@pytest.fixture
def update_subagent_trajectories(
    trajectory_loader: TrajectoryLoader,
) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
    """Provide all discovered update subagent (doc, span) pairs."""
    return trajectory_loader.find_agent_spans("update_subagent")


@pytest.fixture
def workaround_subagent_trajectories(
    trajectory_loader: TrajectoryLoader,
) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
    """Provide all discovered workaround subagent (doc, span) pairs."""
    return trajectory_loader.find_agent_spans("workaround_subagent")


@pytest.fixture
def qa_critic_trajectories(
    trajectory_loader: TrajectoryLoader,
) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
    """Provide all discovered QA critic (doc, span) pairs."""
    return trajectory_loader.find_agent_spans("qa_critic")


@pytest.fixture
def report_trajectories(
    trajectory_loader: TrajectoryLoader,
) -> list[tuple[TrajectoryDocument, TrajectorySpan]]:
    """Provide all discovered report narrative (doc, span) pairs."""
    return trajectory_loader.find_agent_spans("report")


# ---------------------------------------------------------------------------
# Golden Dataset Helper
# ---------------------------------------------------------------------------


@pytest.fixture
def load_golden_cases() -> Callable[[str], list[dict[str, Any]]]:
    """Fixture returning a callable to load curated evaluation cases from golden JSON files."""

    def _loader(dataset_name: str) -> list[dict[str, Any]]:
        filename = dataset_name if dataset_name.endswith(".json") else f"{dataset_name}.json"
        path = _DEFAULT_GOLDEN_DIR / filename
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "cases" in data:
            return data["cases"]
        return []

    return _loader


@pytest.fixture
def report_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated report evaluation cases from golden/report_cases.json."""
    cases = load_golden_cases("report_cases")
    if not cases:
        pytest.skip("No golden report cases found in tests/evals/golden/report_cases.json")
    return cases


@pytest.fixture
def triage_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated triage evaluation cases from golden/triage_cases.json."""
    all_cases = load_golden_cases("triage_cases")
    cases = [c for c in all_cases if c.get("eval_type", "triage") == "triage"]
    if not cases:
        pytest.skip("No golden triage cases found in tests/evals/golden/triage_cases.json")
    return cases


@pytest.fixture
def fix_planner_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated fix planner evaluation cases from golden/triage_cases.json."""
    all_cases = load_golden_cases("triage_cases")
    cases = [c for c in all_cases if c.get("eval_type") == "fix_planner"]
    if not cases:
        pytest.skip("No golden fix planner cases found in tests/evals/golden/triage_cases.json")
    return cases


@pytest.fixture
def subagent_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated subagent evaluation cases from golden/subagent_cases.json."""
    cases = load_golden_cases("subagent_cases")
    if not cases:
        pytest.skip("No golden subagent cases found in tests/evals/golden/subagent_cases.json")
    return cases


@pytest.fixture
def update_subagent_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated update subagent evaluation cases from golden/subagent_cases.json."""
    all_cases = load_golden_cases("subagent_cases")
    cases = [c for c in all_cases if c.get("eval_type") == "update_subagent"]
    if not cases:
        pytest.skip(
            "No golden update subagent cases found in tests/evals/golden/subagent_cases.json"
        )
    return cases


@pytest.fixture
def workaround_subagent_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated workaround subagent evaluation cases from golden/subagent_cases.json."""
    all_cases = load_golden_cases("subagent_cases")
    cases = [c for c in all_cases if c.get("eval_type") == "workaround_subagent"]
    if not cases:
        pytest.skip(
            "No golden workaround subagent cases found in tests/evals/golden/subagent_cases.json"
        )
    return cases


@pytest.fixture
def qa_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated QA critic evaluation cases from golden/qa_cases.json."""
    cases = load_golden_cases("qa_cases")
    if not cases:
        pytest.skip("No golden QA cases found in tests/evals/golden/qa_cases.json")
    return cases
