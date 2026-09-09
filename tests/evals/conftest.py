"""Shared pytest configuration, fixtures, and TrajectoryLoader for DeepEval evaluations."""

from __future__ import annotations

import datetime
import json
import logging
import os
import re
import subprocess
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
_EVAL_TEST_ROOT = _PROJECT_ROOT / "tests" / "evals"
_EVAL_TEST_FILE_NAMES = {
    "test_business_rules.py",
    "test_fix_planner_eval.py",
    "test_qa_critic_eval.py",
    "test_report_eval.py",
    "test_triage_eval.py",
    "test_update_subagent_eval.py",
    "test_workaround_subagent_eval.py",
}

# Pytest reports include every item that ran, while DeepEval only registers
# cases passed through ``assert_test``. Keep the former separately so the
# session-finish hook can make pytest's item set authoritative for the UI.
_EVAL_NODE_IDS: set[str] = set()
_EVAL_TEST_REPORTS: dict[str, dict[str, pytest.TestReport]] = {}

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
    parser.addoption(
        "--eval-tag",
        action="store",
        default=None,
        help="Tag the persisted evaluation run with a friendly identifier.",
    )
    parser.addoption(
        "--eval-baseline",
        action="store",
        default=None,
        help="Compare the run with a previous run ID, tag, or the reserved 'latest' reference.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the eval marker if not already present."""
    config.addinivalue_line(
        "markers",
        "eval: DeepEval LLM evaluation tests (requires OPENAI_API_KEY when running live)",
    )


def _normalise_nodeid(nodeid: str) -> str:
    """Return a case-insensitive, slash-normalized pytest node ID."""
    return nodeid.replace("\\", "/").casefold()


def _is_eval_nodeid(nodeid: str) -> bool:
    """Return whether a node ID belongs to a dedicated evaluation file."""
    normalized = _normalise_nodeid(nodeid)
    return any(f"tests/evals/{filename}" in normalized for filename in _EVAL_TEST_FILE_NAMES)


def _is_eval_item(item: pytest.Item) -> bool:
    """Return whether a collected pytest item should appear in the eval dashboard."""
    if item.get_closest_marker("eval") is not None:
        return True

    item_path = getattr(item, "path", None) or getattr(item, "fspath", None)
    if item_path is not None:
        try:
            resolved_path = Path(str(item_path)).resolve()
            if resolved_path.is_relative_to(_EVAL_TEST_ROOT):
                return resolved_path.name in _EVAL_TEST_FILE_NAMES
        except (OSError, ValueError):
            pass

    return _is_eval_nodeid(str(getattr(item, "nodeid", "")))


def pytest_sessionstart(session: pytest.Session) -> None:
    """Reset the in-memory pytest observations at the start of each session."""
    del session
    _EVAL_NODE_IDS.clear()
    _EVAL_TEST_REPORTS.clear()


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Record the selected eval item order for session-level persistence."""
    del session, config
    _EVAL_NODE_IDS.clear()
    _EVAL_NODE_IDS.update(str(item.nodeid) for item in items if _is_eval_item(item))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Capture setup, call, and teardown outcomes for selected eval items."""
    nodeid = str(report.nodeid)
    if nodeid not in _EVAL_NODE_IDS and not _is_eval_nodeid(nodeid):
        return
    _EVAL_TEST_REPORTS.setdefault(nodeid, {})[str(report.when)] = report


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a possibly absent metric value to a float fallback."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _nodeid_parameter_id(nodeid: str) -> str | None:
    """Extract a pytest parameter ID from a node ID when one is present."""
    item_name = nodeid.rsplit("::", 1)[-1]
    opening = item_name.rfind("[")
    if opening >= 0 and item_name.endswith("]"):
        return item_name[opening + 1 : -1]
    return None


def _case_id_from_test_case(tc: Any) -> str | None:
    """Extract a stable case ID from DeepEval metadata or its display name."""
    metadata = getattr(tc, "additional_metadata", None)
    if isinstance(metadata, dict) and metadata.get("case_id"):
        return str(metadata["case_id"])

    name = str(getattr(tc, "name", "") or "")
    if name:
        return name.split(" [", 1)[0]
    return None


def _case_id_from_nodeid(nodeid: str) -> str:
    """Extract a dashboard case ID from a pytest node ID."""
    return _nodeid_parameter_id(nodeid) or nodeid.rsplit("::", 1)[-1]


def _suite_from_nodeid(nodeid: str) -> str:
    """Derive the dashboard suite name from a pytest node ID."""
    normalized = _normalise_nodeid(nodeid)
    suite_by_file = (
        ("test_workaround_subagent_eval.py", "subagent"),
        ("test_update_subagent_eval.py", "subagent"),
        ("test_qa_critic_eval.py", "qa_critic"),
        ("test_report_eval.py", "report"),
        ("test_triage_eval.py", "triage"),
        ("test_fix_planner_eval.py", "fix_planner"),
        ("test_business_rules.py", "business_rules"),
    )
    for filename, suite in suite_by_file:
        if filename in normalized:
            return suite
    return "general"


def _suite_from_test_case(tc: Any) -> str:
    """Derive a fallback suite name from a DeepEval test-case display name."""
    name = str(getattr(tc, "name", "") or "").casefold()
    if "qa critic" in name or "critic" in name:
        return "qa_critic"
    if "triage" in name:
        return "triage"
    if "fix extraction" in name or "workaround extraction" in name or "planner" in name:
        return "fix_planner"
    if "report" in name or "narrative" in name:
        return "report"
    if "subagent" in name or "update" in name or "workaround" in name:
        return "subagent"
    if "budget" in name or "sla" in name:
        return "business_rules"
    return "general"


def _pytest_item_observation(nodeid: str) -> dict[str, Any]:
    """Summarize all pytest phases recorded for one eval item."""
    reports = _EVAL_TEST_REPORTS.get(nodeid, {})
    ordered_reports = [reports[name] for name in ("setup", "call", "teardown") if name in reports]
    outcomes = [str(getattr(report, "outcome", "")) for report in ordered_reports]

    if "failed" in outcomes:
        status = "FAILED"
    elif "skipped" in outcomes:
        status = "SKIPPED"
    elif outcomes:
        status = "PASSED"
    else:
        status = "SKIPPED"

    details = [
        str(getattr(report, "longrepr", ""))
        for report in ordered_reports
        if getattr(report, "longrepr", None) is not None
        and getattr(report, "outcome", "") in {"failed", "skipped"}
    ]
    error_message = "\n\n".join(detail for detail in details if detail) or None
    duration = sum(_safe_float(getattr(report, "duration", 0.0)) for report in ordered_reports)

    return {
        "status": status,
        "error_message": error_message,
        "latency_seconds": duration,
        "phase_outcomes": {
            name: str(getattr(report, "outcome", "")) for name, report in reports.items()
        },
        "has_reports": bool(reports),
    }


def _test_case_metadata(tc: Any) -> dict[str, Any]:
    """Return a mutable metadata copy from a DeepEval test case."""
    metadata = getattr(tc, "additional_metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else {}


def _metric_names(tc: Any) -> list[str]:
    """Return metric names recorded on a DeepEval test case."""
    return [
        str(getattr(metric, "name", ""))
        for metric in (getattr(tc, "metrics_data", None) or [])
        if getattr(metric, "name", None)
    ]


def _deep_eval_node_score(tc: Any, nodeid: str) -> int:
    """Score how likely a pytest node is the source of a DeepEval test case."""
    normalized_nodeid = _normalise_nodeid(nodeid)
    score = 0
    case_id = _case_id_from_test_case(tc)
    parameter_id = _nodeid_parameter_id(nodeid)
    if case_id and parameter_id:
        if case_id.casefold() == parameter_id.casefold():
            score += 10_000
    elif case_id and case_id.casefold() in normalized_nodeid:
        score += 1_000

    test_case_name = str(getattr(tc, "name", "") or "")
    for token in re.findall(r"[a-z0-9]+", test_case_name.casefold()):
        if len(token) >= 4 and token in normalized_nodeid:
            score += min(len(token), 20)

    metric_hints = {
        "tool correctness": "tool_correctness",
        "task completion": "task_completion",
        "hallucination": "narrative_no_hallucination",
        "finding coverage": "narrative_covers_key_findings",
        "constraint adherence": "narrative_respects_negative_constraints",
        "fix extraction": "version_extraction_from_advisory",
        "workaround extraction quality": "workaround_extraction_from_issues",
        "extraction faithfulness": "no_hallucinated_versions",
    }
    for metric_name in _metric_names(tc):
        metric_lower = metric_name.casefold()
        for label, hint in metric_hints.items():
            if label in metric_lower and hint in normalized_nodeid:
                score += 500

    return score


def _match_deep_eval_case_to_nodeid(
    tc: Any,
    nodeids: list[str],
    assigned_nodeids: set[str],
) -> str | None:
    """Match one DeepEval case to an unassigned pytest item."""
    candidates = [
        nodeid
        for nodeid in nodeids
        if nodeid not in assigned_nodeids
        and _pytest_item_observation(nodeid)["status"] != "SKIPPED"
    ]
    if not candidates:
        return None

    scored = [(_deep_eval_node_score(tc, nodeid), nodeid) for nodeid in candidates]
    best_score = max(score for score, _ in scored)
    if best_score <= 0:
        return candidates[0] if len(candidates) == 1 else None
    return next(nodeid for score, nodeid in scored if score == best_score)


def _record_from_deep_eval_case(
    tc: Any,
    nodeid: str | None,
    judge_model: str,
) -> Any:
    """Convert a DeepEval case to a dashboard record and apply pytest status."""
    from remediation_engine.evals.models import EvalTestCaseRecord, MetricRecord

    metadata = _test_case_metadata(tc)
    observation = (
        _pytest_item_observation(nodeid)
        if nodeid
        else {
            "status": "PASSED" if getattr(tc, "success", True) else "FAILED",
            "error_message": None,
            "latency_seconds": 0.0,
        }
    )
    metrics = [
        MetricRecord(
            metric_name=str(metric.name),
            score=_safe_float(getattr(metric, "score", 0.0)),
            threshold=_safe_float(getattr(metric, "threshold", 0.70), 0.70),
            success=bool(getattr(metric, "success", False)),
            reason=getattr(metric, "reason", None),
            evaluation_model=getattr(metric, "evaluation_model", None) or judge_model,
            verbose_logs=getattr(metric, "verbose_logs", None),
        )
        for metric in (getattr(tc, "metrics_data", None) or [])
    ]

    raw_context = getattr(tc, "context", None)
    context_text = (
        "\n---\n".join(str(value) for value in raw_context)
        if isinstance(raw_context, list)
        else str(raw_context)
        if raw_context
        else None
    )
    raw_retrieval = getattr(tc, "retrieval_context", None)
    retrieval_context = (
        "\n---\n".join(str(value) for value in raw_retrieval)
        if isinstance(raw_retrieval, list)
        else str(raw_retrieval)
        if raw_retrieval
        else None
    )

    if nodeid:
        metadata.setdefault("pytest_nodeid", nodeid)
        metadata["pytest_status"] = observation["status"]

    return EvalTestCaseRecord(
        case_id=metadata.get("case_id") or (_case_id_from_nodeid(nodeid) if nodeid else None),
        test_name=nodeid or str(getattr(tc, "name", "unnamed_test")),
        suite=_suite_from_nodeid(nodeid) if nodeid else _suite_from_test_case(tc),
        status=observation["status"],
        input_text=str(getattr(tc, "input", "") or ""),
        actual_output=str(getattr(tc, "actual_output", "") or ""),
        expected_output=getattr(tc, "expected_output", None),
        context_text=context_text,
        retrieval_context=retrieval_context,
        latency_seconds=_safe_float(getattr(tc, "run_duration", 0.0))
        or observation["latency_seconds"],
        cost=_safe_float(getattr(tc, "evaluation_cost", 0.0)),
        error_message=observation["error_message"],
        additional_metadata=metadata,
        metrics=metrics,
    )


def _record_from_pytest_item(nodeid: str) -> Any:
    """Create a dashboard record for an eval item without a DeepEval case."""
    from remediation_engine.evals.models import EvalTestCaseRecord

    observation = _pytest_item_observation(nodeid)
    metadata = {
        "pytest_nodeid": nodeid,
        "pytest_status": observation["status"],
        "pytest_phase_outcomes": observation["phase_outcomes"],
        "pytest_deepeval_case": False,
    }
    if not observation["has_reports"]:
        metadata["pytest_not_run"] = True
        error_message = (
            observation["error_message"] or "Collected eval item produced no pytest report."
        )
    else:
        error_message = observation["error_message"]

    return EvalTestCaseRecord(
        case_id=_case_id_from_nodeid(nodeid),
        test_name=nodeid,
        suite=_suite_from_nodeid(nodeid),
        status=observation["status"],
        latency_seconds=observation["latency_seconds"],
        error_message=error_message,
        additional_metadata=metadata,
    )


def _load_deep_eval_test_cases() -> tuple[Any | None, list[Any]]:
    """Return the current DeepEval run and its registered cases if available."""
    try:
        from deepeval.test_run import global_test_run_manager
    except ImportError:
        return None, []

    try:
        test_run = global_test_run_manager.get_test_run()
    except Exception:
        return None, []
    if not test_run:
        return None, []
    return test_run, list(getattr(test_run, "test_cases", None) or [])


def _build_eval_test_case_records(
    session: pytest.Session,
    deep_eval_cases: list[Any],
    judge_model: str,
) -> list[Any]:
    """Build exactly one dashboard record for every selected eval pytest item."""
    nodeids = [str(item.nodeid) for item in getattr(session, "items", []) if _is_eval_item(item)]
    assigned_nodeids: set[str] = set()
    records_by_nodeid: dict[str, Any] = {}

    for tc in deep_eval_cases:
        nodeid = _match_deep_eval_case_to_nodeid(tc, nodeids, assigned_nodeids)
        if nodeid is None:
            logger.warning(
                "Could not associate DeepEval case %r with a pytest eval item; "
                "the item itself will still be persisted.",
                getattr(tc, "name", None),
            )
            continue
        records_by_nodeid[nodeid] = _record_from_deep_eval_case(tc, nodeid, judge_model)
        assigned_nodeids.add(nodeid)

    for nodeid in nodeids:
        records_by_nodeid.setdefault(nodeid, _record_from_pytest_item(nodeid))

    return [records_by_nodeid[nodeid] for nodeid in nodeids]


def _git_command_output(arguments: list[str]) -> str | None:
    """Run one bounded Git metadata command without invoking a shell."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Git metadata command failed: %s", exc)
        return None
    return completed.stdout.strip()


def _git_metadata() -> dict[str, Any]:
    """Collect branch, commit, and dirty-worktree metadata for an eval run."""
    branch = _git_command_output(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git_command_output(["rev-parse", "HEAD"])
    status = _git_command_output(["status", "--porcelain", "--untracked-files=all"])
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _write_session_message(session: pytest.Session, message: str) -> None:
    """Write a session-level message through pytest's terminal reporter."""
    terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminal_reporter is not None:
        terminal_reporter.write_line(message)
    else:
        print(message)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Persist one SQLite record for every selected eval pytest item."""
    del exitstatus
    from remediation_engine.evals.db import EvalDatabase
    from remediation_engine.evals.models import EvalRunRecord

    test_run, deep_eval_cases = _load_deep_eval_test_cases()
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "").strip() or _INITIAL_JUDGE_MODEL or "gpt-4o"
    is_live = bool(session.config.getoption("--run-eval-live", default=False))
    test_case_records = _build_eval_test_case_records(session, deep_eval_cases, judge_model)
    if not test_case_records:
        return

    total_tests = len(test_case_records)
    passed_tests = sum(1 for record in test_case_records if record.status == "PASSED")
    failed_tests = sum(1 for record in test_case_records if record.status == "FAILED")
    skipped_tests = sum(1 for record in test_case_records if record.status == "SKIPPED")
    deep_eval_duration = _safe_float(getattr(test_run, "run_duration", 0.0)) if test_run else 0.0
    deep_eval_cost = _safe_float(getattr(test_run, "evaluation_cost", 0.0)) if test_run else 0.0
    duration = deep_eval_duration or sum(record.latency_seconds for record in test_case_records)
    cost = deep_eval_cost or sum(record.cost for record in test_case_records)
    run_id = (
        getattr(test_run, "identifier", None) if test_run else None
    ) or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    suite_name = (getattr(test_run, "test_file", None) if test_run else None) or "tests/evals"
    eval_tag = session.config.getoption("--eval-tag", default=None)
    eval_tag = eval_tag.strip() if isinstance(eval_tag, str) else None
    eval_tag = eval_tag or None
    baseline_reference = session.config.getoption("--eval-baseline", default=None)
    baseline_reference = baseline_reference.strip() if isinstance(baseline_reference, str) else None
    baseline_reference = baseline_reference or None

    run_record = EvalRunRecord(
        run_id=run_id,
        timestamp=datetime.datetime.now().isoformat(),
        tag=eval_tag,
        suite_name=suite_name,
        judge_model=judge_model,
        is_live=is_live,
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        skipped_tests=skipped_tests,
        duration_seconds=duration,
        total_cost=cost,
        metadata={
            **_git_metadata(),
            "pytest_items_recorded": total_tests,
            "deep_eval_cases_recorded": len(deep_eval_cases),
        },
        test_cases=test_case_records,
    )

    db: EvalDatabase | None = None
    baseline_run: dict[str, Any] | None = None
    baseline_error: Exception | None = None
    try:
        db = EvalDatabase()
        if baseline_reference:
            try:
                baseline_run = db.resolve_run_reference(
                    baseline_reference,
                    suite_name=suite_name,
                )
            except Exception as exc:
                baseline_error = exc
        db.save_run(run_record)
    except Exception as exc:
        logger.warning("Failed to auto-persist evaluation run to SQLite: %s", exc)
        return

    logger.info("Persisted %d pytest eval items to SQLite at %s", total_tests, db.db_path)

    if not baseline_reference:
        return
    if baseline_error is not None:
        _write_session_message(
            session,
            f"WARNING: Could not resolve eval baseline {baseline_reference!r}: {baseline_error}",
        )
        return
    if baseline_run is None:
        _write_session_message(
            session,
            f"WARNING: Eval baseline {baseline_reference!r} was not found; current run was saved.",
        )
        return

    try:
        from remediation_engine.evals.comparison import format_run_comparison

        comparison = db.get_run_comparison(baseline_run["run_id"], run_record.run_id)
        _write_session_message(session, format_run_comparison(comparison))
    except Exception as exc:
        _write_session_message(
            session,
            f"WARNING: Could not compare eval baseline {baseline_reference!r}: {exc}",
        )


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
    """Provide curated fix planner cases from golden/fix_planner_cases.json."""
    all_cases = load_golden_cases("fix_planner_cases")
    cases = [c for c in all_cases if c.get("eval_type") == "fix_planner"]
    if not cases:
        pytest.skip(
            "No golden fix planner cases found in tests/evals/golden/fix_planner_cases.json"
        )
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
    """Provide curated update cases from golden/update_subagent_cases.json."""
    cases = load_golden_cases("update_subagent_cases")
    if not cases:
        pytest.skip(
            "No golden update subagent cases found in tests/evals/golden/update_subagent_cases.json"
        )
    return cases


@pytest.fixture
def workaround_subagent_golden_cases(
    load_golden_cases: Callable[[str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Provide curated workaround cases from the dedicated golden dataset."""
    cases = load_golden_cases("workaround_subagent_cases")
    if not cases:
        pytest.skip(
            "No golden workaround subagent cases found in tests/evals/golden/workaround_subagent_cases.json"
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
