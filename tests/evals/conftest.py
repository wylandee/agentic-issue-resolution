"""Shared pytest configuration, fixtures, and TrajectoryLoader for DeepEval evaluations."""

from __future__ import annotations

import json
import logging
import os
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


@pytest.fixture
def eval_settings(request: pytest.FixtureRequest) -> EvalSettings:
    """Provide configuration settings for DeepEval evaluation runs.

    Reads ``EVAL_JUDGE_MODEL`` (defaults to 'gpt-4o') and respects ``--run-eval-live``.
    """
    judge_model = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o").strip() or "gpt-4o"
    is_live = bool(request.config.getoption("--run-eval-live", default=False))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

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
