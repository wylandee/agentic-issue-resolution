from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """
    Workspace-local tmp path fixture.

    Some environments deny access to pytest's default Windows temp root, so we
    create per-test temp directories inside the writable repository workspace.
    """
    base_dir = Path(__file__).resolve().parents[1] / ".pytest-tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest-", dir=base_dir))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolate_external_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off external tracing and service credentials.

    Individual tests may opt into a specific setting with ``monkeypatch``.
    The default must be deterministic even when a developer's shell exports
    LangSmith, LLM, registry, or repository credentials.
    """
    for name in (
        "OPENAI_API_KEY",
        "SERPER_API_KEY",
        "GITHUB_TOKEN",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_ENDPOINT",
        "TRIAGE_CACHE_DIR",
        "REMEDIATION_TRAJECTORY_DIR",
        "REMEDIATION_REPORT_DIR",
        "REMEDY_LLM_MODEL",
        "TRIAGE_LLM_MODEL",
        "SUPERVISOR_LLM_MODEL",
        "UPDATE_LLM_MODEL",
        "WORKAROUND_LLM_MODEL",
        "QA_LLM_MODEL",
        "REPORT_LLM_ENABLED",
        "REPORT_LLM_MODEL",
        "REMEDY_BYPASS_WORKAROUND_SUBAGENT",
        "ODC_EXTRA_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setenv("TRIAGE_LLM_ENABLED", "false")
    monkeypatch.setenv("REMEDY_DISABLE_POST_QA_TRIAGE", "false")
    monkeypatch.setenv("REMEDY_DISABLE_RETRIAGE", "false")
