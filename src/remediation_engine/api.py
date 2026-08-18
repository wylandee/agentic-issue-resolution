"""Small typed API around ingestion, triage, and the Phase 5 graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts.schemas import SystemContext, VulnerabilityGroup, VulnerabilityIssue
from .orchestration.graph import run_orchestrator
from .settings import AppSettings
from .triage.pipeline import run_triage_pipeline


class RemediationRequest(BaseModel):
    """Input required to run remediation against a repository."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path
    valid_groups: list[VulnerabilityGroup] = Field(default_factory=list)
    issues: list[VulnerabilityIssue] = Field(default_factory=list)
    system_context: SystemContext | None = None


class RemediationResult(BaseModel):
    """Stable result projection returned by the public API."""

    status: str
    changed_files: list[str] = Field(default_factory=list)
    diff: str = ""
    errors: list[str] = Field(default_factory=list)
    trajectory_path: str | None = None
    report_path: str | None = None
    raw_state: dict[str, Any] = Field(default_factory=dict, exclude=True)


def triage_issues(
    issues: list[VulnerabilityIssue],
    *,
    repo_root: Path | None = None,
    system_context: SystemContext | None = None,
) -> list[VulnerabilityGroup]:
    """Return actionable vulnerability groups for a scanner finding list."""
    context = system_context or SystemContext(
        public_facing=True,
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs",
    )
    results = run_triage_pipeline(issues, context, str(repo_root) if repo_root else None)
    return [group for group, verdict in results if verdict.is_valid]


def run_remediation(
    request: RemediationRequest,
    *,
    settings: AppSettings | None = None,
) -> RemediationResult:
    """Run the current task-queue workflow and return a typed patch result.

    The host repository is not edited. ``settings`` is accepted as an explicit
    dependency-injection boundary for CLI and embedding applications.
    """
    # Initial triage belongs to the graph's ``initial_triage`` node.  Passing
    # an empty group list is intentional: it tells the graph to triage the
    # supplied issue set exactly once instead of performing a hidden
    # preprocessing pass here and then reporting ``triage_skipped``.
    groups = list(request.valid_groups)
    orchestrator_kwargs: dict[str, Any] = {
        "repo_root": str(request.repo_root),
        "valid_groups": groups,
        "issues": request.issues,
        "system_context": request.system_context,
    }
    if settings is not None:
        orchestrator_kwargs["settings"] = settings
    state = run_orchestrator(**orchestrator_kwargs)
    errors = list(state.get("errors", []) or [])
    status = state.get("status", "failed")
    if errors and status == "completed":
        status = "completed_with_errors"
    return RemediationResult(
        status=status,
        changed_files=list(state.get("changed_files", [])),
        diff=state.get("diff", ""),
        errors=errors,
        trajectory_path=state.get("trajectory_path"),
        report_path=state.get("report_path"),
        raw_state=dict(state),
    )
