"""Small typed API around ingestion, triage, and the Phase 5 graph."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .contracts.schemas import SystemContext, VulnerabilityGroup, VulnerabilityIssue
from .orchestration.graph import run_orchestrator
from .orchestration.task_utils import terminal_outcome_issues
from .settings import AppSettings
from .triage.pipeline import run_triage_pipeline


class RemediationRequest(BaseModel):
    """Input required to run remediation against a repository."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path
    valid_groups: list[VulnerabilityGroup] = Field(default_factory=list)
    issues: list[VulnerabilityIssue] = Field(default_factory=list)
    system_context: SystemContext | None = None

    @field_validator("repo_root")
    @classmethod
    def validate_repo_root(cls, value: Path) -> Path:
        """Require an existing absolute repository directory at the API boundary."""
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("repo_root must be an absolute path")
        resolved = path.resolve()
        if not resolved.is_dir():
            raise ValueError(f"repo_root must be an existing directory: {value}")
        return resolved


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
    settings: AppSettings | None = None,
) -> list[VulnerabilityGroup]:
    """Return actionable vulnerability groups for a scanner finding list."""
    resolved_settings = settings or AppSettings.from_env()
    context = system_context or SystemContext(
        public_facing=True,
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs",
    )
    results = run_triage_pipeline(
        issues,
        context,
        str(repo_root) if repo_root else None,
        settings=resolved_settings,
    )
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
    resolved_settings = settings or AppSettings.from_env()
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
        "settings": resolved_settings,
    }
    state = run_orchestrator(**orchestrator_kwargs)
    errors = list(state.get("errors", []) or [])
    status = state.get("status", "failed")
    outcome_issues = terminal_outcome_issues(state)
    for issue in outcome_issues:
        message = f"remediation outcome: {issue}"
        if message not in errors:
            errors.append(message)
    if (errors or outcome_issues) and status == "completed":
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
