"""Typed public API for ingesting findings and running Phase 5 remediation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts.schemas import SystemContext, VulnerabilityGroup, VulnerabilityIssue
from .orchestration.graph import run_orchestrator
from .orchestration.state import normalize_target_packages, validate_target_package_scope
from .orchestration.task_utils import terminal_outcome_issues
from .settings import AppSettings
from .triage.pipeline import run_triage_pipeline


class RemediationRequest(BaseModel):
    """Request for a remediation run against an existing repository directory.

    ``repo_root`` is normalized to an absolute path during validation. The
    orchestrator runs against an isolated workspace; callers supply either
    pre-grouped findings or raw typed issues for the graph's triage node.
    ``target_packages`` is an explicit development-only allowlist; leaving it
    empty preserves the normal full-repository behavior.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path
    valid_groups: list[VulnerabilityGroup] = Field(default_factory=list)
    issues: list[VulnerabilityIssue] = Field(default_factory=list)
    system_context: SystemContext | None = None
    target_packages: list[str] = Field(
        default_factory=list,
        description=(
            "Optional development-only package allowlist. Empty means the full "
            "repository is in scope."
        ),
    )

    @field_validator("repo_root")
    @classmethod
    def validate_repo_root(cls, value: Path) -> Path:
        """Validate and resolve an existing absolute repository directory."""
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("repo_root must be an absolute path")
        resolved = path.resolve()
        if not resolved.is_dir():
            raise ValueError(f"repo_root must be an existing directory: {value}")
        return resolved

    @field_validator("target_packages")
    @classmethod
    def normalize_target_package_names(cls, value: list[str]) -> list[str]:
        """Normalize and de-duplicate development package scope names."""
        return normalize_target_packages(value)

    @model_validator(mode="after")
    def validate_development_package_scope(self) -> RemediationRequest:
        """Reject package scoping unless the request explicitly targets dev."""
        validate_target_package_scope(self.target_packages, self.system_context)
        return self


class RemediationResult(BaseModel):
    """Typed status, patch, and diagnostic projection of one remediation run.

    ``diff`` and ``changed_files`` describe the proposed patch produced during
    teardown. ``raw_state`` is retained internally for callers that need the
    complete graph projection and is excluded from serialized public output.
    """

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
    """Return valid vulnerability groups produced from typed scanner findings.

    Args:
        issues: Scanner findings to normalize and group.
        repo_root: Optional repository path used for path normalization.
        system_context: Optional deployment context for triage.
        settings: Optional settings dependency for embedding callers.

    Returns:
        Groups whose triage verdict is actionable.
    """
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
    """Run remediation in an isolated workspace and return its typed result.

    The supplied host repository is never modified. The graph owns task
    creation, committed attempt selection, worker/QA routing, final full-scan
    gating, and Docker workspace cleanup. The result contains status, changed
    paths, a unified diff, and any accumulated errors.

    Args:
        request: Validated repository path, typed findings or groups, and an
            optional development-only package scope.
        settings: Optional settings dependency for CLI and embedding callers.

    Returns:
        A typed status and patch projection. ``raw_state`` is available only
        through the in-memory model and is excluded from serialization.
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
        "target_packages": request.target_packages,
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
