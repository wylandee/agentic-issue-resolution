"""Pydantic schemas for the Remediation Engine demo web application."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FindingModel(BaseModel):
    """Normalized finding representation displayed on the web dashboard."""

    id: str = Field(..., description="Stable unique identifier for this finding.")
    vuln_id: str = Field(..., description="CVE or GHSA or rule identifier.")
    package_name: str = Field(..., description="Affected package component name.")
    package_version: str | None = Field(default=None, description="Installed or affected version.")
    severity: str = Field(default="UNKNOWN", description="Severity label (CRITICAL, HIGH, etc.).")
    file_path: str | None = Field(default=None, description="Repository-relative file path.")
    cwes: list[str] = Field(default_factory=list, description="Associated CWE identifier strings.")
    aliases: list[str] = Field(default_factory=list, description="Linked CVE or GHSA identifiers.")
    description: str | None = Field(
        default=None, description="Vulnerability description or finding message."
    )
    status: Literal["flagged", "remediating", "remediated", "newly_detected", "unresolved"] = Field(
        default="flagged",
        description="Display status of the finding.",
    )
    is_new: bool = Field(default=False, description="True if detected during post-baseline scans.")
    group_id: str | None = Field(default=None, description="Associated triage group ID if mapped.")
    task_id: str | None = Field(default=None, description="Currently active task ID if any.")


class WorkspaceInfo(BaseModel):
    """Metadata and initial findings for an attached repository folder."""

    workspace_id: str
    repo_name: str
    repo_root: str
    odc_report_path: str | None = None
    findings: list[FindingModel] = Field(default_factory=list)
    total_findings: int = 0
    severity_counts: dict[str, int] = Field(default_factory=dict)


class ConsoleEvent(BaseModel):
    """Filtered high-signal event shown in the console log view."""

    id: str
    timestamp: str
    category: Literal["NODE_TRANSITION", "TOOL_CALL", "DETERMINISTIC_FN", "KEY_RESULT"]
    title: str
    details: str | None = None
    status: Literal["running", "success", "error", "incomplete"] | None = None
    duration_seconds: float | None = None
    outcome: str | None = None
    task_label: str | None = None
    data: dict[str, Any] | None = None


class DashboardUpdate(BaseModel):
    """Real-time update payload informing the frontend of finding changes."""

    type: Literal[
        "finding_removed",
        "finding_added",
        "finding_status_change",
        "finding_updated",
        "full_refresh",
    ]
    finding_id: str | None = None
    finding: FindingModel | None = None
    active_findings: list[FindingModel] = Field(default_factory=list)
    remediated_count: int = 0
    new_count: int = 0
    severity_counts: dict[str, int] = Field(default_factory=dict)


class RunState(BaseModel):
    """Current state of an orchestration run."""

    run_id: str
    workspace_id: str
    status: Literal["pending", "running", "completed", "completed_with_errors", "failed"] = "pending"
    current_node: str | None = None
    active_findings: list[FindingModel] = Field(default_factory=list)
    remediated_count: int = 0
    new_count: int = 0
    logs: list[ConsoleEvent] = Field(default_factory=list)
    report_available: bool = False
    report_path: str | None = None
    diff: str | None = None
    errors: list[str] = Field(default_factory=list)


class ReportResponse(BaseModel):
    """Report and patch contents returned upon run completion."""

    report_markdown: str
    report_path: str | None = None
    diff: str = ""
    changed_files: list[str] = Field(default_factory=list)
    status: str = "completed"
