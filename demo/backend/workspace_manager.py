"""Workspace and ODC report management for the demo application."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import IssueSource, Severity, VulnerabilityIssue
from remediation_engine.orchestration.issue_identity import stable_issue_fingerprint
from remediation_engine.tools.odc_parser import load_report, parse_vulnerabilities

from .models import FindingModel, WorkspaceInfo

logger = logging.getLogger(__name__)

DEMO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = DEMO_ROOT / ".workspaces"
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory registry for workspace sessions
_WORKSPACES: dict[str, dict[str, Any]] = {}


def _generate_finding_id(issue: VulnerabilityIssue) -> str:
    """Generate a clean, stable deterministic identifier for a finding."""
    identifier = issue.cve_id or issue.ghsa_id or issue.rule_id or str(issue.id)
    pkg = (issue.package_name or "unknown").strip().lower()
    ver = (issue.package_version or "").strip()
    path = (issue.file_path or "").strip().lower()
    seed = f"{identifier}:{pkg}:{ver}:{path}"
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    clean_id = f"{identifier}_{pkg}_{h}"
    return clean_id.replace("/", "_").replace("\\", "_")


def issue_to_finding_model(issue: VulnerabilityIssue, status: str = "flagged", is_new: bool = False) -> FindingModel:
    """Convert a contracts VulnerabilityIssue to an API FindingModel."""
    vuln_id = issue.cve_id or issue.ghsa_id or issue.rule_id or str(issue.id)
    cwes = [c.id for c in issue.cwe if c.id]
    severity_str = str(getattr(issue.severity, "value", issue.severity)).upper()
    
    return FindingModel(
        id=_generate_finding_id(issue),
        vuln_id=vuln_id,
        package_name=issue.package_name or "Unknown package",
        package_version=issue.package_version,
        severity=severity_str,
        file_path=issue.file_path,
        cwes=cwes,
        description=issue.message,
        status=status,
        is_new=is_new,
    )


def find_odc_report(repo_root: Path) -> Path | None:
    """Locate an ODC JSON report or issues JSONL inside the repository, prioritizing reports with vulnerabilities."""
    patterns = [
        "*dependency-check*report*.json",
        "*dependency-check*.json",
        "*odc*report*.json",
        "*odc*.json",
        "*baseline*issues*.jsonl",
        "*.jsonl",
    ]

    candidates: list[tuple[int, Path]] = []

    def evaluate_candidate(p: Path) -> int | None:
        """Return estimated finding count if file is a valid ODC report, else None."""
        if p.suffix.lower() in {".jsonl", ".ndjson"}:
            try:
                lines = [line for line in p.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
                return len(lines)
            except Exception:
                return None
        elif p.suffix.lower() == ".json":
            try:
                content = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(content, dict) and "dependencies" in content:
                    deps = content.get("dependencies") or []
                    vuln_count = sum(
                        len(d.get("vulnerabilities", [])) for d in deps if isinstance(d, dict) and "vulnerabilities" in d
                    )
                    return vuln_count
            except Exception:
                return None
        return None

    for pattern in patterns:
        for match in sorted(repo_root.rglob(pattern)):
            if match.is_file():
                rel_parts = match.relative_to(repo_root).parts
                if any(part in {".git", "node_modules", ".venv", "dist", "build"} for part in rel_parts):
                    continue
                score = evaluate_candidate(match)
                if score is not None:
                    candidates.append((score, match))

    if not candidates:
        # General JSON scan looking for ODC schema
        for json_file in sorted(repo_root.rglob("*.json")):
            rel_parts = json_file.relative_to(repo_root).parts
            if any(part in {".git", "node_modules", ".venv", "dist", "build"} for part in rel_parts):
                continue
            score = evaluate_candidate(json_file)
            if score is not None:
                candidates.append((score, json_file))

    if not candidates:
        return None

    # Pick candidate with most findings, or first valid candidate if all 0
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_path = candidates[0]
    logger.info("Selected ODC report: %s (estimated findings: %d)", best_path, best_score)
    return best_path


def parse_findings_from_file(report_path: Path) -> list[VulnerabilityIssue]:
    """Parse vulnerability issues from an ODC JSON report or JSONL file."""
    if report_path.suffix.lower() in {".jsonl", ".ndjson"}:
        text = report_path.read_text(encoding="utf-8")
        issues: list[VulnerabilityIssue] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                issues.append(VulnerabilityIssue.model_validate_json(line))
            except Exception as exc:
                logger.warning("Failed to parse JSONL line in %s: %s", report_path, exc)
        return issues

    # ODC JSON report
    report_dict = load_report(report_path)
    return parse_vulnerabilities(report_dict)


def calculate_severity_counts(findings: list[FindingModel]) -> dict[str, int]:
    """Calculate counts per severity level."""
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0, "UNKNOWN": 0}
    for f in findings:
        sev = f.severity.upper()
        if sev in counts:
            counts[sev] += 1
        else:
            counts["UNKNOWN"] += 1
    return counts


def create_workspace_from_local_path(local_path_str: str) -> WorkspaceInfo:
    """Register an existing local directory on the host machine as a workspace."""
    repo_path = Path(local_path_str).expanduser().resolve()
    if not repo_path.is_dir():
        raise ValueError(f"Path does not exist or is not a directory: {local_path_str}")

    odc_path = find_odc_report(repo_path)
    if not odc_path:
        raise ValueError(
            f"No OWASP Dependency-Check report found in '{repo_path.name}'. "
            "Please ensure the folder contains an ODC report JSON or baseline issues JSONL."
        )

    issues = parse_findings_from_file(odc_path)
    findings = [issue_to_finding_model(issue, status="flagged") for issue in issues]
    workspace_id = f"ws_{uuid.uuid4().hex[:10]}"

    info = WorkspaceInfo(
        workspace_id=workspace_id,
        repo_name=repo_path.name,
        repo_root=str(repo_path),
        odc_report_path=str(odc_path.relative_to(repo_path)),
        findings=findings,
        total_findings=len(findings),
        severity_counts=calculate_severity_counts(findings),
    )

    _WORKSPACES[workspace_id] = {
        "info": info,
        "repo_root": repo_path,
        "odc_path": odc_path,
        "issues": issues,
    }
    return info


def create_workspace_from_upload(root_folder_name: str) -> tuple[str, Path]:
    """Allocate directory for incoming multipart upload."""
    workspace_id = f"ws_{uuid.uuid4().hex[:10]}"
    ws_dir = WORKSPACES_DIR / workspace_id / root_folder_name
    ws_dir.mkdir(parents=True, exist_ok=True)
    return workspace_id, ws_dir


def finalize_uploaded_workspace(workspace_id: str, repo_path: Path) -> WorkspaceInfo:
    """Analyze uploaded directory, parse findings, and store workspace session."""
    odc_path = find_odc_report(repo_path)
    if not odc_path:
        # Fall back to root directory
        raise ValueError(
            f"No OWASP Dependency-Check report found in uploaded folder '{repo_path.name}'. "
            "Please ensure the attached folder contains an ODC report JSON file."
        )

    issues = parse_findings_from_file(odc_path)
    findings = [issue_to_finding_model(issue, status="flagged") for issue in issues]

    info = WorkspaceInfo(
        workspace_id=workspace_id,
        repo_name=repo_path.name,
        repo_root=str(repo_path),
        odc_report_path=str(odc_path.relative_to(repo_path)),
        findings=findings,
        total_findings=len(findings),
        severity_counts=calculate_severity_counts(findings),
    )

    _WORKSPACES[workspace_id] = {
        "info": info,
        "repo_root": repo_path,
        "odc_path": odc_path,
        "issues": issues,
    }
    return info


def get_workspace_data(workspace_id: str) -> dict[str, Any] | None:
    """Retrieve internal workspace context dict."""
    return _WORKSPACES.get(workspace_id)
