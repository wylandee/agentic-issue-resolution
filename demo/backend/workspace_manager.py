"""Workspace and ODC report management for the demo application."""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

from remediation_engine.contracts.schemas import VulnerabilityIssue
from remediation_engine.orchestration.issue_identity import (
    canonical_issue_components,
    finding_occurrence_key,
)
from remediation_engine.tools.odc_parser import load_report, parse_vulnerabilities

from .models import FindingModel, WorkspaceInfo

logger = logging.getLogger(__name__)

DEMO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACES_DIR = DEMO_ROOT / ".workspaces"
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)

# In-memory registry for workspace sessions
_WORKSPACES: dict[str, dict[str, Any]] = {}


_SEVERITY_RANK = {
    "UNKNOWN": 0,
    "INFO": 1,
    "LOW": 2,
    "MEDIUM": 3,
    "HIGH": 4,
    "CRITICAL": 5,
}


def _generate_finding_id(issue: VulnerabilityIssue, identifier: str | None = None) -> str:
    """Generate a stable identifier for one vulnerability package occurrence."""
    vuln_id = identifier or issue.cve_id or issue.ghsa_id or issue.rule_id or str(issue.id)
    package_name, package_version, package_path = finding_occurrence_key(issue)
    seed = json.dumps(
        [vuln_id, package_name, package_version, package_path],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    clean_id = f"{vuln_id}_{package_name}_{digest}"
    return clean_id.replace("/", "_").replace("\\", "_")


def _finding_from_linked_issues(
    issues: list[VulnerabilityIssue],
    *,
    status: str,
    is_new: bool,
) -> FindingModel:
    """Project one linked issue component into a dashboard finding."""
    cve_ids = sorted({issue.cve_id.strip().upper() for issue in issues if issue.cve_id})
    ghsa_ids = sorted({issue.ghsa_id.strip().upper() for issue in issues if issue.ghsa_id})
    primary_id = (
        cve_ids[0]
        if cve_ids
        else ghsa_ids[0]
        if ghsa_ids
        else (issues[0].rule_id or str(issues[0].id))
    )
    aliases = sorted((set(cve_ids) | set(ghsa_ids)) - {primary_id})

    representative = next(
        (issue for issue in issues if issue.cve_id and issue.cve_id.strip().upper() == primary_id),
        None,
    )
    if representative is None:
        representative = next(
            (
                issue
                for issue in issues
                if issue.ghsa_id and issue.ghsa_id.strip().upper() == primary_id
            ),
            issues[0],
        )

    severities = [str(getattr(issue.severity, "value", issue.severity)).upper() for issue in issues]
    severity = max(severities, key=lambda value: _SEVERITY_RANK.get(value, 0), default="UNKNOWN")
    cwes = sorted({cwe.id for issue in issues for cwe in issue.cwe if cwe.id})
    description = next(
        (
            issue.message
            for issue in [representative, *issues]
            if issue.message and issue.message.strip()
        ),
        None,
    )

    return FindingModel(
        id=_generate_finding_id(representative, primary_id),
        vuln_id=primary_id,
        package_name=(representative.package_name or "").strip() or "Unknown package",
        package_version=(
            representative.package_version.strip() if representative.package_version else None
        ),
        severity=severity,
        file_path=representative.file_path,
        aliases=aliases,
        cwes=cwes,
        description=description,
        status=status,  # type: ignore[arg-type]
        is_new=is_new,
    )


def normalize_findings(
    issues: list[VulnerabilityIssue],
    *,
    status: str = "flagged",
    is_new: bool = False,
) -> list[FindingModel]:
    """Canonicalize scanner records into one row per linked vulnerability occurrence.

    Records are linked by shared CVE/GHSA identifiers, including transitive
    links through an issue that carries both identifiers. Identifier links are
    scoped to the same package, installed version, and repository location.
    The input issues are only read and are never modified.

    Args:
        issues: Original scanner issue records.
        status: Dashboard status assigned to each canonical row.
        is_new: Whether the rows were discovered after the baseline scan.

    Returns:
        Canonical findings in the order of their first source record.
    """
    return [
        _finding_from_linked_issues(group, status=status, is_new=is_new)
        for group in canonical_issue_components(issues)
    ]


def issue_to_finding_model(
    issue: VulnerabilityIssue,
    status: str = "flagged",
    is_new: bool = False,
) -> FindingModel:
    """Convert one contracts issue to its canonical API finding representation."""
    return normalize_findings([issue], status=status, is_new=is_new)[0]


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
                lines = [
                    line
                    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.strip()
                ]
                return len(lines)
            except Exception:
                return None
        elif p.suffix.lower() == ".json":
            try:
                content = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                if isinstance(content, dict) and "dependencies" in content:
                    deps = content.get("dependencies") or []
                    vuln_count = sum(
                        len(d.get("vulnerabilities", []))
                        for d in deps
                        if isinstance(d, dict) and "vulnerabilities" in d
                    )
                    return vuln_count
            except Exception:
                return None
        return None

    for pattern in patterns:
        for match in sorted(repo_root.rglob(pattern)):
            if match.is_file():
                rel_parts = match.relative_to(repo_root).parts
                if any(
                    part in {".git", "node_modules", ".venv", "dist", "build"} for part in rel_parts
                ):
                    continue
                score = evaluate_candidate(match)
                if score is not None:
                    candidates.append((score, match))

    if not candidates:
        # General JSON scan looking for ODC schema
        for json_file in sorted(repo_root.rglob("*.json")):
            rel_parts = json_file.relative_to(repo_root).parts
            if any(
                part in {".git", "node_modules", ".venv", "dist", "build"} for part in rel_parts
            ):
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
    findings = normalize_findings(issues, status="flagged")
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
    findings = normalize_findings(issues, status="flagged")

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
