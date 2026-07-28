"""
Semgrep Parser - typed ingestion layer for local Semgrep JSON output.

Reads findings from ``data/semgrep.json``, normalizes each raw dict into a
typed ``VulnerabilityIssue`` Pydantic model, and exports both JSONL
(canonical agent input) and CSV (human inspection).

Public API
----------
``setup_session(api_token) -> None``
``load_findings_from_json(json_path) -> List[Dict]``
``fetch_findings(session=None, deployment_slug=None, json_path=None) -> List[Dict]``
``normalize_finding(finding) -> Optional[VulnerabilityIssue]``
``export_to_jsonl(issues, output_path) -> None``
``export_to_csv(issues, output_path) -> None``
``main() -> None``
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        """Provide a no-op fallback when python-dotenv is unavailable."""
        return False

from remediation_engine.contracts import (
    CWEEntry,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.contracts.schemas import LineRange

log = logging.getLogger(__name__)

CSV_HEADERS = [
    "Repository",
    "Issue_Type",
    "Rule_ID",
    "Severity",
    "File_Path",
    "Line_Start",
    "Line_End",
    "Message",
    "Finding_URL",
]

_SEMGREP_SEVERITY_MAP: dict[str, Severity] = {
    "CRITICAL": Severity.CRITICAL,
    "ERROR": Severity.HIGH,
    "HIGH": Severity.HIGH,
    "WARNING": Severity.MEDIUM,
    "MEDIUM": Severity.MEDIUM,
    "LOW": Severity.LOW,
    "INFO": Severity.INFO,
    "UNKNOWN": Severity.UNKNOWN,
}


def _default_input_json_path() -> Path:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    return project_root / "data" / "semgrep.json"


def setup_session(api_token: str) -> None:
    """Compatibility helper retained for callers from the old API-based flow."""
    del api_token
    return None


def _extract_findings_page(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a list of findings from known Semgrep response shapes."""
    if isinstance(payload.get("findings"), list):
        return payload["findings"]
    if isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    return []


def load_findings_from_json(json_path: Path | str) -> List[Dict[str, Any]]:
    """Load findings from a local Semgrep JSON file."""
    path = Path(json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return _extract_findings_page(payload)
    return []


def fetch_findings(
    session: Any = None,
    deployment_slug: Optional[str] = None,
    json_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Compatibility wrapper that now loads findings from local JSON."""
    del session, deployment_slug
    input_path = Path(json_path) if json_path is not None else _default_input_json_path()
    return load_findings_from_json(input_path)


def _parse_cwe_entries(values: Any) -> List[CWEEntry]:
    cwe_list: List[CWEEntry] = []
    for raw_cwe in values or []:
        raw_str = str(raw_cwe).strip()
        if not raw_str:
            continue
        parts = raw_str.split(":", 1)
        cwe_id = parts[0].strip()
        if not cwe_id.startswith("CWE-"):
            digits = "".join(ch for ch in cwe_id if ch.isdigit())
            if not digits:
                continue
            cwe_id = f"CWE-{digits}"
        cwe_name = parts[1].strip() if len(parts) > 1 else None
        try:
            cwe_list.append(CWEEntry(id=cwe_id, name=cwe_name))
        except Exception:
            continue
    return cwe_list


def _parse_owasp_entries(values: Any) -> List[str]:
    owasp_list: List[str] = []
    for raw_owasp in values or []:
        raw_str = str(raw_owasp).strip()
        if not raw_str:
            continue
        match = re.match(r"^(A\d{1,2}:\d{4})", raw_str)
        if match:
            owasp_list.append(match.group(1))
        else:
            token = re.split(r"[\s\-]+", raw_str)[0].strip()
            if token:
                owasp_list.append(token)
    return owasp_list


def _coerce_severity(value: Any) -> Severity:
    raw = str(value or "").strip().upper()
    return _SEMGREP_SEVERITY_MAP.get(raw, Severity.UNKNOWN)


def _build_line_range(
    start_line: Any,
    end_line: Any,
) -> Optional[LineRange]:
    try:
        start = int(start_line) if start_line is not None else 0
        end = int(end_line) if end_line is not None else 0
        if start >= 1:
            return LineRange(start=start, end=max(end, start))
    except Exception:
        return None
    return None


def normalize_finding(finding: Dict[str, Any]) -> Optional[VulnerabilityIssue]:
    """Normalise one raw Semgrep finding into a typed ``VulnerabilityIssue``."""
    status = str(finding.get("status", "")).upper()
    if status and status != "OPEN":
        return None

    extra = finding.get("extra") if isinstance(finding.get("extra"), dict) else {}
    if extra.get("is_ignored") is True:
        return None

    repository = finding.get("repository") if isinstance(finding.get("repository"), dict) else {}
    location = finding.get("location") if isinstance(finding.get("location"), dict) else {}
    rule = finding.get("rule") if isinstance(finding.get("rule"), dict) else {}
    metadata = extra.get("metadata") if isinstance(extra.get("metadata"), dict) else {}
    sca_info = extra.get("sca_info") if isinstance(extra.get("sca_info"), dict) else None

    cwe_list = _parse_cwe_entries(metadata.get("cwe") or rule.get("cwe_names"))
    owasp_list = _parse_owasp_entries(metadata.get("owasp") or rule.get("owasp_names"))

    repo_url: Optional[str] = repository.get("url") or None
    base_ref: Optional[str] = repository.get("ref") or repository.get("branch") or None

    rule_id: Optional[str] = (
        finding.get("check_id")
        or finding.get("rule_name")
        or rule.get("name")
        or finding.get("rule_id")
        or None
    )

    finding_id: Optional[str] = None
    for candidate in (
        finding.get("id"),
        extra.get("fingerprint"),
        finding.get("finding_id"),
        finding.get("uuid"),
    ):
        if candidate is not None:
            finding_id = str(candidate)
            break

    severity = _coerce_severity(extra.get("severity") or finding.get("severity"))
    confidence = metadata.get("confidence") or finding.get("confidence")

    start = finding.get("start") if isinstance(finding.get("start"), dict) else {}
    end = finding.get("end") if isinstance(finding.get("end"), dict) else {}
    file_path: Optional[str] = (
        finding.get("path")
        or location.get("file_path")
        or finding.get("file_path")
        or None
    )
    line_range = _build_line_range(
        start.get("line") or location.get("line"),
        end.get("line") or location.get("end_line"),
    )

    message: Optional[str] = (
        extra.get("message")
        or finding.get("rule_message")
        or rule.get("message")
        or finding.get("message")
        or None
    )
    finding_url: Optional[str] = (
        metadata.get("semgrep.url")
        or finding.get("line_of_code_url")
        or finding.get("url")
        or finding.get("finding_url")
        or None
    )

    issue_type = IssueType.SCA if sca_info is not None else IssueType.SAST

    package_name: Optional[str] = finding.get("package_name")
    package_version: Optional[str] = finding.get("package_version") or finding.get("found_version")
    fixed_version: Optional[str] = finding.get("fixed_version")
    purl: Optional[str] = finding.get("purl")
    ecosystem: Optional[str] = finding.get("ecosystem")
    cve_id: Optional[str] = finding.get("cve_id")

    if sca_info is not None:
        dependency_match = sca_info.get("dependency_match") if isinstance(sca_info.get("dependency_match"), dict) else {}
        found_dependency = dependency_match.get("found_dependency") if isinstance(dependency_match.get("found_dependency"), dict) else {}
        dependency_pattern = dependency_match.get("dependency_pattern") if isinstance(dependency_match.get("dependency_pattern"), dict) else {}

        if not package_name:
            package_name = found_dependency.get("package") or dependency_pattern.get("package")
        if not package_version:
            package_version = found_dependency.get("version")
        if not ecosystem:
            ecosystem = found_dependency.get("ecosystem") or dependency_pattern.get("ecosystem")
        if not fixed_version:
            explicit_fix = (
                sca_info.get("fixed_version")
                or extra.get("fixed_version")
                or finding.get("fixed_version")
            )
            if explicit_fix:
                fixed_version = str(explicit_fix)
        if not cve_id:
            explicit_cve = (
                sca_info.get("cve_id")
                or sca_info.get("vulnerability_id")
                or extra.get("cve_id")
                or finding.get("vulnerability_identifier")
            )
            if explicit_cve and re.match(r"^CVE-\d{4}-\d{4,}$", str(explicit_cve).strip(), re.IGNORECASE):
                cve_id = str(explicit_cve).strip().upper()

    if not purl and ecosystem and package_name and package_version:
        purl = f"pkg:{ecosystem}/{package_name}@{package_version}"

    dataflow_trace = extra.get("dataflow_trace") if issue_type == IssueType.SAST else None

    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        finding_id=finding_id,
        rule_id=rule_id,
        cve_id=cve_id,
        cwe=cwe_list,
        owasp=owasp_list,
        severity=severity,
        confidence=str(confidence) if confidence is not None else None,
        repo_url=repo_url,
        base_ref=base_ref,
        file_path=file_path,
        line_range=line_range,
        package_name=package_name,
        package_version=package_version,
        fixed_version=fixed_version,
        purl=purl,
        ecosystem=ecosystem,
        message=message,
        finding_url=finding_url,
        dataflow_trace=dataflow_trace if isinstance(dataflow_trace, dict) else None,
        raw_payload=finding,
    )


def export_to_jsonl(issues: List[VulnerabilityIssue], output_path: Path) -> None:
    """Write issues as JSONL - one JSON object per line."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(issue.model_dump_json())
            handle.write("\n")
    log.info("Wrote %d issues to %s", len(issues), output_path)


def export_to_csv(issues: List[VulnerabilityIssue], output_path: Path) -> None:
    """Export a flat CSV projection of typed issues for human inspection."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for issue in issues:
            lr = issue.line_range
            repository = (issue.raw_payload or {}).get("repository") or {}
            repo_name = repository.get("name", "") if isinstance(repository, dict) else ""
            writer.writerow(
                {
                    "Repository": repo_name,
                    "Issue_Type": issue.issue_type.value,
                    "Rule_ID": issue.rule_id or "",
                    "Severity": issue.severity.value,
                    "File_Path": issue.file_path or "",
                    "Line_Start": lr.start if lr else 0,
                    "Line_End": lr.end if lr else 0,
                    "Message": issue.message or "",
                    "Finding_URL": issue.finding_url or "",
                }
            )
    log.info("Wrote CSV to %s", output_path)


def main() -> None:
    """Run the local Semgrep JSON -> JSONL + CSV ingestion pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    input_json = _default_input_json_path()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_jsonl = project_root / "data" / "semgrep_issues.jsonl"
    output_csv = project_root / "data" / "semgrep_issues.csv"

    log.info("Starting Semgrep findings ingestion from %s", input_json)
    raw_findings = load_findings_from_json(input_json)
    log.info("Loaded %s total findings", len(raw_findings))

    issues: List[VulnerabilityIssue] = []
    skipped = 0
    for raw in raw_findings:
        issue = normalize_finding(raw)
        if issue is not None:
            issues.append(issue)
        else:
            skipped += 1

    log.info("Parsed %d issues, skipped %d findings", len(issues), skipped)
    export_to_jsonl(issues, output_jsonl)
    export_to_csv(issues, output_csv)
    log.info("Semgrep ingestion complete.")


if __name__ == "__main__":
    main()


