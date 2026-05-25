"""
Semgrep Parser — typed ingestion layer for the Semgrep AppSec Platform API.

Fetches findings from the Semgrep findings endpoint, normalises each raw dict
into a typed ``VulnerabilityIssue`` Pydantic model, and exports both JSONL
(canonical agent input) and CSV (human inspection).

Public API
----------
``setup_session(api_token) -> requests.Session``
``fetch_findings(session, deployment_slug) -> List[Dict]``
``normalize_finding(finding) -> Optional[VulnerabilityIssue]``
``export_to_jsonl(issues, output_path) -> None``
``export_to_csv(issues, output_path) -> None``
``main() -> None``

Environment variables
---------------------
  SEMGREP_API_TOKEN           Required.
  SEMGREP_DEPLOYMENT_SLUG     Required.
  SEMGREP_API_BASE_URL        Default: https://semgrep.dev/api/v1
"""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.contracts import (
    CWEEntry,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityIssue,
)
from src.contracts.schemas import LineRange

log = logging.getLogger(__name__)

API_BASE_URL = os.getenv("SEMGREP_API_BASE_URL", "https://semgrep.dev/api/v1")
FINDINGS_ENDPOINT_TEMPLATE = "/deployments/{deployment_slug}/findings"

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


def setup_session(api_token: str) -> requests.Session:
    """Create an authenticated requests session with retry/backoff support."""
    retry_strategy = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        }
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _extract_findings_page(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract a list of findings from known Semgrep response shapes."""
    if isinstance(payload.get("findings"), list):
        return payload["findings"]
    if isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload.get("data"), list):
        return payload["data"]
    return []


def fetch_findings(
    session: requests.Session,
    deployment_slug: str,
) -> List[Dict[str, Any]]:
    """Fetch all findings page-by-page from the Semgrep API.

    Iterates over ``sast`` and ``sca`` issue types separately, paging each
    until an empty page is returned.
    """
    endpoint = FINDINGS_ENDPOINT_TEMPLATE.format(deployment_slug=deployment_slug)
    url = f"{API_BASE_URL.rstrip('/')}{endpoint}"

    findings: List[Dict[str, Any]] = []

    issue_types = ["sast", "sca"]

    for issue_type in issue_types:
        log.info("--- Starting ingestion for issue type: %s ---", issue_type.upper())
        page = 0

        while True:
            log.info("Fetching Semgrep findings page=%s", page)
            response = session.get(
                url,
                params={"page": page, "issue_type": issue_type},
                timeout=30,
            )
            if response.status_code >= 400:
                response.raise_for_status()

            payload = response.json()
            page_findings = _extract_findings_page(payload)
            if not page_findings:
                break

            for finding in page_findings:
                finding["issue_type"] = issue_type

            findings.extend(page_findings)
            page += 1

    return findings



def normalize_finding(finding: Dict[str, Any]) -> Optional[VulnerabilityIssue]:
    """Normalise one raw Semgrep finding into a typed ``VulnerabilityIssue``.

    Returns ``None`` for non-OPEN findings (fixed, ignored, removed, etc.).
    """
    status = str(finding.get("status", "")).upper()
    if status and status != "OPEN":
        return None

    repository = finding.get("repository") or {}
    location = finding.get("location") or {}
    rule = finding.get("rule") or {}

    # --- CWE & OWASP ---
    cwe_list: List[CWEEntry] = []
    cwe_names = rule.get("cwe_names") or []
    for raw_cwe in cwe_names:
        raw_str = str(raw_cwe).strip()
        if not raw_str:
            continue
        parts = raw_str.split(":", 1)
        cwe_id = parts[0].strip()
        if not cwe_id.startswith("CWE-"):
            digits = "".join(c for c in cwe_id if c.isdigit())
            if digits:
                cwe_id = f"CWE-{digits}"
            else:
                continue
        cwe_name = parts[1].strip() if len(parts) > 1 else None
        try:
            cwe_list.append(CWEEntry(id=cwe_id, name=cwe_name))
        except Exception:
            pass

    owasp_list: List[str] = []
    owasp_names = rule.get("owasp_names") or []
    for raw_owasp in owasp_names:
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

    # --- Repository context ---
    if isinstance(repository, dict):
        repo_url: Optional[str] = repository.get("url") or None
        base_ref: Optional[str] = (
            repository.get("ref") or repository.get("branch") or None
        )
    else:
        repo_url = None
        base_ref = None

    # --- Identifiers ---
    rule_id: Optional[str] = (
        finding.get("rule_name") or rule.get("name") or finding.get("rule_id") or None
    )
    finding_id: Optional[str] = None
    for id_key in ("id", "finding_id", "uuid"):
        if finding.get(id_key) is not None:
            finding_id = str(finding[id_key])
            break

    # --- Severity ---
    severity_raw = str(finding.get("severity", "")).strip().upper()
    try:
        severity = Severity(severity_raw) if severity_raw else Severity.UNKNOWN
    except ValueError:
        severity = Severity.UNKNOWN

    # --- Location ---
    file_path: Optional[str] = (
        location.get("file_path") or finding.get("path") or None
    )
    line_range: Optional[LineRange] = None
    line_start = location.get("line", 0)
    line_end = location.get("end_line", 0)
    try:
        start = int(line_start) if isinstance(line_start, (int, float)) else 0
        end = int(line_end) if isinstance(line_end, (int, float)) else 0
        if start >= 1:
            line_range = LineRange(start=start, end=max(end, start))
    except Exception:
        pass

    # --- Message / URL ---
    message: Optional[str] = (
        finding.get("rule_message") or rule.get("message") or finding.get("message") or None
    )
    finding_url: Optional[str] = (
        finding.get("line_of_code_url") or finding.get("url") or finding.get("finding_url") or None
    )

    # --- Issue type ---
    raw_type = finding.get("issue_type", "sast")
    issue_type = IssueType.SCA if raw_type == "sca" else IssueType.SAST

    # --- SCA-specific fields ---
    package_name: Optional[str] = finding.get("package_name")
    package_version: Optional[str] = finding.get("package_version") or finding.get("found_version")
    fixed_version: Optional[str] = finding.get("fixed_version")
    purl: Optional[str] = finding.get("purl")
    ecosystem: Optional[str] = finding.get("ecosystem")
    cve_id: Optional[str] = finding.get("cve_id")

    # Attempt to extract from sca_info/extra/rule nesting
    sca_info = (
        finding.get("sca_info")
        or rule.get("sca_info")
        or finding.get("extra", {}).get("sca_info")
        or {}
    )

    if isinstance(sca_info, dict):
        if not package_name:
            package_name = sca_info.get("dependency_name") or sca_info.get("package")
        if not package_version:
            package_version = sca_info.get("found_version") or sca_info.get("version")
        if not fixed_version:
            fix_versions = sca_info.get("fix_versions")
            if isinstance(fix_versions, list) and fix_versions:
                fixed_version = str(fix_versions[0])
            elif fix_versions:
                fixed_version = str(fix_versions)
        if not purl:
            purl = sca_info.get("purl")
        if not ecosystem:
            ecosystem = sca_info.get("ecosystem")
        if not cve_id:
            cve_id = sca_info.get("vulnerability_id") or sca_info.get("cve_id")

    # Attempt to extract from dependency_matches list
    dep_matches = finding.get("dependency_matches") or finding.get("extra", {}).get("dependency_matches")
    if isinstance(dep_matches, list) and dep_matches:
        match_zero = dep_matches[0] or {}
        if isinstance(match_zero, dict):
            dep_info = match_zero.get("dependency") or {}
            if isinstance(dep_info, dict):
                pkg_info = dep_info.get("package") or {}
                if isinstance(pkg_info, dict):
                    if not package_name:
                        package_name = pkg_info.get("name")
                    if not ecosystem:
                        ecosystem = pkg_info.get("ecosystem")
                if not package_version:
                    package_version = dep_info.get("version")
            if not fixed_version:
                fix_versions = match_zero.get("fix_versions")
                if isinstance(fix_versions, list) and fix_versions:
                    fixed_version = str(fix_versions[0])

    # Attempt to extract from found_dependency, fix_recommendations, and vulnerability_identifier (Platform API)
    found_dep = finding.get("found_dependency")
    if isinstance(found_dep, dict):
        if not package_name:
            package_name = found_dep.get("package")
        if not package_version:
            package_version = found_dep.get("version")
        if not ecosystem:
            ecosystem = found_dep.get("ecosystem")

    fix_recs = finding.get("fix_recommendations")
    if isinstance(fix_recs, list) and fix_recs:
        rec_zero = fix_recs[0]
        if isinstance(rec_zero, dict):
            if not fixed_version:
                fixed_version = rec_zero.get("version")

    vuln_ident = finding.get("vulnerability_identifier")
    if vuln_ident and not cve_id:
        val_str = str(vuln_ident).strip().upper()
        if re.match(r"^CVE-\d{4}-\d{4,}$", val_str):
            cve_id = val_str

    # Infer ecosystem from file_path if missing
    if not ecosystem and file_path:
        lower_path = file_path.lower()
        if any(k in lower_path for k in ("package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml")):
            ecosystem = "npm"
        elif any(k in lower_path for k in ("requirements.txt", "pipfile", "poetry.lock", "setup.py")):
            ecosystem = "pypi"
        elif any(k in lower_path for k in ("pom.xml", "build.gradle")):
            ecosystem = "maven"
        elif any(k in lower_path for k in ("go.mod", "go.sum")):
            ecosystem = "go"
        elif any(k in lower_path for k in ("cargo.toml", "cargo.lock")):
            ecosystem = "cargo"

    # Construct PURL if missing and ecosystem, name, and version are available
    if not purl and ecosystem and package_name and package_version:
        purl = f"pkg:{ecosystem}/{package_name}@{package_version}"

    # Extract CVE ID from rule name or message if missing
    if not cve_id:
        search_targets = [
            rule_id,
            message,
            finding.get("rule_name"),
            rule.get("name"),
            rule.get("message"),
        ]
        for target in search_targets:
            if target:
                match = re.search(r"(CVE-\d{4}-\d{4,})", str(target), re.IGNORECASE)
                if match:
                    cve_id = match.group(1).upper()
                    break

    return VulnerabilityIssue(
        source=IssueSource.SEMGREP,
        issue_type=issue_type,
        finding_id=finding_id,
        rule_id=rule_id,
        cve_id=cve_id,
        severity=severity,
        repo_url=repo_url,
        base_ref=base_ref,
        cwe=cwe_list,
        owasp=owasp_list,
        file_path=file_path,
        line_range=line_range,
        package_name=package_name,
        package_version=package_version,
        fixed_version=fixed_version,
        purl=purl,
        ecosystem=ecosystem,
        message=message,
        finding_url=finding_url,
        raw_payload=finding,
    )


def export_to_jsonl(issues: List[VulnerabilityIssue], output_path: Path) -> None:
    """Write issues as JSONL — one JSON object per line (canonical agent format)."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for issue in issues:
            f.write(issue.model_dump_json())
            f.write("\n")
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
            repo_name = (
                repository.get("name", "") if isinstance(repository, dict) else ""
            )
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
    """Run the Semgrep API → JSONL + CSV ingestion pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    api_token = os.getenv("SEMGREP_API_TOKEN", "")
    deployment_slug = os.getenv("SEMGREP_DEPLOYMENT_SLUG", "")

    if not api_token:
        raise ValueError("Missing required environment variable: SEMGREP_API_TOKEN")
    if not deployment_slug:
        raise ValueError("Missing required environment variable: SEMGREP_DEPLOYMENT_SLUG")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    output_jsonl = project_root / "data" / "semgrep_issues.jsonl"
    output_csv = project_root / "data" / "semgrep_issues.csv"

    log.info("Starting Semgrep findings ingestion")
    session = setup_session(api_token)

    raw_findings = fetch_findings(session, deployment_slug)
    log.info("Fetched %s total findings", len(raw_findings))

    issues: List[VulnerabilityIssue] = []
    skipped = 0
    for raw in raw_findings:
        issue = normalize_finding(raw)
        if issue is not None:
            issues.append(issue)
        else:
            skipped += 1

    log.info("Parsed %d OPEN issues, skipped %d non-OPEN findings", len(issues), skipped)

    export_to_jsonl(issues, output_jsonl)
    export_to_csv(issues, output_csv)
    log.info("Semgrep ingestion complete.")


if __name__ == "__main__":
    main()
