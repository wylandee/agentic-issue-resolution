import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
        return payload.get("findings", [])
    if isinstance(payload.get("results"), list):
        return payload.get("results", [])
    if isinstance(payload.get("data"), list):
        return payload.get("data", [])
    return []


def fetch_findings(session: requests.Session, deployment_slug: str) -> List[Dict[str, Any]]:
    """Fetch all findings page-by-page from the Semgrep API."""
    endpoint = FINDINGS_ENDPOINT_TEMPLATE.format(deployment_slug=deployment_slug)
    url = f"{API_BASE_URL.rstrip('/')}{endpoint}"

    findings: List[Dict[str, Any]] = []

    issue_types = ["sast", "sca"]
    
    for issue_type in issue_types:
        logging.info("--- Starting ingestion for issue type: %s ---", issue_type.upper())

        page = 0

        while True:
            logging.info("Fetching Semgrep findings page=%s", page)
            response = session.get(url, params={"page": page, "issue_type": issue_type}, timeout=30)
            if response.status_code >= 400:
                response.raise_for_status()

            payload = response.json()
            page_findings = _extract_findings_page(payload)
            if not page_findings:
                break

            findings.extend(page_findings)
            page += 1

    return findings


def infer_issue_type(finding: Dict[str, Any]) -> str:
    """Classify issue as sast/sca with conservative defaults."""
    lower_haystack = " ".join(
        [
            str(finding.get("rule_name", "")),
            str((finding.get("rule") or {}).get("name", "")),
            str((finding.get("rule") or {}).get("category", "")),
            " ".join(map(str, finding.get("categories", []) or [])),
        ]
    ).lower()

    dependency_keys = {"dependency", "package", "vulnerable library", "supply chain", "sca"}
    if any(token in lower_haystack for token in dependency_keys):
        return "sca"

    if any(key in finding for key in ("dependency_matches", "reachability", "lockfile_path", "package_name")):
        return "sca"

    return "sast"


def normalize_finding(finding: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one finding into the flat CSV schema; return None for non-OPEN findings."""
    status = str(finding.get("status", "")).upper()
    if status != "OPEN": # Filters out provisionally ignored findings
        return None

    repository = finding.get("repository") or {}
    location = finding.get("location") or {}
    rule = finding.get("rule") or {}

    line_start = location.get("line", 0)
    line_end = location.get("end_line", 0)

    return {
        "Repository": repository.get("name", "") if isinstance(repository, dict) else finding.get("repository_name", ""),
        "Issue_Type": infer_issue_type(finding),
        "Rule_ID": finding.get("rule_name", "") or rule.get("name", "") or finding.get("rule_id", ""),
        "Severity": str(finding.get("severity", "")).upper(),
        "File_Path": location.get("file_path", "") if isinstance(location, dict) else "",
        "Line_Start": int(line_start) if isinstance(line_start, int) else 0,
        "Line_End": int(line_end) if isinstance(line_end, int) else 0,
        "Message": finding.get("rule_message", "") or rule.get("message", "") or finding.get("message", ""),
        "Finding_URL": finding.get("line_of_code_url", "") or finding.get("url", "") or finding.get("finding_url", ""),
    }


def export_to_csv(rows: Iterable[Dict[str, Any]], output_path: Path) -> None:
    """Write normalized findings to CSV."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
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
    output_csv = project_root / "data" / "semgrep_issues.csv"

    logging.info("Starting Semgrep findings ingestion")
    session = setup_session(api_token)

    raw_findings = fetch_findings(session, deployment_slug)
    logging.info("Fetched %s total findings", len(raw_findings))

    normalized: List[Dict[str, Any]] = []
    for finding in raw_findings:
        normalized_finding = normalize_finding(finding)
        if normalized_finding is not None:
            normalized.append(normalized_finding)

    logging.info("Filtered to %s OPEN findings", len(normalized))
    export_to_csv(normalized, output_csv)
    logging.info("Wrote normalized findings to %s", output_csv)


if __name__ == "__main__":
    main()
