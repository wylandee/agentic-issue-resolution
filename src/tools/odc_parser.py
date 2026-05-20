import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List


CSV_HEADERS = [
    "Dependency_Name",
    "File_Path",
    "Vulnerability_ID",
    "Severity",
    "Description",
]


def load_report(report_path: Path) -> Dict[str, Any]:
    """Load and parse the OWASP Dependency-Check JSON report from disk."""
    if not report_path.exists():
        raise FileNotFoundError(f"Input report not found: {report_path}")

    with report_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _extract_severity(vulnerability: Dict[str, Any]) -> str:
    """Extract severity using ODC fallback order.

    Fallback order:
    1. vulnerability.highestSeverity
    2. vulnerability.cvssv3.baseSeverity
    3. vulnerability.cvssv2.severity
    """
    highest = vulnerability.get("highestSeverity", "")
    if highest:
        return str(highest)

    # ODC sometimes omits highestSeverity, so we inspect nested CVSS blocks.
    cvssv3 = vulnerability.get("cvssv3", {})
    if isinstance(cvssv3, dict):
        base_severity = cvssv3.get("baseSeverity", "")
        if base_severity:
            return str(base_severity)

    # Older records may only include CVSS v2 severity.
    cvssv2 = vulnerability.get("cvssv2", {})
    if isinstance(cvssv2, dict):
        v2_severity = cvssv2.get("severity", "")
        if v2_severity:
            return str(v2_severity)

    return ""


def parse_vulnerabilities(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten dependency vulnerabilities into normalized row dictionaries."""
    normalized_rows: List[Dict[str, Any]] = []
    dependencies = report.get("dependencies", [])

    if not isinstance(dependencies, list):
        return normalized_rows

    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue

        vulnerabilities = dependency.get("vulnerabilities", [])
        if not vulnerabilities or not isinstance(vulnerabilities, list):
            continue

        for vulnerability in vulnerabilities:
            if not isinstance(vulnerability, dict):
                continue

            normalized_rows.append(
                {
                    "Dependency_Name": dependency.get("fileName", ""),
                    "File_Path": dependency.get("filePath", ""),
                    "Vulnerability_ID": vulnerability.get("name", ""),
                    "Severity": _extract_severity(vulnerability),
                    "Description": vulnerability.get("description", ""),
                }
            )

    return normalized_rows


def export_to_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Export normalized vulnerability rows to CSV."""
    os.makedirs(output_path.parent, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Run the ODC JSON -> CSV ingestion pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    input_report = project_root / "data" / "dependency-check-report.json"
    output_csv = project_root / "data" / "odc_issues.csv"

    logging.info("Loading ODC report from %s", input_report)
    report = load_report(input_report)

    logging.info("Parsing vulnerable dependencies")
    normalized_rows = parse_vulnerabilities(report)
    logging.info("Parsed %s vulnerability rows", len(normalized_rows))

    logging.info("Writing CSV to %s", output_csv)
    export_to_csv(normalized_rows, output_csv)
    logging.info("ODC ingestion completed")


if __name__ == "__main__":
    main()
