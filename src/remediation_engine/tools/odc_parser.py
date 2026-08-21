"""
ODC Parser â€” upgraded SCA ingestion layer.

Changes from v1
---------------
* Emits typed ``VulnerabilityIssue`` Pydantic models (contracts layer).
* Parses PURL from the ``packages`` list, extracts ecosystem from it.
* Extracts CWE entries and OWASP labels from raw vulnerability data.
* Applies the severity fallback chain: highestSeverity â†’ cvssv3.baseSeverity
  â†’ cvssv2.severity â†’ vulnerability.severity.
* Exports canonical JSONL (``data/odc_issues.jsonl``) for agent consumption
  PLUS CSV (``data/odc_issues.csv``) for human inspection.
* ``parse_vulnerabilities`` now returns ``List[VulnerabilityIssue]`` instead
  of plain dicts.  The CSV exporter falls back to a flat projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from remediation_engine.contracts import (
    CWEEntry,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityIssue,
)
from remediation_engine.tools.package_identity import package_name_from_purl

try:
    from packageurl import PackageURL

    _PURL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PURL_AVAILABLE = False


log = logging.getLogger(__name__)

CSV_HEADERS = [
    "Dependency_Name",
    "File_Path",
    "Vulnerability_ID",
    "Severity",
    "PURL",
    "Ecosystem",
    "Package_Name",
    "Package_Version",
    "Description",
    "CWEs",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_severity(vulnerability: dict[str, Any]) -> Severity:
    """Extract severity using ODC fallback order, returning a typed Severity."""
    candidates = [
        vulnerability.get("highestSeverity", ""),
        (vulnerability.get("cvssv3") or {}).get("baseSeverity", ""),
        (vulnerability.get("cvssv2") or {}).get("severity", ""),
        vulnerability.get("severity", ""),
    ]
    for raw in candidates:
        if raw:
            try:
                return Severity(str(raw).strip().upper())
            except ValueError:
                pass
    return Severity.UNKNOWN


def _parse_purl(dep: dict[str, Any]) -> str | None:
    """Return the first PURL string from the dependency's packages list."""
    packages = dep.get("packages") or []
    for pkg in packages:
        purl_str = pkg.get("id", "")
        if purl_str.startswith("pkg:"):
            return purl_str
    return None


def _ecosystem_from_purl(purl_str: str | None) -> str | None:
    """Extract ecosystem (type) from a PURL string."""
    if not purl_str:
        return None
    if _PURL_AVAILABLE:
        try:
            purl = PackageURL.from_string(purl_str)
            return purl.type or None
        except Exception:
            pass
    # Fallback: parse manually â†’ pkg:<type>/...
    try:
        after_pkg = purl_str[4:]  # strip "pkg:"
        return after_pkg.split("/")[0] or None
    except Exception:
        return None


def _package_name_from_purl(purl_str: str | None) -> str | None:
    """Extract canonical package name from a PURL string.

    For npm: packageurl-python puts '@scope/name' directly in purl.name,
    with namespace=None.  For maven: namespace is the groupId and name is
    the artifactId; we join them with ':' (maven convention).
    """
    return package_name_from_purl(purl_str)


def _version_from_purl(purl_str: str | None) -> str | None:
    """Extract package version from a PURL string."""
    if not purl_str:
        return None
    if _PURL_AVAILABLE:
        try:
            purl = PackageURL.from_string(purl_str)
            return purl.version or None
        except Exception:
            pass
    try:
        segment = purl_str.split("/")[-1]
        parts = segment.split("@")
        return parts[-1] if len(parts) > 1 else None
    except Exception:
        return None


def _parse_cwes(vulnerability: dict[str, Any]) -> list[CWEEntry]:
    """Extract CWE entries from vulnerability.cwes list."""
    cwes_raw = vulnerability.get("cwes") or []
    result: list[CWEEntry] = []
    for raw in cwes_raw:
        if isinstance(raw, dict):
            raw_id = raw.get("id") or raw.get("cwe") or raw.get("name") or ""
            display_name = raw.get("description") or raw.get("name")
        else:
            raw_id = raw
            display_name = None
        raw_text = str(raw_id).strip()
        match = re.search(r"(?i)\bCWE[- ]?(\d+)\b", raw_text)
        if not match:
            match = re.search(r"\b(\d+)\b", raw_text)
        if not match:
            continue
        cwe_str = f"CWE-{match.group(1)}"
        if display_name is None and match.end() < len(raw_text):
            suffix = raw_text[match.end() :].lstrip(" :-")
            display_name = suffix or None
        with suppress(Exception):
            result.append(
                CWEEntry(id=cwe_str, name=str(display_name).strip() if display_name else None)
            )
    return result


def _extract_cve_id(vulnerability: dict[str, Any]) -> str | None:
    """Return CVE ID if the vuln name matches CVE format, else check references."""
    name = (vulnerability.get("name") or "").strip()
    if name.upper().startswith("CVE-"):
        return name.upper()

    import re

    cve_regex = re.compile(r"\b(CVE-\d{4}-\d{4,7})\b", re.IGNORECASE)
    for ref in vulnerability.get("references") or []:
        for key in ("url", "name"):
            val = ref.get(key) or ""
            match = cve_regex.search(val)
            if match:
                return match.group(1).upper()
    return None


def _extract_ghsa_id(vulnerability: dict[str, Any]) -> str | None:
    """Return GHSA ID if the vulnerability name or references contain one."""
    import re

    name = (vulnerability.get("name") or "").strip()
    ghsa_regex = re.compile(
        r"\b(GHSA-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4})\b",
        re.IGNORECASE,
    )

    match = ghsa_regex.search(name)
    if match:
        return match.group(1).upper()

    for ref in vulnerability.get("references") or []:
        for key in ("url", "name"):
            val = ref.get(key) or ""
            match = ghsa_regex.search(val)
            if match:
                return match.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_report(report_path: Path) -> dict[str, Any]:
    """Load and parse the OWASP Dependency-Check JSON report from disk."""
    if not report_path.exists():
        raise FileNotFoundError(f"Input report not found: {report_path}")
    with report_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_vulnerabilities(
    report: dict[str, Any],
    repo_url: str | None = None,
    base_ref: str | None = None,
) -> list[VulnerabilityIssue]:
    """
    Flatten dependency vulnerabilities into typed ``VulnerabilityIssue`` objects.

    Parameters
    ----------
    report:
        Parsed ODC JSON report dict.
    repo_url:
        Optional HTTPS clone URL of the scanned repository.
    base_ref:
        Optional git branch or commit SHA that was scanned.
    """
    issues: list[VulnerabilityIssue] = []
    dependencies = report.get("dependencies") or []

    if not isinstance(dependencies, list):
        return issues

    for dep in dependencies:
        if not isinstance(dep, dict):
            continue

        vulnerabilities = dep.get("vulnerabilities") or []
        if not vulnerabilities:
            continue

        # Dependency-level context
        file_name = dep.get("fileName", "")
        file_path = dep.get("filePath", "")
        purl_str = _parse_purl(dep)
        ecosystem = _ecosystem_from_purl(purl_str)
        package_name = _package_name_from_purl(purl_str)
        package_version = _version_from_purl(purl_str)

        for vuln in vulnerabilities:
            if not isinstance(vuln, dict):
                continue

            severity = _extract_severity(vuln)
            cwes = _parse_cwes(vuln)
            cve_id = _extract_cve_id(vuln)
            ghsa_id = _extract_ghsa_id(vuln)
            vuln_id = (vuln.get("name") or "").strip()

            issue = VulnerabilityIssue(
                source=IssueSource.ODC,
                issue_type=IssueType.SCA,
                repo_url=repo_url,
                base_ref=base_ref,
                severity=severity,
                cve_id=cve_id if cve_id else None,
                ghsa_id=ghsa_id if ghsa_id else None,
                rule_id=vuln_id if not cve_id else None,
                cwe=cwes,
                package_name=package_name or file_name or None,
                package_version=package_version,
                purl=purl_str,
                ecosystem=ecosystem,
                file_path=file_path or None,
                message=vuln.get("description", "") or None,
                raw_payload={
                    "fileName": file_name,
                    "filePath": file_path,
                    "vulnerability": vuln,
                    "packages": dep.get("packages", []),
                    "includedBy": dep.get("includedBy", []),
                },
            )
            issues.append(issue)

    return issues


def export_to_jsonl(issues: list[VulnerabilityIssue], output_path: Path) -> None:
    """Write issues as JSONL â€” one JSON object per line (canonical agent format)."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for issue in issues:
            f.write(issue.model_dump_json())
            f.write("\n")
    log.info("Wrote %d issues to %s", len(issues), output_path)


def export_to_csv(issues: list[VulnerabilityIssue], output_path: Path) -> None:
    """Export a flat CSV projection of issues for human inspection."""
    os.makedirs(output_path.parent, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for issue in issues:
            raw = issue.raw_payload or {}
            writer.writerow(
                {
                    "Dependency_Name": raw.get("fileName", ""),
                    "File_Path": raw.get("filePath", ""),
                    "Vulnerability_ID": issue.cve_id or issue.ghsa_id or issue.rule_id or "",
                    "Severity": issue.severity.value,
                    "PURL": issue.purl or "",
                    "Ecosystem": issue.ecosystem or "",
                    "Package_Name": issue.package_name or "",
                    "Package_Version": issue.package_version or "",
                    "Description": issue.message or "",
                    "CWEs": " ".join(c.id for c in issue.cwe),
                }
            )
    log.info("Wrote CSV to %s", output_path)


def main(argv: list[str] | None = None) -> None:
    """Run the ODC JSON â†’ JSONL + CSV ingestion pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    parser = argparse.ArgumentParser(description="Parse an OWASP Dependency-Check JSON report.")
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "dependency-check-report.json",
        help="Dependency-Check JSON report path.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=project_root / "data" / "odc_issues.jsonl",
        help="Canonical issue JSONL output path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "data" / "odc_issues.csv",
        help="Human-readable CSV output path.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write only JSONL and skip CSV generation.",
    )
    args = parser.parse_args(argv)

    input_report = args.input.expanduser()
    output_jsonl = args.output_jsonl.expanduser()
    output_csv = args.output_csv.expanduser()

    log.info("Loading ODC report from %s", input_report)
    report = load_report(input_report)

    log.info("Parsing vulnerable dependencies")
    issues = parse_vulnerabilities(report)
    log.info("Parsed %d vulnerability issues", len(issues))

    export_to_jsonl(issues, output_jsonl)
    if not args.no_csv:
        export_to_csv(issues, output_csv)
    log.info("ODC ingestion complete")


if __name__ == "__main__":
    main()
