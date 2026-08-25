"""Unit tests for the NodeGoat batch remediation execution script."""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from examples.NodeGoat.run_batch import (
    copy_suppressions_to_repo,
    filter_issues_for_packages,
    get_unique_packages,
    load_baseline_issues,
    parse_package_name_from_url,
    run_batch,
    sample_package_batches,
    update_suppressions_xml,
    update_suppressions_xml_content,
    write_suppressed_issues,
)

from remediation_engine.api import RemediationResult
from remediation_engine.contracts.schemas import VulnerabilityIssue

_SAMPLE_ISSUES_JSONL = """{"id":"11111111-1111-1111-1111-111111111111","finding_id":null,"source":"odc","issue_type":"sca","package_name":"adm-zip","package_version":"0.4.4","purl":"pkg:npm/adm-zip@0.4.4","ecosystem":"npm","message":"msg 1","cve_id":"CVE-2018-1002204"}
{"id":"22222222-2222-2222-2222-222222222222","finding_id":null,"source":"odc","issue_type":"sca","package_name":"growl","package_version":"1.9.2","purl":"pkg:npm/growl@1.9.2","ecosystem":"npm","message":"msg 2","cve_id":"CVE-2017-16042"}
{"id":"33333333-3333-3333-3333-333333333333","finding_id":null,"source":"odc","issue_type":"sca","package_name":"ini","package_version":"1.3.4","purl":"pkg:npm/ini@1.3.4","ecosystem":"npm","message":"msg 3","cve_id":"CVE-2020-7788"}
{"id":"44444444-4444-4444-4444-444444444444","finding_id":null,"source":"odc","issue_type":"sca","package_name":"marked","package_version":"0.3.5","purl":"pkg:npm/marked@0.3.5","ecosystem":"npm","message":"msg 4","cve_id":"CVE-2017-16114"}
{"id":"55555555-5555-5555-5555-555555555555","finding_id":null,"source":"odc","issue_type":"sca","package_name":"mongodb","package_version":"2.2.36","purl":"pkg:npm/mongodb@2.2.36","ecosystem":"npm","message":"msg 5","cve_id":"CVE-2021-32036"}
"""

_SAMPLE_SUPPRESSIONS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<suppressions xmlns="https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.4.xsd">
    <suppress>
        <notes>Suppress all vulnerabilities for bootstrap (javascript)</notes>
        <packageUrl regex="true">^pkg:javascript/bootstrap@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
<!--
    <suppress>
        <notes>Keep adm-zip as a selected NodeGoat fixture</notes>
        <packageUrl regex="true">^pkg:npm/adm\\-zip@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
-->
    <suppress>
        <notes>Suppress all vulnerabilities for growl (npm)</notes>
        <packageUrl regex="true">^pkg:npm/growl@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
    <suppress>
        <notes>Suppress all vulnerabilities for ini (npm)</notes>
        <packageUrl regex="true">^pkg:npm/ini@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
    <suppress>
        <notes>Suppress all vulnerabilities for marked (npm)</notes>
        <packageUrl regex="true">^pkg:npm/marked@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
</suppressions>
"""


def test_load_baseline_issues(tmp_path: Path) -> None:
    """Test loading issues from a JSONL fixture."""
    fixture_path = tmp_path / "test_baseline.jsonl"
    fixture_path.write_text(_SAMPLE_ISSUES_JSONL, encoding="utf-8")

    issues = load_baseline_issues(fixture_path)
    assert len(issues) == 5
    assert issues[0].package_name == "adm-zip"
    assert issues[1].package_name == "growl"


def test_get_unique_packages() -> None:
    """Test extracting unique package names."""
    issues = [
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="pkg-a",
            purl="pkg:npm/pkg-a@1.0.0",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="pkg-b",
            purl="pkg:npm/pkg-b@1.0.0",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="pkg-a",
            purl="pkg:npm/pkg-a@2.0.0",
        ),
    ]
    pkgs = get_unique_packages(issues)
    assert pkgs == ["pkg-a", "pkg-b"]


def test_sample_package_batches_uniqueness() -> None:
    """Test that package batches are distinct and properly sized."""
    pkgs = [f"pkg-{i}" for i in range(10)]
    batches = sample_package_batches(pkgs, batch_size=3, num_batches=10, seed=42)

    assert len(batches) == 10
    for batch in batches:
        assert len(batch) == 3
        assert len(set(batch)) == 3

    # Check that all batches are unique tuples
    batch_tuples = [tuple(b) for b in batches]
    assert len(set(batch_tuples)) == 10


def test_sample_package_batches_insufficient_packages() -> None:
    """Test error when available packages are fewer than batch size."""
    with pytest.raises(ValueError, match="Cannot sample batch"):
        sample_package_batches(["pkg-a", "pkg-b"], batch_size=3, num_batches=1)


def test_filter_issues_for_packages() -> None:
    """Test filtering issues matching selected package names."""
    issues = [
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="adm-zip",
            purl="pkg:npm/adm-zip@0.4.4",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="growl",
            purl="pkg:npm/growl@1.9.2",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="ini",
            purl="pkg:npm/ini@1.3.4",
        ),
    ]
    filtered = filter_issues_for_packages(issues, {"adm-zip", "ini"})
    assert len(filtered) == 2
    assert {i.package_name for i in filtered} == {"adm-zip", "ini"}


def test_write_suppressed_issues(tmp_path: Path) -> None:
    """Test writing filtered issues to JSONL fixture."""
    issues = [
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="adm-zip",
            purl="pkg:npm/adm-zip@0.4.4",
        ),
    ]
    out_path = tmp_path / "odc_suppressed_issues.jsonl"
    write_suppressed_issues(issues, out_path)

    assert out_path.is_file()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["package_name"] == "adm-zip"


def test_parse_package_name_from_url() -> None:
    """Test unescaping package names from packageUrl patterns."""
    assert parse_package_name_from_url(r"^pkg:npm/adm\-zip@.*$") == "adm-zip"
    assert parse_package_name_from_url(r"^pkg:javascript/moment\.js@.*$") == "moment.js"
    assert parse_package_name_from_url(r"^pkg:npm/growl@.*$") == "growl"
    assert parse_package_name_from_url(r"invalid-url") is None


def test_update_suppressions_xml_content() -> None:
    """Test commenting out selected packages and uncommenting all others."""
    selected = {"growl", "ini"}
    updated_xml = update_suppressions_xml_content(_SAMPLE_SUPPRESSIONS_XML, selected)

    # Verify updated string is valid XML
    root = ET.fromstring(updated_xml)
    assert root.tag.endswith("suppressions")

    # In XML comments, growl and ini should appear
    assert "Keep growl as a selected NodeGoat fixture" in updated_xml
    assert "Keep ini as a selected NodeGoat fixture" in updated_xml

    # adm-zip was previously commented out, now should be uncommented (suppressed)
    assert "Suppress all vulnerabilities for adm-zip (npm)" in updated_xml
    assert "Suppress all vulnerabilities for bootstrap (javascript)" in updated_xml


def test_update_suppressions_xml_file(tmp_path: Path) -> None:
    """Test update_suppressions_xml reads and writes to disk correctly."""
    xml_path = tmp_path / "suppressions.xml"
    xml_path.write_text(_SAMPLE_SUPPRESSIONS_XML, encoding="utf-8")

    update_suppressions_xml(xml_path, xml_path, {"adm-zip", "marked"})
    content = xml_path.read_text(encoding="utf-8")

    assert "Keep adm-zip as a selected NodeGoat fixture" in content
    assert "Keep marked as a selected NodeGoat fixture" in content
    assert "Suppress all vulnerabilities for growl (npm)" in content


def test_copy_suppressions_to_repo(tmp_path: Path) -> None:
    """Test copying suppressions.xml to repository root."""
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    suppressions_file = fixtures_dir / "suppressions.xml"
    suppressions_file.write_text("<suppressions/>", encoding="utf-8")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    dest = copy_suppressions_to_repo(suppressions_file, repo_dir)
    assert dest == repo_dir / "suppressions.xml"
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == "<suppressions/>"


def test_run_batch_dry_run(tmp_path: Path) -> None:
    """Test full batch dry-run without invoking Docker or LLMs."""
    repo_dir = tmp_path / "NodeGoat"
    repo_dir.mkdir()

    baseline_file = tmp_path / "baseline_issues.jsonl"
    baseline_file.write_text(_SAMPLE_ISSUES_JSONL, encoding="utf-8")

    suppressed_issues_file = tmp_path / "odc_suppressed_issues.jsonl"

    suppressions_file = tmp_path / "suppressions.xml"
    suppressions_file.write_text(_SAMPLE_SUPPRESSIONS_XML, encoding="utf-8")

    output_dir = tmp_path / "trajectories"

    summaries = run_batch(
        iterations=3,
        batch_size=2,
        repo_root=repo_dir,
        baseline_path=baseline_file,
        suppressed_issues_path=suppressed_issues_file,
        suppressions_xml_path=suppressions_file,
        output_dir=output_dir,
        seed=123,
        dry_run=True,
    )

    assert len(summaries) == 3
    for s in summaries:
        assert s.status == "dry_run"
        assert len(s.selected_packages) == 2

    # Verify baseline file remained untouched
    assert baseline_file.read_text(encoding="utf-8") == _SAMPLE_ISSUES_JSONL

    # Verify summary JSON was created
    summary_json = output_dir / "nodegoat-batch-runs-summary.json"
    assert summary_json.is_file()
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert data["total_iterations"] == 3
    assert data["dry_run"] is True


@patch("examples.NodeGoat.run_batch.run_remediation")
def test_run_batch_mocked_remediation(mock_run: MagicMock, tmp_path: Path) -> None:
    """Test full batch execution with mocked run_remediation."""
    mock_run.return_value = RemediationResult(
        status="completed",
        changed_files=["package.json", "package-lock.json"],
        diff="--- a/package.json\n+++ b/package.json\n",
        errors=[],
    )

    repo_dir = tmp_path / "NodeGoat"
    repo_dir.mkdir()

    baseline_file = tmp_path / "baseline_issues.jsonl"
    baseline_file.write_text(_SAMPLE_ISSUES_JSONL, encoding="utf-8")

    suppressed_issues_file = tmp_path / "odc_suppressed_issues.jsonl"

    suppressions_file = tmp_path / "suppressions.xml"
    suppressions_file.write_text(_SAMPLE_SUPPRESSIONS_XML, encoding="utf-8")

    output_dir = tmp_path / "trajectories"

    summaries = run_batch(
        iterations=2,
        batch_size=2,
        repo_root=repo_dir,
        baseline_path=baseline_file,
        suppressed_issues_path=suppressed_issues_file,
        suppressions_xml_path=suppressions_file,
        output_dir=output_dir,
        seed=42,
        dry_run=False,
    )

    assert len(summaries) == 2
    assert mock_run.call_count == 2
    for s in summaries:
        assert s.status == "completed"
        assert s.changed_files == ["package.json", "package-lock.json"]

    # Verify iteration output files
    assert (output_dir / "nodegoat-run-01-result.json").is_file()
    assert (output_dir / "nodegoat-run-01.patch").is_file()
    assert (output_dir / "nodegoat-run-02-result.json").is_file()
    assert (output_dir / "nodegoat-run-02.patch").is_file()
