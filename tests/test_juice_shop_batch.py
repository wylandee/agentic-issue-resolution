"""Unit tests for the Juice Shop batch remediation execution script."""

from __future__ import annotations

import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest
from examples.juice_shop.run_batch import (
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

_SAMPLE_ISSUES_JSONL = """{"id":"11111111-1111-1111-1111-111111111111","finding_id":null,"source":"odc","issue_type":"sca","package_name":"@angular/common","package_version":"14.0.0","purl":"pkg:npm/%40angular%2Fcommon@14.0.0","ecosystem":"npm","message":"msg 1","cve_id":"CVE-2022-25869"}
{"id":"22222222-2222-2222-2222-222222222222","finding_id":null,"source":"odc","issue_type":"sca","package_name":"@sigstore/core","package_version":"1.0.0","purl":"pkg:npm/%40sigstore%2Fcore@1.0.0","ecosystem":"npm","message":"msg 2","cve_id":"CVE-2023-33958"}
{"id":"33333333-3333-3333-3333-333333333333","finding_id":null,"source":"odc","issue_type":"sca","package_name":"cookie","package_version":"0.4.1","purl":"pkg:npm/cookie@0.4.1","ecosystem":"npm","message":"msg 3","cve_id":"CVE-2024-47764"}
{"id":"44444444-4444-4444-4444-444444444444","finding_id":null,"source":"odc","issue_type":"sca","package_name":"diff","package_version":"1.4.0","purl":"pkg:npm/diff@1.4.0","ecosystem":"npm","message":"msg 4","cve_id":"CVE-2020-7788"}
{"id":"55555555-5555-5555-5555-555555555555","finding_id":null,"source":"odc","issue_type":"sca","package_name":"commons-io:commons-io","package_version":"2.4","purl":"pkg:maven/commons-io/commons-io@2.4","ecosystem":"maven","message":"msg 5","cve_id":"CVE-2021-29425"}
"""

_SAMPLE_SUPPRESSIONS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<suppressions xmlns="https://jeremylong.github.io/DependencyCheck/dependency-suppression.1.4.xsd">
    <suppress>
        <notes>Suppress all vulnerabilities for @angular/common</notes>
        <packageUrl regex="true">^pkg:npm/%40angular%2Fcommon@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
    <suppress>
        <notes>Suppress all vulnerabilities for @sigstore/core</notes>
        <packageUrl regex="true">^pkg:npm/%40sigstore%2Fcore@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
<!--
    <suppress>
        <notes>Keep cookie as a selected Juice Shop fixture</notes>
        <packageUrl regex="true">^pkg:npm/cookie@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
-->
    <suppress>
        <notes>Suppress all vulnerabilities for diff (npm)</notes>
        <packageUrl regex="true">^pkg:npm/diff@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>
    <suppress>
        <notes>Suppress all vulnerabilities for commons-io:commons-io (maven)</notes>
        <packageUrl regex="true">^pkg:maven/commons-io/commons-io@.*$</packageUrl>
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
    assert issues[0].package_name == "@angular/common"
    assert issues[1].package_name == "@sigstore/core"


def test_get_unique_packages() -> None:
    """Test extracting unique package names including scoped and maven coordinates."""
    issues = [
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="@angular/common",
            purl="pkg:npm/%40angular%2Fcommon@14.0.0",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="cookie",
            purl="pkg:npm/cookie@0.4.1",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="@angular/common",
            purl="pkg:npm/%40angular%2Fcommon@14.0.1",
        ),
    ]
    pkgs = get_unique_packages(issues)
    assert pkgs == ["@angular/common", "cookie"]


def test_sample_package_batches_uniqueness() -> None:
    """Test that package batches are distinct and properly sized."""
    pkgs = [f"pkg-{i}" for i in range(10)]
    batches = sample_package_batches(pkgs, batch_size=3, num_batches=10, seed=42)

    assert len(batches) == 10
    for batch in batches:
        assert len(batch) == 3
        assert len(set(batch)) == 3

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
            package_name="@angular/common",
            purl="pkg:npm/%40angular%2Fcommon@14.0.0",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="cookie",
            purl="pkg:npm/cookie@0.4.1",
        ),
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="diff",
            purl="pkg:npm/diff@1.4.0",
        ),
    ]
    filtered = filter_issues_for_packages(issues, {"@angular/common", "diff"})
    assert len(filtered) == 2
    assert {i.package_name for i in filtered} == {"@angular/common", "diff"}


def test_write_suppressed_issues(tmp_path: Path) -> None:
    """Test writing filtered issues to JSONL fixture."""
    issues = [
        VulnerabilityIssue(
            id=str(uuid.uuid4()),
            source="odc",
            issue_type="sca",
            package_name="@angular/common",
            purl="pkg:npm/%40angular%2Fcommon@14.0.0",
        ),
    ]
    out_path = tmp_path / "odc_suppressed_issues.jsonl"
    write_suppressed_issues(issues, out_path)

    assert out_path.is_file()
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["package_name"] == "@angular/common"


def test_parse_package_name_from_url() -> None:
    """Test unescaping scoped npm, maven, and javascript package URLs."""
    assert parse_package_name_from_url(r"^pkg:npm/%40angular%2Fcommon@.*$") == "@angular/common"
    assert parse_package_name_from_url(r"^pkg:npm/%40sigstore%2Fcore@.*$") == "@sigstore/core"
    assert (
        parse_package_name_from_url(r"^pkg:maven/commons-io/commons-io@.*$")
        == "commons-io:commons-io"
    )
    assert parse_package_name_from_url(r"^pkg:javascript/underscore\.js@.*$") == "underscore.js"
    assert parse_package_name_from_url(r"^pkg:npm/cookie@.*$") == "cookie"
    assert parse_package_name_from_url(r"invalid-url") is None


def test_update_suppressions_xml_content() -> None:
    """Test commenting out selected packages and uncommenting all others."""
    selected = {"@angular/common", "diff"}
    updated_xml = update_suppressions_xml_content(_SAMPLE_SUPPRESSIONS_XML, selected)

    root = ET.fromstring(updated_xml)
    assert root.tag.endswith("suppressions")

    assert "Keep @angular/common as a selected Juice Shop fixture" in updated_xml
    assert "Keep diff as a selected Juice Shop fixture" in updated_xml

    # cookie was previously commented out, now should be uncommented (suppressed)
    assert "Suppress all vulnerabilities for cookie (npm)" in updated_xml
    assert "Suppress all vulnerabilities for @sigstore/core (npm)" in updated_xml


def test_update_suppressions_xml_file(tmp_path: Path) -> None:
    """Test update_suppressions_xml reads and writes to disk correctly."""
    xml_path = tmp_path / "suppressions.xml"
    xml_path.write_text(_SAMPLE_SUPPRESSIONS_XML, encoding="utf-8")

    update_suppressions_xml(xml_path, xml_path, {"@sigstore/core", "cookie"})
    content = xml_path.read_text(encoding="utf-8")

    assert "Keep @sigstore/core as a selected Juice Shop fixture" in content
    assert "Keep cookie as a selected Juice Shop fixture" in content
    assert "Suppress all vulnerabilities for @angular/common (npm)" in content


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
    repo_dir = tmp_path / "juice-shop"
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
        assert s.issue_count > 0
        assert Path(s.result_path).name.startswith("juice-shop-run-")
        assert Path(s.patch_path).name.startswith("juice-shop-run-")

    # Check that repo copy was performed
    assert (repo_dir / "suppressions.xml").is_file()

    # Check aggregate summary JSON
    summary_json = output_dir / "juice-shop-batch-runs-summary.json"
    assert summary_json.is_file()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["total_iterations"] == 3
    assert payload["dry_run"] is True
    assert len(payload["results"]) == 3


def test_run_batch_mocked_remediation(tmp_path: Path) -> None:
    """Test run_batch iteration flow with mocked run_remediation."""
    repo_dir = tmp_path / "juice-shop"
    repo_dir.mkdir()

    baseline_file = tmp_path / "baseline_issues.jsonl"
    baseline_file.write_text(_SAMPLE_ISSUES_JSONL, encoding="utf-8")

    suppressed_issues_file = tmp_path / "odc_suppressed_issues.jsonl"
    suppressions_file = tmp_path / "suppressions.xml"
    suppressions_file.write_text(_SAMPLE_SUPPRESSIONS_XML, encoding="utf-8")

    output_dir = tmp_path / "trajectories"

    mock_result = RemediationResult(
        run_id="test-run-id",
        status="completed",
        diff="--- a/package.json\n+++ b/package.json\n",
        changed_files=["package.json", "package-lock.json"],
        task_count=2,
        fixed_task_count=2,
        failed_task_count=0,
        errors=[],
        raw_state={},
    )

    with patch(
        "examples.juice_shop.run_batch.run_remediation", return_value=mock_result
    ) as mock_run:
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
            assert s.errors == []

        # Verify result and patch files on disk
        for i in range(1, 3):
            res_file = output_dir / f"juice-shop-run-{i:02d}-result.json"
            patch_file = output_dir / f"juice-shop-run-{i:02d}.patch"
            assert res_file.is_file()
            assert patch_file.is_file()
            assert (
                patch_file.read_text(encoding="utf-8") == "--- a/package.json\n+++ b/package.json\n"
            )
