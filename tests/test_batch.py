"""Contract tests for the shared four-codebase batch interface."""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from scripts import run_batch as batch_runner

from remediation_engine.api import RemediationResult
from remediation_engine.contracts.schemas import IssueSource, IssueType, VulnerabilityIssue
from remediation_engine.settings import AppSettings

_BASELINE_HASHES = {
    "juice_shop": "788a4835917b346dd07489a040861fb33976f38eb6b55a519e5a03bc3b7ded7b",
    "nodegoat": "9a9601d3ad66ab2947a9151c02579c54512a3bd5aa45ad2a9b78850bcb4812f6",
    "vulnado": "02dcf49f108a8f80084a7080601f39846a9d41a60e61bbdddb823dd36e60b459",
    "pygoat": "c7ba2059400de6e2a6878e2f7b6896826c1de69f3ed5d2a7adeb4af90e375a66",
}
_PACKAGE_COUNTS = {"juice_shop": 53, "nodegoat": 92, "vulnado": 12, "pygoat": 6}


def _issue(
    package_name: str,
    *,
    ecosystem: str,
    purl: str | None,
    index: int = 1,
) -> VulnerabilityIssue:
    """Build one deterministic issue for profile-selection tests."""
    return VulnerabilityIssue(
        id=f"00000000-0000-0000-0000-{index:012d}",
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name=package_name,
        package_version="1.0.0",
        purl=purl,
        ecosystem=ecosystem,
        message=f"Finding for {package_name}",
    )


@pytest.mark.parametrize("codebase", tuple(batch_runner.PROFILES))
def test_profile_points_to_unchanged_actionable_baseline(codebase: str) -> None:
    """Each profile exposes the expected unique names from its retained JSONL."""
    profile = batch_runner.PROFILES[codebase]
    baseline = profile.default_baseline

    assert baseline.is_file()
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == _BASELINE_HASHES[codebase]
    issues = batch_runner.load_baseline_issues(baseline)
    assert len(batch_runner.actionable_packages(profile, issues)) == _PACKAGE_COUNTS[codebase]


def test_baseline_loader_rejects_a_json_array(tmp_path: Path) -> None:
    """The shared runner accepts JSONL records, not a JSON array document."""
    baseline = tmp_path / "array.jsonl"
    baseline.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid baseline issue"):
        batch_runner.load_baseline_issues(baseline)


def test_sampling_is_seeded_unique_finite_and_validated() -> None:
    """Sampling returns repeatable unique batches and rejects impossible requests."""
    packages = [f"pkg-{index}" for index in range(5)]

    sampled = batch_runner.sample_package_batches(packages, batch_size=2, iterations=5, seed=11)

    assert sampled == batch_runner.sample_package_batches(
        packages, batch_size=2, iterations=5, seed=11
    )
    assert len({tuple(batch) for batch in sampled}) == 5
    assert all(len(batch) == 2 and batch == sorted(batch) for batch in sampled)
    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        batch_runner.sample_package_batches(packages, batch_size=0, iterations=1, seed=0)
    with pytest.raises(ValueError, match="only 10 combinations exist"):
        batch_runner.sample_package_batches(packages, batch_size=2, iterations=11, seed=0)


def test_profile_selection_uses_ecosystem_identity_and_canonical_names() -> None:
    """Node and PyPI names normalize for matching while Maven needs its PURL."""
    node = batch_runner.PROFILES["nodegoat"]
    node_issues = [
        _issue(
            "@Angular/Common",
            ecosystem="javascript",
            purl="pkg:javascript/%40Angular%2FCommon@1.0.0",
        ),
        _issue(
            "commons-io:commons-io",
            ecosystem="maven",
            purl="pkg:maven/commons-io/commons-io@1.0.0",
            index=2,
        ),
    ]
    node_packages = batch_runner.actionable_packages(node, node_issues)
    assert list(node_packages.values()) == ["@Angular/Common"]
    assert [
        item.package_name
        for item in batch_runner.filter_issues(node, node_issues, {"@angular/common"})
    ] == ["@Angular/Common"]

    python_profile = batch_runner.PROFILES["pygoat"]
    python_issue = _issue("My_Package", ecosystem="pypi", purl="pkg:pypi/my-package@1.0.0")
    python_packages = batch_runner.actionable_packages(python_profile, [python_issue])
    assert python_packages == {"my-package": "My_Package"}
    python_batches, _available = batch_runner._selected_batches(
        python_profile,
        [python_issue],
        packages=["my.package"],
        iterations=1,
        batch_size=1,
        seed=0,
    )
    assert python_batches == [["My_Package"]]
    assert batch_runner.filter_issues(
        python_profile,
        [python_issue],
        {batch_runner._package_key(python_profile, python_batches[0][0])},
    ) == [python_issue]

    java_profile = batch_runner.PROFILES["vulnado"]
    java_issue = _issue(
        "org.postgresql:postgresql",
        ecosystem="maven",
        purl="pkg:maven/org.postgresql/postgresql@1.0.0",
    )
    mismatched = java_issue.model_copy(update={"purl": "pkg:npm/postgresql@1.0.0"})
    assert batch_runner.actionable_packages(java_profile, [java_issue]) == {
        "org.postgresql:postgresql": "org.postgresql:postgresql"
    }
    assert batch_runner.actionable_packages(java_profile, [mismatched]) == {}


@pytest.mark.parametrize("codebase", tuple(batch_runner.PROFILES))
def test_dry_run_writes_selected_batch_without_repo_or_services(
    codebase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All profiles can prepare artifacts while the default clone is absent."""
    profile = batch_runner.PROFILES[codebase]
    first_package = next(
        iter(
            batch_runner.actionable_packages(
                profile, batch_runner.load_baseline_issues(profile.default_baseline)
            ).values()
        )
    )
    output_root = tmp_path / "batch-runs"

    def unexpected_live_call(*_args: object, **_kwargs: object) -> RemediationResult:
        pytest.fail("dry-run must not call run_remediation")

    monkeypatch.setattr(batch_runner, "run_remediation", unexpected_live_call)
    summary = batch_runner.run_batch(
        codebase,
        repo=tmp_path / "missing-repository",
        packages=[first_package],
        output_dir=output_root,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["batch_count"] == 1
    batch = summary["batches"][0]
    assert batch["status"] == "dry_run"
    run_dir = Path(summary["run_dir"])
    assert (run_dir / batch["issues_path"]).is_file()
    written_issues = batch_runner.load_baseline_issues(run_dir / batch["issues_path"])
    assert written_issues == batch_runner.filter_issues(
        profile,
        batch_runner.load_baseline_issues(profile.default_baseline),
        {batch_runner._package_key(profile, first_package)},
    )
    assert json.loads((run_dir / batch["result_path"]).read_text())["status"] == "dry_run"
    assert (run_dir / batch["patch_path"]).read_text() == ""
    assert not (tmp_path / "missing-repository").exists()


@pytest.mark.parametrize("codebase", tuple(batch_runner.PROFILES))
def test_invalid_selection_fails_before_output_creation(codebase: str, tmp_path: Path) -> None:
    """Unknown targets and impossible batch sizes produce no run directory."""
    output_root = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Unknown or unsupported"):
        batch_runner.run_batch(
            codebase,
            packages=["not-a-baseline-package"],
            output_dir=output_root,
            dry_run=True,
        )
    assert not output_root.exists()

    profile = batch_runner.PROFILES[codebase]
    available = batch_runner.actionable_packages(
        profile, batch_runner.load_baseline_issues(profile.default_baseline)
    )
    with pytest.raises(ValueError, match="Cannot sample batch"):
        batch_runner.run_batch(
            codebase,
            iterations=1,
            batch_size=len(available) + 1,
            output_dir=output_root,
            dry_run=True,
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("codebase", "package_name", "ecosystem", "purl", "selection", "canonical"),
    [
        ("juice_shop", "cookie", "npm", "pkg:npm/cookie@1.0.0", "COOKIE", "cookie"),
        (
            "nodegoat",
            "@Scope/Package",
            "javascript",
            "pkg:javascript/%40Scope%2FPackage@1.0.0",
            "@scope/package",
            "@Scope/Package",
        ),
        (
            "vulnado",
            "org.example:artifact",
            "maven",
            "pkg:maven/org.example/artifact@1.0.0",
            "org.example:artifact",
            "org.example:artifact",
        ),
        (
            "pygoat",
            "My_Package",
            "pypi",
            "pkg:pypi/my-package@1.0.0",
            "MY.PACKAGE",
            "My_Package",
        ),
    ],
)
def test_live_run_uses_scoped_snapshot_and_preserves_caller_repository(
    codebase: str,
    package_name: str,
    ecosystem: str,
    purl: str,
    selection: str,
    canonical: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each live profile scopes a temporary snapshot without changing the caller."""
    profile = batch_runner.PROFILES[codebase]
    issue = _issue(package_name, ecosystem=ecosystem, purl=purl)
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(issue.model_dump_json(exclude_none=False) + "\n", encoding="utf-8")
    repo = tmp_path / "caller-repo"
    repo.mkdir()
    (repo / "source.txt").write_text("caller source\n", encoding="utf-8")
    (repo / ".env").write_text("SECRET=value\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.js").write_text("vendor\n", encoding="utf-8")
    original_entries = sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*"))
    settings = AppSettings(odc_extra_args="--foo bar --suppression /old/path.xml")
    observed: dict[str, object] = {}

    def mocked_run(request: object, *, settings: AppSettings) -> RemediationResult:
        repo_snapshot = request.repo_root
        observed["snapshot"] = repo_snapshot
        observed["target_packages"] = request.target_packages
        observed["language"] = request.system_context.primary_language
        observed["environment"] = request.system_context.environment
        observed["args"] = settings.odc_extra_args
        assert (repo_snapshot / "source.txt").is_file()
        assert not (repo_snapshot / ".env").exists()
        assert not (repo_snapshot / "node_modules").exists()
        suppression = (repo_snapshot / batch_runner.SUPPRESSION_FILENAME).read_text()
        active_component = batch_runner._encoded_purl_component(
            profile, canonical, profile.purl_types[0]
        )
        assert re.escape(active_component) in suppression
        assert "--suppression" in settings.odc_extra_args
        assert "/scan/suppressions.xml" in settings.odc_extra_args
        assert "/old/path.xml" not in settings.odc_extra_args
        return RemediationResult(status="completed", changed_files=["source.txt"], diff="diff\n")

    monkeypatch.setattr(batch_runner, "run_remediation", mocked_run)
    summary = batch_runner.run_batch(
        codebase,
        repo=repo,
        baseline=baseline,
        packages=[selection],
        output_dir=tmp_path / "outputs",
        settings=settings,
    )

    assert observed["target_packages"] == [canonical]
    assert observed["language"] == profile.language.value
    assert observed["environment"] == "development"
    assert observed["args"] == "--foo bar --suppression /scan/suppressions.xml"
    assert not Path(observed["snapshot"]).exists()
    assert sorted(path.relative_to(repo).as_posix() for path in repo.rglob("*")) == original_entries
    assert not (repo / batch_runner.SUPPRESSION_FILENAME).exists()
    run_dir = Path(summary["run_dir"])
    batch = summary["batches"][0]
    assert batch["status"] == "completed", batch["errors"]
    assert (run_dir / batch["patch_path"]).read_text() == "diff\n"
    assert json.loads((run_dir / batch["result_path"]).read_text())["changed_files"] == [
        "source.txt"
    ]


@pytest.mark.parametrize(
    ("codebase", "package", "target_purl", "other_purl"),
    [
        (
            "juice_shop",
            "@scope/pkg",
            "pkg:npm/%40scope%2Fpkg@1.0.0",
            "pkg:npm/lodash@4.0.0",
        ),
        (
            "vulnado",
            "org.example:artifact",
            "pkg:maven/org.example/artifact@1.0.0",
            "pkg:maven/org.example/other@1.0.0",
        ),
        ("pygoat", "PyYAML", "pkg:pypi/pyyaml@1.0.0", "pkg:pypi/requests@2.0.0"),
    ],
)
def test_scoped_suppression_keeps_targets_and_suppresses_other_packages(
    codebase: str,
    package: str,
    target_purl: str,
    other_purl: str,
) -> None:
    """Generated Dependency-Check regexes suppress only non-target PURLs."""
    profile = batch_runner.PROFILES[codebase]
    xml_text = batch_runner.generate_suppressions_xml(profile, [package])
    root = ET.fromstring(xml_text)
    package_url_tag = f"{{{batch_runner._SUPPRESSION_NAMESPACE}}}packageUrl"
    patterns = [node.text for node in root.iter(package_url_tag)]
    purl_type = target_purl.split(":", 1)[1].split("/", 1)[0]
    pattern = patterns[profile.purl_types.index(purl_type)]

    assert pattern is not None
    assert re.match(pattern, target_purl) is None
    assert re.match(pattern, other_purl) is not None


def test_main_lists_packages_without_loading_a_repository(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The package listing CLI needs only the retained baseline file."""
    assert batch_runner.main(["--codebase", "pygoat", "--list-packages", "--repo", "/missing"]) == 0
    assert "pyyaml" in capsys.readouterr().out


def test_main_runs_an_explicit_dry_run_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The documented CLI syntax writes one dry-run batch without a clone."""
    output_root = tmp_path / "runs"
    assert (
        batch_runner.main(
            [
                "--codebase",
                "juice_shop",
                "--packages",
                "cookie",
                "--dry-run",
                "--repo",
                str(tmp_path / "missing-clone"),
                "--output-dir",
                str(output_root),
            ]
        )
        == 0
    )
    assert "Saved 1 batches" in capsys.readouterr().out
    run_dir = next(output_root.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["batches"][0]["selected_packages"] == ["cookie"]
    assert summary["batches"][0]["status"] == "dry_run"


def test_odc_pypi_ingestion_feeds_the_common_dry_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical Dependency-Check ingestion output is accepted by the PyPI profile."""
    from remediation_engine import cli

    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    report = {
        "reportSchema": "1.1",
        "projectInfo": {"name": "sample-python"},
        "dependencies": [
            {
                "fileName": "django-2.2.9.dist-info/METADATA",
                "filePath": "requirements.txt",
                "packages": [{"id": "pkg:pypi/django@2.2.9"}],
                "vulnerabilities": [
                    {
                        "source": "NVD",
                        "name": "CVE-2025-12345",
                        "severity": "HIGH",
                        "description": "Synthetic ingestion contract finding.",
                    }
                ],
            }
        ],
    }
    report_path = tmp_path / "odc-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    baseline_path = tmp_path / "ingested.jsonl"

    assert (
        cli.main(
            [
                "ingest",
                str(report_path),
                "--format",
                "odc-json",
                "--output",
                str(baseline_path),
            ]
        )
        == 0
    )
    issues = batch_runner.load_baseline_issues(baseline_path)
    assert len(issues) == 1
    assert issues[0].issue_type is IssueType.SCA
    assert issues[0].ecosystem == "pypi"
    assert issues[0].purl == "pkg:pypi/django@2.2.9"

    summary = batch_runner.run_batch(
        "pygoat",
        baseline=baseline_path,
        packages=["DJANGO"],
        output_dir=tmp_path / "runs",
        dry_run=True,
    )
    batch = summary["batches"][0]
    assert batch["selected_packages"] == ["django"]
    assert batch["issue_count"] == 1
