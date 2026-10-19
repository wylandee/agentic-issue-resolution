"""Tests for explicit Maven package selection in the Vulnado batch runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from examples.Vulnado.run_batch import main, run_batch

from remediation_engine.api import RemediationResult

_PACKAGES = [
    "org.apache.tomcat.embed:tomcat-embed-core",
    "org.apache.tomcat.embed:tomcat-embed-websocket",
]


def _write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_dir = tmp_path / "vulnado"
    repo_dir.mkdir()

    baseline_path = tmp_path / "baseline_issues.jsonl"
    rows = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "source": "odc",
            "issue_type": "sca",
            "package_name": _PACKAGES[0],
            "package_version": "9.0.14",
            "purl": "pkg:maven/org.apache.tomcat.embed/tomcat-embed-core@9.0.14",
            "ecosystem": "maven",
            "message": "core finding",
            "cve_id": "CVE-2019-0199",
        },
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "source": "odc",
            "issue_type": "sca",
            "package_name": _PACKAGES[1],
            "package_version": "9.0.14",
            "purl": "pkg:maven/org.apache.tomcat.embed/tomcat-embed-websocket@9.0.14",
            "ecosystem": "maven",
            "message": "websocket finding",
            "cve_id": "CVE-2019-0199",
        },
        {
            "id": "33333333-3333-3333-3333-333333333333",
            "source": "odc",
            "issue_type": "sca",
            "package_name": "org.yaml:snakeyaml",
            "package_version": "1.23",
            "purl": "pkg:maven/org.yaml/snakeyaml@1.23",
            "ecosystem": "maven",
            "message": "unselected finding",
            "cve_id": "CVE-2022-1471",
        },
    ]
    baseline_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    suppressions_path = tmp_path / "suppressions.xml"
    entries = [
        f"""    <suppress>
        <notes>Suppress {name}</notes>
        <packageUrl regex="true">^pkg:maven/{name.replace(":", "/")}@.*$</packageUrl>
        <vulnerabilityName regex="true">.*</vulnerabilityName>
    </suppress>"""
        for name in [*_PACKAGES, "org.yaml:snakeyaml"]
    ]
    suppressions_path.write_text(
        "<suppressions>\n" + "\n".join(entries) + "\n</suppressions>\n",
        encoding="utf-8",
    )
    return repo_dir, baseline_path, suppressions_path


def test_run_batch_accepts_one_explicit_maven_package_batch(tmp_path: Path) -> None:
    repo_dir, baseline_path, suppressions_path = _write_fixture_inputs(tmp_path)
    suppressed_issues_path = tmp_path / "suppressed" / "issues.jsonl"
    output_dir = tmp_path / "output"

    summaries = run_batch(
        iterations=5,
        batch_size=1,
        target_packages=[_PACKAGES[1], f"  {_PACKAGES[0]}  ", _PACKAGES[0]],
        repo_root=repo_dir,
        baseline_path=baseline_path,
        suppressed_issues_path=suppressed_issues_path,
        suppressions_xml_path=suppressions_path,
        output_dir=output_dir,
        dry_run=True,
    )

    assert len(summaries) == 1
    assert summaries[0].selected_packages == sorted(_PACKAGES)
    assert summaries[0].issue_count == 2
    selected_rows = [
        json.loads(line) for line in suppressed_issues_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {row["package_name"] for row in selected_rows} == set(_PACKAGES)
    assert (repo_dir / "suppressions.xml").is_file()


def test_run_batch_passes_package_allowlist_to_java_development_request(
    tmp_path: Path,
) -> None:
    repo_dir, baseline_path, suppressions_path = _write_fixture_inputs(tmp_path)
    with patch(
        "examples.Vulnado.run_batch.run_remediation",
        return_value=RemediationResult(status="completed"),
    ) as mocked_run:
        run_batch(
            target_packages=_PACKAGES,
            repo_root=repo_dir,
            baseline_path=baseline_path,
            suppressed_issues_path=tmp_path / "suppressed" / "issues.jsonl",
            suppressions_xml_path=suppressions_path,
            output_dir=tmp_path / "output",
        )

    assert mocked_run.call_count == 1
    request = mocked_run.call_args.args[0]
    assert request.target_packages == _PACKAGES
    assert request.system_context.environment == "development"
    assert request.system_context.primary_language == "java"


@pytest.mark.parametrize(
    ("target_packages", "message"),
    [
        (["", "  "], "at least one non-empty package name"),
        (["org.apache.tomcat.embed:missing"], "not present in the baseline"),
    ],
)
def test_run_batch_rejects_empty_or_unknown_explicit_packages(
    tmp_path: Path,
    target_packages: list[str],
    message: str,
) -> None:
    repo_dir, baseline_path, suppressions_path = _write_fixture_inputs(tmp_path)

    with pytest.raises(ValueError, match=message):
        run_batch(
            target_packages=target_packages,
            repo_root=repo_dir,
            baseline_path=baseline_path,
            suppressed_issues_path=tmp_path / "suppressed" / "issues.jsonl",
            suppressions_xml_path=suppressions_path,
            output_dir=tmp_path / "output",
            dry_run=True,
        )


def test_main_forwards_packages_option_to_batch_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = [_PACKAGES[0], _PACKAGES[1]]
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch.py", "--packages", *selected, "--dry-run"],
    )
    with (
        patch("examples.Vulnado.run_batch.load_dotenv"),
        patch("examples.Vulnado.run_batch.run_batch", return_value=[]) as mocked_run,
    ):
        assert main() == 0

    assert mocked_run.call_args.kwargs["target_packages"] == selected
