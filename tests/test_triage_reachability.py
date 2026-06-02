from __future__ import annotations

from types import SimpleNamespace

from src.contracts.schemas import IssueSource, IssueType, VulnerabilityGroup, VulnerabilityIssue
from src.triage.reachability import analyze_reachability


def _sca_issue(package_name: str) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        package_name=package_name,
    )


def _sca_group(package_name: str) -> VulnerabilityGroup:
    issue = _sca_issue(package_name)
    return VulnerabilityGroup(
        group_id=f"sca:package.json:{package_name}",
        issue_type=IssueType.SCA,
        vulnerable_component=package_name,
        representative_issue_id=issue.id,
        issues=[issue],
    )


def test_analyze_reachability_distinguishes_direct_and_transitive(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"lodash":"1.0.0","express":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.js").write_text("use_lodash", encoding="utf-8")
    (tmp_path / "node_modules" / "vendor").mkdir(parents=True)
    (tmp_path / "node_modules" / "vendor" / "index.js").write_text(
        "use_express",
        encoding="utf-8",
    )

    monkeypatch.setattr("src.triage.reachability.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "src.triage.reachability.load_source_bytes",
        lambda path: path.read_bytes(),
    )
    monkeypatch.setattr(
        "src.triage.reachability.parse_source",
        lambda source_bytes, language: SimpleNamespace(
            root_node=source_bytes.decode("utf-8")
        ),
    )
    monkeypatch.setattr(
        "src.triage.reachability.extract_imports",
        lambda root_node, source_bytes: (
            ['import lodash from "lodash";']
            if "use_lodash" in root_node
            else ['import express from "express";']
            if "use_express" in root_node
            else []
        ),
    )

    groups = [
        _sca_group("lodash"),
        _sca_group("express"),
        _sca_group("minimist"),
    ]

    analyze_reachability(groups, tmp_path)

    assert groups[0].is_reachable is True
    assert groups[1].is_reachable is False
    assert groups[2].is_reachable is None


def test_analyze_reachability_continues_after_parse_failure(monkeypatch, tmp_path):
    (tmp_path / "package.json").write_text(
        '{"dependencies":{"lodash":"1.0.0","express":"1.0.0"}}',
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.js").write_text("broken", encoding="utf-8")
    (tmp_path / "src" / "ok.js").write_text("ok", encoding="utf-8")

    monkeypatch.setattr("src.triage.reachability.language_for_path", lambda _: object())
    monkeypatch.setattr(
        "src.triage.reachability.load_source_bytes",
        lambda path: path.read_bytes(),
    )

    def _parse(source_bytes, language):
        marker = source_bytes.decode("utf-8")
        if marker == "broken":
            raise RuntimeError("parse failed")
        return SimpleNamespace(root_node=marker)

    monkeypatch.setattr("src.triage.reachability.parse_source", _parse)
    monkeypatch.setattr(
        "src.triage.reachability.extract_imports",
        lambda root_node, source_bytes: ['import lodash from "lodash";']
        if root_node == "ok"
        else [],
    )

    groups = [_sca_group("lodash"), _sca_group("express")]
    analyze_reachability(groups, tmp_path)

    assert groups[0].is_reachable is True
    assert groups[1].is_reachable is False
