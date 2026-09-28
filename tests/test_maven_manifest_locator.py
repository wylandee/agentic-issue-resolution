from __future__ import annotations

from pathlib import Path

import pytest

from remediation_engine.contracts import IssueSource, IssueType, Severity, VulnerabilityIssue
from remediation_engine.tools.maven_manifest_locator import (
    MavenManifestError,
    find_dependency_in_pom,
    locate_maven_from_issue,
    locate_maven_manifests,
    parse_pom_xml,
)

_NS = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def _issue(**kwargs) -> VulnerabilityIssue:
    values = {
        "source": IssueSource.ODC,
        "issue_type": IssueType.SCA,
        "severity": Severity.HIGH,
        "ecosystem": "maven",
        "package_name": "org.example:widget",
        "package_version": "1.0",
        "purl": "pkg:maven/org.example/widget@1.0",
        "raw_payload": {"filePath": "/scan/src/main/java/App.java"},
    }
    values.update(kwargs)
    return VulnerabilityIssue(**values)


def _pom(*body: str) -> str:
    return f"<project {_NS}><modelVersion>4.0.0</modelVersion>{''.join(body)}</project>"


def _coordinates(group: str = "com.acme", artifact: str = "app") -> str:
    return f"<groupId>{group}</groupId><artifactId>{artifact}</artifactId><version>1.0</version>"


def _dep(*, version: str | None = "2.0", extra: str = "") -> str:
    version_xml = f"<version>{version}</version>" if version is not None else ""
    return (
        "<dependency><groupId>org.example</groupId><artifactId>widget</artifactId>"
        f"{version_xml}{extra}</dependency>"
    )


def test_manifest_discovery_prunes_vendor_and_build_but_keeps_mvn(tmp_path: Path):
    for relative in (
        "pom.xml",
        ".mvn/pom.xml",
        ".git/vendor/pom.xml",
        "node_modules/vendor/pom.xml",
        "target/generated/pom.xml",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_pom(_coordinates()), encoding="utf-8")

    assert [path.relative_to(tmp_path).as_posix() for path in locate_maven_manifests(tmp_path)] == [
        ".mvn/pom.xml",
        "pom.xml",
    ]


def test_parse_namespace_inheritance_and_property_owner(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _pom(
            _coordinates("com.acme", "reactor"),
            "<properties><widget.version>2.4</widget.version></properties>",
            "<modules><module>service</module></modules>",
        ),
        encoding="utf-8",
    )
    child = tmp_path / "service" / "pom.xml"
    child.parent.mkdir()
    child.write_text(
        _pom(
            "<parent><groupId>com.acme</groupId><artifactId>reactor</artifactId>"
            "<version>1.0</version><relativePath>../pom.xml</relativePath></parent>",
            "<artifactId>service</artifactId>",
            f"<dependencies>{_dep(version='${widget.version}')}</dependencies>",
        ),
        encoding="utf-8",
    )

    parsed = parse_pom_xml(child)
    dependency = find_dependency_in_pom("org.example", "widget", parsed)

    assert parsed.group_id == "com.acme"
    assert parsed.parent_path == root.resolve()
    assert parsed.modules == ()
    assert dependency is not None
    assert dependency.is_direct is True
    assert dependency.declared_version == "${widget.version}"
    assert dependency.property_name == "widget.version"
    assert (
        Path(__file__).resolve().parents[1] / dependency.property_file
    ).resolve() == root.resolve()


def test_raw_odc_scan_path_selects_nested_module_and_direct_explicit_target(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _pom(_coordinates(), "<modules><module>service</module></modules>"),
        encoding="utf-8",
    )
    module = tmp_path / "service" / "pom.xml"
    module.parent.mkdir()
    module.write_text(
        _pom(
            "<parent><groupId>com.acme</groupId><artifactId>app</artifactId>"
            "<version>1.0</version><relativePath>../pom.xml</relativePath></parent>",
            "<artifactId>service</artifactId>",
            f"<dependencies>{_dep(version='2.2')}</dependencies>",
        ),
        encoding="utf-8",
    )
    issue = _issue(raw_payload={"filePath": "/scan/service/src/main/java/Widget.java"})

    localized = locate_maven_from_issue(issue, tmp_path)

    assert localized.package_manager == "maven"
    assert localized.manifest_file == "service/pom.xml"
    assert localized.is_direct_dependency is True
    assert localized.declaration_type == "dependencies"
    assert localized.localization_confidence > 0
    assert localized.dependency_ancestry == []
    assert localized.parent_package_name is None


def test_local_parent_relative_path_must_match_declared_coordinates(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        _pom(_coordinates("com.acme", "unrelated")),
        encoding="utf-8",
    )
    module = tmp_path / "service" / "pom.xml"
    module.parent.mkdir()
    module.write_text(
        _pom(
            "<parent><groupId>com.acme</groupId><artifactId>app</artifactId>"
            "<version>1.0</version><relativePath>../pom.xml</relativePath></parent>",
            "<artifactId>service</artifactId>",
            f"<dependencies>{_dep()}</dependencies>",
        ),
        encoding="utf-8",
    )

    localized = locate_maven_from_issue(
        _issue(raw_payload={"filePath": "/scan/service/src/main/Widget.java"}),
        tmp_path,
    )

    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0


def test_direct_managed_and_transitive_targets_use_dependency_management(tmp_path: Path):
    direct_issue = _issue(raw_payload={"filePath": "/scan/pom.xml"})
    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            f"<dependencyManagement><dependencies>{_dep(version='2.3')}</dependencies>"
            "</dependencyManagement>",
            f"<dependencies>{_dep(version=None)}</dependencies>",
        ),
        encoding="utf-8",
    )

    managed_direct = locate_maven_from_issue(direct_issue, tmp_path)
    transitive = locate_maven_from_issue(
        _issue(package_name="org.example:transitive", purl="pkg:maven/org.example/transitive@1"),
        tmp_path,
    )

    assert managed_direct.is_direct_dependency is True
    assert managed_direct.declaration_type == "dependencyManagement"
    assert transitive.is_direct_dependency is False
    assert transitive.declaration_type == "dependencyManagement"
    assert transitive.manifest_file == "pom.xml"


def test_bom_managed_direct_dependency_targets_dependency_management(tmp_path: Path):
    bom = (
        "<dependency><groupId>org.example</groupId><artifactId>widgets-bom</artifactId>"
        "<version>3.0</version><type>pom</type><scope>import</scope></dependency>"
    )
    content = _pom(
        _coordinates(),
        f"<dependencyManagement><dependencies>{bom}</dependencies></dependencyManagement>",
        f"<dependencies>{_dep(version=None)}</dependencies>",
    )
    (tmp_path / "pom.xml").write_text(content, encoding="utf-8")

    localized = locate_maven_from_issue(
        _issue(raw_payload={"filePath": "/scan/pom.xml"}),
        tmp_path,
    )

    assert localized.is_direct_dependency is True
    assert localized.declaration_type == "dependencyManagement"


def test_inherited_parent_direct_dependency_targets_local_management_override(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _pom(
            _coordinates(),
            "<modules><module>service</module></modules>",
            f"<dependencies>{_dep(version='2.0')}</dependencies>",
        ),
        encoding="utf-8",
    )
    module = tmp_path / "service" / "pom.xml"
    module.parent.mkdir()
    module.write_text(
        _pom(
            "<parent><groupId>com.acme</groupId><artifactId>app</artifactId>"
            "<version>1.0</version><relativePath>../pom.xml</relativePath></parent>",
            "<artifactId>service</artifactId>",
        ),
        encoding="utf-8",
    )

    localized = locate_maven_from_issue(
        _issue(raw_payload={"filePath": "/scan/service/src/Service.java"}),
        tmp_path,
    )

    assert localized.manifest_file == "service/pom.xml"
    assert localized.is_direct_dependency is True
    assert localized.declaration_type == "dependencyManagement"


def test_scan_path_prefers_raw_payload_over_normalized_issue_path(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _pom(_coordinates(), "<modules><module>one</module><module>two</module></modules>"),
        encoding="utf-8",
    )
    for module_name in ("one", "two"):
        module = tmp_path / module_name / "pom.xml"
        module.parent.mkdir()
        module.write_text(
            _pom(
                "<parent><groupId>com.acme</groupId><artifactId>app</artifactId>"
                "<version>1.0</version><relativePath>../pom.xml</relativePath></parent>",
                f"<artifactId>{module_name}</artifactId>",
                f"<dependencies>{_dep()}</dependencies>",
            ),
            encoding="utf-8",
        )

    issue = _issue(
        file_path="one/src/App.java",
        raw_payload={"filePath": "/scan/two/src/App.java"},
    )
    assert locate_maven_from_issue(issue, tmp_path).manifest_file == "two/pom.xml"


def test_unknown_path_uses_unique_root_pom(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(_pom(_coordinates()), encoding="utf-8")
    issue = _issue(raw_payload={"filePath": "/scan/unknown/path/file.jar"})
    assert locate_maven_from_issue(issue, tmp_path).manifest_file == "pom.xml"


@pytest.mark.parametrize(
    "content",
    [
        "<project><dependencies>",
        '<!DOCTYPE project [<!ENTITY x "boom">]><project><artifactId>&x;</artifactId></project>',
    ],
)
def test_malformed_or_entity_pom_fails_closed(tmp_path: Path, content: str):
    (tmp_path / "pom.xml").write_text(content, encoding="utf-8")
    localized = locate_maven_from_issue(_issue(), tmp_path)
    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0
    assert localized.package_manager == "maven"


def test_missing_maven_root_fails_closed(tmp_path: Path):
    localized = locate_maven_from_issue(_issue(), tmp_path)
    assert localized.package_manager == "maven"
    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0


def test_duplicate_classifier_profile_and_multiple_roots_fail_closed(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        _pom(_coordinates(), f"<dependencies>{_dep()} {_dep()}</dependencies>"),
        encoding="utf-8",
    )
    duplicate = locate_maven_from_issue(_issue(), tmp_path)
    assert duplicate.manifest_file is None

    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            "<profiles><profile><id>optional</id><dependencies>"
            f"{_dep()}</dependencies></profile></profiles>",
        ),
        encoding="utf-8",
    )
    profile = locate_maven_from_issue(_issue(), tmp_path)
    assert profile.manifest_file is None

    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            f"<dependencies>{_dep(extra='<classifier>tests</classifier>')}</dependencies>",
        ),
        encoding="utf-8",
    )
    classifier = locate_maven_from_issue(_issue(), tmp_path)
    assert classifier.manifest_file is None

    other = tmp_path / "independent" / "pom.xml"
    other.parent.mkdir()
    other.write_text(_pom(_coordinates("com.other", "standalone")), encoding="utf-8")
    roots = locate_maven_from_issue(_issue(), tmp_path)
    assert roots.manifest_file is None


def test_outside_scan_path_does_not_fall_back_to_root(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(_pom(_coordinates()), encoding="utf-8")
    issue = _issue(raw_payload={"filePath": "/etc/maven/pom.xml"})
    localized = locate_maven_from_issue(issue, tmp_path)
    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0


def test_find_dependency_rejects_duplicate_declarations(tmp_path: Path):
    path = tmp_path / "pom.xml"
    path.write_text(
        _pom(_coordinates(), f"<dependencies>{_dep()} {_dep()}</dependencies>"), encoding="utf-8"
    )
    parsed = parse_pom_xml(path)
    with pytest.raises(MavenManifestError):
        find_dependency_in_pom("org.example", "widget", parsed)


def test_java_context_dispatches_the_same_maven_finding_to_pom(tmp_path: Path):
    from remediation_engine.language import ProjectLanguage
    from remediation_engine.tools.manifest_locator import locate_from_issue

    (tmp_path / "pom.xml").write_text(
        _pom(_coordinates(), f"<dependencies>{_dep()}</dependencies>"),
        encoding="utf-8",
    )
    issue = _issue()

    localized = locate_from_issue(issue, tmp_path, project_language=ProjectLanguage.JAVA)

    assert localized.package_manager == "maven"
    assert localized.manifest_file == "pom.xml"
    assert localized.declaration_type == "dependencies"


def test_localized_issue_exposes_maven_version_property_owner(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            "<properties><widget.version>2.3.0</widget.version></properties>",
            f"<dependencies>{_dep(version='${widget.version}')}</dependencies>",
        ),
        encoding="utf-8",
    )

    localized = locate_maven_from_issue(_issue(), tmp_path)

    assert localized.manifest_file == "pom.xml"
    assert localized.version_property_name == "widget.version"
    assert localized.version_property_file == "pom.xml"
