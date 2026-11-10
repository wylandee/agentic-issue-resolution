"""Focused tests for manifests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from packaging.requirements import Requirement

from remediation_engine.contracts import (
    IssueSource,
    IssueType,
    NoFixMitigationStage,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.schemas import (
    CommandResult,
    MavenTargetOperation,
    MultiPackageAction,
    PackageMutation,
    TacticalStrategy,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.state import normalize_group_paths
from remediation_engine.tools.identity import normalize_python_package_name, package_name_from_purl
from remediation_engine.tools.maven_manifest_locator import (
    MavenManifestError,
    add_dependency_management_entry,
    find_dependency_in_pom,
    locate_maven_from_issue,
    locate_maven_manifests,
    parse_pom_xml,
    remove_direct_dependency_entry,
    update_pom_dependency_version,
    update_pom_property_value,
)
from remediation_engine.tools.maven_mutations import (
    _make_modify_and_validate_maven_dependency_tool,
    _MavenPackageCheckpoint,
    _remove_maven_no_fix_dependency_transaction,
    apply_multi_maven_package_action,
)
from remediation_engine.tools.npm_mutations import (
    _normalize_python_manifest_targets,
    _package_checkpoint_paths,
)
from remediation_engine.tools.python_manifest_locator import (
    RequirementEntry,
    locate_from_issue,
    locate_python_manifests,
    parse_requirements_txt,
    update_requirements_dependency,
)
from remediation_engine.tools.python_mutations import (
    _make_modify_and_validate_python_dependency_tool,
)
from remediation_engine.tools.workers import build_workaround_toolbelt

_NS = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def _maven_manifest_issue(**kwargs) -> VulnerabilityIssue:
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
    issue = _maven_manifest_issue(
        raw_payload={"filePath": "/scan/service/src/main/java/Widget.java"}
    )

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
        _maven_manifest_issue(raw_payload={"filePath": "/scan/service/src/main/Widget.java"}),
        tmp_path,
    )

    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0


def test_direct_managed_and_transitive_targets_use_dependency_management(tmp_path: Path):
    direct_issue = _maven_manifest_issue(raw_payload={"filePath": "/scan/pom.xml"})
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
        _maven_manifest_issue(
            package_name="org.example:transitive", purl="pkg:maven/org.example/transitive@1"
        ),
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
        _maven_manifest_issue(raw_payload={"filePath": "/scan/pom.xml"}),
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
        _maven_manifest_issue(raw_payload={"filePath": "/scan/service/src/Service.java"}),
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

    issue = _maven_manifest_issue(
        file_path="one/src/App.java",
        raw_payload={"filePath": "/scan/two/src/App.java"},
    )
    assert locate_maven_from_issue(issue, tmp_path).manifest_file == "two/pom.xml"


def test_unknown_path_uses_unique_root_pom(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(_pom(_coordinates()), encoding="utf-8")
    issue = _maven_manifest_issue(raw_payload={"filePath": "/scan/unknown/path/file.jar"})
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
    localized = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0
    assert localized.package_manager == "maven"


def test_missing_maven_root_fails_closed(tmp_path: Path):
    localized = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert localized.package_manager == "maven"
    assert localized.manifest_file is None
    assert localized.localization_confidence == 0.0


def test_duplicate_classifier_profile_and_multiple_roots_fail_closed(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(
        _pom(_coordinates(), f"<dependencies>{_dep()} {_dep()}</dependencies>"),
        encoding="utf-8",
    )
    duplicate = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert duplicate.manifest_file is None

    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            "<profiles><profile><id>optional</id><dependencies>"
            f"{_dep()}</dependencies></profile></profiles>",
        ),
        encoding="utf-8",
    )
    profile = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert profile.manifest_file is None

    (tmp_path / "pom.xml").write_text(
        _pom(
            _coordinates(),
            f"<dependencies>{_dep(extra='<classifier>tests</classifier>')}</dependencies>",
        ),
        encoding="utf-8",
    )
    classifier = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert classifier.manifest_file is None

    other = tmp_path / "independent" / "pom.xml"
    other.parent.mkdir()
    other.write_text(_pom(_coordinates("com.other", "standalone")), encoding="utf-8")
    roots = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)
    assert roots.manifest_file is None


def test_outside_scan_path_does_not_fall_back_to_root(tmp_path: Path):
    (tmp_path / "pom.xml").write_text(_pom(_coordinates()), encoding="utf-8")
    issue = _maven_manifest_issue(raw_payload={"filePath": "/etc/maven/pom.xml"})
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
    issue = _maven_manifest_issue()

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

    localized = locate_maven_from_issue(_maven_manifest_issue(), tmp_path)

    assert localized.manifest_file == "pom.xml"
    assert localized.version_property_name == "widget.version"
    assert localized.version_property_file == "pom.xml"


def _python_manifest_issue(
    name: str, *, file_path: str = "/src/requirements.txt", ecosystem: str = "pypi"
) -> VulnerabilityIssue:
    return VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        severity=Severity.HIGH,
        package_name=name,
        package_version="1.0.0",
        ecosystem=ecosystem,
        purl=f"pkg:pypi/{name}@1.0.0" if ecosystem == "pypi" else None,
        file_path=file_path,
        raw_payload={"filePath": file_path},
    )


def test_pep503_normalization_and_pypi_purl_identity() -> None:
    assert normalize_python_package_name("Requests._Tool-Kit") == "requests-tool-kit"
    assert package_name_from_purl("pkg:pypi/Requests._Tool-Kit@2.31.0") == "requests-tool-kit"


def test_requirements_parser_retains_lines_and_ignores_directives(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# managed dependencies\n--index-url https://packages.example/simple\n"
        "  Requests_Toolkit >= 2.0  # keep this note\n-e git+https://example.invalid/repo\n"
        "other==1.0 \\\n  --hash=sha256:deadbeef\n",
        encoding="utf-8",
    )

    entries = parse_requirements_txt(path)

    assert len(entries) == 2
    first, second = entries
    assert isinstance(first, RequirementEntry)
    assert (first.path, first.line_number, first.name) == (path, 3, "Requests_Toolkit")
    assert isinstance(first.requirement, Requirement)
    assert first.raw_line == "  Requests_Toolkit >= 2.0  # keep this note"
    assert second.line_number == 5
    assert second.name == "other"


def test_requirements_include_directives_are_ignored_without_escaping_repo(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "outer" / "repo"
    repo.mkdir(parents=True)
    (repo / "requirements.txt").write_text("-r ../../outside.txt\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private-package==9.9\n", encoding="utf-8")

    assert parse_requirements_txt(repo / "requirements.txt") == []
    localized = locate_from_issue(_python_manifest_issue("private-package"), repo)
    assert localized.manifest_file == "requirements.txt"
    assert localized.is_direct_dependency is None


def test_requirements_update_preserves_comments_spacing_and_other_lines(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    original = "# top\n  Requests_Toolkit  >=  2.0  # pinned by policy\nflask==3.0\n"
    path.write_text(original, encoding="utf-8")

    updated = update_requirements_dependency(path, "requests.toolkit", "2.32.0")

    assert updated == "# top\n  Requests_Toolkit  ==  2.32.0  # pinned by policy\nflask==3.0\n"
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "declaration",
    [
        "requests==2.31.0 --hash=sha256:0123\n",
        "requests @ https://example.invalid/requests.whl\n",
        "requests==2.31.0; python_version < '3.12'\n",
        "requests[security]==2.31.0\n",
        "requests>=2.0,!=2.5\n",
    ],
)
def test_requirements_update_rejects_hash_and_unsupported_syntax(
    tmp_path: Path, declaration: str
) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text(declaration, encoding="utf-8")

    with pytest.raises(ValueError):
        update_requirements_dependency(path, "requests", "2.32.0")

    assert path.read_text(encoding="utf-8") == declaration


def test_requirements_lookup_fails_closed_for_ambiguous_names(tmp_path: Path) -> None:
    path = tmp_path / "requirements.txt"
    path.write_text("Requests_toolkit==1.0\nrequests-toolkit==2.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one"):
        update_requirements_dependency(path, "requests.toolkit", "3.0")


def test_duplicate_requirements_localization_is_not_an_authorized_direct_target(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "Requests_toolkit==1.0\nrequests-toolkit==2.0\n",
        encoding="utf-8",
    )

    localized = locate_from_issue(_python_manifest_issue("requests.toolkit"), tmp_path)

    assert localized.manifest_file == "requirements.txt"
    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None


def test_odc_scan_mount_path_localizes_to_requirements_file(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("PyJWT==2.4.0\n", encoding="utf-8")

    localized = locate_from_issue(
        _python_manifest_issue("PyJWT", file_path="/scan/requirements.txt:PyJWT/2.4.0"),
        tmp_path,
    )

    assert localized.manifest_file == "requirements.txt"
    assert localized.package_manager == "pip"
    assert localized.declaration_type == "requirements"
    assert localized.is_direct_dependency is True
    assert localized.manifest_line == 1


def test_nearest_requirements_manifest_wins_and_returns_typed_localization(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("requests==1.0\n", encoding="utf-8")
    nested = tmp_path / "services" / "api"
    nested.mkdir(parents=True)
    (nested / "requirements.txt").write_text("Requests_Toolkit==2.31.0\n", encoding="utf-8")

    localized = locate_from_issue(
        _python_manifest_issue(
            "Requests.Toolkit", file_path="/src/services/api/.venv/lib/site-packages/x.py"
        ),
        tmp_path,
    )

    assert localized.manifest_file == "services/api/requirements.txt"
    assert localized.package_manager == "pip"
    assert localized.declaration_type == "requirements"
    assert localized.is_direct_dependency is True
    assert localized.manifest_line == 1
    assert "Requests_Toolkit==2.31.0" in (localized.manifest_snippet or "")
    assert localized.issue.package_name == "requests-toolkit"
    assert localized.dependency_ancestry == []


def test_nearer_python_project_does_not_fall_through_to_ancestor_declaration(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    nested = tmp_path / "services" / "api"
    nested.mkdir(parents=True)
    (nested / "requirements.txt").write_text("flask==3.0\n", encoding="utf-8")

    localized = locate_from_issue(
        _python_manifest_issue("requests", file_path="/src/services/api/site-packages/pkg.py"),
        tmp_path,
    )

    assert localized.manifest_file == "services/api/requirements.txt"
    assert localized.package_manager == "pip"
    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None


def test_pyproject_direct_and_optional_declarations(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sample"\ndependencies = ["Requests_Toolkit>=2.0"]\n'
        '[project.optional-dependencies]\ndev = ["pytest>=8"]\n',
        encoding="utf-8",
    )

    direct = locate_from_issue(_python_manifest_issue("requests.toolkit"), tmp_path)
    optional = locate_from_issue(_python_manifest_issue("pytest"), tmp_path)

    assert direct.manifest_file == "pyproject.toml"
    assert direct.declaration_type == "dependencies"
    assert direct.is_direct_dependency is True
    assert direct.manifest_line == 3
    assert optional.declaration_type == "optional-dependencies"
    assert optional.is_direct_dependency is True
    assert optional.manifest_line == 5


def test_setup_cfg_install_and_extra_requirements_are_static_declarations(tmp_path: Path) -> None:
    (tmp_path / "setup.cfg").write_text(
        "[metadata]\nname = sample\n[options]\ninstall_requires =\n"
        "    Requests_Toolkit==2.0\n[options.extras_require]\ntest =\n"
        "    pytest>=8\n",
        encoding="utf-8",
    )

    install = locate_from_issue(_python_manifest_issue("requests-toolkit"), tmp_path)
    extra = locate_from_issue(_python_manifest_issue("pytest"), tmp_path)

    assert install.manifest_file == "setup.cfg"
    assert install.declaration_type == "install_requires"
    assert install.manifest_line == 5
    assert extra.declaration_type == "extras_require"
    assert extra.manifest_line == 8


def test_pipfile_direct_and_lock_versions_do_not_create_parent_edges(tmp_path: Path) -> None:
    (tmp_path / "Pipfile").write_text(
        '[packages]\nRequests_Toolkit = "*"\n[dev-packages]\nblack = "*"\n',
        encoding="utf-8",
    )
    (tmp_path / "Pipfile.lock").write_text(
        json.dumps(
            {
                "_meta": {},
                "default": {
                    "requests-toolkit": {"version": "==2.31.0"},
                    "urllib3": {"version": "==1.26.18"},
                    "unversioned": {},
                },
                "develop": {"black": {"version": "==24.2.0"}},
            }
        ),
        encoding="utf-8",
    )

    direct = locate_from_issue(
        _python_manifest_issue("REQUESTS.toolkit", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    transitive = locate_from_issue(
        _python_manifest_issue("urllib3", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    dev = locate_from_issue(
        _python_manifest_issue("black", file_path="/src/Pipfile.lock?/develop"), tmp_path
    )
    unversioned = locate_from_issue(
        _python_manifest_issue("unversioned", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    unknown = locate_from_issue(_python_manifest_issue("absent"), tmp_path)

    assert direct.manifest_file == "Pipfile"
    assert direct.package_manager == "pipenv"
    assert direct.declaration_type == "packages"
    assert direct.is_direct_dependency is True
    assert direct.manifest_line == 2
    assert direct.dependency_versions == {"requests-toolkit": "2.31.0"}
    assert transitive.is_direct_dependency is False
    assert transitive.declaration_type == "packages"
    assert transitive.dependency_versions == {"urllib3": "1.26.18"}
    assert transitive.dependency_ancestry == []
    assert transitive.parent_package_name is None
    assert unversioned.is_direct_dependency is False
    assert unversioned.declaration_type == "packages"
    assert unversioned.dependency_versions == {}
    assert dev.is_direct_dependency is True
    assert dev.declaration_type == "dev-packages"
    assert dev.dependency_versions == {"black": "24.2.0"}
    assert unknown.is_direct_dependency is None
    assert unknown.declaration_type is None


def test_pipfile_lock_membership_without_matching_category_hint_is_fail_closed(
    tmp_path: Path,
) -> None:
    (tmp_path / "Pipfile").write_text('[packages]\nknown = "*"\n', encoding="utf-8")
    (tmp_path / "Pipfile.lock").write_text(
        json.dumps(
            {
                "default": {"locked": {"version": "==1.2.3"}},
                "develop": {"locked": {"version": "==2.0.0"}},
            }
        ),
        encoding="utf-8",
    )

    localized = locate_from_issue(
        _python_manifest_issue("locked", file_path="/src/Pipfile.lock"), tmp_path
    )

    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None
    assert localized.dependency_versions == {}


def test_manifest_discovery_excludes_dynamic_setup_py_and_tool_only_pyproject(
    tmp_path: Path,
) -> None:
    (tmp_path / "setup.py").write_text("setup(install_requires=dynamic())\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = "-q"\n', encoding="utf-8"
    )
    (tmp_path / "unsupported.txt").write_text("requests==2.0\n", encoding="utf-8")

    assert locate_python_manifests(tmp_path) == []
    assert locate_from_issue(_python_manifest_issue("requests"), tmp_path).manifest_file is None


def test_manifest_selection_rejects_outside_and_symlink_escape_paths(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "requirements.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("requests==9.9\n", encoding="utf-8")
    outside_issue = _python_manifest_issue("requests", file_path=str(outside))
    escaped = tmp_path / "linked"

    escaped.symlink_to(outside.parent, target_is_directory=True)
    symlink_issue = _python_manifest_issue("requests", file_path="linked/requirements.txt")

    assert locate_from_issue(outside_issue, tmp_path).manifest_file is None
    assert locate_from_issue(symlink_issue, tmp_path).manifest_file is None


@pytest.mark.parametrize(
    ("manifest_name", "content"),
    [
        (
            "pyproject.toml",
            '[project]\ndependencies = ["requests==1.0"]\n'
            '[project.optional-dependencies]\ntest = ["requests==2.0"]\n',
        ),
        (
            "setup.cfg",
            "[options]\ninstall_requires = requests==1.0\n"
            "[options.extras_require]\ntest = requests==2.0\n",
        ),
        (
            "Pipfile",
            '[packages]\nrequests = "*"\n[dev-packages]\nrequests = "*"\n',
        ),
    ],
)
def test_ambiguous_manifest_declarations_fail_closed(
    tmp_path: Path,
    manifest_name: str,
    content: str,
) -> None:
    (tmp_path / manifest_name).write_text(content, encoding="utf-8")

    localized = locate_from_issue(_python_manifest_issue("requests"), tmp_path)

    assert localized.manifest_file == manifest_name
    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None


def test_non_pypi_issue_is_not_localized_through_python_manifests(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("lodash==1.0\n", encoding="utf-8")

    localized = locate_from_issue(_python_manifest_issue("lodash", ecosystem="npm"), tmp_path)

    assert localized.manifest_file is None
    assert localized.is_direct_dependency is None
    assert localized.issue.package_name == "lodash"


GAV = "org.example:widget"
_JAVA_TOOLS_NAMESPACE = 'xmlns="http://maven.apache.org/POM/4.0.0"'


def _java_tool_issue(**updates) -> VulnerabilityIssue:
    values = {
        "source": IssueSource.ODC,
        "issue_type": IssueType.SCA,
        "severity": Severity.HIGH,
        "ecosystem": "maven",
        "package_name": GAV,
        "package_version": "1.0.0",
        "purl": "pkg:maven/org.example/widget@1.0.0",
        "raw_payload": {"filePath": "/scan/pom.xml"},
    }
    values.update(updates)
    return VulnerabilityIssue(**values)


def _dependency(version: str | None = "1.0.0", extra: str = "") -> str:
    declared = f"<version>{version}</version>" if version is not None else ""
    return (
        "<dependency><groupId>org.example</groupId><artifactId>widget</artifactId>"
        f"{declared}{extra}</dependency>"
    )


def _java_tool_pom(body: str = "", *, artifact: str = "app") -> str:
    return (
        f"<project {_JAVA_TOOLS_NAMESPACE}>\n"
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <groupId>com.acme</groupId>\n"
        f"  <artifactId>{artifact}</artifactId>\n"
        "  <version>1.0.0</version>\n"
        f"{body}"
        "</project>\n"
    )


class _Sandbox:
    def __init__(
        self, files: dict[str, str], *, sync_exit: int = 0, raise_sync: Exception | None = None
    ):
        self.files = dict(files)
        self.sync_exit = sync_exit
        self.raise_sync = raise_sync
        self.commands: list[str] = []
        self.writes: list[tuple[str, str]] = []

    def read_file(self, path: str) -> str | None:
        return self.files.get(path)

    def write_file(self, path: str, content: str) -> None:
        self.files[path] = content
        self.writes.append((path, content))

    def run(self, command: str, timeout: int = 0):
        self.commands.append(command)
        if command.startswith("rm -f -- "):
            return SimpleNamespace(exit_code=0, stdout="", stderr="")
        if self.raise_sync is not None:
            raise self.raise_sync
        return SimpleNamespace(
            exit_code=self.sync_exit,
            stdout="simulated stdout",
            stderr="simulated stderr" if self.sync_exit else "",
        )


def _updater(
    sandbox: _Sandbox,
    touched: set[str],
    *,
    paths: tuple[str, ...] = ("pom.xml",),
    version: str = "2.0.0",
    dependency_type: str = "dependencies",
    state: dict | None = None,
    operation: MavenTargetOperation = MavenTargetOperation.UPDATE_DECLARATION,
):
    return _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        touched,
        {GAV: paths},
        {GAV: {version}},
        {GAV: {dependency_type}},
        execution_state=state,
        maven_target_operations_by_package={GAV: {operation.value}},
    )


def _multi_maven_action() -> MultiPackageAction:
    return MultiPackageAction(
        cluster_id="cluster-maven",
        dispatch_batch_id="batch-maven",
        selected_strategy=TacticalStrategy.VERSION_BUMP,
        package_mutations=[
            PackageMutation(
                task_id="task-widget",
                package_name="org.example:widget",
                ecosystem="maven",
                manifest_path="module-a/pom.xml",
                target_manifest_paths=["module-a/pom.xml"],
                maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
                target_version="2.0.0",
                dependency_type="dependencies",
            ),
            PackageMutation(
                task_id="task-library",
                package_name="org.example:library",
                ecosystem="maven",
                manifest_path="module-b/pom.xml",
                target_manifest_paths=["module-b/pom.xml"],
                maven_target_operation=MavenTargetOperation.UPDATE_DECLARATION,
                target_version="2.0.0",
                dependency_type="dependencies",
            ),
        ],
        rationale="Update the committed Maven package cluster atomically.",
    )


def _two_module_poms() -> dict[str, str]:
    widget = (
        "<dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>widget</artifactId><version>1.0.0</version>"
        "</dependency></dependencies>"
    )
    library = (
        "<dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>library</artifactId><version>1.0.0</version>"
        "</dependency></dependencies>"
    )
    return {
        "module-a/pom.xml": _java_tool_pom(widget, artifact="module-a"),
        "module-b/pom.xml": _java_tool_pom(library, artifact="module-b"),
    }


class _FailSecondModuleSync(_Sandbox):
    def __init__(self, files: dict[str, str], *, fail_first_restore: bool = False):
        super().__init__(files)
        self.original_files = dict(files)
        self.fail_first_restore = fail_first_restore
        self.second_sync_saw_first_edit = False

    def run(self, command: str, timeout: int = 0):
        self.commands.append(command)
        if "-f module-b/pom.xml" in command:
            self.second_sync_saw_first_edit = (
                "<version>2.0.0</version>" in self.files["module-a/pom.xml"]
            )
            return SimpleNamespace(
                exit_code=1,
                stdout="second reactor failed",
                stderr="dependency resolution rejected",
            )
        return SimpleNamespace(exit_code=0, stdout="ok", stderr="")

    def write_file(self, path: str, content: str) -> None:
        if (
            self.fail_first_restore
            and path == "module-a/pom.xml"
            and content == self.original_files[path]
        ):
            self.fail_first_restore = False
            raise OSError("simulated first-module restore failure")
        super().write_file(path, content)


def test_multi_maven_action_rolls_back_all_poms_after_later_package_failure() -> None:
    baseline = _two_module_poms()
    sandbox = _FailSecondModuleSync(baseline)
    touched_files = {"preexisting.java"}
    checkpoints: dict[str, object] = {}

    succeeded, error = apply_multi_maven_package_action(
        sandbox,
        _multi_maven_action(),
        touched_files,
        checkpoint_store=checkpoints,
    )

    assert succeeded is False
    assert "Rollback complete" in error
    assert sandbox.second_sync_saw_first_edit is True
    assert sandbox.files == baseline
    assert touched_files == {"preexisting.java"}
    assert checkpoints == {}


def test_multi_maven_action_preserves_partial_restore_evidence() -> None:
    baseline = _two_module_poms()
    sandbox = _FailSecondModuleSync(baseline, fail_first_restore=True)
    touched_files: set[str] = set()
    checkpoints: dict[str, object] = {}

    succeeded, error = apply_multi_maven_package_action(
        sandbox,
        _multi_maven_action(),
        touched_files,
        checkpoint_store=checkpoints,
    )

    assert succeeded is False
    assert "Rollback incomplete" in error
    assert "module-a/pom.xml" in touched_files
    assert "__multi_package_action__" in checkpoints
    assert "<version>2.0.0</version>" in sandbox.files["module-a/pom.xml"]


def _invoke(updater, *, version="2.0.0", kind="dependencies", path="pom.xml", package=GAV):
    return updater.invoke(
        {
            "package_name": package,
            "target_version": version,
            "dependency_type": kind,
            "manifest_path": path,
        }
    )


def _module_pom(name: str, body: str, *, parent: str = "../pom.xml") -> str:
    return (
        f"<project {_JAVA_TOOLS_NAMESPACE}>\n"
        "  <modelVersion>4.0.0</modelVersion>\n"
        "  <parent><groupId>com.acme</groupId><artifactId>reactor</artifactId>"
        f"<version>1.0.0</version><relativePath>{parent}</relativePath></parent>\n"
        f"  <artifactId>{name}</artifactId>\n"
        f"{body}"
        "</project>\n"
    )


def test_direct_explicit_version_is_updated_in_place_and_synced_with_quoted_pom():
    path = "service with space/pom.xml"
    original = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({path: original})
    touched = {"src/App.java"}
    updater = _updater(sandbox, touched, paths=(path,))

    result = _invoke(updater, path=path)

    assert result.startswith("SUCCESS:")
    assert "<version>2.0.0</version>" in sandbox.files[path]
    assert "<artifactId>app</artifactId>" in sandbox.files[path]
    assert touched == {"src/App.java", path}
    assert sandbox.commands == ["mvn -B -q -f 'service with space/pom.xml' dependency:resolve"]
    assert updater.name == "modify_and_validate_maven_dependency"
    assert "manifest_path" in updater.args


def test_direct_managed_and_transitive_override_use_dependency_management():
    managed_pom = _java_tool_pom(
        "  <dependencyManagement><dependencies>"
        f"{_dependency('1.0.0')}"
        "</dependencies></dependencyManagement>\n"
        f"  <dependencies>{_dependency(None)}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": managed_pom})
    result = _invoke(
        _updater(sandbox, set(), dependency_type="dependencyManagement"),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert sandbox.files["pom.xml"].count("<version>2.0.0</version>") == 1
    assert "<dependencies>" in sandbox.files["pom.xml"]

    transitive = _java_tool_pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>other</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": transitive})
    result = _invoke(
        _updater(
            sandbox,
            set(),
            dependency_type="dependencyManagement",
            operation=MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT,
        ),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert "<dependencyManagement>" in sandbox.files["pom.xml"]
    assert (
        "<groupId>org.example</groupId><artifactId>widget</artifactId>" in sandbox.files["pom.xml"]
    )
    assert "<version>2.0.0</version>" in sandbox.files["pom.xml"]


def test_external_bom_is_overridden_by_new_local_managed_entry_only():
    bom_import = (
        "<dependency><groupId>org.example</groupId><artifactId>widgets-bom</artifactId>"
        "<version>5.0.0</version><type>pom</type><scope>import</scope></dependency>"
    )
    root = _java_tool_pom(
        "  <dependencyManagement><dependencies>"
        f"{bom_import}"
        "</dependencies></dependencyManagement>\n"
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>widget</artifactId></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": root})
    result = _invoke(
        _updater(
            sandbox,
            set(),
            dependency_type="dependencyManagement",
            operation=MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT,
        ),
        kind="dependencyManagement",
    )

    assert result.startswith("SUCCESS:")
    assert "widgets-bom" in sandbox.files["pom.xml"]
    assert sandbox.files["pom.xml"].count("<artifactId>widget</artifactId>") == 2
    assert sandbox.files["pom.xml"].count("<version>2.0.0</version>") == 1


def test_unsafe_transitive_insertion_fails_without_rewriting_inline_pom():
    source = (
        f"<project {_JAVA_TOOLS_NAMESPACE}><modelVersion>4.0.0</modelVersion><groupId>com.acme</groupId>"
        "<artifactId>app</artifactId><version>1.0.0</version></project>"
    )
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(
        _updater(
            sandbox,
            set(),
            dependency_type="dependencyManagement",
            operation=MavenTargetOperation.ENSURE_DEPENDENCY_MANAGEMENT,
        ),
        kind="dependencyManagement",
    )

    assert "ERROR_CODE: POM_INSERTION_UNSAFE:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_parent_property_owner_and_multi_module_different_properties_are_updated():
    root = _java_tool_pom(
        "  <modules><module>one</module><module>two</module></modules>\n",
        artifact="reactor",
    )
    one = _module_pom(
        "one",
        "  <properties><one.version>1.0.0</one.version></properties>\n"
        f"  <dependencies>{_dependency('${one.version}')}</dependencies>\n",
    )
    two = _module_pom(
        "two",
        "  <properties><two.version>1.0.0</two.version></properties>\n"
        f"  <dependencies>{_dependency('${two.version}')}</dependencies>\n",
    )
    paths = ("pom.xml", "one/pom.xml", "two/pom.xml")
    sandbox = _Sandbox({"pom.xml": root, "one/pom.xml": one, "two/pom.xml": two})
    updater = _updater(sandbox, set(), paths=paths)

    result = _invoke(updater, path="one/pom.xml")

    assert result.startswith("SUCCESS:")
    assert "<one.version>2.0.0</one.version>" in sandbox.files["one/pom.xml"]
    assert "<two.version>2.0.0</two.version>" in sandbox.files["two/pom.xml"]


def test_parent_property_owner_is_updated_only_in_authorized_parent_pom():
    root = _java_tool_pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        "  <modules><module>module</module></modules>\n",
        artifact="reactor",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
    )
    sandbox = _Sandbox({"pom.xml": root, "module/pom.xml": child})
    touched: set[str] = set()
    updater = _updater(sandbox, touched, paths=("pom.xml", "module/pom.xml"))

    result = _invoke(updater, path="module/pom.xml")

    assert result.startswith("SUCCESS:")
    assert "<widget.version>2.0.0</widget.version>" in sandbox.files["pom.xml"]
    assert sandbox.files["module/pom.xml"] == child
    assert touched == {"pom.xml"}


def test_parent_owned_property_is_localized_and_normalized_as_repo_relative(tmp_path: Path):
    root = tmp_path / "pom.xml"
    root.write_text(
        _java_tool_pom(
            "  <properties><widget.version>1.0.0</widget.version></properties>\n"
            "  <modules><module>module</module></modules>\n",
            artifact="reactor",
        ),
        encoding="utf-8",
    )
    module_dir = tmp_path / "module"
    module_dir.mkdir()
    (module_dir / "pom.xml").write_text(
        _module_pom(
            "module",
            f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
        ),
        encoding="utf-8",
    )
    localized = locate_maven_from_issue(
        _java_tool_issue(raw_payload={"filePath": "/scan/module/src/main/java/Widget.java"}),
        tmp_path,
    )
    group = VulnerabilityGroup(
        group_id="sca:maven:org.example:widget",
        issue_type=IssueType.SCA,
        vulnerable_component=GAV,
        file_path=localized.manifest_file,
        file_paths=[localized.manifest_file],
        representative_issue_id=localized.issue.id,
        issues=[localized.issue],
        localized_issues=[localized.model_copy(update={"version_property_file": str(root)})],
    )

    normalized = normalize_group_paths([group], str(tmp_path))[0]

    assert localized.version_property_name == "widget.version"
    assert localized.version_property_file == "pom.xml"
    assert normalized.localized_issues[0].version_property_file == "pom.xml"


def test_property_shared_and_unresolved_fail_closed_without_writes():
    shared = _java_tool_pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
        "  <name>${widget.version}</name>\n"
    )
    sandbox = _Sandbox({"pom.xml": shared})
    touched = {"src/Main.java"}
    result = _invoke(_updater(sandbox, touched))

    assert "ERROR_CODE: POM_PROPERTY_SHARED:" in result
    assert sandbox.files["pom.xml"] == shared
    assert touched == {"src/Main.java"}
    assert sandbox.commands == []

    unresolved = _java_tool_pom(
        f"  <dependencies>{_dependency('${external.version}')}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": unresolved})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_PROPERTY_UNRESOLVED:" in result
    assert sandbox.files["pom.xml"] == unresolved

    cyclic = _java_tool_pom(
        "  <properties><widget.version>${widget.version}</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": cyclic})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_PROPERTY_UNRESOLVED:" in result
    assert sandbox.files["pom.xml"] == cyclic


@pytest.mark.parametrize(
    "dependency, extra_body",
    [
        (_dependency() + _dependency(), ""),
        (
            _dependency(),
            "  <profiles><profile><id>active</id><dependencies>"
            + _dependency()
            + "</dependencies></profile></profiles>\n",
        ),
        (_dependency(extra="<classifier>tests</classifier>"), ""),
    ],
)
def test_duplicate_profile_and_classifier_targets_are_ambiguous(dependency, extra_body):
    source = _java_tool_pom(f"  <dependencies>{dependency}</dependencies>\n{extra_body}")
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_TARGET_AMBIGUOUS:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


@pytest.mark.parametrize(
    "source",
    [
        '<!DOCTYPE project [<!ENTITY x "boom">]><project><modelVersion>4.0.0</modelVersion></project>',
        "<project><modelVersion>4.0.0</project>",
    ],
)
def test_dtd_and_malformed_pom_fail_with_invalid_xml(source):
    sandbox = _Sandbox({"pom.xml": source})
    result = _invoke(_updater(sandbox, set()))

    assert "ERROR_CODE: POM_INVALID_XML:" in result
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_version_type_gav_and_path_authorization_fail_closed():
    source = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    updater = _updater(sandbox, set())

    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, package="org.example:widget;id")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, version="2.0.0;touch /tmp/x")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, version="2.1.0-rc1")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, path="not-authorized/pom.xml")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, version="3.0.0")
    assert "ERROR_CODE: INVALID_ARGUMENT:" in _invoke(updater, path="../pom.xml")
    assert "ERROR_CODE: TARGET_NOT_ALLOWED:" in _invoke(updater, kind="dependencyManagement")
    assert sandbox.files["pom.xml"] == source
    assert sandbox.commands == []


def test_failed_maven_sync_restores_every_authorized_pom_and_touched_files():
    root = _java_tool_pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        "  <modules><module>module</module></modules>\n",
        artifact="reactor",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n",
    )
    sandbox = _Sandbox(
        {"pom.xml": root, "module/pom.xml": child},
        sync_exit=1,
    )
    touched = {"src/AlreadyChanged.java", "module/pom.xml"}
    result = _invoke(
        _updater(sandbox, touched, paths=("pom.xml", "module/pom.xml")),
        path="module/pom.xml",
    )

    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files == {"pom.xml": root, "module/pom.xml": child}
    assert touched == {"src/AlreadyChanged.java", "module/pom.xml"}
    assert len(sandbox.commands) == 1
    assert "dependency:resolve" in sandbox.commands[0]


def test_maven_sync_timeout_restores_the_pom_checkpoint():
    source = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source}, raise_sync=TimeoutError("timed out"))
    touched = {"src/Existing.java"}

    result = _invoke(_updater(sandbox, touched))

    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files["pom.xml"] == source
    assert touched == {"src/Existing.java"}


def test_success_marks_only_changed_authorized_poms_touched():
    source = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    touched = {"src/Existing.java"}
    result = _invoke(_updater(sandbox, touched))

    assert result.startswith("SUCCESS:")
    assert touched == {"src/Existing.java", "pom.xml"}
    assert len(sandbox.commands) == 1


def test_direct_only_no_fix_removal_and_sync_rollback():
    source = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    plan = {
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
        "recorded": True,
        "package_removal_planned": True,
    }
    sandbox = _Sandbox({"pom.xml": source})
    touched: set[str] = set()
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox,
        touched,
        plan,
        GAV,
        ["pom.xml"],
        GAV,
        "pom.xml",
    )

    assert result.startswith("SUCCESS:")
    assert "<artifactId>widget</artifactId>" not in sandbox.files["pom.xml"]
    assert touched == {"pom.xml"}
    assert plan["package_removal_completed"] is True
    assert plan["phase"] == "VALIDATE"

    transitive = _java_tool_pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>other</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": transitive})
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox, set(), plan, GAV, ["pom.xml"], GAV, "pom.xml"
    )
    assert "ERROR_CODE: EDIT_FAILED:" in result
    assert sandbox.files["pom.xml"] == transitive
    assert sandbox.commands == []

    sandbox = _Sandbox({"pom.xml": source}, sync_exit=1)
    touched = {"src/Main.java"}
    result = _remove_maven_no_fix_dependency_transaction(
        sandbox, touched, plan, GAV, ["pom.xml"], GAV, "pom.xml"
    )
    assert "ERROR_CODE: MANIFEST_SYNC_FAILED:" in result
    assert sandbox.files["pom.xml"] == source
    assert touched == {"src/Main.java"}


def test_byte_span_helpers_escape_and_preserve_unrelated_xml():
    source = _java_tool_pom(
        "  <properties><widget.version>1.0.0</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    updated = update_pom_property_value(source, "widget.version", "2.0.0")
    assert "<widget.version>2.0.0</widget.version>" in updated
    assert "<artifactId>app</artifactId>" in updated

    escaped = update_pom_dependency_version(
        _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n"),
        "org.example",
        "widget",
        "2<&0",
    )
    assert "<version>2&lt;&amp;0</version>" in escaped
    managed = add_dependency_management_entry(_java_tool_pom(), "org.example", "widget", "2.0.0")
    assert "<dependencyManagement>" in managed
    removed = remove_direct_dependency_entry(
        _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n"),
        "org.example",
        "widget",
    )
    assert "<artifactId>widget</artifactId>" not in removed


def test_update_retry_signature_and_three_attempt_limit_are_per_gav():
    source = _java_tool_pom(f"  <dependencies>{_dependency()}</dependencies>\n")
    sandbox = _Sandbox({"pom.xml": source})
    touched: set[str] = set()
    updater = _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        touched,
        {GAV: ["pom.xml"]},
        {GAV: {"2.0.0", "2.1.0", "2.2.0", "2.3.0"}},
        {GAV: {"dependencies"}},
        execution_state={},
    )

    assert _invoke(updater, version="2.0.0").startswith("SUCCESS:")
    assert "ERROR_CODE: RETRY_PARAMETERS_UNCHANGED:" in _invoke(updater, version="2.0.0")
    assert _invoke(updater, version="2.1.0").startswith("SUCCESS:")
    assert _invoke(updater, version="2.2.0").startswith("SUCCESS:")
    assert "ERROR_CODE: RETRY_LIMIT_REACHED:" in _invoke(updater, version="2.3.0")
    assert len(sandbox.commands) == 3


def test_checkpoint_is_frozen_and_captures_pretransaction_touched_set():
    checkpoint = _MavenPackageCheckpoint({"pom.xml": "<project/>"}, {"A.java"})

    assert checkpoint.files == {"pom.xml": "<project/>"}
    assert checkpoint.touched_files_before == {"A.java"}


def test_repeated_property_references_are_resolved_without_false_cycle() -> None:
    source = _java_tool_pom(
        "  <properties><base.version>1.0.0</base.version>"
        "<widget.version>${base.version}${base.version}</widget.version></properties>\n"
        f"  <dependencies>{_dependency('${widget.version}')}</dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": source})

    result = _invoke(_updater(sandbox, set()))

    assert result.startswith("SUCCESS:")
    assert "<widget.version>2.0.0</widget.version>" in sandbox.files["pom.xml"]
    assert "<base.version>1.0.0</base.version>" in sandbox.files["pom.xml"]


def test_local_parent_relative_path_must_match_parent_coordinates() -> None:
    root = _java_tool_pom(
        "  <modules><module>module</module></modules>\n",
        artifact="unrelated",
    )
    child = _module_pom(
        "module",
        f"  <dependencies>{_dependency()}</dependencies>\n",
    )
    sandbox = _Sandbox({"pom.xml": root, "module/pom.xml": child})
    updater = _updater(sandbox, set(), paths=("pom.xml", "module/pom.xml"))

    result = _invoke(updater, path="module/pom.xml")

    assert "ERROR_CODE: POM_TARGET_AMBIGUOUS:" in result
    assert sandbox.files == {"pom.xml": root, "module/pom.xml": child}
    assert sandbox.commands == []


def test_single_character_artifact_coordinate_is_authorized() -> None:
    coordinate = "org.example:x"
    source = _java_tool_pom(
        "  <dependencies><dependency><groupId>org.example</groupId>"
        "<artifactId>x</artifactId><version>1.0.0</version></dependency></dependencies>\n"
    )
    sandbox = _Sandbox({"pom.xml": source})
    updater = _make_modify_and_validate_maven_dependency_tool(
        sandbox,
        set(),
        {coordinate: ["pom.xml"]},
        {coordinate: {"2.0.0"}},
        {coordinate: {"dependencies"}},
    )

    result = _invoke(updater, package=coordinate)

    assert result.startswith("SUCCESS:")
    assert "<version>2.0.0</version>" in sandbox.files["pom.xml"]


def _result(exit_code: int = 0, stdout: str = "ok", stderr: str = "") -> CommandResult:
    return CommandResult(
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=0.1,
    )


def _python_update_tool(
    sandbox,
    *,
    manifest: str = "requirements.txt",
    allowed_versions=None,
    allowed_types=None,
    execution_state=None,
    package_checkpoints=None,
    touched_files=None,
):
    tool = _make_modify_and_validate_python_dependency_tool(
        sandbox,
        touched_files if touched_files is not None else set(),
        {"requests": [manifest]},
        allowed_target_versions_by_package=(
            allowed_versions if allowed_versions is not None else {"requests": ["2.32.0"]}
        ),
        allowed_dependency_types_by_package=(
            allowed_types if allowed_types is not None else {"requests": ["requirements"]}
        ),
        execution_state=execution_state,
        package_checkpoints=package_checkpoints,
    )
    return tool


def _files_sandbox(files: dict[str, str], *, run_handler=None, write_handler=None):
    sandbox = MagicMock()
    sandbox.read_file.side_effect = lambda path: files.get(path)

    def write(path, content):
        if write_handler is not None:
            write_handler(path, content)
        files[path] = content

    sandbox.write_file.side_effect = write
    sandbox.run.side_effect = run_handler or (lambda *_args, **_kwargs: _result())
    return sandbox


def _projection(result: str) -> list[str]:
    marker = "JSON: "
    assert marker in result
    return json.loads(result.split(marker, 1)[1])["changed_files"]


def test_python_manifest_targets_and_checkpoints_exclude_npm_locks():
    assert _normalize_python_manifest_targets(
        ["requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg", "Pipfile"]
    ) == ["Pipfile", "pyproject.toml", "requirements-dev.txt", "requirements.txt", "setup.cfg"]
    assert _package_checkpoint_paths(["requirements-dev.txt"], package_ecosystem="pypi") == [
        "requirements-dev.txt"
    ]
    assert _package_checkpoint_paths(["Pipfile"], package_ecosystem="pypi") == [
        "Pipfile",
        "Pipfile.lock",
    ]
    with pytest.raises(ValueError):
        _normalize_python_manifest_targets(["setup.py"])


def test_authorized_requirements_update_validates_and_reports_changed_files():
    files = {"requirements.txt": "# keep\nrequests>=2.31.0\n"}
    sandbox = _files_sandbox(files)
    touched: set[str] = set()
    tool = _python_update_tool(sandbox, touched_files=touched)

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "requirements",
            "manifest_path": "requirements.txt",
        }
    )

    assert result.startswith("SUCCESS:")
    assert files["requirements.txt"] == "# keep\nrequests==2.32.0\n"
    assert "pip install --dry-run -r /workspace/requirements.txt" in sandbox.run.call_args.args[0]
    assert touched == {"requirements.txt"}
    assert _projection(result) == ["requirements.txt"]


def test_pep621_and_setup_cfg_transactions_edit_only_the_approved_declaration():
    cases = (
        (
            "pyproject.toml",
            '[project]\nname = "sample"\ndependencies = ["requests>=2.31.0"]\n',
            "dependencies",
        ),
        (
            "pyproject.toml",
            '[project]\nname = "sample"\n[project.optional-dependencies]\ntest = ["requests>=2.31.0"]\n',
            "optional-dependencies",
        ),
        (
            "setup.cfg",
            "[metadata]\nname = sample\n[options]\ninstall_requires =\n    requests>=2.31.0\n",
            "install_requires",
        ),
    )

    for manifest, original, declaration_type in cases:
        files = {manifest: original}
        sandbox = _files_sandbox(files)
        tool = _python_update_tool(
            sandbox,
            manifest=manifest,
            allowed_types={"requests": [declaration_type]},
        )

        result = tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": declaration_type,
                "manifest_path": manifest,
            }
        )

        assert result.startswith("SUCCESS:")
        assert "requests==2.32.0" in files[manifest]
        assert _projection(result) == [manifest]


def test_unapproved_manifest_candidate_and_declaration_type_do_not_mutate():
    files = {"requirements.txt": "requests==2.31.0\n", "other.txt": "requests==2.31.0\n"}
    sandbox = _files_sandbox(files)
    tool = _python_update_tool(sandbox)

    rejected = [
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.33.0",
                "dependency_type": "requirements",
                "manifest_path": "requirements.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": "dependencies",
                "manifest_path": "requirements.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "requests",
                "target_version": "2.32.0",
                "dependency_type": "requirements",
                "manifest_path": "other.txt",
            }
        ),
        tool.invoke(
            {
                "package_name": "Requests",
                "target_version": "2.32.0",
                "dependency_type": "requirements",
                "manifest_path": "requirements.txt",
            }
        ),
    ]

    assert all(result.startswith("ERROR_CODE:") for result in rejected)
    assert files == {"requirements.txt": "requests==2.31.0\n", "other.txt": "requests==2.31.0\n"}
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_not_called()


def test_failed_python_validation_restores_manifest_and_touched_projection():
    baseline = "requests==2.31.0\n"
    files = {"requirements.txt": baseline}
    failure = _result(1, "", "resolution failed")
    sandbox = _files_sandbox(files, run_handler=lambda *_args, **_kwargs: failure)
    touched = {"src/module.py"}
    tool = _python_update_tool(sandbox, touched_files=touched)

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "requirements",
            "manifest_path": "requirements.txt",
        }
    )

    assert result.startswith("ERROR_CODE: PYTHON_TRANSACTION_FAILED:")
    assert "resolution failed" in result
    assert "Rollback: package checkpoint restored." in result
    assert files["requirements.txt"] == baseline
    assert touched == {"src/module.py"}


def test_pipfile_update_locks_then_syncs_and_attributes_both_files():
    files = {
        "Pipfile": '[packages]\nrequests = "*"\n',
        "Pipfile.lock": '{"default": {"requests": {"version": "==2.31.0"}}}\n',
    }
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = '{"default": {"requests": {"version": "==2.32.0"}}}\n'
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    touched: set[str] = set()
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
        touched_files=touched,
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert result.startswith("SUCCESS:")
    assert files["Pipfile"] == '[packages]\nrequests = "==2.32.0"\n'
    assert "pipenv lock" in commands[0]
    assert "PIPENV_VENV_IN_PROJECT=1" in commands[1]
    assert "pipenv sync --dev" in commands[1]
    assert touched == {"Pipfile", "Pipfile.lock"}
    assert _projection(result) == ["Pipfile", "Pipfile.lock"]


def test_pipfile_sync_failure_restores_manifest_lock_and_resyncs_saved_lock():
    baseline_pipfile = '[packages]\nrequests = "*"\n'
    baseline_lock = '{"default": {"requests": {"version": "==2.31.0"}}}\n'
    files = {"Pipfile": baseline_pipfile, "Pipfile.lock": baseline_lock}
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = "generated lock\n"
        if "pipenv sync --dev" in command and len(commands) == 2:
            return _result(1, "partial sync", "sync failure")
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    touched = {"src/app.py"}
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
        touched_files=touched,
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert "sync failure" in result
    assert "Rollback: package checkpoint restored." in result
    assert files == {"Pipfile": baseline_pipfile, "Pipfile.lock": baseline_lock}
    assert "pipenv sync --dev" in commands[-1]
    assert touched == {"src/app.py"}


def test_pipfile_rollback_restore_failure_is_reported_explicitly():
    baseline_lock = '{"default": {"requests": {"version": "==2.31.0"}}}\n'
    files = {"Pipfile": '[packages]\nrequests = "*"\n', "Pipfile.lock": baseline_lock}

    def write(path, _content):
        if path == "Pipfile.lock":
            raise OSError("lock restore denied")

    def run(command, **_kwargs):
        if "pipenv lock" in command:
            files["Pipfile.lock"] = "changed lock\n"
        if "pipenv sync --dev" in command and files["Pipfile.lock"] != baseline_lock:
            return _result(1, "", "rollback sync failed")
        if "pipenv sync --dev" in command:
            return _result(1, "", "update sync failed")
        return _result()

    sandbox = _files_sandbox(files, run_handler=run, write_handler=write)
    execution_state = {}
    package_checkpoints = {}
    tool = _python_update_tool(
        sandbox,
        manifest="Pipfile",
        allowed_types={"requests": ["packages"]},
        execution_state=execution_state,
        package_checkpoints=package_checkpoints,
    )

    result = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )

    assert "Rollback failed" in result
    assert "lock restore denied" in result
    assert "Pipenv rollback sync failed" in result
    assert "requests" in package_checkpoints
    assert execution_state["restoration_failed"] is True
    blocked = tool.invoke(
        {
            "package_name": "requests",
            "target_version": "2.32.0",
            "dependency_type": "packages",
            "manifest_path": "Pipfile",
        }
    )
    assert "further dependency edits are blocked" in blocked


def test_pipfile_manifest_removal_is_direct_only_and_runs_lock_sync():
    files = {
        "Pipfile": '[packages]\nrequests = "*"\n[dev-packages]\npytest = "*"\n',
        "Pipfile.lock": '{"default": {}, "develop": {}}\n',
    }
    commands: list[str] = []

    def run(command, **_kwargs):
        commands.append(command)
        if "pipenv lock" in command:
            files["Pipfile.lock"] = '{"default": {}, "develop": {}}\\n'
        return _result()

    sandbox = _files_sandbox(files, run_handler=run)
    plan_state = {
        "recorded": True,
        "package_removal_planned": True,
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
    }
    toolbelt = build_workaround_toolbelt(
        sandbox,
        set(),
        Path("/tmp/repo"),
        plan_state=plan_state,
        no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        no_fix_package_name="requests",
        no_fix_manifest_paths=["Pipfile"],
        no_fix_package_manager="pipenv",
        language=ProjectLanguage.PYTHON,
    )
    tools = {item.name: item for item in toolbelt}
    assert "remove_no_fix_python_dependency" in tools
    assert "remove_no_fix_dependency" not in tools

    result = tools["remove_no_fix_python_dependency"].invoke(
        {"package_name": "requests", "manifest_path": "Pipfile"}
    )

    assert result.startswith("SUCCESS:")
    assert "requests" not in files["Pipfile"]
    assert "pipenv lock" in commands[0]
    assert "pipenv sync --dev" in commands[1]
    assert _projection(result) == ["Pipfile", "Pipfile.lock"]


def test_python_removal_rejects_undeclared_package_without_editing():
    files = {"requirements.txt": "flask==3.0\n"}
    sandbox = _files_sandbox(files)
    plan_state = {
        "recorded": True,
        "package_removal_planned": True,
        "no_fix_stage": NoFixMitigationStage.PACKAGE_REMOVAL.value,
    }
    toolbelt = build_workaround_toolbelt(
        sandbox,
        set(),
        Path("/tmp/repo"),
        plan_state=plan_state,
        no_fix_stage=NoFixMitigationStage.PACKAGE_REMOVAL,
        no_fix_package_name="requests",
        no_fix_manifest_paths=["requirements.txt"],
        no_fix_package_manager="pip",
        language=ProjectLanguage.PYTHON,
    )
    tool = next(item for item in toolbelt if item.name == "remove_no_fix_python_dependency")

    result = tool.invoke({"package_name": "requests", "manifest_path": "requirements.txt"})

    assert result.startswith("NOT_APPLICABLE:")
    assert files["requirements.txt"] == "flask==3.0\n"
    sandbox.write_file.assert_not_called()
    sandbox.run.assert_not_called()
