from __future__ import annotations

import json
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from remediation_engine.contracts import IssueSource, IssueType, Severity, VulnerabilityIssue
from remediation_engine.tools.package_identity import (
    normalize_python_package_name,
    package_name_from_purl,
)
from remediation_engine.tools.python_manifest_locator import (
    RequirementEntry,
    find_dependency_in_requirements,
    locate_from_issue,
    locate_python_manifests,
    parse_requirements_txt,
    update_requirements_dependency,
)


def _issue(
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
    assert find_dependency_in_requirements("requests-toolkit", path) == first
    assert find_dependency_in_requirements("not-present", path) is None


def test_requirements_include_directives_are_ignored_without_escaping_repo(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text("-r ../../outside.txt\n", encoding="utf-8")
    outside = tmp_path.parent.parent / "outside.txt"
    outside.write_text("private-package==9.9\n", encoding="utf-8")

    assert parse_requirements_txt(tmp_path / "requirements.txt") == []
    localized = locate_from_issue(_issue("private-package"), tmp_path)
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

    assert find_dependency_in_requirements("requests.toolkit", path) is None
    with pytest.raises(ValueError, match="exactly one"):
        update_requirements_dependency(path, "requests.toolkit", "3.0")


def test_duplicate_requirements_localization_is_not_an_authorized_direct_target(
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements.txt").write_text(
        "Requests_toolkit==1.0\nrequests-toolkit==2.0\n",
        encoding="utf-8",
    )

    localized = locate_from_issue(_issue("requests.toolkit"), tmp_path)

    assert localized.manifest_file == "requirements.txt"
    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None


def test_odc_scan_mount_path_localizes_to_requirements_file(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("PyJWT==2.4.0\n", encoding="utf-8")

    localized = locate_from_issue(
        _issue("PyJWT", file_path="/scan/requirements.txt:PyJWT/2.4.0"),
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
        _issue("Requests.Toolkit", file_path="/src/services/api/.venv/lib/site-packages/x.py"),
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
        _issue("requests", file_path="/src/services/api/site-packages/pkg.py"), tmp_path
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

    direct = locate_from_issue(_issue("requests.toolkit"), tmp_path)
    optional = locate_from_issue(_issue("pytest"), tmp_path)

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

    install = locate_from_issue(_issue("requests-toolkit"), tmp_path)
    extra = locate_from_issue(_issue("pytest"), tmp_path)

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
        _issue("REQUESTS.toolkit", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    transitive = locate_from_issue(
        _issue("urllib3", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    dev = locate_from_issue(_issue("black", file_path="/src/Pipfile.lock?/develop"), tmp_path)
    unversioned = locate_from_issue(
        _issue("unversioned", file_path="/src/Pipfile.lock?/default"), tmp_path
    )
    unknown = locate_from_issue(_issue("absent"), tmp_path)

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

    localized = locate_from_issue(_issue("locked", file_path="/src/Pipfile.lock"), tmp_path)

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
    assert locate_from_issue(_issue("requests"), tmp_path).manifest_file is None


def test_manifest_selection_rejects_outside_and_symlink_escape_paths(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("requests==2.0\n", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside" / "requirements.txt"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("requests==9.9\n", encoding="utf-8")
    outside_issue = _issue("requests", file_path=str(outside))
    escaped = tmp_path / "linked"

    escaped.symlink_to(outside.parent, target_is_directory=True)
    symlink_issue = _issue("requests", file_path="linked/requirements.txt")

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

    localized = locate_from_issue(_issue("requests"), tmp_path)

    assert localized.manifest_file == manifest_name
    assert localized.is_direct_dependency is None
    assert localized.declaration_type is None


def test_non_pypi_issue_is_not_localized_through_python_manifests(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text("lodash==1.0\n", encoding="utf-8")

    localized = locate_from_issue(_issue("lodash", ecosystem="npm"), tmp_path)

    assert localized.manifest_file is None
    assert localized.is_direct_dependency is None
    assert localized.issue.package_name == "lodash"
