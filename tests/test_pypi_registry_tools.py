"""Offline behavior tests for bounded PyPI registry access and selection."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from requests import RequestException

from remediation_engine.tools import pypi_registry_tools as pypi


def _metadata() -> dict:
    return {
        "info": {"requires_dist": []},
        "releases": {
            "1.3.9": [{"filename": "old.whl", "yanked": False}],
            "1.4.0rc1": [{"filename": "pre.whl", "yanked": False}],
            "1.4.0.dev1": [{"filename": "dev.whl", "yanked": False}],
            "1.4.0": [{"filename": "floor.whl", "yanked": False}],
            "1.4.1": [
                {"filename": "partly-yanked.whl", "yanked": True},
                {"filename": "ok.whl", "yanked": False},
            ],
            "1.5.0": [{"filename": "yanked.whl", "yanked": True}],
            "1.6.0": [],
            "2.0.0": [{"filename": "next-major.whl", "yanked": False}],
        },
    }


def _response(payload: dict, status_code: int = 200) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_candidates_use_only_stable_file_backed_releases_and_distinct_roles():
    with patch.object(pypi.requests, "get", return_value=_response(_metadata())) as get:
        candidates = pypi.fetch_pypi_registry_candidates("Example_Pkg.Name", "1.4.0")

    assert [candidate.version for candidate in candidates] == ["1.4.0", "1.4.1", "2.0.0"]
    assert [candidate.selection_roles for candidate in candidates] == [
        ("osv_minimum",),
        ("same_major",),
        ("pypi_latest",),
    ]
    assert all(
        candidate.ecosystem == "pypi" and candidate.semver_key is None for candidate in candidates
    )
    assert get.call_args.args[0] == "https://pypi.org/pypi/example-pkg-name/json"
    assert get.call_args.kwargs["timeout"] > 0


def test_candidate_roles_collapse_when_one_version_fulfills_multiple_roles():
    metadata = {"releases": {"1.0.0": [{"yanked": False}], "1.0.1": [{"yanked": False}]}}
    with patch.object(pypi.requests, "get", return_value=_response(metadata)):
        candidates = pypi.fetch_pypi_registry_candidates("demo", "1.0.0")

    assert [candidate.version for candidate in candidates] == ["1.0.0", "1.0.1"]
    assert candidates[0].selection_roles == ("osv_minimum",)
    assert candidates[1].selection_roles == ("pypi_latest", "same_major")


def test_attempted_versions_are_normalized_before_candidate_role_selection():
    with patch.object(
        pypi.requests,
        "get",
        side_effect=[_response(_metadata()), _response(_metadata())],
    ):
        candidates = pypi.fetch_pypi_registry_candidates(
            "demo",
            "1.4.0",
            {"1.4.0", "v1.4.1"},
        )
        safe_version = pypi.select_python_safe_version("demo", "1.4.0", {"1.4.0"})

    assert [candidate.version for candidate in candidates] == ["2.0.0"]
    assert candidates[0].selection_roles == ("osv_minimum", "pypi_latest")
    assert safe_version == "1.4.1"

    with (
        patch.object(pypi.requests, "get") as get,
        pytest.raises(ValueError, match="INVALID SECURITY FLOOR"),
    ):
        pypi.fetch_pypi_registry_candidates("demo", "1.4.0rc1")
    get.assert_not_called()


def test_404_malformed_network_and_empty_results_are_distinct():
    missing = _response({}, status_code=404)
    with (
        patch.object(pypi.requests, "get", return_value=missing),
        pytest.raises(ValueError, match="PACKAGE NOT FOUND"),
    ):
        pypi.fetch_pypi_registry_candidates("missing", "1.0")

    malformed = _response({"info": {}})
    with (
        patch.object(pypi.requests, "get", return_value=malformed),
        pytest.raises(ValueError, match="MALFORMED PYPI METADATA"),
    ):
        pypi.fetch_pypi_registry_candidates("demo", "1.0")

    with (
        patch.object(pypi.requests, "get", side_effect=RequestException("offline")),
        pytest.raises(ValueError, match="NETWORK ERROR"),
    ):
        pypi.fetch_pypi_registry_candidates("demo", "1.0")

    no_eligible = {"releases": {"1.0.0": [], "2.0.0": [{"yanked": True}]}}
    with patch.object(pypi.requests, "get", return_value=_response(no_eligible)):
        assert pypi.fetch_pypi_registry_candidates("demo", "1.0.0") == []


def test_version_helpers_and_requires_dist_normalize_the_release_request():
    package_metadata = _metadata()
    release_metadata = {"info": {"requires_dist": ["child_pkg>=2.0"]}}
    with patch.object(
        pypi.requests,
        "get",
        side_effect=[
            _response(package_metadata),
            _response(package_metadata),
            _response(release_metadata),
        ],
    ) as get:
        assert pypi.get_pypi_versions("Example_Pkg.Name")[0] == "1.3.9"
        assert pypi.get_pypi_latest_version("Example_Pkg.Name") == "2.0.0"
        assert pypi.get_pypi_release_requires_dist("Example_Pkg.Name", "1.0") == ["child_pkg>=2.0"]

    assert get.call_args_list[-1].args[0] == "https://pypi.org/pypi/example-pkg-name/1.0/json"


def test_python_parent_report_preserves_selection_parser_fields_and_is_read_only():
    data = {
        "releases": {
            "1.0.0": [{"yanked": False}],
            "1.1.0": [{"yanked": False}],
        }
    }
    with (
        patch.object(pypi, "_fetch_json", return_value=data),
        patch.object(pypi, "get_pypi_release_requires_dist", return_value=["child>=2.0"]),
    ):
        report = pypi.plan_python_parent_version.invoke(
            {
                "parent_package_name": "Parent_Pkg",
                "child_package_name": "child",
                "child_fixed_version": "2.1.0",
                "installed_parent_version": "1.0.0",
                "selection": "minimum",
            }
        )

    assert "- Selected Version: 1.1.0" in report
    assert "- Eligible Candidates: 1.1.0" in report
    assert "- Compatible Parent Versions: 1.1.0" in report
    assert "- Latest Stable: 1.1.0" in report
    assert "- PyPI Latest: 1.1.0" in report
    assert "- Selected: 1.1.0" in report


def test_parent_latest_role_is_the_highest_compatible_candidate():
    data = {
        "releases": {
            "1.0.0": [{"yanked": False}],
            "1.1.0": [{"yanked": False}],
            "1.2.0": [{"yanked": False}],
        }
    }
    with (
        patch.object(pypi, "_fetch_json", return_value=data),
        patch.object(
            pypi,
            "get_pypi_release_requires_dist",
            side_effect=[["child>=2.0"], ["child<2.0"]],
        ),
    ):
        report = pypi.plan_python_parent_version.invoke(
            {
                "parent_package_name": "parent",
                "child_package_name": "child",
                "child_fixed_version": "2.1",
                "installed_parent_version": "1.0",
                "selection": "latest",
            }
        )

    assert "- Selected Version: 1.1.0" in report
    assert "- Eligible Candidates: 1.1.0" in report
    assert "- Compatible Parent Versions: 1.1.0" in report
    assert "- PyPI Latest: 1.1.0" in report


def test_parent_minimum_and_same_major_roles_are_reassigned_from_compatible_pool():
    data = {
        "releases": {
            "1.0.0": [{"yanked": False}],
            "1.1.0": [{"yanked": False}],
            "1.2.0": [{"yanked": False}],
            "2.0.0": [{"yanked": False}],
        }
    }
    requirements = [
        ["child<2.0"],
        ["child>=2.0"],
        ["child<2.0"],
        ["child<2.0"],
        ["child>=2.0"],
        ["child<2.0"],
    ]
    with (
        patch.object(pypi, "_fetch_json", return_value=data),
        patch.object(pypi, "get_pypi_release_requires_dist", side_effect=requirements),
    ):
        minimum_report = pypi.plan_python_parent_version.invoke(
            {
                "parent_package_name": "parent",
                "child_package_name": "child",
                "child_fixed_version": "2.1",
                "installed_parent_version": "1.0",
                "selection": "minimum",
            }
        )
        same_major_report = pypi.plan_python_parent_version.invoke(
            {
                "parent_package_name": "parent",
                "child_package_name": "child",
                "child_fixed_version": "2.1",
                "installed_parent_version": "1.0",
                "selection": "same_major",
            }
        )

    assert "- Selected Version: 1.2.0" in minimum_report
    assert "- Selected Version: 1.2.0" in same_major_report
    assert "- Compatible Parent Versions: 1.2.0" in minimum_report
    assert "- PyPI Latest: 1.2.0" in minimum_report
    assert "- Latest Stable: 2.0.0" in minimum_report
