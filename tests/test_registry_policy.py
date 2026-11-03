"""Focused tests for registry policy."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
import requests
from packaging.version import Version
from pydantic import ValidationError
from requests import RequestException

import remediation_engine.tools.registry as pypi
from remediation_engine.agents.supervisor_policy import (
    _canonical_security_floor,
    _next_sca_stage,
    _selection_for_stage,
)
from remediation_engine.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    RegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
    registry_version_key,
    select_maven_version,
    select_version,
)
from remediation_engine.orchestration.state import _is_exhausted_update_pivot_candidate
from remediation_engine.tools.registry import fetch_maven_registry_candidates


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class _XmlResponse:
    def __init__(self, content: str) -> None:
        self.content = content.encode()

    def raise_for_status(self) -> None:
        return None


def test_metadata_candidate_lookup_prefers_canonical_metadata_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_url = "https://repo.maven.apache.org/maven2/org/safe/widget/maven-metadata.xml"
    calls: list[dict[str, Any]] = []

    def fake_get(
        url: str,
        *,
        timeout: int,
        params: dict[str, Any] | None = None,
    ) -> _XmlResponse:
        calls.append({"url": url, "params": params, "timeout": timeout})
        assert params is None
        return _XmlResponse(
            """<metadata>
  <versioning>
    <latest>100.0</latest>
    <release>99.0</release>
    <lastUpdated>20261020123456</lastUpdated>
    <versions>
      <version>3.0</version>
      <version>1.0</version>
      <version>2.0</version>
      <version>2.1-rc1</version>
      <version>4.0-SNAPSHOT</version>
    </versions>
  </versioning>
</metadata>"""
        )

    monkeypatch.setattr("remediation_engine.tools.registry.requests.get", fake_get)
    candidates = fetch_maven_registry_candidates("org.safe:widget", "2.0", {"3.0"})

    assert [candidate.version for candidate in candidates] == ["2.0", "3.0"]
    assert candidates[0].selection_roles == ("maven_minimum",)
    assert candidates[1].selection_roles == ("maven_latest",)
    assert [candidate.already_attempted for candidate in candidates] == [False, True]
    assert all(candidate.security_floor_met and candidate.is_stable for candidate in candidates)
    assert len(calls) == 1
    assert calls[0]["url"] == metadata_url
    assert calls[0]["timeout"] > 0


def test_solr_candidate_lookup_uses_params_paginates_and_returns_safe_extremes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_url = "https://repo.maven.apache.org/maven2/org/safe/widget/maven-metadata.xml"
    calls: list[dict[str, Any]] = []

    def fake_get(
        url: str,
        *,
        timeout: int,
        params: dict[str, Any] | None = None,
    ) -> _Response:
        calls.append({"url": url, "params": params, "timeout": timeout})
        if url == metadata_url:
            raise requests.Timeout("repository metadata timed out")
        assert params is not None
        start = params["start"]
        versions = ["1.8", "2.0-SNAPSHOT", "2.0-rc1", "2.0", "1.9"]
        docs = [
            {"v": version, "timestamp": 100000 - index} for index, version in enumerate(versions)
        ]
        if start == 0:
            docs.extend({"v": "0.1"} for _ in range(195))
            return _Response({"response": {"numFound": 201, "docs": docs}})
        return _Response({"response": {"numFound": 201, "docs": [{"v": "3.0"}]}})

    monkeypatch.setattr(
        "remediation_engine.tools.registry.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.registry.time.sleep",
        lambda *_args: None,
    )
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.9", {"3.0"})
    solr_calls = [call for call in calls if call["params"] is not None]

    assert [candidate.version for candidate in candidates] == ["1.9", "3.0"]
    assert candidates[0].selection_roles == ("maven_minimum",)
    assert candidates[1].selection_roles == ("maven_latest",)
    assert candidates[0].already_attempted is False
    assert candidates[1].already_attempted is True
    assert [call["url"] for call in calls[:3]] == [metadata_url] * 3
    assert all(call["url"] == "https://search.maven.org/solrsearch/select" for call in solr_calls)
    assert solr_calls[0]["params"]["q"] == 'g:"org.safe" AND a:"widget"'
    assert solr_calls[0]["params"]["core"] == "gav"
    assert solr_calls[0]["params"]["rows"] == 200
    assert solr_calls[0]["params"]["wt"] == "json"
    assert all(call["timeout"] > 0 for call in calls)


def test_solr_candidate_lookup_deduplicates_equal_roles_and_marks_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(
        _url: str,
        *,
        params: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> _Response:
        if params is None:
            raise requests.Timeout("repository metadata unavailable")
        return _Response({"response": {"numFound": 1, "docs": [{"v": "1.0"}]}})

    monkeypatch.setattr(
        "remediation_engine.tools.registry.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.registry.time.sleep",
        lambda *_args: None,
    )
    candidates = fetch_maven_registry_candidates("org.safe:widget", "1.0", {"1.0.0"})
    assert len(candidates) == 1
    assert candidates[0].selection_roles == ("maven_minimum", "maven_latest")
    assert candidates[0].already_attempted is True


def test_maven_registry_rejects_unsafe_coordinates_and_reports_both_source_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def failed_request(
        url: str,
        *,
        params: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        calls.append((url, params))
        if params is None:
            raise requests.Timeout("repository metadata timed out")
        raise requests.ConnectionError("Solr unavailable")

    monkeypatch.setattr(
        "remediation_engine.tools.registry.requests.get",
        failed_request,
    )
    monkeypatch.setattr(
        "remediation_engine.tools.registry.time.sleep",
        lambda *_args: None,
    )
    with pytest.raises(ValueError):
        fetch_maven_registry_candidates("org.safe:widget?x=1", "1.0")
    assert calls == []

    with pytest.raises(ValueError, match="Maven Central") as error:
        fetch_maven_registry_candidates("org.safe:widget", "1.0")
    assert "repository metadata timed out" in str(error.value)
    assert "Solr unavailable" in str(error.value)
    assert sum(params is None for _url, params in calls) == 3
    assert sum(params is not None for _url, params in calls) == 3


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


def _candidate(version: str, role: str, *, attempted: bool = False) -> MavenRegistryCandidate:
    return MavenRegistryCandidate(
        version=version,
        security_floor_met=True,
        is_stable=True,
        already_attempted=attempted,
        selection_roles=(role,),
    )


def test_comparable_version_numeric_qualifier_alias_and_service_pack_ordering() -> None:
    assert compare_maven_versions("1.9", "1.10") < 0
    assert compare_maven_versions("1.0-alpha1", "1.0-beta1") < 0
    assert compare_maven_versions("1.0-alpha1", "1.0-alpha-1") == 0
    assert compare_maven_versions("1.0-a1", "1.0-alpha-1") == 0
    assert compare_maven_versions("1.0-1", "1.0.1") < 0
    assert compare_maven_versions("1.0-rc1", "1.0") < 0
    assert compare_maven_versions("1.0", "1.0-sp1") < 0
    assert compare_maven_versions("1.0-ga", "1.0-final") == 0
    assert compare_maven_versions("1.0-release", "1.0") == 0
    assert compare_maven_versions("1.0-RC1", "1.0-rc1") == 0


def test_safe_maven_stability_and_malformed_versions() -> None:
    assert is_stable_maven_version("3.5.0")
    assert is_stable_maven_version("3.5.0-sp1")
    assert not is_stable_maven_version("FINAL")
    assert not is_stable_maven_version("sp1")
    assert not is_stable_maven_version("3.5.0-SNAPSHOT")
    assert not is_stable_maven_version("3.5.0-rc1")
    assert not is_stable_maven_version("3.5.0-redhat")
    with pytest.raises(ValueError):
        compare_maven_versions("1..0", "1.0")
    with pytest.raises(ValueError):
        compare_maven_versions("1.0-", "1.0")


def test_maven_selection_requires_stage_role_and_unattempted_candidate() -> None:
    candidates = [
        _candidate("1.4", "maven_minimum"),
        _candidate("2.0", "maven_latest"),
    ]
    assert select_maven_version(candidates, SCARemediationStage.OSV_MINIMUM, set()) == "1.4"
    assert select_maven_version(candidates, SCARemediationStage.MAVEN_LATEST, set()) == "2.0"
    assert (
        select_maven_version(
            [_candidate("1.4", "maven_minimum", attempted=True)],
            SCARemediationStage.OSV_MINIMUM,
            set(),
        )
        is None
    )
    assert (
        select_maven_version(
            [_candidate("2.0", "maven_latest")],
            SCARemediationStage.OSV_MINIMUM,
            set(),
        )
        is None
    )
    assert (
        select_maven_version(
            candidates,
            SCARemediationStage.MAVEN_LATEST,
            {"2.0"},
        )
        is None
    )
    assert (
        select_maven_version(
            [_candidate("1.0", "maven_minimum")],
            SCARemediationStage.OSV_MINIMUM,
            {"1.0.0"},
        )
        is None
    )


def _npm(version: str, roles: tuple[str, ...] = ()) -> RegistryCandidate:
    parts = tuple(int(part) for part in version.split("."))
    return RegistryCandidate(
        version=version,
        semver_key=parts,
        security_floor_met=True,
        is_stable=True,
        same_major=parts[0] == 1,
        already_attempted=False,
        selection_roles=roles,
    )


def _pypi(version: str, roles: tuple[str, ...] = ()) -> RegistryCandidate:
    return RegistryCandidate(
        version=version,
        ecosystem="pypi",
        semver_key=None,
        security_floor_met=True,
        is_stable=True,
        same_major=version.split(".", 1)[0] == "1",
        already_attempted=False,
        selection_roles=roles,
    )


def test_pypi_candidate_version_is_canonicalized_with_packaging():
    assert _pypi("v1.2").version == "1.2"


def test_registry_candidate_enforces_ecosystem_key_invariants():
    with pytest.raises(ValidationError, match="three-part semver_key"):
        RegistryCandidate(
            version="1.2.3",
            security_floor_met=True,
            is_stable=True,
            same_major=True,
            already_attempted=False,
        )

    with pytest.raises(ValidationError, match="must not carry a semver_key"):
        RegistryCandidate(
            version="1.2.3",
            ecosystem="pypi",
            semver_key=(1, 2, 3),
            security_floor_met=True,
            is_stable=True,
            same_major=True,
            already_attempted=False,
        )


def test_registry_version_key_uses_its_ecosystem_ordering_domain():
    assert registry_version_key("2.0.0", "npm", (2, 0, 0)) == (0, (2, 0, 0))
    lower = registry_version_key("1.9", "pypi")[1]
    higher = registry_version_key("1.10", "pypi")[1]
    assert isinstance(lower, Version)
    assert higher > lower


def test_python_selection_uses_pep440_order_roles_and_attempt_normalization():
    candidates = [
        _pypi("1.9.0", ("osv_minimum",)),
        _pypi("1.10.0", ("same_major",)),
        _pypi("2.0.0", ("pypi_latest",)),
    ]

    assert select_version(candidates, SCARemediationStage.OSV_MINIMUM, set()) == "1.9.0"
    assert select_version(candidates, SCARemediationStage.PYPI_SAME_MAJOR, set()) == "1.10.0"
    assert select_version(candidates, SCARemediationStage.PYPI_LATEST, set()) == "2.0.0"
    assert select_version(candidates, SCARemediationStage.OSV_MINIMUM, {"v1.9.0"}) == "1.10.0"


def test_npm_ordering_and_latest_role_behavior_remain_unchanged():
    candidates = [_npm("1.10.0"), _npm("1.9.0"), _npm("2.0.0", ("npm_latest",))]
    assert select_version(candidates, SCARemediationStage.OSV_MINIMUM, set()) == "1.9.0"
    assert select_version(candidates, SCARemediationStage.NPM_SAME_MAJOR, set()) == "1.10.0"
    assert select_version(candidates, SCARemediationStage.NPM_LATEST, set()) == "2.0.0"
    assert select_version(candidates, SCARemediationStage.NPM_LATEST, {"2.0.0"}) == "1.10.0"


def test_mixed_candidate_ecosystems_fail_closed():
    with pytest.raises(ValueError, match="only one registry ecosystem"):
        select_version(
            [_npm("1.0.0"), _pypi("1.0.0")],
            SCARemediationStage.OSV_MINIMUM,
            set(),
        )


def _group(
    *,
    ecosystem: str,
    issue_fixed_version: str | None,
    plan_fixed_version: str | None,
) -> VulnerabilityGroup:
    issue = VulnerabilityIssue(
        source=IssueSource.ODC,
        issue_type=IssueType.SCA,
        ecosystem=ecosystem,
        purl="pkg:pypi/example-package@1.0.0" if ecosystem == "pypi" else None,
        package_name="example-package",
        fixed_version=issue_fixed_version,
    )
    plan = (
        FixPlan(
            status=FixPlanStatus.VERSION_FOUND,
            fixed_version=plan_fixed_version,
            instruction="Upgrade to the fixed version.",
            strategy_used="osv_api",
        )
        if plan_fixed_version is not None
        else None
    )
    return VulnerabilityGroup(
        group_id="test-group",
        issue_type=IssueType.SCA,
        representative_issue_id=issue.id or uuid4(),
        issues=[issue],
        fix_plan=plan,
    )


def test_pypi_stage_chain_never_enters_npm_override():
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM, ecosystem="pypi") == (
        SCARemediationStage.PYPI_SAME_MAJOR
    )
    assert _next_sca_stage(SCARemediationStage.PYPI_SAME_MAJOR, ecosystem="pypi") == (
        SCARemediationStage.PYPI_LATEST
    )
    assert _next_sca_stage(SCARemediationStage.PYPI_LATEST, transitive=True, ecosystem="pypi") == (
        SCARemediationStage.CODE_WORKAROUND
    )
    assert _next_sca_stage(SCARemediationStage.PACKAGE_OVERRIDE, ecosystem="pypi") == (
        SCARemediationStage.CODE_WORKAROUND
    )
    assert _selection_for_stage(SCARemediationStage.PYPI_SAME_MAJOR) == "same_major"
    assert _selection_for_stage(SCARemediationStage.PYPI_LATEST) == "latest"


def test_default_npm_stage_progression_is_unchanged():
    assert _next_sca_stage(SCARemediationStage.OSV_MINIMUM) == SCARemediationStage.NPM_SAME_MAJOR
    assert _next_sca_stage(SCARemediationStage.NPM_SAME_MAJOR) == SCARemediationStage.NPM_LATEST
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST, transitive=True) == (
        SCARemediationStage.PACKAGE_OVERRIDE
    )
    assert _next_sca_stage(SCARemediationStage.NPM_LATEST) == SCARemediationStage.CODE_WORKAROUND


def test_exhausted_pypi_latest_parent_can_pivot_but_npm_parent_cannot():
    pypi_task = RemediationTask(
        task_id="pypi-task",
        parent_group_id="group",
        strategy=RoutingStrategy.VERSION_BUMP,
        strategy_stage=SCARemediationStage.PYPI_LATEST,
        status=TaskStatus.NEEDS_RETRY,
        parent_package_name="direct-parent",
    )
    npm_task = pypi_task.model_copy(
        update={
            "task_id": "npm-task",
            "strategy_stage": SCARemediationStage.NPM_LATEST,
        }
    )
    diagnostics = UpdateRetryDiagnostics(
        task_id="pypi-task",
        exhausted_update_path=True,
    )

    assert _is_exhausted_update_pivot_candidate(pypi_task, diagnostics) is True
    assert (
        _is_exhausted_update_pivot_candidate(
            npm_task,
            diagnostics.model_copy(update={"task_id": "npm-task"}),
        )
        is False
    )


def test_canonical_security_floor_uses_stable_pep440_for_pypi():
    group = _group(
        ecosystem="pypi",
        issue_fixed_version="1.4",
        plan_fixed_version="1.4.0",
    )
    assert _canonical_security_floor(group) == ("1.4.0", None)

    group.fix_plan = group.fix_plan.model_copy(update={"fixed_version": "1.4.0rc1"})
    floor, error = _canonical_security_floor(group)
    assert floor is None
    assert error is not None and "PEP 440" in error


def test_pypi_member_floor_conflicts_compare_with_pep440_ordering():
    group = _group(
        ecosystem="python",
        issue_fixed_version="1.10.0",
        plan_fixed_version="1.9.0",
    )
    floor, error = _canonical_security_floor(group)
    assert floor is None
    assert error is not None and "exceeds" in error
    assert Version("1.10.0") > Version("1.9.0")


def test_npm_floor_remains_strict_semver():
    assert _canonical_security_floor(
        _group(ecosystem="npm", issue_fixed_version="1.4.0", plan_fixed_version="1.4.0")
    ) == ("1.4.0", None)
    floor, error = _canonical_security_floor(
        _group(ecosystem="npm", issue_fixed_version=None, plan_fixed_version="1.4")
    )
    assert floor is None
    assert error is not None and "semver" in error
