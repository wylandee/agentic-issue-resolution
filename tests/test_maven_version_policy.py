from __future__ import annotations

import pytest

from remediation_engine.contracts.schemas import SCARemediationStage
from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
    select_maven_version,
)


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
