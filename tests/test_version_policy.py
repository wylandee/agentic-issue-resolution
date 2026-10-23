"""Focused policy tests for mixed npm and PyPI version ecosystems."""

from __future__ import annotations

import pytest
from packaging.version import Version
from pydantic import ValidationError

from remediation_engine.contracts.schemas import SCARemediationStage
from remediation_engine.contracts.version_policy import (
    RegistryCandidate,
    registry_version_key,
    select_version,
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
