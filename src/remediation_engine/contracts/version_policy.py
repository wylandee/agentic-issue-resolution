"""Deterministic registry candidate filtering and version selection."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from remediation_engine.contracts.schemas import SCARemediationStage


class RegistryCandidate(BaseModel):
    """Typed registry result consumed by :func:`select_version`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    semver_key: tuple[int, int, int]
    security_floor_met: bool
    is_stable: bool
    same_major: bool
    already_attempted: bool


def select_version(
    candidates: list[RegistryCandidate],
    stage: SCARemediationStage,
    attempted: set[str],
) -> str | None:
    """Select the next eligible version using a stable, pure policy."""

    if stage == SCARemediationStage.CODE_WORKAROUND:
        return None

    attempted_normalized = {str(value).strip().lstrip("vV") for value in attempted}
    eligible = [
        candidate
        for candidate in candidates
        if candidate.is_stable
        and candidate.security_floor_met
        and not candidate.already_attempted
        and candidate.version not in attempted_normalized
    ]
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        eligible = [candidate for candidate in eligible if candidate.same_major]
    if stage == SCARemediationStage.OSV_MINIMUM:
        eligible.sort(key=lambda candidate: (candidate.semver_key, candidate.version))
    else:
        eligible.sort(key=lambda candidate: (candidate.semver_key, candidate.version), reverse=True)
    return eligible[0].version if eligible else None


def is_version_space_exhausted(
    candidates: list[RegistryCandidate],
    stage: SCARemediationStage,
    attempted: set[str],
) -> bool:
    """Return whether no unattempted candidate remains for ``stage``."""

    return select_version(candidates, stage, attempted) is None


__all__ = ["RegistryCandidate", "select_version", "is_version_space_exhausted"]
