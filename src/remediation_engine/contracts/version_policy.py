"""Deterministic registry candidate filtering and version selection."""

from __future__ import annotations

from typing import Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, model_validator

from remediation_engine.contracts.schemas import SCARemediationStage


class RegistryCandidate(BaseModel):
    """Typed registry result consumed by :func:`select_version`."""

    version: str
    ecosystem: Literal["npm", "pypi"] = "npm"
    semver_key: tuple[int, int, int] | None = None
    security_floor_met: bool
    is_stable: bool
    same_major: bool
    already_attempted: bool
    selection_roles: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _canonicalize_pypi_version(cls, values: object) -> object:
        if not isinstance(values, dict) or values.get("ecosystem", "npm") != "pypi":
            return values
        version_value = values.get("version")
        if version_value is None:
            return values
        try:
            version = Version(str(version_value))
        except InvalidVersion:
            return values
        canonical = dict(values)
        canonical["version"] = str(version)
        return canonical

    @model_validator(mode="after")
    def _validate_version_key(self) -> RegistryCandidate:
        if self.ecosystem == "npm":
            if (
                not isinstance(self.semver_key, tuple)
                or len(self.semver_key) != 3
                or any(
                    not isinstance(part, int) or isinstance(part, bool) for part in self.semver_key
                )
            ):
                raise ValueError("npm candidates require a three-part semver_key")
        else:
            if self.semver_key is not None:
                raise ValueError("pypi candidates must not carry a semver_key")
            try:
                version = Version(self.version)
            except InvalidVersion as exc:
                raise ValueError("pypi candidates require a valid PEP 440 version") from exc
            if version.is_prerelease or version.is_devrelease:
                raise ValueError("pypi candidates must be stable PEP 440 versions")
        return self


def registry_version_key(
    version: str,
    ecosystem: Literal["npm", "pypi"],
    semver_key: tuple[int, int, int] | None = None,
) -> tuple[int, object]:
    """Return a source-orderable key without conflating ecosystem versions."""
    if ecosystem == "npm":
        if (
            not isinstance(semver_key, tuple)
            or len(semver_key) != 3
            or any(not isinstance(part, int) or isinstance(part, bool) for part in semver_key)
        ):
            raise ValueError("npm versions require a three-part semver_key")
        return 0, semver_key
    if ecosystem == "pypi":
        try:
            return 1, Version(version)
        except InvalidVersion as exc:
            raise ValueError(f"Invalid PEP 440 version: {version}") from exc
    raise ValueError(f"Unsupported registry ecosystem: {ecosystem}")


def _normalized_attempted_versions(
    attempted: set[str],
    ecosystem: Literal["npm", "pypi"],
) -> set[str | Version]:
    if ecosystem == "npm":
        return {str(value).strip().lstrip("vV") for value in attempted}
    normalized: set[Version] = set()
    for value in attempted:
        try:
            version = Version(str(value).strip())
        except InvalidVersion:
            continue
        if not version.is_prerelease and not version.is_devrelease:
            normalized.add(version)
    return normalized


def select_version(
    candidates: list[RegistryCandidate],
    stage: SCARemediationStage,
    attempted: set[str],
) -> str | None:
    """Select the next eligible version using a stable, pure policy."""

    ecosystems = {candidate.ecosystem for candidate in candidates}
    if len(ecosystems) > 1:
        raise ValueError("candidate list must contain only one registry ecosystem")
    ecosystem: Literal["npm", "pypi"] = next(iter(ecosystems), "npm")
    attempted_normalized = _normalized_attempted_versions(attempted, ecosystem)
    if stage == SCARemediationStage.CODE_WORKAROUND:
        return None

    eligible = [
        candidate
        for candidate in candidates
        if candidate.is_stable
        and candidate.security_floor_met
        and not candidate.already_attempted
        and (
            Version(candidate.version) not in attempted_normalized
            if ecosystem == "pypi"
            else candidate.version not in attempted_normalized
        )
    ]
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        eligible = [candidate for candidate in eligible if candidate.same_major]
    elif stage == SCARemediationStage.NPM_LATEST:
        # Preserve npm's authoritative latest-role and legacy fixture behavior.
        npm_latest = [
            candidate for candidate in eligible if "npm_latest" in candidate.selection_roles
        ]
        if npm_latest:
            eligible = npm_latest
    elif stage == SCARemediationStage.PYPI_SAME_MAJOR:
        eligible = [candidate for candidate in eligible if candidate.same_major]
    elif stage == SCARemediationStage.PYPI_LATEST:
        pypi_latest = [
            candidate for candidate in eligible if "pypi_latest" in candidate.selection_roles
        ]
        if pypi_latest:
            eligible = pypi_latest

    reverse = stage != SCARemediationStage.OSV_MINIMUM
    eligible.sort(
        key=lambda candidate: (
            registry_version_key(candidate.version, ecosystem, candidate.semver_key),
            candidate.version,
        ),
        reverse=reverse,
    )
    return eligible[0].version if eligible else None


__all__ = ["RegistryCandidate", "registry_version_key", "select_version"]
