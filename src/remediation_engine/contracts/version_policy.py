"""Deterministic registry candidate filtering and version selection."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, model_validator

from remediation_engine.contracts.schemas import SCARemediationStage


class MavenRegistryCandidate(BaseModel):
    """Typed Maven Central candidate consumed by Maven stage selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    security_floor_met: bool
    is_stable: bool
    already_attempted: bool
    selection_roles: tuple[str, ...] = ()


_MAVEN_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]*\Z")
_MAVEN_QUALIFIER_ORDER = ("alpha", "beta", "milestone", "rc", "snapshot", "", "sp")
_MAVEN_QUALIFIER_INDEX = {name: index for index, name in enumerate(_MAVEN_QUALIFIER_ORDER)}
_MAVEN_QUALIFIER_ALIASES = {
    "a": "alpha",
    "b": "beta",
    "m": "milestone",
    "cr": "rc",
    "ga": "",
    "final": "",
    "release": "",
}


def _maven_string_item(value: str, *, followed_by_digit: bool = False) -> str:
    """Normalize one ComparableVersion string item and its short aliases."""
    qualifier = value.lower()
    if not followed_by_digit and len(qualifier) == 1 and qualifier in {"a", "b", "m"}:
        return qualifier
    return _MAVEN_QUALIFIER_ALIASES.get(qualifier, qualifier)


def _parse_maven_version(version: str) -> tuple[object, ...]:
    """Parse a Maven version into ComparableVersion-compatible nested items."""
    value = str(version).strip().lower()
    if (
        not value
        or len(value) > 512
        or not _MAVEN_VERSION_RE.fullmatch(value)
        or value[-1] in ".-_+"
        or ".." in value
        or ".-" in value
        or "-." in value
        or "--" in value
    ):
        raise ValueError(f"Malformed Maven version: {version!r}")

    root: list[object] = []
    current = root
    stack = [root]
    is_digit = False
    start_index = 0

    for index, char in enumerate(value):
        if char == ".":
            current.append(
                int(value[start_index:index])
                if index > start_index and is_digit
                else _maven_string_item(value[start_index:index])
                if index > start_index
                else 0
            )
            start_index = index + 1
        elif char == "-":
            current.append(
                int(value[start_index:index])
                if index > start_index and is_digit
                else _maven_string_item(value[start_index:index])
                if index > start_index
                else 0
            )
            start_index = index + 1
            current = []
            stack[-1].append(current)
            stack.append(current)
        elif char.isdigit():
            if not is_digit and index > start_index:
                if current:
                    current = []
                    stack[-1].append(current)
                    stack.append(current)
                current.append(_maven_string_item(value[start_index:index], followed_by_digit=True))
                start_index = index
                current = []
                stack[-1].append(current)
                stack.append(current)
            is_digit = True
        else:
            if is_digit and index > start_index:
                current.append(int(value[start_index:index]))
                start_index = index
                current = []
                stack[-1].append(current)
                stack.append(current)
            is_digit = False

    if len(value) > start_index:
        if not is_digit and current:
            current = []
            stack[-1].append(current)
            stack.append(current)
        current.append(
            int(value[start_index:]) if is_digit else _maven_string_item(value[start_index:])
        )

    for items in reversed(stack):
        index = len(items) - 1
        while index >= 0:
            item = items[index]
            is_null = (
                isinstance(item, int)
                and item == 0
                or isinstance(item, str)
                and _compare_maven_qualifiers(item, "") == 0
                or isinstance(item, list)
                and not item
            )
            if is_null:
                del items[index]
            elif not isinstance(item, list):
                break
            index -= 1

    def freeze(items: list[object]) -> tuple[object, ...]:
        return tuple(freeze(item) if isinstance(item, list) else item for item in items)

    return freeze(root)


def _compare_maven_qualifiers(left: str, right: str) -> int:
    left = left.lower()
    right = right.lower()
    left_index = _MAVEN_QUALIFIER_INDEX.get(left)
    right_index = _MAVEN_QUALIFIER_INDEX.get(right)
    left_key = (len(_MAVEN_QUALIFIER_ORDER), left) if left_index is None else (left_index, "")
    right_key = (len(_MAVEN_QUALIFIER_ORDER), right) if right_index is None else (right_index, "")
    return (left_key > right_key) - (left_key < right_key)


def _compare_maven_items(left: object | None, right: object | None) -> int:
    if left is None:
        return 0 if right is None else -_compare_maven_items(right, None)
    if right is None:
        if isinstance(left, int):
            return 0 if left == 0 else 1
        if isinstance(left, str):
            return _compare_maven_qualifiers(left, "")
        if isinstance(left, tuple):
            for item in left:
                comparison = _compare_maven_items(item, None)
                if comparison:
                    return comparison
            return 0
        return 0
    if isinstance(left, int):
        if isinstance(right, int):
            return (left > right) - (left < right)
        return 1
    if isinstance(left, str):
        if isinstance(right, str):
            return _compare_maven_qualifiers(left, right)
        return -1
    if isinstance(left, tuple):
        if isinstance(right, int):
            return -1
        if isinstance(right, str):
            return 1
        if isinstance(right, tuple):
            return _compare_maven_lists(left, right)
    return 0


def _compare_maven_lists(left: tuple[object, ...], right: tuple[object, ...]) -> int:
    for index in range(max(len(left), len(right))):
        left_item = left[index] if index < len(left) else None
        right_item = right[index] if index < len(right) else None
        comparison = (
            -_compare_maven_items(right_item, None)
            if left_item is None
            else _compare_maven_items(left_item, right_item)
        )
        if comparison:
            return comparison
    return 0


def compare_maven_versions(left: str, right: str) -> int:
    """Compare Maven versions using numeric and nested qualifier semantics.

    Malformed values raise ``ValueError`` rather than falling back to lexical
    or semantic-version ordering.
    """
    return _compare_maven_lists(_parse_maven_version(left), _parse_maven_version(right))


def is_stable_maven_version(version: str) -> bool:
    """Return whether a Maven version is a stable release usable for selection."""
    try:
        parsed = _parse_maven_version(version)
    except (TypeError, ValueError):
        return False

    def stable(items: tuple[object, ...]) -> bool:
        return all(
            stable(item)
            if isinstance(item, tuple)
            else not isinstance(item, str) or item in {"", "sp"}
            for item in items
        )

    return stable(parsed) and any(isinstance(item, int) for item in parsed)


def select_maven_version(
    candidates: Sequence[MavenRegistryCandidate],
    stage: SCARemediationStage,
    attempted: set[str],
) -> str | None:
    """Select only the candidate role authorized for the active Maven stage."""
    if stage == SCARemediationStage.OSV_MINIMUM:
        role = "maven_minimum"
        choose_minimum = True
    elif stage == SCARemediationStage.MAVEN_LATEST:
        role = "maven_latest"
        choose_minimum = False
    else:
        return None

    attempted_versions = [str(value).strip() for value in attempted if str(value).strip()]

    def was_attempted(version: str) -> bool:
        for attempted_version in attempted_versions:
            try:
                if compare_maven_versions(version, attempted_version) == 0:
                    return True
            except ValueError:
                if version == attempted_version:
                    return True
        return False

    eligible: list[MavenRegistryCandidate] = []
    for candidate in candidates:
        version = candidate.version.strip()
        if (
            role not in candidate.selection_roles
            or not candidate.security_floor_met
            or not candidate.is_stable
            or not is_stable_maven_version(version)
            or candidate.already_attempted
            or was_attempted(version)
        ):
            continue
        eligible.append(candidate)
    if not eligible:
        return None
    selected = eligible[0]
    for candidate in eligible[1:]:
        try:
            order = compare_maven_versions(candidate.version, selected.version)
        except ValueError:
            continue
        if (choose_minimum and order < 0) or (not choose_minimum and order > 0):
            selected = candidate
    return selected.version


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


__all__ = [
    "MavenRegistryCandidate",
    "RegistryCandidate",
    "compare_maven_versions",
    "is_stable_maven_version",
    "registry_version_key",
    "select_maven_version",
    "select_version",
]
