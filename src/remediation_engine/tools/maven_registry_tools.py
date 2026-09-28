"""Supervisor-only Maven Central version candidate lookup."""

from __future__ import annotations

import re
from functools import cmp_to_key
from typing import Any

import requests

from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
)

_MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
_REQUEST_TIMEOUT_SECONDS = 15
_ROWS_PER_PAGE = 200
_COORDINATE_PART_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


def _canonical_coordinate(package_name: str) -> tuple[str, str]:
    """Validate a canonical group:artifact coordinate before querying Solr."""
    value = str(package_name or "").strip()
    parts = value.split(":")
    if (
        len(parts) != 2
        or any(not _COORDINATE_PART_RE.fullmatch(part) for part in parts)
        or any(part.endswith(".") or ".." in part for part in parts)
    ):
        raise ValueError("package_name must be a canonical Maven group:artifact coordinate")
    return parts[0], parts[1]


def fetch_maven_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[MavenRegistryCandidate]:
    """Fetch only the lowest and highest stable Maven releases meeting a floor.

    Coordinates are passed only through Solr's parameter interface. Registry
    timestamps and response ordering never participate in version selection.
    HTTP and malformed-response errors fail closed as ``ValueError``.
    """
    group_id, artifact_id = _canonical_coordinate(package_name)
    floor = str(security_floor or "").strip()
    if not is_stable_maven_version(floor):
        raise ValueError(f"Invalid Maven security floor: {security_floor!r}")

    attempted = tuple(str(value).strip() for value in (attempted_versions or set()))

    def already_attempted(version: str) -> bool:
        for attempted_version in attempted:
            try:
                if compare_maven_versions(version, attempted_version) == 0:
                    return True
            except ValueError:
                if version == attempted_version:
                    return True
        return False

    versions: list[str] = []
    seen_versions: set[str] = set()
    start = 0
    num_found: int | None = None
    while num_found is None or start < num_found:
        params = {
            "q": f'g:"{group_id}" AND a:"{artifact_id}"',
            "core": "gav",
            "rows": _ROWS_PER_PAGE,
            "start": start,
            "wt": "json",
        }
        try:
            response = requests.get(
                _MAVEN_SEARCH_URL,
                params=params,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except requests.RequestException as exc:
            raise ValueError(f"Could not query Maven Central for {package_name}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Maven Central returned invalid JSON for {package_name}: {exc}"
            ) from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
            raise ValueError(f"Maven Central returned an invalid response for {package_name}")
        registry_response = payload["response"]
        raw_num_found = registry_response.get("numFound")
        docs = registry_response.get("docs")
        if (
            not isinstance(raw_num_found, int)
            or isinstance(raw_num_found, bool)
            or raw_num_found < 0
            or not isinstance(docs, list)
        ):
            raise ValueError(f"Maven Central returned invalid pagination data for {package_name}")
        if num_found is None or raw_num_found != num_found:
            num_found = raw_num_found
        if not docs and start < num_found:
            raise ValueError(f"Maven Central pagination stopped early for {package_name}")
        for document in docs:
            if isinstance(document, dict) and isinstance(document.get("v"), str):
                version = document["v"].strip()
                if version and version not in seen_versions:
                    seen_versions.add(version)
                    versions.append(version)
        start += len(docs)
        if not docs:
            break

    eligible: list[str] = []
    for version in versions:
        if not is_stable_maven_version(version):
            continue
        try:
            if compare_maven_versions(version, floor) >= 0:
                eligible.append(version)
        except ValueError:
            continue
    if not eligible:
        return []
    eligible.sort(key=cmp_to_key(compare_maven_versions))
    minimum = eligible[0]
    latest = eligible[-1]
    if compare_maven_versions(minimum, latest) == 0:
        return [
            MavenRegistryCandidate(
                version=minimum,
                security_floor_met=True,
                is_stable=True,
                already_attempted=already_attempted(minimum),
                selection_roles=("maven_minimum", "maven_latest"),
            )
        ]
    return [
        MavenRegistryCandidate(
            version=minimum,
            security_floor_met=True,
            is_stable=True,
            already_attempted=already_attempted(minimum),
            selection_roles=("maven_minimum",),
        ),
        MavenRegistryCandidate(
            version=latest,
            security_floor_met=True,
            is_stable=True,
            already_attempted=already_attempted(latest),
            selection_roles=("maven_latest",),
        ),
    ]


__all__ = ["fetch_maven_registry_candidates"]
