"""Supervisor-only Maven Central version candidate lookup."""

from __future__ import annotations

import re
import time
from functools import cmp_to_key
from typing import Any
from xml.etree import ElementTree as ET

import requests

from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
)

_MAVEN_SEARCH_URL = "https://search.maven.org/solrsearch/select"
_MAVEN_REPOSITORY_BASE_URL = "https://repo.maven.apache.org/maven2"
_REQUEST_TIMEOUT_SECONDS = 15
_REQUEST_MAX_ATTEMPTS = 3
_REQUEST_RETRY_BACKOFF_SECONDS = 0.5
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
_ROWS_PER_PAGE = 200
_COORDINATE_PART_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


def _canonical_coordinate(package_name: str) -> tuple[str, str]:
    """Validate a canonical group:artifact coordinate before querying Maven Central."""
    value = str(package_name or "").strip()
    parts = value.split(":")
    if (
        len(parts) != 2
        or any(not _COORDINATE_PART_RE.fullmatch(part) for part in parts)
        or any(part.endswith(".") or ".." in part for part in parts)
    ):
        raise ValueError("package_name must be a canonical Maven group:artifact coordinate")
    return parts[0], parts[1]


def _request_with_retries(
    url: str,
    *,
    package_name: str,
    source: str,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    """Issue one bounded Maven Central request using the shared retry policy."""
    for attempt in range(_REQUEST_MAX_ATTEMPTS):
        try:
            if params is None:
                response = requests.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
            else:
                response = requests.get(
                    url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError)) or (
                status_code in _RETRYABLE_HTTP_STATUS_CODES
            )
            if not retryable or attempt + 1 == _REQUEST_MAX_ATTEMPTS:
                raise ValueError(
                    f"Could not query Maven Central {source} for {package_name} "
                    f"after {attempt + 1} attempt(s): {exc}"
                ) from exc
            time.sleep(_REQUEST_RETRY_BACKOFF_SECONDS * (2**attempt))
    raise ValueError(f"Could not query Maven Central {source} for {package_name}")


def _parse_metadata_versions(response: requests.Response, package_name: str) -> list[str]:
    """Parse and validate the canonical repository's version list."""
    content = getattr(response, "content", None)
    if content is None:
        content = getattr(response, "text", None)
    if not content:
        raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
    try:
        root = ET.fromstring(content)
    except (ET.ParseError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Maven Central returned invalid repository metadata for {package_name}: {exc}"
        ) from exc

    versioning_nodes = [element for element in root if element.tag == "versioning"]
    if root.tag != "metadata" or len(versioning_nodes) != 1:
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")
    versions_nodes = [element for element in versioning_nodes[0] if element.tag == "versions"]
    if len(versions_nodes) != 1:
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")

    versions_node = versions_nodes[0]
    version_nodes = list(versions_node)
    if not version_nodes:
        raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
    if versions_node.attrib or (versions_node.text and versions_node.text.strip()):
        raise ValueError(f"Maven Central returned malformed repository metadata for {package_name}")

    versions: list[str] = []
    seen_versions: set[str] = set()
    for version_node in version_nodes:
        if (
            version_node.tag != "version"
            or version_node.attrib
            or len(version_node)
            or (version_node.tail and version_node.tail.strip())
        ):
            raise ValueError(
                f"Maven Central returned malformed repository metadata for {package_name}"
            )
        version = (version_node.text or "").strip()
        if not version:
            raise ValueError(f"Maven Central returned empty repository metadata for {package_name}")
        try:
            compare_maven_versions(version, version)
        except ValueError as exc:
            raise ValueError(
                f"Maven Central returned an invalid Maven version for {package_name}: {version!r}"
            ) from exc
        if version not in seen_versions:
            seen_versions.add(version)
            versions.append(version)
    return versions


def _fetch_solr_versions(
    group_id: str,
    artifact_id: str,
    package_name: str,
) -> list[str]:
    """Fetch and deduplicate all Solr versions, preserving its pagination rules."""
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
        response = _request_with_retries(
            _MAVEN_SEARCH_URL,
            package_name=package_name,
            source="Solr",
            params=params,
        )
        try:
            payload: Any = response.json()
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
    return versions


def _metadata_url(group_id: str, artifact_id: str) -> str:
    group_path = group_id.replace(".", "/")
    return f"{_MAVEN_REPOSITORY_BASE_URL}/{group_path}/{artifact_id}/maven-metadata.xml"


def fetch_maven_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str] | None = None,
) -> list[MavenRegistryCandidate]:
    """Fetch only the lowest and highest stable Maven releases meeting a floor.

    Canonical repository metadata is preferred, with the Solr API as a bounded
    fallback. Registry timestamps, response ordering, and declared latest/release
    values never participate in version selection.
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

    metadata_url = _metadata_url(group_id, artifact_id)
    try:
        metadata_response = _request_with_retries(
            metadata_url,
            package_name=package_name,
            source="repository metadata",
        )
        versions = _parse_metadata_versions(metadata_response, package_name)
    except ValueError as metadata_error:
        try:
            versions = _fetch_solr_versions(group_id, artifact_id, package_name)
        except ValueError as solr_error:
            raise ValueError(
                f"Could not fetch Maven Central versions for {package_name}; "
                f"repository metadata failed: {metadata_error}; "
                f"Solr fallback failed: {solr_error}"
            ) from solr_error
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
