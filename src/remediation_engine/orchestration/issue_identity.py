"""Stable identity helpers for vulnerability findings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any, TypeVar

from remediation_engine.contracts.accessors import model_or_dict_value
from remediation_engine.contracts.schemas import VulnerabilityIssue


def stable_issue_fingerprint(issue: VulnerabilityIssue) -> str:
    """Return a stable JSON fingerprint, ignoring generated ingestion metadata.

    Args:
        issue: Typed vulnerability finding to identify.

    Returns:
        Canonical JSON containing issue content except its generated ID and
        ingestion timestamp. The value is suitable for deterministic equality
        and de-duplication within one remediation run.
    """
    payload = issue.model_dump(mode="json")
    payload.pop("id", None)
    payload.pop("ingested_at", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


_IssueT = TypeVar("_IssueT")


def finding_occurrence_key(record: Any) -> tuple[str, str, str]:
    """Return the normalized package, version, and path identifying an occurrence.

    Args:
        record: A typed issue, dashboard projection, or mapping.

    Returns:
        A tuple used to keep distinct package occurrences separate.
    """
    package_name = (
        str(model_or_dict_value(record, "package_name") or "").strip().casefold()
        or "unknown package"
    )
    package_version = str(model_or_dict_value(record, "package_version") or "").strip()
    raw_path = str(model_or_dict_value(record, "file_path") or "").strip().replace("\\", "/")
    file_path = PurePosixPath(raw_path).as_posix() if raw_path else ""
    return package_name, package_version, file_path


def _linked_identifier_tokens(issue: Any, index: int) -> set[str]:
    """Return identifiers that safely link issue records for one occurrence."""
    tokens = {
        f"{prefix}:{str(value).strip().upper()}"
        for field, prefix in (("cve_id", "CVE"), ("ghsa_id", "GHSA"))
        if (value := model_or_dict_value(issue, field)) and str(value).strip()
    }
    if tokens:
        return tokens

    rule_id = str(model_or_dict_value(issue, "rule_id") or "").strip()
    if rule_id:
        return {f"RULE:{rule_id.upper()}"}
    issue_id = str(model_or_dict_value(issue, "id") or "").strip()
    return {f"ISSUE:{issue_id}" if issue_id else f"INDEX:{index}"}


def canonical_issue_components(issues: Sequence[_IssueT]) -> list[list[_IssueT]]:
    """Group linked issue records into canonical package-occurrence findings.

    Records are linked transitively by shared CVE/GHSA identifiers, but only
    when package, installed version, and repository path match. Inputs are
    read-only and component order follows the first source record.

    Args:
        issues: Scanner issue records, either typed models or mappings.

    Returns:
        One list of linked records per canonical finding occurrence.
    """
    if not issues:
        return []

    parents = list(range(len(issues)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    first_index_by_occurrence_and_token: dict[tuple[str, str, str], dict[str, int]] = {}
    for index, issue in enumerate(issues):
        occurrence_tokens = first_index_by_occurrence_and_token.setdefault(
            finding_occurrence_key(issue), {}
        )
        for token in _linked_identifier_tokens(issue, index):
            if token in occurrence_tokens:
                union(index, occurrence_tokens[token])
            else:
                occurrence_tokens[token] = index

    components: dict[int, list[_IssueT]] = {}
    for index, issue in enumerate(issues):
        components.setdefault(find(index), []).append(issue)
    return list(components.values())


__all__ = [
    "canonical_issue_components",
    "finding_occurrence_key",
    "stable_issue_fingerprint",
]
