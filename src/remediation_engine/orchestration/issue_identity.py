"""Stable identity helpers for vulnerability findings."""

from __future__ import annotations

import json

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


__all__ = ["stable_issue_fingerprint"]
