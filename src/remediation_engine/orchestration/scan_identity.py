"""Canonical identity helpers for incremental scan occurrences."""

from __future__ import annotations

import json


def package_scan_occurrence_key(
    source_input_path: str,
    lockfile_key: str | None,
    package_name: str,
    version: str,
    occurrence_index: int = 0,
) -> str:
    """Return the stable key for one physical ODC package occurrence.

    Args:
        source_input_path: Relative input path scanned by ODC.
        lockfile_key: Physical npm ``packages`` key, or ``None`` for non-npm input.
        package_name: Reported package name; identity comparison is case-insensitive.
        version: Reported package version.
        occurrence_index: Deterministic ordinal for repeated report entries.

    Returns:
        Compact canonical JSON suitable as a ledger-map key.
    """
    return json.dumps(
        [
            source_input_path,
            lockfile_key,
            package_name.casefold(),
            version,
            occurrence_index,
        ],
        separators=(",", ":"),
    )


__all__ = ["package_scan_occurrence_key"]
