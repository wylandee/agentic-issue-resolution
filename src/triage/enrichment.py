"""
enrichment.py — CVE threat-intelligence enrichment for the triage layer.

Public API
----------
enrich_cves(cve_ids: List[str]) -> Dict[str, CVEEnrichment]
    Returns a ``CVEEnrichment`` for every requested CVE ID.
    Never raises — on API failure, safe defaults (epss=0.0, in_kev=False) are
    returned and a warning is logged.

Data sources
------------
EPSS (Exploit Prediction Scoring System)
    https://api.first.org/data/v1/epss
    Queried in chunks of up to 100 CVEs (comma-separated).
    Timeout: 10 s.

CISA Known Exploited Vulnerabilities (KEV)
    https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
    Cached locally at data/cache/kev_cache.json with a 24-hour TTL.
    On download failure, a stale cache (if present) is used; otherwise an
    empty KEV set is assumed and a warning is logged.

Cache location
--------------
The cache directory is resolved relative to this file's location:
    <repo_root>/data/cache/kev_cache.json

Override with the ``TRIAGE_CACHE_DIR`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import requests

from src.contracts.schemas import CVEEnrichment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPSS_API_URL = "https://api.first.org/data/v1/epss"
KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
)
EPSS_CHUNK_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 10
KEV_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    """Return the cache directory, honouring TRIAGE_CACHE_DIR env var."""
    env_override = os.environ.get("TRIAGE_CACHE_DIR")
    if env_override:
        return Path(env_override)
    # Walk up from this file to find repo root (contains requirements.txt)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "requirements.txt").exists():
            return parent / "data" / "cache"
    # Fallback: sibling data/cache relative to CWD
    return Path("data") / "cache"


def _kev_cache_path() -> Path:
    return _cache_dir() / "kev_cache.json"


def _is_cache_fresh(path: Path) -> bool:
    """Return True if the cache file exists and was written within the TTL."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < KEV_CACHE_TTL_SECONDS


# ---------------------------------------------------------------------------
# CISA KEV
# ---------------------------------------------------------------------------


def _load_kev_set() -> tuple[Set[str], Dict[str, str]]:
    """
    Return (kev_cve_set, kev_date_map).

    Tries the local cache first; falls back to a live download.
    On any failure returns empty structures and logs a warning.
    """
    cache_path = _kev_cache_path()

    def _parse_catalog(data: Dict[str, Any]) -> tuple[Set[str], Dict[str, str]]:
        kev_set: Set[str] = set()
        date_map: Dict[str, str] = {}
        for entry in data.get("vulnerabilities", []):
            cve_id = entry.get("cveID", "")
            date_added = entry.get("dateAdded", "")
            if cve_id:
                kev_set.add(cve_id.upper())
                if date_added:
                    date_map[cve_id.upper()] = date_added
        return kev_set, date_map

    # --- Cache hit ---
    if _is_cache_fresh(cache_path):
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.debug("KEV: loaded %d entries from cache.", len(raw.get("vulnerabilities", [])))
            return _parse_catalog(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KEV cache read failed (%s); attempting live download.", exc)

    # --- Live download ---
    try:
        resp = requests.get(KEV_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        raw = resp.json()
        # Persist to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        logger.info(
            "KEV: downloaded %d entries and cached at %s.",
            len(raw.get("vulnerabilities", [])),
            cache_path,
        )
        return _parse_catalog(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("KEV download failed (%s); using stale cache if available.", exc)

    # --- Stale cache fallback ---
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.warning("KEV: using stale cache (%s).", cache_path)
            return _parse_catalog(raw)
        except Exception as exc2:  # noqa: BLE001
            logger.warning("KEV stale cache also failed (%s).", exc2)

    logger.warning("KEV: no data available; treating all CVEs as not-in-KEV.")
    return set(), {}


# ---------------------------------------------------------------------------
# FIRST EPSS
# ---------------------------------------------------------------------------


def _fetch_epss_chunk(cve_ids: List[str]) -> Dict[str, tuple[float, float]]:
    """
    Query the EPSS API for a chunk of CVE IDs.

    Returns a dict of cve_id → (epss_score, epss_percentile).
    On failure, returns an empty dict (caller fills in defaults).
    """
    cve_param = ",".join(cve_ids)
    try:
        resp = requests.get(
            EPSS_API_URL,
            params={"cve": cve_param},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        data = resp.json()
        result: Dict[str, tuple[float, float]] = {}
        for entry in data.get("data", []):
            cve = entry.get("cve", "").upper()
            try:
                epss = float(entry.get("epss", 0.0))
                pct = float(entry.get("percentile", 0.0))
            except (TypeError, ValueError):
                epss, pct = 0.0, 0.0
            if cve:
                result[cve] = (epss, pct)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "EPSS query failed for %d CVEs (%s); returning defaults.",
            len(cve_ids),
            exc,
        )
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_cves(cve_ids: List[str]) -> Dict[str, CVEEnrichment]:
    """
    Enrich a list of CVE IDs with EPSS scores and CISA KEV membership.

    Parameters
    ----------
    cve_ids:
        List of CVE identifiers (e.g. ['CVE-2021-44228', ...]).
        Duplicates are handled gracefully.

    Returns
    -------
    Dict[str, CVEEnrichment]
        A ``CVEEnrichment`` for every requested CVE ID.
        Keys are upper-cased CVE IDs.
        Safe defaults are returned for any CVE when API calls fail.
    """
    if not cve_ids:
        return {}

    # Normalise and deduplicate
    normalised = list(dict.fromkeys(c.strip().upper() for c in cve_ids if c))
    if not normalised:
        return {}

    logger.debug("Enriching %d unique CVEs.", len(normalised))

    # --- KEV ---
    kev_set, kev_date_map = _load_kev_set()

    # --- EPSS (chunked) ---
    epss_map: Dict[str, tuple[float, float]] = {}
    for i in range(0, len(normalised), EPSS_CHUNK_SIZE):
        chunk = normalised[i : i + EPSS_CHUNK_SIZE]
        epss_map.update(_fetch_epss_chunk(chunk))

    # --- Assemble results ---
    enriched_at = datetime.now(timezone.utc)
    results: Dict[str, CVEEnrichment] = {}
    for cve in normalised:
        in_kev = cve in kev_set
        epss_score, epss_pct = epss_map.get(cve, (0.0, 0.0))

        # Determine which sources contributed
        sources: list[str] = []
        if epss_score > 0.0:
            sources.append("epss")
        if in_kev:
            sources.append("kev")
        enrichment_source = "+".join(sources) if sources else "none"

        results[cve] = CVEEnrichment(
            cve_id=cve,
            epss=epss_score,
            epss_percentile=epss_pct,
            in_kev=in_kev,
            kev_date_added=kev_date_map.get(cve),
            enriched_at=enriched_at,
            enrichment_source=enrichment_source,
        )

    logger.debug(
        "Enrichment complete: %d KEV hits, %d EPSS hits.",
        sum(1 for r in results.values() if r.in_kev),
        sum(1 for r in results.values() if r.epss > 0.0),
    )
    return results
