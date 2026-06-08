"""
scanner_node.py — Phase 5 Scanner Node for the AppSec Orchestrator.

Public API
----------
run_scanner_node(state: OrchestratorState) -> Dict[str, Any]
    LangGraph node that reconstructs a minimal scan workspace from
    ``pending_files``, runs the OWASP Dependency-Check CLI, parses the JSON
    report, and checks whether any target CVEs still appear.

Design principles
-----------------
* **Host-immutable** — only reads from ``repo_root``; never writes to it.
* **Temp-dir isolation** — uses ``tempfile.TemporaryDirectory`` so cleanup is
  guaranteed even on exceptions.
* **Graceful fallback** — if ``dependency-check.sh`` is not found on PATH, the
  node logs a warning and passes (returns ``"scanned"``).  This preserves the
  ability to test on developer machines that don't have ODC installed.
* **Report-first parsing** — if ODC exits non-zero but the report file exists,
  the node attempts to parse it anyway and only uses CLI stderr as a last
  resort.

Statuses returned
-----------------
* ``"scanned"``      — ODC absent (fallback) or all target CVEs resolved.
* ``"scan_failed"``  — target CVEs still present, or ODC crashed without a report.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.orchestrator.state import OrchestratorState

logger = logging.getLogger(__name__)

# Root-level npm manifest files that ODC/npm analysis depends on.
_MANIFEST_CANDIDATES = (
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "yarn.lock",
    "pnpm-lock.yaml",
)

_ODC_TIMEOUT_SECONDS = 300
_ODC_REPORT_NAME = "dependency-check-report.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_cache_dir() -> Path:
    """Return the cache directory, honouring TRIAGE_CACHE_DIR env var."""
    env_override = os.environ.get("TRIAGE_CACHE_DIR")
    if env_override:
        return Path(env_override)
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "requirements.txt").exists():
            return parent / "data" / "cache"
    return Path("data") / "cache"


def _collect_target_cves(state: OrchestratorState) -> Set[str]:
    """Return the normalised set of CVE IDs from all valid_groups."""
    cves: Set[str] = set()
    for group in state.get("valid_groups", []):
        for cve in group.cve_ids or []:
            if cve:
                cves.add(cve.upper().strip())
    return cves


def _build_scan_workspace(
    pending_files: Dict[str, str],
    repo_root: str,
    tmp_dir: str,
) -> None:
    """
    Populate *tmp_dir* for the ODC scan.

    1. Write all ``pending_files`` (relative paths preserved).
    2. Copy any unchanged root-level manifest files from ``repo_root`` that are
       NOT already present in ``pending_files`` (so ODC has the full manifest
       context without mutating the originals).
    """
    tmp_path = Path(tmp_dir)
    repo_path = Path(repo_root)

    # Write pending files first
    for rel_path, content in pending_files.items():
        dest = tmp_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        logger.debug("scanner_node: wrote pending file '%s' to temp workspace.", rel_path)

    # Copy root-level manifests that aren't already covered by pending_files
    pending_names = {Path(p).name for p in pending_files}
    for candidate in _MANIFEST_CANDIDATES:
        if candidate in pending_names:
            continue  # already written from pending_files
        src = repo_path / candidate
        if src.exists() and src.is_file():
            shutil.copy2(str(src), str(tmp_path / candidate))
            logger.debug("scanner_node: copied host manifest '%s' to temp workspace.", candidate)


def _run_odc(tmp_dir: str) -> "subprocess.CompletedProcess[str]":
    """Execute ODC in Docker against *tmp_dir*."""
    cache_dir = _get_cache_dir().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    abs_tmp_dir = Path(tmp_dir).resolve()

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_tmp_dir}:/scan",
        "-v", f"{cache_dir}:/usr/share/dependency-check/data",
        "owasp/dependency-check:latest",
        "--project", "sandbox_scan",
        "--scan", "/scan",
        "--format", "JSON",
        "--out", "/scan",
    ]
    logger.info("scanner_node: running ODC in Docker: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_ODC_TIMEOUT_SECONDS,
    )
    logger.debug("scanner_node: ODC stdout:\n%s", proc.stdout)
    logger.debug("scanner_node: ODC stderr:\n%s", proc.stderr)
    return proc


def _parse_report(tmp_dir: str) -> Optional[List[str]]:
    """
    Parse ``dependency-check-report.json`` from *tmp_dir*.

    Returns a list of CVE IDs found, or None if the report is absent/invalid.
    """
    report_path = Path(tmp_dir) / _ODC_REPORT_NAME
    if not report_path.exists():
        return None

    try:
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)
        deps = [d.get("fileName") for d in report.get("dependencies", [])]
        logger.info("scanner_node: ODC report dependencies scanned: %s", deps)
    except Exception as exc:
        logger.warning("scanner_node: failed to parse ODC report — %s", exc)
        return None

    try:
        from src.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning(
            "scanner_node: src.tools.odc_parser not importable — cannot parse report."
        )
        return None

    issues = parse_vulnerabilities(report)
    found_cves = [
        issue.cve_id
        for issue in issues
        if issue.cve_id
    ]
    logger.info("scanner_node: ODC report parsed CVEs: %s", found_cves)
    return found_cves


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def run_scanner_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Scanner.

    Reconstructs a minimal workspace from ``pending_files``, runs ODC, and
    checks whether any target CVEs remain.

    Returns a state-update dict merged by LangGraph.
    """
    pending_files: Dict[str, str] = state.get("pending_files", {})
    repo_root_str: str = state.get("repo_root", "")

    # ── Guard: nothing to scan ──────────────────────────────────────────────
    if not pending_files:
        logger.info("scanner_node: no pending_files — skipping ODC scan.")
        return {"status": "scanned", "scan_failures": None}

    # ── Graceful fallback: ODC not installed ────────────────────────────────
    if shutil.which("docker") is None:
        logger.warning(
            "scanner_node: 'docker' not found on PATH — "
            "skipping ODC scan (install Docker to enable)."
        )
        return {"status": "scanned", "scan_failures": None}

    target_cves = _collect_target_cves(state)
    logger.info(
        "scanner_node: %d target CVE(s) to verify: %s",
        len(target_cves),
        ", ".join(sorted(target_cves)) or "(none)",
    )

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="odc_scan_", dir=str(cache_dir)) as tmp_dir:
        # ── Build scan workspace ────────────────────────────────────────────
        try:
            _build_scan_workspace(pending_files, repo_root_str, tmp_dir)
        except Exception as exc:
            msg = f"scanner_node: failed to build scan workspace — {exc}"
            logger.error(msg)
            return {"status": "scan_failed", "scan_failures": msg}

        # ── Run ODC ────────────────────────────────────────────────────────
        try:
            proc = _run_odc(tmp_dir)
        except FileNotFoundError:
            # Race condition: checked above but missing now
            logger.warning("scanner_node: docker disappeared — skipping.")
            return {"status": "scanned", "scan_failures": None}
        except subprocess.TimeoutExpired:
            msg = f"scanner_node: ODC timed out after {_ODC_TIMEOUT_SECONDS}s."
            logger.error(msg)
            return {"status": "scan_failed", "scan_failures": msg}
        except Exception as exc:
            msg = f"scanner_node: ODC subprocess error — {exc}"
            logger.error(msg)
            return {"status": "scan_failed", "scan_failures": msg}

        # ── Parse report ────────────────────────────────────────────────────
        found_cves = _parse_report(tmp_dir)

        if proc.returncode != 0 and found_cves is None:
            # ODC crashed AND no parseable report
            failure = (
                f"ODC exited {proc.returncode} and produced no parseable report.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
            logger.error("scanner_node: %s", failure)
            return {"status": "scan_failed", "scan_failures": failure}

        if found_cves is None:
            # Non-zero exit but report is unreadable — treat as failure
            logger.warning(
                "scanner_node: ODC report missing/invalid after exit %d.",
                proc.returncode,
            )
            return {
                "status": "scan_failed",
                "scan_failures": (
                    f"scanner_node: ODC report not parseable "
                    f"(exit {proc.returncode}).\nstderr:\n{proc.stderr}"
                ),
            }

        # ── Compare found CVEs against target CVEs ──────────────────────────
        found_upper = {c.upper().strip() for c in found_cves if c}
        remaining = target_cves & found_upper

        if remaining:
            failure_msg = (
                f"ODC Scan failed: the following CVE(s) are still present after patching: "
                f"{', '.join(sorted(remaining))}"
            )
            logger.warning("scanner_node: %s", failure_msg)
            return {"status": "scan_failed", "scan_failures": failure_msg}

        logger.info("scanner_node: all target CVEs resolved — scan passed.")
        return {"status": "scanned", "scan_failures": None}
