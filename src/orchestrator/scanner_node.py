"""
scanner_node.py - Phase 5 Scanner Node for the AppSec Orchestrator.

This node runs OWASP Dependency-Check directly against the shared Docker named
volume created by the workspace builder, then reads the JSON report back out of
that volume for CVE and GHSA verification.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)

_ODC_TIMEOUT_SECONDS = 300
_ODC_REPORT_NAME = "dependency-check-report.json"
_ODC_CACHE_VOLUME = "odc-cache"


def _collect_target_identifiers(state: OrchestratorState) -> Set[str]:
    """Return the normalised set of CVE/GHSA identifiers from all valid_groups."""
    identifiers: Set[str] = set()
    for group in state.get("valid_groups", []):
        for cve in group.cve_ids or []:
            if cve:
                identifiers.add(cve.upper().strip())
        for ghsa in group.ghsa_ids or []:
            if ghsa:
                identifiers.add(ghsa.upper().strip())
    return identifiers

def _run_odc(workspace_volume: str) -> "subprocess.CompletedProcess[str]":
    """Execute ODC in Docker against the named workspace volume."""
    cmd = [
        "docker",
        "run",
        "--rm",
        "-u",
        "root",
        "-v",
        f"{workspace_volume}:/scan",
        "-v",
        f"{_ODC_CACHE_VOLUME}:/usr/share/dependency-check/data",
        "owasp/dependency-check:latest",
        "--project",
        "sandbox_scan",
        "--scan",
        "/scan",
        "--format",
        "JSON",
        "--out",
        "/scan",
        "--noupdate",
    ]

    extra_args = os.environ.get("ODC_EXTRA_ARGS", "").strip()
    if extra_args:
        cmd.extend(shlex.split(extra_args))

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


def _read_report_from_volume(workspace_volume: str) -> Optional[str]:
    """Read the ODC JSON report from the named volume via a tiny sandbox."""
    try:
        with DockerSandbox(
            repo_root=None,
            workspace_volume=workspace_volume,
        ) as sandbox:
            return sandbox.read_file(_ODC_REPORT_NAME)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner_node: failed to read report from volume - %s", exc)
        return None


def _parse_report(report_text: str) -> Optional[Set[str]]:
    """Parse a Dependency-Check report into the found CVE/GHSA identifiers."""
    try:
        report = json.loads(report_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner_node: failed to decode ODC report JSON - %s", exc)
        return None

    try:
        from src.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning(
            "scanner_node: src.tools.odc_parser not importable - cannot parse report."
        )
        return None

    issues = parse_vulnerabilities(report)
    found_identifiers: Set[str] = set()
    for issue in issues:
        if issue.cve_id:
            found_identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            found_identifiers.add(issue.ghsa_id.upper().strip())
    logger.info(
        "scanner_node: ODC report parsed vulnerability identifiers: %s",
        sorted(found_identifiers),
    )
    return found_identifiers


def run_scanner_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node - Scanner.

    Runs ODC against ``workspace_volume`` and checks whether any target CVE/GHSA
    identifiers from ``valid_groups`` remain present in the report.
    """
    workspace_volume: Optional[str] = state.get("workspace_volume")
    if not workspace_volume:
        msg = "scanner_node: workspace_volume is missing from state."
        logger.error(msg)
        return {"status": "scan_failed", "scan_failures": msg}

    if shutil.which("docker") is None:
        logger.warning("scanner_node: 'docker' not found on PATH - skipping ODC scan.")
        return {"status": "scanned", "scan_failures": None}

    target_identifiers = _collect_target_identifiers(state)
    logger.info(
        "scanner_node: %d target vulnerability identifier(s) to verify: %s",
        len(target_identifiers),
        ", ".join(sorted(target_identifiers)) or "(none)",
    )

    try:
        proc = _run_odc(workspace_volume)
    except FileNotFoundError:
        logger.warning("scanner_node: docker disappeared - skipping.")
        return {"status": "scanned", "scan_failures": None}
    except subprocess.TimeoutExpired:
        msg = f"scanner_node: ODC timed out after {_ODC_TIMEOUT_SECONDS}s."
        logger.error(msg)
        return {"status": "scan_failed", "scan_failures": msg}
    except Exception as exc:  # noqa: BLE001
        msg = f"scanner_node: ODC subprocess error - {exc}"
        logger.error(msg)
        return {"status": "scan_failed", "scan_failures": msg}

    report_text = _read_report_from_volume(workspace_volume)
    found_identifiers = _parse_report(report_text) if report_text is not None else None

    if proc.returncode != 0 and found_identifiers is None:
        failure = (
            f"ODC exited {proc.returncode} and produced no parseable report.\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        logger.error("scanner_node: %s", failure)
        return {"status": "scan_failed", "scan_failures": failure}

    if found_identifiers is None:
        failure = (
            f"scanner_node: ODC report not parseable (exit {proc.returncode}).\n"
            f"stderr:\n{proc.stderr}"
        )
        logger.warning(failure)
        return {"status": "scan_failed", "scan_failures": failure}

    remaining = target_identifiers & found_identifiers
    if remaining:
        failure_msg = (
            "ODC Scan failed: the following vulnerability identifier(s) are still "
            f"present after patching: {', '.join(sorted(remaining))}"
        )
        logger.warning("scanner_node: %s", failure_msg)
        return {"status": "scan_failed", "scan_failures": failure_msg}

    logger.info("scanner_node: all target vulnerability identifiers resolved - scan passed.")
    return {"status": "scanned", "scan_failures": None}
