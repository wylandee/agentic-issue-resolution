"""
tester_node.py — Phase 5 Tester Node for the AppSec Orchestrator.

Public API
----------
run_tester_node(state: OrchestratorState) -> Dict[str, Any]
    LangGraph node that injects ``pending_files`` into a fresh Docker sandbox,
    runs ``npm install`` (or ``npm ci``) followed by ``npm test``, and reports
    whether the test suite passes.

Design principles
-----------------
* **Host-immutable** — never writes to ``repo_root`` directly.
* **Injection order** — all pending files are written into the sandbox *before*
  the install/test commands, so the container always runs against the patched
  state.
* **Lockfile preference** — prefers ``npm ci --ignore-scripts`` when a
  ``package-lock.json`` is present in either ``pending_files`` or the host repo;
  falls back to ``npm install --ignore-scripts`` otherwise.
* **Failure isolation** — install and test failures are surfaced as
  ``test_failures`` strings so the Remedy Agent can self-correct on the next
  retry.

Statuses returned
-----------------
* ``"tested"``       — install and ``npm test`` both passed.
* ``"test_failed"``  — install or test failed; ``test_failures`` is populated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from src.orchestrator.state import OrchestratorState
from src.runtime.sandbox_mgr import DockerSandbox

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_lockfile(pending_files: Dict[str, str], repo_root: str) -> bool:
    """Return True if package-lock.json is in pending_files or the host repo."""
    if "package-lock.json" in pending_files:
        return True
    host_lockfile = Path(repo_root) / "package-lock.json"
    return host_lockfile.is_file()


# ---------------------------------------------------------------------------
# Public node function
# ---------------------------------------------------------------------------


def run_tester_node(state: OrchestratorState) -> Dict[str, Any]:
    """
    LangGraph node — Tester.

    Injects ``pending_files`` into a fresh Docker sandbox, runs install and
    ``npm test``, and returns a state-update dict.
    """
    pending_files: Dict[str, str] = state.get("pending_files", {})
    repo_root_str: str = state.get("repo_root", "")

    # ── Guard: nothing to inject ────────────────────────────────────────────
    if not pending_files:
        logger.info("tester_node: no pending_files — skipping test run.")
        return {"status": "tested", "test_failures": None}

    # ── Guard: repo_root must exist ─────────────────────────────────────────
    if not repo_root_str or not Path(repo_root_str).is_dir():
        msg = f"tester_node: repo_root '{repo_root_str}' is not a valid directory."
        logger.error(msg)
        return {"status": "test_failed", "test_failures": msg, "errors": [msg]}

    logger.info(
        "tester_node: starting sandbox to test %d pending file(s).", len(pending_files)
    )

    try:
        with DockerSandbox(repo_root_str) as sandbox:

            # ── Inject all pending files ────────────────────────────────────
            logger.info("tester_node: injecting %d file(s) into sandbox.", len(pending_files))
            for rel_path, content in pending_files.items():
                logger.debug("tester_node: injecting '%s' (%d bytes).", rel_path, len(content))
                sandbox.write_file(rel_path, content)

            # ── Install dependencies ────────────────────────────────────────
            use_ci = _has_lockfile(pending_files, repo_root_str)
            if use_ci:
                install_cmd = "npm ci"
            else:
                install_cmd = "npm install"

            logger.info("tester_node: running '%s'.", install_cmd)
            install_result = sandbox.run(install_cmd, timeout=300)

            if install_result.exit_code != 0:
                failure = (
                    f"Install failed (exit {install_result.exit_code}) "
                    f"with '{install_cmd}'.\n"
                    f"stdout:\n{install_result.stdout}\n"
                    f"stderr:\n{install_result.stderr}"
                )
                logger.warning("tester_node: %s", failure)
                return {"status": "test_failed", "test_failures": failure}

            logger.info("tester_node: install succeeded — running npm test.")

            # ── Run test suite ──────────────────────────────────────────────
            test_result = sandbox.run("npm test", timeout=300)

            if test_result.exit_code != 0:
                failure = (
                    f"npm test failed (exit {test_result.exit_code}).\n"
                    f"stdout:\n{test_result.stdout}\n"
                    f"stderr:\n{test_result.stderr}"
                )
                logger.warning("tester_node: %s", failure)
                return {"status": "test_failed", "test_failures": failure}

    except Exception as exc:  # noqa: BLE001
        msg = f"tester_node: sandbox error — {exc}"
        logger.exception("tester_node: unexpected sandbox error.")
        return {"status": "test_failed", "test_failures": msg, "errors": [msg]}

    logger.info("tester_node: all tests passed.")
    return {"status": "tested", "test_failures": None}
