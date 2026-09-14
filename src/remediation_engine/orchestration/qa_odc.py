"""OWASP Dependency-Check execution and scan result handling for QA."""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from remediation_engine.contracts.schemas import ScannerExecutionStatus, VulnerabilityIssue
from remediation_engine.orchestration.runtime_context import get_runtime_settings
from remediation_engine.runtime.sandbox_mgr import DockerSandbox

from .qa_types import (
    _exception_stream,
    _QALogRecord,
    _SecurityScanResult,
    _subprocess_text,
    _validate_qa_path,
)

logger = logging.getLogger(__name__)


_ODC_TIMEOUT_SECONDS = 300
_ODC_REPORT_NAME = "dependency-check-report.json"
_ODC_HTML_REPORT_NAME = "dependency-check-report.html"
_ODC_CACHE_VOLUME = "odc-cache"
_ODC_DEBUG_DIR = Path("data/cache/qa_reports")
_ODC_LOG_DIR = _ODC_DEBUG_DIR / "odc_logs"
_ODC_CLEANUP_TIMEOUT_SECONDS = 30
_ODC_INTERNAL_FULL_SCAN_EXCLUDES = (
    "**/.remedy-attempt-snapshots/**",
    "**/.odc-targeted/**",
)


@dataclass(frozen=True)
class _ODCExecution:
    """Host-side identity and diagnostics location for one ODC invocation."""

    run_id: str
    scope: str
    container_name: str
    log_path: Path
    started_at_utc: str


@dataclass(frozen=True)
class _ODCCleanupResult:
    """Outcome of the best-effort forced removal of an ODC container."""

    container_name: str
    succeeded: bool
    already_absent: bool
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None


class _ODCScanTimeout(subprocess.TimeoutExpired):
    """Timeout carrying ODC diagnostics and container-cleanup evidence."""

    def __init__(
        self,
        original: subprocess.TimeoutExpired,
        execution: _ODCExecution,
        cleanup: _ODCCleanupResult,
        log_path: Path | None,
    ) -> None:
        super().__init__(
            original.cmd,
            original.timeout,
            output=getattr(original, "output", None),
            stderr=getattr(original, "stderr", None),
        )
        self.run_id = execution.run_id
        self.scope = execution.scope
        self.container_name = execution.container_name
        self.cleanup = cleanup
        self.log_path = log_path


def _new_odc_execution(scope: str) -> _ODCExecution:
    """Allocate a unique Docker container name and host diagnostic path."""
    run_id = uuid4().hex
    return _ODCExecution(
        run_id=run_id,
        scope=scope,
        container_name=f"remedy-odc-{run_id}",
        log_path=_ODC_LOG_DIR / f"{run_id}-{scope}.json",
        started_at_utc=datetime.now(UTC).isoformat(),
    )


def _redact_odc_command(command: Sequence[str]) -> list[str]:
    """Redact values for the small set of secret-bearing ODC options."""
    sensitive_options = {
        "--apikey",
        "--nvdapikey",
        "--proxypass",
        "--password",
    }
    redacted: list[str] = []
    redact_next = False
    for argument in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        redacted.append(argument)
        if argument.lower() in sensitive_options:
            redact_next = True
    return redacted


def _cleanup_odc_container(container_name: str) -> _ODCCleanupResult:
    """Force-remove one named ODC container after a scan timeout.

    ``docker rm -f`` both stops a running container and removes it.  A
    ``No such container`` response is treated as success because ``--rm`` may
    have won a cleanup race.  All other failures are returned to the caller so
    the timeout remains a hard QA failure while preserving the cleanup issue.
    """
    command = ["docker", "rm", "-f", container_name]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_ODC_CLEANUP_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _ODCCleanupResult(
            container_name=container_name,
            succeeded=False,
            already_absent=False,
            returncode=None,
            stdout=_subprocess_text(getattr(exc, "stdout", None)),
            stderr=_subprocess_text(getattr(exc, "stderr", None)),
            error=(f"docker rm -f timed out after {_ODC_CLEANUP_TIMEOUT_SECONDS}s"),
        )
    except Exception as exc:  # noqa: BLE001
        return _ODCCleanupResult(
            container_name=container_name,
            succeeded=False,
            already_absent=False,
            returncode=None,
            stdout="",
            stderr="",
            error=f"docker rm -f failed to start: {exc}",
        )

    stdout = _subprocess_text(getattr(result, "stdout", None))
    stderr = _subprocess_text(getattr(result, "stderr", None))
    returncode = getattr(result, "returncode", None)
    detail = f"{stdout}\n{stderr}".strip()
    if returncode == 0:
        return _ODCCleanupResult(
            container_name=container_name,
            succeeded=True,
            already_absent=False,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    if "no such container" in detail.lower():
        return _ODCCleanupResult(
            container_name=container_name,
            succeeded=True,
            already_absent=True,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return _ODCCleanupResult(
        container_name=container_name,
        succeeded=False,
        already_absent=False,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=detail or f"docker rm -f exited {returncode}",
    )


def _write_odc_diagnostic_log(
    execution: _ODCExecution,
    command: Sequence[str],
    *,
    status: str,
    process: subprocess.CompletedProcess[str] | None = None,
    error: BaseException | None = None,
    cleanup: _ODCCleanupResult | None = None,
) -> Path | None:
    """Persist one ODC invocation's command, output, and lifecycle evidence."""
    stdout = _subprocess_text(
        getattr(process, "stdout", None) if process is not None else getattr(error, "stdout", None)
    )
    stderr = _subprocess_text(
        getattr(process, "stderr", None) if process is not None else getattr(error, "stderr", None)
    )
    exit_code = getattr(process, "returncode", None) if process is not None else None
    cleanup_payload: dict[str, Any] | None = None
    if cleanup is not None:
        cleanup_payload = {
            "container_name": cleanup.container_name,
            "succeeded": cleanup.succeeded,
            "already_absent": cleanup.already_absent,
            "returncode": cleanup.returncode,
            "stdout": cleanup.stdout,
            "stderr": cleanup.stderr,
            "error": cleanup.error,
        }
    payload = {
        "run_id": execution.run_id,
        "scope": execution.scope,
        "container_name": execution.container_name,
        "started_at_utc": execution.started_at_utc,
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "exit_code": exit_code,
        "timeout_seconds": _ODC_TIMEOUT_SECONDS,
        "command": _redact_odc_command(command),
        "stdout": stdout,
        "stderr": stderr,
        "error": str(error) if error is not None else None,
        "cleanup": cleanup_payload,
    }
    try:
        execution.log_path.parent.mkdir(parents=True, exist_ok=True)
        execution.log_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        resolved_path = execution.log_path.resolve()
        logger.info("qa_critic: ODC diagnostic log saved to %s", resolved_path)
        return resolved_path
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "qa_critic: failed to persist ODC diagnostic log to %s - %s",
            execution.log_path,
            exc,
        )
        return None


def _odc_process_metadata(process: Any, key: str) -> Any:
    """Read metadata attached by the ODC execution helper without MagicMock leakage."""
    metadata = getattr(process, "__dict__", {})
    return metadata.get(key) if isinstance(metadata, dict) else None


def _odc_log_note(process: Any) -> str:
    """Return a user-facing note for a persisted ODC diagnostic log."""
    log_path = _odc_process_metadata(process, "odc_log_path")
    return f"\nODC diagnostic log saved to: {log_path}" if log_path else ""


def _odc_exception_log_note(error: BaseException) -> str:
    """Return a user-facing diagnostic-log note attached to a scan exception."""
    log_path = getattr(error, "odc_log_path", None)
    return f"\nODC diagnostic log saved to: {log_path}" if log_path else ""


def _odc_timeout_summary(exc: subprocess.TimeoutExpired) -> str:
    """Build a timeout summary that preserves log and cleanup diagnostics."""
    summary = f"FAILURE: Dependency-Check timed out after {_ODC_TIMEOUT_SECONDS}s."
    log_path = getattr(exc, "log_path", None)
    if log_path:
        summary += f"\nODC diagnostic log saved to: {log_path}"
    cleanup = getattr(exc, "cleanup", None)
    if isinstance(cleanup, _ODCCleanupResult):
        if cleanup.succeeded:
            cleanup_state = "already absent" if cleanup.already_absent else "removed"
            summary += (
                f"\nODC container cleanup completed ({cleanup_state}): {cleanup.container_name}"
            )
        else:
            summary += (
                f"\nODC container cleanup FAILED for {cleanup.container_name}: "
                f"{cleanup.error or 'unknown cleanup error'}"
            )
    return summary


def _run_odc_process(
    workspace_volume: str,
    scan_subdir: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ODC with a unique container identity and durable diagnostics."""
    scope = "targeted" if scan_subdir else "full"
    execution = _new_odc_execution(scope)
    cmd = _odc_command(workspace_volume, scan_subdir, execution.container_name)
    logger.info("qa_critic: running %s ODC in Docker: %s", scope, " ".join(cmd))
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_ODC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        cleanup = _cleanup_odc_container(execution.container_name)
        log_path = _write_odc_diagnostic_log(
            execution,
            cmd,
            status="timeout",
            error=exc,
            cleanup=cleanup,
        )
        if not cleanup.succeeded:
            logger.error(
                "qa_critic: failed to remove timed-out ODC container %s: %s",
                execution.container_name,
                cleanup.error,
            )
        raise _ODCScanTimeout(exc, execution, cleanup, log_path) from exc
    except Exception as exc:  # noqa: BLE001
        log_path = _write_odc_diagnostic_log(
            execution,
            cmd,
            status="subprocess_error",
            error=exc,
        )
        try:
            exc.__dict__.update(
                {
                    "odc_run_id": execution.run_id,
                    "odc_container_name": execution.container_name,
                    "odc_log_path": log_path,
                }
            )
        except Exception:  # noqa: BLE001
            logger.debug(
                "qa_critic: unable to attach ODC error metadata",
                exc_info=True,
            )
        raise

    log_path = _write_odc_diagnostic_log(
        execution,
        cmd,
        status="completed",
        process=process,
    )
    try:
        process.__dict__.update(
            {
                "odc_run_id": execution.run_id,
                "odc_container_name": execution.container_name,
                "odc_log_path": log_path,
            }
        )
    except Exception:  # noqa: BLE001
        logger.debug("qa_critic: unable to attach ODC diagnostic metadata", exc_info=True)
    return process


def _read_report_from_workspace(
    sandbox: DockerSandbox,
    relative_dir: str = "",
) -> str | None:
    """Read an ODC JSON report from a workspace-relative report directory."""
    try:
        clean_dir = relative_dir.strip("/\\")
        report_path = f"{clean_dir}/{_ODC_REPORT_NAME}".lstrip("/")
        return sandbox.read_file(report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to read ODC report from workspace â€” %s", exc)
        return None


def _persist_workspace_report_to_host(
    sandbox: DockerSandbox,
    workspace_name: str,
    host_path: Path,
) -> Path | None:
    """Copy a Dependency-Check report from the workspace volume onto the host."""
    try:
        content = sandbox.read_file(workspace_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to read %s from workspace - %s", workspace_name, exc)
        return None

    if content is None:
        return None

    try:
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "qa_critic: failed to persist %s to host path %s - %s",
            workspace_name,
            host_path,
            exc,
        )
        return None

    return host_path.resolve()


def _next_html_report_host_path() -> Path:
    """Return a unique host-side HTML report path for one QA scan."""
    timestamp_ms = int(time.time() * 1000)
    return _ODC_DEBUG_DIR / f"dependency-check-report-{timestamp_ms}.html"


def _parse_report_identifiers(report_text: str) -> set[str] | None:
    """Parse CVE/GHSA identifiers from the ODC JSON report text."""
    issues = _parse_report_issues(report_text)
    if issues is None:
        return None

    identifiers: set[str] = set()
    for issue in issues:
        if issue.cve_id:
            identifiers.add(issue.cve_id.upper().strip())
        if issue.ghsa_id:
            identifiers.add(issue.ghsa_id.upper().strip())
    return identifiers


def _parse_report_issues(report_text: str) -> list[VulnerabilityIssue] | None:
    """Parse the complete typed vulnerability snapshot from an ODC report."""
    try:
        report = json.loads(report_text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to decode ODC report JSON â€” %s", exc)
        return None

    try:
        from remediation_engine.tools.odc_parser import parse_vulnerabilities
    except ImportError:
        logger.warning("qa_critic: remediation_engine.tools.odc_parser not importable.")
        return None

    try:
        return parse_vulnerabilities(report)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qa_critic: failed to parse ODC vulnerabilities â€” %s", exc)
        return None


def _odc_command(
    workspace_volume: str,
    scan_subdir: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    """Build a safe ODC Docker command for the full or targeted workspace."""
    scan_path = "/scan"
    if scan_subdir:
        scan_subdir = _validate_qa_path(scan_subdir)
        scan_path = f"/scan/{scan_subdir}"
    if container_name is None:
        container_name = _new_odc_execution("targeted" if scan_subdir else "full").container_name
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
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
        scan_path,
        "--format",
        "JSON",
        "--format",
        "HTML",
        "--out",
        scan_path,
        "--noupdate",
    ]

    if scan_subdir is None:
        for pattern in _ODC_INTERNAL_FULL_SCAN_EXCLUDES:
            cmd.extend(["--exclude", pattern])

    extra_args = get_runtime_settings().odc_extra_args
    if extra_args:
        cmd.extend(shlex.split(extra_args))

    return cmd


def _run_odc(workspace_volume: str) -> subprocess.CompletedProcess[str]:
    """Execute OWASP Dependency-Check in Docker against the shared workspace volume."""
    return _run_odc_process(workspace_volume)


def _run_targeted_odc(
    workspace_volume: str,
    targeted_subdir: str,
) -> subprocess.CompletedProcess[str]:
    """Execute ODC against a validated workspace-relative targeted directory."""
    return _run_odc_process(workspace_volume, targeted_subdir)


def _record_scan_result(
    result: _SecurityScanResult,
    *,
    label: str,
    exit_code: int | None,
    stdout: str | None,
    stderr: str | None,
    diagnostic_log_path: Any = None,
    error: str | None = None,
) -> _SecurityScanResult:
    """Attach private process metadata and one immutable scan log record."""
    raw_stdout = _subprocess_text(stdout)
    raw_stderr = _subprocess_text(stderr)
    path = str(diagnostic_log_path) if diagnostic_log_path else None
    record = _QALogRecord(
        phase="scan",
        label=label,
        exit_code=exit_code,
        stdout=raw_stdout,
        stderr=raw_stderr,
        error=error,
    )
    return replace(
        result,
        exit_code=exit_code,
        raw_stdout=raw_stdout,
        raw_stderr=raw_stderr,
        diagnostic_log_path=path,
        scan_records=(record,),
    )


def _run_security_scan(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str] | None = None,
) -> _SecurityScanResult:
    """Run Dependency-Check and return typed scan evidence.

    Args:
        sandbox: Active QA sandbox containing the post-remediation workspace.
        workspace_volume: Docker volume mounted into Dependency-Check.
        target_identifiers: Vulnerability identifiers owned by the active QA
            attempt.
        baseline_identifiers: Pre-remediation identifiers used to distinguish
            remaining target findings from newly introduced findings. When
            omitted, the target identifiers provide the comparison baseline.

    Returns:
        An ``_SecurityScanResult`` with named identifier sets, execution
        status, process streams, diagnostic-log path, and immutable scan
        records.
    """
    baseline = {
        identifier.upper().strip()
        for identifier in (
            baseline_identifiers if baseline_identifiers is not None else target_identifiers
        )
        if identifier and identifier.strip()
    }

    def finish(
        result: _SecurityScanResult,
        *,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        diagnostic_log_path: Any = None,
        error: str | None = None,
    ) -> _SecurityScanResult:
        return _record_scan_result(
            result,
            label="odc:full",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=error,
        )

    if shutil.which("docker") is None:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        logger.warning("qa_critic: %s", msg)
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.DOCKER_UNAVAILABLE,
            ),
            error=msg,
        )

    try:
        proc = _run_odc(workspace_volume)
    except FileNotFoundError as exc:
        msg = (
            "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
            + _odc_exception_log_note(exc)
        )
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.DOCKER_UNAVAILABLE,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "odc_log_path", None),
            error=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        msg = _odc_timeout_summary(exc)
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.TIMEOUT,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "log_path", None),
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"FAILURE: Dependency-Check subprocess error — {exc}"
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "odc_log_path", None),
            error=str(exc),
        )

    try:
        exit_code = int(getattr(proc, "returncode", 1))
    except (TypeError, ValueError):
        exit_code = None
    stdout = _subprocess_text(getattr(proc, "stdout", ""))
    stderr = _subprocess_text(getattr(proc, "stderr", ""))
    diagnostic_log_path = _odc_process_metadata(proc, "odc_log_path")
    saved_html_report = _persist_workspace_report_to_host(
        sandbox,
        _ODC_HTML_REPORT_NAME,
        _next_html_report_host_path(),
    )
    saved_json_report = _persist_workspace_report_to_host(
        sandbox,
        _ODC_REPORT_NAME,
        _ODC_DEBUG_DIR / _ODC_REPORT_NAME,
    )
    report_location_note = _odc_log_note(proc)
    if saved_html_report is not None:
        report_location_note += f"\nHTML report saved to: {saved_html_report}"
        if saved_json_report is not None:
            report_location_note += f"\nJSON report saved to: {saved_json_report}"

    report_text = _read_report_from_workspace(sandbox)
    found_identifiers = _parse_report_identifiers(report_text) if report_text is not None else None
    found_issues = _parse_report_issues(report_text) if report_text is not None else None
    # Legacy direct callers/tests may replace the identifier-only parser. In
    # that compatibility mode there is no typed issue snapshot to propagate,
    # but the identifier scan can still be evaluated normally.
    if found_issues is None and found_identifiers is not None:
        found_issues = []

    if exit_code != 0 and (found_identifiers is None or found_issues is None):
        summary = (
            f"FAILURE: Dependency-Check exited {exit_code} and produced "
            "no parseable report.\n"
            f"stdout:\n{stdout[:2000]}\n"
            f"stderr:\n{stderr[:2000]}"
        )
        summary += report_location_note
        return finish(
            _SecurityScanResult(
                False,
                summary,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=summary,
        )

    if found_identifiers is None or found_issues is None:
        summary = (
            "FAILURE: Dependency-Check report was not parseable "
            f"(exit {exit_code}).\n"
            f"stderr:\n{stderr[:2000]}"
        )
        summary += report_location_note
        return finish(
            _SecurityScanResult(
                False,
                summary,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=summary,
        )

    found_identifiers = {
        identifier.upper().strip() for identifier in found_identifiers if identifier
    }
    remaining = {ident.upper().strip() for ident in target_identifiers if ident}
    remaining &= found_identifiers
    new_identifiers = found_identifiers - baseline

    if remaining:
        remaining_text = ", ".join(sorted(remaining))
        summary = (
            "FAILURE: Dependency-Check found unresolved target vulnerabilities. "
            f"Remaining identifiers: {remaining_text}"
        )
        if new_identifiers:
            summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
        summary += report_location_note
        return finish(
            _SecurityScanResult(
                False,
                summary,
                remaining,
                found_identifiers,
                new_identifiers,
                found_issues,
                execution_status=ScannerExecutionStatus.SUCCESS,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
        )

    summary = "Dependency-Check found no remaining target vulnerability identifiers."
    if new_identifiers:
        summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
    summary += report_location_note
    return finish(
        _SecurityScanResult(
            True,
            summary,
            set(),
            found_identifiers,
            new_identifiers,
            found_issues,
            execution_status=ScannerExecutionStatus.SUCCESS,
        ),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        diagnostic_log_path=diagnostic_log_path,
    )


def _targeted_report_host_path(targeted_subdir: str, suffix: str) -> Path:
    """Return an ignored, unique host path for a targeted ODC report."""
    stamp = f"{time.time_ns()}-{abs(hash(targeted_subdir))}"
    return _ODC_DEBUG_DIR / "targeted" / f"{stamp}-{suffix}"


def _run_targeted_security_scan(
    sandbox: DockerSandbox,
    workspace_volume: str,
    target_identifiers: set[str],
    baseline_identifiers: set[str],
    targeted_subdir: str,
) -> _SecurityScanResult:
    """Run and classify ODC against a synthetic targeted workspace."""
    baseline = {
        identifier.upper().strip()
        for identifier in baseline_identifiers
        if identifier and identifier.strip()
    }

    def finish(
        result: _SecurityScanResult,
        *,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        diagnostic_log_path: Any = None,
        error: str | None = None,
    ) -> _SecurityScanResult:
        return _record_scan_result(
            result,
            label="odc:targeted",
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=error,
        )

    if shutil.which("docker") is None:
        msg = "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.DOCKER_UNAVAILABLE,
            ),
            error=msg,
        )

    try:
        proc = _run_targeted_odc(workspace_volume, _validate_qa_path(targeted_subdir))
    except FileNotFoundError as exc:
        msg = (
            "FAILURE: docker is not available on PATH; Dependency-Check cannot run."
            + _odc_exception_log_note(exc)
        )
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.DOCKER_UNAVAILABLE,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "odc_log_path", None),
            error=str(exc),
        )
    except subprocess.TimeoutExpired as exc:
        msg = _odc_timeout_summary(exc)
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.TIMEOUT,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "log_path", None),
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"FAILURE: Dependency-Check subprocess error — {exc}"
        return finish(
            _SecurityScanResult(
                False,
                msg,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            stdout=_exception_stream(exc, "stdout"),
            stderr=_exception_stream(exc, "stderr"),
            diagnostic_log_path=getattr(exc, "odc_log_path", None),
            error=str(exc),
        )

    try:
        exit_code = int(getattr(proc, "returncode", 1))
    except (TypeError, ValueError):
        exit_code = None
    stdout = _subprocess_text(getattr(proc, "stdout", ""))
    stderr = _subprocess_text(getattr(proc, "stderr", ""))
    diagnostic_log_path = _odc_process_metadata(proc, "odc_log_path")
    report_dir = _validate_qa_path(targeted_subdir)
    report_json = f"{report_dir}/{_ODC_REPORT_NAME}"
    report_html = f"{report_dir}/{_ODC_HTML_REPORT_NAME}"
    saved_html = _persist_workspace_report_to_host(
        sandbox,
        report_html,
        _targeted_report_host_path(report_dir, _ODC_HTML_REPORT_NAME),
    )
    saved_json = _persist_workspace_report_to_host(
        sandbox,
        report_json,
        _targeted_report_host_path(report_dir, _ODC_REPORT_NAME),
    )
    report_location_note = _odc_log_note(proc)
    if saved_html is not None:
        report_location_note += f"\nHTML report saved to: {saved_html}"
        if saved_json is not None:
            report_location_note += f"\nJSON report saved to: {saved_json}"

    report_text = _read_report_from_workspace(sandbox, report_dir)
    found_identifiers = _parse_report_identifiers(report_text) if report_text is not None else None
    found_issues = _parse_report_issues(report_text) if report_text is not None else None
    if found_issues is None and found_identifiers is not None:
        found_issues = []
    if exit_code != 0 and (found_identifiers is None or found_issues is None):
        summary = (
            f"FAILURE: Dependency-Check exited {exit_code} and produced no parseable report.\n"
            f"stdout:\n{stdout[:2000]}\n"
            f"stderr:\n{stderr[:2000]}"
        )
        return finish(
            _SecurityScanResult(
                False,
                summary + report_location_note,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=summary,
        )
    if found_identifiers is None or found_issues is None:
        summary = (
            "FAILURE: Dependency-Check report was not parseable "
            f"(exit {exit_code}).\n"
            f"stderr:\n{stderr[:2000]}"
        )
        return finish(
            _SecurityScanResult(
                False,
                summary + report_location_note,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=summary,
        )
    if exit_code != 0:
        summary = (
            f"FAILURE: targeted Dependency-Check exited {exit_code}.\nstderr:\n{stderr[:2000]}"
        )
        return finish(
            _SecurityScanResult(
                False,
                summary + report_location_note,
                set(),
                set(),
                set(),
                execution_status=ScannerExecutionStatus.UNPARSEABLE,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
            error=summary,
        )

    found = {identifier.upper().strip() for identifier in found_identifiers if identifier}
    remaining = {
        identifier.upper().strip() for identifier in target_identifiers if identifier
    } & found
    new_identifiers = found - baseline
    if remaining:
        summary = (
            "FAILURE: Dependency-Check found unresolved target vulnerabilities. "
            f"Remaining identifiers: {', '.join(sorted(remaining))}"
        )
        if new_identifiers:
            summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
        return finish(
            _SecurityScanResult(
                False,
                summary + report_location_note,
                remaining,
                found,
                new_identifiers,
                found_issues,
                execution_status=ScannerExecutionStatus.SUCCESS,
            ),
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            diagnostic_log_path=diagnostic_log_path,
        )

    summary = "Dependency-Check found no remaining target vulnerability identifiers."
    if new_identifiers:
        summary += f" Newly introduced identifiers: {', '.join(sorted(new_identifiers))}."
    return finish(
        _SecurityScanResult(
            True,
            summary + report_location_note,
            set(),
            found,
            new_identifiers,
            found_issues,
            execution_status=ScannerExecutionStatus.SUCCESS,
        ),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        diagnostic_log_path=diagnostic_log_path,
    )
