"""Strict package-manager certification for solver-backed npm portfolios.

The solver-only plan is a useful preview, but it is not safe to dispatch until
npm resolves the complete selected assignment in an isolated copy of the
workspace. This module owns that certification boundary and never writes to the
host repository.
"""

from __future__ import annotations

import json
import math
import shlex
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from remediation_engine.contracts.schemas import (
    PackageMutation,
    PortfolioPlan,
    QAAttemptResult,
    RemediationTask,
    TaskStatus,
    VulnerabilityGroup,
)
from remediation_engine.contracts.solver_models import (
    CertificationStatistics,
    PackageResolutionCertificate,
    PackageResolutionStatus,
    PortfolioReplanRequest,
    QAPassedWorkspacePrefix,
    SolverBatch,
    SolverCandidateConflict,
    SolverCandidateCutKind,
    SolverCandidateLiteral,
    SolverCandidatePlan,
    SolverCandidateRejectionReason,
    SolverFindingRequirement,
    SolverMutation,
    SolverRemediationPlan,
    SolverRuntimeFingerprint,
    SolverStatus,
    SolverVersionCandidate,
)
from remediation_engine.orchestration._tool_support import _workspace_dir_for_manifest
from remediation_engine.orchestration.portfolio_solver import (
    _digest,
    _issue_identity,
    _prepare_portfolio_problem,
    _PreparedPortfolioProblem,
    _project_prepared_portfolio_plan,
)
from remediation_engine.orchestration.qa_test_parsing import (
    _install_error_category,
    parse_peer_conflict_evidence,
)
from remediation_engine.orchestration.task_utils import (
    TERMINAL_TASK_STATUSES,
    effective_group_status,
)
from remediation_engine.orchestration.tools_manifest import (
    _capture_package_checkpoint,
    _override_target_was_pruned,
    _package_checkpoint_paths,
    _PackageCheckpoint,
    _restore_package_checkpoint,
    _validate_workspace_path,
    _verify_lockfile_mutations,
)
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.settings import AppSettings
from remediation_engine.solver.cpsat import solve_portfolio
from remediation_engine.tools.npm_graph import (
    NpmGraphSnapshot,
    check_npm_range,
    load_npm_graph_snapshot,
    load_npm_graph_snapshot_from_documents,
    resolve_lockfile_dependency_package,
)
from remediation_engine.tools.registry_cache import PackumentFetcher

_NPM_INSTALL = (
    "npm install --package-lock-only --ignore-scripts --no-audit --no-fund "
    "--legacy-peer-deps --engine-strict=true --force=false"
)


class _CertificationUnknown(RuntimeError):
    """Raised when resolver evidence is missing or cannot be trusted."""


class _AssignmentRejected(RuntimeError):
    """Raised when typed resolver evidence proves a candidate incompatible."""

    def __init__(self, conflict: SolverCandidateConflict) -> None:
        self.conflict = conflict
        super().__init__(conflict.summary)


@dataclass(frozen=True)
class _AssignmentResult:
    """Resolver result retained long enough to make a certificate or no-good."""

    status: PackageResolutionStatus
    snapshot: NpmGraphSnapshot | None
    covered_coverage_ids: tuple[str, ...]
    workaround_coverage_ids: tuple[str, ...]
    unresolved_coverage_ids: tuple[str, ...]
    resolved_lockfile_digests: Mapping[str, str]
    batch_prefix_graph_digests: Mapping[str, str]
    diagnostics: tuple[str, ...]
    rejection_conflict: SolverCandidateConflict | None = None


def _remaining_seconds(deadline: float) -> float:
    """Return non-negative time remaining in the shared certification budget."""
    return max(0.0, deadline - time.monotonic())


def _new_certification_metrics() -> dict[str, Any]:
    """Create bounded mutable counters scoped to one certification invocation."""
    return {
        "solver_calls": 0,
        "candidate_attempts": 0,
        "strict_install_invocations": 0,
        "rejection_counts_by_reason": {},
        "unknown_count": 0,
        "metadata_files_read": 0,
        "metadata_bytes_read": 0,
        "archive_bytes": 0,
        "archive_build_time_seconds": 0.0,
        "extraction_time_seconds": 0.0,
        "package_manager_time_seconds": 0.0,
        "lockfile_read_time_seconds": 0.0,
        "evidence_model_guard_hit": False,
    }


def _certification_statistics(
    prepared: _PreparedPortfolioProblem,
    metrics: Mapping[str, Any],
    started: float,
) -> CertificationStatistics:
    """Project invocation-local counters into the bounded public certificate."""
    evidence_relations = prepared.subgraph.candidate_relations
    evidence_model_guard_hit = bool(metrics.get("evidence_model_guard_hit"))
    modeled = sum(
        relation.is_modelled
        and not (evidence_model_guard_hit and relation.evidence_variable_id is not None)
        for relation in evidence_relations
    )
    unmodeled = len(evidence_relations) - modeled
    return CertificationStatistics(
        solver_calls=int(metrics.get("solver_calls", 0)),
        candidate_attempts=int(metrics.get("candidate_attempts", 0)),
        strict_install_invocations=int(metrics.get("strict_install_invocations", 0)),
        rejection_counts_by_reason=dict(metrics.get("rejection_counts_by_reason", {})),
        unknown_count=int(metrics.get("unknown_count", 0)),
        evidence_relations_modeled=modeled,
        evidence_relations_unmodeled=unmodeled,
        metadata_files_read=int(metrics.get("metadata_files_read", 0)),
        metadata_bytes_read=int(metrics.get("metadata_bytes_read", 0)),
        archive_bytes=int(metrics.get("archive_bytes", 0)),
        archive_build_time_seconds=float(metrics.get("archive_build_time_seconds", 0.0)),
        extraction_time_seconds=float(metrics.get("extraction_time_seconds", 0.0)),
        package_manager_time_seconds=float(metrics.get("package_manager_time_seconds", 0.0)),
        lockfile_read_time_seconds=float(metrics.get("lockfile_read_time_seconds", 0.0)),
        total_certification_time_seconds=min(86_400.0, max(0.0, time.monotonic() - started)),
    )


def _candidate_conflict(
    assignment: Mapping[str, str],
    reason_code: SolverCandidateRejectionReason,
    *,
    cut_kind: SolverCandidateCutKind = SolverCandidateCutKind.EXACT_ASSIGNMENT,
    literal_ids: Sequence[str] | None = None,
    evidence: Mapping[str, Any],
    summary: str,
) -> SolverCandidateConflict:
    """Bind normalized, structured rejection evidence to a complete assignment."""
    normalized_assignment = {
        str(variable_id): str(version).strip().lstrip("vV")
        for variable_id, version in sorted(assignment.items())
    }
    selected_ids = sorted(literal_ids if literal_ids is not None else normalized_assignment)
    literals = [
        SolverCandidateLiteral(variable_id=variable_id, version=normalized_assignment[variable_id])
        for variable_id in selected_ids
    ]
    evidence_payload = {
        "reason_code": reason_code.value,
        "cut_kind": cut_kind.value,
        "literals": [literal.model_dump(mode="json") for literal in literals],
        "evidence": evidence,
    }
    return SolverCandidateConflict(
        assignment_digest=_digest(normalized_assignment),
        reason_code=reason_code,
        cut_kind=cut_kind,
        literals=literals,
        evidence_digest=_digest(evidence_payload),
        summary=str(summary).strip()[:500] or "candidate assignment rejected",
    )


def _task_backed_literal_ids(
    prepared: _PreparedPortfolioProblem,
    assignment: Mapping[str, str],
    occurrence_ids: Sequence[str],
) -> tuple[str, ...] | None:
    """Return exact task-backed mutation IDs, or refuse generalized learning."""
    requested = tuple(occurrence_ids)
    if not requested or len(set(requested)) != len(requested):
        return None
    eligible_ids = {
        target.occurrence_id for target in prepared.targets if target.eligible_for_atomic_update
    }
    if any(item not in eligible_ids or item not in assignment for item in requested):
        return None
    return tuple(sorted(requested))


def _reject_assignment(
    assignment: Mapping[str, str],
    reason_code: SolverCandidateRejectionReason,
    *,
    evidence: Mapping[str, Any],
    summary: str,
    cut_kind: SolverCandidateCutKind = SolverCandidateCutKind.EXACT_ASSIGNMENT,
    literal_ids: Sequence[str] | None = None,
) -> _AssignmentRejected:
    """Create a typed rejection, defaulting to a complete-assignment no-good."""
    return _AssignmentRejected(
        _candidate_conflict(
            assignment,
            reason_code,
            cut_kind=cut_kind,
            literal_ids=literal_ids,
            evidence=evidence,
            summary=summary,
        )
    )


def _run_command(
    sandbox: DockerSandbox,
    command: str,
    deadline: float,
    stage: str,
    *,
    allow_nonzero: bool = False,
):
    """Run one checked sandbox command within the shared deadline."""
    remaining = _remaining_seconds(deadline)
    if remaining < 1.0:
        raise _CertificationUnknown(f"certification deadline expired before {stage}")
    timeout = max(1, math.floor(remaining))
    try:
        result = sandbox.run(command, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        raise _CertificationUnknown(f"{stage} command failed: {exc}") from exc
    if time.monotonic() >= deadline:
        raise _CertificationUnknown(f"certification deadline expired during {stage}")
    if result.exit_code == 124:
        raise _CertificationUnknown(f"{stage} command timed out")
    if not allow_nonzero and result.exit_code != 0:
        raise _CertificationUnknown(
            f"{stage} command exited {result.exit_code}: "
            f"{result.stderr or result.stdout or 'no diagnostic output'}"
        )
    return result


def _build_workspace_archive(
    sandbox: DockerSandbox,
    archive_path: str,
    deadline: float,
) -> tuple[int, float]:
    """Create and validate one immutable archive of the verified workspace."""
    started = time.monotonic()
    tar_command = shlex.join(
        [
            "tar",
            "-cf",
            archive_path,
            "--exclude=node_modules",
            "--exclude=*/node_modules",
            "--exclude=.git",
            "--exclude=*/.git",
            "--exclude=.remedy-attempt-snapshots",
            "--exclude=*/.remedy-attempt-snapshots",
            "--exclude=.remedy-plan-cert",
            "--exclude=*/.remedy-plan-cert",
            "-C",
            "/workspace",
            ".",
        ]
    )
    _run_command(sandbox, tar_command, deadline, "workspace archive creation")
    _run_command(
        sandbox,
        f"test -s {shlex.quote(archive_path)} && tar -tf {shlex.quote(archive_path)} >/dev/null",
        deadline,
        "workspace archive validation",
    )
    size_result = _run_command(
        sandbox,
        f"wc -c < {shlex.quote(archive_path)}",
        deadline,
        "workspace archive sizing",
    )
    try:
        archive_bytes = int(size_result.stdout.strip())
    except ValueError as exc:
        raise _CertificationUnknown("workspace archive size output is invalid") from exc
    return archive_bytes, max(0.0, time.monotonic() - started)


@contextmanager
def _certification_artifact_scope(
    sandbox: DockerSandbox,
    workspace_volume: str,
    sandbox_factory: Callable[..., DockerSandbox],
    scratch_root: str,
    archive_path: str,
    deadline: float,
) -> Iterator[None]:
    """Remove one run's archive and scratch root on every exit path."""
    try:
        yield
    finally:
        scratch_path = _validate_workspace_path(scratch_root)
        cleanup_command = (
            f"rm -rf -- /workspace/{shlex.quote(scratch_path)} {shlex.quote(archive_path)}"
        )
        cleanup_failed = False
        try:
            cleanup_result = sandbox.run(
                cleanup_command, timeout=max(1, math.floor(_remaining_seconds(deadline)))
            )
            cleanup_failed = cleanup_result.exit_code != 0
        except Exception:  # noqa: BLE001 - cleanup status is the only public evidence
            cleanup_failed = True
        if cleanup_failed:
            try:
                with sandbox_factory(
                    repo_root=None, workspace_volume=workspace_volume
                ) as cleanup_sandbox:
                    cleanup_result = cleanup_sandbox.run(
                        cleanup_command,
                        timeout=max(1, math.floor(_remaining_seconds(deadline))),
                    )
                    if cleanup_result.exit_code != 0:
                        cleanup_failed = True
            except Exception:  # noqa: BLE001
                cleanup_failed = True
            raise _CertificationUnknown("shared certification archive or scratch cleanup failed")


def _unknown_runtime_fingerprint() -> SolverRuntimeFingerprint:
    """Return an explicit missing-runtime marker for UNKNOWN certificates."""
    return SolverRuntimeFingerprint(
        node_version="unknown",
        npm_version="unknown",
        platform="unknown",
        architecture="unknown",
    )


def _read_runtime_fingerprint(
    sandbox: DockerSandbox,
    deadline: float,
) -> SolverRuntimeFingerprint:
    """Capture Node/npm/platform identity from the certification container."""
    command = (
        "node --version && npm --version && "
        "node -p 'JSON.stringify({platform:process.platform,architecture:process.arch})'"
    )
    result = _run_command(sandbox, command, deadline, "runtime fingerprint")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 3:
        raise _CertificationUnknown("runtime fingerprint output is incomplete")
    try:
        runtime = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise _CertificationUnknown("runtime fingerprint platform output is invalid JSON") from exc
    if not isinstance(runtime, Mapping):
        raise _CertificationUnknown("runtime fingerprint platform output is not an object")
    try:
        return SolverRuntimeFingerprint(
            node_version=lines[0],
            npm_version=lines[1],
            platform=runtime.get("platform"),
            architecture=runtime.get("architecture"),
        )
    except Exception as exc:  # noqa: BLE001
        raise _CertificationUnknown(f"runtime fingerprint is incomplete: {exc}") from exc


def _read_npm_documents(
    sandbox: DockerSandbox,
    root_prefix: str,
    deadline: float,
    metrics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Batch-read discovered npm metadata from one workspace subtree."""
    normalized_prefix = _validate_workspace_path(root_prefix) if root_prefix else ""
    root_absolute = "/workspace" + (f"/{normalized_prefix}" if normalized_prefix else "")
    quoted_root = shlex.quote(root_absolute)
    compiled_prunes = " ".join(
        f"-path {shlex.quote(f'{root_absolute}/{directory}')} -prune -o"
        for directory in ("build", "dist")
    )
    command = (
        f"find {quoted_root} "
        r"-type d \( -name node_modules -o -name .git "
        r"-o -name .remedy-attempt-snapshots -o -name .remedy-plan-cert \) -prune -o "
        f"{compiled_prunes} "
        r"-type f \( -name package.json -o -name package-lock.json "
        r"-o -name npm-shrinkwrap.json \) -print | sort"
    )
    result = _run_command(sandbox, command, deadline, "npm metadata discovery")
    prefix = root_absolute.rstrip("/") + "/"
    relative_paths_by_workspace_path: dict[str, set[str]] = {}
    for raw_path in sorted({line.strip() for line in result.stdout.splitlines() if line.strip()}):
        path = raw_path.replace("\\", "/")
        if path.startswith(prefix):
            relative_to_root = path[len(prefix) :]
        elif path.startswith("./"):
            relative_to_root = path[2:]
        elif not path.startswith("/"):
            relative_to_root = path
        else:
            raise _CertificationUnknown("metadata discovery escaped workspace")
        relative_to_root = _validate_workspace_path(relative_to_root)
        workspace_path = (
            _validate_workspace_path(f"{normalized_prefix}/{relative_to_root}")
            if root_prefix
            else relative_to_root
        )
        relative_paths_by_workspace_path.setdefault(workspace_path, set()).add(relative_to_root)

    if _remaining_seconds(deadline) < 0.001:
        raise _CertificationUnknown("certification deadline expired before metadata batch read")
    read_started = time.monotonic()
    try:
        contents = sandbox.read_files(sorted(relative_paths_by_workspace_path))
    except Exception as exc:  # noqa: BLE001 - batch read errors fail certification closed
        raise _CertificationUnknown("npm metadata batch read failed") from exc
    if not isinstance(contents, Mapping):
        raise _CertificationUnknown("npm metadata batch read returned no complete document set")
    if _remaining_seconds(deadline) < 0.001:
        raise _CertificationUnknown("certification deadline expired during metadata batch read")

    documents: dict[str, str] = {}
    for workspace_path in sorted(relative_paths_by_workspace_path):
        content = contents.get(workspace_path)
        if not isinstance(content, str):
            raise _CertificationUnknown("npm metadata batch read omitted a discovered file")
        for relative_to_root in sorted(relative_paths_by_workspace_path[workspace_path]):
            previous = documents.get(relative_to_root)
            if previous is not None and previous != content:
                raise _CertificationUnknown(
                    "conflicting workspace metadata paths normalize to the same document"
                )
            documents[relative_to_root] = content
    if metrics is not None:
        reported_bytes = getattr(contents, "total_bytes", None)
        if not isinstance(reported_bytes, int):
            reported_bytes = sum(len(content.encode("utf-8")) for content in documents.values())
        metrics["metadata_files_read"] += len(relative_paths_by_workspace_path)
        metrics["metadata_bytes_read"] += reported_bytes
        metrics["lockfile_read_time_seconds"] += max(0.0, time.monotonic() - read_started)
    return documents


def _manifest_fingerprints(snapshot: NpmGraphSnapshot) -> dict[str, str]:
    """Return canonical package-manifest inputs keyed by repository path."""
    return {
        path: content
        for path, content in snapshot.fingerprint_inputs
        if PurePosixPath(path).name == "package.json"
    }


def _lockfile_pair_conflict(snapshot: NpmGraphSnapshot) -> str | None:
    """Reject ambiguous npm lockfile precedence when contents differ."""
    by_manifest: dict[str, dict[str, str]] = {}
    for lockfile in snapshot.lockfiles:
        name = PurePosixPath(lockfile.path).name
        if name not in {"package-lock.json", "npm-shrinkwrap.json"}:
            continue
        parent = PurePosixPath(lockfile.path).parent.as_posix()
        manifest_path = f"{parent}/package.json" if parent != "." else "package.json"
        by_manifest.setdefault(manifest_path, {})[name] = lockfile.raw_digest
    for manifest_path, digests in sorted(by_manifest.items()):
        package_lock = digests.get("package-lock.json")
        shrinkwrap = digests.get("npm-shrinkwrap.json")
        if package_lock and shrinkwrap and package_lock != shrinkwrap:
            return (
                f"manifest {manifest_path!r} has conflicting package-lock.json and "
                "npm-shrinkwrap.json content digests"
            )
    return None


def _last_qa_passed_prefix_provenance(
    prior_plan: PortfolioPlan | None,
    task_queue: Mapping[str, RemediationTask],
    request: PortfolioReplanRequest | None,
    host_repository_fingerprint: str,
    *,
    qa_results_by_attempt: Mapping[str, QAAttemptResult] | None = None,
    qa_passed_workspace_prefix: QAPassedWorkspacePrefix | None = None,
    diagnostics: list[str] | None = None,
) -> QAPassedWorkspacePrefix | None:
    """Return the verified workspace prefix represented by QA-passed batches.

    Args:
        prior_plan: Previously committed and certified portfolio plan.
        task_queue: Current Supervisor-owned task projection.
        request: Typed request that identifies the source plan.
        host_repository_fingerprint: Fingerprint of current host manifests.
        qa_passed_workspace_prefix: Latest cumulative workspace checkpoint recorded by QA.
        diagnostics: Optional list receiving why a trusted prefix was unavailable.

    Returns:
        Provenance for the last exact QA-passed prefix, or ``None`` when no
        prefix can be independently verified.
    """

    def reject(reason: str) -> None:
        if diagnostics is not None:
            diagnostics.append(f"QA-passed workspace prefix not trusted: {reason}")

    if request is None:
        return None
    if prior_plan is None:
        reject("the source portfolio plan is unavailable")
        return None
    if not host_repository_fingerprint:
        reject("the current host repository fingerprint is unavailable")
        return None
    if isinstance(qa_passed_workspace_prefix, Mapping):
        try:
            qa_passed_workspace_prefix = QAPassedWorkspacePrefix.model_validate(
                qa_passed_workspace_prefix
            )
        except Exception as exc:  # noqa: BLE001 - invalid persisted state is untrusted evidence
            reject(f"the latest QA workspace checkpoint is malformed: {exc}")
            qa_passed_workspace_prefix = None
    prior_plan_id = prior_plan.portfolio_plan_id
    if request.source_portfolio_plan_id != prior_plan_id:
        reject(
            "request source plan ID "
            f"{request.source_portfolio_plan_id!r} does not match committed plan {prior_plan_id!r}"
        )
        return None
    certificate = prior_plan.resolution_certificate
    solver_plan = prior_plan.solver_plan
    if certificate is None:
        reject("the source plan has no package-resolution certificate")
        return None
    if certificate.status != PackageResolutionStatus.CERTIFIED:
        reject(f"the source certificate status is {certificate.status!s}, not CERTIFIED")
        return None
    if certificate.portfolio_plan_id != prior_plan_id:
        reject("the source certificate is bound to a different portfolio plan ID")
        return None
    if certificate.solver_input_digest != prior_plan.solver_input_digest:
        reject("the source certificate solver-input digest is stale")
        return None
    if certificate.repository_fingerprint != prior_plan.repository_fingerprint:
        reject("the source certificate repository fingerprint differs from its plan")
        return None
    if certificate.repository_fingerprint != host_repository_fingerprint:
        reject("the source plan was certified against a different host repository")
        return None
    if certificate.workspace_graph_digest != prior_plan.workspace_graph_digest:
        reject("the source certificate workspace graph digest differs from its plan")
        return None
    if certificate.task_revisions != prior_plan.task_revisions:
        reject("the source certificate task revisions differ from its plan")
        return None
    if solver_plan is None:
        reject("the source plan has no solver plan")
        return None
    if solver_plan.status not in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}:
        reject(f"the source solver status is {solver_plan.status!s}, not accepted")
        return None
    if not solver_plan.candidate_catalog_complete:
        reject("the source solver candidate catalog was incomplete")
        return None
    if certificate.candidate_catalog_digest != solver_plan.candidate_catalog_digest:
        reject("the source certificate candidate catalog digest is stale")
        return None
    selected = solver_plan.selected_plan
    if selected is None:
        reject("the source solver plan has no selected assignment")
        return None
    expected_assignment_digest = _digest(dict(sorted(selected.selected_candidate_versions.items())))
    if certificate.candidate_plan_id != selected.candidate_plan_id:
        reject("the source certificate candidate plan ID differs from its selected plan")
        return None
    if certificate.candidate_assignment_digest != expected_assignment_digest:
        reject("the source certificate candidate assignment digest is stale")
        return None

    if qa_passed_workspace_prefix is not None:
        if qa_passed_workspace_prefix.certified_by_portfolio_plan_id != prior_plan_id:
            reject("the latest QA workspace checkpoint is bound to a different portfolio plan")
            return None
        evidence_errors = _qa_prefix_evidence_errors(
            qa_passed_workspace_prefix,
            task_queue,
            qa_results_by_attempt or {},
        )
        if evidence_errors:
            reject(
                "the latest QA workspace checkpoint does not match its QA evidence: "
                f"{evidence_errors}"
            )
            return None
        # Prefer QA's latest full-batch checkpoint over the older prefix copied
        # into the source plan's certificate.
        return qa_passed_workspace_prefix

    last_prefix_digest: str | None = None
    last_prefix_attempt_ids_by_task: dict[str, str] = {}
    last_prefix_task_revisions: dict[str, int] = {}
    last_prefix_snapshot_attempt_id: str | None = None
    last_prefix_batch_ids: list[str] = []
    last_prefix_task_ids: list[str] = []
    inherited_prefix = getattr(certificate, "workspace_prefix_provenance", None)
    if inherited_prefix is not None:
        if inherited_prefix.certified_by_portfolio_plan_id != prior_plan_id:
            reject("the inherited workspace prefix is bound to a different portfolio plan")
            return None
        if inherited_prefix.qa_attempt_ids_by_task:
            evidence_errors = _qa_prefix_evidence_errors(
                inherited_prefix,
                task_queue,
                qa_results_by_attempt or {},
            )
        else:
            invalid_task_ids = [
                task_id
                for task_id in inherited_prefix.task_ids
                if (task := task_queue.get(task_id)) is None
                or getattr(task.status, "value", task.status) != TaskStatus.QA_PASSED.value
                or (
                    task_id in inherited_prefix.task_revisions
                    and task.task_revision != inherited_prefix.task_revisions[task_id]
                )
            ]
            evidence_errors = (
                ["legacy workspace prefix no longer matches the current task revisions"]
                if invalid_task_ids
                else []
            )
            if inherited_prefix.snapshot_id:
                evidence_errors.append("retained snapshot has no immutable QA attempt provenance")
        if evidence_errors:
            reject(
                f"the inherited workspace prefix does not match its QA evidence: {evidence_errors}"
            )
            return None
        last_prefix_digest = inherited_prefix.graph_digest
        last_prefix_batch_ids = list(inherited_prefix.batch_ids)
        last_prefix_task_ids = list(inherited_prefix.task_ids)
    last_prefix_snapshot_id = inherited_prefix.snapshot_id if inherited_prefix is not None else None
    if inherited_prefix is not None:
        last_prefix_attempt_ids_by_task = dict(inherited_prefix.qa_attempt_ids_by_task)
        last_prefix_task_revisions = dict(inherited_prefix.task_revisions)
        last_prefix_snapshot_attempt_id = inherited_prefix.snapshot_attempt_id

    batches = list(selected.batches)
    batches_by_id = {batch.batch_id: batch for batch in batches}
    if len(batches_by_id) != len(batches):
        reject("the source solver plan contains duplicate batch IDs")
        return None
    ordered_batch_ids: list[str] = []
    seen_batch_ids: set[str] = set()
    for phase in sorted(selected.phases, key=lambda item: item.phase_number):
        for batch_id in phase.batch_ids:
            if batch_id not in batches_by_id or batch_id in seen_batch_ids:
                reject(f"phase order contains an unknown or duplicate batch {batch_id!r}")
                return None
            ordered_batch_ids.append(batch_id)
            seen_batch_ids.add(batch_id)
    ordered_batch_ids.extend(sorted(set(batches_by_id) - seen_batch_ids))

    def qa_workspace_checkpoint_for_batch(
        batch: SolverBatch,
    ) -> tuple[str, str | None, dict[str, str], dict[str, int], str | None] | None:
        """Return one source-plan-bound successful QA checkpoint for a batch."""
        if not qa_results_by_attempt:
            return None
        digests_by_task: dict[str, set[str]] = {task_id: set() for task_id in batch.task_ids}
        snapshots_by_task: dict[str, set[str]] = {task_id: set() for task_id in batch.task_ids}
        attempts_by_task: dict[str, set[str]] = {task_id: set() for task_id in batch.task_ids}
        revisions_by_task: dict[str, set[int]] = {task_id: set() for task_id in batch.task_ids}
        for attempt_id, qa_result in qa_results_by_attempt.items():
            result_attempt_id = (
                qa_result.get("attempt_id")
                if isinstance(qa_result, Mapping)
                else qa_result.attempt_id
            )
            task_id = (
                qa_result.get("task_id") if isinstance(qa_result, Mapping) else qa_result.task_id
            )
            if result_attempt_id != attempt_id or task_id not in digests_by_task:
                continue
            result_plan_id = (
                qa_result.get("portfolio_plan_id")
                if isinstance(qa_result, Mapping)
                else qa_result.portfolio_plan_id
            )
            result_revision = (
                qa_result.get("task_revision")
                if isinstance(qa_result, Mapping)
                else qa_result.task_revision
            )
            if result_plan_id != prior_plan_id or result_revision != certificate.task_revisions.get(
                task_id
            ):
                continue
            evaluation = (
                qa_result.get("evaluation")
                if isinstance(qa_result, Mapping)
                else qa_result.evaluation
            )
            passed = (
                evaluation.get("passed")
                if isinstance(evaluation, Mapping)
                else getattr(evaluation, "passed", False)
            )
            digest = (
                qa_result.get("workspace_graph_digest")
                if isinstance(qa_result, Mapping)
                else qa_result.workspace_graph_digest
            )
            if passed and isinstance(digest, str) and digest.strip():
                digests_by_task[task_id].add(digest.strip())
                attempts_by_task[task_id].add(attempt_id)
                if isinstance(result_revision, int):
                    revisions_by_task[task_id].add(result_revision)
                snapshot_id = (
                    qa_result.get("workspace_snapshot_id")
                    if isinstance(qa_result, Mapping)
                    else qa_result.workspace_snapshot_id
                )
                if isinstance(snapshot_id, str) and snapshot_id.strip():
                    snapshots_by_task[task_id].add(snapshot_id.strip())
        if any(
            len(task_digests) != 1
            or len(attempts_by_task[task_id]) != 1
            or len(revisions_by_task[task_id]) != 1
            for task_id, task_digests in digests_by_task.items()
        ):
            return None
        batch_digests = {next(iter(task_digests)) for task_digests in digests_by_task.values()}
        if len(batch_digests) != 1:
            return None
        batch_snapshot_ids = {
            next(iter(task_snapshot_ids))
            for task_snapshot_ids in snapshots_by_task.values()
            if len(task_snapshot_ids) == 1
        }
        snapshot_id = (
            next(iter(batch_snapshot_ids))
            if len(batch_snapshot_ids) == 1
            and all(len(task_snapshot_ids) == 1 for task_snapshot_ids in snapshots_by_task.values())
            else None
        )
        task_attempt_ids = {
            task_id: next(iter(task_attempts))
            for task_id, task_attempts in attempts_by_task.items()
        }
        task_revisions = {
            task_id: next(iter(revisions)) for task_id, revisions in revisions_by_task.items()
        }
        snapshot_attempt_id = (
            task_attempt_ids[sorted(task_attempt_ids)[0]] if snapshot_id is not None else None
        )
        return (
            next(iter(batch_digests)),
            snapshot_id,
            task_attempt_ids,
            task_revisions,
            snapshot_attempt_id,
        )

    diagnostic_start = len(diagnostics) if diagnostics is not None else 0
    for batch_id in ordered_batch_ids:
        batch = batches_by_id[batch_id]
        if not batch.task_ids or len(set(batch.task_ids)) != len(batch.task_ids):
            reject(f"batch {batch_id!r} has no task IDs or duplicate task IDs")
            return None
        if not batch.dispatchable:
            invalid_terminal_members = [
                task_id
                for task_id in batch.task_ids
                if (task := task_queue.get(task_id)) is None
                or task.status not in TERMINAL_TASK_STATUSES
            ]
            if batch.mutations or invalid_terminal_members:
                reject(
                    f"non-dispatchable batch {batch_id!r} is not a mutation-free "
                    f"terminal-only batch (invalid members: {invalid_terminal_members})"
                )
                return None
            continue

        member_tasks = [task_queue.get(task_id) for task_id in batch.task_ids]
        if any(task is None for task in member_tasks):
            reject(f"batch {batch_id!r} references a task missing from the current queue")
            return None
        member_group_ids = {task.parent_group_id for task in member_tasks if task is not None}
        if not member_group_ids:
            reject(f"batch {batch_id!r} has no member groups")
            return None
        non_qa_groups = sorted(
            group_id
            for group_id in member_group_ids
            if effective_group_status(task_queue, group_id) != TaskStatus.QA_PASSED.value
        )
        if non_qa_groups:
            if diagnostics is not None:
                if last_prefix_batch_ids:
                    diagnostics.append(
                        f"QA-passed workspace prefix ends at batch "
                        f"{last_prefix_batch_ids[-1]!r}; batch {batch_id!r} has "
                        f"non-passed groups {non_qa_groups}"
                    )
                else:
                    diagnostics.append(
                        f"QA-passed workspace prefix stops before batch {batch_id!r}: "
                        f"groups {non_qa_groups} are not QA_PASSED"
                    )
            break
        qa_workspace_checkpoint = qa_workspace_checkpoint_for_batch(batch)
        qa_workspace_digest = (
            qa_workspace_checkpoint[0] if qa_workspace_checkpoint is not None else None
        )
        resolver_prefix_digest = certificate.batch_prefix_graph_digests.get(batch_id)
        prefix_digest = qa_workspace_digest or resolver_prefix_digest
        if (
            qa_workspace_digest is not None
            and resolver_prefix_digest is not None
            and qa_workspace_digest != resolver_prefix_digest
            and diagnostics is not None
        ):
            diagnostics.append(
                f"QA-passed workspace fingerprint for batch {batch_id!r} differs from "
                "the resolver-predicted prefix; using the attempt-bound QA fingerprint"
            )
        if not prefix_digest:
            if diagnostics is not None:
                diagnostics.append(
                    f"QA-passed workspace prefix stops at batch {batch_id!r}: "
                    "the certified prefix graph digest is missing"
                )
            break
        last_prefix_digest = prefix_digest
        last_prefix_snapshot_id = (
            qa_workspace_checkpoint[1]
            if qa_workspace_checkpoint is not None and qa_workspace_digest is not None
            else None
        )
        if qa_workspace_checkpoint is not None:
            last_prefix_attempt_ids_by_task.update(qa_workspace_checkpoint[2])
            last_prefix_task_revisions.update(qa_workspace_checkpoint[3])
            last_prefix_snapshot_attempt_id = qa_workspace_checkpoint[4]
        else:
            last_prefix_snapshot_attempt_id = None
        last_prefix_batch_ids = list(dict.fromkeys([*last_prefix_batch_ids, batch_id]))
        last_prefix_task_ids = list(dict.fromkeys([*last_prefix_task_ids, *batch.task_ids]))

    if (
        last_prefix_digest is None
        and diagnostics is not None
        and len(diagnostics) == diagnostic_start
    ):
        reject("the source plan contains no executable QA-passed batch prefix")
    if last_prefix_digest is None:
        return None
    return QAPassedWorkspacePrefix(
        graph_digest=last_prefix_digest,
        certified_by_portfolio_plan_id=prior_plan_id,
        batch_ids=last_prefix_batch_ids,
        task_ids=last_prefix_task_ids,
        task_revisions=(last_prefix_task_revisions if last_prefix_snapshot_id is not None else {}),
        qa_attempt_ids_by_task=(
            last_prefix_attempt_ids_by_task if last_prefix_snapshot_id is not None else {}
        ),
        snapshot_id=last_prefix_snapshot_id,
        snapshot_attempt_id=(
            last_prefix_snapshot_attempt_id if last_prefix_snapshot_id is not None else None
        ),
    )


def _last_qa_passed_prefix_digest(
    prior_plan: PortfolioPlan | None,
    task_queue: Mapping[str, RemediationTask],
    request: PortfolioReplanRequest | None,
    host_repository_fingerprint: str,
    *,
    qa_passed_workspace_prefix: QAPassedWorkspacePrefix | None = None,
    diagnostics: list[str] | None = None,
) -> str | None:
    """Return only the graph digest from verified QA-passed prefix provenance."""
    prefix = _last_qa_passed_prefix_provenance(
        prior_plan,
        task_queue,
        request,
        host_repository_fingerprint,
        qa_passed_workspace_prefix=qa_passed_workspace_prefix,
        diagnostics=diagnostics,
    )
    return prefix.graph_digest if prefix is not None else None


def _assert_workspace_matches_host(
    host_snapshot: NpmGraphSnapshot,
    workspace_snapshot: NpmGraphSnapshot,
    workspace_documents: Mapping[str, str],
    qa_passed_prefix_digest: str | None = None,
) -> None:
    """Require host manifests or one independently verified QA-passed prefix."""
    if host_snapshot.diagnostics:
        raise _CertificationUnknown(
            "host npm graph is malformed or unsupported: " + " | ".join(host_snapshot.diagnostics)
        )
    if workspace_snapshot.diagnostics:
        raise _CertificationUnknown(
            "workspace npm graph is malformed or unsupported: "
            + " | ".join(workspace_snapshot.diagnostics)
        )
    actual_manifest_paths = {
        path for path in workspace_documents if PurePosixPath(path).name == "package.json"
    }
    host_manifest_paths = {manifest.path for manifest in host_snapshot.manifests}
    workspace_manifest_paths = {manifest.path for manifest in workspace_snapshot.manifests}
    if (
        actual_manifest_paths != host_manifest_paths
        or workspace_manifest_paths != host_manifest_paths
    ):
        raise _CertificationUnknown(
            "workspace package manifest paths differ from the host repository snapshot"
        )
    workspace_fingerprint = workspace_snapshot.repository_fingerprint
    if _manifest_fingerprints(workspace_snapshot) != _manifest_fingerprints(host_snapshot) and (
        qa_passed_prefix_digest is None or workspace_fingerprint != qa_passed_prefix_digest
    ):
        if qa_passed_prefix_digest is None:
            detail = "no certified QA-passed workspace prefix digest was available"
        else:
            detail = "the live fingerprint did not match the certified QA-passed prefix"
        raise _CertificationUnknown(
            "workspace package manifest contents differ from the host repository snapshot; "
            f"{detail}"
        )


def _candidate_for_assignment(
    prepared: _PreparedPortfolioProblem,
    occurrence_id: str,
    assignment: Mapping[str, str],
) -> SolverVersionCandidate:
    """Return the exact catalog candidate selected for one mutation target."""
    version = assignment.get(occurrence_id)
    if not version:
        raise _CertificationUnknown(f"selected candidate assignment is missing {occurrence_id!r}")
    for candidate in prepared.candidate_domains.get(occurrence_id, ()):
        if candidate.version == version:
            return candidate
    raise _CertificationUnknown(
        f"selected candidate {version!r} is absent from domain {occurrence_id!r}"
    )


def _platform_matches(allowed_values: Sequence[str], actual: str) -> bool:
    """Apply npm's positive and negated platform allowlist semantics."""
    if not allowed_values:
        return True
    excluded = {value[1:] for value in allowed_values if value.startswith("!")}
    included = {value for value in allowed_values if not value.startswith("!")}
    if actual in excluded:
        return False
    return not included or actual in included


def _validate_candidate_runtime(
    prepared: _PreparedPortfolioProblem,
    assignment: Mapping[str, str],
    mutated_occurrence_ids: set[str],
    runtime: SolverRuntimeFingerprint,
) -> None:
    """Validate selected release engine and platform metadata under Node 22."""
    for occurrence_id in sorted(mutated_occurrence_ids):
        candidate = _candidate_for_assignment(prepared, occurrence_id, assignment)
        for engine_name, version_range in candidate.engines.items():
            if engine_name == "node":
                actual = runtime.node_version
            elif engine_name == "npm":
                actual = runtime.npm_version
            else:
                raise _CertificationUnknown(
                    f"runtime fingerprint cannot validate engine {engine_name!r}"
                )
            if actual == "unknown":
                raise _CertificationUnknown(f"runtime value for engine {engine_name!r} is missing")
            checked = check_npm_range(version_range, actual)
            if checked.matches is None:
                raise _CertificationUnknown(
                    f"invalid engine range {version_range!r} for {candidate.version}"
                )
            if not checked.matches:
                literal_ids = _task_backed_literal_ids(prepared, assignment, (occurrence_id,))
                raise _reject_assignment(
                    assignment,
                    SolverCandidateRejectionReason.RUNTIME_ENGINE,
                    cut_kind=(
                        SolverCandidateCutKind.UNARY
                        if literal_ids is not None
                        else SolverCandidateCutKind.EXACT_ASSIGNMENT
                    ),
                    literal_ids=literal_ids,
                    evidence={
                        "candidate_version": candidate.version,
                        "engine": engine_name,
                        "required_range": version_range,
                        "runtime_version": actual,
                    },
                    summary=(
                        f"candidate {candidate.version} requires {engine_name} "
                        f"{version_range}; certification runtime is {actual}"
                    ),
                )
        if not _platform_matches(candidate.os, runtime.platform):
            literal_ids = _task_backed_literal_ids(prepared, assignment, (occurrence_id,))
            raise _reject_assignment(
                assignment,
                SolverCandidateRejectionReason.RUNTIME_PLATFORM,
                cut_kind=(
                    SolverCandidateCutKind.UNARY
                    if literal_ids is not None
                    else SolverCandidateCutKind.EXACT_ASSIGNMENT
                ),
                literal_ids=literal_ids,
                evidence={
                    "candidate_version": candidate.version,
                    "constraint": "os",
                    "allowed": candidate.os,
                    "actual": runtime.platform,
                },
                summary=f"candidate {candidate.version} does not support platform {runtime.platform}",
            )
        if not _platform_matches(candidate.cpu, runtime.architecture):
            literal_ids = _task_backed_literal_ids(prepared, assignment, (occurrence_id,))
            raise _reject_assignment(
                assignment,
                SolverCandidateRejectionReason.RUNTIME_PLATFORM,
                cut_kind=(
                    SolverCandidateCutKind.UNARY
                    if literal_ids is not None
                    else SolverCandidateCutKind.EXACT_ASSIGNMENT
                ),
                literal_ids=literal_ids,
                evidence={
                    "candidate_version": candidate.version,
                    "constraint": "cpu",
                    "allowed": candidate.cpu,
                    "actual": runtime.architecture,
                },
                summary=(
                    f"candidate {candidate.version} does not support "
                    f"architecture {runtime.architecture}"
                ),
            )


def _scratch_workspace_path(scratch_prefix: str, manifest_path: str) -> str:
    """Return one validated manifest path inside the assignment scratch root."""
    return _validate_workspace_path(f"{scratch_prefix}/{manifest_path}")


def _pruning_baseline_checkpoint(
    scratch_prefix: str,
    manifest_paths: Iterable[str],
    archived_documents: Mapping[str, str],
) -> _PackageCheckpoint:
    """Build immutable same-scope manifest and lockfile evidence from the archive."""
    files: dict[str, str | None] = {}
    for relative_manifest in sorted({_validate_workspace_path(path) for path in manifest_paths}):
        scratch_manifest = _scratch_workspace_path(scratch_prefix, relative_manifest)
        files[scratch_manifest] = archived_documents.get(relative_manifest)
        relative_parent = PurePosixPath(relative_manifest).parent
        scratch_parent = PurePosixPath(scratch_manifest).parent
        for name in ("package-lock.json", "npm-shrinkwrap.json"):
            relative_lockfile = _validate_workspace_path((relative_parent / name).as_posix())
            scratch_lockfile = _validate_workspace_path((scratch_parent / name).as_posix())
            files[scratch_lockfile] = archived_documents.get(relative_lockfile)
    return _PackageCheckpoint(files=files, touched_files_before=set())


def _stage_batch_mutations(
    sandbox: DockerSandbox,
    scratch_prefix: str,
    batch: SolverBatch,
    deadline: float,
) -> tuple[list[SolverMutation], list[PackageMutation], _PackageCheckpoint | None, set[str]]:
    """Stage one atomic batch with npm pkg set and a complete checkpoint."""
    solver_mutations = list(batch.mutations)
    scratch_mutations = [
        PackageMutation(
            task_id=mutation.task_id,
            package_name=mutation.package_name,
            manifest_path=_scratch_workspace_path(scratch_prefix, mutation.manifest_path),
            target_version=mutation.target_version,
            dependency_type=mutation.dependency_type,
        )
        for mutation in solver_mutations
    ]
    touched_files: set[str] = set()
    if not scratch_mutations:
        return solver_mutations, scratch_mutations, None, touched_files
    manifest_paths = sorted({mutation.manifest_path for mutation in scratch_mutations})
    expected_checkpoint_paths = set(_package_checkpoint_paths(manifest_paths))
    checkpoint = _capture_package_checkpoint(sandbox, manifest_paths, touched_files)
    if set(checkpoint.files) != expected_checkpoint_paths:
        raise _CertificationUnknown("package checkpoint did not cover every staging path")
    try:
        for mutation in scratch_mutations:
            dependency_path = (
                "pnpm.overrides"
                if mutation.dependency_type == "pnpm_overrides"
                else mutation.dependency_type
            )
            package_expr = f"{dependency_path}[{mutation.package_name}]={mutation.target_version}"
            command = shlex.join(["npm", "pkg", "set", package_expr])
            workspace_dir = _workspace_dir_for_manifest(mutation.manifest_path)
            if workspace_dir != "/workspace":
                command = f"cd {shlex.quote(workspace_dir)} && {command}"
            _run_command(sandbox, command, deadline, "manifest staging")
            touched_files.add(mutation.manifest_path)
        return solver_mutations, scratch_mutations, checkpoint, touched_files
    except Exception as exc:
        rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
        if rollback_error:
            raise _CertificationUnknown(rollback_error) from exc
        raise


def _manifest_mutation_value(payload: Mapping[str, Any], mutation: PackageMutation) -> Any:
    """Read the exact direct or override field changed by one mutation."""
    if mutation.dependency_type == "pnpm_overrides":
        pnpm = payload.get("pnpm")
        values = pnpm.get("overrides") if isinstance(pnpm, Mapping) else None
    else:
        values = payload.get(mutation.dependency_type)
    return values.get(mutation.package_name) if isinstance(values, Mapping) else None


def _verify_staged_mutations(
    sandbox: DockerSandbox,
    scratch_mutations: Sequence[PackageMutation],
    checkpoint: _PackageCheckpoint | None,
    *,
    pruning_baseline: _PackageCheckpoint | None = None,
) -> None:
    """Verify staged manifests and lockfiles against the certified candidate."""
    for mutation in scratch_mutations:
        content = sandbox.read_file(mutation.manifest_path)
        if not isinstance(content, str):
            raise _CertificationUnknown(
                f"could not read mutated manifest {mutation.manifest_path!r}"
            )
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise _CertificationUnknown(
                f"mutated manifest is invalid JSON: {mutation.manifest_path!r}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise _CertificationUnknown(
                f"mutated manifest is not an object: {mutation.manifest_path!r}"
            )
        if _manifest_mutation_value(payload, mutation) != mutation.target_version:
            raise _CertificationUnknown(
                f"manifest staging verification failed for {mutation.package_name!r}"
            )
    if checkpoint is None or not scratch_mutations:
        return
    try:
        _verify_lockfile_mutations(
            sandbox,
            checkpoint,
            scratch_mutations,
            pruning_baseline=pruning_baseline,
        )
    except RuntimeError as exc:
        detail = str(exc).strip()
        if len(detail) > 400:
            detail = f"{detail[:397]}..."
        suffix = f": {detail}" if detail else ""
        raise _CertificationUnknown(f"npm lockfile verification failed{suffix}") from exc


def _mutation_lockfile_path(
    checkpoint: _PackageCheckpoint,
    mutation: PackageMutation,
) -> str | None:
    """Return the lockfile checkpoint beside one mutation manifest."""
    parent = PurePosixPath(mutation.manifest_path).parent
    for name in ("npm-shrinkwrap.json", "package-lock.json"):
        path = (parent / name).as_posix()
        if checkpoint.files.get(path) is not None:
            return path
    return None


def _pruned_override_mutations(
    snapshot: NpmGraphSnapshot,
    solver_mutations: Sequence[SolverMutation],
    scratch_mutations: Sequence[PackageMutation],
    pruning_baseline: _PackageCheckpoint | None,
) -> set[str]:
    """Identify override targets whose installed package was proven pruned."""
    if pruning_baseline is None:
        return set()
    present = {
        (package.manifest_path, package.package_name) for package in snapshot.lockfile_packages
    }
    pruned: set[str] = set()
    for solver_mutation, scratch_mutation in zip(solver_mutations, scratch_mutations, strict=False):
        if scratch_mutation.dependency_type not in {"overrides", "resolutions", "pnpm_overrides"}:
            continue
        manifest_path = solver_mutation.manifest_path
        if any(
            lock_manifest == manifest_path and package_name == solver_mutation.package_name
            for lock_manifest, package_name in present
        ):
            continue
        lockfile_path = _mutation_lockfile_path(pruning_baseline, scratch_mutation)
        original_lockfile = pruning_baseline.files.get(lockfile_path) if lockfile_path else None
        if _override_target_was_pruned(
            pruning_baseline,
            scratch_mutation,
            original_lockfile if isinstance(original_lockfile, str) else None,
        ):
            pruned.add(solver_mutation.occurrence_id)
    return pruned


def _validate_candidate_relations(
    prepared: _PreparedPortfolioProblem,
    snapshot: NpmGraphSnapshot,
    assignment: Mapping[str, str],
    mutated_occurrence_ids: set[str],
    pruned_occurrence_ids: set[str],
) -> None:
    """Check required candidate dependency and peer ranges in a resolved lockfile."""
    prepared_occurrences = {
        occurrence.occurrence_id: occurrence for occurrence in prepared.npm_snapshot.occurrences
    }
    for relation in prepared.subgraph.candidate_relations:
        if not relation.is_range_supported:
            continue
        if relation.is_optional or relation.kind == "optional_dependency":
            continue
        if relation.source_occurrence_id not in mutated_occurrence_ids:
            continue
        if assignment.get(relation.source_occurrence_id) != relation.source_candidate_version:
            continue
        expected_source = prepared_occurrences.get(relation.source_occurrence_id)
        if expected_source is None:
            raise _CertificationUnknown(
                f"prepared source occurrence {relation.source_occurrence_id!r} is missing"
            )
        source_matches = [
            occurrence
            for occurrence in snapshot.occurrences
            if occurrence.manifest_path == expected_source.manifest_path
            and occurrence.package_name == expected_source.package_name
        ]
        selected_version = (
            str(assignment.get(relation.source_occurrence_id, "")).strip().lstrip("vV")
        )
        source = next(
            (
                occurrence
                for occurrence in source_matches
                if occurrence.installed_version
                and occurrence.installed_version.strip().lstrip("vV") == selected_version
            ),
            source_matches[0] if len(source_matches) == 1 else None,
        )
        if source is None and relation.source_occurrence_id in pruned_occurrence_ids:
            continue
        if source is None:
            raise _CertificationUnknown(
                "resolved graph omitted prepared source package "
                f"({expected_source.manifest_path!r}, {expected_source.package_name!r})"
            )
        package = resolve_lockfile_dependency_package(
            snapshot, source.occurrence_id, relation.package_name
        )
        if package is None:
            raise _reject_assignment(
                assignment,
                SolverCandidateRejectionReason.REQUIRED_DEPENDENCY_MISSING,
                evidence={
                    "kind": "required_dependency_missing",
                    "source_occurrence_id": relation.source_occurrence_id,
                    "source_version": relation.source_candidate_version,
                    "package_name": relation.package_name,
                    "version_range": relation.version_range,
                    "dependency_kind": relation.kind,
                },
                summary=f"required {relation.kind} {relation.package_name!r} is absent",
            )
        if not package.version:
            raise _CertificationUnknown(
                f"resolved package {relation.package_name!r} has no version metadata"
            )
        checked = check_npm_range(relation.version_range, package.version)
        if checked.matches is None:
            raise _CertificationUnknown(
                f"invalid required range {relation.version_range!r} for {relation.package_name!r}"
            )
        if not checked.matches:
            expected_target = prepared_occurrences.get(relation.target_occurrence_id or "")
            if (
                expected_target is not None
                and expected_target.manifest_path == package.manifest_path
                and expected_target.package_name == package.package_name
            ):
                literal_ids = _task_backed_literal_ids(
                    prepared,
                    assignment,
                    (relation.source_occurrence_id, relation.target_occurrence_id),
                )
            reason_code = (
                SolverCandidateRejectionReason.PEER_CONFLICT
                if relation.kind == "peer"
                else SolverCandidateRejectionReason.DEPENDENCY_RANGE
            )
            raise _reject_assignment(
                assignment,
                reason_code,
                cut_kind=(
                    SolverCandidateCutKind.PAIR
                    if literal_ids is not None
                    else SolverCandidateCutKind.EXACT_ASSIGNMENT
                ),
                literal_ids=literal_ids,
                evidence={
                    "kind": relation.kind,
                    "source_occurrence_id": relation.source_occurrence_id,
                    "source_version": relation.source_candidate_version,
                    "target_occurrence_id": relation.target_occurrence_id,
                    "package_name": relation.package_name,
                    "required_range": relation.version_range,
                    "resolved_version": package.version,
                },
                summary=(
                    f"resolved {relation.kind} {relation.package_name!r}@{package.version} "
                    f"does not satisfy {relation.version_range!r}"
                ),
            )


def _vulnerable_fixed_version(
    prepared: _PreparedPortfolioProblem,
    finding: SolverFindingRequirement,
) -> str | None:
    """Return the vulnerable leaf's OSV floor, not a transitive parent floor."""
    target_by_id = {target.occurrence_id: target for target in prepared.targets}
    target = target_by_id.get(finding.target_occurrence_id)
    group_by_id = {group.group_id: group for group in prepared.groups}
    group = group_by_id.get(target.group_id) if target else None
    if group is not None:
        for issue in group.issues:
            try:
                identity = _issue_identity(issue)
            except ValueError:
                continue
            if identity == finding.finding_id and issue.fixed_version:
                return issue.fixed_version
        if group.fix_plan and group.fix_plan.fixed_version:
            return group.fix_plan.fixed_version
    return finding.fixed_version


def _authorized_workaround(
    finding: SolverFindingRequirement,
    prepared: _PreparedPortfolioProblem,
    selected_plan: SolverCandidatePlan,
) -> bool:
    """Return whether this finding has an already-QA-passed authorized workaround."""
    if not finding.workaround_available or not finding.workaround_plan_ids:
        return False
    target = next(
        (item for item in prepared.targets if item.occurrence_id == finding.target_occurrence_id),
        None,
    )
    if (
        target is None
        or not target.is_terminal
        or target.terminal_status != "qa_passed"
        or target.strategy.replace("-", "_").lower() not in {"code_workaround", "workaround"}
    ):
        return False
    decision = next(
        (item for item in selected_plan.task_decisions if item.task_id == target.task_id),
        None,
    )
    return bool(
        decision
        and str(decision.selected_strategy).lower().replace("-", "_")
        in {"code_workaround", "workaround"}
        and set(finding.workaround_plan_ids) & set(decision.selected_plan_issue_ids)
    )


def _validate_final_coverage(
    prepared: _PreparedPortfolioProblem,
    snapshot: NpmGraphSnapshot,
    selected_plan: SolverCandidatePlan,
    assignment: Mapping[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Validate every known occurrence and reject newly introduced vulnerable copies."""
    original_occurrences = {
        occurrence.occurrence_id: occurrence for occurrence in prepared.npm_snapshot.occurrences
    }
    original_versions_by_package: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    floors_by_package: dict[tuple[str, str], list[str]] = {}
    for occurrence in prepared.npm_snapshot.occurrences:
        if occurrence.installed_version:
            original_versions_by_package[(occurrence.manifest_path, occurrence.package_name)][
                occurrence.installed_version.strip().lstrip("vV")
            ] += 1
    for finding in prepared.findings:
        floor = _vulnerable_fixed_version(prepared, finding)
        if floor:
            target = next(
                item
                for item in prepared.targets
                if item.occurrence_id == finding.target_occurrence_id
            )
            floors_by_package.setdefault(
                (target.manifest_path, finding.vulnerable_package), []
            ).append(floor)
    resolved_versions_by_package: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for occurrence in snapshot.occurrences:
        key = (occurrence.manifest_path, occurrence.package_name)
        version = str(occurrence.installed_version or "").strip().lstrip("vV")
        resolved_versions_by_package[key][version] += 1
        if resolved_versions_by_package[key][version] <= original_versions_by_package.get(
            key, Counter()
        ).get(version, 0):
            continue
        floors = floors_by_package.get(key, ())
        if not floors:
            continue
        if not occurrence.installed_version:
            raise _CertificationUnknown(
                f"new vulnerable package copy {occurrence.occurrence_id!r} has no version"
            )
        for floor in floors:
            checked = check_npm_range(f">={floor}", occurrence.installed_version)
            if checked.matches is None:
                raise _CertificationUnknown(
                    f"invalid vulnerable floor {floor!r} for {occurrence.occurrence_id!r}"
                )
            if not checked.matches:
                raise _reject_assignment(
                    assignment,
                    SolverCandidateRejectionReason.NEW_VULNERABLE_COPY,
                    evidence={
                        "kind": "new_vulnerable_copy",
                        "occurrence_id": occurrence.occurrence_id,
                        "package_name": occurrence.package_name,
                        "resolved_version": occurrence.installed_version,
                        "required_floors": sorted(floors),
                    },
                    summary=(
                        f"new vulnerable package copy {occurrence.occurrence_id!r} "
                        f"remains below {floor}"
                    ),
                )

    covered: list[str] = []
    workaround: list[str] = []
    unresolved: list[str] = []
    for finding in prepared.findings:
        target = next(
            (
                item
                for item in prepared.targets
                if item.occurrence_id == finding.target_occurrence_id
            ),
            None,
        )
        if target is None:
            raise _CertificationUnknown(
                f"finding {finding.coverage_id!r} has no prepared solver target"
            )
        if target.is_terminal and target.terminal_status != "qa_passed":
            unresolved.append(finding.coverage_id)
            continue
        prepared_occurrence = original_occurrences.get(finding.vulnerable_occurrence_id)
        if prepared_occurrence is None:
            raise _CertificationUnknown(
                f"prepared vulnerable occurrence {finding.vulnerable_occurrence_id!r} is missing"
            )
        key = (prepared_occurrence.manifest_path, finding.vulnerable_package)
        occurrences = [
            occurrence
            for occurrence in snapshot.occurrences
            if (occurrence.manifest_path, occurrence.package_name) == key
        ]
        floor = _vulnerable_fixed_version(prepared, finding)
        if not occurrences:
            covered.append(finding.coverage_id)
            continue
        if floor:
            below_floor = False
            for occurrence in occurrences:
                if not occurrence.installed_version:
                    raise _CertificationUnknown(f"known vulnerable package {key!r} has no version")
                checked = check_npm_range(f">={floor}", occurrence.installed_version)
                if checked.matches is None:
                    raise _CertificationUnknown(
                        f"invalid vulnerable floor {floor!r} for package {key!r}"
                    )
                below_floor = below_floor or not checked.matches
            if not below_floor:
                covered.append(finding.coverage_id)
                continue
        if _authorized_workaround(finding, prepared, selected_plan):
            workaround.append(finding.coverage_id)
        else:
            unresolved.append(finding.coverage_id)
    return sorted(covered), sorted(workaround), sorted(unresolved)


def _read_candidate_assignment(selected_plan: SolverCandidatePlan) -> dict[str, str]:
    """Return one complete deterministic map of target occurrence to candidate version."""
    return dict(sorted(selected_plan.selected_candidate_versions.items()))


def _assignment_mutations(selected_plan: SolverCandidatePlan) -> list[SolverMutation]:
    """Flatten the selected batches to the exact package mutations."""
    return [mutation for batch in selected_plan.batches for mutation in batch.mutations]


def _peer_conflict_literal_ids(
    prepared: _PreparedPortfolioProblem,
    assignment: Mapping[str, str],
    peer_evidence: Sequence[Any],
) -> tuple[str, str] | None:
    """Map parsed npm peer evidence only to one exact mutable physical relation."""
    targets_by_id = {
        target.occurrence_id: target
        for target in prepared.targets
        if target.eligible_for_atomic_update
    }
    mapped_pairs: set[tuple[str, str]] = set()
    for evidence in peer_evidence:
        requester = str(getattr(evidence, "requester_package", "") or "").strip()
        requester_version = (
            str(getattr(evidence, "requester_version", "") or "").strip().lstrip("vV")
        )
        peer = str(getattr(evidence, "peer_package", "") or "").strip()
        peer_version = str(getattr(evidence, "observed_version", "") or "").strip().lstrip("vV")
        required_range = str(getattr(evidence, "required_range", "") or "").strip()
        if not all((requester, requester_version, peer, peer_version, required_range)):
            continue
        source_targets = [
            target
            for target in targets_by_id.values()
            if target.target_package_name == requester
            and assignment.get(target.occurrence_id, "").strip().lstrip("vV") == requester_version
        ]
        for source in source_targets:
            for relation in prepared.subgraph.candidate_relations:
                if (
                    not relation.is_modelled
                    or not relation.is_range_supported
                    or relation.source_occurrence_id != source.occurrence_id
                    or relation.source_candidate_version != requester_version
                    or relation.package_name != peer
                    or relation.version_range != required_range
                    or relation.kind != "peer"
                    or relation.is_optional
                    or relation.target_occurrence_id is None
                ):
                    continue
                target = targets_by_id.get(relation.target_occurrence_id)
                if (
                    target is not None
                    and target.target_package_name == peer
                    and assignment.get(target.occurrence_id, "").strip().lstrip("vV")
                    == peer_version
                ):
                    mapped_pairs.add((source.occurrence_id, target.occurrence_id))
    if len(mapped_pairs) != 1:
        return None
    return next(iter(mapped_pairs))


def _peer_evidence_payload(peer_evidence: Sequence[Any]) -> list[dict[str, str | None]]:
    """Project only parsed peer fields; omit raw npm excerpts."""
    return [
        {
            "requester_package": getattr(item, "requester_package", None),
            "requester_version": getattr(item, "requester_version", None),
            "peer_package": getattr(item, "peer_package", None),
            "required_range": getattr(item, "required_range", None),
            "observed_version": getattr(item, "observed_version", None),
        }
        for item in peer_evidence
    ]


def _peer_conflict_rejection(
    prepared: _PreparedPortfolioProblem,
    assignment: Mapping[str, str],
    peer_evidence: Sequence[Any],
) -> _AssignmentRejected:
    """Build a peer pair cut only when requester and physical peer map uniquely."""
    literal_ids = _peer_conflict_literal_ids(prepared, assignment, peer_evidence)
    return _reject_assignment(
        assignment,
        SolverCandidateRejectionReason.PEER_CONFLICT,
        cut_kind=(
            SolverCandidateCutKind.PAIR
            if literal_ids is not None
            else SolverCandidateCutKind.EXACT_ASSIGNMENT
        ),
        literal_ids=literal_ids,
        evidence={
            "kind": "npm_peer_conflict",
            "peer_conflicts": _peer_evidence_payload(peer_evidence),
        },
        summary="npm resolution rejected the assignment with a peer conflict",
    )


def _certify_assignment(
    sandbox: DockerSandbox,
    workspace_volume: str,
    sandbox_factory: Callable[..., DockerSandbox],
    prepared: _PreparedPortfolioProblem,
    selected_plan: SolverCandidatePlan,
    assignment: Mapping[str, str],
    runtime: SolverRuntimeFingerprint,
    archive_path: str,
    scratch_root: str,
    metrics: dict[str, Any],
    deadline: float,
) -> _AssignmentResult:
    """Resolve one candidate in a fresh subtree of the shared workspace archive."""
    assignment_digest = _digest(dict(sorted(assignment.items())))
    scratch_prefix = _validate_workspace_path(f"{scratch_root}/{assignment_digest}")
    scratch_absolute = f"/workspace/{scratch_prefix}"
    diagnostics: list[str] = []
    prefix_graph_digests: dict[str, str] = {}
    resolved_snapshot: NpmGraphSnapshot | None = None
    covered: list[str] = []
    workaround: list[str] = []
    unresolved: list[str] = []
    status = PackageResolutionStatus.UNKNOWN
    cleanup_errors: list[str] = []
    rejection_conflict: SolverCandidateConflict | None = None

    try:
        assignment_mutations = _assignment_mutations(selected_plan)
        _validate_candidate_runtime(
            prepared,
            assignment,
            {mutation.occurrence_id for mutation in assignment_mutations},
            runtime,
        )
        scratch_root_path = _validate_workspace_path(scratch_root)
        _run_command(
            sandbox,
            f"mkdir -p -- /workspace/{shlex.quote(scratch_root_path)} && "
            f"rm -rf -- {shlex.quote(scratch_absolute)} && "
            f"mkdir -p -- {shlex.quote(scratch_absolute)}",
            deadline,
            "scratch workspace creation",
        )
        extraction_started = time.monotonic()
        try:
            _run_command(
                sandbox,
                f"tar -xf {shlex.quote(archive_path)} -C {shlex.quote(scratch_absolute)}",
                deadline,
                "workspace archive extraction",
            )
        finally:
            metrics["extraction_time_seconds"] += max(0.0, time.monotonic() - extraction_started)
        archived_documents = _read_npm_documents(sandbox, scratch_prefix, deadline, metrics=metrics)
        archived_snapshot = load_npm_graph_snapshot_from_documents(archived_documents)
        if (
            archived_snapshot.diagnostics
            or archived_snapshot.repository_fingerprint
            != prepared.npm_snapshot.repository_fingerprint
        ):
            raise _CertificationUnknown(
                "extracted workspace fingerprint does not match prepared snapshot"
            )
        pruning_baseline = _pruning_baseline_checkpoint(
            scratch_prefix,
            (mutation.manifest_path for mutation in assignment_mutations),
            archived_documents,
        )

        targets_by_task = {target.task_id: target for target in prepared.targets}
        applied_occurrence_ids: set[str] = set()
        batches_by_id = {batch.batch_id: batch for batch in selected_plan.batches}
        ordered_batch_ids = [
            batch_id
            for phase in sorted(selected_plan.phases, key=lambda item: item.phase_number)
            for batch_id in phase.batch_ids
            if batch_id in batches_by_id
        ]
        ordered_batch_ids.extend(
            batch_id for batch_id in sorted(batches_by_id) if batch_id not in set(ordered_batch_ids)
        )
        audit_only_batch_ids: set[str] = set()
        executable_batch_count = 0
        for batch in selected_plan.batches:
            if batch.dispatchable:
                executable_batch_count += 1
                continue
            terminal_only_batch = bool(batch.task_ids) and all(
                (target := targets_by_task.get(task_id)) is not None and target.is_terminal
                for task_id in batch.task_ids
            )
            if batch.mutations or not terminal_only_batch:
                raise _CertificationUnknown(
                    "selected plan contains a non-dispatchable batch that is not terminal-only"
                )
            audit_only_batch_ids.add(batch.batch_id)
        if executable_batch_count == 0:
            raise _CertificationUnknown("selected plan contains no executable package batch")

        for batch_id in ordered_batch_ids:
            if batch_id in audit_only_batch_ids:
                continue
            batch = batches_by_id[batch_id]
            solver_mutations, scratch_mutations, checkpoint, touched_files = _stage_batch_mutations(
                sandbox, scratch_prefix, batch, deadline
            )
            try:
                affected_manifests = {
                    _scratch_workspace_path(scratch_prefix, targets_by_task[task_id].manifest_path)
                    for task_id in batch.task_ids
                    if task_id in targets_by_task
                }
                affected_manifests.update(mutation.manifest_path for mutation in scratch_mutations)
                workspace_dirs = sorted(
                    {_workspace_dir_for_manifest(path) for path in affected_manifests}
                )
                if not workspace_dirs:
                    raise _CertificationUnknown(
                        f"batch {batch_id!r} has no affected npm workspace directory"
                    )
                for workspace_dir in workspace_dirs:
                    command = f"cd {shlex.quote(workspace_dir)} && {_NPM_INSTALL}"
                    metrics["strict_install_invocations"] += 1
                    package_manager_started = time.monotonic()
                    try:
                        result = _run_command(
                            sandbox,
                            command,
                            deadline,
                            "npm resolution",
                            allow_nonzero=True,
                        )
                    finally:
                        metrics["package_manager_time_seconds"] += max(
                            0.0, time.monotonic() - package_manager_started
                        )
                    if result.exit_code != 0:
                        category = _install_error_category(
                            result.stdout, result.stderr, result.exit_code
                        )
                        peer_evidence = parse_peer_conflict_evidence(result.stdout, result.stderr)
                        if peer_evidence or category == "PEER_CONFLICT":
                            raise _peer_conflict_rejection(prepared, assignment, peer_evidence)
                        conflict_text = f"{result.stdout}\n{result.stderr}".casefold()
                        deterministic_rejection = category == "ENGINE_CONFLICT" or any(
                            marker in conflict_text
                            for marker in ("eresolve", "eoverride", "ebadplatform")
                        )
                        if deterministic_rejection:
                            raise _reject_assignment(
                                assignment,
                                SolverCandidateRejectionReason.UNCLASSIFIED,
                                evidence={
                                    "kind": "npm_install_rejection",
                                    "category": category,
                                    "exit_code": result.exit_code,
                                },
                                summary=(
                                    f"npm resolution rejected the complete assignment ({category})"
                                ),
                            )
                        raise _CertificationUnknown(f"npm installation failed ({category})")

                documents = _read_npm_documents(sandbox, scratch_prefix, deadline, metrics=metrics)
                resolved_snapshot = load_npm_graph_snapshot_from_documents(documents)
                if resolved_snapshot.diagnostics:
                    raise _CertificationUnknown(
                        "resolved npm graph is malformed or unsupported: "
                        + " | ".join(resolved_snapshot.diagnostics)
                    )
                if not resolved_snapshot.lockfiles:
                    raise _CertificationUnknown("npm resolution produced no supported lockfile")
                lock_conflict = _lockfile_pair_conflict(resolved_snapshot)
                if lock_conflict:
                    raise _CertificationUnknown(lock_conflict)
                _verify_staged_mutations(
                    sandbox,
                    scratch_mutations,
                    checkpoint,
                    pruning_baseline=pruning_baseline,
                )
                applied_occurrence_ids.update(
                    mutation.occurrence_id for mutation in solver_mutations
                )
                pruned_occurrence_ids = _pruned_override_mutations(
                    resolved_snapshot,
                    solver_mutations,
                    scratch_mutations,
                    pruning_baseline,
                )
                _validate_candidate_relations(
                    prepared,
                    resolved_snapshot,
                    assignment,
                    applied_occurrence_ids,
                    pruned_occurrence_ids,
                )
                prefix_graph_digests[batch_id] = resolved_snapshot.repository_fingerprint
            except _AssignmentRejected as exc:
                if checkpoint is not None:
                    rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                    if rollback_error:
                        raise _CertificationUnknown(rollback_error) from exc
                raise
            except Exception as exc:
                if checkpoint is not None:
                    rollback_error = _restore_package_checkpoint(sandbox, checkpoint, touched_files)
                    if rollback_error:
                        raise _CertificationUnknown(rollback_error) from exc
                raise

        if resolved_snapshot is None:
            raise _CertificationUnknown("no package-manager graph was resolved")
        covered, workaround, unresolved = _validate_final_coverage(
            prepared, resolved_snapshot, selected_plan, assignment
        )
        status = PackageResolutionStatus.CERTIFIED
    except _AssignmentRejected as exc:
        diagnostics.append(exc.conflict.summary)
        rejection_conflict = exc.conflict
        status = PackageResolutionStatus.REJECTED
    except _CertificationUnknown as exc:
        diagnostics.append(str(exc))
        status = PackageResolutionStatus.UNKNOWN
    except Exception as exc:  # noqa: BLE001
        diagnostics.append(f"unexpected package-certification failure: {exc}")
        status = PackageResolutionStatus.UNKNOWN
    finally:
        cleanup_command = f"rm -rf -- {shlex.quote(scratch_absolute)}"
        remaining = _remaining_seconds(deadline)
        try:
            cleanup_result = sandbox.run(cleanup_command, timeout=max(1, math.floor(remaining)))
            if cleanup_result.exit_code != 0:
                cleanup_errors.append("assignment scratch cleanup failed")
        except Exception:  # noqa: BLE001
            cleanup_errors.append("assignment scratch cleanup failed")
        if cleanup_errors:
            try:
                with sandbox_factory(
                    repo_root=None, workspace_volume=workspace_volume
                ) as cleanup_sandbox:
                    cleanup_result = cleanup_sandbox.run(
                        cleanup_command,
                        timeout=max(1, math.floor(_remaining_seconds(deadline))),
                    )
                    if cleanup_result.exit_code != 0:
                        cleanup_errors.append(
                            "retry scratch cleanup failed: "
                            f"{cleanup_result.stderr or cleanup_result.stdout or cleanup_result.exit_code}"
                        )
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"retry scratch cleanup failed: {exc}")
            diagnostics.extend(cleanup_errors)
            status = PackageResolutionStatus.UNKNOWN
            rejection_conflict = None
            covered = []
            workaround = []
            unresolved = sorted(finding.coverage_id for finding in prepared.findings)

    return _AssignmentResult(
        status=status,
        snapshot=resolved_snapshot,
        covered_coverage_ids=tuple(sorted(covered)),
        workaround_coverage_ids=tuple(sorted(workaround)),
        unresolved_coverage_ids=tuple(sorted(unresolved)),
        resolved_lockfile_digests=(
            {lockfile.path: lockfile.raw_digest for lockfile in resolved_snapshot.lockfiles}
            if resolved_snapshot is not None
            else {}
        ),
        batch_prefix_graph_digests=dict(sorted(prefix_graph_digests.items())),
        diagnostics=tuple(diagnostics),
        rejection_conflict=(
            rejection_conflict if status == PackageResolutionStatus.REJECTED else None
        ),
    )


def _unknown_solver_plan(
    prepared: _PreparedPortfolioProblem,
    diagnostics: Sequence[str],
) -> SolverRemediationPlan:
    """Build a no-selection UNKNOWN plan for a failed certification prerequisite."""
    domain_payload = {
        key: [candidate.model_dump(mode="json") for candidate in values]
        for key, values in sorted(prepared.candidate_domains.items())
    }
    return SolverRemediationPlan(
        status=SolverStatus.UNKNOWN,
        input_digest=_digest(
            {
                "subgraph": prepared.subgraph.model_dump(mode="json"),
                "candidate_catalog_digest": prepared.candidate_catalog_digest,
                "diagnostics": sorted(set(diagnostics)),
            }
        ),
        domain_digest=_digest(
            {
                "domains": domain_payload,
                "candidate_catalog_digest": prepared.candidate_catalog_digest,
            }
        ),
        repository_digest=prepared.workspace_repository_fingerprint,
        candidate_catalog_complete=prepared.candidate_catalog_complete,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
        task_revisions={target.task_id: 0 for target in prepared.targets},
        unresolved_finding_ids=sorted({finding.finding_id for finding in prepared.findings}),
        diagnostics=sorted(set(diagnostics)),
    )


def _make_certificate(
    plan: PortfolioPlan,
    prepared: _PreparedPortfolioProblem,
    runtime: SolverRuntimeFingerprint,
    *,
    status: PackageResolutionStatus,
    assignment: Mapping[str, str] | None = None,
    result: _AssignmentResult | None = None,
    rejected_assignment_digests: Sequence[str] = (),
    diagnostics: Sequence[str] = (),
    candidate_plan_id: str | None = None,
    rejection_conflicts: Sequence[SolverCandidateConflict] = (),
    certification_statistics: CertificationStatistics | None = None,
    workspace_prefix_provenance: QAPassedWorkspacePrefix | None = None,
) -> PackageResolutionCertificate:
    """Bind one resolver result to the selected plan and current task revisions."""
    unresolved = (
        result.unresolved_coverage_ids
        if result is not None
        else tuple(sorted(finding.coverage_id for finding in prepared.findings))
    )
    if candidate_plan_id is None and plan.solver_plan is not None:
        selected = plan.solver_plan.selected_plan
        candidate_plan_id = selected.candidate_plan_id if selected is not None else None
    return PackageResolutionCertificate(
        status=status,
        portfolio_plan_id=plan.portfolio_plan_id,
        candidate_plan_id=candidate_plan_id,
        solver_input_digest=plan.solver_input_digest or plan.solver_plan.input_digest,
        repository_fingerprint=plan.repository_fingerprint,
        task_revisions=plan.task_revisions,
        workspace_graph_digest=prepared.workspace_repository_fingerprint,
        candidate_catalog_digest=prepared.candidate_catalog_digest,
        candidate_assignment_digest=_digest(dict(sorted((assignment or {}).items()))),
        resolved_graph_digest=(
            result.snapshot.repository_fingerprint
            if result is not None and result.snapshot is not None
            else _digest({})
        ),
        resolved_lockfile_digests=(
            dict(result.resolved_lockfile_digests) if result is not None else {}
        ),
        runtime_fingerprint=runtime,
        covered_coverage_ids=(list(result.covered_coverage_ids) if result is not None else []),
        workaround_coverage_ids=(
            list(result.workaround_coverage_ids) if result is not None else []
        ),
        unresolved_coverage_ids=list(unresolved),
        rejected_assignment_digests=list(rejected_assignment_digests),
        rejection_conflicts=list(rejection_conflicts),
        certification_statistics=certification_statistics or CertificationStatistics(),
        batch_prefix_graph_digests=(
            dict(result.batch_prefix_graph_digests) if result is not None else {}
        ),
        workspace_prefix_provenance=workspace_prefix_provenance,
        diagnostics=sorted(set([*diagnostics, *(result.diagnostics if result else ())])),
    )


def _attach_resolution_certificate(
    plan: PortfolioPlan,
    certificate: PackageResolutionCertificate,
) -> PortfolioPlan:
    """Attach a certificate, reconcile coverage, and derive final plan identity."""
    solver_plan = plan.solver_plan
    if solver_plan is None:
        raise ValueError("cannot attach a resolution certificate without a solver plan")
    if certificate.portfolio_plan_id != plan.portfolio_plan_id:
        raise ValueError("resolution certificate is bound to a different portfolio plan")
    if certificate.solver_input_digest != plan.solver_input_digest:
        raise ValueError("resolution certificate solver input digest does not match the plan")
    if certificate.repository_fingerprint != plan.repository_fingerprint:
        raise ValueError("resolution certificate host fingerprint does not match the plan")
    if certificate.workspace_graph_digest != plan.workspace_graph_digest:
        raise ValueError("resolution certificate workspace graph digest does not match the plan")
    if certificate.candidate_catalog_digest != solver_plan.candidate_catalog_digest:
        raise ValueError("resolution certificate candidate catalog digest does not match the plan")
    if certificate.task_revisions != plan.task_revisions:
        raise ValueError("resolution certificate task revisions do not match the plan")

    selected = solver_plan.selected_plan
    if certificate.status == PackageResolutionStatus.CERTIFIED:
        if (
            solver_plan.status not in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
            or selected is None
        ):
            raise ValueError("a certified certificate requires an accepted selected solver plan")
        if not solver_plan.candidate_catalog_complete:
            raise ValueError("a certified certificate requires a complete candidate catalog")
        if certificate.candidate_plan_id != selected.candidate_plan_id:
            raise ValueError("resolution certificate candidate plan ID does not match the plan")
        assignment_digest = _digest(dict(sorted(selected.selected_candidate_versions.items())))
        if certificate.candidate_assignment_digest != assignment_digest:
            raise ValueError(
                "resolution certificate selected assignment digest does not match the plan"
            )

    resolved_coverage = set(certificate.covered_coverage_ids)
    workaround_coverage = set(certificate.workaround_coverage_ids)
    unresolved_coverage = set(certificate.unresolved_coverage_ids)
    if selected is not None:
        known_coverage_ids = {
            coverage_id for batch in selected.batches for coverage_id in batch.coverage_finding_ids
        }
        certified_coverage_ids = resolved_coverage | workaround_coverage | unresolved_coverage
        if (
            certificate.status == PackageResolutionStatus.CERTIFIED
            and known_coverage_ids != certified_coverage_ids
        ):
            raise ValueError("certificate coverage IDs do not match the selected batch projections")

    status_by_issue: dict[str, set[str]] = defaultdict(set)
    for batch in selected.batches if selected is not None else []:
        for coverage_id, finding_id in batch.coverage_finding_ids.items():
            if coverage_id in resolved_coverage:
                status_by_issue[finding_id].add("resolved")
            elif coverage_id in workaround_coverage:
                status_by_issue[finding_id].add("workaround")
            else:
                status_by_issue[finding_id].add("unresolved")
    issue_unresolved = sorted(
        finding_id for finding_id, statuses in status_by_issue.items() if "unresolved" in statuses
    )
    if selected is not None:
        projected_batches: list[SolverBatch] = []
        for batch in selected.batches:
            batch_coverage_ids = set(batch.coverage_finding_ids)
            resolved = sorted(batch_coverage_ids & resolved_coverage)
            workaround = sorted(batch_coverage_ids & workaround_coverage)
            unresolved = sorted(batch_coverage_ids - set(resolved) - set(workaround))
            issue_ids = set(batch.coverage_finding_ids.values())
            resolved_findings = sorted(
                finding_id
                for finding_id in issue_ids
                if status_by_issue.get(finding_id) == {"resolved"}
            )
            workaround_findings = sorted(
                finding_id
                for finding_id in issue_ids
                if status_by_issue.get(finding_id) == {"workaround"}
            )
            unresolved_findings = sorted(
                (set(batch.unresolved_finding_ids) - issue_ids)
                | {
                    finding_id
                    for finding_id in issue_ids
                    if "unresolved" in status_by_issue.get(finding_id, set())
                }
            )
            projected_batches.append(
                batch.model_copy(
                    update={
                        "resolved_finding_ids": resolved_findings,
                        "workaround_finding_ids": workaround_findings,
                        "unresolved_finding_ids": unresolved_findings,
                        "resolved_coverage_ids": resolved,
                        "workaround_coverage_ids": workaround,
                        "unresolved_coverage_ids": unresolved,
                    }
                )
            )
        task_decisions = [
            decision.model_copy(
                update={"allowed_alternative_versions": []}
                if certificate.status == PackageResolutionStatus.CERTIFIED
                else {}
            )
            for decision in selected.task_decisions
        ]
        selected = selected.model_copy(
            update={
                "coverage_ids": sorted(resolved_coverage | workaround_coverage),
                "unresolved_ids": sorted(unresolved_coverage),
                "task_decisions": task_decisions,
                "batches": projected_batches,
            }
        )
        if certificate.status == PackageResolutionStatus.CERTIFIED:
            candidate_plans = [selected]
        else:
            candidate_plans = [
                selected if candidate.candidate_plan_id == selected.candidate_plan_id else candidate
                for candidate in solver_plan.candidate_plans
            ]
        solver_plan = solver_plan.model_copy(
            update={
                "candidate_plans": candidate_plans,
                "selected_plan": selected,
                "unresolved_finding_ids": issue_unresolved,
            }
        )

    certificate_payload = certificate.model_dump(mode="json")
    # Aggregate timings are observability only; they must not change plan identity.
    certificate_payload.pop("certification_statistics", None)
    certificate_digest = _digest(certificate_payload)
    certified_dispatch = bool(
        certificate.status == PackageResolutionStatus.CERTIFIED
        and solver_plan.status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
        and solver_plan.candidate_catalog_complete
        and selected is not None
    )
    batch_by_id = {
        batch.batch_id: batch for batch in (selected.batches if selected is not None else [])
    }
    clusters = [
        cluster.model_copy(
            update={
                "dispatchable": bool(
                    certified_dispatch
                    and (batch := batch_by_id.get(cluster.cluster_id)) is not None
                    and batch.dispatchable
                )
            }
        )
        for cluster in plan.clusters
    ]
    final_plan_digest = _digest(
        {
            "base_plan_digest": plan.plan_digest,
            "portfolio_plan_id": plan.portfolio_plan_id,
            "resolution_certificate_digest": certificate_digest,
        }
    )
    return plan.model_copy(
        update={
            "plan_id": f"portfolio-{final_plan_digest[:24]}",
            "plan_digest": final_plan_digest,
            "resolution_certificate": certificate,
            "solver_plan": solver_plan,
            "clusters": clusters,
        }
    )


def _downgrade_solver_plan(
    solver_plan: SolverRemediationPlan,
    diagnostics: Sequence[str],
) -> SolverRemediationPlan:
    """Make a failed package-manager proof non-selected and non-dispatchable."""
    return solver_plan.model_copy(
        update={
            "status": SolverStatus.UNKNOWN,
            "selected_plan": None,
            "candidate_plans": [],
            "diagnostics": sorted(set([*solver_plan.diagnostics, *diagnostics])),
        }
    )


def _host_lockfile_diagnostics(snapshot: NpmGraphSnapshot) -> list[str]:
    """Return unsupported host npm graph conditions relevant to certification."""
    diagnostics = list(snapshot.diagnostics)
    conflict = _lockfile_pair_conflict(snapshot)
    if conflict:
        diagnostics.append(conflict)
    return sorted(set(diagnostics))


def _qa_prefix_evidence_errors(
    prefix: QAPassedWorkspacePrefix,
    task_queue: Mapping[str, RemediationTask],
    qa_results_by_attempt: Mapping[str, QAAttemptResult],
) -> list[str]:
    """Validate a checkpoint against its immutable, attempt-correlated QA evidence.

    Current task revisions may advance when a certified replan rebinds terminal
    tasks to a new portfolio plan. The checkpoint remains valid only when every
    task is still QA-passed and its recorded attempt and QA revision still match
    the stored result, and the designated result identifies the exact snapshot
    and workspace digest being restored.

    Args:
        prefix: Cumulative QA-passed workspace checkpoint under review.
        task_queue: Current Supervisor-owned task projection.
        qa_results_by_attempt: QA evidence indexed by immutable attempt ID.

    Returns:
        A list of provenance errors. An empty list means the checkpoint matches
        its recorded QA evidence.
    """

    def value(record: Any, field_name: str) -> Any:
        if isinstance(record, Mapping):
            return record.get(field_name)
        return getattr(record, field_name, None)

    errors: list[str] = []
    task_ids = set(prefix.task_ids)
    attempt_ids = prefix.qa_attempt_ids_by_task
    revisions = prefix.task_revisions
    if set(attempt_ids) != task_ids:
        errors.append("task-to-attempt mapping does not cover the checkpoint task IDs")
    if set(revisions) != task_ids:
        errors.append("QA task revisions do not cover the checkpoint task IDs")
    if not prefix.snapshot_id or not prefix.snapshot_attempt_id:
        errors.append("the checkpoint does not identify an exact QA-passed workspace snapshot")
    elif prefix.snapshot_attempt_id not in set(attempt_ids.values()):
        errors.append("snapshot attempt is not one of the checkpoint task attempts")

    for task_id in prefix.task_ids:
        task = task_queue.get(task_id)
        if task is None:
            errors.append(f"task {task_id!r} is missing from the current task queue")
            continue
        if getattr(task.status, "value", task.status) != TaskStatus.QA_PASSED.value:
            errors.append(f"task {task_id!r} is no longer QA_PASSED")
        attempt_id = attempt_ids.get(task_id)
        if not attempt_id:
            continue
        current_attempt_id = task.current_attempt_id
        if current_attempt_id is not None and current_attempt_id != attempt_id:
            errors.append(f"task {task_id!r} has a different active attempt")
        result = qa_results_by_attempt.get(attempt_id)
        if result is None:
            errors.append(f"QA result for task {task_id!r} attempt {attempt_id!r} is missing")
            continue
        if value(result, "attempt_id") != attempt_id or value(result, "task_id") != task_id:
            errors.append(f"QA result identity does not match task {task_id!r}")
            continue
        if value(result, "task_revision") != revisions.get(task_id):
            errors.append(f"QA result revision does not match task {task_id!r}")
        evaluation = value(result, "evaluation")
        if value(evaluation, "passed") is not True:
            errors.append(f"QA result for task {task_id!r} did not pass")

    if prefix.snapshot_attempt_id:
        snapshot_result = qa_results_by_attempt.get(prefix.snapshot_attempt_id)
        if snapshot_result is None:
            errors.append("QA result for the checkpoint snapshot attempt is missing")
        else:
            if value(snapshot_result, "workspace_snapshot_id") != prefix.snapshot_id:
                errors.append("checkpoint snapshot ID differs from its QA attempt evidence")
            if value(snapshot_result, "workspace_graph_digest") != prefix.graph_digest:
                errors.append("checkpoint graph digest differs from its QA attempt evidence")

    return errors


def _timed_registry_fetcher(
    registry_fetcher: PackumentFetcher | None,
    deadline: float,
) -> PackumentFetcher:
    """Wrap the explicit registry seam with the shared certification deadline."""
    from remediation_engine.tools.registry_tools import _fetch_package_data

    def fetch(package_name: str) -> Mapping[str, Any]:
        if _remaining_seconds(deadline) < 1.0:
            raise TimeoutError("solver certification deadline expired during registry fetch")
        if registry_fetcher is not None:
            result = registry_fetcher(package_name)
        else:
            result = _fetch_package_data(package_name, cache=None)
        if time.monotonic() >= deadline:
            raise TimeoutError("solver certification deadline expired during registry fetch")
        return result

    return fetch


def build_certified_portfolio_plan(
    repo_root: str | Path,
    workspace_volume: str,
    groups: Iterable[VulnerabilityGroup],
    task_queue: Mapping[str, RemediationTask],
    *,
    target_packages: Iterable[str] | None = None,
    peer_conflict_pairs: Iterable[tuple[str, str]] = (),
    forced_singleton_task_ids: Iterable[str] = (),
    settings: AppSettings | None = None,
    portfolio_iteration: int = 0,
    portfolio_replan_request: PortfolioReplanRequest | None = None,
    prior_portfolio_plan: PortfolioPlan | None = None,
    qa_results_by_attempt: Mapping[str, QAAttemptResult] | None = None,
    qa_passed_workspace_prefix: QAPassedWorkspacePrefix | None = None,
    registry_fetcher: PackumentFetcher | None = None,
    sandbox_factory: Callable[..., DockerSandbox] = DockerSandbox,
) -> PortfolioPlan:
    """Build and certify one complete package assignment in the live workspace.

    Args:
        repo_root: Host repository used only for fingerprints and stale-plan checks.
        workspace_volume: Docker volume populated by ``workspace_builder``.
        groups: Current triaged and prepared vulnerability groups.
        task_queue: Supervisor-owned task revisions for this portfolio iteration.
        target_packages: Optional synthetic mutation-task allowlist.
        peer_conflict_pairs: QA peer conflicts retained from an earlier iteration.
        forced_singleton_task_ids: Tasks that cannot join atomic package batches.
        settings: Explicit runtime settings, including the certification deadline.
        portfolio_iteration: Outer portfolio iteration number.
        portfolio_replan_request: Optional Supervisor-owned replan constraints.
        prior_portfolio_plan: Current committed plan for verifying an earlier QA-passed prefix.
        qa_results_by_attempt: Attempt-correlated QA evidence from the source plan.
        qa_passed_workspace_prefix: Latest cumulative QA-passed workspace checkpoint.
        registry_fetcher: Injectable fresh raw-packument fetcher for deterministic tests.
        sandbox_factory: Injectable DockerSandbox-compatible context manager.

    Returns:
        A solver-backed plan with a matching package-resolution certificate. Any
        missing evidence returns an UNKNOWN or REJECTED plan with no selected plan.

    Raises:
        ValueError: If no active SCA package tasks exist.
    """
    certification_started = time.monotonic()
    metrics = _new_certification_metrics()
    resolved_settings = settings or AppSettings.from_env()
    deadline = time.monotonic() + resolved_settings.solver_certification_timeout_seconds
    root = Path(repo_root).resolve()
    group_list = [group.model_copy(deep=True) for group in groups]
    queue = {task_id: task.model_copy(deep=True) for task_id, task in task_queue.items()}
    host_snapshot = load_npm_graph_snapshot(root)
    host_fingerprint = host_snapshot.repository_fingerprint
    prefix_diagnostics: list[str] = []
    qa_passed_prefix_provenance = _last_qa_passed_prefix_provenance(
        prior_portfolio_plan,
        queue,
        portfolio_replan_request,
        host_fingerprint,
        qa_results_by_attempt=qa_results_by_attempt,
        qa_passed_workspace_prefix=qa_passed_workspace_prefix,
        diagnostics=prefix_diagnostics,
    )
    qa_passed_prefix_digest = (
        qa_passed_prefix_provenance.graph_digest
        if qa_passed_prefix_provenance is not None
        else None
    )
    host_diagnostics = _host_lockfile_diagnostics(host_snapshot)
    host_prepared = _prepare_portfolio_problem(
        root,
        group_list,
        queue,
        target_packages=target_packages,
        peer_conflict_pairs=tuple(peer_conflict_pairs),
        forced_singleton_task_ids=tuple(forced_singleton_task_ids),
        portfolio_replan_request=portfolio_replan_request,
        npm_snapshot=host_snapshot,
        repository_fingerprint=host_fingerprint,
        settings=resolved_settings,
    )

    def make_certificate(
        plan: PortfolioPlan,
        prepared_problem: _PreparedPortfolioProblem,
        runtime_fingerprint: SolverRuntimeFingerprint,
        **values: Any,
    ) -> PackageResolutionCertificate:
        """Attach invocation-local aggregate measurements to one certificate."""
        values["diagnostics"] = sorted(
            set([*(values.get("diagnostics") or ()), *prefix_diagnostics])
        )
        if (
            values.get("status") == PackageResolutionStatus.CERTIFIED
            and qa_passed_prefix_provenance is not None
        ):
            values["workspace_prefix_provenance"] = qa_passed_prefix_provenance.model_copy(
                update={"certified_by_portfolio_plan_id": plan.portfolio_plan_id}
            )
        return _make_certificate(
            plan,
            prepared_problem,
            runtime_fingerprint,
            certification_statistics=_certification_statistics(
                prepared_problem, metrics, certification_started
            ),
            **values,
        )

    fallback_runtime = _unknown_runtime_fingerprint()
    if not workspace_volume or not str(workspace_volume).strip():
        metrics["unknown_count"] += 1
        diagnostics = [*host_diagnostics, "workspace volume is missing"]
        solver_plan = _unknown_solver_plan(host_prepared, diagnostics)
        plan = _project_prepared_portfolio_plan(
            host_prepared, solver_plan, portfolio_iteration=portfolio_iteration
        )
        certificate = make_certificate(
            plan,
            host_prepared,
            fallback_runtime,
            status=PackageResolutionStatus.UNKNOWN,
            diagnostics=diagnostics,
        )
        return _attach_resolution_certificate(plan, certificate)

    prepared = host_prepared
    runtime = fallback_runtime
    run_id = uuid.uuid4().hex
    archive_path = f"/tmp/remedy-plan-cert-{run_id}.tar"
    scratch_root = f".remedy-plan-cert/{run_id}"
    try:
        with (
            sandbox_factory(repo_root=None, workspace_volume=workspace_volume) as sandbox,
            _certification_artifact_scope(
                sandbox,
                workspace_volume,
                sandbox_factory,
                scratch_root,
                archive_path,
                deadline,
            ),
        ):
            try:
                runtime = _read_runtime_fingerprint(sandbox, deadline)
                if (
                    portfolio_replan_request is not None
                    and qa_passed_prefix_provenance is not None
                    and qa_passed_prefix_provenance.snapshot_id
                ):
                    sandbox.restore_workspace_snapshot(qa_passed_prefix_provenance.snapshot_id)
                    prefix_diagnostics.append(
                        "restored the shared workspace from the latest QA-passed snapshot "
                        f"{qa_passed_prefix_provenance.snapshot_id!r} before portfolio replan"
                    )
                live_documents = _read_npm_documents(sandbox, "", deadline, metrics=metrics)
                workspace_snapshot = load_npm_graph_snapshot_from_documents(live_documents)
                _assert_workspace_matches_host(
                    host_snapshot,
                    workspace_snapshot,
                    live_documents,
                    qa_passed_prefix_digest=qa_passed_prefix_digest,
                )
                lock_diagnostic = _lockfile_pair_conflict(workspace_snapshot)
                if lock_diagnostic:
                    raise _CertificationUnknown(lock_diagnostic)
                if _remaining_seconds(deadline) < 1.0:
                    raise _CertificationUnknown(
                        "certification deadline expired before solver preparation"
                    )
                prepared = _prepare_portfolio_problem(
                    root,
                    group_list,
                    queue,
                    target_packages=target_packages,
                    peer_conflict_pairs=tuple(peer_conflict_pairs),
                    forced_singleton_task_ids=tuple(forced_singleton_task_ids),
                    portfolio_replan_request=portfolio_replan_request,
                    npm_snapshot=workspace_snapshot,
                    repository_fingerprint=host_fingerprint,
                    registry_fetcher=_timed_registry_fetcher(registry_fetcher, deadline),
                    settings=resolved_settings,
                    runtime_fingerprint=runtime,
                )
                if not prepared.candidate_catalog_complete:
                    raise _CertificationUnknown(
                        "candidate catalog is incomplete: " + " | ".join(prepared.diagnostics)
                    )
                if _remaining_seconds(deadline) < 1.0:
                    raise _CertificationUnknown(
                        "certification deadline expired after registry preparation"
                    )
            except _CertificationUnknown as exc:
                metrics["unknown_count"] += 1
                if prepared is host_prepared and "workspace_snapshot" in locals():
                    prepared = _prepare_portfolio_problem(
                        root,
                        group_list,
                        queue,
                        target_packages=target_packages,
                        peer_conflict_pairs=tuple(peer_conflict_pairs),
                        repository_fingerprint=host_fingerprint,
                        registry_fetcher=(
                            _timed_registry_fetcher(registry_fetcher, deadline)
                            if registry_fetcher is not None
                            else None
                        ),
                        settings=resolved_settings,
                        runtime_fingerprint=runtime,
                    )
                diagnostics = sorted(
                    set(
                        [
                            *host_diagnostics,
                            *prepared.diagnostics,
                            *prefix_diagnostics,
                            str(exc),
                        ]
                    )
                )
                unknown_solver = _unknown_solver_plan(prepared, diagnostics)
                unknown_plan = _project_prepared_portfolio_plan(
                    prepared, unknown_solver, portfolio_iteration=portfolio_iteration
                )
                certificate = make_certificate(
                    unknown_plan,
                    prepared,
                    runtime,
                    status=PackageResolutionStatus.UNKNOWN,
                    diagnostics=diagnostics,
                )
                return _attach_resolution_certificate(unknown_plan, certificate)
            archive_bytes, archive_build_time = _build_workspace_archive(
                sandbox, archive_path, deadline
            )
            metrics["archive_bytes"] = archive_bytes
            metrics["archive_build_time_seconds"] = archive_build_time

            rejected_assignments: list[dict[str, str]] = []
            rejected_digests: list[str] = []
            rejection_conflicts: list[SolverCandidateConflict] = []
            generalized_conflicts: dict[tuple[tuple[str, str], ...], SolverCandidateConflict] = {}
            last_rejection: _AssignmentResult | None = None
            last_rejected_assignment: dict[str, str] = {}
            last_rejected_plan_id: str | None = None
            last_rejection_summary: str | None = None

            def unknown_certificate(
                source_solver_plan: SolverRemediationPlan,
                failure_diagnostics: Sequence[str],
                *,
                assignment: Mapping[str, str] | None = None,
                candidate_plan_id: str | None = None,
                result: _AssignmentResult | None = None,
            ) -> PortfolioPlan:
                """Build a no-selection UNKNOWN result retaining prior rejection proof."""
                messages = [*prepared.diagnostics, *failure_diagnostics]
                metrics["unknown_count"] += 1
                if last_rejection_summary:
                    messages.append(f"last rejected candidate: {last_rejection_summary}")
                messages = sorted(set(messages))
                downgraded = _downgrade_solver_plan(source_solver_plan, messages)
                failed_plan = _project_prepared_portfolio_plan(
                    prepared, downgraded, portfolio_iteration=portfolio_iteration
                )
                certificate = make_certificate(
                    failed_plan,
                    prepared,
                    runtime,
                    status=PackageResolutionStatus.UNKNOWN,
                    assignment=assignment,
                    result=result,
                    candidate_plan_id=candidate_plan_id or last_rejected_plan_id,
                    rejected_assignment_digests=rejected_digests,
                    rejection_conflicts=rejection_conflicts,
                    diagnostics=messages,
                )
                return _attach_resolution_certificate(failed_plan, certificate)

            while True:
                remaining = _remaining_seconds(deadline)
                if remaining < 1.0:
                    raise _CertificationUnknown(
                        "certification deadline expired during solver re-runs"
                    )
                solve_settings = replace(
                    resolved_settings,
                    solver_timeout_seconds=max(1, math.floor(remaining)),
                )
                metrics["solver_calls"] += 1
                solver_plan = solve_portfolio(
                    prepared.subgraph,
                    prepared.candidate_domains,
                    settings=solve_settings,
                    candidate_catalog_complete=prepared.candidate_catalog_complete,
                    candidate_catalog_digest=prepared.candidate_catalog_digest,
                    forbidden_assignments=rejected_assignments,
                    forbidden_conflicts=list(generalized_conflicts.values()),
                )
                if any(
                    "evidence-only model variable guard" in item for item in solver_plan.diagnostics
                ):
                    metrics["evidence_model_guard_hit"] = True
                if _remaining_seconds(deadline) < 0.001:
                    raise _CertificationUnknown("certification deadline expired during CP-SAT")
                if solver_plan.status == SolverStatus.INFEASIBLE:
                    plan = _project_prepared_portfolio_plan(
                        prepared, solver_plan, portfolio_iteration=portfolio_iteration
                    )
                    resolution_status = (
                        PackageResolutionStatus.REJECTED
                        if rejected_digests
                        else PackageResolutionStatus.UNKNOWN
                    )
                    if resolution_status == PackageResolutionStatus.UNKNOWN:
                        metrics["unknown_count"] += 1
                    certificate = make_certificate(
                        plan,
                        prepared,
                        runtime,
                        status=resolution_status,
                        assignment=last_rejected_assignment,
                        result=last_rejection,
                        candidate_plan_id=last_rejected_plan_id,
                        rejected_assignment_digests=rejected_digests,
                        rejection_conflicts=rejection_conflicts,
                        diagnostics=prepared.diagnostics,
                    )
                    return _attach_resolution_certificate(plan, certificate)

                accepted_solver_statuses = {SolverStatus.OPTIMAL}
                if resolved_settings.solver_accept_feasible:
                    accepted_solver_statuses.add(SolverStatus.FEASIBLE)
                if (
                    solver_plan.status not in accepted_solver_statuses
                    or not solver_plan.candidate_plans
                ):
                    return unknown_certificate(
                        solver_plan,
                        [
                            *solver_plan.diagnostics,
                            "certified portfolio requires accepted solver candidates",
                        ],
                        assignment=last_rejected_assignment,
                        candidate_plan_id=last_rejected_plan_id,
                        result=last_rejection,
                    )

                eligible_occurrence_ids = {
                    target.occurrence_id
                    for target in prepared.targets
                    if target.eligible_for_atomic_update
                }
                restart_after_learned_cut = False
                for candidate in solver_plan.candidate_plans:
                    if candidate.status not in accepted_solver_statuses:
                        return unknown_certificate(
                            solver_plan,
                            [
                                *solver_plan.diagnostics,
                                "candidate "
                                f"{candidate.candidate_plan_id!r} has an unaccepted solver status",
                            ],
                            assignment=candidate.selected_candidate_versions,
                            candidate_plan_id=candidate.candidate_plan_id,
                        )
                    candidate_solver_plan = solver_plan.model_copy(
                        update={"selected_plan": candidate}
                    )
                    plan = _project_prepared_portfolio_plan(
                        prepared, candidate_solver_plan, portfolio_iteration=portfolio_iteration
                    )
                    selected_plan = plan.solver_plan.selected_plan
                    if selected_plan is None:
                        return unknown_certificate(
                            candidate_solver_plan,
                            ["prepared portfolio projection lost a candidate assignment"],
                            assignment=candidate.selected_candidate_versions,
                            candidate_plan_id=candidate.candidate_plan_id,
                        )
                    assignment = _read_candidate_assignment(selected_plan)
                    if set(assignment) != eligible_occurrence_ids:
                        return unknown_certificate(
                            candidate_solver_plan,
                            ["selected candidate plan does not contain a complete assignment"],
                            assignment=assignment,
                            candidate_plan_id=selected_plan.candidate_plan_id,
                        )

                    metrics["candidate_attempts"] += 1
                    result = _certify_assignment(
                        sandbox,
                        workspace_volume,
                        sandbox_factory,
                        prepared,
                        selected_plan,
                        assignment,
                        runtime,
                        archive_path,
                        scratch_root,
                        metrics,
                        deadline,
                    )
                    if result.status == PackageResolutionStatus.REJECTED:
                        assignment_digest = _digest(dict(sorted(assignment.items())))
                        conflict = result.rejection_conflict
                        if conflict is None or conflict.assignment_digest != assignment_digest:
                            return unknown_certificate(
                                candidate_solver_plan,
                                ["npm resolver rejection lacked matching typed evidence"],
                                assignment=assignment,
                                candidate_plan_id=selected_plan.candidate_plan_id,
                                result=result,
                            )
                        if assignment_digest in rejected_digests:
                            return unknown_certificate(
                                candidate_solver_plan,
                                [
                                    "solver repeated an assignment already rejected by npm resolution"
                                ],
                                assignment=assignment,
                                candidate_plan_id=selected_plan.candidate_plan_id,
                                result=result,
                            )
                        if any(
                            literal.variable_id not in eligible_occurrence_ids
                            or assignment.get(literal.variable_id) != literal.version
                            for literal in conflict.literals
                        ):
                            return unknown_certificate(
                                candidate_solver_plan,
                                ["resolver conflict did not match task-backed candidate literals"],
                                assignment=assignment,
                                candidate_plan_id=selected_plan.candidate_plan_id,
                                result=result,
                            )
                        rejection_counts = metrics["rejection_counts_by_reason"]
                        reason_key = conflict.reason_code.value
                        rejection_counts[reason_key] = rejection_counts.get(reason_key, 0) + 1
                        rejected_digests.append(assignment_digest)
                        rejection_conflicts.append(conflict)
                        last_rejection = result
                        last_rejected_assignment = dict(assignment)
                        last_rejected_plan_id = selected_plan.candidate_plan_id
                        last_rejection_summary = conflict.summary
                        if conflict.cut_kind == SolverCandidateCutKind.EXACT_ASSIGNMENT:
                            conflict_assignment = {
                                literal.variable_id: literal.version
                                for literal in conflict.literals
                            }
                            if conflict_assignment != assignment:
                                return unknown_certificate(
                                    candidate_solver_plan,
                                    [
                                        "exact resolver conflict did not name the complete assignment"
                                    ],
                                    assignment=assignment,
                                    candidate_plan_id=selected_plan.candidate_plan_id,
                                    result=result,
                                )
                            rejected_assignments.append(dict(sorted(assignment.items())))
                            continue

                        conflict_key = tuple(
                            (literal.variable_id, literal.version) for literal in conflict.literals
                        )
                        if conflict_key in generalized_conflicts:
                            return unknown_certificate(
                                candidate_solver_plan,
                                [
                                    "resolver repeated a conjunction already excluded by a learned cut"
                                ],
                                assignment=assignment,
                                candidate_plan_id=selected_plan.candidate_plan_id,
                                result=result,
                            )
                        generalized_conflicts[conflict_key] = conflict
                        restart_after_learned_cut = True
                        break

                    if result.status == PackageResolutionStatus.UNKNOWN:
                        return unknown_certificate(
                            candidate_solver_plan,
                            result.diagnostics,
                            assignment=assignment,
                            candidate_plan_id=selected_plan.candidate_plan_id,
                            result=result,
                        )

                    if _remaining_seconds(deadline) < 1.0:
                        raise _CertificationUnknown(
                            "certification deadline expired before final host fingerprint check"
                        )
                    final_host_snapshot = load_npm_graph_snapshot(root)
                    if final_host_snapshot.repository_fingerprint != host_fingerprint:
                        raise _CertificationUnknown(
                            "host repository changed while package resolution was being certified"
                        )
                    certificate = make_certificate(
                        plan,
                        prepared,
                        runtime,
                        status=PackageResolutionStatus.CERTIFIED,
                        assignment=assignment,
                        result=result,
                        candidate_plan_id=selected_plan.candidate_plan_id,
                        rejected_assignment_digests=rejected_digests,
                        rejection_conflicts=rejection_conflicts,
                        diagnostics=prepared.diagnostics,
                    )
                    return _attach_resolution_certificate(plan, certificate)

                if restart_after_learned_cut:
                    continue
    except _CertificationUnknown as exc:
        metrics["unknown_count"] += 1
        last_rejection_summary = locals().get("last_rejection_summary")
        if last_rejection_summary:
            diagnostics = sorted(
                set(
                    [
                        *prepared.diagnostics,
                        str(exc),
                        f"last rejected candidate: {last_rejection_summary}",
                    ]
                )
            )
        else:
            diagnostics = sorted(set([*prepared.diagnostics, str(exc)]))
        unknown_solver = _unknown_solver_plan(prepared, diagnostics)
        plan = _project_prepared_portfolio_plan(
            prepared, unknown_solver, portfolio_iteration=portfolio_iteration
        )
        certificate = make_certificate(
            plan,
            prepared,
            runtime,
            status=PackageResolutionStatus.UNKNOWN,
            rejected_assignment_digests=locals().get("rejected_digests", ()),
            assignment=locals().get("last_rejected_assignment"),
            result=locals().get("last_rejection"),
            candidate_plan_id=locals().get("last_rejected_plan_id"),
            rejection_conflicts=locals().get("rejection_conflicts", ()),
            diagnostics=diagnostics,
        )
        return _attach_resolution_certificate(plan, certificate)
    except Exception as exc:  # noqa: BLE001 - the production boundary fails closed
        metrics["unknown_count"] += 1
        failure_message = f"unexpected certification failure: {exc}"
        last_rejection_summary = locals().get("last_rejection_summary")
        messages = [*prepared.diagnostics, failure_message]
        if last_rejection_summary:
            messages.append(f"last rejected candidate: {last_rejection_summary}")
        diagnostics = sorted(set(messages))
        unknown_solver = _unknown_solver_plan(prepared, diagnostics)
        plan = _project_prepared_portfolio_plan(
            prepared, unknown_solver, portfolio_iteration=portfolio_iteration
        )
        certificate = make_certificate(
            plan,
            prepared,
            runtime,
            status=PackageResolutionStatus.UNKNOWN,
            rejected_assignment_digests=locals().get("rejected_digests", ()),
            assignment=locals().get("last_rejected_assignment"),
            result=locals().get("last_rejection"),
            candidate_plan_id=locals().get("last_rejected_plan_id"),
            rejection_conflicts=locals().get("rejection_conflicts", ()),
            diagnostics=diagnostics,
        )
        return _attach_resolution_certificate(plan, certificate)


__all__ = [
    "_attach_resolution_certificate",
    "build_certified_portfolio_plan",
]
