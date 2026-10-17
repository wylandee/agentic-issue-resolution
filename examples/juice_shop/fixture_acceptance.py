"""Bounded evidence projection and verification for maintained Juice Shop fixtures."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from remediation_engine.orchestration.task_utils import effective_group_status, task_group_lineage
from remediation_engine.orchestration.trajectory_exporter import to_jsonable

_TASK_EVIDENCE_FIELDS = (
    "task_id",
    "task_revision",
    "current_attempt_id",
    "parent_task_id",
    "parent_group_id",
    "status",
    "strategy",
    "target_package_name",
    "selected_version",
)
_SNAPSHOT_EVIDENCE_FIELDS = (
    "attempt_id",
    "task_id",
    "task_revision",
    "attempt_number",
    "cluster_id",
    "dispatch_batch_id",
    "action_digest",
    "portfolio_plan_id",
    "plan_id",
    "target_package_name",
    "manifest_path",
    "selected_version",
    "allowed_target_versions",
    "dispatch_node",
    "instruction_digest",
    "state_revision",
)
_WORKER_EVIDENCE_FIELDS = (
    "attempt_id",
    "task_id",
    "task_revision",
    "cluster_id",
    "dispatch_batch_id",
    "action_digest",
    "status",
    "executed_versions",
    "changed_files",
    "instruction_digest",
)
_QA_ENVELOPE_FIELDS = (
    "attempt_id",
    "task_id",
    "task_revision",
    "portfolio_plan_id",
    "cluster_id",
    "dispatch_batch_id",
    "action_digest",
    "qa_policy_source",
)
_EVALUATION_FIELDS = (
    "task_id",
    "passed",
    "failure_category",
    "contract_error",
    "evidence_inconclusive",
)


def _as_mapping(value: Any) -> dict[str, Any]:
    """Return a model or mapping as a plain dictionary for bounded projection."""
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _select_fields(value: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    mapping = _as_mapping(value)
    return {field: mapping[field] for field in fields if field in mapping}


def _project_attempt_evidence(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Keep only task, attempt, verdict, and scan-coverage provenance.

    The projection deliberately omits instructions, messages, diagnostics,
    scanner output, source contents, and exception text.
    """
    task_queue = raw_state.get("task_queue") or {}
    snapshots = raw_state.get("attempt_snapshots_by_id") or {}
    workers = raw_state.get("worker_results_by_attempt") or {}
    qa_results = raw_state.get("qa_results_by_attempt") or {}
    evaluations = raw_state.get("qa_evaluations") or {}
    target_packages = {
        str(_as_mapping(task).get("target_package_name"))
        for task in task_queue.values()
        if _as_mapping(task).get("target_package_name")
    }

    payload: dict[str, Any] = {
        "run_id": raw_state.get("run_id"),
        "target_packages": raw_state.get("target_packages", []),
        "status": raw_state.get("status"),
        "task_queue": {
            str(key): _select_fields(value, _TASK_EVIDENCE_FIELDS)
            for key, value in task_queue.items()
        },
        "attempt_snapshots_by_id": {
            str(key): _select_fields(value, _SNAPSHOT_EVIDENCE_FIELDS)
            for key, value in snapshots.items()
        },
        "worker_results_by_attempt": {
            str(key): _select_fields(value, _WORKER_EVIDENCE_FIELDS)
            for key, value in workers.items()
        },
        "qa_results_by_attempt": {
            str(key): {
                **_select_fields(value, _QA_ENVELOPE_FIELDS),
                "evaluation": _select_fields(
                    _as_mapping(value).get("evaluation"), _EVALUATION_FIELDS
                ),
                "batch_scan_evidence": _project_scan_evidence(
                    _as_mapping(value).get("batch_scan_evidence"), target_packages
                ),
            }
            for key, value in qa_results.items()
        },
        "qa_evaluations": {
            str(key): _select_fields(value, _EVALUATION_FIELDS)
            for key, value in evaluations.items()
        },
        "delta_isolation_by_cluster": {
            str(key): _select_fields(
                value,
                (
                    "status",
                    "responsible_task_ids",
                    "candidate_restored",
                    "cluster_id",
                    "dispatch_batch_id",
                    "action_digest",
                    "source_portfolio_plan_id",
                    "attempt_ids_by_task",
                    "tested_subsets",
                    "executions",
                ),
            )
            for key, value in (raw_state.get("delta_isolation_by_cluster") or {}).items()
        },
        "portfolio_plan": _project_portfolio_plan(raw_state.get("portfolio_plan")),
        "portfolio_solver_plan": _project_solver_plan(raw_state.get("portfolio_solver_plan")),
        "setup_scan_evidence": _project_scan_evidence(
            raw_state.get("setup_scan_evidence"), target_packages
        ),
        "latest_incremental_scan_by_input": {
            str(key): _select_fields(
                value, ("source_input_path", "task_ids", "attempt_ids", "complete")
            )
            for key, value in (raw_state.get("latest_incremental_scan_by_input") or {}).items()
        },
        "latest_successful_scan_by_task": {
            str(key): _project_task_scan(value, target_packages)
            for key, value in (raw_state.get("latest_successful_scan_by_task") or {}).items()
        },
        "final_full_scan_result": _select_fields(
            raw_state.get("final_full_scan_result"),
            ("completed", "authoritative", "coverage_scope", "coverage_complete", "status"),
        ),
        "final_full_scan_enabled": raw_state.get("final_full_scan_enabled", False),
    }
    return to_jsonable(payload)


def _project_occurrences(value: Any, target_packages: set[str]) -> list[dict[str, Any]]:
    mapping = _as_mapping(value)
    return [
        _select_fields(
            occurrence,
            (
                "source_input_path",
                "package_name",
                "version",
                "lockfile_key",
                "occurrence_index",
                "found_identifiers",
            ),
        )
        for occurrence in mapping.get("covered_occurrences", [])
        if _as_mapping(occurrence).get("package_name") in target_packages
    ]


def _project_scan_evidence(value: Any, target_packages: set[str]) -> dict[str, Any]:
    mapping = _as_mapping(value)
    if not mapping:
        return {}
    return {
        **_select_fields(
            value,
            (
                "requested_scope",
                "effective_scope",
                "authoritative",
                "covered_task_ids",
                "closure_package_names",
                "closure_lockfile_keys",
                "covered_input_paths",
                "scan_performed",
                "complete",
            ),
        ),
        "covered_occurrences": _project_occurrences(mapping, target_packages),
    }


def _project_task_scan(value: Any, target_packages: set[str]) -> dict[str, Any]:
    mapping = _as_mapping(value)
    return {
        **_select_fields(
            mapping,
            (
                "task_id",
                "attempt_id",
                "complete",
                "closure_package_names",
                "closure_lockfile_keys",
                "covered_input_paths",
            ),
        ),
        "covered_occurrences": _project_occurrences(mapping, target_packages),
    }


def _normalized_relative_path(raw_path: Any) -> str:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"invalid repository-relative path: {raw_path!r}")
    normalized = raw_path.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe repository-relative path: {raw_path!r}")
    return path.as_posix()


def _patch_paths(patch_text: str) -> set[str]:
    paths: set[str] = set()
    for old_path, new_path in _patch_sections(patch_text):
        paths.update(path for path in (old_path, new_path) if path is not None)
    return paths


def _patch_sections(patch_text: str) -> list[tuple[str | None, str | None]]:
    sections: list[tuple[str | None, str | None]] = []
    lines = patch_text.splitlines()
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- ") or not lines[index + 1].startswith("+++ "):
            continue
        paths: list[str | None] = []
        for raw_path in (line[4:], lines[index + 1][4:]):
            raw_path = raw_path.split("\t", 1)[0].strip()
            if raw_path == "/dev/null":
                paths.append(None)
                continue
            prefix, separator, relative = raw_path.partition("/")
            if not separator or prefix not in {"a", "b"}:
                raise ValueError(f"patch contains an unexpected path header: {raw_path!r}")
            paths.append(_normalized_relative_path(relative))
        sections.append((paths[0], paths[1]))
    return sections


def _replay_patch(
    repo: Path, patch_path: Path, result: dict[str, Any]
) -> tuple[tempfile.TemporaryDirectory[str], set[str]]:
    repo = repo.expanduser().resolve(strict=True)
    patch_path = patch_path.expanduser().resolve(strict=True)
    patch_text = patch_path.read_text(encoding="utf-8")
    changed_paths = {_normalized_relative_path(path) for path in result.get("changed_files", [])}
    patch_paths = _patch_paths(patch_text)
    if not patch_paths or patch_paths != changed_paths:
        raise ValueError("result changed_files do not match the files represented by the patch")
    patch_executable = shutil.which("patch")
    if patch_executable is None:
        raise RuntimeError("the system patch utility is required for isolated patch replay")

    temporary = tempfile.TemporaryDirectory(prefix="juice-shop-fixture-replay-")
    replay_root = Path(temporary.name)
    try:
        source_paths = set(changed_paths)
        source_paths.update(
            lockfile.relative_to(repo).as_posix() for lockfile in _package_lock_paths(repo)
        )
        for relative in source_paths:
            source = repo / relative
            if not source.exists():
                continue
            resolved_source = source.resolve(strict=True)
            if not resolved_source.is_relative_to(repo):
                raise ValueError(f"changed file resolves outside the repository: {relative}")
            destination = replay_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolved_source, destination)
        applied = subprocess.run(
            [patch_executable, "--batch", "--forward", "-p1", "-i", str(patch_path)],
            cwd=replay_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if applied.returncode != 0:
            details = (applied.stderr or applied.stdout).strip()
            raise ValueError(
                "patch could not be replayed against a temporary repository copy"
                + (f": {details}" if details else "")
            )
        for old_path, new_path in _patch_sections(patch_text):
            relative = old_path or new_path
            if relative is None:
                raise ValueError("patch section does not identify a repository file")
            final_path = replay_root / relative
            if new_path is None:
                # The system patch utility updates a deletion to an empty
                # file but does not unlink it; finish the deletion represented
                # by the patch header after verifying the resulting content.
                if final_path.is_file() and final_path.stat().st_size == 0:
                    final_path.unlink()
                if final_path.exists():
                    raise ValueError(f"deleted file remains after patch replay: {relative}")
            elif not final_path.is_file():
                raise ValueError(f"patch replay omitted resulting file {relative}")
    except BaseException:
        temporary.cleanup()
        raise
    return temporary, changed_paths


def _package_lock_paths(root: Path) -> list[Path]:
    root = root.resolve()
    matches: list[Path] = []
    for current, directories, files in os.walk(root):
        directories[:] = [
            name for name in directories if name not in {".git", "node_modules", ".venv", "venv"}
        ]
        if "package-lock.json" in files:
            lockfile = Path(current) / "package-lock.json"
            if lockfile.is_symlink() and not lockfile.resolve().is_relative_to(root):
                raise ValueError(f"package lockfile resolves outside the repository: {lockfile}")
            if lockfile.is_file():
                matches.append(lockfile)
    return matches


def _package_occurrences(root: Path, package_name: str) -> dict[tuple[str, str], str]:
    root = root.resolve(strict=True)
    occurrences: dict[tuple[str, str], str] = {}
    suffix = f"/node_modules/{package_name}"
    exact = f"node_modules/{package_name}"
    for lockfile in _package_lock_paths(root):
        relative = lockfile.relative_to(root).as_posix()
        try:
            document = json.loads(lockfile.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read package lockfile {relative}") from exc
        package_entries = document.get("packages")
        if not isinstance(package_entries, dict):
            continue
        for key, metadata in package_entries.items():
            if not isinstance(key, str) or not isinstance(metadata, dict):
                continue
            normalized_key = key.strip("/")
            if normalized_key == exact or normalized_key.endswith(suffix):
                occurrences[(relative, normalized_key)] = str(metadata.get("version", ""))
    return occurrences


def _status(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value", value.get("status", ""))).lower()
    return str(value or "").lower()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_result_binding(
    result: dict[str, Any], evidence: dict[str, Any], patch_text: str
) -> None:
    binding = evidence.get("result_binding") or {}
    _assert(binding.get("status") == result.get("status"), "result/evidence statuses differ")
    _assert(
        binding.get("changed_files") == sorted(result.get("changed_files") or []),
        "result/evidence changed-file lists differ",
    )
    _assert(
        binding.get("diff_sha256") == hashlib.sha256(patch_text.encode("utf-8")).hexdigest(),
        "result/evidence patch digests differ",
    )
    _assert(
        binding.get("trajectory_path") == result.get("trajectory_path"),
        "result/evidence trajectory paths differ",
    )
    if result.get("run_id") is not None and evidence.get("run_id") is not None:
        _assert(result["run_id"] == evidence["run_id"], "result/evidence run IDs differ")


def _load_acceptance_inputs(
    repo: Path,
    result_path: Path,
    evidence_path: Path,
    patch_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tempfile.TemporaryDirectory[str],
    set[str],
]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(evidence, dict):
        raise ValueError("result and acceptance evidence must contain JSON objects")
    _assert(
        result.get("status") in {"completed", "completed_with_errors"},
        f"fixture runner did not complete: status={result.get('status')!r}",
    )
    patch_text = patch_path.read_text(encoding="utf-8")
    _assert(result.get("diff", "") == patch_text, "saved patch differs from typed result diff")
    _assert_result_binding(result, evidence, patch_text)
    if not patch_text.strip() and not result.get("changed_files"):
        raise ValueError("fixture completed with no accepted patch")
    replay_temporary, changed_paths = _replay_patch(repo, patch_path, result)
    return result, evidence, replay_temporary, changed_paths


def _correlated_records(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    snapshots = evidence.get("attempt_snapshots_by_id") or {}
    workers = evidence.get("worker_results_by_attempt") or {}
    qa_results = evidence.get("qa_results_by_attempt") or {}
    for map_name, records in (("worker", workers), ("QA", qa_results)):
        for attempt_key, record in records.items():
            if record.get("attempt_id") != attempt_key:
                raise ValueError(f"{map_name} result is stored under a foreign attempt key")
    correlated: dict[str, dict[str, Any]] = {}
    for attempt_key, qa_result in qa_results.items():
        snapshot = snapshots.get(attempt_key)
        worker = workers.get(attempt_key)
        if snapshot is None or worker is None:
            continue
        task_id = snapshot.get("task_id")
        evaluation = qa_result.get("evaluation") or {}
        _assert(attempt_key == snapshot.get("attempt_id"), "snapshot map key is miskeyed")
        _assert(
            task_id == worker.get("task_id") == qa_result.get("task_id"), "attempt task IDs differ"
        )
        _assert(task_id == evaluation.get("task_id"), "QA evaluation task ID differs")
        _assert(
            snapshot.get("task_revision")
            == worker.get("task_revision")
            == qa_result.get("task_revision"),
            "attempt revisions differ across immutable snapshot and envelopes",
        )
        for field in ("cluster_id", "dispatch_batch_id", "action_digest"):
            expected = snapshot.get(field)
            _assert(worker.get(field) == expected, f"worker {field} provenance differs")
            _assert(qa_result.get(field) == expected, f"QA {field} provenance differs")
        if qa_result.get("portfolio_plan_id") is not None:
            _assert(
                qa_result["portfolio_plan_id"] == snapshot.get("portfolio_plan_id"),
                "QA portfolio plan provenance differs from committed snapshot",
            )
        correlated[attempt_key] = {
            "snapshot": snapshot,
            "worker": worker,
            "qa": qa_result,
            "evaluation": evaluation,
        }
    return correlated


def _accepted_attempts_by_task(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    correlated = _correlated_records(evidence)
    accepted: dict[str, dict[str, Any]] = {}
    for _attempt_id, record in correlated.items():
        evaluation = record["evaluation"]
        if (
            evaluation.get("passed") is not True
            or _status(record["worker"].get("status")) != "success"
        ):
            continue
        if record["qa"].get("qa_policy_source") != "attempt_snapshot":
            raise ValueError("passing QA record lacks committed snapshot policy provenance")
        task_id = record["snapshot"]["task_id"]
        current = accepted.get(task_id)
        if current is None or int(record["snapshot"].get("attempt_number", 0)) >= int(
            current["snapshot"].get("attempt_number", 0)
        ):
            accepted[task_id] = record
    return accepted


def _first_attempts_by_package(
    evidence: dict[str, Any], task_ids: dict[str, list[str]]
) -> dict[str, list[dict[str, Any]]]:
    snapshots = evidence.get("attempt_snapshots_by_id") or {}
    first: dict[str, list[dict[str, Any]]] = {}
    for package_name, root_task_ids in task_ids.items():
        first[package_name] = []
        for task_id in root_task_ids:
            task_snapshots = [
                snapshot for snapshot in snapshots.values() if snapshot.get("task_id") == task_id
            ]
            if not task_snapshots:
                raise ValueError(f"no committed attempt snapshot for package {package_name}")
            first[package_name].append(
                min(task_snapshots, key=lambda item: int(item.get("attempt_number", 0)))
            )
    return first


def _task_ids_by_package(evidence: dict[str, Any], packages: set[str]) -> dict[str, list[str]]:
    roots: dict[str, list[str]] = defaultdict(list)
    for task_id, task in (evidence.get("task_queue") or {}).items():
        package_name = task.get("target_package_name")
        if package_name not in packages or task.get("parent_task_id") is not None:
            continue
        roots[package_name].append(task_id)
    missing = sorted(packages - set(roots))
    if missing:
        raise ValueError("fixture has no terminal task for packages: " + ", ".join(missing))
    return {package: sorted(task_ids) for package, task_ids in roots.items()}


def _lineage_task_ids_by_package(
    evidence: dict[str, Any], root_task_ids: dict[str, list[str]]
) -> dict[str, list[list[str]]]:
    task_queue = evidence.get("task_queue") or {}
    lineages: dict[str, list[list[str]]] = {}
    for package, package_roots in root_task_ids.items():
        package_lineages = []
        for root_task_id in package_roots:
            group_id = task_queue[root_task_id].get("parent_group_id")
            lineage = task_group_lineage(task_queue, group_id)
            task_ids = [str(task.get("task_id")) for task in lineage]
            _assert(
                root_task_id in task_ids, f"task lineage for {package} omitted its initial root"
            )
            package_lineages.append(task_ids)
        lineages[package] = package_lineages
    return lineages


def _effective_package_statuses(
    evidence: dict[str, Any], root_task_ids: dict[str, list[str]]
) -> dict[str, list[str]]:
    task_queue = evidence.get("task_queue") or {}
    return {
        package: [
            effective_group_status(task_queue, task_queue[task_id].get("parent_group_id"))
            for task_id in package_roots
        ]
        for package, package_roots in root_task_ids.items()
    }


def _accepted_attempt_for_lineage(
    accepted_by_task: dict[str, dict[str, Any]], lineage_task_ids: list[str], package: str
) -> dict[str, Any]:
    records = [
        accepted_by_task[task_id] for task_id in lineage_task_ids if task_id in accepted_by_task
    ]
    if not records:
        raise ValueError(f"{package} has no correlated passing attempt in its task lineage")
    return max(records, key=lambda record: int(record["snapshot"].get("attempt_number", 0)))


def _assert_singleton_attempts(
    evidence: dict[str, Any],
    task_ids: dict[str, list[str]],
    first_attempts: dict[str, list[dict[str, Any]]],
    atomic_pair_batch_id: str,
) -> None:
    """Prove got and esbuild were dispatched independently of the atomic pair."""
    solver_batches = (evidence.get("portfolio_solver_plan") or {}).get("batches", [])
    all_batch_ids: list[str] = []
    for package in ("got", "esbuild"):
        _assert(bool(first_attempts[package]), f"{package} has no initial attempts")
        for snapshot in first_attempts[package]:
            task_id = str(snapshot.get("task_id", ""))
            _assert(
                snapshot.get("cluster_id") is None,
                f"{package} root {task_id} was not dispatched as a singleton cluster",
            )
            batch_id = snapshot.get("dispatch_batch_id")
            _assert(
                batch_id is None or batch_id != atomic_pair_batch_id,
                f"{package} root {task_id} shared the initial base64url/jws atomic batch",
            )
            if batch_id is None:
                continue
            all_batch_ids.append(batch_id)
            plan_batch = next(
                (item for item in solver_batches if item.get("batch_id") == batch_id), None
            )
            _assert(
                plan_batch is not None,
                f"{package} root {task_id} singleton batch lacks plan evidence",
            )
            _assert(
                set(plan_batch.get("task_ids", [])) == {task_id},
                f"{package} root {task_id} plan batch contains multiple tasks",
            )
    _assert(len(all_batch_ids) == len(set(all_batch_ids)), "singleton roots shared a batch")


def _require_complete_scan(evidence: dict[str, Any], task_id: str, attempt_id: str) -> None:
    scans = evidence.get("latest_successful_scan_by_task") or {}
    record = scans.get(task_id)
    if record is not None:
        _assert(
            record.get("attempt_id") == attempt_id, f"task {task_id} scan is from another attempt"
        )
        _assert(record.get("complete") is True, f"task {task_id} accepted scan is incomplete")
        return
    qa = (evidence.get("qa_results_by_attempt") or {}).get(attempt_id, {})
    batch_scan = qa.get("batch_scan_evidence") or {}
    _assert(bool(batch_scan), f"task {task_id} has no persisted accepted scan evidence")
    _assert(batch_scan.get("complete") is True, f"task {task_id} batch scan is incomplete")
    _assert(
        task_id in (batch_scan.get("covered_task_ids") or []),
        f"task {task_id} is absent from scan coverage",
    )


def _assert_patch_target_occurrences(
    original_repo: Path,
    replay_repo: Path,
    packages: set[str],
    rejected_package: str | None = None,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for package in sorted(packages):
        before = _package_occurrences(original_repo, package)
        after = _package_occurrences(replay_repo, package)
        _assert(bool(before), f"baseline has no physical lockfile occurrence for {package}")
        if package == rejected_package:
            _assert(before == after, f"rejected package {package} changed in the final patch")
            summary[package] = {"baseline": len(before), "final": len(after), "changed": 0}
            continue
        changed = sum(before.get(key) != after.get(key) for key in before.keys() | after.keys())
        _assert(changed > 0, f"accepted package {package} has no physical lockfile mutation")
        summary[package] = {"baseline": len(before), "final": len(after), "changed": changed}
    return summary


def _assert_scanned_target_occurrences(
    evidence: dict[str, Any],
    package: str,
    task_id: str,
    accepted_attempt_id: str,
    replay_repo: Path,
) -> None:
    record = (evidence.get("latest_successful_scan_by_task") or {}).get(task_id)
    if not record:
        record = (
            (evidence.get("qa_results_by_attempt") or {})
            .get(accepted_attempt_id, {})
            .get("batch_scan_evidence")
        )
        _assert(bool(record), f"accepted scan for {package} has no persisted occurrence evidence")
        _assert(
            task_id in (record.get("covered_task_ids") or []),
            f"accepted scan for {package} does not cover task {task_id}",
        )
    else:
        _assert(
            record.get("attempt_id") == accepted_attempt_id,
            f"accepted scan for {package} is from another attempt",
        )
    _assert(record.get("complete") is True, f"accepted scan for {package} is incomplete")
    covered_input_paths = record.get("covered_input_paths")
    _assert(
        isinstance(covered_input_paths, list) and bool(covered_input_paths),
        f"accepted scan for {package} has no covered input paths",
    )
    covered_paths = {_normalized_relative_path(path) for path in covered_input_paths}
    occurrences = [
        occurrence
        for occurrence in record.get("covered_occurrences", [])
        if occurrence.get("package_name") == package
    ]
    _assert(bool(occurrences), f"accepted scan did not cover physical occurrences of {package}")
    _assert(
        all(not occurrence.get("found_identifiers") for occurrence in occurrences),
        f"accepted scan still reports findings for {package}",
    )
    final_occurrences = _package_occurrences(replay_repo, package)
    scanned = {
        (
            _normalized_relative_path(occurrence.get("source_input_path")),
            str(occurrence.get("lockfile_key", "")),
            str(occurrence.get("version", "")),
        )
        for occurrence in occurrences
    }
    _assert(
        all(path in covered_paths for path, _, _ in scanned),
        f"accepted scan for {package} reports occurrences outside its covered inputs",
    )
    actual = {
        (path, lockfile_key, version)
        for (path, lockfile_key), version in final_occurrences.items()
        if path in covered_paths
    }
    _assert(
        scanned == actual,
        f"accepted scan occurrence set for {package} does not match the replayed physical lockfile state within its covered inputs",
    )


def verify_shared_name_fixture(
    repo: Path,
    result_path: Path,
    evidence_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    """Validate the live shared-name fixture's task, batch, scan, and patch evidence."""
    _, evidence, replay_temporary, changed_paths = _load_acceptance_inputs(
        repo, result_path, evidence_path, patch_path
    )
    try:
        return _verify_shared_name_fixture_loaded(repo, evidence, replay_temporary, changed_paths)
    finally:
        replay_temporary.cleanup()


def _verify_shared_name_fixture_loaded(
    repo: Path,
    evidence: dict[str, Any],
    replay_temporary: tempfile.TemporaryDirectory[str],
    changed_paths: set[str],
) -> dict[str, Any]:
    replay_root = Path(replay_temporary.name)
    expected = {
        "@angular/animations",
        "@angular/common",
        "@angular/compiler",
        "@angular/compiler-cli",
        "@angular/core",
        "@angular/forms",
        "@angular/platform-browser",
        "@angular/platform-browser-dynamic",
        "@angular/router",
    }
    _assert(
        set(evidence.get("target_packages", []))
        == {"@angular/common", "@angular/compiler", "@angular/core"},
        "shared-name run did not retain the three requested target packages",
    )
    tasks = _task_ids_by_package(evidence, expected)
    lineages = _lineage_task_ids_by_package(evidence, tasks)
    accepted_by_task = _accepted_attempts_by_task(evidence)
    effective_statuses = _effective_package_statuses(evidence, tasks)
    accepted = {
        package: [
            _accepted_attempt_for_lineage(accepted_by_task, lineage, package)
            for lineage in lineages[package]
        ]
        for package in expected
    }
    for package in tasks:
        _assert(
            bool(effective_statuses[package])
            and all(status == "qa_passed" for status in effective_statuses[package]),
            f"Angular task lineage {package} is not effectively QA-passed",
        )
        for attempt in accepted[package]:
            task_id = attempt["snapshot"]["task_id"]
            _require_complete_scan(evidence, task_id, attempt["snapshot"]["attempt_id"])
            _assert_scanned_target_occurrences(
                evidence, package, task_id, attempt["snapshot"]["attempt_id"], replay_root
            )

    accepted_records = [record for package in tasks for record in accepted[package]]
    batch_ids = {record["snapshot"].get("dispatch_batch_id") for record in accepted_records}
    cluster_ids = {record["snapshot"].get("cluster_id") for record in accepted_records}
    action_digests = {record["snapshot"].get("action_digest") for record in accepted_records}
    _assert(
        len(batch_ids) == 1 and None not in batch_ids,
        "Angular tasks were not accepted in one batch",
    )
    _assert(
        len(cluster_ids) == 1 and None not in cluster_ids,
        "Angular tasks do not share an atomic cluster",
    )
    _assert(
        len(action_digests) == 1 and None not in action_digests, "Angular action digests differ"
    )
    batch_id = next(iter(batch_ids))
    cluster_id = next(iter(cluster_ids))
    accepted_ids = {record["snapshot"]["task_id"] for record in accepted_records}
    plan = evidence.get("portfolio_plan") or {}
    cluster = next(
        (item for item in plan.get("clusters", []) if item.get("cluster_id") == cluster_id),
        None,
    )
    if cluster is not None:
        _assert(cluster.get("atomic") is True, "Angular portfolio cluster is not atomic")
        _assert(
            cluster.get("dispatchable") is True, "Angular portfolio cluster is not dispatchable"
        )
        _assert(
            set(cluster.get("task_ids", [])) == accepted_ids, "Angular cluster membership differs"
        )
    else:
        solver_batches = (evidence.get("portfolio_solver_plan") or {}).get("batches", [])
        solver_batch = next(
            (item for item in solver_batches if item.get("batch_id") == batch_id), None
        )
        _assert(solver_batch is not None, "no persisted plan proves the Angular batch is atomic")
        _assert(solver_batch.get("atomic") is True, "Angular solver batch is not atomic")
        _assert(
            solver_batch.get("dispatchable") is True, "Angular solver batch is not dispatchable"
        )
        _assert(
            set(solver_batch.get("task_ids", [])) == accepted_ids,
            "Angular solver batch membership differs",
        )

    occurrence_summary = _assert_patch_target_occurrences(repo.resolve(), replay_root, expected)
    _assert(bool(changed_paths), "shared-name patch has no changed files")
    return {
        "fixture": "shared-name-angular",
        "passed": True,
        "run_id": evidence.get("run_id"),
        "packages": sorted(expected),
        "task_count": len(tasks),
        "dispatch_batch_id": batch_id,
        "cluster_id": cluster_id,
        "scan_verified_task_count": len(tasks),
        "changed_files": sorted(changed_paths),
        "lockfile_occurrences": occurrence_summary,
    }


def verify_four_package_fixture(
    repo: Path,
    result_path: Path,
    evidence_path: Path,
    patch_path: Path,
) -> dict[str, Any]:
    """Validate the live four-package batch, jws blame, rollback, and final patch."""
    _, evidence, replay_temporary, changed_paths = _load_acceptance_inputs(
        repo, result_path, evidence_path, patch_path
    )
    try:
        return _verify_four_package_fixture_loaded(repo, evidence, replay_temporary, changed_paths)
    finally:
        replay_temporary.cleanup()


def _verify_four_package_fixture_loaded(
    repo: Path,
    evidence: dict[str, Any],
    replay_temporary: tempfile.TemporaryDirectory[str],
    changed_paths: set[str],
) -> dict[str, Any]:
    replay_root = Path(replay_temporary.name)
    expected = {"base64url", "got", "esbuild", "jws"}
    successful = {"base64url", "got", "esbuild"}
    rejected = "jws"
    _assert(
        set(evidence.get("target_packages", [])) == expected, "four-package target scope differs"
    )
    tasks = _task_ids_by_package(evidence, expected)
    lineages = _lineage_task_ids_by_package(evidence, tasks)
    effective_statuses = _effective_package_statuses(evidence, tasks)
    accepted_by_task = _accepted_attempts_by_task(evidence)
    for package in successful:
        _assert(
            bool(effective_statuses[package])
            and all(status == "qa_passed" for status in effective_statuses[package]),
            f"{package} task lineage is not effectively QA-passed",
        )
        for lineage in lineages[package]:
            attempt = _accepted_attempt_for_lineage(accepted_by_task, lineage, package)
            task_id = attempt["snapshot"]["task_id"]
            _require_complete_scan(evidence, task_id, attempt["snapshot"]["attempt_id"])
            _assert_scanned_target_occurrences(
                evidence, package, task_id, attempt["snapshot"]["attempt_id"], replay_root
            )
    _assert(
        len(tasks["base64url"]) == len(tasks[rejected]) == 1,
        "initial pair must have one root per package",
    )
    rejected_task_id = tasks[rejected][0]
    _assert(
        bool(effective_statuses[rejected])
        and all(status in {"unfixable", "inconclusive"} for status in effective_statuses[rejected]),
        "jws task lineage did not finish in a terminal unresolved state",
    )

    first = _first_attempts_by_package(evidence, tasks)
    pair_batch = first["base64url"][0].get("dispatch_batch_id")
    _assert(pair_batch, "initial base64url/jws batch ID is missing")
    _assert(
        first[rejected][0].get("dispatch_batch_id") == pair_batch,
        "base64url and jws did not start in the same atomic batch",
    )
    _assert(
        first["base64url"][0].get("cluster_id") == first[rejected][0].get("cluster_id"),
        "base64url and jws initial cluster IDs differ",
    )
    _assert_singleton_attempts(evidence, tasks, first, pair_batch)

    correlated = _correlated_records(evidence)
    for package in ("base64url", rejected):
        task_id = tasks[package][0]
        attempt_id = first[package][0].get("attempt_id")
        qa = correlated.get(attempt_id)
        _assert(qa is not None, f"initial {package} attempt lacks worker/QA evidence")
        _assert(qa["evaluation"].get("passed") is False, f"initial {package} QA did not fail")

    blamed_records = [
        record
        for record in (evidence.get("delta_isolation_by_cluster") or {}).values()
        if record.get("status") == "IDENTIFIED"
        and record.get("candidate_restored") is True
        and record.get("responsible_task_ids") == [rejected_task_id]
    ]
    _assert(bool(blamed_records), "delta isolation did not restore and blame only jws")
    blamed_record = blamed_records[-1]
    _assert(
        (blamed_record.get("attempt_ids_by_task") or {}).get(rejected_task_id)
        == first[rejected][0].get("attempt_id"),
        "jws blame does not refer to its initial failed attempt",
    )

    occurrence_summary = _assert_patch_target_occurrences(
        repo.resolve(), replay_root, expected, rejected_package=rejected
    )
    _assert(bool(changed_paths), "four-package patch has no changed files")
    return {
        "fixture": "four-package-delta-isolation",
        "passed": True,
        "run_id": evidence.get("run_id"),
        "initial_atomic_batch_id": pair_batch,
        "initial_pair_task_ids": [tasks["base64url"], rejected_task_id],
        "singleton_batch_ids": {
            "got": [item.get("dispatch_batch_id") for item in first["got"]],
            "esbuild": [item.get("dispatch_batch_id") for item in first["esbuild"]],
        },
        "blamed_task_id": rejected_task_id,
        "candidate_restored": True,
        "remediated_packages": sorted(successful),
        "rejected_package": rejected,
        "changed_files": sorted(changed_paths),
        "lockfile_occurrences": occurrence_summary,
    }


def _project_portfolio_plan(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    if not mapping:
        return {}
    return {
        **_select_fields(mapping, ("plan_id", "portfolio_plan_id", "task_ids")),
        "clusters": [
            _select_fields(cluster, ("cluster_id", "task_ids", "atomic", "dispatchable"))
            for cluster in mapping.get("clusters", [])
        ],
    }


def _project_solver_plan(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    selected = _as_mapping(mapping.get("selected_plan"))
    if not selected:
        return _select_fields(mapping, ("status",)) if mapping else {}
    batches = []
    for batch in selected.get("batches", []):
        batches.append(
            {
                **_select_fields(batch, ("batch_id", "task_ids", "atomic", "dispatchable")),
                "mutations": [
                    _select_fields(
                        mutation,
                        (
                            "task_id",
                            "occurrence_id",
                            "package_name",
                            "manifest_path",
                            "target_version",
                            "dependency_type",
                        ),
                    )
                    for mutation in _as_mapping(batch).get("mutations", [])
                ],
            }
        )
    return {**_select_fields(mapping, ("status",)), "batches": batches}


def write_fixture_evidence(
    raw_state: dict[str, Any],
    path: Path,
    *,
    status: str,
    changed_files: list[str],
    diff: str,
    trajectory_path: str | None,
) -> None:
    """Write bounded, secret-redacted state evidence for fixture verification."""
    payload = _project_attempt_evidence(raw_state)
    payload["result_binding"] = {
        "status": status,
        "changed_files": sorted(changed_files),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "trajectory_path": trajectory_path,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
