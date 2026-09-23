"""Deterministic CP-SAT portfolio assignment for occurrence-aware npm targets.

This module is intentionally a pure solver boundary.  It consumes immutable
solver contracts and an explicit settings object; registry access, graph-state
access, and worker execution belong to the portfolio orchestration layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langsmith import traceable

from remediation_engine.contracts.solver_models import (
    SolverCandidatePlan,
    SolverFindingRequirement,
    SolverPeerConstraint,
    SolverRemediationPlan,
    SolverStatistics,
    SolverStatus,
    SolverSubgraph,
    SolverTarget,
    SolverTaskDecision,
    SolverVersionCandidate,
)

_MAX_DIAGNOSTICS = 64
_MAX_DIAGNOSTIC_LENGTH = 500


def _trace_field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from a contract object or its serialized mapping."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _trace_solver_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded, non-sensitive summary for the solver child span."""
    subgraph = inputs.get("subgraph")
    targets = list(_trace_field(subgraph, "targets", ()) or ())
    findings = list(_trace_field(subgraph, "findings", ()) or ())
    edges = list(_trace_field(subgraph, "edges", ()) or ())
    peer_constraints = list(_trace_field(subgraph, "peer_constraints", ()) or ())
    candidate_domains = inputs.get("candidate_domains") or {}
    domain_counts = (
        {str(key): len(values or ()) for key, values in candidate_domains.items()}
        if isinstance(candidate_domains, Mapping)
        else {}
    )
    settings = inputs.get("settings")
    setting_names = (
        "solver_timeout_seconds",
        "solver_top_k",
        "solver_num_search_workers",
        "solver_accept_feasible",
        "solver_max_candidates_per_target",
        "solver_max_model_variables",
    )
    return {
        "target_count": len(targets),
        "eligible_target_count": sum(
            bool(_trace_field(target, "eligible_for_atomic_update", False)) for target in targets
        ),
        "finding_count": len(findings),
        "edge_count": len(edges),
        "peer_constraint_count": len(peer_constraints),
        "subgraph_valid": bool(_trace_field(subgraph, "valid", True)),
        "candidate_domain_count": len(domain_counts),
        "candidate_total_count": sum(domain_counts.values()),
        "candidate_counts": domain_counts,
        "settings": {
            name: _trace_field(settings, name)
            for name in setting_names
            if _trace_field(settings, name) is not None
        },
    }


def _trace_solver_outputs(output: Any) -> dict[str, Any]:
    """Return bounded solver results and raw execution statistics."""
    if not hasattr(output, "status"):
        return {"result_type": type(output).__name__}
    status = getattr(output.status, "value", output.status)
    selected = getattr(output, "selected_plan", None)
    statistics = getattr(output, "solver_statistics", None)
    return {
        "status": str(status),
        "candidate_plan_count": len(getattr(output, "candidate_plans", ()) or ()),
        "selected_plan_id": getattr(selected, "candidate_plan_id", None),
        "unresolved_finding_count": len(getattr(output, "unresolved_finding_ids", ()) or ()),
        "task_revision_count": len(getattr(output, "task_revisions", {}) or {}),
        "diagnostics": list(getattr(output, "diagnostics", ()) or ()),
        "solver_statistics": (
            statistics.model_dump(mode="json") if statistics is not None else None
        ),
    }


def _digest(value: Any) -> str:
    """Return a stable digest for JSON-compatible contract data."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _diagnostic(values: list[str], message: str) -> None:
    """Append one bounded diagnostic while retaining deterministic order."""
    message = str(message).strip()
    if not message:
        return
    message = message[:_MAX_DIAGNOSTIC_LENGTH]
    if message not in values and len(values) < _MAX_DIAGNOSTICS:
        values.append(message)


def _candidate_key(candidate: SolverVersionCandidate) -> tuple[Any, ...]:
    """Sort candidates by semantic version and then by stable source/version."""
    key = tuple(candidate.semver_key)
    if not key:
        parts: list[int] = []
        for token in candidate.version.lstrip("vV").split(".")[:3]:
            digits = "".join(char for char in token if char.isdigit())
            parts.append(int(digits or 0))
        key = tuple(parts)
    return key, candidate.version, candidate.source


def _version_key(value: str | None) -> tuple[int, ...] | None:
    """Parse the stable numeric portion of an npm version."""
    if not value:
        return None
    try:
        from semantic_version import Version

        parsed = Version.coerce(value.strip().lstrip("vV"))
        return (parsed.major, parsed.minor, parsed.patch)
    except (ImportError, TypeError, ValueError):
        parts = value.strip().lstrip("vV").split(".")
        if len(parts) < 3 or any(not part.isdigit() for part in parts[:3]):
            return None
        return tuple(int(part) for part in parts[:3])


def _at_least(version: str, floor: str | None) -> bool:
    """Return whether a candidate version is at least a security floor."""
    if not floor:
        return True
    left = _version_key(version)
    right = _version_key(floor)
    return left is not None and right is not None and left >= right


def _distance_above(version: str, floor: str | None) -> int:
    """Return a bounded numeric distance above a floor for objective ranking."""
    if not floor:
        return 0
    current = _version_key(version)
    minimum = _version_key(floor)
    if current is None or minimum is None:
        return 1_000_000_000
    return min(
        1_000_000_000,
        max(
            0,
            (current[0] - minimum[0]) * 1_000_000_000
            + (current[1] - minimum[1]) * 1_000_000
            + (current[2] - minimum[2]),
        ),
    )


def _range_matches(version_range: str | None, version: str) -> bool | None:
    """Check an npm range without turning malformed ranges into compatibility."""
    if not version_range or version_range.strip() in {"*", "latest"}:
        return True
    try:
        from remediation_engine.tools.npm_graph import npm_range_contains

        return npm_range_contains(version_range, version)
    except (ImportError, TypeError, ValueError):
        return None


def _setting(settings: Any, name: str, default: Any) -> Any:
    """Read a setting attribute without consulting the process environment."""
    value = getattr(settings, name, default)
    return default if value is None else value


def _objective_weights(
    finding_list: Sequence[SolverFindingRequirement],
    eligible: Sequence[SolverTarget],
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
    floors: Mapping[str, str | None],
    severity_weight: Mapping[str, int],
) -> tuple[tuple[int, int, int, int, int, int], bool]:
    """Return bounded lexicographic weights safe for CP-SAT int64 objectives.

    The mixed-radix encoding preserves the exact ordering for ordinary-sized
    portfolios.  For very large portfolios the mathematically exact radix would
    exceed CP-SAT's int64 coefficient/offset range, so every weight is scaled
    together to a conservative bounded score.  Scaling is deterministic and
    prevents a native solver exception from turning an otherwise valid plan into
    ``FALLBACK``.
    """
    max_distance = sum(
        max(
            (
                _distance_above(candidate.version, floors[target.occurrence_id])
                for candidate in domains[target.occurrence_id]
            ),
            default=0,
        )
        for target in eligible
    )
    max_stable = sum(len(domains[target.occurrence_id]) for target in eligible)
    distance_bound = max_distance + 1
    stable_bound = max_stable + 1
    changed_bound = len(eligible) + 1
    workaround_bound = len(finding_list) + 1
    unresolved_bound = (
        sum(severity_weight.get(finding.severity.upper(), 0) for finding in finding_list) + 1
    )

    weight_distance = stable_bound
    weight_changed = distance_bound * weight_distance + stable_bound
    weight_workaround = (
        changed_bound * weight_changed + distance_bound * weight_distance + stable_bound
    )
    weight_unresolved = (
        workaround_bound * weight_workaround
        + changed_bound * weight_changed
        + distance_bound * weight_distance
        + stable_bound
    )
    weight_coverage = (
        unresolved_bound * weight_unresolved
        + workaround_bound * weight_workaround
        + changed_bound * weight_changed
        + distance_bound * weight_distance
        + stable_bound
    )
    weights = (
        weight_coverage,
        weight_unresolved,
        weight_workaround,
        weight_changed,
        weight_distance,
        1,
    )

    def objective_bound(values: tuple[int, int, int, int, int, int]) -> int:
        """Bound the absolute objective including affine constant offsets."""
        coverage, unresolved, workaround, changed, distance, stable = values
        return (
            len(finding_list) * coverage
            + unresolved_bound * unresolved
            + len(finding_list) * workaround
            + len(eligible) * changed
            + max_distance * distance
            + max_stable * stable
        )

    limit = 1 << 60
    total = objective_bound(weights)
    if total <= limit:
        return weights, False
    scale = max(2, (total + limit - 1) // limit)
    while True:
        scaled = tuple(max(1, value // scale) for value in weights)
        if objective_bound(scaled) <= limit:
            return scaled, True
        scale += 1


def _as_domain(
    target: SolverTarget,
    candidate_domains: Mapping[str, Sequence[SolverVersionCandidate]],
    *,
    max_candidates: int,
    diagnostics: list[str],
) -> list[SolverVersionCandidate]:
    """Normalize, sort, and bound one occurrence's candidate domain."""
    raw = candidate_domains.get(target.occurrence_id)
    if raw is None:
        raw = candidate_domains.get(target.task_id, ())
        if raw:
            _diagnostic(
                diagnostics, f"candidate domain used task-id compatibility key for {target.task_id}"
            )
    candidates = sorted(tuple(raw or ()), key=_candidate_key)
    unique: list[SolverVersionCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, SolverVersionCandidate):
            _diagnostic(diagnostics, f"invalid candidate for occurrence {target.occurrence_id}")
            continue
        if candidate.version in seen:
            _diagnostic(
                diagnostics,
                f"duplicate candidate {candidate.version} ignored for {target.occurrence_id}",
            )
            continue
        seen.add(candidate.version)
        unique.append(candidate)
    if len(unique) > max_candidates:
        _diagnostic(
            diagnostics,
            f"candidate domain for {target.occurrence_id} truncated from {len(unique)} to {max_candidates}",
        )
        eligible = [candidate for candidate in unique if candidate.meets_security_floor]
        required: list[SolverVersionCandidate] = [(eligible or unique)[-1]]
        installed = target.installed_version.strip().lstrip("vV")
        for candidate in unique:
            if candidate.version == installed and candidate not in required:
                required.append(candidate)
                break
        for candidate in reversed(unique):
            if len(required) >= max_candidates:
                break
            if candidate not in required:
                required.append(candidate)
        unique = sorted(required[:max_candidates], key=_candidate_key)
    return unique


def _floor_for_target(
    target: SolverTarget,
    findings: Mapping[str, SolverFindingRequirement],
    diagnostics: list[str],
) -> str | None:
    """Return the greatest valid fixed version required by a target's findings."""
    referenced = set(target.finding_ids)
    referenced.update(
        finding.finding_id
        for finding in findings.values()
        if finding.target_occurrence_id == target.occurrence_id
    )
    floors: list[str] = []
    for finding_id in sorted(referenced):
        finding = findings.get(finding_id)
        if finding is None or not finding.fixed_version:
            continue
        if _version_key(finding.fixed_version) is None:
            _diagnostic(
                diagnostics, f"invalid security floor {finding.fixed_version!r} for {finding_id}"
            )
            continue
        floors.append(finding.fixed_version)
    if not floors:
        return None
    return max(floors, key=lambda item: _version_key(item) or ())


def _range_for_candidate(
    source: SolverTarget,
    target: SolverTarget,
    candidate: SolverVersionCandidate,
    *,
    edge_range: str | None,
    peer: SolverPeerConstraint | None,
) -> bool | None:
    """Evaluate all known source-to-target range requirements for one pair."""
    if (
        peer is not None
        and peer.candidate_specific_source_version
        and candidate.version != peer.candidate_specific_source_version
    ):
        return False
    ranges: list[str] = []
    if edge_range:
        ranges.append(edge_range)
    for mapping in (candidate.dependency_ranges, candidate.peer_ranges):
        for name in (target.package_name, target.target_package_name):
            if name in mapping:
                ranges.append(mapping[name])
                break
    if peer is not None:
        ranges.append(peer.version_range)
    result = True
    for requirement in ranges:
        matched = _range_matches(requirement, target.installed_version)
        if matched is None:
            # The target's installed version is not the selected value here;
            # callers perform the full selected-pair check below.
            continue
        result = result and matched
    return result


def _pair_allowed(
    source: SolverTarget,
    target: SolverTarget,
    source_candidate: SolverVersionCandidate,
    target_candidate: SolverVersionCandidate,
    *,
    edge_range: str | None,
    peer: SolverPeerConstraint | None,
) -> bool | None:
    """Evaluate a candidate pair using explicit npm compatibility tables."""
    if peer is not None and peer.is_optional:
        return True
    if (
        peer is not None
        and peer.candidate_specific_source_version
        and source_candidate.version != peer.candidate_specific_source_version
    ):
        return False
    ranges: list[tuple[str, str]] = []
    candidate_range_found = False
    for mapping in (source_candidate.dependency_ranges, source_candidate.peer_ranges):
        for name in (target.package_name, target.target_package_name):
            if name in mapping:
                ranges.append((mapping[name], target_candidate.version))
                candidate_range_found = True
                break
    # The installed lockfile range is only a fallback.  Once a candidate's
    # published metadata is available it describes the selected pair and must
    # supersede the stale installed range.
    if edge_range and not candidate_range_found:
        ranges.append((edge_range, target_candidate.version))
    if peer is not None and not candidate_range_found:
        ranges.append((peer.version_range, target_candidate.version))
    # Published peer metadata on the target can constrain the source as well.
    for name in (source.package_name, source.target_package_name):
        requirement = target_candidate.peer_ranges.get(name)
        if requirement:
            ranges.append((requirement, source_candidate.version))
            break
    result = True
    for requirement, version in ranges:
        matched = _range_matches(requirement, version)
        if matched is None:
            return None
        if not matched:
            return False
    return result


def _constraint_pairs(
    source: SolverTarget,
    target: SolverTarget,
    source_candidates: Sequence[SolverVersionCandidate],
    target_candidates: Sequence[SolverVersionCandidate],
    *,
    edge_range: str | None,
    peer: SolverPeerConstraint | None,
    diagnostics: list[str],
) -> list[tuple[int, int]]:
    """Precompute the integer allowed-pair table for one graph relation."""
    pairs: list[tuple[int, int]] = []
    for left_index, left in enumerate(source_candidates):
        for right_index, right in enumerate(target_candidates):
            allowed = _pair_allowed(
                source,
                target,
                left,
                right,
                edge_range=edge_range,
                peer=peer,
            )
            if allowed is None:
                _diagnostic(
                    diagnostics,
                    f"invalid range in constraint {source.occurrence_id}->{target.occurrence_id}",
                )
            elif allowed:
                pairs.append((left_index, right_index))
    return pairs


def _compatible_alternatives(
    target: SolverTarget,
    candidates: Sequence[SolverVersionCandidate],
    selected_index: int,
    *,
    target_by_id: Mapping[str, SolverTarget],
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
    selected_indices: Mapping[str, int],
    relations: Sequence[tuple[str, str, str | None, SolverPeerConstraint | None]],
    floor: str | None = None,
) -> list[str]:
    """Return candidate versions compatible with the selected neighboring assignment."""
    if target.strategy.replace("-", "_").lower() in {"code_workaround", "workaround"}:
        return []
    compatible: list[str] = []
    for index, candidate in enumerate(candidates):
        if (
            index == selected_index
            or not candidate.meets_security_floor
            or not _at_least(candidate.version, floor)
        ):
            continue
        is_compatible = True
        for source_id, target_id, edge_range, peer in relations:
            if peer is not None and peer.is_optional:
                continue
            if target.occurrence_id not in {source_id, target_id}:
                continue
            neighbor_id = target_id if source_id == target.occurrence_id else source_id
            neighbor_index = selected_indices.get(neighbor_id)
            neighbor = target_by_id.get(neighbor_id)
            if neighbor is None or neighbor_index is None:
                continue
            neighbor_candidates = domains.get(neighbor_id, ())
            if not 0 <= neighbor_index < len(neighbor_candidates):
                is_compatible = False
                break
            neighbor_candidate = neighbor_candidates[neighbor_index]
            allowed = (
                _pair_allowed(
                    target,
                    neighbor,
                    candidate,
                    neighbor_candidate,
                    edge_range=edge_range,
                    peer=peer,
                )
                if source_id == target.occurrence_id
                else _pair_allowed(
                    neighbor,
                    target,
                    neighbor_candidate,
                    candidate,
                    edge_range=edge_range,
                    peer=peer,
                )
            )
            if allowed is not True:
                is_compatible = False
                break
        if is_compatible:
            compatible.append(candidate.version)
    return compatible


def _project_candidate_plan(
    solver: Any,
    *,
    alternative_index: int,
    status: SolverStatus,
    targets: Sequence[SolverTarget],
    eligible: Sequence[SolverTarget],
    vars_by_id: Mapping[str, Any],
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
    target_by_id: Mapping[str, SolverTarget],
    finding_by_target: Mapping[str, Sequence[SolverFindingRequirement]],
    findings_covered: Mapping[str, Any],
    findings_workaround: Mapping[str, Any],
    finding_list: Sequence[SolverFindingRequirement],
    severity_weight: Mapping[str, int],
    floors: Mapping[str, str | None],
    relations: Sequence[tuple[str, str, str | None, SolverPeerConstraint | None]],
    diagnostics: Sequence[str],
) -> tuple[SolverCandidatePlan, dict[str, int]]:
    """Project one CP-SAT assignment into the immutable candidate contract."""
    selected_indices = {
        occurrence_id: int(solver.Value(variable)) for occurrence_id, variable in vars_by_id.items()
    }
    covered_values = {
        finding_id: bool(solver.Value(value)) for finding_id, value in findings_covered.items()
    }
    workaround_values = {
        finding_id: bool(solver.Value(value)) for finding_id, value in findings_workaround.items()
    }
    compatible_alternatives_by_target = {
        target.occurrence_id: _compatible_alternatives(
            target,
            domains[target.occurrence_id],
            selected_indices.get(target.occurrence_id, -1),
            target_by_id=target_by_id,
            domains=domains,
            selected_indices=selected_indices,
            relations=relations,
            floor=floors.get(target.occurrence_id),
        )
        for target in eligible
    }
    decisions = [
        _build_decision(
            target,
            domains.get(target.occurrence_id, ()),
            selected_indices.get(target.occurrence_id, -1),
            finding_by_target,
            covered_values,
            workaround_values,
            compatible_alternatives=compatible_alternatives_by_target.get(target.occurrence_id),
        )
        for target in targets
    ]
    coverage_ids = sorted(finding_id for finding_id, value in covered_values.items() if value)
    unresolved_ids = sorted(finding_id for finding_id, value in covered_values.items() if not value)
    unresolved_critical_high = sum(
        severity_weight.get(finding.severity.upper(), 0)
        for finding in finding_list
        if finding.finding_id in unresolved_ids
    )
    workaround_count = sum(workaround_values.values())
    changed_count = sum(
        bool(
            domains[occurrence_id][selected_indices[occurrence_id]].version
            != target_by_id[occurrence_id].installed_version
        )
        for occurrence_id in selected_indices
    )
    distance = sum(
        _distance_above(
            domains[occurrence_id][selected_indices[occurrence_id]].version,
            floors[occurrence_id],
        )
        for occurrence_id in selected_indices
    )
    stable = sum(selected_indices.values())
    candidate = SolverCandidatePlan(
        candidate_plan_id=f"candidate-{alternative_index + 1:04d}",
        objective_vector=(
            len(coverage_ids),
            unresolved_critical_high,
            workaround_count,
            changed_count,
            distance,
            -stable,
        ),
        coverage_ids=coverage_ids,
        unresolved_ids=unresolved_ids,
        task_decisions=decisions,
        batches=[],
        phases=[],
        status=status,
        diagnostics=list(diagnostics),
    )
    return candidate, selected_indices


def _build_decision(
    target: SolverTarget,
    candidates: Sequence[SolverVersionCandidate],
    selected_index: int,
    findings_by_target: Mapping[str, Sequence[SolverFindingRequirement]],
    finding_covered: Mapping[str, bool],
    finding_workaround: Mapping[str, bool],
    *,
    compatible_alternatives: Sequence[str] | None = None,
) -> SolverTaskDecision:
    """Project one assignment into a solver-approved task decision."""
    requirements = list(findings_by_target.get(target.occurrence_id, ()))
    requirements.sort(key=lambda item: item.finding_id)
    selected = candidates[selected_index] if 0 <= selected_index < len(candidates) else None
    all_by_version = bool(requirements) and all(
        finding_covered.get(item.finding_id, False)
        and not finding_workaround.get(item.finding_id, False)
        for item in requirements
    )
    all_by_workaround = bool(requirements) and all(
        finding_workaround.get(item.finding_id, False) for item in requirements
    )
    preferred_strategy = target.strategy.replace("-", "_").lower()
    if preferred_strategy in {"code_workaround", "workaround"}:
        if all_by_workaround:
            strategy = "code_workaround"
            route = "code_workaround"
        else:
            strategy = "no_fix"
            route = "no_fix"
    elif preferred_strategy == "no_fix":
        strategy = "no_fix"
        route = "no_fix"
    elif not requirements or all_by_version:
        strategy = "version_bump"
        route = "version_bump"
    elif all_by_workaround:
        strategy = "code_workaround"
        route = "code_workaround"
    else:
        strategy = "no_fix"
        route = "no_fix"
    if target.is_terminal or target.has_open_attempt:
        selected = None
        alternatives: list[str] = []
    elif strategy == "code_workaround":
        alternatives = []
    else:
        alternatives = list(
            compatible_alternatives
            if compatible_alternatives is not None
            else (
                candidate.version
                for index, candidate in enumerate(candidates)
                if index != selected_index and candidate.meets_security_floor
            )
        )
    selected_plan_ids = sorted(
        {
            plan_id
            for finding in requirements
            for plan_id in finding.workaround_plan_ids
            if strategy == "code_workaround" and finding_workaround.get(finding.finding_id, False)
        }
    )
    stage = requirements[0].strategy_stage if requirements else "osv_minimum"
    return SolverTaskDecision(
        task_id=target.task_id,
        selected_strategy=strategy,
        selected_route=route,
        selected_version=selected.version
        if selected is not None and strategy == "version_bump"
        else None,
        allowed_alternative_versions=alternatives,
        allowed_dependency_types=[target.dependency_type],
        strategy_stage=stage,
        selected_plan_issue_ids=selected_plan_ids,
        instruction_source="deterministic_solver",
        exact_instruction=None,
        target_occurrence_id=target.occurrence_id,
        target_group_id=target.group_id,
        target_package_name=target.target_package_name,
        manifest_path=target.manifest_path,
        lockfile_package_key=target.lockfile_package_key,
        dependency_type=target.dependency_type,
    )


def _status_from_cp_code(cp_model: Any, status_code: Any) -> SolverStatus:
    """Map a CP-SAT status code to the typed solver status."""
    if status_code == cp_model.OPTIMAL:
        return SolverStatus.OPTIMAL
    if status_code == cp_model.FEASIBLE:
        return SolverStatus.FEASIBLE
    if status_code == cp_model.INFEASIBLE:
        return SolverStatus.INFEASIBLE
    return SolverStatus.UNKNOWN


def _raw_status_name(solver: Any, status_code: Any) -> str:
    """Return the native CP-SAT status name without collapsing it."""
    try:
        name = solver.StatusName(status_code)
    except Exception:  # pragma: no cover - native API compatibility guard
        name = status_code
    return str(name).strip().upper() or "UNKNOWN"


def _raw_status_code(status_code: Any) -> int | None:
    """Return a serializable native status code when one is available."""
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _safe_solver_float(solver: Any, method_name: str) -> float | None:
    """Read one finite floating-point statistic from a CP-SAT solver."""
    try:
        value = float(getattr(solver, method_name)())
    except (AttributeError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _safe_solver_nonnegative_float(solver: Any, method_name: str) -> float:
    """Read one finite non-negative CP-SAT timing statistic."""
    value = _safe_solver_float(solver, method_name)
    return max(0.0, value) if value is not None else 0.0


def _safe_solver_int(solver: Any, method_name: str) -> int:
    """Read one non-negative integer statistic from a CP-SAT solver."""
    try:
        return max(0, int(getattr(solver, method_name)()))
    except (AttributeError, TypeError, ValueError):
        return 0


def _record_solver_result(
    cp_model: Any,
    solver: Any,
    status_code: Any,
    observations: list[dict[str, Any]],
) -> SolverStatus:
    """Record a native solve result before mapping it to the public status."""
    status = _status_from_cp_code(cp_model, status_code)
    observations.append(
        {
            "raw_status_name": _raw_status_name(solver, status_code),
            "raw_status_code": _raw_status_code(status_code),
            "wall_time_seconds": _safe_solver_nonnegative_float(solver, "WallTime"),
            "user_time_seconds": _safe_solver_nonnegative_float(solver, "UserTime"),
            "deterministic_time_seconds": _safe_solver_nonnegative_float(
                solver, "DeterministicTime"
            ),
            "num_conflicts": _safe_solver_int(solver, "NumConflicts"),
            "num_branches": _safe_solver_int(solver, "NumBranches"),
            "objective_value": _safe_solver_float(solver, "ObjectiveValue"),
            "best_objective_bound": _safe_solver_float(solver, "BestObjectiveBound"),
        }
    )
    return status


def _build_solver_statistics(
    observations: Sequence[Mapping[str, Any]],
) -> SolverStatistics | None:
    """Project native solve observations into the bounded contract model."""
    if not observations:
        return None
    last = observations[-1]
    status_codes = [
        code for code in (item.get("raw_status_code") for item in observations) if code is not None
    ]
    return SolverStatistics(
        solve_calls=len(observations),
        raw_status_name=last.get("raw_status_name"),
        raw_status_code=last.get("raw_status_code"),
        status_sequence=[str(item.get("raw_status_name") or "UNKNOWN") for item in observations],
        status_code_sequence=status_codes,
        wall_time_seconds=sum(float(item.get("wall_time_seconds") or 0.0) for item in observations),
        user_time_seconds=sum(float(item.get("user_time_seconds") or 0.0) for item in observations),
        deterministic_time_seconds=sum(
            float(item.get("deterministic_time_seconds") or 0.0) for item in observations
        ),
        num_conflicts=sum(int(item.get("num_conflicts") or 0) for item in observations),
        num_branches=sum(int(item.get("num_branches") or 0) for item in observations),
        objective_value=last.get("objective_value"),
        best_objective_bound=last.get("best_objective_bound"),
    )


def _add_statistics_diagnostic(diagnostics: list[str], statistics: SolverStatistics | None) -> None:
    """Add one concise native status summary to the bounded diagnostics."""
    if statistics is None:
        return
    _diagnostic(
        diagnostics,
        "CP-SAT raw status "
        f"{statistics.raw_status_name} (code={statistics.raw_status_code}, "
        f"calls={statistics.solve_calls})",
    )


def _solve_lexicographic(
    cp_model: Any,
    base_model: Any,
    solver: Any,
    *,
    vars_by_id: Mapping[str, Any],
    metrics: Sequence[tuple[Any, bool]],
    top_k: int,
    timeout_seconds: float,
    accepted_feasible: bool,
    project: Callable[[Any, int, SolverStatus], tuple[SolverCandidatePlan, dict[str, int]]],
    observations: list[dict[str, Any]],
) -> tuple[list[SolverCandidatePlan], SolverStatus | None]:
    """Enumerate top-K assignments with exact bounded lexicographic stages."""
    candidate_plans: list[SolverCandidatePlan] = []
    forbidden_assignments: list[dict[str, int]] = []
    aggregate_status: SolverStatus | None = None
    deadline = time.monotonic() + max(0.001, timeout_seconds)
    ordered_occurrences = sorted(vars_by_id)
    for alternative_index in range(top_k):
        model = base_model.Clone()
        for assignment in forbidden_assignments:
            model.AddForbiddenAssignments(
                [vars_by_id[occurrence_id] for occurrence_id in ordered_occurrences],
                [[assignment[occurrence_id] for occurrence_id in ordered_occurrences]],
            )
        final_status = SolverStatus.OPTIMAL
        for expression, maximize in metrics:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                remaining = 0.001
            model.ClearObjective()
            if maximize:
                model.Maximize(expression)
            else:
                model.Minimize(expression)
            solver.parameters.max_time_in_seconds = max(0.001, remaining)
            status_code = solver.Solve(model)
            status = _record_solver_result(cp_model, solver, status_code, observations)
            if status == SolverStatus.FEASIBLE:
                aggregate_status = SolverStatus.FEASIBLE
            elif status == SolverStatus.OPTIMAL and aggregate_status is None:
                aggregate_status = SolverStatus.OPTIMAL
            elif (
                status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}
                and aggregate_status is None
            ):
                aggregate_status = status
            if status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}:
                final_status = status
                break
            if status == SolverStatus.FEASIBLE:
                if not accepted_feasible:
                    if candidate_plans:
                        return candidate_plans, candidate_plans[0].status
                    return candidate_plans, SolverStatus.UNKNOWN
                final_status = SolverStatus.FEASIBLE
            try:
                metric_value = int(solver.Value(expression))
            except (TypeError, ValueError):
                metric_value = int(expression)
            model.Add(expression == metric_value)
        if final_status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}:
            if not candidate_plans:
                return candidate_plans, final_status
            break
        candidate, selected_indices = project(solver, alternative_index, final_status)
        candidate_plans.append(candidate)
        if not ordered_occurrences:
            break
        forbidden_assignments.append(selected_indices)
    return candidate_plans, aggregate_status


@traceable(
    name="portfolio_solver",
    run_type="chain",
    process_inputs=_trace_solver_inputs,
    process_outputs=_trace_solver_outputs,
)
def solve_portfolio(
    subgraph: SolverSubgraph,
    candidate_domains: Mapping[str, Sequence[SolverVersionCandidate]],
    *,
    settings: Any,
) -> SolverRemediationPlan:
    """Solve one occurrence-aware portfolio with deterministic CP-SAT.

    Args:
        subgraph: Immutable target, finding, peer, and dependency graph.
        candidate_domains: Bounded candidates keyed by occurrence ID (task ID is
            accepted as a compatibility key).
        settings: Explicit AppSettings-like object. No environment variables are
            read by this function.

    Returns:
        A typed plan with top-K deterministic assignments, or a typed failure
        status. Native solver errors and model guards return ``FALLBACK`` and
        never include a dispatchable selected plan.
    """
    diagnostics = list(subgraph.diagnostics)
    targets = sorted(subgraph.targets, key=lambda item: item.occurrence_id)
    findings = {finding.finding_id: finding for finding in subgraph.findings}
    finding_list = sorted(subgraph.findings, key=lambda item: item.finding_id)
    max_candidates = max(1, int(_setting(settings, "solver_max_candidates_per_target", 64)))
    max_variables = max(1, int(_setting(settings, "solver_max_model_variables", 10_000)))
    domains: dict[str, list[SolverVersionCandidate]] = {
        target.occurrence_id: _as_domain(
            target,
            candidate_domains,
            max_candidates=max_candidates,
            diagnostics=diagnostics,
        )
        for target in targets
        if target.eligible_for_atomic_update
    }
    floors = {
        target.occurrence_id: _floor_for_target(target, findings, diagnostics) for target in targets
    }
    for target in targets:
        if target.eligible_for_atomic_update and not domains.get(target.occurrence_id):
            _diagnostic(
                diagnostics, f"no candidates for eligible occurrence {target.occurrence_id}"
            )
    domain_payload = {
        occurrence_id: [candidate.model_dump(mode="json") for candidate in values]
        for occurrence_id, values in sorted(domains.items())
    }
    input_digest = _digest(subgraph.model_dump(mode="json"))
    domain_digest = _digest(domain_payload)
    repository_digest = _digest(
        {
            "targets": [target.model_dump(mode="json") for target in targets],
            "edges": [edge.model_dump(mode="json") for edge in subgraph.edges],
        }
    )
    revisions = {target.task_id: 0 for target in targets}
    if not getattr(subgraph, "valid", True):
        _diagnostic(diagnostics, "invalid solver subgraph; no dispatchable plan")
        return SolverRemediationPlan(
            status=SolverStatus.UNKNOWN,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
            diagnostics=diagnostics,
        )

    # Import lazily so importing the contracts and CLI remains possible in an
    # environment where the production dependency has not yet been installed.
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover - exercised by packaging failures
        _diagnostic(diagnostics, f"CP-SAT import failed: {exc}")
        return SolverRemediationPlan(
            status=SolverStatus.FALLBACK,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
            diagnostics=diagnostics,
        )

    eligible = [target for target in targets if target.eligible_for_atomic_update]
    # One integer variable per eligible occurrence, plus bounded coverage and
    # workaround booleans. Empty domains are a typed infeasible result.
    if any(not domains.get(target.occurrence_id) for target in eligible):
        return SolverRemediationPlan(
            status=SolverStatus.INFEASIBLE,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
            diagnostics=diagnostics,
        )
    estimated_variables = (
        len(eligible)
        + sum(len(domains[target.occurrence_id]) for target in eligible)
        + (len(finding_list) * 3)
        + sum(
            sum(
                1
                for candidate in domains.get(finding.target_occurrence_id, ())
                if finding.fixed_version
                and candidate.meets_security_floor
                and _at_least(candidate.version, finding.fixed_version)
            )
            for finding in finding_list
        )
    )
    if estimated_variables > max_variables:
        _diagnostic(
            diagnostics,
            f"solver model variable guard exceeded: {estimated_variables}>{max_variables}",
        )
        return SolverRemediationPlan(
            status=SolverStatus.FALLBACK,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
            diagnostics=diagnostics,
        )

    target_by_id = {target.occurrence_id: target for target in targets}
    vars_by_id: dict[str, Any] = {}
    finding_by_target: dict[str, list[SolverFindingRequirement]] = {}
    for finding in finding_list:
        finding_by_target.setdefault(finding.target_occurrence_id, []).append(finding)
    model = cp_model.CpModel()
    for target in eligible:
        values = domains[target.occurrence_id]
        target_findings = target.finding_ids or [
            finding.finding_id
            for finding in finding_list
            if finding.target_occurrence_id == target.occurrence_id
        ]
        target_requirements = finding_by_target.get(target.occurrence_id, [])
        preferred_strategy = target.strategy.replace("-", "_").lower()
        floor_required = preferred_strategy not in {"code_workaround", "workaround", "no_fix"}
        allowed = [
            index
            for index, candidate in enumerate(values)
            if not target_findings
            or (
                not floor_required
                or all(
                    candidate.meets_security_floor
                    and _at_least(candidate.version, floors[target.occurrence_id])
                    for _ in target_findings
                )
            )
        ]
        if (
            not allowed
            and floor_required
            and target_requirements
            and all(
                finding.workaround_available and finding.workaround_plan_ids
                for finding in target_requirements
            )
        ):
            # A validated workaround is allowed to replace a version bump
            # when the registry domain cannot satisfy the security floor.
            allowed = list(range(len(values)))
        if not allowed:
            _diagnostic(
                diagnostics, f"security floor removes every candidate for {target.occurrence_id}"
            )
            return SolverRemediationPlan(
                status=SolverStatus.INFEASIBLE,
                input_digest=input_digest,
                domain_digest=domain_digest,
                repository_digest=repository_digest,
                task_revisions=revisions,
                unresolved_finding_ids=[finding.finding_id for finding in finding_list],
                diagnostics=diagnostics,
            )
        vars_by_id[target.occurrence_id] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(allowed), f"occurrence_{len(vars_by_id):04d}"
        )

    # Add explicit integer allowed-pair tables for every constrained relation.
    peer_by_pair: dict[tuple[str, str], SolverPeerConstraint] = {
        (peer.source_occurrence_id, peer.target_occurrence_id): peer
        for peer in subgraph.peer_constraints
    }
    relations: list[tuple[str, str, str | None, SolverPeerConstraint | None]] = []
    for peer in subgraph.peer_constraints:
        relations.append(
            (peer.source_occurrence_id, peer.target_occurrence_id, peer.version_range, peer)
        )
    for edge in subgraph.edges:
        if edge.is_optional:
            continue
        if edge.edge_kind in {
            "runtime",
            "dependency",
            "ancestry",
            "workspace",
            "scope",
            "pinned",
            "peer",
        }:
            relations.append(
                (
                    edge.source_occurrence_id,
                    edge.target_occurrence_id,
                    edge.version_range,
                    peer_by_pair.get((edge.source_occurrence_id, edge.target_occurrence_id)),
                )
            )
    seen_relations: set[tuple[str, str, str | None]] = set()
    for source_id, target_id, edge_range, peer in relations:
        if peer is not None and peer.is_optional:
            continue
        relation_key = (source_id, target_id, edge_range)
        if (
            relation_key in seen_relations
            or source_id not in vars_by_id
            or target_id not in vars_by_id
        ):
            continue
        seen_relations.add(relation_key)
        source = target_by_id.get(source_id)
        target = target_by_id.get(target_id)
        if source is None or target is None:
            continue
        pairs = _constraint_pairs(
            source,
            target,
            domains[source_id],
            domains[target_id],
            edge_range=edge_range,
            peer=peer,
            diagnostics=diagnostics,
        )
        if not pairs:
            _diagnostic(diagnostics, f"no compatible candidate pair for {source_id}->{target_id}")
            model.AddBoolOr([])
        else:
            model.AddAllowedAssignments([vars_by_id[source_id], vars_by_id[target_id]], pairs)

    findings_covered: dict[str, Any] = {}
    findings_version: dict[str, Any] = {}
    findings_workaround: dict[str, Any] = {}
    coverage_meta: dict[str, tuple[list[int], str | None]] = {}
    for finding in finding_list:
        covered = model.NewBoolVar(f"finding_covered_{finding.finding_id}")
        workaround = model.NewBoolVar(f"finding_workaround_{finding.finding_id}")
        version_bool = model.NewBoolVar(f"finding_version_{finding.finding_id}")
        findings_covered[finding.finding_id] = covered
        findings_workaround[finding.finding_id] = workaround
        findings_version[finding.finding_id] = version_bool
        occurrence_var = vars_by_id.get(finding.target_occurrence_id)
        candidates = domains.get(finding.target_occurrence_id, ())
        good_indices: list[int] = []
        if occurrence_var is not None and finding.fixed_version:
            good_indices = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.meets_security_floor
                and _at_least(candidate.version, finding.fixed_version)
            ]
        if occurrence_var is not None and good_indices:
            literals = []
            for index in good_indices:
                literal = model.NewBoolVar(f"finding_{finding.finding_id}_version_{index}")
                model.Add(occurrence_var == index).OnlyEnforceIf(literal)
                model.Add(occurrence_var != index).OnlyEnforceIf(literal.Not())
                literals.append(literal)
            model.AddMaxEquality(version_bool, literals)
        else:
            model.Add(version_bool == 0)
        has_workaround = finding.workaround_available and bool(finding.workaround_plan_ids)
        preferred_strategy = (
            target_by_id[finding.target_occurrence_id].strategy.replace("-", "_").lower()
            if finding.target_occurrence_id in target_by_id
            else ""
        )
        if not has_workaround:
            model.Add(workaround == 0)
        elif preferred_strategy in {"code_workaround", "workaround"}:
            # A workaround target is only dispatchable when every selected
            # workaround carries the plan IDs that authorize it.
            model.Add(workaround == 1)
        model.AddMaxEquality(covered, [version_bool, workaround])
        coverage_meta[finding.finding_id] = (good_indices, finding.fixed_version)

    candidate_literals: dict[str, list[Any]] = {}
    for target in eligible:
        variable = vars_by_id[target.occurrence_id]
        literals = []
        for index in range(len(domains[target.occurrence_id])):
            literal = model.NewBoolVar(f"candidate_{len(candidate_literals):04d}_{index}")
            model.Add(variable == index).OnlyEnforceIf(literal)
            model.Add(variable != index).OnlyEnforceIf(literal.Not())
            literals.append(literal)
        candidate_literals[target.occurrence_id] = literals

    # Lexicographic objective encoded as bounded integer weights. Coverage is
    # maximized; every subsequent term is minimized in the required order.
    severity_weight = {"CRITICAL": 1000, "HIGH": 100}
    (
        (
            weight_coverage,
            weight_unresolved,
            weight_workaround,
            weight_changed,
            weight_distance,
            _weight_stable,
        ),
        weights_scaled,
    ) = _objective_weights(
        finding_list,
        eligible,
        domains,
        floors,
        severity_weight,
    )
    if weights_scaled:
        _diagnostic(
            diagnostics,
            "objective weights scaled to remain within CP-SAT int64 bounds",
        )
    coverage_expression = sum(findings_covered.values())
    unresolved_expression = sum(
        severity_weight.get(finding.severity.upper(), 0)
        * (1 - findings_covered[finding.finding_id])
        for finding in finding_list
    )
    workaround_expression = sum(findings_workaround.values())
    changed_terms: list[Any] = []
    distance_terms: list[Any] = []
    stable_terms: list[Any] = []
    for target in eligible:
        values = domains[target.occurrence_id]
        floor = floors[target.occurrence_id]
        literals = candidate_literals[target.occurrence_id]
        changed_terms.append(
            sum(
                int(candidate.version != target.installed_version) * literals[index]
                for index, candidate in enumerate(values)
            )
        )
        distance_terms.append(
            sum(
                _distance_above(candidate.version, floor) * literals[index]
                for index, candidate in enumerate(values)
            )
        )
        stable_terms.append(sum((index + 1) * literals[index] for index in range(len(values))))
    changed_expression = sum(changed_terms)
    distance_expression = sum(distance_terms)
    stable_expression = sum(stable_terms)
    metrics = (
        (coverage_expression, True),
        (unresolved_expression, False),
        (workaround_expression, False),
        (changed_expression, False),
        (distance_expression, False),
        (stable_expression, False),
    )
    objective_terms: list[Any] = [
        weight_coverage * coverage_expression,
        -weight_unresolved * unresolved_expression,
        -weight_workaround * workaround_expression,
        -weight_changed * changed_expression,
        -weight_distance * distance_expression,
        -stable_expression,
    ]
    base_model = model.Clone()
    if not weights_scaled:
        model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(_setting(settings, "solver_timeout_seconds", 10))
    solver.parameters.random_seed = int(_setting(settings, "solver_random_seed", 0))
    solver.parameters.num_search_workers = int(_setting(settings, "solver_num_search_workers", 1))
    accepted_feasible = bool(_setting(settings, "solver_accept_feasible", True))
    top_k = max(1, int(_setting(settings, "solver_top_k", 3)))
    candidate_plans: list[SolverCandidatePlan] = []
    first_status: SolverStatus | None = None
    observations: list[dict[str, Any]] = []
    weighted_deadline = time.monotonic() + max(
        0.001, float(_setting(settings, "solver_timeout_seconds", 10))
    )

    def project(
        solver_instance: Any,
        alternative_index: int,
        status: SolverStatus,
    ) -> tuple[SolverCandidatePlan, dict[str, int]]:
        """Project a solver assignment without duplicating local wiring."""
        return _project_candidate_plan(
            solver_instance,
            alternative_index=alternative_index,
            status=status,
            targets=targets,
            eligible=eligible,
            vars_by_id=vars_by_id,
            domains=domains,
            target_by_id=target_by_id,
            finding_by_target=finding_by_target,
            findings_covered=findings_covered,
            findings_workaround=findings_workaround,
            finding_list=finding_list,
            severity_weight=severity_weight,
            floors=floors,
            relations=relations,
            diagnostics=diagnostics,
        )

    try:
        if weights_scaled:
            candidate_plans, first_status = _solve_lexicographic(
                cp_model,
                base_model,
                solver,
                vars_by_id=vars_by_id,
                metrics=metrics,
                top_k=top_k,
                timeout_seconds=float(_setting(settings, "solver_timeout_seconds", 10)),
                accepted_feasible=accepted_feasible,
                project=project,
                observations=observations,
            )
        else:
            for alternative_index in range(top_k):
                remaining = weighted_deadline - time.monotonic()
                if remaining <= 0 and candidate_plans:
                    break
                solver.parameters.max_time_in_seconds = max(0.001, remaining)
                status_code = solver.Solve(model)
                status = _record_solver_result(cp_model, solver, status_code, observations)
                if first_status is None:
                    first_status = status
                elif (
                    status == SolverStatus.FEASIBLE
                    and first_status == SolverStatus.OPTIMAL
                    and not candidate_plans
                ):
                    first_status = SolverStatus.FEASIBLE
                if status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}:
                    if not candidate_plans:
                        solver_statistics = _build_solver_statistics(observations)
                        _add_statistics_diagnostic(diagnostics, solver_statistics)
                        return SolverRemediationPlan(
                            status=status,
                            input_digest=input_digest,
                            domain_digest=domain_digest,
                            repository_digest=repository_digest,
                            task_revisions=revisions,
                            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
                            diagnostics=diagnostics,
                            solver_statistics=solver_statistics,
                        )
                    break
                if status == SolverStatus.FEASIBLE and not accepted_feasible:
                    _diagnostic(
                        diagnostics, "feasible CP-SAT result rejected by solver_accept_feasible"
                    )
                    if candidate_plans:
                        break
                    solver_statistics = _build_solver_statistics(observations)
                    _add_statistics_diagnostic(diagnostics, solver_statistics)
                    return SolverRemediationPlan(
                        status=SolverStatus.UNKNOWN,
                        input_digest=input_digest,
                        domain_digest=domain_digest,
                        repository_digest=repository_digest,
                        task_revisions=revisions,
                        unresolved_finding_ids=[finding.finding_id for finding in finding_list],
                        diagnostics=diagnostics,
                        solver_statistics=solver_statistics,
                    )
                candidate, selected_indices = project(solver, alternative_index, status)
                candidate_plans.append(candidate)
                if not vars_by_id:
                    break
                model.AddForbiddenAssignments(
                    [vars_by_id[occurrence_id] for occurrence_id in sorted(vars_by_id)],
                    [[selected_indices[occurrence_id] for occurrence_id in sorted(vars_by_id)]],
                )
    except Exception as exc:  # pragma: no cover - native CP-SAT failures are environment-specific
        _diagnostic(diagnostics, f"native CP-SAT error: {exc}")
        solver_statistics = _build_solver_statistics(observations)
        _add_statistics_diagnostic(diagnostics, solver_statistics)
        return SolverRemediationPlan(
            status=SolverStatus.FALLBACK,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=[finding.finding_id for finding in finding_list],
            diagnostics=diagnostics,
            solver_statistics=solver_statistics,
        )

    final_status = first_status or SolverStatus.UNKNOWN
    solver_statistics = _build_solver_statistics(observations)
    _add_statistics_diagnostic(diagnostics, solver_statistics)
    selected_plan = (
        candidate_plans[0]
        if final_status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}
        else None
    )
    return SolverRemediationPlan(
        status=final_status,
        input_digest=input_digest,
        domain_digest=domain_digest,
        repository_digest=repository_digest,
        task_revisions=revisions,
        candidate_plans=candidate_plans,
        selected_plan=selected_plan,
        unresolved_finding_ids=(
            selected_plan.unresolved_ids
            if selected_plan
            else [finding.finding_id for finding in finding_list]
        ),
        diagnostics=diagnostics,
        solver_statistics=solver_statistics,
    )


__all__ = ["solve_portfolio"]
