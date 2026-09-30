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
    SolverCandidateConflict,
    SolverCandidateCutKind,
    SolverCandidatePlan,
    SolverCandidateRejectionReason,
    SolverCandidateRelation,
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
from remediation_engine.settings import DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET

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
    evidence_domains = list(_trace_field(subgraph, "evidence_domains", ()) or ())
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
        "evidence_domain_count": len(evidence_domains),
        "evidence_candidate_total_count": sum(
            len(_trace_field(domain, "candidate_versions", ()) or ()) for domain in evidence_domains
        ),
        "candidate_total_count": sum(domain_counts.values()),
        "forbidden_assignment_count": len(inputs.get("forbidden_assignments") or ()),
        "forbidden_assignments": [
            {
                "digest": _digest(dict(sorted(item.items()))),
                "values": dict(sorted(item.items())[:32]),
            }
            for item in (inputs.get("forbidden_assignments") or ())[:32]
            if isinstance(item, Mapping)
        ],
        "forbidden_conflict_count": len(inputs.get("forbidden_conflicts") or ()),
        "forbidden_conflicts": [
            {
                "assignment_digest": _trace_field(item, "assignment_digest"),
                "reason_code": _trace_field(item, "reason_code"),
                "cut_kind": _trace_field(item, "cut_kind"),
                "evidence_digest": _trace_field(item, "evidence_digest"),
                "literals": [
                    {
                        "variable_id": _trace_field(literal, "variable_id"),
                        "version": _trace_field(literal, "version"),
                    }
                    for literal in list(_trace_field(item, "literals", ()) or ())[:16]
                ],
                "summary": str(_trace_field(item, "summary", "") or "")[:160],
            }
            for item in (inputs.get("forbidden_conflicts") or ())[:32]
        ],
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


def _stable_semver_major(value: str | None) -> int | None:
    """Return a major version only for valid stable semantic versions."""
    if not value:
        return None
    try:
        from semantic_version import Version

        parsed = Version(value.strip().lstrip("vV"))
    except (ImportError, TypeError, ValueError):
        return None
    return None if parsed.prerelease else parsed.major


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
    security_floor: str | None,
    diagnostics: list[str],
) -> list[SolverVersionCandidate]:
    """Normalize and goal-directly bound one target's candidate domain.

    The installed version is always retained as the status-quo assignment.
    Large catalogs retain the security floor and the newest patch release from
    each minor branch at or above that floor.
    """
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

    installed_version = str(target.installed_version or "").strip().lstrip("vV")
    if installed_version and installed_version not in seen:
        installed_key = _version_key(installed_version)
        unique.append(
            SolverVersionCandidate(
                version=installed_version,
                semver_key=installed_key or (),
                source="current",
                meets_security_floor=_at_least(installed_version, security_floor),
            )
        )
        seen.add(installed_version)
        if installed_key is None:
            _diagnostic(
                diagnostics,
                f"installed version {installed_version!r} is non-semver; retained only as "
                f"the status-quo candidate for {target.occurrence_id}",
            )

    if len(unique) > max_candidates:
        baseline_versions = {installed_version} if installed_version in seen else set()
        floor_version = str(security_floor or "").strip().lstrip("vV")
        retained_versions = set(baseline_versions)
        if floor_version in seen:
            retained_versions.add(floor_version)

        latest_by_minor: dict[tuple[int, int], SolverVersionCandidate] = {}
        for candidate in unique:
            if not candidate.meets_security_floor or not _at_least(
                candidate.version, security_floor
            ):
                continue
            version_key = _version_key(candidate.version)
            if version_key is None:
                continue
            branch = (version_key[0], version_key[1])
            previous = latest_by_minor.get(branch)
            if previous is None or _candidate_key(candidate) > _candidate_key(previous):
                latest_by_minor[branch] = candidate
        branch_candidates = sorted(latest_by_minor.values(), key=_candidate_key, reverse=True)
        room = max(0, max_candidates - len(retained_versions))
        retained_versions.update(candidate.version for candidate in branch_candidates[:room])
        unique = [candidate for candidate in unique if candidate.version in retained_versions]
        _diagnostic(
            diagnostics,
            f"candidate catalog pruned for {target.occurrence_id}: "
            f"{len(candidates)}>{max_candidates}; retained installed, security-floor, "
            "and latest-per-minor candidates",
        )
    return sorted(unique, key=_candidate_key)


def _floor_for_target(
    target: SolverTarget,
    findings: Mapping[str, SolverFindingRequirement],
    diagnostics: list[str],
) -> str | None:
    """Return the greatest valid fixed version required by a target's findings."""
    requirements = sorted(
        (
            finding
            for finding in findings.values()
            if finding.target_occurrence_id == target.occurrence_id
        ),
        key=lambda item: item.coverage_id,
    )
    floors: list[str] = []
    for finding in requirements:
        if not finding.fixed_version:
            continue
        if _version_key(finding.fixed_version) is None:
            _diagnostic(
                diagnostics,
                f"invalid security floor {finding.fixed_version!r} for {finding.coverage_id}",
            )
            continue
        floors.append(finding.fixed_version)
    if not floors:
        return None
    return max(floors, key=lambda item: _version_key(item) or ())


def _pair_allowed(
    source: SolverTarget,
    target: SolverTarget,
    source_candidate: SolverVersionCandidate,
    target_candidate: SolverVersionCandidate,
    *,
    edge_range: str | None,
    peer: SolverPeerConstraint | None,
    candidate_relations: Sequence[SolverCandidateRelation] = (),
) -> bool | None:
    """Evaluate static edges and exact candidate-specific dependency ranges."""
    if peer is not None and peer.is_optional:
        return True
    if (
        peer is not None
        and peer.candidate_specific_source_version
        and source_candidate.version != peer.candidate_specific_source_version
    ):
        return False
    override_target = peer is None and target.dependency_type.strip().lower() in {
        "overrides",
        "resolutions",
        "pnpm_overrides",
    }
    candidate_requirement_found = False
    for relation in candidate_relations:
        if relation.is_optional or relation.kind == "optional_dependency":
            continue
        package_matches_target = relation.package_name in {
            target.package_name,
            target.target_package_name,
        }
        package_matches_source = relation.package_name in {
            source.package_name,
            source.target_package_name,
        }
        if (
            relation.source_occurrence_id == source.occurrence_id
            and relation.source_candidate_version == source_candidate.version
            and package_matches_target
        ):
            candidate_requirement_found = True
            if not relation.is_modelled or not relation.is_range_supported:
                continue
            if relation.target_occurrence_id == target.occurrence_id:
                matched = _range_matches(relation.version_range, target_candidate.version)
                if matched is not True:
                    return matched
        elif (
            relation.source_occurrence_id == target.occurrence_id
            and relation.source_candidate_version == target_candidate.version
            and package_matches_source
        ):
            candidate_requirement_found = True
            if not relation.is_modelled or not relation.is_range_supported:
                continue
            if relation.target_occurrence_id == source.occurrence_id:
                matched = _range_matches(relation.version_range, source_candidate.version)
                if matched is not True:
                    return matched

    ranges: list[tuple[str, str]] = []
    if edge_range and not candidate_requirement_found and not override_target:
        ranges.append((edge_range, target_candidate.version))
    if peer is not None and not candidate_requirement_found:
        ranges.append((peer.version_range, target_candidate.version))
    for requirement, version in ranges:
        matched = _range_matches(requirement, version)
        if matched is None:
            return None
        if not matched:
            return False
    return True


def _constraint_pairs(
    source: SolverTarget,
    target: SolverTarget,
    source_candidates: Sequence[SolverVersionCandidate],
    target_candidates: Sequence[SolverVersionCandidate],
    *,
    edge_range: str | None,
    peer: SolverPeerConstraint | None,
    diagnostics: list[str],
    candidate_relations: Sequence[SolverCandidateRelation] = (),
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
                candidate_relations=candidate_relations,
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
    candidate_relations: Sequence[SolverCandidateRelation],
    floor: str | None = None,
) -> list[str]:
    """Return candidate versions compatible with the selected assignment."""
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
                    candidate_relations=candidate_relations,
                )
                if source_id == target.occurrence_id
                else _pair_allowed(
                    neighbor,
                    target,
                    neighbor_candidate,
                    candidate,
                    edge_range=edge_range,
                    peer=peer,
                    candidate_relations=candidate_relations,
                )
            )
            if allowed is not True:
                is_compatible = False
                break
        if is_compatible:
            compatible.append(candidate.version)
    return compatible


def _assignment_value(solver_or_snapshot: Any, variable: Any) -> int:
    """Read one integer variable value from a live solver or saved assignment."""
    if isinstance(solver_or_snapshot, Mapping):
        return int(solver_or_snapshot[int(variable.Index())])
    return int(solver_or_snapshot.Value(variable))


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
    candidate_relations: Sequence[SolverCandidateRelation],
    diagnostics: Sequence[str],
) -> tuple[SolverCandidatePlan, dict[str, int]]:
    """Project one CP-SAT assignment into the immutable candidate contract."""
    selected_indices = {
        occurrence_id: _assignment_value(solver, variable)
        for occurrence_id, variable in vars_by_id.items()
    }
    covered_values = {
        finding_id: bool(_assignment_value(solver, value))
        for finding_id, value in findings_covered.items()
    }
    workaround_values = {
        finding_id: bool(_assignment_value(solver, value))
        for finding_id, value in findings_workaround.items()
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
            candidate_relations=candidate_relations,
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
    coverage_ids = sorted(coverage_id for coverage_id, value in covered_values.items() if value)
    unresolved_ids = sorted(
        coverage_id for coverage_id, value in covered_values.items() if not value
    )
    unresolved_critical_high = sum(
        severity_weight.get(finding.severity.upper(), 0)
        for finding in finding_list
        if finding.coverage_id in unresolved_ids
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
        selected_candidate_versions={
            occurrence_id: domains[occurrence_id][candidate_index].version
            for occurrence_id, candidate_index in sorted(selected_indices.items())
        },
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
    requirements.sort(key=lambda item: item.coverage_id)
    selected = candidates[selected_index] if 0 <= selected_index < len(candidates) else None
    all_by_version = bool(requirements) and all(
        finding_covered.get(item.coverage_id, False)
        and not finding_workaround.get(item.coverage_id, False)
        for item in requirements
    )
    all_by_workaround = bool(requirements) and all(
        finding_workaround.get(item.coverage_id, False) for item in requirements
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
            if strategy == "code_workaround" and finding_workaround.get(finding.coverage_id, False)
        }
    )
    stage = requirements[0].strategy_stage if requirements else "osv_minimum"
    selected_version = (
        selected.version if selected is not None and strategy == "version_bump" else None
    )
    installed_major = _stable_semver_major(target.installed_version)
    selected_major = _stable_semver_major(selected_version)
    requires_source_migration = bool(
        selected_version
        and (installed_major is None or selected_major is None or installed_major != selected_major)
    )
    exact_instruction = None
    if requires_source_migration:
        exact_instruction = (
            f"Upgrade package {target.target_package_name} from installed version "
            f"{target.installed_version} to selected version {selected_version}.\n"
            "Migrate all affected production and test code to the selected package API "
            "while preserving behavior; do not change the solver-approved package or version."
        )
    return SolverTaskDecision(
        task_id=target.task_id,
        selected_strategy=strategy,
        selected_route=route,
        selected_version=selected_version,
        allowed_alternative_versions=alternatives,
        allowed_dependency_types=[target.dependency_type],
        strategy_stage=stage,
        selected_plan_issue_ids=selected_plan_ids,
        instruction_source="deterministic_solver",
        exact_instruction=exact_instruction,
        requires_source_migration=requires_source_migration,
        installed_version=target.installed_version,
        target_occurrence_id=target.occurrence_id,
        target_group_id=target.group_id,
        target_package_name=target.target_package_name,
        manifest_path=target.manifest_path,
        lockfile_package_key=target.lockfile_package_key,
        dependency_type=target.dependency_type,
    )


def _validate_forbidden_assignments(
    assignments: Sequence[Mapping[str, str]],
    eligible: Sequence[SolverTarget],
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
) -> tuple[list[dict[str, str]], list[str]]:
    """Validate exact no-goods against the current complete variable set."""
    expected_ids = {target.occurrence_id for target in eligible}
    normalized: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    diagnostics: list[str] = []
    for index, assignment in enumerate(assignments):
        if not isinstance(assignment, Mapping):
            diagnostics.append(f"forbidden assignment {index} is not a mapping")
            continue
        if set(assignment) != expected_ids:
            diagnostics.append(
                f"forbidden assignment {index} must specify exactly the eligible occurrence IDs"
            )
            continue
        values: dict[str, str] = {}
        malformed = False
        for occurrence_id in sorted(expected_ids):
            raw_version = assignment.get(occurrence_id)
            if not isinstance(raw_version, str) or not raw_version.strip():
                diagnostics.append(
                    f"forbidden assignment {index} has an invalid version for {occurrence_id!r}"
                )
                malformed = True
                break
            version = raw_version.strip().lstrip("vV")
            if _version_key(version) is None or version not in {
                candidate.version for candidate in domains.get(occurrence_id, ())
            }:
                diagnostics.append(
                    f"forbidden assignment {index} references an unknown version "
                    f"{version!r} for {occurrence_id!r}"
                )
                malformed = True
                break
            values[occurrence_id] = version
        if not malformed:
            normalized[tuple(sorted(values.items()))] = values
    return [normalized[key] for key in sorted(normalized)], diagnostics


def _validate_forbidden_conflicts(
    conflicts: Sequence[SolverCandidateConflict],
    eligible: Sequence[SolverTarget],
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
) -> tuple[
    list[SolverCandidateConflict],
    list[dict[str, str]],
    list[str],
]:
    """Validate rejection cuts against the current task-backed candidate catalog."""
    expected_ids = {target.occurrence_id for target in eligible}
    canonical: dict[str, SolverCandidateConflict] = {}
    exact_assignments: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
    diagnostics: list[str] = []
    for conflict_index, raw_conflict in enumerate(conflicts):
        try:
            conflict = (
                raw_conflict
                if isinstance(raw_conflict, SolverCandidateConflict)
                else SolverCandidateConflict.model_validate(raw_conflict)
            )
        except Exception as exc:  # noqa: BLE001 - malformed input must fail closed
            diagnostics.append(f"forbidden conflict {conflict_index} is invalid: {exc}")
            continue
        literal_ids = {literal.variable_id for literal in conflict.literals}
        if len(literal_ids) != len(conflict.literals):
            diagnostics.append(
                f"forbidden conflict {conflict_index} contains duplicate variable IDs"
            )
            continue
        literal_order = [(literal.variable_id, literal.version) for literal in conflict.literals]
        if literal_order != sorted(literal_order):
            diagnostics.append(f"forbidden conflict {conflict_index} literals are not canonical")
            continue
        if not literal_ids or not literal_ids <= expected_ids:
            diagnostics.append(
                f"forbidden conflict {conflict_index} references a non-mutation variable"
            )
            continue
        malformed = False
        values: dict[str, str] = {}
        for literal in conflict.literals:
            version = literal.version
            if version != version.strip().lstrip("vV") or version not in {
                candidate.version for candidate in domains.get(literal.variable_id, ())
            }:
                diagnostics.append(
                    f"forbidden conflict {conflict_index} references an unknown version "
                    f"{version!r} for {literal.variable_id!r}"
                )
                malformed = True
                break
            values[literal.variable_id] = version
        if malformed:
            continue
        if conflict.cut_kind == SolverCandidateCutKind.UNARY and (
            len(conflict.literals) != 1
            or conflict.reason_code
            not in {
                SolverCandidateRejectionReason.RUNTIME_ENGINE,
                SolverCandidateRejectionReason.RUNTIME_PLATFORM,
            }
        ):
            diagnostics.append(f"forbidden conflict {conflict_index} has an invalid unary cut")
            continue
        if conflict.cut_kind == SolverCandidateCutKind.PAIR and (
            len(conflict.literals) != 2
            or conflict.reason_code
            not in {
                SolverCandidateRejectionReason.DEPENDENCY_RANGE,
                SolverCandidateRejectionReason.PEER_CONFLICT,
            }
        ):
            diagnostics.append(f"forbidden conflict {conflict_index} has an invalid pair cut")
            continue
        if conflict.cut_kind == SolverCandidateCutKind.EXACT_ASSIGNMENT:
            if literal_ids != expected_ids:
                diagnostics.append(
                    f"forbidden conflict {conflict_index} must specify every eligible occurrence ID"
                )
                continue
            if _digest(dict(sorted(values.items()))) != conflict.assignment_digest:
                diagnostics.append(
                    f"forbidden conflict {conflict_index} assignment digest does not match its map"
                )
                continue
            exact_assignments[tuple(sorted(values.items()))] = dict(sorted(values.items()))
            continue
        canonical_key = json.dumps(
            conflict.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        canonical[canonical_key] = conflict
    unique_cuts: dict[tuple[tuple[str, str], ...], SolverCandidateConflict] = {}
    for conflict in (canonical[key] for key in sorted(canonical)):
        cut_key = tuple((literal.variable_id, literal.version) for literal in conflict.literals)
        unique_cuts.setdefault(cut_key, conflict)
    return (
        sorted(
            unique_cuts.values(),
            key=lambda conflict: json.dumps(
                conflict.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ),
        ),
        [exact_assignments[key] for key in sorted(exact_assignments)],
        diagnostics,
    )


def _candidate_relations_complete(
    subgraph: SolverSubgraph,
    domains: Mapping[str, Sequence[SolverVersionCandidate]],
    diagnostics: list[str],
    *,
    pruned_candidate_versions: Mapping[str, set[str]] | None = None,
) -> bool:
    """Require relations for every retained candidate and its requirements."""
    by_source_candidate: dict[tuple[str, str], list[SolverCandidateRelation]] = {}
    intentionally_pruned = pruned_candidate_versions or {}
    complete = True
    for relation in subgraph.candidate_relations:
        source_candidates = domains.get(relation.source_occurrence_id, ())
        candidate = next(
            (
                value
                for value in source_candidates
                if value.version == relation.source_candidate_version
            ),
            None,
        )
        if candidate is None:
            if relation.source_candidate_version in intentionally_pruned.get(
                relation.source_occurrence_id, set()
            ):
                continue
            complete = False
            _diagnostic(
                diagnostics,
                f"candidate relation references unknown source candidate "
                f"{relation.source_occurrence_id!r}@{relation.source_candidate_version}",
            )
            continue
        if relation.target_occurrence_id == relation.source_occurrence_id:
            complete = False
            _diagnostic(
                diagnostics,
                f"candidate relation for {relation.source_occurrence_id!r} resolves to itself",
            )
        if relation.is_range_supported and _range_matches(relation.version_range, "1.0.0") is None:
            complete = False
            _diagnostic(
                diagnostics,
                f"invalid candidate range {relation.version_range!r} for "
                f"{relation.source_occurrence_id!r}@{relation.source_candidate_version}",
            )
        if not any(
            requirement.package_name == relation.package_name
            and requirement.version_range == relation.version_range
            and requirement.kind == relation.kind
            and requirement.is_optional == relation.is_optional
            and requirement.is_range_supported == relation.is_range_supported
            for requirement in candidate.requirements
        ):
            complete = False
            _diagnostic(
                diagnostics,
                f"candidate relation does not match published metadata for "
                f"{relation.source_occurrence_id!r}@{relation.source_candidate_version}",
            )
        by_source_candidate.setdefault(
            (relation.source_occurrence_id, relation.source_candidate_version), []
        ).append(relation)

    for target in subgraph.targets:
        if not target.eligible_for_atomic_update:
            continue
        candidates = domains.get(target.occurrence_id, ())
        for candidate in candidates:
            relations = by_source_candidate.get((target.occurrence_id, candidate.version), ())
            for requirement in candidate.requirements:
                matches = [
                    relation
                    for relation in relations
                    if relation.package_name == requirement.package_name
                    and relation.version_range == requirement.version_range
                    and relation.kind == requirement.kind
                    and relation.is_optional == requirement.is_optional
                    and relation.is_range_supported == requirement.is_range_supported
                ]
                if len(matches) != 1:
                    complete = False
                    _diagnostic(
                        diagnostics,
                        f"incomplete candidate relation for {target.occurrence_id!r}@"
                        f"{candidate.version} -> {requirement.package_name!r}",
                    )
    return complete


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
    findings_covered: Mapping[str, Any],
    findings_workaround: Mapping[str, Any],
    metrics: Sequence[tuple[str, Any, bool]],
    top_k: int,
    timeout_seconds: float,
    accepted_feasible: bool,
    project: Callable[[Any, int, SolverStatus], tuple[SolverCandidatePlan, dict[str, int]]],
    observations: list[dict[str, Any]],
    diagnostics: list[str],
) -> tuple[list[SolverCandidatePlan], SolverStatus | None]:
    """Enumerate lexicographic assignments and retain incumbents across stages.

    If a later objective stage becomes infeasible or times out after an earlier
    stage produced an assignment, restore that model snapshot, constrain the
    failed objective to be no worse than the saved assignment, and continue
    with lower-priority metrics. Such a recovered candidate is feasible but
    cannot be reported as globally optimal.
    """
    candidate_plans: list[SolverCandidatePlan] = []
    forbidden_assignments: list[dict[str, int]] = []
    primary_status: SolverStatus | None = None
    deadline = time.monotonic() + max(0.001, timeout_seconds)
    ordered_occurrences = sorted(vars_by_id)

    def stage_diagnostic(message: str) -> None:
        """Retain stage diagnostics even when earlier diagnostics filled the cap."""
        message = str(message).strip()[:_MAX_DIAGNOSTIC_LENGTH]
        if not message or message in diagnostics:
            return
        if len(diagnostics) >= _MAX_DIAGNOSTICS:
            diagnostics.pop()
        diagnostics.append(message)

    def snapshot_values(solver_instance: Any) -> dict[int, int]:
        """Capture every value needed to project a candidate plan later."""
        variables = [
            *vars_by_id.values(),
            *findings_covered.values(),
            *findings_workaround.values(),
        ]
        return {
            int(variable.Index()): int(solver_instance.Value(variable))
            for variable in variables
        }

    def read_metric(solver_instance: Any, expression: Any) -> int:
        """Read an objective expression, including constant-only metrics."""
        try:
            return int(solver_instance.Value(expression))
        except (AttributeError, TypeError, ValueError, OverflowError):
            return int(expression)

    def metric_values(solver_instance: Any) -> dict[str, int]:
        """Evaluate every lexicographic objective on one saved assignment."""
        return {
            name: read_metric(solver_instance, expression)
            for name, expression, _maximize in metrics
        }

    def add_no_worse_bound(model: Any, expression: Any, maximize: bool, value: int) -> None:
        """Preserve an incumbent metric without requiring an exact lock."""
        if maximize:
            model.Add(expression >= value)
        else:
            model.Add(expression <= value)

    for alternative_index in range(top_k):
        model = base_model.Clone()
        for assignment in forbidden_assignments:
            model.AddForbiddenAssignments(
                [vars_by_id[occurrence_id] for occurrence_id in ordered_occurrences],
                [[assignment[occurrence_id] for occurrence_id in ordered_occurrences]],
            )
        candidate_status = SolverStatus.OPTIMAL
        latest_assignment: dict[int, int] | None = None
        latest_metric_values: dict[str, int] = {}
        latest_model_snapshot: Any | None = None
        recovered_stage = False
        abandon_alternative = False

        for stage_index, (metric_name, expression, maximize) in enumerate(metrics):
            remaining = deadline - time.monotonic()
            model.ClearObjective()
            if maximize:
                model.Maximize(expression)
            else:
                model.Minimize(expression)

            if remaining <= 0:
                status = SolverStatus.UNKNOWN
                stage_wall_time = 0.0
                objective_value = None
                best_objective_bound = None
                stage_diagnostic(
                    f"lexicographic candidate {alternative_index + 1} stage "
                    f"{stage_index + 1}/{len(metrics)} '{metric_name}' timed out before solve",
                )
            else:
                solver.parameters.max_time_in_seconds = max(0.001, remaining)
                status_code = solver.Solve(model)
                status = _record_solver_result(cp_model, solver, status_code, observations)
                observation = observations[-1]
                stage_wall_time = float(observation.get("wall_time_seconds") or 0.0)
                objective_value = observation.get("objective_value")
                best_objective_bound = observation.get("best_objective_bound")

            direction = "maximize" if maximize else "minimize"
            value_text = "n/a"
            if status in {SolverStatus.OPTIMAL, SolverStatus.FEASIBLE}:
                stage_metric_value = read_metric(solver, expression)
                value_text = str(stage_metric_value)
            stage_diagnostic(
                f"lexicographic candidate {alternative_index + 1} stage "
                f"{stage_index + 1}/{len(metrics)} '{metric_name}' {direction}: "
                f"{status.value}; metric={value_text}; "
                f"objective={objective_value if objective_value is not None else 'n/a'}; "
                f"bound={best_objective_bound if best_objective_bound is not None else 'n/a'}; "
                f"wall={stage_wall_time:.3f}s",
            )

            if status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}:
                if latest_assignment is None:
                    if candidate_plans:
                        stage_diagnostic(
                            "no further lexicographic candidate is feasible; retaining "
                            "the completed candidate plan",
                        )
                        abandon_alternative = True
                        break
                    stage_diagnostic(
                        f"lexicographic stage '{metric_name}' failed before any "
                        "candidate assignment was available",
                    )
                    return candidate_plans, status
                if candidate_plans:
                    stage_diagnostic(
                        f"lexicographic candidate {alternative_index + 1} stopped at "
                        f"stage '{metric_name}'; retaining previously completed candidate(s)",
                    )
                    abandon_alternative = True
                    break
                if not accepted_feasible:
                    stage_diagnostic(
                        "lexicographic recovery requires a FEASIBLE plan, but "
                        "solver_accept_feasible is disabled",
                    )
                    return candidate_plans, SolverStatus.UNKNOWN

                # The most recent saved assignment is a witness that this
                # restored model remains feasible. Preserve its value for the
                # failed objective, then continue with lower-priority metrics.
                fallback_value = latest_metric_values[metric_name]
                model = latest_model_snapshot.Clone()
                add_no_worse_bound(model, expression, maximize, fallback_value)
                latest_model_snapshot = model.Clone()
                candidate_status = SolverStatus.FEASIBLE
                recovered_stage = True
                bound = ">=" if maximize else "<="
                stage_diagnostic(
                    f"restored candidate snapshot after stage '{metric_name}' returned "
                    f"{status.value}; preserving {metric_name} {bound} {fallback_value} "
                    "and continuing with the next objective",
                )
                continue

            if status == SolverStatus.FEASIBLE:
                if not accepted_feasible:
                    stage_diagnostic(
                        "feasible lexicographic stage rejected by "
                        "solver_accept_feasible=false",
                    )
                    if candidate_plans:
                        abandon_alternative = True
                        break
                    return candidate_plans, SolverStatus.UNKNOWN
                candidate_status = SolverStatus.FEASIBLE

            latest_assignment = snapshot_values(solver)
            latest_metric_values = metric_values(solver)
            metric_value = latest_metric_values[metric_name]
            if status == SolverStatus.OPTIMAL:
                # Exact locks preserve a proven lexicographic optimum.
                model.Add(expression == metric_value)
            else:
                # An unproven incumbent must not worsen while lower-priority
                # metrics are optimized.
                add_no_worse_bound(model, expression, maximize, metric_value)
            latest_model_snapshot = model.Clone()

        if abandon_alternative:
            break
        if latest_assignment is None:
            # All metrics are present in the model, so this is defensive only.
            if candidate_plans:
                break
            return candidate_plans, SolverStatus.UNKNOWN
        if recovered_stage:
            candidate_status = SolverStatus.FEASIBLE
        if primary_status is None:
            primary_status = candidate_status
        candidate, selected_indices = project(
            latest_assignment, alternative_index, candidate_status
        )
        candidate_plans.append(candidate)
        if not ordered_occurrences:
            break
        forbidden_assignments.append(selected_indices)
        if recovered_stage or candidate_status == SolverStatus.FEASIBLE:
            # Once a candidate required fallback recovery, later alternatives
            # cannot improve the certified primary assignment reliably.
            break
    return candidate_plans, primary_status


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
    candidate_catalog_complete: bool = True,
    candidate_catalog_digest: str | None = None,
    forbidden_assignments: Sequence[Mapping[str, str]] = (),
    forbidden_conflicts: Sequence[SolverCandidateConflict] = (),
) -> SolverRemediationPlan:
    """Solve one occurrence-aware portfolio with deterministic CP-SAT.

    Args:
        subgraph: Immutable target, finding, peer, and dependency graph.
        candidate_domains: Bounded candidates keyed by occurrence ID (task ID is
            accepted as a compatibility key).
        settings: Explicit AppSettings-like object. No environment variables are
            read by this function.
        forbidden_assignments: Complete mutation maps excluded by exact no-goods.
        forbidden_conflicts: Structured unary, pair, or exact assignment evidence.

    Returns:
        A typed plan with top-K deterministic assignments, or a typed failure
        status. Native solver errors and model guards return ``FALLBACK`` and
        never include a dispatchable selected plan.
    """
    diagnostics = list(subgraph.diagnostics)
    targets = sorted(subgraph.targets, key=lambda item: item.occurrence_id)
    findings = {finding.coverage_id: finding for finding in subgraph.findings}
    finding_list = sorted(subgraph.findings, key=lambda item: item.coverage_id)
    issue_finding_ids = sorted({finding.finding_id for finding in finding_list})
    max_candidates = max(
        1,
        int(
            _setting(
                settings,
                "solver_max_candidates_per_target",
                DEFAULT_SOLVER_MAX_CANDIDATES_PER_TARGET,
            )
        ),
    )
    max_variables = max(1, int(_setting(settings, "solver_max_model_variables", 10_000)))
    floors = {
        target.occurrence_id: _floor_for_target(target, findings, diagnostics) for target in targets
    }
    eligible = [target for target in targets if target.eligible_for_atomic_update]
    domains: dict[str, list[SolverVersionCandidate]] = {}
    pruned_candidate_versions: dict[str, set[str]] = {}
    for target in eligible:
        raw_domain = candidate_domains.get(target.occurrence_id)
        if raw_domain is None:
            raw_domain = candidate_domains.get(target.task_id, ())
        raw_versions = {
            candidate.version
            for candidate in raw_domain or ()
            if isinstance(candidate, SolverVersionCandidate)
        }
        domain = _as_domain(
            target,
            candidate_domains,
            max_candidates=max_candidates,
            security_floor=floors[target.occurrence_id],
            diagnostics=diagnostics,
        )
        domains[target.occurrence_id] = domain
        removed_versions = raw_versions - {candidate.version for candidate in domain}
        if removed_versions:
            pruned_candidate_versions[target.occurrence_id] = removed_versions
    validated_conflicts, exact_conflict_assignments, conflict_diagnostics = (
        _validate_forbidden_conflicts(forbidden_conflicts, eligible, domains)
    )
    normalized_forbidden, forbidden_diagnostics = _validate_forbidden_assignments(
        [*forbidden_assignments, *exact_conflict_assignments], eligible, domains
    )
    diagnostics.extend([*forbidden_diagnostics, *conflict_diagnostics])
    candidate_relations_complete = _candidate_relations_complete(
        subgraph,
        domains,
        diagnostics,
        pruned_candidate_versions=pruned_candidate_versions,
    )
    candidate_catalog_complete = bool(candidate_catalog_complete)
    for target in targets:
        if target.eligible_for_atomic_update and not domains.get(target.occurrence_id):
            _diagnostic(
                diagnostics, f"no candidates for eligible occurrence {target.occurrence_id}"
            )
    mutation_domain_payload = {
        occurrence_id: [candidate.model_dump(mode="json") for candidate in values]
        for occurrence_id, values in sorted(domains.items())
    }
    evidence_domain_payload = {
        domain.variable_id: list(domain.candidate_versions)
        for domain in sorted(subgraph.evidence_domains, key=lambda item: item.variable_id)
    }
    domain_payload = {
        "mutation_domains": mutation_domain_payload,
        "evidence_domains": evidence_domain_payload,
    }
    candidate_catalog_digest = (
        candidate_catalog_digest.strip()
        if isinstance(candidate_catalog_digest, str) and candidate_catalog_digest.strip()
        else _digest(
            {
                "candidate_domains": domain_payload,
                "complete": candidate_catalog_complete,
            }
        )
    )
    input_digest = _digest(
        {
            "subgraph": subgraph.model_dump(mode="json"),
            "candidate_catalog_complete": candidate_catalog_complete,
            "candidate_catalog_digest": candidate_catalog_digest,
            "forbidden_assignments": normalized_forbidden,
            "forbidden_assignment_diagnostics": forbidden_diagnostics,
            "forbidden_conflicts": [
                conflict.model_dump(mode="json") for conflict in validated_conflicts
            ],
            "forbidden_conflict_diagnostics": conflict_diagnostics,
            "candidate_relations_complete": candidate_relations_complete,
        }
    )
    domain_digest = _digest(
        {
            "domains": domain_payload,
            "candidate_catalog_complete": candidate_catalog_complete,
            "candidate_catalog_digest": candidate_catalog_digest,
        }
    )
    repository_digest = _digest(
        {
            "targets": [target.model_dump(mode="json") for target in targets],
            "occurrences": [
                occurrence.model_dump(mode="json") for occurrence in subgraph.occurrences
            ],
            "edges": [edge.model_dump(mode="json") for edge in subgraph.edges],
            "candidate_relations": [
                relation.model_dump(mode="json") for relation in subgraph.candidate_relations
            ],
            "evidence_domains": [
                domain.model_dump(mode="json") for domain in subgraph.evidence_domains
            ],
        }
    )
    revisions = {target.task_id: 0 for target in targets}

    def make_plan(**values: Any) -> SolverRemediationPlan:
        """Attach immutable candidate-catalog evidence to every solver result."""
        return SolverRemediationPlan(
            candidate_catalog_complete=candidate_catalog_complete,
            candidate_catalog_digest=candidate_catalog_digest,
            **values,
        )

    if (
        not getattr(subgraph, "valid", True)
        or not candidate_relations_complete
        or forbidden_diagnostics
        or conflict_diagnostics
    ):
        if not getattr(subgraph, "valid", True):
            _diagnostic(diagnostics, "invalid solver subgraph; no dispatchable plan")
        if not candidate_relations_complete:
            _diagnostic(diagnostics, "candidate dependency relations are incomplete")
        if forbidden_diagnostics or conflict_diagnostics:
            _diagnostic(diagnostics, "invalid forbidden-assignment or conflict input")
        return make_plan(
            status=SolverStatus.UNKNOWN,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=issue_finding_ids,
            diagnostics=diagnostics,
        )

    # Import lazily so importing the contracts and CLI remains possible in an
    # environment where the production dependency has not yet been installed.
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:  # pragma: no cover - exercised by packaging failures
        _diagnostic(diagnostics, f"CP-SAT import failed: {exc}")
        return make_plan(
            status=SolverStatus.FALLBACK,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=issue_finding_ids,
            diagnostics=diagnostics,
        )

    # One integer variable per eligible occurrence, plus bounded coverage and
    # workaround booleans. Empty domains are a typed infeasible result.
    if any(not domains.get(target.occurrence_id) for target in eligible):
        return make_plan(
            status=SolverStatus.INFEASIBLE if candidate_catalog_complete else SolverStatus.UNKNOWN,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=issue_finding_ids,
            diagnostics=diagnostics,
        )
    base_estimated_variables = (
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
    evidence_domains_by_id = {domain.variable_id: domain for domain in subgraph.evidence_domains}
    nonempty_evidence_domains = sum(
        bool(domain.candidate_versions) for domain in evidence_domains_by_id.values()
    )
    evidence_model_enabled = True
    if base_estimated_variables + nonempty_evidence_domains > max_variables:
        evidence_model_enabled = False
        _diagnostic(
            diagnostics,
            "evidence-only model variable guard exceeded; required dependency relations "
            "left unmodeled without pruning mutation candidates",
        )
    estimated_variables = base_estimated_variables + (
        nonempty_evidence_domains if evidence_model_enabled else 0
    )
    if estimated_variables > max_variables:
        _diagnostic(
            diagnostics,
            f"solver model variable guard exceeded: {estimated_variables}>{max_variables}",
        )
        return make_plan(
            status=SolverStatus.UNKNOWN,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=issue_finding_ids,
            diagnostics=diagnostics,
        )

    target_by_id = {target.occurrence_id: target for target in targets}
    evidence_by_id = {occurrence.occurrence_id: occurrence for occurrence in subgraph.occurrences}
    vars_by_id: dict[str, Any] = {}
    finding_by_target: dict[str, list[SolverFindingRequirement]] = {}
    for finding in finding_list:
        finding_by_target.setdefault(finding.target_occurrence_id, []).append(finding)
    model = cp_model.CpModel()

    def installed_index(occurrence_id: str) -> int | None:
        """Return the status-quo candidate index for one mutation target."""
        target = target_by_id.get(occurrence_id)
        if target is None or not target.installed_version:
            return None
        installed_version = target.installed_version.strip().lstrip("vV")
        return next(
            (
                index
                for index, candidate in enumerate(domains.get(occurrence_id, ()))
                if candidate.version == installed_version
            ),
            None,
        )

    for target in eligible:
        values = domains[target.occurrence_id]
        target_findings = [
            finding.coverage_id
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
        has_floor_candidate = bool(allowed)
        if (
            not has_floor_candidate
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
        elif not has_floor_candidate:
            baseline_index = installed_index(target.occurrence_id)
            if baseline_index is not None:
                # Keep the status quo available when there is no secure release.
                allowed = [baseline_index]
                _diagnostic(
                    diagnostics,
                    f"no candidate meets the security floor for {target.occurrence_id}; "
                    "retaining the installed version",
                )
        else:
            baseline_index = installed_index(target.occurrence_id)
            if baseline_index is not None and baseline_index not in allowed:
                # Keep the status quo available as a portfolio escape hatch when
                # otherwise-safe upgrades conflict with another package constraint.
                allowed.append(baseline_index)
        if not allowed:
            _diagnostic(
                diagnostics, f"security floor removes every candidate for {target.occurrence_id}"
            )
            return make_plan(
                status=SolverStatus.INFEASIBLE,
                input_digest=input_digest,
                domain_digest=domain_digest,
                repository_digest=repository_digest,
                task_revisions=revisions,
                unresolved_finding_ids=issue_finding_ids,
                diagnostics=diagnostics,
            )
        vars_by_id[target.occurrence_id] = model.NewIntVarFromDomain(
            cp_model.Domain.FromValues(allowed), f"occurrence_{len(vars_by_id):04d}"
        )

    def add_resilient_pair_constraint(
        left_id: str,
        right_id: str,
        pairs: Sequence[tuple[int, int]],
        diagnostic: str,
    ) -> None:
        """Constrain compatible candidates while preserving the baseline pair."""
        normalized_pairs = set(pairs)
        left_baseline = installed_index(left_id)
        right_baseline = installed_index(right_id)
        if not pairs and (left_baseline is None or right_baseline is None):
            _diagnostic(
                diagnostics,
                f"{diagnostic}; baseline unavailable, leaving peer relation relaxed",
            )
            return
        if left_baseline is not None and right_baseline is not None:
            normalized_pairs.add((left_baseline, right_baseline))
        if not pairs:
            _diagnostic(diagnostics, diagnostic)

        allowed_left = {left for left, _right in pairs}
        allowed_right = {right for _left, right in pairs}
        if left_baseline is not None:
            allowed_left.add(left_baseline)
        if right_baseline is not None:
            allowed_right.add(right_baseline)
        for index, _candidate in enumerate(domains[left_id]):
            if index not in allowed_left:
                model.Add(vars_by_id[left_id] != index)
        for index, _candidate in enumerate(domains[right_id]):
            if index not in allowed_right:
                model.Add(vars_by_id[right_id] != index)

        if normalized_pairs:
            model.AddAllowedAssignments(
                [vars_by_id[left_id], vars_by_id[right_id]], sorted(normalized_pairs)
            )

    evidence_vars_by_id: dict[str, Any] = {}
    if evidence_model_enabled:
        for evidence_index, evidence_domain in enumerate(
            sorted(subgraph.evidence_domains, key=lambda item: item.variable_id)
        ):
            if not evidence_domain.candidate_versions:
                continue
            evidence_vars_by_id[evidence_domain.variable_id] = model.NewIntVar(
                0,
                len(evidence_domain.candidate_versions) - 1,
                f"evidence_{evidence_index:04d}",
            )
    ordered_occurrences = sorted(vars_by_id)
    for assignment in normalized_forbidden:
        if ordered_occurrences:
            indices = [
                next(
                    index
                    for index, candidate in enumerate(domains[occurrence_id])
                    if candidate.version == assignment[occurrence_id]
                )
                for occurrence_id in ordered_occurrences
            ]
            model.AddForbiddenAssignments(
                [vars_by_id[occurrence_id] for occurrence_id in ordered_occurrences],
                [indices],
            )
    seen_conflict_tuples: set[tuple[tuple[str, str], ...]] = set()
    for conflict in validated_conflicts:
        literal_key = tuple((literal.variable_id, literal.version) for literal in conflict.literals)
        if literal_key in seen_conflict_tuples:
            continue
        seen_conflict_tuples.add(literal_key)
        literal_variables = [vars_by_id[literal.variable_id] for literal in conflict.literals]
        literal_indices = [
            next(
                index
                for index, candidate in enumerate(domains[literal.variable_id])
                if candidate.version == literal.version
            )
            for literal in conflict.literals
        ]
        model.AddForbiddenAssignments(literal_variables, [literal_indices])

    # Peer constraints and range-bearing graph edges are the solver's hard
    # package compatibility requirements. Range-free workspace and scope edges
    # remain useful for grouping and ordering, but adding them as compatibility
    # tables would materialize every pair in two candidate domains even though
    # every pair is allowed.
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
        if edge.edge_kind in {"workspace", "scope", "peer", "strict_peer", "peer_conflict"}:
            peer = peer_by_pair.get((edge.source_occurrence_id, edge.target_occurrence_id))
            if not edge.version_range and peer is None:
                continue
            relations.append(
                (
                    edge.source_occurrence_id,
                    edge.target_occurrence_id,
                    edge.version_range,
                    peer,
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
            candidate_relations=subgraph.candidate_relations,
        )
        add_resilient_pair_constraint(
            source_id,
            target_id,
            pairs,
            f"no compatible candidate pair for {source_id}->{target_id}; "
            "retaining installed baseline",
        )
    for relation in subgraph.candidate_relations:
        if (
            relation.is_optional
            or not relation.is_modelled
            or not relation.is_range_supported
            or relation.target_occurrence_id is None
            or relation.source_occurrence_id not in vars_by_id
            or relation.target_occurrence_id not in vars_by_id
        ):
            continue
        source_id = relation.source_occurrence_id
        target_id = relation.target_occurrence_id
        pairs: list[tuple[int, int]] = []
        invalid_range = False
        for source_index, source_candidate in enumerate(domains[source_id]):
            for target_index, target_candidate in enumerate(domains[target_id]):
                if source_candidate.version != relation.source_candidate_version:
                    pairs.append((source_index, target_index))
                    continue
                matched = _range_matches(relation.version_range, target_candidate.version)
                if matched is None:
                    invalid_range = True
                    break
                if matched:
                    pairs.append((source_index, target_index))
            if invalid_range:
                break
        if invalid_range:
            _diagnostic(
                diagnostics,
                f"invalid candidate range {relation.version_range!r} in relation "
                f"{source_id}->{target_id}",
            )
            return make_plan(
                status=SolverStatus.UNKNOWN,
                input_digest=input_digest,
                domain_digest=domain_digest,
                repository_digest=repository_digest,
                task_revisions=revisions,
                unresolved_finding_ids=issue_finding_ids,
                diagnostics=diagnostics,
            )
        add_resilient_pair_constraint(
            source_id,
            target_id,
            pairs,
            f"no compatible candidate pair for {source_id}->{target_id}; "
            "retaining installed baseline",
        )
    for relation in subgraph.candidate_relations:
        if (
            relation.is_optional
            or not relation.is_modelled
            or not relation.is_range_supported
            or relation.evidence_variable_id is None
            or relation.source_occurrence_id not in vars_by_id
            or not evidence_model_enabled
        ):
            continue
        source_id = relation.source_occurrence_id
        source_variable = vars_by_id[source_id]
        evidence_domain = evidence_domains_by_id[relation.evidence_variable_id]
        if not evidence_domain.candidate_versions:
            _diagnostic(
                diagnostics,
                f"dependency witness {relation.package_name!r} is unmodeled; "
                f"relaxed constraint for {source_id}@{relation.source_candidate_version}",
            )
            continue
        evidence_variable = evidence_vars_by_id.get(relation.evidence_variable_id)
        if evidence_variable is None:
            return make_plan(
                status=SolverStatus.UNKNOWN,
                input_digest=input_digest,
                domain_digest=domain_digest,
                repository_digest=repository_digest,
                task_revisions=revisions,
                unresolved_finding_ids=issue_finding_ids,
                diagnostics=[*diagnostics, "evidence variable map is incomplete"],
            )
        allowed_pairs: list[tuple[int, int]] = []
        invalid_range = False
        for source_index, source_candidate in enumerate(domains[source_id]):
            for evidence_index, version in enumerate(evidence_domain.candidate_versions):
                if source_candidate.version != relation.source_candidate_version:
                    allowed_pairs.append((source_index, evidence_index))
                    continue
                matched = _range_matches(relation.version_range, version)
                if matched is None:
                    invalid_range = True
                    break
                if matched:
                    allowed_pairs.append((source_index, evidence_index))
            if invalid_range:
                break
        if invalid_range:
            _diagnostic(
                diagnostics,
                f"invalid modeled evidence range {relation.version_range!r} for "
                f"{source_id}->{relation.package_name}",
            )
            return make_plan(
                status=SolverStatus.UNKNOWN,
                input_digest=input_digest,
                domain_digest=domain_digest,
                repository_digest=repository_digest,
                task_revisions=revisions,
                unresolved_finding_ids=issue_finding_ids,
                diagnostics=diagnostics,
            )
        source_baseline = installed_index(source_id)
        if source_baseline is not None:
            allowed_pairs.extend(
                (source_baseline, evidence_index)
                for evidence_index in range(len(evidence_domain.candidate_versions))
            )
        if allowed_pairs:
            model.AddAllowedAssignments(
                [source_variable, evidence_variable], sorted(set(allowed_pairs))
            )
        elif source_baseline is not None:
            source_target = target_by_id[source_id]
            installed_version = str(source_target.installed_version or "").strip().lstrip("vV")
            for source_index, source_candidate in enumerate(domains[source_id]):
                if (
                    source_candidate.version == relation.source_candidate_version
                    and source_candidate.version != installed_version
                ):
                    model.Add(source_variable != source_index)
            _diagnostic(
                diagnostics,
                f"dependency witness {relation.package_name!r} has no compatible release; "
                f"disqualified non-baseline candidate {relation.source_candidate_version!r}",
            )
        else:
            _diagnostic(
                diagnostics,
                f"dependency witness {relation.package_name!r} has no compatible release; "
                "constraint left relaxed because no installed baseline is available",
            )

    for relation in subgraph.candidate_relations:
        if (
            relation.kind != "peer"
            or relation.is_optional
            or not relation.is_modelled
            or not relation.is_range_supported
            or relation.target_occurrence_id is None
            or relation.target_occurrence_id in vars_by_id
            or relation.source_occurrence_id not in vars_by_id
        ):
            continue
        physical_peer = evidence_by_id.get(relation.target_occurrence_id)
        if physical_peer is None or not physical_peer.installed_version:
            _diagnostic(
                diagnostics,
                f"modeled peer {relation.package_name!r} has no physical version; "
                "peer constraint left unmodeled",
            )
            continue
        source_target = target_by_id[relation.source_occurrence_id]
        installed_version = str(source_target.installed_version or "").strip().lstrip("vV")
        if installed_index(relation.source_occurrence_id) is None:
            _diagnostic(
                diagnostics,
                f"installed baseline is unavailable for {relation.source_occurrence_id}; "
                f"physical peer {relation.package_name!r} left unmodeled",
            )
            continue
        for source_index, source_candidate in enumerate(domains[relation.source_occurrence_id]):
            if source_candidate.version != relation.source_candidate_version:
                continue
            matched = _range_matches(relation.version_range, physical_peer.installed_version)
            if matched is None:
                return make_plan(
                    status=SolverStatus.UNKNOWN,
                    input_digest=input_digest,
                    domain_digest=domain_digest,
                    repository_digest=repository_digest,
                    task_revisions=revisions,
                    unresolved_finding_ids=issue_finding_ids,
                    diagnostics=[
                        *diagnostics,
                        f"invalid modeled peer range {relation.version_range!r}",
                    ],
                )
            if matched:
                continue
            if source_candidate.version == installed_version:
                _diagnostic(
                    diagnostics,
                    f"installed baseline {installed_version!r} conflicts with physical peer "
                    f"{relation.package_name!r}; retaining baseline fallback",
                )
                continue
            model.Add(vars_by_id[relation.source_occurrence_id] != source_index)

    findings_covered: dict[str, Any] = {}
    findings_version: dict[str, Any] = {}
    findings_workaround: dict[str, Any] = {}
    coverage_meta: dict[str, tuple[list[int], str | None]] = {}
    for finding in finding_list:
        covered = model.NewBoolVar(f"finding_covered_{finding.coverage_id}")
        workaround = model.NewBoolVar(f"finding_workaround_{finding.coverage_id}")
        version_bool = model.NewBoolVar(f"finding_version_{finding.coverage_id}")
        findings_covered[finding.coverage_id] = covered
        findings_workaround[finding.coverage_id] = workaround
        findings_version[finding.coverage_id] = version_bool
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
                literal = model.NewBoolVar(f"finding_{finding.coverage_id}_version_{index}")
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
        if preferred_strategy in {"code_workaround", "workaround", "no_fix"}:
            model.Add(version_bool == 0)
        if not has_workaround or preferred_strategy == "no_fix":
            model.Add(workaround == 0)
        elif preferred_strategy in {"code_workaround", "workaround"}:
            # A workaround target is only dispatchable when every selected
            # workaround carries the plan IDs that authorize it.
            model.Add(workaround == 1)
        model.AddMaxEquality(covered, [version_bool, workaround])
        coverage_meta[finding.coverage_id] = (good_indices, finding.fixed_version)

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
        * (1 - findings_covered[finding.coverage_id])
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
        ("coverage", coverage_expression, True),
        ("unresolved", unresolved_expression, False),
        ("workaround", workaround_expression, False),
        ("changed", changed_expression, False),
        ("distance", distance_expression, False),
        ("stability", stable_expression, False),
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
            candidate_relations=subgraph.candidate_relations,
            diagnostics=diagnostics,
        )

    try:
        if weights_scaled:
            candidate_plans, first_status = _solve_lexicographic(
                cp_model,
                base_model,
                solver,
                vars_by_id=vars_by_id,
                findings_covered=findings_covered,
                findings_workaround=findings_workaround,
                metrics=metrics,
                top_k=top_k,
                timeout_seconds=float(_setting(settings, "solver_timeout_seconds", 10)),
                accepted_feasible=accepted_feasible,
                project=project,
                observations=observations,
                diagnostics=diagnostics,
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
                if status in {SolverStatus.INFEASIBLE, SolverStatus.UNKNOWN}:
                    if not candidate_plans:
                        solver_statistics = _build_solver_statistics(observations)
                        _add_statistics_diagnostic(diagnostics, solver_statistics)
                        return make_plan(
                            status=status,
                            input_digest=input_digest,
                            domain_digest=domain_digest,
                            repository_digest=repository_digest,
                            task_revisions=revisions,
                            unresolved_finding_ids=issue_finding_ids,
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
                    return make_plan(
                        status=SolverStatus.UNKNOWN,
                        input_digest=input_digest,
                        domain_digest=domain_digest,
                        repository_digest=repository_digest,
                        task_revisions=revisions,
                        unresolved_finding_ids=issue_finding_ids,
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
        return make_plan(
            status=SolverStatus.FALLBACK,
            input_digest=input_digest,
            domain_digest=domain_digest,
            repository_digest=repository_digest,
            task_revisions=revisions,
            unresolved_finding_ids=issue_finding_ids,
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
    return make_plan(
        status=final_status,
        input_digest=input_digest,
        domain_digest=domain_digest,
        repository_digest=repository_digest,
        task_revisions=revisions,
        candidate_plans=candidate_plans,
        selected_plan=selected_plan,
        unresolved_finding_ids=(
            sorted(
                {
                    finding.finding_id
                    for finding in finding_list
                    if finding.coverage_id in set(selected_plan.unresolved_ids)
                }
            )
            if selected_plan
            else issue_finding_ids
        ),
        diagnostics=diagnostics,
        solver_statistics=solver_statistics,
    )


__all__ = ["solve_portfolio"]
