"""Deterministic occurrence graph, batching, and phase scheduling.

This module contains only pure transformations over the immutable solver
contracts.  Package and task IDs are never used interchangeably: occurrence
IDs identify graph vertices while task IDs identify executable mutations.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from remediation_engine.contracts.schemas import MAX_MULTI_PACKAGE_ACTION_SIZE
from remediation_engine.contracts.solver_models import (
    DAGBuildResult,
    SolverBatch,
    SolverEdge,
    SolverFindingRequirement,
    SolverMutation,
    SolverPhase,
    SolverSubgraph,
    SolverTarget,
    SolverTaskDecision,
)

_HARD_ATOMIC_OVERFLOW_REASON = (
    f"hard atomic component exceeds {MAX_MULTI_PACKAGE_ACTION_SIZE} tasks"
)
_SOFT_PARTITION_REASON = (
    f"soft coupling component partitioned at {MAX_MULTI_PACKAGE_ACTION_SIZE} tasks"
)

# Lower numbers are scheduled first.  Keeping this table local avoids relying
# on enum ordering or lexical ordering of scanner-provided severity strings.
_SEVERITY_RANK = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
    "unknown": 5,
}

_EXACT_VERSION = re.compile(r"^[=vV]?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _digest(parts: Iterable[str]) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _edge_kind(edge: SolverEdge | str) -> str:
    return _text(edge if isinstance(edge, str) else edge.edge_kind).lower().replace("-", "_")


def _is_peer_kind(kind: str) -> bool:
    return kind in {"peer", "strict_peer", "peer_conflict", "peer_coupling"}


def _is_hard_kind(kind: str) -> bool:
    return kind in {
        "peer",
        "strict_peer",
        "peer_conflict",
        "peer_coupling",
        "workspace",
        "workspace_coupling",
        "pinned",
        "pinned_dependency",
        "exact_pinned",
    }


def _is_exact_range(value: str | None) -> bool:
    return bool(value and _EXACT_VERSION.fullmatch(value.strip()))


def _dependency_order_kind(kind: str) -> bool:
    """Return whether an edge's source depends on its target."""
    return kind in {
        "runtime",
        "dependency",
        "dependency_ancestry",
        "ancestry",
        "pinned",
        "pinned_dependency",
        "exact_pinned",
    }


def _scope(name: str) -> str | None:
    name = _text(name)
    if not name.startswith("@") or "/" not in name:
        return None
    return name.split("/", 1)[0]


def _version_key(value: str | None) -> tuple[int, int, int, str] | None:
    if not value:
        return None
    match = re.match(r"^[=vV]?(\d+)\.(\d+)\.(\d+)(.*)$", value.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), match.group(4)


class DisjointSetUnion:
    """A deterministic union-find with path compression.

    Roots are selected lexicographically, rather than by insertion order or
    rank.  Consequently the same set of relations has the same representative
    regardless of input iteration order.
    """

    def __init__(self, items: Iterable[str] = ()) -> None:
        """Initialize a disjoint-set forest from optional item names."""
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}
        for item in sorted({_text(item) for item in items if _text(item)}):
            self.add(item)

    def add(self, item: str) -> None:
        """Add ``item`` as a singleton if it is not present."""
        item = _text(item)
        if not item:
            raise ValueError("DSU items must be non-empty")
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: str) -> str:
        """Return the stable root for ``item`` and compress its path."""
        if item not in self._parent:
            raise KeyError(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != item:
            parent = self._parent[item]
            self._parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> bool:
        """Join two sets, choosing the lexicographically smaller root."""
        self.add(left)
        self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size.pop(right_root)
        return True

    def components(self) -> tuple[tuple[str, ...], ...]:
        """Return all components sorted by root and then member ID."""
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in sorted(self._parent):
            grouped[self.find(item)].append(item)
        return tuple(
            sorted((tuple(sorted(members)) for members in grouped.values()), key=lambda x: x[0])
        )

    @property
    def parent(self) -> Mapping[str, str]:
        """Expose a read-only copy of parent links for diagnostics."""
        return dict(self._parent)


def _target_indexes(
    targets: Sequence[SolverTarget],
) -> tuple[dict[str, SolverTarget], dict[str, SolverTarget]]:
    by_occurrence: dict[str, SolverTarget] = {}
    by_task: dict[str, SolverTarget] = {}
    for target in targets:
        if target.occurrence_id in by_occurrence:
            raise ValueError(f"duplicate target occurrence ID: {target.occurrence_id!r}")
        if target.task_id in by_task:
            raise ValueError(f"duplicate target task ID: {target.task_id!r}")
        by_occurrence[target.occurrence_id] = target
        by_task[target.task_id] = target
    return by_occurrence, by_task


def _decision_indexes(
    decisions: Sequence[SolverTaskDecision],
) -> dict[str, SolverTaskDecision]:
    """Index one solver decision per executable task ID."""
    indexed: dict[str, SolverTaskDecision] = {}
    for decision in decisions:
        if decision.task_id in indexed:
            raise ValueError(f"duplicate solver task decision: {decision.task_id!r}")
        indexed[decision.task_id] = decision
    return indexed


def _finding_sets(
    subgraph: SolverSubgraph,
) -> tuple[dict[str, list[SolverFindingRequirement]], dict[str, set[str]]]:
    by_occurrence: dict[str, list[SolverFindingRequirement]] = defaultdict(list)
    ids_by_occurrence: dict[str, set[str]] = defaultdict(set)
    for finding in subgraph.findings:
        by_occurrence[finding.target_occurrence_id].append(finding)
        ids_by_occurrence[finding.target_occurrence_id].add(finding.finding_id)
    for target in subgraph.targets:
        ids_by_occurrence[target.occurrence_id].update(target.finding_ids)
    for values in by_occurrence.values():
        values.sort(key=lambda item: item.finding_id)
    return by_occurrence, ids_by_occurrence


def _candidate_finding_ids(
    target: SolverTarget,
    decision: SolverTaskDecision | None,
    findings: Mapping[str, Sequence[SolverFindingRequirement]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify findings as resolved, workaround, or unresolved."""
    requirements = findings.get(target.occurrence_id, ())
    explicit_ids = set(target.finding_ids)
    resolved: list[str] = []
    workaround: list[str] = []
    unresolved: list[str] = []
    strategy = _text(getattr(decision, "selected_strategy", "")).lower() if decision else ""
    selected = getattr(decision, "selected_version", None) if decision else None
    for requirement in requirements:
        explicit_ids.add(requirement.finding_id)
        if (
            strategy in {"workaround", "code_workaround", "no_fix", "no-fix"}
            and requirement.workaround_available
        ):
            workaround.append(requirement.finding_id)
        elif strategy in {"version_bump", "version-bump", "version bump"} and selected:
            floor = _version_key(requirement.fixed_version)
            actual = _version_key(selected)
            if floor is None or (actual is not None and actual[:3] >= floor[:3]):
                resolved.append(requirement.finding_id)
            else:
                unresolved.append(requirement.finding_id)
        else:
            unresolved.append(requirement.finding_id)
    accounted = set(resolved) | set(workaround) | set(unresolved)
    unresolved.extend(sorted(explicit_ids - accounted))
    return sorted(set(resolved)), sorted(set(workaround)), sorted(set(unresolved))


def _boundary(left: SolverTarget, right: SolverTarget) -> bool:
    if left.manifest_path == right.manifest_path:
        return True
    return bool(left.workspace_id and left.workspace_id == right.workspace_id)


def _resolve_pair(
    pair: tuple[str, str],
    by_occurrence: Mapping[str, SolverTarget],
    by_task: Mapping[str, SolverTarget],
) -> tuple[str, str] | None:
    resolved: list[str] = []
    for value in pair:
        if value in by_occurrence:
            resolved.append(value)
        elif value in by_task:
            resolved.append(by_task[value].occurrence_id)
        else:
            return None
    if resolved[0] == resolved[1]:
        return None
    return resolved[0], resolved[1]


def _supported(
    target: SolverTarget, decision: SolverTaskDecision | None
) -> tuple[bool, str | None, bool]:
    """Return (eligible, diagnostic, dispatchable) for one target."""
    if target.is_terminal:
        return False, "terminal task retained as non-dispatchable singleton", False
    if target.has_open_attempt:
        return False, "task has an open attempt; no new mutation assigned", False
    if not target.eligible_for_atomic_update:
        return False, "target is unsupported or ambiguous for atomic update", True
    if decision is None:
        return False, "missing solver task decision; retained as singleton", True
    strategy = _text(decision.selected_strategy).lower()
    if strategy not in {"version_bump", "version-bump", "version bump"}:
        return (
            False,
            f"unsupported strategy {decision.selected_strategy!r}; retained as singleton",
            True,
        )
    return True, None, True


def _severity_for(
    finding_ids: Iterable[str],
    by_id: Mapping[str, SolverFindingRequirement],
) -> int:
    ranks = [
        _SEVERITY_RANK.get(_text(by_id[finding_id].severity).lower(), 5)
        for finding_id in finding_ids
        if finding_id in by_id
    ]
    return min(ranks, default=5)


def _new_batch(
    task_ids: Sequence[str],
    targets: Mapping[str, SolverTarget],
    decisions: Mapping[str, SolverTaskDecision],
    findings: SolverSubgraph,
    finding_by_occurrence: Mapping[str, set[SolverFindingRequirement]],
    *,
    atomic: bool,
    dispatchable: bool,
    batch_diagnostic: str | None = None,
) -> SolverBatch:
    ordered = sorted(task_ids)
    mutations: list[SolverMutation] = []
    resolved: set[str] = set()
    workaround: set[str] = set()
    unresolved: set[str] = set()
    for task_id in ordered:
        target = targets[task_id]
        decision = decisions.get(task_id)
        r, w, u = _candidate_finding_ids(target, decision, finding_by_occurrence)
        resolved.update(r)
        workaround.update(w)
        unresolved.update(u)
        if (
            dispatchable
            and decision is not None
            and decision.selected_version
            and decision.selected_strategy.lower().replace("-", "_") == "version_bump"
        ):
            mutations.append(
                SolverMutation(
                    task_id=task_id,
                    occurrence_id=target.occurrence_id,
                    package_name=target.target_package_name,
                    manifest_path=target.manifest_path,
                    target_version=decision.selected_version,
                    dependency_type=decision.dependency_type or target.dependency_type,
                )
            )
    all_finding_ids = sorted(resolved | workaround | unresolved)
    return SolverBatch(
        batch_id=f"batch-{_digest(ordered)}",
        task_ids=ordered,
        mutations=mutations,
        resolved_finding_ids=sorted(resolved),
        workaround_finding_ids=sorted(workaround),
        unresolved_finding_ids=sorted(unresolved),
        severity_rank=_severity_for(all_finding_ids, {f.finding_id: f for f in findings.findings}),
        atomic=atomic,
        dispatchable=dispatchable,
        diagnostic=batch_diagnostic,
    )


def cluster_packages(
    subgraph: SolverSubgraph,
    decisions: Sequence[SolverTaskDecision],
    *,
    forced_singleton_task_ids: Sequence[str] = (),
    peer_conflict_pairs: Sequence[tuple[str, str]] = (),
    scope_coupling: bool = True,
) -> tuple[list[SolverBatch], list[SolverEdge], list[str]]:
    """Hard relationships are never split.  Same-scope relationships are soft
    coupling signals and may be partitioned at the configured action-size cap.
    Unsupported, terminal, and open-attempt tasks are preserved as singleton
    diagnostics rather than silently dropped.

    Args:
        subgraph: Occurrence graph to cluster.
        decisions: Solver decisions for the graph targets.
        forced_singleton_task_ids: Tasks that must not join an atomic batch.
        peer_conflict_pairs: Explicit peer-conflict pairs to couple.
        scope_coupling: Whether same-package-scope targets may form a soft
            batch. Development-scoped runs disable this because namespace
            membership is not proof that packages must be mutated together.

    Returns:
        Solver batches, retained occurrence edges, and deterministic diagnostics.
    """
    by_occurrence, by_task = _target_indexes(subgraph.targets)
    decision_by_task = _decision_indexes(decisions)
    finding_by_occurrence, _ = _finding_sets(subgraph)
    forced = {_text(item) for item in forced_singleton_task_ids if _text(item)} | set(
        subgraph.forced_singleton_task_ids
    )
    forced_occurrences = {by_task[item].occurrence_id for item in forced if item in by_task}
    diagnostics = list(subgraph.diagnostics)
    eligible: dict[str, bool] = {}
    dispatchable: dict[str, bool] = {}
    for task_id, target in sorted(by_task.items()):
        ok, diagnostic, can_dispatch = _supported(target, decision_by_task.get(task_id))
        eligible[task_id], dispatchable[task_id] = ok, can_dispatch
        if diagnostic:
            diagnostics.append(f"task {task_id!r}: {diagnostic}")

    hard = DisjointSetUnion(by_occurrence)
    soft = DisjointSetUnion(by_occurrence)
    hard_relations: list[tuple[str, str, str]] = []
    soft_relations: list[tuple[str, str, str]] = []

    def add_relation(left: str, right: str, kind: str, hard_relation: bool) -> None:
        if left not in by_occurrence or right not in by_occurrence or left == right:
            return
        if left in forced_occurrences or right in forced_occurrences:
            diagnostics.append(
                f"forced singleton prevented {kind} coupling between {left!r} and {right!r}"
            )
            return
        # Unsupported/open/terminal targets cannot participate in newly built
        # atomic mutations, even when a graph relation is present.
        if not eligible.get(by_occurrence[left].task_id, False) or not eligible.get(
            by_occurrence[right].task_id, False
        ):
            return
        if hard_relation:
            hard.union(left, right)
            hard_relations.append((left, right, kind))
        else:
            soft_relations.append((left, right, kind))
            # Runtime, ancestry, ordinary dependency, and loose peer edges
            # describe ordering/compatibility evidence, not one atomic edit.
            if kind != "scope":
                return
        soft.union(left, right)
        if hard_relation:
            soft_relations.append((left, right, kind))

    # Candidate compatibility constraints are atomic only when strict and
    # non-optional. Optional or loose peers still remain graph evidence.
    for constraint in sorted(
        subgraph.peer_constraints,
        key=lambda item: (item.source_occurrence_id, item.target_occurrence_id, item.version_range),
    ):
        if (
            constraint.source_occurrence_id in by_occurrence
            and constraint.target_occurrence_id in by_occurrence
        ):
            add_relation(
                constraint.source_occurrence_id,
                constraint.target_occurrence_id,
                "peer",
                constraint.is_strict and not constraint.is_optional,
            )

    for edge in sorted(
        subgraph.edges,
        key=lambda item: (item.source_occurrence_id, item.target_occurrence_id, item.edge_kind),
    ):
        left, right, kind = edge.source_occurrence_id, edge.target_occurrence_id, _edge_kind(edge)
        if not scope_coupling and kind == "scope":
            continue
        if left not in by_occurrence or right not in by_occurrence:
            continue
        hard_relation = (
            not edge.is_optional
            and _is_hard_kind(kind)
            and (
                kind not in {"pinned", "pinned_dependency", "exact_pinned"}
                or _is_exact_range(edge.version_range)
            )
        )
        add_relation(left, right, kind, hard_relation)

    # Same-scope coupling is a conservative full-repository fallback. A
    # development-scoped plan must rely on actual peer/workspace evidence so a
    # target such as @angular/core cannot pull every @angular/* package into
    # one atomic mutation.
    if scope_coupling:
        ordered_targets = sorted(subgraph.targets, key=lambda item: item.occurrence_id)
        for index, left in enumerate(ordered_targets):
            left_scope = _scope(left.target_package_name)
            if not left_scope:
                continue
            for right in ordered_targets[index + 1 :]:
                if left_scope == _scope(right.target_package_name) and _boundary(left, right):
                    add_relation(left.occurrence_id, right.occurrence_id, "scope", False)

    for pair in sorted(peer_conflict_pairs):
        resolved = _resolve_pair(pair, by_occurrence, by_task)
        if resolved:
            add_relation(resolved[0], resolved[1], "peer_conflict", True)
            add_relation(resolved[1], resolved[0], "peer_conflict", True)
        else:
            diagnostics.append(f"peer conflict references unknown target pair {pair!r}")

    # Turn soft components into groups of hard components.  A hard component
    # larger than the action cap cannot safely be dispatched and is rejected.
    hard_components = {
        root: tuple(members)
        for root, members in (
            (hard.find(component[0]), component) for component in hard.components()
        )
    }
    soft_components = [list(component) for component in soft.components()]
    groups: list[tuple[list[str], bool, bool, str | None]] = []
    assigned: set[str] = set()
    for soft_component in sorted(soft_components, key=lambda items: items[0]):
        hard_units: list[list[str]] = []
        seen_roots: set[str] = set()
        for occurrence_id in sorted(soft_component):
            root = hard.find(occurrence_id)
            if root not in seen_roots:
                seen_roots.add(root)
                hard_units.append(list(hard_components[root]))
        hard_units.sort(
            key=lambda unit: (
                not any(by_occurrence[item].is_finding_backed for item in unit),
                min(unit),
            )
        )
        if any(len(unit) > MAX_MULTI_PACKAGE_ACTION_SIZE for unit in hard_units):
            members = sorted(
                item
                for unit in hard_units
                if len(unit) > MAX_MULTI_PACKAGE_ACTION_SIZE
                for item in unit
            )
            diagnostics.append(f"{_HARD_ATOMIC_OVERFLOW_REASON}; rejected component {members!r}")
            for occurrence_id in members:
                task_id = by_occurrence[occurrence_id].task_id
                groups.append(([task_id], False, False, _HARD_ATOMIC_OVERFLOW_REASON))
                assigned.add(occurrence_id)
            continue
        current: list[str] = []
        for unit in hard_units:
            task_count = len(current) + len(unit)
            if current and task_count > MAX_MULTI_PACKAGE_ACTION_SIZE:
                groups.append(
                    (
                        [by_occurrence[item].task_id for item in current],
                        True,
                        True,
                        _SOFT_PARTITION_REASON,
                    )
                )
                diagnostics.append(
                    f"{_SOFT_PARTITION_REASON}: {[by_occurrence[item].task_id for item in current]!r}"
                )
                current = []
            current.extend(unit)
        if current:
            task_ids = [by_occurrence[item].task_id for item in current]
            groups.append(
                (
                    task_ids,
                    len(hard_units) > 1 or len(task_ids) > 1,
                    all(dispatchable[task_id] for task_id in task_ids),
                    None,
                )
            )
        assigned.update(soft_component)

    for target in sorted(subgraph.targets, key=lambda item: item.occurrence_id):
        if target.occurrence_id not in assigned:
            groups.append(([target.task_id], False, dispatchable.get(target.task_id, True), None))

    # Forced and unsupported tasks are always singleton batches, even if they
    # slipped into a soft group through a malformed incoming edge.
    normalized_groups: list[tuple[list[str], bool, bool, str | None]] = []
    for task_ids, atomic, can_dispatch, diagnostic in groups:
        if len(task_ids) == 1:
            normalized_groups.append((task_ids, atomic, can_dispatch, diagnostic))
            continue
        split = [
            task_id for task_id in task_ids if task_id in forced or not eligible.get(task_id, False)
        ]
        if split:
            keep = [task_id for task_id in task_ids if task_id not in split]
            if keep:
                normalized_groups.append((keep, atomic, can_dispatch, diagnostic))
            for task_id in split:
                normalized_groups.append(
                    (
                        [task_id],
                        False,
                        dispatchable.get(task_id, True),
                        "forced/unsupported task retained as singleton",
                    )
                )
                diagnostics.append(f"task {task_id!r} retained as singleton despite coupling")
        else:
            normalized_groups.append((task_ids, atomic, can_dispatch, diagnostic))

    batches: list[SolverBatch] = []
    task_to_batch: dict[str, str] = {}
    for task_ids, atomic, can_dispatch, diagnostic in normalized_groups:
        batch = _new_batch(
            task_ids,
            by_task,
            decision_by_task,
            subgraph,
            finding_by_occurrence,
            atomic=atomic,
            dispatchable=can_dispatch,
            batch_diagnostic=diagnostic,
        )
        if diagnostic:
            diagnostics.append(f"batch {batch.batch_id}: {diagnostic}")
        batches.append(batch)
        for task_id in batch.task_ids:
            task_to_batch[task_id] = batch.batch_id
    batches.sort(key=lambda batch: batch.batch_id)

    # Preserve occurrence edges and add explicit coupling metadata.  The latter
    # is bidirectional so SCC scheduling can diagnose non-peer coupling cycles.
    edge_map: dict[tuple[str, str, str], SolverEdge] = {}
    for edge in subgraph.edges:
        if not scope_coupling and _edge_kind(edge) == "scope":
            # The boundary extractor retains namespace edges for traceability,
            # but development-scoped plans treat namespace membership as
            # validation context rather than mutation or ordering evidence.
            continue
        source = by_occurrence.get(edge.source_occurrence_id)
        target = by_occurrence.get(edge.target_occurrence_id)
        if source is None or target is None:
            diagnostics.append(
                f"edge references unknown occurrence: {edge.source_occurrence_id!r}->{edge.target_occurrence_id!r}"
            )
            continue
        normalized = edge.model_copy(
            update={
                "source_task_id": edge.source_task_id or source.task_id,
                "target_task_id": edge.target_task_id or target.task_id,
            }
        )
        edge_map[
            (
                normalized.source_occurrence_id,
                normalized.target_occurrence_id,
                _edge_kind(normalized),
            )
        ] = normalized
    for left, right, kind in sorted(set(hard_relations + soft_relations)):
        for source_id, target_id in ((left, right), (right, left)):
            edge = SolverEdge(
                source_occurrence_id=source_id,
                target_occurrence_id=target_id,
                edge_kind=kind,
                source_task_id=by_occurrence[source_id].task_id,
                target_task_id=by_occurrence[target_id].task_id,
                is_peer_coupling=_is_peer_kind(kind),
            )
            edge_map.setdefault((source_id, target_id, kind), edge)
    edges = sorted(
        edge_map.values(),
        key=lambda item: (item.source_occurrence_id, item.target_occurrence_id, _edge_kind(item)),
    )
    return batches, edges, sorted(set(diagnostics))


def build_dependency_dag(
    subgraph: SolverSubgraph,
    batches: Sequence[SolverBatch],
    edges: Sequence[SolverEdge] = (),
) -> DAGBuildResult:
    """Map occurrence edges to batch edges and validate external prerequisites.

    ``edges`` retain occurrence identities for diagnostics and candidate
    provenance. ``batch_edges`` is the compact directed graph consumed by
    scheduling, with each tuple ordered as ``(upstream_batch, downstream_batch)``.
    """
    by_occurrence, _ = _target_indexes(subgraph.targets)
    task_to_batch: dict[str, str] = {}
    occurrence_to_task: dict[str, str] = {}
    for batch in batches:
        for task_id in batch.task_ids:
            if task_id in task_to_batch:
                raise ValueError(f"task belongs to multiple batches: {task_id!r}")
            task_to_batch[task_id] = batch.batch_id
        for mutation in batch.mutations:
            occurrence_to_task[mutation.occurrence_id] = mutation.task_id
    for target in subgraph.targets:
        occurrence_to_task.setdefault(target.occurrence_id, target.task_id)

    external = set(subgraph.external_prerequisite_task_ids)
    diagnostics = list(subgraph.diagnostics)
    edges_to_validate = list(edges) if edges else list(subgraph.edges)
    mapped: dict[tuple[str, str, str], SolverEdge] = {}
    batch_edge_set: set[tuple[str, str]] = set()
    used_external: set[str] = set()
    valid = bool(getattr(subgraph, "valid", True))
    for edge in sorted(
        edges_to_validate,
        key=lambda item: (item.source_occurrence_id, item.target_occurrence_id, _edge_kind(item)),
    ):
        kind = _edge_kind(edge)
        source_task = edge.source_task_id or occurrence_to_task.get(edge.source_occurrence_id)
        target_task = edge.target_task_id or occurrence_to_task.get(edge.target_occurrence_id)
        if target_task not in task_to_batch:
            diagnostics.append(
                f"unknown downstream edge endpoint {target_task or edge.target_occurrence_id!r}"
            )
            valid = False
            continue
        source_batch: str | None = None
        if source_task in task_to_batch:
            source_batch = task_to_batch[source_task]
        elif source_task and source_task in external:
            used_external.add(source_task)
        else:
            diagnostics.append(
                f"unknown upstream edge endpoint {source_task or edge.source_occurrence_id!r}"
            )
            valid = False
            continue
        target_batch = task_to_batch[target_task]
        if source_batch is not None and source_batch != target_batch:
            batch_edge_set.add(
                (target_batch, source_batch)
                if _dependency_order_kind(kind)
                else (source_batch, target_batch)
            )
        mapped_edge = edge.model_copy(
            update={
                "source_task_id": source_task,
                "target_task_id": target_task,
            }
        )
        mapped_key = (source_task or source_batch or "", target_task, _edge_kind(mapped_edge))
        mapped[mapped_key] = mapped_edge

    # ``edges`` are occurrence relationships. Keep them in the DAG result,
    # while batch_edges supplies the graph after endpoint projection.
    return DAGBuildResult(
        batches=sorted(batches, key=lambda batch: batch.batch_id),
        edges=sorted(
            mapped.values(),
            key=lambda item: (
                item.source_occurrence_id,
                item.target_occurrence_id,
                _edge_kind(item),
            ),
        ),
        batch_edges=sorted(batch_edge_set),
        task_to_batch=dict(sorted(task_to_batch.items())),
        external_prerequisite_task_ids=sorted(used_external),
        diagnostics=sorted(set(diagnostics)),
        valid=valid,
    )


def _sccs(
    nodes: Sequence[str], batch_edges: Sequence[tuple[str, str]]
) -> tuple[tuple[str, ...], ...]:
    """Return Tarjan SCCs with every traversal sorted for determinism."""
    successors: dict[str, set[str]] = {node: set() for node in nodes}
    for source, target in batch_edges:
        if source in successors and target in successors and source != target:
            successors[source].add(target)
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    active: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for successor in sorted(successors[node]):
            if successor not in indices:
                visit(successor)
                low[node] = min(low[node], low[successor])
            elif successor in active:
                low[node] = min(low[node], indices[successor])
        if low[node] == indices[node]:
            members: list[str] = []
            while True:
                member = stack.pop()
                active.remove(member)
                members.append(member)
                if member == node:
                    break
            components.append(tuple(sorted(members)))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)
    return tuple(components)


def schedule_batches(
    dag: DAGBuildResult,
    *,
    severity_rank: Mapping[str, int] | None = None,
    phase_budget: int = 8,
) -> tuple[list[SolverPhase], list[str]]:
    """Condense SCCs and schedule ready batches into bounded phases.

    Atomic batches are indivisible.  An SCC is one scheduling component, so a
    cyclic component may exceed the soft phase budget but is never split.
    """
    if phase_budget < 1:
        return [], ["phase budget must be at least one"]
    diagnostics = list(dag.diagnostics)
    severity_rank = severity_rank or {}
    if not dag.valid:
        return [], sorted(
            set(diagnostics + ["invalid dependency DAG; no dispatchable phases emitted"])
        )

    batches = tuple(sorted(getattr(dag, "batches", ()), key=lambda batch: batch.batch_id))
    batch_by_id = {batch.batch_id: batch for batch in batches}
    batch_ids = sorted(batch_by_id)
    raw_batch_edges = getattr(dag, "batch_edges", ())
    batch_edges = sorted(
        {
            (str(source), str(target))
            for source, target in raw_batch_edges
            if str(source) in batch_by_id
            and str(target) in batch_by_id
            and str(source) != str(target)
        }
    )
    # Compatibility with a hand-built DAG from an older contract: derive
    # projected endpoints from task IDs where batch_edges was not populated.
    if not batch_edges:
        task_to_batch = getattr(dag, "task_to_batch", {})
        derived: set[tuple[str, str]] = set()
        for edge in dag.edges:
            source = task_to_batch.get(edge.source_task_id or "")
            target = task_to_batch.get(edge.target_task_id or "")
            if source and target and source != target:
                derived.add((source, target))
        batch_edges = sorted(derived)
    components = _sccs(batch_ids, batch_edges)
    component_of = {node: component for component in components for node in component}
    successors: dict[tuple[str, ...], set[tuple[str, ...]]] = {
        component: set() for component in components
    }
    predecessors: dict[tuple[str, ...], set[tuple[str, ...]]] = {
        component: set() for component in components
    }
    for source, target in batch_edges:
        left, right = component_of[source], component_of[target]
        if left != right:
            successors[left].add(right)
            predecessors[right].add(left)

    # Recover relationship kinds from occurrence edges to distinguish peer-only
    # cycles from ordinary dependency cycles.
    task_to_batch = getattr(dag, "task_to_batch", {})
    pair_kinds: dict[tuple[str, str], set[str]] = defaultdict(set)
    for edge in dag.edges:
        source = task_to_batch.get(edge.source_task_id or "", edge.source_occurrence_id)
        target = task_to_batch.get(edge.target_task_id or "", edge.target_occurrence_id)
        if source in batch_by_id and target in batch_by_id:
            pair_kinds[(source, target)].add(_edge_kind(edge))
    for component in components:
        if len(component) <= 1:
            continue
        internal_kinds = [
            kind
            for source, target in batch_edges
            if source in component and target in component
            for kind in pair_kinds.get((source, target), {"runtime"})
        ]
        if internal_kinds and all(_is_peer_kind(kind) for kind in internal_kinds):
            diagnostics.append(f"peer cycle collapsed for {list(component)!r}")
        else:
            diagnostics.append(
                f"non-peer dependency cycle detected; collapsed SCC {list(component)!r}"
            )
        if len(component) > phase_budget:
            diagnostics.append(f"scc_over_budget: {list(component)!r}")

    def batch_rank(batch_id: str) -> int:
        batch = batch_by_id[batch_id]
        rank = severity_rank.get(batch_id)
        if rank is not None:
            return int(rank)
        candidates = [
            int(severity_rank[finding_id])
            for finding_id in (
                batch.resolved_finding_ids
                + batch.workaround_finding_ids
                + batch.unresolved_finding_ids
            )
            if finding_id in severity_rank
        ]
        return min(candidates, default=batch.severity_rank)

    def component_key(
        component: tuple[str, ...],
    ) -> tuple[int, int, tuple[str, ...], tuple[str, ...]]:
        finding_first = any(
            bool(
                batch_by_id[batch_id].resolved_finding_ids
                or batch_by_id[batch_id].workaround_finding_ids
                or batch_by_id[batch_id].unresolved_finding_ids
            )
            for batch_id in component
        )
        task_key = tuple(
            task_id for batch_id in component for task_id in sorted(batch_by_id[batch_id].task_ids)
        )
        return (
            min((batch_rank(batch_id) for batch_id in component), default=5),
            0 if finding_first else 1,
            task_key,
            component,
        )

    ready = sorted(
        (component for component in components if not predecessors[component]), key=component_key
    )
    phases: list[SolverPhase] = []
    phase_number = 1
    while ready:
        selected: list[tuple[str, ...]] = []
        acyclic_count = 0
        for component in ready:
            if selected and len(component) == 1 and acyclic_count >= phase_budget:
                break
            selected.append(component)
            if len(component) == 1:
                acyclic_count += 1
            if len(component) == 1 and acyclic_count >= phase_budget:
                break
        if not selected:
            diagnostics.append("dependency scheduling made no progress")
            return [], sorted(set(diagnostics))
        for component in selected:
            ready.remove(component)
        phase_batches = [batch_id for component in selected for batch_id in component]
        collapsed = any(len(component) > 1 for component in selected)
        phases.append(
            SolverPhase(
                phase_number=phase_number,
                batch_ids=phase_batches,
                scc_collapsed=collapsed,
            )
        )
        phase_number += 1
        selected_set = set(selected)
        for component in selected:
            for successor in sorted(successors[component], key=component_key):
                predecessors[successor].discard(component)
                if (
                    not predecessors[successor]
                    and successor not in ready
                    and successor not in selected_set
                ):
                    ready.append(successor)
        ready.sort(key=component_key)
    return phases, sorted(set(diagnostics))


__all__ = [
    "DisjointSetUnion",
    "MAX_MULTI_PACKAGE_ACTION_SIZE",
    "build_dependency_dag",
    "cluster_packages",
    "schedule_batches",
]
