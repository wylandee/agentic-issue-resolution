"""Pure dependency graph, scheduling, and occurrence-subgraph transforms.

Package and task IDs are never interchangeable: occurrence IDs identify
graph vertices, while task IDs identify executable mutations.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypeVar

from remediation_engine.contracts.schemas import MAX_MULTI_PACKAGE_ACTION_SIZE
from remediation_engine.contracts.solver_models import (
    DAGBuildResult,
    SolverBatch,
    SolverCandidateRelation,
    SolverEdge,
    SolverEvidenceDomain,
    SolverFindingRequirement,
    SolverMutation,
    SolverPackageOccurrence,
    SolverPeerConstraint,
    SolverPhase,
    SolverSubgraph,
    SolverTarget,
    SolverTaskDecision,
    SolverVersionCandidate,
)
from remediation_engine.tools.npm import (
    NpmGraphSnapshot,
    NpmLockfilePackage,
    check_npm_range,
    make_occurrence_id,
    normalize_dependency_ancestry,
    resolve_lockfile_dependency_package,
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
        "dependency_ancestry",
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
        values.sort(key=lambda item: (item.finding_id, item.coverage_id))
    return by_occurrence, ids_by_occurrence


def _candidate_finding_ids(
    target: SolverTarget,
    decision: SolverTaskDecision | None,
    findings: Mapping[str, Sequence[SolverFindingRequirement]],
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Classify occurrence coverage and its issue-level projection."""
    requirements = findings.get(target.occurrence_id, ())
    explicit_ids = set(target.finding_ids)
    status_by_issue: dict[str, set[str]] = defaultdict(set)
    resolved_coverage: list[str] = []
    workaround_coverage: list[str] = []
    unresolved_coverage: list[str] = []
    strategy = _text(getattr(decision, "selected_strategy", "")).lower() if decision else ""
    selected = getattr(decision, "selected_version", None) if decision else None
    for requirement in requirements:
        explicit_ids.add(requirement.finding_id)
        if (
            strategy in {"workaround", "code_workaround"}
            and requirement.workaround_available
            and requirement.workaround_plan_ids
        ):
            workaround_coverage.append(requirement.coverage_id)
            status_by_issue[requirement.finding_id].add("workaround")
        elif strategy in {"version_bump", "version-bump", "version bump"} and selected:
            floor = requirement.direct_parent_minimum_version or requirement.fixed_version
            if _candidate_meets_finding_floor(selected, floor):
                resolved_coverage.append(requirement.coverage_id)
                status_by_issue[requirement.finding_id].add("resolved")
            else:
                unresolved_coverage.append(requirement.coverage_id)
                status_by_issue[requirement.finding_id].add("unresolved")
        else:
            unresolved_coverage.append(requirement.coverage_id)
            status_by_issue[requirement.finding_id].add("unresolved")
    for finding_id in explicit_ids:
        status_by_issue.setdefault(finding_id, {"unresolved"})
    resolved: list[str] = []
    workaround: list[str] = []
    unresolved: list[str] = []
    for finding_id, statuses in status_by_issue.items():
        if statuses == {"resolved"}:
            resolved.append(finding_id)
        elif statuses == {"workaround"}:
            workaround.append(finding_id)
        elif "unresolved" in statuses:
            unresolved.append(finding_id)
    return (
        sorted(resolved),
        sorted(workaround),
        sorted(unresolved),
        sorted(resolved_coverage),
        sorted(workaround_coverage),
        sorted(unresolved_coverage),
    )


def _candidate_meets_finding_floor(candidate: str, floor: str | None) -> bool:
    """Compare a selected version with its finding or parent-version floor.

    Most solver inputs are npm three-part versions. Python parent updates may
    use stable PEP 440 post releases, which require PEP 440 ordering instead
    of comparing only the numeric release prefix.
    """
    if not floor:
        return False
    if ".post" in candidate.casefold() or ".post" in floor.casefold():
        try:
            from packaging.version import Version

            return Version(candidate) >= Version(floor)
        except Exception:  # noqa: BLE001 - malformed version evidence cannot resolve a finding
            return False
    floor_key = _version_key(floor)
    candidate_key = _version_key(candidate)
    return (
        floor_key is not None and candidate_key is not None and candidate_key[:3] >= floor_key[:3]
    )


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
        strategy = _text(target.strategy).lower().replace("-", "_")
        decision_strategy = (
            _text(decision.selected_strategy).lower().replace("-", "_")
            if decision is not None
            else ""
        )
        if (
            target.is_finding_backed
            and decision is not None
            and strategy == "no_fix"
            and decision_strategy == "no_fix"
            and decision.selected_version is None
        ):
            # A committed no-fix task has no package mutation variable, but its
            # mitigation worker still needs a dispatchable unresolved task.
            return False, None, True
        if (
            target.is_finding_backed
            and decision is not None
            and strategy in {"code_workaround", "workaround"}
            and decision_strategy in {"code_workaround", "workaround"}
            and bool(decision.selected_plan_issue_ids)
            and decision.selected_version is None
        ):
            # The Supervisor may dispatch an authorized source workaround
            # without assigning a package mutation variable.
            return False, None, True
        return False, "target is unsupported or ambiguous for atomic update", False
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
    finding_by_occurrence: Mapping[str, Sequence[SolverFindingRequirement]],
    *,
    atomic: bool,
    dispatchable: bool,
    batch_diagnostic: str | None = None,
) -> SolverBatch:
    ordered = sorted(task_ids)
    mutations: list[SolverMutation] = []
    issue_statuses: dict[str, set[str]] = defaultdict(set)
    resolved_coverage: set[str] = set()
    workaround_coverage: set[str] = set()
    unresolved_coverage: set[str] = set()
    coverage_finding_ids = {
        requirement.coverage_id: requirement.finding_id
        for task_id in ordered
        for requirement in finding_by_occurrence.get(targets[task_id].occurrence_id, ())
    }
    for task_id in ordered:
        target = targets[task_id]
        decision = decisions.get(task_id)
        r, w, u, rc, wc, uc = _candidate_finding_ids(target, decision, finding_by_occurrence)
        for finding_id in r:
            issue_statuses[finding_id].add("resolved")
        for finding_id in w:
            issue_statuses[finding_id].add("workaround")
        for finding_id in u:
            issue_statuses[finding_id].add("unresolved")
        resolved_coverage.update(rc)
        workaround_coverage.update(wc)
        unresolved_coverage.update(uc)
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
    resolved = sorted(
        finding_id for finding_id, statuses in issue_statuses.items() if statuses == {"resolved"}
    )
    workaround = sorted(
        finding_id for finding_id, statuses in issue_statuses.items() if statuses == {"workaround"}
    )
    unresolved = sorted(
        finding_id for finding_id, statuses in issue_statuses.items() if "unresolved" in statuses
    )
    all_finding_ids = sorted(issue_statuses)
    return SolverBatch(
        batch_id=f"batch-{_digest(ordered)}",
        task_ids=ordered,
        mutations=mutations,
        resolved_finding_ids=resolved,
        workaround_finding_ids=workaround,
        unresolved_finding_ids=unresolved,
        resolved_coverage_ids=sorted(resolved_coverage),
        workaround_coverage_ids=sorted(workaround_coverage),
        unresolved_coverage_ids=sorted(unresolved_coverage),
        coverage_finding_ids=coverage_finding_ids,
        severity_rank=_severity_for(all_finding_ids, {f.finding_id: f for f in findings.findings}),
        atomic=atomic,
        dispatchable=dispatchable,
        diagnostic=batch_diagnostic,
    )


def _reconcile_batch_issue_projections(batches: Sequence[SolverBatch]) -> list[SolverBatch]:
    """Project occurrence coverage to issue IDs only after all batches are known."""
    statuses_by_issue: dict[str, set[str]] = defaultdict(set)
    for batch in batches:
        for coverage_id, finding_id in batch.coverage_finding_ids.items():
            if coverage_id in batch.resolved_coverage_ids:
                status = "resolved"
            elif coverage_id in batch.workaround_coverage_ids:
                status = "workaround"
            else:
                status = "unresolved"
            statuses_by_issue[finding_id].add(status)

    reconciled: list[SolverBatch] = []
    for batch in batches:
        issue_ids = set(batch.coverage_finding_ids.values())
        unprojected_unresolved = set(batch.unresolved_finding_ids) - issue_ids
        resolved = sorted(
            finding_id for finding_id in issue_ids if statuses_by_issue[finding_id] == {"resolved"}
        )
        workaround = sorted(
            finding_id
            for finding_id in issue_ids
            if statuses_by_issue[finding_id] == {"workaround"}
        )
        unresolved = sorted(
            unprojected_unresolved
            | {
                finding_id
                for finding_id in issue_ids
                if "unresolved" in statuses_by_issue[finding_id]
            }
        )
        reconciled.append(
            batch.model_copy(
                update={
                    "resolved_finding_ids": resolved,
                    "workaround_finding_ids": workaround,
                    "unresolved_finding_ids": unresolved,
                }
            )
        )
    return reconciled


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
            # Runtime, generic ancestry, ordinary dependency, and loose peer
            # edges describe ordering/compatibility evidence, not one atomic
            # edit. Finding-backed dependency ancestry is hard above.
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
        hard_relation = (
            not edge.is_optional
            and _is_hard_kind(kind)
            and (
                kind != "dependency_ancestry"
                or (
                    by_occurrence[left].is_finding_backed and by_occurrence[right].is_finding_backed
                )
            )
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
    batches = _reconcile_batch_issue_projections(batches)

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


_EXACT_VERSION_RE = re.compile(r"^[=vV]?\d+\.\d+\.\d+(?:[-+]\w[\w.-]*)?$")
_DEPENDENCY_SECTIONS = ("dependencies", "optionalDependencies", "peerDependencies")


def _validated_target(value: SolverTarget | Mapping[str, Any]) -> SolverTarget:
    """Validate one target while accepting JSON-compatible callers."""
    if isinstance(value, SolverTarget):
        return value
    return SolverTarget.model_validate(value)


def _validated_finding(
    value: SolverFindingRequirement | Mapping[str, Any],
) -> SolverFindingRequirement:
    """Validate one finding requirement while accepting JSON-compatible callers."""
    if isinstance(value, SolverFindingRequirement):
        return value
    return SolverFindingRequirement.model_validate(value)


_ModelT = TypeVar("_ModelT")


def _deduplicate_models(values: Sequence[_ModelT], identity: Any, label: str) -> list[_ModelT]:
    """Deduplicate identical models and reject conflicting identities.

    A repeated JSON record is harmless and is removed.  Reusing an occurrence,
    task, or finding identity for a different record is an input error rather
    than an arbitrary last-write-wins choice.
    """
    by_identity: dict[str, _ModelT] = {}
    result: list[_ModelT] = []
    for value in values:
        key = str(identity(value))
        previous = by_identity.get(key)
        if previous is not None:
            if previous != value:
                raise ValueError(f"{label} identity {key!r} has conflicting records.")
            continue
        by_identity[key] = value
        result.append(value)
    return result


def _target_identity(target: SolverTarget) -> str:
    return target.occurrence_id


def _finding_identity(finding: SolverFindingRequirement) -> str:
    return finding.coverage_id


def _normalise_pairs(
    pairs: Sequence[tuple[str, str]],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Normalize undirected peer-conflict pairs and report malformed entries."""
    result: set[tuple[str, str]] = set()
    diagnostics: list[str] = []
    for raw_pair in pairs:
        if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
            diagnostics.append(f"invalid peer conflict pair {raw_pair!r}")
            continue
        left, right = (str(raw_pair[0]).strip(), str(raw_pair[1]).strip())
        if not left or not right or left == right:
            diagnostics.append(f"invalid peer conflict pair {raw_pair!r}")
            continue
        result.add(tuple(sorted((left, right))))
    return sorted(result), diagnostics


def _package_key_for_target(target: SolverTarget, package: NpmLockfilePackage | None) -> str:
    """Return a normalized physical key for matching nested lock entries."""
    if package is not None:
        return package.package_key.replace("\\", "/")
    return target.lockfile_package_key.replace("\\", "/")


def _metadata_for_target(
    target: SolverTarget,
    lock_packages: Mapping[tuple[str, str], NpmLockfilePackage],
) -> NpmLockfilePackage | None:
    """Find exact physical lock metadata for a target, if present."""
    key = (target.manifest_path, target.lockfile_package_key.replace("\\", "/"))
    return lock_packages.get(key)


def _child_target(
    source: SolverTarget,
    child_name: str,
    targets_by_id: Mapping[str, SolverTarget],
    lock_packages: Mapping[tuple[str, str], NpmLockfilePackage],
) -> SolverTarget | None:
    """Resolve a represented child without creating a nested mutation target."""
    candidates = [
        target
        for target in targets_by_id.values()
        if target.manifest_path == source.manifest_path and target.target_package_name == child_name
    ]
    if not candidates:
        return None
    source_package = _metadata_for_target(source, lock_packages)
    source_key = _package_key_for_target(source, source_package)
    nested_key = f"{source_key.rstrip('/')}/node_modules/{child_name}"
    exact = [
        target
        for target in candidates
        if target.lockfile_package_key.replace("\\", "/") == nested_key
    ]
    if exact:
        return min(exact, key=lambda target: (target.occurrence_id, target.task_id))
    # A nested lockfile package is a distinct physical dependency.  It is not
    # safe to bind that edge to an unrelated direct occurrence merely because
    # the nested package is not itself a mutation target.
    if (source.manifest_path, nested_key) in lock_packages:
        return None
    # Hoisted children are represented by the direct lockfile key.  Prefer a
    # physical entry at the shallowest depth and then deterministic identity.
    return min(
        candidates,
        key=lambda target: (
            target.lockfile_package_key.replace("\\", "/").count("node_modules/"),
            target.occurrence_id,
            target.task_id,
        ),
    )


def _workspace_roots(snapshot: NpmGraphSnapshot, target: SolverTarget) -> set[str]:
    """Return workspace roots for one target, including its explicit override."""
    roots = set(snapshot.workspace_membership_map.get(target.manifest_path, set()))
    if target.workspace_id:
        roots.add(target.workspace_id)
    return roots


def _same_boundary(snapshot: NpmGraphSnapshot, left: SolverTarget, right: SolverTarget) -> bool:
    """Whether two targets share a manifest or a workspace boundary."""
    if left.manifest_path == right.manifest_path:
        return True
    return bool(_workspace_roots(snapshot, left) & _workspace_roots(snapshot, right))


def _edge(
    source: SolverTarget,
    target: SolverTarget,
    edge_kind: str,
    *,
    version_range: str | None = None,
    is_optional: bool = False,
    is_peer_coupling: bool = False,
) -> SolverEdge | None:
    """Create an edge while preventing invalid self relationships."""
    if source.occurrence_id == target.occurrence_id:
        return None
    return SolverEdge(
        source_occurrence_id=source.occurrence_id,
        target_occurrence_id=target.occurrence_id,
        source_task_id=source.task_id,
        target_task_id=target.task_id,
        edge_kind=edge_kind,
        version_range=version_range,
        is_optional=is_optional,
        is_peer_coupling=is_peer_coupling,
    )


def _dependency_range(metadata: Mapping[str, Any], section: str, package_name: str) -> str | None:
    values = metadata.get(section)
    if not isinstance(values, Mapping) or package_name not in values:
        return None
    value = values.get(package_name)
    return str(value).strip() if value is not None else ""


def _build_metadata_edges(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    *,
    diagnostics: list[str],
) -> tuple[list[SolverEdge], list[SolverPeerConstraint]]:
    """Extract represented runtime and peer relationships from lock metadata."""
    targets_by_id = {target.occurrence_id: target for target in targets}
    lock_packages = {
        (package.manifest_path, package.package_key.replace("\\", "/")): package
        for package in snapshot.lockfile_packages
    }
    edges: dict[tuple[Any, ...], SolverEdge] = {}
    peer_constraints: dict[tuple[Any, ...], SolverPeerConstraint] = {}
    for source in sorted(targets, key=lambda item: item.occurrence_id):
        package = _metadata_for_target(source, lock_packages)
        if package is None:
            continue
        metadata = package.metadata
        for section in _DEPENDENCY_SECTIONS:
            values = metadata.get(section)
            if not isinstance(values, Mapping):
                continue
            for raw_name in sorted(values, key=str):
                child_name = str(raw_name).strip()
                if not child_name:
                    continue
                child = _child_target(source, child_name, targets_by_id, lock_packages)
                requirement = _dependency_range(metadata, section, child_name)
                optional = section == "optionalDependencies"
                if section == "peerDependencies":
                    peer_meta = metadata.get("peerDependenciesMeta")
                    peer_meta_value = (
                        peer_meta.get(child_name) if isinstance(peer_meta, Mapping) else None
                    )
                    optional = optional or (
                        isinstance(peer_meta_value, Mapping)
                        and bool(peer_meta_value.get("optional"))
                    )
                    if not requirement:
                        requirement = "<invalid-empty-peer-range>"
                        diagnostics.append(
                            f"invalid peer range for {source.occurrence_id!r} -> {child_name!r}: empty range"
                        )
                    elif (
                        child is not None
                        and check_npm_range(requirement, child.installed_version).matches is None
                    ):
                        diagnostics.append(
                            f"invalid peer range {requirement!r} for {source.occurrence_id!r} -> {child_name!r}"
                        )
                if child is None:
                    if section == "peerDependencies":
                        diagnostics.append(
                            f"unrepresented peer dependency {child_name!r} for {source.occurrence_id!r}"
                        )
                    continue
                if section == "peerDependencies":
                    constraint = SolverPeerConstraint(
                        source_occurrence_id=source.occurrence_id,
                        target_occurrence_id=child.occurrence_id,
                        version_range=requirement,
                        is_strict=not optional,
                        is_optional=optional,
                    )
                    peer_key = (
                        constraint.source_occurrence_id,
                        constraint.target_occurrence_id,
                        constraint.version_range,
                        constraint.is_strict,
                        constraint.is_optional,
                    )
                    peer_constraints[peer_key] = constraint
                    relation = _edge(
                        source,
                        child,
                        "peer",
                        version_range=requirement,
                        is_optional=optional,
                        is_peer_coupling=True,
                    )
                else:
                    relation = _edge(
                        source,
                        child,
                        "runtime",
                        version_range=requirement,
                        is_optional=optional,
                    )
                if relation is not None:
                    edges[
                        (
                            relation.source_occurrence_id,
                            relation.target_occurrence_id,
                            relation.edge_kind,
                            relation.version_range,
                            relation.is_optional,
                        )
                    ] = relation
                if (
                    section != "peerDependencies"
                    and requirement
                    and _EXACT_VERSION_RE.fullmatch(requirement)
                ):
                    pinned = _edge(
                        source,
                        child,
                        "pinned",
                        version_range=requirement,
                    )
                    if pinned is not None:
                        edges[
                            (
                                pinned.source_occurrence_id,
                                pinned.target_occurrence_id,
                                pinned.edge_kind,
                                pinned.version_range,
                                pinned.is_optional,
                            )
                        ] = pinned
    return list(edges.values()), list(peer_constraints.values())


def _build_fixed_peer_constraints(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    evidence_occurrences: Sequence[SolverPackageOccurrence],
    *,
    diagnostics: list[str],
) -> tuple[list[SolverPeerConstraint], bool]:
    """Constrain target versions by required peers of unchanged packages.

    A candidate update can violate a peer range declared by a package that is
    intentionally outside the mutation scope. Keep that installed consumer as
    evidence and constrain the target candidate to its declared range instead
    of silently upgrading or ignoring the consumer.
    """
    target_ids = {target.occurrence_id for target in targets}
    evidence_ids = {occurrence.occurrence_id for occurrence in evidence_occurrences}
    fixed_constraints: dict[tuple[str, str, str], SolverPeerConstraint] = {}
    valid = True

    for source_package in snapshot.lockfile_packages:
        source_occurrence = next(
            (
                occurrence
                for occurrence in snapshot.occurrences
                if occurrence.manifest_path == source_package.manifest_path
                and occurrence.package_name == source_package.package_name
                and (occurrence.lockfile_package_key or "").replace("\\", "/")
                == source_package.package_key.replace("\\", "/")
            ),
            None,
        )
        if source_occurrence is None:
            continue
        source_id = source_occurrence.occurrence_id
        if source_id in target_ids or source_id not in evidence_ids:
            continue
        peer_dependencies = source_package.metadata.get("peerDependencies", {})
        if not isinstance(peer_dependencies, Mapping):
            diagnostics.append(f"malformed peerDependencies for fixed package {source_id!r}")
            valid = False
            continue
        peer_metadata = source_package.metadata.get("peerDependenciesMeta", {})
        if not isinstance(peer_metadata, Mapping):
            diagnostics.append(f"malformed peerDependenciesMeta for fixed package {source_id!r}")
            valid = False
            continue

        for target in targets:
            if target.manifest_path != source_package.manifest_path:
                continue
            raw_range = peer_dependencies.get(target.target_package_name)
            if raw_range is None:
                continue
            if not isinstance(raw_range, str) or not raw_range.strip():
                diagnostics.append(
                    f"invalid fixed peer range for {source_id!r} -> {target.occurrence_id!r}"
                )
                valid = False
                continue
            peer_meta = peer_metadata.get(target.target_package_name, {})
            if isinstance(peer_meta, Mapping) and bool(peer_meta.get("optional")):
                continue

            resolved = resolve_lockfile_dependency_package(
                snapshot,
                source_id,
                target.target_package_name,
            )
            if (
                resolved is None
                or resolved.manifest_path != target.manifest_path
                or resolved.package_key.replace("\\", "/")
                != target.lockfile_package_key.replace("\\", "/")
            ):
                continue

            constraint = SolverPeerConstraint(
                source_occurrence_id=source_id,
                target_occurrence_id=target.occurrence_id,
                version_range=raw_range.strip(),
                is_strict=True,
            )
            fixed_constraints[(source_id, target.occurrence_id, raw_range.strip())] = constraint

    return list(fixed_constraints.values()), valid


def _build_ancestry_edges(
    targets: Sequence[SolverTarget],
) -> list[SolverEdge]:
    """Connect adjacent represented package names in declared ancestry paths."""
    by_manifest_name: dict[tuple[str, str], list[SolverTarget]] = {}
    for target in targets:
        by_manifest_name.setdefault((target.manifest_path, target.target_package_name), []).append(
            target
        )
    edges: dict[tuple[str, str, str], SolverEdge] = {}
    for leaf in targets:
        ancestry = normalize_dependency_ancestry(leaf.dependency_ancestry)
        if len(ancestry) < 2:
            continue
        for parent_name, child_name in zip(ancestry, ancestry[1:], strict=False):
            parents = by_manifest_name.get((leaf.manifest_path, parent_name), ())
            children = by_manifest_name.get((leaf.manifest_path, child_name), ())
            for parent in parents:
                for child in children:
                    relation = _edge(parent, child, "dependency_ancestry")
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
    return list(edges.values())


def _build_boundary_edges(
    snapshot: NpmGraphSnapshot, targets: Sequence[SolverTarget]
) -> list[SolverEdge]:
    """Add deterministic workspace and same-scope coupling within safe boundaries."""
    edges: dict[tuple[str, str, str], SolverEdge] = {}
    ordered = sorted(targets, key=lambda item: item.occurrence_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if not _same_boundary(snapshot, left, right):
                continue
            if left.manifest_path != right.manifest_path and _workspace_roots(
                snapshot, left
            ) & _workspace_roots(snapshot, right):
                for source, target in ((left, right), (right, left)):
                    relation = _edge(source, target, "workspace", is_peer_coupling=True)
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
            left_scope, right_scope = (
                _scope(left.target_package_name),
                _scope(right.target_package_name),
            )
            if left_scope is not None and left_scope == right_scope:
                for source, target in ((left, right), (right, left)):
                    relation = _edge(source, target, "scope", is_peer_coupling=True)
                    if relation is not None:
                        edges[
                            (
                                relation.source_occurrence_id,
                                relation.target_occurrence_id,
                                relation.edge_kind,
                            )
                        ] = relation
    return list(edges.values())


def _bounded_occurrences(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
) -> list[SolverPackageOccurrence]:
    """Retain affected manifests, workspaces, and their complete lockfile closure."""
    manifests = {target.manifest_path for target in targets}
    memberships_by_manifest = snapshot.workspace_membership_map
    members_by_root: dict[str, set[str]] = {}
    for membership in snapshot.workspace_memberships:
        members_by_root.setdefault(membership.workspace_root, set()).add(membership.member_manifest)
    active_targets = [
        target for target in targets if target.is_finding_backed and not target.is_terminal
    ]
    for target in active_targets:
        roots = set(memberships_by_manifest.get(target.manifest_path, set()))
        if target.workspace_id:
            roots.add(target.workspace_id)
        for root in roots:
            manifests.add(root)
            manifests.update(members_by_root.get(root, set()))
    evidence = {
        occurrence.occurrence_id: SolverPackageOccurrence(
            occurrence_id=occurrence.occurrence_id,
            manifest_path=occurrence.manifest_path,
            package_name=occurrence.package_name,
            lockfile_package_key=occurrence.lockfile_package_key,
            installed_version=occurrence.installed_version,
            dependency_type=occurrence.dependency_type,
            is_direct=occurrence.is_direct,
            ancestry=list(occurrence.ancestry),
        )
        for occurrence in snapshot.occurrences
        if occurrence.manifest_path in manifests
    }
    return [evidence[key] for key in sorted(evidence)]


def _nearest_candidate_child(
    source: SolverPackageOccurrence,
    package_name: str,
    evidence: Sequence[SolverPackageOccurrence],
) -> tuple[SolverPackageOccurrence | None, bool]:
    """Resolve the nearest same-manifest child by its physical npm key."""
    source_key = (source.lockfile_package_key or "").replace("\\", "/").strip("/")
    prefixes: list[str] = []
    current = source_key
    while current:
        prefixes.append(current)
        marker = current.rfind("/node_modules/")
        current = current[:marker] if marker >= 0 else ""
    prefixes.append("")
    for prefix in prefixes:
        child_key = (
            f"{prefix}/node_modules/{package_name}" if prefix else f"node_modules/{package_name}"
        )
        matches = [
            occurrence
            for occurrence in evidence
            if occurrence.manifest_path == source.manifest_path
            and occurrence.package_name == package_name
            and (occurrence.lockfile_package_key or "").replace("\\", "/") == child_key
        ]
        if matches:
            if len(matches) == 1:
                return matches[0], False
            return None, True
    return None, False


def expand_candidate_relations(
    subgraph: SolverSubgraph,
    candidate_domains: Mapping[str, Sequence[SolverVersionCandidate]],
    *,
    npm_snapshot: NpmGraphSnapshot | None = None,
    evidence_domains: Sequence[SolverEvidenceDomain] = (),
) -> SolverSubgraph:
    """Expand published dependency requirements into physical or witness relations.

    Evidence domains are only used for supported, required runtime dependencies
    without an eligible physical mutation target. They never create package
    occurrences or worker tasks.
    """
    diagnostics = list(subgraph.diagnostics)
    valid = subgraph.valid
    evidence_by_id = {occurrence.occurrence_id: occurrence for occurrence in subgraph.occurrences}
    evidence_domains_by_key = {
        (domain.source_occurrence_id, domain.package_name, domain.dependency_kind): domain
        for domain in evidence_domains
    }
    eligible_target_ids = {
        target.occurrence_id for target in subgraph.targets if target.eligible_for_atomic_update
    }
    relations: dict[tuple[Any, ...], SolverCandidateRelation] = {}
    for target in sorted(subgraph.targets, key=lambda item: item.occurrence_id):
        if not target.eligible_for_atomic_update:
            continue
        domain = candidate_domains.get(target.occurrence_id)
        if domain is None:
            domain = candidate_domains.get(target.task_id)
        if domain is None:
            valid = False
            diagnostics.append(f"candidate relations missing domain for {target.occurrence_id!r}")
            continue
        source = evidence_by_id.get(target.occurrence_id)
        if source is None:
            valid = False
            diagnostics.append(
                f"candidate relation source {target.occurrence_id!r} is absent from evidence"
            )
            continue
        for candidate in domain:
            if not isinstance(candidate, SolverVersionCandidate):
                valid = False
                diagnostics.append(f"invalid candidate metadata for {target.occurrence_id!r}")
                continue
            for requirement in candidate.requirements:
                range_supported = requirement.is_range_supported
                if (
                    range_supported
                    and check_npm_range(requirement.version_range, "1.0.0").matches is None
                ):
                    valid = False
                    diagnostics.append(
                        f"invalid candidate range {requirement.version_range!r} for "
                        f"{candidate.version} of {target.target_package_name!r}"
                    )
                    range_supported = False
                if npm_snapshot is None:
                    child, ambiguous = _nearest_candidate_child(
                        source, requirement.package_name, subgraph.occurrences
                    )
                else:
                    package = resolve_lockfile_dependency_package(
                        npm_snapshot,
                        target.occurrence_id,
                        requirement.package_name,
                    )
                    child = None
                    ambiguous = False
                    if package is not None:
                        matching_occurrence = next(
                            (
                                occurrence
                                for occurrence in npm_snapshot.occurrences
                                if occurrence.manifest_path == package.manifest_path
                                and occurrence.package_name == package.package_name
                                and occurrence.lockfile_package_key == package.package_key
                            ),
                            None,
                        )
                        child_id = (
                            matching_occurrence.occurrence_id
                            if matching_occurrence is not None
                            else make_occurrence_id(
                                package.manifest_path,
                                package.package_name,
                                package.package_key,
                            )
                        )
                        child = evidence_by_id.get(child_id)
                        if child is None:
                            child = SolverPackageOccurrence(
                                occurrence_id=child_id,
                                manifest_path=package.manifest_path,
                                package_name=package.package_name,
                                lockfile_package_key=package.package_key,
                                installed_version=package.version,
                                is_direct=package.package_key.count("node_modules/") == 1,
                                ancestry=list(package.ancestry),
                            )
                            evidence_by_id[child_id] = child
                if child is not None and child.occurrence_id == target.occurrence_id:
                    diagnostics.append(
                        f"candidate dependency {requirement.package_name!r} resolves to its source "
                        f"{target.occurrence_id!r}; left unmodeled"
                    )
                    child = None
                if ambiguous:
                    diagnostics.append(
                        f"ambiguous physical dependency {requirement.package_name!r} for "
                        f"{target.occurrence_id!r}@{candidate.version}; left unmodeled"
                    )
                    child = None

                evidence_variable_id: str | None = None
                relation_modelled = False
                if range_supported and not requirement.is_optional:
                    if requirement.kind == "dependency":
                        if child is not None and child.occurrence_id in eligible_target_ids:
                            relation_modelled = True
                        else:
                            evidence_domain = evidence_domains_by_key.get(
                                (
                                    target.occurrence_id,
                                    requirement.package_name,
                                    requirement.kind,
                                )
                            )
                            if evidence_domain is not None:
                                evidence_variable_id = evidence_domain.variable_id
                                relation_modelled = True
                    elif requirement.kind == "peer":
                        relation_modelled = bool(child is not None and child.installed_version)

                relation = SolverCandidateRelation(
                    source_occurrence_id=target.occurrence_id,
                    source_candidate_version=candidate.version,
                    package_name=requirement.package_name,
                    version_range=requirement.version_range,
                    kind=requirement.kind,
                    is_optional=requirement.is_optional,
                    target_occurrence_id=child.occurrence_id if child else None,
                    evidence_variable_id=evidence_variable_id,
                    is_modelled=relation_modelled,
                    is_range_supported=range_supported,
                )
                key = (
                    relation.source_occurrence_id,
                    relation.source_candidate_version,
                    relation.package_name,
                    relation.version_range,
                    relation.kind,
                    relation.is_optional,
                    relation.target_occurrence_id,
                    relation.evidence_variable_id,
                    relation.is_modelled,
                    relation.is_range_supported,
                )
                relations[key] = relation
    return subgraph.model_copy(
        update={
            "candidate_relations": [
                relations[key]
                for key in sorted(
                    relations,
                    key=lambda value: tuple(str(item or "") for item in value),
                )
            ],
            "evidence_domains": sorted(evidence_domains, key=lambda item: item.variable_id),
            "occurrences": [evidence_by_id[key] for key in sorted(evidence_by_id)],
            "valid": valid,
            "diagnostics": sorted(set(diagnostics)),
        }
    )


def extract_solver_subgraph(
    snapshot: NpmGraphSnapshot,
    targets: Sequence[SolverTarget],
    findings: Sequence[SolverFindingRequirement],
    *,
    peer_conflict_pairs: Sequence[tuple[str, str]] = (),
    forced_singleton_task_ids: Sequence[str] = (),
    additional_occurrences: Sequence[SolverPackageOccurrence] = (),
) -> SolverSubgraph:
    """Extract a bounded immutable occurrence graph for portfolio solving.

    Args:
        snapshot: Neutral, immutable npm graph metadata.
        targets: Finding-backed, synthetic, or direct task occurrences already
            deemed safe by the outer preparation boundary.
        findings: Finding coverage requirements, including GHSA-only findings.
        peer_conflict_pairs: Explicit occurrence (or task) pairs that must not
            be silently lost between portfolio iterations.
        forced_singleton_task_ids: Task IDs that cannot be atomically coupled.

    Returns:
        A deterministic :class:`SolverSubgraph`. Finding uniqueness is keyed
        by occurrence-level coverage identity; malformed references make the
        returned subgraph invalid and non-dispatchable.

    Raises:
        ValueError: If a duplicate identity has conflicting fields or a model
            fails typed validation.
    """
    if not isinstance(snapshot, NpmGraphSnapshot):
        raise TypeError("snapshot must be an NpmGraphSnapshot")
    normalized_targets = _deduplicate_models(
        [_validated_target(value) for value in targets], _target_identity, "target"
    )
    normalized_findings = _deduplicate_models(
        [_validated_finding(value) for value in findings], _finding_identity, "finding coverage"
    )
    by_task: dict[str, SolverTarget] = {}
    for target in normalized_targets:
        previous = by_task.get(target.task_id)
        if previous is not None and previous != target:
            raise ValueError(f"target task identity {target.task_id!r} has conflicting records.")
        by_task[target.task_id] = target
    normalized_targets.sort(key=lambda item: (item.occurrence_id, item.task_id))
    normalized_findings.sort(
        key=lambda item: (
            item.target_occurrence_id,
            item.vulnerable_occurrence_id,
            item.coverage_id,
        )
    )

    diagnostics = list(snapshot.diagnostics)
    valid = True
    target_ids = {target.occurrence_id for target in normalized_targets}
    evidence_occurrences = _bounded_occurrences(snapshot, normalized_targets)
    additional_by_id: dict[str, SolverPackageOccurrence] = {}
    for raw_occurrence in additional_occurrences:
        occurrence = SolverPackageOccurrence.model_validate(raw_occurrence)
        previous = additional_by_id.get(occurrence.occurrence_id)
        if previous is not None and previous != occurrence:
            valid = False
            diagnostics.append(
                f"additional occurrence {occurrence.occurrence_id!r} has conflicting evidence"
            )
            continue
        additional_by_id[occurrence.occurrence_id] = occurrence
    existing_ids = {occurrence.occurrence_id for occurrence in evidence_occurrences}
    evidence_occurrences.extend(
        occurrence
        for occurrence_id, occurrence in sorted(additional_by_id.items())
        if occurrence_id not in existing_ids
    )
    evidence_occurrences.sort(key=lambda item: item.occurrence_id)
    occurrence_index = {occurrence.occurrence_id: occurrence for occurrence in evidence_occurrences}
    for target in normalized_targets:
        occurrence = occurrence_index.get(target.occurrence_id)
        if occurrence is None:
            valid = False
            diagnostics.append(
                f"target occurrence {target.occurrence_id!r} is not present in bounded graph evidence"
            )
        elif (
            occurrence.manifest_path != target.manifest_path
            or occurrence.package_name != target.target_package_name
        ):
            valid = False
            diagnostics.append(
                f"target occurrence {target.occurrence_id!r} does not match graph snapshot identity"
            )
    for finding in normalized_findings:
        if finding.target_occurrence_id not in target_ids:
            valid = False
            diagnostics.append(
                f"finding coverage {finding.coverage_id!r} references unknown mutation target "
                f"{finding.target_occurrence_id!r}"
            )
        occurrence = occurrence_index.get(finding.vulnerable_occurrence_id)
        if occurrence is None:
            valid = False
            diagnostics.append(
                f"finding coverage {finding.coverage_id!r} references unknown vulnerable "
                f"occurrence {finding.vulnerable_occurrence_id!r}"
            )
        elif occurrence.package_name != finding.vulnerable_package:
            valid = False
            diagnostics.append(
                f"finding coverage {finding.coverage_id!r} package does not match vulnerable "
                f"occurrence {finding.vulnerable_occurrence_id!r}"
            )

    pairs, pair_diagnostics = _normalise_pairs(peer_conflict_pairs)
    diagnostics.extend(pair_diagnostics)
    valid = valid and not pair_diagnostics
    task_to_occurrence = {target.task_id: target.occurrence_id for target in normalized_targets}
    external_prerequisites: set[str] = set()
    edges, peer_constraints = _build_metadata_edges(
        snapshot, normalized_targets, diagnostics=diagnostics
    )
    fixed_peer_constraints, fixed_peers_valid = _build_fixed_peer_constraints(
        snapshot,
        normalized_targets,
        evidence_occurrences,
        diagnostics=diagnostics,
    )
    peer_constraints.extend(fixed_peer_constraints)
    valid = valid and fixed_peers_valid
    edges.extend(_build_ancestry_edges(normalized_targets))
    edges.extend(_build_boundary_edges(snapshot, normalized_targets))
    for raw_left, raw_right in pairs:
        left = task_to_occurrence.get(raw_left, raw_left)
        right = task_to_occurrence.get(raw_right, raw_right)
        if left not in target_ids:
            valid = False
            diagnostics.append(f"peer conflict endpoint {raw_left!r} is external or unknown")
            external_prerequisites.add(raw_left)
        if right not in target_ids:
            valid = False
            diagnostics.append(f"peer conflict endpoint {raw_right!r} is external or unknown")
            external_prerequisites.add(raw_right)
        if left not in target_ids or right not in target_ids or left == right:
            continue
        source = next(target for target in normalized_targets if target.occurrence_id == left)
        target = next(target for target in normalized_targets if target.occurrence_id == right)
        relation = _edge(source, target, "peer_conflict", is_peer_coupling=True)
        if relation is not None:
            edges.append(relation)
        reverse = _edge(target, source, "peer_conflict", is_peer_coupling=True)
        if reverse is not None:
            edges.append(reverse)
    forced = sorted(
        {str(task_id).strip() for task_id in forced_singleton_task_ids if str(task_id).strip()}
    )
    for task_id in forced:
        if task_id not in by_task:
            valid = False
            diagnostics.append(f"forced singleton task ID {task_id!r} is external or unknown")
    # Explicit task-level conflicts are represented as occurrence-level graph
    # edges above. Deduplicate edges before model creation for stable digests.
    unique_edges: dict[tuple[Any, ...], SolverEdge] = {}
    for edge in edges:
        key = (
            edge.source_occurrence_id,
            edge.target_occurrence_id,
            edge.edge_kind,
            edge.version_range,
            edge.is_optional,
            edge.is_peer_coupling,
        )
        unique_edges[key] = edge
    unique_peers: dict[tuple[Any, ...], SolverPeerConstraint] = {}
    for peer in peer_constraints:
        key = (
            peer.source_occurrence_id,
            peer.target_occurrence_id,
            peer.version_range,
            peer.is_strict,
            peer.is_optional,
            peer.candidate_specific_source_version,
        )
        unique_peers[key] = peer
    return SolverSubgraph(
        targets=normalized_targets,
        occurrences=evidence_occurrences,
        findings=normalized_findings,
        peer_constraints=sorted(
            unique_peers.values(),
            key=lambda item: (
                item.source_occurrence_id,
                item.target_occurrence_id,
                item.version_range,
                item.is_optional,
            ),
        ),
        edges=sorted(
            unique_edges.values(),
            key=lambda item: (
                item.source_occurrence_id,
                item.target_occurrence_id,
                item.edge_kind,
                item.version_range or "",
                item.is_optional,
            ),
        ),
        forced_singleton_task_ids=forced,
        external_prerequisite_task_ids=sorted(external_prerequisites),
        valid=valid,
        diagnostics=sorted(set(diagnostics)),
    )


__all__ = [
    "DisjointSetUnion",
    "MAX_MULTI_PACKAGE_ACTION_SIZE",
    "build_dependency_dag",
    "cluster_packages",
    "expand_candidate_relations",
    "extract_solver_subgraph",
    "schedule_batches",
]
