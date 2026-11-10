"""Pure, occurrence-aware portfolio solver collaborators.

The solver package intentionally contains no orchestration or worker imports.  Its
public functions consume immutable graph snapshots and typed solver contracts.
"""

from .cpsat import solve_portfolio
from .graph import (
    build_dependency_dag,
    cluster_packages,
    expand_candidate_relations,
    extract_solver_subgraph,
    schedule_batches,
)

__all__ = [
    "build_dependency_dag",
    "cluster_packages",
    "expand_candidate_relations",
    "extract_solver_subgraph",
    "schedule_batches",
    "solve_portfolio",
]
