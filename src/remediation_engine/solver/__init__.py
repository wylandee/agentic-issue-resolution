"""Pure, occurrence-aware portfolio solver collaborators.

The solver package intentionally contains no orchestration or worker imports.  Its
public functions consume immutable graph snapshots and typed solver contracts.
"""

from .cpsat import solve_portfolio
from .graph import build_dependency_dag, cluster_packages, schedule_batches
from .subgraph import extract_solver_subgraph

__all__ = [
    "build_dependency_dag",
    "cluster_packages",
    "extract_solver_subgraph",
    "schedule_batches",
    "solve_portfolio",
]
