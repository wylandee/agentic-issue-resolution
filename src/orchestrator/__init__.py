"""
src.orchestrator - LangGraph remediation package for Phase 4.1 and Phase 5.

Public entry points
-------------------
Phase 4.1:
``build_remediation_graph()``
``remediation_engine``
``run_remediation(...)``

Phase 5:
``build_orchestrator_graph()``
``orchestrator_engine``
``run_orchestrator(...)``
"""

__all__ = [
    "build_orchestrator_graph",
    "build_remediation_graph",
    "orchestrator_engine",
    "remediation_engine",
    "run_orchestrator",
    "run_remediation",
]


def __getattr__(name: str):
    """Load graph entry points lazily to avoid triage/exporter import cycles."""
    if name in __all__:
        from src.orchestrator import graph

        return getattr(graph, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
