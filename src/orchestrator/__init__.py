"""
src/orchestrator — Phase 4.1 LangGraph remediation orchestrator.

Public entry points
-------------------
``build_remediation_graph()``  — compile the StateGraph into a runnable app.
``run_remediation(...)``       — convenience wrapper: invoke the graph for one issue.

Import examples::

    from src.orchestrator import run_remediation
    from src.orchestrator.graph import build_remediation_graph, remediation_engine
    from src.orchestrator.state import RemediationState
"""

from src.orchestrator.graph import build_remediation_graph, remediation_engine, run_remediation

__all__ = ["build_remediation_graph", "remediation_engine", "run_remediation"]
