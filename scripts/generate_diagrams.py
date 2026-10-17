"""Generate architecture diagrams for the current remediation graph."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def generate_mermaid_code() -> str:
    """Return Mermaid source matching the registered LangGraph topology."""
    return """flowchart TD
    Start([START]) --> Initial[initial_triage]
    Initial -->|actionable| Builder[workspace_builder]
    Initial -->|no work or failure| Teardown[teardown]
    Builder -->|ready| SetupScan[workspace_setup_scan]
    Builder -->|setup failure| Teardown
    SetupScan --> Supervisor[Supervisor<br/>plans and certifies inside this node]

    Supervisor -->|committed update| Update[update_subagent]
    Supervisor -->|committed workaround| Workaround[workaround_subagent]
    Supervisor -->|validate attempt| QA[qa_critic]
    Supervisor -->|post-QA reconcile| Triage[triage]
    Supervisor -->|optional full scan| FullScan[final_full_scan]
    Supervisor -->|finished| Teardown

    Update --> Supervisor
    Workaround --> Supervisor
    QA --> Supervisor
    Triage -->|reconciled| Supervisor
    Triage -->|no work or failure| Teardown
    FullScan --> Supervisor
    Teardown --> Report[report]
    Report --> End([END])
"""


def generate_dot_code() -> str:
    """Return Graphviz source matching the registered LangGraph topology."""
    return r"""digraph AgentFlow {
    rankdir=TB;
    graph [fontname="Helvetica,Arial,sans-serif", fontsize=11, pad="0.4", nodesep="0.55", ranksep="0.65"];
    node [fontname="Helvetica,Arial,sans-serif", fontsize=10, shape=box, style="rounded,filled", margin="0.16,0.10"];
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=9];

    start [label="START", shape=circle, fillcolor="#1F2937", color="#111827", fontcolor="#FFFFFF"];
    end [label="END", shape=doublecircle, fillcolor="#1F2937", color="#111827", fontcolor="#FFFFFF"];
    initial_triage [label="initial_triage", fillcolor="#FEF3C7", color="#F59E0B"];
    workspace_builder [label="workspace_builder", fillcolor="#DBEAFE", color="#3B82F6"];
    workspace_setup_scan [label="workspace_setup_scan", fillcolor="#F3F4F6", color="#6B7280"];
    supervisor [label="supervisor\n(planning and certification are functions inside this node)", fillcolor="#EDE9FE", color="#8B5CF6", fontcolor="#5B21B6", penwidth=2.0];
    update_subagent [label="update_subagent", fillcolor="#DBEAFE", color="#3B82F6"];
    workaround_subagent [label="workaround_subagent", fillcolor="#DBEAFE", color="#3B82F6"];
    qa_critic [label="qa_critic", fillcolor="#FEE2E2", color="#EF4444"];
    triage [label="triage\n(post-QA reconciliation)", fillcolor="#FEF3C7", color="#F59E0B"];
    final_full_scan [label="final_full_scan\n(optional)", fillcolor="#F3F4F6", color="#6B7280"];
    teardown [label="teardown", fillcolor="#D1FAE5", color="#10B981"];
    report [label="report", fillcolor="#D1FAE5", color="#10B981"];

    start -> initial_triage;
    initial_triage -> workspace_builder [label="actionable"];
    initial_triage -> teardown [label="no work / failure"];
    workspace_builder -> workspace_setup_scan [label="ready"];
    workspace_builder -> teardown [label="failure"];
    workspace_setup_scan -> supervisor;
    supervisor -> update_subagent [label="committed update"];
    supervisor -> workaround_subagent [label="committed workaround"];
    supervisor -> qa_critic [label="validate attempt"];
    supervisor -> triage [label="post-QA reconcile"];
    supervisor -> final_full_scan [label="optional full scan"];
    supervisor -> teardown [label="finished"];
    update_subagent -> supervisor;
    workaround_subagent -> supervisor;
    qa_critic -> supervisor;
    triage -> supervisor [label="reconciled"];
    triage -> teardown [label="no work / failure"];
    final_full_scan -> supervisor;
    teardown -> report;
    report -> end;
}
"""


def main() -> None:
    """Write Mermaid and Graphviz sources and render PNG when Graphviz exists."""
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    mmd_path = docs_dir / "agent_flow.mmd"
    mmd_path.write_text(generate_mermaid_code(), encoding="utf-8")
    print(f"Generated Mermaid file: {mmd_path}")

    dot_path = docs_dir / "agent_flow.dot"
    dot_path.write_text(generate_dot_code(), encoding="utf-8")
    print(f"Generated DOT file: {dot_path}")

    dot_binary = shutil.which("dot")
    if dot_binary is None:
        print("Graphviz dot is unavailable; PNG was not regenerated.")
        return
    png_path = docs_dir / "agent_flow.png"
    try:
        subprocess.run(
            [dot_binary, "-Tpng", "-Gdpi=200", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Generated PNG file: {png_path}")
    except subprocess.CalledProcessError as exc:
        print(f"Could not render with dot: {exc.stderr.strip()}")


if __name__ == "__main__":
    main()
