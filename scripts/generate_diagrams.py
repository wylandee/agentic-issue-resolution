"""Generate pure multi-agent flow diagrams in Mermaid (.mmd), Graphviz (.dot), and PNG formats.

Simplified presentation architecture:
1. Triage Agent (initial triage & post-QA re-triage)
2. Supervisor (central attempt authority & router)
3. Update Agent (atomic dependency bump)
4. Code Workaround Agent (phase-gated code/AST fixes)
5. QA Agent (deterministic verification gates)
6. Report Agent (markdown audit report & patch)
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def generate_mermaid_code() -> str:
    return """flowchart TD
    Start([START]):::boundary --> Triage["Triage Agent<br/>(Parse, Group & Reconcile)"]:::triage
    
    Triage -->|Initial Triage Done| Supervisor["SUPERVISOR<br/>(Central Attempt Authority)"]:::supervisor
    Triage -.->|No Actionable Work| Report["Report Agent<br/>(Markdown & Diff Summary)"]:::report
    
    Supervisor -.->|Committed Update| Update["Update Agent<br/>(Atomic Dep Bump)"]:::worker
    Update --> Supervisor
    
    Supervisor -.->|Committed Workaround| Workaround["Code Workaround Agent<br/>(Phase-Gated Code Fix)"]:::worker
    Workaround --> Supervisor
    
    Supervisor -.->|Validate Attempt| QA["QA Agent<br/>(Deterministic QA Gates)"]:::qa
    QA -->|Attempt Evaluation| Supervisor
    
    Supervisor -.->|Post-QA Re-Triage| Triage
    Triage -->|Reconciled| Supervisor
    
    Supervisor -->|All Tasks Evaluated| Report
    Report --> End([END]):::boundary

    classDef boundary fill:#1f2937,stroke:#111827,stroke-width:2px,color:#ffffff;
    classDef supervisor fill:#ede9fe,stroke:#8b5cf6,stroke-width:2.5px,color:#5b21b6;
    classDef worker fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e40af;
    classDef qa fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b;
    classDef triage fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#92400e;
    classDef report fill:#d1fae5,stroke:#10b981,stroke-width:2px,color:#065f46;
"""


def generate_dot_code() -> str:
    return """digraph AgentFlow {
    rankdir=TB;
    graph [fontname="Helvetica,Arial,sans-serif", fontsize=12, pad="0.5", nodesep="0.7", ranksep="0.9"];
    node [fontname="Helvetica,Arial,sans-serif", fontsize=11, shape=box, style="rounded,filled", margin="0.22,0.16"];
    edge [fontname="Helvetica,Arial,sans-serif", fontsize=10];

    __start__ [label="START", shape=circle, fillcolor="#1F2937", color="#111827", fontcolor="#FFFFFF", width=0.8];
    __end__ [label="END", shape=doublecircle, fillcolor="#1F2937", color="#111827", fontcolor="#FFFFFF", width=0.8];

    triage_agent [label="Triage Agent\\n(Parse, Group & Reconcile)", fillcolor="#FEF3C7", color="#F59E0B", fontcolor="#92400E", penwidth=1.8];
    supervisor [label="SUPERVISOR\\n(Central Attempt Authority)", fillcolor="#EDE9FE", color="#8B5CF6", fontcolor="#5B21B6", penwidth=2.5];
    
    update_agent [label="Update Agent\\n(Atomic Dep Bump)", fillcolor="#DBEAFE", color="#3B82F6", fontcolor="#1E40AF", penwidth=1.8];
    workaround_agent [label="Code Workaround Agent\\n(Phase-Gated Code Fix)", fillcolor="#DBEAFE", color="#3B82F6", fontcolor="#1E40AF", penwidth=1.8];
    qa_agent [label="QA Agent\\n(Deterministic QA Gates)", fillcolor="#FEE2E2", color="#EF4444", fontcolor="#991B1B", penwidth=1.8];
    report_agent [label="Report Agent\\n(Markdown & Diff Summary)", fillcolor="#D1FAE5", color="#10B981", fontcolor="#065F46", penwidth=1.8];

    { rank = same; update_agent; workaround_agent; qa_agent; }

    __start__ -> triage_agent [color="#1F2937", penwidth=1.5];
    triage_agent -> supervisor [color="#1F2937", penwidth=1.5, label="Initial Triage Done"];
    triage_agent -> report_agent [style=dashed, color="#6B7280", label="No Actionable Work"];

    supervisor -> update_agent [style=dashed, color="#3B82F6", label="Committed Update"];
    update_agent -> supervisor [color="#1F2937", penwidth=1.2];

    supervisor -> workaround_agent [style=dashed, color="#3B82F6", label="Committed Workaround"];
    workaround_agent -> supervisor [color="#1F2937", penwidth=1.2];

    supervisor -> qa_agent [style=dashed, color="#EF4444", label="Validate Attempt"];
    qa_agent -> supervisor [color="#1F2937", penwidth=1.2, label="Attempt Evaluation"];

    supervisor -> triage_agent [style=dashed, color="#F59E0B", label="Post-QA Re-Triage"];
    triage_agent -> supervisor [color="#1F2937", penwidth=1.2, label="Reconciled"];

    supervisor -> report_agent [color="#1F2937", penwidth=1.5, label="All Tasks Evaluated"];
    report_agent -> __end__ [color="#1F2937", penwidth=1.5];
}
"""


def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)

    mmd_path = docs_dir / "agent_flow.mmd"
    mmd_path.write_text(generate_mermaid_code())
    print(f"Generated Mermaid file: {mmd_path}")

    dot_path = docs_dir / "agent_flow.dot"
    dot_path.write_text(generate_dot_code())
    print(f"Generated DOT file: {dot_path}")

    png_path = docs_dir / "agent_flow.png"
    try:
        subprocess.run(
            ["/opt/homebrew/bin/dot", "-Tpng", "-Gdpi=200", str(dot_path), "-o", str(png_path)],
            check=True,
            capture_output=True,
        )
        print(f"Generated High-Res PNG: {png_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Could not render with dot: {exc}")


if __name__ == "__main__":
    main()
