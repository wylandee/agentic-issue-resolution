import os
import logging
from dotenv import load_dotenv
from pydantic import TypeAdapter

from langchain_core.messages import AIMessage, ToolMessage

from src.contracts.schemas import (
    VulnerabilityGroup,
)
from src.orchestrator.graph import run_orchestrator
import src.orchestrator.remedy_agent as remedy_agent

load_dotenv()

# Keep basic logging configuration active so dependencies (Docker, LangGraph) can still log
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ============================================================================
# 🪄 1. PROMPT INTERCEPTOR (Updated for Bulk resolved_groups)
# ============================================================================
original_build_prompt = remedy_agent._build_prompt

def prompt_interceptor(*args, **kwargs):
    prompt = original_build_prompt(*args, **kwargs)
    
    # Unpack resolved_groups (List[Tuple[VulnerabilityGroup, str]])
    resolved_groups = kwargs.get("resolved_groups") or args[0]
    group_ids = [g[0].group_id for g in resolved_groups]

    log_msg = [
        "\n" + "═"*70,
        f"🧠 [LLM SYSTEM PROMPT INTERCEPTED] Targeting {len(group_ids)} groups:",
        f"   {', '.join(group_ids)}",
        "═"*70,
        prompt,
        "═"*70 + "\n"
    ]
    print("\n".join(log_msg))
    return prompt

remedy_agent._build_prompt = prompt_interceptor


# ============================================================================
# 🪄 2. LIVE TOOL CALL INTERCEPTOR (Updated with Conditional Truncation)
# Prints file reads compactly, but outputs full test/scan stdout in real-time
# ============================================================================
original_invoke_bound_tool = remedy_agent._invoke_bound_tool

def invoke_bound_tool_interceptor(*args, **kwargs):
    tool_call = kwargs.get("tool_call") or args[1]
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    
    print("\n" + "─"*60)
    print(f"🛠️  [LIVE TOOL CALL]: {tool_name}")
    print(f"   Args: {tool_args}")
    print("─"*60)
    
    # Execute the actual tool
    tool_message = original_invoke_bound_tool(*args, **kwargs)
    
    # 🎛️ CONDITIONAL TRUNCATION:
    if tool_name == "read_workspace_file":
        # Keep file reads short and single-line
        content_preview = tool_message.content[:200].replace("\n", " ")
        if len(tool_message.content) > 200:
            content_preview += " ... [TRUNCATED]"
        print(f"   ✅ [LIVE TOOL RESULT]: {content_preview}")
    else:
        # Print the FULL output (retaining newlines, stack traces, and formatting)
        # for install, scan, and unit test tools
        print(f"   ✅ [LIVE TOOL RESULT]:\n{tool_message.content}")
        
    print("─"*60 + "\n")
    return tool_message

# Apply the patch
remedy_agent._invoke_bound_tool = invoke_bound_tool_interceptor


def test_react_full_graph_odc():
    print("==================================================")
    print("🤖 STARTING FULLY CONNECTED REACT LANGGRAPH TEST (TRIAGE + EXECUTION)")
    print("==================================================")

    repo_root = os.path.abspath("./data/clones/juice-shop")
    triaged_groups_path = "./data/cache/triaged_groups.json"

    if not os.path.exists(triaged_groups_path):
        print(f"Error: Could not find triaged groups file at {triaged_groups_path}")
        return

    # 1. INGESTION - load pre-triaged groups from cache
    with open(triaged_groups_path, "r", encoding="utf-8") as f:
        valid_groups = TypeAdapter(list[VulnerabilityGroup]).validate_json(f.read())

    print(f"\n[STEP 1] Ingested {len(valid_groups)} triaged vulnerability groups.")
    for idx, group in enumerate(valid_groups):
        representative = next(
            (issue for issue in group.issues if issue.id == group.representative_issue_id),
            group.issues[0] if group.issues else None,
        )
        if representative is None:
            print(f"   {idx+1}. {group.group_id} - No representative issue found")
            continue
        vuln_id = representative.cve_id or representative.ghsa_id or representative.rule_id or "None"
        print(
            f"   {idx+1}. {group.vulnerable_component or group.group_id} "
            f"(Vuln: {vuln_id}) - Severity: {representative.severity.value}"
        )

    # ---------------------------------------------------------
    # GRAPH EXECUTION
    # ---------------------------------------------------------
    print(f"\n▶️  INVOKING ORCHESTRATOR FOR {repo_root}")
    print("   Providing pre-triaged Vulnerability Groups directly to execution")
    print("   " + "-"*60)
    
    final_state = run_orchestrator(
        repo_root=repo_root,
        valid_groups=valid_groups,
    )
    
    print("   " + "-"*60)
    print("\n✅ GRAPH EXECUTION COMPLETE")

    # ---------------------------------------------------------
    # OUTPUT INSPECTION
    # ---------------------------------------------------------
    final_status = final_state.get("status")
    print(f"\n[📊 FINAL GRAPH STATUS]: {final_status}")
    
    retries = final_state.get("retry_count", 0)
    if retries > 0:
        print(f"   ↳ 🔄 The Agent had to self-correct {retries} time(s).")
        
    errors = final_state.get("errors", [])
    if errors:
        print("\n   [⚠️ ACCUMULATED ERRORS]")
        for err in errors:
            print(f"   | {err}")

    # =========================================================
    # 🕵️‍♂️ INSPECT THE LLM's ReAct CONVERSATION (Conditional Truncation)
    # =========================================================
    messages = final_state.get("messages", [])
    if messages:
        print("\n   [💬 AGENT TRANSCRIPT (Summary of final history)]")
        print("   " + "-"*60)
        for msg in messages:
            # LLM's Tool Calls or final text
            if isinstance(msg, AIMessage):
                if msg.content:
                    print(f"   🤖 LLM: {msg.content}")
                for tool_call in getattr(msg, "tool_calls", []):
                    print(f"   🛠️  TOOL INSTRUCTION: {tool_call['name']}")
                    print(f"       Args: {tool_call['args']}")
            
            # Python's Tool Responses (with conditional truncation)
            elif isinstance(msg, ToolMessage):
                if msg.name == "read_workspace_file":
                    content_preview = msg.content[:200].replace("\n", " ")
                    if len(msg.content) > 200:
                        content_preview += " ... [TRUNCATED]"
                    print(f"   ✅ TOOL OUTCOME: {content_preview}")
                else:
                    print(f"   ✅ TOOL OUTCOME:\n{msg.content}")
        print("   " + "-"*60)

    # Inspect the final diff generated by teardown_node
    diff = final_state.get("diff")
    if diff:
        print("\n   [🔍 GENERATED GIT DIFF]")
        lines = diff.split("\n")
        for line in lines[:50]:
            print(f"   | {line}")
    elif final_status == "completed":
        print("\n   [ℹ️] Graph completed, but no diff was generated (no files changed).")
    else:
        print("\n   [❌] Graph did not complete successfully; no diff generated.")

if __name__ == "__main__":
    test_react_full_graph_odc()
