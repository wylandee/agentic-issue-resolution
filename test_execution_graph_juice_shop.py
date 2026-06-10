import os
import uuid
import logging
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, ToolMessage

from src.contracts.schemas import (
    VulnerabilityGroup, 
    VulnerabilityIssue, 
    IssueType, 
    Severity, 
    FixPlan, 
    FixPlanStatus
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
# 🪄 2. LIVE TOOL CALL INTERCEPTOR (NEW)
# Prints each tool execution and its outcome LIVE as the LLM executes them
# ============================================================================
original_invoke_bound_tool = remedy_agent._invoke_bound_tool

def invoke_bound_tool_interceptor(*args, **kwargs):
    # args[0] is tool_map, args[1] is tool_call dict
    tool_call = kwargs.get("tool_call") or args[1]
    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})
    
    print("\n" + "─"*60)
    print(f"🛠️  [LIVE TOOL CALL]: {tool_name}")
    print(f"   Args: {tool_args}")
    print("─"*60)
    
    # Execute the actual tool
    tool_message = original_invoke_bound_tool(*args, **kwargs)
    
    # Truncate long tool responses (like large file reads) for clean terminal printing
    content_preview = tool_message.content[:200].replace("\n", " ")
    if len(tool_message.content) > 200:
        content_preview += " ... [TRUNCATED]"
        
    print(f"✅ [LIVE TOOL RESULT]: {content_preview}")
    print("─"*60 + "\n")
    
    return tool_message

# Apply the patch
remedy_agent._invoke_bound_tool = invoke_bound_tool_interceptor


def test_react_phase5_graph():
    print("==================================================")
    print("🤖 STARTING PHASE 5 REACT LANGGRAPH TEST (MULTI-CVE)")
    print("==================================================")

    repo_root = os.path.abspath("./data/clones/juice-shop")

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 1: lodash
    # ---------------------------------------------------------
    issue_lodash = VulnerabilityIssue(
        id=str(uuid.uuid4()), issue_type=IssueType.SCA, source="odc",
        file_path="package.json", package_name="lodash", cve_id="CVE-2019-10744", severity=Severity.HIGH
    )
    plan_lodash = FixPlan(
        status=FixPlanStatus.VERSION_FOUND, strategy_used="osv_api", fixed_version="4.17.21",
        instruction='Add or update "overrides": {"lodash": "4.17.21"} in package.json to pin the transitive dependency via npm overrides.',
        workaround_snippets=None
    )
    group_lodash = VulnerabilityGroup(
        group_id="sca:package.json:lodash:UPDATE", issue_type=IssueType.SCA, vulnerable_component="lodash",
        file_path="package.json", cve_ids=["CVE-2019-10744"], sources=["odc"],
        representative_issue_id=issue_lodash.id, issues=[issue_lodash], fix_plan=plan_lodash
    )

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 2: lodash.set
    # ---------------------------------------------------------
    issue_lodash_set = VulnerabilityIssue(
        id=str(uuid.uuid4()), issue_type=IssueType.SCA, source="odc",
        file_path="package.json", package_name="lodash.set", ghsa_id="GHSA-p6mc-m468-83gw", severity=Severity.HIGH
    )
    plan_lodash_set = FixPlan(
        status=FixPlanStatus.VERSION_FOUND, strategy_used="npm_registry", fixed_version="4.3.2",
        instruction='Add or update "overrides": {"lodash.set": "4.3.2"} in package.json to pin the transitive dependency via npm overrides.',
        workaround_snippets=None
    )
    group_lodash_set = VulnerabilityGroup(
        group_id="sca:package.json:lodash.set:UPDATE", issue_type=IssueType.SCA, vulnerable_component="lodash.set",
        file_path="package.json", ghsa_ids=["GHSA-rpr9-rxv7-x643"], sources=["odc"],
        representative_issue_id=issue_lodash_set.id, issues=[issue_lodash_set], fix_plan=plan_lodash_set
    )

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 3: sanitize-html
    # ---------------------------------------------------------
    issue_sanitize = VulnerabilityIssue(
        id=str(uuid.uuid4()), issue_type=IssueType.SCA, source="odc",
        file_path="package.json", package_name="sanitize-html", ghsa_id="GHSA-rpr9-rxv7-x643", severity=Severity.HIGH
    )
    plan_sanitize = FixPlan(
        status=FixPlanStatus.VERSION_FOUND, strategy_used="osv_api", fixed_version="1.4.3",
        instruction='Update "sanitize-html" in package.json to version "1.4.3".',
        workaround_snippets=None
    )
    group_sanitize = VulnerabilityGroup(
        group_id="sca:package.json:sanitize-html:UPDATE", issue_type=IssueType.SCA, vulnerable_component="sanitize-html",
        file_path="package.json", ghsa_ids=["GHSA-rpr9-rxv7-x643"], sources=["odc"],
        representative_issue_id=issue_sanitize.id, issues=[issue_sanitize], fix_plan=plan_sanitize
    )

    # ---------------------------------------------------------
    # GRAPH EXECUTION
    # ---------------------------------------------------------
    print(f"\n▶️  INVOKING ORCHESTRATOR FOR {repo_root}")
    print("   Providing 3 Vulnerability Groups:")
    print("   1. lodash (transitive)")
    print("   2. lodash.set (transitive)")
    print("   3. sanitize-html (direct)")
    print("   " + "-"*60)
    
    final_state = run_orchestrator(
        repo_root=repo_root,
        valid_groups=[group_lodash, group_lodash_set, group_sanitize]
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
    # 🕵️‍♂️ INSPECT THE LLM's ReAct CONVERSATION
    # =========================================================
    messages = final_state.get("messages", [])
    if messages:
        print("\n   [💬 AGENT TRANSCRIPT (Summary of final history)]")
        print("   " + "-"*60)
        for msg in messages:
            if isinstance(msg, AIMessage):
                if msg.content:
                    print(f"   🤖 LLM: {msg.content}")
                for tool_call in getattr(msg, "tool_calls", []):
                    print(f"   🛠️  TOOL INSTRUCTION: {tool_call['name']}")
                    print(f"       Args: {tool_call['args']}")
            elif isinstance(msg, ToolMessage):
                content_preview = msg.content[:200].replace("\n", " ")
                if len(msg.content) > 200:
                    content_preview += " ... [TRUNCATED]"
                print(f"   ✅ TOOL OUTCOME: {content_preview}")
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
    test_react_phase5_graph()