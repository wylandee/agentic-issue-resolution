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

# Keep basic config so dependency modules (like LangGraph/Docker) can still log if needed
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ============================================================================
# 🪄 PROMPT INTERCEPTOR (Updated for Bulk resolved_groups)
# ============================================================================
original_build_prompt = remedy_agent._build_prompt

def prompt_interceptor(*args, **kwargs):
    prompt = original_build_prompt(*args, **kwargs)
    
    # args[0] is now resolved_groups: List[Tuple[VulnerabilityGroup, str]]
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
        file_path="package.json", package_name="lodash.set", cve_id=None, severity=Severity.HIGH
    )
    plan_lodash_set = FixPlan(
        status=FixPlanStatus.VERSION_FOUND, strategy_used="npm_registry", fixed_version="4.3.2",
        instruction='Add or update "overrides": {"lodash.set": "4.3.2"} in package.json to pin the transitive dependency via npm overrides.',
        workaround_snippets=None
    )
    group_lodash_set = VulnerabilityGroup(
        group_id="sca:package.json:lodash.set:UPDATE", issue_type=IssueType.SCA, vulnerable_component="lodash.set",
        file_path="package.json", cve_ids=[], sources=["odc"],
        representative_issue_id=issue_lodash_set.id, issues=[issue_lodash_set], fix_plan=plan_lodash_set
    )

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 3: sanitize-html
    # ---------------------------------------------------------
    issue_sanitize = VulnerabilityIssue(
        id=str(uuid.uuid4()), issue_type=IssueType.SCA, source="odc",
        file_path="package.json", package_name="sanitize-html", cve_id=None, severity=Severity.HIGH
    )
    plan_sanitize = FixPlan(
        status=FixPlanStatus.VERSION_FOUND, strategy_used="osv_api", fixed_version="1.4.3",
        instruction='Update "sanitize-html" in package.json to version "1.4.3".',
        workaround_snippets=None
    )
    group_sanitize = VulnerabilityGroup(
        group_id="sca:package.json:sanitize-html:UPDATE", issue_type=IssueType.SCA, vulnerable_component="sanitize-html",
        file_path="package.json", cve_ids=[], sources=["odc"],
        representative_issue_id=issue_sanitize.id, issues=[issue_sanitize], fix_plan=plan_sanitize
    )

    # ---------------------------------------------------------
    # GRAPH EXECUTION
    # ---------------------------------------------------------
    print(f"\n▶️  INVOKING ORCHESTRATOR FOR {repo_root}")
    
    final_state = run_orchestrator(
        repo_root=repo_root,
        valid_groups=[group_lodash, group_lodash_set, group_sanitize]
    )
    
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
    #🕵️‍♂️ INSPECT THE LLM's ReAct CONVERSATION
    # =========================================================
    messages = final_state.get("messages", [])
    if messages:
        print("\n   [💬 AGENT TRANSCRIPT (Tools & Actions)]")
        print("   " + "-"*60)
        for msg in messages:
            # LLM's Tool Calls or final text
            if isinstance(msg, AIMessage):
                if msg.content:
                    print(f"   🤖 LLM: {msg.content}")
                for tool_call in getattr(msg, "tool_calls", []):
                    print(f"   🛠️  CALLING TOOL: {tool_call['name']}")
                    print(f"       Args: {tool_call['args']}")
            
            # Python's Tool Responses
            elif isinstance(msg, ToolMessage):
                # Truncate long tool outputs (like file reads) for console viewing
                content_preview = msg.content[:200].replace("\n", " ")
                if len(msg.content) > 200:
                    content_preview += " ... [TRUNCATED]"
                print(f"   ✅ TOOL RESULT: {content_preview}")
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