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
    FixPlanStatus,
    IssueSource
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


def test_react_phase5_workaround_graph():
    print("==================================================")
    print("🤖 STARTING PHASE 5 REACT LANGGRAPH WORKAROUND TEST")
    print("==================================================")

    repo_root = os.path.abspath("./data/clones/juice-shop")

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 1: jsonwebtoken (Algorithm Confusion workaround)
    # ---------------------------------------------------------
    issue_jwt = VulnerabilityIssue(
        id=str(uuid.uuid4()), 
        issue_type=IssueType.SCA, 
        source=IssueSource.ODC,
        file_path="routes/verify.ts", 
        package_name="jsonwebtoken", 
        cve_id="CVE-2015-9235", 
        severity=Severity.HIGH
    )
    plan_jwt = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND, 
        strategy_used="serper", 
        fixed_version=None,
        instruction='Analyze the provided workaround_snippets to determine if a code edit can safely mitigate this vulnerability. Modify routes/verify.ts to explicitly specify algorithms: ["RS256"] when verifying JWTs to prevent algorithm confusion attacks.',
        workaround_snippets=['jwt.verify(token, publicKey, { algorithms: ["RS256"] }, callback)']
    )
    group_jwt = VulnerabilityGroup(
        group_id="sca:routes/verify.ts:jsonwebtoken:WORKAROUND", 
        issue_type=IssueType.SCA, 
        vulnerable_component="jsonwebtoken",
        file_path="routes/verify.ts", 
        cve_ids=["CVE-2015-9235"], 
        sources=[IssueSource.ODC],
        representative_issue_id=issue_jwt.id, 
        issues=[issue_jwt], 
        fix_plan=plan_jwt
    )

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 2: express-jwt (Missing validation workaround)
    # ---------------------------------------------------------
    issue_express_jwt = VulnerabilityIssue(
        id=str(uuid.uuid4()), 
        issue_type=IssueType.SCA, 
        source=IssueSource.ODC,
        file_path="lib/insecurity.ts", 
        package_name="express-jwt", 
        cve_id="CVE-2020-15084", 
        severity=Severity.CRITICAL
    )
    plan_express_jwt = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND, 
        strategy_used="serper", 
        fixed_version=None,
        instruction='Analyze the provided workaround_snippets to determine if a code edit can safely mitigate this vulnerability. In lib/insecurity.ts, update the expressJwt options to explicitly enforce algorithms (e.g. ["RS256"]).',
        workaround_snippets=['expressJwt({ secret: publicKey, algorithms: ["RS256"] })']
    )
    group_express_jwt = VulnerabilityGroup(
        group_id="sca:lib/insecurity.ts:express-jwt:WORKAROUND", 
        issue_type=IssueType.SCA, 
        vulnerable_component="express-jwt",
        file_path="lib/insecurity.ts", 
        cve_ids=["CVE-2020-15084"], 
        sources=[IssueSource.ODC],
        representative_issue_id=issue_express_jwt.id, 
        issues=[issue_express_jwt], 
        fix_plan=plan_express_jwt
    )

    # ---------------------------------------------------------
    # MOCK VULNERABILITY 3: sanitize-html (Bypass workaround)
    # ---------------------------------------------------------
    issue_sanitize = VulnerabilityIssue(
        id=str(uuid.uuid4()), 
        issue_type=IssueType.SCA, 
        source=IssueSource.ODC,
        file_path="lib/insecurity.ts", 
        package_name="sanitize-html", 
        cve_id="CVE-2021-23424", 
        severity=Severity.HIGH
    )
    plan_sanitize = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND, 
        strategy_used="serper", 
        fixed_version=None,
        instruction='Analyze the provided workaround_snippets to determine if a code edit can safely mitigate this vulnerability. In lib/insecurity.ts, supply custom options (such as disabling allowedTags and allowedAttributes) to sanitizeHtmlLib to avoid sanitization bypass.',
        workaround_snippets=['sanitizeHtml(html, { allowedTags: [], allowedAttributes: {} })']
    )
    group_sanitize = VulnerabilityGroup(
        group_id="sca:lib/insecurity.ts:sanitize-html:WORKAROUND", 
        issue_type=IssueType.SCA, 
        vulnerable_component="sanitize-html",
        file_path="lib/insecurity.ts", 
        cve_ids=["CVE-2021-23424"], 
        sources=[IssueSource.ODC],
        representative_issue_id=issue_sanitize.id, 
        issues=[issue_sanitize], 
        fix_plan=plan_sanitize
    )

    # ---------------------------------------------------------
    # GRAPH EXECUTION
    # ---------------------------------------------------------
    print(f"\n▶️  INVOKING ORCHESTRATOR FOR {repo_root}")
    print("   Providing 3 Vulnerability Groups:")
    print("   1. jsonwebtoken (routes/verify.ts - code-level workaround)")
    print("   2. express-jwt (lib/insecurity.ts - code-level workaround)")
    print("   3. sanitize-html (lib/insecurity.ts - code-level workaround)")
    print("   " + "-"*60)
    
    final_state = run_orchestrator(
        repo_root=repo_root,
        valid_groups=[group_jwt, group_express_jwt, group_sanitize]
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
    test_react_phase5_workaround_graph()
