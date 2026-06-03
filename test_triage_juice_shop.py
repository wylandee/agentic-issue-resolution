import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# --- Contracts ---
from src.contracts.schemas import VulnerabilityIssue, SystemContext, IssueType, FixPlan

# --- Senses (Locators & Planners) ---
from src.tools.manifest_locator import locate_from_issue as locate_sca
from src.tools.fix_planner import plan_fix

# --- Brain (Triage Layer) ---
from src.triage.grouper import group_issues
from src.triage.enrichment import enrich_cves
from src.triage.reachability import analyze_reachability
from src.triage.agent import run_triage

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_triage_only_test(jsonl_path: str, repo_root: str, max_issues: int = 5):
    print("==================================================")
    print(f"STARTING SHIFT-LEFT TRIAGE TEST (Capped at {max_issues})")
    print("==================================================")

    if not os.path.exists(jsonl_path):
        print(f"Error: Could not find JSONL file at {jsonl_path}")
        return

    # 1. INGESTION
    raw_issues = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            raw_issues.append(VulnerabilityIssue(**json.loads(line)))
            if len(raw_issues) >= max_issues:
                break
    print(f"\n[STEP 1] Ingested {len(raw_issues)} raw vulnerabilities.")

    # 2. LOCATE & PLAN (Shift-Left)
    print("\n[STEP 2] Locating & Planning Fixes...")
    sca_plans = []
    for issue in raw_issues:
        if issue.issue_type == IssueType.SCA:
            localized = locate_sca(issue, Path(repo_root))
            plan_dict = plan_fix(localized)
            plan = FixPlan(**plan_dict)
            sca_plans.append((localized, plan))
            print(f"   ↳ {issue.package_name}: {plan.status.value} ({plan.fixed_version or 'No Version'})")

    # 3. STRATEGY-BASED GROUPING
    print("\n[STEP 3] Grouping by Strategy & Calculating High-Water Mark...")
    groups = group_issues(issues=raw_issues, sca_issue_plans=sca_plans)
    print(f"   Reduced {len(raw_issues)} raw issues into {len(groups)} distinct action groups.")

    # 4. ENRICHMENT & REACHABILITY
    print("\n🌐 [STEP 4] Enrichment (AST Reachability & EPSS/KEV)...")
    analyze_reachability(groups, repo_root)
    
    all_cves = list({cve for group in groups for cve in group.cve_ids})
    enrichment_data = enrich_cves(all_cves)
    
    for group in groups:
        if group.cve_ids:
            # Attach highest EPSS enrichment
            group.enrichment = max(
                (enrichment_data.get(cve) for cve in group.cve_ids if cve in enrichment_data),
                key=lambda x: x.epss if x else 0,
                default=None
            )

    # 5. LLM TRIAGE
    print("\n[STEP 5] Executing AI Triage & RBVM Guardrails...")
    context = SystemContext(
        is_public_facing=True,
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs"
    )

    print("\n" + "="*50)
    print("TRIAGE RESULTS")
    print("="*50)

    for group in groups:
        result = run_triage(group, context)
        
        status_icon = "VALID" if result.is_valid else "FALSE POSITIVE"
        
        print(f"\n Group: {group.vulnerable_component} (in {group.file_path})")
        print(f"   ↳ Unified Plan: {group.fix_plan.status.value if group.fix_plan else 'UNKNOWN'}")
        print(f"   ↳ Target Ver. : {group.fix_plan.fixed_version if group.fix_plan else 'N/A'}")
        print(f"   ↳ Status      : {status_icon}")
        
        if not result.is_valid:
            print(f"   ↳ FP Reason   : {result.false_positive_reason}")
            print(f"   ↳ Val. Conf.  : {result.validity_confidence_score} / 1.0")
        else:
            print(f"   ↳ Priority    : {result.revised_priority.name}")
            print(f"   ↳ Pri. Conf.  : {result.priority_confidence_score} / 1.0")
            print(f"   ↳ Reasoning   : {result.priority_reasoning}")
        
        print(f"\n   [Chain of Thought]\n   {result.chain_of_thought}\n")

if __name__ == "__main__":
    os.environ["TRIAGE_LLM_ENABLED"] = "true"
    os.environ["TRIAGE_LLM_MODEL"] = "gpt-4o-mini"
    
    INGESTED_JSONL_PATH = "./data/odc_issues.jsonl"
    JUICE_SHOP_REPO = os.path.abspath("./data/clones/juice-shop")
    
    run_triage_only_test(INGESTED_JSONL_PATH, JUICE_SHOP_REPO, max_issues=5)