import json
from pathlib import Path
import os
from dotenv import load_dotenv

# Import our Contracts and Tools
from src.contracts import VulnerabilityIssue, FixPlan
from src.tools.manifest_locator import locate_from_issue
from src.tools.fix_planner import plan_fix

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
load_dotenv()

def test_planner_pipeline():
    project_root = Path(__file__).resolve().parent
    jsonl_path = project_root / "data" / "odc_issues.jsonl"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not jsonl_path.exists() or not repo_path.exists():
        print("❌ Error: Missing jsonl file or juice-shop clone.")
        return

    print("✅ Found JSONL and Codebase. Starting Fix Planner Test...\n")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            # 1. Ingestion: Parse the raw finding into a strict Contract
            issue = VulnerabilityIssue.model_validate_json(line)
            print(f"--- 🔍 Processing: {issue.package_name} (Vuln: {issue.rule_id or issue.cve_id}) ---")
            
            # 2. Triage Phase: Locate the issue in the codebase
            # (Note: Assuming you applied the markdown prompt to strip OSV from the locator!)
            localized_issue = locate_from_issue(issue, repo_path)
            print(f"📍 Localized in: {localized_issue.manifest_file} (Direct: {localized_issue.is_direct_dependency})")
            
            # 3. Plan Phase: Run the 5-Step Waterfall to find the fix
            raw_plan_dict = plan_fix(localized_issue)
            
            # 4. Strict Validation: Convert the raw dict into a Pydantic FixPlan model
            final_plan = FixPlan(**raw_plan_dict)
            
            # Print the beautifully planned instructions!
            print(final_plan.model_dump_json(indent=2))
            print("-" * 60 + "\n")
            
            count += 1

def test_planner_serper_path():
    project_root = Path(__file__).resolve().parent
    jsonl_path = project_root / "data" / "odc_issues.jsonl"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not jsonl_path.exists() or not repo_path.exists():
        print("❌ Error: Missing jsonl file or juice-shop clone.")
        return

    # Check if Serper API key is in the environment!
    if not os.environ.get("SERPER_API_KEY"):
        print("⚠️ WARNING: SERPER_API_KEY is not set!")
        print("The script will skip Step 4 and jump to Step 5 (Graceful Abort).")
        print("Please export your key in the terminal before running.\n")

    print("✅ Found JSONL and Codebase. Testing Serper Path...\n")

    targets = ["blaze", "moment"]
    tested_packages = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            issue = VulnerabilityIssue.model_validate_json(line)
            pkg_name = (issue.package_name or "").lower()

            # Check if this package is one of our targets
            is_target = any(target in pkg_name for target in targets)
            
            # If it is a target, and we haven't tested it yet
            if is_target and pkg_name not in tested_packages:
                print(f"--- 🔍 Processing: {issue.package_name} (Vuln: {issue.rule_id or issue.cve_id}) ---")
                
                localized_issue = locate_from_issue(issue, repo_path)
                print(f"📍 Localized in: {localized_issue.manifest_file}")
                
                raw_plan_dict = plan_fix(localized_issue)
                final_plan = FixPlan(**raw_plan_dict)
                
                print(final_plan.model_dump_json(indent=2))
                print("-" * 60 + "\n")
                
                # Mark as tested so we don't repeat the same package
                tested_packages.add(pkg_name)
                
                # Stop once we have found and tested both targets
                if len(tested_packages) >= 2:
                    break

if __name__ == "__main__":
    #test_planner_pipeline()
    test_planner_serper_path()