import json
from pathlib import Path
from dotenv import load_dotenv

# Import our Contracts and Tools
from src.contracts import VulnerabilityIssue, FixPlan, EditRequest
from src.tools.manifest_locator import locate_from_issue
from src.tools.fix_planner import plan_fix
from src.tools.edit_tools import apply_edit

def test_full_pipeline():
    load_dotenv()
    
    project_root = Path(__file__).resolve().parent
    jsonl_path = project_root / "data" / "odc_issues.jsonl"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not jsonl_path.exists() or not repo_path.exists():
        print("❌ Error: Missing jsonl file or juice-shop clone.")
        return

    print("✅ Starting End-to-End Edit Test...\n")

    # We will specifically hunt for 'express-jwt' since it's a known direct dependency
    target_package = "file-type"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            issue = VulnerabilityIssue.model_validate_json(line)
            
            if (issue.package_name or "").lower() == target_package:
                print(f"--- 1️⃣ TRIAGE: Locating {issue.package_name}... ---")
                localized = locate_from_issue(issue, repo_path)
                print(f"📍 Found in: {localized.manifest_file} on line {localized.manifest_line}\n")
                
                print(f"--- 2️⃣ PLAN: Finding safe version... ---")
                raw_plan = plan_fix(localized)
                plan = FixPlan(**raw_plan)
                
                # If OSV or NPM couldn't find a version, we'll fake one for the test
                safe_version = plan.fixed_version or "99.9.9"
                print(f"💡 Strategy used: {plan.strategy_used}")
                print(f"💡 Instruction: {plan.instruction}\n")
                
                print(f"--- 3️⃣ THE DUMMY AGENT (Simulating LLM) ---")
                # The LLM would normally read the snippet and generate this old/new text
                old_text = localized.manifest_snippet
                old_version = issue.package_version
                
                # We simulate the LLM replacing the old version with the safe version
                new_text = old_text.replace(old_version, safe_version)
                
                print("Old Text:\n" + old_text)
                print("\nNew Text:\n" + new_text + "\n")
                
                print(f"--- 4️⃣ EDIT: Applying to hard drive... ---")
                request = EditRequest(
                    repo_root=str(repo_path),
                    file_path=localized.manifest_file,
                    old_text=old_text,
                    new_text=new_text,
                    dry_run=False  # Set to False so it actually saves to disk!
                )
                
                result = apply_edit(request)
                
                print(f"✅ Status: {result.status.value}")
                if result.rejection_reason:
                    print(f"❌ Rejection Reason: {result.rejection_reason}")
                
                if result.unified_diff:
                    print("\n--- 🛠️ THE GIT DIFF ---")
                    print(result.unified_diff)
                
                # Break after fixing one so we don't spam the terminal
                break

if __name__ == "__main__":
    test_full_pipeline()