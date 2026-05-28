import os
import json
from collections import Counter
import logging

# Adjust these imports based on your exact src module structure
from src.contracts.schemas import VulnerabilityIssue
from src.orchestrator.graph import run_remediation

# Set up logging to show INFO level from our graph nodes
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

def run_batch_remediation(jsonl_path: str, repo_root: str, dry_run: bool = True):
    """
    Reads parsed vulnerabilities from a JSONL file and processes them 
    through the Phase 4.1 LangGraph remediation engine.
    """
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: Could not find JSONL file at {jsonl_path}")
        return

    if not os.path.exists(repo_root):
        print(f"❌ Error: Could not find repository at {repo_root}")
        return

    print(f"🚀 Starting Batch Remediation Test")
    print(f"📁 Repo: {repo_root}")
    print(f"📄 Data: {jsonl_path}")
    print(f"🛡️  Mode: {'DRY RUN' if dry_run else 'LIVE EDITING'}\n")

    results_counter = Counter()
    processed_issues = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 1. Parse the JSON and initialize the Pydantic Contract
                issue_data = json.loads(line)
                issue = VulnerabilityIssue(**issue_data)
                
                print(f"\n--- [Issue {line_num}] Processing {issue.id} ({issue.issue_type.value}) in {issue.file_path} ---")
                
                # 2. Run the LangGraph Orchestrator
                final_state = run_remediation(
                    issue=issue,
                    repo_root=repo_root,
                    dry_run=dry_run
                )
                
                # 3. Track the final status
                status = final_state.get("status", "unknown")
                results_counter[status] += 1
                
                processed_issues.append({
                    "id": issue.id,
                    "type": issue.issue_type.value,
                    "status": status,
                    "errors": final_state.get("errors", [])
                })
                
                # Print quick result for this issue
                if status == "failed":
                    print(f"❌ Failed: {final_state.get('errors', ['Unknown error'])}")
                elif status == "localized_needs_remedy_agent":
                    print(f"⏸️  Paused: SAST issue deferred for Phase 4.2 LLM Agent.")
                elif status == "dry_run":
                    print(f"✅ Success (Dry Run): Ready to edit!")
                else:
                    print(f"ℹ️  Status: {status}")

            except json.JSONDecodeError:
                print(f"⚠️ Error parsing JSON on line {line_num}. Skipping.")
                results_counter["json_error"] += 1
            except Exception as e: # Catch Pydantic validation errors, etc.
                print(f"⚠️ Error initializing contract on line {line_num}: {e}")
                results_counter["contract_validation_error"] += 1

            if results_counter["status"] <= 10:  # Limit to first 10 issues for testing
                print("\n⚠️ Reached processing limit of 10 issues for this test run.")
                break

    # 4. Print Summary Report
    print("\n=============================================")
    print("📊 BATCH REMEDIATION SUMMARY")
    print("=============================================")
    print(f"Total Issues Processed : {len(processed_issues)}")
    print("-" * 45)
    
    for status, count in results_counter.items():
        print(f"{status.ljust(30)}: {count}")
    print("=============================================\n")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to match your local setup
    INGESTED_JSONL_PATH = ""
    JUICE_SHOP_REPO_PATH = ""
    
    # Run the batch processor!
    run_batch_remediation(
        jsonl_path=INGESTED_JSONL_PATH,
        repo_root=JUICE_SHOP_REPO_PATH,
        dry_run=True # Keep this True until you are ready to actually write to the files
    )