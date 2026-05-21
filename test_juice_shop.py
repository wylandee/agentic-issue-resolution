import json
from pathlib import Path

# Import our Pydantic schema and the upgraded locator
from src.contracts import VulnerabilityIssue
from src.tools.manifest_locator import locate_from_issue

def test_v2_pipeline():
    project_root = Path(__file__).resolve().parent
    jsonl_path = project_root / "data" / "odc_issues.jsonl"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not jsonl_path.exists() or not repo_path.exists():
        print("❌ Error: Missing jsonl file or juice-shop clone.")
        return

    print("✅ Found JSONL and Codebase. Starting V2 Test...\n")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            # 1. Magically convert the JSON string back into a strict Pydantic object!
            issue = VulnerabilityIssue.model_validate_json(line)
            
            print(f"--- 🔍 Processing: {issue.package_name} (Vuln: {issue.rule_id or issue.cve_id}) ---")
            
            # 2. Pass the typed issue to our new locator
            # enrich_osv=True means it will actually call the Google OSV API!
            localized_issue = locate_from_issue(issue, repo_path)
            
            # 3. Print the resulting LocalizedIssue Pydantic object as JSON
            print(localized_issue.model_dump_json(indent=2))
            print("-" * 60 + "\n")
            
            count += 1

if __name__ == "__main__":
    test_v2_pipeline()