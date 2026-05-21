import csv
import json
from pathlib import Path

# Import your manifest locator!
from src.tools.manifest_locator import locate_dependency

def test_real_repo():
    # 1. Define paths
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "odc_issues.csv"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    # 2. Safety checks
    if not csv_path.exists():
        print(f"❌ Error: CSV not found at {csv_path}")
        return
    if not repo_path.exists():
        print(f"❌ Error: Codebase not found at {repo_path}")
        return

    print(f"✅ Found CSV and Codebase. Starting test...\n")

    # 3. Read the CSV and process the first 5 findings
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # Keep track of how many we process so we don't spam your terminal
        count = 0 
        
        for row in reader:
            # Grab the dependency name from the CSV column
            raw_dep_name = row.get("Dependency_Name") 
            
            if not raw_dep_name:
                continue

            print(f"--- 🔍 Processing: {raw_dep_name} ---")
            
            # Call your Manifest Locator tool
            result = locate_dependency(repo_path, raw_dep_name)
            
            # Print the output formatted nicely
            print(json.dumps(result, indent=2))
            print("-" * 50 + "\n")
            
            count += 1

if __name__ == "__main__":
    test_real_repo()