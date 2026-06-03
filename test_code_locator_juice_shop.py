import os
from pathlib import Path
from dotenv import load_dotenv

# Import our Contracts and Tools
from src.contracts import VulnerabilityIssue
from src.contracts.schemas import IssueType
from src.tools.code_locator import locate_sast

def test_sast_locator():
    load_dotenv()
    
    project_root = Path(__file__).resolve().parent
    jsonl_path = project_root / "data" / "semgrep_issues.jsonl"
    repo_path = project_root / "data" / "clones" / "juice-shop"

    if not jsonl_path.exists() or not repo_path.exists():
        print("Error: Missing semgrep_issues.jsonl file or juice-shop clone.")
        return

    print("Found JSONL and Codebase. Starting SAST Code Locator Test...\n")

    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            # 1. Parse the Semgrep finding into our strict Pydantic Contract
            issue = VulnerabilityIssue.model_validate_json(line)
            
            # We only want to test SAST (Code) issues, not dependencies
            if issue.issue_type != IssueType.SAST:
                continue
                
            print(f"---Processing: {issue.rule_id} in {issue.file_path} ---")
            
            # 2. Run the SAST Locator (Tree-Sitter AST Parsing)
            localized_issue = locate_sast(issue, str(repo_path))
            
            # 3. Print the highlights to the terminal!
            print(f"Target Line: {localized_issue.issue.line_range.start if localized_issue.issue.line_range else 'Unknown'}")
            print(f"Enclosing Function: {localized_issue.enclosing_symbol} ({localized_issue.enclosing_node_type.value})")
            print(f"Sink Expression: {localized_issue.sink_expression}")
            print(f"Imports Found: {len(localized_issue.imports)}")
            print(f"Data Flow Hints: {localized_issue.data_flow_hints}")
            print(f"Confidence Score: {localized_issue.localization_confidence}")
            
            print("\nSnippet Preview (First 5 lines):")
            if localized_issue.snippet:
                preview = "\n".join(localized_issue.snippet.splitlines()[:5])
                print(preview + "\n[...] ")
            
            print("-" * 60 + "\n")
            
            count += 1

if __name__ == "__main__":
    test_sast_locator()