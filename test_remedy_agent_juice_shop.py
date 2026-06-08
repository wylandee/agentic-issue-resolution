import os
import uuid
import logging
from pathlib import Path
from src.contracts.schemas import VulnerabilityIssue, IssueType, IssueSource, Severity, VulnerabilityGroup, FixPlan, FixPlanStatus
from src.orchestrator.remedy_agent import run_remedy_agent, _build_prompt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_remedy_test():
    print("==================================================")
    print("🧠 STARTING REMEDY AGENT TEST")
    print("==================================================")

    repo_root = os.path.abspath("./data/clones/juice-shop")

    # 1. Create a mock vulnerability that we know exists in Juice Shop
    mock_id = str(uuid.uuid4()) 
    mock_issue = VulnerabilityIssue(
        id=mock_id,
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path="package.json", 
        package_name="express-jwt",
        package_version="0.1.3",
        cve_id="CVE-2020-15084",
        severity=Severity.CRITICAL
    )

    # 2. Mock the FixPlan telling the LLM what to do
    plan = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        strategy_used="osv_api",
        fixed_version="6.0.0",
        instruction='Update "express-jwt" in package.json to version "6.0.0".'
    )

    # 3. Create the Vulnerability Group
    mock_group = VulnerabilityGroup(
        group_id="sca:package.json:express-jwt:UPDATE_VERSION",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        file_path="package.json",
        cve_ids=["CVE-2020-15084"],
        versions=["0.1.3"],
        sources=[IssueSource.ODC],
        representative_issue_id=mock_issue.id,
        issues=[mock_issue],
        fix_plan=plan
    )

    # 4. Construct the OrchestratorState
    state = {
        "repo_root": repo_root,
        "valid_groups": [mock_group],
        "retry_count": 0,
        "max_retries": 3,
        "test_failures": None,
        "scan_failures": None,
    }

    # -------------------------------------------------------------
    # PRINT THE PROMPT
    # -------------------------------------------------------------
    print("\n   [🔍 PROMPT SENT TO LLM]")
    try:
        # Read the file content to build the prompt exactly as the agent sees it
        file_content = Path(repo_root, "package.json").read_text(encoding="utf-8")
        prompt_text = _build_prompt(
            group=mock_group,
            rel_path="package.json",
            repo_root=repo_root,
            file_content=file_content,
            test_failures=None,
            scan_failures=None,
            retry_count=0,
            max_retries=3
        )
        print("   " + "-"*60)
        # We will truncate the file content part if it gets too long, but let's print the structure
        for line in prompt_text.split('\n'):
            print(f"   | {line}")
        print("   " + "-"*60 + "\n")
    except Exception as e:
        print(f"   [Error generating prompt text: {e}]")

    # 5. Run the Agent!
    print(f"⚙️  Prompting LLM to fix 'express-jwt' in package.json -> v6.0.0...")
    result_state = run_remedy_agent(state)

    # 6. Evaluate the Output
    print("\n" + "="*50)
    print("📊 REMEDY AGENT OUTPUT")
    print("="*50)
    
    status = result_state.get("status")
    print(f"Status: {status}\n")

    if status == "edits_generated":
        edits = result_state.get("edit_requests", [])
        for i, edit in enumerate(edits, 1):
            print(f"📝 EDIT {i} (File: {edit.file_path})")
            print(f"Rationale: {edit.rationale}")
            print("-" * 40)
            print("🔴 OLD TEXT (Must match exactly):")
            print(edit.old_text)
            print("-" * 40)
            print("🟢 NEW TEXT:")
            print(edit.new_text)
            print("=" * 50)

            # Print the raw JSON Pydantic output
            print("📦 RAW JSON (EditRequest):")
            print(edit.model_dump_json(indent=2))
            print("=" * 50)
    else:
        print("❌ FAILED TO GENERATE EDITS:")
        for err in result_state.get("errors", []):
            print(f" - {err}")

if __name__ == "__main__":
    os.environ["REMEDY_LLM_MODEL"] = "gpt-4o-mini"
    run_remedy_test()