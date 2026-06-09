import os
import uuid
import logging
from src.contracts.schemas import EditRequest, VulnerabilityGroup, VulnerabilityIssue, IssueType, Severity, IssueSource
from src.orchestrator.editor_node import run_editor_node
from src.orchestrator.scanner_node import run_scanner_node
from src.orchestrator.tester_node import run_tester_node
from src.orchestrator.teardown_node import run_teardown_node

from dotenv import load_dotenv

load_dotenv()
# Set logging to INFO
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def test_stateless_execution_pipeline():
    print("==================================================")
    print("🛠️  STARTING DOCKER VOLUME EXECUTION TEST")
    print("==================================================")

    repo_root = os.path.abspath("./data/clones/juice-shop")

    print("\n[MOCK] Simulating LLM EditRequest...")
    mock_edit = EditRequest(
        repo_root=repo_root,
        file_path="package.json",
        old_text='    "express-jwt": "0.1.3",',
        new_text='    "express-jwt": "6.0.0",',
        dry_run=False,
        rationale="Update express-jwt to safe version 6.0.0"
    )

    mock_issue = VulnerabilityIssue(
        id=str(uuid.uuid4()),
        issue_type=IssueType.SCA,
        source="odc",
        file_path="package.json",
        package_name="express-jwt",
        cve_id="CVE-2020-15084",
        severity=Severity.CRITICAL
    )
    mock_group = VulnerabilityGroup(
        group_id="sca:package.json:express-jwt:UPDATE_VERSION",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        file_path="package.json",
        cve_ids=["CVE-2020-15084"],
        sources=["odc"],
        representative_issue_id=mock_issue.id,
        issues=[mock_issue]
    )

    # 3. INITIALIZE STATE (Updated for Docker Volumes)
    state = {
        "repo_root": repo_root,
        "valid_groups": [mock_group],
        "edit_requests": [mock_edit],
        "retry_count": 0,
        "max_retries": 3,
        "test_failures": None,
        "scan_failures": None,
        "status": "pending",
        "workspace_volume": None,  # <--- NEW FIELD
        "changed_files": [],
        "change_diff": None,
        "final_status": None
    }

    # ---------------------------------------------------------
    # NODE 1: THE EDITOR
    # ---------------------------------------------------------
    print("\n▶️  RUNNING EDITOR NODE...")
    editor_result = run_editor_node(state)
    state.update(editor_result)  # Merges {"workspace_volume": "...", "status": "edited"} into state
    
    print(f"   ↳ Status: {state.get('status')}")
    if state.get("test_failures"):
        print(f"   ↳ Error : {state['test_failures']}")
        run_teardown_node(state)
        return 

    print(f"   ↳ Docker Volume Created: {state.get('workspace_volume')}")

    # ---------------------------------------------------------
    # NODE 2: THE SCANNER
    # ---------------------------------------------------------
    print("\n▶️  RUNNING SCANNER NODE...")
    scanner_result = run_scanner_node(state)
    state.update(scanner_result)
    
    print(f"   ↳ Status: {state.get('status')}")
    if state.get("scan_failures"):
        print(f"   ↳ Error : {state['scan_failures']}")
    else:
        print("   ↳ SUCCESS: ODC Scan passed! CVE is gone.")

    # ---------------------------------------------------------
    # NODE 3: THE TESTER
    # ---------------------------------------------------------
    print("\n▶️  RUNNING TESTER NODE...")
    tester_result = run_tester_node(state)
    state.update(tester_result)
    
    print(f"   ↳ Status: {state.get('status')}")
    if state.get("test_failures"):
        print(f"   ↳ Error : {state['test_failures']}")
    else:
        print("   ↳ SUCCESS: npm test passed!")

    # ---------------------------------------------------------
    # NODE 4: THE TEARDOWN / DIFF REPORTER
    # ---------------------------------------------------------
    print("\n▶️  RUNNING TEARDOWN NODE...")
    teardown_result = run_teardown_node(state)
    state.update(teardown_result)
    
    print(f"   ↳ Final Status : {state.get('final_status')}")
    
    diff = state.get("change_diff")
    if diff:
        print("\n   [🔍 GENERATED GIT DIFF]")
        print("   " + "-"*60)
        lines = diff.split("\n")
        for line in lines[:30]:
            print(f"   | {line}")
        if len(lines) > 30:
            print("   | ... (diff truncated for terminal) ...")
        print("   " + "-"*60)

    print("\n==================================================")
    print("🏁 EXECUTION PIPELINE TEST COMPLETE")
    print("==================================================")

if __name__ == "__main__":
    test_stateless_execution_pipeline()