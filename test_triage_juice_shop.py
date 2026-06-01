import os
import json
import logging
from src.contracts.schemas import VulnerabilityIssue, SystemContext
from src.triage.pipeline import run_triage_pipeline
from src.triage.agent import _build_triage_prompt  # <-- IMPORT ADDED HERE
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def run_jsonl_fp_test(jsonl_path: str):
    print("==================================================")
    print("🕵️  STARTING REAL-WORLD LLM FALSE POSITIVE TEST")
    print("==================================================")

    if not os.path.exists(jsonl_path):
        print(f"❌ Error: Could not find JSONL file at {jsonl_path}")
        return

    # 1. Iterate through the JSONL and pick out our 2 targets
    test_issues = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            
            data = json.loads(line)
            issue = VulnerabilityIssue(**data)
            
            # Target 1: The Apache Ivy Hallucination
            if issue.package_name == "blaze.jar":
                test_issues.append(issue)
            
            # Target 2: The Typosquatting Mix-up
            elif issue.package_name == "grunt-cli":
                test_issues.append(issue)

    print(f"📥 Extracted {len(test_issues)} specific raw alerts from the JSONL.\n")

    # 2. Define Context (Explicitly stating it is a Node.js app!)
    # Note: Double check if your schema expects `public_facing` or `is_public_facing`
    context = SystemContext(
        is_public_facing=True,
        deployment_os="linux",          
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs",
        description="This is a Node.js / JavaScript web application." 
    )

    # 3. Run the pipeline
    print("⚙️  Context: Node.js, Linux, Production")
    print("⚙️  Running Triage Pipeline...\n")
    triage_results = run_triage_pipeline(test_issues, context)
    
    print("="*50)
    print("📊 LLM FALSE POSITIVE RESULTS")
    print("="*50)
    
    for group, result in triage_results:
        status_icon = "✅ VALID" if result.is_valid else "❌ FALSE POSITIVE"
        
        print(f"\n📦 Group: {group.vulnerable_component} (in {group.file_path})")
        print(f"   ↳ Method      -> 🤖 {result.triage_method.upper()}")
        print(f"   ↳ Status      -> {status_icon}")
        
        if not result.is_valid:
            print(f"   ↳ FP Reason   -> {result.false_positive_reason}")
        else:
            print(f"   ↳ Priority    -> {result.revised_priority.name}")
            print(f"   ↳ Reasoning   -> {result.priority_reasoning}")

        print(f"COT:  {result.chain_of_thought}")

        # -------------------------------------------------------------
        # PRINT THE PROMPT HERE
        # -------------------------------------------------------------
        print("\n   [🔍 PROMPT SENT TO LLM]")
        prompt_text = _build_triage_prompt(group, context)
        print("   " + "-"*60)
        for line in prompt_text.split('\n'):
            print(f"   | {line}")
        print("   " + "-"*60 + "\n")

if __name__ == "__main__":
    os.environ["TRIAGE_LLM_ENABLED"] = "true"
    os.environ["TRIAGE_LLM_MODEL"] = "gpt-4o-mini"
    
    INGESTED_JSONL_PATH = "./data/odc_issues.jsonl"
    run_jsonl_fp_test(jsonl_path=INGESTED_JSONL_PATH)