import os
import json
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic import TypeAdapter

from src.contracts.schemas import VulnerabilityIssue, SystemContext, VulnerabilityGroup
from src.triage.pipeline import run_triage_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preprocess_triage")

def main():
    load_dotenv()
    
    # 1. Define paths
    input_path = os.path.abspath("./data/odc_issues.jsonl")
    output_dir = os.path.abspath("./data/cache")
    output_path = os.path.join(output_dir, "triaged_groups_latest.json")
    repo_root = os.path.abspath("./data/clones/juice-shop")
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Repo root for reachability: {repo_root}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found at {input_path}")
        return
        
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Ingest issues
    logger.info("Ingesting issues from JSONL...")
    issues = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                issue = VulnerabilityIssue.model_validate_json(line.strip())
                issues.append(issue)
            except Exception as e:
                logger.error(f"Failed to parse line: {line.strip()}. Error: {e}")
                
    logger.info(f"Ingested {len(issues)} issues.")
    
    # 3. Create SystemContext
    context = SystemContext(
        public_facing=True,
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs"
    )
    
    # 4. Run triage pipeline
    logger.info("Running triage pipeline...")
    results = run_triage_pipeline(issues, context, repo_root)
    
    # 5. Filter for valid groups
    valid_groups = []
    for group, result in results:
        if result.is_valid:
            valid_groups.append(group)
            
    logger.info(f"Triage completed: {len(valid_groups)}/{len(results)} groups are valid.")
    
    # 6. Save valid groups to cache
    logger.info("Caching triaged groups to JSON...")
    try:
        adapter = TypeAdapter(List[VulnerabilityGroup])
        json_data = adapter.dump_json(valid_groups, indent=2)
        with open(output_path, "wb") as f:
            f.write(json_data)
        logger.info(f"Successfully cached {len(valid_groups)} groups to {output_path}")
    except Exception as e:
        logger.error(f"Failed to serialize/write cache: {e}")

if __name__ == "__main__":
    main()
