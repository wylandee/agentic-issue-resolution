import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import TypeAdapter

from src.contracts.schemas import VulnerabilityIssue, SystemContext, VulnerabilityGroup
from src.triage.pipeline import run_triage_pipeline
from src.tools.odc_parser import parse_vulnerabilities

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preprocess_triage")


def ingest_issues(input_path: Path, input_format: str = "auto") -> List[VulnerabilityIssue]:
    """Load issues from a Dependency-Check JSON report or legacy JSONL output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    detected_format = input_format
    if detected_format == "auto":
        detected_format = "jsonl" if input_path.suffix.lower() in {".jsonl", ".ndjson"} else "odc-json"

    if detected_format == "odc-json":
        with input_path.open("r", encoding="utf-8") as input_file:
            report = json.load(input_file)
        return parse_vulnerabilities(report)

    if detected_format == "jsonl":
        issues: List[VulnerabilityIssue] = []
        with input_path.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    issues.append(VulnerabilityIssue.model_validate_json(line.strip()))
                except Exception as exc:  # noqa: BLE001 - keep processing records
                    logger.error(
                        "Failed to parse JSONL record %d in %s: %s",
                        line_number,
                        input_path,
                        exc,
                    )
        return issues

    raise ValueError(
        f"Unsupported input format '{input_format}'. "
        "Expected 'auto', 'odc-json', or 'jsonl'."
    )


def main(argv: Optional[List[str]] = None):
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Ingest a Dependency-Check JSON report and run triage."
    )
    parser.add_argument(
        "--input",
        default="data/dependency-check-report.json",
        help="Dependency-Check JSON report path (default: data/dependency-check-report.json).",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "odc-json", "jsonl"),
        default="auto",
        help="Input format; auto treats .jsonl/.ndjson as legacy JSONL.",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent

    def resolve_project_path(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        return path if path.is_absolute() else project_root / path
    
    # 1. Define paths
    input_path = resolve_project_path(args.input)
    output_dir = project_root / "data" / "cache"
    output_path = output_dir / "triaged_groups_latest.json"
    repo_root = project_root / "data" / "clones" / "juice-shop"
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Repo root for reachability: {repo_root}")
    
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        return
        
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Ingest issues
    logger.info("Ingesting issues from %s...", input_path)
    try:
        issues = ingest_issues(input_path, args.input_format)
    except Exception as exc:  # noqa: BLE001 - report malformed input cleanly
        logger.error("Failed to ingest %s: %s", input_path, exc)
        return
                
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
    results = run_triage_pipeline(issues, context, str(repo_root))
    
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
        with output_path.open("wb") as output_file:
            output_file.write(json_data)
        logger.info(f"Successfully cached {len(valid_groups)} groups to {output_path}")
    except Exception as e:
        logger.error(f"Failed to serialize/write cache: {e}")

if __name__ == "__main__":
    main()
