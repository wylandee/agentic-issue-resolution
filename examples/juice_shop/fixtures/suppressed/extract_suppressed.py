#!/usr/bin/env python3
"""Helper script to extract suppressed ODC issues based on target package names.

This script parses baseline issues in JSONL format and extracts entries matching
a batch of target package names into a suppressed issues JSONL file.
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_suppressed_issues(
    packages: set[str],
    input_path: Path,
    output_path: Path,
) -> int:
    """Extract issues matching package names from input JSONL to output JSONL.

    Args:
        packages: Set of package names to match.
        input_path: Path to the source baseline_issues.jsonl file.
        output_path: Path to write the filtered JSONL entries to.

    Returns:
        The number of matching issue entries extracted.
    """
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as err:
                logger.warning("Skipping invalid JSON line: %s", err)
                continue

            pkg_name = data.get("package_name")
            if pkg_name in packages:
                outfile.write(line_str + "\n")
                extracted_count += 1

    logger.info(
        "Extracted %d matching issues for packages %s to %s",
        extracted_count,
        packages,
        output_path,
    )
    return extracted_count


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the extraction tool.

    Args:
        args: List of command line arguments, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir.parent / "baseline_issues.jsonl"
    default_output = script_dir / "odc_suppressed_issues.jsonl"

    parser = argparse.ArgumentParser(
        description="Extract entries from baseline_issues.jsonl given target package names."
    )
    parser.add_argument(
        "packages",
        nargs="+",
        help="One or more target package names to extract (e.g. @angular/common)",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        help=f"Path to input baseline JSONL file (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=default_output,
        help=f"Path to output JSONL file (default: {default_output})",
    )

    return parser.parse_args(args)


def main() -> None:
    """Entry point for the extract script."""
    parsed = parse_args()
    target_packages = set(parsed.packages)
    extract_suppressed_issues(
        packages=target_packages,
        input_path=parsed.input,
        output_path=parsed.output,
    )


if __name__ == "__main__":
    main()
