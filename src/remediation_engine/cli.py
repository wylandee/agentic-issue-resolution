"""Command-line interface for the remediation engine."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from .api import RemediationRequest, run_remediation, triage_issues
from .contracts.schemas import SystemContext, VulnerabilityIssue
from .settings import AppSettings
from .tools.odc_parser import parse_vulnerabilities
from .tools.semgrep_parser import load_findings_from_json, normalize_finding

log = logging.getLogger(__name__)


def _load_issues(path: Path, input_format: str) -> list[VulnerabilityIssue]:
    """Load canonical findings or a scanner JSON report.

    Canonical findings are normally newline-delimited JSON objects.  For
    compatibility with older exports, a file containing one JSON array of
    canonical objects is accepted as well; both forms produce the same typed
    issue list.
    """
    if input_format == "auto":
        input_format = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "odc-json"
    if input_format == "jsonl":
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return []
        # A previous version wrote ``.jsonl`` files as pretty-printed arrays.
        # Detect that shape before attempting line-by-line parsing so those
        # fixtures remain ingestible while all output is canonical JSONL.
        if text.lstrip().startswith("["):
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("Canonical JSONL array payload must be a list.")
            return [VulnerabilityIssue.model_validate(item) for item in payload]
        return [
            VulnerabilityIssue.model_validate_json(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if input_format == "odc-json":
        return parse_vulnerabilities(json.loads(path.read_text(encoding="utf-8")))
    if input_format == "semgrep-json":
        return [
            issue
            for raw in load_findings_from_json(path)
            if (issue := normalize_finding(raw)) is not None
        ]
    raise ValueError(f"Unsupported input format: {input_format}")


def _write_json(path: Path | None, value: object) -> None:
    """Write indented JSON to a file or stdout."""
    payload = json.dumps(value, indent=2, default=str)
    if path is None:
        print(payload)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")


def _write_jsonl(path: Path | None, values: list[object]) -> None:
    """Write one compact JSON object per line to a file or stdout."""
    payload = "".join(
        json.dumps(value, separators=(",", ":"), default=str) + "\n" for value in values
    )
    if path is None:
        print(payload, end="")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser without reading environment state."""
    parser = argparse.ArgumentParser(prog="remedy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="normalize a scanner report")
    ingest.add_argument("input", type=Path)
    ingest.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    ingest.add_argument("--output", type=Path)
    triage = sub.add_parser("triage", help="load canonical findings for triage")
    triage.add_argument("input", type=Path)
    triage.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    triage.add_argument("--repo", type=Path)
    triage.add_argument("--output", type=Path)
    run = sub.add_parser("run", help="run remediation and emit a patch result")
    run.add_argument("input", type=Path)
    run.add_argument("--repo", required=True, type=Path)
    run.add_argument(
        "--format", choices=("auto", "odc-json", "semgrep-json", "jsonl"), default="auto"
    )
    run.add_argument("--output", type=Path)
    run.add_argument("--patch-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute a CLI command and return a process exit code."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        issues = _load_issues(args.input, args.format)
        if args.command == "ingest":
            _write_jsonl(args.output, [issue.model_dump(mode="json") for issue in issues])
            return 0
        if args.command == "triage":
            groups = triage_issues(issues, repo_root=args.repo)
            _write_json(args.output, [group.model_dump(mode="json") for group in groups])
            return 0
        settings = AppSettings.from_env()
        request = RemediationRequest(
            repo_root=args.repo,
            issues=issues,
            valid_groups=[],
            system_context=SystemContext(
                public_facing=True,
                deployment_os="linux",
                deployment_architecture="containerized",
                environment="production",
                primary_language="javascript/nodejs",
            ),
        )
        result = run_remediation(request, settings=settings)
        if args.patch_out:
            args.patch_out.parent.mkdir(parents=True, exist_ok=True)
            args.patch_out.write_text(result.diff, encoding="utf-8")
        _write_json(args.output, result.model_dump(exclude={"raw_state"}))
        return 0 if result.status == "completed" and not result.errors else 1
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2
