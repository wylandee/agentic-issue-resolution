"""Compare persisted evaluation runs from the command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if _SOURCE_ROOT.exists() and str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    """Build the comparison CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="List and compare persisted remediation-engine evaluation runs."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        help="SQLite database path; defaults to EVAL_DB_PATH or data/evals/eval_results.db.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", dest="list_runs", help="List recorded runs.")
    mode.add_argument("--latest", action="store_true", help="Compare the two newest runs.")
    mode.add_argument("--run-a", metavar="REF", help="Baseline run ID or tag.")
    parser.add_argument("--run-b", metavar="REF", help="Candidate run ID or tag.")
    return parser


def _resolve_comparison_runs(
    database: Any,
    run_a_reference: str,
    run_b_reference: str,
) -> tuple[dict[str, object] | None, dict[str, object] | None, str | None]:
    """Resolve two user-provided run references and return an error if needed."""
    run_a = database.resolve_run_reference(run_a_reference)
    if run_a is None:
        return None, None, f"Run or tag {run_a_reference!r} was not found."

    run_b = database.resolve_run_reference(run_b_reference)
    if run_b is None:
        return None, None, f"Run or tag {run_b_reference!r} was not found."

    if run_a["run_id"] == run_b["run_id"]:
        return None, None, "Baseline and candidate must refer to different runs."
    return run_a, run_b, None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation comparison CLI.

    Args:
        argv: Optional argument sequence. Defaults to ``sys.argv[1:]``.

    Returns:
        Zero for a successful operation, otherwise a nonzero CLI error code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.list_runs or args.latest:
        if args.run_b is not None:
            parser.error("--run-b can only be used with --run-a")
    elif args.run_b is None:
        parser.error("--run-a requires --run-b")

    try:
        from remediation_engine.evals.comparison import format_run_comparison, format_run_list
        from remediation_engine.evals.db import EvalDatabase

        database = EvalDatabase(db_path=args.db_path)
        if args.list_runs:
            print(format_run_list(database.get_runs(limit=None)))
            return 0

        if args.latest:
            runs = database.get_runs(limit=None)
            if len(runs) < 2:
                print(
                    "ERROR: At least two evaluation runs are required for --latest.",
                    file=sys.stderr,
                )
                return 1
            run_a, run_b = runs[1], runs[0]
        else:
            run_a, run_b, error = _resolve_comparison_runs(
                database,
                args.run_a,
                args.run_b,
            )
            if error:
                print(f"ERROR: {error}", file=sys.stderr)
                return 1
            assert run_a is not None and run_b is not None

        comparison = database.get_run_comparison(run_a["run_id"], run_b["run_id"])
        if comparison.get("error"):
            print(f"ERROR: {comparison['error']}", file=sys.stderr)
            return 1
        print(format_run_comparison(comparison))
        return 0
    except Exception as exc:
        print(f"ERROR: Could not read evaluation database: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
