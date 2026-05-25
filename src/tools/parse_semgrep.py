"""
parse_semgrep — compatibility shim.

All logic has moved to ``src/tools/semgrep_parser.py``.
This module re-exports the public API so existing callers do not break.
"""

from src.tools.semgrep_parser import (  # noqa: F401
    API_BASE_URL,
    CSV_HEADERS,
    FINDINGS_ENDPOINT_TEMPLATE,
    _extract_findings_page,
    export_to_csv,
    export_to_jsonl,
    fetch_findings,
    main,
    normalize_finding,
    setup_session,
)

__all__ = [
    "API_BASE_URL",
    "CSV_HEADERS",
    "FINDINGS_ENDPOINT_TEMPLATE",
    "export_to_csv",
    "export_to_jsonl",
    "fetch_findings",
    "main",
    "normalize_finding",
    "setup_session",
    "_extract_findings_page",
]
