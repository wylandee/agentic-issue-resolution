"""Atomic persistence helpers for deterministic remediation reports."""

from __future__ import annotations

import re
from pathlib import Path

from remediation_engine.settings import AppSettings


def resolve_report_dir(settings: AppSettings, default_dir: Path) -> Path:
    """Resolve the configured report directory without reading the environment."""
    return settings.remediation_report_dir or default_dir


def report_filename(run_id: str) -> str:
    """Return a stable, filesystem-safe Markdown filename for a run ID."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-") or "local-run"
    return f"remediation_{safe_id}.md"


def write_report_atomic(path: Path, markdown: str) -> None:
    """Write report text through a sibling temporary file and replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["report_filename", "resolve_report_dir", "write_report_atomic"]
