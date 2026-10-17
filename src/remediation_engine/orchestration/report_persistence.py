"""Atomic persistence helpers for deterministic remediation reports."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from remediation_engine.settings import AppSettings


def resolve_report_dir(settings: AppSettings, default_dir: Path) -> Path:
    """Resolve the configured report directory without reading the environment."""
    return settings.remediation_report_dir or default_dir


def report_filename(run_id: str) -> str:
    """Return a stable, filesystem-safe Markdown filename for a run ID."""
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_id).strip(".-") or "local-run"
    return f"remediation_{safe_id}.md"


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text through an independently owned sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(text)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_report_atomic(path: Path, markdown: str) -> None:
    """Write report text through the shared atomic text writer."""
    write_text_atomic(path, markdown)


__all__ = ["report_filename", "resolve_report_dir", "write_report_atomic", "write_text_atomic"]
