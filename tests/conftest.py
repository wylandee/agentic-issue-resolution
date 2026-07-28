from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """
    Workspace-local tmp path fixture.

    Some environments deny access to pytest's default Windows temp root, so we
    create per-test temp directories inside the writable repository workspace.
    """
    base_dir = Path(__file__).resolve().parents[1] / ".pytest-tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="pytest-", dir=base_dir))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


