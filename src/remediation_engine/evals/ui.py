"""Launch the local DeepEval dashboard application."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_APP_PATH = Path(__file__).resolve().with_name("dashboard.py")


def main() -> None:
    """Launch the Streamlit DeepEval dashboard."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [sys.executable, "-m", "streamlit", "run", str(_APP_PATH), *sys.argv[1:]]
    returncode = subprocess.call(cmd, env=env)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
