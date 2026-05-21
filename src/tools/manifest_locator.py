import json
import logging
import re
import shutil
import subprocess
import os
from pathlib import Path
from typing import Any, Dict, Optional


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def normalize_package_name(raw_name: str) -> str:
    """Normalize noisy package identifiers into a package name.

    Examples:
    - lodash-4.17.21.tgz -> lodash
    - log4j-core-2.17.2.jar -> log4j-core
    """
    name = (raw_name or "").strip()
    if not name:
        return ""

    # Strip ODC formatting like version colons (package:1.2.3) and parenthesis
    name = name.split(":")[0]
    name = re.sub(r"\s*\(.*?\)", "", name)

    # Keep npm scoped packages intact if already clean (e.g. @types/node).
    if name.startswith("@") and "/" in name and not re.search(r"\.(tgz|jar|zip)$", name, re.IGNORECASE):
        return name

    # Remove common archive/binary/js suffixes.
    name = re.sub(r"\.(tgz|jar|zip|js)$", "", name, flags=re.IGNORECASE)

    # Remove trailing version-ish segment: -1.2.3, _1.2.3, .1.2.3, -v1.2.3
    name = re.sub(r"[-_.]v?\d+(?:\.\d+)*(?:[-+][A-Za-z0-9._-]+)?$", "", name)

    return name.strip()


def _read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _find_dependency_line(lines: list[str], package_name: str) -> Optional[int]:
    """Find 1-indexed line number of dependency declaration in package.json text."""
    pattern = re.compile(rf'^\s*"{re.escape(package_name)}"\s*:\s*')
    for idx, line in enumerate(lines, start=1):
        if pattern.search(line):
            return idx
    return None


def _build_snippet(lines: list[str], line_number: int) -> str:
    start = max(1, line_number - 1)
    end = min(len(lines), line_number + 1)
    return "\n".join(lines[start - 1 : end])


def _run_package_lock_generation(repo_path: Path) -> None:
    logging.info("Running npm install --package-lock-only in %s", repo_path)

    npm_executable = shutil.which("npm") or "npm"
    
    # Check if we are on Windows (os.name == 'nt')
    is_windows = os.name == 'nt'

    try:
        subprocess.run(
            [npm_executable, "install", "--package-lock-only"],
            cwd=str(repo_path),
            check=True,
            capture_output=True,
            text=True,
            shell=is_windows
        )
        logging.info("package-lock.json generation command completed")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to generate package-lock.json")
        logging.error("stderr: %s", (e.stderr or "").strip())
    except FileNotFoundError:
        logging.error("Failed to generate package-lock.json")
        logging.error("npm executable not found on PATH")


def handle_node_project(repo_path: Path, package_name: str) -> Dict[str, Any]:
    """Locate dependency context in a Node.js project manifest."""
    manifest_path = repo_path / "package.json"
    if not manifest_path.exists():
        return {"status": "error", "message": "No supported manifest found in repository."}

    package_json = _read_json_file(manifest_path)
    dependencies = package_json.get("dependencies", {}) or {}
    dev_dependencies = package_json.get("devDependencies", {}) or {}

    is_direct = package_name in dependencies or package_name in dev_dependencies

    if is_direct:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
        line_number = _find_dependency_line(lines, package_name)

        if line_number is None:
            return {
                "status": "not_found",
                "manifest_file": "package.json",
                "package_name": package_name,
                "is_direct": True,
                "line_number": None,
                "snippet": None,
                "ai_instruction": f"Dependency '{package_name}' appears direct but exact line was not found. Update its version in package.json.",
            }

        snippet = _build_snippet(lines, line_number)
        return {
            "status": "success",
            "manifest_file": "package.json",
            "package_name": package_name,
            "is_direct": True,
            "line_number": line_number,
            "snippet": snippet,
            "ai_instruction": f"Change the version of {package_name} on line {line_number}.",
        }

    lockfile_path = repo_path / "package-lock.json"
    if not lockfile_path.exists():
        _run_package_lock_generation(repo_path)

    return {
        "status": "success",
        "manifest_file": "package.json",
        "package_name": package_name,
        "is_direct": False,
        "line_number": None,
        "snippet": None,
        "ai_instruction": (
            "This is a transitive dependency. Do not edit existing dependencies. "
            f"Instead, add an 'overrides' block to the root of package.json to force the version of '{package_name}'."
        ),
    }


def locate_dependency(repo_path: Path, raw_dependency_name: str) -> Dict[str, Any]:
    """Entry point for locating where a vulnerable package should be remediated."""
    normalized_name = normalize_package_name(raw_dependency_name)
    if not normalized_name:
        return {"status": "error", "message": "Dependency name is empty after normalization."}

    return handle_node_project(repo_path=repo_path, package_name=normalized_name)
