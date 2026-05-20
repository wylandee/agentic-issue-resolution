import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _project_root() -> Path:
    """Return the project root based on this file location."""
    return Path(__file__).resolve().parent.parent.parent


def _clones_dir() -> Path:
    """Return the clones directory and ensure it exists."""
    clones = _project_root() / "data" / "clones"
    os.makedirs(clones, exist_ok=True)
    return clones


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from HTTPS or SSH git URLs.

    Examples:
    - https://github.com/org/my-app.git -> my-app
    - git@github.com:org/my-app.git -> my-app
    - ssh://git@github.com/org/my-app.git -> my-app
    """
    repo = repo_url.strip().rstrip("/")

    # Handle SCP-like SSH syntax: git@host:org/repo.git
    if ":" in repo and "//" not in repo and "@" in repo:
        path_part = repo.split(":", 1)[1]
    else:
        parsed = urlparse(repo)
        path_part = parsed.path

    name = Path(path_part).name
    if name.endswith(".git"):
        name = name[:-4]

    if not name:
        raise ValueError(f"Unable to determine repository name from URL: {repo_url}")

    return name


def get_workspace_path(repo_name: str) -> Path:
    """Return the absolute path for a repository workspace under data/clones."""
    return (_clones_dir() / repo_name).resolve()


def _run_git_command(args: list[str], cwd: Path | None = None) -> None:
    """Run a git command and raise a clear error on failure."""
    try:
        subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logging.error("Git command failed: %s", " ".join(args))
        logging.error("stderr: %s", (e.stderr or "").strip())
        raise RuntimeError(f"Git command failed: {' '.join(args)}") from e


def setup_clean_workspace(repo_url: str, branch: str = "main") -> Path:
    """Create or reset a clean workspace for the target repository.

    If the repo does not exist locally, clone it and checkout the branch.
    If it exists, fetch and hard-reset to origin/<branch>, then clean untracked files.
    """
    repo_name = _extract_repo_name(repo_url)
    target_path = get_workspace_path(repo_name)

    if not target_path.exists():
        logging.info("Workspace missing. Cloning %s into %s", repo_url, target_path)
        _run_git_command(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                "--single-branch",
                repo_url,
                str(target_path),
            ]
        )
        logging.info("Clone complete and checked out branch '%s'", branch)
    else:
        logging.info("Workspace exists. Resetting %s to origin/%s", target_path, branch)
        _run_git_command(["git", "fetch", "origin"], cwd=target_path)
        _run_git_command(["git", "reset", "--hard", f"origin/{branch}"], cwd=target_path)
        _run_git_command(["git", "clean", "-fd"], cwd=target_path)
        logging.info("Workspace reset complete")

    return target_path.resolve()


if __name__ == "__main__":
    # Example manual usage:
    # python src/runtime/workspace_manager.py https://github.com/octocat/Hello-World.git main
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python src/runtime/workspace_manager.py <repo_url> [branch]")

    repo_url_arg = sys.argv[1]
    branch_arg = sys.argv[2] if len(sys.argv) > 2 else "main"
    workspace = setup_clean_workspace(repo_url_arg, branch_arg)
    print(workspace)


