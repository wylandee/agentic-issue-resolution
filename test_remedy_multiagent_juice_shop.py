from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.contracts.schemas import VulnerabilityGroup
from src.orchestrator.graph import run_orchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("remedy_multiagent_test")

TRIAGED_GROUPS_JSON = Path("data/cache/triaged_groups.json")
TEST_REPO_ROOT = Path(
    os.environ.get(
        "TEST_REPO_ROOT",
        "data/clones/juice-shop",
    )
)

# 3 Direct Dependencies & 2 Transitive Dependencies selected from triaged_groups.json
TARGET_GROUP_IDS = (
    # Direct
     "sca:package.json:jsonwebtoken:UPDATE_VERSION",
     "sca:package.json:express-jwt:UPDATE_VERSION",
    # "sca:package.json:sanitize-html:UPDATE_VERSION",
    # "sca:package.json:socket.io:UPDATE_VERSION",
    # Transitive
    #"sca:package.json:@tootallnate/once:UPDATE_VERSION", # Happy path
    #"sca:package.json:base64url:UPDATE_VERSION", # Happy path
    "sca:frontend/package.json:ws:UPDATE_VERSION",
    "sca:frontend/package.json:elliptic:UPDATE_VERSION",
    #"sca:package.json:cookie:UPDATE_VERSION", # Happy path
    #"sca:package.json:lodash.set:UPDATE_VERSION",
    #"sca:package.json:nanoid:UPDATE_VERSION", # Happy path
    #"sca:package.json:http-cache-semantics:UPDATE_VERSION", # Happy path
    "sca:package.json:serialize-javascript:UPDATE_VERSION",

)


def _load_vulnerability_groups() -> list[VulnerabilityGroup]:
    if not TRIAGED_GROUPS_JSON.is_file():
        raise FileNotFoundError(f"Missing triage cache: {TRIAGED_GROUPS_JSON}")

    raw_groups = json.loads(TRIAGED_GROUPS_JSON.read_text(encoding="utf-8"))
    groups = [VulnerabilityGroup.model_validate(group) for group in raw_groups]
    by_id = {group.group_id: group for group in groups}

    selected: list[VulnerabilityGroup] = []
    missing: list[str] = []
    for group_id in TARGET_GROUP_IDS:
        group = by_id.get(group_id)
        if group is None:
            missing.append(group_id)
            continue
        selected.append(group)

    if missing:
        raise ValueError(f"Could not find these target groups in triaged_groups.json: {missing}")

    return selected


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. The supervisor and subagents will fail.")
        return

    repo_root = TEST_REPO_ROOT.resolve()
    if not repo_root.is_dir():
        logger.error("Repo root does not exist: %s", repo_root)
        return

    try:
        valid_groups = _load_vulnerability_groups()
    except Exception as exc:
        logger.exception("Failed to load target groups: %s", exc)
        return

    logger.info("Selected %d target groups from triaged_groups.json for the Remedy Phase:", len(valid_groups))
    for group in valid_groups:
        dep_type = "Direct" if "package.json:" in group.group_id and group.vulnerable_component in (
            "jsonwebtoken", "express-jwt", "sanitize-html"
        ) else "Transitive"
        logger.info(" - %s (%s) [%s]", group.group_id, group.vulnerable_component, dep_type)

    logger.info("=" * 60)
    logger.info("STARTING PHASE 5 REMEDY ENGINE (Multi-Agent Flow)")
    logger.info("Repo Root: %s", repo_root)
    if os.environ.get("LANGSMITH_API_KEY"):
        logger.info("LangSmith Tracing: ENABLED")
    else:
        logger.info("LangSmith Tracing: DISABLED (set LANGSMITH_API_KEY to enable)")
    logger.info("=" * 60)

    try:
        # run_orchestrator automatically triggers LangSmith tracing via runnable config
        result = run_orchestrator(
            repo_root=str(repo_root),
            valid_groups=valid_groups,
            issues=None,           # Skip Triage Node by passing None
            system_context=None,   # Skip Triage Node by passing None
        )
    except Exception as exc:
        logger.exception("Orchestration graph crashed: %s", exc)
        return

    logger.info("=" * 60)
    logger.info("ORCHESTRATOR GRAPH EXECUTION COMPLETE")
    logger.info("=" * 60)
    logger.info("Final Status : %s", result.get("status"))
    logger.info("LangSmith Run ID: %s", result.get("langsmith_run_id", "N/A"))
    logger.info("LangSmith URL   : %s", result.get("langsmith_trace_url", "N/A"))
    logger.info("Changed Files   : %s", result.get("changed_files", []))
    
    errors = result.get("errors", [])
    if errors:
        logger.error("Errors encountered (%d):", len(errors))
        for err in errors:
            logger.error(" - %s", err)
    else:
        logger.info("Errors: None")

    diff = result.get("diff", "").strip()
    if diff:
        logger.info("Unified Git Diff generated:")
        logger.info("\n%s", diff)
    else:
        logger.warning("No Git Diff generated.")


if __name__ == "__main__":
    main()
