from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.contracts.schemas import VulnerabilityGroup
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.state import (
    initial_orchestrator_state,
    initial_update_subagent_state,
)
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.update_subagent import run_update_subagent_node


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRIAGED_GROUPS_JSON = Path("data/cache/triaged_groups.json")
TEST_REPO_ROOT = Path(
    os.environ.get(
        "TEST_REPO_ROOT",
        "data/clones/juice-shop",
    )
)
TARGET_GROUP_IDS = (
    "sca:package.json:express-jwt:UPDATE_VERSION",
    "sca:package.json:@tootallnate/once:UPDATE_VERSION",
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
        logger.error("OPENAI_API_KEY is not set. The subagent LLM will fail.")
        return

    repo_root = TEST_REPO_ROOT.resolve()
    if not repo_root.is_dir():
        logger.error("Repo root does not exist: %s", repo_root)
        return

    try:
        target_groups = _load_vulnerability_groups()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load target groups: %s", exc)
        return

    logger.info("Selected %d target groups from triaged_groups.json.", len(target_groups))
    for group in target_groups:
        logger.info(" - %s (%s)", group.group_id, group.vulnerable_component)

    workspace_state = initial_orchestrator_state(str(repo_root), target_groups)
    logger.info("Starting workspace builder...")
    workspace_result = run_workspace_builder_node(workspace_state)
    workspace_volume = workspace_result.get("workspace_volume")

    if not workspace_volume:
        logger.error("Workspace builder failed: %s", workspace_result.get("errors", []))
        return

    logger.info("Workspace ready: %s", workspace_volume)

    initial_state = initial_update_subagent_state(
        repo_root=str(repo_root),
        workspace_volume=workspace_volume,
        target_groups=target_groups,
        constraints_ledger=[],
        feedback_by_group={},
    )

    result = None
    try:
        logger.info("Starting Update Subagent...")
        logger.info("-" * 40)
        result = run_update_subagent_node(initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Subagent crashed entirely: %s", exc)
    finally:
        action_summary = (result or {}).get("action_summary")
        teardown_state = {
            "repo_root": str(repo_root),
            "workspace_volume": workspace_volume,
            "changed_files": (result or {}).get("changed_files", []),
            "status": action_summary.status.value if action_summary is not None else "completed",
            "diff": "",
            "errors": (workspace_result.get("errors", []) or []) + ((result or {}).get("errors", []) or []),
        }
        teardown_result = run_teardown_node(teardown_state)
        logger.info("Teardown status: %s", teardown_result.get("status"))

    if result is None:
        return

    logger.info("-" * 40)
    logger.info("SUBAGENT EXECUTION COMPLETE")

    action_summary = result.get("action_summary")
    if action_summary:
        logger.info("STATUS  : %s", action_summary.status.value)
        logger.info("SUMMARY : %s", action_summary.summary)
    else:
        logger.warning("No action_summary returned!")

    logger.info("CHANGED FILES : %s", result.get("changed_files", []))

    errors = result.get("errors", [])
    if errors:
        logger.error("ERRORS DETECTED (%d):", len(errors))
        for err in errors:
            logger.error(" - %s", err)
    else:
        logger.info("ERRORS DETECTED : None")


if __name__ == "__main__":
    main()
