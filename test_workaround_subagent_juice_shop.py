from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.contracts.schemas import (
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.state import (
    initial_orchestrator_state,
    initial_workaround_subagent_state,
)
from src.orchestrator.teardown_node import run_teardown_node
from src.orchestrator.workaround_subagent import run_workaround_subagent_node


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


TEST_REPO_ROOT = Path(
    os.environ.get(
        "TEST_REPO_ROOT",
        "data/clones/juice-shop",
    )
)


def _build_test_groups() -> list[VulnerabilityGroup]:
    """Create the three workaround stub groups used by the existing execution test."""
    issue_jwt = VulnerabilityIssue(
        id=str(uuid.uuid4()),
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path="package.json",
        package_name="jsonwebtoken",
        cve_id="CVE-2015-9235",
        severity=Severity.HIGH,
    )
    plan_jwt = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND,
        strategy_used="serper",
        fixed_version=None,
        instruction=(
            "Analyze the provided workaround_snippets to determine if a code edit "
            "can safely mitigate this vulnerability."
        ),
        workaround_snippets=[
            'jwt.verify(token, publicKey, { algorithms: ["RS256"] }, callback)'
        ],
    )
    group_jwt = VulnerabilityGroup(
        group_id="sca:package.json:jsonwebtoken:WORKAROUND",
        issue_type=IssueType.SCA,
        vulnerable_component="jsonwebtoken",
        file_path="package.json",
        cve_ids=["CVE-2015-9235"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue_jwt.id,
        issues=[issue_jwt],
        fix_plan=plan_jwt,
    )

    issue_express_jwt = VulnerabilityIssue(
        id=str(uuid.uuid4()),
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path="package.json",
        package_name="express-jwt",
        cve_id="CVE-2020-15084",
        severity=Severity.CRITICAL,
    )
    plan_express_jwt = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND,
        strategy_used="serper",
        fixed_version=None,
        instruction=(
            "Analyze the provided workaround_snippets to determine if a code edit "
            "can safely mitigate this vulnerability."
        ),
        workaround_snippets=[
            'expressJwt({ secret: publicKey, algorithms: ["RS256"] })'
        ],
    )
    group_express_jwt = VulnerabilityGroup(
        group_id="sca:package.json:express-jwt:WORKAROUND",
        issue_type=IssueType.SCA,
        vulnerable_component="express-jwt",
        file_path="package.json",
        cve_ids=["CVE-2020-15084"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue_express_jwt.id,
        issues=[issue_express_jwt],
        fix_plan=plan_express_jwt,
    )

    issue_sanitize = VulnerabilityIssue(
        id=str(uuid.uuid4()),
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path="package.json",
        package_name="sanitize-html",
        cve_id="CVE-2021-23424",
        severity=Severity.HIGH,
    )
    plan_sanitize = FixPlan(
        status=FixPlanStatus.WORKAROUND_FOUND,
        strategy_used="serper",
        fixed_version=None,
        instruction=(
            "Analyze the provided workaround_snippets to determine if a code edit "
            "can safely mitigate this vulnerability."
        ),
        workaround_snippets=[
            'sanitizeHtml(html, { allowedTags: [], allowedAttributes: {} })'
        ],
    )
    group_sanitize = VulnerabilityGroup(
        group_id="sca:package.json:sanitize-html:WORKAROUND",
        issue_type=IssueType.SCA,
        vulnerable_component="sanitize-html",
        file_path="package.json",
        cve_ids=["CVE-2021-23424"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue_sanitize.id,
        issues=[issue_sanitize],
        fix_plan=plan_sanitize,
    )

    return [group_jwt, group_express_jwt, group_sanitize]


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. The subagent LLM will fail.")
        return

    repo_root = TEST_REPO_ROOT.resolve()
    if not repo_root.is_dir():
        logger.error("Repo root does not exist: %s", repo_root)
        return

    target_groups = _build_test_groups()
    logger.info("Prepared %d workaround stub groups.", len(target_groups))
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

    results: list[tuple[VulnerabilityGroup, dict]] = []
    try:
        for group in target_groups:
            logger.info("Starting Workaround Subagent for %s...", group.group_id)
            initial_state = initial_workaround_subagent_state(
                repo_root=str(repo_root),
                workspace_volume=workspace_volume,
                target_group=group,
                constraints_ledger=[],
                previous_feedback=None,
            )
            result = run_workaround_subagent_node(initial_state)
            results.append((group, result))
            action_summary = result.get("action_summary")
            if action_summary:
                logger.info("STATUS  : %s", action_summary.status.value)
                logger.info("SUMMARY : %s", action_summary.summary)
            logger.info("CHANGED FILES : %s", result.get("changed_files", []))
            errors = result.get("errors", [])
            if errors:
                logger.error("ERRORS DETECTED (%d):", len(errors))
                for err in errors:
                    logger.error(" - %s", err)
            else:
                logger.info("ERRORS DETECTED : None")
            logger.info("-" * 40)
    finally:
        teardown_state = {
            "repo_root": str(repo_root),
            "workspace_volume": workspace_volume,
            "changed_files": [
                path
                for _, result in results
                for path in result.get("changed_files", [])
            ],
            "status": "completed",
            "diff": "",
            "errors": (workspace_result.get("errors", []) or [])
            + [
                err
                for _, result in results
                for err in result.get("errors", [])
            ],
        }
        teardown_result = run_teardown_node(teardown_state)
        logger.info("Teardown status: %s", teardown_result.get("status"))


if __name__ == "__main__":
    main()
