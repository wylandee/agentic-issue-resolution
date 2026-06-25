from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv

from src.contracts.schemas import (
    AgentActionStatus,
    AgentActionSummary,
    FixPlan,
    FixPlanStatus,
    IssueSource,
    IssueType,
    RoutingStrategy,
    Severity,
    VulnerabilityGroup,
    VulnerabilityIssue,
)
from src.orchestrator.editor_node import run_workspace_builder_node
from src.orchestrator.qa_critic import run_qa_critic_node
from src.orchestrator.state import initial_orchestrator_state
from src.orchestrator.teardown_node import run_teardown_node

# Configure logging to write to both a file and console
log_file = Path("qa_critic_execution_loop.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TEST_REPO_ROOT = Path(
    os.environ.get(
        "TEST_REPO_ROOT",
        "data/clones/juice-shop",
    )
)


def _build_test_groups() -> list[VulnerabilityGroup]:
    """Create test groups for verifying the QA critic node against Juice Shop."""
    # Group 1: jsonwebtoken (SCA workaround)
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

    # Group 2: express-jwt (SCA workaround)
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

    # Group 3: sanitize-html (SCA workaround)
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

    # Group 4: ws (SCA version bump)
    issue_ws = VulnerabilityIssue(
        id=str(uuid.uuid4()),
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path="frontend/package.json",
        package_name="ws",
        cve_id="CVE-2021-32831",
        severity=Severity.HIGH,
    )
    plan_ws = FixPlan(
        status=FixPlanStatus.VERSION_FOUND,
        strategy_used="osv_api",
        fixed_version="7.4.6",
        instruction="Upgrade ws to 7.4.6.",
    )
    group_ws = VulnerabilityGroup(
        group_id="sca:frontend/package.json:ws:UPDATE_VERSION",
        issue_type=IssueType.SCA,
        vulnerable_component="ws",
        file_path="frontend/package.json",
        cve_ids=["CVE-2021-32831"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue_ws.id,
        issues=[issue_ws],
        fix_plan=plan_ws,
    )

    return [group_jwt, group_express_jwt, group_sanitize, group_ws]


def main() -> None:
    load_dotenv()

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. The QA Critic LLM will fail.")
        return

    repo_root = TEST_REPO_ROOT.resolve()
    if not repo_root.is_dir():
        logger.error("Repo root does not exist: %s", repo_root)
        return

    target_groups = []
    logger.info("Prepared %d test groups for QA evaluation.", len(target_groups))

    workspace_state = initial_orchestrator_state(str(repo_root), target_groups)
    logger.info("Starting workspace builder...")
    workspace_result = run_workspace_builder_node(workspace_state)
    workspace_volume = workspace_result.get("workspace_volume")

    if not workspace_volume:
        logger.error("Workspace builder failed: %s", workspace_result.get("errors", []))
        return

    logger.info("Workspace ready: %s", workspace_volume)

    # Construct the state for run_qa_critic_node
    qa_state = {
        **workspace_state,
        "workspace_volume": workspace_volume,
        "group_strategies": {},
        "action_summaries": [],
        "changed_files": [],
        "force_qa": True,
    }

    qa_result = None
    try:
        logger.info("Starting QA Critic Evaluation node...")
        logger.info("-" * 50)
        qa_result = run_qa_critic_node(qa_state)
    except Exception as exc:
        logger.exception("QA Critic crashed: %s", exc)
    finally:
        changed_files = (qa_result or {}).get("changed_files", [])
        errors = (qa_result or {}).get("errors", [])

        teardown_state = {
            "repo_root": str(repo_root),
            "workspace_volume": workspace_volume,
            "changed_files": changed_files,
            "status": "completed",
            "diff": "",
            "errors": (workspace_result.get("errors", []) or []) + errors,
        }
        teardown_result = run_teardown_node(teardown_state)
        logger.info("Teardown status: %s", teardown_result.get("status"))

    if qa_result is None:
        return

    logger.info("-" * 50)
    logger.info("QA CRITIC EXECUTION COMPLETE")
    logger.info("Status      : %s", qa_result.get("status"))
    logger.info("Eval Status : %s", qa_result.get("eval_status"))
    logger.info("Changed Files: %s", qa_result.get("changed_files", []))

    evals = qa_result.get("qa_evaluations", {})
    logger.info("\nEvaluations:")
    for group_id, evaluation in evals.items():
        logger.info("Group: %s", group_id)
        logger.info("  Passed: %s", evaluation.passed)
        if not evaluation.passed:
            logger.info("  Failure Category: %s", evaluation.failure_category)
            logger.info("  Retry Feedback  : %s", evaluation.retry_feedback)

    errors = qa_result.get("errors", [])
    if errors:
        logger.error("\nErrors detected:")
        for err in errors:
            logger.error(" - %s", err)


if __name__ == "__main__":
    main()
