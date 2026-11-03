"""Deterministic Supervisor policy, retry planning, routing, and task spawning."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from packaging.version import InvalidVersion, Version

from remediation_engine.contracts.decision_codes import DecisionCode
from remediation_engine.contracts.schemas import (
    AgentActionSummary,
    AuditRecord,
    FailureCategory,
    FixPlanStatus,
    NoFixMitigationStage,
    PortfolioPlan,
    QAEvaluation,
    QAPolicy,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    StateConsistencyEvent,
    SupervisorDecision,
    SupervisorRetryPlan,
    TaskSpawnRequest,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.contracts.version_policy import (
    MavenRegistryCandidate,
    RegistryCandidate,
    compare_maven_versions,
    is_stable_maven_version,
    select_maven_version,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.orchestration.state import (
    _TERMINAL_STATUSES,
    _WORKABLE_STATUSES,
    QA_DISPATCH_LIMIT,
    UPDATE_DISPATCH_LIMIT,
    _commit_task_transition,
    _is_exhausted_update_pivot_candidate,
    _qa_ready_task_ids,
    _task_sort_key,
    build_no_fix_retry_instruction,
    group_parent_context,
    instruction_digest,
    is_maven_group,
    select_package_fix_plan,
)
from remediation_engine.runtime.telemetry import invoke_with_trajectory
from remediation_engine.tools.registry import (
    fetch_maven_registry_candidates,
    fetch_pypi_registry_candidates,
    fetch_registry_candidates,
    plan_npm_parent_version,
    plan_python_parent_version,
)

MAX_RETRIES: int = 3
"""Maximum number of QA-fail-retry cycles before a task is unfixable."""

MAX_QA_EVIDENCE_RERUNS: int = 1
"""Maximum same-attempt QA reruns after inconclusive evidence or a bad contract."""


def _normalise_security_floor(
    value: Any,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool = False,
    ecosystem: str | None = None,
) -> tuple[str | None, Any]:
    """Normalize a stable security floor with ecosystem-specific semantics."""
    if value is None:
        return None, None
    raw = str(value).strip()
    normalized_ecosystem = (
        str(
            ecosystem
            or (
                "maven"
                if project_language == ProjectLanguage.JAVA and maven_mode
                else "pypi"
                if project_language == ProjectLanguage.PYTHON
                else "npm"
            )
        )
        .strip()
        .casefold()
    )
    if normalized_ecosystem == "maven":
        if not is_stable_maven_version(raw):
            return None, None
        return raw, raw
    if normalized_ecosystem == "pypi":
        try:
            version = Version(raw)
        except InvalidVersion:
            return None, None
        if version.is_prerelease or version.is_devrelease:
            return None, None
        return str(version), version
    normalized = raw.lstrip("vV")
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", normalized)
    if match is None:
        return None, None
    return normalized, tuple(int(part) for part in match.groups())


def _group_uses_pypi(group: VulnerabilityGroup) -> bool:
    """Identify Python package findings from their issue ecosystem or PURL."""
    for issue in group.issues or []:
        if str(getattr(issue, "ecosystem", "") or "").strip().casefold() in {"pypi", "python"}:
            return True
        purl = str(getattr(issue, "purl", "") or "").strip().casefold()
        if purl.startswith("pkg:pypi/"):
            return True
    return False


def _task_uses_pypi(
    task: RemediationTask | None,
    group: VulnerabilityGroup | None,
) -> bool:
    """Resolve the package ecosystem from finding evidence or committed stage."""
    if group is not None and _group_uses_pypi(group):
        return True
    return task is not None and task.strategy_stage in {
        SCARemediationStage.PYPI_SAME_MAJOR,
        SCARemediationStage.PYPI_LATEST,
    }


def _latest_sca_stage(
    task: RemediationTask | None,
    group: VulnerabilityGroup | None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> SCARemediationStage:
    """Return the latest bounded update stage for one package ecosystem."""
    if is_maven_group(group, project_language):
        return SCARemediationStage.MAVEN_LATEST
    if _task_uses_pypi(task, group):
        return SCARemediationStage.PYPI_LATEST
    return SCARemediationStage.NPM_LATEST


def _canonical_security_floor(
    group: VulnerabilityGroup | None,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> tuple[str | None, str | None]:
    """Return the validated group security floor and any conflict reason.

    The Supervisor-selected package candidate is the canonical floor. Member
    finding versions are corroborating evidence: lower member floors are
    subsumed, while a higher member floor or malformed value is contradictory.

    Args:
        group: Vulnerability group whose fix metadata is being authorized.

    Returns:
        ``(floor, error)``.  Exactly one value is populated: ``floor`` for a
        valid floor, or ``error`` when the evidence is unavailable or
        contradictory.
    """
    if group is None:
        return None, "vulnerability group metadata is unavailable"

    fix_plan = select_package_fix_plan(group).plan
    if fix_plan is not None and fix_plan.status != FixPlanStatus.VERSION_FOUND:
        return None, "the fix plan does not provide an authoritative fixed version"

    maven_mode = is_maven_group(group, project_language)
    ecosystem = (
        "maven"
        if maven_mode
        else "pypi"
        if project_language == ProjectLanguage.PYTHON or _group_uses_pypi(group)
        else "npm"
    )
    plan_floor, plan_key = _normalise_security_floor(
        fix_plan.fixed_version if fix_plan is not None else None,
        project_language=project_language,
        maven_mode=maven_mode,
        ecosystem=ecosystem,
    )
    if fix_plan is not None and fix_plan.fixed_version is not None and plan_floor is None:
        if ecosystem == "maven":
            return None, "the fix-plan fixed version is not a stable Maven version"
        if ecosystem == "pypi":
            return None, "the fix-plan fixed version is not a complete stable PEP 440 version"
        return None, "the fix-plan fixed version is not a complete stable semver"

    member_values: list[tuple[str, Any]] = []
    for issue in group.issues or []:
        value, key = _normalise_security_floor(
            getattr(issue, "fixed_version", None),
            project_language=project_language,
            maven_mode=maven_mode,
            ecosystem=ecosystem,
        )
        raw = getattr(issue, "fixed_version", None)
        if raw is None:
            continue
        if value is None or key is None:
            if ecosystem == "maven":
                return None, "a vulnerability fixed version is not a stable Maven version"
            if ecosystem == "pypi":
                return (
                    None,
                    "a vulnerability fixed version is not a complete stable PEP 440 version",
                )
            return None, "a vulnerability fixed version is not a complete stable semver"
        member_values.append((value, key))

    def compare_keys(left: Any, right: Any) -> int:
        if maven_mode:
            return compare_maven_versions(left, right)
        return (left > right) - (left < right)

    if plan_floor is not None and plan_key is not None:
        conflicting = [value for value, key in member_values if compare_keys(key, plan_key) > 0]
        if conflicting:
            return (
                None,
                "member vulnerability floor exceeds the canonical fix-plan floor: "
                + ", ".join(sorted(set(conflicting))),
            )
        return plan_floor, None

    if member_values:
        highest_value, highest_key = member_values[0]
        for value, key in member_values[1:]:
            if compare_keys(key, highest_key) > 0:
                highest_value, highest_key = value, key
        return highest_value, None
    return None, "no authoritative security floor is available"


def _has_existing_workaround_child(
    task: RemediationTask,
    task_queue: dict[str, RemediationTask],
) -> bool:
    """Return whether a task already owns a code-workaround child."""
    return any(
        candidate.parent_task_id == task.task_id
        and candidate.strategy == RoutingStrategy.CODE_WORKAROUND
        for candidate in task_queue.values()
    )


def _next_sca_stage(
    stage: SCARemediationStage,
    transitive: bool = False,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool = False,
    ecosystem: str | None = None,
) -> SCARemediationStage:
    """Advance one ordered SCA stage without changing the Node ladder."""
    resolved_ecosystem = (
        str(
            ecosystem
            or (
                "maven"
                if project_language == ProjectLanguage.JAVA and maven_mode
                else "pypi"
                if project_language == ProjectLanguage.PYTHON
                else "npm"
            )
        )
        .strip()
        .casefold()
    )
    if resolved_ecosystem == "maven":
        if stage == SCARemediationStage.OSV_MINIMUM:
            return SCARemediationStage.MAVEN_LATEST
        return SCARemediationStage.CODE_WORKAROUND
    if resolved_ecosystem == "pypi":
        if stage == SCARemediationStage.OSV_MINIMUM:
            return SCARemediationStage.PYPI_SAME_MAJOR
        if stage == SCARemediationStage.PYPI_SAME_MAJOR:
            return SCARemediationStage.PYPI_LATEST
        return SCARemediationStage.CODE_WORKAROUND
    if stage == SCARemediationStage.OSV_MINIMUM:
        return SCARemediationStage.NPM_SAME_MAJOR
    if stage == SCARemediationStage.NPM_SAME_MAJOR:
        return SCARemediationStage.NPM_LATEST
    if stage == SCARemediationStage.NPM_LATEST and transitive:
        return SCARemediationStage.PACKAGE_OVERRIDE
    if stage == SCARemediationStage.PACKAGE_OVERRIDE:
        return SCARemediationStage.CODE_WORKAROUND
    return SCARemediationStage.CODE_WORKAROUND


def _selection_for_stage(stage: SCARemediationStage) -> str | None:
    """Return the registry selection mode for a strategy stage."""
    if stage in {SCARemediationStage.NPM_SAME_MAJOR, SCARemediationStage.PYPI_SAME_MAJOR}:
        return "same_major"
    if stage in {SCARemediationStage.NPM_LATEST, SCARemediationStage.PYPI_LATEST}:
        return "latest"
    if stage == SCARemediationStage.MAVEN_LATEST:
        return "maven_latest"
    return None


def _worker_node_for_strategy(strategy: RoutingStrategy) -> str:
    """Return the worker node that handles a routing strategy."""
    if strategy == RoutingStrategy.VERSION_BUMP:
        return "update_subagent"
    return "workaround_subagent"


def _parent_status_for_strategy_pivot(
    parent_task: RemediationTask,
    new_strategy: RoutingStrategy,
    qa_evaluations: dict[str, QAEvaluation],
) -> TaskStatus:
    """Choose the terminal parent status when a pivot creates a child task."""
    evaluation = qa_evaluations.get(parent_task.task_id)
    if (
        parent_task.strategy == RoutingStrategy.VERSION_BUMP
        and new_strategy == RoutingStrategy.CODE_WORKAROUND
        and evaluation is not None
    ):
        gates = evaluation.deterministic_gates
        if (
            parent_task.qa_policy == QAPolicy.VERSION_BUMP
            and gates is not None
            and gates.target_scanner_cleared is True
            and gates.tests_passed is False
        ):
            return TaskStatus.QA_PASSED
        if evaluation.failure_category == FailureCategory.BREAKING_CHANGE:
            return TaskStatus.PIVOTED
    return TaskStatus.UNFIXABLE


logger = logging.getLogger(__name__)

_SCA_STAGE_ORDER: dict[SCARemediationStage, int] = {
    SCARemediationStage.OSV_MINIMUM: 0,
    SCARemediationStage.NPM_SAME_MAJOR: 1,
    SCARemediationStage.PYPI_SAME_MAJOR: 1,
    SCARemediationStage.NPM_LATEST: 2,
    SCARemediationStage.MAVEN_LATEST: 1,
    SCARemediationStage.PYPI_LATEST: 2,
    SCARemediationStage.PACKAGE_OVERRIDE: 3,
    SCARemediationStage.CODE_WORKAROUND: 4,
}
_OVERRIDE_DEPENDENCY_TYPES = frozenset({"overrides", "resolutions", "pnpm_overrides"})
_MAX_REGISTRY_CANDIDATES = 3


def _supervisor_fetch_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str],
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool = False,
    ecosystem: str | None = None,
) -> list[RegistryCandidate | MavenRegistryCandidate]:
    """Fetch registry candidates as a Supervisor-owned traced operation.

    Args:
        package_name: Exact package identity committed by the Supervisor.
        security_floor: Minimum non-vulnerable version to consider.
        attempted_versions: Previously attempted versions to exclude.
        project_language: Language profile associated with the task.
        maven_mode: Whether Java dependency coordinates use Maven semantics.
        ecosystem: Optional explicit registry ecosystem for non-Maven tasks.

    Returns:
        Candidates returned by the selected registry adapter.
    """
    normalized_ecosystem = (
        str(
            ecosystem
            or (
                "maven"
                if project_language == ProjectLanguage.JAVA and maven_mode
                else "pypi"
                if project_language == ProjectLanguage.PYTHON
                else "npm"
            )
        )
        .strip()
        .casefold()
    )
    if normalized_ecosystem not in {"npm", "maven", "pypi"}:
        raise ValueError(f"Unsupported registry ecosystem: {normalized_ecosystem}")
    inputs = {
        "package_name": package_name,
        "security_floor": security_floor,
        "attempted_versions": attempted_versions,
        "project_language": project_language.value,
        "ecosystem": normalized_ecosystem,
    }

    def fetch() -> list[RegistryCandidate | MavenRegistryCandidate]:
        if normalized_ecosystem == "maven":
            return fetch_maven_registry_candidates(
                package_name,
                security_floor,
                attempted_versions,
            )
        if normalized_ecosystem == "pypi":
            return fetch_pypi_registry_candidates(
                package_name,
                security_floor,
                attempted_versions,
            )
        return fetch_registry_candidates(package_name, security_floor, attempted_versions)

    return invoke_with_trajectory(
        "supervisor.fetch_registry_candidates",
        fetch,
        inputs,
        run_type="tool",
    )


def _normalise_candidate_version(value: Any, *, maven_mode: bool = False) -> str:
    """Normalize a registry version without rewriting Maven coordinates."""
    normalized = str(value).strip()
    return normalized if maven_mode else normalized.lstrip("vV")


def _candidate_version_key(value: Any) -> tuple[int, int, int]:
    """Return a stable semantic-version sort key for Node report values."""
    parts = _normalise_candidate_version(value).split(".")
    try:
        numbers = tuple(int(part) for part in parts)
    except ValueError:
        return (0, 0, 0)
    return (numbers + (0, 0, 0))[:3]


def _select_report_candidate(
    values: Iterable[str],
    stage: SCARemediationStage,
    attempted_versions: set[str],
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool = False,
) -> str | None:
    """Select an approved report candidate without inventing a version."""
    attempted = {
        _normalise_candidate_version(version, maven_mode=maven_mode)
        for version in attempted_versions
    }
    eligible = [
        _normalise_candidate_version(value, maven_mode=maven_mode)
        for value in values
        if _normalise_candidate_version(value, maven_mode=maven_mode) not in attempted
    ]
    if project_language == ProjectLanguage.JAVA and maven_mode:
        eligible = [version for version in eligible if is_stable_maven_version(version)]
        if not eligible:
            return None
        selected = eligible[0]
        for version in eligible[1:]:
            try:
                comparison = compare_maven_versions(version, selected)
            except ValueError:
                continue
            if (stage == SCARemediationStage.OSV_MINIMUM and comparison < 0) or (
                stage == SCARemediationStage.MAVEN_LATEST and comparison > 0
            ):
                selected = version
        return selected
    if not eligible:
        return None
    return (
        min(eligible, key=_candidate_version_key)
        if stage == SCARemediationStage.OSV_MINIMUM
        else max(eligible, key=_candidate_version_key)
    )


def _approved_candidate_pool(
    diagnostics: UpdateRetryDiagnostics,
    target_package_name: str | None,
    *,
    maven_mode: bool = False,
) -> tuple[str, ...]:
    """Return the previously committed candidate pool for one target."""
    if (
        target_package_name
        and diagnostics.target_package_name
        and diagnostics.target_package_name != target_package_name
    ):
        return ()
    return tuple(
        dict.fromkeys(
            _normalise_candidate_version(version, maven_mode=maven_mode)
            for version in diagnostics.candidate_versions_considered
            if _normalise_candidate_version(version, maven_mode=maven_mode)
        )
    )


def _fetch_registry_candidates_for_task(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str],
    diagnostics: UpdateRetryDiagnostics,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    maven_mode: bool = False,
) -> list[RegistryCandidate | MavenRegistryCandidate]:
    """Revalidate only the task's committed candidate pool on retries."""
    approved_pool = _approved_candidate_pool(
        diagnostics,
        package_name,
        maven_mode=maven_mode,
    )
    candidates = _supervisor_fetch_registry_candidates(
        package_name,
        security_floor,
        set() if approved_pool else attempted_versions,
        project_language=project_language,
        maven_mode=maven_mode,
    )
    if not approved_pool:
        return candidates
    approved = set(approved_pool)
    return [
        candidate
        for candidate in candidates
        if _normalise_candidate_version(
            getattr(candidate, "version", ""),
            maven_mode=maven_mode,
        )
        in approved
    ]


def _supervisor_plan_npm_parent_version(inputs: dict[str, Any]) -> str:
    """Run the npm parent planner under an explicit Supervisor tool span."""
    return invoke_with_trajectory(
        "supervisor.plan_npm_parent_version",
        lambda: plan_npm_parent_version.invoke(inputs),
        inputs,
        run_type="tool",
    )


def _supervisor_plan_python_parent_version(inputs: dict[str, Any]) -> str:
    """Run the PyPI parent planner under an explicit Supervisor tool span."""
    return invoke_with_trajectory(
        "supervisor.plan_python_parent_version",
        lambda: plan_python_parent_version.invoke(inputs),
        inputs,
        run_type="tool",
    )


def _supervisor_dependency_type_candidates(
    strategy_stage: SCARemediationStage,
    target_dependency_type: str | None,
) -> list[str]:
    """Return dependency types permitted by the committed strategy."""
    if strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        candidates = (
            [target_dependency_type] if target_dependency_type in _OVERRIDE_DEPENDENCY_TYPES else []
        )
        candidates.extend(sorted(_OVERRIDE_DEPENDENCY_TYPES))
    else:
        candidates = [target_dependency_type] if target_dependency_type else []
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _build_high_level_retry_instruction(
    task: RemediationTask,
    group: VulnerabilityGroup | None,
    evaluation: QAEvaluation | None,
    diagnostics: UpdateRetryDiagnostics | None,
    *,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> str:
    """Synthesize a retry instruction without consulting package metadata."""
    component = group.vulnerable_component if group else task.parent_group_id
    parent_name, _, parent_type = (
        group_parent_context(group) if group is not None else (None, None, None)
    )
    target = task.target_package_name or (
        parent_name if task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE else component
    )
    dependency_type = task.target_dependency_type or parent_type
    if diagnostics and diagnostics.selected_version:
        manifest = group.file_paths[0] if group and group.file_paths else "package.json"
        return (
            f"Apply the outer-solver-approved dependency decision for {component}: "
            f"update only {target or component} in {manifest}"
            f"{f' ({dependency_type})' if dependency_type else ''} "
            f"to exact version {diagnostics.selected_version}; "
            "do not query a registry or choose another version; "
            "use modify_and_validate_npm_dependency so synchronization runs immediately after the edit."
        )
    if diagnostics and diagnostics.package_abandoned:
        return (
            f"The outer portfolio plan has no remaining approved update candidate for {component}. "
            "Do not select another version; report the bounded update failure so the Supervisor can pivot safely."
        )
    if evaluation and evaluation.failure_category == FailureCategory.PEER_CONFLICT:
        return (
            f"Use only the outer-solver-approved peer-compatible candidate for {component}; "
            "preserve the committed package and manifest targets."
        )
    return (
        f"Execute the next outer-solver-approved manifest candidate for {component}. "
        "Do not query a registry or invent a candidate."
    )


def _registry_selected_version(report: str) -> str | None:
    """Extract a planner-selected stable version from a registry report."""
    match = re.search(
        r"^-\s*Selected Version:\s*(\S+)",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match or match.group(1).upper() == "NONE":
        return None
    return match.group(1).strip().lstrip("vV")


def _registry_report_versions(report: str, label: str) -> list[str]:
    """Extract a comma-separated version list from a registry-tool report."""
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(.*)$",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    return list(
        dict.fromkeys(
            version.strip().lstrip("vV")
            for version in match.group(1).split(",")
            if version.strip() and version.strip().upper() != "NONE"
        )
    )


def _override_dependency_type(group: VulnerabilityGroup | None) -> str:
    """Return the package-manager-native override field for an SCA group."""
    managers = {
        (issue.package_manager or "").strip().lower()
        for issue in (group.localized_issues if group else [])
    }
    if "yarn" in managers:
        return "resolutions"
    if "pnpm" in managers:
        return "pnpm_overrides"
    return "overrides"


def _planner_plan_violations(
    plans: dict[str, SupervisorRetryPlan],
    task_queue: dict[str, RemediationTask],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> list[str]:
    """Validate retry-plan semantics before a plan can mutate routing state.

    Free-form plan evidence is intentionally not treated as an authority. These
    checks enforce the small set of invariants that must hold for an exact
    worker instruction to be safe.  Returning human-readable violations also
    makes the corrective replan visible in LangSmith through the supervisor's
    accumulated ``errors`` field.
    """
    violations: list[str] = []
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        if task is None:
            violations.append(f"task {task_id}: planner returned an unknown task")
            continue
        group = (group_by_id or {}).get(task.parent_group_id)
        maven_task = is_maven_group(group, project_language)
        python_task = not maven_task and _task_uses_pypi(task, group)
        latest_stage = _latest_sca_stage(task, group, project_language)
        allowed_stages = (
            {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.MAVEN_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
            if maven_task
            else {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.PYPI_SAME_MAJOR,
                SCARemediationStage.PYPI_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
            if python_task
            else {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.PACKAGE_OVERRIDE,
                SCARemediationStage.CODE_WORKAROUND,
            }
        )
        if plan.strategy_stage not in allowed_stages:
            ecosystem = "maven" if maven_task else "pypi" if python_task else "npm"
            violations.append(
                f"task {task_id}: stage {plan.strategy_stage.value} is not valid for {ecosystem}"
            )
        if task.status in _TERMINAL_STATUSES:
            violations.append(f"task {task_id}: planner returned a plan for terminal task")
        if plan.source_task_revision != task.task_revision:
            violations.append(
                f"task {task_id}: planner snapshot revision {plan.source_task_revision} "
                f"does not match current revision {task.task_revision}"
            )

        def version_key(value: str, *, maven_mode: bool = plan.version_scheme == "maven") -> str:
            normalized = value.strip()
            if not maven_mode:
                normalized = normalized.lstrip("vV")
            return normalized.lower()

        attempted = {version_key(version) for version in plan.attempted_versions if version}
        diagnostics = diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            attempted.update(version_key(version) for version in diagnostics.attempted_versions)

        selected = version_key(plan.selected_version) if plan.selected_version else None
        if selected and selected in attempted:
            violations.append(
                f"task {task_id}: selected version {plan.selected_version} was already attempted"
            )
        if (
            plan.strategy_stage == latest_stage
            and selected
            and plan.latest_version_seen
            and selected != plan.latest_version_seen.strip().lstrip("vV").lower()
        ):
            violations.append(
                f"task {task_id}: {latest_stage.value} selected {plan.selected_version}, "
                f"but registry latest is {plan.latest_version_seen}"
            )
        if plan.action == "retry_update" and selected is None:
            violations.append(
                f"task {task_id}: retry_update requires an unattempted exact selected_version"
            )
        if plan.action == "retry_update" and plan.exhausted_update_path:
            violations.append(f"task {task_id}: exhausted update path cannot retry update")
        if (
            plan.action == "retry_update"
            and plan.strategy_stage == SCARemediationStage.CODE_WORKAROUND
        ):
            violations.append(f"task {task_id}: retry_update cannot use code_workaround stage")
        if (
            plan.action == "retry_update"
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage] < _SCA_STAGE_ORDER[task.strategy_stage]
        ):
            violations.append(
                f"task {task_id}: retry plan stage {plan.strategy_stage.value} regresses "
                f"from committed stage {task.strategy_stage.value}"
            )
        required_pivot_stage = (
            SCARemediationStage.MAVEN_LATEST if plan.version_scheme == "maven" else latest_stage
        )
        if plan.action == "pivot_workaround" and plan.strategy_stage != required_pivot_stage:
            violations.append(
                f"task {task_id}: workaround pivot must be committed at "
                f"{required_pivot_stage.value}"
            )
        if plan.action == "pivot_workaround" and selected is not None:
            violations.append(
                f"task {task_id}: workaround pivot cannot retain selected version {plan.selected_version}"
            )
    return violations


def _repair_invalid_planner_plans(
    plans: dict[str, SupervisorRetryPlan],
    diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    violations: list[str] | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Apply a deterministic, fail-closed repair after corrective replanning.

    A valid unattempted candidate already present in planner evidence is safe to
    commit.  If no such candidate exists at the latest stage, the only safe
    action is the existing workaround pivot.  In particular, this function
    never preserves an invalid selected version merely to keep the graph
    moving.
    """
    repaired_diagnostics = dict(diagnostics_by_task)
    repaired_plans: dict[str, SupervisorRetryPlan] = {}
    invalid_task_ids = {
        match.group(1)
        for violation in (violations or [])
        if (match := re.match(r"task\s+(task-[\w-]+):", violation))
    }
    for task_id, plan in plans.items():
        if invalid_task_ids and task_id not in invalid_task_ids:
            repaired_plans[task_id] = plan
            continue
        task = task_queue[task_id]
        diagnostics = repaired_diagnostics.get(task_id)
        attempted = {
            version.strip().lstrip("vV").lower()
            for version in (diagnostics.attempted_versions if diagnostics else [])
            if version
        }
        group = group_by_id.get(task.parent_group_id)
        maven_task = is_maven_group(group, project_language)
        python_task = not maven_task and _task_uses_pypi(task, group)
        if python_task:
            attempted.update(
                version.strip().lstrip("vV").lower()
                for version in plan.attempted_versions
                if version
            )
        if maven_task and group is not None:
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            target_package = group.vulnerable_component or task_queue[task_id].target_package_name
            target_type = _maven_dependency_type(task_queue[task_id], group)
            floor, _floor_error = _canonical_security_floor(
                group,
                project_language=project_language,
            )
            approved = tuple(
                dict.fromkeys(
                    diagnostics.candidate_versions_considered or plan.candidate_versions_considered
                )
            )
            attempted_maven = {
                str(version).strip()
                for version in (
                    *(diagnostics.attempted_versions if diagnostics is not None else ()),
                    *plan.attempted_versions,
                )
                if str(version).strip()
            }
            stage = plan.strategy_stage
            selected = None
            if (
                floor
                and target_package
                and stage
                in {
                    SCARemediationStage.OSV_MINIMUM,
                    SCARemediationStage.MAVEN_LATEST,
                }
            ):
                pool = [
                    version
                    for version in approved
                    if is_stable_maven_version(version)
                    and not any(
                        compare_maven_versions(version, attempted) == 0
                        for attempted in attempted_maven
                        if is_stable_maven_version(attempted)
                    )
                    and compare_maven_versions(version, floor) >= 0
                ]
                if stage == SCARemediationStage.MAVEN_LATEST:
                    latest = plan.latest_version_seen or diagnostics.latest_version_seen
                    if latest in approved and latest in pool:
                        selected = latest
                elif pool:
                    selected = _select_report_candidate(
                        pool,
                        SCARemediationStage.OSV_MINIMUM,
                        attempted_maven,
                        project_language=project_language,
                        maven_mode=True,
                    )
                if selected is None and stage == SCARemediationStage.OSV_MINIMUM:
                    latest = plan.latest_version_seen or diagnostics.latest_version_seen
                    if latest in approved and latest in pool:
                        selected = latest
                        stage = SCARemediationStage.MAVEN_LATEST
            exhausted = selected is None
            if exhausted:
                stage = (
                    SCARemediationStage.CODE_WORKAROUND
                    if plan.strategy_stage == SCARemediationStage.CODE_WORKAROUND
                    else SCARemediationStage.MAVEN_LATEST
                )
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": stage,
                    "selected_version": selected,
                    "candidate_versions_considered": list(approved),
                    "candidate_dependency_types": [target_type],
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                    "security_floor": floor,
                    "exhausted_update_path": exhausted,
                    "parent_package_name": None,
                    "parent_minimum_version": None,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
            if exhausted:
                instruction = (
                    f"No approved unattempted Maven Central release remains for exact GAV "
                    f"{target_package}; implement a code workaround without changing a parent POM or BOM."
                )
            else:
                instruction = _build_high_level_retry_instruction(
                    task_queue[task_id].model_copy(
                        update={
                            "strategy_stage": stage,
                            "target_package_name": target_package,
                            "target_dependency_type": target_type,
                        }
                    ),
                    group,
                    None,
                    diagnostics,
                    project_language=project_language,
                )
            repaired_plans[task_id] = plan.model_copy(
                update={
                    "strategy_stage": stage,
                    "selected_version": selected,
                    "candidate_versions_considered": list(approved),
                    "candidate_dependency_types": [target_type],
                    "exhausted_update_path": exhausted,
                    "action": "pivot_workaround" if exhausted else "retry_update",
                    "exact_instruction": instruction,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                    "parent_minimum_version": None,
                }
            )
            continue

        candidate = None
        group = group_by_id.get(task.parent_group_id)
        latest_stage = _latest_sca_stage(task, group, project_language)
        python_task = not maven_task and _task_uses_pypi(task, group)
        valid_stages = (
            {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.PYPI_SAME_MAJOR,
                SCARemediationStage.PYPI_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
            if python_task
            else {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.PACKAGE_OVERRIDE,
                SCARemediationStage.CODE_WORKAROUND,
            }
        )
        plan_regresses = (
            task.strategy == RoutingStrategy.VERSION_BUMP
            and _SCA_STAGE_ORDER[plan.strategy_stage] < _SCA_STAGE_ORDER[task.strategy_stage]
        ) or plan.strategy_stage not in valid_stages
        # A correction cannot reopen an earlier stage.  In particular, a
        # stale same-major proposal must not revive a version-bump parent that
        # has already reached code_workaround.  Let the fail-closed branch
        # below create the deterministic latest-stage pivot instead.
        if not plan_regresses and plan.strategy_stage != SCARemediationStage.CODE_WORKAROUND:
            # At the ecosystem's latest stage, only the registry's declared latest version is
            # safe to repair into a retry. Older candidates are already
            # covered by the bounded update stages and selecting one here can
            # produce a null/stale dispatch state after the guardrail clears
            # the invalid plan. If latest was attempted, fail closed into the
            # deterministic workaround pivot.
            repair_candidates = (
                [plan.latest_version_seen]
                if plan.strategy_stage == latest_stage
                else [plan.latest_version_seen, *plan.candidate_versions_considered]
            )
            for version in repair_candidates:
                if version and version.strip().lstrip("vV").lower() not in attempted:
                    candidate = version.strip().lstrip("vV")
                    break

        if candidate:
            effective_stage = plan.strategy_stage
            same_major_stage = (
                SCARemediationStage.PYPI_SAME_MAJOR
                if python_task
                else SCARemediationStage.NPM_SAME_MAJOR
            )
            if (
                effective_stage == same_major_stage
                and plan.latest_version_seen
                and candidate == plan.latest_version_seen.strip().lstrip("vV")
            ):
                effective_stage = latest_stage
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            group = group_by_id.get(task_queue[task_id].parent_group_id)
            target_package = task_queue[task_id].target_package_name or (
                group_parent_context(group)[0] if group is not None else None
            )
            target_type = task_queue[task_id].target_dependency_type
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": list(
                        dict.fromkeys([*diagnostics.candidate_versions_considered, candidate])
                    ),
                    "registry_query_performed": False,
                    "exhausted_update_path": False,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            repaired_diagnostics[task_id] = diagnostics
            instruction = _build_high_level_retry_instruction(
                task_queue[task_id].model_copy(update={"strategy_stage": effective_stage}),
                group,
                None,
                diagnostics,
            )
            repaired_plans[task_id] = plan.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": diagnostics.candidate_versions_considered,
                    "action": "retry_update",
                    "exact_instruction": instruction,
                    "exhausted_update_path": False,
                    "target_package_name": target_package,
                    "target_dependency_type": target_type,
                }
            )
            continue

        # No direct unattempted candidate can be proven. Clear stale selection
        # and pivot at the terminal update stage so no guessed/old version is
        # sent to the dumb update worker.
        effective_stage = latest_stage
        if diagnostics is None:
            diagnostics = UpdateRetryDiagnostics(task_id=task_id)
        diagnostics = diagnostics.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
            }
        )
        repaired_diagnostics[task_id] = diagnostics
        component = group.vulnerable_component if group else task_queue[task_id].parent_group_id
        instruction = (
            f"Implement a code workaround or isolation strategy for {component} "
            "because the manifest-based update path is exhausted."
        )
        repaired_plans[task_id] = plan.model_copy(
            update={
                "strategy_stage": effective_stage,
                "selected_version": None,
                "exhausted_update_path": True,
                "action": "pivot_workaround",
                "exact_instruction": instruction,
                "target_package_name": task_queue[task_id].target_package_name,
                "target_dependency_type": task_queue[task_id].target_dependency_type,
            }
        )
    return repaired_diagnostics, repaired_plans


def _maven_dependency_type(
    task: RemediationTask,
    group: VulnerabilityGroup,
) -> str:
    """Return only the localized Maven declaration edit target."""
    if task.target_dependency_type in {"dependencies", "dependencyManagement"}:
        return task.target_dependency_type
    for localized in group.localized_issues:
        if localized.declaration_type in {"dependencies", "dependencyManagement"}:
            return localized.declaration_type
    return "dependencyManagement"


def _build_maven_retry_plan(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics,
    group: VulnerabilityGroup,
    *,
    requested_stage: SCARemediationStage | None,
) -> SupervisorRetryPlan:
    """Build an exact-GAV Maven plan without entering npm parent/override paths."""
    requested = requested_stage or task.strategy_stage
    effective_stage = (
        requested
        if _SCA_STAGE_ORDER.get(requested, 99) >= _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
        else task.strategy_stage
    )
    exact_gav = group.vulnerable_component or task.target_package_name or ""
    dependency_type = _maven_dependency_type(task, group)
    security_floor, floor_error = _canonical_security_floor(
        group,
        project_language=ProjectLanguage.JAVA,
    )
    attempted = set(diagnostics.attempted_versions)
    candidate_versions: list[str] = []
    latest_version_seen: str | None = None
    selected_version: str | None = None
    failure_reason = floor_error or ""
    registry_ran = False

    if (
        effective_stage
        in {
            SCARemediationStage.OSV_MINIMUM,
            SCARemediationStage.MAVEN_LATEST,
        }
        and security_floor
        and exact_gav
    ):
        registry_ran = True
        try:
            candidates = _fetch_registry_candidates_for_task(
                exact_gav,
                security_floor,
                attempted,
                diagnostics,
                project_language=ProjectLanguage.JAVA,
                maven_mode=True,
            )
            candidate_versions = [candidate.version for candidate in candidates]
            latest_version_seen = next(
                (
                    candidate.version
                    for candidate in candidates
                    if "maven_latest" in candidate.selection_roles
                ),
                None,
            )
            selected_version = select_maven_version(
                candidates,
                effective_stage,
                attempted,
            )
            if selected_version is None and not failure_reason:
                failure_reason = (
                    "No eligible unattempted Maven candidate has the role required "
                    f"for stage {effective_stage.value}."
                )
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"Deterministic Maven Central planning failed: {exc}"

    approved_pool = _approved_candidate_pool(
        diagnostics,
        exact_gav,
        maven_mode=True,
    )
    provenance_versions = list(dict.fromkeys(approved_pool or candidate_versions))
    if selected_version and selected_version not in set(provenance_versions):
        selected_version = None
        failure_reason = "Selected Maven version is outside the committed candidate whitelist."
    exhausted = selected_version is None and effective_stage in {
        SCARemediationStage.MAVEN_LATEST,
        SCARemediationStage.CODE_WORKAROUND,
    }
    effective_task = task.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected_version,
            "target_package_name": exact_gav,
            "target_dependency_type": dependency_type,
            "parent_package_name": None,
            "parent_package_version": None,
            "parent_minimum_version": None,
        }
    )
    effective_diagnostics = diagnostics.model_copy(
        update={
            "strategy_stage": effective_stage,
            "candidate_versions_considered": provenance_versions,
            "selected_version": selected_version,
            "latest_version_seen": latest_version_seen,
            "registry_query_performed": registry_ran,
            "exhausted_update_path": exhausted,
            "target_package_name": exact_gav,
            "target_dependency_type": dependency_type,
            "candidate_dependency_types": [dependency_type],
            "parent_package_name": None,
            "parent_minimum_version": None,
            "failure_reason": failure_reason,
        }
    )
    if exhausted:
        instruction = (
            f"No eligible Maven Central candidate remains for exact GAV {exact_gav}; "
            "implement a code workaround without changing a parent POM or BOM."
        )
    else:
        instruction = _build_high_level_retry_instruction(
            effective_task,
            group,
            None,
            effective_diagnostics,
            project_language=ProjectLanguage.JAVA,
        )
    return SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=effective_stage,
        version_scheme="maven",
        selected_version=selected_version,
        attempted_versions=list(diagnostics.attempted_versions),
        candidate_versions_considered=provenance_versions,
        candidate_dependency_types=[dependency_type],
        latest_version_seen=latest_version_seen,
        exhausted_update_path=exhausted,
        package_abandoned=diagnostics.package_abandoned,
        target_package_name=exact_gav,
        target_dependency_type=dependency_type,
        parent_minimum_version=None,
        action="pivot_workaround" if exhausted else "retry_update",
        exact_instruction=instruction,
    )


def _build_deterministic_retry_plan(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics,
    group: VulnerabilityGroup | None,
    *,
    requested_stage: SCARemediationStage | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
) -> SupervisorRetryPlan:
    """Build a retry plan solely from the outer solver's committed allowlist."""
    if (
        project_language == ProjectLanguage.JAVA
        and group is not None
        and is_maven_group(group, project_language)
    ):
        return _build_maven_retry_plan(
            task,
            diagnostics,
            group,
            requested_stage=requested_stage,
        )
    requested = requested_stage or task.strategy_stage
    effective_stage = (
        requested
        if _SCA_STAGE_ORDER.get(requested, 99) >= _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
        else task.strategy_stage
    )
    attempted = {
        str(version).strip().lstrip("vV").lower()
        for version in (*diagnostics.attempted_versions, *diagnostics.executed_versions)
        if str(version).strip()
    }
    approved_versions = list(
        dict.fromkeys(
            str(version).strip().lstrip("vV")
            for version in task.allowed_target_versions
            if str(version).strip()
        )
    )
    approved_version_keys = {version.lower() for version in approved_versions}
    selected_version = next(
        (
            version
            for version in [task.selected_version, *approved_versions]
            if version
            and version.strip().lstrip("vV").lower() in approved_version_keys
            and version.strip().lstrip("vV").lower() not in attempted
        ),
        None,
    )
    exhausted = task.strategy == RoutingStrategy.VERSION_BUMP and selected_version is None
    if task.strategy == RoutingStrategy.CODE_WORKAROUND:
        exhausted = False
    target_package_name = task.target_package_name
    target_dependency_type = task.target_dependency_type
    candidate_dependency_types = list(
        dict.fromkeys(
            str(value).strip() for value in task.allowed_dependency_types if str(value).strip()
        )
    )
    failure_reason = (
        "The outer portfolio plan has no remaining approved version." if exhausted else ""
    )
    effective_task = task.model_copy(update={"selected_version": selected_version})
    effective_diagnostics = diagnostics.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected_version,
            "candidate_versions_considered": approved_versions,
            "latest_version_seen": approved_versions[-1] if approved_versions else None,
            "registry_query_performed": False,
            "exhausted_update_path": exhausted,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "candidate_dependency_types": candidate_dependency_types,
            "parent_package_name": (
                group.parent_package_name if group is not None else diagnostics.parent_package_name
            ),
            "parent_minimum_version": task.parent_minimum_version,
            "failure_reason": failure_reason,
        }
    )
    if exhausted:
        component = group.vulnerable_component if group else task.parent_group_id
        instruction = (
            f"Implement the committed workaround path for {component}; "
            "the outer portfolio plan has no remaining approved update candidate."
        )
    else:
        instruction = _build_high_level_retry_instruction(
            effective_task,
            group,
            None,
            effective_diagnostics,
        )
    return SupervisorRetryPlan(
        task_id=task.task_id,
        source_task_revision=task.task_revision,
        strategy_stage=effective_stage,
        selected_version=selected_version,
        attempted_versions=list(diagnostics.attempted_versions),
        candidate_versions_considered=approved_versions,
        candidate_dependency_types=candidate_dependency_types,
        latest_version_seen=approved_versions[-1] if approved_versions else None,
        exhausted_update_path=exhausted,
        package_abandoned=diagnostics.package_abandoned,
        target_package_name=target_package_name,
        target_dependency_type=target_dependency_type,
        parent_minimum_version=task.parent_minimum_version,
        action="pivot_workaround" if exhausted else "retry_update",
        exact_instruction=instruction,
    )


def _needs_planner(
    task_queue: dict[str, RemediationTask],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    current_status: str,
) -> bool:
    """Return True when deterministic retry planning should run.

    Retry planning is reserved for VERSION_BUMP retry analysis and playbook selection.
    Both npm and PyPI candidates come only from the committed outer allowlist;
    CODE_WORKAROUND tasks do not use the version planner.
    """
    version_bump_retries = [
        t
        for t in task_queue.values()
        if (
            t.status == TaskStatus.NEEDS_RETRY
            and t.strategy == RoutingStrategy.VERSION_BUMP
            and (
                t.strategy_stage
                in {
                    SCARemediationStage.OSV_MINIMUM,
                    SCARemediationStage.NPM_SAME_MAJOR,
                    SCARemediationStage.NPM_LATEST,
                    SCARemediationStage.MAVEN_LATEST,
                    SCARemediationStage.PYPI_SAME_MAJOR,
                    SCARemediationStage.PYPI_LATEST,
                }
                or (
                    t.strategy_stage == SCARemediationStage.CODE_WORKAROUND
                    and not t.parent_package_name
                    and t.target_dependency_type
                    not in {
                        "overrides",
                        "resolutions",
                        "pnpm_overrides",
                    }
                )
            )
        )
    ]
    if not version_bump_retries:
        return False
    return (
        current_status == "qa_completed" or bool(qa_evaluations) or bool(retry_diagnostics_by_task)
    )


def _run_deterministic_retry_planner(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    *,
    target_task_ids: Iterable[str] | None = None,
    project_language: ProjectLanguage = ProjectLanguage.NODEJS,
    include_initial_maven: bool = False,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Plan retries from the outer portfolio's committed candidate allowlists.

    The portfolio plan owns version and retry-stage transitions. ``target_task_ids``
    narrows planning to tasks selected for the current Supervisor turn.
    """
    updated_diagnostics = dict(retry_diagnostics_by_task)
    plans: dict[str, SupervisorRetryPlan] = {}
    target_ids = set(target_task_ids) if target_task_ids is not None else None
    retry_tasks = sorted(
        (
            task
            for task in task_queue.values()
            if task.strategy == RoutingStrategy.VERSION_BUMP
            and (target_ids is None or task.task_id in target_ids)
            and task.strategy_stage
            in {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.MAVEN_LATEST,
                SCARemediationStage.PYPI_SAME_MAJOR,
                SCARemediationStage.PYPI_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
            and (
                task.status == TaskStatus.NEEDS_RETRY
                or (
                    include_initial_maven
                    and task.status == TaskStatus.PENDING
                    and not task.selected_version
                    and is_maven_group(
                        group_by_id.get(task.parent_group_id),
                        project_language,
                    )
                )
            )
        ),
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    for task in retry_tasks:
        diagnostics = updated_diagnostics.get(
            task.task_id,
            UpdateRetryDiagnostics(task_id=task.task_id, strategy_stage=task.strategy_stage),
        )
        # The deterministic router owns the exhausted-update pivot. Never
        # reopen a path that its guardrail has already proven exhausted.
        if _is_exhausted_update_pivot_candidate(task, diagnostics):
            continue
        group = group_by_id.get(task.parent_group_id)
        maven_mode = bool(
            project_language == ProjectLanguage.JAVA
            and group is not None
            and is_maven_group(group, project_language)
        )
        plan = _build_deterministic_retry_plan(
            task,
            diagnostics,
            group,
            project_language=project_language,
        )
        while maven_mode and plan.selected_version is None and not plan.exhausted_update_path:
            next_stage = _next_sca_stage(
                plan.strategy_stage,
                project_language=ProjectLanguage.JAVA,
                maven_mode=True,
            )
            if next_stage == SCARemediationStage.CODE_WORKAROUND:
                break
            plan = _build_deterministic_retry_plan(
                task,
                diagnostics,
                group,
                requested_stage=next_stage,
                project_language=project_language,
            )
        plans[task.task_id] = plan
        parent_group = group_by_id.get(task.parent_group_id)
        parent_plan = select_package_fix_plan(parent_group).plan if parent_group else None
        security_floor = diagnostics.security_floor or (
            parent_plan.fixed_version if parent_plan is not None else None
        )
        maven_registry_query_attempted = bool(
            maven_mode
            and security_floor
            and plan.target_package_name
            and plan.strategy_stage
            in {SCARemediationStage.OSV_MINIMUM, SCARemediationStage.MAVEN_LATEST}
        )
        updated_diagnostics[task.task_id] = diagnostics.model_copy(
            update={
                "strategy_stage": plan.strategy_stage,
                "security_floor": diagnostics.security_floor,
                "selected_version": plan.selected_version,
                "candidate_versions_considered": plan.candidate_versions_considered,
                "latest_version_seen": plan.latest_version_seen,
                "registry_query_performed": maven_registry_query_attempted,
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name,
                "target_dependency_type": plan.target_dependency_type,
                "candidate_dependency_types": plan.candidate_dependency_types,
                "parent_minimum_version": plan.parent_minimum_version,
            }
        )
    return updated_diagnostics, plans


def _commit_retry_plans(
    task_queue: dict[str, RemediationTask],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    retry_plans_by_task: dict[str, SupervisorRetryPlan],
    plans: dict[str, SupervisorRetryPlan],
) -> None:
    """Commit deterministic retry inputs before the next worker dispatch."""
    for task_id, plan in sorted(plans.items()):
        task = task_queue.get(task_id)
        if task is None:
            continue
        committed_task = _commit_task_transition(
            task_queue,
            task_id,
            updates={
                "strategy_stage": plan.strategy_stage,
                "instruction": plan.exact_instruction,
                "selected_version": plan.selected_version,
                "exhausted_update_path": plan.exhausted_update_path,
                "target_package_name": plan.target_package_name or task.target_package_name,
                "target_dependency_type": plan.target_dependency_type
                or task.target_dependency_type,
                "parent_minimum_version": plan.parent_minimum_version
                or task.parent_minimum_version,
            },
            close_attempt=True,
            clear_selected_version=plan.selected_version is None,
        )
        if committed_task is None:
            continue
        diagnostics = retry_diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            retry_diagnostics_by_task[task_id] = diagnostics.model_copy(
                update={
                    "committed_attempt_id": committed_task.current_attempt_id,
                    "strategy_stage": committed_task.strategy_stage,
                    "selected_version": committed_task.selected_version,
                    "candidate_dependency_types": plan.candidate_dependency_types,
                    "exhausted_update_path": committed_task.exhausted_update_path,
                    "target_package_name": committed_task.target_package_name,
                    "target_dependency_type": committed_task.target_dependency_type,
                    "parent_minimum_version": committed_task.parent_minimum_version,
                    "instruction_digest": instruction_digest(committed_task.instruction),
                }
            )
        retry_plans_by_task[task_id] = plan.model_copy(
            update={"source_task_revision": committed_task.task_revision}
        )


def _portfolio_cluster_targets(
    portfolio_plan: PortfolioPlan | None,
    task_queue: dict[str, RemediationTask],
    *,
    qa: bool,
    preferred_ids: list[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Return the first active cluster only when its next action is ready.

    A blocked head cluster is returned with no targets so callers can prevent
    fallback routing from dispatching a later batch out of portfolio order.
    """
    if portfolio_plan is None:
        return None, []
    preferred = set(preferred_ids or [])
    clusters_by_id = {cluster.cluster_id: cluster for cluster in portfolio_plan.clusters}
    cluster_positions = {
        cluster_id: position for position, cluster_id in enumerate(portfolio_plan.cluster_order)
    }
    has_cycle_fallback = any(
        "cycle" in diagnostic.casefold() for diagnostic in portfolio_plan.diagnostics
    )

    def upstreams_terminal(cluster: Any) -> bool:
        members = set(cluster.task_ids)
        return all(
            dependency.upstream_task_id in members
            or task_queue.get(dependency.upstream_task_id) is None
            or task_queue[dependency.upstream_task_id].status in _TERMINAL_STATUSES
            or (
                has_cycle_fallback
                and cluster_positions.get(
                    portfolio_plan.task_to_cluster.get(dependency.upstream_task_id, ""),
                    -1,
                )
                > cluster_positions.get(cluster.cluster_id, -1)
            )
            for dependency in cluster.dependencies
        )

    ordered_cluster_ids = list(portfolio_plan.cluster_order)
    if preferred:
        preferred_cluster_ids = [
            portfolio_plan.task_to_cluster[task_id]
            for task_id in portfolio_plan.task_order
            if task_id in preferred and task_id in portfolio_plan.task_to_cluster
        ]
        ordered_cluster_ids = list(dict.fromkeys(preferred_cluster_ids + ordered_cluster_ids))

    for cluster_id in ordered_cluster_ids:
        cluster = clusters_by_id.get(cluster_id)
        if cluster is None:
            return cluster_id, []
        tasks = [task_queue.get(task_id) for task_id in cluster.task_ids]
        if any(task is None for task in tasks):
            return cluster_id, []
        concrete_tasks = [task for task in tasks if task is not None]
        if all(task.status in _TERMINAL_STATUSES for task in concrete_tasks):
            continue
        if not getattr(cluster, "dispatchable", True) or not upstreams_terminal(cluster):
            logger.warning(
                "portfolio head cluster %s is blocked; preserving later batches",
                cluster_id,
            )
            return cluster_id, []
        if qa:
            if all(task.status == TaskStatus.OPTIMISTICALLY_FIXED for task in concrete_tasks):
                return cluster_id, list(cluster.task_ids)
            return cluster_id, []
        if any(task.status in _TERMINAL_STATUSES for task in concrete_tasks):
            return cluster_id, []
        if any(task.current_attempt_id is not None for task in concrete_tasks):
            return cluster_id, []
        if not all(task.status in _WORKABLE_STATUSES for task in concrete_tasks):
            return cluster_id, []
        if any(_has_existing_workaround_child(task, task_queue) for task in concrete_tasks):
            return cluster_id, []
        strategies = {task.strategy for task in concrete_tasks}
        if len(strategies) != 1 or not strategies.issubset(
            {RoutingStrategy.VERSION_BUMP, RoutingStrategy.CODE_WORKAROUND}
        ):
            return cluster_id, []
        if strategies == {RoutingStrategy.CODE_WORKAROUND} and len(concrete_tasks) != 1:
            logger.warning(
                "portfolio workaround cluster %s has multiple tasks; preserving later batches",
                cluster_id,
            )
            return cluster_id, []
        return cluster_id, list(cluster.task_ids)
    return None, []


def _build_workaround_retry_instruction(
    task: RemediationTask,
    evaluation: QAEvaluation | None,
    group: VulnerabilityGroup | None = None,
) -> str:
    """Synthesize a high-level retry instruction for a generic workaround task."""
    category_str = (
        evaluation.failure_category.value
        if evaluation and evaluation.failure_category
        else "unknown"
    )
    feedback_str = (
        evaluation.retry_feedback
        if evaluation and evaluation.retry_feedback
        else "No feedback provided."
    )
    component = (group.vulnerable_component if group else None) or task.parent_group_id
    return (
        f"RETRY: Your previous code workaround attempt for {component} failed QA.\n"
        f"Failure category: {category_str}\n"
        f"QA feedback: {feedback_str}\n\n"
        f"Original instruction: {task.instruction}\n\n"
        f"Fix the issues identified by QA and re-apply a valid code workaround."
    )


def _deterministic_routing(
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup],
    qa_evaluations: dict[str, QAEvaluation],
    retry_diagnostics_by_task: dict[str, UpdateRetryDiagnostics],
    action_summaries: list[AgentActionSummary] | None = None,
    active_target_task_ids: list[str] | None = None,
    current_status: str = "",
    triage_required: bool = False,
    portfolio_plan: PortfolioPlan | None = None,
    delta_replay_task_ids: Iterable[str] = (),
) -> SupervisorDecision:
    """
    Pure-Python routing used as the Supervisor's authoritative state machine.

    Implements the fixed Supervisor priority rules.
    """
    tasks = sorted(task_queue.values(), key=lambda task: _task_sort_key(task, group_by_id))
    non_terminal = [t for t in tasks if t.status not in _TERMINAL_STATUSES]
    # Defer incremental scan triage until ordinary remediation work is exhausted.
    if not non_terminal and triage_required:
        return SupervisorDecision(
            decision_code=DecisionCode.TRIAGE_REQUIRED,
            next_node="triage",
            target_task_ids=[],
            instructions="Route cumulative accepted scan findings to post-remediation triage.",
            decision_reason=(
                "No actionable remediation tasks remain and accepted scans contain "
                "baseline-absent or remaining target findings."
            ),
        )

    if not group_by_id and not task_queue:
        return SupervisorDecision(
            decision_code=DecisionCode.NO_VALID_GROUPS,
            next_node="teardown",
            target_task_ids=[],
            instructions="No valid vulnerability groups are available for remediation.",
            decision_reason="Deterministic routing found no valid groups.",
        )

    # No final full-workspace scan node remains; terminal tasks go to triage
    # when accepted incremental findings exist, otherwise directly to teardown.
    if not non_terminal:
        return SupervisorDecision(
            decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
            next_node="teardown",
            target_task_ids=[],
            instructions="All tasks are terminal. Proceeding to teardown.",
            decision_reason="No actionable tasks remain.",
        )

    # If an active task is optimistically_fixed â†’ route it to qa_critic.
    current_task_qa_ready = _qa_ready_task_ids(
        task_queue,
        preferred_ids=list(active_target_task_ids or []),
        group_by_id=group_by_id,
        limit=QA_DISPATCH_LIMIT,
    )
    portfolio_cluster_id, portfolio_qa_targets = _portfolio_cluster_targets(
        portfolio_plan,
        task_queue,
        qa=True,
        preferred_ids=list(active_target_task_ids or []),
    )
    if portfolio_cluster_id and portfolio_qa_targets:
        is_batch = len(portfolio_qa_targets) > 1
        return SupervisorDecision(
            decision_code=(DecisionCode.QA_READY_BATCH if is_batch else DecisionCode.QA_READY),
            next_node="qa_critic",
            target_task_ids=portfolio_qa_targets,
            cluster_id=portfolio_cluster_id if is_batch else None,
            instructions=(
                "Run QA atomically across the completed package cluster."
                if is_batch
                else "Run QA on the next completed portfolio task."
            ),
            decision_reason=(
                f"All package tasks in cluster '{portfolio_cluster_id}' are ready for QA."
            ),
        )
    if current_status != "qa_completed" and current_task_qa_ready:
        return SupervisorDecision(
            decision_code=DecisionCode.QA_READY,
            next_node="qa_critic",
            target_task_ids=current_task_qa_ready,
            instructions="Run QA on the current remediated task before starting more remediation.",
            decision_reason=(
                f"Routing task '{current_task_qa_ready[0]}' to QA after a successful worker attempt."
            ),
        )

    all_qa_ready = _qa_ready_task_ids(
        task_queue,
        group_by_id=group_by_id,
        limit=QA_DISPATCH_LIMIT,
    )
    if all_qa_ready and portfolio_cluster_id is None:
        return SupervisorDecision(
            decision_code=DecisionCode.QA_READY_BATCH,
            next_node="qa_critic",
            target_task_ids=all_qa_ready,
            instructions="Run QA on the next remaining optimistically fixed task.",
            decision_reason=f"Routing task '{all_qa_ready[0]}' to QA.",
        )

    # Collect tasks that still need work
    workable = [t for t in non_terminal if t.status in _WORKABLE_STATUSES]

    portfolio_work_cluster_id, portfolio_work_targets = _portfolio_cluster_targets(
        portfolio_plan,
        task_queue,
        qa=False,
        preferred_ids=list(active_target_task_ids or []),
    )
    portfolio_work_target_ids = set(portfolio_work_targets)
    # NO_FIX is a deterministic same-task state machine.  Keep it ahead of
    # generic workaround routing so an untrusted decision cannot skip a
    # mitigation stage or
    # let MAX_RETRIES terminate the lifecycle early.
    no_fix_workable = sorted(
        [
            task
            for task in workable
            if task.strategy == RoutingStrategy.CODE_WORKAROUND
            and (
                (
                    task.no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL
                    and task.status == TaskStatus.PENDING
                )
                or (
                    task.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL
                    and task.status == TaskStatus.NEEDS_RETRY
                )
            )
            and (
                portfolio_plan is None
                or task.task_id not in portfolio_plan.task_ids
                or task.task_id in portfolio_work_target_ids
            )
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if no_fix_workable:
        target = no_fix_workable[0]
        evaluation = qa_evaluations.get(target.task_id)
        revised_instructions: dict[str, str] = {}
        feedback_by_task: dict[str, str] = {}
        if target.no_fix_stage == NoFixMitigationStage.VULNERABLE_CODE_REMOVAL:
            revised_instructions[target.task_id] = build_no_fix_retry_instruction(
                target,
                group_by_id.get(target.parent_group_id),
                evaluation=evaluation,
            )
        if evaluation and evaluation.retry_feedback:
            feedback_by_task[target.task_id] = evaluation.retry_feedback
        return SupervisorDecision(
            decision_code=DecisionCode.NO_FIX_LIFECYCLE,
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions=("Advance the NO_FIX task through its supervisor-owned mitigation stage."),
            decision_reason=(
                f"Deterministic NO_FIX routing selected {target.no_fix_stage.value} "
                f"for task '{target.task_id}'."
            ),
        )

    delta_replay_task_ids = set(delta_replay_task_ids)
    delta_replay_targets = [
        task_id
        for task_id in portfolio_work_targets
        if task_id in delta_replay_task_ids
        and task_queue[task_id].strategy == RoutingStrategy.VERSION_BUMP
    ]
    if delta_replay_targets:
        target = task_queue[delta_replay_targets[0]]
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[target.task_id],
            instructions=(
                "Run the restored delta peer's committed candidate as a singleton "
                "so authoritative QA can validate it independently."
            ),
            decision_reason=(
                f"Dispatching restored delta peer '{target.task_id}' from the "
                "committed portfolio head before unrelated retries."
            ),
        )

    # Tactical reasoning gets the first opportunity to pivot after a QA
    # failure. This deterministic ladder is fallback-only, so a
    # BREAKING_CHANGE can become a workaround without burning later versions.

    exhausted_retries = sorted(
        [
            task
            for task in workable
            if _is_exhausted_update_pivot_candidate(
                task,
                retry_diagnostics_by_task.get(task.task_id),
            )
            and not _has_existing_workaround_child(task, task_queue)
            and (
                portfolio_plan is None
                or task.task_id not in portfolio_plan.task_ids
                or task.task_id in portfolio_work_target_ids
            )
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if exhausted_retries:
        spawn_requests: list[TaskSpawnRequest] = []
        feedback_by_task: dict[str, str] = {}
        for task in exhausted_retries:
            component = (
                group_by_id.get(task.parent_group_id).vulnerable_component
                if task.parent_group_id in group_by_id
                else task.parent_group_id
            )
            eval_ = qa_evaluations.get(task.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[task.task_id] = eval_.retry_feedback
            spawn_requests.append(
                TaskSpawnRequest(
                    parent_task_id=task.task_id,
                    strategy=RoutingStrategy.CODE_WORKAROUND,
                    instruction=(
                        f"Original Context: {task.instruction}\n\n"
                        f"Pivot Directive: Implement a code workaround or isolation strategy for {component} "
                        "because manifest-based update remediation appears exhausted after bounded registry-guided retries."
                    ),
                    reason=(
                        "Deterministic fallback: exhausted manifest remediation must pivot "
                        "to a workaround child task."
                    ),
                )
            )
        return SupervisorDecision(
            decision_code=DecisionCode.EXHAUSTED_UPDATE_PIVOT,
            next_node="workaround_subagent",
            target_task_ids=[exhausted_retries[0].task_id],
            spawn_requests=spawn_requests,
            feedback_by_task=feedback_by_task,
            instructions="Pivot exhausted update remediation to workaround child tasks.",
            decision_reason=(
                f"Retry diagnostics show {len(exhausted_retries)} update task(s) no longer have a remaining manifest-based update path."
            ),
        )

    # A conclusive QA failure is still the active task's responsibility, even
    # when the portfolio plan has already advanced to another ready cluster.
    # Give a singleton retry to tactical planning before dispatching unrelated
    # work. Never split a committed multi-task cluster.
    active_retry_task = None
    active_retry_evaluation = None
    if len(active_target_task_ids or ()) == 1:
        active_task_id = active_target_task_ids[0]
        candidate_task = task_queue.get(active_task_id)
        candidate_evaluation = qa_evaluations.get(active_task_id)
        if (
            candidate_task is not None
            and candidate_task.strategy == RoutingStrategy.VERSION_BUMP
            and candidate_task.status == TaskStatus.NEEDS_RETRY
            and candidate_task.current_attempt_id is None
            and candidate_evaluation is not None
            and not candidate_evaluation.passed
            and not candidate_evaluation.contract_error
            and not candidate_evaluation.evidence_inconclusive
            and not _has_existing_workaround_child(candidate_task, task_queue)
        ):
            cluster_id = (
                portfolio_plan.task_to_cluster.get(active_task_id)
                if portfolio_plan is not None
                else None
            )
            active_cluster = (
                next(
                    (
                        cluster
                        for cluster in portfolio_plan.clusters
                        if cluster.cluster_id == cluster_id
                    ),
                    None,
                )
                if portfolio_plan is not None and cluster_id is not None
                else None
            )
            if active_cluster is None or len(active_cluster.task_ids) == 1:
                active_retry_task = candidate_task
                active_retry_evaluation = candidate_evaluation

    if active_retry_task is not None and active_retry_evaluation is not None:
        retry_instruction = _build_high_level_retry_instruction(
            active_retry_task,
            group_by_id.get(active_retry_task.parent_group_id),
            active_retry_evaluation,
            retry_diagnostics_by_task.get(active_retry_task.task_id),
        )
        retry_feedback = (
            {active_retry_task.task_id: active_retry_evaluation.retry_feedback}
            if active_retry_evaluation.retry_feedback
            else {}
        )
        return SupervisorDecision(
            decision_code=DecisionCode.RETRY_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[active_retry_task.task_id],
            feedback_by_task=retry_feedback,
            revised_instructions={active_retry_task.task_id: retry_instruction},
            instructions=(
                "Route the active QA-failed dependency task through tactical/retry planning "
                "before dispatching unrelated portfolio work."
            ),
            decision_reason=(
                f"Prioritizing active QA retry task '{active_retry_task.task_id}' "
                "before unrelated portfolio dispatch."
            ),
        )

    portfolio_cluster_id = portfolio_work_cluster_id
    portfolio_update_targets = portfolio_work_targets
    if portfolio_cluster_id and portfolio_update_targets:
        target = task_queue[portfolio_update_targets[0]]
        if target.strategy == RoutingStrategy.CODE_WORKAROUND:
            if target.no_fix_stage is not None or target.retry_count >= MAX_RETRIES:
                return SupervisorDecision(
                    decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                    next_node="teardown",
                    target_task_ids=[],
                    instructions="Stop safely; the committed workaround batch cannot be dispatched.",
                    decision_reason=(
                        f"Portfolio head cluster '{portfolio_cluster_id}' contains a "
                        f"non-dispatchable workaround task '{target.task_id}'."
                    ),
                )
            evaluation = qa_evaluations.get(target.task_id)
            feedback_by_task: dict[str, str] = {}
            revised_instructions: dict[str, str] = {}
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[target.task_id] = evaluation.retry_feedback
            if target.status == TaskStatus.NEEDS_RETRY and evaluation:
                revised_instructions[target.task_id] = _build_workaround_retry_instruction(
                    target,
                    evaluation,
                    group_by_id.get(target.parent_group_id),
                )
            return SupervisorDecision(
                decision_code=DecisionCode.WORKAROUND_DISPATCH,
                next_node="workaround_subagent",
                target_task_ids=[target.task_id],
                feedback_by_task=feedback_by_task,
                revised_instructions=revised_instructions,
                instructions="Apply the next committed portfolio code workaround.",
                decision_reason=(
                    f"Dispatching the ordered workaround task '{target.task_id}' "
                    f"from cluster '{portfolio_cluster_id}'."
                ),
            )
    if portfolio_cluster_id and len(portfolio_update_targets) > 1:
        return SupervisorDecision(
            decision_code=DecisionCode.ATOMIC_CLUSTER_DISPATCH,
            next_node="update_subagent",
            target_task_ids=portfolio_update_targets,
            cluster_id=portfolio_cluster_id,
            instructions="Apply the committed package-cluster update atomically.",
            decision_reason=(
                f"Dispatching the next dependency cluster '{portfolio_cluster_id}' "
                "after all upstream package groups completed."
            ),
        )
    if portfolio_cluster_id and portfolio_update_targets:
        target = task_queue[portfolio_update_targets[0]]
        if target.status == TaskStatus.NEEDS_RETRY:
            evaluation = qa_evaluations.get(target.task_id)
            feedback_by_task: dict[str, str] = {}
            revised_instructions = {
                target.task_id: _build_high_level_retry_instruction(
                    target,
                    group_by_id.get(target.parent_group_id),
                    evaluation,
                    retry_diagnostics_by_task.get(target.task_id),
                )
            }
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[target.task_id] = evaluation.retry_feedback
            return SupervisorDecision(
                decision_code=DecisionCode.RETRY_VERSION_BUMP,
                next_node="update_subagent",
                target_task_ids=[target.task_id],
                feedback_by_task=feedback_by_task,
                revised_instructions=revised_instructions,
                instructions=(
                    "Route the next committed portfolio task back to the update worker "
                    "with its high-level retry goal."
                ),
                decision_reason=(
                    f"Retrying the ready singleton portfolio task '{target.task_id}' "
                    f"from cluster '{portfolio_cluster_id}' before dependent clusters."
                ),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[target.task_id],
            instructions=(
                "Apply the next committed portfolio task update before dispatching "
                "dependent package clusters."
            ),
            decision_reason=(
                f"Dispatching the ready singleton portfolio task '{target.task_id}' "
                f"from cluster '{portfolio_cluster_id}' before dependent clusters."
            ),
        )
    if portfolio_plan is not None:
        active_planned_tasks = [
            task_queue[task_id]
            for task_id in portfolio_plan.task_ids
            if task_id in task_queue and task_queue[task_id].status not in _TERMINAL_STATUSES
        ]
        if portfolio_cluster_id is not None or active_planned_tasks:
            return SupervisorDecision(
                decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
                next_node="teardown",
                target_task_ids=[],
                instructions="Stop safely; do not dispatch past the committed portfolio head.",
                decision_reason=(
                    "The committed portfolio head is not dispatchable; later batches remain blocked."
                ),
            )

    retry_version_bump = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.VERSION_BUMP
            and t.status == TaskStatus.NEEDS_RETRY
            and not _has_existing_workaround_child(t, task_queue)
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if retry_version_bump:
        batch = retry_version_bump[:UPDATE_DISPATCH_LIMIT]
        feedback_by_task: dict[str, str] = {}
        revised_instructions: dict[str, str] = {}
        for task in batch:
            evaluation = qa_evaluations.get(task.task_id)
            if evaluation and evaluation.retry_feedback:
                feedback_by_task[task.task_id] = evaluation.retry_feedback
            revised_instructions[task.task_id] = _build_high_level_retry_instruction(
                task,
                group_by_id.get(task.parent_group_id),
                evaluation,
                retry_diagnostics_by_task.get(task.task_id),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.RETRY_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            revised_instructions=revised_instructions,
            instructions="Route the retry-bound dependency task back to the update worker with its high-level retry goal.",
            decision_reason=(
                f"Routing retry VERSION_BUMP task '{batch[0].task_id}' to update_subagent for registry-guided evidence gathering."
            ),
        )

    # VERSION_BUMP tasks route to update_subagent one at a time for non-retry work.
    version_bump = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.VERSION_BUMP
            and t.status != TaskStatus.NEEDS_RETRY
            and not _has_existing_workaround_child(t, task_queue)
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if version_bump:
        batch = version_bump[:UPDATE_DISPATCH_LIMIT]
        feedback_by_task: dict[str, str] = {}
        for t in batch:
            eval_ = qa_evaluations.get(t.task_id)
            if eval_ and eval_.retry_feedback:
                feedback_by_task[t.task_id] = eval_.retry_feedback
        return SupervisorDecision(
            decision_code=DecisionCode.NEW_VERSION_BUMP,
            next_node="update_subagent",
            target_task_ids=[t.task_id for t in batch],
            feedback_by_task=feedback_by_task,
            instructions="Apply the required version bump in the package manifest for this task only.",
            decision_reason=(f"Routing VERSION_BUMP task '{batch[0].task_id}' to update_subagent."),
        )

    # CODE_WORKAROUND tasks: send exactly one at a time to workaround_subagent
    workaround = sorted(
        [
            t
            for t in workable
            if t.strategy == RoutingStrategy.CODE_WORKAROUND
            and t.no_fix_stage is None
            and t.retry_count < MAX_RETRIES
        ],
        key=lambda task: _task_sort_key(task, group_by_id),
    )
    if workaround:
        target = workaround[0]
        eval_ = qa_evaluations.get(target.task_id)
        feedback: dict[str, str] = {}
        revised_instructions: dict[str, str] = {}
        if eval_ and eval_.retry_feedback:
            feedback[target.task_id] = eval_.retry_feedback
        if target.status == TaskStatus.NEEDS_RETRY and eval_:
            revised_instructions[target.task_id] = _build_workaround_retry_instruction(
                target,
                eval_,
                group_by_id.get(target.parent_group_id),
            )
        return SupervisorDecision(
            decision_code=DecisionCode.WORKAROUND_DISPATCH,
            next_node="workaround_subagent",
            target_task_ids=[target.task_id],
            feedback_by_task=feedback,
            revised_instructions=revised_instructions,
            instructions="Apply the minimal safe code workaround for this vulnerability.",
            decision_reason=(f"Routing task '{target.task_id}' to workaround_subagent."),
        )

    # Unexpected: no workable tasks found â†’ teardown as safe default
    return SupervisorDecision(
        decision_code=DecisionCode.NO_ACTIONABLE_TASKS,
        next_node="teardown",
        target_task_ids=[],
        instructions="No actionable tasks remain.",
        decision_reason=("Deterministic fallback: no workable tasks found, routing to teardown."),
    )


def _emit_audit(
    decision: SupervisorDecision,
    consistency_events: list[StateConsistencyEvent],
    state_revision: int,
) -> AuditRecord:
    """Build the typed audit record for a deterministic decision."""

    return AuditRecord(
        decision_code=decision.decision_code or DecisionCode.INVALID_LLM_DECISION,
        next_node=decision.next_node,
        target_task_ids=list(decision.target_task_ids),
        reasoning=decision.decision_reason,
        state_revision=state_revision,
        consistency_events=list(consistency_events),
    )


def _no_fix_decision_requires_fallback(
    decision: SupervisorDecision,
    task_queue: dict[str, RemediationTask],
    group_by_id: dict[str, VulnerabilityGroup] | None = None,
) -> bool:
    """Return whether an untrusted router decision violates NO_FIX routing."""
    actionable = sorted(
        [
            task
            for task in task_queue.values()
            if task.status in _WORKABLE_STATUSES
            and task.strategy == RoutingStrategy.CODE_WORKAROUND
            and task.no_fix_stage
            in {
                NoFixMitigationStage.PACKAGE_REMOVAL,
                NoFixMitigationStage.VULNERABLE_CODE_REMOVAL,
            }
        ],
        key=lambda task: _task_sort_key(task, group_by_id or {}),
    )
    if not actionable:
        return False

    expected_task_id = actionable[0].task_id
    if actionable[0].status == TaskStatus.OPTIMISTICALLY_FIXED:
        return (
            decision.next_node != "qa_critic"
            or expected_task_id not in decision.target_task_ids
            or expected_task_id in decision.unfixable_task_ids
            or decision.task_status_updates.get(expected_task_id) == TaskStatus.UNFIXABLE
            or decision.updated_task_strategies.get(expected_task_id)
            not in (None, RoutingStrategy.CODE_WORKAROUND)
        )
    if decision.next_node != "workaround_subagent":
        return True
    if decision.target_task_ids != [expected_task_id]:
        return True
    if expected_task_id in decision.unfixable_task_ids:
        return True
    if decision.task_status_updates.get(expected_task_id) == TaskStatus.UNFIXABLE:
        return True
    if decision.updated_task_strategies.get(expected_task_id) not in (
        None,
        RoutingStrategy.CODE_WORKAROUND,
    ):
        return True
    return any(request.parent_task_id == expected_task_id for request in decision.spawn_requests)
