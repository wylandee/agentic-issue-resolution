"""Deterministic retry planning and Supervisor-owned registry selection."""

from __future__ import annotations

import json
import logging
import re
import tomllib
from collections.abc import Iterable
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.version import InvalidVersion, Version
from semantic_version import Version as SemVerVersion

from remediation_engine.contracts.schemas import (
    FailureCategory,
    QAEvaluation,
    RemediationTask,
    RoutingStrategy,
    SCARemediationStage,
    SupervisorRetryPlan,
    TaskStatus,
    UpdateRetryDiagnostics,
    VulnerabilityGroup,
)
from remediation_engine.contracts.version_policy import (
    RegistryCandidate,
    registry_version_key,
    select_version,
)
from remediation_engine.orchestration.supervisor_policy import (
    _TERMINAL_STATUSES,
    _canonical_security_floor,
    _is_exhausted_update_pivot_candidate,
    _next_sca_stage,
    _task_sort_key,
    instruction_digest,
)
from remediation_engine.orchestration.task_utils import group_parent_context, is_transitive_group
from remediation_engine.orchestration.trajectory_exporter import invoke_with_trajectory
from remediation_engine.runtime.sandbox_mgr import DockerSandbox
from remediation_engine.tools.package_identity import normalize_python_package_name
from remediation_engine.tools.pypi_registry_tools import (
    fetch_pypi_registry_candidates,
    get_pypi_release_requires_dist,
    plan_python_parent_version,
)
from remediation_engine.tools.registry_tools import (
    fetch_registry_candidates,
    plan_npm_parent_version,
)

logger = logging.getLogger(__name__)

UPDATE_DISPATCH_LIMIT: int = 1
QA_DISPATCH_LIMIT: int = 1
_SCA_STAGE_ORDER: dict[SCARemediationStage, int] = {
    SCARemediationStage.OSV_MINIMUM: 0,
    SCARemediationStage.NPM_SAME_MAJOR: 1,
    SCARemediationStage.PYPI_SAME_MAJOR: 1,
    SCARemediationStage.NPM_LATEST: 2,
    SCARemediationStage.PYPI_LATEST: 2,
    SCARemediationStage.PACKAGE_OVERRIDE: 3,
    SCARemediationStage.CODE_WORKAROUND: 4,
}
_OVERRIDE_DEPENDENCY_TYPES = frozenset({"overrides", "resolutions", "pnpm_overrides"})
_MAX_REGISTRY_CANDIDATES = 3


def _group_ecosystem(group: VulnerabilityGroup | None) -> str:
    """Resolve the vulnerable package ecosystem from issue and PURL evidence."""
    if group is None:
        return "npm"
    issues = [*group.issues, *(localized.issue for localized in group.localized_issues)]
    for issue in issues:
        ecosystem = str(issue.ecosystem or "").strip().casefold()
        if ecosystem in {"pypi", "python"}:
            return "pypi"
        if ecosystem == "npm":
            return "npm"
        purl = str(issue.purl or "").strip().casefold()
        if purl.startswith("pkg:pypi/"):
            return "pypi"
        if purl.startswith("pkg:npm/"):
            return "npm"
    return "npm"


def _normalize_package_target(package_name: str, ecosystem: str) -> str:
    return normalize_python_package_name(package_name) if ecosystem == "pypi" else package_name


def _normalise_candidate_version(value: Any, ecosystem: str = "npm") -> str:
    """Normalize a candidate without changing npm's legacy ``v`` handling."""
    normalized = str(value).strip().lstrip("vV")
    if ecosystem != "pypi":
        return normalized
    try:
        return str(Version(normalized))
    except InvalidVersion:
        return ""


def _registry_version_key(value: str, ecosystem: str) -> tuple[int, object]:
    """Build the ecosystem key using the canonical source ordering primitive."""
    if ecosystem == "npm":
        parsed = SemVerVersion(value)
        semver_key = (parsed.major, parsed.minor, parsed.patch)
        return registry_version_key(value, ecosystem, semver_key)
    return registry_version_key(value, ecosystem)


def _candidate_version_key(value: Any, ecosystem: str = "npm") -> tuple[int, object]:
    """Return the cross-ecosystem policy key for an exact registry version."""
    normalized = _normalise_candidate_version(value, ecosystem)
    return _registry_version_key(normalized, ecosystem)


def _supervisor_fetch_registry_candidates(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str],
    ecosystem: str = "npm",
) -> list[RegistryCandidate]:
    """Fetch registry candidates as a Supervisor-owned traced operation."""
    ecosystem = "pypi" if str(ecosystem).casefold() in {"pypi", "python"} else "npm"
    normalized_name = _normalize_package_target(package_name, ecosystem)
    normalized_floor = _normalise_candidate_version(security_floor, ecosystem)
    normalized_attempted = {
        _normalise_candidate_version(version, ecosystem)
        for version in attempted_versions
        if _normalise_candidate_version(version, ecosystem)
    }
    inputs = {
        "package_name": normalized_name,
        "security_floor": normalized_floor,
        "attempted_versions": normalized_attempted,
        "ecosystem": ecosystem,
    }

    def fetch_candidates() -> list[RegistryCandidate]:
        if ecosystem == "pypi":
            return fetch_pypi_registry_candidates(
                normalized_name,
                normalized_floor,
                normalized_attempted,
            )
        return fetch_registry_candidates(
            normalized_name,
            normalized_floor,
            normalized_attempted,
        )

    return invoke_with_trajectory(
        "supervisor.fetch_registry_candidates",
        fetch_candidates,
        inputs,
        run_type="tool",
    )


def _select_report_candidate(
    values: Iterable[str],
    stage: SCARemediationStage,
    attempted_versions: set[str],
    ecosystem: str = "npm",
) -> str | None:
    """Select an approved report candidate without inventing a version."""
    attempted = {
        _normalise_candidate_version(version, ecosystem)
        for version in attempted_versions
        if _normalise_candidate_version(version, ecosystem)
    }
    eligible = [
        normalized
        for value in values
        if (normalized := _normalise_candidate_version(value, ecosystem))
        and normalized not in attempted
    ]
    if not eligible:
        return None

    def version_key(version: str) -> tuple[int, object]:
        return _candidate_version_key(version, ecosystem)

    return (
        min(eligible, key=version_key)
        if stage == SCARemediationStage.OSV_MINIMUM
        else max(
            eligible,
            key=version_key,
        )
    )


def _supervisor_plan_parent_version(inputs: dict[str, Any], ecosystem: str) -> str:
    """Run the ecosystem-matched read-only parent planner under a tool span."""
    if ecosystem == "pypi":
        return invoke_with_trajectory(
            "supervisor.plan_python_parent_version",
            lambda: plan_python_parent_version.invoke(inputs),
            inputs,
            run_type="tool",
        )
    return _supervisor_plan_npm_parent_version(inputs)


def _registry_report_version(value: str, ecosystem: str = "npm") -> str:
    normalized = _normalise_candidate_version(value, ecosystem)
    if not normalized:
        return ""
    if ecosystem == "pypi":
        parsed = Version(normalized)
        if parsed.is_prerelease or parsed.is_devrelease:
            return ""
    _registry_version_key(normalized, ecosystem)
    return normalized


def _registry_report_versions(
    report: str,
    label: str,
    ecosystem: str = "npm",
) -> list[str]:
    """Extract and canonicalize stable versions from a planner report."""
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(.*)$",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match:
        return []
    versions: list[str] = []
    for raw in match.group(1).split(","):
        raw_version = raw.strip()
        if not raw_version or raw_version.upper() == "NONE":
            continue
        value = _registry_report_version(raw_version, ecosystem)
        if value and value not in versions:
            versions.append(value)
    if ecosystem == "npm":
        return versions
    return sorted(versions, key=lambda version: _candidate_version_key(version, ecosystem))


def _approved_candidate_pool(
    diagnostics: UpdateRetryDiagnostics,
    target_package_name: str | None,
    ecosystem: str = "npm",
) -> tuple[str, ...]:
    """Return the previously committed candidate pool for one target."""
    normalized_target = (
        _normalize_package_target(target_package_name, ecosystem) if target_package_name else None
    )
    diagnostic_target = diagnostics.target_package_name
    if ecosystem == "pypi" and diagnostic_target:
        diagnostic_target = _normalize_package_target(diagnostic_target, ecosystem)
    if normalized_target and diagnostic_target and diagnostic_target != normalized_target:
        return ()
    return tuple(
        dict.fromkeys(
            normalized
            for version in diagnostics.candidate_versions_considered
            if (normalized := _normalise_candidate_version(version, ecosystem))
        )
    )


def _fetch_registry_candidates_for_task(
    package_name: str,
    security_floor: str,
    attempted_versions: set[str],
    diagnostics: UpdateRetryDiagnostics,
    ecosystem: str = "npm",
) -> list[RegistryCandidate]:
    """Revalidate only the task's committed candidate pool on retries."""
    package_name = _normalize_package_target(package_name, ecosystem)
    attempted = {
        normalized
        for version in attempted_versions
        if (normalized := _normalise_candidate_version(version, ecosystem))
    }
    approved_pool = _approved_candidate_pool(diagnostics, package_name, ecosystem)
    candidates = _supervisor_fetch_registry_candidates(
        package_name,
        security_floor,
        set() if approved_pool else attempted,
        ecosystem,
    )
    if not approved_pool:
        return candidates
    approved = set(approved_pool)
    return [
        candidate
        for candidate in candidates
        if _normalise_candidate_version(candidate.version, ecosystem) in approved
    ]


def _attempted_versions_for_target(
    diagnostics: UpdateRetryDiagnostics,
    target_package_name: str | None,
    ecosystem: str,
) -> set[str]:
    """Canonicalize attempts while preferring evidence scoped to this target."""
    normalized_target = (
        _normalize_package_target(target_package_name, ecosystem) if target_package_name else None
    )
    scoped: list[str] = []
    for package_name, versions in diagnostics.attempted_versions_by_target.items():
        normalized_name = _normalize_package_target(package_name, ecosystem)
        if normalized_target and normalized_name == normalized_target:
            scoped.extend(versions)
    values = scoped or diagnostics.attempted_versions
    return {
        normalized
        for version in values
        if (normalized := _normalise_candidate_version(version, ecosystem))
    }


def _pipfile_data(workspace_volume: str) -> dict[str, Any] | None:
    """Read and parse the authorized root Pipfile from the shared volume."""
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            content = sandbox.read_file("Pipfile")
        parsed = tomllib.loads(content or "")
    except Exception as exc:  # noqa: BLE001 - parent proof must fail closed
        logger.debug("Unable to read or parse Pipfile for parent proof: %s", exc)
        return None
    return parsed if isinstance(parsed, dict) else None


def _pipfile_declaration_is_unconditional(raw_name: str, value: Any) -> bool:
    """Accept only unconditional index-style Pipfile declarations."""
    try:
        parsed_name = Requirement(raw_name)
    except InvalidRequirement:
        return False
    if parsed_name.extras or parsed_name.marker:
        return False
    if isinstance(value, str):
        version_specifier = value.strip()
    elif isinstance(value, dict):
        if set(value) - {"version"}:
            return False
        version_specifier = value.get("version", "*")
        if not isinstance(version_specifier, str):
            return False
        version_specifier = version_specifier.strip()
    else:
        return False
    if version_specifier == "*":
        return True
    try:
        requirement = Requirement(f"{parsed_name.name}{version_specifier}")
    except InvalidRequirement:
        return False
    return not requirement.extras and requirement.marker is None


def _pipfile_parent_declaration_type(
    workspace_volume: str,
    parent_package_name: str,
) -> str | None:
    """Return the unique Pipfile category containing a directly declared parent."""
    pipfile = _pipfile_data(workspace_volume)
    if pipfile is None:
        return None
    normalized_parent = normalize_python_package_name(parent_package_name)
    matches: list[str] = []
    for category, declaration_type in (("packages", "packages"), ("dev-packages", "dev-packages")):
        entries = pipfile.get(category)
        if entries is None:
            continue
        if not isinstance(entries, dict):
            return None
        for name, value in entries.items():
            if normalize_python_package_name(str(name)) != normalized_parent:
                continue
            if not _pipfile_declaration_is_unconditional(str(name), value):
                return None
            matches.append(declaration_type)
    return matches[0] if len(matches) == 1 else None


def _locked_python_version(category: Any, normalized_name: str) -> str | None:
    if not isinstance(category, dict):
        return None
    matches = [
        value
        for name, value in category.items()
        if normalize_python_package_name(str(name)) == normalized_name
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        return None
    raw_version = matches[0].get("version")
    if not isinstance(raw_version, str):
        return None
    exact = raw_version.strip()
    if exact.startswith("=="):
        exact = exact[2:]
    elif exact.startswith("="):
        exact = exact[1:]
    try:
        parsed = Version(exact)
    except InvalidVersion:
        return None
    if parsed.is_prerelease or parsed.is_devrelease:
        return None
    return str(parsed)


def _resolve_pipfile_parent_context(
    workspace_volume: str,
    group: VulnerabilityGroup,
) -> tuple[str, str, list[str]] | None:
    """Prove one unconditional, category-matched Pipfile.lock parent edge."""
    child_name = group.vulnerable_component
    if not child_name:
        return None
    normalized_child = normalize_python_package_name(child_name)
    ancestry = [normalize_python_package_name(str(name)) for name in group.dependency_ancestry]
    if len(ancestry) > 2 or (ancestry and ancestry[-1] != normalized_child):
        return None
    try:
        with DockerSandbox(repo_root=None, workspace_volume=workspace_volume) as sandbox:
            pipfile_content = sandbox.read_file("Pipfile")
            lock_content = sandbox.read_file("Pipfile.lock")
        pipfile = tomllib.loads(pipfile_content or "")
        lockfile = json.loads(lock_content or "")
    except Exception as exc:  # noqa: BLE001 - malformed/missing evidence is not proof
        logger.debug("Unable to read Pipenv parent evidence: %s", exc)
        return None
    if not isinstance(pipfile, dict) or not isinstance(lockfile, dict):
        return None

    possible_edges: list[tuple[str, str, str, str, bool]] = []
    for pipfile_category, lock_category_name in (
        ("packages", "default"),
        ("dev-packages", "develop"),
    ):
        entries = pipfile.get(pipfile_category)
        if entries is None:
            continue
        if not isinstance(entries, dict):
            return None
        lock_category = lockfile.get(lock_category_name)
        locked_child_version = _locked_python_version(lock_category, normalized_child)
        if locked_child_version is None:
            # A child lock from the other Pipfile category cannot prove an edge
            # from this category's directly declared parent.
            continue
        for raw_name, declaration in entries.items():
            try:
                parsed_name = Requirement(str(raw_name))
            except InvalidRequirement:
                return None
            parent_name = normalize_python_package_name(parsed_name.name)
            parent_version = _locked_python_version(lock_category, parent_name)
            if parent_version is None:
                continue
            possible_edges.append(
                (
                    parent_name,
                    parent_version,
                    locked_child_version,
                    pipfile_category,
                    _pipfile_declaration_is_unconditional(str(raw_name), declaration),
                )
            )

    proven: list[tuple[str, str, str]] = []
    unproven_child_edge = False
    try:
        for (
            parent_name,
            parent_version,
            raw_child_version,
            category,
            unconditional_parent,
        ) in possible_edges:
            locked_child = Version(raw_child_version)
            requirements = get_pypi_release_requires_dist(parent_name, parent_version)
            for raw_requirement in requirements:
                requirement = Requirement(raw_requirement)
                if normalize_python_package_name(requirement.name) != normalized_child:
                    continue
                if not requirement.specifier.contains(locked_child, prereleases=False):
                    continue
                if requirement.marker is not None or requirement.extras or not unconditional_parent:
                    unproven_child_edge = True
                    continue
                proven.append((parent_name, parent_version, category))
    except (InvalidRequirement, InvalidVersion, TypeError, ValueError) as exc:
        logger.debug("Malformed Pipfile or PyPI Requires-Dist evidence: %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001 - registry failure is not ancestry proof
        logger.debug("Unable to query PyPI Requires-Dist evidence: %s", exc)
        return None
    if unproven_child_edge:
        return None
    unique_edges = list(dict.fromkeys(proven))
    if len(unique_edges) != 1:
        return None
    parent_name, parent_version, _category = unique_edges[0]
    if ancestry and len(ancestry) == 2 and ancestry[0] != parent_name:
        return None
    return parent_name, parent_version, [parent_name, normalized_child]


def _supervisor_plan_npm_parent_version(inputs: dict[str, Any]) -> str:
    """Run the npm parent planner under an explicit Supervisor tool span."""
    return invoke_with_trajectory(
        "supervisor.plan_npm_parent_version",
        lambda: plan_npm_parent_version.invoke(inputs),
        inputs,
        run_type="tool",
    )


def _commit_task_transition(*args: Any, **kwargs: Any) -> Any:
    from remediation_engine.orchestration import supervisor_node

    return supervisor_node._commit_task_transition(*args, **kwargs)


def _supervisor_dependency_type_candidates(
    strategy_stage: SCARemediationStage,
    target_dependency_type: str | None,
) -> list[str]:
    """Return dependency-type candidates permitted by the committed strategy."""
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
) -> str:
    ecosystem = _group_ecosystem(group)
    is_python = ecosystem == "pypi"
    component = group.vulnerable_component if group else task.parent_group_id
    parent_name, _, parent_type = (
        group_parent_context(group) if group is not None else (None, None, None)
    )
    parent_name = parent_name or task.parent_package_name
    target = task.target_package_name
    if not target and task.strategy_stage != SCARemediationStage.PACKAGE_OVERRIDE:
        target = parent_name
    target = target or component
    dependency_type = task.target_dependency_type or parent_type
    if task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        dependency_type = dependency_type or "overrides"
    transaction_tool = (
        "modify_and_validate_python_dependency"
        if is_python
        else "modify_and_validate_npm_dependency"
    )
    manifest = (
        "Pipfile"
        if is_python and group is not None and is_transitive_group(group)
        else (
            group.file_paths[0]
            if group and group.file_paths
            else ("Pipfile" if is_python else "package.json")
        )
    )
    category = evaluation.failure_category if evaluation else None
    if diagnostics and diagnostics.selected_version:
        is_override = not is_python and (
            task.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE
            or diagnostics.used_overrides
            or dependency_type in {"overrides", "resolutions", "pnpm_overrides"}
        )
        dependency_action = (
            f"package-manager override for {component}"
            if is_override
            else f"{target} dependency version"
        )
        target_clause = (
            f"edit only the {target} declaration" if target != component else f"update {target}"
        )
        if dependency_type:
            target_clause += f" in {dependency_type}"
        return (
            f"Apply the supervisor-selected {dependency_action} for {component}; "
            f"during strategy stage {task.strategy_stage.value}: "
            f"{target_clause} in {manifest} to exact version {diagnostics.selected_version}; "
            "do not edit any other dependency target; "
            f"use {transaction_tool} so synchronization runs immediately after the edit."
        )
    if task.strategy_stage == SCARemediationStage.OSV_MINIMUM and group and group.fix_plan:
        floor = group.fix_plan.fixed_version
        if floor:
            if parent_name and target == parent_name:
                return (
                    f"Apply strategy stage {task.strategy_stage.value} for transitive package {component}: "
                    f"update only directly declared parent {parent_name} in {manifest} to the "
                    "supervisor-selected compatible parent version; do not use a child override; "
                    f"use {transaction_tool} so synchronization runs immediately after the edit."
                )
            return (
                f"Apply strategy stage {task.strategy_stage.value} for {component}: "
                f"update {manifest} to exact OSV minimum fixed version {floor}; "
                f"use {transaction_tool} so synchronization runs immediately after the edit."
            )
    if diagnostics and task.strategy in {RoutingStrategy.VERSION_BUMP}:
        attempted = set(diagnostics.attempted_versions)
        candidates = [
            version
            for version in diagnostics.candidate_versions_considered
            if version not in attempted
        ]
        candidate = next(
            (
                version
                for version in [diagnostics.latest_version_seen, *candidates]
                if version and version not in attempted
            ),
            None,
        )
        if candidate:
            manifest = group.file_paths[0] if group and group.file_paths else "package.json"
            return (
                f"Apply strategy stage {task.strategy_stage.value} for {component}: "
                f"update only {target} in {manifest} to exact version {candidate}; "
                f"use {transaction_tool} so synchronization runs immediately after the edit."
            )
    if diagnostics and diagnostics.package_abandoned:
        return (
            f"The Supervisor found no remaining supported manifest-based update candidate for {component}. "
            "Do not select another version; report the bounded update failure so the Supervisor can "
            "pivot this task safely."
        )
    if category == FailureCategory.PEER_CONFLICT:
        return (
            f"Use only the Supervisor-approved peer-compatible candidate or override path for {component}. "
            "Preserve the committed package and manifest targets while retrying the combined transaction."
        )
    if category == FailureCategory.SECURITY_FLAG:
        return (
            f"Use only the Supervisor-approved patched manifest candidate for {component}. "
            "Do not query a registry or invent a newer release; retry the combined transaction with "
            "a different committed candidate when instructed."
        )
    if category == FailureCategory.BREAKING_CHANGE:
        return (
            f"Use only the Supervisor-approved compatible candidate for {component} and preserve the "
            "committed manifest target. Do not search for another version or change source code."
        )
    return (
        f"Execute the next Supervisor-approved manifest candidate for {component}. "
        "Use prior validation failures only to follow the committed retry instruction; do not query "
        "a registry or choose a candidate."
    )


def _registry_selected_version(report: str, ecosystem: str = "npm") -> str | None:
    """Extract a planner-selected stable version from a registry report."""
    match = re.search(
        r"^-\s*Selected(?: Version)?:\s*(\S+)",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match or match.group(1).upper() == "NONE":
        return None
    return _registry_report_version(match.group(1), ecosystem) or None


def _registry_report_value(
    report: str,
    label: str,
    ecosystem: str = "npm",
) -> str | None:
    """Extract one normalized value from a deterministic registry-tool report."""
    match = re.search(
        rf"^-\s*{re.escape(label)}:\s*(\S+)",
        report or "",
        re.IGNORECASE | re.MULTILINE,
    )
    if not match or match.group(1).upper() == "NONE":
        return None
    return _registry_report_version(match.group(1), ecosystem) or None


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
) -> list[str]:
    """Validate retry-plan semantics before a plan can mutate routing state."""
    violations: list[str] = []
    groups = group_by_id or {}
    for task_id, plan in plans.items():
        task = task_queue.get(task_id)
        if task is None:
            violations.append(f"task {task_id}: planner returned an unknown task")
            continue
        if task.status in _TERMINAL_STATUSES:
            violations.append(f"task {task_id}: planner returned a plan for terminal task")
        if plan.source_task_revision != task.task_revision:
            violations.append(
                f"task {task_id}: planner snapshot revision {plan.source_task_revision} "
                f"does not match current revision {task.task_revision}"
            )
        group = groups.get(task.parent_group_id)
        ecosystem = _group_ecosystem(group)
        attempted = {
            normalized
            for version in plan.attempted_versions
            if (normalized := _normalise_candidate_version(version, ecosystem))
        }
        diagnostics = diagnostics_by_task.get(task_id)
        if diagnostics is not None:
            attempted.update(
                normalized
                for version in diagnostics.attempted_versions
                if (normalized := _normalise_candidate_version(version, ecosystem))
            )
        selected = (
            _normalise_candidate_version(plan.selected_version, ecosystem)
            if plan.selected_version
            else ""
        )
        if selected and selected in attempted:
            violations.append(
                f"task {task_id}: selected version {plan.selected_version} was already attempted"
            )
        latest_stage = (
            SCARemediationStage.PYPI_LATEST
            if ecosystem == "pypi"
            else SCARemediationStage.NPM_LATEST
        )
        latest = (
            _normalise_candidate_version(plan.latest_version_seen, ecosystem)
            if plan.latest_version_seen
            else ""
        )
        if plan.strategy_stage == latest_stage and selected and latest and selected != latest:
            label = "pypi_latest" if ecosystem == "pypi" else "npm_latest"
            violations.append(
                f"task {task_id}: {label} selected {plan.selected_version}, "
                f"but registry latest is {plan.latest_version_seen}"
            )
        if ecosystem == "pypi" and plan.strategy_stage == SCARemediationStage.PACKAGE_OVERRIDE:
            violations.append(f"task {task_id}: PyPI tasks cannot use package_override")
        if plan.action == "retry_update" and selected == "":
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
            and _SCA_STAGE_ORDER.get(plan.strategy_stage, 99)
            < _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
        ):
            violations.append(
                f"task {task_id}: retry plan stage {plan.strategy_stage.value} regresses "
                f"from committed stage {task.strategy_stage.value}"
            )
        if plan.action == "pivot_workaround" and plan.strategy_stage != latest_stage:
            violations.append(
                f"task {task_id}: workaround pivot must be committed at {latest_stage.value}"
            )
        if plan.action == "pivot_workaround" and selected:
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
        diagnostics = repaired_diagnostics.get(task_id)
        task = task_queue[task_id]
        group = group_by_id.get(task.parent_group_id)
        ecosystem = _group_ecosystem(group)
        attempted = {
            normalized
            for version in plan.attempted_versions
            if (normalized := _normalise_candidate_version(version, ecosystem))
        }
        if diagnostics is not None:
            attempted.update(
                normalized
                for version in diagnostics.attempted_versions
                if (normalized := _normalise_candidate_version(version, ecosystem))
            )
        latest_stage = (
            SCARemediationStage.PYPI_LATEST
            if ecosystem == "pypi"
            else SCARemediationStage.NPM_LATEST
        )

        candidate = None
        plan_regresses = task.strategy == RoutingStrategy.VERSION_BUMP and _SCA_STAGE_ORDER.get(
            plan.strategy_stage, 99
        ) < _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
        if not plan_regresses and plan.strategy_stage != SCARemediationStage.CODE_WORKAROUND:
            repair_candidates = (
                [plan.latest_version_seen]
                if plan.strategy_stage == latest_stage
                else [plan.latest_version_seen, *plan.candidate_versions_considered]
            )
            for version in repair_candidates:
                normalized = _normalise_candidate_version(version, ecosystem) if version else ""
                if normalized and normalized not in attempted:
                    candidate = normalized
                    break

        if candidate:
            effective_stage = plan.strategy_stage
            same_major_stage = (
                SCARemediationStage.PYPI_SAME_MAJOR
                if ecosystem == "pypi"
                else SCARemediationStage.NPM_SAME_MAJOR
            )
            latest_version = (
                _normalise_candidate_version(plan.latest_version_seen, ecosystem)
                if plan.latest_version_seen
                else ""
            )
            if (
                effective_stage == same_major_stage
                and latest_version
                and candidate == latest_version
            ):
                effective_stage = latest_stage
            if diagnostics is None:
                diagnostics = UpdateRetryDiagnostics(task_id=task_id)
            target_package = task.target_package_name or (
                task.parent_package_name
                if ecosystem == "pypi"
                else (group_parent_context(group)[0] if group is not None else None)
            )
            if target_package:
                target_package = _normalize_package_target(target_package, ecosystem)
            target_type = task.target_dependency_type
            diagnostics = diagnostics.model_copy(
                update={
                    "strategy_stage": effective_stage,
                    "selected_version": candidate,
                    "candidate_versions_considered": list(
                        dict.fromkeys([*diagnostics.candidate_versions_considered, candidate])
                    ),
                    "registry_query_performed": True,
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

        if ecosystem != "pypi" and group is not None and is_transitive_group(group):
            # Parent registry exhaustion is the deterministic handoff to the
            # native child override stage, but the child version must be
            # verified independently. Never reuse the child's fix-plan floor
            # as if it were a registry candidate.
            child_floor, _floor_error = _canonical_security_floor(group)
            target_type = _override_dependency_type(group)
            child_candidate: str | None = None
            if child_floor and group.vulnerable_component:
                try:
                    child_candidates = _supervisor_fetch_registry_candidates(
                        group.vulnerable_component,
                        child_floor,
                        set(attempted),
                        ecosystem,
                    )
                    child_candidate = select_version(
                        child_candidates,
                        SCARemediationStage.OSV_MINIMUM,
                        set(attempted),
                    )
                except Exception:  # noqa: BLE001 - fall through to safe pivot
                    child_candidate = None
            if child_candidate is not None:
                if diagnostics is None:
                    diagnostics = UpdateRetryDiagnostics(task_id=task_id)
                diagnostics = diagnostics.model_copy(
                    update={
                        "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                        "security_floor": child_floor,
                        "selected_version": child_candidate,
                        "candidate_versions_considered": [child_candidate],
                        "registry_query_performed": True,
                        "target_package_name": group.vulnerable_component,
                        "target_dependency_type": target_type,
                        "exhausted_update_path": False,
                    }
                )
                repaired_diagnostics[task_id] = diagnostics
                override_task = task_queue[task_id].model_copy(
                    update={
                        "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                        "selected_version": child_candidate,
                        "target_package_name": group.vulnerable_component,
                        "target_dependency_type": target_type,
                    }
                )
                instruction = _build_high_level_retry_instruction(
                    override_task,
                    group,
                    None,
                    diagnostics,
                )
                repaired_plans[task_id] = plan.model_copy(
                    update={
                        "strategy_stage": SCARemediationStage.PACKAGE_OVERRIDE,
                        "selected_version": child_candidate,
                        "candidate_versions_considered": [child_candidate],
                        "exhausted_update_path": False,
                        "action": "retry_update",
                        "exact_instruction": instruction,
                        "target_package_name": group.vulnerable_component,
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
        group = group_by_id.get(task_queue[task_id].parent_group_id)
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


def _build_deterministic_retry_plan(
    task: RemediationTask,
    diagnostics: UpdateRetryDiagnostics,
    group: VulnerabilityGroup | None,
    *,
    requested_stage: SCARemediationStage | None = None,
    workspace_volume: str | None = None,
) -> SupervisorRetryPlan:
    """Build an exact retry plan from committed state and registry facts."""
    ecosystem = _group_ecosystem(group)
    is_python = ecosystem == "pypi"
    requested = requested_stage or task.strategy_stage
    if is_python and requested == SCARemediationStage.PACKAGE_OVERRIDE:
        requested = SCARemediationStage.PYPI_LATEST
    requested_order = _SCA_STAGE_ORDER.get(requested, 99)
    current_order = _SCA_STAGE_ORDER.get(task.strategy_stage, 0)
    effective_stage = requested if requested_order >= current_order else task.strategy_stage
    if is_python and effective_stage == SCARemediationStage.PACKAGE_OVERRIDE:
        effective_stage = SCARemediationStage.PYPI_LATEST

    attempted: set[str] = set()
    security_floor, floor_error = _canonical_security_floor(group)
    transitive = bool(group and is_transitive_group(group))
    candidate_versions: list[str] = []
    latest_version_seen: str | None = None
    selected_version: str | None = None
    failure_reason = floor_error or ""
    parent_minimum_version = task.parent_minimum_version
    target_package_name = task.target_package_name
    target_dependency_type = task.target_dependency_type
    parent_name: str | None = task.parent_package_name
    parent_version: str | None = task.parent_package_version

    if is_python and group is not None:
        if transitive:
            target_package_name = None
            target_dependency_type = None
            parent_name = None
            parent_version = None
            if workspace_volume:
                parent_context = _resolve_pipfile_parent_context(workspace_volume, group)
                if parent_context:
                    parent_name, parent_version, _ancestry = parent_context
                    parent_name = _normalize_package_target(parent_name, ecosystem)
                    parent_version = _normalise_candidate_version(parent_version, ecosystem)
                    target_package_name = parent_name
                    target_dependency_type = _pipfile_parent_declaration_type(
                        workspace_volume,
                        parent_name,
                    )
                    if target_dependency_type is None:
                        parent_name = None
                        parent_version = None
                        target_package_name = None
                        failure_reason = (
                            "The proven Pipfile parent has no unique editable declaration category."
                        )
                else:
                    failure_reason = "No unique active one-hop Pipfile.lock parent was proven."
            else:
                failure_reason = (
                    "The shared workspace volume is unavailable for Pipfile parent proof."
                )
        else:
            target_package_name = (
                _normalize_package_target(
                    group.vulnerable_component or target_package_name or "",
                    ecosystem,
                )
                or None
            )
    if transitive and not is_python and group is not None:
        parent_name, parent_version, parent_type = group_parent_context(group)
        target_package_name = target_package_name or parent_name
        target_dependency_type = target_dependency_type or parent_type
    if is_python:
        attempted = _attempted_versions_for_target(
            diagnostics,
            target_package_name,
            ecosystem,
        )
    else:
        attempted = {
            normalized
            for version in diagnostics.attempted_versions
            if (normalized := _normalise_candidate_version(version, ecosystem))
        }

    # An unproved Pipfile.lock child is never an editable target. Continue
    # through the bounded PyPI stages only to commit the final workaround.
    selected_stage_allowed = not (is_python and transitive and not parent_name)

    if effective_stage == SCARemediationStage.PACKAGE_OVERRIDE and not is_python:
        if group is not None:
            target_package_name = group.vulnerable_component
            target_dependency_type = _override_dependency_type(group)
            if security_floor and target_package_name:
                try:
                    child_candidates = _fetch_registry_candidates_for_task(
                        target_package_name,
                        security_floor,
                        attempted,
                        diagnostics,
                        ecosystem,
                    )
                    candidate_versions = [
                        candidate.version
                        for candidate in child_candidates[:_MAX_REGISTRY_CANDIDATES]
                    ]
                    selected_version = select_version(
                        child_candidates,
                        SCARemediationStage.OSV_MINIMUM,
                        attempted,
                    )
                    if selected_version is None:
                        failure_reason = "No verified vulnerable-child override candidate meets the security floor."
                except Exception as exc:  # noqa: BLE001
                    failure_reason = f"Deterministic child override verification failed: {exc}"
    elif (
        effective_stage != SCARemediationStage.CODE_WORKAROUND
        and security_floor
        and selected_stage_allowed
    ):
        stages = [effective_stage]
        if transitive:
            if not parent_name or not parent_version or not group or not group.vulnerable_component:
                failure_reason = (
                    failure_reason
                    or "Missing parent context for deterministic transitive planning."
                )
            else:
                parent_approved_pool = _approved_candidate_pool(
                    diagnostics,
                    parent_name,
                    ecosystem,
                )
                for stage in stages:
                    selection = {
                        SCARemediationStage.OSV_MINIMUM: "minimum",
                        SCARemediationStage.NPM_SAME_MAJOR: "same_major",
                        SCARemediationStage.PYPI_SAME_MAJOR: "same_major",
                        SCARemediationStage.NPM_LATEST: "latest",
                        SCARemediationStage.PYPI_LATEST: "latest",
                    }.get(stage)
                    if selection is None:
                        continue
                    parent_inputs = {
                        "parent_package_name": _normalize_package_target(parent_name, ecosystem),
                        "child_package_name": _normalize_package_target(
                            group.vulnerable_component,
                            ecosystem,
                        ),
                        "child_fixed_version": security_floor,
                        "installed_parent_version": _normalise_candidate_version(
                            parent_version,
                            ecosystem,
                        ),
                        "selection": selection,
                        "attempted_versions": (
                            "" if parent_approved_pool else ",".join(sorted(attempted))
                        ),
                        "dependency_ancestry": ",".join(
                            [
                                parent_name,
                                _normalize_package_target(group.vulnerable_component, ecosystem),
                            ]
                            if is_python
                            else group.dependency_ancestry
                        ),
                    }
                    try:
                        report = _supervisor_plan_parent_version(parent_inputs, ecosystem)
                    except Exception as exc:  # noqa: BLE001
                        failure_reason = f"Deterministic parent registry planning failed: {exc}"
                        continue
                    report_candidates = _registry_report_versions(
                        report,
                        "Eligible Candidates",
                        ecosystem,
                    ) or _registry_report_versions(
                        report,
                        "Compatible Parent Versions",
                        ecosystem,
                    )
                    if parent_approved_pool:
                        approved = set(parent_approved_pool)
                        report_candidates = [
                            version
                            for version in report_candidates
                            if version in approved and version not in attempted
                        ]
                    candidate_versions = list(
                        dict.fromkeys([*candidate_versions, *report_candidates])
                    )
                    latest_version_seen = (
                        _registry_report_value(
                            report,
                            "PyPI Latest" if is_python else "Npm Latest",
                            ecosystem,
                        )
                        or _registry_report_value(report, "Latest Compatible", ecosystem)
                        or _registry_report_value(report, "Latest Stable", ecosystem)
                        or latest_version_seen
                    )
                    selected = _registry_selected_version(report, ecosystem)
                    if selected not in set(report_candidates):
                        selected = _select_report_candidate(
                            report_candidates,
                            stage,
                            attempted,
                            ecosystem,
                        )
                    if selected:
                        selected_version = selected
                        effective_stage = stage
                        if stage == SCARemediationStage.OSV_MINIMUM:
                            parent_minimum_version = selected
                        break
        else:
            try:
                candidates = _fetch_registry_candidates_for_task(
                    target_package_name
                    or (
                        _normalize_package_target(group.vulnerable_component, ecosystem)
                        if group and group.vulnerable_component
                        else ""
                    ),
                    security_floor,
                    attempted,
                    diagnostics,
                    ecosystem,
                )
                candidate_versions = [
                    candidate.version for candidate in candidates[:_MAX_REGISTRY_CANDIDATES]
                ]
                latest_role = "pypi_latest" if is_python else "npm_latest"
                latest_version_seen = next(
                    (
                        candidate.version
                        for candidate in candidates
                        if latest_role in candidate.selection_roles
                    ),
                    candidates[-1].version if candidates else None,
                )
                selected_version = select_version(candidates, effective_stage, attempted)
            except Exception as exc:  # noqa: BLE001
                failure_reason = f"Deterministic registry planning failed: {exc}"
    elif effective_stage != SCARemediationStage.CODE_WORKAROUND and not failure_reason:
        failure_reason = "No security floor is available for deterministic version selection."

    latest_stage = SCARemediationStage.PYPI_LATEST if is_python else SCARemediationStage.NPM_LATEST
    if (
        selected_version is None
        and not is_python
        and transitive
        and security_floor
        and effective_stage == latest_stage
    ):
        effective_stage = SCARemediationStage.PACKAGE_OVERRIDE
        target_package_name = group.vulnerable_component if group else task.parent_group_id
        target_dependency_type = _override_dependency_type(group)
        try:
            child_candidates = _fetch_registry_candidates_for_task(
                target_package_name,
                security_floor,
                attempted,
                diagnostics,
                ecosystem,
            )
            candidate_versions = [
                candidate.version for candidate in child_candidates[:_MAX_REGISTRY_CANDIDATES]
            ]
            selected_version = select_version(
                child_candidates,
                SCARemediationStage.OSV_MINIMUM,
                attempted,
            )
            if selected_version is None:
                failure_reason = (
                    "Parent update stages are exhausted and no verified "
                    "vulnerable-child override candidate remains."
                )
        except Exception as exc:  # noqa: BLE001
            failure_reason = f"Child override verification failed: {exc}"
        if selected_version is None:
            effective_stage = SCARemediationStage.NPM_LATEST

    exhausted = selected_version is None and effective_stage == latest_stage
    effective_task = task.model_copy(
        update={
            "strategy_stage": effective_stage,
            "selected_version": selected_version,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "parent_package_name": parent_name,
            "parent_package_version": parent_version,
            "parent_minimum_version": parent_minimum_version,
        }
    )
    safe_candidate_versions = (
        candidate_versions[:_MAX_REGISTRY_CANDIDATES] if selected_version or exhausted else []
    )
    approved_pool = _approved_candidate_pool(diagnostics, target_package_name, ecosystem)
    provenance_candidate_versions = list(approved_pool or safe_candidate_versions)
    safe_latest_version = latest_version_seen if selected_version or exhausted else None
    candidate_dependency_types = _supervisor_dependency_type_candidates(
        effective_stage,
        target_dependency_type,
    )
    effective_diagnostics = diagnostics.model_copy(
        update={
            "strategy_stage": effective_stage,
            "candidate_versions_considered": provenance_candidate_versions,
            "selected_version": selected_version,
            "latest_version_seen": safe_latest_version,
            "registry_query_performed": bool(security_floor and (not transitive or parent_name)),
            "exhausted_update_path": exhausted,
            "target_package_name": target_package_name,
            "target_dependency_type": target_dependency_type,
            "candidate_dependency_types": candidate_dependency_types,
            "parent_package_name": parent_name,
            "parent_minimum_version": parent_minimum_version,
            "failure_reason": failure_reason,
        }
    )
    if exhausted:
        component = group.vulnerable_component if group else task.parent_group_id
        instruction = (
            f"Implement a code workaround or isolation strategy for {component} "
            "because the deterministic registry policy found no remaining eligible update version."
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
        attempted_versions=sorted(attempted),
        candidate_versions_considered=provenance_candidate_versions,
        candidate_dependency_types=candidate_dependency_types,
        latest_version_seen=safe_latest_version,
        exhausted_update_path=exhausted,
        package_abandoned=diagnostics.package_abandoned,
        target_package_name=target_package_name,
        target_dependency_type=target_dependency_type,
        parent_minimum_version=parent_minimum_version,
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
    CODE_WORKAROUND tasks do not use the npm version planner.
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
    advance_failed_stage: bool = False,
    target_task_ids: Iterable[str] | None = None,
    workspace_volume: str | None = None,
) -> tuple[dict[str, UpdateRetryDiagnostics], dict[str, SupervisorRetryPlan]]:
    """Plan retries from state and registry facts.

    ``advance_failed_stage`` is used only by the deterministic fallback after
    tactical reasoning is unavailable or rejected.  Tactical callers inspect
    the failed stage before this advancement occurs.
    ``target_task_ids`` optionally narrows planning to the task selected by
    deterministic routing, preventing registry work for tasks that are not
    active yet.
    """
    updated_diagnostics = dict(retry_diagnostics_by_task)
    plans: dict[str, SupervisorRetryPlan] = {}
    target_ids = set(target_task_ids) if target_task_ids is not None else None
    retry_tasks = sorted(
        (
            task
            for task in task_queue.values()
            if task.status == TaskStatus.NEEDS_RETRY
            and task.strategy == RoutingStrategy.VERSION_BUMP
            and (target_ids is None or task.task_id in target_ids)
            and task.strategy_stage
            in {
                SCARemediationStage.OSV_MINIMUM,
                SCARemediationStage.NPM_SAME_MAJOR,
                SCARemediationStage.NPM_LATEST,
                SCARemediationStage.PYPI_SAME_MAJOR,
                SCARemediationStage.PYPI_LATEST,
                SCARemediationStage.CODE_WORKAROUND,
            }
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
        requested_stage = None
        if advance_failed_stage:
            requested_stage = _next_sca_stage(
                task.strategy_stage,
                transitive=bool(group and is_transitive_group(group)),
                ecosystem=_group_ecosystem(group),
            )
        plan = _build_deterministic_retry_plan(
            task,
            diagnostics,
            group,
            requested_stage=requested_stage,
            workspace_volume=workspace_volume,
        )
        # An empty stage is deterministic evidence to advance to the next
        # bounded stage, not a request for the worker to inspect the registry.
        # Keep the worker execution-only: every update dispatch must end with
        # an exact unattempted version, or with the terminal update pivot.
        while plan.selected_version is None and not plan.exhausted_update_path:
            transitive = bool(group and is_transitive_group(group))
            next_stage = _next_sca_stage(
                plan.strategy_stage,
                transitive=transitive,
                ecosystem=_group_ecosystem(group),
            )
            if next_stage == SCARemediationStage.CODE_WORKAROUND:
                break
            if _SCA_STAGE_ORDER.get(next_stage, 99) <= _SCA_STAGE_ORDER.get(
                plan.strategy_stage, 99
            ):
                break
            plan = _build_deterministic_retry_plan(
                task,
                diagnostics,
                group,
                requested_stage=next_stage,
                workspace_volume=workspace_volume,
            )

        if plan.selected_version is None and not plan.exhausted_update_path:
            # This is a malformed or incomplete registry state (for example a
            # package-override stage without a security floor). Never preserve
            # it as retry_update: commit the same fail-closed latest-stage
            # pivot used by the deterministic guardrail.
            component = group.vulnerable_component if group else task.parent_group_id
            plan = plan.model_copy(
                update={
                    "strategy_stage": (
                        SCARemediationStage.PYPI_LATEST
                        if _group_ecosystem(group) == "pypi"
                        else SCARemediationStage.NPM_LATEST
                    ),
                    "selected_version": None,
                    "exhausted_update_path": True,
                    "action": "pivot_workaround",
                    "exact_instruction": (
                        f"Implement a code workaround or isolation strategy for {component} "
                        "because the manifest-based update path is exhausted."
                    ),
                }
            )
        plans[task.task_id] = plan
        updated_diagnostics[task.task_id] = diagnostics.model_copy(
            update={
                "strategy_stage": plan.strategy_stage,
                "security_floor": diagnostics.security_floor
                or (
                    group_by_id[task.parent_group_id].fix_plan.fixed_version
                    if task.parent_group_id in group_by_id
                    and group_by_id[task.parent_group_id].fix_plan is not None
                    else None
                ),
                "selected_version": plan.selected_version,
                "candidate_versions_considered": plan.candidate_versions_considered,
                "latest_version_seen": plan.latest_version_seen,
                "registry_query_performed": bool(
                    group_by_id.get(task.parent_group_id)
                    and group_by_id[task.parent_group_id].fix_plan
                    and group_by_id[task.parent_group_id].fix_plan.fixed_version
                ),
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
