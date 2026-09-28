"""Assembly helpers for the focused remediation-subagent tool categories.

Each builder returns the tool set for one worker contract. Implementations
remain in their owning modules; this facade only composes the update and
workaround toolbelts used by current Phase 5 dispatches.
"""

from typing import Literal

from remediation_engine.language import ProjectLanguage

# These imports are the intentional compatibility surface for callers that
# historically patched or imported tool factories from this facade.
from ._tool_support import (
    Any,
    DockerSandbox,
    Iterable,
    Mapping,
    NoFixMitigationStage,
    Path,
    Sequence,
    WorkaroundExecutionPhase,
    _detect_newline_style,
    _is_authoritative_evidence_source,  # noqa: F401
    _is_infrastructure_failure,  # noqa: F401
    _normalise_newlines,
    _normalize_manifest_targets,
    _normalize_package_manifest_targets,
    _restore_newlines,
    get_runtime_settings,  # noqa: F401
    requests,  # noqa: F401
)
from .tools_edit import (
    _apply_replacements_to_content,
    _make_deterministic_apply_edit_set_tool,
    _make_deterministic_replace_ast_symbol_tool,
    _make_deterministic_search_replace_tool,
)
from .tools_manifest import (
    _is_allowlisted_no_fix_package_file,
    _is_prohibited_target,
    _make_modify_and_validate_npm_dependency_tool,
    _make_remove_no_fix_dependency_tool,
    _normalize_python_manifest_targets,
    _package_checkpoint_paths,
    _PackageCheckpoint,
    rollback_pending_package_updates,
)
from .tools_manifest_python import (
    _make_modify_and_validate_python_dependency_tool,
    _make_remove_no_fix_python_dependency_tool,
    _normalize_python_package_manifest_targets,
)
from .tools_validation import (
    _make_record_plan_tool,
    _make_record_targeted_test_substitution_tool,
    _make_run_targeted_python_test_tool,
    _make_run_targeted_test_tool,
    _make_validate_code_syntax_tool,
    _make_validate_python_syntax_tool,
    _make_validate_python_workaround_tool,
    _make_validate_workaround_tool,
)
from .tools_web import (
    _make_read_web_page_tool,
    _make_search_web_tool,
)
from .tools_workspace import (
    _make_inspect_ast_symbol_tool,
    _make_read_repository_map_tool,
    _make_read_workspace_file_tool,
    _make_revert_workspace_file_tool,
    _make_search_codebase_pattern_tool,
)


def build_update_toolbelt(
    sandbox: DockerSandbox,
    touched_files: set[str],
    target_manifest_paths: Iterable[str],
    package_manifest_paths: Mapping[str, Iterable[str]],
    allowed_target_versions_by_package: Mapping[str, Iterable[str]] | None = None,
    override_required_packages: Iterable[str] | None = None,
    allowed_dependency_types_by_package: Mapping[str, Iterable[str]] | None = None,
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
    *,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
    package_ecosystem: Literal["npm", "pypi"] = "npm",
) -> list:
    """Build the strict, manager-matched update-only toolbelt."""
    if not (
        (language == ProjectLanguage.NODEJS and package_ecosystem == "npm")
        or (language == ProjectLanguage.PYTHON and package_ecosystem == "pypi")
    ):
        raise ValueError("Unsupported project-language/package-ecosystem pairing.")

    if language == ProjectLanguage.NODEJS:
        _normalize_manifest_targets(target_manifest_paths)
        normalized_package_manifest_paths = _normalize_package_manifest_targets(
            package_manifest_paths
        )
        if package_checkpoints is None:
            package_checkpoints = {}
        return [
            _make_modify_and_validate_npm_dependency_tool(
                sandbox,
                touched_files,
                normalized_package_manifest_paths,
                allowed_target_versions_by_package=allowed_target_versions_by_package,
                override_required_packages=override_required_packages,
                allowed_dependency_types_by_package=allowed_dependency_types_by_package,
                execution_state=execution_state,
                package_checkpoints=package_checkpoints,
            )
        ]

    allowed_manifests = set(_normalize_python_manifest_targets(target_manifest_paths))
    normalized_python_manifests = _normalize_python_package_manifest_targets(package_manifest_paths)
    outside_targets = sorted(
        {
            manifest
            for manifests in normalized_python_manifests.values()
            for manifest in manifests
            if manifest not in allowed_manifests
        }
    )
    if outside_targets:
        raise ValueError(
            "Python package manifest targets must be included in target_manifest_paths. "
            f"Unauthorized values: {outside_targets}"
        )
    if package_checkpoints is None:
        package_checkpoints = {}
    return [
        _make_modify_and_validate_python_dependency_tool(
            sandbox,
            touched_files,
            normalized_python_manifests,
            allowed_target_versions_by_package=allowed_target_versions_by_package,
            allowed_dependency_types_by_package=allowed_dependency_types_by_package,
            execution_state=execution_state,
            package_checkpoints=package_checkpoints,
        )
    ]


def build_workaround_toolbelt(
    sandbox: DockerSandbox,
    touched_files: set[str],
    host_repo_root: Path,
    plan_state: dict[str, Any] | None = None,
    mandatory_search_terms: dict[str, str] | None = None,
    preferred_test_files: Sequence[str] | None = None,
    no_fix_stage: NoFixMitigationStage | None = None,
    no_fix_package_name: str | None = None,
    no_fix_manifest_paths: Sequence[str] | None = None,
    no_fix_package_manager: str | None = None,
    *,
    language: ProjectLanguage = ProjectLanguage.NODEJS,
):
    """Build the strict workaround-only toolbelt.

    The optional NO_FIX arguments add one narrowly scoped package-removal
    capability. They never relax the source-edit or generic manifest guards.
    """
    if plan_state is None:
        plan_state = {}

    if plan_state.get("recorded") and not plan_state.get("phase"):
        plan_state["phase"] = WorkaroundExecutionPhase.EXECUTE.value
    else:
        plan_state.setdefault("phase", WorkaroundExecutionPhase.INVESTIGATE.value)

    plan_state.setdefault("iteration", 1)
    plan_state.setdefault("local_investigation_complete", False)
    plan_state.setdefault("plan_revision", 0)
    plan_state.setdefault("successful_edit_count", 0)
    plan_state.setdefault("successful_edit_count_this_iteration", 0)
    plan_state.setdefault("last_validation_result", None)
    plan_state.setdefault("last_infrastructure_diagnostics", None)
    plan_state.setdefault("infrastructure_failure_details", None)
    plan_state.setdefault("validation_calls", 0)
    plan_state.setdefault("validation_input_errors", 0)
    plan_state.setdefault("last_validation_input_error", None)
    plan_state.setdefault(
        "original_targeted_test",
        preferred_test_files[0] if preferred_test_files else None,
    )
    plan_state.setdefault("accepted_alternative_test", None)
    plan_state.setdefault("validated_files", [])
    plan_state.setdefault("inspected_symbols", set())
    plan_state.setdefault("inspected_files", set())
    plan_state.setdefault("fallback_files", set())
    plan_state.setdefault("read_files", set())
    plan_state.setdefault("edit_records", [])
    plan_state.setdefault("web_search_performed", False)
    plan_state.setdefault("recorded", False)
    # The real workaround toolbelt never permits a successful validation with
    # an inferred smoke target or a skipped QA-targeted test.
    plan_state.setdefault("runtime_smoke_required", True)
    plan_state.setdefault("targeted_test_required", True)

    effective_no_fix_stage = no_fix_stage or plan_state.get("no_fix_stage")
    if isinstance(effective_no_fix_stage, NoFixMitigationStage):
        effective_no_fix_stage = effective_no_fix_stage.value
    if effective_no_fix_stage:
        plan_state["no_fix_stage"] = str(effective_no_fix_stage)
    if no_fix_package_name:
        plan_state["no_fix_package_name"] = str(no_fix_package_name).strip()
        if effective_no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL.value:
            plan_state["targeted_test_required"] = bool(preferred_test_files)

    if language == ProjectLanguage.PYTHON:
        normalized_no_fix_manifests = _normalize_python_manifest_targets(
            no_fix_manifest_paths or []
        )
        python_lockfiles = {
            path
            for manifest in normalized_no_fix_manifests
            for path in _package_checkpoint_paths([manifest], package_ecosystem="pypi")
            if Path(path).name.casefold() == "pipfile.lock"
        }
        related_package_files = python_lockfiles
    else:
        normalized_no_fix_manifests = _normalize_manifest_targets(no_fix_manifest_paths or [])
        related_package_files = {
            path
            for manifest in normalized_no_fix_manifests
            for path in _package_checkpoint_paths([manifest])
            if Path(path).name in {"package-lock.json", "npm-shrinkwrap.json"}
        }
    if language == ProjectLanguage.PYTHON:
        plan_state["no_fix_manifest_paths"] = normalized_no_fix_manifests
        plan_state["no_fix_package_files"] = sorted(
            set(normalized_no_fix_manifests) | related_package_files
        )
    else:
        plan_state.setdefault("no_fix_manifest_paths", normalized_no_fix_manifests)
        plan_state.setdefault(
            "no_fix_package_files",
            sorted(set(normalized_no_fix_manifests) | related_package_files),
        )

    python_removal_manager = (
        "pipenv"
        if any(Path(path).name == "Pipfile" for path in normalized_no_fix_manifests)
        else "pip"
    )
    if language == ProjectLanguage.PYTHON:
        validation_tools = [
            _make_validate_python_syntax_tool(sandbox),
            _make_run_targeted_python_test_tool(
                sandbox,
                preferred_test_files,
                plan_state,
            ),
            _make_validate_python_workaround_tool(
                sandbox,
                touched_files,
                plan_state,
                preferred_test_files,
            ),
        ]
    else:
        validation_tools = [
            _make_validate_workaround_tool(
                sandbox,
                touched_files,
                plan_state,
                preferred_test_files,
                language=language,
            )
        ]

    toolbelt = [
        _make_record_plan_tool(plan_state, language=language),
        _make_record_targeted_test_substitution_tool(
            sandbox,
            plan_state,
            language=language,
        ),
        _make_search_web_tool(mandatory_search_terms=mandatory_search_terms, plan_state=plan_state),
        _make_read_web_page_tool(plan_state, language=language),
        _make_read_repository_map_tool(sandbox),
        _make_read_workspace_file_tool(sandbox, plan_state),
        _make_search_codebase_pattern_tool(sandbox, plan_state, language=language),
        _make_inspect_ast_symbol_tool(sandbox, plan_state),
        _make_deterministic_apply_edit_set_tool(sandbox, touched_files, plan_state),
        _make_revert_workspace_file_tool(sandbox, touched_files, host_repo_root),
        *validation_tools,
    ]
    if (
        effective_no_fix_stage == NoFixMitigationStage.PACKAGE_REMOVAL.value
        and no_fix_package_name
        and normalized_no_fix_manifests
        and (
            language == ProjectLanguage.NODEJS
            or (
                language == ProjectLanguage.PYTHON
                and (no_fix_package_manager or "").strip().casefold() == python_removal_manager
            )
        )
    ):
        # Keep these tools absent in every other language, manager, and stage.
        remover = (
            _make_remove_no_fix_python_dependency_tool
            if language == ProjectLanguage.PYTHON
            else _make_remove_no_fix_dependency_tool
        )
        toolbelt.insert(
            1,
            remover(
                sandbox,
                touched_files,
                plan_state,
                no_fix_package_name,
                normalized_no_fix_manifests,
                no_fix_package_manager or "",
            ),
        )
    return toolbelt


__all__ = [
    "build_update_toolbelt",
    "build_workaround_toolbelt",
    "_apply_replacements_to_content",
    "_make_deterministic_apply_edit_set_tool",
    "_make_deterministic_replace_ast_symbol_tool",
    "_make_deterministic_search_replace_tool",
    "_detect_newline_style",
    "_normalise_newlines",
    "_normalize_manifest_targets",
    "_normalize_package_manifest_targets",
    "_normalize_python_manifest_targets",
    "_restore_newlines",
    "_is_allowlisted_no_fix_package_file",
    "_is_prohibited_target",
    "_make_modify_and_validate_npm_dependency_tool",
    "_make_modify_and_validate_python_dependency_tool",
    "_make_remove_no_fix_dependency_tool",
    "_make_remove_no_fix_python_dependency_tool",
    "_package_checkpoint_paths",
    "_PackageCheckpoint",
    "rollback_pending_package_updates",
    "_make_record_plan_tool",
    "_make_record_targeted_test_substitution_tool",
    "_make_run_targeted_test_tool",
    "_make_run_targeted_python_test_tool",
    "_make_validate_python_syntax_tool",
    "_make_validate_python_workaround_tool",
    "_make_validate_code_syntax_tool",
    "_make_validate_workaround_tool",
    "_make_read_web_page_tool",
    "_make_search_web_tool",
    "_make_inspect_ast_symbol_tool",
    "_make_read_repository_map_tool",
    "_make_read_workspace_file_tool",
    "_make_revert_workspace_file_tool",
    "_make_search_codebase_pattern_tool",
]
