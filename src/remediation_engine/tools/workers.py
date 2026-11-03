"""Worker tool construction and deterministic source mutation tools."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from remediation_engine.contracts.schemas import (
    NoFixMitigationStage,
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundExecutionPhase,
    WorkaroundPlannedReplacement,
)
from remediation_engine.language import ProjectLanguage
from remediation_engine.runtime.sandbox import DockerSandbox
from remediation_engine.tools.maven_mutations import (
    _make_modify_and_validate_maven_dependency_tool,
    _make_modify_batch_maven_dependencies_tool,
)
from remediation_engine.tools.npm_mutations import (
    _is_allowlisted_no_fix_package_file,
    _is_prohibited_target,
    _make_modify_and_validate_npm_dependency_tool,
    _make_modify_batch_npm_dependencies_tool,
    _make_remove_no_fix_dependency_tool,
    _normalize_python_manifest_targets,
    _package_checkpoint_paths,
    _PackageCheckpoint,
    _remember_stage_baseline,
    apply_multi_package_action,
    rollback_pending_package_updates,
)
from remediation_engine.tools.python_mutations import (
    _make_modify_and_validate_python_dependency_tool,
    _make_modify_batch_python_dependencies_tool,
    _make_remove_no_fix_python_dependency_tool,
    _normalize_python_package_manifest_targets,
)
from remediation_engine.tools.validation import (
    _make_record_plan_tool,
    _make_record_targeted_test_substitution_tool,
    _make_run_targeted_python_test_tool,
    _make_run_targeted_test_tool,
    _make_validate_code_syntax_tool,
    _make_validate_python_syntax_tool,
    _make_validate_python_workaround_tool,
    _make_validate_workaround_tool,
)
from remediation_engine.tools.workspace import (
    _detect_newline_style,
    _make_inspect_ast_symbol_tool,
    _make_read_repository_map_tool,
    _make_read_web_page_tool,
    _make_read_workspace_file_tool,
    _make_revert_workspace_file_tool,
    _make_search_codebase_pattern_tool,
    _make_search_web_tool,
    _normalise_newlines,
    _normalize_manifest_targets,
    _normalize_package_manifest_targets,
    _restore_newlines,
    _validate_workspace_path,
)


def _restoration_failure_message(plan_state: dict[str, Any] | None) -> str | None:
    """Block further mutations after an incomplete local rollback."""
    if not plan_state or not plan_state.get("restoration_failed"):
        return None
    paths = ", ".join(plan_state.get("restoration_failure_paths", []) or [])
    return (
        "ERROR: [RESTORATION_INCOMPLETE] A previous rollback did not restore "
        f"the workspace files ({paths}); this worker cannot edit until the "
        "outer attempt snapshot restores the workspace."
    )


def _restore_source_snapshots(
    sandbox: DockerSandbox,
    snapshots: dict[str, str],
    touched_files: set[str],
    plan_state: dict[str, Any] | None,
) -> list[str]:
    """Restore local source snapshots and retain failed paths as candidates."""
    errors: list[str] = []
    for path, content in snapshots.items():
        try:
            sandbox.write_file(path, content)
        except Exception as exc:  # noqa: BLE001 - retain the primary edit failure
            touched_files.add(path)
            errors.append(f"{path}: {exc}")
    if errors and plan_state is not None:
        plan_state["restoration_failed"] = True
        plan_state["restoration_failure_paths"] = sorted(
            set(plan_state.get("restoration_failure_paths", []) or [])
            | {error.split(":", 1)[0] for error in errors}
        )
        plan_state["restoration_failure_diagnostics"] = errors
        plan_state["validation_passed"] = False
    return errors


def _apply_replacements_to_content(
    content: str,
    replacements: list[WorkaroundPlannedReplacement],
) -> tuple[str, str | None]:
    """Apply a list of non-overlapping planned replacements to file content."""
    newline_style = _detect_newline_style(content)
    norm_content = _normalise_newlines(content)

    all_intervals: list[tuple[int, int, str]] = []

    for r in replacements:
        old_norm = _normalise_newlines(r.old_text)
        new_norm = _normalise_newlines(r.new_text)

        if old_norm == new_norm:
            return (
                content,
                "ERROR: [NO_OP_EDIT] Replacement cannot be a no-op (old_text equals new_text).",
            )

        indices: list[int] = []
        start_idx = 0
        while True:
            idx = norm_content.find(old_norm, start_idx)
            if idx == -1:
                break
            indices.append(idx)
            start_idx = idx + max(1, len(old_norm))

        if len(indices) != r.expected_occurrences:
            return (
                content,
                f"ERROR: [OCCURRENCE_MISMATCH] Expected {r.expected_occurrences} occurrence(s) of old_text, but found {len(indices)}.",
            )

        for idx in indices:
            all_intervals.append((idx, idx + len(old_norm), new_norm))

    sorted_intervals = sorted(all_intervals, key=lambda x: x[0])
    for i in range(1, len(sorted_intervals)):
        prev_start, prev_end, _ = sorted_intervals[i - 1]
        curr_start, curr_end, _ = sorted_intervals[i]
        if curr_start < prev_end:
            return (
                content,
                "ERROR: [OVERLAPPING_REPLACEMENTS] Overlapping replacement spans detected.",
            )

    cur = norm_content
    for start, end, replacement_text in sorted(all_intervals, key=lambda x: x[0], reverse=True):
        cur = cur[:start] + replacement_text + cur[end:]

    updated_content = _restore_newlines(cur, newline_style)
    return (updated_content, None)


def _make_deterministic_apply_edit_set_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_apply_edit_set(
        replacements: list[WorkaroundPlannedReplacement],
    ) -> str:
        """Apply one atomic edit set containing exact planned replacements.

        The edit set is written immediately but remains pending until
        ``validate_workaround`` returns ``PASS``. A code-failing validation
        restores the pre-iteration snapshot, so a revised plan must include
        every required change from the failed set again. Infrastructure or
        blocked validation results retain the pending set for recovery; a
        passing validation promotes it to the validated cumulative patch.

        Each item in ``replacements`` must be a flat replacement object. Do not
        nest replacements under a file-level ``path`` or ``replacements`` key.

        Example payload::

            {
                "replacements": [
                    {
                        "file_path": "src/auth.ts",
                        "old_text": "const oldName = 1",
                        "new_text": "const newName = 1",
                        "expected_occurrences": 1
                    }
                ]
            }
        """
        restoration_error = _restoration_failure_message(plan_state)
        if restoration_error:
            return restoration_error
        if plan_state is not None:
            if not plan_state.get("recorded", False) or "planned_replacements" not in plan_state:
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits with deterministic_apply_edit_set."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit set is permitted per iteration before validation. Call validate_workaround now."

            if plan_state.get("require_authoritative_evidence") and not plan_state.get(
                "has_authoritative_evidence"
            ):
                return (
                    "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                    "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                    "or the installed package's README/types in node_modules."
                )

        raw_list = replacements
        if isinstance(raw_list, str):
            try:
                raw_list = json.loads(raw_list)
            except Exception:
                return (
                    "ERROR: [INVALID_REPLACEMENTS] replacements must be a valid JSON array or list."
                )

        if not isinstance(raw_list, list) or not raw_list:
            return "ERROR: [INVALID_REPLACEMENTS] replacements must be a non-empty list."

        submitted_planned: list[WorkaroundPlannedReplacement] = []
        for idx, item in enumerate(raw_list):
            try:
                if isinstance(item, WorkaroundPlannedReplacement):
                    r_obj = item
                elif isinstance(item, dict):
                    r_obj = WorkaroundPlannedReplacement(**item)
                else:
                    return f"ERROR: [INVALID_REPLACEMENTS] Item at index {idx} is invalid."
                submitted_planned.append(r_obj)
            except Exception as exc:
                return f"ERROR: [INVALID_REPLACEMENTS] Item at index {idx} failed validation: {exc}"

        recorded_data = plan_state.get("planned_replacements", []) if plan_state else []
        recorded_planned = [WorkaroundPlannedReplacement(**d) for d in recorded_data]

        if [r.model_dump() for r in submitted_planned] != [
            r.model_dump() for r in recorded_planned
        ]:
            return "ERROR: [PLAN_MISMATCH] Submitted replacements do not match the recorded plan's planned_replacements exactly."

        affected_files_set: set[str] = set()
        for r in submitted_planned:
            try:
                rel_path = _validate_workspace_path(r.file_path)
            except ValueError as exc:
                return f"ERROR: {exc}"
            if _is_prohibited_target(rel_path):
                return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."
            affected_files_set.add(rel_path)

        if len(submitted_planned) > 16:
            return f"ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 16 replacements per edit set (got {len(submitted_planned)})."
        if len(affected_files_set) > 8:
            return f"ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 8 affected files per edit set (got {len(affected_files_set)})."
        total_text_bytes = sum(len(r.new_text.encode("utf-8")) for r in submitted_planned)
        if total_text_bytes > 65536:
            return (
                "ERROR: [PLAN_LIMIT_EXCEEDED] Maximum 64 KiB of combined replacement text allowed."
            )

        if plan_state is not None:
            inspected = (
                plan_state.get("inspected_files", set())
                | plan_state.get("read_files", set())
                | plan_state.get("fallback_files", set())
            )
            for rel_path in affected_files_set:
                suffix = Path(rel_path).suffix.lower()
                if (
                    suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".py"}
                    and rel_path not in inspected
                ):
                    return (
                        f"ERROR: [MISSING_INSPECTION] AST inspection required before edit on '{rel_path}'. "
                        "Use inspect_ast_symbol or read_workspace_file first."
                    )

        file_snapshots: dict[str, str] = {}
        for rel_path in affected_files_set:
            curr = sandbox.read_file(rel_path)
            if curr is None:
                return f"ERROR: Could not read '{rel_path}'."
            file_snapshots[rel_path] = curr
        if plan_state is not None:
            attempt_snapshots = plan_state.setdefault("attempt_file_snapshots", {})
            for rel_path, content in file_snapshots.items():
                attempt_snapshots.setdefault(rel_path, content)
        _remember_stage_baseline(sandbox, plan_state, affected_files_set)

        replacements_by_file: dict[str, list[WorkaroundPlannedReplacement]] = {}
        for r in submitted_planned:
            norm_p = _validate_workspace_path(r.file_path)
            replacements_by_file.setdefault(norm_p, []).append(r)

        new_file_contents: dict[str, str] = {}
        for rel_path, file_repls in replacements_by_file.items():
            curr = file_snapshots[rel_path]
            updated, err = _apply_replacements_to_content(curr, file_repls)
            if err:
                return f"ERROR: Failed applying replacements to '{rel_path}': {err}"
            new_file_contents[rel_path] = updated

        written_files: list[str] = []
        try:
            for rel_path, new_content in new_file_contents.items():
                sandbox.write_file(rel_path, new_content)
                written_files.append(rel_path)
        except Exception as exc:
            restore_errors = _restore_source_snapshots(
                sandbox, file_snapshots, touched_files, plan_state
            )
            restoration_note = (
                " All files restored."
                if not restore_errors
                else " RESTORATION_INCOMPLETE: " + " | ".join(restore_errors)
            )
            return f"ERROR: [WRITE_FAILURE] Write failed on '{rel_path}': {exc}.{restoration_note}"

        for rel_path in new_file_contents:
            language = (
                ProjectLanguage.PYTHON
                if Path(rel_path).suffix.lower() == ".py"
                else ProjectLanguage.NODEJS
            )
            syntax_tool = _make_validate_code_syntax_tool(sandbox, language=language)
            syntax_res = syntax_tool.invoke({"file_path": rel_path})
            if "FAILURE" in syntax_res or "ERROR" in syntax_res:
                restore_errors = _restore_source_snapshots(
                    sandbox, file_snapshots, touched_files, plan_state
                )
                if plan_state is not None:
                    plan_state["phase"] = WorkaroundExecutionPhase.INVESTIGATE.value
                    plan_state["local_investigation_complete"] = False
                    plan_state["edit_failure_requires_replan"] = True
                    plan_state["validation_passed"] = False

                restoration_note = (
                    " All files in the edit set were restored."
                    if not restore_errors
                    else " RESTORATION_INCOMPLETE: " + " | ".join(restore_errors)
                )
                return (
                    f"ERROR: [SYNTAX_FAILURE] Replacement produced invalid syntax in '{rel_path}'."
                    f"{restoration_note}\n{syntax_res}"
                )

        plan_rev = plan_state.get("plan_revision", 1) if plan_state else 1
        iteration = plan_state.get("iteration", 1) if plan_state else 1
        patch_id = f"patch_r{plan_rev}_i{iteration}_{uuid.uuid4().hex[:8]}"

        edits: list[WorkaroundEdit] = []
        for idx, r in enumerate(submitted_planned):
            norm_p = _validate_workspace_path(r.file_path)
            edits.append(
                WorkaroundEdit(
                    file_path=norm_p,
                    old_text=r.old_text,
                    new_text=r.new_text,
                    symbol_name=r.symbol_name,
                    patch_id=patch_id,
                    replacement_index=idx,
                    expected_occurrences=r.expected_occurrences,
                    edit_index=idx,
                )
            )

        edit_set = WorkaroundEditSet(
            patch_id=patch_id,
            plan_revision=plan_rev,
            iteration=iteration,
            affected_files=sorted(list(new_file_contents.keys())),
            replacements=edits,
        )

        touched_files.update(new_file_contents.keys())
        if plan_state is not None:
            plan_state["pending_edit_set"] = edit_set
            plan_state["pending_snapshots"] = file_snapshots
            plan_state["current_iteration_edit"] = edits[0] if edits else None
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value

        res_dict = {
            "status": "SUCCESS",
            "patch_id": patch_id,
            "affected_files": sorted(list(new_file_contents.keys())),
            "replacement_count": len(edits),
            "phase_transition": "EXECUTE -> VALIDATE",
        }
        return (
            f"SUCCESS: Atomic edit set '{patch_id}' applied successfully and is pending validation. "
            "A CODE_FAILURE will restore the pre-iteration checkpoint; a PASS will commit this edit set.\n"
            f"JSON: {json.dumps(res_dict)}"
        )

    return deterministic_apply_edit_set


def _make_deterministic_search_replace_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_search_replace(
        file_path: str,
        old_text: str,
        new_text: str,
        symbol_name: str | None = None,
    ) -> str:
        """
        Apply an exact one-time search/replace to a workspace file.
        """
        restoration_error = _restoration_failure_message(plan_state)
        if restoration_error:
            return restoration_error
        if plan_state is not None:
            if not plan_state.get("recorded", False):
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits with deterministic_search_replace."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit is permitted per iteration before validation. Call validate_workaround now."

        if (
            plan_state is not None
            and plan_state.get("require_authoritative_evidence")
            and not plan_state.get("has_authoritative_evidence")
        ):
            return (
                "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                "or the installed package's README/types in node_modules."
            )

        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        if _is_prohibited_target(rel_path):
            return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

        planned_files = {
            str(path).replace("\\", "/").lstrip("/")
            for path in (plan_state or {}).get("planned_files", [])
        }
        if (
            plan_state is not None
            and plan_state.get("recorded")
            and planned_files
            and rel_path not in planned_files
        ):
            return (
                f"ERROR: [PLAN_VIOLATION] '{rel_path}' is outside the recorded workaround plan. "
                "Re-run record_plan with every causally related source file before editing it; "
                "do not apply an isolated or unrelated fix."
            )

        suffix = Path(rel_path).suffix.lower()
        if suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs", ".py"}:
            inspected_files = plan_state.get("inspected_files", set()) if plan_state else set()
            fallback_files = plan_state.get("fallback_files", set()) if plan_state else set()
            if rel_path not in inspected_files and rel_path not in fallback_files:
                return (
                    f"ERROR: [MISSING_INSPECTION] AST inspection required before first edit on '{rel_path}'. "
                    "Use inspect_ast_symbol for the target symbol, or use read_workspace_file "
                    "and document a no-symbol fallback in record_plan before editing."
                )

        current = sandbox.read_file(rel_path)
        if current is None:
            return (
                f"ERROR: Could not read '{rel_path}'. Use inspect_ast_symbol or "
                "search_codebase_pattern to verify the current file content."
            )
        if plan_state is not None:
            plan_state.setdefault("attempt_file_snapshots", {}).setdefault(rel_path, current)
        _remember_stage_baseline(sandbox, plan_state, [rel_path])

        newline_style = _detect_newline_style(current)
        current_norm = _normalise_newlines(current)
        old_norm = _normalise_newlines(old_text)
        new_norm = _normalise_newlines(new_text)

        count = current_norm.count(old_norm)
        if count == 0:
            return (
                "ERROR: old_text not found. Use inspect_ast_symbol or "
                "search_codebase_pattern to verify your anchor."
            )
        if count > 1:
            return "ERROR: old_text found multiple times. Make anchor more specific."

        updated = current_norm.replace(old_norm, new_norm, 1)

        sandbox.write_file(rel_path, _restore_newlines(updated, newline_style))

        language = (
            ProjectLanguage.PYTHON
            if Path(rel_path).suffix.lower() == ".py"
            else ProjectLanguage.NODEJS
        )
        syntax_tool = _make_validate_code_syntax_tool(sandbox, language=language)
        syntax_res = syntax_tool.invoke({"file_path": rel_path})
        if "FAILURE" in syntax_res or "ERROR" in syntax_res:
            sandbox.write_file(rel_path, current)
            return f"ERROR: Replacement produced invalid syntax in '{rel_path}'. Edit reverted.\n{syntax_res}"

        touched_files.add(rel_path)
        if plan_state is not None:
            edits = plan_state.setdefault("edit_records", [])
            edit_obj = WorkaroundEdit(
                file_path=rel_path,
                old_text=old_text,
                new_text=new_text,
                symbol_name=symbol_name,
                edit_index=len(edits),
            )
            edits.append(edit_obj)
            plan_state["current_iteration_edit"] = edit_obj
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value
        return f"SUCCESS: File modified: {rel_path}"

    return deterministic_search_replace


def _make_deterministic_replace_ast_symbol_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
):
    @tool
    def deterministic_replace_ast_symbol(
        file_path: str,
        symbol_name: str,
        replacement: str,
        line_hint: int = 0,
    ) -> str:
        """
        Replace the complete AST body of a declared function, class, or method.
        """
        restoration_error = _restoration_failure_message(plan_state)
        if restoration_error:
            return restoration_error
        if plan_state is not None:
            if not plan_state.get("recorded", False):
                return "ERROR: [PLAN_VIOLATION] You MUST call record_plan before making any code edits."
            phase = plan_state.get("phase")
            if not phase and plan_state.get("recorded"):
                phase = WorkaroundExecutionPhase.EXECUTE.value
            phase = phase or WorkaroundExecutionPhase.INVESTIGATE.value
            if phase != WorkaroundExecutionPhase.EXECUTE.value:
                return f"ERROR: [PHASE_VIOLATION] Code edits are only allowed in the EXECUTE phase (current phase: '{phase}'). Call record_plan first."
            if plan_state.get("successful_edit_count_this_iteration", 0) >= 1:
                return "ERROR: [ITERATION_LIMIT] Only one source edit is permitted per iteration before validation. Call validate_workaround now."

        if (
            plan_state is not None
            and plan_state.get("require_authoritative_evidence")
            and not plan_state.get("has_authoritative_evidence")
        ):
            return (
                "ERROR: [MISSING_EVIDENCE] Authoritative evidence required before editing for QA_REGRESSION_REPAIR. "
                "Gather evidence from an official advisory, package repo/docs, npm registry metadata, "
                "or the installed package's README/types in node_modules."
            )

        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        if _is_prohibited_target(rel_path):
            return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

        planned_files = {
            str(path).replace("\\", "/").lstrip("/")
            for path in (plan_state or {}).get("planned_files", [])
        }
        if (
            plan_state is not None
            and plan_state.get("recorded")
            and planned_files
            and rel_path not in planned_files
        ):
            return (
                f"ERROR: [PLAN_VIOLATION] '{rel_path}' is outside the recorded workaround plan. "
                "Re-run record_plan with every causally related source file before editing it; "
                "do not apply an isolated or unrelated fix."
            )

        if Path(rel_path).suffix.lower() == ".py":
            inspected_files = plan_state.get("inspected_files", set()) if plan_state else set()
            fallback_files = plan_state.get("fallback_files", set()) if plan_state else set()
            if rel_path not in inspected_files and rel_path not in fallback_files:
                return (
                    f"ERROR: [MISSING_INSPECTION] AST inspection required before first edit on '{rel_path}'. "
                    "Use inspect_ast_symbol for the target symbol, or use read_workspace_file "
                    "and document a no-symbol fallback in record_plan before editing."
                )
        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: Could not read '{rel_path}'."
        if plan_state is not None:
            plan_state.setdefault("attempt_file_snapshots", {}).setdefault(rel_path, content)
        _remember_stage_baseline(sandbox, plan_state, [rel_path])

        try:
            from remediation_engine.tools.source import (
                find_named_symbol,
                language_for_path,
                parse_source,
            )
        except ImportError:
            return "ERROR: code_map module is unavailable."

        lang = language_for_path(rel_path)
        if lang is None:
            return f"ERROR: No AST parser available for '{rel_path}'."

        source_bytes = content.encode("utf-8", errors="replace")
        tree = parse_source(source_bytes, lang)
        if tree is None:
            return "ERROR: tree-sitter unavailable."

        hint = int(line_hint) if line_hint else None
        try:
            res = find_named_symbol(tree.root_node, symbol_name, source_bytes, line_hint=hint)
        except (ValueError, RuntimeError) as exc:
            return f"ERROR: {exc}"

        if res is None:
            return (
                f"NOT FOUND: No declared function, class, or method named '{symbol_name}' found in '{rel_path}'. "
                "Imported identifiers and package names are not AST symbols. "
                "Use search_codebase_pattern to find the call site and inspect its "
                "enclosing declared symbol. For an arrow-function binding, pass the "
                "variable name only after confirming it is declared in this file; "
                "this tool accepts either the expression body or a complete enclosing declaration."
            )

        old_text = res["text"]

        # tree-sitter reports an arrow function's expression as the symbol,
        # but models often correctly reason in terms of the complete exported
        # declaration.  If the replacement is declaration-shaped, expand the
        # replacement scope to the enclosing export/lexical declaration.  This
        # avoids producing invalid text such as ``export const name = const
        # name = ...`` while retaining expression-only replacement semantics.
        replacement_stripped = replacement.strip()
        declaration_shaped = bool(
            re.match(
                r"^(?:export\s+(?:default\s+)?)?(?:const|let|var|function|class)\b",
                replacement_stripped,
            )
        )
        if res.get("node_type") == "arrow_function" and declaration_shaped:
            start_byte = int(res.get("start_byte", -1))
            end_byte = int(res.get("end_byte", -1))
            enclosing_nodes = []
            if start_byte >= 0 and end_byte >= 0:
                stack = [tree.root_node]
                while stack:
                    node = stack.pop()
                    stack.extend(getattr(node, "children", []) or [])
                    if (
                        node.start_byte <= start_byte
                        and node.end_byte >= end_byte
                        and node.type
                        in {
                            "variable_declarator",
                            "lexical_declaration",
                            "variable_declaration",
                            "export_statement",
                        }
                    ):
                        node_text = node.text
                        if isinstance(node_text, bytes):
                            node_text = node_text.decode("utf-8", errors="replace")
                        if symbol_name in str(node_text):
                            enclosing_nodes.append(node)
            if enclosing_nodes:
                replacement_scope = max(
                    enclosing_nodes,
                    key=lambda node: node.end_byte - node.start_byte,
                )
                raw_scope = replacement_scope.text
                old_text = (
                    raw_scope.decode("utf-8", errors="replace")
                    if isinstance(raw_scope, bytes)
                    else str(raw_scope)
                )
            else:
                return (
                    f"ERROR: Replacement for arrow symbol '{symbol_name}' is a complete declaration, "
                    "but its enclosing declaration could not be identified. "
                    "Retry with only the arrow/function expression body."
                )
        if old_text not in content:
            return f"ERROR: Symbol text for '{symbol_name}' could not be cleanly anchored in '{rel_path}'."

        count = content.count(old_text)
        if count > 1:
            return f"ERROR: Symbol text for '{symbol_name}' appears {count} times in '{rel_path}'. Provide a line_hint."

        updated = content.replace(old_text, replacement, 1)

        sandbox.write_file(rel_path, updated)

        language = (
            ProjectLanguage.PYTHON
            if Path(rel_path).suffix.lower() == ".py"
            else ProjectLanguage.NODEJS
        )
        syntax_tool = _make_validate_code_syntax_tool(sandbox, language=language)
        syntax_res = syntax_tool.invoke({"file_path": rel_path})
        if "FAILURE" in syntax_res or "ERROR" in syntax_res:
            sandbox.write_file(rel_path, content)
            return f"ERROR: Replacement produced invalid syntax in '{rel_path}'. Edit reverted.\n{syntax_res}"

        touched_files.add(rel_path)
        if plan_state is not None:
            plan_state.setdefault("inspected_files", set()).add(rel_path)
            edits = plan_state.setdefault("edit_records", [])
            edit_obj = WorkaroundEdit(
                file_path=rel_path,
                old_text=old_text,
                new_text=replacement,
                symbol_name=symbol_name,
                edit_index=len(edits),
            )
            edits.append(edit_obj)
            plan_state["current_iteration_edit"] = edit_obj
            plan_state["successful_edit_count"] = (
                int(plan_state.get("successful_edit_count", 0)) + 1
            )
            plan_state["successful_edit_count_this_iteration"] = 1
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value
        return f"SUCCESS: Symbol '{symbol_name}' in '{rel_path}' successfully replaced."

    return deterministic_replace_ast_symbol


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
    project_language: ProjectLanguage | str = ProjectLanguage.NODEJS,
    maven_target_operations_by_package: Mapping[str, Iterable[str]] | None = None,
    *,
    language: ProjectLanguage | str | None = None,
    package_ecosystem: str | None = None,
) -> list:
    """Build the strict update-only toolbelt for the resolved run language."""
    if language is not None:
        project_language = language
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    expected_ecosystem = {
        ProjectLanguage.NODEJS: "npm",
        ProjectLanguage.JAVA: "maven",
        ProjectLanguage.PYTHON: "pypi",
    }[project_language]
    if package_ecosystem is not None and package_ecosystem.strip().casefold() != expected_ecosystem:
        raise ValueError("Unsupported project-language/package-ecosystem pairing.")

    if project_language == ProjectLanguage.PYTHON:
        allowed_manifests = set(_normalize_python_manifest_targets(target_manifest_paths))
        normalized_package_manifest_paths = _normalize_python_package_manifest_targets(
            package_manifest_paths
        )
        outside_targets = sorted(
            {
                path
                for paths in normalized_package_manifest_paths.values()
                for path in paths
                if path not in allowed_manifests
            }
        )
        if outside_targets:
            raise ValueError(
                "Python package manifest targets must be included in target_manifest_paths. "
                f"Unauthorized values: {outside_targets}"
            )
        return [
            _make_modify_and_validate_python_dependency_tool(
                sandbox,
                touched_files,
                normalized_package_manifest_paths,
                allowed_target_versions_by_package=allowed_target_versions_by_package,
                allowed_dependency_types_by_package=allowed_dependency_types_by_package,
                execution_state=execution_state,
                package_checkpoints=package_checkpoints,
            )
        ]

    allowed_manifest_names = (
        ("pom.xml",) if project_language == ProjectLanguage.JAVA else ("package.json",)
    )
    _normalize_manifest_targets(
        target_manifest_paths,
        allowed_manifest_names=allowed_manifest_names,
    )
    normalized_package_manifest_paths = _normalize_package_manifest_targets(
        package_manifest_paths,
        allowed_manifest_names=allowed_manifest_names,
    )
    if package_checkpoints is None:
        package_checkpoints = {}
    if project_language == ProjectLanguage.JAVA:
        return [
            _make_modify_and_validate_maven_dependency_tool(
                sandbox,
                touched_files,
                normalized_package_manifest_paths,
                allowed_target_versions_by_package or {},
                allowed_dependency_types_by_package or {},
                execution_state=execution_state,
                maven_target_operations_by_package=maven_target_operations_by_package or {},
            )
        ]
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


def build_multi_package_update_toolbelt(
    sandbox: DockerSandbox,
    touched_files: set[str],
    action: Any,
    execution_state: dict[str, Any] | None = None,
    package_checkpoints: dict[str, _PackageCheckpoint] | None = None,
    project_language: ProjectLanguage | str = ProjectLanguage.NODEJS,
) -> list:
    """Build the single committed-action toolbelt for a package cluster.

    The returned tool exposes no mutation arguments to the LLM. It closes over
    the already validated ``MultiPackageAction`` and delegates to the
    deterministic atomic executor when called.

    Args:
        sandbox: Running Docker workspace sandbox.
        touched_files: Mutable changed-file projection for the worker result.
        action: Supervisor-validated multi-package action.
        execution_state: Mutable per-worker execution state.
        package_checkpoints: Optional store retaining the successful union
            checkpoint until the worker result is accepted.

    Returns:
        A one-tool list for the bounded update-subagent loop.
    """
    if not isinstance(project_language, ProjectLanguage):
        project_language = ProjectLanguage(project_language)
    mutations = list(getattr(action, "package_mutations", ()) or ())
    if not mutations:
        raise ValueError("Multi-package actions must contain at least one mutation.")
    if project_language == ProjectLanguage.JAVA:
        if any(getattr(mutation, "ecosystem", "") != "maven" for mutation in mutations):
            raise ValueError("Java multi-package actions must contain only Maven mutations.")
        return [
            _make_modify_batch_maven_dependencies_tool(
                sandbox,
                action,
                touched_files,
                execution_state=execution_state,
                package_checkpoints=package_checkpoints,
            )
        ]
    expected_ecosystem = "pypi" if project_language == ProjectLanguage.PYTHON else "npm"
    if any(getattr(mutation, "ecosystem", "npm") != expected_ecosystem for mutation in mutations):
        raise ValueError(
            f"{project_language.value} batch toolbelt requires a non-empty "
            f"{expected_ecosystem} action."
        )
    if project_language == ProjectLanguage.PYTHON:
        return [
            _make_modify_batch_python_dependencies_tool(
                sandbox,
                action,
                touched_files,
                execution_state=execution_state,
                package_checkpoints=package_checkpoints,
            )
        ]
    return [
        _make_modify_batch_npm_dependencies_tool(
            sandbox,
            action,
            touched_files,
            execution_state=execution_state,
            package_checkpoints=package_checkpoints,
        )
    ]


def build_workaround_toolbelt(
    sandbox: DockerSandbox,
    touched_files: set[str],
    host_repo_root: Path,
    plan_state: dict[str, Any] | None = None,
    preferred_test_files: Sequence[str] | None = None,
    no_fix_stage: NoFixMitigationStage | None = None,
    no_fix_package_name: str | None = None,
    no_fix_manifest_paths: Sequence[str] | None = None,
    no_fix_package_manager: str | None = None,
    project_language: ProjectLanguage | str = ProjectLanguage.NODEJS,
    *,
    language: ProjectLanguage | str | None = None,
) -> list:
    """Build the strict workaround-only toolbelt.

    The optional NO_FIX arguments add one narrowly scoped package-removal
    capability. They never relax the source-edit or generic manifest guards.
    Web queries are supplied by the worker and are not silently augmented.
    """
    if plan_state is None:
        plan_state = {}
    language = project_language if language is None else language
    if not isinstance(language, ProjectLanguage):
        language = ProjectLanguage(language)
    project_language = language

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

    if no_fix_package_manager:
        package_manager = str(no_fix_package_manager).strip().casefold()
    elif language == ProjectLanguage.JAVA:
        package_manager = "maven"
    elif language == ProjectLanguage.PYTHON:
        package_manager = (
            "pipenv"
            if any(Path(path).name.casefold() == "pipfile" for path in no_fix_manifest_paths or [])
            else "pip"
        )
    else:
        package_manager = "npm"

    if package_manager == "maven" and language != ProjectLanguage.JAVA:
        raise ValueError("Maven package removal requires a Java project run.")
    if package_manager in {"pip", "pipenv"} and language != ProjectLanguage.PYTHON:
        raise ValueError("Python package removal requires a Python project run.")
    if language == ProjectLanguage.PYTHON and package_manager not in {"pip", "pipenv"}:
        raise ValueError("Python package removal requires pip or pipenv metadata.")
    if language == ProjectLanguage.JAVA and package_manager != "maven":
        raise ValueError("Java package removal requires the Maven package manager.")

    plan_state["no_fix_package_manager"] = package_manager
    if language == ProjectLanguage.PYTHON:
        normalized_no_fix_manifests = _normalize_python_manifest_targets(
            no_fix_manifest_paths or []
        )
        related_package_files = {
            path
            for manifest in normalized_no_fix_manifests
            for path in _package_checkpoint_paths([manifest], package_ecosystem="pypi")
            if Path(path).name.casefold() == "pipfile.lock"
        }
        plan_state["no_fix_manifest_paths"] = normalized_no_fix_manifests
        plan_state["no_fix_package_files"] = sorted(
            set(normalized_no_fix_manifests) | related_package_files
        )
    elif language == ProjectLanguage.JAVA:
        normalized_no_fix_manifests = _normalize_manifest_targets(
            no_fix_manifest_paths or [],
            allowed_manifest_names=("pom.xml",),
        )
        plan_state["no_fix_manifest_paths"] = normalized_no_fix_manifests
        plan_state["no_fix_package_files"] = normalized_no_fix_manifests.copy()
    elif language == ProjectLanguage.NODEJS:
        normalized_no_fix_manifests = _normalize_manifest_targets(
            no_fix_manifest_paths or [],
            allowed_manifest_names=("package.json",),
        )
        plan_state.setdefault("no_fix_manifest_paths", normalized_no_fix_manifests)
        package_files = set(normalized_no_fix_manifests)
        package_files.update(
            path
            for manifest in normalized_no_fix_manifests
            for path in _package_checkpoint_paths([manifest])
            if Path(path).name in {"package-lock.json", "npm-shrinkwrap.json"}
        )
        plan_state.setdefault("no_fix_package_files", sorted(package_files))
    else:
        normalized_no_fix_manifests = []
        plan_state["no_fix_manifest_paths"] = []
        plan_state["no_fix_package_files"] = []

    if language == ProjectLanguage.PYTHON:
        validation_tools = [
            _make_validate_python_syntax_tool(sandbox),
            _make_run_targeted_python_test_tool(sandbox, preferred_test_files, plan_state),
            _make_validate_python_workaround_tool(
                sandbox, touched_files, plan_state, preferred_test_files
            ),
        ]
    else:
        validation_tools = [
            _make_validate_workaround_tool(
                sandbox,
                touched_files,
                plan_state,
                preferred_test_files,
                project_language=language,
            )
        ]

    toolbelt = [
        _make_record_plan_tool(plan_state, language=language),
        _make_record_targeted_test_substitution_tool(sandbox, plan_state, language=language),
        _make_search_web_tool(plan_state=plan_state),
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
        and language
        in {
            ProjectLanguage.NODEJS,
            ProjectLanguage.JAVA,
            ProjectLanguage.PYTHON,
        }
    ):
        toolbelt.insert(
            1,
            (
                _make_remove_no_fix_python_dependency_tool
                if language == ProjectLanguage.PYTHON
                else _make_remove_no_fix_dependency_tool
            )(
                sandbox,
                touched_files,
                plan_state,
                no_fix_package_name,
                normalized_no_fix_manifests,
                package_manager,
            ),
        )
    return toolbelt


__all__ = [
    "build_update_toolbelt",
    "build_multi_package_update_toolbelt",
    "build_workaround_toolbelt",
    "apply_multi_package_action",
    "_apply_replacements_to_content",
    "_make_deterministic_apply_edit_set_tool",
    "_make_deterministic_replace_ast_symbol_tool",
    "_make_deterministic_search_replace_tool",
    "_detect_newline_style",
    "_normalise_newlines",
    "_normalize_manifest_targets",
    "_normalize_package_manifest_targets",
    "_restore_newlines",
    "_is_allowlisted_no_fix_package_file",
    "_is_prohibited_target",
    "_make_modify_and_validate_npm_dependency_tool",
    "_make_remove_no_fix_dependency_tool",
    "_package_checkpoint_paths",
    "_PackageCheckpoint",
    "rollback_pending_package_updates",
    "_make_record_plan_tool",
    "_make_record_targeted_test_substitution_tool",
    "_make_run_targeted_test_tool",
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
