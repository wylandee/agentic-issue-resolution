"""Deterministic source-edit tools for exact replacement and AST edits."""

from __future__ import annotations

from ._tool_support import (
    Any,
    DockerSandbox,
    Path,
    WorkaroundEdit,
    WorkaroundEditSet,
    WorkaroundExecutionPhase,
    WorkaroundPlannedReplacement,
    _detect_newline_style,
    _normalise_newlines,
    _restore_newlines,
    _validate_workspace_path,
    json,
    re,
    tool,
    uuid,
)
from .tools_manifest import _is_prohibited_target, _remember_stage_baseline
from .tools_validation import _make_validate_code_syntax_tool


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
                    suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
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
            for rel_path, orig in file_snapshots.items():
                sandbox.write_file(rel_path, orig)
            return (
                f"ERROR: [WRITE_FAILURE] Write failed on '{rel_path}': {exc}. All files restored."
            )

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
        for rel_path in new_file_contents:
            syntax_res = syntax_tool.invoke({"file_path": rel_path})
            if "FAILURE" in syntax_res or "ERROR" in syntax_res:
                for p, orig in file_snapshots.items():
                    sandbox.write_file(p, orig)
                if plan_state is not None:
                    plan_state["phase"] = WorkaroundExecutionPhase.INVESTIGATE.value
                    plan_state["local_investigation_complete"] = False
                    plan_state["edit_failure_requires_replan"] = True
                    plan_state["validation_passed"] = False

                return f"ERROR: [SYNTAX_FAILURE] Replacement produced invalid syntax in '{rel_path}'. All files in edit set restored.\n{syntax_res}"

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
        if suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}:
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

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
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

        content = sandbox.read_file(rel_path)
        if content is None:
            return f"ERROR: Could not read '{rel_path}'."
        if plan_state is not None:
            plan_state.setdefault("attempt_file_snapshots", {}).setdefault(rel_path, content)
        _remember_stage_baseline(sandbox, plan_state, [rel_path])

        try:
            from remediation_engine.tools.code_map import (
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

        syntax_tool = _make_validate_code_syntax_tool(sandbox)
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


__all__ = [name for name in globals() if not name.startswith("__")]
