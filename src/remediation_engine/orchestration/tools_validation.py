"""Targeted test, syntax, typecheck, and workaround validation tools."""

from __future__ import annotations

from ._tool_support import (
    _LINT_CHECK_TIMEOUT_SECONDS,
    _NPM_TEST_TIMEOUT_SECONDS,
    _RUNTIME_SMOKE_TIMEOUT_SECONDS,
    _SOURCE_MODULE_SUFFIXES,
    _SYNTAX_CHECK_TIMEOUT_SECONDS,
    Any,
    DockerSandbox,
    FailureCategory,
    NoFixMitigationStage,
    Path,
    Sequence,
    WorkaroundExecutionPhase,
    WorkaroundPlannedReplacement,
    WorkaroundValidationResult,
    WorkaroundValidationStatus,
    _is_authoritative_evidence_source,
    _is_infrastructure_failure,
    _is_test_file_path,
    _run_readonly,
    _runtime_smoke_path_error,
    _select_lightweight_runtime_smoke_target,
    _select_targeted_test_file,
    _validate_workspace_path,
    json,
    logger,
    re,
    shlex,
    tool,
)
from .tools_manifest import _is_allowlisted_no_fix_package_file, _is_prohibited_target


def _parse_mocha_json_output(stdout: str) -> dict[str, Any] | None:
    """Parse Mocha's JSON reporter output, tolerating leading tool output.

    Args:
        stdout: Captured stdout from a Mocha invocation.

    Returns:
        The decoded Mocha result object, or ``None`` when the output is not
        valid JSON reporter output.
    """
    text = str(stdout or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _mocha_entry_title(entry: Any) -> str:
    """Return the canonical-or-leaf title from one Mocha JSON entry."""
    if not isinstance(entry, dict):
        return ""
    return str(entry.get("fullTitle") or entry.get("title") or "").strip()


def _mocha_titles_match(left: str, right: str) -> bool:
    """Match full and leaf Mocha titles without accepting unrelated tests."""
    lhs = re.sub(r"\s+", " ", left or "").strip().casefold()
    rhs = re.sub(r"\s+", " ", right or "").strip().casefold()
    return bool(lhs and rhs and (lhs == rhs or lhs.endswith(f" {rhs}") or rhs.endswith(f" {lhs}")))


def _mocha_test_hint_matches_title(title: str, hint: str) -> bool:
    """Return whether a requested Mocha hint identifies a test or suite.

    Mocha's ``fullTitle`` includes parent suite names. Workers frequently
    provide only a suite name, such as ``b2bOrder``. That is a valid bounded
    target when it is an exact title component, but it is not an exact full
    test title. Prefix matching is therefore limited to a whitespace boundary
    and does not accept arbitrary substrings.
    """
    from remediation_engine.orchestration.qa_test_parsing import _mocha_test_name_variants

    lhs = re.sub(r"\s+", " ", title or "").strip().casefold()
    for variant in _mocha_test_name_variants(hint):
        rhs = re.sub(r"\s+", " ", variant or "").strip().casefold()
        if lhs and rhs and (lhs == rhs or lhs.startswith(f"{rhs} ") or lhs.endswith(f" {rhs}")):
            return True
    return False


def _select_mocha_tests(payload: dict[str, Any], test_name: str | None) -> list[dict[str, Any]]:
    """Select the tests represented by a requested Mocha test hint."""
    tests = [entry for entry in payload.get("tests", []) if isinstance(entry, dict)]
    if not test_name:
        return tests

    return [
        entry
        for entry in tests
        if _mocha_test_hint_matches_title(_mocha_entry_title(entry), test_name)
    ]


def _mocha_entries_contain_titles(
    entries: list[Any],
    selected: list[dict[str, Any]],
) -> bool:
    """Return whether result entries contain one of the selected tests."""
    selected_titles = [_mocha_entry_title(entry) for entry in selected]
    return any(
        any(_mocha_titles_match(_mocha_entry_title(entry), title) for title in selected_titles)
        for entry in entries
        if isinstance(entry, dict)
    )


def _format_mocha_failure_output(payload: dict[str, Any]) -> str:
    """Render useful failure text from Mocha JSON output."""
    lines: list[str] = []
    for failure in payload.get("failures", []):
        if not isinstance(failure, dict):
            continue
        title = _mocha_entry_title(failure)
        error = failure.get("err") if isinstance(failure.get("err"), dict) else {}
        message = str(error.get("message") or "").strip()
        stack = str(error.get("stack") or "").strip()
        lines.extend(value for value in (title, message, stack) if value)
    return "\n".join(lines)


def _make_run_targeted_test_tool(
    sandbox: DockerSandbox,
    preferred_test_files: Sequence[str] | None = None,
    plan_state: dict[str, Any] | None = None,
):
    preferred = tuple(
        path.replace("\\", "/").strip().lstrip("/")
        for path in (preferred_test_files or [])
        if isinstance(path, str) and path.strip()
    )

    @tool
    def run_targeted_test(
        test_file: str,
        test_name: str | None = None,
    ) -> str:
        """
        Run a bounded targeted test for fast diagnostic feedback.
        Only accepts repository-relative test file paths.
        """
        if not test_file or not test_file.strip():
            return "ERROR: test_file is required."

        try:
            norm_path = _validate_workspace_path(test_file)
        except ValueError as exc:
            return f"ERROR: Invalid test file path '{test_file}': {exc}"
        if norm_path.startswith(("build/", "dist/")):
            return (
                f"ERROR: Compiled test path '{norm_path}' is not supported. "
                "Use the original source test path under test/, tests/, or the source package directory."
            )

        accepted_alt = (plan_state or {}).get("accepted_alternative_test")
        selected_path, correction_note, selection_error = _select_targeted_test_file(
            norm_path,
            preferred,
            accepted_alt,
            sandbox,
        )
        if selection_error:
            return f"ERROR: [INVALID_VALIDATION_INPUT] {selection_error}"
        if selected_path:
            norm_path = selected_path
            if plan_state is not None and correction_note:
                plan_state["last_targeted_test_selection"] = correction_note

        if preferred and norm_path not in preferred and norm_path != accepted_alt:
            preferred_text = ", ".join(preferred)
            return (
                f"ERROR: Targeted test '{norm_path}' is not the QA-recommended target. "
                f"Run one of these source test files instead: {preferred_text}."
            )

        if sandbox.read_file(norm_path) is None:
            return (
                f"BLOCKED: Targeted test path '{norm_path}' could not be verified in the workspace. "
                "Resolve the source test path from read_repository_map before choosing a replacement."
            )

        from remediation_engine.orchestration.qa_test_parsing import (
            _detect_targeted_test_context,
            build_targeted_test_command,
            extract_qa_failure_evidence,
        )

        runner, package_cwd, npm_invocation = _detect_targeted_test_context(sandbox, norm_path)
        if runner == "npm_text_fallback":
            return (
                f"BLOCKED: Cannot run targeted test on '{norm_path}'. "
                "No safe runner-specific target command can be constructed for runner 'npm_text_fallback'."
            )

        relative_test_file = norm_path
        if package_cwd and norm_path.startswith(package_cwd.rstrip("/") + "/"):
            relative_test_file = norm_path[len(package_cwd.rstrip("/")) + 1 :]
        cmd = build_targeted_test_command(
            runner,
            relative_test_file,
            test_name,
            npm_invocation=npm_invocation,
            package_cwd=package_cwd,
        )
        if not cmd:
            return f"BLOCKED: Could not construct targeted test command for runner '{runner}' and file '{norm_path}'."

        try:
            result = _run_readonly(sandbox, cmd, timeout=_NPM_TEST_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: Targeted test execution was unavailable - {exc}"

        output = (result.stdout + "\n" + result.stderr).strip()
        lowered_output = output.lower()
        if (
            result.exit_code == 124
            or "command timed out after" in lowered_output
            or "sandbox is not running" in lowered_output
        ):
            return (
                f"FAILURE: Targeted test infrastructure failed ({norm_path}).\n"
                f"Diagnostic:\n{output[:1500]}"
            )

        mocha_failure_output = ""
        if runner == "mocha":
            mocha_payload = _parse_mocha_json_output(result.stdout)
            if mocha_payload is not None:
                selected_tests = _select_mocha_tests(mocha_payload, test_name)
                if not selected_tests:
                    if test_name:
                        # A suite-only hint can be valid even when the first
                        # filtered invocation produces no JSON test entries.
                        # Discover titles from the same file without a grep,
                        # then apply the bounded selector locally. This keeps
                        # a naming mismatch from becoming a source-code
                        # failure or consuming another validation attempt.
                        discovery_cmd = build_targeted_test_command(
                            runner,
                            relative_test_file,
                            None,
                            npm_invocation=npm_invocation,
                            package_cwd=package_cwd,
                        )
                        if discovery_cmd and discovery_cmd != cmd:
                            try:
                                discovery_result = _run_readonly(
                                    sandbox,
                                    discovery_cmd,
                                    timeout=_NPM_TEST_TIMEOUT_SECONDS,
                                )
                            except Exception as exc:  # noqa: BLE001
                                return f"BLOCKED: Targeted test discovery was unavailable - {exc}"
                            discovery_output = (
                                discovery_result.stdout + "\n" + discovery_result.stderr
                            ).strip()
                            discovery_payload = _parse_mocha_json_output(discovery_result.stdout)
                            if discovery_payload is not None:
                                discovered_tests = _select_mocha_tests(
                                    discovery_payload,
                                    test_name,
                                )
                                if discovered_tests:
                                    result = discovery_result
                                    output = discovery_output
                                    mocha_payload = discovery_payload
                                    selected_tests = discovered_tests
                                else:
                                    available_titles = [
                                        _mocha_entry_title(entry)
                                        for entry in discovery_payload.get("tests", [])
                                        if isinstance(entry, dict)
                                    ]
                                    available_text = ", ".join(available_titles[:8])
                                    if len(available_titles) > 8:
                                        available_text += ", ..."
                                    return (
                                        "ERROR: [INVALID_VALIDATION_INPUT] Requested Mocha "
                                        f"test hint '{test_name}' did not match any test "
                                        f"in '{norm_path}'."
                                        + (
                                            f" Available titles: {available_text}"
                                            if available_text
                                            else ""
                                        )
                                    )
                            elif discovery_result.exit_code == 0:
                                return (
                                    "ERROR: [INVALID_VALIDATION_INPUT] Mocha test discovery "
                                    f"for '{norm_path}' returned no parseable test titles for "
                                    f"hint '{test_name}'."
                                )
                    requested = test_name or norm_path
                    return f"ERROR: [INVALID_VALIDATION_INPUT] Targeted Mocha test hint '{requested}' matched no executed tests in '{norm_path}'."

                failures = mocha_payload.get("failures", [])
                passes = mocha_payload.get("passes", [])
                if _mocha_entries_contain_titles(failures, selected_tests):
                    mocha_failure_output = _format_mocha_failure_output(mocha_payload)
                elif _mocha_entries_contain_titles(passes, selected_tests):
                    canonical_title = _mocha_entry_title(selected_tests[0])
                    return (
                        f"SUCCESS: Targeted test passed (mocha): {norm_path}"
                        f" [{canonical_title or test_name or 'selected test'}]\n"
                        f"Tests executed: {len(selected_tests)}\n"
                    )
                else:
                    return (
                        f"FAILURE: Targeted test did not pass (mocha): {norm_path}\n"
                        f"Requested test: {test_name or norm_path}\n"
                        "Diagnostic: The selected test was pending or had no passing result."
                    )
            else:
                passing = re.search(r"\b(\d+)\s+passing\b", output, re.IGNORECASE)
                failing = re.search(r"\b(\d+)\s+failing\b", output, re.IGNORECASE)
                passing_count = int(passing.group(1)) if passing else 0
                failing_count = int(failing.group(1)) if failing else 0
                if not passing and not failing or passing_count == 0 and failing_count == 0:
                    return (
                        f"FAILURE: Targeted test result could not be verified (mocha): {norm_path}\n"
                        "Diagnostic: Mocha output did not report any executed tests.\n"
                        f"Raw output:\n{output[:1500]}"
                    )
                if test_name and passing_count == 0:
                    return (
                        f"FAILURE: Targeted test did not pass (mocha): {norm_path}\n"
                        f"Requested test: {test_name}\n"
                        "Diagnostic: The requested test did not produce a passing result."
                    )

        if result.exit_code == 0 and not mocha_failure_output:
            name_str = f" [{test_name}]" if test_name else ""
            return f"SUCCESS: Targeted test passed ({runner}): {norm_path}{name_str}\n\nstdout:\n{result.stdout[:1000]}"

        evidence_stdout = mocha_failure_output or result.stdout
        evidence = extract_qa_failure_evidence(
            result.exit_code,
            evidence_stdout,
            result.stderr,
            sandbox=sandbox,
        )
        diag_str = (
            "\n".join(evidence.exact_diagnostics[:5]) or result.stderr[:500] or result.stdout[:500]
        )
        loc_str = "\n".join(evidence.source_locations[:5])
        test_str = test_name or (evidence.failed_tests[0] if evidence.failed_tests else "unknown")

        return (
            f"FAILURE: Targeted test failed ({runner}): {norm_path} (exit {result.exit_code})\n"
            f"Failing Test: {test_str}\n"
            f"Exact Diagnostics:\n{diag_str}\n"
            f"Source Locations:\n{loc_str}\n"
            f"Raw Excerpt:\n{evidence.raw_excerpt[:1000]}"
        )

    return run_targeted_test


def _make_validate_code_syntax_tool(sandbox: DockerSandbox):
    @tool
    def validate_code_syntax(file_path: str) -> str:
        """Validate syntax for a JS/TS-family source file inside the workspace."""
        try:
            rel_path = _validate_workspace_path(file_path)
        except ValueError as exc:
            return f"ERROR: {exc}"

        suffix = Path(rel_path).suffix.lower()
        if suffix in {".js", ".mjs", ".cjs"}:
            cmd = f"node -c {shlex.quote(f'/workspace/{rel_path}')}"
        elif suffix in {".ts", ".tsx", ".jsx"}:
            cmd = f"npx --yes esbuild {shlex.quote(rel_path)} --outfile=/dev/null"
        else:
            return (
                f"ERROR: validate_code_syntax does not support '{rel_path}'. "
                "Supported extensions are .js, .mjs, .cjs, .ts, .tsx, and .jsx."
            )

        result = _run_readonly(sandbox, cmd, timeout=_SYNTAX_CHECK_TIMEOUT_SECONDS)
        if result.exit_code == 0:
            return f"SUCCESS: Syntax validation passed for {rel_path}."
        return (
            f"FAILURE: Syntax validation failed for {rel_path} "
            f"(exit {result.exit_code}).\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return validate_code_syntax


def _revert_current_iteration_edit(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any],
    touched_files: set[str],
) -> None:
    """Revert the unvalidated edit set made in the current iteration."""
    pending_snapshots = plan_state.get("pending_snapshots", {})
    pending_edit_set = plan_state.get("pending_edit_set")

    if pending_snapshots:
        for file_path, orig_content in pending_snapshots.items():
            try:
                sandbox.write_file(file_path, orig_content)
                touched_files.discard(file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to revert unvalidated snapshot edit on %s: %s", file_path, exc
                )
    elif pending_edit_set:
        for edit in reversed(pending_edit_set.replacements):
            try:
                curr = sandbox.read_file(edit.file_path)
                if curr and edit.new_text in curr:
                    updated = curr.replace(edit.new_text, edit.old_text, 1)
                    sandbox.write_file(edit.file_path, updated)
                    touched_files.discard(edit.file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to revert unvalidated edit on %s: %s", edit.file_path, exc)
    else:
        last_edit = plan_state.get("current_iteration_edit")
        if last_edit:
            try:
                curr = sandbox.read_file(last_edit.file_path)
                if curr and last_edit.new_text in curr:
                    updated = curr.replace(last_edit.new_text, last_edit.old_text, 1)
                    sandbox.write_file(last_edit.file_path, updated)
                    touched_files.discard(last_edit.file_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to revert unvalidated edit on %s: %s", last_edit.file_path, exc
                )

    plan_state["pending_edit_set"] = None
    plan_state["pending_snapshots"] = {}
    plan_state["current_iteration_edit"] = None

    plan_state["phase"] = WorkaroundExecutionPhase.INVESTIGATE.value
    plan_state["local_investigation_complete"] = False
    plan_state["iteration"] = int(plan_state.get("iteration", 1)) + 1
    plan_state["successful_edit_count_this_iteration"] = 0
    plan_state["validated_files"] = []
    plan_state["validation_passed"] = False
    plan_state["accepted_alternative_test"] = None


def _make_validate_workaround_tool(
    sandbox: DockerSandbox,
    touched_files: set[str],
    plan_state: dict[str, Any] | None = None,
    preferred_test_files: Sequence[str] | None = None,
):
    """Build one short-circuiting validation gate for source workarounds.

    The gate runs syntax, typecheck, lint, runtime-import, and targeted-test
    checks in that order. It returns immediately after the first failed gate,
    preserving the exact diagnostic for the worker's next reasoning turn.
    Individual check implementations remain private helpers so update workers
    can retain their manifest-specific validation contract, while workaround
    workers receive one atomic validation tool.
    """
    if plan_state is None:
        plan_state = {}
    plan_state.setdefault("validation_calls", 0)
    plan_state.setdefault("validation_input_errors", 0)
    plan_state.setdefault("last_validation_input_error", None)

    syntax_tool = _make_validate_code_syntax_tool(sandbox)
    targeted_test_tool = _make_run_targeted_test_tool(sandbox, preferred_test_files, plan_state)

    def _failure(gate: str, detail: str) -> str:
        return f"FAILURE: Workaround validation gate '{gate}' failed.\n{detail[:4000]}"

    def _invalid_validation_request(
        detail: str,
        *,
        level: str = "ERROR",
        code: str = "INVALID_VALIDATION_INPUT",
    ) -> str:
        """Record a preflight validation-input error without consuming a gate attempt."""
        plan_state["validation_input_errors"] = (
            int(plan_state.get("validation_input_errors", 0)) + 1
        )
        plan_state["last_validation_input_error"] = detail
        marker = "" if code == "INVALID_VALIDATION_INPUT" else " [INVALID_VALIDATION_INPUT]"
        return f"{level}: [{code}]{marker} {detail}"

    def _record_result(result: WorkaroundValidationResult) -> WorkaroundValidationResult:
        """Persist the complete gate result for the supervisor success contract."""
        plan_state["last_validation_result"] = result
        plan_state["validation_passed"] = result.overall_status == WorkaroundValidationStatus.PASS
        plan_state["validated_files"] = list(result.validated_files)
        if result.infrastructure_diagnostics:
            plan_state["last_infrastructure_diagnostics"] = result.infrastructure_diagnostics
            plan_state["infrastructure_failure_details"] = result.infrastructure_diagnostics
        return result

    def _run_typecheck_gate() -> str:
        tsconfig = sandbox.read_file("tsconfig.json")
        if not isinstance(tsconfig, str) or not tsconfig.strip():
            return "SKIPPED: TypeScript gate (tsconfig.json not found)."
        try:
            result = _run_readonly(sandbox, "npx --no-install tsc --noEmit", timeout=60)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: TypeScript gate execution failed: {exc}"
        if result.exit_code == 0:
            return "SUCCESS: TypeScript compilation passed cleanly."
        output = (result.stdout + "\n" + result.stderr).strip()
        lowered = output.lower()
        if (
            result.exit_code == 127
            or "command not found" in lowered
            or "cannot find module" in lowered
            or "npx: not found" in lowered
            or "err_module_not_found" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return f"BLOCKED: TypeScript gate blocked.\nCommand: npx --no-install tsc --noEmit\nDiagnostic:\n{output}"
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: TypeScript compilation failed (exit {result.exit_code}).\n{output}"

    def _run_lint_gate(source_files: Sequence[str]) -> str:
        source_files = [
            path
            for path in source_files
            if Path(path).suffix.lower() in {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"}
        ]
        if not source_files:
            return "SKIPPED: Lint gate (no lintable source files were modified)."
        try:
            eslint_probe = _run_readonly(sandbox, "test -x node_modules/.bin/eslint", timeout=10)
        except Exception as exc:  # noqa: BLE001
            return f"FAILURE: Lint gate probe failed: {exc}"
        if eslint_probe.exit_code != 0:
            return "SKIPPED: Lint gate (the repository does not provide eslint)."
        command = "npx --no-install eslint --no-error-on-unmatched-pattern " + " ".join(
            shlex.quote(path) for path in source_files
        )
        try:
            result = _run_readonly(sandbox, command, timeout=_LINT_CHECK_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"FAILURE: Lint gate execution failed: {exc}"
        if result.exit_code == 0:
            return f"SUCCESS: Lint passed for {', '.join(source_files)}."
        output = (result.stdout + "\n" + result.stderr).strip()
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: Lint failed (exit {result.exit_code}).\n{output}"

    def _run_runtime_smoke_gate(runtime_file: str | None) -> str:
        if not runtime_file or not runtime_file.strip():
            return "FAILURE: Runtime smoke gate failed: runtime_smoke_file must be explicitly supplied for workaround validation."
        rel_path, path_error = _runtime_smoke_path_error(runtime_file)
        if path_error:
            return f"ERROR: [INVALID_RUNTIME_SMOKE] {path_error}"
        assert rel_path is not None
        suffix = Path(rel_path).suffix.lower()

        # Resolve the source path before launching Node. A missing path is a
        # target-selection error, not evidence that the code change is broken.
        source_content = sandbox.read_file(rel_path)
        if source_content is None:
            return (
                f"ERROR: [INVALID_RUNTIME_SMOKE] Source module '{rel_path}' could not be found. "
                "Use read_repository_map or read_workspace_file to resolve a source path."
            )

        import json

        import_expression = json.dumps(f"./{rel_path}")
        script = (
            f"import({import_expression}).catch((error) => {{ "
            "console.error(error?.stack || error); process.exitCode = 1; })"
        )
        loader = "--import tsx " if suffix in {".ts", ".tsx", ".jsx"} else ""
        command = f"node {loader}--input-type=module -e {shlex.quote(script)}"
        try:
            result = _run_readonly(sandbox, command, timeout=_RUNTIME_SMOKE_TIMEOUT_SECONDS)
        except Exception as exc:  # noqa: BLE001
            return f"BLOCKED: Runtime smoke gate execution failed: {exc}"
        if result.exit_code == 0:
            return f"SUCCESS: Runtime smoke import passed for {rel_path}."
        output = (result.stdout + "\n" + result.stderr).strip()
        lowered = output.lower()
        if (
            result.exit_code == 124
            or "command timed out after" in lowered
            or "sandbox is not running" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return (
                f"BLOCKED: Runtime smoke gate unavailable for {rel_path}.\n"
                f"Command: {command}\nDiagnostic:\n{output}"
            )
        if suffix in {".ts", ".tsx", ".jsx"} and (
            result.exit_code == 127
            or "cannot find module 'tsx'" in lowered
            or "tsx: command not found" in lowered
            or "err_module_not_found" in lowered
            or "npx: not found" in lowered
        ):
            if len(output) > 3000:
                output = output[:3000] + "\n... (truncated)"
            return (
                f"BLOCKED: Runtime smoke gate blocked.\nCommand: {command}\nDiagnostic:\n{output}"
            )
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return f"FAILURE: Runtime smoke import failed for {rel_path}.\n{output}"

    @tool
    def validate_workaround(
        modified_files: list[str],
        runtime_smoke_file: str | None = None,
        targeted_test_file: str | None = None,
        targeted_test_name: str | None = None,
    ) -> str:
        """Run all workaround validation gates and stop at the first failure.

        ``modified_files`` must include every source file changed by the
        current cumulative patch. A QA-recommended test is selected
        automatically when the caller omits ``targeted_test_file``.
        ``runtime_smoke_file`` must be a lightweight repository source module;
        test/spec files and compiled ``build/`` or ``dist/`` paths are rejected,
        and the smoke module must be distinct from the targeted test.
        """
        requested: list[str] = []
        for path in modified_files or []:
            if not str(path).strip():
                continue
            try:
                requested.append(_validate_workspace_path(str(path)))
            except ValueError as exc:
                return _invalid_validation_request(f"Invalid modified file path: {exc}")

        current: list[str] = []
        for path in touched_files:
            if not str(path).strip():
                continue
            try:
                current.append(_validate_workspace_path(str(path)))
            except ValueError as exc:
                return _invalid_validation_request(f"Invalid touched file path: {exc}")
        files = list(dict.fromkeys([*requested, *current]))
        if not files:
            return _invalid_validation_request("No modified files were supplied.")

        prohibited = [
            path
            for path in files
            if _is_prohibited_target(path)
            and not _is_allowlisted_no_fix_package_file(path, plan_state)
        ]
        if prohibited:
            return _invalid_validation_request(
                f"Prohibited files included: {', '.join(prohibited)}."
            )

        package_removal_mode = (
            plan_state.get("no_fix_stage") == NoFixMitigationStage.PACKAGE_REMOVAL.value
        )
        if package_removal_mode:
            if not plan_state.get("package_removal_planned"):
                return _invalid_validation_request(
                    "Record and execute the scoped package-removal plan before validation.",
                    code="PLAN_VIOLATION",
                )
            missing_operations: list[str] = []
            if not plan_state.get("no_fix_package_removed"):
                missing_operations.append("remove_no_fix_dependency")
            if (
                plan_state.get("planned_replacements")
                and plan_state.get("pending_edit_set") is None
            ):
                missing_operations.append("deterministic_apply_edit_set")
            if missing_operations:
                return _invalid_validation_request(
                    "NO_FIX PACKAGE_REMOVAL requires successful calls to both "
                    "remove_no_fix_dependency and deterministic_apply_edit_set when source "
                    "replacements are planned; either order is valid. Missing operation(s): "
                    + ", ".join(missing_operations)
                    + ".",
                    level="FAILURE",
                    code="PACKAGE_REMOVAL",
                )
            package_name = str(plan_state.get("no_fix_package_name", "") or "")
            for manifest_path in plan_state.get("no_fix_manifest_paths", []):
                manifest_text = sandbox.read_file(manifest_path)
                try:
                    manifest_data = json.loads(manifest_text or "")
                except (TypeError, json.JSONDecodeError):
                    return _invalid_validation_request(
                        f"The authorized manifest could not be parsed after removal: {manifest_path}.",
                        level="FAILURE",
                        code="PACKAGE_REMOVAL",
                    )
                if any(
                    isinstance(manifest_data.get(dep_type), dict)
                    and package_name in manifest_data[dep_type]
                    for dep_type in ("dependencies", "devDependencies", "optionalDependencies")
                ):
                    return _invalid_validation_request(
                        f"The vulnerable package remains in a direct declaration in {manifest_path}.",
                        level="FAILURE",
                        code="PACKAGE_REMOVAL",
                    )

        accepted_alt = plan_state.get("accepted_alternative_test")
        test_file, target_selection_note, target_selection_error = _select_targeted_test_file(
            accepted_alt or targeted_test_file,
            preferred_test_files,
            accepted_alt,
            sandbox,
        )
        if target_selection_error:
            return _invalid_validation_request(target_selection_error)
        if target_selection_note:
            plan_state["last_targeted_test_selection"] = target_selection_note
        if test_file and preferred_test_files:
            preferred_targets: set[str] = set()
            for path in preferred_test_files:
                if not str(path).strip():
                    continue
                try:
                    preferred_targets.add(_validate_workspace_path(str(path)))
                except ValueError:
                    continue
            if test_file not in preferred_targets and test_file != accepted_alt:
                return _invalid_validation_request(
                    f"Targeted test '{test_file}' is not the QA-recommended target. "
                    f"Run one of these source test files instead: {', '.join(sorted(preferred_targets))}."
                )

        if not runtime_smoke_file or not runtime_smoke_file.strip():
            return _invalid_validation_request(
                "Runtime smoke gate failed: runtime_smoke_file must be explicitly supplied for workaround validation.",
                level="FAILURE",
            )

        smoke_target = runtime_smoke_file
        smoke_selection_note = ""
        if smoke_target is not None and smoke_target.strip():
            normalized_smoke, smoke_selection_message = _select_lightweight_runtime_smoke_target(
                smoke_target,
                files,
                targeted_test_file=test_file,
                sandbox=sandbox,
            )
            if normalized_smoke is None:
                return _invalid_validation_request(
                    str(smoke_selection_message),
                    code="INVALID_RUNTIME_SMOKE",
                )
            smoke_target = normalized_smoke
            smoke_selection_note = smoke_selection_message or ""

        if (
            not test_file
            and plan_state.get("targeted_test_required", False)
            and not (package_removal_mode and not preferred_test_files)
        ):
            return _invalid_validation_request(
                "targeted_test_file must be supplied when a targeted test is required."
            )

        if test_file:
            normalized_test = test_file
            if Path(normalized_test).suffix.lower() not in _SOURCE_MODULE_SUFFIXES:
                return _invalid_validation_request(
                    f"Targeted test '{normalized_test}' must be a JavaScript/TypeScript source test file."
                )
            if not _is_test_file_path(normalized_test):
                return _invalid_validation_request(
                    f"Targeted test '{normalized_test}' is a source module, not a test/spec file. "
                    "Use a repository-relative path under test/, tests/, __tests__, or a *.test/spec.* file."
                )
            if sandbox.read_file(normalized_test) is None:
                return _invalid_validation_request(
                    f"Targeted test path '{normalized_test}' could not be verified in the workspace. "
                    "Resolve the source test path from read_repository_map before choosing a replacement."
                )
            test_file = normalized_test

        # Count only validations that made it through deterministic preflight
        # and are about to execute at least one validation gate.
        plan_state["validation_calls"] = int(plan_state.get("validation_calls", 0)) + 1

        syntax_msg = ""
        for file_path in files:
            if _is_allowlisted_no_fix_package_file(file_path, plan_state):
                continue
            current_content = sandbox.read_file(file_path)
            if current_content is None:
                syntax_msg = f"FAILURE: Modified file '{file_path}' could not be re-read."
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                        syntax=syntax_msg,
                        validated_files=[],
                        failure_category=FailureCategory.BREAKING_CHANGE,
                    )
                )
                _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                return f"FAILURE: Workaround validation gate 'syntax' failed.\n{syntax_msg}\nJSON: {res.model_dump_json()}"
            syntax_result = syntax_tool.invoke({"file_path": file_path})
            if str(syntax_result).startswith("BLOCKED:"):
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.BLOCKED,
                        syntax=str(syntax_result),
                        validated_files=[],
                    )
                )
                return f"{syntax_result}\nJSON: {res.model_dump_json()}"
            if not str(syntax_result).startswith("SUCCESS:"):
                syntax_msg = str(syntax_result)
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                        syntax=syntax_msg,
                        validated_files=[],
                        failure_category=FailureCategory.BREAKING_CHANGE,
                    )
                )
                _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                return f"FAILURE: Workaround validation gate 'syntax' failed.\n{syntax_msg}\nJSON: {res.model_dump_json()}"
            syntax_msg = str(syntax_result)

        typecheck_msg = _run_typecheck_gate()
        if typecheck_msg.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    typecheck=typecheck_msg,
                    validated_files=[],
                )
            )
            return f"{typecheck_msg}\nJSON: {res.model_dump_json()}"
        if typecheck_msg.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'typecheck' failed.\n{typecheck_msg}\nJSON: {res.model_dump_json()}"

        lint_msg = _run_lint_gate(files)
        if lint_msg.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    lint=lint_msg,
                    validated_files=[],
                )
            )
            return f"{lint_msg}\nJSON: {res.model_dump_json()}"
        if lint_msg.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    lint=lint_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'lint' failed.\n{lint_msg}\nJSON: {res.model_dump_json()}"

        if smoke_target is None and not plan_state.get("runtime_smoke_required", False) and files:
            smoke_target = next(
                (
                    path
                    for path in files
                    if Path(path).suffix.lower() in _SOURCE_MODULE_SUFFIXES
                    and not _is_test_file_path(path)
                ),
                None,
            )
        if package_removal_mode and not any(
            Path(path).suffix.lower() in _SOURCE_MODULE_SUFFIXES and not _is_test_file_path(path)
            for path in files
        ):
            runtime_smoke_result = (
                "SKIPPED: Runtime smoke gate (package removal changed no source modules)."
            )
        else:
            runtime_smoke_result = _run_runtime_smoke_gate(smoke_target)
        smoke_msg = runtime_smoke_result
        if smoke_selection_note:
            smoke_msg = f"{smoke_selection_note}\n{smoke_msg}"
        if smoke_target:
            plan_state["runtime_smoke_file"] = smoke_target
        if runtime_smoke_result.startswith("BLOCKED:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.BLOCKED,
                    runtime_smoke=smoke_msg,
                    validated_files=[],
                    infrastructure_diagnostics=smoke_msg,
                )
            )
            return f"{smoke_msg}\nJSON: {res.model_dump_json()}"
        if runtime_smoke_result.startswith("FAILURE:"):
            res = _record_result(
                WorkaroundValidationResult(
                    overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                    syntax=syntax_msg,
                    typecheck=typecheck_msg,
                    lint=lint_msg,
                    runtime_smoke=smoke_msg,
                    validated_files=[],
                    failure_category=FailureCategory.BREAKING_CHANGE,
                )
            )
            _revert_current_iteration_edit(sandbox, plan_state, touched_files)
            return f"FAILURE: Workaround validation gate 'runtime_smoke' failed.\n{smoke_msg}\nJSON: {res.model_dump_json()}"

        test_msg = ""
        alt_used = bool(accepted_alt)
        if test_file:
            test_res_str = targeted_test_tool.invoke(
                {"test_file": test_file, "test_name": targeted_test_name}
            )
            test_msg = str(test_res_str)
            if test_msg.startswith("ERROR: [INVALID_VALIDATION_INPUT]"):
                # The gate reached the test runner, but the requested test
                # identity was invalid. Do not treat this as a code failure or
                # consume a validation-gate attempt; retain the pending source
                # edits so the worker can retry with the canonical title.
                plan_state["validation_calls"] = max(
                    0,
                    int(plan_state.get("validation_calls", 0)) - 1,
                )
                plan_state["latest_failed_targeted_test"] = None
                plan_state["latest_test_failure_infra"] = False
                plan_state["latest_infra_diagnostics"] = None
                return _invalid_validation_request(test_msg)
            if test_msg.startswith("BLOCKED:"):
                is_infra = _is_infrastructure_failure(test_msg)
                if is_infra:
                    plan_state["latest_failed_targeted_test"] = test_file
                    plan_state["latest_test_failure_infra"] = True
                    plan_state["latest_infra_diagnostics"] = test_msg
                res = _record_result(
                    WorkaroundValidationResult(
                        overall_status=WorkaroundValidationStatus.BLOCKED,
                        syntax=syntax_msg,
                        typecheck=typecheck_msg,
                        lint=lint_msg,
                        runtime_smoke=smoke_msg,
                        targeted_test=test_msg,
                        targeted_test_file=test_file,
                        alternative_used=alt_used,
                        validated_files=[],
                        infrastructure_diagnostics=test_msg,
                    )
                )
                return f"{test_msg}\nJSON: {res.model_dump_json()}"

            if not test_msg.startswith("SUCCESS:"):
                is_infra = _is_infrastructure_failure(test_msg)
                plan_state["latest_failed_targeted_test"] = test_file
                plan_state["latest_test_failure_infra"] = is_infra
                plan_state["latest_infra_diagnostics"] = test_msg

                if is_infra:
                    res = _record_result(
                        WorkaroundValidationResult(
                            overall_status=WorkaroundValidationStatus.INFRA_FAILURE,
                            syntax=syntax_msg,
                            typecheck=typecheck_msg,
                            lint=lint_msg,
                            runtime_smoke=smoke_msg,
                            targeted_test=test_msg,
                            targeted_test_file=test_file,
                            alternative_used=alt_used,
                            validated_files=[],
                            infrastructure_diagnostics=test_msg,
                        )
                    )
                    return f"FAILURE: Workaround validation gate 'targeted_test' failed (INFRASTRUCTURE_FAILURE).\n{test_msg}\nJSON: {res.model_dump_json()}"
                else:
                    res = _record_result(
                        WorkaroundValidationResult(
                            overall_status=WorkaroundValidationStatus.CODE_FAILURE,
                            syntax=syntax_msg,
                            typecheck=typecheck_msg,
                            lint=lint_msg,
                            runtime_smoke=smoke_msg,
                            targeted_test=test_msg,
                            targeted_test_file=test_file,
                            alternative_used=alt_used,
                            validated_files=[],
                            failure_category=FailureCategory.BREAKING_CHANGE,
                        )
                    )
                    _revert_current_iteration_edit(sandbox, plan_state, touched_files)
                    return f"FAILURE: Workaround validation gate 'targeted_test' failed.\n{test_msg}\nJSON: {res.model_dump_json()}"

        res = _record_result(
            WorkaroundValidationResult(
                overall_status=WorkaroundValidationStatus.PASS,
                syntax=syntax_msg,
                typecheck=typecheck_msg,
                lint=lint_msg,
                runtime_smoke=smoke_msg,
                targeted_test=test_msg,
                targeted_test_file=test_file,
                alternative_used=alt_used,
                alternative_test_mapping_details=dict(
                    plan_state.get("original_to_alternative_test_details", {})
                ),
                validated_files=files,
            )
        )
        plan_state["validated_files"] = files
        plan_state["validation_passed"] = True
        if package_removal_mode:
            plan_state["package_removal_sync_succeeded"] = True
        if plan_state.pop("edit_failure_requires_replan", False):
            plan_state["validation_passed"] = False
            plan_state["phase"] = WorkaroundExecutionPhase.INVESTIGATE.value
        else:
            plan_state["phase"] = WorkaroundExecutionPhase.VALIDATE.value

        pending_edit_set = plan_state.get("pending_edit_set")
        if pending_edit_set is not None:
            plan_state.setdefault("successful_edit_sets", []).append(pending_edit_set)
            plan_state.setdefault("edit_records", []).extend(pending_edit_set.replacements)
            plan_state["pending_edit_set"] = None
            plan_state["pending_snapshots"] = {}
        return (
            "SUCCESS: Workaround validation gate passed. "
            f"Validated files: {', '.join(files)}; "
            f"runtime smoke: {smoke_target}; "
            f"targeted test: {test_file or 'skipped'}. "
            f"{target_selection_note + ' ' if target_selection_note else ''}"
            "The pending edit set is now committed as part of the validated cumulative patch.\n"
            f"JSON: {res.model_dump_json()}"
        )

    return validate_workaround


def _make_record_plan_tool(plan_state: dict[str, Any]):
    @tool
    def record_plan(
        affected_files: list[str],
        affected_symbols: list[str],
        security_invariant: str,
        causal_hypothesis: str,
        planned_replacements: list[WorkaroundPlannedReplacement],
        evidence_source: str | None = None,
        package_removal_requested: bool = False,
    ) -> str:
        """Record the evidence-backed plan before applying source edits.

        ``planned_replacements`` must be a flat list of exact replacement
        objects. Do not wrap the list in a file-level ``path`` or
        ``replacements`` object. Each replacement must include
        ``file_path``, ``old_text``, and ``new_text``; ``expected_occurrences``
        defaults to 1.

        Example payload::

            {
                "affected_files": ["src/auth.ts"],
                "affected_symbols": ["authorize"],
                "security_invariant": "JWT authorization remains enforced",
                "causal_hypothesis": "The dependency changed its export shape",
                "planned_replacements": [
                    {
                        "file_path": "src/auth.ts",
                        "old_text": "import oldName from 'auth-lib'",
                        "new_text": "import { newName } from 'auth-lib'",
                        "expected_occurrences": 1
                    }
                ],
                "evidence_source": "workspace:src/auth.ts"
            }

        MUST be called after local investigation is complete and BEFORE
        executing any code edits. Test and spec files are read-only: they must
        not appear in ``affected_files`` or ``planned_replacements``. For a
        NO_FIX package-removal plan, set ``package_removal_requested`` to true,
        list only the configured manifest/lockfile paths plus any source files
        in ``affected_files``, and keep manifest/lockfile mutations out of
        ``planned_replacements``; the dedicated removal tool owns them.
        """
        if plan_state.get("inspected_files") or plan_state.get("read_files"):
            plan_state["local_investigation_complete"] = True

        if not plan_state.get("local_investigation_complete", False):
            return "ERROR: [PLAN_REJECTED] Local codebase investigation must complete before recording a plan."

        phase = plan_state.get("phase", WorkaroundExecutionPhase.INVESTIGATE.value)
        if phase == WorkaroundExecutionPhase.VALIDATE.value:
            return (
                "ERROR: [PHASE_VIOLATION] A plan cannot be recorded while awaiting validation. "
                "Call validate_workaround now; a code failure will restart the loop in INVESTIGATE."
            )
        if phase not in {
            WorkaroundExecutionPhase.INVESTIGATE.value,
            WorkaroundExecutionPhase.PLAN.value,
            WorkaroundExecutionPhase.EXECUTE.value,
        }:
            return f"ERROR: [PHASE_VIOLATION] Cannot record a plan in phase '{phase}'."
        if plan_state.get("pending_edit_set") is not None:
            return (
                "ERROR: [PHASE_VIOLATION] The current edit set has not been validated. "
                "Call validate_workaround before recording another plan."
            )

        sec_inv = str(security_invariant or "").strip()
        causal_hyp = str(causal_hypothesis or "").strip()
        ev_source = str(evidence_source or "").strip()

        if not sec_inv:
            return "ERROR: [PLAN_REJECTED] security_invariant cannot be empty."
        if not causal_hyp:
            return "ERROR: [PLAN_REJECTED] causal_hypothesis cannot be empty."
        if not ev_source:
            return "ERROR: [PLAN_REJECTED] evidence_source is required and cannot be empty."

        if plan_state.get("require_authoritative_evidence"):
            evidence_is_authoritative = bool(
                plan_state.get("has_authoritative_evidence")
                or _is_authoritative_evidence_source(ev_source)
            )
            if not evidence_is_authoritative:
                return (
                    "ERROR: [MISSING_EVIDENCE] Authoritative evidence is required before accepting a "
                    "QA_REGRESSION_REPAIR plan. Set evidence_source to an official advisory, package "
                    "repository/docs URL, npm registry metadata, or an installed package README/types path."
                )

        raw_replacements = planned_replacements
        if isinstance(raw_replacements, str):
            try:
                raw_replacements = json.loads(raw_replacements)
            except Exception:
                return "ERROR: [PLAN_REJECTED] planned_replacements must be a valid JSON array or list."

        if not isinstance(raw_replacements, list):
            return (
                "ERROR: [PLAN_REJECTED] planned_replacements must be a valid list of replacements."
            )
        if not raw_replacements and not package_removal_requested:
            return "ERROR: [PLAN_REJECTED] planned_replacements must be a non-empty list of replacements."

        replacements: list[WorkaroundPlannedReplacement] = []
        for idx, item in enumerate(raw_replacements):
            try:
                if isinstance(item, WorkaroundPlannedReplacement):
                    r_obj = item
                elif isinstance(item, dict):
                    r_obj = WorkaroundPlannedReplacement(**item)
                else:
                    return f"ERROR: [PLAN_REJECTED] Item at index {idx} in planned_replacements is invalid."
                replacements.append(r_obj)
            except Exception as exc:
                return f"ERROR: [PLAN_REJECTED] Item at index {idx} in planned_replacements failed validation: {exc}"

        if len(replacements) > 16:
            return f"ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 16 replacements (got {len(replacements)})."

        files_list = affected_files if isinstance(affected_files, list) else [str(affected_files)]
        symbols_list = (
            affected_symbols if isinstance(affected_symbols, list) else [str(affected_symbols)]
        )

        declared_files = set()
        for f in files_list:
            f_str = str(f).strip()
            if f_str:
                try:
                    declared_files.add(_validate_workspace_path(f_str))
                except ValueError:
                    declared_files.add(f_str.replace("\\", "/").lstrip("/"))

        if not declared_files:
            return "ERROR: [PLAN_REJECTED] At least one affected file must be specified."

        declared_test_files = sorted(
            file_path for file_path in declared_files if _is_test_file_path(file_path)
        )
        if declared_test_files:
            return (
                "ERROR: [PROHIBITED_TARGET] Test and spec files are read-only and cannot be "
                "declared in record_plan affected_files: "
                f"{declared_test_files}."
            )

        replacement_files = set()
        seen_specs = set()
        total_text_bytes = 0

        for r in replacements:
            try:
                norm_p = _validate_workspace_path(r.file_path)
            except ValueError as exc:
                return f"ERROR: [PLAN_REJECTED] {exc}"

            if _is_prohibited_target(norm_p):
                return "ERROR: [PROHIBITED_TARGET] Workaround workers cannot modify dependency manifests or test files."

            if not r.old_text:
                return f"ERROR: [PLAN_REJECTED] Anchor old_text cannot be empty for replacement on '{norm_p}'."

            if r.old_text == r.new_text:
                return f"ERROR: [PLAN_REJECTED] Replacement cannot be a no-op (old_text equals new_text) on '{norm_p}'."

            if r.expected_occurrences <= 0:
                return f"ERROR: [PLAN_REJECTED] expected_occurrences must be positive for replacement on '{norm_p}'."

            spec_key = (norm_p, r.old_text, r.new_text, r.symbol_name)
            if spec_key in seen_specs:
                return f"ERROR: [PLAN_REJECTED] Duplicate replacement specification found for file '{norm_p}'."
            seen_specs.add(spec_key)

            replacement_files.add(norm_p)
            total_text_bytes += len(r.new_text.encode("utf-8"))

        if len(replacement_files) > 8:
            return f"ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 8 affected files (got {len(replacement_files)})."

        if total_text_bytes > 65536:
            return "ERROR: [PLAN_REJECTED] Plan exceeds maximum limit of 64 KiB of combined replacement text."

        manifest_allowlist = set(plan_state.get("no_fix_manifest_paths", []))
        allowlisted_package_files = set(
            plan_state.get("no_fix_package_files", []) or manifest_allowlist
        )
        if package_removal_requested:
            if plan_state.get("no_fix_stage") != NoFixMitigationStage.PACKAGE_REMOVAL.value:
                return "ERROR: [PLAN_REJECTED] package_removal_requested is valid only during PACKAGE_REMOVAL."
            if not manifest_allowlist:
                return "ERROR: [PLAN_REJECTED] No exact manifest allowlist is configured for package removal."
            if not declared_files.intersection(manifest_allowlist):
                return (
                    "ERROR: [PLAN_REJECTED] A package-removal plan must declare at least one "
                    "configured manifest path in affected_files."
                )
            unexpected_files = declared_files - replacement_files - allowlisted_package_files
            if unexpected_files:
                return (
                    "ERROR: [PLAN_REJECTED] Package-removal affected_files contain paths outside "
                    f"the source replacement set and manifest allowlist: {sorted(unexpected_files)}."
                )
            plan_state["package_removal_planned"] = True
        elif declared_files != replacement_files:
            return (
                f"ERROR: [PLAN_REJECTED] Mismatch between declared affected_files ({sorted(declared_files)}) "
                f"and files in planned_replacements ({sorted(replacement_files)})."
            )

        inspected_files = (
            plan_state.get("inspected_files", set())
            | plan_state.get("read_files", set())
            | plan_state.get("fallback_files", set())
        )
        for f in replacement_files:
            if f not in inspected_files:
                return (
                    f"ERROR: [PLAN_REJECTED] Target file '{f}' has not been inspected. "
                    "Inspect all planned files using read_workspace_file or inspect_ast_symbol before recording a plan."
                )

        plan_state["recorded"] = True
        plan_state["plan_revision"] = int(plan_state.get("plan_revision", 0)) + 1
        plan_state["phase"] = WorkaroundExecutionPhase.EXECUTE.value
        plan_state.pop("edit_failure_requires_replan", None)
        plan_state["successful_edit_count_this_iteration"] = 0

        plan_state["planned_replacements"] = [r.model_dump() for r in replacements]
        plan_state["planned_files"] = sorted(
            list(declared_files if package_removal_requested else replacement_files)
        )

        new_symbols = [str(s).strip() for s in symbols_list if str(s).strip()]
        existing_symbols = list(plan_state.get("planned_symbols", []))
        all_symbols = list(dict.fromkeys(existing_symbols + new_symbols))
        plan_state["planned_symbols"] = all_symbols

        plan_state["security_invariant"] = sec_inv
        plan_state["causal_hypothesis"] = causal_hyp
        plan_state["evidence_source"] = ev_source
        plan_state.setdefault("evidence_ledger", []).append(ev_source)
        if _is_authoritative_evidence_source(ev_source):
            plan_state["has_authoritative_evidence"] = True

        for f in replacement_files:
            plan_state.setdefault("fallback_files", set()).add(f)

        logger.debug(
            "Workaround subagent plan recorded (revision %s): %s",
            plan_state["plan_revision"],
            plan_state,
        )
        resp_json = {
            "status": "SUCCESS",
            "plan_revision": plan_state["plan_revision"],
            "phase_transition": "PLAN -> EXECUTE",
            "evidence_source": ev_source,
            "planned_targets": sorted(
                list(declared_files if package_removal_requested else replacement_files)
            ),
            "planned_replacements": [r.model_dump() for r in replacements],
        }
        return (
            f"SUCCESS: Plan revision {plan_state['plan_revision']} recorded successfully.\n"
            f"Evidence Source: {ev_source}\n"
            f"Phase Transition: PLAN -> EXECUTE\n"
            f"Planned Targets: {', '.join(sorted(replacement_files))}\n"
            f"JSON: {json.dumps(resp_json)}"
        )

    return record_plan


def _make_record_targeted_test_substitution_tool(
    sandbox: DockerSandbox,
    plan_state: dict[str, Any],
):
    @tool
    def record_targeted_test_substitution(
        original_test: str,
        alternative_test: str,
        infrastructure_failure_evidence: str,
        shared_behavior_explanation: str,
        evidence_sources: Any,
        infrastructure_avoidance_explanation: str,
    ) -> str:
        """Register an existing repository alternative test as a substitution for an infrastructure-failed targeted test."""
        if not original_test or not str(original_test).strip():
            return "ERROR: [SUBSTITUTION_REJECTED] original_test is required."
        if not alternative_test or not str(alternative_test).strip():
            return "ERROR: [SUBSTITUTION_REJECTED] alternative_test is required."

        infra_ev = str(infrastructure_failure_evidence or "").strip()
        shared_exp = str(shared_behavior_explanation or "").strip()
        avoid_exp = str(infrastructure_avoidance_explanation or "").strip()

        if not infra_ev:
            return "ERROR: [SUBSTITUTION_REJECTED] infrastructure_failure_evidence cannot be empty."
        if not shared_exp:
            return "ERROR: [SUBSTITUTION_REJECTED] shared_behavior_explanation cannot be empty."
        if not avoid_exp:
            return "ERROR: [SUBSTITUTION_REJECTED] infrastructure_avoidance_explanation cannot be empty."

        ev_sources = (
            evidence_sources if isinstance(evidence_sources, list) else [str(evidence_sources)]
        )
        cleaned_ev_sources = [str(s).strip() for s in ev_sources if str(s).strip()]
        if not cleaned_ev_sources:
            return "ERROR: [SUBSTITUTION_REJECTED] At least one evidence source must be provided."

        phase = plan_state.get("phase")
        if phase and phase != WorkaroundExecutionPhase.VALIDATE.value:
            return (
                "ERROR: [SUBSTITUTION_REJECTED] An alternative targeted test may only be "
                "registered after the targeted validation gate reports an infrastructure failure."
            )

        try:
            norm_orig = _validate_workspace_path(original_test)
            norm_alt = _validate_workspace_path(alternative_test)
        except ValueError as exc:
            return f"ERROR: [SUBSTITUTION_REJECTED] {exc}"

        # 1. Verify original test was the latest failed targeted test
        latest_failed = plan_state.get("latest_failed_targeted_test")
        if not latest_failed or norm_orig != latest_failed:
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] original_test '{norm_orig}' does not match "
                f"the latest failed targeted test '{latest_failed}'."
            )

        # 2. Verify failure was classified as infrastructure-only
        if not plan_state.get("latest_test_failure_infra", False):
            return (
                "ERROR: [SUBSTITUTION_REJECTED] Substitution is allowed only for infrastructure failures. "
                "Assertion failures, syntax failures, type failures, and application runtime errors do not qualify."
            )

        alt_content = sandbox.read_file(norm_alt)
        if alt_content is None:
            return f"ERROR: [SUBSTITUTION_REJECTED] alternative_test file '{norm_alt}' does not exist in workspace."

        # 3. Verify alternative is different from original
        if norm_alt == norm_orig:
            return "ERROR: [SUBSTITUTION_REJECTED] alternative_test must be different from original_test."

        if not _is_test_file_path(norm_alt):
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] alternative_test '{norm_alt}' must be an existing "
                "repository test/spec file. A source module cannot replace the targeted test."
            )

        # 4. Verify alternative test file has been locally inspected
        inspected_files = plan_state.get("inspected_files", set()) | plan_state.get(
            "read_files", set()
        )
        if norm_alt not in inspected_files:
            return (
                f"ERROR: [SUBSTITUTION_REJECTED] alternative_test file '{norm_alt}' has not been inspected. "
                "Read or inspect the test file using read_workspace_file before registering it."
            )

        # 5. Verify alternative directly imports or invokes neither unavailable dep nor failed infra setup
        infra_diag = (plan_state.get("latest_infra_diagnostics") or "").lower()
        alt_lowered = alt_content.lower()

        forbidden_keywords = []
        if "sqlite" in infra_diag:
            forbidden_keywords.extend(["sqlite3", "better-sqlite3"])
        if "native" in infra_diag or "bindings" in infra_diag:
            forbidden_keywords.extend(["node-gyp", "prebuild-install", "bindings"])

        for kw in forbidden_keywords:
            if kw in alt_lowered:
                return (
                    f"ERROR: [SUBSTITUTION_REJECTED] alternative_test '{norm_alt}' directly imports/invokes "
                    f"unavailable infrastructure component '{kw}'."
                )

        # 6. Verify only one alternative per iteration
        curr_iter = int(plan_state.get("iteration", 1))
        if plan_state.get("substitution_registered_for_iteration") == curr_iter:
            return f"ERROR: [SUBSTITUTION_REJECTED] Only one alternative test substitution is permitted for iteration {curr_iter}."

        plan_state["accepted_alternative_test"] = norm_alt
        plan_state["substitution_registered_for_iteration"] = curr_iter
        plan_state.setdefault("original_to_alternative_test_mapping", {})[norm_orig] = norm_alt
        plan_state.setdefault("original_to_alternative_test_evidence", {})[norm_orig] = (
            cleaned_ev_sources
        )
        plan_state.setdefault("original_to_alternative_test_details", {})[norm_orig] = {
            "infrastructure_failure_evidence": infra_ev,
            "shared_behavior_explanation": shared_exp,
            "infrastructure_avoidance_explanation": avoid_exp,
            "evidence_sources": ", ".join(cleaned_ev_sources),
        }

        return (
            f"SUCCESS: Registered alternative targeted test '{norm_alt}' as full replacement for "
            f"'{norm_orig}' in iteration {curr_iter}. Mapping evidence recorded."
        )

    return record_targeted_test_substitution


__all__ = [name for name in globals() if not name.startswith("__")]
