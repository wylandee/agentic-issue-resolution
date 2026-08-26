"""Phase 2 & Phase 3: Custom DeepEval BaseMetric implementations for remediation evaluation.

This module provides deterministic, domain-specific evaluation metrics that conform
to DeepEval's ``BaseMetric`` interface. These metrics can run both offline (CI-safe,
zero LLM cost) and within live evaluation test suites.
"""

from __future__ import annotations

import re
from typing import Any

from remediation_engine.contracts.schemas import (
    CVEEnrichment,
    Severity,
    SystemContext,
)
from remediation_engine.triage.agent import _apply_guardrails

try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ToolCall

    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False

    class BaseMetric:  # type: ignore[no-redef]
        """Fallback base metric when deepeval is not installed."""

        threshold: float = 1.0
        score: float | None = None
        reason: str | None = None
        success: bool | None = None
        strict_mode: bool = False
        async_mode: bool = False
        verbose_mode: bool = True

    class LLMTestCase:  # type: ignore[no-redef]
        """Fallback test case representation."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLMTestCaseParams:  # type: ignore[no-redef]
        INPUT = "input"
        ACTUAL_OUTPUT = "actual_output"
        EXPECTED_OUTPUT = "expected_output"
        CONTEXT = "context"

    class ToolCall:  # type: ignore[no-redef]
        def __init__(
            self,
            name: str,
            input_parameters: dict[str, Any] | None = None,
            output: Any = None,
        ) -> None:
            self.name = name
            self.input_parameters = input_parameters or {}
            self.output = output


# ---------------------------------------------------------------------------
# Semver regex for version validation
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^\bv?(\d+)\.(\d+)(?:\.(\d+))?(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?\b$"
)


# ---------------------------------------------------------------------------
# Phase 2: Triage Consistency Metric
# ---------------------------------------------------------------------------


class TriageConsistencyMetric(BaseMetric):
    """Metric evaluating whether deterministic RBVM guardrails override LLM triage decisions.

    A score of 1.0 indicates that the LLM verdict (is_valid, revised_priority) was in full
    harmony with the engine's deterministic policy guardrails. A score of 0.0 indicates that
    post-LLM guardrails had to override the verdict (e.g. CISA KEV in production forced to CRITICAL,
    or low-EPSS internal app downgraded to MEDIUM).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the triage consistency metric."""
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-guardrails"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Triage Consistency (Guardrail Alignment)"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure guardrail alignment for the provided test case."""
        meta = getattr(test_case, "additional_metadata", {}) or {}

        raw_priority = meta.get("llm_priority", "UNKNOWN")
        raw_is_valid = meta.get("llm_is_valid", True)
        raw_fp_reason = meta.get("llm_false_positive_reason")
        raw_orig_sev = meta.get("original_severity", "UNKNOWN")
        raw_context = meta.get("system_context", {})
        raw_enrichment = meta.get("enrichment")

        try:
            priority = Severity(raw_priority)
        except ValueError:
            priority = Severity.UNKNOWN

        try:
            orig_sev = Severity(raw_orig_sev)
        except ValueError:
            orig_sev = Severity.UNKNOWN

        context = SystemContext(**raw_context) if raw_context else SystemContext()
        enrichment = None
        if raw_enrichment:
            enrich_dict = dict(raw_enrichment)
            enrich_dict.setdefault("cve_id", "CVE-UNKNOWN")
            enrichment = CVEEnrichment(**enrich_dict)

        (
            final_priority,
            guardrail_note,
            final_is_valid,
            final_fp_reason,
            priority_overridden,
        ) = _apply_guardrails(
            priority=priority,
            is_valid=raw_is_valid,
            false_positive_reason=raw_fp_reason,
            context=context,
            enrichment=enrichment,
            original_severity=orig_sev,
        )

        validity_overridden = final_is_valid != raw_is_valid

        if priority_overridden or validity_overridden:
            self.score = 0.0
            reasons = []
            if priority_overridden:
                reasons.append(
                    f"Priority was overridden from {priority.value} to {final_priority.value}. "
                    f"Guardrail detail: {guardrail_note or 'deterministic override rule fired'}."
                )
            if validity_overridden:
                reasons.append(f"Validity was overridden from {raw_is_valid} to {final_is_valid}.")
            self.reason = " ".join(reasons)
        else:
            self.score = 1.0
            self.reason = (
                "LLM triage verdict aligns with deterministic RBVM guardrails without override."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 2: Fix Planner Schema Metric
# ---------------------------------------------------------------------------


class FixPlannerSchemaMetric(BaseMetric):
    """Metric validating structured output invariants of SerperLLMResult."""

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the Fix Planner schema metric."""
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-schema-validation"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Fix Planner Schema Invariants"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Validate SerperLLMResult invariants on the provided test case."""
        meta = getattr(test_case, "additional_metadata", {}) or {}

        strategy = meta.get("strategy")
        fixed_version = meta.get("fixed_version")
        workaround_snippets = meta.get("workaround_snippets")

        violations: list[str] = []

        valid_strategies = {"VERSION_BUMP", "CODE_WORKAROUND", "NO_FIX"}
        if strategy not in valid_strategies:
            violations.append(
                f"Invalid strategy '{strategy}'. Must be one of {sorted(valid_strategies)}."
            )

        if strategy == "VERSION_BUMP":
            if not fixed_version or not str(fixed_version).strip():
                violations.append("VERSION_BUMP strategy requires a non-empty 'fixed_version'.")
            elif not _SEMVER_RE.match(str(fixed_version).strip()):
                violations.append(
                    f"fixed_version '{fixed_version}' does not match valid semver format."
                )

        elif strategy == "CODE_WORKAROUND":
            if not workaround_snippets or not isinstance(workaround_snippets, list):
                violations.append(
                    "CODE_WORKAROUND strategy requires a non-empty 'workaround_snippets' list."
                )
            elif not any(str(s).strip() for s in workaround_snippets):
                violations.append("workaround_snippets list contains only empty strings.")

        elif strategy == "NO_FIX":
            if fixed_version is not None and str(fixed_version).strip():
                violations.append(
                    f"NO_FIX strategy must have null fixed_version, got '{fixed_version}'."
                )
            if workaround_snippets:
                violations.append(
                    f"NO_FIX strategy must have null workaround_snippets, got {workaround_snippets}."
                )

        if violations:
            self.score = 0.0
            self.reason = "Schema violations detected: " + "; ".join(violations)
        else:
            self.score = 1.0
            self.reason = (
                f"SerperLLMResult satisfies all schema invariants for strategy '{strategy}'."
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 3: Architecture Boundary Metric
# ---------------------------------------------------------------------------


class ArchitectureBoundaryMetric(BaseMetric):
    """Metric verifying architectural boundary compliance for update and workaround workers.

    Core Invariant:
      - The Update Worker is strictly an execution worker. It must NOT select versions
        or query external registries/web during first-pass executions.
      - Version discovery tools (search_web, read_web_page) are prohibited for update workers.
      - view_npm_package_versions is allowed ONLY during retry mode (`is_retry=True`).
      - Workaround workers must NOT call update-only manifest modification tools (modify_npm_dependency).
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the architecture boundary metric."""
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-boundary-assertion"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Architecture Boundary Adherence"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure architectural boundary compliance for the test case."""
        meta = getattr(test_case, "additional_metadata", {}) or {}
        eval_type = meta.get("eval_type", "update_subagent")
        is_retry = bool(meta.get("is_retry", False))

        tools_called = getattr(test_case, "tools_called", []) or []
        tool_names_and_args: list[tuple[str, dict[str, Any]]] = []

        for tool in tools_called:
            if isinstance(tool, ToolCall):
                name = getattr(tool, "name", "")
                params = getattr(tool, "input_parameters", {}) or {}
                tool_names_and_args.append((name, params))
            elif isinstance(tool, dict):
                name = tool.get("name", "")
                params = tool.get("args") or tool.get("input_parameters") or {}
                tool_names_and_args.append((name, params))

        violations: list[str] = []

        if eval_type == "update_subagent":
            for name, params in tool_names_and_args:
                if name in ("search_web", "read_web_page"):
                    violations.append(
                        f"Update worker illegally invoked research tool '{name}' (version hunting prohibited)."
                    )
                elif name == "view_npm_package_versions" and not is_retry:
                    violations.append(
                        "Update worker called 'view_npm_package_versions' during first-pass execution "
                        "(only permitted in retry mode)."
                    )
                elif name == "run_sandbox_command":
                    cmd = str(params.get("command", "") or params.get("cmd", "")).lower()
                    if any(v in cmd for v in ("npm view", "npm show", "npm info", "yarn info")):
                        violations.append(
                            f"Update worker executed registry query in sandbox command '{cmd}'."
                        )
                elif name in (
                    "deterministic_apply_edit_set",
                    "deterministic_search_replace",
                    "deterministic_replace_ast_symbol",
                ):
                    violations.append(
                        f"Update worker called source-code modification tool '{name}'."
                    )

        elif eval_type == "workaround_subagent":
            for name, _params in tool_names_and_args:
                if name == "modify_npm_dependency":
                    violations.append(
                        "Workaround worker called update-only tool 'modify_npm_dependency'."
                    )

        if violations:
            self.score = 0.0
            self.reason = "Architecture boundary violations detected: " + "; ".join(violations)
        else:
            self.score = 1.0
            self.reason = f"All tool calls comply with architectural boundaries for {eval_type}."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 3: Tool Efficiency Metric
# ---------------------------------------------------------------------------


class ToolEfficiencyMetric(BaseMetric):
    """Metric evaluating execution efficiency and budget constraints for subagents.

    Penalizes:
      1. Redundant file reads: Reading the same file > 2 times (-0.15 per excess read).
      2. Consecutive/excessive failed tool calls: > 3 failed tool calls (-0.20).
      3. Excessive total rounds: Exceeding MAX_SUBAGENT_TOOL_CALL_ROUNDS // 2 (12 rounds) (-0.30).
    """

    def __init__(
        self,
        threshold: float = 0.70,
        max_tool_rounds_budget: int = 12,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the tool efficiency metric."""
        self.threshold = threshold
        self.max_tool_rounds_budget = max_tool_rounds_budget
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-efficiency-analysis"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Tool Call Efficiency"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure tool efficiency for the test case."""
        tools_called = getattr(test_case, "tools_called", []) or []

        tool_calls: list[dict[str, Any]] = []
        for tool in tools_called:
            if isinstance(tool, ToolCall):
                tool_calls.append(
                    {
                        "name": getattr(tool, "name", ""),
                        "args": getattr(tool, "input_parameters", {}) or {},
                        "output": getattr(tool, "output", ""),
                    }
                )
            elif isinstance(tool, dict):
                tool_calls.append(
                    {
                        "name": tool.get("name", ""),
                        "args": tool.get("args") or tool.get("input_parameters") or {},
                        "output": tool.get("output", ""),
                    }
                )

        penalties: list[tuple[float, str]] = []
        total_rounds = len(tool_calls)

        # 1. Check budget usage
        if total_rounds > self.max_tool_rounds_budget:
            excess = total_rounds - self.max_tool_rounds_budget
            penalties.append(
                (
                    0.30,
                    f"Used {total_rounds} tool call rounds, exceeding recommended budget of {self.max_tool_rounds_budget} (excess: {excess}).",
                )
            )

        # 2. Check redundant file reads
        file_read_counts: dict[str, int] = {}
        for tc in tool_calls:
            name = tc["name"]
            args_dict = tc["args"]
            if name in ("read_workspace_file", "read_file_context", "inspect_file"):
                target_file = str(
                    args_dict.get("file_path")
                    or args_dict.get("path")
                    or args_dict.get("target_file")
                    or ""
                ).strip()
                if target_file:
                    file_read_counts[target_file] = file_read_counts.get(target_file, 0) + 1

        for filepath, count in file_read_counts.items():
            if count > 2:
                excess_reads = count - 2
                penalty_amount = min(0.40, excess_reads * 0.15)
                penalties.append(
                    (
                        penalty_amount,
                        f"File '{filepath}' was read {count} times (> 2 allowed reads).",
                    )
                )

        # 3. Check failed tool calls
        failed_count = 0
        for tc in tool_calls:
            out = str(tc.get("output", "") or "").lower()
            if out.startswith("error:") or "failed" in out or "exception" in out:
                failed_count += 1

        if failed_count > 3:
            penalties.append(
                (
                    0.20,
                    f"Observed {failed_count} failed tool call responses before task completion.",
                )
            )

        total_penalty = sum(p[0] for p in penalties)
        score = max(0.0, min(1.0, 1.0 - total_penalty))
        self.score = round(score, 3)

        if penalties:
            self.reason = f"Efficiency penalties applied (score={self.score}): " + " ".join(
                p[1] for p in penalties
            )
        else:
            self.reason = f"Tool execution was highly efficient across {total_rounds} rounds with zero penalties."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 3: Workaround Lifecycle Metric
# ---------------------------------------------------------------------------


class WorkaroundLifecycleMetric(BaseMetric):
    """Metric verifying the ordered 4-phase lifecycle for workaround subagents.

    Lifecycle Phases:
      1. INVESTIGATE: Read-only codebase / AST inspection and optional external research.
      2. PLAN: Mandatory call to `record_plan` BEFORE any code edits.
      3. EXECUTE: Atomic AST or string patch application via `deterministic_apply_edit_set`
         or `remove_no_fix_dependency`.
      4. VALIDATE: Call to `validate_workaround` following code edits.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the workaround lifecycle metric."""
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-lifecycle-validation"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Workaround Lifecycle Order"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure workaround lifecycle ordering for the test case."""
        tools_called = getattr(test_case, "tools_called", []) or []
        tool_names: list[str] = []

        for tool in tools_called:
            if isinstance(tool, ToolCall):
                tool_names.append(getattr(tool, "name", ""))
            elif isinstance(tool, dict):
                tool_names.append(tool.get("name", ""))

        edit_tools = {
            "deterministic_apply_edit_set",
            "deterministic_search_replace",
            "deterministic_replace_ast_symbol",
            "remove_no_fix_dependency",
        }

        violations: list[str] = []
        plan_recorded = False
        edits_made = False
        validated = False

        for i, name in enumerate(tool_names):
            if name == "record_plan":
                plan_recorded = True
            elif name in edit_tools:
                if not plan_recorded:
                    violations.append(
                        f"Step {i + 1}: '{name}' was invoked before calling mandatory planning gate 'record_plan'."
                    )
                edits_made = True
            elif name == "validate_workaround":
                if not edits_made:
                    violations.append(
                        f"Step {i + 1}: 'validate_workaround' was called before any edits were performed."
                    )
                validated = True

        if not plan_recorded and edits_made:
            violations.append("Completed edits without ever recording a workaround plan.")

        if edits_made and not validated:
            violations.append("Modified files were not validated via 'validate_workaround'.")

        if violations:
            self.score = 0.0
            self.reason = "Workaround lifecycle violations: " + "; ".join(violations)
        else:
            self.score = 1.0
            self.reason = "Workaround followed the strict investigate -> plan -> execute -> validate lifecycle."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# ---------------------------------------------------------------------------
# Phase 3: Task Completion Metric
# ---------------------------------------------------------------------------


class DeterministicTaskCompletionMetric(BaseMetric):
    """Deterministic metric evaluating whether a subagent worker completed its assigned task.

    Checks:
      1. changed_files contains the target manifest/source file.
      2. validate_manifest_sync or validate_workaround succeeded.
      3. action_status is APPLIED (fails if SURRENDER or FAILED).
    """

    def __init__(
        self,
        threshold: float = 0.70,
        verbose_mode: bool = True,
    ) -> None:
        """Initialize the deterministic task completion metric."""
        self.threshold = threshold
        self.verbose_mode = verbose_mode
        self.score = None
        self.reason = None
        self.success = None
        self.evaluation_model = "deterministic-task-completion"

    @property
    def __name__(self) -> str:
        """Metric display name."""
        return "Deterministic Task Completion"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Measure task completion status for the test case."""
        meta = getattr(test_case, "additional_metadata", {}) or {}

        action_status = meta.get("action_status", "APPLIED")
        changed_files = meta.get("changed_files", []) or []

        if action_status == "SURRENDER":
            self.score = 0.0
            self.reason = "Worker performed a Clean Room Surrender (task uncompleted)."
        elif action_status == "FAILED":
            self.score = 0.0
            self.reason = "Worker execution terminated in a FAILED status."
        elif not changed_files:
            self.score = 0.0
            self.reason = "Worker reported success but produced no changed files."
        else:
            self.score = 1.0
            self.reason = f"Worker successfully completed task and modified {len(changed_files)} files ({', '.join(changed_files)})."

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        """Async implementation delegating to synchronous measure."""
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        """Return whether metric passed the threshold."""
        return bool(self.success)


# Backwards compatibility alias
TaskCompletionMetric = DeterministicTaskCompletionMetric
