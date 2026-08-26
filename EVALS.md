# DeepEval Evaluation Layer — Phased Implementation Plan

## Background

The `remediation_engine` is a multi-agent AppSec remediation service with a hub-and-spoke LangGraph orchestrator. Six distinct LLM-using components need evaluation:

| Component | LLM Usage | Key Output Contract |
|:--|:--|:--|
| **Triage Agent** | Conditional (`TRIAGE_LLM_ENABLED`) | `TriageResult` (verdict, strategy, confidence) |
| **Update Subagent** | Always (ReAct tool loop) | `AgentActionSummary`, `WorkerAttemptResult` |
| **Workaround Subagent** | Always (ReAct tool loop) | `AgentActionSummary`, `WorkerAttemptResult`, `WorkaroundReplayPlan` |
| **QA Critic** | Hybrid (deterministic gates + LLM investigator) | `QAEvaluation`, `BatchQAResult` |
| **Report Node** | Conditional (`REPORT_LLM_ENABLED`) | Executive narrative Markdown |
| **Fix Planner** | Optional (web page extraction) | `SerperLLMResult` |

> [!NOTE]
> The **Supervisor** is 100% deterministic Python — no LLM evaluation needed. It is already well-covered by existing unit tests (`test_deterministic_supervisor.py`, `test_supervisor_policy.py`).

### Evaluation of the Previous Response

The previous analysis correctly identified the four evaluation axes and mapped them to DeepEval metrics. However, there are several corrections and refinements needed before implementation:

1. **DeepEval integration model**: DeepEval provides a native `CallbackHandler` for LangChain/LangGraph that hooks into the existing callback system — no `@observe` decorators required. This aligns perfectly with the engine's existing `TrajectoryRecorder` callback pattern.

2. **Trajectory replay vs. live evaluation**: The previous response proposed trajectory-driven offline replay but underspecified _how_ to bridge the `TrajectoryRecorder` span format (65+ existing trajectory files under `data/trajectories/`) to DeepEval's `LLMTestCase` schema. This adapter is the most critical infrastructure piece.

3. **Existing infrastructure to leverage**:
   - 65+ real trajectory Markdown files already exist in `data/trajectories/`
   - 9 curated Juice Shop fixture JSON files under `examples/juice_shop/fixtures/` covering deterministic routing, shared dependencies, transitive upgrades, workaround replay, and suppressed packages
   - Root `conftest.py` already isolates all external services (LangSmith, OpenAI keys stripped, `TRIAGE_LLM_ENABLED=false` forced)
   - Existing `@traceable` decorators and `invoke_with_trajectory` wrappers

4. **Custom metric corrections**: The `LatencyAndTokenBudgetMetric` example used `additional_metadata` which is not a standard DeepEval `LLMTestCase` field. Should use DeepEval's built-in `cost` and `latency` fields on test cases, or subclass `BaseMetric` with explicit state injection.

5. **Missing evaluation dimension**: The previous response omitted evaluation of the **QA Critic**, which is the most complex LLM component (hybrid deterministic + LLM, with structured output via `emit_qa_evaluation`). It requires both tool correctness (did it use the read-only tools properly?) and classification accuracy (did it correctly categorize failures into `FailureCategory`?).

---

## Proposed Changes

### Phase 0 — Foundation Infrastructure (Week 1)

> Goal: Establish the evaluation framework, test harness, and trajectory adapter without touching any production code.

---

#### [NEW] `pyproject.toml` — Add `eval` optional dependency group

Add DeepEval as an optional dependency alongside an `eval` pytest marker:

```diff
[project.optional-dependencies]
 dev = ["pytest>=8,<9", "pytest-mock>=3,<4", "ruff>=0.15,<1"]
+eval = ["deepeval>=2,<3"]

[tool.pytest.ini_options]
 markers = [
   "integration: tests spanning multiple internal subsystems",
   "docker: requires a live Docker daemon",
   "network: requires external network access",
   "llm: requires an LLM provider",
+  "eval: DeepEval LLM evaluation tests (requires OPENAI_API_KEY)",
 ]
```

---

#### [NEW] `tests/evals/__init__.py`

Empty package marker.

---

#### [NEW] `tests/evals/conftest.py` — Shared eval fixtures and trajectory adapter

This is the most critical new file. It provides:

1. **`TrajectoryLoader`** — Parses saved trajectory Markdown files from `data/trajectories/` and extracts structured span data (inputs, outputs, tool calls, token counts, timing) into Python dataclasses.
2. **`trajectory_to_test_case()`** — Converts a `TrajectoryLoader` span into a DeepEval `LLMTestCase` with correct `input`, `actual_output`, `context`, `tools_called`, `expected_tools`, `cost`, and `latency` fields.
3. **Fixture factories** — `@pytest.fixture` wrappers that yield loaded trajectories for each agent type, filtered by span name prefixes (`triage.llm`, `update_subagent`, `workaround_subagent`, `qa.batch_evaluate`, `report.narrative`).
4. **Golden dataset loader** — Reads curated evaluation fixtures from `tests/evals/golden/` (to be created in Phase 1). Datasets are sourced from Juice Shop fixtures, synthetic CVE scenarios, and additional real-world project scans.
5. **`eval_settings` fixture** — Provides `AppSettings` configured for evaluation (real `OPENAI_API_KEY` from env, evaluation-specific model overrides). Also reads `EVAL_JUDGE_MODEL` env var (default: `gpt-4o`) and exposes it to all DeepEval metric constructors via their `model` parameter.

Key design decisions:
- **Offline-first**: All trajectory-based tests run without network, mocking the DeepEval judge LLM with recorded evaluations for CI. Live LLM judge runs are gated behind `--run-eval-live` flag.
- **No production code changes**: The adapter reads the existing trajectory Markdown format and `TrajectoryRecorder.to_dict()` JSON.

---

#### [NEW] `tests/evals/adapters.py` — TrajectoryRecorder-to-DeepEval bridge

Core adapter module containing:

- `TrajectorySpan` dataclass mirroring `_LocalSpan` but with additional computed fields (`duration_seconds`, `is_llm`, `is_tool`, `parent_name`)
- `parse_trajectory_markdown(path: Path) -> TrajectoryDocument` — Parses the Markdown trajectory format (spans table, JSON blocks, token summary)
- `parse_trajectory_dict(data: dict) -> TrajectoryDocument` — Parses `TrajectoryRecorder.to_dict()` output
- `spans_to_test_cases(spans: list[TrajectorySpan], agent_filter: str) -> list[LLMTestCase]` — Converts filtered spans into DeepEval test cases
- `extract_tool_calls(spans: list[TrajectorySpan], parent_span_id: str) -> list[ToolCall]` — Extracts tool call sequences for a given agent invocation

---

### Phase 1 — Report Evaluation: Hallucination & Faithfulness (Week 2)

> Goal: Evaluate the Report Node's executive narrative against deterministic evidence. This is the safest starting point because the report node has the clearest input/output contract and the evidence payload is fully deterministic.

---

#### [NEW] `tests/evals/test_report_eval.py`

Metrics applied:
- **`HallucinationMetric(threshold=0.9)`** — Compares narrative against `_evidence_payload` context
- **`FaithfulnessMetric(threshold=0.85)`** — Ensures no claims exist outside evidence
- **`GEval("Report Constraint Adherence")`** — Custom criteria: "The output must not invent CVE IDs, change task statuses, calculate metrics not in the evidence, or recommend actions"
- **`SummarizationMetric(threshold=0.8)`** — Verifies coverage of key findings

Test structure:
```python
@pytest.mark.eval
class TestReportNodeEval:
    def test_narrative_no_hallucination(self, report_evidence_fixture):
        """Narrative only contains facts from _evidence_payload."""
        
    def test_narrative_covers_key_findings(self, report_evidence_fixture):
        """Narrative mentions all resolved and unresolved groups."""
        
    def test_narrative_respects_negative_constraints(self, report_evidence_fixture):
        """Narrative does not recommend actions or invent CVEs."""
```

Data source: Existing trajectory files that contain report node spans with both the evidence JSON and the generated narrative text.

---

#### [NEW] `tests/evals/golden/report_cases.json`

5–8 curated test cases from diverse sources:
- `evidence_payload`: Deterministic evidence dict (extracted from real runs and synthetic scenarios)
- `generated_narrative`: The actual LLM output
- `expected_coverage`: List of CVE IDs, package names, and statuses that must appear
- `forbidden_claims`: Strings that must NOT appear (hallucinated CVEs, prescriptive language)
- `provenance`: Source description (Juice Shop run, synthetic multi-group scenario, edge case with zero resolved groups, etc.)

---

### Phase 2 — Triage & Fix Planner Evaluation: Classification Accuracy (Week 3)

> Goal: Evaluate LLM triage decisions and Fix Planner web-extraction against deterministic guardrail baselines and curated ground truth. Both components share a classification structure (enum-based strategy output from vulnerability context).

---

#### [NEW] `tests/evals/test_triage_eval.py`

Metrics applied:
- **`GEval("Triage Verdict Accuracy")`** — Criteria: "Given the vulnerability group context (CVE ID, CVSS, EPSS, reachability, ecosystem), evaluate whether the triage verdict (ACTIONABLE/FALSE_POSITIVE/DEFERRED) is correct"
- **`GEval("Strategy Selection Quality")`** — Criteria: "Evaluate whether the recommended strategy (VERSION_UPDATE vs CODE_WORKAROUND) is appropriate given the available fix data"
- **Custom `TriageConsistencyMetric(BaseMetric)`** — Deterministic check: LLM triage result must pass the same `_apply_guardrails()` that the engine applies post-LLM. If guardrails override the LLM verdict, that's a quality signal.

Test structure:
```python
@pytest.mark.eval
class TestTriageEval:
    def test_triage_accuracy_against_golden_set(self, triage_golden_cases):
        """LLM triage matches expected verdicts on curated CVE cases."""
    
    def test_triage_guardrail_alignment(self, triage_golden_cases):
        """LLM verdict is not overridden by deterministic guardrails."""
    
    def test_triage_false_positive_rate(self, triage_golden_cases):
        """LLM does not classify actionable CVEs as false positives."""
```

---

#### [NEW] `tests/evals/test_fix_planner_eval.py`

Metrics applied:
- **`GEval("Fix Extraction Accuracy")`** — Criteria: "Given web page content about a vulnerable package, evaluate whether the extracted strategy (VERSION_BUMP, CODE_WORKAROUND, NO_FIX) and fixed_version are correct"
- **`FaithfulnessMetric(threshold=0.85)`** — Ensures the extracted `fixed_version` and `workaround_snippets` actually appear in the provided web page content (no hallucinated version numbers)
- **Custom `FixPlannerSchemaMetric(BaseMetric)`** — Validates `SerperLLMResult` schema: `VERSION_BUMP` must have a non-empty `fixed_version` in valid semver; `CODE_WORKAROUND` must have non-empty `workaround_snippets`; `NO_FIX` must have both empty

Test structure:
```python
@pytest.mark.eval
class TestFixPlannerEval:
    def test_version_extraction_from_advisory(self, fix_planner_golden_cases):
        """Correctly extracts patched version from GitHub advisory pages."""
    
    def test_workaround_extraction_from_issues(self, fix_planner_golden_cases):
        """Correctly extracts code workaround snippets from issue threads."""
    
    def test_no_hallucinated_versions(self, fix_planner_golden_cases):
        """Extracted versions exist in the source web page content."""
```

---

#### [NEW] `tests/evals/golden/triage_cases.json`

15–20 cases from diverse sources:
- **Juice Shop derived** (from `triaged_groups_baseline.json`, `triaged_groups_deterministic.json`): Known `CRITICAL` CVEs that must be `ACTIONABLE`, known false-positives (test-only dependencies, suppressed packages) that should be `FALSE_POSITIVE` or `DEFERRED`
- **Synthetic scenarios**: fabricated transitive conflicts, SAST code injection patterns, disputed CVEs, packages with no upstream fix, reachable vs. unreachable code paths
- **Additional real-world project scans**: Cases from other npm/Node.js projects with different dependency topologies
- **Fix Planner cases**: Web page content from GitHub advisories, npm release notes, and issue threads with expected `SerperLLMResult` outputs

Each case documents its provenance (source fixture, synthetic rationale, or external project reference).

---

### Phase 3 — Subagent Tool Use Evaluation: Update & Workaround Workers (Week 4–5)

> Goal: Evaluate tool-calling correctness, architectural boundary adherence, and execution efficiency for the two specialist ReAct agents.

---

#### [NEW] `tests/evals/test_update_subagent_eval.py`

Metrics applied:
- **`ToolCorrectnessMetric(threshold=0.9)`** — Verifies correct tool sequence (`modify_npm_dependency` → `validate_manifest_sync`)
- **`TaskCompletionMetric(threshold=0.7)`** — Did the worker produce a valid patch matching the supervisor instruction?
- **Custom `ArchitectureBoundaryMetric(BaseMetric)`** — Negative assertion: worker must NOT have called tools that select versions (no `search_web` for version hunting, no `run_sandbox_command npm view` to pick versions). Only the Supervisor owns version selection.
- **Custom `ToolEfficiencyMetric(BaseMetric)`** — Penalizes: (a) reading the same file >2 times, (b) >3 failed tool calls before success, (c) total tool rounds > `MAX_SUBAGENT_TOOL_CALL_ROUNDS / 2`

Test structure:
```python
@pytest.mark.eval
class TestUpdateSubagentEval:
    def test_tool_sequence_correctness(self, update_trajectory_cases):
        """Update worker calls modify_npm_dependency then validate_manifest_sync."""
    
    def test_no_version_selection_boundary_violation(self, update_trajectory_cases):
        """Worker does not attempt to discover or select dependency versions."""
    
    def test_tool_call_efficiency(self, update_trajectory_cases):
        """Worker completes task within reasonable tool call budget."""
    
    def test_task_completion(self, update_trajectory_cases):
        """Worker produces a valid patch matching the supervisor instruction."""
```

---

#### [NEW] `tests/evals/test_workaround_subagent_eval.py`

Metrics applied (in addition to the above):
- **`GEval("Workaround Minimality")`** — Criteria: "The code change should only modify the vulnerable sink/call site. It should not refactor unrelated code, rename variables unnecessarily, or restructure control flow beyond what is needed for the security fix"
- **Custom `WoraroundLifecycleMetric(BaseMetric)`** — Enforces the `record_plan` → edit → `validate_workaround` lifecycle. Plans must be recorded before any `deterministic_apply_edit_set` calls.
- **`ToolCorrectnessMetric`** — Includes `record_plan` as mandatory first expected tool

---

#### [NEW] `tests/evals/golden/subagent_cases.json`

8–12 cases from diverse sources:
- **Juice Shop trajectories**: Simple version bump (lodash, express), transitive dependency with parent-first strategy
- **Synthetic scenarios**: Code workaround with AST-targeted edit, failed validation → retry sequence, workaround replay (pivot from failed update)
- **Additional real-world projects**: Cases with different package managers, monorepo layouts, and dependency conflict patterns

Each case documents its provenance and includes the supervisor instruction that produced it.

---

### Phase 4 — QA Critic Evaluation: Diagnostic Accuracy (Week 5–6)

> Goal: Evaluate the QA Critic's ability to correctly categorize build/scan/test failures and attribute them to the correct remediation task.

---

#### [NEW] `tests/evals/test_qa_critic_eval.py`

Metrics applied:
- **`ToolCorrectnessMetric(threshold=0.5)`** — Evaluates QA critic ordered tool sequence (`list_changed_files`, `query_qa_logs`, `generate_workspace_diff`, `read_file_context`, `search_codebase_pattern`, `inspect_ast_symbol`) and verifies termination with `emit_qa_evaluation`
- **`TaskCompletionMetric(threshold=0.7)`** — Evaluates whether the QA Critic successfully completed its diagnostic review, emitted a valid verdict, and provided actionable retry feedback (evaluated via live LLM judge under `--run-eval-live`)
- **`GEval("Failure Attribution Accuracy")`** — Criteria: "Given the install log, scan diff, and test output, evaluate whether the QA critic correctly attributed the failure to the changed package vs. a pre-existing repo issue"
- **`GEval("Failure Category Precision")`** — Criteria: "Evaluate whether the assigned FailureCategory (SECURITY_FLAG, PEER_CONFLICT, BREAKING_CHANGE, TEST_REGRESSION, etc.) matches the evidence"
- **Custom `QAStructuredOutputMetric(BaseMetric)`** — Validates that `QAEvaluation` fields are internally consistent (e.g., `passed=False` must have non-empty `failure_category` and `retry_feedback`)
- **Custom `QAGuardrailConsistencyMetric(BaseMetric)`** — Evaluates whether deterministic policy guardrails override LLM QA decisions

Test structure:
```python
@pytest.mark.eval
class TestQACriticEval:
    def test_qa_tool_sequence_correctness(self, case):
        """QA Critic uses only authorized read-only review tools and terminates with emit_qa_evaluation."""

    def test_qa_tool_correctness_deepeval(self, case, eval_settings):
        """DeepEval built-in ToolCorrectnessMetric evaluates QA Critic ordered tool execution."""

    def test_qa_deterministic_task_completion(self, case, eval_settings):
        """QA Critic produces a complete diagnostic evaluation matching ground truth."""

    def test_qa_live_task_completion_deepeval(self, case, eval_settings):
        """DeepEval built-in TaskCompletionMetric evaluates QA completion with LLM judge (requires --run-eval-live)."""

    def test_qa_structured_output_validity(self, case, eval_settings):
        """QACriticLLMOutput conforms to strict Pydantic invariants."""

    def test_qa_failure_category_accuracy(self, case, eval_settings):
        """LLM assigns the correct failure category (SECURITY_FLAG, PEER_CONFLICT, BREAKING_CHANGE)."""

    def test_qa_guardrail_consistency(self, case, eval_settings):
        """LLM QA verdict survives deterministic policy guardrails without override."""

    def test_qa_semantic_security_review(self, case, eval_settings):
        """Semantic security review verdicts and evidence references are sound."""

    def test_qa_retry_feedback_actionability(self, case, eval_settings):
        """When passed=False, retry feedback provides clear and actionable diagnostic guidance."""
```

---

#### [NEW] `tests/evals/golden/qa_cases.json`

8–10 cases from diverse sources with structured `tool_calls`, `expected_tools`, `expected_output`, and diagnostic logs:
- **Juice Shop derived**: Clean pass (all gates green), `PEER_CONFLICT` failure from `npm install`
- **Synthetic scenarios**: `SECURITY_FLAG` — ODC scan finds new/persistent issues, `BREAKING_CHANGE` — test failure attributed to the update, pre-existing test failure misattributed to the remediation (false positive QA failure)
- **Additional real-world projects**: Cases with ambiguous test output, multi-group shared dependency conflicts

Each case documents its provenance and includes the raw install/scan/test logs fed to the QA Critic along with full tool execution sequences.

---

### Phase 5 — Business Rules: Latency, Token Budget & Cost (Week 6–7)

> Goal: Enforce operational SLAs on LLM usage as regression tests.

---

#### [NEW] `tests/evals/test_business_rules.py`

Metrics applied (all custom `BaseMetric` subclasses):

- **`TokenBudgetMetric`** — Per-agent token ceiling:
  | Agent | Max Prompt Tokens | Max Completion Tokens |
  |:--|:--|:--|
  | Triage | 4,000 | 1,000 |
  | Update Subagent | 20,000 | 8,000 |
  | Workaround Subagent | 30,000 | 12,000 |
  | QA Critic (per group) | 15,000 | 5,000 |
  | Report Narrative | 4,000 | 1,500 |

- **`LatencySLAMetric`** — Per-agent wall-clock ceiling:
  | Agent | Max Latency (seconds) |
  |:--|:--|
  | Triage (per group) | 10 |
  | Update Subagent (full loop) | 60 |
  | Workaround Subagent (full loop) | 120 |
  | QA Critic (per group) | 45 |
  | Report Narrative | 15 |

- **`ToolCallBudgetMetric`** — Enforces `tool_calls_made <= MAX_SUBAGENT_TOOL_CALL_ROUNDS` and flags runs using >75% of budget as warnings

Data source: Token counts and span durations from `TrajectoryRecorder` (already tracked via `_total_prompt_tokens`, `_total_completion_tokens`, span `started_at`/`ended_at`).

Test structure:
```python
@pytest.mark.eval
class TestBusinessRules:
    @pytest.mark.parametrize("trajectory_path", TRAJECTORY_SAMPLE_PATHS)
    def test_token_budgets(self, trajectory_path):
        """Each agent stays within its allocated token budget."""

    @pytest.mark.parametrize("trajectory_path", TRAJECTORY_SAMPLE_PATHS)
    def test_latency_sla(self, trajectory_path):
        """Each agent completes within its latency SLA."""

    @pytest.mark.parametrize("trajectory_path", TRAJECTORY_SAMPLE_PATHS)
    def test_tool_call_budget(self, trajectory_path):
        """Workers don't exhaust their tool call budget."""
```

---

### Phase 6 — CI Integration & Regression Dashboard (Week 7–8)

> Goal: Wire evaluations into the development workflow with tiered execution.

---

#### [MODIFY] `pyproject.toml` — Add eval test configuration

```diff
[tool.pytest.ini_options]
 addopts = "--strict-markers"
+# Run evals with: pytest -m eval --run-eval-live
+# Offline evals (CI-safe): pytest -m eval
```

---

#### [NEW] `tests/evals/conftest.py` additions — `--run-eval-live` flag

```python
def pytest_addoption(parser):
    parser.addoption(
        "--run-eval-live",
        action="store_true",
        help="Run DeepEval metrics with live LLM judge (requires OPENAI_API_KEY)",
    )
```

When `--run-eval-live` is not set, the DeepEval judge model is mocked with cached evaluation responses. This allows the structural assertions (token budgets, tool sequences, Pydantic schema validation) to run in CI without cost.

---

#### Tiered Execution Model

| Tier | Trigger | What Runs | Network Required? | Est. Cost/Run |
|:--|:--|:--|:--|:--|
| **Tier 0: Offline CI** | Every PR | Business rules (token/latency/tool budgets), schema validation, lifecycle assertions | No | $0 |
| **Tier 1: Judge Evals** | Nightly / `--run-eval-live` | Hallucination, Faithfulness, GEval criteria on golden datasets | Yes (OpenAI) | ~$0.50 |
| **Tier 2: End-to-End** | Pre-release / manual | Full Juice Shop remediation → trajectory export → DeepEval sweep | Yes (OpenAI + Docker) | ~$5–15 |

---

## Resolved Decisions

> [!NOTE]
> **D1: Golden dataset curation scope.** Golden datasets will be curated from **additional examples beyond the Juice Shop fixtures** to cover a wide variety of scenarios. Sources will include: Juice Shop fixtures (9 existing files), synthetic CVE scenarios (fabricated transitive conflicts, SAST code injection patterns, disputed CVEs), and additional real-world project scans. Each golden dataset file will document its provenance.

> [!NOTE]
> **D2: DeepEval judge model selection.** Support **both** via an `EVAL_JUDGE_MODEL` environment variable. Default to `gpt-4o` for higher evaluation quality. Override with `EVAL_JUDGE_MODEL=gpt-4o-mini` for cheaper local/CI runs. The `eval_settings` fixture in `conftest.py` reads this env var and passes it to all DeepEval metric constructors via their `model` parameter.

> [!NOTE]
> **D3: Token budget thresholds.** Keep the **initial estimates** as proposed in Phase 5. Thresholds will be calibrated iteratively as eval data accumulates rather than derived statistically from the existing trajectory corpus upfront.

> [!NOTE]
> **D4: Fix Planner evaluation.** Include Fix Planner evaluation **in Phase 2 alongside Triage**. Both share a classification structure (enum-based strategy output) and similar input patterns (vulnerability context → LLM → structured result). The `triage_cases.json` golden dataset will include Fix Planner web-extraction cases.

---

## Verification Plan

### Automated Tests
```bash
# Phase 0: Verify adapter parses existing trajectories
pytest tests/evals/test_adapters.py -v

# Phases 1–4: Offline structural assertions (CI-safe)
pytest -m eval -v

# Phases 1–4: Live judge evaluations (requires OPENAI_API_KEY)
pytest -m eval --run-eval-live -v

# Phase 5: Business rules on trajectory corpus
pytest tests/evals/test_business_rules.py -v

# Full suite including existing tests (regression check)
python -m pytest
ruff check .
ruff format --check .
```

### Manual Verification
- Inspect DeepEval HTML report (`deepeval test run` generates a local report)
- Verify trajectory adapter correctly parses at least 5 of the 65 existing trajectory files
- Confirm CI pipeline runs Tier 0 evals in <30 seconds without network
- Validate golden dataset cases produce expected pass/fail outcomes with live judge

---

## Final File Tree

```text
tests/evals/
├── __init__.py
├── conftest.py                    # Shared fixtures, --run-eval-live flag, eval_settings,
│                                  # EVAL_JUDGE_MODEL env var support
├── adapters.py                    # TrajectoryRecorder → DeepEval LLMTestCase bridge
├── custom_metrics.py              # ArchitectureBoundaryMetric, ToolEfficiencyMetric,
│                                  # WorkaroundLifecycleMetric, QAStructuredOutputMetric,
│                                  # TokenBudgetMetric, LatencySLAMetric, ToolCallBudgetMetric,
│                                  # TriageConsistencyMetric, FixPlannerSchemaMetric
├── test_report_eval.py            # Phase 1: Hallucination, Faithfulness, Summarization
├── test_triage_eval.py            # Phase 2: Triage classification accuracy
├── test_fix_planner_eval.py       # Phase 2: Fix Planner extraction accuracy
├── test_update_subagent_eval.py   # Phase 3: Update worker tool correctness
├── test_workaround_subagent_eval.py  # Phase 3: Workaround worker eval
├── test_qa_critic_eval.py         # Phase 4: QA diagnostic accuracy
├── test_business_rules.py         # Phase 5: Token/latency/cost SLAs
└── golden/                        # Curated evaluation datasets (multi-source, provenance-tracked)
    ├── report_cases.json           # 5–8 cases (Juice Shop + synthetic + real-world)
    ├── triage_cases.json           # 15–20 cases (includes Fix Planner web-extraction)
    ├── subagent_cases.json         # 8–12 cases (version bumps, workarounds, retries)
    └── qa_cases.json               # 8–10 cases (pass/fail, misattribution edge cases)
```
