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

The previous analysis identified several evaluation axes and mapped them to DeepEval metrics. The implementation keeps the QA Critic scope deliberately narrow: only the two requested DeepEval metrics are run for QA.

1. **DeepEval integration model**: DeepEval provides a native `CallbackHandler` for LangChain/LangGraph that hooks into the existing callback system — no `@observe` decorators required. This aligns perfectly with the engine's existing `TrajectoryRecorder` callback pattern.

2. **Trajectory replay vs. live evaluation**: The previous response proposed trajectory-driven offline replay but underspecified _how_ to bridge the `TrajectoryRecorder` span format (65+ existing trajectory files under `data/trajectories/`) to DeepEval's `LLMTestCase` schema. This adapter is the most critical infrastructure piece.

3. **Existing infrastructure to leverage**:
   - 65+ real trajectory Markdown files already exist in `data/trajectories/`
   - 9 curated Juice Shop fixture JSON files under `examples/juice_shop/fixtures/` covering deterministic routing, shared dependencies, transitive upgrades, workaround replay, and suppressed packages
   - Root `conftest.py` already isolates all external services (LangSmith, OpenAI keys stripped, `TRIAGE_LLM_ENABLED=false` forced)
   - Existing `@traceable` decorators and `invoke_with_trajectory` wrappers

4. **Custom metric corrections**: The `LatencyAndTokenBudgetMetric` example used `additional_metadata` which is not a standard DeepEval `LLMTestCase` field. Should use DeepEval's built-in `cost` and `latency` fields on test cases, or subclass `BaseMetric` with explicit state injection.

5. **QA Critic scope**: The QA Critic is evaluated with `ToolCorrectnessMetric` for its read-only investigation/terminal tool trace and `TaskCompletionMetric` for the evidence-backed structured decision. Failure categorization, semantic review, and attribution remain part of the completion contract and replay evidence, not separate metrics.

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

### Phase 1 — Report Evaluation: Four-metric Markdown contract (Week 2)

> Goal: Evaluate the final deterministic Report Node Markdown against historical task, QA, attempt, and patch evidence. The report has a typed output contract, so the suite tests both factual grounding and preservation of critical remediation details.

---

#### [IMPLEMENTED] `tests/evals/test_report_eval.py`

Metrics applied:
- **`HallucinationMetric(threshold=0.30)`** — Lower-is-better check for invented facts, packages, versions, files, or statuses
- **`FaithfulnessMetric(threshold=0.85)`** — Ensures report claims are supported by final task, QA, attempt, and patch evidence
- **`SummarizationMetric(threshold=0.80)`** — Verifies that the report preserves the important outcome, follow-up, and remediation details
- **`GEval("Report Constraint Adherence", threshold=0.70)`** — One custom criterion combining aggregate counts, attempted-remediation coverage, version/workaround/pivot rules, transitive-package identity, and no invented claims or recommendations

Test structure:
```python
@pytest.mark.eval
class TestReportNodeEval:
    def test_report_uses_only_the_four_requested_metrics(self, report_case):
        """Run Hallucination, Faithfulness, Summarization, and one GEval."""
```

Data source: Ten compact cases mapped to historical trajectories and explicitly marked synthetic cases. The fixture adapter reconstructs a compatible graph state, and `generate_report()` supplies the actual Markdown output. Offline contract checks validate the fixture wiring; only the four DeepEval metrics are recorded as report metrics.

---

#### [IMPLEMENTED] `tests/evals/golden/report_cases.json`

Ten curated cases from historical trajectories and synthetic coverage fixtures:
- `fixture_type`: `historical`, `historical_contract_adjusted`, or `synthetic`
- `fixture`: Compact final task, QA, attempt, package-diff, and source-diff evidence
- `expected_output`: Canonical summary used by `SummarizationMetric`
- `expected_contract`: Offline cardinality and evidence-fragment checks
- `historical_evidence`: Short trace-backed evidence notes
- `provenance`: Source trace or explicit synthetic-fixture rationale

---

### Phase 2 — Triage & Fix Planner Evaluation (Week 3)

> Goal: Evaluate triage task completion and Fix Planner web-extraction against curated, provenance-tracked cases. Triage and Fix Planner use separate golden datasets and evaluation contracts.

---

#### [UPDATED] tests/evals/test_triage_eval.py

The triage suite uses exactly one metric:

- DeepEval TaskCompletionMetric with a 0.70 threshold judges whether the
  final triage task was completed.

The completion contract is embedded in each golden's completion_task so the
metric can judge validity, priority, reachability, guardrail recovery, and
pipeline handoff as one task outcome. The replayed actual_output is the final
post-guardrail result. No GEval, custom triage metric, schema assertion, or
separate guardrail-alignment metric is run by this suite.

The cases retain trajectory provenance labels in the golden JSON and in
docs/triage-eval-mapping.md. Missing historical evidence is explicitly marked
synthetic rather than being treated as a recorded production outcome.

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

#### [UPDATED] `tests/evals/golden/triage_cases.json`

15 curated triage task-completion cases:
- **Juice Shop derived** (from `triaged_groups_baseline.json`, `triaged_groups_deterministic.json`): Known `CRITICAL` CVEs that must be `ACTIONABLE`, known false-positives (test-only dependencies, suppressed packages) that should be `FALSE_POSITIVE` or `DEFERRED`
- **Synthetic scenarios**: fabricated transitive conflicts, SAST code injection patterns, disputed CVEs, packages with no upstream fix, reachable vs. unreachable code paths
- **Additional real-world project scans**: Cases from other npm/Node.js projects with different dependency topologies
- **Historical mapping**: Each case records whether the trajectories provide an exact match, an analogue, or no evidence.

The completion contract and replayed final outcome are stored in each case's
completion_task and actual_output fields. Fix Planner cases live in the
dedicated tests/evals/golden/fix_planner_cases.json dataset.

---

### Phase 3 — Update & Workaround Workers (Week 4–5)

> Goal: Evaluate update task completion and exact combined-tool calls for the update worker. Workaround evaluation remains a separate suite.

---

#### [NEW] `tests/evals/test_update_subagent_eval.py`

Metrics applied:
- **`ToolCorrectnessMetric(threshold=1.0)`** — Requires the combined manifest transaction (`modify_and_validate_npm_dependency`) with exact input parameters and exact order.
- **`TaskCompletionMetric(threshold=0.7)`** — Judges whether the worker completed the supervisor instruction. It runs only with `--run-eval-live`; an intentional surrender is expected to remain incomplete.

The update prompt includes the deterministic repository map as read-only
context. Error-coded transaction results are represented in retry cases as
ordered combined-tool calls, with exact package, version, dependency type, and
manifest arguments. Historical trajectories used the former split update and
validation tools, so their semantic sequences are normalized to the current
combined tool and marked with provenance in the dedicated dataset.
Retry goldens retain the bad or repeated call in `tools_called` but omit it
from `expected_tools`; this intentionally makes ToolCorrectnessMetric fail the
tool trace while TaskCompletionMetric can independently pass a recovered
update. Surrender cases likewise test the bounded outcome separately from the
tool trace.

Test structure:
```python
@pytest.mark.eval
class TestUpdateSubagentEval:
    def test_tool_correctness_deepeval(self, update_golden_cases):
        """Exact combined-tool names, arguments, and order."""

    def test_task_completion_deepeval(self, update_golden_cases):
        """Task completion judged by DeepEval's live LLM metric."""
```

---

#### [NEW] `tests/evals/test_workaround_subagent_eval.py`

Metrics applied:
- **`ToolCorrectnessMetric(threshold=1.0)`** — Checks exact workaround tool names, current input schemas, and order. Recovery goldens retain the erroneous call in the observed trace and omit it from the expected trace so the metric exposes the tool error.
- **`TaskCompletionMetric(threshold=0.7)`** — Judges whether the worker completed the supervisor's workaround instruction. It runs only with `--run-eval-live`; bounded surrender cases intentionally remain incomplete.

The active dataset is `tests/evals/golden/workaround_subagent_cases.json` and contains 16 trajectory-backed or deterministic-contract cases: clean code changes, pivots, validation recovery, no-fix removal, guardrail retries, infrastructure substitution, and bounded surrender paths. The clean first-attempt case and the validation-failure retry case are separate goldens.

Each case documents its provenance, attempt identity, exact observed and expected tool arguments, final validation output, and expected result for both metrics. Optional tool branches such as web research and AST versus file inspection are represented by case-specific traces rather than being mandatory in every case.

---

#### [NEW] `tests/evals/golden/update_subagent_cases.json`

11 update cases:
- **Historical normalization**: direct, transitive override, development dependency, parent update, invalid dependency type, and invalid manifest retries mapped to exact trajectory spans.
- **Runtime-contract cases**: invalid version syntax, manifest-sync retry, stagnation recovery, retry-limit surrender, and next-candidate fallback where historical traces predate the combined tool or do not contain the proposed values.
- The original max-round surrender is omitted because it is a global runtime-boundary test rather than a discriminative update golden; the runtime limit remains authoritative in `subagent_runtime.py`.

Each case documents its provenance, evidence status, supervisor instruction,
exact tool arguments, and expected completion status. The older shared
`subagent_cases.json` remains only as legacy shared data and is loaded by neither
of the dedicated worker suites.

---

### Phase 4 — QA Critic Evaluation: Task Completion and Tool Correctness (Week 5–6)

> Goal: Evaluate the QA Critic only on whether it completes the assigned evidence-backed decision and follows the expected read-only investigation path.

---

#### [UPDATED] `tests/evals/test_qa_critic_eval.py`

Metrics applied:
- **`ToolCorrectnessMetric(threshold=1.0)`** — Compares the observed QA tool trace with an independently authored `expected_tool_calls` trace, including tool input parameters, order, and the terminal `emit_qa_evaluation` call.
- **`TaskCompletionMetric(threshold=0.70)`** — Judges whether the Critic completed the assigned QA decision from the deterministic evidence. It runs only with `--run-eval-live`; intentional surrender after the runtime tool-call limit is expected to remain incomplete.

No separate category-accuracy, schema-validity, guardrail, semantic-review, or
retry-feedback metrics are part of the QA suite. Those details remain evidence
in the replay prompt and expected QA output used by `TaskCompletionMetric`.

Test structure:
```python
@pytest.mark.eval
class TestQACriticEval:
    def test_tool_correctness_deepeval(self, case):
        """DeepEval checks the expected QA tool names, arguments, and ordering."""

    def test_task_completion_deepeval(self, case, eval_settings):
        """DeepEval judges whether the QA decision was completed."""
```

---

#### [UPDATED] `tests/evals/golden/qa_cases.json`

20 cases cover direct update decisions, initial workaround decisions, workaround
pivots, no-fix package-removal/pivot decisions, and the 24-round surrender
boundary. The cases include pass and intentional-fail outcomes for install,
scanner, unit-test, and no-fix dependency-tree gates, plus semantic-review
branches. Install-gate failures keep post-install graph state unknown, while
package-removal graph-retention is represented as a separate scanner failure.
Each case keeps
observed `tool_calls` separate from independently authored `expected_tool_calls`
and records historical trajectory provenance and evidence status. The surrender
boundary is evaluated only by `TaskCompletionMetric`, because its full QA tool
trace is a runtime-limit observation rather than a complete expected sequence.

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

# Phases 1–4: Offline collection and fixture checks (CI-safe)
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
│                                  # WorkaroundLifecycleMetric,
│                                  # TokenBudgetMetric, LatencySLAMetric, ToolCallBudgetMetric,
│                                  # FixPlannerSchemaMetric
├── test_report_eval.py            # Phase 1: Hallucination, Faithfulness, Summarization
├── test_triage_eval.py            # Phase 2: Triage TaskCompletionMetric
├── test_fix_planner_eval.py       # Phase 2: Fix Planner extraction accuracy
├── test_update_subagent_eval.py   # Phase 3: Update worker DeepEval metrics
├── test_workaround_subagent_eval.py  # Phase 3: Workaround worker eval
├── test_qa_critic_eval.py         # Phase 4: QA task completion/tool correctness
├── test_business_rules.py         # Phase 5: Token/latency/cost SLAs
└── golden/                        # Curated evaluation datasets (multi-source, provenance-tracked)
    ├── report_cases.json           # 5–8 cases (Juice Shop + synthetic + real-world)
    ├── triage_cases.json           # 15 triage TaskCompletionMetric cases
    ├── fix_planner_cases.json      # Fix Planner web-extraction cases
    ├── update_subagent_cases.json  # 11 update cases (two DeepEval metrics)
    ├── subagent_cases.json         # Legacy shared update/workaround data
    ├── workaround_subagent_cases.json  # 16 workaround cases (two DeepEval metrics)
    └── qa_cases.json               # 20 QA cases (two DeepEval metrics)
```
