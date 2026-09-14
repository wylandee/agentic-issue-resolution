# Evaluation runbook

The evaluation suites live under `tests/evals/`. They combine deterministic contract checks with optional DeepEval judgments. The six maintained golden datasets are JSON lists with a shared envelope (`case_id`, `input`, `context`, `expected_output`, and `expected_tools`):

| Dataset | Cases | Consumer |
| --- | ---: | --- |
| `tests/evals/golden/report_cases.json` | 10 | `test_report_eval.py` |
| `tests/evals/golden/triage_cases.json` | 15 | `test_triage_eval.py` |
| `tests/evals/golden/fix_planner_cases.json` | 15 | `test_fix_planner_eval.py` |
| `tests/evals/golden/update_subagent_cases.json` | 11 | `test_update_subagent_eval.py` |
| `tests/evals/golden/workaround_subagent_cases.json` | 16 | `test_workaround_subagent_eval.py` |
| `tests/evals/golden/qa_cases.json` | 20 | `test_qa_critic_eval.py` |

Every dataset is validated by `tests/evals/golden_schema.py`. Agent replay cases contain a typed `replay.input` and an `offline_fixture`; provenance fields identify the source trajectory or explain why a case is synthetic. Historical evidence is input context, not an unmarked live result.

## Offline and live modes

The default invocation does not pass `--run-eval-live`:

```bash
python -m pytest tests/evals -q
```

This is the CI-safe mode. It runs schema, evidence, renderer, adapter, replay-boundary, and business-rule assertions without a live model judge. The worker and QA offline replay tests use `ScriptedReplayModel` driven by explicit `offline_fixture` tool traces; they do not silently turn recorded observations into model output. Tests whose only purpose is a DeepEval live judgment skip with a message directing the operator to `--run-eval-live`.

`--run-eval-live` opts into external model calls. It requires the `deepeval` extra and a usable `OPENAI_API_KEY`; `EVAL_JUDGE_MODEL` selects the judge model and defaults to `gpt-4o` (it can be set to another supported model). Agent live-replay tests also invoke the production model at the node boundary. Deterministic execution doubles still replace Docker, repository commands, scanner execution, and advisory HTTP calls, so the replay does not modify a host repository or require a live Docker daemon.

```bash
# One live suite
python -m pytest tests/evals/test_report_eval.py --run-eval-live -q

# All suites, including live-judged cases
python -m pytest tests/evals --run-eval-live -q

# Only tests marked eval (live-judge classes and business rules)
python -m pytest -m eval --run-eval-live -q
```

The marker selection is not the same as the complete offline run: several deterministic replay and contract tests are intentionally unmarked. Do not pass `--run-eval-live` in CI or in a no-network environment. In CI, omit the flag and do not provide a live API credential. Installing the optional evaluation dependencies is otherwise sufficient:

```bash
python -m pip install -e '.[dev,eval]'
```

## Replay boundary and provenance

`tests/evals/replay_adapters.py` invokes real production nodes and captures typed outputs, tool events, task/attempt identifiers, and side effects. `tests/evals/replay_harness.py` provides the deterministic boundary:

- `ScriptedReplayModel` supplies controlled assistant messages or structured results at the model boundary.
- `ReplaySandbox` keeps workspace files in memory, records commands, and returns case-registered `CommandResult` values. It never writes the host repository, starts Docker, or invokes subprocesses.
- Advisory and scanner responses are case-backed doubles. New unregistered commands fail rather than escaping the replay contract.
- A worker replay constructs a `RemediationTask` and committed attempt snapshot. Captures retain `attempt_id` and `task_revision`; QA results and report evidence remain associated with task lineage.
- A case builder uses `offline_fixture.actual_output` and `offline_fixture.actual_tools` only when no production capture is supplied. A production capture is labeled `production_live` and is never mixed with the fixture observation.

Golden cases keep independently authored expectations separate from observations. `expected_tools` is the canonical expected trace. Observed calls are stored in `offline_fixture.actual_tools` or in a live `ReplayCapture.actual_tools`. The adapters preserve tool names, argument objects, output text, and order; each suite decides whether order and exact arguments are required. This prevents a historical call sequence from becoming the expected result by accident.

Report cases expand compact evidence into task queue, attempt, QA, patch, and diff state, then call the production `generate_report()` renderer. Supervisor decisions and report rendering are deterministic; they are checked as typed/state contracts rather than judged as LLM behavior.

## Suites and implemented metric scope

### Adapters and replay contracts

- `test_adapters.py` checks trajectory Markdown/dictionary parsing, span classification, token/tool extraction, conversion to DeepEval-compatible test cases, loader behavior, and evaluation settings.
- `test_replay_harness.py` checks the in-memory sandbox, command recording, rollback, tool traces, and replay caching/deduplication.
- `test_golden_schema.py` checks all six datasets, required fields, replay inputs, duplicate IDs, and the separation of `offline_fixture` observations from expected/live fields.

These tests are deterministic and do not invoke a model judge.

### Report (`test_report_eval.py`)

The ten cases render deterministic Markdown from typed remediation evidence and validate counts, package/task identity, changed-file and diff evidence, successful versus follow-up sections, and forbidden unsupported claims. In live mode, the same rendered report is evaluated with exactly these four DeepEval metrics:

- `HallucinationMetric`, threshold `0.30`;
- `FaithfulnessMetric`, threshold `0.85`;
- `SummarizationMetric`, threshold `0.80`;
- `GEval` named `Report Constraint Adherence`, threshold `0.70`.

Offline mode does not substitute a cached judge: it runs the deterministic report contract only.

### Triage (`test_triage_eval.py`)

The 15 cases cover actionable/false-positive/deferred decisions, priority and reachability handling, missing advisory data, and deterministic guardrail recovery. Offline tests exercise the real triage path with deterministic inputs, including timeout fallback and protection against a hallucinated selected issue. Live mode runs the production triage replay and one DeepEval `TaskCompletionMetric` at threshold `0.70`. Triage has no expected tool trace.

### Fix Planner (`test_fix_planner_eval.py`)

The 15 page-backed cases cover `VERSION_BUMP`, `CODE_WORKAROUND`, and `NO_FIX` extraction, semver and source evidence, multi-page content, and workaround snippets. Offline checks validate the typed result and that versions/snippets occur in the supplied page content. Live mode adds:

- `GEval` `Fix Extraction Accuracy`, threshold `0.70`;
- `GEval` `Workaround Extraction Quality`, threshold `0.70`, for workaround cases;
- `FaithfulnessMetric`, threshold `0.85`, for non-`NO_FIX` results.

The pages are fixtures; live judging does not fetch advisory pages.

### Update worker (`test_update_subagent_eval.py`)

The 11 cases cover direct and transitive updates, dependency sections, invalid inputs, manifest-sync recovery, retries, fallback, and bounded surrender. Offline tests run the real worker loop against a scripted model and recording sandbox, then assert the typed attempt/result, changed files, final workspace, and ordered tool trace. Live mode replays the production worker and evaluates:

- `ToolCorrectnessMetric`, threshold `1.0`, exact input parameters and ordering;
- `TaskCompletionMetric`, threshold `0.70`, against the Supervisor instruction.

Expected-negative cases intentionally retain an observed bad call while omitting it from `expected_tools`, so tool correctness and eventual task completion remain independent signals.

### Workaround worker (`test_workaround_subagent_eval.py`)

The 16 cases cover clean edits, pivots, validation recovery, no-fix removal, guardrail retries, infrastructure substitution, and bounded surrender. Offline tests execute the worker and lifecycle against the deterministic workspace boundary and verify typed attempt provenance, final files, and tool calls. Live mode evaluates:

- `ToolCorrectnessMetric`, threshold `0.50`, with case-specific expected calls and ordering;
- `TaskCompletionMetric`, threshold `0.70`.

Optional research and inspection branches are represented by each case's trace; they are not mandatory for every case.

### QA critic (`test_qa_critic_eval.py`)

The 20 cases cover update and workaround decisions, pivots, package-removal decisions, install/scanner/test gates, semantic review, and the bounded tool-call surrender path. Offline tests run the real QA node against deterministic execution logs and workspace files, verify typed deterministic gates and failure evidence, and require task-keyed QA output. Live mode evaluates:

- `ToolCorrectnessMetric`, threshold `0.50`, when the case marks tool correctness applicable;
- `TaskCompletionMetric`, threshold `0.70`, for the evidence-backed QA decision.

The surrender case is a runtime-limit observation and is judged only for task completion; it does not require a complete expected tool sequence.

### Business rules (`test_business_rules.py`)

This suite reads up to 25 trajectory Markdown files under `data/trajectories/` (skipping files at or above 10 MB) and evaluates available spans for `triage`, `update_subagent`, `workaround_subagent`, and `qa_critic`. It uses the implemented deterministic custom metrics:

- `TokenBudgetMetric` (default prompt/completion ceilings: triage `15,000/1,500`, update `35,000/8,000`, workaround `350,000/15,000`, QA critic `60,000/5,000`);
- `LatencySLAMetric` (triage `30s`, update `60s`, workaround `120s`, with a `60s` fallback for other agent names);
- `ToolCallBudgetMetric` for worker tool calls: hard limit `24`, score `1.0` through 18 rounds, warning score `0.75` for rounds 19–24, and failure above 24.

Token and latency tests require at least an 80% pass rate among evaluated spans. Tool-call budget checks report clean passes, warnings, and hard failures. These checks use recorded metadata and do not call a judge model.

## Targeted commands

Run an individual suite without live judging:

```bash
python -m pytest tests/evals/test_adapters.py tests/evals/test_replay_harness.py -q
python -m pytest tests/evals/test_report_eval.py -q
python -m pytest tests/evals/test_triage_eval.py -q
python -m pytest tests/evals/test_fix_planner_eval.py -q
python -m pytest tests/evals/test_update_subagent_eval.py -q
python -m pytest tests/evals/test_workaround_subagent_eval.py -q
python -m pytest tests/evals/test_qa_critic_eval.py -q
python -m pytest tests/evals/test_business_rules.py -q
```

Add `--run-eval-live` to a targeted command only when the suite's live requirements are available. Use `-k CASE_ID` to narrow a parametrized golden case, for example:

```bash
python -m pytest tests/evals/test_qa_critic_eval.py -k qa_update_pass --run-eval-live -q
```

## Persisted runs and comparison

The evaluation session finish hook records selected evaluation items and metric results in SQLite at `data/evals/eval_results.db`, or at the path in `EVAL_DB_PATH`. Runs record mode, judge model, suite, pass/fail/skip counts, provenance metadata, and available token/cost information. `--eval-tag` gives a run a stable reference; `--eval-baseline` prints a comparison after saving the current run.

```bash
python -m pytest tests/evals/test_report_eval.py --eval-tag report-baseline -q
python -m pytest tests/evals/test_report_eval.py --eval-tag report-candidate \
  --eval-baseline report-baseline -q

python scripts/eval_compare.py --list
python scripts/eval_compare.py --latest
python scripts/eval_compare.py --run-a report-baseline --run-b report-candidate
# Use a non-default database when required:
python scripts/eval_compare.py --db-path /tmp/evals.db --list
```

Comparison is diagnostic: it displays pass-rate changes, regressions, improvements, and status changes between distinct persisted runs. It does not turn an offline run into a live judgment and does not replace pytest's exit status.
