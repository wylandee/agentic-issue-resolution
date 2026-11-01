# Evaluation runbook

The evaluation suites live under `tests/evals/`. They combine deterministic contract checks with optional DeepEval judgments. The seven maintained golden datasets are JSON lists with a shared envelope (`case_id`, `input`, `context`, `expected_output`, and `expected_tools`):

| Dataset | Cases | Consumer |
| --- | ---: | --- |
| `tests/evals/golden/report_cases.json` | 14 | `test_report_eval.py` |
| `tests/evals/golden/triage_cases.json` | 20 | `test_triage_eval.py` |
| `tests/evals/golden/fix_planner_cases.json` | 18 | `test_fix_planner_eval.py` |
| `tests/evals/golden/update_subagent_cases.json` | 22 | `test_update_subagent_eval.py` |
| `tests/evals/golden/workaround_subagent_cases.json` | 19 | `test_workaround_subagent_eval.py` |
| `tests/evals/golden/qa_cases.json` | 31 | `test_qa_critic_eval.py` |
| `tests/evals/golden/supervisor_cases.json` | 17 | `test_supervisor_eval.py` |

Every dataset is validated by `tests/evals/golden_schema.py`. Agent replay cases contain a typed `replay.input` and an `offline_fixture`; provenance fields identify the source trajectory or explain why a case is synthetic. Historical evidence is input context, not an unmarked live result.

## Offline and live modes

The default invocation does not pass `--run-eval-live`:

```bash
python -m pytest tests/evals -q
```

This is the CI-safe mode. It runs schema, evidence, renderer, adapter, replay-boundary, and business-rule assertions without a live model judge. The worker and QA offline replay tests use `ScriptedReplayModel` driven by explicit `offline_fixture` tool traces; they do not silently turn recorded observations into model output. Tests whose only purpose is a DeepEval live judgment skip with a message directing the operator to `--run-eval-live`.

`--run-eval-live` opts into external model calls. It requires the `deepeval` extra and a usable `OPENAI_API_KEY`; `EVAL_JUDGE_MODEL` selects the judge model and defaults to `gpt-4o` (it can be set to another supported model). Agent live-replay tests also invoke the production model at the node boundary. Deterministic execution doubles still replace Docker, repository commands, scanner execution, and advisory HTTP calls, so the replay does not modify a host repository or require a live Docker daemon.

Live judge requests use a 60-second request timeout with automatic retries disabled. Fix Planner's production extraction keeps a separate 30-second bound. Provider or network failures surface as evaluation errors instead of indefinite retries.

In a restricted local environment where `pytest-rerunfailures` cannot bind its listener, disable only that plugin and opt out of DeepEval telemetry for the offline run:

```bash
DEEPEVAL_TELEMETRY_OPT_OUT=YES python -m pytest -p no:rerunfailures tests/evals -q
```

This keeps collection and deterministic checks offline. Live judging still needs network access and a configured key; a fresh whole-run command also needs a working Docker daemon.

```bash
# One live suite
python -m pytest tests/evals/test_report_eval.py --run-eval-live -q

# All suites, including live-judged node cases and one explicitly selected whole run
python -m pytest tests/evals --run-eval-live \
  --e2e-trajectory data/trajectories/selected-run.md -q

# Only tests marked eval (live-judge classes and business rules)
python -m pytest -m eval --run-eval-live -q
```

The marker selection is not the same as the complete offline run: several deterministic replay and contract tests are intentionally unmarked. A live whole-run score requires explicit input; without an `--e2e-trajectory` or the fresh-run pair below, the whole-run live item skips. The suite never discovers local trajectories automatically. Do not pass `--run-eval-live` in CI or in a no-network environment. In CI, omit the flag and do not provide a live API credential. Installing the optional evaluation dependencies is otherwise sufficient:

```bash
python -m pip install -e '.[dev,eval]'
```

## Whole-run evaluation

Node suites measure a single agent boundary. The `end_to_end` suite measures one complete remediation run and keeps task completion separate from safe system behavior and report faithfulness. It accepts only explicitly selected trajectories; it never discovers or scores the local trajectory corpus automatically.

Score one or more recorded, complete trajectories:

```bash
python -m remediation_engine.evals.end_to_end score \
  --trajectory data/trajectories/selected-run-a.md \
  --trajectory data/trajectories/selected-run-b.md \
  --eval-tag historical-review
```

Launch one fresh production run and evaluate its typed result and exported trajectory:

```bash
python -m remediation_engine.evals.end_to_end run \
  --repo data/clones/juice-shop \
  --input examples/juice_shop/fixtures/baseline_issues.jsonl \
  --format jsonl \
  --eval-tag juice-shop-check
```

The `run` command invokes the real system once, so it uses the configured model, scanner, and Docker boundaries. The repository path is fingerprinted before and after the run, and recorded workspace/cache volumes are checked after teardown. The original host repository must remain unchanged. For either command, `--judge-model` selects the DeepEval model and `--db-path` selects the existing evaluation SQLite database. `--strict` makes built-in metric threshold failures affect the command exit status. Without it, judge scores are diagnostic; evidence-integrity failures and judge/API errors still return nonzero. Each case records source-run identity, recorded/fresh provenance, metric reasons, coverage and compaction counts, and DeepEval/model versions. Remediation completion is recorded separately from evaluation status.

The same scorer is available inside pytest's `end_to_end` selection. It scores only explicit inputs and writes metric scores into the evaluation database with suite `end_to_end`:

```bash
# One or more recorded whole-run trajectories
python -m pytest tests/evals/test_end_to_end_eval.py --run-eval-live \
  --e2e-trajectory data/trajectories/selected-run-a.md \
  --e2e-trajectory data/trajectories/selected-run-b.md -q

# One fresh system run; this explicitly launches remediation
python -m pytest tests/evals/test_end_to_end_eval.py --run-eval-live \
  --e2e-run-repo data/clones/juice-shop \
  --e2e-input examples/juice_shop/fixtures/baseline_issues.jsonl \
  --e2e-input-format jsonl -q
```

Use `--e2e-strict` to make the three judge thresholds fail pytest items. The dashboard runner also exposes `END_TO_END` as an independent suite and asks for recorded trajectory paths or a fresh-run repository/findings pair before starting live judging. The programmatic runner selection is `end_to_end`; pass the same input options through `run_eval_subprocess(..., extra_args=[...])` to choose it independently from node suites. The one-shot `end_to_end run` CLI remains available for a separate persisted evaluation run.

The whole-run metrics use DeepEval 2.x built-ins:

- `TaskCompletionMetric`, threshold `0.70`, judges whether the original requested remediation was achieved from factual outcomes and executed tools. Unresolved findings reduce completion; safe surrender is evaluated separately by the safety metric.
- `GEval` named `System Safety and Contract Adherence`, threshold `0.70`, judges authorization, attempt lineage, validation, rollback, and cleanup evidence. Missing evidence remains unknown.
- `FaithfulnessMetric`, threshold `0.85`, compares the final report with independently projected task, worker, QA, scan, and patch evidence.

Deterministic checks enforce task and attempt identity, current QA, candidate-version authorization, incremental scan coverage, patch support, rollback evidence, contradictory success claims, and fresh-run isolation. Recorded trajectories report only what they establish. The `end_to_end` runner selection includes deterministic contracts and, when supplied explicit inputs, the complete-run live scorer.

## Replay boundary and provenance

`tests/evals/replay_adapters.py` invokes real production nodes and captures typed outputs, tool events, task/attempt identifiers, and side effects. `tests/evals/replay_harness.py` provides the deterministic boundary:

- `ScriptedReplayModel` supplies controlled assistant messages or structured results at the model boundary.
- `ReplaySandbox` keeps workspace files in memory, records commands, and returns case-registered `CommandResult` values. It never writes the host repository, starts Docker, or invokes subprocesses.
- Advisory and scanner responses are case-backed doubles. New unregistered commands fail rather than escaping the replay contract.
- A worker replay constructs a `RemediationTask` and committed attempt snapshot. Captures retain `attempt_id` and `task_revision`; QA results and report evidence remain associated with task lineage.
- A case builder uses `offline_fixture.actual_output` and `offline_fixture.actual_tools` only when no production capture is supplied. A production capture is labeled `production_live` and is never mixed with the fixture observation.

Golden cases keep independently authored expectations separate from observations. `expected_tools` is the canonical expected trace. Observed calls are stored in `offline_fixture.actual_tools` or in a live `ReplayCapture.actual_tools`. The adapters preserve tool names, argument objects, output text, and order; each suite decides whether order and exact arguments are required. This prevents a historical call sequence from becoming the expected result by accident.

## Python coverage

The seven datasets include 25 Python cases selected by `python` in their case IDs. Historical evidence comes from PyGoat trajectories: certifi and PyYAML updates, a certifi/sqlparse atomic batch, incomplete ODC coverage, and a PyJWT update/workaround path that did not clear all findings. The PyYAML 5.1 to 6.0.3 trajectory exercises the PEP 440 validation path; synthetic QA test-failure and successful-workaround scenarios remain explicit policy cases. The historical PyJWT workaround remains failed; the separate synthetic success case does not claim historical success.

| Historical trajectory | Evidence used by the Python cases |
| --- | --- |
| `phase5_20261016T075629Z_ba0b44c1-7a62-4523-b00e-7797df5e29c9.md` | certifi 2022.12.7 was updated to 2024.7.4 and passed task-scoped QA; the overall run still reported 19 new incremental-scan identifiers. |
| `phase5_20261019T120726Z_cdef0f21-e8b5-44bd-a160-dbcd055766b6.md` | PyYAML 5.1 was updated to 6.0.3 for three reported CVEs and passed task-scoped QA. |
| `phase5_20261019T165913Z_9ede9474-e210-431f-bb8b-4def9e00cca8.md` | certifi and sqlparse were updated in one atomic requirements transaction; both tasks passed QA, while the run retained an incremental-scan warning. |
| `phase5_20261016T074329Z_2da20b0f-ee1b-4365-b63d-04000305773c.md` | Targeted ODC did not provide verifiable `requirements.txt` coverage; QA returned inconclusive without consuming a remediation retry. |
| `phase5_20261019T155736Z_2f772cbc-204b-44b7-a1b5-3f2a0f177a25.md` | The PyJWT update left CVE-2025-45770 open. The proposed workaround was not accepted: its runtime import failed and another validation attempt used `introduction/tests.py` as a test target. |

Each historical case links its source trajectory under `historical_evidence`. Cases that exercise policy without an observed production attempt are labeled `synthetic_policy_case`; the PyJWT tactical workaround proposal is labeled as a historical scenario projection, and its failed validation is preserved.

Python replay cases set `replay.input.project_language` to `python`, use PyPI findings and `requirements.txt`, and preserve task-scoped QA evidence. The in-memory replay boundary handles pip and Python validation commands; it does not add Python metadata to existing Node.js cases. The evidence map and each golden's `historical_evidence` distinguish task-level QA success from an overall run that still reports incremental scan findings.

Report cases expand compact evidence into task queue, attempt, QA, patch, and diff state, then call the production `generate_report()` renderer. Offline mode uses the evidence-derived summary fallback and checks fixture-authored count pairs, scanner-ID alias relationships, special remediation outcomes, and the report contract without model calls. Supervisor decisions and report facts remain deterministic.

## Suites and implemented metric scope

### Adapters and replay contracts

- `test_adapters.py` checks trajectory Markdown/dictionary parsing, span classification, token/tool extraction, conversion to DeepEval-compatible test cases, loader behavior, and evaluation settings.
- `test_replay_harness.py` checks the in-memory sandbox, command recording, rollback, tool traces, and replay caching/deduplication.
- `test_golden_schema.py` checks all seven datasets, required fields, replay inputs, duplicate IDs, and the separation of `offline_fixture` observations from expected/live fields.

These tests are deterministic and do not invoke a model judge.

### Report (`test_report_eval.py`)

The 14 cases validate the Executive Summary, package/task identity and transitions, failed-attempt versions/outcomes, changed-file and diff evidence, successful versus follow-up sections, and forbidden unsupported claims. The first report section uses connected natural-language prose without category labels followed by colons, bolds numeric figures, and includes original, accepted-remediation, follow-up, and new finding/group counts plus evidence-backed special actions. Golden expectations are authored independently of the production summary projection; scanner aliases are represented together on explicit issue records so the fixture exercises canonical deduplication.

In live mode, each case first calls the production Executive Summary generation boundary with the configured report model, validates the generated report against the deterministic fixture contract, then scores the whole report with exactly these four DeepEval metrics. A usable `OPENAI_API_KEY` and configured report model are required in addition to the live judge configuration. The structured response supplies one clause per fact enum in the required order. Each clause selects a short, natural, fact-specific sentence lead-in from the schema's closed set of choices; new-count choices adapt to scan scope and complete, partial, or unknown evidence. Counts, identities, claims, category labels, colons, and Markdown cannot be authored in lead-ins; special-remediation wording is renderer-owned and receives an empty lead-in. The renderer inserts all counts, identities, outcomes, details, and scan limits, scopes incremental-scan counts to dependency inputs, and bolds numeric counts. If the first draft omits, repeats, or reorders a required fact reference, or selects a lead-in that does not match its fact/evidence, the boundary makes at most one repair call; only a validated model response counts as generated. Judge source evidence is built from fixture-authored facts and identifies canonical CVE/GHSA alias counting, absent final scans as unknown coverage/counts, no-special-action cases, dependency sections and package lineage, and attempt IDs with outcomes. Generation status and source fingerprint are recorded in pytest metadata and both judge cases; the persistence adapter reads the legacy `additional_metadata` field and the current serialized-case `metadata` field. A fallback still runs the judges, then fails the item's live-generation coverage check so a live run cannot silently evaluate deterministic fallback prose. Metric scores and reasons remain diagnostic; deterministic contract assertions and actual-generation coverage are the pass/fail gates:

- `HallucinationMetric`, threshold `0.70` (DeepEval 4's pass direction, preserving the prior maximum hallucination score of `0.30`);
- `FaithfulnessMetric`, threshold `0.85`;
- `SummarizationMetric`, threshold `0.80`;
- `Report Constraint Adherence`, threshold `0.70`.

The report's run-duration and token-total rows are captured before the separate summary-generation call. They describe the remediation run and its recorded agent usage; they exclude summary-generation latency/tokens and subsequent eval-judge usage. Offline mode makes no model call; live mode adds persisted, non-gating metric diagnostics plus a required summary-generation status.

Latest natural-prose validation run `run_20261031_040009_6ca027` (`llm-executive-summary-natural-prose-report-final-scoped`) completed **34/34 pytest items passed**. All **14/14** report summaries were model-generated, with no fallback errors, and all **56** metric records were persisted. The metric means were 0.9889 Faithfulness (0/14 below 0.85), 0.4286 Hallucination (8/14 below 0.70), 0.8629 Constraint Adherence (0/14 below 0.70), and 0.7894 Summarization (6/14 below 0.80). Scores remain diagnostic; generation coverage and deterministic report contracts gate the live suite. The associated live E2E golden run `run_20261031_014146_911066` (`llm-executive-summary-natural-prose-e2e-verified`) completed **47/47 pytest items passed**, including all ten goldens. Its Task Completion mean was 0.806 (2/10 below 0.70) and System Safety/Contract Adherence mean was 0.854 (1/10 below 0.70); judge scores remain diagnostic. Source evidence and metadata persistence details are described above and documented with the run results in `docs/llm-executive-summary-implementation-plan.md`.

### Triage (`test_triage_eval.py`)

The 18 cases cover actionable/false-positive/deferred decisions, priority and reachability handling, missing advisory data, ecosystem identity, and deterministic guardrail recovery. Python additions use PyGoat certifi and PyYAML evidence. Offline tests exercise the real triage path with deterministic inputs, including timeout fallback and protection against a hallucinated selected issue. Live mode runs the production triage replay and one DeepEval `TaskCompletionMetric` at threshold `0.70`. Triage has no expected tool trace.

### Tactical Supervisor (`test_supervisor_eval.py`)

The 13 cases exercise bounded single-task model proposals and Python-side
verification, including retry/repair behavior, candidate authorization,
peer-conflict referral, workaround pivots, and inconclusive QA. Offline replay
tests call the production tactical boundary with scripted model/tool output and
assert decision codes, attempt identity, selected version, staged resolution,
and state preservation. Live mode uses task-completion and, where applicable,
tool-correctness judgments. These cases evaluate a proposal boundary; the
Supervisor and deterministic portfolio functions remain the orchestration
authorities.

### Fix Planner (`test_fix_planner_eval.py`)

The 18 page-backed cases cover `VERSION_BUMP`, `CODE_WORKAROUND`, and `NO_FIX` extraction, semver and PEP 440 version evidence, multi-page content, and workaround snippets. Offline checks validate the typed result and that versions/snippets occur in the supplied page content. Live mode adds:

- `GEval` `Fix Extraction Accuracy`, threshold `0.70`;
- `GEval` `Workaround Extraction Quality`, threshold `0.70`, for workaround cases;
- `FaithfulnessMetric`, threshold `0.85`, for non-`NO_FIX` results.

Live mode runs the production structured extraction boundary against the page fixtures without fetching advisory pages. Judges see only the contract fields (strategy, fixed version, and workaround snippets), not free-form reasoning. The historical PyJWT workaround is judged for extraction fidelity only; it does not imply the proposal was accepted or validated.

### Update worker (`test_update_subagent_eval.py`)

The 16 cases cover direct and transitive updates, dependency sections, invalid inputs, manifest-sync recovery, retries, atomic batches, Python requirements updates, and bounded surrender. Offline tests run the real worker loop against a scripted model and recording sandbox, then assert the typed attempt/result, changed files, final workspace, and ordered tool trace. Live mode replays the production worker and evaluates:

- `ToolCorrectnessMetric`, threshold `1.0`, exact input parameters and ordering;
- `TaskCompletionMetric`, threshold `0.70`, against the Supervisor instruction.

Expected-negative cases intentionally retain an observed bad call while omitting it from `expected_tools`, so tool correctness and eventual task completion remain independent signals.

### Workaround worker (`test_workaround_subagent_eval.py`)

The 19 cases cover clean edits, pivots, validation recovery, no-fix removal, guardrail retries, infrastructure substitution, bounded surrender, and Python validation. Offline tests execute the worker and lifecycle against the deterministic workspace boundary and verify typed attempt provenance, final files, and tool calls. Live mode evaluates:

- `ToolCorrectnessMetric`, threshold `0.50`, with case-specific expected calls and ordering;
- `TaskCompletionMetric`, threshold `0.70`.

Historical failure-injection cases with intentionally invalid calls may mark tool correctness inapplicable; deterministic replay asserts those known failure paths. Validation-failure cases may also mark TaskCompletion inapplicable because their expected failure is asserted deterministically. Optional research and inspection branches are represented by each case's trace; they are not mandatory for every case.

### QA critic (`test_qa_critic_eval.py`)

The 25 cases cover update and workaround decisions, pivots, package-removal decisions, install/scanner/test gates, semantic review, task-scoped Python scan evidence, and the bounded tool-call surrender path. Offline tests run the real QA node against deterministic execution logs and workspace files, verify typed deterministic gates and failure evidence, and require task-keyed QA output. Live mode evaluates:

- `ToolCorrectnessMetric`, threshold `0.50`, when the case marks tool correctness applicable;
- `TaskCompletionMetric`, threshold `0.70`, for the evidence-backed QA decision.

The surrender case is a runtime-limit observation and is judged only for task completion; it does not require a complete expected tool sequence.

### Business rules (`test_business_rules.py`)

This suite reads up to 25 trajectory Markdown files under `data/trajectories/` (skipping files at or above 10 MB) and evaluates available spans for `triage`, `update_subagent`, `workaround_subagent`, and `qa_critic`. Whole-run historical scoring is separate and always requires explicit trajectory arguments. The business rules use deterministic custom metrics:

- `TokenBudgetMetric` (default prompt/completion ceilings: triage `15,000/1,500`, update `35,000/8,000`, workaround `350,000/15,000`, QA critic `60,000/5,000`);
- `LatencySLAMetric` (triage `30s`, update `60s`, workaround `120s`, with a `60s` fallback for other agent names);
- `ToolCallBudgetMetric` for worker tool calls: hard limit `24`, score `1.0` through 18 rounds, warning score `0.75` for rounds 19–24, and failure above 24.

Token and latency tests require at least an 80% pass rate among spans with telemetry. Tool-call budget checks report clean passes, warnings, and hard failures. Missing token, latency, or loop-round evidence is unscored and excluded from the compliance denominator; each business-rule result logs eligible-versus-scored evidence coverage. Zero-token and zero-duration measurements are valid observations. Tool executions are collected from nested descendant spans, proposed model calls are excluded, and the loop budget counts distinct `react.llm` rounds. These checks use recorded metadata and do not call a judge model.

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
