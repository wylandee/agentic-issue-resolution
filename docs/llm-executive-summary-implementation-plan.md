# LLM Executive Summary implementation plan

## Requested outcome

Make the first report section `## 1. Executive Summary`, containing a short LLM-authored account of the run. Keep the existing metrics table immediately below that prose. Preserve the content, numbering, layout, and evidence rules of Follow up Actions, Successful Remediations, and References.

The summary covers original findings and groups, successfully remediated findings and groups, follow-up findings and groups, notable special remediations, and newly detected findings and groups. The request repeated the findings/groups item; this implementation interprets the second occurrence as successfully remediated counts.

Implementation was completed by the user-authorized GPT-6 Luna Max agent. This coding-model choice did not change the application's configured report or judge models.

## Current implementation and constraints

- `orchestration/report_context.py` normalizes graph state into `ReportContext`, reconciles scan evidence, and calculates canonical finding counts. The initial-triage `issues` and `initial_valid_groups` are retained at the state boundary, including an explicit empty baseline; no separate duplicate snapshot was needed.
- `orchestration/report_node.py` renders all four sections deterministically. `_render_summary` already calculates unique effective group statuses, followed by the metrics table. Group lineage, CVE/GHSA aliases, untriaged scan findings, and transitive package identity require the existing deduplication rules.
- `generate_report` is a pure renderer. `run_report_node` produces a preliminary report after teardown. `finalize_report` builds the final context and persists the final Markdown; inspect its caller to ensure the returned report, state, trajectory evidence, and API output remain consistent.
- `_successful_remediation_evidence`, accepted attempt QA, final diff, and replay-backed source changes already constrain special-remediation claims. Reuse these projections rather than treating worker prose or a selected version as proof of success.
- Report goldens live in `tests/evals/golden/report_cases.json`. `tests/evals/test_report_eval.py` owns format parsing, independent evidence, deterministic contracts, four live metrics, and judge prompts. Live Report Evals now call the production Executive Summary generator before scoring the full report and require actual-generation coverage.
- The E2E golden loader defines ten cases across npm, Python, and Java. `--e2e-golden all` selects fresh fixture-backed runs. Recorded-trajectory scoring alone does not verify this change.

## 1. Add a typed summary evidence projection

Create a focused report-summary module with typed evidence and generation contracts. Build it from the same normalized final context used by the table and detail sections. Avoid moving or rewriting unrelated report code.

Evidence must include:

| Fact | Definition |
| --- | --- |
| Original findings | Canonical baseline scanner findings, retaining existing CVE/GHSA alias identity rules. Never count newly detected findings as originals. |
| Original groups | Unique nonsynthetic baseline vulnerability groups. Pivot children and retry attempts do not create new original groups. |
| Remediated groups | Unique effective groups with accepted final QA-passed outcomes, consistent with successful-remediation rows and authoritative scan reconciliation. |
| Remediated findings | Canonical findings associated with those outcomes. A finding shared by unresolved groups must not be claimed completely remediated. |
| Follow-up groups | Unique effective outstanding groups from the report's actual follow-up projection, including confirmed untriaged findings where applicable. |
| Follow-up findings | Canonical findings associated with those outstanding groups, deduplicated across aliases and overlapping groups. |
| Newly detected findings | Confirmed canonical discoveries across accepted scans/reconciliation, even if later remediated. Distinguish these from findings still present in the terminal scan. |
| Newly detected groups | Actual new groups established by reconciliation, excluding existing groups reopened or reappearing and excluding pivot children. Untriaged identifiers are discoveries without an established group count. |
| Special remediations | Evidence-supported code workarounds, API migrations, package removal with follow-on source edits, and other supported nontrivial actions. Include attempt identity, outcome, package identity, and concise accepted changes. Distinguish successful, attempted/failed, and subsequently rolled-back work. |
| Evidence limits | Scan scope/completeness and missing grouping or finding-identity evidence. Unknown counts remain unknown, never fabricated zeros. |

Keep provenance or stable evidence references with each derived item. If baseline state does not preserve enough information, add the smallest explicit state snapshot at the input/initial-triage boundary; do not reconstruct originals from mutable final groups. Findings and groups need not have a one-to-one relationship, so do not impose false arithmetic across them.

Use a single group-count projection for the existing table and summary evidence. Preserve the table's current rows and meanings. Add tests for any genuine existing inconsistency that must be corrected to satisfy state consistency.

## 2. Generate bounded, evidence-grounded prose

- Add an injectable generator using the project's existing LangChain/OpenAI configuration and key handling. Read configuration through `AppSettings`; use the existing configured default model unless an explicit report override is added and documented. Do not hardcode GPT-6 Luna into application behavior merely because it is the requested coding model.
- Use a compact structured evidence payload and structured response schema. Ask for concise executive prose, all requested facts, special actions when supported, and explicit uncertainty. Treat source text as data and prohibit new remediation advice, unsupported identities, and invented successes.
- Keep authoritative counts and identities outside model arithmetic. Prefer structured references/placeholders for factual values, resolved against the evidence by the renderer, plus validation of allowed references, required coverage, and Markdown shape. Do not claim that a numeric regex alone establishes factual correctness.
- Reject empty, malformed, contradictory, missing-required-fact, or structurally disruptive output. Bound call duration, response size, and retries. A generation failure must preserve a useful report and remediation result through an evidence-derived fallback summary, with generation status/error recorded separately.
- Keep `generate_report` usable without network calls through an explicit optional validated summary input or injected boundary. Deterministic offline callers receive the same-format fallback. The final production report must attempt LLM generation when usable credentials/configuration exist.

## 3. Integrate generation into finalization

- Generate against one final evidence snapshot after remediation and teardown, before the final Markdown is persisted and returned. The preliminary report may use the deterministic fallback; avoid charging for a duplicate preliminary summary.
- Render `## 1. Executive Summary`, then validated narrative, then the unchanged existing metrics table and scan coverage note.
- Preserve sections 2–4 byte-for-byte for the same context. Add a regression assertion comparing these sections with their prior renderers.
- Record minimal summary provenance/status and source fingerprint if needed so finalization cannot reuse stale prose after state changes. Do not append report-generation errors to remediation success/failure decisions.
- Check the orchestrator finalization sequence so the API result, canonical report file, final state, and exported evidence agree. Preserve return signatures where possible; account for report generation telemetry without corrupting duration/token facts or causing recursive regeneration.

## 4. Update Report Evals without changing metrics

- Update format version, heading constants, parser, expected outputs, golden contracts/schema, judge policy, and case-specific summarization questions for the Executive Summary.
- Add independent fixture expectations for the new counts and special remediations. Do not derive expected facts from the generated report or use the production summary projection as the only correctness oracle.
- Preserve exactly these metrics, their thresholds, and current diagnostic/gating behavior:
  - Hallucination: `0.70` minimum under the installed DeepEval scoring direction.
  - Faithfulness: `0.85`.
  - Summarization: `0.80`.
  - Report Constraint Adherence: `0.70`.
- Keep factual source evidence separate from output-format policies in judge inputs.
- In live mode, generate the summary through the production boundary before scoring the whole report. Record whether an actual LLM summary or fallback was evaluated; an unnoticed fallback cannot be described as live-generation coverage.
- Keep offline tests deterministic with injected model doubles. Extend goldens for mixed outcomes, no groups/failed setup, partial and absent scan coverage, alias duplication, overlapping groups, pivot lineage, fixed discoveries, reopened old findings, stale QA/results, API migration, and package removal with source edits.

## 5. Verification and required execution order

Use the repository `.venv`, workspace-local temporary paths, and existing evaluation persistence. Do not print secrets or commit runtime outputs.

1. Run focused report/context/persistence/settings and Report Eval contract tests, plus relevant orchestrator finalization tests. Confirm valid model output, invalid output, timeout/failure fallback, stale evidence, and unchanged sections 2–4.
2. Run the full offline test suite and required lint/format checks. Avoid silently weakening existing assertions or metric thresholds.
3. Run the complete live E2E golden suite first:

   ```text
   DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python -m pytest -p no:rerunfailures tests/evals/test_end_to_end_eval.py --run-eval-live --e2e-golden all --e2e-strict --eval-tag llm-executive-summary-e2e -q
   ```

   Verify prerequisites through safe checks: configured credentials without displaying them, Docker availability, golden source clones, scanner artifacts, and required dependency services. Request sandbox escalation only when needed. Inspect persisted results for all ten live cases, metric thresholds, evidence integrity, cleanup, and host-source isolation. Compare with a comparable available baseline; without one, distinguish passing regression contracts from a measured no-regression claim.

4. After the E2E run completes, run updated live Report Evals:

   ```text
   DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python -m pytest -p no:rerunfailures tests/evals/test_report_eval.py --run-eval-live --eval-tag llm-executive-summary-report -q
   ```

   Inspect all four persisted metric scores and diagnostics, not just pytest exit status: existing Report Evals intentionally record low scores as diagnostics. Verify every expected case ran and actual LLM summary generation occurred.

5. Resolve change-related failures, rerun affected checks, and report concrete pass/fail/error/skip counts, evaluation run identifiers, notable metric changes, and remaining limitations. If external services block execution, record the exact blocker and still attempt independent requested checks; never label skipped or failed live execution as successful.

## Completion criteria

- The final report begins with an LLM-authored Executive Summary when generation succeeds, covers all requested categories, and retains the table below it.
- All summary facts reconcile with authoritative state and the detailed report, including uncertainty and new-discovery lineage.
- Existing sections and public remediation behavior remain intact.
- Updated Report Evals use the same four metrics and exercise the new generation path.
- Full offline verification, the live ten-case E2E golden suite, and then live Report Evals have been executed and their real results documented, or concrete external blockers are explicitly reported.

## Baseline observations before implementation

- Focused baseline command: `.venv/bin/python -m pytest -p no:rerunfailures tests/test_report_node.py tests/test_report_persistence.py tests/test_report_eval_metrics.py -q`, with telemetry disabled. Result: **51 passed, 1 failed**. The existing failure is `test_summarization_metric_receives_case_specific_questions`: it attempts to monkeypatch `test_report_eval.GEval`, which no longer exists after the suite switched to `build_judge_score_metric`. Repair this stale test boundary as part of updating Report Evals; do not classify it as a newly introduced report regression.
- `orchestration/graph.py` finalizes the report before recording `phase5.root_output` and exporting the trajectory. This is the correct integration boundary for keeping final report evidence, state, and persisted output aligned.
- `orchestration/portfolio_orchestrator.py` updates a group's nested `issues` when preparing coordinated candidates; this is not itself a write to top-level `state.issues`. Verify top-level write paths separately before deciding whether a new baseline snapshot is needed.
- Live prerequisites: an API credential is configured, and the Juice Shop, PyGoat, and Vulnado source clone directories are present. The initial sandbox probe could not access the Docker socket; a scoped elevated probe later confirmed Docker 29.2.1 and enabled the authorized live E2E run. Credentials were checked only for presence.

## Implementation and verification record

The report now renders a first `## 1. Executive Summary` section, followed by the existing metrics table. A typed evidence projection derives original, accepted-remediation, follow-up, and newly detected findings/groups from normalized report state; unknown identities and incomplete coverage remain explicit. Special-action outcomes are supported by current QA, committed attempt identity, and retained final patch evidence. The LLM selects fact-specific natural sentence starters from a closed structured schema; the renderer owns all counts, identities, outcomes, and evidence-limit wording. One bounded repair is permitted for missing or repeated fact references or a lead-in that does not match its fact/evidence, and deterministic fallback status is recorded separately. Sections 2–4 were independently compared with the unchanged renderer for 14 expanded golden states and matched byte-for-byte.

The final focused report, persistence, summary projection/generation, report metric adapter, metadata compatibility, and Report Eval contract checks passed: **104 passed**. An independent run of the full eval-persistence contract file added **15 passed** using an isolated temporary database and cache. The fixture-authored judge evidence explicitly describes canonical CVE/GHSA aliases, unknown counts when final scan evidence is absent, no-special-action cases, dependency sections and parent/target identity, and every attempt ID and outcome. DeepEval metadata reads support both legacy `additional_metadata` and current serialized-case `metadata`; persisted status/fingerprint coverage was confirmed for all final report cases. Changed-file `ruff check`, `ruff format --check`, and `git diff --check` passed. A configured-model smoke on the API-migration fixture returned `generated` from `gpt-5.4-nano-2026-03-17`, produced 817 characters, and differed from the deterministic fallback.

The final exact-code offline suite reported **2358 passed, 142 skipped, and 82 failed**. Every current failure ID was matched to the isolated unchanged-HEAD failure set, four independently verified Phase 5 failures, or the separately reproduced CP-SAT timing diagnostic; there were **zero unexplained or newly introduced failure IDs**. The full suite is not green. The original focused baseline was 51 passed and one stale `GEval` monkeypatch failure; the final focused set repairs that boundary.

The first complete live E2E cohort, run `run_20261030_120847_aa8b52` (`llm-executive-summary-e2e`), selected all ten goldens and finished with six passed, three failed, and one error. The failures were the preexisting npm code-migration case, npm single-update (20 current-attempt package tests failed after the requested update), and Python single-update (pip setup failed before mutation/QA); the mixed-strategies case retained an incomplete Task Completion score of 0.50 and had a System Safety judge transport error. A separate Safety-only rescore of the unchanged mixed-case evidence, `run_20261030_045229_c007a7` (`llm-executive-summary-e2e-mixed-safety-recheck`), scored 0.92 against the 0.70 threshold; it preserves the original Task Completion 0.50 and does not represent a fresh remediation run. A targeted two-case recheck, `run_20261030_123006_554433` (`llm-executive-summary-e2e-targeted-single-recheck`), passed both npm and Python cases: Task Completion 1.00 for each, with System Safety 0.88 and 0.94. This selected rerun occurred before the final structured fact-reference boundary adjustment; the final boundary is covered by focused tests and the actual-model smoke, and no broad no-regression claim is made. Historical E2E baselines include seven passes and three failures among the same ten cases; the original cohort's additional single-update outcomes are not represented as proven regressions because the targeted rerun passed.

The first live Report Eval run, `run_20261030_131225_1a2408`, had 30 passed and one timed-out test. Its score diagnostics exposed missing source facts for judges and metadata stored in DeepEval 4's current serialized-case field; these evidence and compatibility issues were corrected before the final full suite. Final run `run_20261030_135928_2b3308` (`llm-executive-summary-report-final`) completed **33/33 pytest items passed** in 2,115.97 seconds with a bounded 360-second per-task timeout. All **14/14** reports were generated by `gpt-5.4-nano-2026-03-17`, had no generation errors, and their source fingerprints matched the current expanded fixture states. The persisted database contains all **56** requested metric records with no judge execution errors:

| Metric | Count | Mean | Minimum | Below threshold |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness (0.85) | 14 | 0.9935 | 0.9091 | 0 |
| Hallucination (0.70) | 14 | 0.9286 | 0.0000 | 1 |
| Report Constraint Adherence (0.70) | 14 | 0.9064 | 0.8200 | 0 |
| Summarization (0.80) | 14 | 0.8545 | 0.7059 | 3 |

Scores remain diagnostic; thresholds and gating behavior are unchanged. The saved `data/evals/llm-executive-summary-validation/report-final-evidence-audit.json` records the remaining low-score reasons: one Hallucination judge interpreted the existing table's “Known new vulnerability identifiers: None” as a zero claim despite unknown final scan coverage; three Summarization cases alleged unsupported claims or omitted failed-attempt details (all-failed, multi-manifest, and PyJWT). Sections 2–4 and the existing table were not rewritten to chase judge scores. A persisted API-migration report and summary excerpt are available at `data/evals/llm-executive-summary-validation/api-migration-final-report-example.md` and `data/evals/llm-executive-summary-validation/api-migration-executive-summary-example.md`. Report duration and token rows continue to describe remediation only; they exclude separate summary-generation and eval-judge usage.

## Natural-language summary refinement and final live verification

The follow-up wording change removes category-label-and-colon framing, asks for connected narrative, and bolds all renderer-owned numeric counts. A live run exposed a real limitation in free-form sentence starters: one retry repeated an unsupported factual opening for `python-report-atomic-batch-success-with-warning`, causing fallback. The schema now restricts each fact to a small, fact-specific set of natural openers, selects new-count wording according to evidence completeness and scan scope, and rejects any other phrasing before rendering. The renderer explicitly describes complete incremental results as counts from the incremental dependency scan and states when coverage is incomplete or unavailable. Special-remediation details and all factual values still come from the final normalized state. Sections 2–4 and the table retain their existing format.

The final live E2E golden run `run_20261031_014146_911066` (`llm-executive-summary-natural-prose-e2e-verified`) completed **47/47 pytest items passed** in 3,906.46 seconds. All ten live goldens passed: npm single, batch, transitive, migration, and no-fix removal; mixed strategies; Python single and batch; Java single and batch. The persisted judge metrics were Task Completion mean **0.806** (2/10 below 0.70) and System Safety and Contract Adherence mean **0.854** (1/10 below 0.70). Judge thresholds remain diagnostic and were not changed.

After the E2E suite, the unrestricted lead-in version of Report Evals found one live-generation failure on the atomic-batch case. Replacing it with the fact-specific closed choices fixed the failure in a targeted live check. The final scoped Report Eval run `run_20261031_040009_6ca027` (`llm-executive-summary-natural-prose-report-final-scoped`) completed **34/34 pytest items passed** in 2,170.54 seconds. All **14/14** report summaries were actually generated by `gpt-5.4-nano-2026-03-17`, with 56 metric rows and no fallback errors. Its unchanged metrics were:

| Metric | Count | Mean | Minimum | Below threshold |
| --- | ---: | ---: | ---: | ---: |
| Faithfulness (0.85) | 14 | 0.9889 | 0.9167 | 0 |
| Hallucination (0.70) | 14 | 0.4286 | 0.0000 | 8 |
| Report Constraint Adherence (0.70) | 14 | 0.8629 | 0.7200 | 0 |
| Summarization (0.80) | 14 | 0.7894 | 0.6000 | 6 |

Metric scores remain non-gating diagnostics; the deterministic report contracts and actual-generation coverage passed. A final targeted live check for the two incremental-scan cases and the repaired atomic-batch case completed **4/4 selected pytest items passed**, and confirmed the generated text says the incremental dependency scan recorded its scoped counts. The focused local suite completed **97 passed**; changed-file Ruff lint and formatting checks passed. The lower Hallucination and Summarization scores are recorded rather than hidden: judges still flag existing “None” wording in the unchanged findings table when scan evidence is unknown, and omissions or unsupported details in a few whole-report comparisons. No metric or threshold changed.
