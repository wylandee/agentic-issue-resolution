# UI, language support, and live evaluation merge plan

Created 2026-11-01 for `feat/add-end-to-end-eval`.

Execution evidence and current results are tracked in [merge-ui-e2e-validation.md](merge-ui-e2e-validation.md).

## Objective and baseline

Complete the existing merge of `ui/create-demo-ui` into the current branch, preserving the current branch's language handling, supervisor-owned fix planning, executive summaries, and ten live end-to-end goldens. Integrate the incoming npm installation and installed-package evidence fixes, make every active engine LLM and the DeepEval judge support the requested Luna configuration, validate the demo UI, and execute all ten fresh live goldens.

This section records the original planning snapshot: no source, tests, configuration, or live prerequisites had been changed or verified at that point. Execution is now complete; actual results, source revisions, and remaining gaps are in the linked validation report. The plan itself is not evidence that validation passed.

- Working checkout: `/Users/wylandee/Documents/Work/agentic-issue-resolution`.
- HEAD: `dbfa99a` (`refactor: add executive summary for report`).
- Prior retained commits: `84be56a` (move fix planner into supervisor), `3520e67` (end-to-end goldens), and `4ddafce` (Luna judge support).
- MERGE_HEAD and incoming branch: `3cf37bd`; incoming commits since the existing common history include `19efcce` (Luna support), `929c1ce`, and `3cf37bd` (transitive dependency fixes).
- Seven unmerged paths; other incoming changes are already staged. Preserve those changes and review their semantic integration rather than selecting an entire side.
- Use the current checkout because its index contains the active merge. Record HEAD, MERGE_HEAD, status, staged diff, and unmerged index stages under ignored `data/` before edits. Do not abort/reset the merge or replace unrelated local changes.
- Read and follow `AGENTS.md`. The task queue, committed attempts, current attempt IDs/revisions, supervisor authorization, task-specific scan evidence, rollback semantics, host isolation, and workspace volume cleanup remain required contracts.

## Requirement coverage and evidence

| Requirement | Observed implementation | Required action |
| --- | --- | --- |
| Supervisor owns fix planning | `orchestration/supervisor_node.py:3437` invokes `prepare_supervisor_fix_plan_state`; `orchestration/supervisor_fix_plans.py:26` calls the advisory `tools.fix_planner.plan_fix` helper | Preserve the HEAD ownership and refresh/cache logic. Do not restore a standalone planner graph node or planner work in triage. The helper's file under `tools/` is compatible with supervisor ownership. |
| LLM Executive Summary and Report Evals | `report_summary.py:582`, `report_node.py:2821`, `graph.py:1216`; `tests/evals/test_report_eval.py:47` uses format v4, `Executive Summary`, and checks actual live generation at approximately line 1420 | Retain this implementation and its independently authored golden evidence. Migrate the production summary client to the Luna-compatible factory. |
| Ten E2E goldens and evaluation | `evals/end_to_end_goldens.py:33` requires exactly ten cases; `tests/evals/golden/end_to_end_cases.json`; `tests/evals/test_end_to_end_eval.py:161` runs selected goldens | Preserve dataset, fresh-run path, task/attempt contracts, persisted results, and judge adapters. Run all ten after merge completion. |
| UI | `demo/backend`, `demo/frontend`, `demo/start_demo.sh`; evaluation UI also exists under `evals/ui` | Preserve both UIs and validate the demo integration. Fix hardcoded JavaScript in `demo/backend/engine_bridge.py:700`. |
| JS, TS, Java, Python | `language.py:13` defines Node/Java/Python; Node suffixes include `.js`, `.ts`, `.tsx`, `.jsx`; language-aware installer exists on HEAD | Follow HEAD language detection, images, manifests, validation, tools, candidate policy, and QA. TypeScript stays within the Node profile. |
| Incoming stale node_modules fix | Incoming `_run_install` removes validated adjacent `node_modules/.package-lock.json`; staged `_qa_runtime.py:2436` / approximately 3187 query installed `npm ls` without `--package-lock-only`; `sandbox_mgr.py:46` pins npm | Combine the installation fix with HEAD language support and retain current-attempt installed dependency verification. |
| Luna models and efforts | Staged `llm_client.py:13` uses Responses and omits temperature for Luna; settings add five effort fields. HEAD judge already has a bounded Responses path for `max`. Summary still directly uses `ChatOpenAI` at `report_summary.py:626` | Complete client coverage, preserve call bounds, distinguish supervisor effort from report/default effort, configure local `.env`, test actual resolved settings. |
| Clone preparation and live runs | Local Juice Shop exists; local PyGoat/Vulnado are missing. Both requested directories exist at the supplied old worktree | Copy both directories, prepare dependencies/images/scanner services, then execute and audit all ten runs. |

Paths in the tables are relative to `src/remediation_engine/` unless they begin with `tests/`, `demo/`, or another repository directory. Line numbers describe the conflicted planning snapshot and will shift during resolution.

## Phase 0: Preserve the merge and establish validation boundaries

1. Capture the merge baseline and save the combined conflict diff under ignored `data/evals/merge-ui-e2e/`. Record the seven conflict paths and existing staged changes. Do not capture `.env` contents or secrets.
2. Confirm the local `.venv` imports this checkout rather than another editable installation. Install this checkout's `.[dev,eval]` requirements if needed. Use workspace-local temporary directories for tests.
3. Keep runtime files, clones, model credentials, generated reports, and trajectories ignored. Use explicit paths when staging the merge; never stage `.env` or `data/`.
4. Source files currently contain conflict markers, so a full pre-edit baseline cannot run. Historical documentation records prior unrelated suite failures; those are context, not proof that a current failure is preexisting. Reproduce unrelated failures against an isolated HEAD snapshot if necessary.

Acceptance: reproducible merge baseline, local interpreter/dependency plan, and no loss of existing work.

## Phase 1: Resolve the seven conflicted files

| File and conflict | Resolution |
| --- | --- |
| `orchestration/qa_critic.py`, `_run_global_execution` signature and install call around lines 250–301 | Retain both `project_language` and keyword `package_lock_paths`. Preserve language-specific install logging. Pass changed npm locks into the Node install path, and preserve Java/Python dispatch and package-state collection. Thread both arguments from `run_qa_critic_node`; do not silently drop either. |
| `orchestration/qa_test_parsing.py`, imports and `_run_install` around lines 203–315 | Keep HEAD `_python_install_command` and language-aware installation. Add incoming normalized, shell-quoted changed-lockfile handling only to the Node path. Use `npm install --package-lock=true` and remove only each changed lock's adjacent hidden npm metadata before installation. Preserve nested lock paths, bounded output/log records, errors, Python venv/Pipenv behavior, and Maven commands. Do not replace the fix with a blanket deletion of all dependencies. |
| `orchestration/qa_policy_engine.py`, `_evaluate_policy_gates` around lines 238–316 | Keep HEAD's tri-state scan coverage semantics: missing task projection cannot clear a task; incomplete evidence is inconclusive rather than clean. Add `_package_state_checks_enabled(policy)` and incoming version-bump dependency evidence gates. Retain both scan-incomplete diagnostics and package-state diagnostics. Remove unused `target_scan_complete` if HEAD's explicit coverage state supersedes it. Preserve language-specific dependency evidence rather than requiring npm evidence for non-Node tasks. |
| `orchestration/tactical_supervisor.py`, imports around line 43 | Keep HEAD Maven candidates/comparators, registry policy utilities, and `ProjectLanguage`; add `create_chat_model`. Keep supervisor action tools and authorization checks. Add the supervisor-specific settings described below to `_model_action`. |
| `orchestration/update_subagent.py`, imports around line 33 | Keep HEAD language profiles and Maven scope resolver; add `create_chat_model`. Verify both worker construction paths use the configured update effort without changing committed instructions or allowing registry queries. |
| `tools/fix_planner.py`, imports and `_llm_extract_remediation` around lines 46 and 881 | Preserve HEAD language/package identity and Maven version handling and supervisor-owned call sites. Resolve the schemas import to one consistent source. Use incoming shared model construction with the planner's existing triage model/effort settings, retaining the 30-second request timeout, disabled retries, injected constructor, and structured result validation. |
| `tests/test_targeted_qa_scans.py`, cleanup assertion around line 3154 | The shown test no longer binds a `cleanup` mock. Match the actual boundary: assert validated sandbox cleanup (`rm -rf -- .odc-targeted`, timeout 30), or deliberately inject the helper and assert its call consistently. Retain scan-evidence assertions. Do not leave an undefined `cleanup` variable. |

Review already merged files too: `_qa_runtime.py`, `sandbox_mgr.py`, `qa_evaluator.py`, `portfolio_orchestrator.py`, settings, triage, workaround worker, dependency metadata, and suppression fixture. In particular:

- Preserve installed-tree `npm ls --all --json` evidence; lock-only data cannot certify the installed transitive package.
- Preserve the incoming npm pin and failure cleanup, scoped to Node images. Verify image startup failure removes containers/temporary workspace volumes through existing lifecycle paths.
- Review installed transitive-parent version evidence against the actual executed candidate and committed attempt, including multi-package atomic batches.
- Retain QA task-specific scan projections and HEAD Python/Java execution paths despite automatically merged surrounding changes.
- Check the changed suppression fixture still selects intended vulnerable packages and does not hide the target of any maintained golden.
- Reconcile the duplicate `tomlkit` dependency entries while preserving the tighter existing requirement and the other language dependencies.

Acceptance: all seven files have coherent merged implementations; no conflict markers remain; all imported symbols and call signatures agree. Stage resolved paths explicitly only after review.

## Phase 2: Complete Luna configuration across engine and judges

Official OpenAI documentation confirms that `gpt-6-luna` supports `medium`, `xhigh`, and `max`. Reasoning with tools uses Responses, and reasoning requests omit unsupported sampling parameters. Sources: [Luna model](https://developers.openai.com/api/docs/models/gpt-6-luna) and [GPT-6 migration parameters](https://developers.openai.com/api/docs/guides/latest-model#update-api-and-model-parameters).

1. Keep one engine construction boundary in `orchestration/llm_client.py`. Extend its explicit interface for bounded timeout/retry/output settings where callers need them. Preserve the existing constructor injection seam. Do not permit arbitrary kwargs to overwrite model, API key, endpoint selection, or validated reasoning accidentally.
2. Add optional `SUPERVISOR_LLM_MODEL` and `SUPERVISOR_LLM_REASONING_EFFORT` overrides in `AppSettings`, following existing per-node patterns. Existing deployments without these overrides retain the old supervisor model/effort behavior. Keep the global/default workload at `medium`; the explicit supervisor override supplies `xhigh`.
3. Update `_model_action` to resolve the supervisor override. Keep planner advisory calls on the triage role at `medium`, even though the supervisor invokes them.
4. Migrate `report_summary.generate_executive_summary` to the shared factory with global/default `medium` effort. Retain source fingerprints, closed fact-reference schema, renderer-owned values, one bounded repair, and explicit fallback metadata. Its current 550-token total budget and 30-second timeout were chosen for a different model: verify whether Luna reasoning consumes that budget and adjust the bounded generation budget/timeout if the smoke proves it necessary. Keep rendered prose length bounded independently. Do not accept fallback as successful live-generation coverage.
5. Audit every active `ChatOpenAI`/shared factory call: initial/post-QA triage, both update paths, workaround, QA evaluator, tactical supervisor, fix-planner advisory extraction, and report summary. Optional solver/portfolio LLM configuration must also inherit the requested model/medium effort if an active construction path exists; do not invent a new LLM path for deterministic portfolio functions.
6. Preserve HEAD `evals/judge.py` bounded Responses implementation for max-effort Luna, sync/async schemas, nested Verdicts JSON handling, cost/usage accounting, and schema-scored metrics that avoid unavailable log probabilities. Do not reintroduce G-Eval's raw-logprob requirement. Verify the installed DeepEval version accepts the model and that dashboard/CLI/pytest all resolve the same configured judge.
7. Update `tests/conftest.py` environment isolation and relevant eval configuration capture for any new settings so local `.env` cannot leak into offline tests. Validate effective runtime settings without printing keys.
8. Update only the required non-secret local `.env` keys while preserving unrelated settings and credentials. Ensure duplicate keys or inherited shell overrides do not defeat the effective role mapping. Document the new optional supervisor settings in `.env.example` and configuration docs; preserve existing public env names and backward-compatible defaults.

Required local `.env` mapping:

```dotenv
REMEDY_LLM_MODEL=gpt-6-luna
REMEDY_LLM_REASONING_EFFORT=medium
SUPERVISOR_LLM_MODEL=gpt-6-luna
SUPERVISOR_LLM_REASONING_EFFORT=xhigh
TRIAGE_LLM_MODEL=gpt-6-luna
TRIAGE_LLM_REASONING_EFFORT=medium
UPDATE_LLM_MODEL=gpt-6-luna
UPDATE_LLM_REASONING_EFFORT=medium
WORKAROUND_LLM_MODEL=gpt-6-luna
WORKAROUND_LLM_REASONING_EFFORT=xhigh
QA_LLM_MODEL=gpt-6-luna
QA_LLM_REASONING_EFFORT=xhigh
SOLVER_LLM_MODEL=gpt-6-luna
EVAL_JUDGE_MODEL=gpt-6-luna
EVAL_JUDGE_REASONING_EFFORT=max
```

`SOLVER_LLM_MODEL` is an existing optional setting, not a request to enable solver LLM assistance. If an active solver model constructor needs its own effort field, add and validate `SOLVER_LLM_REASONING_EFFORT=medium` through `AppSettings`; otherwise it remains a deterministic subsystem. Report generation uses the global model/medium effort and needs no new report role.

Acceptance: model-construction tests inspect endpoint and effort for every role; invalid efforts fail early; existing supported non-Luna clients retain their behavior; required bounds persist. Actual-model smoke proves structured supervisor tool calls, a generated report summary, and a max-effort judge response work before launching ten long runs.

## Phase 3: Preserve language and UI behavior

1. Follow `feat/add-end-to-end-eval` language detection and execution throughout API, CLI, workspace setup, manifest tools, version policy, workers, QA, and reports. Preserve Node/TS manifest support, Maven coordinates/version ordering, and Python normalized package names and supported manifests.
2. Replace the JavaScript literal in `demo/backend/engine_bridge.py::_execute_run_thread` with the existing language-detection contract or API auto-detection. Inspect API behavior first and avoid adding another language detector. Attach Java, Python, and Node repositories to the correct profile; keep TypeScript as Node.
3. Preserve the UI's repo attachment, findings filters/reconciliation, SSE console updates, report availability, Markdown Executive Summary rendering, and unified patch display. Update any telemetry mappings that still expect a standalone fix-planner graph node.
4. Keep the evaluation dashboard's end-to-end suite and live/golden selection functional with the retained HEAD evaluation contracts.
5. Install demo backend dependencies, run `npm ci` in `demo/frontend`, build with its existing TypeScript/Vite build script, and smoke the local backend/frontend. Use mocked engine boundaries for deterministic backend tests; use the authorized live goldens for engine execution.

Acceptance: demo builds and serves; attachment detects all supported profiles; SSE/status/findings/report/patch flow works; report summary renders; UI integration retains the public engine API and host repository isolation.

## Phase 4: Offline verification and complete the merge

Use `.venv/bin/python`, `.venv/bin/ruff`, `DEEPEVAL_TELEMETRY_OPT_OUT=YES`, and a workspace-local `--basetemp`. Add focused tests only where behavior or merged interactions require protection.

Focused areas:

- `tests/test_settings.py`, `tests/test_llm_client.py`, `tests/test_tactical_supervisor.py`, `tests/test_fix_planner.py`, `tests/test_supervisor_fix_plans.py`.
- `tests/test_targeted_qa_scans.py`, `tests/test_qa_policy.py`, `tests/test_qa_odc.py`, relevant QA and sandbox tests: stale hidden npm locks, nested lock paths, actual installed transitive versions, rejected paths, incomplete scans, cleanup, and non-Node dispatch.
- `tests/test_language_detection.py`, `tests/test_language_agent_parity.py`, Java/Python supervisor/manifest tests, `tests/test_demo_dashboard_telemetry.py` and focused UI language integration coverage.
- `tests/test_report_summary.py`, `tests/test_report_node.py`, `tests/test_report_persistence.py`, `tests/test_report_eval_metrics.py`; Report Eval format/generation metadata and independent golden facts.
- `tests/evals/test_end_to_end_goldens.py`, `tests/evals/test_end_to_end_eval.py`, judge/settings/persistence contracts and schema compatibility tests.

Then run the full offline suite, `ruff check .`, `ruff format --check .`, `git diff --check`, and `.venv/bin/remedy --help`. Resolve introduced failures. If broad checks reveal unrelated preexisting issues, establish their baseline explicitly and record them; do not weaken unrelated contracts or claim the suite is green.

Review the complete staged merge, confirm no unmerged index entries or conflict markers, and confirm `.env` and runtime outputs are excluded. Complete the existing local merge with a merge commit after focused checks and UI build are sound. Record its SHA for live run provenance. No remote push or publication is required.

Checkpoint: retain saved merge inputs and use focused follow-up commits for fixes found by live verification. Avoid resetting a completed merge or discarding the user's work.

## Phase 5: Prepare clones and live services

The source root explicitly supplied by the user is `/Users/wylandee/.omp/wt/wt-20260928-120732-a8f1544/data/clones`.

1. Copy `vulnado` and `pygoat` from that root into this checkout's `data/clones/`, preserving repository content and metadata. Both local destinations were absent during planning; recheck before copying and preserve any newly present local content. Do not modify the source worktree. Validate destination manifests, git status/revision, and fixture package identities. Keep copied repositories ignored.
2. Confirm existing `data/clones/juice-shop` and its fixture-compatible revision. Snapshot host clone status and relevant file hashes before runs. Golden suppression staging temporarily writes a case-specific suppression file and restores its bytes/mode in `finally`; preserve and verify this existing harness behavior. The engine itself only mutates Docker workspaces.
3. Install this checkout's `.[dev,eval]` and demo dependencies, if not already done. Confirm imports resolve here and optional DeepEval/OpenAI dependencies are compatible.
4. Check Docker daemon access, disk capacity, the Node 22 and Maven 3.9/JDK 17 images, Dependency-Check image/cache/data freshness, and npm/PyPI/Maven network access needed by the actual pipeline. Build the Python image with:

   ```sh
   docker build -f docker/python-workspace.Dockerfile -t remediation-engine-python:3.11 .
   ```

5. Check credential presence and actual model access without logging secrets. If Docker socket/network access is denied by the sandbox, request only the scoped escalation required by the authorized live run; report an actual auto-review denial if one occurs.
6. Load all ten golden cases and validate fixture selection/suppression generation. Let `golden_settings` preserve the established golden execution policy. Do not reuse generated outputs as new ground truth, disable current-attempt checks, or modify expectations to manufacture success.

Acceptance: all three destination clones exist, fixture inputs resolve, environment/model smoke works, language images and scanner are usable, and host clone baselines are recorded.

## Phase 6: Run all ten fresh live goldens and inspect evidence

Run from the merged checkout with local `.env` loaded by the existing evaluation boundary:

```sh
DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python -m pytest -p no:rerunfailures \
  tests/evals/test_end_to_end_eval.py --run-eval-live --e2e-golden all \
  --e2e-strict --eval-tag merge-ui-e2e-luna-max \
  --basetemp=data/tmp/merge-ui-e2e-live -q
```

Run the existing sequential cohort rather than concurrent fresh mutations of a shared clone. Stream output to an ignored log, poll long-running commands, and keep the user informed. Validate pytest collection/options first if plugin/version differences require an equivalent command. Do not silently drop any live case.

| Golden ID | Scenario |
| --- | --- |
| `e2e-npm-single-update` | `@tootallnate/once` |
| `e2e-npm-batch-update` | Angular common/compiler/core |
| `e2e-npm-transitive-parent-update` | `crypto-js` through authorized parent |
| `e2e-npm-code-migration` | `express-jwt` with code migration |
| `e2e-npm-no-fix-removal` | `notevil` removal and required code edits |
| `e2e-npm-mixed-strategies` | Angular, express-jwt, and notevil |
| `e2e-python-single-update` | `pyyaml` |
| `e2e-python-batch-update` | `certifi`, `sqlparse` |
| `e2e-java-single-update` | `org.postgresql:postgresql` |
| `e2e-java-batch-update` | Logback classic/core |

For every case, record its fresh source-run ID, evaluation-run ID, merged code SHA, fixture hash, actual model and role efforts, duration, final task states, current-attempt worker/QA evidence, patch/report/trajectory locations, actual summary generation status, and both judge scores/reasons. The maintained metrics are Task Completion and System Safety and Contract Adherence, each threshold 0.70. Strict mode makes score failures visible; inspect deterministic remediation outcomes too because non-strict PASSED alone does not prove completion.

Audit all ten cases rather than relying on aggregate pytest output: no live skips, missing metric rows, stale attempt passes, unsupported target versions, incomplete task scan evidence claimed clean, missing patch evidence, silent summary fallback, host clone mutations, or leaked temporary Docker workspace volumes. Retain the persistent scanner cache; cleanup assertions concern run-owned temporary resources.

Fix change-related failures and rerun affected cases. Preserve original failed/error records and identify each rerun separately. The initial ten-case cohort completed on the merged revision; subsequent confirmed QA-prompt authority and report/runtime changes require corrected-source scoped coverage. The parent-approved scope is the Angular batch (its recorded judge score is complete), mixed-strategies (completed with a strict Task Completion failure), code migration, Python single and batch, Java single and batch, plus the updated Report Eval suite. All selected work is complete; migration and Java batch original Safety timeouts have separate successful metric-only recorded retries. The original ERROR records remain unchanged, and the mixed-strategy completion gap remains disclosed. Keep the parent-floor strict recheck and initial ten-case results attributed to their own revisions. Do not claim that all ten cases validate the final source unless a full final-source cohort actually runs. Do not lower judge thresholds, weaken safety contracts, or rewrite golden expectations merely to pass. If a failure is a judge transport/service issue or a disputed low score, retain the evidence and distinguish that from engine completion; report unresolved failures explicitly.

External-evidence authorization: the user explicitly authorized sending repository-derived report, patch, and trace evidence to `gpt-6-luna` at `https://api.openai.com/v1` for the live evaluations on 2026-11-02 at 15:24 SGT. The earlier automatic-review rejection and sandbox transport error remain preserved as historical failed attempts. Run the authorized evaluations sequentially, each with an isolated database and log; do not use another endpoint to bypass this scope.

After the ten-case run, execute updated live Report Evals to verify requirement 2 on Luna:

```sh
DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python -m pytest -p no:rerunfailures \
  tests/evals/test_report_eval.py --run-eval-live \
  --eval-tag merge-ui-e2e-luna-report \
  --basetemp=data/tmp/merge-ui-e2e-report -q
```

Retain the four Report Eval metrics and their current scoring/gating behavior: Hallucination 0.70, Faithfulness 0.85, Summarization 0.80, and Report Constraint Adherence 0.70. Verify 14 actual generated summaries and all 56 metric records; diagnostic low scores must be reported even if pytest passes. Keep independent source facts separate from format policy and do not treat fallback summaries as live generation.

## Completion and execution handoff

Execution owner: a `gpt-6-luna` subagent with `max` reasoning, working in this same checkout after this plan is saved. The user has authorized implementation, model configuration, copying the two clones, dependency/setup work, merge completion, and live evaluations. Proceed through these phases without asking for routine reconfirmation. Follow repository and tool approval boundaries if a concrete blocker arises.

Done for implementation requires: the merge is complete, all seven conflicts are resolved, supervisor-owned planning and executive summaries remain functional, JS/TS/Java/Python and UI language selection work, the incoming npm fix is covered, effective `.env` settings match the requested mapping, offline/UI checks are documented, and the merged-revision ten-case cohort plus corrected-source scoped verification are reported with real persisted evidence where available. Report the initial cohort, interrupted follow-up, selected corrected-source cases, and report generation separately; never describe historical, skipped, failed, unscored, or merely planned cases as successful verification. Provide concise pass/fail/error/skip counts, run IDs, both metric results when persisted, summary-generation coverage, artifact paths, and any remaining external authorization or service blockers.


### Execution closeout

Implementation and live execution were handed to `gpt-6-luna` with `max` reasoning in this checkout. Merge commit `43e7a42` and follow-ups through evaluator source `ed6d72c` are complete. The initial ten-case cohort ran; selected corrected-source checks and 14 live Report Evals also completed. Two Safety-only recorded retries passed without rewriting original timeout records or claiming whole-case reruns. The no-fix/mixed expectations, 81 baseline offline failures, and non-gating report quality diagnostics remain visible in the validation report. Final documentation was consolidated by the parent agent after live work completed. No source changes, live reruns, remote push, or publication were added during closeout.
