# Codebase Refactoring Implementation Plan

Prepared 15 October 2026 against commit `ae6b80c` and the existing dirty working tree.

## Executive Summary

The engine currently combines typed ingestion and triage, Supervisor-owned portfolio planning and attempt commits, isolated workers, deterministic QA execution, incremental scan reconciliation, and patch/report generation. The highest-risk findings concern accepting a worker envelope under the wrong attempt key, losing task diagnostics through replacement reducers, claiming successful candidate restoration after a destructive restore failure, and discarding local rollback evidence before restoration succeeds.

At review time, the graph already had a setup scan, invoked portfolio planning inside the Supervisor, and used incremental default final evidence with a full scan opt-in. Several documents then described an outer portfolio graph node, target-only scans without dependency edges, or mandatory final full scans. Those baseline descriptions were corrected without changing runtime behavior.

Implement in small checkpoints: protect the required fixture outcomes; reconcile Markdown; repair demonstrated invariant defects; remove proven obsolete private code; deepen the rollback and evidence boundaries only where this reduces duplicate policy. Preserve public imports, configuration names, serialized contracts, tool schemas, prompts' behavioral instructions, retry budgets, solver ordering, batching, and successful remediation behavior.

## Implementation Progress Ledger

Updated 15 October 2026. The read-only audit baseline above remains a historical record; implementation changes are now present in the shared working tree.

| Workstream | Status | Evidence / remaining gate |
|---|---|---|
| Task 1: Markdown truth reconciliation | **Complete** | All 17 pre-existing operational Markdown files were inventoried and reconciled; this implementation plan is the new 18th non-runtime Markdown file. Current engine claims were updated across `AGENTS.md`, `README.md`, `EVALS.md`, architecture/ODC/evaluation docs, Juice Shop and NodeGoat examples, six fixture READMEs, and the two engine-specific skills that needed edits. The generic `review-and-refactor` workflow remains unchanged because its instructions are still accurate. Mermaid/DOT generator and rendered diagrams now reflect setup scan, Supervisor-owned planning, optional final scan, report and teardown. `.env.example` was scrubbed; no secret values were inspected or printed, and no manual `.env` inspection was performed. Fixture runners retain normal `load_dotenv()` behavior. |
| Phase 0 offline behavior gates | **Mocked behavior gates complete; Gate A live pass; Gate B accepted by the user's explicit PasteBin exception** | Mocked workflow regressions protect the Angular task closure and four-package atomic failure/isolation/replay. The latest Gate A live run passed the nine-task one-batch verifier. Gate B reproduced pair dispatch, QA failure, candidate restore and jws-only blame; `got` and both esbuild occurrences passed QA. The raw checker flags base64url because one remote PasteBin integration test failed after its targeted scan cleared. The user explicitly directed that this isolated transient test failure counts as passed. No QA, fixture-checker, or remediation logic was changed to accommodate PasteBin. The raw checker result remains recorded below rather than being represented as a raw pass. |
| State, path, ingestion, triage, QA-target, rollback and lifecycle fixes | **Focused fixes implemented; 68 baseline failures remain** | STATE-01/02/03, RB-01/02/03/04, LIFE-01/02, QA-01/02, CFG-01/02/03, path validation, scanner ingestion, triage, ODC redaction, solver candidate completeness, required-peer constraints, evidence constraints, and Supervisor registry-error routing have focused changes. STATE-03 identity/revision rejection passed 3 new tests; the triage/path/ingestion/code-locator family passed 244 tests. Latest offline run (`-m "not integration"`, with the socket-dependent rerun plugin disabled): **1,769 passed, 68 failed, 89 skipped**. Exact node-ID comparison against the original 82-failure baseline found no new failing IDs and 14 original failures now passing. Remaining failures are grouped and individually listed below; the largest family calls `run_supervisor_node` directly with pre-portfolio or uncommitted attempt inputs. Focused QA, scan and fixture-checker tests pass; production QA offline replay passes 20 cases after its adapter was corrected to supply task-owned evidence. Ruff, formatting and compile checks are rerun at final verification. |
| ODC diagnostics redaction (SEC-01) | **Complete** | Redacts split/inline options in command logs, persisted process/cleanup diagnostics, timeout exceptions, summaries and QA records. Dummy-only regressions include quoted/backslash values and false-positive option-like words; `test_targeted_qa_scans.py` passes 38 tests and targeted Ruff checks pass. |
| Evaluation tooling | **Focused changes complete; full baseline remains** | EVAL-01/02/03 close DB connections, compare all cases by stable occurrence identity, clean up abandoned subprocesses and report typed exit status. Offline mode overrides ambient `EVAL_RUN_LIVE=true`; EVAL-04 distinguishes ERROR and derives status/suite filters from persisted values. The latest focused runs passed 17 DB/runner tests and 53 report/persistence/trajectory tests. `compileall` passed after the UI edits. |
| Dead-code/periphery findings and full baseline classification | **Targeted cleanup complete; facade simplification and the 68 baseline failures remain open** | DEAD-04 removed six unused eval metrics; `tests/evals/test_business_rules.py` passed 16 tests with 3 expected skips. DEAD-01/02 exact private planner cones were removed while active synthetic dependency handling and OSV/Serper/LLM paths remain. DEAD-03 removed duplicate phase-transition helpers after moving the substantive assertion to the active deterministic route/audit boundary; public `EligibleActions` and `ReconciliationResult` contract types remain. CLEAN-01/02 and RANGE-01 have focused coverage; TOOL-01 defaults resolve from repository root; TOOL-02 reports missing Graphviz. DOC-07 prose/comments describe committed atomic dispatch and scoped QA. ARCH-01 remains deferred because dynamic dependency facades preserve established import/monkeypatch seams; TYPE-01 remains a mutability capability without demonstrated production alias mutation. Fourteen original failing IDs now pass, including candidate-resource, terminal-task certification, exact no-good iteration, and registry-error routing cases. |
| Live acceptance | **Gate A verified pass; Gate B accepted with the explicit PasteBin exception** | Latest Gate A run `f9eb4d00-1f98-4d72-a332-a66c95adb233` expanded three requested roots into nine Angular mutation tasks; all nine lineages passed QA in one atomic dispatch. The runner exited 0 and the explicit verifier passed. Latest Gate B run `3dd77d3a-bcc8-41f0-b76d-a364094efa12` sent base64url+jws in one atomic batch, failed QA, restored the candidate, and identified only jws as responsible. `got` and both esbuild occurrence tasks passed QA as singletons; jws remained rejected/unremediated. Base64url's targeted scan cleared the target, but one test in `test/api/internet-resources.test.ts` failed while reaching PasteBin. The raw `verify_four_package_fixture` consequently reports base64url as not QA-passed. Per the user's explicit instruction, count this one transient PasteBin test failure as a pass for acceptance; do not change code logic or claim that the raw verifier passed. Run artifacts are under `/private/tmp/juice-gate-a-20261015-peerfix/` and `/private/tmp/juice-gate-b-20261015-peerfix/`. Host clone protection and cleanup checks completed; LangSmith tracing was disabled. |

Task 1 documentation reconciliation is complete. The Gate B security-attribution fix is present and the latest trace does not show the prior sibling-finding `security_flag`; the only base64url blocker in this run was the PasteBin integration test, handled under the user's explicit exception without a logic change. The latest offline run has 68 failing IDs, all present in the original baseline; 14 baseline failures now pass. Gate A was rerun after the solver changes. Gate B was also rerun; its raw base64url verifier failure is preserved as evidence and accepted only by user direction. Implementation remains in progress because baseline test migration and ARCH-01 review remain; TYPE-01 has no demonstrated production alias defect.

### Baseline failure triage

The latest offline run has 68 failures, all present in the original 82-failure baseline. This is a disposition ledger, not a claim that those tests or behaviors are fixed. No production guard was loosened to clear them. The node IDs below are individually classified by the stale setup/assertion they need migrated; each remains visible debt until rerun against current graph preconditions.

- `test_supervisor_node.py` (51), `test_deterministic_supervisor.py` (2), and `test_sca_strategy_refactor.py` (3) contain legacy direct-node assumptions. Many tests supply findings without the initialized task queue, accepted portfolio plan, selected candidate, or current attempt snapshot that the active graph commits before Supervisor routing. A direct reproduction with bare SCA groups returns an unknown portfolio and no worker targets; bare SAST groups with no task queue also do not dispatch. Other failures expect Supervisor to query/choose a version, patch the removed `supervisor_spawn.plan_npm_parent_version` seam, or require an internal stage name in worker instructions. These tests should be migrated to the current graph/portfolio boundary before they can prove a production routing defect. Do not relax dispatch validation to make them pass.
- `test_portfolio_orchestrator.py` (10) splits into three contract families. Five synthetic-task assertions expect triage's proposed version to be written into `selected_version` before an accepted plan, or expect a current resolved version despite a range-only manifest with no lockfile. Three Supervisor integration cases use incomplete task/plan fixtures and do not exercise the committed-plan route they assert. Two direct `apply_portfolio_plan` cases omit an accepted complete solver plan and are rejected before their narrower occurrence assertions. Candidate resource-bound and final-scan route cases that appeared in the original audit now pass focused regressions.
- `test_portfolio_certifier.py` (2 offline failures) expects candidate rejection in two fixtures where the selected candidate is certified immediately, so no rejection history is produced. The terminal-task fixture now models nested `jsonwebtoken` resolution, and the exact no-good test verifies both rejected complete assignments plus the final infeasible solve. The Docker-based strict-resolution test is explicitly marked integration and skips when the daemon or `node:22` image is unavailable.


### Per-node-ID disposition

These dispositions describe the next correction to each failing test's setup or assertion; they do **not** mark a failing test as passing. Re-run each case through the current graph/plan contract before concluding that its behavior is correct.

| Node ID | Disposition |
|---|---|
| `tests.test_deterministic_supervisor.test_deterministic_retry_planner_preserves_committed_stage` | DS-SETUP |
| `tests.test_deterministic_supervisor.test_deterministic_retry_planner_does_not_widen_approved_candidate_pool` | DS-SETUP |
| `tests.test_portfolio_certifier.test_exact_rejection_uses_next_optimal_candidate_without_resolving[True-new_vulnerable_copy]` | CERT-SETUP |
| `tests.test_portfolio_certifier.test_exact_rejection_uses_next_optimal_candidate_without_resolving[False-coverage_unresolved]` | CERT-SETUP |
| `tests.test_portfolio_orchestrator.test_missing_namespace_dependency_gets_a_coordination_task` | PORT-SYNTH |
| `tests.test_portfolio_orchestrator.test_unflagged_direct_dependency_gets_a_synthetic_task_without_a_seed` | PORT-SYNTH |
| `tests.test_portfolio_orchestrator.test_runtime_related_synthetic_package_keeps_its_resolved_version` | PORT-SYNTH |
| `tests.test_portfolio_orchestrator.test_synthetic_dependency_materialization_is_idempotent` | PORT-SYNTH |
| `tests.test_portfolio_orchestrator.test_synthetic_target_refresh_keeps_group_identity` | PORT-SYNTH |
| `tests.test_portfolio_orchestrator.test_supervisor_leaves_synthetic_materialization_to_outer_portfolio` | PORT-PLAN |
| `tests.test_portfolio_orchestrator.test_supervisor_commits_shared_atomic_provenance_for_cluster` | PORT-PLAN |
| `tests.test_portfolio_orchestrator.test_invalid_atomic_action_requests_portfolio_replan` | PORT-PLAN |
| `tests.test_portfolio_orchestrator.test_apply_rejects_a_mismatched_physical_lockfile_occurrence` | PORT-ASSERT |
| `tests.test_portfolio_orchestrator.test_scoped_transitive_findings_become_override_solver_tasks` | PORT-PLAN |
| `tests.test_sca_strategy_refactor.test_transitive_initial_planner_recovers_parent_version_from_group_evidence` | SCA-API |
| `tests.test_sca_strategy_refactor.test_qa_failure_advances_task_and_diagnostics_to_next_supervisor_stage` | SCA-API |
| `tests.test_sca_strategy_refactor.test_retry_instruction_contains_exact_selected_version_and_stage` | SCA-API |
| `tests.test_supervisor_node.TestRunSupervisorNodeNormalization.test_initial_dispatch_commits_policy_to_task_and_attempt_snapshot` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeNormalization.test_candidate_less_update_is_rejected_before_snapshot_creation` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeVersionBump.test_first_active_update_task_is_the_only_registry_target` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeVersionBump.test_version_bump_tasks_dispatch_one_at_a_time` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeVersionBump.test_deterministic_routing_caps_update_dispatch_at_one` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeWorkaround.test_code_workaround_deterministic_routes_one_task` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeToQA.test_optimistically_fixed_task_routes_to_qa` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeToQA.test_all_optimistically_fixed_tasks_route_to_qa_within_limit` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeToQA.test_current_task_routes_to_qa_before_more_updates` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeToQA.test_terminal_tasks_in_active_batch_are_omitted_from_qa_targets` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeToQA.test_optimistically_fixed_task_routes_to_qa_before_retry_planning` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_qa_failed_retry_uses_deterministic_planning` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_qa_completed_passed_marks_task_qa_passed` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_no_fix_package_removal_pass_reaches_qa_passed` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_qa_completed_passed_adds_constraint` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_qa_completed_passed_does_not_duplicate_existing_constraint` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_qa_completed_failed_marks_task_needs_retry` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_failed_task_keyed_qa_eval_replans_active_child_task` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_retry_update_without_revised_instruction_falls_back_to_high_level_retry` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_missing_retry_instruction_replans_before_dispatch` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeQAUpdates.test_pending_update_task_can_seed_instruction_from_revised_instructions` | SUP-INPUT |
| `tests.test_supervisor_node.test_exhausted_deterministic_guardrail_routes_workaround_child` | SUP-INPUT |
| `tests.test_supervisor_node.test_deterministic_latest_selection_skips_attempted_versions` | SUP-INPUT |
| `tests.test_supervisor_node.test_empty_same_major_stage_advances_or_pivots_without_unversioned_dispatch` | SUP-INPUT |
| `tests.test_supervisor_node.test_failed_update_attempt_is_closed_before_retry_planner_commit` | SUP-INPUT |
| `tests.test_supervisor_node.test_terminal_task_cannot_retain_retry_plan` | SUP-INPUT |
| `tests.test_supervisor_node.test_terminal_pivot_parent_is_not_reseeded_from_initial_osv_plan` | SUP-INPUT |
| `tests.test_supervisor_node.test_pivot_detaches_previous_update_attempt_before_child_dispatch` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeActionSummary.test_qa_passed_subset_is_removed_before_retry_routing` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeActionSummaryUpdates.test_updates_task_to_needs_retry_on_surrender` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeActionSummaryUpdates.test_does_not_update_non_active_targets_from_action_summary` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodeTargetGuardrails.test_llm_update_targets_drop_terminal_tasks` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorMaxRetries.test_max_retries_marks_task_unfixable_and_removes_from_targets` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorMaxRetries.test_qa_failure_at_retry_cap_pivots_before_terminalization` | SUP-INPUT |
| `tests.test_supervisor_node.test_newly_unfixable_sca_transition_replans_before_dispatch` | SUP-INPUT |
| `tests.test_supervisor_node.test_existing_unfixable_sca_does_not_trigger_another_replan` | SUP-INPUT |
| `tests.test_supervisor_node.test_pending_unfixable_replan_waits_for_open_attempt_reconciliation` | SUP-INPUT |
| `tests.test_supervisor_node.test_pending_unfixable_replan_routes_when_attempt_barrier_is_clear` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_workaround_spawn_with_missing_target_is_normalized_to_child` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_llm_strategy_pivot_spawns_child_and_routes_child` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_breaking_change_pivot_marks_parent_pivoted_and_routes_child` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_strategy_pivot_without_child_instruction_fails_closed` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_exhausted_update_path_fallback_spawns_child_and_terminalizes_parent` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_exhausted_update_at_retry_cap_pivots_before_terminal_cap` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_deterministic_guardrail_routes_exhausted_update_to_workaround` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorNodePeerConflict.test_deterministic_routing_ignores_multi_spawn_advisory` | SUP-INPUT |
| `tests.test_supervisor_node.TestRunSupervisorDeterministicRouting.test_version_bump_routes_without_a_supervisor_llm` | SUP-INPUT |
| `tests.test_supervisor_node.TestBugFixes.test_terminal_pivot_child_releases_other_pending_tasks` | SUP-INPUT |
| `tests.test_supervisor_node.TestBugFixes.test_bug4_programmatic_feedback_injection_for_workaround_retries` | SUP-INPUT |
| `tests.test_supervisor_node.test_atomic_cluster_without_delta_attribution_keeps_all_member_retry_behavior` | SUP-INPUT |
| `tests.test_supervisor_node.test_verified_absence_does_not_dispatch_remaining_members_while_worker_attempt_is_open` | SUP-INPUT |

Disposition codes:

- **DS-SETUP:** provide an explicit committed candidate pool and typed stage expectation to the deterministic retry-planner test; it must continue proving that retries do not widen authority.
- **CERT-SETUP:** construct candidate catalogs that actually produce the intended certification rejection, then retain assertions for no-good progression without adding resolver calls to immediately certified assignments.
- **PORT-SYNTH:** assert the prepared synthetic task separately from a solver-accepted version; provide complete manifest/lock evidence and test idempotent materialization at the current outer portfolio boundary.
- **PORT-PLAN:** provide an accepted, complete solver plan and current task revisions before asserting Supervisor dispatch, atomic provenance, replan requests, or transitive override tasks.
- **PORT-ASSERT:** retain the physical-occurrence rejection assertion but match the typed error contract and verify that application made no mutation.
- **SCA-API:** remove the obsolete `supervisor_spawn.plan_npm_parent_version` patch seam, use the current plan/task result shape, and assert the typed strategy stage rather than a retired internal label.
- **SUP-INPUT:** invoke the real graph/portfolio setup or provide its accepted plan, selected candidate, task revision, and current attempt snapshot before asserting routing or transitions. Preserve the fail-closed behavior for incomplete state.
Retain these 68 failures as visible review debt. Do not weaken current attempt, portfolio certification, candidate-authority, occurrence-identity, or rollback gates to satisfy a test whose input omits its production preconditions. Migrate legacy direct-node tests to committed graph inputs and revise the two certifier expectations against the catalogs those fixtures construct.

Implementation is in progress under the user’s authorization. Before each substantial work chunk and during long-running work, check the account's **300-minute window usedPercent**. At **98% or greater**, stop commands and agents, preserve unfinished work, report the checkpoint, and await the user's next instruction. Do not interpret this as a 98-point allowance from the starting percentage. Finding line references and dirty-file counts above refer to the reviewed baseline; recheck them against current files before relying on line numbers.

## Review Scope and Baseline

- Scope: 95 Python source files; 84 test Python files; repository-owned Markdown; example runners/fixtures; scripts; `pyproject.toml`; diagram sources; public configuration template. No `.github` directory exists.
- Exclude installed environments, dependency code, caches, ignored clones, generated scans, trajectories, runtime reports, and actual secrets. The clone is used only for explicitly authorized fixture execution during implementation.
- Python package: `src/remediation_engine`; Pydantic contracts, LangGraph orchestration, Docker SDK, requests, OR-Tools CP-SAT, LangChain/OpenAI adapters, optional SQLite/Streamlit evaluation tooling.
- Entry points: `api.py:run_remediation`, `api.py:triage_issues`, `cli.py:main`, `orchestration/graph.py:run_orchestrator`, solver public functions in `orchestration/__init__.py`, and the example scripts.
- `.venv` exists; `venv` does not. Use `.venv/bin/python` for the supplied commands and record this interpreter-path normalization; do not change their packages, repository scope, or expected results.
- Existing modifications must be preserved: `examples/juice_shop/fixtures/suppressed/{odc_suppressed_issues.jsonl,suppressions.xml}`, orchestration `{_qa_runtime.py,portfolio_solver.py,supervisor_node.py}`, runtime `sandbox_mgr.py`, and tests `{test_portfolio_orchestrator.py,test_sandbox_mgr.py,test_supervisor_node.py,test_targeted_odc_docker.py,test_targeted_qa_scans.py}`. Initial diff: 11 files, 571 insertions, 63 deletions. Record a new status/diff before implementation in case it changes.

### Completed baseline checks

| Check | Result | Implication |
|---|---|---|
| `.venv/bin/python -m pytest -q -m 'not docker and not network and not llm and not eval'` | Startup internal error: installed rerunfailures plugin tries binding a localhost socket denied by this environment | Use `-p no:rerunfailures` locally; do not rewrite project configuration to accommodate this unrelated installed plugin |
| Same command with `-p no:rerunfailures` | **1,663 passed, 82 failed, 142 deselected**, 2 warnings, 16.68 seconds | Baseline is not green; classify each failure and preserve its test intent |
| Failed-test rerun with `--lf --tb=short -p no:rerunfailures` | Same **82 failures** | Diagnostic log: `/private/tmp/remediation-review-baseline-failures.txt`; includes unmocked Docker calls and old standalone registry-planning assumptions |
| `.venv/bin/ruff check . --output-format concise` | 3 existing D103 errors at `scripts/generate_diagrams.py:18,49,91` | Fix actual missing docstrings |
| `.venv/bin/ruff format --check .` | 4 files need formatting: `workspace_builder.py`, `lockfile_closure.py`, `test_portfolio_orchestrator.py`, and a Python example in `docs/additive_targeted_odc_plan.md` | Keep formatting changes isolated and preserve dirty edits |

Tests that fail because a boundary is not mocked must gain a correct mock. Tests encoding an obsolete planner assumption must be moved to the current boundary with the same substantive contract. Do not delete or relax failing tests merely to produce a green suite. Solver/certification failures require separate diagnosis; they are not all presumed stale.

## Current Architecture

```text
API / CLI -> typed issues / groups
  -> initial_triage -> workspace_builder -> workspace_setup_scan
  -> Supervisor (portfolio preparation, solving, certification, task/attempt commit)
     -> update worker -> Supervisor
     -> workaround worker -> Supervisor
     -> QA execution/evaluation -> Supervisor
     -> post-QA triage -> Supervisor / portfolio replan
     -> optional final_full_scan -> Supervisor
  -> teardown / host-relative patch -> deterministic report -> result
```

The graph registrations at `graph.py:792–815` are authoritative for topology. Portfolio solving is a function/tool invoked by `supervisor_node.py:6068–6117`, not a registered graph node. The pure solver models assignments; certification owns scratch resolver evidence; the Supervisor decides which certified action becomes executable.

| Boundary | Ownership and dependencies |
|---|---|
| `contracts/{schemas,solver_models,decision_codes,supervisor_phases,accessors,version_policy}.py` | Typed findings, tasks, attempts, worker/QA envelopes, package occurrence and plan provenance; serialization is a compatibility boundary |
| `triage/*`, `tools/{odc_parser,semgrep_parser,manifest_locator,code_locator,fix_planner,reachability-related helpers}` | Finding normalization, localization, enrichment and advisory/workaround proposals; these do not authorize worker dependency mutations |
| `orchestration/state.py`, `task_utils.py`, Supervisor modules | Authoritative task queue, immutable attempt identity, retry/pivot/spawn policy, complete state projections and terminal lineage |
| `portfolio_{orchestrator,solver,certifier}.py`, `solver/*`, `tools/{npm_graph,registry_cache,registry_tools,lockfile_closure}.py` | Occurrence-aware graph, candidate domains, atomic clusters/DAG, pure assignments, resolver certification and approved mutation allowlists |
| `graph_wrappers.py`, `runtime/sandbox_mgr.py`, `tools_{manifest,edit,validation,workspace}.py` | Attempt/batch rollback anchors, cumulative dependency candidate preservation, temporary source edit rollback, Docker filesystem/cache lifecycle |
| `update_subagent.py`, `workaround_subagent.py`, `subagent_runtime.py`, toolbelts | Execute sealed instructions, bounded tool loops, typed execution evidence; workers do not acquire version-selection authority |
| `_qa_runtime.py`, `qa_{critic,evaluator,policy_engine,odc,test_parsing,types}.py`, `final_full_scan.py` | Install/scan/test execution, task/attempt evidence, evaluator proposals and deterministic gates; canaries attribute failure but do not confer QA success |
| `report_{context,diff,node,persistence}.py`, `trajectory_exporter.py`, `langsmith_config.py` | Evidence normalization, pure rendering, artifact persistence and optional tracing; rendering/telemetry must not mask the original run failure |
| `runtime/{path_policy,docker_client}.py`, `workspace_builder.py`, `teardown_node.py` | Host containment, shared-volume creation, container commands, diff and resource teardown; engine reads host repository without applying patches |
| `settings.py`, `runtime_context.py`, `api.py`, `cli.py` | Settings parsing at boundaries and per-run binding; credentials stay outside serializable orchestration state |
| `evals/*`, `tests/evals/*`, scripts/examples | Optional evaluation persistence/UI and deterministic replay; live execution must remain separated from offline unit tests |

Important active behavior:

- Default `AppSettings.remedy_final_full_scan_enabled=False` (`settings.py:81,154`); `REMEDY_FINAL_FULL_SCAN_ENABLED` opts in. Supervisor synthesizes an incremental `FinalFullScanResult` with `authoritative=False` and explicit coverage (`supervisor_node.py:5983–6007`). Keep the existing type/name for consumers.
- Setup scan evidence plus accepted current task/occurrence evidence feeds incremental accounting. Reopened work must invalidate the affected evidence and completion gates; failed/reverted attempts cannot add durable clean status.
- Current targeted artifact staging (`_qa_runtime.py:2521–2613`) creates one project per changed source lockfile, retaining its raw lockfile, applicable actual manifest, and physical package manifests. Closure analysis supplies coverage planning; it does not mean production artifact staging uses the exported sliced-lockfile builder. Describe emitted artifacts and coverage planning separately.
- `_QA_PACKAGE_STATE_CHECKS_ENABLED=False` (`qa_policy_engine.py:39`) and `_QA_NPM_LS_ENABLED=False` (`qa_critic.py:105`) disable those extra checks. Do not enable them as cleanup or describe them as active gates.
- Replacement reducers carry complete Supervisor-owned projections, while attempt history and result envelopes have their own merge/consume semantics (`state.py:474–490`). An incremental worker map cannot safely replace a complete task map.
- Child workaround source snapshots preserve the cumulative dependency candidate; parent rollback anchors survive until success or terminal abandonment. Passing validation terminates the bounded worker loop after tool responses are recorded.
- Example runners may copy suppression files or rewrite fixture state on the host; distinguish these runner side effects from the engine's no-host-patch guarantee.

## Findings Inventory

Locations below refer to the reviewed dirty baseline. Recheck lines after edits. **Observed** means directly established by code or a deterministic mocked reproduction. **Inference** requires the named test before treatment. Classifications: local cleanup, cohesion, or invariant bug fix; none authorizes a policy redesign.

| ID | Priority / confidence | Location / observation | Impact / planned treatment |
|---|---|---|---|
| DOC-01 | P1 observed | `docs/architecture.md:17–55`; actual `graph.py:792–815` and `supervisor_node.py:6068` | Wrong graph and portfolio authority. Rewrite topology/ownership from current registrations and Supervisor tool call |
| DOC-02 | P1 observed | `README.md:86`, `docs/architecture.md:239–260`; `settings.py:81,154`, `supervisor_node.py:5991` | Docs mandate a final scan while default is incremental. Describe default coverage and opt-in authority without changing either |
| DOC-03 | P2 observed | `docs/architecture.md:230–235`; current `_qa_runtime.py` closure preparation and `tools/lockfile_closure.py` | Target-only/no-edges description is obsolete. Document physical occurrence resolution and dependency closure/coverage failure behavior |
| DOC-04 | P2 observed | `AGENTS.md:13`; README references, `examples/NodeGoat/*`, `tests/test_nodegoat_batch.py` | Sole maintained example wording conflicts with tested secondary NodeGoat. Identify primary/secondary roles accurately; retain tested examples |
| DOC-05 | P2 observed | `docs/additive_targeted_odc_plan.md:1–180` | Historical proposal contains dead absolute worktree links and misleading authoritative=True example. Replace with a clearly labeled implementation-status record and repo-relative links; never execute old proposal as current requirements |
| DOC-06 | P2 observed | `EVALS.md:3,57,65`; `tests/evals/golden_schema.py:22–30`, `docs/triage-eval-mapping.md:3–8` | Seven datasets registered, tactical Supervisor replay and deterministic assertions omitted/misdescribed. Derive suite inventory and schema description from current tests |
| STATE-01 | P1 observed, reproduced | `supervisor_node.py:3222–3240` | Worker-result check omits envelope attempt-ID equality with its map key/current ID. Reject foreign IDs even when task revision/digest match; mirror QA guard. Test a foreign envelope stored under current key, plus stale/terminal/atomic cases |
| STATE-02 | P1 observed | `state.py:483`; `update_subagent.py:803–920`; `graph_wrappers.py:1505–1508` | Partial per-dispatch diagnostics replace every task's diagnostics. Overlay validated target results onto the prior full map in the bridge; preserve explicit Supervisor deletion semantics |
| RB-01 | P1 observed, reproduced | `graph_wrappers.py:738–753`; destructive `sandbox_mgr.py:958–977` | Final canary restore can erase workspace then fail; code changes candidate_restored back to True and may retain blame. A successful previous restore does not establish current integrity. Clear attribution, keep the archive and block further mutation until verified restoration |
| RB-02 | P1 observed, reproduced | `tools_manifest.py:65–103,399–403` | Failed local rollback clears touched paths and checkpoints. Keep unresolved paths/anchors, expose typed restoration outcome, preserve successful rollback behavior |
| RB-03 | P1 observed, reproduced | `tools_validation.py:407–460` | Failed restore logs only, clears snapshot/edit state and advances iteration. Stop the worker lifecycle and retain pending anchors when restoration is incomplete |
| RB-04 | P1 observed / production effect inference | `workaround_subagent.py:1461–1485`, `tools_edit.py:244–270` | Failed source restoration can disappear from changed-files evidence; restoration exception can mask the original edit failure. Return unresolved files plus both errors and require outer restore before dispatch |
| LIFE-01 | P1 observed / leak inference | `graph.py:901–980` | Exception fallback uses initial state and renders artifacts without tearing down an already-created volume. Track the run-owned volume/latest usable state outside successful invoke, clean in finally, and re-raise the original exception |
| LIFE-02 | P1 observed / leak inference | `teardown_node.py:466–490,619+` | Reconciliation/path bookkeeping can throw before cleanup try/finally; outer handler retains volume without cleanup. Move cleanup guarantee around the entire teardown body; test pre-diff failure |
| QA-01 | P1 observed, reproduced; fixed | `_qa_runtime.py` task target builder; `tests/test_qa_policy.py` authorized-attempt regression | Target builder preferred `task.selected_version` over current authorized effective execution version. It now resolves current committed attempt evidence first and fails closed when that evidence is invalid; it does not widen the allowlist |
| QA-02 | P1 observed, reproduced; fixed | `_qa_runtime.py` per-task `ODCScanEvidence`; `qa_evaluator.py` task prompt; `qa_policy_engine.py` deterministic gate; Gate B live evidence | Repository-wide scanner identifiers leaked into a task verdict and could make base64url receive a `security_flag` for a sibling occurrence. Evaluator context and deterministic target gates now use task-scoped occurrence evidence; incomplete task coverage fails closed, while cumulative findings remain available to triage. Prompt, policy and four-package live regressions cover the boundary |
| SEC-01 | P1 observed | `qa_odc.py:309,654–657`; redactor `:109` | INFO log exposes secret-bearing ODC extra arguments. Reuse redaction for logs, diagnostics and exception command presentation; test dummy secrets only |
| CFG-01 | P1 observed | `triage/agent.py:401–403` | Injected API key is omitted from ChatOpenAI constructor. Pass the resolved settings credential, retaining model and deterministic fallback; test without any environment key |
| CFG-02 | P1 observed | `api.py:triage_issues`, `triage/pipeline.py:179–206`, `tools/fix_planner.py:plan_fix` | Direct triage settings do not bind/propagate to planning. Propagate the explicit settings dependency consistently without new environment reads; preserve existing default call compatibility |
| CFG-03 | P2 observed | `evals/runner.py:86–88` | `is_live=False` leaves an ambient `EVAL_RUN_LIVE=true` in the child environment, so an unchecked offline run can invoke live services. Set the flag explicitly for both modes and test the ambient-true/offline case |
| PATH-01 | P1 strong inference | `tools/manifest_locator.py:358–360,809,891–909`; manifest reading precedes containment rejection | Symlinked manifests may be read outside the repository before their path is rejected. Apply shared resolved containment before reading; add an external-file sentinel symlink test and inspect equivalent reachability reads |
| SEC-02 | P1 observed template content / credential validity unknown | `.env.example:16–17` | Credential-looking non-empty obsolete template data. Never copy its value into plan/logs. Replace with blank/example-only values and reconcile consumed variable names; do not read actual `.env` or claim a verified credential leak |
| EVAL-01 | P2 observed | `evals/db.py:38–51` and other `with self._get_connection()` sites | sqlite transaction context does not guarantee connection.close, and required PRAGMA failure can leak before the operation boundary receives the connection. Close on both setup failure and transaction exit; test success/failure paths |
| EVAL-02 | P2 observed | `evals/db.py:440,560–561` | Comparison silently sees only default first 500 cases. Fetch all cases for comparison while preserving UI pagination; test a regression beyond row 500 and duplicate-name identity |
| EVAL-03 | P2 observed | `evals/runner.py:73–89`, `evals/ui/app.py:539–544` | Abandoned generator lacks process/pipe cleanup; UI always claims successful persistence despite nonzero exit. Preserve streaming interface, expose typed terminal outcome to consumer, close/wait on abandonment, and report suite success without claiming DB persistence from exit status alone |
| DEAD-01 | P2 observed call graph | `portfolio_orchestrator.py:1917`, `_legacy_build_portfolio_plan` | Uncalled private old planner plus exclusive helper cone creates competing architecture. Delete only verified exclusive helpers, not shared namespace/synthetic/digest/delta logic |
| DEAD-02 | P2 observed call graph | `tools/fix_planner.py:113,136,523` | `_is_npm_issue`, `_extract_local_version`, `_fetch_npm_latest` only serve isolated private tests. Remove them and exclusive constants/tests; retain active OSV/Serper/workaround behavior and meaningful planner contract tests |
| DEAD-03 | P2 observed call graph; fixed | `supervisor_routing.py` former duplicate phase helpers; `tests/test_deterministic_supervisor.py` | Removed the obsolete alternate phase transition/reconciliation implementations and moved the substantive phase/audit assertion to the active deterministic route boundary. Kept the exported `EligibleActions` and `ReconciliationResult` schema types because they remain public contracts |
| CLEAN-01 | P2 observed | `tools_web.py:27–32`, `remedy_tools.py:139,216`, `_tool_support.py:78` | Unused mandatory_search_terms threaded through helpers and duplicate three-call literal. Remove internal dead parameter after caller check, use existing constant; keep public toolbelt compatibility if externally exposed |
| CLEAN-02 | P2 observed | `report_node.py:2484–2497`, `generate_report:2396` | Finalization builds report context twice. Render through one internal context boundary and keep public function signatures/artifact text identical |
| PERSIST-01 | P2 observed, reproduced | `report_persistence.py:25–31`; analogous `trajectory_exporter.py:1077–1079` | Writers to the same destination share a temporary filename. Reproduced interleaving makes writer A fail after writer B consumes the temp. Use uniquely owned sibling temps, atomic replacement and cleanup of only owned temp; test write/replace failures and interleaving |
| DOC-07 | P3 observed | `fix_planner.py:1–30`, `qa_critic.py:4–21,752–758`, `state.py:1–19`, `scripts/generate_diagrams.py:18,49,91` | Contradictory/changelog prose, mojibake, one-task description and stale reducer overview. Rewrite accurate Google-style docs, concise rationale, repair encoding and missing docstrings; preserve model behavior and telemetry |
| TYPE-01 | P2 hypothesis; production alias not demonstrated | `contracts/schemas.py:TaskAttemptSnapshot`; `_create_attempt_snapshot` | Pydantic frozen model leaves nested lists mutable to a caller that mutates the model directly. Production search found no in-place snapshot-field mutation; snapshot construction copies list-valued authorization and issue fields. Preserve the JSON contract and defer a schema migration unless a production alias path is reproduced |
| STATE-03 | P1 observed repair / effect needs test | `_supervisor_execution.py:341–369`, `_impl__validate_committed_state` | Snapshot repair can accept foreign/stale identity and rewind mutable revision. Preserve intentional same-attempt input repair; reject foreign task/attempt/revision before copying snapshot fields. Test monotonic revisions and wrong-task snapshot |
| INGEST-01 | P2 observed, reproduced | `cli.py:177–207`, `tools/odc_parser.py:240`, `tools/semgrep_parser.py:69–87,149` | Malformed ODC root `[]` and Semgrep scalar result raise uncaught AttributeError rather than typed input failure. Validate envelopes/records, raise useful ValueError, preserve valid filtering; CLI must exit 2 |
| TRIAGE-01 | P2 observed, reproduced | `triage/enrichment.py:181–186,256–264`; constrained `schemas.py:904–914` | External EPSS outside 0–1 passes parsing then raises ValidationError. Validate finite bounded scores at response boundary and retain failure-safe defaults; test negative, >1, NaN, infinity and malformed records |
| TRIAGE-02 | P2 observed, reproduced | `triage/reachability.py:121–122`, `tools/code_map.py:380–407` | Substring matching marks got reachable from an unrelated import whose identifier is forgot. Match parsed import package identities/subpaths; preserve True/False/unknown rules |
| TRIAGE-03 | P2 observed, reproduced | `tools/code_map.py:407`; `triage/reachability.py:31,95–96` | Analysis reuses 20-import prompt cap and omits supported .mjs/.cjs. Separate uncapped deterministic import analysis from bounded prompt context; test target import 21 and both extensions |
| TRIAGE-04 | P2 observed | `triage/grouper.py:422` | Source set becomes unsorted list despite deterministic grouping. Sort source enum values, preserve membership/representative policy, test reverse input and hash-seed stability |
| DEAD-04 | P2 observed call graph | `tests/evals/custom_metrics.py:78,184,273,383,519,617` | Six obsolete unreferenced metric classes. Delete TriageConsistencyMetric, FixPlannerSchemaMetric, ArchitectureBoundaryMetric, ToolEfficiencyMetric, WorkaroundLifecycleMetric, DeterministicTaskCompletionMetric and exclusive imports/helpers; retain three metrics consumed by test_business_rules.py |
| ARCH-01 | P2 observed, deferred structural cleanup | `_supervisor_execution.py:73–159`, `supervisor_node.py:2464–2814`, `graph_wrappers.py:46` | Dynamic proxy/facade dependency plumbing remains around active helpers. It preserves graph import-cycle avoidance and facade monkeypatch seams. Removing it is not needed for the QA or state fixes and would require a separate test-seam migration; no behavior-changing rewrite was made in this pass |
| RANGE-01 | P2 strong inference | `tools/registry_tools.py:72–79`; normalized consumers `npm_graph.py:167–178`, `manifest_locator.py:548–550` | Registry range parser may reject legal operator whitespace used elsewhere. Reproduce with `^ 1.2.3` first, then share normalization without changing roles/security floors/candidate order |
| EVAL-04 | P3 observed | `evals/ui/app.py:137–140,238–244` and obsolete branding at `:1,25,116,179` | ERROR labeled SKIP and filter options omit supported suites. Render actual status, derive filters from registered/persisted suites, remove misleading branding; retain dashboard data contracts |
| TOOL-01 | P3 observed | `tools/odc_parser.py:339–340`, `tools/semgrep_parser.py:63–66,341–344` | Direct module CLI default roots point to src/data rather than repository data. Retain supported entrypoints, correct defaults with focused parser tests; canonical remedy ingest remains unchanged |
| TOOL-02 | P3 observed | `scripts/generate_diagrams.py:106` | Renderer path hard-coded to /opt/homebrew/bin/dot. Use explicit/discovered executable and clear unavailable result, preserving generated graph content |

### Finding dependencies, alternatives and non-goals

- STATE-01 precedes QA-01 and any state-map cohesion work. A new check must not mark the valid current attempt consumed merely because a foreign envelope was rejected.
- STATE-02 preserves the replacement reducer; replacing it with a merge reducer would make deletions impossible. Validate worker evidence before adding it to the full projection.
- RB-01 through RB-04 and LIFE-01/02 are invariant repairs. Do not alter normal success/retry policy, canary ranking/budget, atomic membership, or baseline/candidate restoration order. Test partial destructive failures rather than only pre-mutation exceptions.
- QA-01 must preserve exact multi-package mutation targets and require matching task/attempt/revision/digest. Do not select versions from arbitrary diagnostics or broaden the allowlist.
- PATH-01 requires a failing sentinel test before changing behavior. Use `runtime/path_policy.py`, not a second path-validation framework; internal symlinks may stay valid if the current contract permits them.
- DEAD-01 exclusive candidates include `_PackageNode`, `_build_nodes`, `_build_edges`, `_canonical_dependency_kind`, `_cluster_key`, `_eligible_component`, `_bounded_component_sets`, `_stable_task_key`, `_topological_cluster_order`. Recompute callers after deleting the root. Keep `_NAMESPACE_PREFIXES`, `_digest`, synthetic materialization and the tested legacy delta mode where still active.
- DEAD-02 does not authorize removal of Serper/LLM workaround discovery, which remains active in `plan_fix:892–914` despite the top docstring. DEAD-03 must not delete contract assertions with substantive coverage or break a supported import path.
- EVAL-03 may need a small internal result carrier/callback; preserve the streaming generator API for existing consumers. Do not introduce a new subprocess framework.
- PERSIST-01 guarantees whole-file replacement, not crash durability or a semantic winner among concurrent same-run writers. Do not add fsync or distributed locks without evidence of need.
- TYPE-01 is not a proven engine bug. Keep JSON shape and current consumer APIs unless an actual alias mutation is demonstrated.
- TRIAGE-02/03 fix evidence collection, not the reachability policy: unreachable findings must retain their documented validity treatment. Prompt import caps remain bounded. TRIAGE-01 must preserve default enrichment behavior on unavailable data.
- Exported `tools/lockfile_closure.py:build_sliced_lockfile_artifacts` has test callers and an `__all__` entry despite no production QA caller. Do not remove it as private dead code without a compatibility decision. Active eval sample data must stay clearly labeled synthetic.

## Target Design

### 1. One rollback boundary with explicit outcomes

Recommend a small internal restoration result shared by package/source rollback helpers: restored paths, unresolved paths, error diagnostics and a success flag. Keep task decisions and retry/pivot policy in the Supervisor. Callers may erase an anchor only after successful restoration or explicit terminal cleanup. Example internal usage:

```python
outcome = restore_checkpoint(sandbox, checkpoint)
if not outcome.succeeded:
    return restoration_failure(outcome, retained_checkpoint=checkpoint)
clear_checkpoint_after_verified_restore()
```

Hide filesystem restore accounting and partial-failure handling, not orchestration strategy. Prefer extending current helpers first; extracting `orchestration/rollback.py` is justified only when it genuinely removes duplicate loops in `tools_manifest`, `tools_validation`, `tools_edit`, and `workaround_subagent`. An all-purpose transaction manager or merging the entire toolbelt would increase scope without improving ownership.

Keep whole-workspace candidate/parent/child snapshots in the graph/runtime boundary. Do not mix them with local file checkpoints or infer that a failed restore preserved the old candidate.

### 2. Explicit complete projections at worker bridges

Validate envelopes once using current task, snapshot and action provenance; return a complete diagnostics/result projection to Supervisor-owned replacement fields. A narrow pure helper is justified if shared by actual bridges. Keep append-only attempts and terminal deletion/reconciliation distinct. Avoid merging every map indiscriminately or adding a new state service that exposes all fields.

### 3. Current package-planning boundary

Preserve public `prepare_portfolio_inputs`, `build_portfolio_plan`, `apply_portfolio_plan`. Preparation/synthetic coordination remains in portfolio orchestration, solving/projection in `portfolio_solver`, certification in `portfolio_certifier`, pure assignments in `solver`. Remove the unused alternate planner rather than merging the three active owners. Keep exports and copy-on-write behavior. Protect namespace expansion and occurrence-aware batch ancestry at these boundaries.

### 4. Report context as the renderer input

Build `ReportContext` once per final render, keep section rendering pure, and keep persistence separate. Preserve public `generate_report`/`finalize_report` and exact evidence attribution, coverage labels, attempt diffs and token semantics. Small wrappers used as monkeypatch seams are intentional until tests move to the actual injected boundary; do not remove them purely for being short.

## Phased Implementation Plan

### Phase 0: Safety net and invariants

**Objective:** record the dirty baseline, classify failures and protect expected engine behavior before touching policy-sensitive code. **Non-goals:** no new package policy, solver defaults, live services in unit tests, or broad test deletion.

1. Capture status/diff and baseline test failures separately from implementation changes. Store artifacts under ignored runtime state or `/private/tmp`; never stash/reset the user's edits.
2. Classify all 82 failures into missing infrastructure mocks, stale boundary setup, or actual regression. Existing failing test bodies remain contracts unless evidence proves a historical assumption changed.
3. Add a shared-name Angular workflow regression using `test_scoped_angular_mutation_closure_stays_at_nine_packages` (`test_portfolio_orchestrator.py:922`) as setup. Current regression verifies the nine-package planning/terminal projection with seeded snapshots and mocked worker/QA nodes; it does not assert a final accepted patch or claim full runtime coverage.
4. Add the four-package transition regression using `test_jws_base64url_finding_ancestry_batching` (`test_solver_portfolio.py:733`), canary tests (`test_portfolio_orchestrator.py:1384–1499`) and blame tests (`test_supervisor_node.py:3956–4024`): initial `{base64url,jws}` atomic batch; `got` and `esbuild` singleton roots; pair QA fails; isolation blames only `jws`; unaffected base64url survives rollback/replan; all accepted packages finish remediated and final patch excludes jws. The passing harness mocks worker/QA boundaries and seeds a committed plan; it does not cover full portfolio certification or live execution.
5. Add focused failing tests for STATE-01/02, RB-01/02/03/04, LIFE-01/02, QA-01, explicit triage settings, redaction and safe manifest reads. Existing passing stale/revision/atomic tests remain unchanged in intent.
6. Create a fixture result checker or optional acceptance mode; general runner exit semantics remain unchanged. Assert actual tasks, atomic membership, attempt IDs, isolation blame, final files/diff and scan coverage. Count Angular packages, not merely the nine input findings.

**Touched files:** relevant existing tests plus new `tests/test_behavior_preservation.py` and focused failure tests as appropriate; optional `examples/juice_shop/verify_expected_results.py`. No committed runtime outcomes/golden secrets.

**Validation:** offline suite command above; targeted fixture/state tests. **Acceptance:** deterministic complete outcome gates exist and no tests require Docker/LLM/network unintentionally. **Checkpoint/rollback:** independent safety-net diff; leave baseline failures documented until repaired. **Reviewer focus:** coherent mocked evidence, real state transitions and no policy encoded differently by the mocks.

### Phase 1: Documentation reconciliation and local cleanup

**Objective:** make all maintained Markdown reflect actual logic; remove unambiguous dead private code and low-risk prose/lint issues. **Prerequisite:** Phase 0 baseline recorded. **Non-goals:** changing runtime to match stale prose, removing active examples or simplifying behavioral prompts.

1. Build a document-to-code truth matrix. Markdown inventory includes `AGENTS.md`, `README.md`, `EVALS.md`, `docs/{architecture,additive_targeted_odc_plan,triage-eval-mapping}.md`, `examples/NodeGoat/README.md`, `examples/juice_shop/README.md`, fixture READMEs for `deterministic_routing`, `peer_conflict`, `peer_dependencies`, `shared_dependencies`, `shared_name`, `upstream_prerequisites`, and engine-specific statements in `.agents/skills/{rabbit-hole,review-and-refactor,trajectory-reviewer}/SKILL.md`. Skill workflow instructions remain intact; only contradicted engine facts need correction. Respect protected-directory permissions. **Complete:** all 17 existing repository Markdown files inventoried; `review-and-refactor` retained unchanged because its generic workflow remains accurate.
2. **Complete:** corrected DOC-01–07; documented active/opt-in scan gates, disabled checks, portfolio authority, atomic canaries, reducer semantics, parent/child anchors, public statuses, credential/config names and runner side effects. Historical documents distinguish proposals from current implementation, and README config now includes `REMEDY_FINAL_FULL_SCAN_ENABLED` and `REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS`.
3. **Complete:** audited Mermaid/DOT/generator against graph registration and refreshed all diagram sources plus the referenced PNG.
4. **Complete:** removed obsolete credential-looking template data without reproducing or validating it. No secret values were inspected or printed, and no manual `.env` inspection was performed; example runners retain normal dotenv loading.
5. **Complete:** removed DEAD-01 and its exclusive cone, then DEAD-02 and exclusive constants/private-helper tests. Rechecked imports and active solver/synthetic dependency boundaries. Removed DEAD-03 duplicate transition helpers after moving the meaningful route/audit test to the active implementation; retained the public contract schema types.
6. **Complete:** applied CLEAN-01/02, safe context/prose corrections including DOC-07, meaningful diagram-generator docstrings, and formatted the working tree.

**Validation:** full Markdown link/symbol search, graph/docs topology comparison, existing planner/solver/toolbelt tests, ruff check/format check. **Acceptance:** no present-tense behavioral assertion contradicts its cited implementation; no unsupported private symbol removals. **API/config/data migration:** none. **Checkpoint:** documentation and dead-code removals are separate reviewable diffs so either can be restored independently. **Reviewer focus:** preserve shared helpers and current prompts/telemetry.

### Phase 2: Invariant fixes and cohesion

**Objective:** repair demonstrated state, restore, targeting, settings and resource defects behind current interfaces. **Prerequisites:** Phase 0 regressions; source-derived docs in Phase 1. **Non-goals:** task scheduling, candidate selection, enabled QA checks or retry-budget redesign.

Ordered batches:

1. **Envelope/projection:** STATE-01/02 in `supervisor_node.py`, `graph_wrappers.py`; tighten ID validation and preserve unrelated task diagnostics. Use pure validated projection helpers only where they replace repeated logic.
2. **Canary/workspace integrity:** RB-01 in `graph_wrappers.py`; remove false success fallback. Preserve candidate archives and failed restore diagnostics; verified restoration is required before blame is usable or workers resume. Keep existing canary budgets/ranking and exhaustive mode.
3. **Local rollback:** RB-02–04 in `tools_manifest.py`, `tools_validation.py`, `tools_edit.py`, `workaround_subagent.py` and a small internal outcome type if needed. Retain unresolved paths/anchors and both primary/restoration errors; do not advance validation iterations after incomplete restoration. Ensure worker result and outer snapshot owner agree on integrity.
4. **Resource safety:** LIFE-01/02 in `graph.py`, `workspace_builder.py`, `teardown_node.py`. Track run-owned resource identity outside successful graph return, cleanup in a finally encompassing pre-diff failures, preserve last usable state where available, original exception propagation and cleanup error diagnostics. Remove only resources created by this run.
5. **QA targets/config/path/logging and ingestion/triage:** QA-01/02, CFG-01/02/03, PATH-01, SEC-01, INGEST-01 and TRIAGE-01–04. Prefer current authorized version evidence; use task-scoped scanner evidence for each verdict while retaining cumulative findings for triage; pass settings explicitly or bind only at an application boundary; explicitly disable live eval mode for offline child runs; validate before reading; redact all displayed commands; validate scanner envelopes/EPSS and use complete exact import identities. Keep failed/inconclusive scan retry behavior, guardrails and atomic exact targets. Add RANGE-01 only after its focused repro proves a defect.
6. **Rendering/persistence:** CLEAN-02, PERSIST-01 in report modules and trajectory writer, preserving rendering text and best-effort export semantics.
7. **Evaluation tooling:** EVAL-01–04 in `evals/db.py`, `runner.py`, `ui/app.py` and existing eval unit tests; preserve DB schema, UI pagination and streaming callers. Remove DEAD-04 in its own isolated cleanup checkpoint; keep active budget/latency metrics. TOOL-01/02 remain local CLI/renderer corrections.

**Tests:** preserve `test_contracts`, `test_execution_nodes`, `test_supervisor_node`, `test_runtime_context`, `test_sandbox_mgr`, `test_workaround_atomic_patches`, `test_workaround_replay`, `test_targeted_qa_scans`, `test_qa_policy`, `test_portfolio_certifier`, `test_report_node`, `test_report_persistence`, `test_trajectory_exporter`, eval DB/runner tests. Add failure injection at each changed I/O boundary and sequential graph reducer tests.

**Acceptance:** regressions fail before fix and pass after; complete mocked fixture gates retain exact outcomes; no new external access in unit tests; no diagnostics or anchor lost on incomplete restore. **API/config/data migration:** additive internal diagnostics only, stable public wire shapes/imports/config. **Checkpoint:** one coherent bug family per diff/test result; revert that checkpoint using a saved patch without erasing pre-existing edits. **Reviewer focus:** cleanup must not mask the original exception, and infrastructure failures must not consume remediation retries.

### Phase 3: Structural migrations

**Objective:** make only evidence-supported boundary changes left necessary by Phase 2. **Non-goals:** broad renames, file-count reduction, new orchestration engine, new state schema or generic transaction framework.

- Extract a focused rollback outcome/helper module only if shared restore accounting cannot be kept coherent in existing owners. Keep task/attempt/workspace policy in Supervisor/graph.
- If actual production alias mutation is demonstrated for TYPE-01, use defensive copies at boundaries first; a tuple/frozen-map wire migration is deferred absent necessity.
- DEAD-03's private duplicate helpers have been removed and the active route/audit boundary is tested. Keep the public contract types until a separate API compatibility decision supports their removal.
- STATE-03 identity/revision regressions are added and pass; preserve intentional same-attempt repair behavior. Address ARCH-01 only in a separate change after test-seam migration; preserve supported facade exports. Physical npm lookup helpers shared privately between manifest_locator/lockfile_closure/npm_graph are a secondary cohesion candidate: use a neutral existing npm_graph boundary only if tests demonstrate identical optional/peer/override semantics.
- Do not flatten solver, portfolio certification, graph adapters or prompt/tool builders into one module. Their boundaries hide different complexity and external effects.

**Validation:** import/serialization/graph-transition comparisons, full offline tests and both behavior gates after each move. **Acceptance:** fewer competing owners, same public signatures, field defaults and graph behavior. **Checkpoint:** each symbol move separately reversible. No persistence migration is currently planned; if one becomes necessary, update this plan before applying it. **Reviewer focus:** module extraction must reduce repeated caller knowledge, not merely shorten functions.

### Phase 4: Consolidation and removal of compatibility scaffolding

**Objective:** verify all supported paths and complete the evidence record. **Prerequisites:** prior phases pass focused checks.

1. Re-run whole-project callsite/dead-import scan including tests, eval adapters and examples. Remove only newly orphaned private code; preserve current public facades and sparse replay compatibility until proved unnecessary.
2. Reconcile all Markdown and diagram sources again against final implementation; remove stale absolute worktree links, mojibake, speculative/changelog claims and redundant comments. Keep rationale for surprising lifecycle policy.
3. Run complete offline checks and both live fixture commands with assertions below. Record exact settings, clone commit, package lock fingerprint, result/trajectory paths, checker output and remaining failures. Do not expose credentials.
4. Inspect final diff and dirty-file provenance; user changes must survive. Provide a change/validation ledger, resolved findings, deferred hypotheses and external blockers. Do not mark live acceptance passed based solely on mocks.

**Acceptance:** definition of done below. **API/config/data migration:** none expected. **Rollback:** last verified checkpoint if any fixture batch, blame, package outcome, patch, scan authority or cleanup invariant changes. **Reviewer focus:** actual runtime evidence, not shell success or a narrower test suite.

## Verification Strategy

### Required fixture acceptance

The user specified two fixtures; the reference to a third was a typo. These two commands are the complete live acceptance scope.

```bash
.venv/bin/python examples/juice_shop/fixtures/shared_name/run_shared_name.py \
  --repo data/clones/juice-shop

.venv/bin/python examples/juice_shop/run_batch.py \
  --iterations 1 \
  --packages base64url got esbuild jws \
  --repo data/clones/juice-shop
```

Gate A: three target packages expand into exactly nine Angular mutation packages, dispatched and successfully remediated in one atomic batch. Verify complete task/action provenance and every terminal outcome; fixture input finding count is insufficient.

Gate B: base64url and jws initially share one atomic batch, failed QA triggers isolation blaming jws alone, and got/esbuild are singleton batches. Base64url/got/esbuild succeed; jws remains the failed/unremediated package. Verify unaffected changes survive restoration/replanning and final patch excludes rejected jws mutation. Do not create a new blanket clustering rule just to make these package names pass: the current finding-backed ancestry and namespace rules must drive batching.

The scripts' general success exit status includes broad completed states. A checker must inspect typed results/state or persisted trajectory evidence. Verify no leftover run-owned volumes/containers; compare engine host input hashes before/after separately from runner suppression-file writes.

Live prerequisites: existing clone, installed environment, Docker daemon/images, allowed network/scanner data, required model credentials. User authorization covers running these acceptance fixtures, but they cannot be replaced by claims from offline mocks. Use normal approval tooling if restricted resources need escalation; preserve completed work and report exact blocker if access is unavailable.

### Offline and static checks

```bash
.venv/bin/python -m pytest -q -p no:rerunfailures \
  -m 'not docker and not network and not llm and not eval'
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m compileall -q src
.venv/bin/remedy --help
.venv/bin/python scripts/eval_compare.py --help
```

Run focused tests during each family, then one full offline suite after coherent batches. Do not repeatedly rerun broad checks absent new failures/changes. No new type/lint toolchain is needed. Add meaningful serialization checks for unchanged public request/result, attempt/plan contracts and legacy optional fields.

Failure matrices cover: envelope key vs ID, revision/digest/cluster mismatch; stale and duplicate results; partial atomic mutation; missing/corrupt snapshot; restoration failing before/after destructive cleanup; preservation of cumulative dependency candidate and earlier accepted source edits; unrelated task rollback anchor; incomplete/failed scans; retry budget not consumed by infrastructure failure; cancellation after volume creation; teardown exception before diff; file deletion/creation and final changed-file reducer; report/trajectory write and replace failure; explicit per-run settings without environment credentials; external symlink reads; secret redaction; eval connection/process cancellation and comparison past row 500.

No performance regression is asserted from function sizes. Measure snapshot archive volume/time and repeated context construction only if observed live cost warrants optimization. Preserve bounded canary calls, solver budgets, candidate guards and external I/O counts by contract tests.

## Compatibility and Rollout

- Keep package exports, API/CLI commands, statuses/exit behavior, model defaults, operational environment names, task/attempt JSON fields, plan identifiers/digests, tracing names and report coverage labels stable.
- Maintain legacy field defaults and sparse historical replay compatibility where tested; historical support is not automatically dead code.
- Preserve existing dirty improvements to nonmutating-workaround authorization, verified package absence/scan accounting, batched targeted artifact writes and snapshot/read safety.
- Bug fixes correct violated invariants and error paths; they do not authorize changing the requested agent policy. Document each deliberate failure-path change and protect the successful paths with the two acceptance gates.
- Implement directly in the current workspace without reset/stash or unsolicited commit/push. No deployment, PR, schema migration or feature flag is required for this request.
- Check usage before dispatching GPT-6 Luna and throughout implementation. At 98%, interrupt all agents and running tests/fixtures, stop further source edits, retain current state and report work paused. Resume only on the next user instruction.

## Deferred Work and Evidence Gaps

- The two live fixture commands were rerun after the peer-constraint solver fix. Gate A passed its latest explicit outcome verifier. Gate B produced the expected initial atomic pair failure and jws-only isolation blame; got and esbuild passed, and the only base64url test failure was the transient PasteBin integration test that the user explicitly waived. Keep that raw failure visible and do not add a code exception for it. The third-fixture wording was a typo and is not an outstanding requirement.
- The latest offline run has 68 failures, all present in the original 82-failure baseline: Supervisor (51), portfolio orchestrator (10), certifier (2), SCA strategy refactor (3), and deterministic Supervisor (2). Fourteen original failures now pass; no new failed test IDs appeared. Remaining families are classified above. Solver portfolio tests are green. Sanitized failure IDs, JUnit summary and console capture are under `/private/tmp/remediation-review-final-*`; no raw logs or generated reports were added to the repository.
- Large solver/parser/QA modules received call-path, ownership, failure-boundary and test inspection, not a proof over every line or candidate combination. Implementation should keep a coverage ledger and investigate remaining baseline failures rather than claiming the entire codebase is bug-free.
- TYPE-01 remains a mutability capability with no production in-place mutation or aliased snapshot input demonstrated. `_create_attempt_snapshot` copies its list-valued task inputs. Keep the wire shape unchanged unless an alias path is reproduced. Broader symlink exposure is a separate bounded hypothesis; the manifest path sentinel test now covers the identified read-before-containment defect.
- Dormant package-state/npm-ls checks remain disabled. Enabling or removing their supported tested contracts is separate policy work and is deferred.
- No benchmark, concurrent live run, full platform matrix, live evaluation suite or dependency upgrade was performed. Avoid speculative optimization, new frameworks and broad file reordering.
- Credential-looking template data was not validated as a real credential; do not test it or reproduce it. Actual credentials/clones/generated runtime reports are excluded from review artifacts.
- Review depth: state/lifecycle pass covered reducers, result correlation, snapshots, canaries, teardown, task lineage and corresponding tests; QA/portfolio pass covered targets, deterministic gates, certification provenance, atomic closure, worker rollback and relevant dirty diffs; periphery pass covered API/CLI/settings/triage/tool utilities/evals/examples and all 17 existing Markdown files. Root verified key findings and reviewed report/context/persistence/trajectory boundaries. Large Supervisor/schema/solver/QA bodies and replay adapters were inspected selectively. Every area has an owner in this map, but unexamined branches remain evidence gaps rather than claims of no defects.
- ARCH-01 remains an optional structural migration: active dynamic collaborators preserve the graph import cycle and established patch seams. It needs a separately reviewed change with seam-specific tests. Do not present it as required for the already passing QA, state, rollback, or live fixture fixes.

## Definition of Done

- Every repository-owned Markdown is reconciled or explicitly recorded as historical/instruction-only with current engine claims verified; linked diagram sources match the graph.
- Proven unused private implementation and exclusive imports/constants are removed; meaningful behavior tests and public/dynamic consumers remain supported.
- Each demonstrated state/rollback/resource/targeting/settings/redaction defect has a focused regression test and a passing fix; each remaining hypothesis has a documented disposition.
- Complete offline acceptance cases protect both required package/batch/QA/isolation outcomes. The 68 remaining baseline failures must be repaired or individually dispositioned; no weakening/deletion to hide failures.
- Ruff, formatting, compile and CLI checks pass, or remaining limitations are reported precisely.
- Both specified live fixture outcomes are recorded with scan coverage, rollback lineage, host protection and cleanup evidence. Gate A's raw verifier passes; Gate B is accepted by the user's PasteBin exception, while the raw verifier's base64url failure remains disclosed. There is no third fixture gate.
- Original dirty changes survive; no secrets, runtime clones, generated reports or trajectories are committed.
- Do not call work complete if these required gates remain unverified.
