# Remediation Engine architecture

The Remediation Engine turns a canonical scanner finding set into a reviewable
patch. Findings are normalized into typed contracts, grouped by triage, and
processed by a Supervisor-owned task/attempt graph. The public API and CLI
return a typed status, changed-file list, unified diff, and errors; they never
apply a patch to the caller's repository.

## Graph topology

The graph has an initial preprocessing pass, a shared Docker workspace, a
scoped setup scan, and a Supervisor hub. Portfolio planning and certification
run as Supervisor-invoked functions, not as a separate graph node. Every worker
and QA result returns to the Supervisor, which may request post-QA
reconciliation and replan before committing the next transition.

```text
START
  |
  +--> initial_triage --(no work/failure)--------------------------+
  |          |                                                    |
  |          +--> workspace_builder --(failure)-------------------+--> teardown
  |                              |                                 |
  |                              +--> workspace_setup_scan --> supervisor
  |                                                             |  ^
  |                    +----------------------------------------+  |
  |                    |                                           |
  |                    +--> update_subagent -----------------------+
  |                    +--> workaround_subagent -------------------+
  |                    +--> qa_critic ------------------------------+
  |                    +--> triage (post-QA reconciliation) --------+
  |                    +--> final_full_scan (optional) -------------+
  |                    +--> teardown --> report --> END
```

`initial_triage` accepts the caller's groups or runs the initial triage
pipeline. `workspace_builder` creates and populates the shared volume, then
`workspace_setup_scan` records setup evidence before the first Supervisor
transition. The Supervisor invokes the portfolio planning and certification
functions when the active task set needs a plan. Invalid or non-dispatchable
plans are handled by Supervisor policy and can fail closed to teardown.

The `supervisor` dispatches committed tasks subject to the dispatch limits in
the implementation. It can route an update worker, workaround worker, or QA;
request post-QA `triage`; invoke portfolio replanning after reconciliation; or
finish at `teardown`. It requests `final_full_scan` only when
`REMEDY_FINAL_FULL_SCAN_ENABLED` is enabled and the configured workflow reaches
that gate. The optional scan and teardown are graph-level operations rather
than task dispatches.

The `triage` node is reserved for post-QA reconciliation. It consumes the
available parseable scan snapshot, reconciles groups with task lineage, and
returns control to the Supervisor. A reopened cycle invalidates affected scan
and plan evidence before more work is dispatched.

## Package boundaries

- `contracts` contains the Pydantic models and enums exchanged across
  ingestion, triage, orchestration, workers, QA, and the public result
  boundary. `contracts/accessors.py` provides the typed accessor used where
  orchestration accepts model or mapping state.
- Tactical actions are model proposals until Python validates them. The
  occurrence-aware portfolio planner supplies deterministic clusters and
  multi-package batches; the Supervisor commits the selected action, batch
  provenance, and per-task `TaskAttemptSnapshot` records before dispatch.
- `triage` owns scanner normalization, enrichment, reachability analysis,
  grouping, and the initial/post-QA triage pipeline.
- `orchestration/graph.py` builds the LangGraph and owns graph-level triage,
  routing, invocation, and trajectory setup.
- `orchestration/state.py` defines the master `OrchestratorState` and private
  `SubagentState`. The latter is ephemeral and may contain localized messages;
  it is not a second source of orchestration truth.
- `orchestration/supervisor_node.py` is the Supervisor entrypoint. Its focused
  helpers in `supervisor_planner.py`, `supervisor_routing.py`,
  `supervisor_spawn.py`, `_supervisor_execution.py`, and
  `supervisor_policy.py` plan candidates, validate transitions, materialize
  attempts, and route guarded decisions. `tactical_supervisor.py` supplies an
  optional QA-aware structured reasoner using the same stable-system and
  dynamic-context prompt layout as the worker and QA agents; Python validates
  its proposals and falls back to deterministic routing.
- `orchestration/graph_wrappers.py` bridges master state to the update,
  workaround, and QA subagent nodes. It validates task/attempt provenance and
  correlates each result back to its committed attempt.
- `orchestration/qa_critic.py` is the QA node facade. It coordinates the
  deterministic QA runtime and structured evaluator; `qa_odc.py` owns ODC
  execution and typed scan results, `qa_test_parsing.py` owns deterministic
  install/test execution and evidence parsing, `qa_policy_engine.py` applies
  policy gates and guardrails, and `qa_evaluator.py` runs the bounded
  read-only evaluator for each task.
- `orchestration/update_subagent.py` and
  `orchestration/workaround_subagent.py` execute committed attempts.
  `subagent_runtime.py` bounds model/tool loops, while `context_manager.py`
  filters tools by workaround phase and maintains bounded ephemeral context.
- Report rendering is split between `report_node.py`, `report_context.py`,
  and `report_diff.py`; `report_persistence.py` owns durable report-file
  replacement.
- `orchestration/tools_workspace.py`, `tools_edit.py`, `tools_manifest.py`,
  `tools_validation.py`, and `tools_web.py` own the update/workaround tool
  categories. `_tool_support.py` contains shared tool policy and bounded
  helpers. `remedy_tools.py` assembles the category tools for the two worker
  entrypoints; it is not an orchestration or state authority.
- `orchestration/workspace_builder.py` prepares the shared workspace and
  `teardown_node.py` performs terminal diff/cleanup.
- `runtime` owns Docker lifecycle and path policy:
  `sandbox_mgr.py` provides `DockerSandbox`, `docker_client.py` manages client
  acquisition/closure, and `path_policy.py` validates repository-relative and
  workspace-relative paths.
- `solver/subgraph.py`, `solver/cpsat.py`, and `solver/graph.py` contain pure,
  occurrence-aware graph extraction, bounded CP-SAT assignment, atomic batching,
  DAG validation, and SCC phase scheduling. They consume immutable solver
  contracts and never read environment variables or dispatch workers.
- `orchestration/portfolio_solver.py` is the outer copy-on-write adapter. It
  loads the npm graph and optional registry cache, projects active task/finding
  domains into solver contracts, and applies only solver-approved decisions.
- `tools` contains deterministic parsers, lockfile-closure and package
  planning helpers, repository maps, locators, and scanner utilities.
- `api.py` and `cli.py` are the supported Python and command-line boundaries.
  Callers do not construct LangGraph state directly.

`OrchestratorState.task_queue` is the authoritative mapping from task ID to
`RemediationTask`. `portfolio_plan` is the committed outer-plan projection;
its `portfolio_plan_id`, repository/graph/solver digests, task membership, and
planned revisions must match the active queue before the Supervisor dispatches.
`attempt_snapshots_by_id` is the authoritative record of each committed
attempt. Task lineage uses `parent_group_id` and explicit parent-task links;
group identity is triage/report context, while task ID is the execution and QA
correlation key. `active_target_task_ids` identifies the tasks selected for the
current dispatch. Tasks in one atomic multi-package dispatch share the
committed `dispatch_batch_id` and action provenance while retaining one
snapshot per attempt. If a batch fails QA, bounded canary replays may test
candidate subsets against the saved baseline. Blame is usable only after the
candidate workspace is verified restored; an incomplete restore blocks further
mutation and remains visible in the run diagnostics.

Before dispatch, the Supervisor commits an attempt snapshot containing the
task ID and revision, attempt ID, strategy stage, approved target-version and
dependency-type candidates, exact instruction and digest, QA policy, and
dispatch node. The mutable `RemediationTask.selected_version` is a planning
hint; the committed snapshot, exact atomic mutation target, and sealed worker
instruction define execution authorization. A worker may execute an alternate
only when an explicit committed candidate rule permits it; it cannot expand
the set or query a registry. QA checks current attempt evidence for the
effective version actually executed.
Worker and QA results must match the current task revision and attempt
envelope. A stale, detached, or policyless result is rejected before it can
mutate the workspace or enter evaluation.

`worker_results_by_attempt` and `qa_results_by_attempt` retain typed result
envelopes keyed by attempt ID. `qa_evaluations` is a derived, task-keyed view
for routing and reporting; its `QAEvaluation.task_id` is the actual task ID,
not a parent-group identifier. Scan evidence is attached to the task/attempt
that produced it. Scratchpads, conversation messages, summaries, and report
views are derived or ephemeral and cannot select a version, retry, pivot, or
new task.

The Supervisor is the only orchestration policy owner. It invokes the
portfolio-planning adapter when initial preparation or reconciliation requires
an executable plan. The adapter builds the occurrence-aware npm graph, obtains
bounded registry/cache candidates, calls the CP-SAT solver, certifies resolver
evidence, and returns a copy-on-write plan. The Supervisor decides how that
result changes task state and whether to dispatch, retry, pivot, replan, or
stop. Workers execute only the sealed attempts; certification and task
selection are not separate graph nodes.

### Optional tactical Supervisor proposals

The tactical Supervisor is a pure diagnostic classifier layered on top of the
deterministic target selector. The selector chooses one eligible task first;
the tactical call receives exactly that task and cannot independently scan the
queue. NO_FIX, terminal, multi-task, and inconclusive-QA transitions remain
deterministic. The classifier prompt has a byte-stable system message and a
fresh, ordered dynamic context for every proposal and one optional repair.
The dynamic context contains bounded QA evidence, worker diagnostics, prior
attempt summaries, and strategy-specific verified candidate sets. Attempt IDs,
timestamps, full instructions, and conversation replay remain Python-side
correlation data.

The model emits `diagnostic_basis`, strategy fields, file hints, and a concise
rationale; it does not emit worker authorization. Python derives the target
package and dependency type, enforces the canonical security floor and the
lowest eligible stable semver, verifies direct/parent/child candidate sets,
and renders the final update, override, or workaround instruction. The
rendered instruction is committed, hashed, and sealed in the attempt snapshot
before dispatch. A rejected proposal can receive one repair proposal; neither
proposal can mutate task state, create a spawn request, or invoke a worker.

When enabled, the tactical model may propose a single-task action or peer
conflict referral. The Supervisor validates the proposal and remains the sole
authority that commits it. Multi-package clustering, dependency ordering, and
portfolio certification are deterministic functions invoked by the
Supervisor. The stage ladder is used only by deterministic fallback, and
model/registry failures do not change safety boundaries.

## Worker execution

The update worker receives a deterministic repository map and task-scoped
context. Its manifest mutation is the atomic
`modify_and_validate_npm_dependency` transaction: it updates the manifest,
runs `npm install --package-lock-only --ignore-scripts`, and restores the
package checkpoint when the transaction fails. Bounded retry calls can request
another Supervisor-approved candidate; the worker cannot expand the candidate
set.

The workaround worker operates in filtered phases. Read-only workspace
inspection and deterministic web-evidence checks lead to a Supervisor-
committed plan; execution and validation then apply that plan through the
allowed edit and validation tools. Replay plans and attempt snapshots preserve
the committed work across retries. The worker context is bounded, excludes
complete file bodies and credentials, and is not persisted as orchestration
state. The Supervisor's committed workaround instruction includes the bounded
QA evidence, hypothesis, target-file hints, prohibited changes, and validation
criteria before the attempt snapshot is sealed.

Workspace lifecycle is candidate-preserving: an intermediate workaround QA
failure restores only that child attempt's source snapshot, while the parent
dependency candidate remains live for the next hypothesis. An inconclusive QA
result reruns against the candidate without consuming remediation retry
budget. The parent anchor is restored only when the workaround is terminally
abandoned; a successful workaround promotes the cumulative dependency-plus-
source state and clears stale anchors. A valid workaround validation PASS is
terminal for the bounded worker loop after matching tool responses are emitted.

## QA execution and authority

`qa_critic` runs deterministic global execution before any evaluator model
call. The QA runtime performs dependency installation, the applicable ODC
scan, and workspace tests, retaining bounded summaries and private raw
diagnostics. The structured evaluator then receives compact status flags,
task-scoped scanner evidence, bounded action summaries, and read-only
source-review tools. It returns a typed terminal decision for the task.
`qa_policy_engine` applies deterministic gates, attaches test/scan evidence,
and rejects decisions that are not backed by the required evidence.

For supported npm `package-lock.json` workspaces, QA resolves physical target
package occurrences from the live volume. Targeted artifact staging creates
one temporary project per changed source lockfile and preserves that
lockfile's raw content and applicable physical manifests; it does not slice
the artifact down to a target-only synthetic dependency closure. Closure
analysis is used for coverage planning. QA requires the ODC report to account
for the requested target occurrences at the expected versions before it can
confirm target CVE status. Unsupported package managers, missing or invalid
lockfiles, unresolvable targets, and incomplete or unparseable reports produce
inconclusive QA and do not consume a remediation retry. They do not trigger a
full-scan fallback. A NO_FIX package-removal batch can skip ODC only when the
accepted change is verified as deletion-only; deterministic install and test
gates still apply. Manifest and resolved-graph QA checks are currently
disabled.

Each task evaluator receives its task-scoped occurrence projection. Findings
from report-wide cumulative scans remain available to terminal triage and do
not independently fail an unrelated task.

Targeted per-task scans and evaluator decisions are attempt-local evidence.
They do not establish repository-wide security status and do not independently
reopen or close work. By default, when all tasks are terminal, the Supervisor
records the available incremental dependency-input coverage in a
`FinalFullScanResult` compatibility contract with `authoritative=false` and
`coverage_scope=incremental_dependency_inputs`. This state does not claim a
full-repository scan. A full ODC scan is opt-in through
`REMEDY_FINAL_FULL_SCAN_ENABLED=true`; when enabled, the graph routes through:

```text
supervisor -> final_full_scan -> supervisor -> teardown
```

The optional scan compares the post-remediation workspace with the baseline
and records remaining target identifiers, newly found identifiers, typed
issues, scan status, and a workspace fingerprint. A failed or incomplete scan
remains visible to teardown and reporting; it cannot be interpreted as a clean
result. If the scan reports unresolved or newly introduced findings, the
Supervisor may route retryable work through post-QA `triage`. Reopened work
invalidates affected evidence and can cause the optional scan to run again.

## Report lifecycle and public boundary

`teardown` reconciles terminal task state, restores or removes only
task-scoped workspace changes required by the committed outcome, computes a
host-relative unified diff and changed-file list, and cleans the run-owned
Docker volume. The host repository is read for comparison but is never
modified by the API, CLI, graph, workers, or QA.

`report` builds a deterministic `ReportContext` from task lineage, attempt
envelopes, QA evidence, scan state, package changes, diff content, triage
reconciliation, and errors. Markdown rendering is pure and stable. Internal
diagnostics remain typed state/error metadata rather than being copied into
the concise end-user report. `report_persistence.py` writes through a sibling
temporary file and atomic replacement; persistence failure is reported without
discarding an in-memory rendered report. Trajectory export is optional and is
controlled by `REMEDIATION_TRAJECTORY_DIR`.

At the public boundary, `RemediationRequest` requires an existing absolute
repository directory. `RemediationResult` exposes status, changed files,
unified diff, errors, trajectory path, and report path without exposing
credentials or requiring callers to understand internal state.

The CLI's scanner issue interchange is canonical JSONL: one typed issue object
per line, with invalid records reported as typed input errors. The Python API
accepts the already-typed `VulnerabilityIssue` and `VulnerabilityGroup`
contracts.

## Docker lifecycle and path safety

`workspace_builder` creates a run-owned named volume and copies the validated
host repository into it without `.git` or host-specific `node_modules`.
Short-lived `DockerSandbox` containers mount the volume at `/workspace` for
workers, QA, snapshots, and teardown. Container startup is transactional;
command timeouts are bounded and isolated to the active container. Sandbox
teardown is idempotent, removes the container, and closes the Docker client.

`teardown_node` is the final volume owner. It force-removes attached run-owned
containers as needed, retries volume removal, and records cleanup failures
without hiding remediation errors. Temporary targeted-scan artifacts and
workspace snapshots are removed by their owning lifecycle.

All scanner, model, persisted-state, and tool paths pass through
`runtime/path_policy.py`. Workspace paths must be normalized,
POSIX-style, relative, and free of traversal. Host resolution checks
`Path.resolve()` containment, preventing symlink escapes; container paths must
remain beneath `/workspace`. Invalid paths become explicit errors or typed
rejections, never an implicit unchanged-file result.

## Runtime settings and secrets

`AppSettings.from_env()` is evaluated at the application boundary and bound
for one graph run. Business logic uses that bound settings object rather than
reading process environment variables directly. Supported operational names
include `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`, the node-specific
`TRIAGE_LLM_MODEL`, `UPDATE_LLM_MODEL`, `WORKAROUND_LLM_MODEL`, and
`QA_LLM_MODEL`, `TRIAGE_LLM_ENABLED`, `SERPER_API_KEY`, `GITHUB_TOKEN`,
`ODC_EXTRA_ARGS`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
`LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`, `TRIAGE_CACHE_DIR`,
`REMEDIATION_TRAJECTORY_DIR`, `REMEDIATION_REPORT_DIR`,
`REMEDY_FINAL_FULL_SCAN_ENABLED`, `REMEDY_SOLVER_TIMEOUT_SECONDS`,
`REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS`, `REMEDY_SOLVER_TOP_K`,
`REMEDY_SOLVER_PHASE_BUDGET`, `REMEDY_SOLVER_ACCEPT_FEASIBLE`,
`REMEDY_SOLVER_RANDOM_SEED`, `REMEDY_SOLVER_NUM_SEARCH_WORKERS`,
`REMEDY_SOLVER_MAX_CANDIDATES_PER_TARGET`, `REMEDY_SOLVER_MAX_MODEL_VARIABLES`,
`REMEDY_SOLVER_LLM_ENABLED`, `SOLVER_LLM_MODEL`,
`REMEDY_BYPASS_WORKAROUND_SUBAGENT`,
`REMEDY_DISABLE_POST_QA_TRIAGE`, `REMEDY_RETRIAGE_LIMIT_ENABLED`,
`REMEDY_RETRIAGE_LIMIT`, and `REMEDY_MAX_DELTA_CANARY_PROBES`.

Credentials and API keys are used only at their operational boundary. They
are not placed in `OrchestratorState`, task/attempt contracts, reports,
trajectory artifacts, scratchpads, or diagnostics; tool and scanner logging
redacts secret-bearing arguments.
