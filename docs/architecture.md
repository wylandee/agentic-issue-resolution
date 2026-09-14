# Remediation Engine architecture

The Remediation Engine turns a canonical scanner finding set into a reviewable
patch. Findings are normalized into typed contracts, grouped by triage, and
processed by a Supervisor-owned task/attempt graph. The public API and CLI
return a typed status, changed-file list, unified diff, and errors; they never
apply a patch to the caller's repository.

## Graph topology

The graph has one preprocessing pass, a shared Docker workspace, and a
Supervisor hub. Every worker result returns to the Supervisor, which decides
the next committed transition.

```text
START
  |
  +--> initial_triage --(no work/failure)--------------------+
  |          |                                              |
  |          +--> workspace_builder --(failure)-------------+--> teardown
  |                              |                           |
  |                              +--> supervisor <------------+
  |                                     |  ^                  |
  |             +-----------------------+  |                  |
  |             |                          |                  |
  |             +--> update_subagent ------+                  |
  |             +--> workaround_subagent --+                  |
  |             +--> qa_critic ------------+                  |
  |             +--> triage ----------------+                  |
  |             +--> final_full_scan -------+                  |
  |             +--> teardown -------------------------------> report --> END
```

`initial_triage` accepts the caller's groups or runs the initial triage
pipeline. `workspace_builder` creates and populates the shared volume.
`supervisor` then dispatches one committed task at a time (subject to the
dispatch limits in the implementation) to an update worker, workaround
worker, or QA. It can also send the graph through post-QA `triage`, request the
authoritative `final_full_scan`, or finish at `teardown`.

The `triage` node is reserved for post-QA reconciliation. It consumes the
complete parseable scan snapshot, reconciles groups with task lineage, and
returns control to the Supervisor. A reopened cycle clears the final-scan
completion gate before more work is dispatched. The final scan and teardown
are graph-level operations rather than task dispatches.

## Package boundaries

- `contracts` contains the Pydantic models and enums exchanged across
  ingestion, triage, orchestration, workers, QA, and the public result
  boundary. `contracts/accessors.py` provides the typed accessor used where
  orchestration accepts model or mapping state.
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
  attempts, and route deterministic decisions.
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
- `tools` contains deterministic parsers, lockfile-closure and package
  planning helpers, repository maps, locators, and scanner utilities.
- `api.py` and `cli.py` are the supported Python and command-line boundaries.
  Callers do not construct LangGraph state directly.

## Authoritative task and attempt state

`OrchestratorState.task_queue` is the authoritative mapping from task ID to
`RemediationTask`. `attempt_snapshots_by_id` is the authoritative record of
each committed attempt. Task lineage uses `parent_group_id` and explicit
parent-task links; group identity is triage/report context, while task ID is
the execution and QA correlation key. `active_target_task_ids` identifies the
tasks selected for the current dispatch.

Before dispatch, the Supervisor commits an attempt snapshot containing the
task ID and revision, attempt ID, strategy stage, selected version (when
applicable), approved target-version and dependency-type candidates, exact
instruction and digest, QA policy, and dispatch node. Worker and QA results
must match the current task revision and attempt envelope. A stale, detached,
or policyless result is rejected before it can mutate the workspace or enter
evaluation.

`worker_results_by_attempt` and `qa_results_by_attempt` retain typed result
envelopes keyed by attempt ID. `qa_evaluations` is a derived, task-keyed view
for routing and reporting; its `QAEvaluation.task_id` is the actual task ID,
not a parent-group identifier. Scan evidence is attached to the task/attempt
that produced it. Scratchpads, conversation messages, summaries, and report
views are derived or ephemeral and cannot select a version, retry, pivot, or
new task.

The Supervisor alone:

1. selects registry versions and dependency types;
2. chooses retry, pivot, and post-QA reconciliation actions;
3. creates tasks and commits their attempt snapshots; and
4. decides when all actionable work is complete and the final scan gate may
   run.

Workers execute the instruction and candidate set already committed to their
attempt. They do not create tasks or choose the next remediation action.

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
state.

## QA execution and authority

`qa_critic` runs deterministic global execution before any evaluator model
call. The QA runtime performs dependency installation, the applicable ODC
scan, and workspace tests, retaining bounded summaries and private raw
diagnostics. The structured evaluator then receives compact status flags,
identifiers, package-state facts, bounded action summaries, and read-only
source-review tools. It returns a typed terminal decision for the task.
`qa_policy_engine` applies deterministic gates, attaches test/scan evidence,
and rejects decisions that are not backed by the required evidence.

For supported npm `package-lock.json` workspaces, QA may resolve the active
task's target from the live volume, build a temporary exact-key dependency
closure with `tools/lockfile_closure.py`, and run a targeted ODC scan. Nested
package keys, optional/peer edges, and dependency ancestry are preserved.
Unsupported package managers, missing or ambiguous lockfiles, incomplete
closures, and targeted scan/report failures use the full-scan fallback. A
scoped NO_FIX package-removal policy can skip the per-task scan only when its
other deterministic install/test and package-state gates are satisfied.

Targeted per-task scans and evaluator decisions are attempt-local evidence.
They do not establish repository-wide security status and do not independently
reopen or close work. Before teardown, the Supervisor routes the terminal
workspace through exactly one authoritative full ODC scan:

```text
supervisor -> final_full_scan -> supervisor -> teardown
```

The final scan compares the post-remediation workspace with the baseline and
records remaining target identifiers, newly found identifiers, typed issues,
scan status, and a workspace fingerprint. A failed or incomplete authoritative
scan remains visible to teardown and reporting; it cannot be interpreted as a
clean result. If the scan reports unresolved or newly introduced findings,
the Supervisor may route retryable work through post-QA `triage`. Any reopened
cycle resets the final-scan gate and requires another authoritative scan before
teardown.

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
`REMEDY_BYPASS_WORKAROUND_SUBAGENT`,
`REMEDY_DISABLE_POST_QA_TRIAGE`, `REMEDY_RETRIAGE_LIMIT_ENABLED`, and
`REMEDY_RETRIAGE_LIMIT`.

Credentials and API keys are used only at their operational boundary. They
are not placed in `OrchestratorState`, task/attempt contracts, reports,
trajectory artifacts, scratchpads, or diagnostics; tool and scanner logging
redacts secret-bearing arguments.
