# Remediation Engine architecture

The application is a patch-producing security remediation service. Scanner
findings are normalized into typed contracts, triaged into vulnerability
groups, and processed by one Phase 5 task-queue graph.

```text
scanner report -> ingest -> triage/group/enrich -> supervisor
                                             |        |
                                             |        +--> update worker
                                             |        +--> workaround worker
                                             |        +--> QA critic
                                             |        +--> final full ODC scan
                                             v
                                        teardown -> report -> patch result
```

The supervisor owns task planning, retry decisions, and the authoritative task
and attempt snapshots. Workers execute only the committed instruction they are
given. QA evaluates the shared workspace and feeds structured evidence back to
the supervisor. Teardown computes the host-relative diff and removes the
temporary Docker volume.

The host repository is never edited by the public API or CLI. Results contain a
unified diff and changed-file list so a caller can review and apply the patch
through its own workflow. Trajectory files are written only when configured by
`REMEDIATION_TRAJECTORY_DIR`.

## QA scan authority

QA keeps install and full test execution per task. For supported npm
`package-lock.json` workspaces, it may resolve the active task's target from
the live Docker workspace after install, build a temporary exact-key closure,
and run a targeted ODC scan. Nested package keys, optional/peer edges, and
dependency ancestry are preserved by the pure resolver in
`tools/lockfile_closure.py`. Unsupported package managers, missing or
ambiguous lockfiles, incomplete closures, and targeted scan/report failures
fall back to the existing full ODC scan.

Per-task scan evidence is attempt-local and non-authoritative. It is attached
to `QAEvaluation` and stored by task, but it cannot update repo-wide scan
projections or trigger post-QA triage. Before teardown, the Supervisor must
route a terminal workspace through one authoritative full ODC scan:

```text
supervisor -> final_full_scan -> supervisor -> teardown
```

If that scan finds unresolved target identifiers or newly introduced
vulnerabilities, the existing triage node reopens only retryable work. A new
terminal cycle resets the final-scan gate. Docker volumes and temporary
targeted artifacts remain owned by the runtime/QA lifecycle and are removed
on teardown or the targeted-scan cleanup path.

## Package boundaries

- `contracts`: Pydantic models shared across boundaries.
- `triage`: scanner normalization, enrichment, reachability, and grouping.
- `orchestration`: LangGraph nodes, supervisor, workers, QA, and teardown.
- `runtime`: Docker workspace lifecycle.
- `tools`: deterministic parsers, locators, planners, and edit helpers.
- `api.py` and `cli.py`: the supported Python and command-line entrypoints.

## Attempt provenance and ownership

The `task_queue` and `attempt_snapshots_by_id` projections are the
authoritative orchestration state. A newly dispatched attempt records its task
ID, task revision, attempt ID, strategy stage, selected version, exact
instruction and digest, QA policy, and dispatch node. Worker and QA results
must match that envelope; stale or policyless results are rejected before any
workspace mutation or evaluation.

The Supervisor is the only component that selects dependency versions, chooses
retry or pivot strategies, and creates tasks. Update and workaround workers
execute committed instructions. QA runs deterministic install, scan, and test
steps and adds typed evidence; it does not choose the next remediation action.

## Runtime settings and secrets

The CLI and public API resolve `AppSettings` once and bind it for the graph
execution. Runtime dependencies use that bound context rather than reading
environment variables in business logic. Credentials are not serialized into
`OrchestratorState`, reports, or trajectory artifacts. Existing operational
names such as `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`, `TRIAGE_LLM_ENABLED`,
`ODC_EXTRA_ARGS`, `LANGSMITH_*`, and `REMEDIATION_TRAJECTORY_DIR` remain
supported at the application boundary.

## Report lifecycle

The report node consumes the final graph projection after teardown. It first
normalizes statuses, task/attempt evidence, scan state, package changes, diff
content, errors, and triage reconciliation into a deterministic report
context. Rendering is pure Markdown generation. Optional executive narrative
generation receives only that bounded deterministic evidence and cannot alter
statuses or findings. Persistence is isolated in `report_persistence.py` and
uses a sibling temporary file followed by atomic replacement. Report failures
remain visible as typed error metadata while the in-memory Markdown result is
preserved.

The graph performs post-QA triage only when an authoritative scan reports new
or unresolved findings. The final full scan is a separate Supervisor-owned
gate before teardown, so targeted per-attempt scans cannot silently reopen or
close repository-wide work.

## Docker volume ownership and path safety

`workspace_builder` creates the named workspace volume. Short-lived
`DockerSandbox` containers mount it for workers, QA, snapshots, and teardown.
Container startup is transactional, timeout results use exit code `124`, and
timeouts terminate only the current container so a caller can restart against
the preserved volume. Every container/client cleanup path is idempotent and
closes the Docker client. Teardown is the final volume owner and retries
removal after attached containers are force-removed.

Host and container file operations share `runtime/path_policy.py`. Relative
paths reject absolute forms and traversal; host resolution checks
`Path.resolve()` containment to block symlink escapes; container operations
verify canonical paths remain under `/workspace`. Invalid paths become
explicit errors or typed rejected results and are never treated as unchanged
files.

## Recommended next improvements

1. Add a small CI workflow that runs Ruff, the unit suite, and a dependency
   audit on every change; keep Docker/LLM tests as an opt-in integration job.
2. Lock production dependencies with a generated constraints file and review
   upgrades through automated vulnerability scanning.
3. Split the remaining deep Supervisor, QA, tools, and contract façades only
   after characterization tests establish their current behavior.
4. Add structured metrics for task retries, QA failure categories, patch size,
   and stale-result rejection to make remediation quality measurable.
