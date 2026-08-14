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
                                        teardown -> patch result
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

## Recommended next improvements

1. Add a small CI workflow that runs Ruff, the unit suite, and a dependency
   audit on every change; keep Docker/LLM tests as an opt-in integration job.
2. Lock production dependencies with a generated constraints file and review
   upgrades through automated vulnerability scanning.
3. Thread `AppSettings` through graph nodes so runtime configuration is fully
   injectable in tests and embedding applications.
4. Add structured metrics for task retries, QA failure categories, patch size,
   and stale-result rejection to make remediation quality measurable.
