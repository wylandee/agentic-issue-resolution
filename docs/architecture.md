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
