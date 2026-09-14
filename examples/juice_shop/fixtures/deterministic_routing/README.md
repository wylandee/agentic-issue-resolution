# Deterministic Supervisor routing fixture

This pre-triaged fixture contains exactly five findings across five groups. It
lets a run focus on Supervisor's deterministic task ordering and route
transitions instead of making an initial-triage model decision.

| Component | Severity | Strategy | Coverage |
|---|---:|---|---|
| `notevil` | MEDIUM | `NO_FIX` | `NO_FIX_LIFECYCLE` and workaround dispatch |
| `express-jwt` | CRITICAL | `VERSION_BUMP` | Highest-priority update task |
| `sanitize-html` | HIGH | `VERSION_BUMP` | Second-priority update task and QA retry evidence |
| `got` | MEDIUM | `VERSION_BUMP` | Stable batch ordering |
| `@tootallnate/once` | LOW | `VERSION_BUMP` | Lowest-priority update task |

## What is deterministic

`triaged_groups_deterministic.json` is loaded and validated before the public
API call. The Supervisor creates one committed remediation task for each
pre-triaged group, chooses route transitions and task order, and records
attempt snapshots. Expected checkpoints are:

1. `NO_FIX_LIFECYCLE` routes `notevil` to the workaround worker.
2. `QA_READY` routes the completed workaround to QA under its committed QA
   policy.
3. `NEW_VERSION_BUMP` dispatches the four update tasks in severity order:
   critical, high, medium, then low.
4. `QA_READY` routes the active update task batch to QA, preserving task-keyed
   evaluation and attempt evidence.
5. When every task is terminal, `NO_ACTIONABLE_TASKS` routes to teardown and
   the final full scan is gated by Supervisor.

The deterministic property applies to fixture loading and Supervisor routing,
not to the whole execution. Workers, deterministic QA install/scan/tests, and
teardown still use the live Docker-backed workflow. A full run therefore needs
Docker and the credentials required by the configured worker model. No initial
triage call is needed for this fixture.

## Run

Run from the repository root against a disposable Juice Shop clone:

```bash
python examples/juice_shop/fixtures/deterministic_routing/run_deterministic_routing.py \
  --repo "$(pwd)/data/clones/juice-shop"
```

Use `--fixture` to load another validated group fixture, or `--output` and
`--patch-out` to select result and unified-patch destinations. Parent
directories are created as needed. The runner never applies the patch to the
host clone; remediation and QA failures remain in the result for inspection.

Default outputs are:

```text
data/trajectories/juice-shop-deterministic-routing-result.json
data/trajectories/juice-shop-deterministic-routing.patch
```

The result adds a routing summary containing the final decision code, next
route, Supervisor audit, and task statuses. A graph result of `completed` or
`completed_with_errors` returns exit code `0`; another graph status returns
`1`, while missing repository or fixture paths return `2`.
