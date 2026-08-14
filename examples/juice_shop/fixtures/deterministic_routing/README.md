# Deterministic routing fixture

This pre-triaged fixture contains exactly five findings across five groups so a
manual Phase 5 run can focus on Supervisor routing rather than initial-triage
model decisions.

| Component | Severity | Strategy | Intended coverage |
|---|---:|---|---|
| `notevil` | MEDIUM | NO_FIX | `NO_FIX_LIFECYCLE` and workaround dispatch |
| `express-jwt` | CRITICAL | VERSION_BUMP | Highest-priority update task |
| `sanitize-html` | HIGH | VERSION_BUMP | Second-priority update task and QA retry evidence |
| `got` | MEDIUM | VERSION_BUMP | Stable batch ordering |
| `@tootallnate/once` | LOW | VERSION_BUMP | Lowest-priority update task |

Expected deterministic checkpoints during a normal run are:

1. `NO_FIX_LIFECYCLE` routes `notevil` to the workaround worker.
2. `QA_READY` routes the completed workaround to QA.
3. `NEW_VERSION_BUMP` dispatches the four update tasks in severity order:
   critical, high, medium, low.
4. `QA_READY` routes the active update batch to QA.
5. Once all tasks are terminal, `NO_ACTIONABLE_TASKS` routes to teardown.

The fixture intentionally does not claim that every dependency upgrade will pass
the Juice Shop test suite. The runner persists the final task states, decision
audit, trajectory, and patch so routing behavior can be evaluated separately
from remediation success.

Run it with:

```text
python examples/juice_shop/fixtures/deterministic_routing/run_deterministic_routing.py
```

The runner returns success when the graph reaches either `completed` or
`completed_with_errors`; remediation failures remain recorded in the output.
