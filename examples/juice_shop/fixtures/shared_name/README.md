# Shared-name Angular cluster fixture

This fixture keeps only the vulnerable Angular package findings from the Juice
Shop baseline active:

| Package | Manifest | Installed version | Findings |
|---|---|---:|---:|
| `@angular/common` | `frontend/package.json` | `21.2.14` | 4 |
| `@angular/compiler` | `frontend/package.json` | `21.2.14` | 2 |
| `@angular/core` | `frontend/package.json` | `21.2.14` | 3 |

All other baseline packages have active Dependency-Check suppression rules in
`suppressions.xml`. The three target rules are XML-commented, so only these
Angular packages remain visible to the scanner. The file contains rules for
all 55 packages in the current baseline, including packages absent from the
older suppressed fixture.

The three findings share the `@angular/` namespace and the same frontend
manifest. The runner scopes its development request to those three names, then
portfolio discovery includes the required same-manifest peer coordination
closure. For this fixture that produces exactly nine Angular mutation tasks:
`@angular/animations`, `@angular/common`, `@angular/compiler`,
`@angular/compiler-cli`, `@angular/core`, `@angular/forms`,
`@angular/platform-browser`, `@angular/platform-browser-dynamic`, and
`@angular/router`. The solver dispatches those mutations together in one atomic
batch. Unrelated direct frontend dependencies are not materialized; namespace
membership alone does not expand the plan.

## Files

* `baseline_issues_shared_name.jsonl` — nine canonical baseline findings.
* `suppressions.xml` — 55 package suppression rules, with only the three
  Angular target rules disabled.
* `run_shared_name.py` — live runner that copies the suppression file into the
  Juice Shop clone and passes raw issues through triage with the explicit
  development-only package scope.

## Run

Use a disposable clone because the runner overwrites its repository-local
`suppressions.xml` file:

```bash
python examples/juice_shop/fixtures/shared_name/run_shared_name.py \
  --repo data/clones/juice-shop
```

The runner writes:

```text
data/trajectories/juice-shop-shared-name-result.json
data/trajectories/juice-shop-shared-name.patch
```

The run passes raw issues rather than pre-triaged groups, so it exercises the
actual package-only grouper key. Inspect the trajectory for the three
finding-backed `sca:frontend/package.json:@angular/...` groups, nine Angular
mutation tasks, one shared `dispatch_batch_id`, atomic batch provenance, and
QA-passed status for all nine tasks.
