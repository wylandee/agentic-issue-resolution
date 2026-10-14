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

The three targets share the `@angular/` namespace and the same frontend
manifest. The runner explicitly scopes its development request to these three
names. Portfolio discovery is limited to the selected findings and any required
workspace or incompatible-peer coordination closure; unrelated direct frontend
dependencies are not materialized. Namespace membership alone does not couple
or expand a scoped plan.

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
finding-backed `sca:frontend/package.json:@angular/...` groups, any synthetic
tasks in their scoped coordination closure, bounded portfolio batches, a
shared `dispatch_batch_id` when tasks batch together, and cluster-wide QA
evidence.
