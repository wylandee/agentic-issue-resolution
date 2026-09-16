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
manifest. The portfolio planner should therefore keep the finding-backed
targets in a coupled bounded package cluster rather than three unrelated
singleton items. The planner also materializes the other direct frontend
dependencies as synthetic coordination tasks; if the namespace exceeds the
multi-package action cap, those synthetic tasks are partitioned into additional
bounded items while the finding-backed targets remain together when possible.

## Files

* `baseline_issues_shared_name.jsonl` — nine canonical baseline findings.
* `suppressions.xml` — 55 package suppression rules, with only the three
  Angular target rules disabled.
* `run_shared_name.py` — live runner that copies the suppression file into the
  Juice Shop clone before passing the raw issues through triage and
  orchestration.

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
actual package-only grouper key. In a successful cluster dispatch, inspect the
trajectory for the three finding-backed `sca:frontend/package.json:@angular/...`
groups, synthetic direct-dependency tasks, a bounded multi-package portfolio
cluster containing the target groups, a shared `dispatch_batch_id`, and
cluster-wide QA evidence.
