# Upstream prerequisite fixture

This fixture keeps only the intended target package findings from the current
Juice Shop baseline active:

| Package | Manifest | Installed version(s) | Findings |
|---|---|---:|---:|
| `express-jwt` | `package.json` | 0.1.3 | 2 |
| `jsonwebtoken` | `package.json` | 0.1.0, 0.4.0 | 12 |
| `moment` | `package.json` | 2.0.0 | 3 |
| `jws` | `package.json` | 0.2.6 | 2 |
| `base64url` | `package.json` | 0.0.6 | 1 |

The JWT dependency chain and Moment are kept active to exercise upstream prerequisite ordering.

All other baseline packages have active Dependency-Check suppression rules in
`suppressions.xml`. The target rules are XML-commented, so only the listed
packages remain visible to the scanner. The suppression file contains rules
for all 55 packages in the current baseline.

## Files

* `baseline_issues_upstream_prerequisites.jsonl` — canonical baseline findings for the target packages.
* `suppressions.xml` — package suppression rules with only the target rules disabled.
* `run_upstream_prerequisites.py` — live runner that copies the suppression file into the
  Juice Shop clone before passing raw issues through triage and orchestration.

## Run

Use a disposable clone because the runner overwrites its repository-local
`suppressions.xml` file:

~~~bash
python examples/juice_shop/fixtures/upstream_prerequisites/run_upstream_prerequisites.py \
  --repo data/clones/juice-shop
~~~

The runner writes:

~~~text
data/trajectories/juice-shop-upstream-prerequisites-result.json
data/trajectories/juice-shop-upstream-prerequisites.patch
~~~

The run passes raw issues rather than pre-triaged groups, so it exercises the
current package-centric grouper and Supervisor portfolio behavior. Inspect the
trajectory for one package group per target package, dependency/peer
diagnostics, deterministic cluster membership, task revisions, and shared
batch provenance when the selected relationships produce a cluster.

