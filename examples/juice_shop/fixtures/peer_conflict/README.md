# Socket.IO peer-conflict fixture

This fixture keeps only the intended target package findings from the current
Juice Shop baseline active:

| Package | Manifest | Installed version(s) | Findings |
|---|---|---:|---:|
| `socket.io` | `package.json` | 3.1.2 | 1 |
| `engine.io` | `package.json` | 4.1.2 | 3 |

Socket.IO and its Engine.IO dependency are kept active to exercise peer-conflict escalation and cluster expansion.

All other baseline packages have active Dependency-Check suppression rules in
`suppressions.xml`. The target rules are XML-commented, so only the listed
packages remain visible to the scanner. The suppression file contains rules
for all 55 packages in the current baseline.

## Files

* `baseline_issues_peer_conflict.jsonl` — canonical baseline findings for the target packages.
* `suppressions.xml` — package suppression rules with only the target rules disabled.
* `run_peer_conflict.py` — live runner that copies the suppression file into the
  Juice Shop clone before passing raw issues through triage and orchestration.

## Run

Use a disposable clone because the runner overwrites its repository-local
`suppressions.xml` file:

~~~bash
python examples/juice_shop/fixtures/peer_conflict/run_peer_conflict.py \
  --repo data/clones/juice-shop
~~~

The runner writes:

~~~text
data/trajectories/juice-shop-peer-conflict-result.json
data/trajectories/juice-shop-peer-conflict.patch
~~~

The run passes raw issues rather than pre-triaged groups, so it exercises the
current package-centric grouper and Supervisor portfolio behavior. Inspect the
trajectory for one package group per target package, dependency/peer
diagnostics, deterministic cluster membership, task revisions, and shared
batch provenance when the selected relationships produce a cluster.

