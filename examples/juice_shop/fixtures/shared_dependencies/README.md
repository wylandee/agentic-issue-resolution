# Shared dependency closure fixture

This pre-triaged fixture exercises task and attempt correlation when two
remediation tasks share npm dependencies:

| Component | Installed version | Findings | Coverage |
|---|---:|---|---|
| `express-jwt` | `0.1.3` | `CVE-2020-15084` | Parent package closure |
| `jsonwebtoken` | `0.1.0` | `CVE-2022-23539`, `CVE-2022-23540`, `CVE-2022-23541` | Nested child package closure |

The `jsonwebtoken@0.1.0` node is nested beneath `express-jwt` in the Juice
Shop lockfile. Both task closures therefore include that node and the shared
`jws` dependency. The fixture covers closure union, duplicate-safe lockfile
keys, and task-keyed QA evidence when separate tasks share dependency data.

## Inputs

* `triaged_groups_shared_dependencies.json` contains the two validated,
  pre-triaged vulnerability groups and their fix plans.
* `baseline_issues_shared_dependencies.jsonl` contains the five matching
  canonical `VulnerabilityIssue` records, one JSON object per line.

The runner validates the correspondence between these files before dispatch.
The group fixture bypasses initial triage; it does not bypass the current
Supervisor workflow. Supervisor creates the committed tasks, selects versions
and retries, and records each attempt snapshot. Each task's QA policy and
attempt-local evidence remain associated with its task even when package
closures overlap.

## Run

Run from the repository root against a disposable Juice Shop clone:

```bash
python examples/juice_shop/fixtures/shared_dependencies/run_shared_dependencies.py \
  --repo "$(pwd)/data/clones/juice-shop"
```

Optional `--groups` and `--issues` arguments select alternate structured group
and canonical JSONL fixtures. `--output` and `--patch-out` select where the
runner writes the result and unified patch. Parent directories are created as
needed. The runner never applies the patch to the clone; inspect it before
using it in a separate checkout.

A live run invokes the current Docker-backed workspace and worker/QA paths, so
Docker and the configured worker credentials are required. It exits `0` for a
`completed` or `completed_with_errors` graph result and `1` for another graph
status; missing repository or fixture paths exit `2`.

Default outputs are:

```text
data/trajectories/juice-shop-shared-dependencies-result.json
data/trajectories/juice-shop-shared-dependencies.patch
```

The result includes typed status, changed files, errors, and diff fields. The
full task queue, committed attempt snapshots, and task-keyed QA envelopes are
available in the internal state used to generate the result and trajectory.
