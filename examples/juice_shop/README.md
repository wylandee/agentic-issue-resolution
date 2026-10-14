# Juice Shop remediation example

This directory contains the maintained end-to-end examples for remediating OWASP
Juice Shop findings with `remediation_engine`.

Commands below are run from the repository root. The main runners resolve
repository and issue paths before invoking the public API; fixture runners also
resolve their repository and input paths and create output parents as needed. A
normal single run gives the engine an immutable host clone as its source and
performs execution in temporary Docker storage. The API returns a typed result
with status, changed files, errors, and a unified diff; it does not apply that
diff to the host repository.

## Prerequisites

1. Clone Juice Shop at the default path:

   ```bash
   git clone https://github.com/juice-shop/juice-shop.git data/clones/juice-shop
   ```

   Pass `--repo /absolute/path/to/juice-shop` to any runner to use another
   clone.

2. Copy `.env.example` to `.env` and configure the credentials needed by the
   selected path. Live remediation and the replay scenarios require Docker.
   Worker execution normally requires `OPENAI_API_KEY`; deterministic fixture
   loading and batch preparation can be run without making LLM calls.

Temporary Docker volumes are cleaned up during teardown. A final full scan is
Supervisor-owned and runs before teardown for a terminal remediation cycle.
The task queue and committed attempt snapshots are authoritative: Supervisor
creates tasks, selects versions, owns retries and pivots, and carries each
attempt's QA policy into QA. Update/workaround workers execute committed
attempts; QA records task-keyed evidence from install, scan, tests, and bounded
read-only evaluation.

## Single-run workflow

`run.py` loads `fixtures/baseline_issues.jsonl` by default and performs initial
triage before task creation and routing:

```bash
python examples/juice_shop/run.py
```

The result and patch default to:

```text
data/trajectories/juice-shop-result.json
data/trajectories/juice-shop.patch
```

Use another canonical JSONL issue fixture with `--issues`; the runner does not
accept a JSON array or a raw scanner report:

```bash
python examples/juice_shop/run.py \
  --repo "$(pwd)/data/clones/juice-shop" \
  --issues examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl \
  --output /tmp/juice-shop-result.json \
  --patch-out /tmp/juice-shop.patch
```

To convert a raw Dependency-Check report at the supported boundary, normalize
it first and then pass the resulting JSONL file to the runner:

```bash
remedy ingest examples/juice_shop/dependency-check-report-baseline.json \
  --format odc-json --output /tmp/juice-shop-issues.jsonl
python examples/juice_shop/run.py --issues /tmp/juice-shop-issues.jsonl
```

The script exits `0` only for `completed` with no errors, `1` when the run
records remediation errors, and `2` when a repository or issue fixture path is
missing. Review the result and unified diff before applying changes to any
separate checkout.

## Pre-triaged and replay scenarios

Pre-triaged runners load structured group fixtures and derive their canonical
issue baseline from the issues embedded in those groups. They bypass initial
triage but still invoke the current task/attempt graph and therefore are live
runs, not offline simulations.

### Suppressed post-triage run

```bash
python examples/juice_shop/fixtures/suppressed/run_post_triage.py
```

The runner reads `fixtures/suppressed/triaged_groups_suppressed.json`, emits a
result at `data/trajectories/juice-shop-suppressed-result.json`, and emits a
unified patch at `data/trajectories/juice-shop-suppressed.patch`.

### Retriage run

```bash
python examples/juice_shop/fixtures/retriage/run_retriage.py
```

This fixture exercises the Supervisor-owned post-QA retriage path for the
pre-triaged `sanitize-html` group. Its output is written under
`data/trajectories/` as `juice-shop-retriage-result.json` and
`juice-shop-retriage.patch`.

### Workaround replay

The Express-JWT replay is a private, opt-in execution path. It loads an
explicit task with its `qa_policy` and prior attempt evidence, seeds the
post-update state inside a temporary Docker volume, dispatches the workaround
worker, and runs QA against that same volume. It is intended for live
Docker/LLM or LangSmith trials, not for the public API workflow:

```bash
python examples/juice_shop/fixtures/workaround_replay/run_workaround_replay.py
```

The timestamped JSON result and unified patch are written under
`data/trajectories/` unless `--output` and `--patch-output` are provided. The
host clone is used only as the immutable source and diff baseline.

### NO_FIX package-removal retry

This opt-in fixture loads a committed retry task with the
`no_fix_code_removal` QA policy and exercises workaround-worker, Supervisor,
and QA routing against a temporary Docker volume:

```bash
python examples/juice_shop/fixtures/workaround_nofix/run_workaround_nofix.py
```

It is a live Docker/LLM script, not an offline check. Its runner prints the
result and QA status and cleans up the temporary volume; inspect the generated
result and diff before any manual application.

## Deterministic routing and shared closures

### Deterministic Supervisor routing

The deterministic fixture contains five pre-triaged groups: one `NO_FIX` group
and four version-bump groups. Because initial triage is bypassed, group order,
task creation, and Supervisor route decisions can be inspected without an
initial-triage model decision. Worker execution and QA still run through the
live graph, so Docker and the configured worker credentials are required for a
full run:

```bash
python examples/juice_shop/fixtures/deterministic_routing/run_deterministic_routing.py
```

The runner validates the five-group fixture, persists a routing summary with
final decision code, next route, Supervisor audit, and task-keyed statuses, and
writes the result and patch to `data/trajectories/`:

```text
data/trajectories/juice-shop-deterministic-routing-result.json
data/trajectories/juice-shop-deterministic-routing.patch
```

A graph result of `completed` or `completed_with_errors` is a successful process
exit for this routing-focused runner; remediation or QA failures remain in the
result.

### Shared dependency closures

This fixture runs two pre-triaged update groups whose npm dependency closures
overlap. The runner validates the five matching canonical JSONL issue records,
then checks that task-keyed attempts and QA evidence preserve each task's
identity even when lockfile keys are shared:

```bash
python examples/juice_shop/fixtures/shared_dependencies/run_shared_dependencies.py
```

It writes `data/trajectories/juice-shop-shared-dependencies-result.json` and
`data/trajectories/juice-shop-shared-dependencies.patch`. See the fixture README
for the exact components and closure coverage.

## Batch fixture preparation

`run_batch.py` samples distinct package batches from the read-only baseline
JSONL, writes each selected subset as canonical JSONL, and stores per-iteration
results and patches under `data/trajectories/`:

```bash
python examples/juice_shop/run_batch.py \
  --iterations 2 --batch-size 3 --seed 7
```

The batch runner is a development-scoped fixture: each request passes the
selected package names to the engine, so synthetic portfolio tasks are limited
to that package set and its required Angular/workspace/peer coordination
closure. To run one explicit Angular batch instead of sampling:

```bash
python examples/juice_shop/run_batch.py \
  --packages @angular/common @angular/compiler @angular/core
```

The normal `examples/juice_shop/run.py` entrypoint does not set a package
scope and therefore retains full-repository behavior.

`--dry-run` prepares the sampled issue and suppression fixtures without calling
the remediation engine, Docker, or LLMs:

```bash
python examples/juice_shop/run_batch.py \
  --iterations 2 --batch-size 3 --seed 7 --dry-run
```

The batch helper rewrites the suppressed JSONL/XML fixture paths and copies the
selected suppression file into the clone before each iteration, including dry
runs. Use a disposable clone and fixture copy when preserving the source
checkout matters. The aggregate summary is
`data/trajectories/juice-shop-batch-runs-summary.json`; live iteration files are
`juice-shop-run-01-result.json` and `juice-shop-run-01.patch`.

## Fixture layout and refresh

* `fixtures/baseline_issues.jsonl` is the canonical baseline issue input for
  `run.py`.
* `fixtures/suppressed/odc_suppressed_issues.jsonl` is a canonical JSONL subset
  for selected suppressed findings.
* `dependency-check-report-baseline.json` and `.html` are retained raw scan
  provenance, not runner inputs.
* `triaged_groups_baseline.json` and the scenario `triaged_groups_*.json` files
  are pre-triaged group/task planning fixtures. They are not substitutes for
  canonical JSONL issue interchange.
* `fixtures/suppressed/suppressions.xml` contains the selected ODC suppression
  rules.

To refresh a suppressed issue subset, use the maintained extractor with a
canonical JSONL input:

```bash
python examples/juice_shop/fixtures/suppressed/extract_suppressed.py \
  @angular/common @angular/compiler \
  --output examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl
```

To normalize a newly produced ODC report and derive groups, use the CLI:

```bash
remedy ingest path/to/dependency-check-report.json \
  --format odc-json \
  --output examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl
remedy triage examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl \
  --repo "$(pwd)/data/clones/juice-shop" \
  --output examples/juice_shop/fixtures/suppressed/triaged_groups_suppressed.json
```

Keep issue files newline-delimited and preserve task, attempt, and QA-policy
fields when maintaining a pre-triaged or replay fixture.
