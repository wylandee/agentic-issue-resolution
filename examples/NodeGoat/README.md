# NodeGoat remediation example

This directory is a maintained secondary test and evaluation fixture set for
OWASP NodeGoat. It exercises the public `remediation_engine` API with a
repository clone, canonical scanner findings, and the current task/attempt
workflow.

Commands below are run from the repository root. Clone paths are resolved to
absolute paths by the runners before they are passed to the API. A normal
single run copies the clone into a temporary Docker workspace; the host clone
is not edited by the engine. The result is a typed status/changed-files/error
projection and a unified diff, both written only to the paths selected by the
runner.

## Prerequisites

1. Clone NodeGoat at the path used by the default runner:

   ```bash
   git clone https://github.com/OWASP/NodeGoat.git data/clones/NodeGoat
   ```

   Use `--repo /absolute/path/to/NodeGoat` when the clone is elsewhere.

2. Copy `.env.example` to `.env` and set the credentials needed by the
   execution path. Live remediation requires Docker. Worker execution uses
   the configured LLM model and normally requires `OPENAI_API_KEY`; a dry run
   does not start Docker or call the engine.

The raw Dependency-Check reports are retained as scan provenance. The checked-
in suppressed fixture contains five representative findings so a run is small
enough to inspect and replay.

## Single-run workflow

`run.py` loads the suppressed canonical issue fixture by default, then sends
those findings to the public API. The graph performs initial triage when no
pre-triaged groups are supplied; Supervisor then creates and routes committed
remediation tasks, records attempt snapshots, and owns version/retry decisions.
Update/workaround workers execute the selected attempt, and QA evaluates the
same task against its committed QA policy with deterministic install, scan, and
test evidence.

```bash
python examples/NodeGoat/run.py
```

The default inputs and outputs are:

```text
Input:  examples/NodeGoat/fixtures/suppressed/odc_suppressed_issues.jsonl
Result: data/trajectories/nodegoat-result.json
Patch:  data/trajectories/nodegoat.patch
```

Issue input is canonical JSONL: one `VulnerabilityIssue` object per line. Use
`--issues` for another JSONL fixture and `--output` or `--patch-out` to choose
absolute or relative output paths (the runner resolves them before writing):

```bash
python examples/NodeGoat/run.py \
  --repo "$(pwd)/data/clones/NodeGoat" \
  --issues examples/NodeGoat/fixtures/baseline_issues.jsonl \
  --output /tmp/nodegoat-result.json \
  --patch-out /tmp/nodegoat.patch
```

To normalize a raw Dependency-Check report before running it, use the CLI
boundary. The output of `ingest` is canonical JSONL and is suitable for
`run.py`:

```bash
remedy ingest examples/NodeGoat/dependency-check-report-baseline.json \
  --format odc-json --output /tmp/nodegoat-issues.jsonl
python examples/NodeGoat/run.py --issues /tmp/nodegoat-issues.jsonl
```

The single-run script exits `0` only for `completed` with no errors. A
completed run with recorded errors exits `1`; missing repository or fixture
paths exit `2`. Inspect the JSON result and patch before applying any change to
a separate checkout. No runner applies the emitted patch to the host clone.

## Batch fixture preparation

`run_batch.py` samples distinct package batches from the read-only
`fixtures/baseline_issues.jsonl`, writes each selected subset as canonical
JSONL, and records per-iteration result/patch paths under
`data/trajectories/`. Use a seed for reproducible selection:

```bash
python examples/NodeGoat/run_batch.py \
  --iterations 2 --batch-size 3 --seed 7
```

Use `--dry-run` to prepare the sampled JSONL and suppression rules without
calling `run_remediation`; this is the deterministic, no-Docker/no-LLM mode:

```bash
python examples/NodeGoat/run_batch.py \
  --iterations 2 --batch-size 3 --seed 7 --dry-run
```

The batch helper updates the checked-in suppressed JSONL and XML paths and
copies the selected `suppressions.xml` into the clone before each iteration,
including dry runs. Run it only against a disposable clone and fixture copy if
the source checkout must remain untouched. Its aggregate summary is
`data/trajectories/nodegoat-batch-runs-summary.json`; live iteration files are
`nodegoat-run-01-result.json` and `nodegoat-run-01.patch` (with the iteration
number substituted).

## Suppressed fixture

`fixtures/suppressed/odc_suppressed_issues.jsonl` is the runnable five-finding
canonical input. `fixtures/suppressed/suppressions.xml` records the associated
package-level Dependency-Check suppression scope. The full
`fixtures/baseline_issues.jsonl` and raw JSON/HTML reports remain provenance
and are not modified by a single run.

NodeGoat fixtures are intentionally retained as a secondary test/evaluation
set. Refresh or generate issue subsets with `remedy ingest`; do not replace
JSONL issue files with a JSON array or pass a raw scanner report directly to
`run.py`.
