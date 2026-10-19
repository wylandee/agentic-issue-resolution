# Vulnado remediation example

This example runs the Java/Maven remediation workflow against the local Vulnado clone at `data/clones/vulnado`.

`fixtures/baseline_issues.jsonl` contains canonical findings ingested from the retained Dependency-Check report. The runnable suppressed fixture contains five findings across three Maven packages; its suppression rules keep those packages active and suppress the other packages from the same baseline. The raw JSON and HTML reports are retained as scan provenance.

## Single run

Run one remediation using the five-finding fixture:

```bash
python examples/Vulnado/run.py
```

Override the repository, input, or output paths as needed:

```bash
python examples/Vulnado/run.py --repo /absolute/path/to/vulnado --issues examples/Vulnado/fixtures/suppressed/odc_suppressed_issues.jsonl --output data/trajectories/vulnado-result.json --patch-out data/trajectories/vulnado.patch
```

The single-run script writes a typed result and unified patch under `data/trajectories/` by default. The engine uses an isolated workspace and does not apply the patch to the host clone.

## Batch runs

`run_batch.py` samples distinct Maven package batches from the full baseline, writes each selected subset to the suppressed JSONL fixture, and updates package suppression rules:

```bash
python examples/Vulnado/run_batch.py --iterations 2 --batch-size 3 --seed 7 --dry-run
```

Remove `--dry-run` to run remediation. The batch runner copies the generated `suppressions.xml` into the clone before each iteration, including dry runs, so use a disposable clone when you need to preserve the checkout. Results and patches are written under `data/trajectories/`; the aggregate summary is `vulnado-batch-runs-summary.json`.

To run one explicit package batch instead of sampling, pass exact Maven package
coordinates from the baseline. For example, to select the aligned Tomcat
artifacts:

```bash
python examples/Vulnado/run_batch.py \
  --packages \
  org.apache.tomcat.embed:tomcat-embed-core \
  org.apache.tomcat.embed:tomcat-embed-websocket \
  --dry-run
```

Explicit selections are trimmed, deduplicated, sorted, and checked against the
baseline before any fixture files are written. They produce one batch regardless
of `--iterations` or `--batch-size`. Batch requests pass their selected package
names as a development-only engine allowlist; the single-run `run.py` entrypoint
remains full-repository scoped.

The full baseline and the suppressed fixture use canonical JSONL. To regenerate the full baseline from the raw report, run:

```bash
remedy ingest examples/Vulnado/dependency-check-report-baseline.json --format odc-json --output examples/Vulnado/fixtures/baseline_issues.jsonl
```
