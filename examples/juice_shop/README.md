# Juice Shop Remediation Example

This directory contains the maintained end-to-end examples for remediating OWASP Juice Shop findings with `remediation_engine`.

## Prerequisites

1. Clone Juice Shop into `data/clones/juice-shop`:
   ```bash
   git clone https://github.com/juice-shop/juice-shop.git data/clones/juice-shop
   ```
2. Configure environment variables in `.env` (requires Docker and an `OPENAI_API_KEY` for LLM workers).

Note: All execution scripts run against isolated Docker volumes. The host clone is never edited.

---

## Example Scenarios

### 1. Whole-pipeline execution

Runs the full remediation workflow from the canonical baseline issue fixture. The engine performs triage, version selection, subagent execution, and QA evaluation:

```bash
python examples/juice_shop/run.py
```

`run.py` accepts any canonical JSONL issue fixture through `--issues`, so a different static subset can be selected without changing the runner:

```bash
python examples/juice_shop/run.py --issues examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl
```

### 2. Suppressed post-triage execution

Loads the pre-triaged groups from `fixtures/suppressed/triaged_groups_suppressed.json`, bypasses the initial triage node, and runs the post-triage workflow directly against the workspace:

```bash
python examples/juice_shop/fixtures/suppressed/run_post_triage.py
```

### 3. Workaround subagent replay

Runs the isolated `express-jwt` workaround replay from its pre-seeded fixture, followed by QA validation:

```bash
python examples/juice_shop/workaround_replay/run_workaround_replay.py
```

---

## Fixture layout

The fixture directory contains both runnable inputs and scanner/triage provenance:

* **`baseline_issues.jsonl`**: Canonical ODC-derived issues consumed by `run.py` by default.
* **`dependency-check-report-baseline.json`**: Baseline raw ODC JSON report retained as scan provenance.
* **`dependency-check-report-baseline.html`**: Human-readable baseline ODC report.
* **`triaged_groups_baseline.json`**: Precomputed baseline triage output retained for inspection; `run.py` performs triage itself.

The `fixtures/suppressed/` scenario contains:

* **`suppressions.xml`**: ODC suppression rules associated with the selected subset.
* **`odc_suppressed_issues.jsonl`**: Canonical issue subset used when selecting suppressed findings with `run.py --issues`.
* **`triaged_groups_suppressed.json`**: Pre-triaged groups consumed by `run_post_triage.py`.
* **`run_post_triage.py`**: Runner for the post-triage scenario.
* **`extract_suppressed.py`**: Helper script to extract target package issues from `baseline_issues.jsonl` into `odc_suppressed_issues.jsonl`.

The raw suppressed ODC report is not currently included in this directory. The checked-in suppressed issue and group files are static fixtures; changing the suppression rules requires producing a new ODC report externally and refreshing those derived files.

The workaround scenario is self-contained under `workaround_replay/`:

* **`run_workaround_replay.py`**: Workaround-only replay runner.
* **`express_jwt_workaround_replay.json`**: Express-JWT replay state, task, and QA evidence.

### Refreshing derived fixtures manually

To extract target package entries directly from `baseline_issues.jsonl` into `fixtures/suppressed/odc_suppressed_issues.jsonl`:

```bash
python examples/juice_shop/fixtures/suppressed/extract_suppressed.py @angular/common
```

You can pass multiple package names or specify custom `--input` and `--output` paths:

```bash
python examples/juice_shop/fixtures/suppressed/extract_suppressed.py @angular/common @angular/compiler --output custom_issues.jsonl
```

After producing an ODC JSON report with the desired suppression rules, normalize and triage it with the `remedy` CLI:

1. **Ingest the report into canonical JSONL issues:**

   ```bash
   remedy ingest path/to/dependency-check-report.json \
     --output examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl
   ```

2. **Triage the selected issues into groups:**

   ```bash
   remedy triage examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl \
     --repo data/clones/juice-shop \
     --output examples/juice_shop/fixtures/suppressed/triaged_groups_suppressed.json
   ```
