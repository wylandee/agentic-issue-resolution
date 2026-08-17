# Juice Shop Remediation Example

This directory contains the maintained end-to-end examples for remediating OWASP Juice Shop findings with `remediation_engine`.

## Prerequisites

1. Clone Juice Shop into `data/clones/juice-shop`:
   ```bash
   git clone https://github.com/juice-shop/juice-shop.git data/clones/juice-shop
   ```
2. Configure environment variables in `.env` (requires Docker and an `OPENAI_API_KEY` for LLM workers).

Note: All execution scripts run against isolated Docker volumes. The host clone is never edited.

### QA scan behavior

During normal task QA, the engine keeps `npm install` and the full test phase
unchanged. Supported npm lockfiles may use a temporary task-targeted ODC
closure read from the live Docker workspace after install. Yarn, pnpm,
ambiguous or incomplete npm closures, and targeted ODC/report failures use the
existing full-scan fallback. Targeted evidence is attempt-local; it does not
represent the repository-wide security state.

Every terminal remediation cycle with a workspace then runs one authoritative
full ODC scan before teardown. Findings from that scan can route the workflow
back through Supervisor-owned post-QA triage. Temporary targeted files are
removed before tests/final scanning, and Docker volume cleanup remains part of
teardown.

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

Runs the isolated `express-jwt` workaround replay from its pre-seeded fixture, followed by QA validation. The Workspace Builder initializes the baseline codebase; the replay runner then seeds the recorded post-update `express-jwt` state:

```bash
python examples/juice_shop/fixtures/workaround_replay/run_workaround_replay.py
```

### 4. NO_FIX workaround retry

Runs the `notevil` vulnerable-code-removal retry fixture through the Supervisor and workaround worker. The Workspace Builder initializes the baseline codebase before dispatch:

```bash
python examples/juice_shop/fixtures/workaround_nofix/run_workaround_nofix.py
```

### 5. Parent-first transitive dependency remediation

Runs three pre-triaged transitive findings—`@tootallnate/once`, `got`, and
`crypto-js`—whose direct parents are `sqlite3`, `download`, and `pdfkit`.
The fixture exercises parent updates before child overrides:

```bash
python examples/juice_shop/fixtures/transitive_parent_first/run_parent_first.py
```

### 6. Deterministic Supervisor routing

Runs a five-finding pre-triaged fixture containing one NO_FIX group and four
version-bump groups. The runner persists the final decision code, route, audit
record, task statuses, trajectory, and patch for routing-focused review:

```bash
python examples/juice_shop/fixtures/deterministic_routing/run_deterministic_routing.py
```

### 7. Shared dependency closures

Runs two pre-triaged update tasks for `express-jwt` and its nested
`jsonwebtoken` dependency. Their lockfile closures overlap, so the fixture is
useful for validating targeted closure union and duplicate-safe lockfile-key
provenance:

```bash
python examples/juice_shop/fixtures/shared_dependencies/run_shared_dependencies.py
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

The workaround scenarios are self-contained under `fixtures/`:

* **`fixtures/workaround_replay/run_workaround_replay.py`**: Express-JWT workaround-only replay runner.
* **`fixtures/workaround_replay/express_jwt_workaround_replay.json`**: Express-JWT replay state, task, and QA evidence.
* **`fixtures/workaround_nofix/run_workaround_nofix.py`**: `notevil` NO_FIX retry runner.
* **`fixtures/workaround_nofix/notevil_workaround_nofix.json`**: `notevil` Stage 2 retry state, task, and prior QA evidence.
* **`fixtures/transitive_parent_first/run_parent_first.py`**: Parent-first runner for three transitive SCA findings.
* **`fixtures/transitive_parent_first/triaged_groups_transitive_parent_first.json`**: Pre-triaged groups with dependency ancestry and direct-parent targets.

Both workaround runners use the Workspace Builder for initial npm dependency installation. The Express-JWT replay additionally applies the target dependency update inside the isolated volume so it can reproduce the failed update state; the NO_FIX retry starts from the unchanged baseline workspace.

The `fixtures/deterministic_routing/` scenario contains:

* **`triaged_groups_deterministic.json`**: Exactly five findings across five pre-triaged groups.
* **`run_deterministic_routing.py`**: Runner that validates the issue limit and persists routing evidence.
* **`README.md`**: Expected deterministic decision checkpoints and coverage notes.

The `fixtures/shared_dependencies/` scenario contains:

* **`triaged_groups_shared_dependencies.json`**: Two extracted pre-triaged groups with overlapping npm closures.
* **`baseline_issues_shared_dependencies.jsonl`**: The five matching canonical baseline issue records.
* **`run_shared_dependencies.py`**: Runner that validates the group/issue correspondence and executes the post-triage workflow.
* **`README.md`**: Shared-closure coverage and expected usage.

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
