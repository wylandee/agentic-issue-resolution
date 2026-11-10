# Remediation Engine

The Remediation Engine ingests Dependency-Check or Semgrep findings, triages
them, and proposes a patch through an isolated remediation workflow. Its public
Python API and CLI return typed results and a unified diff. The engine does not
modify the host repository.

## Install and configure

Python 3.11 or newer and Docker are required for remediation runs. Install the
engine and development tools, then configure provider credentials locally:

```bash
python -m pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` for model-backed agents. Unset role-specific model values
inherit `REMEDY_LLM_MODEL`. Python projects also need the reusable workspace
image, built once from the repository root:

```bash
docker build -f docker/python-workspace.Dockerfile \
  -t remediation-engine-python:3.11 .
```

`AppSettings` reads configuration at application boundaries. Preserve these
environment interfaces when integrating the package:

- Models and providers: `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`,
  `TRIAGE_LLM_ENABLED`, `TRIAGE_LLM_MODEL`, `SUPERVISOR_LLM_MODEL`,
  `UPDATE_LLM_MODEL`, `WORKAROUND_LLM_MODEL`, `QA_LLM_MODEL`, `SERPER_API_KEY`,
  and `GITHUB_TOKEN`.
- Scans and workflow: `ODC_EXTRA_ARGS`,
  `REMEDY_FINAL_FULL_SCAN_ENABLED`, `REMEDY_BYPASS_WORKAROUND_SUBAGENT`,
  `REMEDY_DISABLE_POST_QA_TRIAGE`, `REMEDY_RETRIAGE_LIMIT_ENABLED`,
  `REMEDY_RETRIAGE_LIMIT`, and `REMEDY_MAX_DELTA_CANARY_PROBES`.
- Storage and tracing: `TRIAGE_CACHE_DIR`, `REMEDIATION_SOLVER_CACHE_DIR`,
  `REMEDIATION_TRAJECTORY_DIR`, `REMEDIATION_REPORT_DIR`,
  `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and
  `LANGSMITH_ENDPOINT`.
- Evaluation: `EVAL_JUDGE_MODEL`, `EVAL_JUDGE_REASONING_EFFORT`, and
  `EVAL_DB_PATH`.
- Solver controls: `REMEDY_SOLVER_TIMEOUT_SECONDS`,
  `REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS`, `REMEDY_SOLVER_TOP_K`,
  `REMEDY_SOLVER_PHASE_BUDGET`, `REMEDY_SOLVER_ACCEPT_FEASIBLE`,
  `REMEDY_SOLVER_RANDOM_SEED`, `REMEDY_SOLVER_NUM_SEARCH_WORKERS`,
  `REMEDY_SOLVER_MAX_CANDIDATES_PER_TARGET`,
  `REMEDY_SOLVER_MAX_MODEL_VARIABLES`, `REMEDY_SOLVER_LLM_ENABLED`, and
  `SOLVER_LLM_MODEL` (the last two are parsed for compatibility but are not
  currently used by the production solver).

Do not commit `.env`, credentials, clones, reports, or trajectories. The final
full-repository Dependency-Check scan is opt-in with
`REMEDY_FINAL_FULL_SCAN_ENABLED=true`; otherwise terminal scan evidence covers
the dependency inputs scanned incrementally.

## Run the CLI

Canonical issue JSONL contains one validated issue object per line. `ingest`
accepts ODC JSON, Semgrep JSON, or canonical JSONL; `triage` writes grouped
findings; `solve` previews a portfolio without dispatching workers; and `run`
executes remediation and writes a typed result, optional patch, and report.

```bash
remedy ingest scanner-report.json --format odc-json --output findings.jsonl
remedy triage findings.jsonl --repo "$REPO_ROOT" --output groups.json
remedy solve groups.json --repo "$REPO_ROOT" --output portfolio.json
remedy run findings.jsonl --format jsonl --repo "$REPO_ROOT" \
  --output result.json --patch-out proposed.patch --report-out report.md
```

`solve` is an offline planning preview: package resolution is `NOT_RUN`, the
result is non-dispatchable, and it does not mutate the repository. `run` uses
temporary Docker workspaces. Exit status is `0` for a clean completion, `1`
for a completed run with remediation errors or unresolved tasks, and `2` for
invalid input or missing prerequisites. Review the returned diff before using
any proposed change.

The package API exports `RemediationRequest`, `RemediationResult`,
`run_remediation`, and `triage_issues`:

```python
from pathlib import Path

from remediation_engine import RemediationRequest, run_remediation

request = RemediationRequest(repo_root=Path("/path/to/checkout"), issues=[])
result = run_remediation(request)
print(result.status, result.diff, result.errors)
```

## Shared example batch runner

The maintained examples are four canonical finding baselines. They are input
data, not runnable clones:

| `--codebase` | Example baseline | Ecosystem |
| --- | --- | --- |
| `juice_shop` | `examples/juice_shop/baseline_issues.jsonl` | npm / JavaScript |
| `nodegoat` | `examples/nodegoat/baseline_issues.jsonl` | npm / JavaScript |
| `vulnado` | `examples/vulnado/baseline_issues.jsonl` | Maven / Java |
| `pygoat` | `examples/pygoat/baseline_issues.jsonl` | PyPI / Python |

One interface lists profile-specific actionable packages, prepares a bounded
seeded sample without external services, or runs a selected batch against a
caller-owned clone:

```bash
python scripts/run_batch.py --codebase juice_shop --list-packages
python scripts/run_batch.py --codebase juice_shop --iterations 3 \
  --batch-size 2 --seed 17 --dry-run
python scripts/run_batch.py --codebase juice_shop --repo data/clones/juice-shop \
  --packages express-jwt lodash --output-dir data/evals/batch-runs
```

Use `--codebase nodegoat`, `vulnado`, or `pygoat` for the other profiles; pass
`--baseline PATH` to override a baseline. Explicit `--packages` selects one
batch. The runner validates ecosystem identity, scopes findings to selected
packages, and stores per-batch issues, result, patch, and a run summary under
the output directory. It does not overwrite the caller's clone. Dry runs do
not need clones, API keys, Docker, or network services.

## Workflow and safety contracts

The graph runs `initial_triage` → `workspace_builder` →
`workspace_setup_scan` → `supervisor` → committed update/workaround workers and
QA → `teardown` → `report`. No-work and setup failures can route directly to
teardown. Post-QA reconciliation and the optional final full scan run only
when required.

The Supervisor invokes portfolio planning and resolver certification. The
task queue and sealed attempt snapshots are the execution authority: every
worker result must match the current attempt and task revision, and workers
cannot query registries or broaden the committed candidate set. Multi-package
mutations are atomic and carry per-attempt provenance. Failed-batch isolation
uses bounded canary checks and assigns blame only after restoration is
verified. Workaround children preserve the approved dependency candidate and
parent rollback anchors remain until success or terminal abandonment.

QA maps scanner observations to physical dependency occurrences and evaluates
each task against its task-scoped evidence. Repository-wide findings are
reconciliation context, not evidence against unrelated tasks. Infrastructure
or incomplete scan coverage is inconclusive and does not consume a remediation
retry. Docker volumes are removed during teardown on every path.

## Tests and evaluations

Unit and offline evaluation tests mock Docker, HTTP, subprocess, and model
boundaries. In restricted environments the repository-local virtualenv,
telemetry opt-out, and disabled pytest rerun listener keep tests self-contained:

```bash
DEEPEVAL_TELEMETRY_OPT_OUT=YES .venv/bin/python -m pytest \
  -p no:rerunfailures -m 'not docker and not network and not llm' -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
remedy --help
python scripts/run_batch.py --help
python scripts/eval_compare.py --help
```

The seven maintained evaluation datasets contain 14 report, 20 triage, 22
update-worker, 19 workaround-worker, 31 QA, 20 Supervisor, and 10 end-to-end
cases. The Supervisor cases include eight fix-planning scenarios. Offline replay invokes production node boundaries
with scripted model, scanner, registry, and workspace doubles. Observed
historical traces are kept separate from independently authored expectations.
Offline evaluation makes no judge calls. Model-judged suites require
`.[eval]`, a usable `OPENAI_API_KEY`, and `--run-eval-live`. Fresh end-to-end
goldens additionally run the real engine with Docker and its configured
scanner/model services; replay or rescoring a saved trajectory is not a fresh
remediation run.

The end-to-end evaluator uses one Task Completion metric at threshold `0.70`.
Deterministic checks enforce attempt lineage, authorized versions, QA evidence,
patch support, rollback, host preservation, and cleanup. Judge scores are
diagnostic unless `--e2e-strict` is selected. The Supervisor suite also uses
one Task Completion metric per case for its combined planning and disposition.
The live report suite requires a generated Executive Summary and records three
diagnostic metrics: Hallucination (`0.70`), Faithfulness (`0.85`), and
Summarization (`0.80`). Operational token, latency, and tool-call limits remain
runtime controls and telemetry, not evaluation scores. A fallback summary does
not count as generated in live mode.

Evaluation runs and metric results are persisted to SQLite at
`data/evals/eval_results.db` by default. Set `EVAL_DB_PATH` to use another
database; use `--eval-tag` and `--eval-baseline` to label and compare pytest
runs. `scripts/eval_compare.py --list`, `--latest`, or `--run-a TAG --run-b TAG`
compares persisted runs. Comparisons are diagnostic, not a substitute for
pytest results or remediation outcome evidence.

Install the optional evaluation dependencies with
`python -m pip install -e ".[dev,eval]"`.
Launch the evaluation dashboard with `remedy-ui` (or
`python -m remediation_engine.evals.ui`).

## Local web demo

The demo provides a local dashboard for attaching a codebase and findings,
viewing run events, and inspecting the final report and patch. Install its
backend dependencies and build the frontend once:

```bash
python -m pip install -r demo/backend/requirements.txt
cd demo/frontend && npm ci && npm run build && cd ../..
./demo/start_demo.sh
```

For Vite hot reload, run `./demo/start_demo.sh --dev`. The demo is a local
operator interface; attach a repository that contains both its source and
findings report or canonical JSONL input.
