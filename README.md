# Remediation Engine

`remediation-engine` is an agentic AppSec workflow that ingests Dependency-Check
or Semgrep findings, triages them, and produces a proposed remediation patch.
The host repository is never edited: work happens in temporary Docker volumes,
which are cleaned up during teardown.

## Install

```bash
python -m pip install -e ".[dev]"
```

Python 3.11 or newer and Docker are required. Copy `.env.example` to `.env`.

Before running a Python project, build the reusable workspace image once from
the repository root:

```bash
docker build -f docker/python-workspace.Dockerfile \
  -t remediation-engine-python:3.11 .
```

The engine uses this local image for Python sandboxes. It includes the
PostgreSQL and Python development headers and compiler toolchain needed by
dependencies that build native extensions.

Each fresh Python virtual environment also upgrades pip to the verified
`26.2.1` release before project dependencies are installed, updating pip's
bundled certificate store for that workspace.

Set `OPENAI_API_KEY` to enable the QA-aware tactical Supervisor and other
LLM-backed agents. The tactical Supervisor uses a stable system prompt plus a
bounded dynamic context, may select an immediate remediation pivot, and falls
back to the deterministic router when the key or model is unavailable. Python
validates targets, authorized candidate versions, and the worker instruction.
The Supervisor owns version policy; a worker may use an alternate only when
the committed attempt explicitly allows it. QA checks the effective version
actually executed. Portfolio planning and certification remain deterministic,
and triage/advisory proposals do not authorize mutations.

The runtime reads these environment names:

* **Provider and node models:** `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`,
  `TRIAGE_LLM_ENABLED`, `TRIAGE_LLM_MODEL`, `UPDATE_LLM_MODEL`,
  `WORKAROUND_LLM_MODEL`, `QA_LLM_MODEL`, `SERPER_API_KEY`, and
  `GITHUB_TOKEN`.
* **Scanning and workflow controls:** `ODC_EXTRA_ARGS`,
  `REMEDY_BYPASS_WORKAROUND_SUBAGENT`, `REMEDY_DISABLE_POST_QA_TRIAGE`,
  `REMEDY_RETRIAGE_LIMIT_ENABLED`, `REMEDY_RETRIAGE_LIMIT`, and
  `REMEDY_MAX_DELTA_CANARY_PROBES`. Set `REMEDY_FINAL_FULL_SCAN_ENABLED=true`
  to opt into a final full-repository Dependency-Check scan; it is disabled by
  default, which leaves the final result scoped to incremental dependency-input
  scan coverage.
* **Caching, reports, and tracing:** `TRIAGE_CACHE_DIR`,
  `REMEDIATION_SOLVER_CACHE_DIR`, `REMEDIATION_TRAJECTORY_DIR`,
  `REMEDIATION_REPORT_DIR`, `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
  `LANGSMITH_PROJECT`, and `LANGSMITH_ENDPOINT`.
* **Portfolio solver controls:** `REMEDY_SOLVER_TIMEOUT_SECONDS`,
  `REMEDY_SOLVER_CERTIFICATION_TIMEOUT_SECONDS`, `REMEDY_SOLVER_TOP_K`,
  `REMEDY_SOLVER_PHASE_BUDGET`,
  `REMEDY_SOLVER_ACCEPT_FEASIBLE`, `REMEDY_SOLVER_RANDOM_SEED`,
  `REMEDY_SOLVER_NUM_SEARCH_WORKERS`,
  `REMEDY_SOLVER_MAX_CANDIDATES_PER_TARGET`,
  `REMEDY_SOLVER_MAX_MODEL_VARIABLES`, `REMEDY_SOLVER_LLM_ENABLED`, and
  `SOLVER_LLM_MODEL`.

Unset node-specific model values use `REMEDY_LLM_MODEL`. The retriage limit is
a development guard and is unlimited by default.

## CLI

The canonical issue interchange format is JSONL: one validated issue object per
line. `ingest` accepts `odc-json`, `semgrep-json`, or canonical `jsonl` input
(`--format auto` detects `.jsonl` and `.ndjson`; use an explicit format for
scanner reports) and writes canonical JSONL. `triage` consumes canonical JSONL
and writes its JSON group result. `solve` consumes pre-triaged groups JSON and
builds a deterministic, offline portfolio/DAG plan without workers or host
repository mutation; pass `--tasks` for an optional task-queue fixture. `run`
consumes canonical JSONL and writes a typed result JSON, with an optional
reviewable unified diff and Markdown report.

For a solver dry run:

```bash
remedy solve /tmp/remediation-groups.json --repo "$REPO_ROOT" \
  --output /tmp/portfolio-plan.json
```

After cloning a target repository, run the maintained Juice Shop fixture
through the complete flow:

```bash
git clone https://github.com/juice-shop/juice-shop.git data/clones/juice-shop
REPO_ROOT="$(pwd)/data/clones/juice-shop"
remedy ingest examples/juice_shop/fixtures/dependency-check-report-baseline.json \
  --format odc-json --output /tmp/remediation-findings.jsonl
remedy triage /tmp/remediation-findings.jsonl --repo "$REPO_ROOT" \
  --output /tmp/remediation-groups.json
remedy run /tmp/remediation-findings.jsonl --format jsonl --repo "$REPO_ROOT" \
  --output /tmp/remediation-result.json \
  --patch-out /tmp/remediation.patch \
  --report-out /tmp/remediation-report.md
```

`run` executes the Phase 5 task queue. Update and workaround workers execute
Supervisor-committed attempts in isolation. QA runs deterministic install,
security-scan, and test gates, then performs a bounded read-only evaluation
for each task. QA results remain keyed to their task. By default, the terminal
result reports incremental dependency-input scan coverage and is not an
authoritative full-repository scan. Set `REMEDY_FINAL_FULL_SCAN_ENABLED=true`
to request the optional final full scan before teardown.

Tactical supervision is single-task and QA-aware. It uses a stable
system prompt plus a bounded, deterministically ordered context so task facts
do not invalidate the reusable prompt prefix. The model returns only a typed
diagnostic basis and strategy proposal; Python verifies the normalized
security floor, candidate whitelist, task identity, dependency type, and
relative file hints, then renders and digests the exact worker instruction.
Breaking-change evidence can pivot directly to a workaround. Peer conflicts
are referred to the Supervisor's deterministic portfolio-planning function,
which owns package clustering and multi-package planning; tactical decisions
remain single-task.
Without an API key, or when the model returns no accepted action, the
deterministic router remains authoritative. If registry verification fails,
the affected task becomes inconclusive with its retry budget preserved, and
the Supervisor re-enters routing for any remaining tasks; no unverified
package version is dispatched.

Workaround retries restore only the child source-edit snapshot when QA rejects
an intermediate hypothesis, preserving the resolved dependency candidate.
The original parent rollback anchor is consumed only after workaround success
or terminal abandonment. A successful `validate_workaround` gate terminates
the bounded worker loop after all tool responses in that model turn are
recorded, so a later max-round guard cannot turn a passing workaround into a
surrender.

`remedy run` exits with `0` for a completed run without errors, `1` for a
completed run with remediation errors or unfixable tasks, and `2` for invalid
input or missing prerequisites. Offline `remedy solve` exits with `0` for an
`OPTIMAL` or `FEASIBLE` solver status without diagnostics, `1` for other solver
statuses or diagnostics, and `2` for invalid input or missing prerequisites.
`solve` is a non-dispatchable preview: package resolution remains `NOT_RUN` and
`dispatchable` is false. `--output` and `--patch-out` write files; without an
output path, serialized output is written to stdout.

## Python API

The public package exports `RemediationRequest`, `RemediationResult`,
`run_remediation`, and `triage_issues`:

```python
from pathlib import Path

from remediation_engine import RemediationRequest, run_remediation

# Run this after cloning the target in the Install/CLI example.
repo_root = (Path.cwd() / "data/clones/juice-shop").resolve()
request = RemediationRequest(repo_root=repo_root, issues=[])
result = run_remediation(request)

print(result.status)
print(result.diff)  # reviewable unified diff; may be empty
print(result.errors)
```

`repo_root` must resolve to an existing absolute directory. The typed
`RemediationResult` includes `status`, `changed_files`, `diff`, `errors`, and
optional trajectory/report paths. The API and CLI never apply changes to the
host repository. Use `triage_issues` when callers need to create actionable
groups explicitly; internal graph state, workers, and Docker clients are not
public API.

The immutable portfolio surface is available from
`remediation_engine.orchestration` for callers that already have validated
triage groups and a repository workspace:

```python
from remediation_engine.orchestration import (
    apply_portfolio_plan,
    build_portfolio_plan,
    prepare_portfolio_inputs,
)

groups, task_queue, diagnostics = prepare_portfolio_inputs(repo_root, groups, task_queue)
plan = build_portfolio_plan(repo_root, groups, task_queue)
groups, task_queue, diagnostics = apply_portfolio_plan(plan, groups, task_queue)
```

These functions are copy-on-write and return typed plan/task projections; they
do not dispatch workers or mutate the repository. The lower-level occurrence
solver entry points remain under `remediation_engine.solver`.

## Development and evaluation

```bash
python -m pytest
ruff check .
ruff format --check .
python -m compileall -q src
remedy --help
python scripts/eval_compare.py --help
```

Evaluation tests and their registered datasets are documented in `EVALS.md`.
For maintained end-to-end workflows, see
[`examples/juice_shop/README.md`](examples/juice_shop/README.md) and
[`examples/NodeGoat/README.md`](examples/NodeGoat/README.md). NodeGoat is a
secondary fixture set; architecture and package-boundary details are in
[`docs/architecture.md`](docs/architecture.md).
