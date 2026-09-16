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
Set `OPENAI_API_KEY` to enable the QA-aware tactical Supervisor and other
LLM-backed agents. The tactical Supervisor uses a stable system prompt plus a
bounded dynamic context, may select an immediate remediation pivot, and falls
back to the deterministic router when the key or model is unavailable. Python
still verifies every target and version and commits the worker instruction.

The runtime reads these environment names:

* **Provider and node models:** `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`,
  `TRIAGE_LLM_ENABLED`, `TRIAGE_LLM_MODEL`, `UPDATE_LLM_MODEL`,
  `WORKAROUND_LLM_MODEL`, `QA_LLM_MODEL`, `SERPER_API_KEY`, and
  `GITHUB_TOKEN`.
* **Scanning and workflow controls:** `ODC_EXTRA_ARGS`,
  `REMEDY_BYPASS_WORKAROUND_SUBAGENT`, `REMEDY_DISABLE_POST_QA_TRIAGE`,
  `REMEDY_RETRIAGE_LIMIT_ENABLED`, and `REMEDY_RETRIAGE_LIMIT`.
* **Caching, reports, and tracing:** `TRIAGE_CACHE_DIR`,
  `REMEDIATION_TRAJECTORY_DIR`, `REMEDIATION_REPORT_DIR`,
  `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and
  `LANGSMITH_ENDPOINT`.

Unset node-specific model values use `REMEDY_LLM_MODEL`. The retriage limit is
a development guard and is unlimited by default.

## CLI

The canonical issue interchange format is JSONL: one validated issue object per
line. `ingest` accepts `odc-json`, `semgrep-json`, or canonical `jsonl` input
(`--format auto` detects `.jsonl` and `.ndjson`; use an explicit format for
scanner reports) and writes canonical JSONL. `triage` consumes canonical JSONL
and writes its JSON group result. `run` consumes canonical JSONL and writes a
typed result JSON, with an optional reviewable unified diff and Markdown
report.

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
for each task. QA results remain keyed to their task, and Supervisor requires
the authoritative final full scan before teardown.

Phase 2 tactical supervision is single-task and QA-aware. It uses a stable
system prompt plus a bounded, deterministically ordered context so task facts
do not invalidate the reusable prompt prefix. The model returns only a typed
diagnostic basis and strategy proposal; Python verifies the normalized
security floor, candidate whitelist, task identity, dependency type, and
relative file hints, then renders and digests the exact worker instruction.
Breaking-change evidence can pivot directly to a workaround. Peer conflicts
may be recorded as a Phase 2 portfolio referral, while actual clustering and
multi-package orchestration remain deferred to Phase 3. Without an API key,
or when the model or registry verification fails, the deterministic router
remains authoritative.

Workaround retries restore only the child source-edit snapshot when QA rejects
an intermediate hypothesis, preserving the resolved dependency candidate.
The original parent rollback anchor is consumed only after workaround success
or terminal abandonment. A successful `validate_workaround` gate terminates
the bounded worker loop after all tool responses in that model turn are
recorded, so a later max-round guard cannot turn a passing workaround into a
surrender.

The CLI exits with `0` for a completed run without errors, `1` for a completed
run with remediation errors or unfixable tasks, and `2` for invalid input or
missing prerequisites. `--output` and `--patch-out` write files; without an
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
