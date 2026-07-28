# Remediation Engine

`remediation-engine` is an agentic AppSec workflow that ingests Dependency-Check
or Semgrep findings, triages them, and produces a proposed remediation patch.
The host repository is never edited by the engine.

## Install

```text
python -m pip install -e ".[dev]"
```

Docker is required for workspace isolation and QA. Set `OPENAI_API_KEY` before
running an LLM-backed remediation. Copy `.env.example` to `.env` and adjust
the scanner, model, tracing, and cache settings as needed.

## CLI

```text
remedy ingest data/sample_odc_report.json --format odc-json --output findings.jsonl
remedy triage findings.jsonl --repo data/clones/juice-shop --output groups.json
remedy run findings.jsonl --format jsonl --repo data/clones/juice-shop \
  --output remediation.json --patch-out remediation.patch
```

`ingest` normalizes scanner output into canonical JSONL (one issue object per
line; legacy JSON-array exports are accepted as input). `triage` produces
actionable groups.
`run` performs triage when groups are not supplied, executes the Phase 5 task
queue, and writes a typed result plus unified patch. Exit code 1 means the run
completed with remediation errors or unfixable tasks; exit code 2 means invalid
input or missing prerequisites.

## Python API

```python
from pathlib import Path
from remediation_engine import RemediationRequest, run_remediation

result = run_remediation(RemediationRequest(repo_root=Path("target"), issues=[]))
print(result.status, result.diff)
```

Use `remediation_engine.triage_issues` to create groups explicitly. Internal
LangGraph state, workers, and Docker clients are not public API.

## Development

```text
python -m pytest
ruff check .
ruff format --check .
```

The maintained end-to-end workflow is documented in `examples/juice_shop`.
Unit tests mock Docker, LLM, registry, and subprocess boundaries. Runtime
clones, caches, trajectories, reports, and credentials are ignored by Git.
