# AGENTS.md

## Purpose

`remediation_engine` is a Python AppSec remediation service. It ingests Dependency-Check and Semgrep findings, enriches and groups them, then runs the Phase 5 task-queue workflow to produce a proposed remediation patch.

## Repository layout

- `src/remediation_engine/`: installable package and public API.
- `tests/`: unit and integration tests; tests must not require live Docker, LLM, or network services.
- `examples/juice_shop/`: the primary maintained end-to-end example and fixtures.
- `examples/NodeGoat/`: a secondary, tested example and fixture set.
- `data/`: ignored runtime state (clones, caches, trajectories, and reports).
- `docs/`: architecture and operational documentation.

## Development commands

```text
python -m pip install -e ".[dev]"
python -m pytest
ruff check .
ruff format --check .
remedy --help
```

Use the repository-local virtual environment where one exists. Tests should use a workspace-local temporary directory so the suite works in restricted environments.

## Architecture invariants

- Current graph flow: `initial_triage` -> `workspace_builder` ->
  `workspace_setup_scan` -> `supervisor` -> committed workers/QA and optional
  post-QA reconciliation -> `teardown` -> `report`. No-work or setup failure
  may route directly to teardown.
- `task_queue` and committed attempt snapshots are the authoritative orchestration state.
- The Supervisor owns dependency-version policy and the authorized candidate
  set, retry planning, strategy pivots, and task spawning. A worker may execute
  an alternate only when the committed attempt explicitly authorizes it; it
  cannot expand the candidate set or query registries.
- Portfolio planning and resolver certification are functions invoked by the
  Supervisor; triage/advisory suggestions are not worker authorization.
- Update and workaround workers execute committed instructions and any
  explicit candidate-selection rule; they do not independently authorize
  versions or query registries.
- Multi-package actions are committed as atomic batches with per-attempt
  provenance. Failed-batch isolation uses bounded canary checks and may assign
  blame only after the candidate has been verified restored.
- QA performs deterministic install/scan/test execution, maps occurrence evidence to tasks, and returns typed evaluations. Each evaluator receives its task-scoped scan projection; repository-wide cumulative findings feed triage and are not attributed to unrelated task verdicts.
- The mutable `RemediationTask.selected_version` is a planning hint. The
  committed attempt snapshot, exact atomic mutation target, and sealed worker
  instruction define what execution is authorized; an explicitly committed
  candidate rule may authorize a bounded alternative. Workers cannot broaden
  that authorization or query registries. QA checks current attempt evidence
  for the effective version actually executed.
- Every worker result must carry the current attempt ID and task revision. Stale results are ignored.
- Incremental dependency-input scan coverage is the default terminal evidence;
  `REMEDY_FINAL_FULL_SCAN_ENABLED` opts into a full-repository final scan.
- Workaround child attempts preserve the cumulative dependency candidate;
  parent rollback anchors are consumed only after success or terminal abandonment.
- Docker volumes are temporary and must be removed on every teardown path.
- Host repositories are never modified by the engine. The public API and CLI emit a typed result and unified patch.

## Configuration and secrets

Environment variable names are part of the operational interface. Preserve existing names such as `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`, `ODC_EXTRA_ARGS`, `TRIAGE_LLM_ENABLED`, `LANGSMITH_*`, and `REMEDIATION_TRAJECTORY_DIR`. Parse them at application boundaries with `AppSettings`; new business logic should receive explicit settings instead of introducing more direct environment reads. Never commit `.env`, API keys, clones, generated reports, or trajectories.

## Testing rules

Unit tests mock Docker, HTTP, LangChain, and subprocess boundaries. Live Docker/LLM runs belong in the Juice Shop or NodeGoat examples, or explicitly marked integration tests. Keep tests deterministic and assert typed contracts, state transitions, cleanup, path validation, and patch output.

## Code quality

Use Google-style docstrings for public APIs and non-trivial helpers. Docstrings must describe the actual arguments, return values, exceptions, mutations, and external side effects. Prefer small pure transition functions, explicit dependency injection, `pathlib.Path`, typed models, and structured logging. Do not reintroduce Phase 4 compatibility shims or generic placeholder modules.
