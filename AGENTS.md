# AGENTS.md

## Purpose

`remediation_engine` is a Python AppSec remediation service. It ingests Dependency-Check and Semgrep findings, enriches and groups them, then runs the Phase 5 task-queue workflow to produce a proposed remediation patch.

## Repository layout

- `src/remediation_engine/`: installable package and public API.
- `tests/`: unit and integration tests; tests must not require live Docker, LLM, or network services.
- `examples/juice_shop/`: the single maintained end-to-end example and small fixtures.
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

- `task_queue` and committed attempt snapshots are the authoritative orchestration state.
- The Supervisor is the only component allowed to select dependency versions, plan retries, pivot strategy, or spawn tasks.
- Update and workaround workers execute committed instructions; they do not choose versions or query registries.
- QA performs deterministic install/scan/test execution, then maps evidence to tasks and returns typed evaluations.
- Every worker result must carry the current attempt ID and task revision. Stale results are ignored.
- Docker volumes are temporary and must be removed on every teardown path.
- Host repositories are never modified by the engine. The public API and CLI emit a typed result and unified patch.

## Configuration and secrets

Environment variable names are part of the operational interface. Preserve existing names such as `OPENAI_API_KEY`, `REMEDY_LLM_MODEL`, `ODC_EXTRA_ARGS`, `TRIAGE_LLM_ENABLED`, `LANGSMITH_*`, and `REMEDIATION_TRAJECTORY_DIR`. Parse them at application boundaries with `AppSettings`; new business logic should receive explicit settings instead of introducing more direct environment reads. Never commit `.env`, API keys, clones, generated reports, or trajectories.

## Testing rules

Unit tests mock Docker, HTTP, LangChain, and subprocess boundaries. Live Docker/LLM runs belong only in the Juice Shop example or explicitly marked integration tests. Keep tests deterministic and assert typed contracts, state transitions, cleanup, path validation, and patch output.

## Code quality

Use Google-style docstrings for public APIs and non-trivial helpers. Docstrings must describe the actual arguments, return values, exceptions, mutations, and external side effects. Prefer small pure transition functions, explicit dependency injection, `pathlib.Path`, typed models, and structured logging. Do not reintroduce Phase 4 compatibility shims or generic placeholder modules.
