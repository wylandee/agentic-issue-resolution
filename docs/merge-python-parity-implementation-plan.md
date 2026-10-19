# Authoritative architecture merge and Python parity implementation plan

## Executive summary

Resolve the merge using `feat/redesign-agent-architecture` as the authority for agent behavior. Preserve Python-specific packaging, runtime, manifest, parser, and command semantics from `feat/extend-to-python`. Consolidate shared functions, prompts, tools, and contracts so both languages execute the same Supervisor-owned workflow.

The primary integration gap is the newly authoritative npm portfolio/certification path: Python support cannot be restored merely by retaining Python worker branches. It must participate in the same planning, certification, committed-attempt, atomic execution, QA, retry, delta-isolation, and teardown contracts through ecosystem adapters. Unsupported packaging features must return explicit typed evidence, never fabricated npm evidence or an alternate, less guarded workflow.

This plan began as a read-only audit and implementation handoff. The user explicitly authorized implementation immediately, without another approval step. The implementation and all three requested live fixture acceptance checks have now been completed; the results and remaining scanner/legacy-suite findings are recorded below.

## Implementation checkpoint — 2026-10-16 (resumed)

- `git diff --name-only --diff-filter=U` is empty: all merge conflicts are resolved. Existing staged merge work, unrelated local changes, and `outside.txt` remain untouched; no merge commit, reset, or cleanup was performed.
- Python worker, Supervisor, portfolio, manifest transaction/rollback, ODC/targeted QA, fix-planning, and stale-plan checks use the authoritative orchestration contracts while retaining PEP 440/PyPI commands and language-scoped tools. Shared npm evidence is scoped to the task's exact package/version closure.
- ODC coverage now recognizes package-suffixed paths such as `requirements.txt:certifi/2024.7.4`. It still reports missing manifests and rejects arbitrary colon suffixes without a package/version path. Regression coverage is in `tests/test_targeted_qa_scans.py`.
- Focused final verification: 336 tests passed and 1 Docker test skipped across Python portfolio/Supervisor parity, language parity, portfolio certification, SCA locator, and QA/ODC suites. Targeted Ruff passed for the changed Python parity and QA files; `compileall`, CLI help, and `git diff --check` passed.
- Full suite `.venv/bin/python -m pytest -p no:rerunfailures --basetemp=.pytest-tmp/merge-python-parity-resume --tb=short -q`: 79 failed, 1,990 passed, 92 skipped. Failures are concentrated in legacy routing/dispatch expectations and Docker-dependent tests in the restricted test environment; the failing tests do not include the focused parity suite. The rerun plugin was disabled because its localhost socket setup is denied.
- Repository-wide Ruff is not clean: `ruff check .` reports 41 issues and `ruff format --check .` reports 23 files needing formatting. Targeted Ruff passes for the modified parity/QA files. No broad autoformat was applied because it would reformat large merge-owned files.
- Shared-name fixture: `verify_shared_name_fixture` passed with 9/9 Angular tasks QA-passed, one atomic dispatch batch, all accepted scan evidence verified, and all nine package occurrences present in the patch. The overall result is `completed_with_errors` because terminal incremental scans found 41 new identifiers and the solver recorded infeasible secondary objectives after finding full coverage.
- Four-package fixture: `verify_four_package_fixture` passed. The initial base64url/jws pair failed QA and shared an atomic batch; restoration was verified and only jws was blamed. base64url, got, and esbuild have accepted QA-passed lineages and updated occurrences; jws remains inconclusive. The overall result is `completed_with_errors` with terminal incremental scan findings (65 identifiers) and expected jws retry exhaustion.
- PyGoat fixture selected certifi, patched `2022.12.7` to `2024.7.4`, and recorded `eval_status=all_passed`. The typed result is `completed_with_errors` because its terminal incremental scan reported 19 new identifiers; this is separate from the task-scoped QA verdict. The earlier coverage failure from ODC's package-suffixed manifest path is fixed and covered by regression tests.
- Both Juice Shop runs restored the clone's `suppressions.xml` to its saved SHA256 `2d5880ad293a8518631d5a1acbae8cb85ae9398e2a11905dd7b3192d45cf1810`; all three runs logged Docker workspace-volume teardown. The last observed five-hour usage was 18%; at 98%, stop all active work and commands, preserve this checkpoint, and wait for a new user instruction.

## Review scope and baseline

- Workspace: `/Users/wylandee/.omp/wt/wt-20260917-123855-a8f1544`.
- Current branch: `feat/extend-to-python`; HEAD `49210b35361497338af53aa4f69a16212f2726a9`.
- Active MERGE_HEAD and authoritative branch tip: `d2f040c0085a59273728aa0bb64cd1e2199b7698`.
- The initial worktree had 36 unmerged paths and many already staged merge changes; all conflict paths are now resolved. Preserve the existing merge and unrelated edits, including skill and documentation changes. Do not reset, abort, globally choose ours/theirs, or replace the checkout.
- Scope: all language seams in `src/remediation_engine`, conflicted source/tests, Python additions since the merge base, authoritative portfolio/solver/batch additions, and the requested Juice Shop/PyGoat examples. Related reporting, settings, runtime, ingestion, and eval boundaries are included for regression review.
- Exclusions: secret contents, ignored clone contents except prerequisites and explicitly requested live execution, unrelated UI/eval refactoring, speculative performance work, and new language implementations.
- Python >=3.11 service using Pydantic, LangGraph/LangChain, Docker, CP-SAT, tree-sitter, requests, packaging, and tomlkit; target ecosystems npm and PyPI.
- `.venv/bin/python` exists and resolves `remediation_engine` to this checkout. `venv/` was not found; use `.venv/bin/python` for the first requested fixture as an interpreter-path substitution only.
- Juice Shop and PyGoat clone directories exist. Docker and provider access were used for the requested live runs; the acceptance outcomes are recorded in the checkpoint above.
- Before resolution, `supervisor_node.py` had mismatched syntax and the engine contained conflict markers, so no valid pre-merge test baseline existed. Do not present that pre-resolution state as a regression baseline.
- Read `AGENTS.md`, `README.md`, `pyproject.toml`, architecture documentation, relevant test definitions, current conflict hunks, and branch file inventories. Commands: `.venv/bin/python -m pytest`, `.venv/bin/ruff check .`, `.venv/bin/ruff format --check .`, `.venv/bin/remedy --help`, plus the three live fixtures below. Use workspace-local pytest temp paths.
- User budget rule: inspect the **300-minute/five-hour used percentage**, not weekly usage; at >=98% immediately interrupt implementation and all subagents, terminate active task-owned long-running commands safely, checkpoint progress, and pause until a new user instruction. Do not auto-resume on reset. Latest reading is recorded in the implementation checkpoint above.

Line references below describe the pre-resolution worktree; use symbols and branch revisions when line numbers move.

### Repository map and ownership

| Area | Entrypoints / owners | Boundary and test areas |
|---|---|---|
| API/CLI | `api.py`, `cli.py`, `__init__.py` | Typed request/result, ingestion, language resolution; `test_public_api.py`, `test_language_detection.py` |
| Intake | `tools/odc_parser.py`, `semgrep_parser.py`, `triage/pipeline.py`, `agent.py`, `grouper.py`, `enrichment.py`, `reachability.py` | Finding identity/localization, advisory information only; triage/parser/locator tests |
| Master graph | `orchestration/graph.py`, `state.py`, `graph_wrappers.py` | Task/attempt state, stale-result guards, graph routes; graph/integration/execution tests |
| Supervisor | `supervisor_node.py`, `supervisor_policy.py`, `supervisor_routing.py`, `supervisor_planner.py`, `supervisor_spawn.py`, `_supervisor_execution.py`, `tactical_supervisor.py` | Sole version/retry/pivot/authorization owner; deterministic/tactical/PyPI-parent tests |
| Portfolio | `supervisor_portfolio.py`, `portfolio_orchestrator.py`, `portfolio_solver.py`, `portfolio_certifier.py`, `solver/*`, `contracts/solver_models.py` | Immutable input graph, bounded candidates, certified assignment, atomic scheduling; portfolio/solver/conflict tests |
| Workers | `update_subagent.py`, `workaround_subagent.py`, `subagent_runtime.py`, `context_manager.py` | Sealed attempt execution, bounded tool loop, phase policy; worker/context/replay/lifecycle tests |
| Tools | `remedy_tools.py`, `tools_manifest*.py`, `tools_validation.py`, `tools_workspace.py`, `tools_edit.py`, `tools_web.py`, `_tool_support.py` | Capability assembly, transaction, validation, source edits, evidence; remedy/manifest/validation/web tests |
| QA | `_qa_runtime.py`, `qa_critic.py`, `qa_odc.py`, `qa_test_parsing.py`, `qa_policy_engine.py`, `qa_evaluator.py`, `qa_types.py` | Deterministic install/scan/test, occurrence mapping, task-scoped read-only evaluation; targeted/ODC/QA/policy tests |
| Workspace/output | `workspace_builder.py`, `runtime/*`, `teardown_node.py`, `final_full_scan.py`, `report_*.py`, `trajectory_exporter.py` | Temporary Docker volume, rollback, incremental/full evidence, unified patch and reports; lifecycle/report/scan tests |
| Language semantics | `language.py`, `tools/package_identity.py`, `python_manifest_locator.py`, `pypi_registry_tools.py`, `manifest_locator.py`, `npm_graph.py`, `lockfile_closure.py`, `code_map.py`, `code_locator.py`, `fix_planner.py`, `contracts/version_policy.py` | Detection, declarations, package/version identity, actual graph and syntax semantics |
| Examples | `examples/juice_shop/fixture_acceptance.py`, fixtures/shared_name, `run_batch.py`; `examples/pygoat/run_batch.py` | Live acceptance separate from mocked offline tests; `test_juice_shop_fixture_outcomes.py`, `test_pygoat_run_batch.py` |

## Current architecture and invariants

Preserve the authoritative flow:

```text
initial_triage -> workspace_builder -> workspace_setup_scan -> supervisor
                                                           <-> committed workers
                                                           <-> deterministic QA
                                                           <-> post-QA triage
                                                           -> optional final_full_scan
                                                           -> teardown -> report
```

Portfolio planning and certification are functions invoked by the Supervisor, not new graph authorities. `task_queue` and committed snapshots own execution state. Tactical and triage outputs are proposals until Python code validates and seals them. Every accepted result must match task ID, revision, attempt ID, authorization, instruction digest, and relevant batch provenance.

Do not change candidate policy, retry limits, fallback semantics, strategy pivots, batching, bounded canary isolation, restore-before-blame, QA attribution, or parent/child rollback ownership to make fixtures pass. The mutable selected-version field is a hint; current committed execution evidence defines the effective version. Workers never query registries or broaden candidates. Preserve terminal PASS handling and complete same-turn tool response recording.

QA runs deterministic commands once for a dispatch and supplies task-scoped scan projections to evaluators. Cumulative repository findings feed triage, not unrelated task verdicts. Incremental dependency-input coverage remains the default terminal evidence; full repository scanning retains its existing opt-in setting. Host target repositories remain unchanged; all mutations occur in temporary Docker volumes that teardown removes on every path.

## Findings inventory

| ID | Priority / confidence | Exact evidence | Observation and impact | Planned treatment / validation |
|---|---|---|---|---|
| MERGE-01 | P1 / observed | `git diff --name-only --diff-filter=U` lists 36 paths; `update_subagent.py:330`, `:481`; `graph.py`, `graph_wrappers.py`, `state.py` conflicts | New batch/provenance/graph behavior collides with older Python integration; choosing whole sides loses required behavior | Use stage 3 as agent-policy authority, port stage 2 language deltas deliberately; preserve both authoritative and Python regression suites |
| ARCH-01 | P1 / observed; runtime failure is an inference | `portfolio_solver.py:588` rejects managers other than empty/npm; `portfolio_solver.py:1792` prepares npm inputs; `supervisor_portfolio.py:278` enters common planning/certification | Incoming portfolio workflow has no Python equivalent; preserving old worker dispatch alone cannot establish common policy/certification | Adapter-driven graph/candidate/certification preparation feeding the same Supervisor/solver; deterministic Python end-to-end graph tests |
| TYPE-01 | P1 / observed | `contracts/schemas.py:2660` `PackageMutation.dependency_type` accepts npm labels only; `contracts/solver_models.py:458` requires Node/npm runtime fields; `SolverVersionCandidate:346` carries semver ordering | Python batch mutation and runtime evidence cannot be represented honestly through npm-only fields | Add backward-compatible typed ecosystem discrimination/variant evidence; prove npm payload round trips and PEP 440 ordering |
| TOOL-01 | P1 / observed | `remedy_tools.py:150-181` multi-package builder always creates npm tool; `tools_manifest.py:245` verifies npm lock mutations; `tools_manifest_python.py:508` is a separate singleton executor | Python does not share the newly authoritative atomic-action execution capability | Reuse common atomic orchestration with language-specific edit/sync/verification hooks; test Python union rollback, exact closed-over action, retained batch checkpoint |
| PROMPT-01 | P1 / observed | `update_subagent.py:481-585` competing system/multi-package prompt builders; `workaround_subagent.py:720-772` separate full Python prompt; `context_manager.py:125-134` names only JS validation/removal tools | Independent full prompt text can omit policy improvements, and phase instructions can ask Python workers to call absent tools | Shared role/phase policy with selected capability names and language fragments; assert all referenced tools are actually exposed and no wrong-language commands leak |
| ARCH-02 | P2 / observed | `language.py:19-49` config contains only image/markers/suffixes; `update_subagent.py:74`, `supervisor_spawn.py:52`, `fix_planner.py:130`, `_qa_runtime.py` independently detect PyPI | Language and package-manager decisions are scattered and may disagree | Keep canonical root detection and centralize issue ecosystem/capability selection; reject mismatched execution pairs; retain mixed-root/default behavior tests |
| LIFE-01 | P1 / observed difference, consequences require testing | `tools_manifest.py:50-62` conflict: Python branch adds Pipfile handling; incoming branch adds ancestor npm lock coverage; batch factory `:456-477` retains union checkpoint; singleton Python `tools_manifest_python.py:649-656` and npm `tools_manifest.py:776-790` both clear successful singleton checkpoints | A merge can lose ancestor locks or apply singleton checkpoint semantics to batches | Keep ancestor npm behavior; enumerate Python manager-owned files through adapter; preserve distinct singleton/batch acceptance lifecycles and outer rollback |
| QA-01 | P1 / observed | `_qa_runtime.py:3103` Python inspect closure and `:3344` resolver; `qa_odc.py:562` conflict; `qa_evaluator.py:529-572` common evaluator plus Python hint; `qa_test_parsing.py:1530` Python test branch | Incoming incremental identity/coverage/attribution logic and Python closure execution must be composed, not substituted | Shared coverage/evidence/projection/guardrails; Python-only closure and test parser adapter; verify overlap, markers, unresolved graphs, task isolation, effective version |
| PY-01 | P2 / observed limitation | `tools_manifest_python.py:144-150` rejects extras/markers/URLs/multiple specifiers; `_SUPPORTED_DECLARATIONS:43-53` excludes dependency groups; `language.py:63-83` root-only detection; `workspace_builder.py:276-378` supports bounded Python 3.11 setup | Detection, localization, writable declarations, and install support are not equivalent; silent broad support claims would be misleading | Publish precise capability matrix; preserve supported Python behavior and explicit unsupported outcomes; expand safe static support only with semantic tests |
| TEST-01 | P1 / observed | `examples/juice_shop/fixture_acceptance.py:729`, `:854` already prove shared-name and four-package outcomes; `tests/test_pygoat_run_batch.py:140` mocks engine | A clean pytest run cannot prove the user's live engine outcomes | Run existing acceptance verifiers on fresh bound evidence and add analogous certifi checks; retain raw command/result evidence and distinguish environment blocks |

### Review-dimension assessment

- Architecture/ownership: shared master state and Supervisor authority are strengths; ARCH-01/02 are the main integration risks.
- Organization/navigability: tool categories and focused QA modules already provide useful ownership; deepen the ecosystem boundary, avoid broad file reshuffles.
- API/type/data: immutable actions and typed results are strengths; TYPE-01 needs narrow evolution with compatibility tests.
- Correctness/resources: committed envelopes, outer anchors, and bounded isolation are core protections; MERGE-01/LIFE-01/QA-01 require preservation.
- Complexity/clarity: duplicate whole prompts and parallel transaction policy are material drift risks; do not combine genuinely different parsers/resolvers.
- Performance/I/O/concurrency: static evidence is insufficient to assert a performance bug. Preserve existing budgets, cache boundaries, solver limits, batched reads, and per-dispatch QA execution. Measure command counts with fakes where extraction could duplicate work.
- Testability: broad deterministic suites exist; add behavioral parity tests at adapter boundaries and use real fixtures for acceptance.
- Security/operability: retain path guards, no worker registry authority, no test weakening, no host edits, typed unsupported evidence, settings injection, bounded logs, and cleanup. Eval/UI and report behavior need regression tests but no unrelated refactoring.

## Python packaging research and implications

These are legitimate language differences, not reasons to fork agent policy:

1. Distribution names normalize case and runs of `-`, `_`, `.`. Reuse `tools/package_identity.py:19` everywhere; keep npm scope semantics separate. [PyPA name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/).
2. Versions use PEP 440, including epochs, variable release lengths, post/dev/pre/local segments. Use `packaging.version.Version` and `SpecifierSet`, never npm semver parsing or lexical sorting. Keep the engine's existing stable-release/candidate-role policy; isolate how eligibility and order are computed. [PEP 440](https://peps.python.org/pep-0440/).
3. Requirement extras, markers, direct references, and environment-sensitive edges carry meaning. Preserve syntax when supported; reject unsupported edits explicitly. Resolve markers against the **target sandbox** environment and selected extras; do not evaluate them against the host. [Dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/).
4. `project.dependencies` and `project.optional-dependencies` use standardized requirement strings; dependency groups are a separate development dependency structure with inclusion semantics. Do not equate npm sections to these declarations or advertise dynamic metadata support. [pyproject specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/), [dependency groups](https://packaging.python.org/en/latest/specifications/dependency-groups/).
5. pip requirements files include options, nested `-r`, `-c`, URLs and hashes. Static localization and writable policy must distinguish each; included paths require containment/cycle handling. Do not execute `setup.py` on the host to discover metadata. [pip requirements format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).
6. pip can resolve with `install --dry-run --ignore-installed --report`; a report describes a resolution/install operation and is **not a lockfile**. Certification must cover the complete relevant assignment/environment, not just one candidate string. [pip install](https://pip.pypa.io/en/stable/cli/pip_install/), [installation report](https://pip.pypa.io/en/stable/reference/installation-report/).
7. `pip inspect` reports installed distributions, metadata, and target environment. Validate its schema version before using it; it is evidence after install, not proof that an unexecuted candidate resolves. Keep pip tooling distributions outside application occurrence attribution unless actually in the application closure. [pip inspect schema](https://pip.pypa.io/en/stable/reference/inspect-report/).

### Required supported-capability matrix

Implementation must document and test, per manifest/manager: detection, localization, update/removal, snapshot paths, resolver/certifier, installed evidence, targeted scanning, targeted/full tests, and unsupported outcome.

- Preserve current requirements variants, static PEP 621, static setup.cfg, and Pipfile/Pipfile.lock behavior.
- `setup.py` root detection/static metadata reading does not imply writable arbitrary Python expressions.
- For extras, markers, compound constraints, nested requirements, dependency groups, hashes, direct references, Poetry, uv, and newer lock formats: inventory actual support first. Support safely expressible forms through existing adapters when needed; otherwise explicitly mark unsupported without mutating. Do not silently route Poetry/uv through pip or rewrite hashes/markers away.
- Preserve Node-first mixed-root detection and unknown-root default unless an existing explicit language override selects another supported profile. Mixed ecosystem findings require explicit pairing validation; full polyglot scheduling is deferred.

## Target design

### One policy core and focused ecosystem adapters

Preserve existing graph nodes and policy symbols. Introduce a meaningful capability boundary in `language.py` plus a focused `ecosystems/` package only where it removes cross-module policy duplication. Suggested files: `ecosystems/contracts.py`, `ecosystems/nodejs.py`, `ecosystems/python.py`, `ecosystems/__init__.py`. Avoid a catch-all strategy class holding the entire engine.

Adapters own: identity/version/specifier semantics; declaration discovery/read/edit; affected package files; runtime/install/test commands; dependency graph projection; bounded registry metadata normalization; candidate resolver certification; installed graph/scan closure extraction; language prompt vocabulary and available capabilities. Shared engine owns authorization, candidate role selection policy, portfolio transitions, solver scheduling/objectives, provenance, attempts, retries, transaction sequencing, phase gates, QA policy/attribution, canaries, and reporting.

Representative use (illustrative interface, refine against actual callers):

```python
adapter = ecosystem_adapter_for(project_language, package_manager)
inputs = adapter.prepare_dependency_inputs(workspace, groups, settings)
plan = build_portfolio_plan_from_inputs(inputs, queue, settings)
certificate = adapter.certify_assignment(plan.selected_assignment, workspace, settings)
# Supervisor applies the same certificate/attempt/dispatch validation to both.
tools = build_update_toolbelt(..., capabilities=adapter.worker_capabilities)
```

Keep immutable typed values at boundaries; inject settings/network/sandbox collaborators. Adapters do not spawn tasks, select retries, seal instructions, query registries from workers, or modify host repositories. Capability registration must cover prompt rendering, exposure, terminal-success detection, scratchpad retention, phase filtering, and event-to-task matching from one consistent definition.

### Portfolio and solver integration

First isolate current npm graph/packument/range/certification logic without changing its results. Add Python graph inputs derived from declarations and actual sandbox resolver/inspect evidence. Feed the same immutable target/domain/coverage/edge/batch contracts and deterministic solver; use ecosystem-specific compatibility/ordering projections where needed. Python has one version per distribution per environment, marker/extras-conditioned edges, and no npm peer/override mechanics. Never synthesize peer constraints or npm overrides for Python.

Extend action/runtime/certificate evidence narrowly, maintaining old npm payload defaults and JSON shape when feasible. Use explicit Python runtime evidence rather than fake Node/npm fields. Candidate assignment/environment/declaration/resolution digests and task revisions must protect Python plans as strictly as npm plans. Do not weaken certification checks, dispatch uncertified Python plans, or bypass portfolio processing just to restore certifi.

### Shared transaction and prompts

Extract atomic batch sequencing, union checkpoint capture/restore, authorization validation, touched-file accounting, and acceptance ownership from `tools_manifest.py`; inject declaration mutation, sync, and resolved-version verification. Keep npm ancestor locks and exact existing command behavior intact. Python edits all approved declarations, resolves the entire assignment, verifies effective installed/lock evidence, and restores environment state where required. A failed restore blocks further mutation and blame.

Use one shared policy template per role/phase and small language fragments for capabilities, manifest vocabulary, commands, syntax, and evidence sources. Preserve stable system prefixes and bounded dynamic context. JS output must remain equivalent to the authoritative branch; Python must retain every applicable policy clause and only reference exposed tools. Do not use global text replacement that can corrupt examples or policy.

### Alternatives considered

- Whole-side conflict resolution: rejected; discards either current engine authority or Python support.
- Separate Python graph/Supervisor/worker policy: rejected; multiplies state machines and drifts from the requested design.
- Conditional branches at every call site: retain small existing branches during merge stabilization, then move meaningful language semantics behind cohesive adapters. Do not extract trivial wrappers solely for a uniform naming scheme.
- New universal packaging framework or all Python managers at once: deferred; expands scope and risks npm regression. Support current bounded capabilities honestly and make extension points explicit.

## Phased implementation plan

### Phase 0: Preserve evidence and define safety net

**Type:** safety/checkpoint work. **Dependency:** none. **Non-goals:** source refactoring or live runs before imports work.

1. Check five-hour usage before implementation. Save read-only `git status`, HEAD/MERGE_HEAD, staged/unstaged diffs, unmerged index stages, and untracked-file names into a task-local ignored/temp checkpoint; do not include secrets. Preserve recovery material for conflicted stage 1/2/3 blobs. Do not rely on `git stash` during an unresolved merge.
2. Build a conflict-resolution ledger with all 36 paths and the authoritative policy/Python delta for each. Compare `git diff feat/redesign-agent-architecture...HEAD` for Python additions and the merge-base-to-authoritative diff for policy changes.
3. Inventory **auto-merged** Python-modified files too: `_tool_support.py`, `context_manager.py`, `subagent_runtime.py`, `supervisor_policy.py`, `teardown_node.py`, tools edit/validation/web/workspace, runtime/sandbox, version policy, package identity, locators, ingestion, and triage. Conflict-free does not mean behaviorally correct.
4. Record existing public contracts, graph edges, default settings, committed envelope fields, candidate rules, batch/delta behavior, and fixture acceptance helpers. Preserve current tests before changing their expectations.

**Validation/acceptance:** recoverable baseline exists; authoritative SHA matches merge target; all invariants named above become review/test checklist. **Rollback:** restore only task-owned edits from the saved material, preserving pre-existing staged work. **Reviewer focus:** no accidental loss of user edits or secrets.

### Phase 1: Resolve merge to a coherent authoritative engine

**Type:** merge repair with local integration. **Dependency:** Phase 0. **Non-goals:** redesigning algorithms or broad cleanup.

Resolve in dependency order, using the incoming behavior as authority and adding bounded Python seams:

1. **Contracts/state:** `contracts/schemas.py`, `orchestration/state.py`; retain authoritative portfolio, multi-package, delta, scan, attempt, result, and policy models plus Python stages/language state.
2. **Intake/context:** `api.py`, `cli.py`, `tools/fix_planner.py`, `tools/code_map.py`; retain new solve/API behavior and source/path safeguards while restoring detection, PEP 440/PyPI localization, and Python AST parsing.
3. **Supervisor:** `supervisor_planner.py`, `supervisor_spawn.py`, `tactical_supervisor.py`, `task_utils.py`, `supervisor_node.py`; preserve incoming preparation, certification, exact target validation, batching, retries, pivots, and delta isolation. Restore Python candidates/parent planning through semantics adapters, not older copied policy functions.
4. **Tools/workers/workspace:** `tools_manifest.py`, `remedy_tools.py`, `update_subagent.py`, `workspace_builder.py`; retain batch closed-over actions/verification and ancestor locks, add Python-specific declarations/commands/image, and preserve common worker runtime.
5. **QA:** `_qa_runtime.py`, `qa_types.py`, `qa_test_parsing.py`, `qa_odc.py`, `qa_policy_engine.py`, `qa_evaluator.py`, `qa_critic.py`; retain occurrence evidence, current-attempt effective versions, structured deterministic logs, scoped projections, and canary restoration while adding Python install/test/closure evidence.
6. **Graph/wrappers:** `graph.py`, `graph_wrappers.py`; preserve authoritative nodes/routes/result guards and carry resolved language consistently through subagent state and sandbox reconstruction.
7. **Conflicted tests:** `test_deterministic_supervisor.py`, `test_execution_nodes.py`, `test_orchestrator_phase5_graph.py`, `test_public_api.py`, `test_qa_odc.py`, `test_remedy_agent.py`, `test_remedy_tools.py`, `test_supervisor_node.py`, `test_targeted_odc_docker.py`, `test_targeted_qa_scans.py`, `test_triage_reachability.py`, `test_workspace_builder.py`. Preserve behavior assertions from both sides; adapt obsolete mocks/signatures without deleting protections.
8. Stage only deliberately resolved paths to clear unmerged entries. Do not create a merge commit or push unless later requested.

**Tests:** syntax/import smoke, contract/public API/language/graph tests, then authoritative Supervisor/portfolio/solver/worker/QA suites. No live services in unit tests. **Acceptance:** no conflict markers/unmerged paths; graph topology and JS defaults match incoming branch; Python paths import without bypassing authority. **Checkpoint:** resolved merge diff and resolution ledger. **Risks:** swallowed incoming changes in auto-merges and tests that accidentally encode older policy.

### Phase 2: Consolidate capabilities and prompts

**Type:** cohesion refactor. **Dependency:** coherent Phase 1. **Non-goals:** new graph nodes or policy changes.

1. Centralize ecosystem identity/capability resolution in `language.py`, `tools/package_identity.py`, and focused adapter contracts. Keep public language aliases/defaults stable.
2. Consolidate language-independent authorization, candidate selection roles, result handling, state transitions, and bounded worker loops. Inspect `contracts/version_policy.py`, `supervisor_policy.py`, `subagent_runtime.py`, `supervisor_spawn.py`, `tactical_supervisor.py` against incoming symbols. Keep different comparison/specifier semantics behind adapters.
3. Refactor `_build_update_system_prompt`, `_build_multi_package_update_prompt`, `_build_workaround_system_prompt`, QA evaluator language rendering, tactical/triage language fragments, and `context_manager.py` phase prompts into shared policy plus explicit capabilities. Update `tools_web.py` evidence-source vocabulary and `tools_validation.py:1346` record-plan guidance so Python receives relevant authoritative sources.
4. Align tool names/capabilities across `remedy_tools.py`, `context_manager.py`, event matching, `_workaround_attempt_succeeded`, runtime terminal-tool handling, and scratchpad critical-event retention. Keep language-neutral tools implemented once.
5. Preserve `code_map.py`/locator/tree-sitter shared traversal and language-specific parser/import details; avoid copying repository-map/path/AST/edit tools for Python.

**Tests:** parameterized role/phase tool exposure and prompt contract tests; npm prompt/default behavior, Python-only tool names/commands, unavailable-tool rejection, investigate/edit/validate boundaries, bounded context and terminal PASS. Keep existing Python web/syntax/AST tests. **Acceptance:** one owner per policy concept, no wrong-language tool exposure, shared prompt policy clauses invariant. **Rollback:** per-boundary extraction checkpoints; keep existing public entrypoints if needed for compatibility, no Phase 4 shims. **Risks:** dynamic context leaking into stable prefix or stale hard-coded tool names.

### Phase 3: Complete shared portfolio, transaction, and QA integration

**Type:** structural adaptation and required Python feature completion. **Dependency:** Phases 1-2. This can be divided by boundary, but implementation initially remains with the requested Luna agent; no unsolicited additional agents.

1. `contracts/schemas.py`, `contracts/solver_models.py`: introduce narrowly compatible ecosystem-aware action/runtime evidence. Keep frozen records, strict validation, path checks, limits, task revisions, digests, and old npm defaults. Avoid accepting arbitrary declaration strings globally.
2. `portfolio_solver.py`, `portfolio_orchestrator.py`, `portfolio_certifier.py`, `supervisor_portfolio.py`, ecosystem adapters: isolate npm graph/range/registry/resolver operations; add Python equivalents feeding identical plan/certificate acceptance. Preserve solver objectives, DAG/SCC scheduling, bounded catalog and replans. Review `solver/cpsat.py`, `graph.py`, `subgraph.py` consumers for semver/peer assumptions; change only language-sensitive projections/ordering.
3. Reuse `pypi_registry_tools.py`, `python_manifest_locator.py`, `contracts/version_policy.py` for Python names, candidate eligibility, secure floors, stable-version selection, declared parent proof, and unsupported states. Keep registry fetches in Supervisor-owned preparation. Enforce target interpreter/platform compatibility and trustworthy metadata; unavailable metadata cannot authorize a candidate.
4. `tools_manifest.py`, `tools_manifest_python.py`, `remedy_tools.py`, `update_subagent.py`: create common atomic batch transaction sequencing and Python mutation/sync/verification capability. Apply the exact sealed action once, preserve union snapshots through result acceptance, restore files/environment on failure, retain failed-restore anchors, and record actual executed targets.
5. `_qa_runtime.py`, `qa_odc.py`, `qa_test_parsing.py`, `qa_policy_engine.py`, `qa_critic.py`, `qa_evaluator.py`: reuse common execution/coverage/occurrence/attribution logic, injecting Python install, inspect/report parsing, syntax/tests and targeted scanner inputs. Keep evaluation scope distinct from cumulative triage snapshot. Do not treat unresolved closure or missing evidence as a pass.
6. `workspace_builder.py`, `runtime/sandbox_mgr.py`, `teardown_node.py`, `final_full_scan.py`: retain Python image/venv/native dependencies, Node workspace inventory and install semantics, all cleanup paths, host immutability, candidate-preserving rollback, and typed patch/report output. Confirm partial failure removes only rejected target changes in Python manifests too.

**Tests:** Python singleton and atomic batch through the shared graph; failed batch isolation restores before blame; stale/revised attempts; mismatched ecosystem; exact action/version/manifest protection; report-schema validation; target environment markers/extras and PEP 440 edge cases; resolver unavailable; unsupported declaration leaves files unchanged; Pipfile lock/sync restore; no npm commands in Python sandboxes; no worker network authorization. Existing JS shared-name/delta, portfolio certificate and targeted scan tests must continue passing.

**Acceptance:** both languages use the same graph, authority checks, policy and transaction/QA contracts. Certifi can receive a real certified current attempt; unsupported cases are explicit, not silently routed around the shared engine. **Migration:** additive/variant model evolution, existing API/CLI/env names unchanged; no durable fixture format changes without necessity and round-trip tests. **Rollback:** retain boundary checkpoints and old serialized npm examples; revert the adapter integration rather than weakening gates. **Risks:** mistakenly treating pip report as lockfile, using host marker context, certifying only a singleton outside the full assignment, and pretending npm peers apply to Python.

### Phase 4: Consolidation and acceptance

**Type:** removal of duplicated internals, docs and validation. **Dependency:** Phase 3. **Non-goals:** unrelated refactors/tooling upgrades.

1. Remove obsolete duplicate internal policy/prompts after callers use shared code. Retain externally used APIs; do not add generic placeholder modules or Phase 4 compatibility layers.
2. Update `docs/architecture.md`, README, PyGoat/fixture documentation with supported capability matrix and actual boundaries. Preserve existing staged documentation changes.
3. Run full deterministic suite and lint/format/CLI smoke below. Fix regressions without relaxing authority or fixture expectations.
4. Run all three requested live fixtures sequentially against unchanged host clones; use fresh evidence bound to each result/patch. Record command exit, typed outcome, paths, verifier result and any narrowly recognized PasteBin failure.
5. Review final diff against incoming authority and original Python deltas. Report every remaining language-independent difference and justify or remove it. Confirm no secrets, runtime artifacts, host edits, unmerged files, or leaked temporary volumes.

**Acceptance/checkpoint:** all Definition of Done conditions satisfied or specific external blockers honestly reported with attempted commands; never claim live success from mocks. **Risks:** nondeterministic provider/package/scanner failures; do not rerun indefinitely or tune policy to fit one trace.

## Verification strategy

### Deterministic verification

Use the repository local environment and local temp root. Start narrow, then run the suite once after the integrated changes; repeat only for failures or new changes.

```sh
.venv/bin/python -m compileall -q src
.venv/bin/python -m pytest --basetemp=.pytest-tmp/merge-python-parity
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/remedy --help
git -c core.fsmonitor=false diff --check
git -c core.fsmonitor=false diff --name-only --diff-filter=U
```

Check pytest configuration/markers before the full run and keep live Docker/HTTP/LLM boundaries mocked as `AGENTS.md` requires; report precisely any deliberately excluded live/eval markers. Preserve authoritative regression tests for candidate whitelists, sealed instruction/digest, revision freshness, portfolio certificates, atomic batches, restore-before-blame, workaround candidate preservation, scoped QA, incremental/full scan, and cleanup. Use parameterized metamorphic tests to compare equivalent JS/Python policy transitions while allowing different package evidence and commands. Do not add tests merely mirroring helpers.

### Exact requested fixtures and assertions

1. Shared-name:

```sh
.venv/bin/python examples/juice_shop/fixtures/shared_name/run_shared_name.py --repo data/clones/juice-shop
```

Expected: three initial target packages expand to nine Angular packages; all nine are remediated in one atomic batch. Use `verify_shared_name_fixture` and assert task/attempt/batch provenance, fresh bound result and patch, effective package versions, relevant scan coverage, and no rejected changes in the patch. The only substitution is `.venv` for the absent `venv` interpreter path.

2. Four-package/delta isolation:

```sh
.venv/bin/python examples/juice_shop/run_batch.py --iterations 1 --packages base64url got esbuild jws --repo data/clones/juice-shop
```

Expected: initial base64url+jws atomic batch fails QA; bounded delta isolation restores the candidate and blames only jws; got and esbuild run in singleton batches; base64url, got and esbuild remediate, jws remains terminally unresolved. Use `verify_four_package_fixture`, including lineage and restore confirmation, independent singleton batches, accepted patch occurrences and current scan evidence. A nonzero runner status because jws is unresolved may be expected; inspect typed outcome/verifier rather than classifying solely by exit code.

3. Python:

```sh
.venv/bin/python examples/pygoat/run_batch.py --iterations 1 --batch-size 1 --seed 2
```

Expected: selection is certifi; update succeeds and QA passes. Verify actual sampled input rather than hard-coding a substituted target, current authorized effective certifi version, installed/declaration/lock evidence as applicable, scan and test gates, typed final result and nonempty valid patch. Preserve seeded sampling; do not edit baseline findings or force a success status. Add a fresh-evidence verifier only if existing output lacks sufficient assertions.

For each live run: confirm prerequisites without printing credentials, verify host clone status before/after, keep outputs in ignored `data/` or task temp paths, and confirm task-owned volumes are removed. If Docker/network access is restricted, request necessary tool escalation with the user's existing fixture authorization. Report an unavailable service accurately after reasonable attempts; do not install unrelated tools or edit global Docker state.

**PasteBin exception:** recognize and record only the requested PasteBin-related fixture unit-test/network failure as non-blocking expected behavior. Do not suppress unrelated failures, globally ignore unit test errors, or label a whole QA failure as passing. If the fixture outcome is obscured by that expected external failure, state the exact evidence and what remained verifiable.

### Performance and operational validation

No performance improvement is claimed. Preserve solver bounds/timeouts and command counts; assert QA installs/scans/tests are not repeated per evaluator, and cache keys include ecosystem/runtime where necessary. Compare trace fields, task lineage, evidence coverage, retries, and cleanup against existing contracts. No new direct environment reads: settings remain parsed at application boundaries and passed explicitly.

## Compatibility and rollout

Work in the active merge checkout; no new checkout, PR, push, deployment, or merge commit is required. Preserve all existing environment names, API/CLI behavior, language overrides, report/result fields, default npm behavior, final-full-scan flag, and Docker lifecycle. Introduce additive typed evidence or discriminated variants only where true language semantics require it; retain npm fixture deserialization and deterministic digests for unchanged npm inputs where possible.

No feature flag should route Python around authorization/certification. No deprecation window is needed for private helper extraction; public signatures need compatible defaults or a deliberate migration covered in tests. Roll back at extraction boundaries if npm behavior differs, shared invariants weaken, or Python reports unsupported evidence as success.

## Deferred work and evidence gaps

- This is a targeted architectural/language audit, not an exhaustive line-by-line review of every solver/eval/report module. Pure solver internals, delta-canary execution details, all teardown package filtering branches, reporting/eval persistence, every Python parser form, and every auto-merged hunk require implementation-time inspection and regression execution. Their ownership/test areas are mapped above; no claim of verified runtime correctness is made yet.
- There was no pre-merge test baseline because conflicts prevent compilation. The immutable incoming revision and its tests are the behavior reference; do not invent historical passing results.
- Docker readiness, cached image behavior, and provider access were exercised by the fixtures. The full test suite and repository-wide Ruff checks remain non-clean as recorded above; the three fixture verifiers pass, with each run's terminal incremental-scan findings preserved rather than suppressed.
- Full polyglot scheduling, new Java support, automatic interpreter matrix expansion, arbitrary dynamic setup.py rewriting, complete Poetry/uv/lock-format support, package-index migration, and solver algorithm changes are deferred. Existing supported Python packaging behavior must be preserved; additional unsupported constructs must be described honestly.
- The precise adapter API and smallest compatible runtime/action model change must be finalized after following every actual caller. The design does not authorize weaker contracts or a parallel Python engine if extraction is difficult.

## Definition of done

- All 36 conflicts resolved and staged intentionally; no conflict markers or syntax/import errors; unrelated staged work preserved.
- Resolution ledger accounts for authoritative policy and retained Python deltas, including auto-merges.
- Both languages use the authoritative graph, Supervisor, sealed attempts, retry/pivot logic, portfolio acceptance, worker loop, QA policy, and rollback ownership.
- Language-independent code/prompts/tools have one implementation; language-specific capabilities and evidence appear only for the detected/explicitly selected supported ecosystem.
- Current Python declaration/runtime behavior survives, npm ancestor lock handling survives, batch execution is atomic, stale results are ignored, and canary blame requires verified restoration.
- Focused deterministic tests, targeted Ruff, compileall, CLI help, and diff checks pass; full-suite and repository-wide Ruff failures are reported with their exact commands and counts.
- All three fixtures were attempted and their requested task/batch behaviors were verified from fresh evidence. Their nonzero overall statuses and incremental-scan findings remain visible; no PasteBin exception was needed.
- Host clones unchanged; temporary volumes removed; public contracts/settings preserved; no secrets or ignored runtime artifacts added to source control.
- Implementation and final report identify remaining evidence gaps. At five-hour usage >=98%, every agent and task-owned long-running process is stopped/paused until a new user instruction.
