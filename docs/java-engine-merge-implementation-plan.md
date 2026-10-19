# Java engine merge and reuse implementation plan

## Executive Summary

Preserve the engine behavior of `feat/redesign-agent-architecture` and integrate Java through language-specific operations inside the same orchestration. Resolve the in-progress merge first; then consolidate language selection, prompts, tool composition, dependency operations, QA evidence, and cleanup. Shared policy, state transitions, authorization, retry planning, provenance, and result handling must have one implementation.

Highest priority: retain portfolio certification and atomic batches, delta isolation after verified restoration, immutable worker authorization, task-scoped QA evidence, incremental scan coverage, and emergency cleanup. A Java-only alternate orchestration would violate the requested design.

Source files were not modified during this review. This plan is the review deliverable; implementation is authorized immediately afterward with `gpt-6-luna` at `max` reasoning.

## Review Scope and Baseline

- Current branch: `feat/extend-to-java`; HEAD `d1746fd9794691ca368ae7bf3da3ae304af494f2`.
- Authoritative merge parent and redesign branch: `d2f040c0085a59273728aa0bb64cd1e2199b7698`.
- The index already contains a large merge and 26 unmerged paths. Preserve existing user changes and merge state. Do not abort, reset, replace the entire worktree, or silently commit the merge.
- Review covered the Java branch delta, conflict inventory, API/CLI, contracts, language configuration, graph/state, Supervisor/planning, portfolio solver/certifier, worker prompts/toolbelts/runtime, QA/scanning, workspace/teardown, triage/localization, Java tools, corresponding tests, and fixture acceptance helpers. Ignored runtime reports, credentials, and generated baseline HTML/JSON were excluded.
- Python service; supported target ecosystems currently JS/TS with npm and Java with Maven. This work does not add Gradle, Kotlin, Python targets, or general polyglot orchestration.
- Repository-local interpreter: `.venv/bin/python`. The requested `venv/bin/python` does not exist. Both `data/clones/juice-shop` and `data/clones/vulnado` exist; live service availability has not been established.
- Read `AGENTS.md`, `README.md`, `docs/architecture.md`, and `pyproject.toml`. No CI directory was found. Ordinary tests must mock external boundaries and use workspace-local temporary directories.
- Five-hour usage was 7% at the final planning checkpoint. Check the 300-minute usage window throughout implementation; at 98% immediately stop tools, live runs, and agents and await the user's next instruction. Do not resume automatically after reset.

## Current Architecture

The public API and CLI resolve a repository and context into the graph. Initial triage localizes and groups scanner issues. Workspace setup copies the repository to run-owned Docker volumes, installs dependencies, and records baseline inputs. The Supervisor invokes portfolio planning and resolver certification, commits tasks and immutable attempts, dispatches update/workaround workers, and reconciles typed QA results. Optional reconciliation and final scanning precede teardown and report emission.

```text
API/CLI -> initial_triage -> workspace_builder -> workspace_setup_scan
                                                     |
                                                     v
             portfolio prepare/solve/certify <-> Supervisor
                                                     |
                       update/workaround -> QA ------+
                                                     |
                   reconciliation / optional final scan
                                                     |
                                               teardown -> report
```

`task_queue` and committed attempt snapshots own execution state. Workers may only execute committed packages, versions, manifest targets, operations, and bounded candidate rules. Registry access belongs to Supervisor functions. QA owns deterministic gates and task-scoped evidence. Docker and HTTP/subprocess operations are external boundaries; host repositories remain unchanged by the engine.

## Findings Inventory

Line references below describe the conflicted planning snapshot and will shift during resolution.

| ID | Priority | Confidence | Location | Problem and impact | Planned treatment and validation |
| --- | --- | --- | --- | --- | --- |
| MERGE-01 | P0 | Observed | `orchestration/supervisor_node.py:832`, `supervisor_planner.py:896`, `update_subagent.py:437`, `contracts/schemas.py:1776` | Older Java edits overlap redesigned authorization, batch execution, and evidence contracts. Choosing one side wholesale loses required behavior. | Treat redesign as semantic baseline and reapply only Java operations; retain all redesign tests and add Java contract equivalents. |
| API-01 | P1 | Observed | `api.py:135`, `api.py:190`, `state.py:643`, `graph.py:660` | Language resolution competes with runtime-settings scope, target packages, inventory defaults, and accepted-evidence reopening guards. | Compose both requirements; test context preservation, explicit settings, target scope, stale results, and guarded reopening. |
| PORT-01 | P1 | Observed | `supervisor_portfolio.py:234`, `portfolio_solver.py:235`, `portfolio_solver.py:588`, `portfolio_certifier.py:2374` | Shared portfolio entrypoints call npm preparation/certification; target building rejects non-npm managers. A marker-only merge cannot give Java the new flow. | Add ecosystem preparation/resolution behind the same Supervisor transaction and certification contract. Never mark an unverified plan dispatchable or skip Java certification to force success. |
| SOLVER-01 | P1 | Observed | `portfolio_solver.py:918`, `solver/cpsat.py:158`, `solver/cpsat.py:228`, `contracts/version_policy.py:201` | Candidate ordering and compatibility use npm semver/ranges. Maven already has a separate comparator that must remain authoritative for Maven versions. | Isolate ecosystem ordering/constraints before neutral solving; test Maven qualifiers, equivalence, floors, and npm ordering without coercing Maven into semver. |
| PROMPT-01 | P1 | Observed | `update_subagent.py:413`, `update_subagent.py:438` | Maven static instructions forbid every source edit; redesign allows migration required by the committed instruction. Separate full prompts can drift in policy. | Compose one shared authorization/policy prompt with selected ecosystem clauses. Preserve exact instruction handling and add parity plus cross-language exclusion tests. |
| TOOL-01 | P1 | Observed | `remedy_tools.py:75`, `remedy_tools.py:137`, `remedy_tools.py:255` | Singleton updates dispatch by language, but committed batch assembly closes over npm. Workaround composition shares tools whose internals remain language-specific. | Keep shared composition and authorization; inject selected mutation and validation operations. Test singleton, atomic batch, failure restoration, and unavailable capability handling. |
| QA-01 | P1 | Observed | `tools_validation.py:375`, `tools_validation.py:565`, `tools_validation.py:619` | Syntax, typecheck, lint, and runtime smoke assume JS/TS/Node. Java tool exposure alone does not supply equivalent validation. | Provide Maven compilation and scoped Java test/runtime operations through shared gate orchestration; preserve typed failure, blocked, and skipped semantics. |
| SCAN-01 | P1 | Observed | `_qa_runtime.py:2731`, `qa_critic.py:1410`, `contracts/schemas.py:1703`, `contracts/schemas.py:1776` | Conflicting scan return signatures and npm-specific setup inventory coexist with non-npm targeted staging. Dropped fields or repo-wide attribution would weaken QA. | Preserve redesign batch/evidence contracts; integrate Maven dependency inputs and exact GAV occurrence evidence. Test incomplete scans, unrelated findings, task projections, and optional full scan. |
| RESOURCE-01 | P1 | Observed | `workspace_builder.py:174`, `teardown_node.py:210` | Java requires a run-owned Maven cache; redesign introduces emergency cleanup and input inventories. A one-sided resolution loses lifecycle guarantees. | Share run-resource cleanup across ordinary, failed setup, and graph-exception paths; preserve owned-name validation and attempt removal of every volume. |
| LOC-01 | P2 | Observed | `language.py:92`, `triage/pipeline.py:274`, `triage/reachability.py:155`, `tools/code_map.py:77` | Detection is root-only with Node default; direct pipeline calls can default Node when language is absent. Maven reachability intentionally stays unknown; AST parsing currently covers JS/TS. | Resolve language consistently at boundaries and audit every caller. Preserve documented detection precedence. Provide safe Java functionality or typed unsupported capability; never label unknown reachability unreachable. |
| TEST-01 | P1 | Observed | `tests/test_java_language.py`, `tests/test_java_supervisor.py`, `tests/test_tools_manifest_java.py`, `examples/juice_shop/fixture_acceptance.py:729`, `:854` | Java tests mostly cover older seams; redesign fixture verifiers already assert task lineage, atomic membership, scan evidence, restoration, and patch effects. | Preserve both suites, parameterize shared contracts, and extend Java graph/portfolio/QA coverage. Verify live outcomes from correlated evidence, not process exit alone. |

No performance regression is asserted from static review. Adapter selection should occur at run/tool construction boundaries; measure live phase timings only if changes create a concrete concern. Existing bounded registry retries, command/path authorization, structured diagnostics, and test seams should be reused.

## Target Design

Use one engine with a small typed ecosystem boundary, selected from the detected or explicitly supplied language. Retain `ProjectLanguage`, `LanguageConfig`, API defaults, CLI options, and operational environment names. Extend their ownership coherently, optionally introducing `ecosystems/base.py`, `ecosystems/nodejs.py`, and `ecosystems/maven.py` if these hide meaningful complexity.

| Shared engine ownership | Selected ecosystem ownership |
| --- | --- |
| Graph, task queue, attempts, revisions, dispatch and stale-result rejection | Manifest discovery, POM/package declaration localization and exact identity |
| Supervisor policy, allowed candidates, retries, strategy pivots and spawning | Version comparison, stable-release semantics, registry and resolver operations |
| Portfolio transaction, plan lifecycle, neutral solving and certification provenance | Resolver graph/input construction and native dependency constraints |
| Atomic action authorization, checkpoints, rollback, delta isolation | Manifest bytes to mutate, native install/synchronization and restoration verification |
| Worker loop and shared prompt policies | Selected tool names, argument schemas, commands and ecosystem prompt clauses |
| QA gate orchestration, task verdicts, evidence projection and report lifecycle | Compilation/test commands, test result parsing, dependency occurrences and scan inputs |
| Path safety, resource ownership, telemetry and result serialization | Source suffixes, excluded build paths, Docker image and Maven cache mount |

Representative shape: resolve an ecosystem once at the boundary, pass its typed operations to shared toolbelt and QA builders, and let the Supervisor invoke the same prepare/solve/certify transaction. This is an internal design sketch, not a required public API rename. Do not create a generic mega-interface with unused methods or move the whole engine into separate language classes.

Prefer composition over copied Java agents and scattered `if java` policy forks. Keep existing Maven XML mutation/localization and registry logic where substantive; share retry, authorization, checkpoint, evidence, and prompt policy around them. Native Maven resolver behavior cannot be implemented by replacing npm command strings.

Do not introduce an alternate Java orchestration, bypass portfolio planning, fake npm lockfiles for Maven, allow workers registry access, broaden authorization, or convert unverified resolution into success. Native npm peer constraints and Angular sibling expansion remain npm-specific, while the portfolio/batch lifecycle remains shared.

## Phased Implementation Plan

### Phase 0: Safety net and invariants

1. Record HEAD, MERGE_HEAD, merge base, status, unmerged paths, index stages, and current diffs in ignored runtime storage or a temporary directory. Record clone state and fixture/suppression file hashes without reading secrets.
2. Inventory all Java-only differences from merge base to HEAD, including files already merged without conflict. Use immutable redesign SHA for comparisons.
3. Preserve authoritative contracts: committed candidate rules, atomic mutation targets, attempt IDs/revisions, sealed instructions, task-specific evidence, restored-candidate prerequisite for blame, parent/child workaround anchors, incremental scanning default, optional full scanning, and every cleanup route.
4. Baseline tests cannot collect while Python conflict markers remain. Record that constraint; do not treat collection errors as behavioral results.

Checkpoint: no tracked source changes before implementation; recoverable snapshot exists. Rollback means restoring only implementation edits, preserving the original merge and user changes. No merge abort or blanket reset.

### Phase 1: Resolve the merge against the redesign

Resolve all 26 paths, preserving redesign semantics and reintegrating language-specific arguments and operations:

- `examples/juice_shop/fixtures/suppressed/odc_suppressed_issues.jsonl`.
- `src/remediation_engine/api.py`, `contracts/schemas.py`.
- `orchestration/_qa_runtime.py`, `graph.py`, `qa_critic.py`, `qa_policy_engine.py`, `qa_test_parsing.py`, `report_node.py`, `state.py`, `supervisor_node.py`, `supervisor_planner.py`, `tactical_supervisor.py`, `teardown_node.py`, `update_subagent.py`, `workaround_subagent.py`, `workspace_builder.py`.
- `tools/fix_planner.py`, `tools/manifest_locator.py`, `triage/reachability.py`.
- `tests/evals/custom_metrics.py`, `tests/test_execution_nodes.py`, `tests/test_remedy_agent.py`, `tests/test_remedy_tools.py`, `tests/test_triage_reachability.py`, `tests/test_workspace_builder.py`.

Order: contracts/state/imports; API/language propagation; workspace/cleanup; Supervisor/planning; workers; QA/scans/policy; report/triage; tests and fixture data. Keep redesign evaluator metrics and golden behavior rather than restoring obsolete metrics wholesale. In API conflicts retain effective language context, runtime-settings scope, and target packages together. In reopening retain accepted-evidence guards as well as language-aware task creation.

Review every resolved file against the redesign SHA and the Java delta, then inspect clean merges in CLI, version policy, `_tool_support`, tool composition, runtime, task utilities, planner policy/routing/spawn, report helpers, sandbox, and triage pipeline. Validate imports/collection, marker absence, JSONL parsing, and `git diff --check`. Stage only resolved merge paths after verification; do not commit without a user request.

Checkpoint: no unmerged paths or markers; redesign tests retained; language fields and Java files retained. Reviewer focus: hidden dropped branches and changed return contracts. This phase depends on Phase 0.

### Phase 2: Consolidate shared policies and selected capabilities

1. Establish the typed ecosystem selection boundary using `language.py` and narrowly scoped adapter modules if justified. Thread it through API/CLI, initial state, graph wrappers, task creation, subagent spawn, QA, setup, final scan, and report context.
2. Consolidate update/workaround/tactical/QA prompt policy. Render only selected-language commands, targets, examples, candidate semantics and tool inventory. Preserve redesign roles, prompt stability, exact instructions, migration authorization, bounded loops and evidence trust rules.
3. Reuse workspace read/search/edit/web tools and runtime event handling. Inject language-specific syntax/compile/test/smoke operations into `tools_validation.py`. Audit AST/edit helpers for Java support; never advertise unsupported JS AST behavior as working Java behavior. If a Java parser is needed, implement it behind the existing parser boundary with focused tests and explicit package metadata changes.
4. Integrate Maven singleton mutations, authorized dependencyManagement insertion, exact GAV handling, local property ownership, multi-module changes, no-fix removal and rollback into common transaction semantics. Native Maven XML preservation remains in `tools_manifest_java.py` and `maven_manifest_locator.py`.
5. Reuse a single resource lifecycle for workspace and Maven cache, including orchestration exceptions and optional final scans. Preserve protected path/command validation and host immutability.

Tests: parameterize shared prompt/tool/attempt invariants across Node and Java; assert foreign ecosystem tools and commands are absent; verify explicit language override, invalid language rejection, context preservation, transactional rollback and cleanup after partial creation. Preserve Node tool schemas/commands where public behavior depends on them. No environment-variable or public result migration is intended.

Checkpoint: same engine contracts and capabilities selected by language; no copied Java agent policy. Depends on Phase 1. Reviewer focus: accidental behavior changes from a broad abstraction; keep changes local when composition suffices.

### Phase 3: Integrate Maven into portfolio and QA contracts

1. Extend `supervisor_portfolio.py`, `portfolio_solver.py`, `portfolio_certifier.py`, and `portfolio_orchestrator.py` at ecosystem boundaries. Retain copy-on-write planning, shared state transitions, plan digests, certification status, and Supervisor-only dispatch.
2. Construct exact Maven targets and candidate domains from POM/managed-property ownership and Maven registry evidence. Use Maven comparison for floors, ordering, equivalence and attempted versions. Adapt neutral solver ordering/compatibility where necessary in `solver/cpsat.py` and typed solver contracts; preserve existing npm objectives and constraints.
3. Certify selected Maven assignments using native resolution in isolated scratch state. Preserve the source workspace, cached resources, budgets, provenance and fail-closed outcomes. Feed certified actions into the same committed attempt/batch lifecycle. Do not claim Java portfolio support from a singleton bypass.
4. Generalize atomic mutation execution by selected ecosystem. Preserve union checkpoint and per-attempt provenance, rollback all mutations on failure, and reuse delta isolation with native canary execution and verified restoration. Test two-task Maven failures with mocks in addition to the requested singleton fixture.
5. Preserve `_TargetedPackageBatch`, new ODC evidence fields, current attempt bindings and task-scoped scan projections. Extend Maven setup/input inventories and non-npm staging without weakening npm lockfile occurrence attribution. Use resolved GAV evidence for effective executed versions; unsupported or incomplete inputs stay incomplete.
6. Reuse QA policy and typed evaluations; parse Maven compile/test outcomes using native result structure. Keep `target_package_absent`, new issues, inventory updates/removals, scan skipped/performed semantics, incremental terminal evidence, cumulative triage findings, and final full-scan opt-in.

Tests: extend Java Supervisor/portfolio/solver/QA tests with exact identities, non-semver qualifiers, certified dispatch, no registry calls from workers, immutable snapshots, stale results, unrelated task findings, incomplete scan rejection, source restoration, child retry anchors, and cleanup exceptions. Preserve all existing portfolio and targeted-scan tests. New tests mock Docker, HTTP, LLM and subprocess boundaries.

Checkpoint: Java traverses the shared graph and portfolio contract; both ecosystems preserve engine invariants. Depends on Phase 2. Reviewer focus: Maven effective dependency semantics, multi-module ownership, and scan authority. Internal model extensions must use compatible defaults and preserve serialized meaning; avoid renaming persisted fields solely for aesthetics.

### Phase 4: Consolidation and regression validation

Remove duplicate language-independent helpers and obsolete prompt copies after all callers use shared operations. Keep thin compatibility imports only where existing repository callers still need them; do not introduce Phase 4 compatibility shims. Update `docs/architecture.md` and Java example documentation to describe implemented support and actual limits. Leave unrelated skill changes and generated diagrams untouched.

Run focused tests after each affected seam, then the full deterministic suite and repository checks. Fix integration failures before starting live fixtures. Compare changed source against the pinned redesign commit and document each difference as an ecosystem operation, compatible plumbing, or a justified shared refactor. Any unexplained orchestration difference fails review.

Checkpoint: tests/checks pass, no unresolved merge state, implementation diff reviewable. Depends on Phases 1–3. Roll back a failed adapter change to the last implementation checkpoint without discarding the user's merge. No release, push or merge commit is required.

## Verification Strategy

Use the local environment and workspace-local temporary directories:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/remedy --help
git -c core.fsmonitor=false diff --check
git -c core.fsmonitor=false diff --name-only --diff-filter=U
```

Run targeted existing suites for contracts/state, public API, Java language/Supervisor, Maven manifest/localization/registry/version policy, worker runtime/prompts/toolbelts, portfolio solver/certifier/orchestrator, QA policy/targeted scans, workspace/sandbox/teardown, report persistence, and fixture acceptance. Add shared contract tests around observed gaps rather than tests mirroring new helpers.

Live fixtures are mandatory end-to-end acceptance attempts after deterministic checks:

```bash
.venv/bin/python examples/juice_shop/fixtures/shared_name/run_shared_name.py --repo data/clones/juice-shop
.venv/bin/python examples/juice_shop/run_batch.py --iterations 1 --packages base64url got esbuild jws --repo data/clones/juice-shop
.venv/bin/python examples/Vulnado/run_batch.py --iterations 1 --batch-size 1 --seed 6
```

| Fixture | Required evidence |
| --- | --- |
| Shared Angular name | Three requested packages expand to the nine Angular packages enumerated in `fixture_acceptance.py:751`; all effective task lineages pass QA in one atomic batch, sharing action provenance. Complete task scan evidence and patch replay prove the requested package effects. |
| Four packages | First base64url/jws attempts share one atomic batch and fail QA; bounded delta isolation records `candidate_restored=true` and blames only jws. All other batches are singleton. base64url, got and esbuild pass; jws remains unresolved. Use `verify_four_package_fixture` and patch replay. |
| Vulnado | Seed 6 selects SnakeYAML. The committed Maven target/version is executed, resolution and task-scoped scan evidence complete, QA passes, and emitted patch changes only authorized POM/source targets. Verify actual selected coordinate rather than assuming seed output. |

Use `verify_shared_name_fixture` and `verify_four_package_fixture` on each run's own result/evidence/patch files. Inspect the helper signatures and runner outputs to bind those paths correctly. Preserve and report logs and run identifiers under ignored `data/`; do not commit trajectories, reports, secrets or clones. Snapshot and restore tracked fixtures/suppression inputs changed by the runners. Fixture runners explicitly copy suppressions into clones; preserve original clone bytes/state around these authorized runs and verify the engine itself never mutates the host checkout.

A PasteBin-related test error is non-blocking per the user. Record it separately and continue; do not hide other failures, broadly skip tests, or change production QA policy to force a pass. Live Docker/network/LLM/prerequisite failures are evidence gaps, never successful fixture outcomes. Request tool escalation when necessary under existing authorization; do not repeatedly retry an unchanged access failure.

## Compatibility and Rollout

Preserve public API/CLI defaults, explicit language behavior, environment names and AppSettings boundaries, typed results/unified patches, persistent task/attempt semantics, telemetry correlation, retry budgets, version authorization and immutable instruction digests. Preserve Java POM byte safety, XML validation, exact coordinate and owner checks, Maven cache lifetime, and npm singleton/batch behavior. No new orchestration mode or feature flag is planned.

Deliver a reviewable uncommitted implementation with resolved merge index and validation evidence. No commit, push, deployment, or external messaging is included. Stop immediately at the requested usage threshold and record incomplete work without marking acceptance complete.

## Deferred Work and Evidence Gaps

- Gradle, Kotlin, additional languages and automatic mixed-language splitting are outside this change. Root-only detection is an existing contract; nested-only Maven detection would need an explicit separate policy decision.
- Java reachability is deliberately unknown where dependency-to-class evidence is absent. Do not invent a package/import equivalence.
- Live fixture outcomes, network/Docker availability and Maven certification behavior remain unverified until implementation and execution. Existing repository tests have not run while conflict markers prevent collection.
- Unrelated eval UI/database internals, generated reports, and deeper performance profiling are outside the Java integration audit; preserve clean redesign changes there.
- The proposed adapter shape is conditional on actual cohesion during implementation. Avoid a broad file reorganization and cosmetic public/persisted-field renames.

## Definition of Done

1. All 26 conflicts resolved, no conflict markers or unmerged paths, original user work preserved.
2. Every change from the pinned redesign baseline is explained; shared agent logic remains authoritative and singular.
3. Language-independent code/prompt policy/tools are reused; only selected-language capabilities and instructions are exposed at every agent boundary.
4. Java follows shared portfolio/attempt/worker/QA/reconciliation/report flow with native Maven behavior and verified authorization; no singleton bypass or uncertified dispatch.
5. Deterministic tests and static checks pass except explicitly recorded PasteBin behavior; any unrelated failure remains visible.
6. All three requested fixtures are attempted and their required outcomes verified from current correlated evidence; any environmental blocker is reported as incomplete acceptance.
7. Run-owned resources are cleaned up, host clone and runner-mutated fixture state restored, generated artifacts stay ignored, and no secrets enter the diff.
8. Implementation is carried out by `gpt-6-luna` at `max`; the five-hour 98% stop rule is enforced for all work.

## Primary Maven References

- [Maven dependency mechanism](https://maven.apache.org/guides/introduction/introduction-to-dependency-mechanism.html): native mediation, dependency management, and transitive dependencies justify an ecosystem resolver boundary.
- [Maven ComparableVersion](https://maven.apache.org/ref/3.9.15/maven-artifact/apidocs/org/apache/maven/artifact/versioning/ComparableVersion.html): native qualifier ordering supports preserving Maven comparison rather than npm semver coercion.
- [Maven Surefire](https://maven.apache.org/components/surefire-archives/surefire-LATEST/maven-surefire-plugin/): native test execution and XML reports belong behind the selected validation operations.
