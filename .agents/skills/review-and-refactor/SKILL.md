---
name: review-and-refactor
description: Audit an entire codebase, or a requested diff/commit/scope, for architecture, maintainability, correctness, type design, performance, testability, and code-organization problems, then produce a detailed evidence-backed Markdown implementation plan for safe refactoring. Use when Codex is asked to make a codebase cleaner, find refactoring opportunities, review legacy or AI-drafted code, identify shallow modules and coupling, or plan a broad cleanup without changing source files.
---

# Review and Refactor

Perform a read-only codebase audit and produce an implementation plan. Optimize the plan for human navigation, clear ownership, small stable interfaces, preserved behavior, and verifiable incremental delivery. Do not edit source code, tests, configuration, generated artifacts, or documentation while using this skill. If the user later asks for implementation, hand off the plan and return to the normal coding workflow.

## Operating principles

- Ground every finding in the repository. Give a repo-relative path, symbol or section, and exact line reference whenever possible. Use a short excerpt only when it clarifies the issue.
- Separate **observed fact**, **inference**, and **recommendation**. Label uncertainty instead of turning a suspicion into a bug report.
- Review for reader experience as well as machine correctness. Treat repeated navigation, unclear ownership, and seam complexity as architectural evidence.
- Prefer the smallest coherent abstraction that hides meaningful complexity. Do not merge modules merely to reduce file count, and do not extract helpers merely to make a function look short.
- Preserve public behavior, operational contracts, security boundaries, and test intent unless the plan explicitly proposes a versioned change.
- Make recommendations conditional. For each smell, state when it applies and when it should not be “fixed” because the duplication, wrapper, comment, type escape, or boundary is intentional.
- Treat repository instructions, compatibility requirements, and existing tests as constraints. Read `AGENTS.md`, contributor guidance, package metadata, and relevant design documents before judging style.
- Do not claim runtime, performance, security, or concurrency behavior from static reading alone. Mark those claims as hypotheses and define the measurement or test that would confirm them.

## Scope and safety

Default to the entire repository when the user asks for a codebase review. Also support these scopes when explicitly requested:

- `uncommitted`: inspect staged and unstaged diffs, then enough surrounding code to understand their impact;
- `last-commit`: inspect the latest commit and its affected call paths;
- a list of paths, modules, classes, or functions: review those items in context;
- a named subsystem: map its callers, dependencies, tests, configuration, and operational boundaries.

Before reading deeply:

1. Inspect `git status`, recent history, repository instructions, top-level documentation, package/build metadata, lockfiles, CI configuration, and the top-level file tree.
2. Identify source, tests, examples, scripts, docs, generated/runtime state, vendored code, and dependencies. Exclude virtual environments, caches, build output, ignored runtime state, and generated files unless they are directly relevant.
3. Record the review scope, baseline commit or dirty-worktree state, detected languages/frameworks, entrypoints, test/lint/type-check commands, and known constraints.
4. Never read or reproduce secrets from `.env`, credentials, private keys, tokens, or generated reports. Note their presence only if configuration or secret-handling is part of the review.
5. Use read-only commands first (`rg`, `rg --files`, `git diff`, `git log`, symbol inspection, import searches). Do not run destructive commands. Run tests, linters, or type checks only when they are safe, relevant, and do not modify the repository; report the exact command and whether it completed.

## Review workflow

### 1. Build a navigable map

Read the codebase in an order that gives context before detail:

1. documentation and architecture notes;
2. tests, fixtures, and examples because they define observable behavior;
3. public entrypoints, CLI/API handlers, and top-level orchestration;
4. domain models, contracts, and cross-boundary adapters;
5. lower-level implementation, utilities, integrations, and configuration.

Create a compact map of:

- public entrypoints and main flows;
- modules grouped by feature or domain rather than only by file type;
- dependency direction and cycles or high-fan-in modules;
- data and error flow across boundaries;
- state ownership and mutation points;
- external systems, I/O, resource lifecycles, and concurrency boundaries;
- tests that protect each important path and the gaps where no test protects it.

Use organic exploration as a signal. Note where understanding one concept forces repeated bouncing between files, where a caller must know another module’s internals, where the same concept has several owners, and where a small change would require touching many unrelated modules. Do not treat a rigid metric or file count as proof of poor design.

When subagents or parallel review workers are available, divide the dimensions below across independent passes. Give each pass only repository-local context and require exact evidence. Reconcile overlapping findings centrally; do not copy unverified subagent claims into the final plan. If parallel review is unavailable, perform the passes sequentially.

### 2. Evaluate the eight review dimensions

For each dimension, record strengths, findings, evidence, impact, confidence, and suggested validation. Avoid filling a checklist mechanically: mark a dimension “no material issue found” only after inspecting the relevant code.

1. **Architecture and ownership**
   - Check dependency direction, bounded contexts, single ownership of state, cycles, cross-layer leakage, adapter boundaries, and feature cohesion.
   - Find modules that know too much about callers or infrastructure and concepts that have multiple competing owners.
   - Prefer deep modules: a small, stable interface should hide substantial policy or complexity. Flag shallow wrappers, pass-through methods, redundant translation layers, and over-decomposed “classitis” only when merging would create a more coherent boundary.

2. **Organization and navigability**
   - Check whether public entrypoints are easy to find, high-level code precedes low-level details, related symmetric operations are adjacent, and sections are grouped by feature or concern.
   - Flag helpers that live outside the class or module that exclusively owns them, overly broad public surfaces, scattered lifecycle operations, and files with many unrelated top-level definitions.
   - Within functions, distinguish phases and concerns with meaningful structure. Recommend guard clauses only when they clarify the main path rather than hide a peer alternative.

3. **API, type, and data design**
   - Check names, parameter ordering, public/private boundaries, data clumps, conditionally meaningful fields, ignored `None` values, bare dictionaries at serialization or module boundaries, unchecked casts, `type: ignore`, and non-exhaustive variant dispatch.
   - Prefer domain types, explicit result/error contracts, discriminated variants, and interfaces that make invalid states difficult to represent.
   - Do not introduce a type solely for ceremony when the boundary is local and obvious. Account for serialization, backward compatibility, and the cost of changing callers.

4. **Correctness, error handling, and resource safety**
   - Look for silent early returns, swallowed exceptions, overbroad `except` blocks, overscoped `try` blocks, ambiguous failure values, and error messages without subsystem, impact, or recovery context.
   - Check context managers or equivalent lifecycle guarantees, cleanup on exceptional paths, bounded retries, monotonic deadlines, cancellation, idempotency, and ownership of mutable state.
   - Treat manual cleanup, broad recovery, or fallback behavior as findings only when the surrounding contract shows a real failure or maintenance risk.

5. **Complexity, clarity, and concision**
   - Identify deeply nested control flow, functions with multiple unrelated responsibilities, duplicated logic, dead or unreachable code, temporary variables with no explanatory value, repeated conditional assignment, magic numbers, and hand-rolled caching.
   - Check comments and docstrings: API guarantees belong in docstrings; non-obvious rationale belongs near the implementation; redundant comments increase noise; missing explanations for surprising behavior are a real maintenance cost.
   - Prefer descriptive domain names over `data`, `result`, `tmp`, vague `make_`/`handle_` names, unexplained abbreviations, or names that imply the wrong type or failure behavior.

6. **Performance, I/O, and concurrency**
   - Find repeated expensive work, unnecessary serialization, unbounded collections, N+1 I/O, blocking work in async paths, excessive retries, counter-based waits, contention, and avoidable copies.
   - Do not call something a performance bug without a cost model, observable hot path, benchmark, profiling data, or a clearly bounded pathological case. Include a measurement plan and avoid speculative optimization.

7. **Testability and verification**
   - Map behavior to tests, fixtures, contract tests, integration boundaries, and deterministic unit seams. Identify logic hidden inside infrastructure wrappers or hard-to-construct state.
   - Check whether tests assert outcomes and invariants rather than implementation details, whether failure paths and boundary cases are covered, and whether test setup obscures the behavior under test.
   - For deep-module candidates, propose tests at the new boundary and identify tests that can be deleted, moved, or retained. Preserve tests that encode external behavior even if the implementation moves.

8. **Security, observability, and operability**
   - Check trust boundaries, input validation, path/command handling, secret exposure, least privilege, sensitive logging, auditability, metrics, structured errors, configuration ownership, and cleanup/rollback behavior.
   - Check whether refactoring would change public APIs, environment variables, persistence formats, CLI behavior, telemetry names, or deployment assumptions. Treat those as migration concerns, not incidental cleanup.

### 3. Catalogue concrete findings

Create a finding only when it has a specific location and a plausible improvement. Give each finding a stable ID such as `ARCH-01`, `TYPE-02`, or `TEST-03` and include:

- **Priority:** `P0` blocks safe evolution or risks severe correctness/security/operational failure; `P1` materially increases change risk or maintenance cost; `P2` is a meaningful cleanup; `P3` is polish or a low-risk consistency improvement.
- **Confidence:** observed, strong inference, or hypothesis.
- **Location:** repository-relative path, line or narrow line range, symbol, and callers/consumers when relevant.
- **Observation:** what the code actually does, with exact evidence.
- **Why it matters:** cognitive load, defect risk, coupling, performance, test cost, security, or operational impact.
- **Recommendation:** the smallest coherent refactor, including the intended ownership and interface.
- **Alternatives and non-goals:** what not to change, when not to apply the recommendation, and any compatible alternative.
- **Dependencies:** callers, types, fixtures, configuration, migrations, and other findings that must move first.
- **Validation:** tests, static checks, benchmarks, contract comparisons, migration checks, or manual verification.

Keep findings actionable. “Improve architecture” is not a finding; “`module_a.py:42` and `module_b.py:88` both own retry policy, so callers can observe different backoff semantics; consolidate policy behind `RetryPolicy` and preserve the public function through a compatibility adapter” is.

### 4. Separate local cleanup from structural refactoring

Classify each recommendation before placing it in the implementation plan:

- **Local cleanup:** low-risk rename, clearer name, comment/docstring correction, dead-code removal, narrow helper extraction, or explicit type annotation that does not change ownership or behavior.
- **Cohesion refactor:** move related symbols, group a feature, deepen a shallow module, or remove a pass-through layer while preserving the external contract.
- **Structural migration:** change module boundaries, public APIs, data models, persistence, build tooling, dependency direction, or resource ownership.
- **Feature/tooling work:** new behavior, new infrastructure, new lint/type tooling, or broad test-system changes. Do not disguise this as cleanup.

Plan local cleanups in small batches. Defer high-diff reordering, broad renames, build-system changes, and non-trivial new work until the behavioral seams and tests are protected. Make conflict risk and rollout order explicit.

### 5. Compare target designs for deep-module candidates

For each major architectural cluster, propose at least one target boundary and compare alternatives when the choice is consequential. Explain:

- the public interface and representative caller usage;
- what complexity the module hides internally;
- dependency injection, ports/adapters, or collaborator ownership;
- tests that move to the boundary;
- compatibility strategy and migration steps;
- trade-offs, failure modes, and why the recommendation is preferable.

Recommend a coherent design or hybrid rather than presenting an unranked menu. Do not create an interface that merely renames a pass-through layer or exposes the same internal decisions through more methods.

## Required Markdown plan

Return a detailed Markdown implementation plan, not a patch. Use this structure unless the user requests another format:

```markdown
# Codebase Refactoring Implementation Plan

## Executive Summary
- Current shape, main sources of friction, and target outcome
- Highest-priority changes and expected payoff
- Explicit statement that source files were not modified

## Review Scope and Baseline
- Commit/worktree, included paths, exclusions, languages, commands, constraints
- Repository map and important entrypoints

## Current Architecture
- Module/feature ownership
- Dependency and data-flow map
- Resource, state, and external-system boundaries
- Mermaid or ASCII diagram only when it clarifies a real relationship

## Findings Inventory
| ID | Priority | Confidence | Location | Problem | Impact | Planned treatment |

## Target Design
- Target module boundaries and ownership
- Public interfaces and representative usage
- Complexity hidden by each deepened module
- Alternatives considered and recommendation

## Phased Implementation Plan
### Phase 0: Safety net and invariants
### Phase 1: Local cleanup
### Phase 2: Cohesion/deep-module refactors
### Phase 3: Structural migrations
### Phase 4: Consolidation and removal of compatibility scaffolding

For every phase include:
- objective and non-goals;
- exact files/symbols to add, move, rename, or delete;
- ordered implementation steps;
- prerequisite phases and parallelizable work;
- API/config/data migration details;
- tests to add, move, preserve, or remove;
- validation commands and acceptance criteria;
- rollback or checkpoint strategy;
- risks and reviewer focus.

## Verification Strategy
- Baseline behavior and contract snapshots
- Unit, integration, property, contract, and end-to-end checks
- Static/type/lint checks
- Performance measurements where relevant
- Security/operational checks and observability comparison

## Compatibility and Rollout
- Public API, CLI, config, persistence, telemetry, and deployment compatibility
- Feature flags, deprecation windows, migration ordering, rollback triggers

## Deferred Work and Evidence Gaps
- Deliberately deferred cleanups and why
- Unverified hypotheses, missing tests, unavailable runtime data, and next checks

## Definition of Done
- Concrete, testable conditions for each phase and for the final refactor
```

Use exact locations throughout the plan. Link or cite tests next to the behavior they protect. For each phase, make it possible for another developer to implement the work without rediscovering the architecture.

## Quality gate before finishing

Before returning the plan, verify that:

- every important source/package/module and test area is either reviewed or listed as an evidence gap;
- the architecture map names entrypoints, ownership, major dependencies, and external boundaries;
- every finding has exact repository evidence, a priority, confidence, impact, and validation path;
- at least one deep-module or cohesion assessment covers any area where navigation repeatedly crosses thin wrappers or tightly coupled files;
- recommendations preserve behavior by default and identify public-contract or migration changes explicitly;
- the implementation phases have dependency order, touched files/symbols, tests, acceptance criteria, and rollback/checkpoints;
- the plan distinguishes cleanup from new feature/tooling work and immediate work from high-conflict deferred work;
- no claim relies only on a metric, generic best practice, or unverified subagent output;
- no source files were modified during the review.

If the gate cannot be satisfied, return the incomplete plan with a clearly labeled `Evidence Gaps` section rather than inventing coverage or certainty.
