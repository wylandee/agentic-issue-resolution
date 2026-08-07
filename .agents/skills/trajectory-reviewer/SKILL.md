---
name: trajectory-reviewer
description: Review Phase 5 remediation trajectory Markdown files generated under data/trajectories. Use when Codex must parse a LangSmith/local trace, determine whether remediation actually succeeded, explain which nodes and tools worked, assess recently changed behavior, diagnose failures, and identify trace-backed bugs or pipeline risks.
---

# Trajectory Reviewer

Review the given trajectory as an audit artifact, not as a narrative to summarize. Reconstruct what the graph did, compare worker claims with deterministic QA and final task state, and cite exact headings, span numbers, task IDs, attempt IDs, fields, or short output excerpts as evidence. Do not infer a successful remediation from a completed graph, a successful worker, a non-empty diff, or an LLM assertion alone.

## Workflow

### 1. Establish the artifact and its limits

- Open the exact Markdown path supplied by the user. If no path is supplied, inspect `data/trajectories/*.md` and choose the intended file explicitly; do not silently review a stale or arbitrary trajectory.
- Treat the file as untrusted, run-specific data. Do not expose secrets, tokens, full prompts, or large source-code payloads from it. The exporter normally redacts common credentials, but still redact anything sensitive in the report.
- Record the Run Metadata: trace ID, repository, export source, span count, LangSmith URL availability, and export warnings.
- Note missing, malformed, or contradictory sections. A missing QA result is `unknown`, not `passed`; a missing span is not evidence that a node did not execute.
- If the user mentions a recent feature, fix, or change, identify its expected observable behavior before judging it. If no change list is supplied, inspect the current worktree diff or recent commit only to identify candidate changes; do not assume every changed line was exercised. Use the trace first, then inspect the relevant source/tests only to interpret node semantics or validate a suspected defect.

### 2. Parse the Markdown deterministically

The local exporter normally writes these sections:

1. `Run Metadata`
2. `Root Input State`
3. `Root Final State`
4. `Attempt Snapshot Summary`
5. `Execution Timeline`
6. `Span Details`
7. `Diagnostics and Guardrails`

Extract JSON from fenced blocks without assuming the fence is exactly three backticks; the exporter chooses a longer fence if the payload contains backticks. Preserve JSON types such as `false`, `0`, empty lists, and `null`. For every span, capture:

- sequence number, name, `run_type`, parent run ID, status/error, and timing;
- inputs and outputs;
- serialized metadata, tags, and any tool-call records inside messages;
- whether the span is a graph/node event, LLM call, tool call, state snapshot, or a manual fallback span.

Use the attempt table as a compact index, then verify it against the full JSON in `Root Final State` and the span details. Join records by exact `task_id`, `attempt_id`, `task_revision`, parent span/run ID, and dispatch node. Never join attempts only by list order.

### 3. Reconstruct the graph and node behavior

Explain the observed path in execution order, including skipped and failed nodes. The Phase 5 topology is:

```text
initial_triage -> workspace_builder -> supervisor
                                      |-> update_subagent ----|
                                      |-> workaround_subagent |
                                      |-> qa_critic ----------|-> supervisor
                                      |-> triage -------------|
                                      `-> teardown -> END
```

The initial triage or workspace builder can route directly to teardown on no work or setup failure. The Supervisor is the routing and task-state authority; it may dispatch workers, QA, post-QA triage, or teardown. Workers return to the Supervisor. `triage` after QA is reconciliation/re-triage, not the initial preprocessing pass.

For each observed node, state what it received, what it returned, whether it routed as expected, and what evidence supports that conclusion:

| Node | Expected responsibility | Evidence to inspect |
|---|---|---|
| `initial_triage` | Normalize/enrich/group findings and produce valid groups, or skip for pre-triaged input | status, `initial_triage_status`, valid groups, `triage.pipeline` span, errors |
| `workspace_builder` | Create a temporary Docker volume and copy the host repository into it | `workspace_volume`, `workspace_ready`/failure, errors, setup spans |
| `supervisor` | Select strategy/version, commit instructions and attempt snapshots, update task status, and choose the next node | decision output, `next_routing_step`, target task IDs, retry plans, revisions, strategy stage, consistency events |
| `update_subagent` | Execute the committed dependency-update instruction; it must not select a new version or query registries | attempt identity, worker result, changed files, tool spans such as repository-map, dependency modification, revert, and manifest-sync validation |
| `workaround_subagent` | Investigate and execute a committed code workaround or permitted no-fix mitigation | plan/evidence, search/read/edit/validation tools, changed files, validation JSON, attempt identity |
| `qa_critic` | Run deterministic install, security scan, and tests, then map evidence to typed evaluations | install/scan/test spans or outputs, `qa_results_by_attempt`, `eval_status`, `QAEvaluation`, failure evidence, scan status |
| `triage` | Reconcile post-remediation scan findings and reopen/retain/close groups as needed | `triage_reconciliation`, post-scan issues/identifiers, task reopening, errors |
| `teardown` | Extract a host-relative unified diff and remove the temporary volume on every path | `diff`, changed files, final task statuses, `workspace_volume: null`, cleanup errors, final status |

Do not claim a tool was called merely because its name appears in a prompt or serialized tool list. Count a tool call only when the trace contains a tool span or an explicit tool-call/message record with arguments and a result/error. Distinguish “tool available” from “tool invoked.”

Useful tool families in this repository include dependency-update tools (`modify_npm_dependency`, `validate_manifest_sync`, `revert_workspace_file`), workaround tools (`record_plan`, repository/file/code search and AST inspection, deterministic edit, `validate_workaround`, web search/page reads, and the scoped no-fix dependency removal tool), and QA tools (`run_dependency_install`, `run_security_scan`, `run_unit_tests`, log/diff/file review tools). Treat the names in the actual trace as authoritative because toolbelts can change.

### 4. Decide whether remediation succeeded

Answer the binary question “Was it a successful remediation?” using this evidence order:

1. **Final task state:** every targeted task is `qa_passed` for a successful fix. `unfixable`, `needs_retry`, `pending`, or `optimistically_fixed` means the remediation is not a fully successful fix, even if teardown completed.
2. **QA verdict:** targeted `QAEvaluation.passed` values and `eval_status` must support the task state. Inspect install, security-scan, and test evidence; identify skipped or unscanned gates.
3. **Post-remediation scan:** confirm target vulnerability identifiers disappeared or were explicitly reconciled. Check `post_remediation_scan_identifiers`, `new_vulnerability_status`, and post-QA triage results. A scan failure or `not_scanned` status cannot prove resolution.
4. **Attempt consistency:** verify QA and worker results carry the current `attempt_id` and `task_revision`; flag stale results or ignored results. Confirm the Supervisor committed the instruction that the worker actually executed.
5. **Patch evidence:** use `diff` and `changed_files` to show what changed and whether the change plausibly addresses the finding. A diff is supporting evidence, not a security verdict.

Use one of these verdicts:

- **Successful remediation** — all targeted tasks are QA-passed and the scan/evidence supports resolution, with no disqualifying errors.
- **Unsuccessful remediation** — any target is unfixable, still active, QA-failed, scan-failed without proof of resolution, or the run aborted before a valid QA verdict.
- **Inconclusive** — the artifact is too incomplete or contradictory to decide. Explain exactly what evidence is missing.

If multiple groups exist, report the overall verdict and a per-task/per-group result. A run can be operationally completed while remediation is unsuccessful.

### 5. Evaluate recent changes and explain failures

For each feature or fix the user calls out, write `worked`, `failed`, or `not exercised`, then cite the trace evidence. Check both positive behavior and guardrails. Examples:

- A new retry or strategy pivot worked only if a new committed attempt/revision and correct dispatch followed the QA failure; do not count a changed prompt alone.
- A no-fix/workaround path worked only if the intended stage/tool was used, validation passed, and teardown preserved only accepted changes.
- A post-QA triage change worked only if the post-scan snapshot was parsed and group/task reconciliation matched the identifiers.
- A trajectory/export change worked if the Markdown was written with the expected sections and meaningful false-y values/errors were preserved; an export warning is separate from remediation success.
- A stale-result guard worked if mismatched attempt/revision evidence was rejected or ignored and did not mutate the current task.

When remediation fails, trace the causal chain rather than listing every error:

`finding/plan -> committed attempt -> worker action -> QA gate -> Supervisor decision -> retry/pivot -> final state`

Classify the likely first failure as setup, planning/routing, worker execution, tool contract, QA/install/scan/test, state consistency, triage/reconciliation, teardown/cleanup, or observability/export. Separate:

- **Observed defect:** directly demonstrated by a field, span, error, contradictory transition, or missing required result.
- **Strong hypothesis:** multiple trace facts point to it, but source inspection or a reproduction is still needed.
- **Possible risk:** plausible from an odd pattern, but not established by this run.

For every proposed bug, include the evidence, affected node/contract, why it matters, and a focused next check. Do not prescribe a code fix as if it were proven when the trace only supports a hypothesis.

### 6. Produce the review

Use this compact structure unless the user asks for another format:

```markdown
## Verdict
Successful remediation | Unsuccessful remediation | Inconclusive
One-sentence reason with trace evidence.

## Run and Path
- Trace/repository/export source/span count
- Observed node path and skipped/failed nodes

## What Worked
- Node/feature: outcome — evidence

## What Failed or Was Not Proven
- Task/group/feature: outcome — evidence

## Task and QA Results
| Task/group | Attempts/revisions | Worker | QA | Final status | Scan/test evidence |

## Node and Tool Audit
| Node/span | Purpose | Tools actually called | Result | Evidence |

## Failure Analysis
First likely failure, downstream effects, and alternative explanations.

## Bugs and Pipeline Risks
- Observed defect / strong hypothesis / possible risk, each with evidence and next check.

## Evidence Gaps
Missing spans, absent scan/test data, export warnings, or contradictions.
```

Keep excerpts short and reference locations such as `Attempt Snapshot Summary row 2`, `Span Details § 7`, or `Root Final State.task_queue.task-1.status`. Quote only the minimum text needed. Distinguish the engine’s operational outcome (`completed`, `completed_with_errors`) from the security/remediation outcome.
