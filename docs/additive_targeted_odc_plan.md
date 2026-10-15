# Targeted Dependency-Check QA: implementation status

This document was originally a design proposal for target-only synthetic
lockfiles and replacing the final full scan. The proposal is historical; the
current behavior is summarized below. It is not a work order to change
production behavior.

## Current behavior

- Workspace setup and task QA collect scoped Dependency-Check evidence. QA
  resolves the physical target package occurrences in the live workspace and
  stages one temporary project per changed source lockfile.
- Each staged project preserves the source lockfile, its applicable physical
  manifests, and the lockfile's dependency graph. Production QA does not replace
  this with a synthetic target-only lockfile or strip dependency edges.
- Lockfile closure analysis informs target coverage planning. It is distinct
  from the bytes emitted for the staged scan project. The existing sliced
  lockfile builder remains a separate utility and should not be described as
  the production QA staging path.
- Scan and QA results retain task/attempt provenance. The Supervisor uses the
  current task queue and incremental scan ledger when reconciling post-QA
  findings; failed or reverted attempts do not become accepted clean evidence.
- The `final_full_scan` graph node remains available as an opt-in. The default
  setting is `REMEDY_FINAL_FULL_SCAN_ENABLED=false`; without it, the terminal
  result reports incremental dependency-input coverage with
  `authoritative=false`, not a full-repository scan.

## Ownership and source

The Supervisor invokes portfolio planning and certification functions; there
is no separate portfolio graph node. Graph registration and routing are in
[`graph.py`](../src/remediation_engine/orchestration/graph.py). Targeted scan
artifact staging and evidence construction are in
[`_qa_runtime.py`](../src/remediation_engine/orchestration/_qa_runtime.py),
with report and deterministic gate handling in
[`qa_critic.py`](../src/remediation_engine/orchestration/qa_critic.py) and
[`qa_policy_engine.py`](../src/remediation_engine/orchestration/qa_policy_engine.py).
The optional full scan and its setting are implemented by
[`final_full_scan.py`](../src/remediation_engine/orchestration/final_full_scan.py),
[`supervisor_node.py`](../src/remediation_engine/orchestration/supervisor_node.py),
and [`settings.py`](../src/remediation_engine/settings.py).

Relevant regression coverage lives in
[`test_targeted_qa_scans.py`](../tests/test_targeted_qa_scans.py),
[`test_qa_policy.py`](../tests/test_qa_policy.py),
[`test_orchestrator_phase5_graph.py`](../tests/test_orchestrator_phase5_graph.py),
and [`test_supervisor_node.py`](../tests/test_supervisor_node.py). These tests
check physical occurrence targeting, staged artifact contents, scan coverage,
attempt correlation, and the optional final-scan route.

## Historical proposal disposition

The earlier proposal suggested generating a synthetic lockfile containing only
the selected target and its transitive closure, then removing the final full
scan node. Neither change describes the current implementation. Keep the raw
source-lockfile project shape and physical package occurrences stable unless a
separate approved behavior change alters those contracts. The final scan stays
opt-in; incremental coverage must not be presented as repository-wide
authoritative evidence.
