# Triage Task-Completion Evaluation Mapping

The triage replay suite in tests/evals/test_triage_eval.py uses only
DeepEval TaskCompletionMetric. Each golden puts its completion contract in
completion_task and the replayed final outcome in actual_output. The
expected_output, provenance, and historical evidence fields are retained for
review and persistence; they are not treated as a second metric or as
deterministic field assertions. Tool calls are empty because triage completion
does not require a worker tool trace.

## Provenance labels

- historical_analogue: the trajectories contain the package, advisory, or
  behavior family, but at least one proposed identifier or signal is a
  scenario variant.
- synthetic_policy_case: no matching finding was found in the trajectories;
  the case tests an explicit triage policy.
- pipeline_boundary_synthetic: the case exercises a failure or handoff
  boundary outside the triage model decision itself.

## Case map

| Golden | Choice | Historical trajectory evidence and implication |
| --- | --- | --- |
| triage-clean-production-actionable | Good core case | phase5_20260721T083757Z_d661e0c3-2466-4cf0-902a-a3215e3affdd.md records a reachable direct cookie group with the same domain/path isolation family and a completed run. The fixture uses a different version, CVE, and EPSS as a scenario variant; its triage span was skipped. |
| triage-clean-ghsa-without-cve | Good after making context explicit | phase5_20260804T091138Z_4c1a02d9-8e2f-46a3-8206-f324e7583405.md has a GHSA-only analogue whose LLM result stayed valid but used UNKNOWN priority when severity/context were missing. The fixture supplies original MEDIUM severity and impact context. The old GHSA-952P-6RRQ-RCJV pairing should not be reused: the 2026-08-25 trace associates it with micromatch, not validator. |
| triage-guardrail-drop-everything-kev-override | Good synthetic recovery case | The @tootallnate/once group in phase5_20260722T063315Z_33cec85b-9906-43bb-9edb-a9a3c7b0ec18.md is a useful package analogue, but it is CVE-2026-3449 with EPSS 0.00112 and KEV false. The proposed CVE-2022-37601 and KEV 0.85 must remain labeled synthetic. |
| triage-guardrail-imminent-threat-public-escalation | Good policy-boundary case | The same trajectory contains the exact express-jwt CVE-2020-15084 group and reachability, but historical severity is HIGH, EPSS 0.01054, and KEV false. Another recorded LLM verdict kept it valid HIGH. The MEDIUM/.45 escalation is a deliberate guardrail variant. |
| triage-guardrail-imminent-threat-internal-escalation | Good synthetic recovery case | No top-level trajectory contains axios with CVE-2023-45857. Keep the case as a synthetic test of the internal high-EPSS floor. |
| triage-guardrail-floor-downgrade-internal | Good synthetic recovery case | No top-level trajectory contains libxmljs with CVE-2015-1823. Keep the low-EPSS internal downgrade explicitly synthetic. |
| triage-guardrail-kev-vetoes-false-positive | Good recovery case with package analogue | Historical jsonwebtoken groups are reachable in phase5_20260722T063315Z_33cec85b-9906-43bb-9edb-a9a3c7b0ec18.md, but contain different CVEs and EPSS 0.08655 with KEV false. The CVE-2022-23529 KEV signal is not historical evidence. |
| triage-rule-a-scanner-tech-mismatch | Good Rule A case with analogue evidence | phase5_20260722T085156Z_6265726c-40d5-4c8e-92f0-73d2f29b0f30.md contains the Java Maven component commons-io:commons-io, but with different CVEs. The npm/Node mismatch remains synthetic and should be judged from the explicit task input. |
| triage-rule-b-contradictory-os | Good focused synthetic case | No trajectory contains CVE-2021-38209 or a Windows/Linux Rule B decision. The advisory's Linux prerequisite and deployment_os=windows must stay explicit. |
| triage-rule-c-dev-test-scope-exclusion | Good focused synthetic case | No trajectory contains the proposed mocha finding. The test/helpers/runner.js path and dev/CI environment are necessary because the production-scope rule is intentionally narrow. |
| triage-rule-d-unreachable-code-preserves-validity | Excellent invariant case | Historical root states record explicit reachability fields, but no sqlite3 CVE-2022-42915 verdict appears. This is a synthetic extension of a real observed field and tests the important valid-plus-unreachable outcome. |
| triage-speculative-defense-rejection | Good policy trap, not a historical replay | No matching Express Semgrep route appears in the trajectories. It is still useful because it tests the explicit rule that an unproven frontend or caller defense is not a false-positive contradiction. |
| triage-fallback-llm-timeout-to-deterministic | Useful boundary case, not triage-only | No trajectory records APITimeoutError or rate-limit recovery. TaskCompletionMetric can judge the stated completed outcome, but cannot prove which exception or fallback branch occurred; retain deterministic mechanism tests outside this suite if needed. |
| triage-pipeline-selection-hallucinated-issue-id | Useful boundary case, not triage-only | No trajectory records TriageSelectionError or a hallucinated recommendation. This belongs in scope only when the triage-to-Supervisor handoff is part of the task being judged. |
| triage-multicve-conflict-signal-resolution | Good grouped-risk case with synthetic signals | The historical jsonwebtoken group in phase5_20260722T063315Z_33cec85b-9906-43bb-9edb-a9a3c7b0ec18.md demonstrates multi-CVE grouping and later reconciliation, but not the proposed low-disputed/high-KEV pair. Primary enrichment selection is a pipeline behavior, not an observed historical verdict. |

The dataset intentionally keeps all 15 proposed case IDs, while labeling
unsupported or boundary cases instead of presenting them as historical
replays.
