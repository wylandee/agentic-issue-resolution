# Java/Maven eval provenance

This file records the trajectory evidence used to design the Java/Maven golden
cases. The JSON fixtures are minimized and self-contained; running the evals
does not read these trajectory files, worktrees, or clones. Historical tool
observations are stored under `offline_fixture`. Expected tools and completion
contracts are authored separately for the current production interfaces.

## Evidence labels

- **Direct historical** means the fixture models an observed project finding or
  task QA result without claiming that a historical tool call is the current
  expected contract.
- **Normalized historical** means a historical worker or Supervisor action is
  represented through the current typed tool, snapshot, or batch contract.
- **Synthetic** means the guardrail or failure branch is an explicit replay
  fixture; it is not presented as a historical observation.

Span numbers below are the trajectory's span-table indices. Task revisions are
the revisions recorded in the final attempt snapshot unless a terminal task
revision is called out separately.

## Historical sources

| Trace ID | Full source path | Observed result and terminal-run qualification |
|---|---|---|
| `acc4a7ea-58ce-4140-becb-f26bc5764064` | `C:\Users\UserAdmin\.codex\worktrees\93fc\agentic-issue-resolution\data\trajectories\phase5_20261020T035317Z_acc4a7ea-58ce-4140-becb-f26bc5764064.md` | Initial Java triage is span 2. PostgreSQL `org.postgresql:postgresql` was changed to `42.7.11` by `modify_and_validate_maven_dependency` at span 82; jsoup `org.jsoup:jsoup` was changed to `1.15.3` at span 137. Task QA passed for both attempts (spans 92 and 147). The root run ended `completed_with_errors`, while `eval_status` was `all_passed`: teardown reported 90 newly detected identifiers, and the run also recorded a CP-SAT `OPTIMAL` status message. Do not describe the full run as clean. |
| `c2b3de3e-bbb6-4d05-a68d-9db7f65027ce` | `C:\Users\UserAdmin\.codex\worktrees\93fc\agentic-issue-resolution\data\trajectories\phase5_20261020T061021Z_c2b3de3e-bbb6-4d05-a68d-9db7f65027ce.md` | `modify_batch_maven_dependencies` applied Logback classic/core in `dependencyManagement` at `1.2.13` at span 80. Both task QA evaluations passed. The root run ended `completed_with_errors`, while `eval_status` was `all_passed`: teardown reported 88 newly detected identifiers, and the run also recorded a CP-SAT `OPTIMAL` status message. |
| `33bca3aa-634e-4f80-9778-3408bcfe4d7b` | `C:\Users\UserAdmin\.codex\worktrees\93fc\agentic-issue-resolution\data\trajectories\phase5_20261020T062231Z_33bca3aa-634e-4f80-9778-3408bcfe4d7b.md` | `modify_and_validate_maven_dependency` inserted AssertJ `org.assertj:assertj-core` at `3.27.7` in `dependencyManagement` at span 68. Its task QA passed. The root run ended `completed_with_errors`, while `eval_status` was `all_passed`: teardown reported 88 newly detected identifiers, and the run recorded a CP-SAT `INFEASIBLE` status message. |
| `d91b4065-fe6e-4cfa-9149-6ec4a6b86e14` | `D:\agentic-issue-resolution\data\trajectories\phase5_20261008T071822Z_d91b4065-fe6e-4cfa-9149-6ec4a6b86e14.md` | QA log query at span 49 reports that target identifiers `CVE-2021-42550` and `CVE-2023-6378` remained after the Logback core update, despite tests passing. The attempt QA verdict is `passed=false`, `security_flag`. The root run ended `completed_with_errors` / `failures_detected`; the terminal task later became inconclusive after retrying. This is a rejection case, not a remediation pass. |

## Case-to-trace mapping

| Golden case ID | Label | Provenance and identity | Normalization or synthetic detail |
|---|---|---|---|
| `triage-maven-actionable` | Direct historical | `acc4a7ea-58ce-4140-becb-f26bc5764064`, initial triage span 2; PostgreSQL finding remains an actionable Maven coordinate, `org.postgresql:postgresql`. | Fixture preserves Maven ecosystem, POM path, observed risk context, and unknown reachability. |
| `triage-maven-unknown-reachability` | Synthetic | Based on the PostgreSQL finding in trace `acc4a7ea-58ce-4140-becb-f26bc5764064`. | Explicitly tests that missing Java reachability proof is not converted into an unreachable-code claim. |
| `supervisor-maven-direct-minimum` | Normalized historical | `acc4a7ea-58ce-4140-becb-f26bc5764064`, span 82; task `task-2`, attempt `05bbd58a-fd5a-5726-9a4c-900fc46b4610`, attempt revision 3; `org.postgresql:postgresql` → `42.7.11`. | Expected action and approved candidate pool are independently authored; the current tactical instruction is checked for exact GAV and version. |
| `supervisor-maven-managed-minimum` | Normalized historical | `c2b3de3e-bbb6-4d05-a68d-9db7f65027ce`, span 80; Logback classic task `task-1`, attempt `c019d0c8-6801-5ba1-92c0-8d08485ca05b`, attempt revision 2; target `ch.qos.logback:logback-classic`, version `1.2.13`, declaration type `dependencyManagement`. | Models the selected minimum action from the historical committed batch as a standalone tactical proposal. |
| `supervisor-maven-qualified-version` | Synthetic | No historical version-selection proposal is claimed. | Tests Maven's native ordering for `1.9`, `1.10.Final`, and release candidates while retaining the exact approved token. |
| `supervisor-maven-reject-unapproved` | Synthetic | Uses the historical PostgreSQL target as context; no out-of-pool proposal is claimed as historical. | Scripts an unapproved proposal through the bounded repair loop and requires production verification to reject it before committing an instruction or staged resolution. |
| `update-maven-direct` | Normalized historical | `acc4a7ea-58ce-4140-becb-f26bc5764064`, span 82; task `task-2`, attempt `05bbd58a-fd5a-5726-9a4c-900fc46b4610`, attempt revision 3; PostgreSQL `42.7.11`. | Historical mutation is represented by current `modify_and_validate_maven_dependency`; the expected GAV, declaration, and file effect are separately specified. |
| `update-maven-property-owner` | Synthetic | No property-owner mutation is claimed as historical. | Parent/module POMs test updating the authorized parent property while preserving the child reference and unrelated XML. |
| `update-maven-managed-insertion` | Normalized historical | `33bca3aa-634e-4f80-9778-3408bcfe4d7b`, span 68; task `task-1`, attempt `a55394e4-7d53-57cb-8bd3-7da4c60a2d21`, attempt revision 3; AssertJ `3.27.7`. | Preserves explicit `ensure_dependency_management` authorization and maps the observed managed insertion to the current combined Maven tool. |
| `update-maven-target-recovery` | Synthetic | No wrong-coordinate or unauthorized-POM sequence is claimed as historical. | Rejected calls are checked for unchanged files before the authorized exact-GAV mutation succeeds. |
| `update-maven-sync-rollback` | Synthetic | No Maven synchronization failure is claimed as historical. | Injects a recorded offline Maven failure and verifies byte-identical POM restoration and an unsuccessful worker result. |
| `update-maven-atomic-logback` | Normalized historical | `c2b3de3e-bbb6-4d05-a68d-9db7f65027ce`, span 80; shared cluster `batch-2b4863dea0feede0d032c4d5`; classic task `task-1` / attempt `c019d0c8-6801-5ba1-92c0-8d08485ca05b` / revision 2; core task `task-2` / attempt `846d6236-8829-5619-902f-ddac3cf81288` / revision 2. Both are `1.2.13` under `dependencyManagement`. | Replays the committed atomic batch through the current multi-task state constructor and retains separate per-task attempt provenance. |
| `qa_maven_target_pass` | Normalized historical | `acc4a7ea-58ce-4140-becb-f26bc5764064`, PostgreSQL task QA span 92; task `task-2`, attempt `05bbd58a-fd5a-5726-9a4c-900fc46b4610`, attempt revision 3. The companion jsoup QA was span 147; task `task-1`, attempt `a55394e4-7d53-57cb-8bd3-7da4c60a2d21`, attempt revision 3. Both attempt evaluations passed. | Uses typed current-attempt Maven dependency evidence and task-scoped scan facts; unrelated cumulative findings are retained separately. |
| `qa_maven_compile_failure` | Synthetic | No compiler failure is claimed as historical. | Models a Maven compile error and confirms tests are not represented as having completed. |
| `qa_maven_test_regression` | Synthetic | No Surefire/JUnit regression is claimed as historical. | Models a post-build, post-scan test regression attributed to the Maven task. |
| `qa_maven_target_remaining` | Direct historical | `d91b4065-fe6e-4cfa-9149-6ec4a6b86e14`, span 49; task `task-1`, attempt `c019d0c8-6801-5ba1-92c0-8d08485ca05b`, attempt revision 2; Logback core at `1.2.13` in `dependencyManagement`. At the terminal root state, the task was revision 7, retry count 1, and inconclusive. | Preserves the historical `security_flag` rejection because both target identifiers remained while the tests passed. The extra terminal retries and 24 newly detected identifiers are not hidden. |
| `qa_maven_incomplete_coverage` | Synthetic | No incomplete-coverage outcome is claimed as historical. | Tests inconclusive QA evidence and the existing QA-only rerun path without a remediation retry. |
| `qa_maven_executed_version_mismatch` | Synthetic | No version mismatch is claimed as historical. | Clean scanner output cannot override resolved Maven dependency evidence that differs from the committed attempt version. |

The historical spans justify scenario selection and observed facts. They are not
used as expected-output assertions, and their overall run statuses are not
substituted for the task-level outcomes in the golden cases.
