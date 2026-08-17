# Shared dependencies fixture

This pre-triaged fixture contains two update tasks whose npm dependency
closures overlap:

| Component | Installed version | Findings | Intended coverage |
|---|---:|---|---|
| `express-jwt` | `0.1.3` | `CVE-2020-15084` | Parent package closure |
| `jsonwebtoken` | `0.1.0` | `CVE-2022-23539`, `CVE-2022-23540`, `CVE-2022-23541` | Nested child package closure |

The selected `jsonwebtoken@0.1.0` node is nested beneath `express-jwt` in the
Juice Shop lockfile. Both task closures therefore include that node and its
shared `jws` dependency. The fixture is intended to exercise closure union,
duplicate lockfile-key preservation, and task-to-evidence mapping when two
active tasks share dependencies.

The fixture contains two vulnerability groups and five canonical baseline
issue records. It uses the corresponding entries extracted from
`fixtures/baseline_issues.jsonl` and `fixtures/triaged_groups_baseline.json`.
The triaged groups' repository manifest fields are normalized to
`package.json`; the original Dependency-Check paths remain only in issue
evidence fields.

Run it with:

```text
python examples/juice_shop/fixtures/shared_dependencies/run_shared_dependencies.py
```

The runner persists the result and patch under `data/trajectories/`. A run is
considered operationally complete when the graph reaches `completed` or
`completed_with_errors`; individual remediation and QA outcomes remain in the
result and trajectory.
