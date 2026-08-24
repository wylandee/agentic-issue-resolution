# NodeGoat Remediation Example

This directory contains a maintained end-to-end example for remediating OWASP
NodeGoat findings with `remediation_engine`.

All commands below assume they are run from the repository root. The runner
resolves the default NodeGoat clone and explicit `--repo` values to absolute
paths before constructing the public API request. The host clone is never
modified by the engine; execution happens in isolated Docker volumes.

## Prerequisites

1. Clone NodeGoat into `data/clones/nodegoat`:

   ```bash
   git clone https://github.com/OWASP/NodeGoat.git data/clones/nodegoat
   ```

2. Configure environment variables in `.env` (Docker and an
   `OPENAI_API_KEY` are required for LLM-backed workers).

The raw baseline reports in this directory are retained as scan provenance.
The runnable fixture intentionally uses five representative findings so the
example remains practical to inspect and replay.

## Example Scenarios

### Whole-pipeline execution

Runs the task-queue workflow against the five-finding suppressed fixture. The
runner performs initial triage before Supervisor routing, worker execution, and
QA evaluation:

```bash
python examples/NodeGoat/run.py
```

`run.py` accepts any canonical JSONL issue fixture through `--issues`, so a
larger normalized report can be supplied without changing the runner:

```bash
remedy ingest examples/NodeGoat/dependency-check-report-baseline.json \
  --format odc-json --output /tmp/nodegoat-issues.jsonl
python examples/NodeGoat/run.py --issues /tmp/nodegoat-issues.jsonl
```

The output result and unified patch default to:

```text
data/trajectories/nodegoat-result.json
data/trajectories/nodegoat.patch
```

## Suppressed Fixture

`fixtures/suppressed/odc_suppressed_issues.jsonl` contains one canonical ODC
record for each of five distinct NodeGoat packages. The selection does not
overlap with the Juice Shop suppressed package set (`@tootallnate/once`,
`express-jwt`, `got`, `notevil`, or `sanitize-html`).

| Package | Version | Advisory | Severity | Representative risk |
| --- | ---: | --- | --- | --- |
| `growl` | `1.9.2` | `CVE-2017-16042` / `GHSA-QH2H-CHJ9-JFFQ` | Critical | Command injection through unsanitized notification input |
| `mongodb` | `2.2.36` | `CVE-2021-32036` | High | Resource exhaustion through repeated feature requests |
| `marked` | `0.3.5` | `CVE-2017-1000427` | Medium | XSS through the `data:` URI parser |
| `adm-zip` | `0.4.4` | `CVE-2018-1002204` | Medium | Zip archive path traversal and arbitrary file write |
| `ini` | `1.3.4` | `CVE-2020-7788` | Critical | Prototype pollution while parsing attacker-controlled INI data |

`suppressions.xml` records the package-level suppression scope associated with
the selected subset. The JSONL file retains the source findings so the public
remediation API can triage and process them.

