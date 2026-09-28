# PyGoat Python remediation example

This example separates offline ingestion/runner-contract tests from a live
remediation run. The checked-in `synthetic-pypi-dependency-check-report.json`
is a small parser fixture only; it is not represented as a scan of PyGoat. A
real `dependency-check-report-baseline.json` and its canonical
`baseline_issues.jsonl` are generated from the user's PyGoat checkout before a
live run.

## Generate a PyGoat baseline

Run these commands from the repository root. The scanner reads PyGoat as a
read-only mount, uses the existing `odc-cache` volume, and writes its report
outside the host clone. Docker needs access to Dependency-Check's database
update source during the scan.

```bash
git clone https://github.com/adeyosemanputra/pygoat.git data/clones/pygoat
mkdir -p /tmp/pygoat-odc
docker run --rm -u root \
  -v "$PWD/data/clones/pygoat:/scan:ro" \
  -v /tmp/pygoat-odc:/out \
  -v odc-cache:/usr/share/dependency-check/data \
  owasp/dependency-check:latest \
  --project pygoat --scan /scan --format JSON --format HTML \
  --out /out --enableExperimental
cp /tmp/pygoat-odc/dependency-check-report.json \
  examples/pygoat/fixtures/dependency-check-report-baseline.json
remedy ingest examples/pygoat/fixtures/dependency-check-report-baseline.json \
  --format odc-json --output examples/pygoat/fixtures/baseline_issues.jsonl
```

Before remediation, confirm that the report produced at least one PyPI finding:

```bash
python -c "import json; rows=[json.loads(line) for line in open('examples/pygoat/fixtures/baseline_issues.jsonl', encoding='utf-8') if line.strip()]; assert any(row.get('ecosystem') == 'pypi' for row in rows)"
```

If the live report has no PyPI findings, do not relabel the synthetic parser
fixture as a PyGoat scan or use it as a claimed PyGoat baseline. The synthetic
fixture is exercised by the offline pytest contract test instead.

## Run live remediation

The runner uses the generated canonical baseline, sets
`SystemContext.primary_language` to `python`, and writes the typed result and
proposed unified patch outside the host clone. First verify that the virtual
environment imports this checkout, not another editable installation:

```bash
.venv/bin/python -c 'import remediation_engine; print(remediation_engine.__file__)'
```

The printed path should be under this checkout's `src/remediation_engine`.
Then run the live example with the same environment:

```bash
.venv/bin/python examples/pygoat/run.py \
  --repo "$PWD/data/clones/pygoat" \
  --issues examples/pygoat/fixtures/baseline_issues.jsonl \
  --output /tmp/pygoat-result.json \
  --patch-out /tmp/pygoat.patch
```

A live run requires Docker, `OPENAI_API_KEY`, and network access needed for
PyPI metadata and Dependency-Check. Python execution uses `python:3.11-slim`.
The engine edits only its isolated workspace; it never applies the proposed
patch to the host clone. Review `/tmp/pygoat.patch` before applying anything.
The default pytest suite does not clone PyGoat, invoke Docker, contact PyPI, or
call an LLM.

## Offline fixture test

The test ingests the separately named synthetic PyPI report into canonical
JSONL and exercises the runner's typed request/result/patch contract with
`run_remediation` mocked:

```bash
.venv/bin/python -m pytest -q tests/test_pygoat_ingestion_example.py
```
