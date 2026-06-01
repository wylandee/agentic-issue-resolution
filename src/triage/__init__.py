"""
src/triage — Phase 4.0 Triage & Enrichment Layer.

Public API
----------
grouper.group_issues          — deterministic grouping of VulnerabilityIssues
enrichment.enrich_cves        — EPSS + CISA KEV enrichment for CVE IDs
agent.run_triage              — deterministic (+ optional LLM) triage verdict
pipeline.run_triage_pipeline  — full triage flow (group → enrich → triage)
pipeline.select_issues_for_remediation — pick one issue per valid group
"""
