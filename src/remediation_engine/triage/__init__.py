"""
Issue triage, enrichment, grouping, and reachability analysis.

Public API
----------
grouper.group_issues          â€” deterministic grouping of VulnerabilityIssues
enrichment.enrich_cves        â€” EPSS + CISA KEV enrichment for CVE IDs
agent.run_triage              â€” deterministic (+ optional LLM) triage verdict
pipeline.run_triage_pipeline  â€” full triage flow (group â†’ enrich â†’ triage)
pipeline.select_issues_for_remediation â€” pick one issue per valid group
"""
