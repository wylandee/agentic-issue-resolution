import os
import logging
from datetime import datetime, timezone
from src.contracts.schemas import VulnerabilityIssue, IssueType, IssueSource, Severity, SystemContext, VulnerabilityGroup, CVEEnrichment
from src.triage.agent import run_triage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def create_mock_group(group_id, component, cve, severity, desc, epss, in_kev, is_reachable=None):
    """Helper to mock a fully formed VulnerabilityGroup with enrichment."""
    issue = VulnerabilityIssue(
        issue_type=IssueType.SCA,
        source=IssueSource.ODC,
        file_path=f"src/package.json?/{component}:1.0.0",
        package_name=component,
        package_version="1.0.0",
        cve_id=cve,
        severity=severity,
        purl=f"pkg:npm/{component}@1.0.0",
        description=desc
    )
    
    group = VulnerabilityGroup(
        group_id=f"sca:{component}",
        issue_type=IssueType.SCA,
        vulnerable_component=component,
        file_path=issue.file_path,
        cve_ids=[cve],
        versions=["1.0.0"],
        sources=[IssueSource.ODC],
        representative_issue_id=issue.id,
        issues=[issue],
        is_reachable=is_reachable
    )
    
    group.enrichment = CVEEnrichment(
        cve_id=cve,
        epss=epss,
        epss_percentile=epss,
        in_kev=in_kev,
        kev_date_added="2024-01-01" if in_kev else None,
        enriched_at=datetime.now(timezone.utc),
        enrichment_source="mock"
    )
    return group

def run_rbvm_test():
    print("==================================================")
    print("🛡️  STARTING RBVM & GUARDRAILS TEST")
    print("==================================================")

    # Context: Internal Production App
    context = SystemContext(
        public_facing=False,          # INTERNAL APP
        deployment_os="linux",
        deployment_architecture="containerized",
        environment="production",
        primary_language="javascript/nodejs"
    )

    test_cases = [
        # TEST 1: The "Drop Everything" KEV Threat
        # Original: HIGH. KEV: YES. Expected: CRITICAL (Guardrail Override)
        create_mock_group("test-1", "jsonwebtoken", "CVE-2015-9235", Severity.HIGH, 
            "Authentication bypass vulnerability.", epss=0.05, in_kev=True, is_reachable=True),
        
        # TEST 2: The "Imminent Threat" High EPSS
        # Original: LOW. EPSS: 0.85 (Massive). Public-Facing: False. Expected: HIGH (Guardrail Override)
        create_mock_group("test-2", "express", "CVE-2024-11111", Severity.LOW, 
            "Denial of service in body parser.", epss=0.85, in_kev=False, is_reachable=True),

        # TEST 3: The "EPSS Floor" Safe Downgrade
        # Original: CRITICAL. EPSS: 0.001. Public-Facing: False. Expected: MEDIUM (Guardrail Override)
        create_mock_group("test-3", "lodash", "CVE-2019-10744", Severity.CRITICAL, 
            "Prototype pollution allowing code execution.", epss=0.001, in_kev=False, is_reachable=True),

        # TEST 4: The Dead Code False Positive
        # Reachability: FALSE. Expected: False Positive (is_valid=False), Validity Conf = 1.0
        create_mock_group("test-4", "grunt-cli", "CVE-2017-16058", Severity.HIGH, 
            "Malicious module hijacks env variables.", epss=0.05, in_kev=False, is_reachable=False),
            
        # TEST 5: The Typosquatting / Mistaken Identity
        # Component: my-safe-lib. Desc: Exploit in "mysafelib" (no hyphens). Expected: FP (Rule A)
        create_mock_group("test-5", "my-safe-lib", "CVE-2024-22222", Severity.HIGH, 
            "Malicious package 'mysafelib' steals tokens.", epss=0.1, in_kev=False, is_reachable=True)
    ]

    print(f"⚙️  Context: Internal (Public=False), Linux, Production")
    print("⚙️  Processing 5 Targeted Scenarios...\n")

    for i, group in enumerate(test_cases, 1):
        print("="*60)
        print(f"🧪 TEST CASE {i}: {group.vulnerable_component}")
        print("="*60)
        
        result = run_triage(group, context)
        
        status_icon = "✅ VALID" if result.is_valid else "❌ FALSE POSITIVE"
        
        print(f"Status       : {status_icon}")
        if not result.is_valid:
            print(f"FP Reason    : {result.false_positive_reason}")
            print(f"Validity Conf: {result.validity_confidence_score} / 1.0")
        else:
            print(f"Priority     : {result.revised_priority.name}")
            print(f"Priority Conf: {result.priority_confidence_score} / 1.0")
            print(f"Reasoning    : {result.priority_reasoning}")
            
        print(f"\n🧠 Chain of Thought Extract:\n{result.chain_of_thought}\n")

if __name__ == "__main__":
    os.environ["TRIAGE_LLM_ENABLED"] = "true"
    os.environ["TRIAGE_LLM_MODEL"] = "gpt-4o-mini"
    
    run_rbvm_test()