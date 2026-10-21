"""Live tracking and reconciliation of vulnerability findings during and after remediation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from remediation_engine.contracts.schemas import TaskStatus, VulnerabilityGroup, VulnerabilityIssue
from remediation_engine.orchestration.issue_identity import stable_issue_fingerprint

from .models import DashboardUpdate, FindingModel
from .workspace_manager import calculate_severity_counts, issue_to_finding_model

logger = logging.getLogger(__name__)


class FindingsTracker:
    """Maintains active, remediated, and newly detected vulnerability findings."""

    def __init__(self, initial_issues: list[VulnerabilityIssue]) -> None:
        self.initial_issues = list(initial_issues)
        # Stable fingerprint/ID -> FindingModel
        self.findings_by_id: dict[str, FindingModel] = {}
        self.remediated_findings_by_id: dict[str, FindingModel] = {}
        self.new_findings_by_id: dict[str, FindingModel] = {}

        # Mappings: group_id -> set of finding_ids
        self.group_to_finding_ids: dict[str, set[str]] = {}
        # identifier (CVE/GHSA upper) -> set of finding_ids
        self.identifier_to_finding_ids: dict[str, set[str]] = {}
        # package_name (lower) -> set of finding_ids
        self.package_to_finding_ids: dict[str, set[str]] = {}

        # Seed initial findings
        for issue in initial_issues:
            model = issue_to_finding_model(issue, status="flagged", is_new=False)
            self.findings_by_id[model.id] = model
            
            # Map identifiers
            if issue.cve_id:
                self.identifier_to_finding_ids.setdefault(issue.cve_id.upper().strip(), set()).add(model.id)
            if issue.ghsa_id:
                self.identifier_to_finding_ids.setdefault(issue.ghsa_id.upper().strip(), set()).add(model.id)
            if issue.package_name:
                self.package_to_finding_ids.setdefault(issue.package_name.lower().strip(), set()).add(model.id)

    @property
    def active_findings(self) -> list[FindingModel]:
        """Return all currently active (unremediated baseline + unresolved new) findings."""
        return sorted(
            self.findings_by_id.values(),
            key=lambda f: (
                0 if f.severity == "CRITICAL" else
                1 if f.severity == "HIGH" else
                2 if f.severity == "MEDIUM" else
                3 if f.severity == "LOW" else 4,
                0 if f.is_new else 1,
                f.package_name,
            ),
        )

    @property
    def remediated_count(self) -> int:
        return len(self.remediated_findings_by_id)

    @property
    def new_count(self) -> int:
        return len(self.new_findings_by_id)

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Return current counters and active list for dashboard."""
        return {
            "active_findings": self.active_findings,
            "remediated_count": self.remediated_count,
            "new_count": self.new_count,
            "severity_counts": calculate_severity_counts(self.active_findings),
        }

    def map_triaged_groups(self, groups: list[VulnerabilityGroup]) -> None:
        """Map triage groups to member findings."""
        for group in groups:
            group_id = group.group_id
            self.group_to_finding_ids.setdefault(group_id, set())

            # Map member issues
            for issue in group.issues or []:
                f_id = issue_to_finding_model(issue).id
                if f_id in self.findings_by_id:
                    self.findings_by_id[f_id].group_id = group_id
                    self.group_to_finding_ids[group_id].add(f_id)

            # Also match by component + cve/ghsa
            comp = (group.vulnerable_component or "").lower().strip()
            cve_ghsa = {cid.upper().strip() for cid in (group.cve_ids or []) + (group.ghsa_ids or [])}
            for fid, model in self.findings_by_id.items():
                if model.package_name.lower() == comp or any(cid in cve_ghsa for cid in [model.vuln_id]):
                    model.group_id = group_id
                    self.group_to_finding_ids[group_id].add(fid)

    def mark_task_in_progress(self, task_id: str, group_id: str) -> list[DashboardUpdate]:
        """Mark findings associated with an active task as currently remediating."""
        updates: list[DashboardUpdate] = []
        target_fids = self.group_to_finding_ids.get(group_id, set())
        for fid in target_fids:
            if fid in self.findings_by_id and self.findings_by_id[fid].status == "flagged":
                self.findings_by_id[fid].status = "remediating"
                self.findings_by_id[fid].task_id = task_id
                updates.append(
                    DashboardUpdate(
                        type="finding_status_change",
                        finding_id=fid,
                        finding=self.findings_by_id[fid],
                        **self.get_dashboard_summary(),
                    )
                )
        return updates

    def mark_group_remediated(self, group_id: str, remaining_target_identifiers: list[str] | None = None) -> list[DashboardUpdate]:
        """Remove remediated findings for a passed group."""
        updates: list[DashboardUpdate] = []
        target_fids = set(self.group_to_finding_ids.get(group_id, set()))
        remaining_set = {rid.upper().strip() for rid in (remaining_target_identifiers or [])}

        for fid in target_fids:
            if fid not in self.findings_by_id:
                continue
            model = self.findings_by_id[fid]
            # If this finding's vuln_id is not in remaining_target_identifiers, it is cleared
            if model.vuln_id.upper() not in remaining_set:
                model.status = "remediated"
                self.remediated_findings_by_id[fid] = model
                del self.findings_by_id[fid]
                updates.append(
                    DashboardUpdate(
                        type="finding_removed",
                        finding_id=fid,
                        finding=model,
                        **self.get_dashboard_summary(),
                    )
                )

        return updates

    def add_new_scan_findings(self, issues: list[VulnerabilityIssue]) -> list[DashboardUpdate]:
        """Add newly discovered findings from targeted or full scans."""
        updates: list[DashboardUpdate] = []
        known_identifiers = {
            m.vuln_id.upper() for m in list(self.findings_by_id.values()) + list(self.remediated_findings_by_id.values())
        }

        for issue in issues:
            model = issue_to_finding_model(issue, status="newly_detected", is_new=True)
            # Avoid duplicating an existing finding
            if model.id in self.findings_by_id or model.id in self.remediated_findings_by_id:
                continue
            if model.vuln_id.upper() in known_identifiers:
                # Same CVE/GHSA on same package
                matching = [f for f in self.findings_by_id.values() if f.vuln_id == model.vuln_id and f.package_name == model.package_name]
                if matching:
                    continue

            self.findings_by_id[model.id] = model
            self.new_findings_by_id[model.id] = model
            updates.append(
                DashboardUpdate(
                    type="finding_added",
                    finding_id=model.id,
                    finding=model,
                    **self.get_dashboard_summary(),
                )
            )

        return updates

    def process_node_state_update(self, node_name: str, state_patch: dict[str, Any]) -> list[DashboardUpdate]:
        """Inspect a LangGraph node's returned state dictionary and emit dashboard updates."""
        updates: list[DashboardUpdate] = []

        if not isinstance(state_patch, Mapping):
            return updates

        # 1. Triage node updates
        if node_name in {"initial_triage", "triage"} and "valid_groups" in state_patch:
            groups = state_patch.get("valid_groups") or []
            if isinstance(groups, list):
                self.map_triaged_groups(groups)

        # 2. Supervisor task updates
        task_queue = state_patch.get("task_queue") or {}
        if isinstance(task_queue, Mapping):
            for task_id, task in task_queue.items():
                status = getattr(task, "status", None)
                status_val = str(getattr(status, "value", status)).lower()
                parent_group_id = getattr(task, "parent_group_id", "")
                
                if status_val == "qa_passed" and parent_group_id:
                    cleared_updates = self.mark_group_remediated(parent_group_id)
                    updates.extend(cleared_updates)

        # 3. QA Critic / Scan outputs (targeted scan results)
        scan_evidence = state_patch.get("scan_evidence")
        if scan_evidence is not None:
            found_issues = getattr(scan_evidence, "found_issues", None) or []
            new_ids = getattr(scan_evidence, "new_identifiers", None) or []
            if new_ids and found_issues:
                # Add only new issues
                new_issues = [
                    iss for iss in found_issues
                    if (iss.cve_id and iss.cve_id.upper() in [x.upper() for x in new_ids])
                    or (iss.ghsa_id and iss.ghsa_id.upper() in [x.upper() for x in new_ids])
                ]
                if new_issues:
                    updates.extend(self.add_new_scan_findings(new_issues))

        # 4. Supervisor post_remediation_scan_issues / new_vulnerability_identifiers
        post_scan_issues = state_patch.get("post_remediation_scan_issues") or []
        new_vuln_ids = state_patch.get("new_vulnerability_identifiers") or []
        if post_scan_issues and new_vuln_ids:
            new_issues = [
                iss for iss in post_scan_issues
                if (getattr(iss, "cve_id", None) and iss.cve_id.upper() in [x.upper() for x in new_vuln_ids])
                or (getattr(iss, "ghsa_id", None) and iss.ghsa_id.upper() in [x.upper() for x in new_vuln_ids])
            ]
            if new_issues:
                updates.extend(self.add_new_scan_findings(new_issues))

        return updates

    def reconcile_final_state(self, final_raw_state: dict[str, Any]) -> DashboardUpdate:
        """Authoritative final reconciliation against the final orchestrator state."""
        task_queue = final_raw_state.get("task_queue") or {}
        remaining_target_ids = {
            tid.upper().strip() for tid in final_raw_state.get("remaining_target_identifiers", []) or []
        }
        
        # 1. Identify all QA passed groups
        qa_passed_group_ids: set[str] = set()
        if isinstance(task_queue, Mapping):
            for task in task_queue.values():
                status = getattr(task, "status", None)
                status_val = str(getattr(status, "value", status)).lower()
                parent_group_id = getattr(task, "parent_group_id", None)
                if status_val == "qa_passed" and parent_group_id:
                    qa_passed_group_ids.add(parent_group_id)

        # 2. Mark findings whose group passed QA and are not in remaining_target_identifiers
        for group_id in qa_passed_group_ids:
            target_fids = set(self.group_to_finding_ids.get(group_id, set()))
            for fid in target_fids:
                if fid in self.findings_by_id:
                    model = self.findings_by_id[fid]
                    if model.vuln_id.upper() not in remaining_target_ids:
                        model.status = "remediated"
                        self.remediated_findings_by_id[fid] = model
                        del self.findings_by_id[fid]

        # 3. Any remaining findings that failed QA are marked 'unresolved'
        for model in self.findings_by_id.values():
            if not model.is_new and model.status != "remediated":
                model.status = "unresolved"

        # 4. Check for any final newly detected issues
        final_scan = final_raw_state.get("final_full_scan_result")
        if final_scan is not None:
            found_issues = getattr(final_scan, "found_issues", None) or []
            new_ids = getattr(final_scan, "new_identifiers", None) or []
            if new_ids and found_issues:
                new_issues = [
                    iss for iss in found_issues
                    if (getattr(iss, "cve_id", None) and iss.cve_id.upper() in [x.upper() for x in new_ids])
                    or (getattr(iss, "ghsa_id", None) and iss.ghsa_id.upper() in [x.upper() for x in new_ids])
                ]
                self.add_new_scan_findings(new_issues)

        return DashboardUpdate(
            type="full_refresh",
            **self.get_dashboard_summary(),
        )
