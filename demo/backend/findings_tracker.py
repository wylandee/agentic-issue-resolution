"""Live tracking and reconciliation of vulnerability findings during and after remediation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from remediation_engine.contracts.schemas import VulnerabilityGroup, VulnerabilityIssue

from .models import DashboardUpdate, FindingModel
from .workspace_manager import calculate_severity_counts, finding_occurrence_key, normalize_findings

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"UNKNOWN": 0, "INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}



class FindingsTracker:
    """Maintains active, remediated, and newly detected vulnerability findings."""

    def __init__(self, initial_issues: list[VulnerabilityIssue]) -> None:
        """Seed the tracker from raw baseline issues without mutating them.

        Args:
            initial_issues: Original vulnerability records for the workspace.
        """
        self.initial_issues = list(initial_issues)
        # Stable finding ID -> canonical FindingModel.
        self.findings_by_id: dict[str, FindingModel] = {}
        self.remediated_findings_by_id: dict[str, FindingModel] = {}
        self.new_findings_by_id: dict[str, FindingModel] = {}

        # Mappings: group_id -> set of finding_ids; identifier -> finding_ids.
        self.group_to_finding_ids: dict[str, set[str]] = {}
        self.identifier_to_finding_ids: dict[str, set[str]] = {}

        for finding in normalize_findings(initial_issues, status="flagged"):
            self.findings_by_id[finding.id] = finding
            self._index_finding(finding)

    @staticmethod
    def _finding_identifiers(finding: FindingModel) -> set[str]:
        """Return the normalized primary and alias identifiers for a row."""
        return {
            identifier.strip().upper()
            for identifier in [finding.vuln_id, *finding.aliases]
            if identifier and identifier.strip()
        }

    @staticmethod
    def _issue_identifiers(issue: VulnerabilityIssue) -> set[str]:
        """Return CVE/GHSA identifiers, or the issue's fallback identity."""
        identifiers = {
            identifier.strip().upper()
            for identifier in [issue.cve_id, issue.ghsa_id]
            if identifier and identifier.strip()
        }
        if identifiers:
            return identifiers
        fallback = issue.rule_id or str(issue.id)
        return {fallback.strip().upper()}

    def _index_finding(self, finding: FindingModel) -> None:
        """Index every identifier retained by a canonical row."""
        for identifier in self._finding_identifiers(finding):
            self.identifier_to_finding_ids.setdefault(identifier, set()).add(finding.id)

    def _matching_finding_ids(self, finding: FindingModel) -> set[str]:
        """Find active or remediated rows for the same identifier and occurrence."""
        occurrence = finding_occurrence_key(finding)
        candidate_ids: set[str] = set()
        for identifier in self._finding_identifiers(finding):
            candidate_ids.update(self.identifier_to_finding_ids.get(identifier, set()))
        matches: set[str] = set()
        for finding_id in candidate_ids:
            existing = self.findings_by_id.get(finding_id)
            if existing is None:
                existing = self.remediated_findings_by_id.get(finding_id)
            if existing is not None and finding_occurrence_key(existing) == occurrence:
                matches.add(finding_id)
        return matches

    def _merge_existing_finding(
        self,
        existing: FindingModel,
        incoming: FindingModel,
    ) -> tuple[FindingModel, bool]:
        """Merge linked identifiers and useful metadata into an existing row."""
        identifiers = self._finding_identifiers(existing) | self._finding_identifiers(incoming)
        cve_ids = sorted(identifier for identifier in identifiers if identifier.startswith("CVE-"))
        ghsa_ids = sorted(
            identifier for identifier in identifiers if identifier.startswith("GHSA-")
        )
        primary_id = cve_ids[0] if cve_ids else ghsa_ids[0] if ghsa_ids else existing.vuln_id
        aliases = sorted(identifiers - {primary_id.upper()})
        severity = max(
            (existing.severity, incoming.severity),
            key=lambda value: _SEVERITY_RANK.get(value.upper(), 0),
        )
        merged = existing.model_copy(
            update={
                "vuln_id": primary_id,
                "aliases": aliases,
                "severity": severity,
                "cwes": sorted(set(existing.cwes) | set(incoming.cwes)),
                "description": existing.description or incoming.description,
            }
        )
        if merged == existing:
            return existing, False

        if existing.id in self.findings_by_id:
            self.findings_by_id[existing.id] = merged
        if existing.id in self.remediated_findings_by_id:
            self.remediated_findings_by_id[existing.id] = merged
        if existing.id in self.new_findings_by_id:
            self.new_findings_by_id[existing.id] = merged
        self._index_finding(merged)
        return merged, True

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
        """Return the number of canonical findings marked remediated."""
        return len(self.remediated_findings_by_id)

    @property
    def new_count(self) -> int:
        """Return the number of canonical findings discovered after baseline."""
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
        """Map triage groups to matching canonical findings."""
        for group in groups:
            group_id = group.group_id
            group_finding_ids = self.group_to_finding_ids.setdefault(group_id, set())

            for issue in group.issues or []:
                issue_identifiers = self._issue_identifiers(issue)
                for identifier in issue_identifiers:
                    candidate_ids = self.identifier_to_finding_ids.get(identifier, set())
                    for finding_id in candidate_ids:
                        finding = self.findings_by_id.get(finding_id)
                        if (
                            finding is not None
                            and issue_identifiers.intersection(self._finding_identifiers(finding))
                            and finding_occurrence_key(issue) == finding_occurrence_key(finding)
                        ):
                            finding.group_id = group_id
                            group_finding_ids.add(finding_id)

            component = (group.vulnerable_component or "").strip().casefold()
            group_identifiers = {
                identifier.strip().upper()
                for identifier in (group.cve_ids or []) + (group.ghsa_ids or [])
                if identifier and identifier.strip()
            }
            for finding_id, finding in self.findings_by_id.items():
                component_matches = bool(
                    component and finding.package_name.strip().casefold() == component
                )
                identifier_matches = bool(
                    group_identifiers.intersection(self._finding_identifiers(finding))
                )
                if component_matches or identifier_matches:
                    finding.group_id = group_id
                    group_finding_ids.add(finding_id)

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
            # Keep a row active if any of its linked identifiers remains.
            if not self._finding_identifiers(model).intersection(remaining_set):
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
        """Merge linked scan results or add previously unseen canonical findings."""
        updates: list[DashboardUpdate] = []

        for finding in normalize_findings(issues, status="newly_detected", is_new=True):
            matching_ids = self._matching_finding_ids(finding)
            if matching_ids:
                ordered_ids = sorted(matching_ids)
                finding_id = next(
                    (
                        candidate_id
                        for candidate_id in ordered_ids
                        if candidate_id in self.findings_by_id
                    ),
                    ordered_ids[0],
                )
                existing = self.findings_by_id.get(finding_id)
                if existing is None:
                    existing = self.remediated_findings_by_id[finding_id]
                merged, changed = self._merge_existing_finding(existing, finding)
                if changed and finding_id in self.findings_by_id:
                    updates.append(
                        DashboardUpdate(
                            type="finding_updated",
                            finding_id=finding_id,
                            finding=merged,
                            **self.get_dashboard_summary(),
                        )
                    )
                continue

            if finding.id in self.findings_by_id or finding.id in self.remediated_findings_by_id:
                continue

            self.findings_by_id[finding.id] = finding
            self.new_findings_by_id[finding.id] = finding
            self._index_finding(finding)
            updates.append(
                DashboardUpdate(
                    type="finding_added",
                    finding_id=finding.id,
                    finding=finding,
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
            for _task_id, task in task_queue.items():
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
                    if not self._finding_identifiers(model).intersection(remaining_target_ids):
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
