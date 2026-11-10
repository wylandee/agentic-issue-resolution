"""Live tracking and reconciliation of vulnerability findings during and after remediation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from remediation_engine.contracts.schemas import (
    VulnerabilityGroup,
    VulnerabilityIssue,
    model_or_dict_value,
)

from .models import DashboardUpdate, FindingModel
from .workspace_manager import calculate_severity_counts, finding_occurrence_key, normalize_findings

_SEVERITY_RANK = {"UNKNOWN": 0, "INFO": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4, "CRITICAL": 5}


class FindingsTracker:
    """Maintains active, remediated, and newly detected vulnerability findings."""

    def __init__(self, initial_issues: list[VulnerabilityIssue]) -> None:
        """Seed the tracker from raw baseline issues without mutating them.

        Args:
            initial_issues: Original vulnerability records for the workspace.
        """
        self._findings_by_id: dict[str, FindingModel] = {}

        # Mappings: group_id -> set of finding_ids; identifier -> finding_ids.
        self.group_to_finding_ids: dict[str, set[str]] = {}
        self.identifier_to_finding_ids: dict[str, set[str]] = {}

        for finding in normalize_findings(initial_issues, status="flagged"):
            self._findings_by_id[finding.id] = finding
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
            existing = self._findings_by_id.get(finding_id)
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

        self._findings_by_id[existing.id] = merged
        self._index_finding(merged)
        return merged, True

    @property
    def active_findings(self) -> list[FindingModel]:
        """Return all currently active (unremediated baseline + unresolved new) findings."""
        return sorted(
            (
                finding
                for finding in self._findings_by_id.values()
                if finding.status != "remediated"
            ),
            key=lambda f: (
                0
                if f.severity == "CRITICAL"
                else 1
                if f.severity == "HIGH"
                else 2
                if f.severity == "MEDIUM"
                else 3
                if f.severity == "LOW"
                else 4,
                0 if f.is_new else 1,
                f.package_name,
            ),
        )

    @property
    def remediated_count(self) -> int:
        """Return the number of canonical findings marked remediated."""
        return sum(finding.status == "remediated" for finding in self._findings_by_id.values())

    @property
    def new_count(self) -> int:
        """Return the number of canonical findings discovered after baseline."""
        return sum(finding.is_new for finding in self._findings_by_id.values())

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Return current counters and active list for dashboard."""
        active_findings = self.active_findings
        return {
            "active_findings": active_findings,
            "remediated_count": self.remediated_count,
            "new_count": self.new_count,
            "severity_counts": calculate_severity_counts(active_findings),
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
                        finding = self._findings_by_id.get(finding_id)
                        if (
                            finding is not None
                            and issue_identifiers.intersection(self._finding_identifiers(finding))
                            and finding_occurrence_key(issue) == finding_occurrence_key(finding)
                        ):
                            finding.group_id = group_id
                            group_finding_ids.add(finding_id)

    def mark_group_remediated(
        self,
        group_id: str,
        *,
        remaining_occurrences: set[tuple[str, str, str]],
        unlocalized_remaining_identifiers: set[str],
    ) -> list[DashboardUpdate]:
        """Remove only occurrences cleared by complete task-scoped scan evidence."""
        updates: list[DashboardUpdate] = []
        target_fids = set(self.group_to_finding_ids.get(group_id, set()))

        for fid in target_fids:
            model = self._findings_by_id.get(fid)
            if model is None:
                continue
            remains = finding_occurrence_key(model) in remaining_occurrences or bool(
                self._finding_identifiers(model).intersection(unlocalized_remaining_identifiers)
            )
            if remains:
                if model.status == "remediated":
                    model.status = "unresolved"
                    updates.append(
                        DashboardUpdate(
                            type="finding_updated",
                            finding_id=fid,
                            finding=model,
                            **self.get_dashboard_summary(),
                        )
                    )
                continue
            if model.status == "remediated":
                continue
            model.status = "remediated"
            updates.append(
                DashboardUpdate(
                    type="finding_removed",
                    finding_id=fid,
                    finding=model,
                    **self.get_dashboard_summary(),
                )
            )

        return updates

    @staticmethod
    def _current_qa_scan_evidence(
        task: Any,
        qa_results_by_attempt: Mapping[str, Any],
    ) -> Any | None:
        """Return complete QA scan evidence for a task's current passing attempt."""
        status = model_or_dict_value(task, "status")
        status_value = str(getattr(status, "value", status)).lower()
        task_id = model_or_dict_value(task, "task_id")
        attempt_id = model_or_dict_value(task, "current_attempt_id")
        if status_value != "qa_passed" or not task_id or not attempt_id:
            return None

        qa_result = qa_results_by_attempt.get(attempt_id)
        if (
            qa_result is None
            or model_or_dict_value(qa_result, "attempt_id") != attempt_id
            or model_or_dict_value(qa_result, "task_id") != task_id
            or model_or_dict_value(qa_result, "task_revision")
            != model_or_dict_value(task, "task_revision")
        ):
            return None

        evaluation = model_or_dict_value(qa_result, "evaluation")
        evidence = model_or_dict_value(evaluation, "scan_evidence")
        if (
            evaluation is None
            or model_or_dict_value(evaluation, "passed") is not True
            or evidence is None
            or model_or_dict_value(evidence, "complete") is not True
        ):
            return None
        return evidence

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
                        if self._findings_by_id[candidate_id].status != "remediated"
                    ),
                    ordered_ids[0],
                )
                existing = self._findings_by_id[finding_id]
                was_remediated = existing.status == "remediated"
                merged, changed = self._merge_existing_finding(existing, finding)
                if was_remediated:
                    merged.status = "unresolved"
                    self._findings_by_id[finding_id] = merged
                    changed = True
                if changed:
                    updates.append(
                        DashboardUpdate(
                            type="finding_updated",
                            finding_id=finding_id,
                            finding=merged,
                            **self.get_dashboard_summary(),
                        )
                    )
                continue

            if finding.id in self._findings_by_id:
                continue

            self._findings_by_id[finding.id] = finding
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

    def process_node_state_update(
        self, node_name: str, state_patch: dict[str, Any]
    ) -> list[DashboardUpdate]:
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
        qa_results_by_attempt = state_patch.get("qa_results_by_attempt") or {}
        if isinstance(task_queue, Mapping) and isinstance(qa_results_by_attempt, Mapping):
            for task in task_queue.values():
                group_id = str(model_or_dict_value(task, "parent_group_id") or "")
                evidence = self._current_qa_scan_evidence(task, qa_results_by_attempt)
                if not group_id or evidence is None:
                    continue
                remaining_ids = {
                    str(identifier).strip().upper()
                    for identifier in (
                        model_or_dict_value(evidence, "remaining_target_identifiers", []) or []
                    )
                    if str(identifier).strip()
                }
                remaining_occurrences: set[tuple[str, str, str]] = set()
                observed_remaining_ids: set[str] = set()
                for issue in model_or_dict_value(evidence, "found_issues", []) or []:
                    issue_ids = self._issue_identifiers(issue)
                    matched_ids = issue_ids.intersection(remaining_ids)
                    if not matched_ids:
                        continue
                    occurrence = finding_occurrence_key(issue)
                    if occurrence[0] == "unknown package" or not occurrence[2]:
                        continue
                    remaining_occurrences.add(occurrence)
                    observed_remaining_ids.update(matched_ids)
                unlocalized_ids = remaining_ids - observed_remaining_ids
                updates.extend(
                    self.mark_group_remediated(
                        group_id,
                        remaining_occurrences=remaining_occurrences,
                        unlocalized_remaining_identifiers=unlocalized_ids,
                    )
                )

        # 3. QA Critic / Scan outputs (targeted scan results)
        scan_evidence = state_patch.get("scan_evidence")
        if scan_evidence is not None:
            found_issues = getattr(scan_evidence, "found_issues", None) or []
            new_ids = getattr(scan_evidence, "new_identifiers", None) or []
            if new_ids and found_issues:
                # Add only new issues
                new_issues = [
                    iss
                    for iss in found_issues
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
                iss
                for iss in post_scan_issues
                if (
                    getattr(iss, "cve_id", None)
                    and iss.cve_id.upper() in [x.upper() for x in new_vuln_ids]
                )
                or (
                    getattr(iss, "ghsa_id", None)
                    and iss.ghsa_id.upper() in [x.upper() for x in new_vuln_ids]
                )
            ]
            if new_issues:
                updates.extend(self.add_new_scan_findings(new_issues))

        return updates

    def reconcile_final_state(self, final_raw_state: dict[str, Any]) -> DashboardUpdate:
        """Authoritative final reconciliation against the final orchestrator state."""
        task_queue = final_raw_state.get("task_queue") or {}
        qa_results_by_attempt = final_raw_state.get("qa_results_by_attempt") or {}
        self.process_node_state_update(
            "final_reconciliation",
            {
                "task_queue": task_queue,
                "qa_results_by_attempt": qa_results_by_attempt,
            },
        )

        # Undo optimistic removals when a group's final current attempt did not
        # produce complete, passing QA evidence. Historical passing attempts do
        # not keep findings cleared after a later pivot or failed workaround.
        qa_passed_group_ids: set[str] = set()
        if isinstance(task_queue, Mapping) and isinstance(qa_results_by_attempt, Mapping):
            for task in task_queue.values():
                group_id = str(model_or_dict_value(task, "parent_group_id") or "")
                if group_id and self._current_qa_scan_evidence(task, qa_results_by_attempt):
                    qa_passed_group_ids.add(group_id)

        qa_passed_finding_ids = {
            finding_id
            for group_id in qa_passed_group_ids
            for finding_id in self.group_to_finding_ids.get(group_id, set())
        }
        for group_id, finding_ids in self.group_to_finding_ids.items():
            if group_id in qa_passed_group_ids:
                continue
            for finding_id in finding_ids - qa_passed_finding_ids:
                finding = self._findings_by_id.get(finding_id)
                if finding is not None and finding.status == "remediated":
                    finding.status = "unresolved"

        final_scan = final_raw_state.get("final_full_scan_result")
        if final_scan is not None and model_or_dict_value(final_scan, "completed") is True:
            remaining_ids = {
                str(identifier).strip().upper()
                for identifier in (
                    model_or_dict_value(final_scan, "remaining_target_identifiers", []) or []
                )
                if str(identifier).strip()
            }
            new_ids = {
                str(identifier).strip().upper()
                for identifier in (model_or_dict_value(final_scan, "new_identifiers", []) or [])
                if str(identifier).strip()
            }
            found_issues = list(model_or_dict_value(final_scan, "found_issues", []) or [])
            relevant_issues = [
                issue
                for issue in found_issues
                if self._issue_identifiers(issue).intersection(remaining_ids | new_ids)
            ]
            self.add_new_scan_findings(relevant_issues)

            observed_remaining_ids: set[str] = set()
            for issue in relevant_issues:
                issue_ids = self._issue_identifiers(issue).intersection(remaining_ids)
                occurrence = finding_occurrence_key(issue)
                if issue_ids and (occurrence[0] == "unknown package" or not occurrence[2]):
                    for finding in self._findings_by_id.values():
                        if finding.status == "remediated" and self._finding_identifiers(
                            finding
                        ).intersection(issue_ids):
                            finding.status = "unresolved"
                observed_remaining_ids.update(issue_ids)
            unlocalized_remaining_ids = remaining_ids - observed_remaining_ids
            for finding in self._findings_by_id.values():
                if finding.status == "remediated" and self._finding_identifiers(
                    finding
                ).intersection(unlocalized_remaining_ids):
                    finding.status = "unresolved"
        elif final_scan is None:
            # Older results may expose only aggregate remaining identifiers.
            # Preserve that conservative signal when no typed terminal scan exists.
            remaining_ids = {
                str(identifier).strip().upper()
                for identifier in (final_raw_state.get("remaining_target_identifiers") or [])
                if str(identifier).strip()
            }
            for finding in self._findings_by_id.values():
                if finding.status == "remediated" and self._finding_identifiers(
                    finding
                ).intersection(remaining_ids):
                    finding.status = "unresolved"

        # Any remaining baseline findings that failed QA stay visible as unresolved.
        for model in self._findings_by_id.values():
            if not model.is_new and model.status != "remediated":
                model.status = "unresolved"

        return DashboardUpdate(
            type="full_refresh",
            **self.get_dashboard_summary(),
        )
