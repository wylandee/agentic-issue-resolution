export type SeverityLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "INFO" | "UNKNOWN";

export type FindingStatus = "flagged" | "remediating" | "remediated" | "newly_detected" | "unresolved";

export interface FindingModel {
  id: string;
  vuln_id: string;
  package_name: string;
  package_version?: string | null;
  severity: SeverityLevel;
  file_path?: string | null;
  aliases?: string[] | null;
  cwes: string[];
  description?: string | null;
  status: FindingStatus;
  is_new: boolean;
  group_id?: string | null;
  task_id?: string | null;
}

export interface WorkspaceInfo {
  workspace_id: string;
  repo_name: string;
  repo_root: string;
  odc_report_path?: string | null;
  findings: FindingModel[];
  total_findings: number;
  severity_counts: Record<string, number>;
  upload_skipped_count?: number;
  upload_skipped_files?: string[];
}

export type ConsoleCategory = "NODE_TRANSITION" | "TOOL_CALL" | "DETERMINISTIC_FN" | "KEY_RESULT";

export interface ConsoleEvent {
  id: string;
  timestamp: string;
  category: ConsoleCategory;
  title: string;
  details?: string | null;
  status?: "running" | "success" | "error" | "incomplete" | null;
  duration_seconds?: number | null;
  outcome?: string | null;
  task_label?: string | null;
  data?: Record<string, unknown> | null;
}

export interface DashboardUpdate {
  type: "finding_removed" | "finding_added" | "finding_status_change" | "finding_updated" | "full_refresh";
  finding_id?: string | null;
  finding?: FindingModel | null;
  active_findings: FindingModel[];
  remediated_count: number;
  new_count: number;
  severity_counts: Record<string, number>;
}

export type RunStatus = "pending" | "running" | "completed" | "completed_with_errors" | "failed";

export interface RunState {
  run_id: string;
  workspace_id: string;
  status: RunStatus;
  current_node?: string | null;
  active_findings: FindingModel[];
  remediated_count: number;
  new_count: number;
  logs: ConsoleEvent[];
  report_available: boolean;
  report_path?: string | null;
  diff?: string | null;
  errors: string[];
}

export interface ReportResponse {
  report_markdown: string;
  report_path?: string | null;
  diff: string;
  changed_files: string[];
  status: string;
}
