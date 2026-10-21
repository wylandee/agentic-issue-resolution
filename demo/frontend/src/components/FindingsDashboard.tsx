import React, { useState } from "react";
import { FindingModel, SeverityLevel, WorkspaceInfo } from "../types";
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Search,
  Filter,
  CheckCircle2,
  RefreshCw,
  Sparkles,
} from "lucide-react";

interface FindingsDashboardProps {
  findings: FindingModel[];
  remediatedCount: number;
  newCount: number;
  isRunActive: boolean;
  attachedWorkspace?: WorkspaceInfo | null;
}

const SEVERITY_COLORS: Record<SeverityLevel, { bg: string; text: string; border: string }> = {
  CRITICAL: { bg: "bg-red-950/80", text: "text-red-400", border: "border-red-600/60" },
  HIGH: { bg: "bg-orange-950/80", text: "text-orange-400", border: "border-orange-600/60" },
  MEDIUM: { bg: "bg-amber-950/80", text: "text-amber-400", border: "border-amber-600/60" },
  LOW: { bg: "bg-blue-950/80", text: "text-blue-400", border: "border-blue-600/60" },
  INFO: { bg: "bg-slate-800", text: "text-slate-300", border: "border-slate-700" },
  UNKNOWN: { bg: "bg-slate-800", text: "text-slate-400", border: "border-slate-700" },
};

export const FindingsDashboard: React.FC<FindingsDashboardProps> = ({
  findings,
  remediatedCount,
  newCount,
  isRunActive,
  attachedWorkspace,
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedSeverity, setSelectedSeverity] = useState<string>("ALL");
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const filteredFindings = findings.filter((f) => {
    const matchesSearch =
      f.vuln_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      f.package_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      (f.file_path && f.file_path.toLowerCase().includes(searchTerm.toLowerCase())) ||
      f.cwes.some((c) => c.toLowerCase().includes(searchTerm.toLowerCase()));

    const matchesSeverity = selectedSeverity === "ALL" || f.severity === selectedSeverity;
    return matchesSearch && matchesSeverity;
  });

  const criticalCount = findings.filter((f) => f.severity === "CRITICAL").length;
  const highCount = findings.filter((f) => f.severity === "HIGH").length;
  const mediumCount = findings.filter((f) => f.severity === "MEDIUM").length;
  const lowCount = findings.filter((f) => f.severity === "LOW").length;

  return (
    <div className="flex flex-col h-full bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      {/* Top Metric Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-4 bg-slate-950/60 border-b border-slate-800">
        <div className="flex flex-col p-3 bg-slate-900/90 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Active Findings</span>
            <ShieldAlert className="w-4 h-4 text-rose-400" />
          </div>
          <div className="text-2xl font-bold text-slate-100">{findings.length}</div>
          <div className="flex items-center gap-1 mt-1 text-[11px] text-slate-400">
            <span className="text-red-400 font-semibold">{criticalCount} Crit</span> ·{" "}
            <span className="text-orange-400 font-semibold">{highCount} High</span> ·{" "}
            <span className="text-amber-400 font-semibold">{mediumCount} Med</span> ·{" "}
            <span className="text-blue-400 font-semibold">{lowCount} Low</span>
          </div>
        </div>

        <div className="flex flex-col p-3 bg-slate-900/90 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Remediated</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400">{remediatedCount}</div>
          <div className="text-[11px] text-slate-400 mt-1">
            <span>Removed upon QA validation</span>
          </div>
        </div>

        <div className="flex flex-col p-3 bg-slate-900/90 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Newly Detected</span>
            <Sparkles className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400">{newCount}</div>
          <div className="text-[11px] text-slate-400 mt-1">
            <span>From targeted/delta scans</span>
          </div>
        </div>

        <div className="flex flex-col p-3 bg-slate-900/90 rounded-lg border border-slate-800">
          <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
            <span>Engine State</span>
            <RefreshCw className={`w-4 h-4 text-blue-400 ${isRunActive ? "animate-spin" : ""}`} />
          </div>
          <div className="text-sm font-semibold mt-1">
            {isRunActive ? (
              <span className="text-blue-400 flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-ping" />
                Remediating
              </span>
            ) : (
              <span className="text-slate-400">Ready / Idle</span>
            )}
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            <span>{isRunActive ? "Live sync active" : "Awaiting run"}</span>
          </div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-slate-900/80 border-b border-slate-800">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="w-4 h-4 text-slate-500 absolute left-2.5 top-2.5" />
          <input
            type="text"
            placeholder="Search CVE, package, or file..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
          />
        </div>

        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <Filter className="w-3.5 h-3.5 text-slate-500" />
          <span>Severity:</span>
          {["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"].map((sev) => (
            <button
              key={sev}
              onClick={() => setSelectedSeverity(sev)}
              className={`px-2 py-1 rounded text-[11px] font-medium transition ${
                selectedSeverity === sev
                  ? "bg-blue-600 text-white"
                  : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
              }`}
            >
              {sev}
            </button>
          ))}
        </div>
      </div>

      {/* Findings List */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2.5 terminal-scroll">
        {filteredFindings.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-center text-slate-500">
            {findings.length === 0 ? (
              remediatedCount > 0 ? (
                <>
                  <CheckCircle2 className="w-10 h-10 text-emerald-400 mb-2 animate-bounce" />
                  <p className="text-sm font-medium text-emerald-300">All findings remediated!</p>
                  <p className="text-xs text-slate-400 mt-1">Zero active vulnerabilities remain in the codebase.</p>
                </>
              ) : attachedWorkspace ? (
                <div className="flex flex-col items-center max-w-md px-4 text-center">
                  <ShieldAlert className="w-9 h-9 text-amber-400 mb-2" />
                  <p className="text-sm font-semibold text-slate-200">
                    Attached <span className="text-blue-400">{attachedWorkspace.repo_name}</span> (0 findings)
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    The report at <code className="text-amber-300 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">{attachedWorkspace.odc_report_path || "report.json"}</code> contains 0 flagged vulnerabilities.
                  </p>
                  <p className="text-xs text-slate-400 mt-2">
                    To test with vulnerabilities, load the benchmark fixture <strong className="text-blue-400">examples/juice_shop</strong> (which contains 173 baseline vulnerabilities).
                  </p>
                </div>
              ) : (
                <>
                  <ShieldAlert className="w-8 h-8 text-slate-600 mb-2" />
                  <p className="text-sm">No findings loaded.</p>
                  <p className="text-xs text-slate-500 mt-1">Attach a folder containing an ODC report to start.</p>
                </>
              )
            ) : (
              <>
                <AlertTriangle className="w-6 h-6 text-slate-600 mb-2" />
                <p className="text-xs">No findings match the current filter or search criteria.</p>
              </>
            )}
          </div>
        ) : (
          filteredFindings.map((finding) => {
            const isExpanded = expandedIds.has(finding.id);
            const colors = SEVERITY_COLORS[finding.severity] || SEVERITY_COLORS.UNKNOWN;

            return (
              <div
                key={finding.id}
                className={`flex flex-col bg-slate-950/70 border rounded-lg transition overflow-hidden ${
                  finding.is_new
                    ? "border-amber-500/60 bg-amber-950/10 shadow-sm"
                    : finding.status === "remediating"
                    ? "border-blue-500/60 bg-blue-950/10 shadow-sm"
                    : "border-slate-800 hover:border-slate-700"
                }`}
              >
                {/* Finding Header Item */}
                <div
                  onClick={() => toggleExpand(finding.id)}
                  className="flex items-center justify-between p-3 cursor-pointer select-none gap-3"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span
                      className={`px-2 py-0.5 text-[10px] font-bold rounded border ${colors.bg} ${colors.text} ${colors.border}`}
                    >
                      {finding.severity}
                    </span>

                    <span className="font-mono text-xs font-semibold text-slate-200 truncate">
                      {finding.vuln_id}
                    </span>

                    <div className="flex items-center gap-1.5 px-2 py-0.5 bg-slate-800/80 rounded text-xs text-slate-300">
                      <span className="font-medium text-blue-300">{finding.package_name}</span>
                      {finding.package_version && (
                        <span className="text-slate-400 text-[11px]">@{finding.package_version}</span>
                      )}
                    </div>

                    {finding.is_new && (
                      <span className="px-1.5 py-0.5 text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 rounded flex items-center gap-1">
                        <Sparkles className="w-3 h-3" />
                        NEW (Targeted Scan)
                      </span>
                    )}

                    {finding.status === "remediating" && (
                      <span className="px-1.5 py-0.5 text-[10px] font-medium bg-blue-500/20 text-blue-300 border border-blue-500/40 rounded flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping" />
                        Remediating
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-3 flex-shrink-0">
                    {finding.file_path && (
                      <span className="text-[11px] font-mono text-slate-500 hidden md:inline truncate max-w-xs">
                        {finding.file_path}
                      </span>
                    )}
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-slate-400" />
                    )}
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-3 pb-3 pt-1 border-t border-slate-900 bg-slate-950/40 text-xs text-slate-300 space-y-2">
                    {finding.description && (
                      <p className="text-slate-400 leading-relaxed">{finding.description}</p>
                    )}

                    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-slate-400 pt-1">
                      {finding.file_path && (
                        <div>
                          <span className="text-slate-500">File: </span>
                          <span className="font-mono text-slate-300">{finding.file_path}</span>
                        </div>
                      )}
                      {finding.cwes.length > 0 && (
                        <div className="flex items-center gap-1">
                          <span className="text-slate-500">CWEs: </span>
                          {finding.cwes.map((cwe) => (
                            <span key={cwe} className="px-1.5 py-0.2 bg-slate-800 text-slate-300 rounded font-mono">
                              {cwe}
                            </span>
                          ))}
                        </div>
                      )}
                      {finding.group_id && (
                        <div>
                          <span className="text-slate-500">Group ID: </span>
                          <span className="font-mono text-slate-400">{finding.group_id}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
