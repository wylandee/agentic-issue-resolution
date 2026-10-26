import React, { useMemo, useState } from "react";
import type {
  ConsoleEvent,
  FindingModel,
  ReportResponse,
  RunStatus,
  WorkspaceInfo,
} from "./types";
import { getRunReport, startRun, subscribeRunStream } from "./api";
import { FolderAttachButton } from "./components/FolderAttachButton";
import { FindingsDashboard } from "./components/FindingsDashboard";
import { ConsoleLogView } from "./components/ConsoleLogView";
import { ReportModal } from "./components/ReportModal";
import { Play, FileText, Shield, Loader2, AlertOctagon } from "lucide-react";

function getTargetPackageOptions(findings: FindingModel[]): { name: string; findingCount: number }[] {
  /**
   * Group workspace findings into package choices for targeted runs.
   * @param findings Findings loaded for the attached workspace.
   * @returns Package names with finding counts, sorted by package name.
   */
  const packages = new Map<string, { name: string; findingCount: number }>();
  for (const finding of findings) {
    const name = finding.package_name.trim();
    if (!name || name.toLowerCase() === "unknown package") continue;
    const key = name.toLowerCase();
    const existing = packages.get(key);
    if (existing) existing.findingCount += 1;
    else packages.set(key, { name, findingCount: 1 });
  }
  return [...packages.values()].sort((left, right) => left.name.localeCompare(right.name));
}

export const App: React.FC = () => {
  const [workspace, setWorkspace] = useState<WorkspaceInfo | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<RunStatus>("pending");
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [activeFindings, setActiveFindings] = useState<FindingModel[]>([]);
  const [remediatedCount, setRemediatedCount] = useState<number>(0);
  const [newCount, setNewCount] = useState<number>(0);
  const [logs, setLogs] = useState<ConsoleEvent[]>([]);
  const [reportAvailable, setReportAvailable] = useState<boolean>(false);
  const [reportModalData, setReportModalData] = useState<ReportResponse | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedTargetPackages, setSelectedTargetPackages] = useState<string[]>([]);
  const [packageFilter, setPackageFilter] = useState("");

  const isRunning = runStatus === "running";
  const targetPackageOptions = useMemo(
    () => getTargetPackageOptions(workspace?.findings ?? []),
    [workspace],
  );

  const filteredTargetPackages = targetPackageOptions.filter((option) =>
    option.name.toLowerCase().includes(packageFilter.trim().toLowerCase()),
  );

  const handleWorkspaceAttached = (ws: WorkspaceInfo) => {
    setWorkspace(ws);
    setActiveFindings(ws.findings);
    setSelectedTargetPackages(getTargetPackageOptions(ws.findings).map((option) => option.name));
    setPackageFilter("");
    setRemediatedCount(0);
    setNewCount(0);
    setRunId(null);
    setRunStatus("pending");
    setCurrentNode(null);
    setLogs([]);
    setReportAvailable(false);
    setReportModalData(null);
    setErrorMessage(null);
  };

  const handleStartRemediation = async () => {
    if (!workspace) return;

    try {
      setErrorMessage(null);
      setRunStatus("running");
      const { run_id } = await startRun(workspace.workspace_id, selectedTargetPackages);
      setRunId(run_id);

      // Subscribe to real-time events
      subscribeRunStream(run_id, {
        onInitialSync: (state) => {
          setActiveFindings(state.active_findings);
          setRemediatedCount(state.remediated_count);
          setNewCount(state.new_count);
          setLogs(state.logs);
          setRunStatus(state.status);
          setCurrentNode(state.current_node ?? null);
          setReportAvailable(state.report_available);
        },
        onConsoleEvent: (event) => {
          setLogs((previous) => {
            const existingIndex = previous.findIndex((entry) => entry.id === event.id);
            if (existingIndex < 0) return [...previous, event];
            const updated = previous.slice();
            updated[existingIndex] = event;
            return updated;
          });
        },
        onDashboardUpdate: (update) => {
          setActiveFindings(update.active_findings);
          setRemediatedCount(update.remediated_count);
          setNewCount(update.new_count);
        },
        onRunStatus: (statusData) => {
          setRunStatus(statusData.status as RunStatus);
          if (statusData.current_node) {
            setCurrentNode(statusData.current_node);
          }
          if (statusData.report_available) {
            setReportAvailable(true);
          }
        },
        onError: () => {
          console.warn("SSE connection error or completed");
        },
      });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to start remediation run";
      setErrorMessage(msg);
      setRunStatus("failed");
    }
  };

  const handleOpenReport = async () => {
    if (!runId) return;
    setIsLoadingReport(true);
    try {
      const report = await getRunReport(runId);
      setReportModalData(report);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load report";
      setErrorMessage(msg);
    } finally {
      setIsLoadingReport(false);
    }
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Top Header / Action Bar */}
      <header className="flex flex-wrap items-center justify-between px-6 py-3.5 bg-slate-900 border-b border-slate-800 gap-4 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-700 text-white rounded-xl shadow-md">
            <Shield className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-slate-100 flex items-center gap-2">
              Remediation Engine
              <span className="text-[10px] font-semibold tracking-wider uppercase px-2 py-0.5 bg-blue-950 text-blue-400 border border-blue-700/50 rounded-full">
                Interactive Demo
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Agentic AppSec vulnerability remediation & deterministic verification
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <FolderAttachButton
            onWorkspaceAttached={handleWorkspaceAttached}
            attachedWorkspace={workspace}
            disabled={isRunning}
          />

          <button
            onClick={handleStartRemediation}
            disabled={
              !workspace ||
              isRunning ||
              selectedTargetPackages.length === 0
            }
            className={`flex items-center gap-2 px-5 py-2 rounded-lg font-semibold text-sm transition shadow-md ${
              isRunning
                ? "bg-blue-950 text-blue-300 border border-blue-700/50 cursor-not-allowed"
                : !workspace || selectedTargetPackages.length === 0
                ? "bg-slate-800 text-slate-500 border border-slate-700 cursor-not-allowed"
                : "bg-blue-600 hover:bg-blue-500 text-white hover:shadow-blue-500/25 cursor-pointer active:scale-95"
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
                <span>Remediating...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Remediate</span>
              </>
            )}
          </button>

          <button
            onClick={handleOpenReport}
            disabled={!reportAvailable || isLoadingReport}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm border transition ${
              reportAvailable
                ? "bg-emerald-950/80 hover:bg-emerald-900 border-emerald-600/60 text-emerald-300 shadow-emerald-950/40 cursor-pointer"
                : "bg-slate-900 text-slate-600 border-slate-800 cursor-not-allowed opacity-60"
            }`}
          >
            {isLoadingReport ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <FileText className="w-4 h-4 text-emerald-400" />
            )}
            <span>View Report</span>
          </button>
        </div>

        {workspace && (
          <div className="w-full border-t border-slate-800 pt-3">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-blue-300">
                Automatic targeted development scope
              </span>
              <span className="text-xs text-slate-300">
                {selectedTargetPackages.length} of {targetPackageOptions.length} detected packages selected
              </span>
            </div>

            <div className="mt-2 rounded-lg border border-blue-800/60 bg-blue-950/30 p-3">
              {targetPackageOptions.length === 0 ? (
                <p className="text-xs text-amber-300">
                  No package findings were detected, so a targeted run cannot start.
                </p>
              ) : (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="mr-auto text-xs text-slate-300">
                      Targets are detected from workspace findings. Dependency closure may include related packages.
                    </p>
                    <input
                      aria-label="Filter target packages"
                      type="search"
                      value={packageFilter}
                      onChange={(event) => setPackageFilter(event.target.value)}
                      disabled={isRunning}
                      placeholder="Filter packages"
                      className="w-44 rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-200 placeholder:text-slate-500 focus:border-blue-500 focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setSelectedTargetPackages(targetPackageOptions.map((option) => option.name))
                      }
                      disabled={isRunning}
                      className="text-xs text-blue-300 hover:text-blue-200 disabled:text-slate-600"
                    >
                      Select all
                    </button>
                    <button
                      type="button"
                      onClick={() => setSelectedTargetPackages([])}
                      disabled={isRunning}
                      className="text-xs text-slate-400 hover:text-slate-200 disabled:text-slate-600"
                    >
                      Clear
                    </button>
                  </div>
                  <div className="mt-2 grid max-h-28 grid-cols-1 gap-x-4 gap-y-1 overflow-y-auto sm:grid-cols-2 lg:grid-cols-3">
                    {filteredTargetPackages.map((option) => (
                      <label
                        key={option.name}
                        className="flex min-w-0 items-center gap-2 text-xs text-slate-300"
                      >
                        <input
                          type="checkbox"
                          checked={selectedTargetPackages.includes(option.name)}
                          onChange={(event) => {
                            setSelectedTargetPackages((current) =>
                              event.target.checked
                                ? [...current, option.name]
                                : current.filter((name) => name !== option.name),
                            );
                          }}
                          disabled={isRunning}
                          className="accent-blue-500"
                        />
                        <span className="truncate" title={option.name}>
                          {option.name}
                        </span>
                        <span className="ml-auto text-slate-500">{option.findingCount}</span>
                      </label>
                    ))}
                    {filteredTargetPackages.length === 0 && (
                      <p className="text-xs text-slate-500">No packages match this filter.</p>
                    )}
                  </div>
                  {selectedTargetPackages.length === 0 && (
                    <p className="mt-2 text-xs text-amber-300">
                      Select at least one detected package to start a targeted run.
                    </p>
                  )}
                </>
              )}
            </div>
          </div>
        )}
      </header>

      {/* Error Banner if any */}
      {errorMessage && (
        <div className="flex items-center justify-between px-6 py-2 bg-red-950/90 border-b border-red-700 text-xs text-red-200">
          <div className="flex items-center gap-2">
            <AlertOctagon className="w-4 h-4 text-red-400" />
            <span>{errorMessage}</span>
          </div>
          <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-white">
            Dismiss
          </button>
        </div>
      )}

      {/* Main 2-Column Content Layout */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 min-h-0 overflow-hidden bg-slate-950">
        {/* Left / Main Column: Findings Dashboard (7 cols on lg) */}
        <section className="lg:col-span-7 flex flex-col min-h-0 h-full">
          <FindingsDashboard
            findings={activeFindings}
            remediatedCount={remediatedCount}
            newCount={newCount}
            isRunActive={isRunning}
            attachedWorkspace={workspace}
          />
        </section>

        {/* Right Column: Filtered Console Log View (5 cols on lg) */}
        <section className="lg:col-span-5 flex flex-col min-h-0 h-full">
          <ConsoleLogView
            logs={logs}
            currentNode={currentNode}
            isRunning={isRunning}
          />
        </section>
      </main>

      {/* Final Markdown Report Modal */}
      {reportModalData && (
        <ReportModal
          report={reportModalData}
          onClose={() => setReportModalData(null)}
        />
      )}
    </div>
  );
};

export default App;
