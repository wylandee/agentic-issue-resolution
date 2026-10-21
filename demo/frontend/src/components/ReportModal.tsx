import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ReportResponse } from "../types";
import { X, FileText, Code2, Download, Copy, Check, GitCommit } from "lucide-react";

interface ReportModalProps {
  report: ReportResponse | null;
  onClose: () => void;
}

export const ReportModal: React.FC<ReportModalProps> = ({ report, onClose }) => {
  const [activeTab, setActiveTab] = useState<"rendered" | "raw" | "diff">("rendered");
  const [copied, setCopied] = useState(false);

  if (!report) return null;

  const handleCopy = () => {
    const textToCopy = activeTab === "diff" ? report.diff : report.report_markdown;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const content = activeTab === "diff" ? report.diff : report.report_markdown;
    const filename = activeTab === "diff" ? "remediation.patch" : "remediation_report.md";
    const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fade-in">
      <div className="flex flex-col w-full max-w-4xl h-[85vh] bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-3.5 bg-slate-950 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-1.5 bg-blue-950 text-blue-400 border border-blue-700/50 rounded-lg">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">Remediation Engine Report</h2>
              <p className="text-xs text-slate-400">
                Status: <span className="text-emerald-400 font-semibold">{report.status}</span>
                {report.changed_files.length > 0 && (
                  <span> · {report.changed_files.length} changed file(s)</span>
                )}
              </p>
            </div>
          </div>

          {/* Action buttons & tabs */}
          <div className="flex items-center gap-2">
            <div className="flex items-center p-1 bg-slate-900 border border-slate-800 rounded-lg text-xs">
              <button
                onClick={() => setActiveTab("rendered")}
                className={`px-3 py-1 rounded font-medium transition ${
                  activeTab === "rendered"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Rendered Report
              </button>
              <button
                onClick={() => setActiveTab("raw")}
                className={`px-3 py-1 rounded font-medium transition flex items-center gap-1 ${
                  activeTab === "raw"
                    ? "bg-blue-600 text-white shadow-sm"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Code2 className="w-3.5 h-3.5" />
                Raw Markdown
              </button>
              {report.diff && (
                <button
                  onClick={() => setActiveTab("diff")}
                  className={`px-3 py-1 rounded font-medium transition flex items-center gap-1 ${
                    activeTab === "diff"
                      ? "bg-blue-600 text-white shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <GitCommit className="w-3.5 h-3.5" />
                  Unified Patch
                </button>
              )}
            </div>

            <button
              onClick={handleCopy}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg border border-slate-800 transition"
              title="Copy to clipboard"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>

            <button
              onClick={handleDownload}
              className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-lg border border-slate-800 transition"
              title="Download file"
            >
              <Download className="w-4 h-4" />
            </button>

            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-red-400 hover:bg-slate-800 rounded-lg border border-slate-800 transition ml-2"
              title="Close modal"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-900/60 terminal-scroll">
          {activeTab === "rendered" && (
            <div className="max-w-none text-slate-200">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  h1: ({ children }) => (
                    <h1 className="text-2xl font-bold text-slate-100 mt-6 mb-4 pb-2.5 border-b border-slate-700/80">
                      {children}
                    </h1>
                  ),
                  h2: ({ children }) => (
                    <h2 className="text-xl font-bold text-slate-100 mt-8 mb-3.5 pb-2 border-b border-slate-800">
                      {children}
                    </h2>
                  ),
                  h3: ({ children }) => (
                    <h3 className="text-base font-semibold text-blue-400 mt-6 mb-2.5">
                      {children}
                    </h3>
                  ),
                  h4: ({ children }) => (
                    <h4 className="text-sm font-semibold text-slate-200 mt-4 mb-2">
                      {children}
                    </h4>
                  ),
                  p: ({ children }) => (
                    <p className="text-xs sm:text-sm text-slate-300 leading-relaxed my-3">
                      {children}
                    </p>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc list-outside ml-5 text-xs sm:text-sm text-slate-300 space-y-1.5 my-3">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal list-outside ml-5 text-xs sm:text-sm text-slate-300 space-y-1.5 my-3">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li className="text-slate-300 text-xs sm:text-sm leading-relaxed">
                      {children}
                    </li>
                  ),
                  table: ({ children }) => (
                    <div className="overflow-x-auto my-5 rounded-lg border border-slate-700/80 bg-slate-950/70 shadow-sm">
                      <table className="min-w-full text-xs sm:text-sm border-collapse divide-y divide-slate-800">
                        {children}
                      </table>
                    </div>
                  ),
                  thead: ({ children }) => (
                    <thead className="bg-slate-950 text-slate-200 font-semibold border-b border-slate-700">
                      {children}
                    </thead>
                  ),
                  tbody: ({ children }) => (
                    <tbody className="divide-y divide-slate-800/80 bg-slate-900/30">
                      {children}
                    </tbody>
                  ),
                  tr: ({ children }) => (
                    <tr className="hover:bg-slate-800/40 transition">
                      {children}
                    </tr>
                  ),
                  th: ({ children }) => (
                    <th className="px-3.5 py-2.5 text-left font-semibold text-slate-200 border-r border-slate-800 last:border-r-0">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="px-3.5 py-2 text-slate-300 border-r border-slate-800/80 last:border-r-0">
                      {children}
                    </td>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-blue-500 bg-slate-950/60 pl-4 py-2 my-4 text-slate-300 rounded-r text-xs sm:text-sm italic">
                      {children}
                    </blockquote>
                  ),
                  code: ({ inline, children, ...props }: any) => {
                    if (inline) {
                      return (
                        <code
                          className="text-blue-300 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800/80 font-mono text-xs"
                          {...props}
                        >
                          {children}
                        </code>
                      );
                    }
                    return (
                      <code
                        className="block p-3 bg-slate-950 border border-slate-800 rounded-lg font-mono text-xs text-slate-200 overflow-x-auto my-3"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  hr: () => <hr className="my-8 border-slate-800" />,
                }}
              >
                {report.report_markdown}
              </ReactMarkdown>
            </div>
          )}

          {activeTab === "raw" && (
            <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
              {report.report_markdown}
            </pre>
          )}

          {activeTab === "diff" && (
            <div className="space-y-3">
              <div className="text-xs text-slate-400 font-mono">
                {report.changed_files.length} file(s) changed in unified diff:
              </div>
              <pre className="p-4 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-200 whitespace-pre-wrap leading-relaxed overflow-x-auto">
                {report.diff ? (
                  report.diff.split("\n").map((line, idx) => {
                    let lineClass = "text-slate-400";
                    if (line.startsWith("+") && !line.startsWith("+++")) {
                      lineClass = "text-emerald-400 bg-emerald-950/30";
                    } else if (line.startsWith("-") && !line.startsWith("---")) {
                      lineClass = "text-red-400 bg-red-950/30";
                    } else if (line.startsWith("@@")) {
                      lineClass = "text-blue-400 font-semibold";
                    }
                    return (
                      <div key={idx} className={lineClass}>
                        {line}
                      </div>
                    );
                  })
                ) : (
                  <span className="text-slate-500">_No patch generated (no code changes required)._</span>
                )}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
