import React, { useEffect, useRef, useState } from "react";
import { ConsoleCategory, ConsoleEvent } from "../types";
import { Terminal, ArrowDown, Ban, Cpu, Wrench, GitFork, CheckCircle2 } from "lucide-react";

interface ConsoleLogViewProps {
  logs: ConsoleEvent[];
  currentNode?: string | null;
  isRunning: boolean;
}

const CATEGORY_STYLES: Record<
  ConsoleCategory,
  { label: string; bg: string; text: string; icon: React.FC<{ className?: string }> }
> = {
  NODE_TRANSITION: {
    label: "NODE",
    bg: "bg-blue-950/80 border-blue-500/40",
    text: "text-blue-300",
    icon: GitFork,
  },
  TOOL_CALL: {
    label: "TOOL",
    bg: "bg-amber-950/80 border-amber-500/40",
    text: "text-amber-300",
    icon: Wrench,
  },
  DETERMINISTIC_FN: {
    label: "DET",
    bg: "bg-purple-950/80 border-purple-500/40",
    text: "text-purple-300",
    icon: Cpu,
  },
  KEY_RESULT: {
    label: "RESULT",
    bg: "bg-emerald-950/80 border-emerald-500/40",
    text: "text-emerald-300",
    icon: CheckCircle2,
  },
};

export const ConsoleLogView: React.FC<ConsoleLogViewProps> = ({ logs, currentNode, isRunning }) => {
  const [autoScroll, setAutoScroll] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState<string>("ALL");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter(
    (log) => selectedCategory === "ALL" || log.category === selectedCategory
  );

  return (
    <div className="flex flex-col h-full bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-xl font-mono text-xs">
      {/* Terminal Title Bar */}
      <div className="flex items-center justify-between px-3 py-2 bg-slate-900 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" />
          <span className="font-semibold text-slate-200">Execution Console</span>
          {currentNode && isRunning && (
            <span className="px-2 py-0.5 text-[10px] bg-blue-950 text-blue-300 border border-blue-700/60 rounded flex items-center gap-1 font-sans">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping" />
              {currentNode}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 font-sans">
          {/* Category Filter Pills */}
          <div className="hidden sm:flex items-center gap-1 text-[10px]">
            {["ALL", "NODE_TRANSITION", "TOOL_CALL", "DETERMINISTIC_FN", "KEY_RESULT"].map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-1.5 py-0.5 rounded transition ${
                  selectedCategory === cat
                    ? "bg-slate-700 text-white font-bold"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {cat === "ALL" ? "All" : CATEGORY_STYLES[cat as ConsoleCategory]?.label || cat}
              </button>
            ))}
          </div>

          <button
            onClick={() => setAutoScroll((prev) => !prev)}
            className={`p-1 rounded transition text-[11px] flex items-center gap-1 ${
              autoScroll ? "text-emerald-400 bg-emerald-950/40" : "text-slate-500 hover:text-slate-300"
            }`}
            title="Auto-scroll to bottom"
          >
            <ArrowDown className="w-3.5 h-3.5" />
            <span className="hidden md:inline">Auto-scroll</span>
          </button>
        </div>
      </div>

      {/* Terminal Body */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-2 terminal-scroll bg-[#0a0f1d]">
        {filteredLogs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-48 text-slate-600 font-sans">
            <Ban className="w-6 h-6 mb-2" />
            <p className="text-xs">No console events recorded yet.</p>
            <p className="text-[11px] text-slate-600 mt-0.5">Click "Remediate" to start the engine run.</p>
          </div>
        ) : (
          filteredLogs.map((log) => {
            const style = CATEGORY_STYLES[log.category] || CATEGORY_STYLES.NODE_TRANSITION;
            const Icon = style.icon;

            return (
              <div
                key={log.id}
                className="flex items-start gap-2.5 p-1.5 rounded hover:bg-slate-900/60 transition group leading-relaxed"
              >
                {/* Timestamp */}
                <span className="text-slate-600 select-none text-[10px] pt-0.5 flex-shrink-0">
                  {log.timestamp}
                </span>

                {/* Category Pill */}
                <span
                  className={`px-1.5 py-0.2 rounded border text-[9px] font-bold tracking-wide flex-shrink-0 flex items-center gap-1 ${style.bg} ${style.text}`}
                >
                  <Icon className="w-2.5 h-2.5" />
                  {style.label}
                </span>

                {/* Log Content */}
                <div className="flex-1 min-w-0">
                  <div className={`font-semibold ${style.text} truncate`}>{log.title}</div>
                  {log.details && (
                    <div className="text-slate-400 text-[11px] whitespace-pre-wrap break-all mt-0.5">
                      {log.details}
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
