import type {
  ConsoleEvent,
  DashboardUpdate,
  ReportResponse,
  RunState,
  WorkspaceInfo,
} from "./types";

const API_BASE = "";

function handleNetworkError(err: unknown, action: string): never {
  if (err instanceof TypeError && err.message.toLowerCase().includes("fetch")) {
    throw new Error(
      "Cannot connect to the backend server. Please make sure the FastAPI server is running on http://127.0.0.1:8000 (e.g. run 'powershell demo/start_demo.ps1' or 'demo/start_demo.sh')."
    );
  }
  if (err instanceof Error) {
    throw err;
  }
  throw new Error(`${action} failed: ${String(err)}`);
}

export async function selectLocalPath(path: string): Promise<WorkspaceInfo> {
  try {
    const res = await fetch(`${API_BASE}/api/workspace/select-path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Failed to select path");
    }
    return res.json();
  } catch (err: unknown) {
    handleNetworkError(err, "Path selection");
  }
}

export async function uploadFolder(files: File[]): Promise<WorkspaceInfo> {
  const formData = new FormData();
  const maxUploadFileBytes = 150 * 1024 * 1024;
  let skippedFileCount = 0;
  const skippedFiles: string[] = [];
  const recordSkippedFile = (path: string) => {
    skippedFileCount += 1;
    if (skippedFiles.length < 5) {
      skippedFiles.push(path);
    }
  };

  const ignoreDirPatterns = [
    "/.git/",
    "/node_modules/",
    "/.venv/",
    "/dist/",
    "/build/",
    "/__pycache__/",
    "/.pytest_cache/",
    "/.idea/",
    "/.vscode/",
  ];

  // Preserve repository assets by default. Tests and application code often
  // read images, fonts, or media directly from the source tree.
  const ignoreExtensions = new Set([".pyc"]);

  for (const file of files) {
    const rawRel = file.webkitRelativePath || file.name;
    const normalized = "/" + rawRel.replace(/\\/g, "/");

    // Skip ignored directories
    if (ignoreDirPatterns.some((pattern) => normalized.includes(pattern))) {
      recordSkippedFile(rawRel);
      continue;
    }

    // Skip generated Python bytecode.
    const ext = rawRel.includes(".") ? rawRel.slice(rawRel.lastIndexOf(".")).toLowerCase() : "";
    if (ignoreExtensions.has(ext)) {
      recordSkippedFile(rawRel);
      continue;
    }

    if (file.size > maxUploadFileBytes) {
      recordSkippedFile(rawRel);
      continue;
    }

    // Skip large HTML report files (e.g. 77MB baseline HTML; ODC engine strictly requires JSON/JSONL)
    if (ext === ".html" && file.size > 2 * 1024 * 1024) {
      recordSkippedFile(rawRel);
      continue;
    }

    formData.append("files", file);
    formData.append("paths", rawRel);
  }

  try {
    const res = await fetch(`${API_BASE}/api/workspace/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Folder upload failed");
    }
    const workspace = (await res.json()) as WorkspaceInfo;
    if (skippedFileCount > 0) {
      workspace.upload_skipped_count = skippedFileCount;
      workspace.upload_skipped_files = skippedFiles;
    }
    return workspace;
  } catch (err: unknown) {
    handleNetworkError(err, "Folder upload");
  }
}

export async function startRun(
  workspaceId: string,
  targetPackages: string[],
): Promise<{ run_id: string; status: string }> {
  const res = await fetch(`${API_BASE}/api/runs/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      workspace_id: workspaceId,
      target_packages: targetPackages,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to start remediation run");
  }
  return res.json();
}

export async function getRunReport(runId: string): Promise<ReportResponse> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/report`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to fetch report");
  }
  return res.json();
}

export async function getRunState(runId: string): Promise<RunState> {
  const res = await fetch(`${API_BASE}/api/runs/${runId}/state`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to fetch run state");
  }
  return res.json();
}

export interface StreamHandlers {
  onInitialSync?: (state: RunState) => void;
  onConsoleEvent?: (event: ConsoleEvent) => void;
  onDashboardUpdate?: (update: DashboardUpdate) => void;
  onRunStatus?: (statusData: { status: string; current_node?: string; report_available?: boolean }) => void;
  onError?: (err: Event) => void;
}

export function subscribeRunStream(runId: string, handlers: StreamHandlers): () => void {
  const eventSource = new EventSource(`${API_BASE}/api/runs/${runId}/stream`);

  eventSource.addEventListener("initial_sync", (e) => {
    try {
      const data = JSON.parse(e.data);
      handlers.onInitialSync?.(data);
    } catch (err) {
      console.error("Failed to parse initial_sync event", err);
    }
  });

  eventSource.addEventListener("console_event", (e) => {
    try {
      const data = JSON.parse(e.data);
      handlers.onConsoleEvent?.(data);
    } catch (err) {
      console.error("Failed to parse console_event", err);
    }
  });

  eventSource.addEventListener("dashboard_update", (e) => {
    try {
      const data = JSON.parse(e.data);
      handlers.onDashboardUpdate?.(data);
    } catch (err) {
      console.error("Failed to parse dashboard_update", err);
    }
  });

  eventSource.addEventListener("run_status", (e) => {
    try {
      const data = JSON.parse(e.data);
      handlers.onRunStatus?.(data);
    } catch (err) {
      console.error("Failed to parse run_status", err);
    }
  });

  eventSource.onerror = (err) => {
    handlers.onError?.(err);
  };

  return () => {
    eventSource.close();
  };
}
