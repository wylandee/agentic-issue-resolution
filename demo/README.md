# Remediation Engine Demo Web UI

A modern, responsive web application for demonstrating the `remediation_engine` autonomous AppSec vulnerability remediation workflow.

## Key Features

1. **Attach Codebase / Scoped Repository**
   - **Directory Browser**: Select any local folder (e.g., a scoped Juice Shop repository) using standard browser folder selection.
   - **Local Path Input**: Quick attachment modal allowing host folder path entry (e.g. `examples/juice_shop` or absolute paths).
   - **Automated ODC Discovery**: Automatically discovers Dependency-Check / ODC JSON reports or issue JSONL files inside the attached repo.

2. **Interactive Findings Dashboard**
   - **Initial State Display**: Immediately parses and lists all baseline vulnerability findings categorized by severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).
   - **Real-Time Live Updates**: As the engine iterates through tasks, remediated issues transition status and disappear from the active findings list.
   - **Newly Detected Issues**: Any regressions or newly introduced vulnerabilities discovered by QA/targeted scans appear dynamically with a `NEW` badge.
   - **Search & Filter**: Search findings by CVE ID, package name, CWE, or description, and filter by severity.

3. **Curated Agent Console Log**
   - **Zero LLM Prompt Noise**: Filters out voluminous prompts and raw chain internals.
   - **Four High-Value Categories**:
     - `[NODE]` Graph transitions (`initial_triage`, `supervisor`, `update_subagent`, `qa_critic`, `report`, etc.).
     - `[TOOL]` Worker tool execution (e.g., npm dependency mutations, AST rewrites, canary builds).
     - `[DET]` Deterministic engine milestones (triage grouping, portfolio candidate selection, canary verification).
     - `[RESULT]` Attempt outcomes (e.g. `QA_PASSED`, `QA_FAILED`, scan clean confirmation).
   - **Live Streaming**: Pushes events via Server-Sent Events (SSE) with auto-scroll and category filters.

4. **Final Remediation Report & Diff Viewer**
   - **View Report Button**: Enabled automatically upon run completion.
   - **Markdown Preview**: Fully rendered GitHub-Flavored Markdown summary with metrics, resolved issues, and unresolved items.
   - **Unified Git Patch Viewer**: Colorized diff viewer showing the exact code mutations proposed by the engine.

5. **Zero Engine Modifications**
   - All demo code is strictly confined to the `demo/` directory.
   - Zero edits or patches were made to `src/remediation_engine/`.

---

## Directory Layout

```text
demo/
├── backend/
│   ├── engine_bridge.py        # Non-intrusive LangGraph telemetry hooks & background runner
│   ├── findings_tracker.py     # Live findings state tracker & reconciliation engine
│   ├── main.py                 # FastAPI application with REST & SSE endpoints
│   ├── models.py               # Pydantic data models for UI and findings
│   ├── requirements.txt        # Backend dependencies (fastapi, uvicorn, sse-starlette)
│   └── workspace_manager.py    # Local path & folder upload handling + ODC report parsing
├── frontend/
│   ├── dist/                   # Production-bundled React static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── ConsoleLogView.tsx      # Terminal-style filtered log viewer
│   │   │   ├── FindingsDashboard.tsx   # Metrics cards, search, & expandable finding items
│   │   │   ├── FolderAttachButton.tsx  # Folder attachment & local path modal
│   │   │   └── ReportModal.tsx         # Rendered Markdown & unified diff viewer
│   │   ├── api.ts              # Fetch client and EventSource SSE subscriber
│   │   ├── App.tsx             # Main dashboard layout and state machine
│   │   ├── index.css           # Tailwind CSS styles
│   │   ├── main.tsx            # React 18 DOM mount
│   │   └── types.ts            # TypeScript interfaces
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
├── start_demo.bat              # Quick-launch batch file for Windows
├── start_demo.ps1              # PowerShell launch script
└── README.md
```

---

## Getting Started

### Prerequisites

- Python virtual environment with dependencies installed:
  ```powershell
  .venv\Scripts\python -m pip install -r demo\backend\requirements.txt
  ```
- Node.js (v18+) for building or developing frontend (already pre-built in `demo/frontend/dist/`).

### Launching the Demo

#### Option 1: Standalone Single-Server (Recommended)
You can launch using **Bash**, **PowerShell**, or the Windows **Batch** file:

- **Bash (Linux / macOS / WSL / Git Bash):**
  ```bash
  chmod +x demo/start_demo.sh
  ./demo/start_demo.sh
  ```
- **PowerShell (Windows):**
  ```powershell
  powershell -ExecutionPolicy Bypass -File demo\start_demo.ps1
  ```
- **Windows Command Prompt:**
  ```cmd
  demo\start_demo.bat
  ```

This starts the FastAPI backend on port `8000` and serves the pre-built React frontend. Your browser will automatically open `http://localhost:8000`.

#### Option 2: Development Mode (Live Hot Reloading)
To run with live hot reloading on both FastAPI and Vite:

- **Bash:**
  ```bash
  ./demo/start_demo.sh --dev
  ```
- **PowerShell:**
  ```powershell
  powershell -ExecutionPolicy Bypass -File demo\start_demo.ps1 -Dev
  ```

- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000` (automatically proxied by Vite)

---

## Demo Walkthrough

1. **Attach Codebase**:
   - Click **Attach Folder**.
   - Either select a directory or click **Enter local folder path** and input `examples/juice_shop` (or an absolute path to your scoped repository containing an ODC report).
   - The dashboard will instantly populate with the parsed baseline findings and severity counters.

2. **Start Remediation**:
   - Click the green **Remediate** button in the top navigation bar.
   - The status updates to **Running**, and the filtered **Console Logs** panel will begin streaming node transitions (`[NODE]`), worker tool invocations (`[TOOL]`), and deterministic milestones (`[DET]`).

3. **Observe Real-Time Dashboard Updates**:
   - As each task progresses and successfully passes QA, its associated findings change status and are removed from the active findings list.
   - Any new issues introduced during targeted validation scans are added with an orange `NEW` badge.

4. **View Final Report & Patch**:
   - When remediation finishes, the **View Report** button lights up.
   - Click it to view the comprehensive Markdown report and inspect the unified diff showing all proposed code patches.
