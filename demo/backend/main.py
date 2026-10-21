"""FastAPI backend application for the Remediation Engine demo."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from .engine_bridge import get_run_session, start_remediation_run
from .models import ReportResponse, RunState, WorkspaceInfo
from .workspace_manager import (
    create_workspace_from_local_path,
    create_workspace_from_upload,
    finalize_uploaded_workspace,
    get_workspace_data,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s]: %(message)s")
logger = logging.getLogger("demo_api")

app = FastAPI(title="Remediation Engine Demo API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SelectPathRequest(BaseModel):
    path: str


class StartRunRequest(BaseModel):
    """Request a development-scoped run for detected finding packages."""

    workspace_id: str
    target_packages: list[str] | None = None


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/workspace/select-path", response_model=WorkspaceInfo)
async def select_local_path(req: SelectPathRequest) -> WorkspaceInfo:
    """Attach a local directory path containing the codebase and ODC report."""
    try:
        return create_workspace_from_local_path(req.path)
    except Exception as exc:
        logger.error("Failed to register local path: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workspace/upload", response_model=WorkspaceInfo)
async def upload_folder(request: Request) -> WorkspaceInfo:
    """Upload folder files preserving relative directory hierarchy."""
    try:
        form = await request.form(
            max_files=10000,
            max_fields=20000,
            max_part_size=150 * 1024 * 1024,
        )
    except Exception as exc:
        logger.error("Failed to parse form: %s", exc)
        raise HTTPException(status_code=400, detail=f"Failed to parse upload form: {exc}") from exc

    raw_files = form.getlist("files")
    raw_paths = form.getlist("paths")

    if not raw_files or not raw_paths:
        raise HTTPException(status_code=400, detail="No files or paths uploaded in payload.")
    if len(raw_files) != len(raw_paths):
        raise HTTPException(
            status_code=400,
            detail=f"Files ({len(raw_files)}) and paths ({len(raw_paths)}) count mismatch.",
        )

    try:
        first_path_item = raw_paths[0]
        if isinstance(first_path_item, UploadFile):
            first_path_str = (await first_path_item.read()).decode("utf-8", errors="replace").strip()
            await first_path_item.seek(0)
        else:
            first_path_str = str(first_path_item).strip()

        first_rel = Path(first_path_str)
        root_name = first_rel.parts[0] if first_rel.parts else "uploaded_repo"
        workspace_id, repo_path = create_workspace_from_upload(root_name)

        for upload_file, rel_path_item in zip(raw_files, raw_paths, strict=False):
            if hasattr(upload_file, "read"):
                content = await upload_file.read()
            elif isinstance(upload_file, bytes):
                content = upload_file
            elif isinstance(upload_file, str):
                content = upload_file.encode("utf-8")
            else:
                continue

            if hasattr(rel_path_item, "read"):
                rel_path_str = (await rel_path_item.read()).decode("utf-8", errors="replace").strip()
            else:
                rel_path_str = str(rel_path_item).strip()

            rel_parts = Path(rel_path_str).parts
            if len(rel_parts) > 1 and rel_parts[0] == root_name:
                sub_rel = Path(*rel_parts[1:])
            else:
                sub_rel = Path(rel_path_str)

            dest_file = repo_path / sub_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_file, "wb") as f:
                f.write(content)

        return finalize_uploaded_workspace(workspace_id, repo_path)
    except Exception as exc:
        logger.error("Folder upload failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workspace/{workspace_id}", response_model=WorkspaceInfo)
async def get_workspace(workspace_id: str) -> WorkspaceInfo:
    """Retrieve workspace details and initial findings."""
    data = get_workspace_data(workspace_id)
    if not data:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return data["info"]


@app.post("/api/runs/start")
async def start_run(req: StartRunRequest) -> dict[str, str]:
    """Start remediation with a validated development package scope.

    Args:
        req: Workspace identifier and optional package allowlist. When omitted,
            targets are detected from the workspace findings.

    Returns:
        The new run identifier and initial status.

    Raises:
        HTTPException: If the workspace has no package findings or the request
            names packages without workspace findings.
    """
    ws_data = get_workspace_data(req.workspace_id)
    if not ws_data:
        raise HTTPException(status_code=404, detail=f"Workspace '{req.workspace_id}' not found.")

    repo_root = str(ws_data["repo_root"])
    issues = ws_data["issues"]
    available_packages: dict[str, str] = {}
    for issue in issues:
        if issue.package_name and issue.package_name.strip():
            available_packages.setdefault(
                issue.package_name.strip().casefold(),
                issue.package_name.strip(),
            )

    if req.target_packages is None:
        requested_packages = list(available_packages.values())
    elif not req.target_packages:
        raise HTTPException(status_code=400, detail="Select at least one detected package.")
    else:
        requested_packages = req.target_packages

    target_packages: list[str] = []
    unknown_packages: list[str] = []
    selected_keys: set[str] = set()
    for requested in requested_packages:
        package_name = requested.strip()
        key = package_name.casefold()
        if not package_name or key not in available_packages:
            unknown_packages.append(requested)
        elif key not in selected_keys:
            selected_keys.add(key)
            target_packages.append(available_packages[key])

    if unknown_packages:
        raise HTTPException(
            status_code=400,
            detail=f"Target packages must have findings in this workspace: {', '.join(unknown_packages)}",
        )
    if not target_packages:
        raise HTTPException(
            status_code=400,
            detail="No package findings were detected; targeted remediation cannot start.",
        )

    session = start_remediation_run(
        req.workspace_id,
        repo_root,
        issues,
        target_packages=target_packages,
    )
    return {"run_id": session.run_id, "status": session.state.status}


@app.get("/api/runs/{run_id}/state", response_model=RunState)
async def get_run_state(run_id: str) -> RunState:
    """Fetch current run status, active findings, and logs."""
    session = get_run_session(run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return session.state


@app.get("/api/runs/{run_id}/stream")
async def stream_run(run_id: str) -> EventSourceResponse:
    """SSE endpoint streaming console logs and real-time dashboard events."""
    session = get_run_session(run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    async def event_generator():
        # First send initial sync snapshot
        yield {
            "event": "initial_sync",
            "data": json.dumps(session.state.model_dump(mode="json")),
        }

        while True:
            # Poll event_queue without blocking asyncio loop
            try:
                item = session.event_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.15)
                # If thread has terminated and queue is empty, exit
                if session.thread and not session.thread.is_alive() and session.event_queue.empty():
                    break
                continue

            if item is None:
                # End of stream sentinel
                yield {
                    "event": "run_status",
                    "data": json.dumps(
                        {
                            "status": session.state.status,
                            "report_available": session.state.report_available,
                        }
                    ),
                }
                break

            yield {
                "event": item["event"],
                "data": json.dumps(item["data"]),
            }

    return EventSourceResponse(event_generator())


@app.get("/api/runs/{run_id}/report", response_model=ReportResponse)
async def get_run_report(run_id: str) -> ReportResponse:
    """Fetch generated final report Markdown and patch."""
    session = get_run_session(run_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    markdown = ""
    diff = session.state.diff or ""
    changed_files = []
    status = session.state.status

    if session.result:
        raw_state = getattr(session.result, "raw_state", {}) or {}
        markdown = raw_state.get("report_markdown", "")
        changed_files = list(session.result.changed_files or [])
        diff = session.result.diff or ""
        status = session.result.status

        if not markdown and session.result.report_path:
            p = Path(session.result.report_path)
            if p.is_file():
                markdown = p.read_text(encoding="utf-8")

    if not markdown and session.state.report_path:
        p = Path(session.state.report_path)
        if p.is_file():
            markdown = p.read_text(encoding="utf-8")

    if not markdown:
        # Fallback summary if report file was not written
        markdown = (
            f"# Remediation Run Report\n\n"
            f"**Run ID:** `{run_id}`\n\n"
            f"**Status:** `{status}`\n\n"
            f"**Remediated Findings:** {session.tracker.remediated_count}\n\n"
            f"**Remaining Active Findings:** {len(session.tracker.active_findings)}\n\n"
            f"## Changed Files\n\n"
            + ("\n".join(f"- `{f}`" for f in changed_files) if changed_files else "_No files modified._")
        )

    return ReportResponse(
        report_markdown=markdown,
        report_path=session.state.report_path,
        diff=diff,
        changed_files=changed_files,
        status=status,
    )


# Mount pre-built frontend distribution if built
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")

