<#
.SYNOPSIS
    Starts the Remediation Engine Demo Web UI.
.DESCRIPTION
    Launches the FastAPI backend (and optionally Vite dev server) for the demo web UI.
.PARAMETER Dev
    If specified, launches both FastAPI and Vite dev server with hot reload.
.PARAMETER Port
    Backend port (default: 8000).
#>
param(
    [switch]$Dev,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
Set-Location $RepoRoot

# Find python interpreter
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Remediation Engine Web UI Demo Launcher" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "Repository root: $RepoRoot"
Write-Host "Python: $PythonExe"
Write-Host "Backend URL: http://localhost:$Port"

if ($Dev) {
    Write-Host "Starting in DEVELOPMENT mode (FastAPI + Vite dev server)..." -ForegroundColor Yellow
    
    # Start FastAPI backend in a background job or separate process
    $backendProcess = Start-Process -FilePath $PythonExe -ArgumentList "-m", "uvicorn", "demo.backend.main:app", "--host", "127.0.0.1", "--port", "$Port", "--reload" -PassThru
    
    Write-Host "FastAPI started (PID $($backendProcess.Id)). Starting Vite dev server..." -ForegroundColor Green
    
    # Start Vite in foreground
    Set-Location (Join-Path $ScriptDir "frontend")
    cmd.exe /c "npm run dev"
} else {
    Write-Host "Starting in PRODUCTION mode (FastAPI serving pre-built UI)..." -ForegroundColor Green
    Write-Host "Opening http://localhost:$Port in browser..." -ForegroundColor Yellow
    Start-Process "http://localhost:$Port"
    
    & $PythonExe -m uvicorn demo.backend.main:app --host 127.0.0.1 --port $Port --reload
}
