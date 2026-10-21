@echo off
setlocal
cd /d "%~dp0\.."

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

echo ==========================================================
echo    Remediation Engine Web UI Demo Launcher
echo ==========================================================
echo Opening http://localhost:8000 in browser...
start http://localhost:8000

"%PYTHON_EXE%" -m uvicorn demo.backend.main:app --host 127.0.0.1 --port 8000 --reload
