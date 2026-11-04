#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PORT=8000
DEV_MODE=false

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dev)
      DEV_MODE=true
      shift
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: ./demo/start_demo.sh [options]"
      echo ""
      echo "Options:"
      echo "  -d, --dev     Start in development mode (FastAPI + Vite dev server with hot reload)"
      echo "  -p, --port    Port for the backend server (default: 8000)"
      echo "  -h, --help    Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# Resolve python interpreter
PYTHON_EXE=""
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXE="${REPO_ROOT}/.venv/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
  PYTHON_EXE="${REPO_ROOT}/.venv/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
  PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
  PYTHON_EXE="python"
else
  echo "Error: Python interpreter not found." >&2
  exit 1
fi

echo "=========================================================="
echo "   Remediation Engine Web UI Demo Launcher"
echo "=========================================================="
echo "Repository root: ${REPO_ROOT}"
echo "Python:          ${PYTHON_EXE}"
echo "Backend URL:     http://localhost:${PORT}"
echo ""

# Helper to open browser if possible
open_browser() {
  local url="$1"
  if command -v xdg-open &>/dev/null; then
    xdg-open "${url}" &>/dev/null || true
  elif command -v open &>/dev/null; then
    open "${url}" &>/dev/null || true
  elif command -v start &>/dev/null; then
    start "${url}" &>/dev/null || true
  fi
}

if [[ "${DEV_MODE}" == true ]]; then
  echo "Starting in DEVELOPMENT mode (FastAPI + Vite dev server)..."
  
  # Cleanup child processes on exit
  cleanup() {
    echo ""
    echo "Shutting down servers..."
    if [[ -n "${BACKEND_PID:-}" ]]; then
      kill "${BACKEND_PID}" 2>/dev/null || true
    fi
    exit 0
  }
  trap cleanup SIGINT SIGTERM EXIT

  # Start FastAPI backend in background
  "${PYTHON_EXE}" -m uvicorn demo.backend.main:app --host 127.0.0.1 --port "${PORT}" --reload &
  BACKEND_PID=$!
  echo "FastAPI started (PID ${BACKEND_PID})."

  # Start Vite dev server in frontend directory
  echo "Starting Vite dev server on http://localhost:5173..."
  cd "${SCRIPT_DIR}/frontend"
  npm run dev
else
  echo "Starting in PRODUCTION mode (FastAPI serving pre-built UI)..."
  open_browser "http://localhost:${PORT}" &
  exec "${PYTHON_EXE}" -m uvicorn demo.backend.main:app --host 127.0.0.1 --port "${PORT}" --reload
fi
