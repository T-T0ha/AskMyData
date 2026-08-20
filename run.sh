#!/usr/bin/env bash
# Start the backend and frontend together for local development.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
  .venv/bin/pip install -q -r backend/requirements.txt
  PYTHON=.venv/bin/python
fi

if [ ! -d frontend/node_modules ]; then
  echo "Installing frontend dependencies..."
  (cd frontend && npm install)
fi

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

(cd backend && "../$PYTHON" -m uvicorn app.main:app --reload --port 8000) &
(cd frontend && npm run dev) &
wait
