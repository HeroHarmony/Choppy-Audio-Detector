#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

echo "Installing/updating dependencies..."
".venv/bin/python" -m pip install -r requirements.txt

echo "Launching GUI..."
exec ".venv/bin/python" app_gui.py "$@"
