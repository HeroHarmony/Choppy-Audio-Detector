#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

echo "Installing runtime dependencies..."
".venv/bin/python" -m pip install -r requirements.txt
echo "Installing build dependencies..."
".venv/bin/python" -m pip install pyinstaller

echo "Building macOS app bundle..."
".venv/bin/python" -m PyInstaller --noconfirm --clean ChoppyAudioDetector.spec

echo "Build complete: dist/ChoppyAudioDetector.app"
