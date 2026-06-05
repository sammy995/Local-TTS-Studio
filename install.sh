#!/usr/bin/env bash
# Local TTS Studio - one-command installer for macOS / Linux
# Usage:  ./install.sh
# Creates a local virtual environment, installs dependencies, checks ffmpeg, and launches the app.

set -euo pipefail

echo ""
echo "  Local TTS Studio - setup"
echo "  ========================="
echo ""

# --- 1. Find a suitable Python (3.10+) -------------------------------------
PY=""
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ver="$("$cmd" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")"
        if [ -n "$ver" ]; then
            maj="${ver%%.*}"; min="${ver##*.}"
            if [ "$maj" -eq 3 ] && [ "$min" -ge 10 ]; then
                PY="$cmd"; break
            fi
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "  ERROR: Python 3.10+ was not found."
    echo "  Install it from https://www.python.org/downloads/ (or your package manager) and re-run."
    exit 1
fi
echo "  Using Python: $($PY --version)"

# --- 2. Create / reuse virtual environment ---------------------------------
if [ ! -d ".venv" ]; then
    echo "  Creating virtual environment (.venv)..."
    "$PY" -m venv .venv
fi
VENV_PY=".venv/bin/python"

# --- 3. Install dependencies -----------------------------------------------
echo "  Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet

echo "  Installing dependencies (this can take a few minutes)..."
"$VENV_PY" -m pip install -r requirements.txt

# --- 4. Check ffmpeg (needed for MP3/M4A) ----------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo ""
    echo "  ffmpeg was not found (needed for MP3/M4A export)."
    if command -v brew >/dev/null 2>&1; then
        echo "  Installing via Homebrew..."
        brew install ffmpeg || echo "  Could not auto-install ffmpeg; install it manually with: brew install ffmpeg"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "  Installing via apt..."
        sudo apt-get update && sudo apt-get install -y ffmpeg || echo "  Could not auto-install ffmpeg; install it manually with: sudo apt-get install ffmpeg"
    else
        echo "  Please install ffmpeg with your package manager. WAV output works without it."
    fi
fi

# --- 5. Launch --------------------------------------------------------------
echo ""
echo "  Setup complete. Starting Local TTS Studio..."
echo "  The first run downloads model weights (~10 GB) - please be patient."
echo "  Open http://localhost:8000 in your browser."
echo ""
exec "$VENV_PY" run_local.py
