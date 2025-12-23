#!/bin/bash
# Brain Widget Launcher
# Double-click this file or run: ./launch_widget.sh

cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

echo "🧠 Launching Brain Widget..."
uv run python src/brain_widget.py
