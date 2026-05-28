#!/usr/bin/env bash
# Stop all Image Text Editor services
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$BASE_DIR/temp/.pids"

echo "Stopping Image Text Editor services..."

if [ -f "$PIDS_FILE" ]; then
    while IFS=: read -r name pid; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "  Stopped $name (PID $pid)"
        fi
    done < "$PIDS_FILE"
    rm -f "$PIDS_FILE"
fi

# Also kill by process name as fallback
pkill -f "iopaint start" 2>/dev/null || true
pkill -f "app/app.py"    2>/dev/null || true
pkill -f "ComfyUI/main.py" 2>/dev/null || true

echo "Done."
