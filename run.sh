#!/usr/bin/env bash
# ============================================================
#  Image Text Editor — Start All Services
# ============================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="$BASE_DIR/temp/.pids"
mkdir -p "$BASE_DIR/temp"

info()    { echo -e "\033[1;34m[INFO]\033[0m $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m $*";   }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*";  }

# ── Check venvs ──────────────────────────────────────────────
if [ ! -d "$BASE_DIR/venv_iopaint" ] || [ ! -d "$BASE_DIR/venv_app" ]; then
    echo "Virtual environments not found. Run ./install.sh first."
    exit 1
fi

echo "================================================================"
echo "  Image Text Editor — Starting Services"
echo "================================================================"
echo ""

# ── Start IOPaint server ─────────────────────────────────────
IOPAINT_PORT=8080
IOPAINT_MODEL="${IOPAINT_MODEL:-lama}"

info "Starting IOPaint ($IOPAINT_MODEL) on port $IOPAINT_PORT..."

MODEL_CACHE="$BASE_DIR/models/iopaint_cache"
mkdir -p "$MODEL_CACHE"

source "$BASE_DIR/venv_iopaint/bin/activate"

# Check if GPU available
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    success "CUDA available — using GPU"
else
    DEVICE="cpu"
    warn "CUDA not available — using CPU (slow)"
fi

nohup iopaint start \
    --model="$IOPAINT_MODEL" \
    --port="$IOPAINT_PORT" \
    --host="127.0.0.1" \
    --model-dir="$MODEL_CACHE" \
    --device="$DEVICE" \
    > "$BASE_DIR/temp/iopaint.log" 2>&1 &

IOPAINT_PID=$!
echo "iopaint:$IOPAINT_PID" > "$PIDS_FILE"
deactivate

# ── Optionally start ComfyUI ─────────────────────────────────
COMFYUI_DIR="$BASE_DIR/ComfyUI"
if [ -d "$COMFYUI_DIR" ]; then
    info "Starting ComfyUI on port 8188..."
    cd "$COMFYUI_DIR"
    source "$COMFYUI_DIR/venv/bin/activate" 2>/dev/null || true
    nohup python3 main.py \
        --port 8188 \
        --cuda-device 0 \
        --output-directory "$BASE_DIR/output" \
        --input-directory "$BASE_DIR/input" \
        > "$BASE_DIR/temp/comfyui.log" 2>&1 &
    COMFYUI_PID=$!
    echo "comfyui:$COMFYUI_PID" >> "$PIDS_FILE"
    deactivate 2>/dev/null || true
    cd "$BASE_DIR"
    success "ComfyUI started (PID $COMFYUI_PID)"
else
    warn "ComfyUI not found — skipping (run setup/install_comfyui.sh to install)"
fi

# ── Wait for IOPaint to be ready ─────────────────────────────
info "Waiting for IOPaint to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://127.0.0.1:$IOPAINT_PORT/" &>/dev/null; then
        success "IOPaint is ready."
        break
    fi
    sleep 2
    printf "."
    if [ $i -eq 60 ]; then
        echo ""
        warn "IOPaint took too long. Check logs: temp/iopaint.log"
    fi
done
echo ""

# ── Start Flask app ───────────────────────────────────────────
info "Starting Flask app on port 5000..."
source "$BASE_DIR/venv_app/bin/activate"

export FLASK_ENV=development
export BASE_DIR="$BASE_DIR"
export IOPAINT_URL="http://127.0.0.1:$IOPAINT_PORT"
export COMFYUI_URL="http://127.0.0.1:8188"
export UPLOAD_FOLDER="$BASE_DIR/temp"
export OUTPUT_FOLDER="$BASE_DIR/output"
export MODELS_FOLDER="$BASE_DIR/models"

nohup python3 "$BASE_DIR/app/app.py" \
    > "$BASE_DIR/temp/flask.log" 2>&1 &

FLASK_PID=$!
echo "flask:$FLASK_PID" >> "$PIDS_FILE"
deactivate

sleep 2

echo ""
echo "================================================================"
success "All services started!"
echo "================================================================"
echo ""
echo "  🖼  Main Editor UI  → http://localhost:5000"
echo "  🔧  IOPaint API     → http://localhost:$IOPAINT_PORT"
[ -d "$COMFYUI_DIR" ] && echo "  🎨  ComfyUI         → http://localhost:8188"
echo ""
echo "  Logs:  tail -f temp/flask.log"
echo "         tail -f temp/iopaint.log"
echo ""
echo "  Stop:  ./stop.sh"
echo "================================================================"

# Open browser
sleep 1
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:5000" &>/dev/null || true
fi
