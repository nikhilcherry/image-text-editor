#!/usr/bin/env bash
# ============================================================
#  Image Text Editor — Master Install Script
#  RTX 5050 / Linux / CUDA 12.x
# ============================================================
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$BASE_DIR/install.log"

info()    { echo -e "\033[1;34m[INFO]\033[0m $*" | tee -a "$LOG"; }
success() { echo -e "\033[1;32m[OK]\033[0m $*"   | tee -a "$LOG"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m $*"  | tee -a "$LOG"; }
error()   { echo -e "\033[1;31m[ERR]\033[0m $*"   | tee -a "$LOG"; exit 1; }

echo "================================================================"
echo "  Image Text Editor – Full Install"
echo "  $(date)"
echo "================================================================" | tee "$LOG"

# ── 1. System deps ───────────────────────────────────────────
info "Checking system dependencies..."

MISSING=()
for cmd in python3 pip3 git curl wget; do
    command -v "$cmd" &>/dev/null || MISSING+=("$cmd")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    warn "Missing: ${MISSING[*]}. Installing via apt..."
    sudo apt-get update -qq
    sudo apt-get install -y python3 python3-pip python3-venv git curl wget \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
        libopencv-dev ffmpeg fonts-liberation fonts-dejavu-core
fi

PYTHON_VER=$(python3 --version | awk '{print $2}')
info "Python: $PYTHON_VER"

# ── 2. CUDA check ────────────────────────────────────────────
info "Checking CUDA..."
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    success "GPU: $GPU_NAME | CUDA: $CUDA_VER"

    # RTX 50xx (Blackwell) requires CUDA ≥ 12.8 + PyTorch nightly
    if echo "$GPU_NAME" | grep -qiE "RTX 50"; then
        warn "RTX 50-series detected. Blackwell GPU requires PyTorch nightly + CUDA 12.8+."
        warn "If torch install fails, run: setup/fix_rtx50.sh"
        TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
        TORCH_PKG="--pre torch torchvision torchaudio --index-url $TORCH_INDEX"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        TORCH_PKG="torch torchvision torchaudio --index-url $TORCH_INDEX"
    fi
else
    warn "nvidia-smi not found — will use CPU mode (slow)."
    TORCH_PKG="torch torchvision torchaudio"
fi

# ── 3. Create virtualenvs ────────────────────────────────────
info "Creating Python virtual environments..."

# IOPaint venv
if [ ! -d "$BASE_DIR/venv_iopaint" ]; then
    python3 -m venv "$BASE_DIR/venv_iopaint"
fi

# App (Flask) venv
if [ ! -d "$BASE_DIR/venv_app" ]; then
    python3 -m venv "$BASE_DIR/venv_app"
fi

# ── 4. Install IOPaint ───────────────────────────────────────
info "Installing IOPaint + PyTorch into venv_iopaint..."
source "$BASE_DIR/venv_iopaint/bin/activate"

pip install --upgrade pip wheel -q

# Install PyTorch first (large, version-sensitive)
eval "pip install $TORCH_PKG -q" || {
    warn "PyTorch CUDA install failed — falling back to CPU build."
    pip install torch torchvision torchaudio -q
}

pip install iopaint -q
pip install diffusers transformers accelerate safetensors -q
pip install opencv-python-headless Pillow -q

success "IOPaint installed: $(iopaint --version 2>/dev/null || echo 'version unknown')"
deactivate

# ── 5. Install Flask app deps ────────────────────────────────
info "Installing Flask app dependencies..."
source "$BASE_DIR/venv_app/bin/activate"
pip install --upgrade pip -q
pip install -r "$BASE_DIR/app/requirements.txt" -q
success "Flask app deps installed."
deactivate

# ── 6. Install ComfyUI ───────────────────────────────────────
info "Installing ComfyUI..."
bash "$BASE_DIR/setup/install_comfyui.sh"

# ── 7. Download models ───────────────────────────────────────
info "Downloading AI models..."
bash "$BASE_DIR/setup/download_models.sh"

# ── 8. Make scripts executable ──────────────────────────────
chmod +x "$BASE_DIR/run.sh" "$BASE_DIR/stop.sh"
chmod +x "$BASE_DIR/setup/"*.sh

echo ""
echo "================================================================"
success "Installation complete!"
echo "================================================================"
echo ""
echo "  Next steps:"
echo "   1. Run:  ./run.sh"
echo "   2. Open: http://localhost:5000"
echo ""
echo "  First-time model download may take a few minutes."
echo "================================================================"
