#!/usr/bin/env bash
# Standalone IOPaint install (called from install.sh but usable alone)
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

info()    { echo -e "\033[1;34m[IOPaint]\033[0m $*"; }
success() { echo -e "\033[1;32m[IOPaint]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[IOPaint]\033[0m $*";  }

if [ -d "$BASE_DIR/venv_iopaint" ]; then
    info "IOPaint venv already exists — reinstalling packages..."
else
    python3 -m venv "$BASE_DIR/venv_iopaint"
fi

source "$BASE_DIR/venv_iopaint/bin/activate"
pip install --upgrade pip wheel -q

# Torch install (version depends on GPU)
if nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" | head -1)
    info "GPU: $GPU  CUDA: $CUDA_VER"

    if echo "$GPU" | grep -qiE "RTX 50"; then
        warn "RTX 50-series (Blackwell) detected."
        warn "Installing PyTorch nightly with CUDA 12.8..."
        pip install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128 -q
    elif echo "$CUDA_VER" | grep -qE "^12\.[4-9]|^1[3-9]\."; then
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124 -q
    else
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu118 -q
    fi
else
    warn "No GPU detected — installing CPU PyTorch."
    pip install torch torchvision torchaudio -q
fi

info "Installing IOPaint and dependencies..."
pip install iopaint -q
pip install "diffusers>=0.27" "transformers>=4.38" accelerate safetensors -q
pip install opencv-python-headless Pillow requests -q

success "IOPaint version: $(iopaint --version 2>/dev/null || python3 -c 'import iopaint; print(iopaint.__version__)')"
deactivate
