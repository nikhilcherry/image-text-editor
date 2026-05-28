#!/usr/bin/env bash
# Install ComfyUI with its own venv inside the project
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFYUI_DIR="$BASE_DIR/ComfyUI"

info()    { echo -e "\033[1;34m[ComfyUI]\033[0m $*"; }
success() { echo -e "\033[1;32m[ComfyUI]\033[0m $*"; }

if [ -d "$COMFYUI_DIR" ]; then
    info "ComfyUI already installed at $COMFYUI_DIR — skipping."
    exit 0
fi

info "Cloning ComfyUI..."
git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"

info "Creating ComfyUI venv..."
python3 -m venv "$COMFYUI_DIR/venv"
source "$COMFYUI_DIR/venv/bin/activate"

pip install --upgrade pip wheel -q

# Detect GPU / RTX 50xx
if nvidia-smi &>/dev/null; then
    GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    if echo "$GPU" | grep -qiE "RTX 50"; then
        info "RTX 50-series: installing PyTorch nightly cu128..."
        pip install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128 -q
    else
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124 -q
    fi
else
    pip install torch torchvision torchaudio -q
fi

pip install -r "$COMFYUI_DIR/requirements.txt" -q

# ── Copy our workflow to ComfyUI's user workflows dir ────────
mkdir -p "$COMFYUI_DIR/user/default/workflows"
cp "$BASE_DIR/workflows/sd_inpaint.json" \
   "$COMFYUI_DIR/user/default/workflows/sd_inpaint.json" 2>/dev/null || true

# ── Install useful custom nodes ──────────────────────────────
CUSTOM_NODES="$COMFYUI_DIR/custom_nodes"
mkdir -p "$CUSTOM_NODES"

info "Installing ComfyUI-Manager..."
git clone --depth=1 \
    https://github.com/ltdrdata/ComfyUI-Manager.git \
    "$CUSTOM_NODES/ComfyUI-Manager" 2>/dev/null || true

deactivate

# ── Symlink model dirs so both tools share models ────────────
MODELS_DIR="$BASE_DIR/models"
mkdir -p "$MODELS_DIR/checkpoints" "$MODELS_DIR/vae" "$MODELS_DIR/loras"

# Symlink ComfyUI model dirs → project models dir
for d in checkpoints vae loras; do
    rm -rf "$COMFYUI_DIR/models/$d"
    ln -sfn "$MODELS_DIR/$d" "$COMFYUI_DIR/models/$d"
done

success "ComfyUI installed at $COMFYUI_DIR"
