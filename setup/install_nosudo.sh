#!/usr/bin/env bash
# No-sudo install path — for environments where apt can't be invoked.
# Uses python3 -m venv (pip comes inside the venv).
set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*" | tee -a install.log; }

log "=== No-sudo install starting ==="
log "Base: $BASE_DIR"

# ── 1. Create venvs ──────────────────────────────────────────
for venv in venv_iopaint venv_app; do
    if [ ! -d "$venv" ]; then
        log "Creating $venv..."
        python3 -m venv "$venv"
    fi
done

# ── 2. IOPaint venv (PyTorch + IOPaint) ──────────────────────
log "=== Installing IOPaint + PyTorch nightly cu128 (large download) ==="
source venv_iopaint/bin/activate
pip install --upgrade pip wheel -q 2>&1 | tail -3 | tee -a install.log

log "Installing PyTorch nightly cu128 (~3 GB, may take 5-10 min)..."
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    2>&1 | tail -5 | tee -a install.log || {
        log "WARN: nightly cu128 failed, trying cu124 stable..."
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124 \
            2>&1 | tail -5 | tee -a install.log
    }

log "Installing IOPaint + diffusers..."
pip install iopaint diffusers transformers accelerate safetensors \
    opencv-python-headless Pillow requests \
    2>&1 | tail -5 | tee -a install.log

log "Testing IOPaint import + CUDA..."
python3 -c "
import torch, iopaint
print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')
print(f'  iopaint module loaded OK')
" 2>&1 | tee -a install.log
deactivate

# ── 3. Flask app venv ────────────────────────────────────────
log "=== Installing Flask app deps ==="
source venv_app/bin/activate
pip install --upgrade pip -q 2>&1 | tail -2 | tee -a install.log
pip install -r app/requirements.txt 2>&1 | tail -5 | tee -a install.log
deactivate

# ── 4. ComfyUI (optional, can skip) ──────────────────────────
if [ "${SKIP_COMFYUI:-}" != "1" ] && [ ! -d "ComfyUI" ]; then
    log "=== Installing ComfyUI ==="
    git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI 2>&1 | tail -3 | tee -a install.log
    python3 -m venv ComfyUI/venv
    source ComfyUI/venv/bin/activate
    pip install --upgrade pip wheel -q 2>&1 | tail -2 | tee -a install.log
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 \
        2>&1 | tail -3 | tee -a install.log || \
    pip install torch torchvision torchaudio -q 2>&1 | tail -3 | tee -a install.log
    pip install -r ComfyUI/requirements.txt 2>&1 | tail -3 | tee -a install.log

    # Symlink shared model dirs
    mkdir -p models/{checkpoints,vae,loras}
    for d in checkpoints vae loras; do
        rm -rf "ComfyUI/models/$d"
        ln -sfn "$BASE_DIR/models/$d" "ComfyUI/models/$d"
    done
    deactivate
fi

log "=== Install complete ==="
log "Skipped: apt packages (need sudo) and SD inpainting model (4 GB download)"
log "Run with: ./run.sh"
