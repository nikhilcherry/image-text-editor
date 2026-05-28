#!/usr/bin/env bash
# Install path using `uv` — no sudo needed, no system python3-venv.
set -eo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

UV="$HOME/.local/bin/uv"
PY="$HOME/.local/bin/python3.11"

ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*" | tee -a install.log; }

if [ ! -x "$UV" ]; then
    log "FATAL: uv not found at $UV"; exit 1
fi

log "=== uv install starting ==="
log "uv:     $($UV --version)"
log "python: $($PY --version)"

# ── 1. Create venvs ──────────────────────────────────────────
for venv in venv_iopaint venv_app; do
    if [ ! -d "$venv" ]; then
        log "Creating $venv with Python 3.11..."
        $UV venv "$venv" --python "$PY" 2>&1 | tail -3 | tee -a install.log
    fi
done

# ── 2. IOPaint venv ──────────────────────────────────────────
log "=== Installing PyTorch (stable cu124) + torch ecosystem ==="
log "    This is a ~3 GB download, may take 5-10 minutes."

VIRTUAL_ENV="$BASE_DIR/venv_iopaint" $UV pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -5 | tee -a install.log

log "=== Installing IOPaint + diffusers ecosystem ==="
VIRTUAL_ENV="$BASE_DIR/venv_iopaint" $UV pip install \
    iopaint diffusers transformers accelerate safetensors \
    opencv-python-headless Pillow requests \
    2>&1 | tail -5 | tee -a install.log

log "=== Verifying IOPaint + CUDA ==="
"$BASE_DIR/venv_iopaint/bin/python" -c "
import torch
print(f'    torch={torch.__version__}')
print(f'    cuda_available={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    device={torch.cuda.get_device_name(0)}')
import iopaint
print(f'    iopaint module loaded OK')
" 2>&1 | tee -a install.log

# ── 3. Flask app venv ────────────────────────────────────────
log "=== Installing Flask app deps ==="
VIRTUAL_ENV="$BASE_DIR/venv_app" $UV pip install \
    -r app/requirements.txt \
    2>&1 | tail -5 | tee -a install.log

# ── 4. ComfyUI ───────────────────────────────────────────────
if [ "${SKIP_COMFYUI:-0}" != "1" ] && [ ! -d "ComfyUI" ]; then
    log "=== Cloning ComfyUI ==="
    git clone --depth=1 https://github.com/comfyanonymous/ComfyUI.git ComfyUI \
        2>&1 | tail -3 | tee -a install.log

    log "=== ComfyUI venv + deps ==="
    $UV venv ComfyUI/venv --python "$PY" 2>&1 | tail -2 | tee -a install.log

    VIRTUAL_ENV="$BASE_DIR/ComfyUI/venv" $UV pip install \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124 \
        2>&1 | tail -3 | tee -a install.log

    VIRTUAL_ENV="$BASE_DIR/ComfyUI/venv" $UV pip install \
        -r ComfyUI/requirements.txt \
        2>&1 | tail -3 | tee -a install.log

    # Symlink shared model dirs
    mkdir -p models/{checkpoints,vae,loras}
    for d in checkpoints vae loras; do
        rm -rf "ComfyUI/models/$d"
        ln -sfn "$BASE_DIR/models/$d" "ComfyUI/models/$d"
    done
fi

log "=== Install complete ==="
log "Skipped: apt packages (sudo needed) and SD inpainting model (4 GB)."
log "Next: ./run.sh"
