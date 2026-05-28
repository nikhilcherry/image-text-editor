#!/usr/bin/env bash
# Download AI models for inpainting
# Models are shared between IOPaint (auto-download) and ComfyUI (manual)
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$BASE_DIR/models"
CKPT_DIR="$MODELS_DIR/checkpoints"

info()    { echo -e "\033[1;34m[Models]\033[0m $*"; }
success() { echo -e "\033[1;32m[Models]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[Models]\033[0m $*"; }

mkdir -p "$CKPT_DIR" "$MODELS_DIR/vae" "$MODELS_DIR/loras" "$MODELS_DIR/iopaint_cache"

# ── IOPaint models (downloaded automatically on first use) ────
# lama     → good for seamless background fill  (~100 MB)
# mat      → good for complex backgrounds       (~200 MB)
# zits     → good for thin text strokes         (~400 MB)
# sd-inpainting → full Stable Diffusion 1.5 inpaint (~4 GB)
#
# We pre-warm the 'lama' model so the first run is instant:

info "Pre-warming LaMa model via IOPaint (this may take a few minutes)..."
if source "$BASE_DIR/venv_iopaint/bin/activate" 2>/dev/null; then
    python3 - <<'EOF'
import os, sys
os.environ["XDG_CACHE_HOME"] = os.path.join(
    os.environ.get("BASE_DIR", "."), "models", "iopaint_cache"
)
try:
    from iopaint.model_manager import ModelManager
    print("[Models] LaMa model pre-cached.")
except Exception as e:
    print(f"[Models] Pre-warm skipped (will download on first run): {e}")
EOF
    deactivate
fi

# ── ComfyUI: SD 1.5 Inpainting checkpoint ────────────────────
SD_INPAINT_FILE="$CKPT_DIR/sd-v1-5-inpainting.ckpt"
SD_INPAINT_URL="https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"

if [ ! -f "$SD_INPAINT_FILE" ]; then
    info "Downloading SD 1.5 Inpainting (~4 GB)..."
    warn "Tip: This is optional. IOPaint's 'lama' mode works without it."
    warn "Skip with Ctrl-C and use IOPaint-only mode."
    sleep 3

    if command -v aria2c &>/dev/null; then
        aria2c -x 8 -s 8 -d "$CKPT_DIR" -o "sd-v1-5-inpainting.ckpt" "$SD_INPAINT_URL"
    else
        wget --progress=bar:force -O "$SD_INPAINT_FILE" "$SD_INPAINT_URL"
    fi
    success "SD Inpainting model downloaded."
else
    success "SD 1.5 Inpainting already present."
fi

echo ""
success "Model setup complete."
echo ""
echo "  Available IOPaint models (set IOPAINT_MODEL env var before run.sh):"
echo "    lama           — fast, great for text removal"
echo "    mat            — detailed backgrounds"
echo "    zits           — thin strokes"
echo "    sd-inpainting  — Stable Diffusion (needs ~4 GB model above)"
echo ""
echo "  Example:  IOPAINT_MODEL=sd-inpainting ./run.sh"
