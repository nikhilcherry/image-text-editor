#!/usr/bin/env bash
# Fix PyTorch for RTX 50-series (Blackwell) GPUs
# Run this if you see CUDA errors on RTX 5050/5060/5070/5080/5090
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Re-installing PyTorch nightly (cu128) for RTX 50-series..."

for VENV in venv_iopaint venv_app; do
    if [ -d "$BASE_DIR/$VENV" ]; then
        echo "  → $VENV"
        source "$BASE_DIR/$VENV/bin/activate"
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
        pip install --pre torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128 -q
        python3 -c "import torch; print('    CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
        deactivate
    fi
done

# Fix ComfyUI venv too
COMFYUI_VENV="$BASE_DIR/ComfyUI/venv"
if [ -d "$COMFYUI_VENV" ]; then
    echo "  → ComfyUI venv"
    source "$COMFYUI_VENV/bin/activate"
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128 -q
    python3 -c "import torch; print('    CUDA:', torch.cuda.is_available())"
    deactivate
fi

echo "Done. Run ./run.sh to start."
