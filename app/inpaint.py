"""
IOPaint client — wraps the IOPaint HTTP API.

IOPaint runs as a separate server (port 8080) and accepts:
  POST /api/v1/inpaint
  { "image": "<base64>", "mask": "<base64>", ... }
→ returns the inpainted image as bytes (PNG).
"""

import base64
import io
import logging
import time
from pathlib import Path

import requests
from PIL import Image

log = logging.getLogger(__name__)


class IOPaintClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers["Accept"] = "image/png"

    # ── Health check ─────────────────────────────────────────
    def health(self) -> str:
        try:
            r = self._session.get(f"{self.base_url}/", timeout=3)
            return "ok" if r.status_code < 400 else f"http {r.status_code}"
        except Exception:
            return "unreachable"

    # ── Main inpaint ─────────────────────────────────────────
    def inpaint(
        self,
        image_path: Path,
        mask_path:  Path,
        output_path: Path,
        model: str   = "lama",
        prompt: str  = "seamless background texture",
        negative_prompt: str = "text, watermark, letters, words",
        sd_steps: int   = 40,
        sd_guidance: float = 7.5,
        sd_seed: int    = 42,
        sd_strength: float = 0.85,
        hd_strategy: str = "Crop",
        timeout: int = 120,
    ):
        """
        Call IOPaint and save the result to output_path.
        Retries once if the server is temporarily busy.
        """
        # Load & encode image + mask
        img_b64  = self._to_b64(image_path)
        mask_b64 = self._mask_to_b64(mask_path, image_path)

        payload = {
            "image": img_b64,
            "mask":  mask_b64,
            # HD strategy (avoid OOM on large images)
            "hd_strategy":                  hd_strategy,
            "hd_strategy_crop_margin":      196,
            "hd_strategy_crop_trigger_size": 1280,
            "hd_strategy_resize_limit":     2048,
            # SD params (ignored for non-SD models)
            "prompt":           prompt,
            "negative_prompt":  negative_prompt,
            "sd_steps":         sd_steps,
            "sd_guidance_scale": sd_guidance,
            "sd_seed":          sd_seed,
            "sd_strength":      sd_strength,
            "sd_sampler":       "uni_pc",
            "sd_mask_blur":     4,
            "sd_match_histograms": False,
        }

        url = f"{self.base_url}/api/v1/inpaint"
        for attempt in range(2):
            try:
                log.info("Calling IOPaint (attempt %d, model=%s)…", attempt + 1, model)
                r = self._session.post(url, json=payload, timeout=timeout)
                if r.status_code == 200:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(r.content)
                    log.info("Inpainted → %s (%d bytes)", output_path, len(r.content))
                    return
                else:
                    msg = r.text[:300]
                    log.warning("IOPaint returned %d: %s", r.status_code, msg)
                    if attempt == 0:
                        time.sleep(2)
            except requests.exceptions.Timeout:
                log.warning("IOPaint timed out (attempt %d)", attempt + 1)
                if attempt == 0:
                    time.sleep(3)
            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    "IOPaint server is not running. "
                    "Start it with ./run.sh or: source venv_iopaint/bin/activate && "
                    f"iopaint start --model={model} --port=8080"
                )

        raise RuntimeError("IOPaint inpainting failed after 2 attempts. Check temp/iopaint.log.")

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _to_b64(path: Path) -> str:
        """Load image → PNG bytes → base64 string."""
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _mask_to_b64(mask_path: Path, ref_path: Path) -> str:
        """
        Load mask, ensure it matches the source image dimensions,
        convert to grayscale (white = inpaint, black = keep).
        """
        ref  = Image.open(ref_path)
        mask = Image.open(mask_path).convert("L")

        if mask.size != ref.size:
            log.warning("Mask size %s != image size %s — resizing mask.", mask.size, ref.size)
            mask = mask.resize(ref.size, Image.NEAREST)

        buf = io.BytesIO()
        mask.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()
