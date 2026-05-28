"""
Image Text Editor — Flask Backend
Orchestrates: image upload → mask → IOPaint inpaint → text overlay → save
"""

import os
import uuid
import json
import logging
from pathlib import Path

# Load .env file (Groq API key, etc) before anything else
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass
from flask import Flask, request, jsonify, send_file, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inpaint import IOPaintClient
from text_overlay import TextOverlay
from comfyui_client import ComfyUIClient
from font_sampler import FontSampler
from auto_detector import AutoTextDetector
from blender import NaturalBlender
from batch_processor import BatchProcessor, DESKTOP

# ── Config ────────────────────────────────────────────────────
BASE_DIR    = Path(os.environ.get("BASE_DIR", Path(__file__).parent.parent))
UPLOAD_DIR  = BASE_DIR / "temp"
OUTPUT_DIR  = BASE_DIR / "output"
INPUT_DIR   = BASE_DIR / "input"

IOPAINT_URL  = os.environ.get("IOPAINT_URL",  "http://127.0.0.1:8080")
COMFYUI_URL  = os.environ.get("COMFYUI_URL",  "http://127.0.0.1:8188")

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

for d in (UPLOAD_DIR, OUTPUT_DIR, INPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

iopaint   = IOPaintClient(IOPAINT_URL)
text_tool = TextOverlay()
comfyui   = ComfyUIClient(COMFYUI_URL)
sampler   = FontSampler()
detector  = AutoTextDetector()
blender   = NaturalBlender()
batch_proc= BatchProcessor(
    iopaint_url = IOPAINT_URL,
    work_dir    = BASE_DIR / "temp" / "_batch",
    log_path    = DESKTOP / "text_replacement_log.json",
)


# ── Helpers ───────────────────────────────────────────────────
def _ext_ok(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


def _session_path(session_id: str) -> Path:
    p = UPLOAD_DIR / session_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def _serve_image(path: Path):
    """Return image as a response."""
    return send_file(path, mimetype="image/png")


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "iopaint": iopaint.health(),
        "comfyui": comfyui.health(),
    })


@app.post("/api/upload")
def upload():
    """Upload source image. Returns session_id + image URL."""
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400

    f = request.files["image"]
    if not f.filename or not _ext_ok(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    session_id = str(uuid.uuid4())
    dest = _session_path(session_id) / "source.png"

    # Save & convert to PNG for uniform handling
    from PIL import Image
    img = Image.open(f.stream).convert("RGBA")
    img.save(dest, "PNG")

    log.info("Uploaded image → session %s (%dx%d)", session_id, img.width, img.height)
    return jsonify({
        "session_id": session_id,
        "width": img.width,
        "height": img.height,
        "image_url": f"/api/session/{session_id}/source.png",
    })


@app.get("/api/session/<session_id>/<filename>")
def serve_session_file(session_id: str, filename: str):
    """Serve any file inside a session directory."""
    safe = secure_filename(filename)
    p = _session_path(session_id) / safe
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(p)


@app.post("/api/inpaint")
def inpaint():
    """
    Run IOPaint inpainting.
    Body (JSON):
      session_id   : str
      mask_data_url: str   (data:image/png;base64,…)
      model        : str   (lama | mat | zits | sd-inpainting)
      prompt       : str   (optional, only for SD model)
      step         : str   (optional 'source' → inpaint source, 'current' → inpaint latest)
    """
    data = request.get_json(force=True)
    session_id    = data.get("session_id")
    mask_data_url = data.get("mask_data_url", "")
    model         = data.get("model", "lama")
    prompt        = data.get("prompt", "seamless background, decorative pattern")
    source_type   = data.get("step", "source")

    if not session_id or not mask_data_url:
        return jsonify({"error": "session_id and mask_data_url required"}), 400

    sess = _session_path(session_id)

    # Determine which image to inpaint
    if source_type == "current" and (sess / "inpainted.png").exists():
        source_path = sess / "inpainted.png"
    else:
        source_path = sess / "source.png"

    if not source_path.exists():
        return jsonify({"error": "Source image not found"}), 404

    # Save the mask from data URL
    import base64, re
    from PIL import Image
    import io

    try:
        b64_data = re.sub(r"^data:image/[^;]+;base64,", "", mask_data_url)
        mask_bytes = base64.b64decode(b64_data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_path = sess / "mask.png"
        mask_img.save(mask_path)
    except Exception as e:
        return jsonify({"error": f"Bad mask data: {e}"}), 400

    # ── Sample original text BEFORE inpainting (so it's still visible) ──
    sample_info = None
    try:
        sample_info = sampler.sample(
            source_path=sess / "source.png",
            mask_path=mask_path,
            sample_out=sess / "text_sample.png",
        )
    except Exception as e:
        log.warning("Font sampling failed (continuing): %s", e)

    # Run inpainting
    log.info("Inpainting session %s with model=%s", session_id, model)
    out_path = sess / "inpainted.png"

    try:
        iopaint.inpaint(
            image_path=source_path,
            mask_path=mask_path,
            output_path=out_path,
            model=model,
            prompt=prompt,
        )
    except Exception as e:
        log.error("IOPaint error: %s", e)
        return jsonify({"error": f"Inpainting failed: {e}"}), 500

    return jsonify({
        "image_url":   f"/api/session/{session_id}/inpainted.png",
        "mask_url":    f"/api/session/{session_id}/mask.png",
        "sample_url":  f"/api/session/{session_id}/text_sample.png" if sample_info else None,
        "sample_info": sample_info,
    })


@app.post("/api/add-text")
def add_text():
    """
    Composite text onto the current image.
    Body (JSON):
      session_id  : str
      source      : 'inpainted' | 'source' | 'with_text'
      text        : str
      x, y        : int   (top-left of text box, 0..width/height)
      font_size   : int
      font_family : str   (font name or path)
      color       : str   (hex #rrggbb or rgba)
      bold        : bool
      italic      : bool
      stroke_width: int   (outline thickness, 0 = none)
      stroke_color: str
      align       : 'left' | 'center' | 'right'
      opacity     : float (0-1)
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)
    source_name = data.get("source", "inpainted")

    if source_name == "inpainted" and (sess / "inpainted.png").exists():
        src = sess / "inpainted.png"
    elif (sess / "with_text.png").exists():
        src = sess / "with_text.png"
    else:
        src = sess / "source.png"

    if not src.exists():
        return jsonify({"error": "Source image not found"}), 404

    # Optional fit_bbox: [x0, y0, x1, y1] — when given, auto-scale to fill
    fit_bbox = data.get("fit_bbox")
    if fit_bbox and isinstance(fit_bbox, list) and len(fit_bbox) == 4:
        fit_bbox = tuple(int(v) for v in fit_bbox)
    else:
        fit_bbox = None

    out_path = sess / "with_text.png"
    try:
        text_tool.add_text(
            source_path=src,
            output_path=out_path,
            text=data.get("text", ""),
            x=int(data.get("x", 50)),
            y=int(data.get("y", 50)),
            font_size=int(data.get("font_size", 48)),
            font_family=data.get("font_family", ""),
            color=data.get("color", "#000000"),
            bold=bool(data.get("bold", False)),
            italic=bool(data.get("italic", False)),
            stroke_width=int(data.get("stroke_width", 0)),
            stroke_color=data.get("stroke_color", "#ffffff"),
            align=data.get("align", "left"),
            opacity=float(data.get("opacity", 1.0)),
            fit_bbox=fit_bbox,
        )
    except Exception as e:
        log.error("Text overlay error: %s", e)
        return jsonify({"error": f"Text overlay failed: {e}"}), 500

    return jsonify({"image_url": f"/api/session/{session_id}/with_text.png"})


@app.post("/api/undo-text")
def undo_text():
    """Revert with_text.png back to the inpainted base."""
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)
    # Check what base to revert to
    if (sess / "inpainted.png").exists():
        return jsonify({"image_url": f"/api/session/{session_id}/inpainted.png"})
    return jsonify({"image_url": f"/api/session/{session_id}/source.png"})


@app.post("/api/save")
def save():
    """
    Copy final image to output/ with a meaningful name.
    Body (JSON):
      session_id: str
      filename  : str  (optional custom name)
    Returns download URL.
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)

    # Pick the most-processed version
    for candidate in ("with_text.png", "inpainted.png", "source.png"):
        p = sess / candidate
        if p.exists():
            final = p
            break
    else:
        return jsonify({"error": "No image to save"}), 404

    # Determine output filename
    custom = data.get("filename", "").strip()
    if custom:
        fname = secure_filename(custom)
        if not fname.lower().endswith(".png"):
            fname += ".png"
    else:
        fname = f"edited_{session_id[:8]}.png"

    out_path = OUTPUT_DIR / fname
    import shutil
    shutil.copy2(final, out_path)
    log.info("Saved → %s", out_path)

    return jsonify({
        "saved_path": str(out_path),
        "download_url": f"/api/download/{fname}",
        "filename": fname,
    })


@app.post("/api/compose-text")
def compose_text():
    """
    Composite a browser-rendered text layer onto the inpainted image.

    The browser renders the text using HTML5 Canvas (so the exact font/size
    shown as the ghost preview becomes the final output — no PIL/Canvas
    rendering mismatch).

    Body (JSON):
      session_id:           str
      text_layer_data_url:  data:image/png;base64,... (full-res RGBA text layer)
    """
    data = request.get_json(force=True)
    session_id     = data.get("session_id")
    text_layer_url = data.get("text_layer_data_url", "")

    if not session_id or not text_layer_url:
        return jsonify({"error": "session_id and text_layer_data_url required"}), 400

    sess = _session_path(session_id)

    # Pick base: prefer inpainted, fall back to source
    if (sess / "inpainted.png").exists():
        base_path = sess / "inpainted.png"
    else:
        base_path = sess / "source.png"

    if not base_path.exists():
        return jsonify({"error": "No base image — upload an image first"}), 404

    # Decode the data-URL text layer
    import base64, re, io
    from PIL import Image
    try:
        b64 = re.sub(r"^data:image/[^;]+;base64,", "", text_layer_url)
        layer_bytes = base64.b64decode(b64)
        text_layer = Image.open(io.BytesIO(layer_bytes)).convert("RGBA")
    except Exception as e:
        return jsonify({"error": f"Bad text_layer_data_url: {e}"}), 400

    # Composite
    base = Image.open(base_path).convert("RGBA")
    if text_layer.size != base.size:
        # Resize to match base if mismatched
        text_layer = text_layer.resize(base.size, Image.LANCZOS)
    out  = Image.alpha_composite(base, text_layer)

    out_path = sess / "with_text.png"
    out.convert("RGBA").save(out_path, "PNG")
    log.info("Composed text layer → %s", out_path)

    return jsonify({"image_url": f"/api/session/{session_id}/with_text.png"})


@app.get("/api/download/<filename>")
def download(filename: str):
    safe = secure_filename(filename)
    p = OUTPUT_DIR / safe
    if not p.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(p, as_attachment=True, download_name=safe)


@app.post("/api/comfyui/inpaint")
def comfyui_inpaint():
    """
    Run the SD-inpainting workflow via ComfyUI API.
    Body: same as /api/inpaint but routed through ComfyUI.
    """
    data = request.get_json(force=True)
    session_id    = data.get("session_id")
    mask_data_url = data.get("mask_data_url", "")
    prompt        = data.get("prompt", "")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)
    source_path = sess / "source.png"
    mask_path   = sess / "mask.png"

    if not source_path.exists():
        return jsonify({"error": "Upload an image first"}), 404

    # Save mask
    import base64, re, io
    from PIL import Image
    try:
        b64_data = re.sub(r"^data:image/[^;]+;base64,", "", mask_data_url)
        mask_bytes = base64.b64decode(b64_data)
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(mask_path)
    except Exception as e:
        return jsonify({"error": f"Bad mask: {e}"}), 400

    # Sample text first
    sample_info = None
    try:
        sample_info = sampler.sample(
            source_path=source_path,
            mask_path=mask_path,
            sample_out=sess / "text_sample.png",
        )
    except Exception as e:
        log.warning("Font sampling failed (continuing): %s", e)

    out_path = sess / "inpainted.png"
    try:
        comfyui.run_inpaint_workflow(
            image_path=source_path,
            mask_path=mask_path,
            output_path=out_path,
            prompt=prompt,
            workflow_path=BASE_DIR / "workflows" / "sd_inpaint.json",
        )
    except Exception as e:
        log.error("ComfyUI error: %s", e)
        return jsonify({"error": f"ComfyUI inpainting failed: {e}"}), 500

    return jsonify({
        "image_url":   f"/api/session/{session_id}/inpainted.png",
        "sample_url":  f"/api/session/{session_id}/text_sample.png" if sample_info else None,
        "sample_info": sample_info,
    })


@app.post("/api/sample-text")
def sample_text():
    """
    Standalone text sampling endpoint — analyzes the masked region of
    the source image to detect font properties (color, size, weight,
    skew, best matching font family).

    Body (JSON):
      session_id   : str
      mask_data_url: str   (data:image/png;base64,…)

    Returns the same sample_info dict that /api/inpaint returns,
    plus a sample_url for the cropped preview image.
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    mask_data_url = data.get("mask_data_url", "")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)
    source_path = sess / "source.png"
    if not source_path.exists():
        return jsonify({"error": "Source image not found"}), 404

    # Save mask
    import base64, re, io
    from PIL import Image
    try:
        b64 = re.sub(r"^data:image/[^;]+;base64,", "", mask_data_url)
        mask_img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        mask_path = sess / "mask.png"
        mask_img.save(mask_path)
    except Exception as e:
        return jsonify({"error": f"Bad mask: {e}"}), 400

    try:
        info = sampler.sample(
            source_path=source_path,
            mask_path=mask_path,
            sample_out=sess / "text_sample.png",
        )
    except Exception as e:
        log.error("Sampling failed: %s", e)
        return jsonify({"error": f"Sampling failed: {e}"}), 500

    return jsonify({
        "sample_url":  f"/api/session/{session_id}/text_sample.png",
        "sample_info": info,
    })


@app.get("/api/fonts")
def list_fonts():
    """Return list of available system fonts."""
    fonts = text_tool.list_fonts()
    return jsonify({"fonts": fonts})


# ── Static file shortcut (for output preview) ─────────────────
@app.get("/output/<path:filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


# ═══════════════════════════════════════════════════════════════
# ADVANCED TEXT REPLACEMENT — new endpoints
# ═══════════════════════════════════════════════════════════════

@app.post("/api/auto-detect")
def auto_detect():
    """
    Automatically detect all text regions in the session's source image.

    Body (JSON):
      session_id : str
      use_groq   : bool  (default true — use Groq vision for OCR)

    Returns:
      boxes        : [[x0,y0,x1,y1], …]  pixel-space bounding boxes
      mask_url     : URL to the generated mask PNG
      ocr          : ["text1", …]          per-box OCR strings (may be "")
      total_regions: int
      method       : "mser+groq" | "mser" | "gradient" | "groq" | "none"
    """
    data       = request.get_json(force=True)
    session_id = data.get("session_id")
    use_groq   = bool(data.get("use_groq", True))

    if not session_id:
        return jsonify({"error": "session_id required"}), 400

    sess = _session_path(session_id)
    src  = sess / "source.png"
    if not src.exists():
        return jsonify({"error": "Upload an image first"}), 404

    mask_out = sess / "auto_mask.png"
    try:
        result = detector.detect(
            image_path = src,
            mask_out   = mask_out,
            use_groq   = use_groq,
        )
    except Exception as e:
        log.error("Auto-detect error: %s", e)
        return jsonify({"error": f"Detection failed: {e}"}), 500

    return jsonify({
        "boxes":         result["boxes"],
        "mask_url":      f"/api/session/{session_id}/auto_mask.png" if mask_out.exists() else None,
        "ocr":           result["ocr"],
        "total_regions": result["total_regions"],
        "method":        result["method"],
        "image_size":    result["image_size"],
    })


@app.post("/api/auto-replace")
def auto_replace():
    """
    Full auto-replace pipeline in a single call (single image, session-based):
      1. Use the pre-computed mask from /api/auto-detect (or auto-detect now)
      2. Inpaint background
      3. Sample font for each region
      4. Render + naturally blend replacement text
      5. Save as with_text.png in the session

    Body (JSON):
      session_id       : str
      replacements     : [{"box":[x0,y0,x1,y1],"text":"…","color":"#hex","font_size":int}, …]
                         — one entry per detected region (use /api/auto-detect first)
      model            : str  (lama | mat | zits | sd-inpainting)
      prompt           : str
      font_size_override: int | null
      color_override   : str | null

    Returns:
      image_url: URL to the composited result
    """
    import base64, re, io as _io

    data             = request.get_json(force=True)
    session_id       = data.get("session_id")
    replacements     = data.get("replacements", [])
    model            = data.get("model", "lama")
    prompt           = data.get("prompt", "")
    font_size_ov     = data.get("font_size_override")
    color_ov         = data.get("color_override")

    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    if not replacements:
        return jsonify({"error": "replacements list required"}), 400

    sess      = _session_path(session_id)
    src       = sess / "source.png"
    mask_path = sess / "auto_mask.png"

    if not src.exists():
        return jsonify({"error": "Source image not found"}), 404

    from PIL import Image as _PIL_Image
    import numpy as _np

    src_img = _PIL_Image.open(src).convert("RGBA")
    W, H    = src_img.size

    # Build a combined mask from the replacement boxes
    m_arr = _np.zeros((H, W), dtype=_np.uint8)
    for rep in replacements:
        b = rep.get("box", [])
        if len(b) == 4:
            x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            m_arr[max(0,y0):min(H,y1), max(0,x0):min(W,x1)] = 255
    _PIL_Image.fromarray(m_arr).save(mask_path)

    # Inpaint
    inpainted = sess / "inpainted.png"
    try:
        iopaint.inpaint(
            image_path = src,
            mask_path  = mask_path,
            output_path= inpainted,
            model      = model,
            prompt     = prompt or "seamless background texture, no text",
            negative_prompt = "text, letters, watermark",
        )
    except Exception as e:
        log.error("Inpaint error in auto-replace: %s", e)
        import shutil as _shutil
        _shutil.copy2(src, inpainted)   # fallback

    # Render + blend each region
    current_base = inpainted
    results      = []

    for idx, rep in enumerate(replacements):
        b    = rep.get("box", [])
        text = str(rep.get("text", "")).strip()
        if len(b) != 4 or not text:
            continue

        x0, y0, x1, y1 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)

        # Per-region mask
        reg_mask = sess / f"reg_mask_{idx}.png"
        rm = _np.zeros((H, W), dtype=_np.uint8)
        rm[max(0,y0):min(H,y1), max(0,x0):min(W,x1)] = 255
        _PIL_Image.fromarray(rm).save(reg_mask)

        # Font sample from original source
        sample = None
        try:
            sample = sampler.sample(
                source_path = src,
                mask_path   = reg_mask,
                sample_out  = sess / f"adv_sample_{idx}.png",
            )
        except Exception as e:
            log.warning("Font sample failed region %d: %s", idx, e)

        font_family = sample.get("best_family", "") if sample else ""
        font_size   = font_size_ov or rep.get("font_size") or (
            sample.get("font_size", max(8, bh)) if sample else max(8, bh)
        )
        color       = color_ov or rep.get("color") or (
            sample.get("color", "#000000") if sample else "#000000"
        )
        bold        = sample.get("bold",   False) if sample else False
        italic      = sample.get("italic", False) if sample else False

        text_layer = sess / f"adv_layer_{idx}.png"
        merged_out = sess / f"adv_merged_{idx}.png"

        try:
            text_tool.add_text(
                source_path = current_base,
                output_path = text_layer,
                text        = text,
                x=x0, y=y0,
                font_size   = int(font_size),
                font_family = font_family,
                color       = color,
                bold        = bold,
                italic      = italic,
                stroke_width= 0,
                stroke_color= "#ffffff",
                align       = "left",
                opacity     = 1.0,
                fit_bbox    = (x0, y0, x1, y1),
            )
            blender.blend(
                base_path        = current_base,
                text_layer_path  = text_layer,
                mask_path        = reg_mask,
                output_path      = merged_out,
                feather_radius   = 2,
                noise_match      = True,
                brightness_match = True,
            )
            current_base = merged_out
            results.append({"region": b, "text": text, "status": "ok"})
        except Exception as e:
            log.error("Render/blend error region %d: %s", idx, e)
            results.append({"region": b, "text": text, "status": "error", "error": str(e)})

    # Copy final to with_text.png
    import shutil as _shutil2
    final_out = sess / "with_text.png"
    _shutil2.copy2(current_base, final_out)

    return jsonify({
        "image_url": f"/api/session/{session_id}/with_text.png",
        "results":   results,
    })


@app.post("/api/batch/process")
def batch_process():
    """
    Run batch replacement on all images in ~/Desktop (or a specified folder).

    Body (JSON):
      replacement_text    : str   — replacement for ALL detected text ("*" wildcard)
      replacement_map     : dict  — optional fine-grained map {"orig":"new", …}
      folder              : str   — optional folder path (default ~/Desktop)
      model               : str   — iopaint model (default "lama")
      prompt              : str
      font_size_override  : int | null
      color_override      : str | null

    Returns a summary dict with per-file results.
    This call may take several minutes — it runs synchronously (use AJAX + polling
    for the log endpoint to check progress).
    """
    data              = request.get_json(force=True)
    replacement_text  = data.get("replacement_text", "").strip()
    replacement_map   = data.get("replacement_map",  {})
    folder_str        = data.get("folder", str(DESKTOP))
    model             = data.get("model",  "lama")
    prompt            = data.get("prompt", "")
    font_size_ov      = data.get("font_size_override")
    color_ov          = data.get("color_override")

    if not replacement_text and not replacement_map:
        return jsonify({"error": "replacement_text or replacement_map required"}), 400

    folder = Path(folder_str).expanduser()
    if not folder.exists():
        return jsonify({"error": f"Folder not found: {folder}"}), 400

    # Build replacement map
    if replacement_text and not replacement_map:
        replacement_map = {"*": replacement_text}
    elif replacement_text and "*" not in replacement_map:
        replacement_map["*"] = replacement_text

    log.info("Batch processing %s with map=%s model=%s", folder, replacement_map, model)

    try:
        summary = batch_proc.process_folder(
            folder           = folder,
            replacement_map  = replacement_map,
            model            = model,
            prompt           = prompt,
            font_size_override = int(font_size_ov) if font_size_ov else None,
            color_override   = color_ov or None,
        )
    except Exception as e:
        log.error("Batch process error: %s", e)
        return jsonify({"error": f"Batch failed: {e}"}), 500

    return jsonify(summary)


@app.get("/api/batch/log")
def batch_log():
    """Return the contents of the JSON batch log file."""
    log_path = DESKTOP / "text_replacement_log.json"
    if not log_path.exists():
        return jsonify({"runs": [], "log_path": str(log_path)})
    try:
        data = json.loads(log_path.read_text("utf-8"))
        if not isinstance(data, list):
            data = [data]
        return jsonify({"runs": data, "log_path": str(log_path)})
    except Exception as e:
        return jsonify({"error": f"Could not read log: {e}"}), 500


@app.get("/api/batch/scan")
def batch_scan():
    """
    Scan ~/Desktop (or ?folder=…) for processable images.
    Returns their names and sizes — useful for previewing what will be processed.
    """
    folder_str = request.args.get("folder", str(DESKTOP))
    folder     = Path(folder_str).expanduser()
    if not folder.exists():
        return jsonify({"error": "Folder not found"}), 400

    from batch_processor import ALLOWED_EXT
    images = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT and not p.stem.endswith("_edited"):
            stat = p.stat()
            images.append({
                "name": p.name,
                "path": str(p),
                "size_kb": round(stat.st_size / 1024, 1),
            })

    return jsonify({
        "folder": str(folder),
        "images": images,
        "count":  len(images),
    })


# ═══════════════════════════════════════════════════════════════
# FIND & REPLACE WORD — browser-facing endpoint
# ═══════════════════════════════════════════════════════════════

@app.post("/api/replace-word")
def replace_word_endpoint():
    """
    Find a word in an uploaded image and replace it (no manual masking).

    Accepts multipart/form-data:
      image       : image file (PNG / JPG / WEBP / BMP / TIFF)
      pairs       : JSON string  [{"find": "...", "replace": "..."}, …]
      model       : "mat" | "lama" | "zits"  (default "mat")
      font        : optional font family name override
      auto_rotate : "true" | "false"         (default "true")

    Returns JSON:
      status      : "success" | "no_match"
      session_id  : str   (use /api/session/<id>/word_replaced.png to fetch)
      image_url   : str
      replaced    : int
      not_found   : [str, …]
      regions     : [{find, replace, bbox, font, color, bold, ocr}, …]
    """
    from word_replace import WordReplacer

    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    f = request.files["image"]
    if not f.filename or not _ext_ok(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    pairs_raw     = request.form.get("pairs", "[]")
    model         = request.form.get("model", "mat")
    font_override = request.form.get("font", "")
    auto_rotate   = request.form.get("auto_rotate", "true").lower() != "false"

    try:
        pairs_list   = json.loads(pairs_raw)
        replacements = {p["find"]: p.get("replace", "") for p in pairs_list if p.get("find")}
    except Exception as e:
        return jsonify({"error": f"Invalid pairs JSON: {e}"}), 400

    if not replacements:
        return jsonify({"error": "No find/replace pairs provided"}), 400

    # Save uploaded image to a fresh session
    session_id = str(uuid.uuid4())
    sess       = _session_path(session_id)

    from PIL import Image as _PIL
    try:
        img = _PIL.open(f.stream).convert("RGB")
        src = sess / "source.png"
        img.save(src)
    except Exception as e:
        return jsonify({"error": f"Could not read image: {e}"}), 400

    out_path = sess / "word_replaced.png"

    log.info("replace-word  session=%s  pairs=%s  model=%s", session_id, replacements, model)
    try:
        replacer = WordReplacer(IOPAINT_URL)
        result   = replacer.replace(
            image_path    = src,
            replacements  = replacements,
            out_path      = out_path,
            model         = model,
            font_override = font_override,
            auto_rotate   = auto_rotate,
        )
    except Exception as e:
        log.error("replace-word error: %s", e)
        return jsonify({"error": f"Replace failed: {e}"}), 500

    status = result.get("status")
    if status == "success":
        return jsonify({
            "status":     "success",
            "session_id": session_id,
            "image_url":  f"/api/session/{session_id}/word_replaced.png",
            "replaced":   result["replaced"],
            "not_found":  result.get("not_found", []),
            "regions":    result.get("regions", []),
        })
    elif status == "no_match":
        return jsonify({
            "status":    "no_match",
            "not_found": result.get("not_found", []),
            "detected":  result.get("detected", []),
        }), 422
    else:
        return jsonify({"error": "Unknown result", "detail": result}), 500


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting Image Text Editor on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
