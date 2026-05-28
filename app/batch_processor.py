"""
Batch Text Replacement Processor

Full pipeline for unattended processing of images from a folder:
  1. Scan folder for image files (skips *_edited.*)
  2. Auto-detect text regions using AutoTextDetector
  3. Build a combined mask → IOPaint inpaints the background
  4. For each region: FontSampler estimates original font properties
  5. SSAA-rendered replacement text is composited with NaturalBlender
  6. Result saved as <original>_edited.png in the SAME folder
  7. JSON log appended to ~/Desktop/text_replacement_log.json

Usage (from Flask route or CLI):
  from batch_processor import BatchProcessor
  bp = BatchProcessor()
  summary = bp.process_folder(
      folder=Path.home()/"Desktop",
      replacement_map={"*": "New Text"},
      model="lama",
  )
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from auto_detector import AutoTextDetector
from blender       import NaturalBlender
from font_sampler  import FontSampler
from inpaint       import IOPaintClient
from text_overlay  import TextOverlay

log = logging.getLogger(__name__)

DESKTOP     = Path.home() / "Desktop"
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


class BatchProcessor:
    """
    Orchestrates the full automated text-replacement pipeline.
    """

    def __init__(
        self,
        iopaint_url: str            = "http://127.0.0.1:8080",
        work_dir:    Optional[Path] = None,
        log_path:    Optional[Path] = None,
    ):
        self.iopaint  = IOPaintClient(iopaint_url)
        self.detector = AutoTextDetector()
        self.sampler  = FontSampler()
        self.overlay  = TextOverlay()
        self.blender  = NaturalBlender()

        self.work_dir = work_dir or (Path.home() / ".image_text_editor_tmp")
        self.log_path = log_path or (DESKTOP / "text_replacement_log.json")
        self.work_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def process_folder(
        self,
        folder:          Path,
        replacement_map: Dict[str, str],
        model:           str                                = "lama",
        prompt:          str                                = "",
        font_size_override: Optional[int]                  = None,
        color_override:  Optional[str]                     = None,
        progress_cb:     Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        """
        Process every image in folder.

        replacement_map examples:
          {"*": "New Text"}           — replace ALL detected text with one string
          {"Hello": "Hola"}           — replace only text matching 'Hello'
          {"Title": "New Title",
           "Subtitle": "New Sub"}     — multiple replacements

        font_size_override / color_override: manual fallbacks if detection fails.
        progress_cb(filename, fraction): optional progress callback.
        """
        images = self._scan_folder(folder)
        if not images:
            log.info("No images found in %s", folder)
            return self._summary([], model)

        total   = len(images)
        results = []

        for idx, img_path in enumerate(images):
            if progress_cb:
                progress_cb(img_path.name, idx / total)

            log.info("[%d/%d] Processing %s …", idx + 1, total, img_path.name)
            try:
                result = self.process_single(
                    img_path          = img_path,
                    replacement_map   = replacement_map,
                    model             = model,
                    prompt            = prompt,
                    font_size_override= font_size_override,
                    color_override    = color_override,
                )
                results.append(result)
            except Exception as exc:
                log.error("✗ %s: %s", img_path.name, exc, exc_info=True)
                results.append({
                    "file":    str(img_path),
                    "status":  "failed",
                    "error":   str(exc),
                    "regions": 0,
                })

        if progress_cb:
            progress_cb("done", 1.0)

        summary = self._summary(results, model)
        self._append_log(summary)
        return summary

    def process_single(
        self,
        img_path:           Path,
        replacement_map:    Dict[str, str],
        model:              str            = "lama",
        prompt:             str            = "",
        font_size_override: Optional[int]  = None,
        color_override:     Optional[str]  = None,
    ) -> dict:
        """
        Full pipeline for ONE image file. Returns a result dict.
        """
        sess = self.work_dir / str(uuid.uuid4())
        sess.mkdir(parents=True, exist_ok=True)

        # ── 1. Load + normalise ──────────────────────────────────────────
        src_img = Image.open(img_path).convert("RGBA")
        W, H    = src_img.size
        src     = sess / "source.png"
        src_img.save(src, "PNG")

        # ── 2. Auto-detect text regions ──────────────────────────────────
        mask_path = sess / "auto_mask.png"
        detection = self.detector.detect(
            image_path = src,
            mask_out   = mask_path,
        )
        boxes     = [tuple(b) for b in detection["boxes"]]
        ocr_texts = detection.get("ocr", [])

        if not boxes:
            log.info("No text detected in %s", img_path.name)
            return {
                "file":    str(img_path),
                "status":  "no_text",
                "regions": 0,
            }

        log.info("Detected %d text region(s) via %s", len(boxes), detection.get("method"))

        # ── 3. Inpaint background ────────────────────────────────────────
        inpainted = sess / "inpainted.png"
        try:
            self.iopaint.inpaint(
                image_path = src,
                mask_path  = mask_path,
                output_path= inpainted,
                model      = model,
                prompt     = prompt or "seamless background, original texture, no text",
                negative_prompt = "text, letters, words, watermark, blurry",
            )
        except Exception as exc:
            log.error("IOPaint failed: %s", exc)
            # Fallback: use source as the base (skip inpainting)
            shutil.copy2(src, inpainted)

        # ── 4. Render replacement text for each region ───────────────────
        current_base = inpainted
        region_results: List[dict] = []

        for idx, bbox in enumerate(boxes):
            x0, y0, x1, y1 = bbox
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)

            # 4a. Build single-region mask for font sampling
            region_mask = sess / f"region_{idx}_mask.png"
            m_arr = np.zeros((H, W), dtype=np.uint8)
            m_arr[y0:y1, x0:x1] = 255
            Image.fromarray(m_arr).save(region_mask)

            # 4b. Sample original font from PRE-inpainted source image
            sample = None
            try:
                sample = self.sampler.sample(
                    source_path = src,
                    mask_path   = region_mask,
                    sample_out  = sess / f"sample_{idx}.png",
                )
            except Exception as exc:
                log.warning("Font sampling failed for region %d: %s", idx, exc)

            # 4c. Decide replacement text
            original_text = (ocr_texts[idx] if idx < len(ocr_texts) else "").strip()
            replacement   = self._pick_replacement(original_text, replacement_map)

            if replacement is None:
                region_results.append({
                    "region":      list(bbox),
                    "original":    original_text,
                    "replacement": None,
                    "status":      "skipped",
                })
                continue

            # 4d. Choose font properties (sample → override → default)
            font_family = (sample or {}).get("best_family", "") if sample else ""
            font_size   = font_size_override or (
                (sample or {}).get("font_size", max(8, bh)) if sample else max(8, bh)
            )
            color       = color_override or (
                (sample or {}).get("color", "#000000") if sample else "#000000"
            )
            bold        = (sample or {}).get("bold",   False) if sample else False
            italic      = (sample or {}).get("italic", False) if sample else False

            # 4e. Render text with exact bbox fit (SSAA through text_overlay)
            text_layer   = sess / f"text_layer_{idx}.png"
            merged_out   = sess / f"merged_{idx}.png"

            try:
                self._render_and_blend(
                    base          = current_base,
                    text          = replacement,
                    bbox          = (x0, y0, x1, y1),
                    region_mask   = region_mask,
                    font_family   = font_family,
                    font_size     = font_size,
                    color         = color,
                    bold          = bold,
                    italic        = italic,
                    text_layer    = text_layer,
                    output        = merged_out,
                    img_w         = W,
                    img_h         = H,
                )
                current_base = merged_out
                region_results.append({
                    "region":      list(bbox),
                    "original":    original_text,
                    "replacement": replacement,
                    "font_size":   int(font_size),
                    "font_family": font_family,
                    "color":       color,
                    "bold":        bold,
                    "italic":      italic,
                    "status":      "replaced",
                })
            except Exception as exc:
                log.error("Render/blend failed for region %d: %s", idx, exc, exc_info=True)
                region_results.append({
                    "region":   list(bbox),
                    "original": original_text,
                    "status":   "render_failed",
                    "error":    str(exc),
                })

        # ── 5. Save _edited output ───────────────────────────────────────
        out_name = img_path.stem + "_edited.png"
        out_path = img_path.parent / out_name
        shutil.copy2(current_base, out_path)

        replaced = sum(1 for r in region_results if r.get("status") == "replaced")
        log.info("✓ %s → %s  (%d/%d regions replaced)",
                 img_path.name, out_name, replaced, len(boxes))

        return {
            "file":           str(img_path),
            "output_path":    str(out_path),
            "status":         "success",
            "regions":        len(boxes),
            "replaced":       replaced,
            "region_results": region_results,
        }

    # ── Render + blend helper ──────────────────────────────────────────────

    def _render_and_blend(
        self,
        base:        Path,
        text:        str,
        bbox:        Tuple[int, int, int, int],
        region_mask: Path,
        font_family: str,
        font_size:   int,
        color:       str,
        bold:        bool,
        italic:      bool,
        text_layer:  Path,
        output:      Path,
        img_w:       int,
        img_h:       int,
    ):
        """
        1. Render text at exact bbox position (auto-fit) onto a transparent layer.
        2. Use NaturalBlender to composite it onto the base with feathering + noise.
        """
        x0, y0, x1, y1 = bbox

        # Render text layer (transparent background, text at bbox coords)
        self.overlay.add_text(
            source_path = base,
            output_path = text_layer,
            text        = text,
            x           = x0,
            y           = y0,
            font_size   = font_size,
            font_family = font_family,
            color       = color,
            bold        = bold,
            italic      = italic,
            stroke_width= 0,
            stroke_color= "#ffffff",
            align       = "left",
            opacity     = 1.0,
            fit_bbox    = (x0, y0, x1, y1),   # ← exact original position + auto-fit
        )

        # Natural blend onto current base
        self.blender.blend(
            base_path        = base,
            text_layer_path  = text_layer,
            mask_path        = region_mask,
            output_path      = output,
            feather_radius   = 2,
            noise_match      = True,
            brightness_match = True,
        )

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _scan_folder(folder: Path) -> List[Path]:
        """Return image files, excluding already-processed *_edited.* files."""
        result = []
        for ext in ALLOWED_EXT:
            for p in folder.glob(f"*{ext}"):
                if not p.stem.endswith("_edited"):
                    result.append(p)
            for p in folder.glob(f"*{ext.upper()}"):
                if not p.stem.endswith("_edited"):
                    result.append(p)
        # Deduplicate (case-insensitive FS may return duplicates)
        seen = set()
        deduped = []
        for p in result:
            k = str(p).lower()
            if k not in seen:
                seen.add(k)
                deduped.append(p)
        return sorted(deduped)

    @staticmethod
    def _pick_replacement(
        original: str, replacement_map: Dict[str, str]
    ) -> Optional[str]:
        """
        Resolve what to replace `original` with.
          "*" key → wildcard, applies to everything
          exact key → exact match
          substring → case-insensitive containment
        Returns None → skip this region.
        """
        if not replacement_map:
            return None
        if "*" in replacement_map:
            return replacement_map["*"]
        if original in replacement_map:
            return replacement_map[original]
        orig_l = original.lower()
        for k, v in replacement_map.items():
            if k.lower() in orig_l or orig_l in k.lower():
                return v
        return None

    @staticmethod
    def _summary(results: List[dict], model: str) -> dict:
        return {
            "processed":  len(results),
            "succeeded":  sum(1 for r in results if r.get("status") == "success"),
            "no_text":    sum(1 for r in results if r.get("status") == "no_text"),
            "failed":     sum(1 for r in results if r.get("status") == "failed"),
            "timestamp":  datetime.now().isoformat(),
            "model":      model,
            "files":      results,
        }

    def _append_log(self, summary: dict):
        """Append this run's summary to the persistent JSON log."""
        existing: List[dict] = []
        if self.log_path.exists():
            try:
                raw = json.loads(self.log_path.read_text("utf-8"))
                existing = raw if isinstance(raw, list) else [raw]
            except Exception:
                existing = []

        existing.append(summary)
        self.log_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info("Log appended → %s", self.log_path)
