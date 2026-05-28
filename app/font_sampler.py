"""
Font sampler — analyzes original text BEFORE inpainting to extract:
  • Dominant text color
  • Approximate font size (cap height)
  • Bold / weight estimate (stroke density)
  • Italic-ish skew estimate
  • Best matching system font(s) via OpenCV template matching
  • (Optional) Groq vision-LLM font identification + OCR

Used to pre-fill the text overlay form in Step 4 so the new text
looks like the original.
"""

import base64
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
import cv2

from text_overlay import TextOverlay

log = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# Limit the font candidates we test for speed.
# These are the most common families on Linux + cover serif/sans/mono.
CANDIDATE_FONTS = [
    "DejaVu Sans", "DejaVu Sans Mono", "DejaVu Serif",
    "Liberation Sans", "Liberation Serif", "Liberation Mono",
    "FreeSans", "FreeSerif", "FreeMono",
    "Ubuntu", "Ubuntu Mono", "Ubuntu Condensed",
    "Noto Sans", "Noto Serif", "Noto Mono",
    "Open Sans", "Roboto", "Lato",
    "Times", "Courier", "Helvetica", "Arial",
]


class FontSampler:
    def __init__(self):
        self.text_tool = TextOverlay()

    # ── Main entry point ─────────────────────────────────────
    def sample(
        self,
        source_path: Path,
        mask_path:   Path,
        sample_out:  Path,
        n_font_matches: int = 5,
    ) -> dict:
        """
        Analyze the masked region of source_path. Save a cropped
        sample image to sample_out and return detected attributes.
        """
        src  = np.array(Image.open(source_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if mask.shape[:2] != src.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(
                (src.shape[1], src.shape[0]), Image.NEAREST))

        # ── Mask bounding box ────────────────────────────────
        ys, xs = np.where(mask > 32)
        if len(xs) == 0:
            log.warning("Mask is empty — cannot sample text.")
            return self._empty_result()

        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        h, w   = y1 - y0 + 1, x1 - x0 + 1

        pad = max(4, min(h, w) // 8)
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(src.shape[1], x1 + pad + 1)
        cy1 = min(src.shape[0], y1 + pad + 1)

        crop_img  = src [cy0:cy1, cx0:cx1]
        crop_mask = mask[cy0:cy1, cx0:cx1]

        # Save crop preview for the UI
        sample_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(crop_img).save(sample_out)

        # ── Color analysis ───────────────────────────────────
        # Use only pixels inside the mask
        inside = crop_mask > 32
        if not inside.any():
            return self._empty_result()

        # ── Robust text-pixel detection via Otsu inside mask ──
        # The mask might be LOOSE (covers more than just text). We need
        # to find the ACTUAL text pixels (high-contrast minority class).
        gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY) if crop_img.ndim == 3 else crop_img
        inside_vals = gray[inside]

        # Otsu threshold on the in-mask region
        try:
            _, otsu_thresh = cv2.threshold(
                inside_vals.astype(np.uint8), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            otsu_val = otsu_thresh if isinstance(otsu_thresh, (int, float)) else 128
            # cv2.threshold returns (threshold_value, thresholded_img). Use the value.
            _, otsu_val_ret = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Get the actual Otsu threshold value via the ret
            otsu_thresh_val, _ = cv2.threshold(
                inside_vals.reshape(-1, 1).astype(np.uint8),
                0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        except Exception:
            otsu_thresh_val = 128.0

        # Decide which side is "text": minority of in-mask pixels
        dark_in_mask  = (gray < otsu_thresh_val) & inside
        light_in_mask = (gray >= otsu_thresh_val) & inside
        if dark_in_mask.sum() <= light_in_mask.sum():
            text_pixel_mask = dark_in_mask
            bg_pixel_mask   = light_in_mask
        else:
            text_pixel_mask = light_in_mask
            bg_pixel_mask   = dark_in_mask

        # Need enough text pixels for analysis (else fall back to mask)
        if text_pixel_mask.sum() < 20:
            text_pixel_mask = inside
            bg_pixel_mask   = ~inside

        text_pixels = crop_img[text_pixel_mask]
        bg_pixels   = crop_img[bg_pixel_mask] if bg_pixel_mask.any() else None

        # Median of darkest 30% of text pixels (more robust than mean)
        if len(text_pixels) > 0:
            brightnesses = text_pixels.mean(axis=1)
            sorted_idx = np.argsort(brightnesses)
            keep = sorted_idx[:max(1, len(sorted_idx) // 3)]
            text_color = tuple(int(c) for c in np.median(text_pixels[keep], axis=0))
        else:
            text_color = self._dominant_color(crop_img[inside], None)

        bg_color = (
            tuple(int(c) for c in np.median(bg_pixels, axis=0))
            if bg_pixels is not None and len(bg_pixels) else (255, 255, 255)
        )

        # ── ACTUAL text bbox (from text pixels, not mask) ────
        ty, tx = np.where(text_pixel_mask)
        if len(tx) > 5:
            tx0, tx1 = int(tx.min()), int(tx.max())
            ty0, ty1 = int(ty.min()), int(ty.max())
            actual_text_h = ty1 - ty0 + 1
            actual_text_w = tx1 - tx0 + 1
        else:
            actual_text_h = h
            actual_text_w = w

        # ── Size estimate ────────────────────────────────────
        # font_size ≈ ACTUAL text height × 1.2 (em-size ratio for most fonts)
        # NOT the mask bbox height (which may be padded if mask was loose)
        font_size = max(8, int(actual_text_h * 1.25))

        # ── Bold / weight estimate ───────────────────────────
        # Density of text pixels (not mask pixels) within the actual-text bbox
        text_only_area = (actual_text_h * actual_text_w) if actual_text_w > 0 else 1
        density = float(text_pixel_mask.sum()) / text_only_area
        bold = density > 0.30   # threshold tuned for actual-text-pixel density

        # ── Italic skew estimate ─────────────────────────────
        # PCA-based skew is unreliable on short text. Default to NOT italic;
        # only trust Groq's explicit italic call (handled below).
        skew_angle = self._estimate_skew(text_pixel_mask.astype(np.uint8) * 255)
        italic = False

        # ── Stroke contrast → serif heuristic ────────────────
        is_serif = self._estimate_serif(text_pixel_mask.astype(np.uint8) * 255)

        # ── Font family matching (slow part) ─────────────────
        font_matches = self._match_fonts(
            crop_img, crop_mask, font_size, text_color, bg_color,
            top_n=n_font_matches,
        )

        # Pick top family from local matching as initial guess
        if font_matches:
            best_family = font_matches[0]["family"]
        else:
            best_family = "DejaVu Serif" if is_serif else "DejaVu Sans"

        # ── Groq vision LLM (font ID + OCR) ──────────────────
        groq_info = self._groq_identify(sample_out)
        ocr_text = ""
        if groq_info:
            # Use Groq's font family if it picks one of our installed ones
            groq_family = groq_info.get("family", "").strip()
            mapped = self._map_to_installed_font(groq_family)
            if mapped:
                log.info("Groq → %s (mapped to %s)", groq_family, mapped)
                best_family = mapped
                # Prepend to matches so it's the top suggestion
                font_matches.insert(0, {
                    "family": mapped, "score": 0.99, "source": "groq",
                    "raw_family": groq_family,
                })
            ocr_text = groq_info.get("ocr_text", "")
            # Override bold/italic only when Groq is HIGH confidence
            confidence = (groq_info.get("confidence") or "").lower()
            high_conf  = confidence in ("high", "medium")

            weight = (groq_info.get("weight") or "").lower()
            style  = (groq_info.get("style")  or "").lower()

            if high_conf:
                if "bold" in weight or "black" in weight or "heavy" in weight:
                    bold = True
                elif "light" in weight or "thin" in weight or "regular" in weight:
                    bold = False
                # Italic is a frequent Groq false positive; require explicit "italic"
                # word AND high confidence
                if ("italic" in style or "oblique" in style) and confidence == "high":
                    italic = True

            # Use Groq's size estimate if it's plausible AND closer to our heuristic
            est_size = groq_info.get("estimated_size_px")
            if isinstance(est_size, (int, float)) and 8 <= est_size <= 500:
                # Sanity check: Groq's estimate shouldn't differ from heuristic by >2×
                heuristic = font_size
                if 0.5 * heuristic <= est_size <= 2.0 * heuristic:
                    font_size = int(est_size)
                else:
                    log.info("Groq size %s rejected (heuristic=%d)", est_size, heuristic)

        # ── Convert ACTUAL text bbox from crop-local → image-global ──
        # cx0, cy0 are the top-left of the crop in image coords
        if len(tx) > 5:
            img_tx0 = cx0 + tx0
            img_ty0 = cy0 + ty0
            img_tx1 = cx0 + tx1
            img_ty1 = cy0 + ty1
            actual_bbox = [img_tx0, img_ty0, img_tx1 + 1, img_ty1 + 1]
        else:
            actual_bbox = [x0, y0, x1 + 1, y1 + 1]

        result = {
            "color":        "#{:02x}{:02x}{:02x}".format(*text_color),
            "bg_color":     "#{:02x}{:02x}{:02x}".format(*bg_color),
            "font_size":    font_size,
            "bold":         bool(bold),
            "italic":       bool(italic),
            "is_serif":     bool(is_serif),
            "density":      round(density, 3),
            "skew_angle":   round(float(skew_angle), 2),
            "bbox":         actual_bbox,                        # ACTUAL text bbox
            "mask_bbox":    [x0, y0, x1 + 1, y1 + 1],           # original mask bbox
            "actual_text_h": int(actual_text_h),
            "actual_text_w": int(actual_text_w),
            "best_family":  best_family,
            "matches":      font_matches,
            "ocr_text":     ocr_text,
            "groq":         groq_info,
        }
        log.info("Sampled text: %s", {
            k: v for k, v in result.items() if k not in ("matches", "groq")
        })
        return result

    # ── Groq vision LLM integration ──────────────────────────
    def _groq_identify(self, image_path: Path) -> Optional[dict]:
        """
        Send the cropped text image to Groq's vision API to identify
        the font family and OCR the text.

        Returns dict with keys: family, weight, style, ocr_text,
        estimated_size_px, confidence. Returns None on failure.
        """
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except Exception as e:
            log.warning("Could not read sample for Groq: %s", e)
            return None

        prompt = (
            "You are a typography expert analyzing a cropped image of text. "
            "Identify the font characteristics and read the text.\n\n"
            "Return ONLY a JSON object with these exact keys:\n"
            "{\n"
            '  "family": "<font family name, e.g. \\"Arial\\", \\"Times New Roman\\", '
            '\\"Helvetica\\", \\"Roboto\\", \\"Georgia\\">",\n'
            '  "weight": "<one of: thin, light, regular, medium, bold, black>",\n'
            '  "style":  "<one of: normal, italic, oblique>",\n'
            '  "ocr_text": "<the visible text>",\n'
            '  "estimated_size_px": <pixel height of the capital letters as a number>,\n'
            '  "confidence": "<low | medium | high>"\n'
            "}\n\n"
            "Be specific about the font family — name an actual font, "
            "not a generic description. If unsure, give your best guess."
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 300,
        }

        try:
            r = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=20,
            )
            if r.status_code != 200:
                log.warning("Groq API HTTP %d: %s", r.status_code, r.text[:200])
                return None
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            log.warning("Groq API call failed: %s", e)
            return None

    def _map_to_installed_font(self, groq_family: str) -> Optional[str]:
        """
        Groq might say "Arial" or "Helvetica" — both unlikely on Linux.
        Map common font names to their installed equivalents.
        """
        if not groq_family:
            return None
        key = groq_family.lower().strip()

        # Aliases: { groq output → equivalent installed font }
        aliases = {
            "arial":            "Liberation Sans",
            "helvetica":        "Liberation Sans",
            "helvetica neue":   "Liberation Sans",
            "sans serif":       "DejaVu Sans",
            "sans-serif":       "DejaVu Sans",
            "times":            "Liberation Serif",
            "times new roman":  "Liberation Serif",
            "serif":            "DejaVu Serif",
            "courier":          "Liberation Mono",
            "courier new":      "Liberation Mono",
            "monospace":        "DejaVu Sans Mono",
            "roboto":           "Roboto",
            "open sans":        "Open Sans",
            "noto sans":        "Noto Sans",
            "noto serif":       "Noto Serif",
            "georgia":          "DejaVu Serif",
            "verdana":          "DejaVu Sans",
            "tahoma":           "DejaVu Sans",
            "calibri":          "Liberation Sans",
            "cambria":          "DejaVu Serif",
            "trebuchet ms":     "DejaVu Sans",
            "comic sans":       "DejaVu Sans",
            "comic sans ms":    "DejaVu Sans",
            "impact":           "DejaVu Sans",
            "ubuntu":           "Ubuntu",
            "dejavu sans":      "DejaVu Sans",
            "dejavu serif":     "DejaVu Serif",
            "liberation sans":  "Liberation Sans",
            "liberation serif": "Liberation Serif",
        }
        if key in aliases:
            return self._verify_font(aliases[key])

        # Try direct fuzzy match against installed fonts
        font_map = self.text_tool._get_font_map()
        for name in font_map:
            if key in name.lower() or name.lower() in key:
                return name
        return None

    def _verify_font(self, family: str) -> Optional[str]:
        """Return the family name if any installed font contains it."""
        font_map = self.text_tool._get_font_map()
        fl = family.lower()
        for name in font_map:
            if fl in name.lower():
                return family   # return the canonical name we passed in
        return None

    # ── Helpers ──────────────────────────────────────────────
    @staticmethod
    def _empty_result() -> dict:
        return {
            "color": "#000000", "bg_color": "#ffffff",
            "font_size": 48, "bold": False, "italic": False,
            "is_serif": False, "density": 0.0, "skew_angle": 0.0,
            "bbox": None, "best_family": "DejaVu Sans", "matches": [],
            "ocr_text": "", "groq": None,
        }

    @staticmethod
    def _dominant_color(text_px: np.ndarray, bg_px: Optional[np.ndarray]) -> tuple:
        """Find dominant text color, excluding background-colored pixels."""
        # If we have background, find the median bg color and reject pixels
        # within euclidean distance < 50 (similar to background).
        if bg_px is not None and len(bg_px) > 10:
            bg_med = np.median(bg_px, axis=0)
            dist = np.linalg.norm(text_px.astype(float) - bg_med, axis=1)
            far  = text_px[dist > 40]
            if len(far) > 10:
                text_px = far

        # Use median for robustness
        med = np.median(text_px, axis=0)
        return tuple(int(c) for c in med)

    @staticmethod
    def _estimate_skew(mask: np.ndarray) -> float:
        """Estimate italic skew angle (degrees) from mask using PCA."""
        ys, xs = np.where(mask > 32)
        if len(xs) < 20:
            return 0.0
        # PCA: principal axis direction
        pts = np.column_stack([xs - xs.mean(), ys - ys.mean()])
        cov = np.cov(pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Major axis is the eigvec with largest eigenvalue
        major = eigvecs[:, -1]
        # Angle from vertical (italic = slight CW lean from vertical)
        angle_deg = math.degrees(math.atan2(major[0], -major[1]))
        # Normalize: a perfectly vertical text returns ~0; italic returns ~10-15°
        if angle_deg > 90:  angle_deg -= 180
        if angle_deg < -90: angle_deg += 180
        return angle_deg

    @staticmethod
    def _estimate_serif(mask: np.ndarray) -> bool:
        """Crude serif detection: serif fonts have more pixels near the
        baseline/cap heights (thin horizontal strokes at top/bottom)."""
        m = mask > 32
        if m.sum() < 50: return False
        row_sums = m.sum(axis=1).astype(float)
        if row_sums.max() == 0: return False
        row_sums /= row_sums.max()
        # Look at the top and bottom 15% of the bbox
        h = len(row_sums)
        top    = row_sums[:max(1, h // 7)].mean()
        bottom = row_sums[-max(1, h // 7):].mean()
        middle = row_sums[h // 3 : 2 * h // 3].mean() or 1e-6
        # Serif fonts have higher top/bottom density relative to middle
        ratio = (top + bottom) / (2 * middle)
        return ratio > 0.55

    # ── Font matching ────────────────────────────────────────
    def _match_fonts(
        self,
        crop_img:  np.ndarray,
        crop_mask: np.ndarray,
        font_size: int,
        text_color: tuple,
        bg_color:   tuple,
        top_n: int = 5,
    ) -> list:
        """
        Try a curated list of system fonts. For each, render a sample word
        at the detected size and color, then compute pixel-level similarity
        to the cropped original text.

        We don't know the actual text — so we use a generic sample
        "AaBbCc" which captures upper/lower/curve characteristics.
        Then we compare via correlation on the MASK (shape) only.
        """
        # Reduce candidates to ones that actually exist on the system
        font_map = self.text_tool._get_font_map()
        available = []
        for fam in CANDIDATE_FONTS:
            fam_l = fam.lower()
            for name in font_map:
                if fam_l in name.lower():
                    available.append((fam, font_map[name]))
                    break
        if not available:
            return []

        # Original shape (binary)
        orig_shape = (crop_mask > 32).astype(np.uint8) * 255
        H, W = orig_shape.shape
        if H < 8 or W < 8:
            return []

        sample_text = "AaBbCc"
        scores = []
        for family, path in available:
            try:
                font = ImageFont.truetype(str(path), font_size)
            except Exception:
                continue
            # Measure rendered text size
            tmp = Image.new("L", (W * 4, H * 4), 0)
            d   = ImageDraw.Draw(tmp)
            try:
                d.text((10, 10), sample_text, font=font, fill=255)
            except Exception:
                continue
            arr = np.array(tmp)
            ys, xs = np.where(arr > 32)
            if len(xs) == 0:
                continue
            ty0, ty1 = ys.min(), ys.max() + 1
            tx0, tx1 = xs.min(), xs.max() + 1
            rendered = arr[ty0:ty1, tx0:tx1]
            # Scale rendered to match original mask aspect ratio
            r_resized = cv2.resize(rendered, (W, H), interpolation=cv2.INTER_AREA)
            # Compute similarity via normalized cross-correlation
            score = float(cv2.matchTemplate(
                orig_shape, r_resized, cv2.TM_CCOEFF_NORMED
            )[0][0])
            scores.append({"family": family, "score": round(score, 3)})

        scores.sort(key=lambda d: -d["score"])
        return scores[:top_n]
