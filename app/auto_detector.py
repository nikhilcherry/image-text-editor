"""
Auto Text Detector
Automatically identifies and masks all text regions in an image.

Pipeline:
  1. MSER (Maximally Stable Extremal Regions) — fast, no model needed
  2. Morphological gradient fallback — catches decorative/outlined text
  3. Groq vision LLM — verifies regions + provides OCR text
  4. Union-find box merging → single clean mask PNG

Returns a list of bounding boxes [x0, y0, x1, y1] and a binary mask.
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import requests
from PIL import Image

log = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


class AutoTextDetector:
    """
    Detects text regions in images without any manual input.
    Combines classical CV (MSER + morphological) with optional Groq vision.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        image_path: Path,
        mask_out:   Optional[Path] = None,
        padding:    int  = 6,
        min_area:   int  = 60,
        use_groq:   bool = True,
    ) -> dict:
        """
        Detect all text regions in image_path.

        Returns:
          {
            "boxes":         [[x0,y0,x1,y1], …],  # pixel coords, image-space
            "mask_path":     str | None,            # path to white-on-black PNG
            "ocr":           [str, …],              # per-box OCR (Groq) or ""
            "total_regions": int,
            "image_size":    [W, H],
            "method":        "mser+groq" | "mser" | "gradient+groq" | "gradient" | "groq"
          }
        """
        img_pil = Image.open(image_path).convert("RGB")
        img     = np.array(img_pil)
        H, W    = img.shape[:2]

        # ── Stage 1: classical CV ──────────────────────────────────────────
        # Solid-ink projection detector first — robust on textured art/photos
        # where MSER/gradient drown in false regions. Falls back to MSER, then
        # morphological gradient, only if it finds nothing.
        solid_boxes = self._solid_text_detect(img, padding)
        mser_boxes  = [] if solid_boxes else self._mser_detect(img, padding, min_area)
        morph_boxes = []
        if not solid_boxes and not mser_boxes:
            morph_boxes = self._gradient_detect(img, padding)

        cv_boxes = solid_boxes or mser_boxes or morph_boxes
        method   = ("solid" if solid_boxes else
                    "mser"  if mser_boxes  else
                    "gradient" if morph_boxes else "")

        # ── Stage 2: Groq vision (primary OCR + verification) ─────────────
        groq_boxes: List[Tuple[int,int,int,int]] = []
        groq_texts: List[str] = []
        if use_groq:
            groq_boxes, groq_texts = self._groq_detect(image_path, W, H)
            if groq_boxes:
                method += ("+groq" if method else "groq")

        # ── Choose geometry source ─────────────────────────────────────────
        # Drop boxes that cover most of the image first, so a single bad
        # whole-image box can't swallow good ones during the union.
        cv_boxes = self._sanity_filter(cv_boxes, W, H)

        if cv_boxes:
            # CV gave precise pixel boxes — trust them. Groq's coordinates are
            # often imprecise, so use Groq ONLY for OCR text (mapped below),
            # not for geometry.
            merged = self._merge_boxes(cv_boxes, W, H, gap=10)
        else:
            # No CV detection — fall back to Groq's boxes as the geometry.
            merged = self._merge_boxes(
                self._sanity_filter(groq_boxes, W, H), W, H, gap=10
            )
        merged = self._sanity_filter(merged, W, H)

        # Align OCR texts to merged boxes (best-effort: Groq texts may not
        # line up 1-to-1 after merging, so map by largest-overlap heuristic)
        ocr_map   = self._align_ocr(groq_boxes, groq_texts, merged)

        # ── Build mask ─────────────────────────────────────────────────────
        mask = np.zeros((H, W), dtype=np.uint8)
        for (x0, y0, x1, y1) in merged:
            mask[y0:y1, x0:x1] = 255

        if mask_out:
            mask_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(mask_out)

        return {
            "boxes":         [list(b) for b in merged],
            "mask_path":     str(mask_out) if mask_out else None,
            "ocr":           [ocr_map.get(i, "") for i in range(len(merged))],
            "total_regions": len(merged),
            "image_size":    [W, H],
            "method":        method or "none",
        }

    # ── MSER detection ─────────────────────────────────────────────────────

    def _mser_detect(
        self, img: np.ndarray, padding: int, min_area: int
    ) -> List[Tuple[int,int,int,int]]:
        """
        MSER finds text-like blobs efficiently.
        We run it on the raw image and a Canny-edge image for robustness.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        H, W  = gray.shape

        # OpenCV 4.x: positional args (delta, min_area, max_area, max_variation, …)
        # We avoid kwargs because the API renamed `_delta`→`delta` between versions.
        try:
            mser = cv2.MSER_create(
                4,                                  # delta
                min_area,                           # min_area
                max(min_area * 300, H * W // 6),    # max_area
                0.30,                               # max_variation
            )
        except TypeError:
            # Newer OpenCV (≥ 4.7) uses kwargs without the underscore prefix
            mser = cv2.MSER_create(
                delta         = 4,
                min_area      = min_area,
                max_area      = max(min_area * 300, H * W // 6),
                max_variation = 0.30,
            )

        all_boxes: List[Tuple[int,int,int,int]] = []

        for source in (gray, cv2.equalizeHist(gray)):
            try:
                regions, _ = mser.detectRegions(source)
            except Exception:
                continue
            for region in regions:
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                if w > W * 0.85 or h > H * 0.70:
                    continue
                ar = w / max(h, 1)
                if ar < 0.15 or ar > 25:
                    continue
                if w < 6 or h < 6:
                    continue
                x0 = max(0, x - padding)
                y0 = max(0, y - padding)
                x1 = min(W, x + w + padding)
                y1 = min(H, y + h + padding)
                all_boxes.append((x0, y0, x1, y1))

        if not all_boxes:
            return []

        # First-pass merge (tight gap = 2 px) to collapse individual characters
        merged = self._merge_boxes(all_boxes, W, H, gap=2)
        # Second-pass merge (larger gap = 8 px) to group words/lines
        merged = self._merge_boxes(merged, W, H, gap=8)

        # Filter: keep only boxes that look like text lines
        # (aspect ratio or height consistent with text)
        filtered = []
        for (x0, y0, x1, y1) in merged:
            bw, bh = x1 - x0, y1 - y0
            if bw < 8 or bh < 6:
                continue
            # Text usually has height < 30% of image height
            if bh > H * 0.35:
                continue
            filtered.append((x0, y0, x1, y1))

        return filtered

    # ── Gradient/morphological detection ──────────────────────────────────

    def _gradient_detect(
        self, img: np.ndarray, padding: int
    ) -> List[Tuple[int,int,int,int]]:
        """
        Gradient-based approach: works well for solid/decorative text.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        H, W  = gray.shape

        # Horizontal morphological gradient → highlights vertical text strokes
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k_h)
        _, thresh = cv2.threshold(
            grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )

        # Connect characters into word-level blobs
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(
            connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 8:
                continue
            ar = w / max(h, 1)
            if ar < 0.4 or ar > 35:
                continue
            if h > H * 0.35:
                continue
            # Text density check
            roi = thresh[y:y+h, x:x+w]
            density = float(roi.mean()) / 255.0
            if density < 0.04 or density > 0.92:
                continue
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(W, x + w + padding)
            y1 = min(H, y + h + padding)
            boxes.append((x0, y0, x1, y1))

        return self._merge_boxes(boxes, W, H, gap=12)

    # ── Solid-ink projection detection ─────────────────────────────────────

    def _solid_text_detect(
        self, img: np.ndarray, padding: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Robust detector for solid, high-contrast text (black/white/strong) on
        busy or textured backgrounds — the case where MSER and gradient drown
        in thousands of false regions (watercolor, gradients, photos).

        Idea: real text is made of *solid extreme-value* strokes that form a
        horizontal band. Decorative line-art (leaves, sketches) is thin and
        mid-tone, so it mostly fails an absolute extreme-value test.

          1. Build an "ink" mask for each polarity (near-black, near-white).
          2. Locate text bands via the horizontal projection profile, using
             hysteresis (high threshold to find a band, low to grow it).
          3. Measure each band's true extent from the *raw* ink (so thin serifs
             at the line edges aren't eroded away).
        """
        arr = img.astype(np.int32)
        H, W = arr.shape[:2]
        ch_min = arr.min(axis=2)
        ch_max = arr.max(axis=2)

        results: List[Tuple[int, int, int, int]] = []

        for polarity in ("dark", "light"):
            raw = (ch_max < 70) if polarity == "dark" else (ch_min > 200)
            if raw.sum() < W * 0.5:          # almost no ink of this polarity
                continue

            # Opened mask de-speckles the projection profile (band *location*).
            opened = cv2.morphologyEx(
                (raw.astype(np.uint8)) * 255,
                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
            ) > 0
            row = opened.sum(axis=1).astype(np.float32)
            if row.max() < W * 0.02:
                continue

            hi = row.max() * 0.25
            lo = row.max() * 0.10
            y = 0
            while y < H:
                if row[y] > hi:
                    y0 = y1 = y
                    while y0 > 0 and row[y0 - 1] > lo:
                        y0 -= 1
                    while y1 < H - 1 and row[y1 + 1] > lo:
                        y1 += 1

                    # Measure extent in an expanded window from RAW ink, so
                    # thin serifs / ascenders / descenders are preserved.
                    ph  = (y1 - y0)
                    wy0 = max(0, y0 - ph)
                    wy1 = min(H, y1 + ph)
                    win = raw[wy0:wy1, :]
                    rr  = win.sum(axis=1)
                    cc  = win.sum(axis=0)
                    ys  = np.where(rr > max(6, W * 0.012))[0]   # rows: solid only
                    xs  = np.where(cc >= 3)[0]                  # cols: inclusive

                    if ys.size and xs.size:
                        fx0, fx1 = int(xs.min()), int(xs.max())
                        fy0 = wy0 + int(ys.min())
                        fy1 = wy0 + int(ys.max())
                        bw, bh = fx1 - fx0, fy1 - fy0
                        if (bw > bh and bw > W * 0.03 and
                                10 < bh < H * 0.45 and bw * bh < 0.55 * W * H):
                            # small safety pad (esp. to catch descenders)
                            px = padding
                            py = max(padding, bh // 12)
                            results.append((
                                max(0, fx0 - px), max(0, fy0 - py),
                                min(W, fx1 + 1 + px), min(H, fy1 + 1 + py),
                            ))
                    y = y1 + 1
                else:
                    y += 1

        # Merge overlapping dark/light detections of the same line.
        merged = self._merge_boxes(results, W, H, gap=6)
        return self._sanity_filter(merged, W, H)

    # ── Size sanity filter ─────────────────────────────────────────────────

    @staticmethod
    def _sanity_filter(
        boxes: List[Tuple[int, int, int, int]], W: int, H: int
    ) -> List[Tuple[int, int, int, int]]:
        """
        Reject boxes that are clearly not a single text line/region:
          • cover > 60% of the image area, or
          • span nearly the full width AND are very tall (whole-image grabs).
        """
        out = []
        for (x0, y0, x1, y1) in boxes:
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0:
                continue
            area_frac = (bw * bh) / float(W * H)
            if area_frac > 0.60:
                continue
            if bw > W * 0.92 and bh > H * 0.55:
                continue
            out.append((x0, y0, x1, y1))
        return out

    # ── Groq vision detection + OCR ────────────────────────────────────────

    def _groq_detect(
        self, image_path: Path, W: int, H: int
    ) -> Tuple[List[Tuple[int,int,int,int]], List[str]]:
        """
        Ask Groq to localise every piece of text in the image.
        Returns (boxes, texts) where texts are the OCR'd strings.
        """
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            return [], []

        try:
            raw = image_path.read_bytes()
            img_b64 = base64.b64encode(raw).decode()
        except Exception as e:
            log.warning("Could not read image for Groq: %s", e)
            return [], []

        prompt = (
            f"Analyze this {W}×{H}-pixel image and detect ALL visible text.\n"
            "For every distinct text region (word, phrase, heading, watermark, label, caption), "
            "output its pixel bounding box and the exact text it contains.\n\n"
            "Return ONLY valid JSON:\n"
            '{"regions": [{"x0":int,"y0":int,"x1":int,"y1":int,"text":"..."},...]}\n\n'
            "Rules:\n"
            "- Coordinates are top-left origin, in pixels\n"
            "- x0 < x1, y0 < y1\n"
            "- Include ALL text (printed, handwritten, watermarks, UI labels)\n"
            "- If no text is visible, return {\"regions\":[]}"
        )

        payload = {
            "model": GROQ_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text",      "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens":  1200,
        }

        try:
            r = requests.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
                json=payload,
                timeout=30,
            )
            if r.status_code != 200:
                log.warning("Groq auto-detect HTTP %d: %s", r.status_code, r.text[:200])
                return [], []

            data   = json.loads(r.json()["choices"][0]["message"]["content"])
            regs   = data.get("regions", [])
            boxes, texts = [], []

            for reg in regs:
                x0 = max(0,  int(reg.get("x0", 0)))
                y0 = max(0,  int(reg.get("y0", 0)))
                x1 = min(W,  int(reg.get("x1", W)))
                y1 = min(H,  int(reg.get("y1", H)))
                if x1 > x0 + 4 and y1 > y0 + 4:
                    boxes.append((x0, y0, x1, y1))
                    texts.append(str(reg.get("text", "")).strip())

            log.info("Groq detected %d text regions", len(boxes))
            return boxes, texts

        except Exception as e:
            log.warning("Groq auto-detect failed: %s", e)
            return [], []

    # ── Box merging ────────────────────────────────────────────────────────

    @staticmethod
    def _merge_boxes(
        boxes: List[Tuple[int,int,int,int]],
        W: int, H: int,
        gap: int = 8,
    ) -> List[Tuple[int,int,int,int]]:
        """
        Greedy union-find: merge boxes that overlap or are within `gap` pixels.
        Runs until stable (no more merges).
        """
        if not boxes:
            return []

        rects = [list(b) for b in boxes]
        changed = True

        while changed:
            changed = False
            used = [False] * len(rects)
            out  = []

            for i, r1 in enumerate(rects):
                if used[i]:
                    continue
                cur = r1[:]
                for j in range(i + 1, len(rects)):
                    if used[j]:
                        continue
                    r2 = rects[j]
                    # Expand by gap and check overlap
                    if (cur[0] - gap < r2[2] and cur[2] + gap > r2[0] and
                            cur[1] - gap < r2[3] and cur[3] + gap > r2[1]):
                        cur[0] = min(cur[0], r2[0])
                        cur[1] = min(cur[1], r2[1])
                        cur[2] = max(cur[2], r2[2])
                        cur[3] = max(cur[3], r2[3])
                        used[j] = True
                        changed  = True
                out.append(cur)
                used[i] = True

            rects = out

        result = []
        for r in rects:
            x0 = max(0, r[0])
            y0 = max(0, r[1])
            x1 = min(W, r[2])
            y1 = min(H, r[3])
            if x1 > x0 + 2 and y1 > y0 + 2:
                result.append((x0, y0, x1, y1))
        return result

    # ── OCR alignment helper ────────────────────────────────────────────────

    @staticmethod
    def _align_ocr(
        src_boxes: List[Tuple[int,int,int,int]],
        src_texts: List[str],
        merged:    List[Tuple[int,int,int,int]],
    ) -> dict:
        """
        Map Groq OCR texts to the final merged boxes by max-IoU overlap.
        Returns { merged_idx: text }.
        """
        result: dict = {}
        if not src_boxes or not src_texts:
            return result

        for mi, mb in enumerate(merged):
            best_iou  = 0.0
            best_text = ""
            mx0, my0, mx1, my1 = mb

            for si, sb in enumerate(src_boxes):
                if si >= len(src_texts):
                    break
                sx0, sy0, sx1, sy1 = sb
                ix0 = max(mx0, sx0)
                iy0 = max(my0, sy0)
                ix1 = min(mx1, sx1)
                iy1 = min(my1, sy1)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = (
                    (mx1 - mx0) * (my1 - my0)
                    + (sx1 - sx0) * (sy1 - sy0)
                    - inter
                )
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou  = iou
                    best_text = src_texts[si]

            if best_iou > 0.15 and best_text:
                result[mi] = best_text

        # ── Fallback: Groq's pixel coords are often imprecise, so IoU can be 0
        # even when the OCR text is correct. For boxes still without text, map
        # any *unused* Groq texts by reading order (top→bottom, left→right).
        used_texts = set(result.values())
        unmapped_boxes = [i for i in range(len(merged)) if i not in result]
        leftover = [
            (sb, st) for sb, st in zip(src_boxes, src_texts)
            if st and st not in used_texts
        ]
        if unmapped_boxes and leftover:
            def _order(b):  # (cy, cx)
                return ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
            unmapped_boxes.sort(key=lambda i: _order(merged[i]))
            leftover.sort(key=lambda t: _order(t[0]))
            for bi, (_, txt) in zip(unmapped_boxes, leftover):
                result[bi] = txt

        return result
