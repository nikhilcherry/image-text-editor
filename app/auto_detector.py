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
        mser_boxes  = self._mser_detect(img, padding, min_area)
        morph_boxes = []
        if not mser_boxes:
            morph_boxes = self._gradient_detect(img, padding)

        cv_boxes = mser_boxes or morph_boxes
        method   = "mser" if mser_boxes else ("gradient" if morph_boxes else "")

        # ── Stage 2: Groq vision (primary OCR + verification) ─────────────
        groq_boxes: List[Tuple[int,int,int,int]] = []
        groq_texts: List[str] = []
        if use_groq:
            groq_boxes, groq_texts = self._groq_detect(image_path, W, H)
            if groq_boxes:
                method += ("+groq" if method else "groq")

        # ── Merge CV + Groq boxes ──────────────────────────────────────────
        all_boxes = list(cv_boxes) + list(groq_boxes)
        merged    = self._merge_boxes(all_boxes, W, H, gap=10)

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

        return result
