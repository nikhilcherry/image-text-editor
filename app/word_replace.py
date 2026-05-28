"""
Word-level find & replace — "replace word X with word Y", no manual masking.

Pipeline per image:
  1. Gemini OCR  → every word + bounding box.
  2. Match the user's "find" text against detected words (exact → substring →
     consecutive-word phrase).
  3. Refine each match to the ACTUAL ink pixels (CV) → tight bbox + stroke mask
     (so only the text is touched, nothing else).
  4. Inpaint the combined mask once (IOPaint: mat / lama / sd).
  5. Re-render each replacement in the sampled font at the same position.
  6. Save <name>_edited.png and append a JSON log.

No bounding boxes are entered by hand — the user only says what to find and
what to replace it with.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from gemini_ocr import GeminiOCR
from inpaint import IOPaintClient
from font_sampler import FontSampler
from text_overlay import TextOverlay
from blender import NaturalBlender

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class WordReplacer:
    def __init__(self, iopaint_url: str = "http://127.0.0.1:8080"):
        self.ocr = GeminiOCR()
        self.iopaint = IOPaintClient(iopaint_url)
        self.sampler = FontSampler()
        self.overlay = TextOverlay()
        self.blender = NaturalBlender()

    # ── word matching ──────────────────────────────────────────────────────
    @staticmethod
    def _match_targets(words: List[Dict], find: str) -> List[List[int]]:
        """Return a list of bboxes (one per occurrence) that match `find`."""
        nf = _norm(find)
        hits = [w["bbox"] for w in words if _norm(w["text"]) == nf]
        if hits:
            return hits
        # substring (handles 'imge' inside a word, or punctuation glued on)
        hits = [w["bbox"] for w in words
                if nf in _norm(w["text"]) or _norm(w["text"]) in nf]
        if hits:
            return hits
        # multi-word phrase: find consecutive words on one line whose join == nf
        if " " in nf:
            toks = nf.split(" ")
            # group words into lines by vertical overlap
            ws = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))
            for i in range(len(ws)):
                acc, boxes = [], []
                for j in range(i, min(i + len(toks) + 2, len(ws))):
                    acc.append(_norm(ws[j]["text"]))
                    boxes.append(ws[j]["bbox"])
                    if _norm(" ".join(acc)) == nf:
                        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
                        x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
                        return [[x0, y0, x1, y1]]
        return []

    # ── bbox refinement → tight box + stroke mask ──────────────────────────
    @staticmethod
    def _refine(img: np.ndarray, box: List[int]) -> Tuple[List[int], np.ndarray]:
        """
        Given an approximate (Gemini) box, find the real text ink inside it.
        Returns (tight_bbox, full_image_mask) where mask is the dilated text
        strokes only (uint8 0/255). VLM boxes are imprecise, so we snap to
        pixels — this is what keeps us from erasing anything but the text.
        """
        H, W = img.shape[:2]
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        # Expand a little to catch ink the VLM box clipped, but keep the
        # vertical window modest so we don't bleed into an adjacent text line.
        px = int(bw * 0.20) + 4
        py = min(int(bh * 0.6) + 4, max(6, bh))
        wx0, wy0 = max(0, x0 - px), max(0, y0 - py)
        wx1, wy1 = min(W, x1 + px), min(H, y1 + py)

        win = img[wy0:wy1, wx0:wx1]
        gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY).astype(np.float32)
        sigma = max(3, (wx1 - wx0) // 8)
        bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        contrast = gray - bg

        dark = contrast < -22
        light = contrast > 22
        ink = dark if int(dark.sum()) >= int(light.sum()) else light
        ink8 = cv2.morphologyEx((ink.astype(np.uint8)) * 255,
                                cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        ys, xs = np.where(ink8 > 0)
        mask = np.zeros((H, W), dtype=np.uint8)
        if xs.size < 5:
            # fall back to the given box
            mask[y0:y1, x0:x1] = 255
            return [x0, y0, x1, y1], mask

        tx0, ty0 = wx0 + int(xs.min()), wy0 + int(ys.min())
        tx1, ty1 = wx0 + int(xs.max()), wy0 + int(ys.max())

        dil = cv2.dilate(ink8, np.ones((3, 3), np.uint8), iterations=2)
        mask[wy0:wy1, wx0:wx1] = dil
        return [tx0, ty0, tx1 + 1, ty1 + 1], mask

    # ── main entry ─────────────────────────────────────────────────────────
    def replace(
        self,
        image_path: Path,
        replacements: Dict[str, str],
        out_path: Optional[Path] = None,
        model: str = "mat",
        prompt: str = "",
        font_override: str = "",
        bold_override: Optional[bool] = None,
        log_path: Optional[Path] = None,
    ) -> dict:
        image_path = Path(image_path)
        out_path = Path(out_path) if out_path else \
            image_path.with_name(image_path.stem + "_edited.png")
        work = Path("/tmp/word_replace") / uuid.uuid4().hex
        work.mkdir(parents=True, exist_ok=True)

        src_img = Image.open(image_path).convert("RGB")
        W, H = src_img.size
        src = work / "source.png"; src_img.save(src)
        img = np.array(src_img)

        # 1. OCR
        words = self.ocr.detect_words(image_path)

        # 2-3. match + refine each target
        jobs = []                       # (find, repl, tight_bbox)
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        not_found = []
        for find, repl in replacements.items():
            boxes = self._match_targets(words, find)
            if not boxes:
                not_found.append(find)
                continue
            for b in boxes:
                tight, m = self._refine(img, b)
                combined_mask |= m
                jobs.append((find, repl, tight))

        if not jobs:
            log.warning("No target words found: %s", list(replacements))
            return {"file": str(image_path), "status": "no_match",
                    "not_found": not_found, "detected": [w["text"] for w in words]}

        mask_path = work / "mask.png"
        Image.fromarray(combined_mask).save(mask_path)

        # 4. inpaint once (erase all targeted text)
        inpainted = work / "inpainted.png"
        try:
            self.iopaint.inpaint(
                image_path=src, mask_path=mask_path, output_path=inpainted,
                model=model,
                prompt=prompt or "clean background, original texture, no text",
                negative_prompt="text, letters, words, numbers, watermark, signature",
            )
        except Exception as e:
            log.error("Inpaint failed (%s) — using source as base", e)
            inpainted = src

        # 5. render each replacement at its position
        base = inpainted
        results = []
        for idx, (find, repl, bbox) in enumerate(jobs):
            x0, y0, x1, y1 = bbox
            rm = np.zeros((H, W), dtype=np.uint8); rm[y0:y1, x0:x1] = 255
            rmask = work / f"r{idx}.png"; Image.fromarray(rm).save(rmask)

            sample = {}
            try:
                sample = self.sampler.sample(src, rmask, work / f"s{idx}.png")
            except Exception as e:
                log.warning("sample failed for %r: %s", find, e)

            family = font_override or sample.get("best_family", "")
            color = sample.get("color", "#000000")
            bold = sample.get("bold", False) if bold_override is None else bold_override
            italic = sample.get("italic", False)

            layer = work / f"t{idx}.png"
            merged = work / f"m{idx}.png"
            self.overlay.add_text(
                source_path=base, output_path=layer, text=repl,
                x=x0, y=y0, font_size=max(8, y1 - y0),
                font_family=family, color=color, bold=bold, italic=italic,
                fit_bbox=(x0, y0, x1, y1),
            )
            self.blender.blend(
                base_path=base, text_layer_path=layer, mask_path=rmask,
                output_path=merged, feather_radius=2,
                noise_match=True, brightness_match=True,
            )
            base = merged
            results.append({"find": find, "replace": repl, "bbox": bbox,
                            "font": family, "color": color, "bold": bool(bold)})

        # 6. save + log
        Image.open(base).convert("RGB").save(out_path)
        summary = {
            "file": str(image_path), "output": str(out_path),
            "status": "success", "replaced": len(results),
            "not_found": not_found, "model": model,
            "timestamp": datetime.now().isoformat(), "regions": results,
        }
        self._append_log(summary, log_path)
        log.info("✓ %s → %s (%d replaced, %d not found)",
                 image_path.name, out_path.name, len(results), len(not_found))
        return summary

    @staticmethod
    def _append_log(summary: dict, log_path: Optional[Path]):
        lp = Path(log_path) if log_path else (Path.home() / "Desktop" / "text_replacement_log.json")
        existing = []
        if lp.exists():
            try:
                raw = json.loads(lp.read_text("utf-8"))
                existing = raw if isinstance(raw, list) else [raw]
            except Exception:
                existing = []
        existing.append(summary)
        try:
            lp.write_text(json.dumps(existing, indent=2, ensure_ascii=False), "utf-8")
        except Exception as e:
            log.warning("Could not write log: %s", e)
