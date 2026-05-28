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
from tesseract_ocr import TesseractOCR
from inpaint import IOPaintClient
from font_sampler import FontSampler
from text_overlay import TextOverlay
from blender import NaturalBlender

log = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class WordReplacer:
    def __init__(self, iopaint_url: str = "http://127.0.0.1:8080"):
        self.tesseract = TesseractOCR()     # primary: pixel-accurate on docs
        self.gemini = GeminiOCR()           # fallback: stylised / artistic text
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
    def _refine(img: np.ndarray, box: List[int],
                precise: bool = False) -> Tuple[List[int], np.ndarray]:
        """
        Find the real text ink inside an OCR box and return
        (tight_bbox, full_image_mask) — the mask is the dilated text strokes
        only (uint8 0/255), so only the text gets inpainted.

        `precise=True` (Tesseract): the box is already accurate, so we only
        look *inside* it (tiny pad) and keep the box as the render target — no
        wide window that could bleed into the line above/below.
        `precise=False` (Gemini): boxes are imprecise, so we expand and snap to
        ink to correct small offsets.
        """
        H, W = img.shape[:2]
        x0, y0, x1, y1 = box

        if precise:
            # Tesseract gives an accurate location but sometimes UNDER-boxes
            # tall/bold text (e.g. it reported h=15 for date digits that are
            # really h=34). So snap the mask to the actual ink extent inside a
            # window around the box — but stop at blank rows/cols so we don't
            # bleed into the line above/below.
            bh = max(1, y1 - y0)
            wy0, wy1 = max(0, y0 - bh), min(H, y1 + bh)
            wx0, wx1 = max(0, x0 - 4), min(W, x1 + 4)
            gray = cv2.cvtColor(img[wy0:wy1, wx0:wx1], cv2.COLOR_RGB2GRAY).astype(np.float32)
            bgw = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(3, (wx1 - wx0) // 8))
            ink = np.abs(gray - bgw) > 22

            cx0, cx1 = x0 - wx0, x1 - wx0          # box x-range in window coords
            rows = ink[:, cx0:cx1].sum(axis=1).astype(float)
            rthr = max(2, (cx1 - cx0) * 0.08)
            cy = (y0 + y1) // 2 - wy0               # box centre in window coords
            cy = min(max(cy, 0), len(rows) - 1)
            # grow up/down from the centre while rows have ink (allow 1px gaps)
            ty0 = cy
            while ty0 > 0 and (rows[ty0 - 1] > rthr or rows[max(0, ty0 - 2)] > rthr):
                ty0 -= 1
            ty1 = cy
            while ty1 < len(rows) - 1 and (rows[ty1 + 1] > rthr or rows[min(len(rows) - 1, ty1 + 2)] > rthr):
                ty1 += 1
            band = ink[ty0:ty1 + 1, :]
            cols = band.sum(axis=0)
            xs = np.where(cols >= 2)[0]
            mask = np.zeros((H, W), dtype=np.uint8)
            if xs.size:
                ix0, ix1 = wx0 + int(xs.min()), wx0 + int(xs.max())
                iy0, iy1 = wy0 + ty0, wy0 + ty1
            else:
                ix0, iy0, ix1, iy1 = x0, y0, x1, y1
            pad = 3
            mask[max(0, iy0 - pad):min(H, iy1 + pad),
                 max(0, ix0 - pad):min(W, ix1 + pad)] = 255
            return [ix0, iy0, ix1 + 1, iy1 + 1], mask

        bw, bh = x1 - x0, y1 - y0
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

        dil = cv2.dilate(ink8, np.ones((3, 3), np.uint8), iterations=2)
        mask[wy0:wy1, wx0:wx1] = dil

        if precise:
            # trust the OCR box for placement/size; mask still = real strokes
            return [x0, y0, x1, y1], mask
        tx0, ty0 = wx0 + int(xs.min()), wy0 + int(ys.min())
        tx1, ty1 = wx0 + int(xs.max()), wy0 + int(ys.max())
        return [tx0, ty0, tx1 + 1, ty1 + 1], mask

    # ── orientation / skew correction ──────────────────────────────────────
    @staticmethod
    def _estimate_skew(pil_img, limit: float = 10.0, step: float = 0.5) -> float:
        """Fine skew angle (deg, CCW-positive to straighten) via projection-
        profile variance: the rotation that makes text rows most distinct."""
        g = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        s = 800.0 / max(h, w)
        if s < 1.0:
            g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
        H, W = bw.shape
        best_a, best = 0.0, -1.0
        a = -limit
        while a <= limit + 1e-9:
            M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
            rot = cv2.warpAffine(bw, M, (W, H), flags=cv2.INTER_NEAREST)
            proj = rot.sum(axis=1, dtype=np.float64)
            d = np.diff(proj)
            score = float((d * d).sum())
            if score > best:
                best, best_a = score, a
            a += step
        return best_a

    def _correct_orientation(self, pil_img, src_path):
        """Straighten a sideways/upside-down or slightly skewed image.
        Returns (upright_img, restore_fn, osd_deg, skew_deg, changed)."""
        osd = self.tesseract.orientation(src_path) if self.tesseract.available() else 0
        img = pil_img.rotate(-osd, expand=True) if osd else pil_img   # rotate(-OSD)=upright
        fine = self._estimate_skew(img)
        if abs(fine) >= 0.5:
            img = img.rotate(fine, expand=True, resample=Image.BICUBIC,
                             fillcolor=(255, 255, 255))
        else:
            fine = 0.0
        orig_w, orig_h = pil_img.size

        def restore(result_pil):
            out = result_pil
            if fine:
                out = out.rotate(-fine, expand=True, resample=Image.BICUBIC,
                                 fillcolor=(255, 255, 255))
            if osd:
                out = out.rotate(osd, expand=True)
            cw, ch = out.size                       # crop back to original framing
            left, top = max(0, (cw - orig_w) // 2), max(0, (ch - orig_h) // 2)
            return out.crop((left, top, left + orig_w, top + orig_h))

        return img, restore, osd, fine, (bool(osd) or bool(fine))

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
        auto_rotate: bool = True,
    ) -> dict:
        image_path = Path(image_path)
        out_path = Path(out_path) if out_path else \
            image_path.with_name(image_path.stem + "_edited.png")
        work = Path("/tmp/word_replace") / uuid.uuid4().hex
        work.mkdir(parents=True, exist_ok=True)

        orig_img = Image.open(image_path).convert("RGB")

        # Auto-straighten sideways / skewed images, then process upright and
        # rotate the result back at the end.
        restore = None
        if auto_rotate:
            raw = work / "raw.png"; orig_img.save(raw)
            src_img, restore, osd, skew, rotated = self._correct_orientation(orig_img, raw)
            if rotated:
                log.info("Auto-rotate applied: OSD=%d°, skew=%.1f°", osd, skew)
            else:
                restore = None
        else:
            src_img = orig_img

        W, H = src_img.size
        src = work / "source.png"; src_img.save(src)
        img = np.array(src_img)

        # 1. OCR — Tesseract first (pixel-accurate on printed/document text).
        #    Gemini is only used as a fallback for words Tesseract can't read
        #    (stylised / artistic text), so we don't even call it unless needed.
        tess_words: List[Dict] = []
        if self.tesseract.available():
            try:
                tess_words = self.tesseract.detect_words(src)
            except Exception as e:
                log.warning("Tesseract OCR failed: %s", e)

        # 2-3. match + refine each target
        jobs = []                       # (find, repl, tight_bbox)
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        not_found = []
        gem_words: Optional[List[Dict]] = None     # lazy-loaded
        used_src = {}
        for find, repl in replacements.items():
            boxes = self._match_targets(tess_words, find)
            ocr_src = "tesseract"
            if not boxes:
                # Fall back to Gemini for this word (load OCR once).
                if gem_words is None:
                    gem_words = []
                    if self.gemini.available():
                        try:
                            gem_words = self.gemini.detect_words(src)
                        except Exception as e:
                            log.warning("Gemini OCR fallback failed: %s", e)
                boxes = self._match_targets(gem_words, find)
                ocr_src = "gemini"
            if not boxes:
                not_found.append(find)
                continue
            used_src[find] = ocr_src
            for b in boxes:
                tight, m = self._refine(img, b, precise=(ocr_src == "tesseract"))
                combined_mask |= m
                jobs.append((find, repl, tight))

        if not jobs:
            detected = [w["text"] for w in tess_words] + \
                       [w["text"] for w in (gem_words or [])]
            log.warning("No target words found: %s", list(replacements))
            return {"file": str(image_path), "status": "no_match",
                    "not_found": not_found, "detected": detected}

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
            color = "#000000"   # always render replacement text in black
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
                            "font": family, "color": color, "bold": bool(bold),
                            "ocr": used_src.get(find, "?")})

        # 6. save + log
        final_img = Image.open(base).convert("RGB")
        if restore is not None:                      # rotate result back to input orientation
            final_img = restore(final_img)
        final_img.save(out_path)
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
