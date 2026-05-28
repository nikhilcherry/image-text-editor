"""
Tesseract OCR — pixel-accurate word boxes for printed/document text.

Calls the `tesseract` binary directly (TSV output) so no Python wrapper package
is required. Returns the same shape as GeminiOCR.detect_words:
  [{"text": str, "bbox": [x0, y0, x1, y1], "conf": float}]

Tesseract is precise on clear printed text (IDs, forms, signs) where VLM boxes
drift. It is weak on stylised/artistic text — that's where Gemini takes over.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)


class TesseractOCR:
    def __init__(self, psm: int = 11, min_conf: float = 40.0):
        # psm 11 = "sparse text": find as much text as possible, any layout.
        self.psm = psm
        self.min_conf = min_conf
        self.bin = shutil.which("tesseract")

    def available(self) -> bool:
        return bool(self.bin)

    def orientation(self, image_path: Path) -> int:
        """
        Coarse page orientation via Tesseract OSD. Returns the degrees the
        image must be rotated *clockwise* to make text upright: 0/90/180/270.
        Returns 0 if OSD is unavailable or low-confidence.
        """
        if not self.bin:
            return 0
        try:
            out = subprocess.check_output(
                [self.bin, str(image_path), "stdout", "--psm", "0"],
                stderr=subprocess.DEVNULL, timeout=30,
            ).decode("utf-8", "ignore")
        except Exception:
            return 0
        rotate, conf = 0, 0.0
        for line in out.splitlines():
            if line.startswith("Rotate:"):
                try:
                    rotate = int(line.split(":")[1].strip())
                except ValueError:
                    pass
            elif line.startswith("Orientation confidence:"):
                try:
                    conf = float(line.split(":")[1].strip())
                except ValueError:
                    pass
        # Require some confidence before trusting a non-zero rotation.
        if rotate in (90, 180, 270) and conf >= 1.0:
            return rotate
        return 0

    def detect_words(self, image_path: Path) -> List[Dict]:
        if not self.bin:
            raise RuntimeError("tesseract binary not found on PATH")
        try:
            out = subprocess.check_output(
                [self.bin, str(image_path), "stdout", "--psm", str(self.psm), "tsv"],
                stderr=subprocess.DEVNULL, timeout=60,
            ).decode("utf-8", "ignore")
        except Exception as e:
            raise RuntimeError(f"tesseract failed: {e}")

        lines = out.splitlines()
        if len(lines) < 2:
            return []
        header = lines[0].split("\t")
        idx = {name: i for i, name in enumerate(header)}
        need = ("level", "left", "top", "width", "height", "conf", "text")
        if not all(k in idx for k in need):
            return []

        words: List[Dict] = []
        for ln in lines[1:]:
            p = ln.split("\t")
            if len(p) <= idx["text"]:
                continue
            try:
                level = int(p[idx["level"]])
                conf = float(p[idx["conf"]])
            except ValueError:
                continue
            text = p[idx["text"]].strip()
            if level != 5 or not text or conf < self.min_conf:
                continue
            l = int(p[idx["left"]]); t = int(p[idx["top"]])
            w = int(p[idx["width"]]); h = int(p[idx["height"]])
            words.append({"text": text, "bbox": [l, t, l + w, t + h], "conf": conf})

        log.info("Tesseract found %d words", len(words))
        return words
