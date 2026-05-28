"""
Gemini vision OCR — word-level text detection with bounding boxes.

Gemini returns reliable per-word locations, which lets us find a specific word
the user wants to replace *without any manual masking*. Coordinates come back
normalised to 0-1000 (Gemini's convention: [ymin, xmin, ymax, xmax]); we convert
them to pixel boxes.

Uses the REST API via `requests` (no extra SDK dependency).
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

import requests

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

_PROMPT = (
    "Detect EVERY piece of text in this image, one entry per word "
    "(split multi-word lines into separate words).\n"
    "Return ONLY a JSON array. Each element:\n"
    '  {"text": "<the exact word>", "box_2d": [ymin, xmin, ymax, xmax]}\n'
    "where box_2d values are integers 0-1000 normalised to the image size "
    "(origin top-left). Include numbers, labels and watermark text. "
    "Do not merge separate words into one box."
)


class GeminiOCR:
    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = model or GEMINI_MODEL

    def available(self) -> bool:
        return bool(self.api_key)

    def detect_words(self, image_path: Path) -> List[Dict]:
        """
        Return [{"text": str, "bbox": [x0, y0, x1, y1]}] in pixel coords.
        Raises RuntimeError on API/parse failure so callers can fall back.
        """
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set (.env)")

        from PIL import Image
        with Image.open(image_path) as im:
            W, H = im.size

        img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
        suffix = Path(image_path).suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        payload = {
            "contents": [{
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": mime, "data": img_b64}},
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        }

        url = GEMINI_URL.format(model=self.model)
        r = requests.post(
            url, params={"key": self.api_key}, json=payload, timeout=120
        )
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")

        try:
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(text)
        except Exception as e:
            raise RuntimeError(f"Gemini parse error: {e}")

        # Gemini sometimes wraps the array in a key
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

        words: List[Dict] = []
        for item in data if isinstance(data, list) else []:
            box = item.get("box_2d") or item.get("box") or item.get("bbox")
            txt = str(item.get("text", "")).strip()
            if not txt or not box or len(box) != 4:
                continue
            ymin, xmin, ymax, xmax = box
            x0 = int(min(xmin, xmax) / 1000.0 * W)
            x1 = int(max(xmin, xmax) / 1000.0 * W)
            y0 = int(min(ymin, ymax) / 1000.0 * H)
            y1 = int(max(ymin, ymax) / 1000.0 * H)
            if x1 > x0 and y1 > y0:
                words.append({"text": txt, "bbox": [x0, y0, x1, y1]})

        log.info("Gemini OCR found %d words", len(words))
        return words
