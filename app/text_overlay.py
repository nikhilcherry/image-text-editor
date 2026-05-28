"""
Text Overlay — renders styled text onto images using Pillow.

Supports:
  • Any system font (auto-discovered)
  • Bold / italic (via fonttools or synthetic slant)
  • Stroke / outline
  • Opacity
  • Multi-line text with configurable alignment
"""

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageFilter

log = logging.getLogger(__name__)

# ── Default font search paths ─────────────────────────────────
FONT_DIRS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path(os.path.expanduser("~/.fonts")),
    Path(os.path.expanduser("~/.local/share/fonts")),
]


class TextOverlay:
    def __init__(self):
        self._font_cache: dict = {}
        self._font_map: Optional[dict] = None    # lazy-loaded (one per family)
        self._font_faces: Optional[list] = None  # lazy-loaded (all weights/styles)

    # ── Public API ────────────────────────────────────────────
    def add_text(
        self,
        source_path: Path,
        output_path: Path,
        text: str,
        x: int = 50,
        y: int = 50,
        font_size: int = 48,
        font_family: str = "",
        color: str = "#000000",
        bold: bool = False,
        italic: bool = False,
        stroke_width: int = 0,
        stroke_color: str = "#ffffff",
        align: str = "left",
        opacity: float = 1.0,
        fit_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Render text onto the source image and save to output_path.

        If `fit_bbox` (x0, y0, x1, y1) is provided, the font size is
        automatically scaled so the rendered text fills that area
        (and `x`, `y` are overridden to the bbox top-left). This is
        the "adapt to masked area" mode.
        """

        base = Image.open(source_path).convert("RGBA")

        # ── Auto-fit text to bbox if requested ────────────────
        if fit_bbox and text:
            bx, by, bx2, by2 = fit_bbox
            target_w = max(1, bx2 - bx)
            target_h = max(1, by2 - by)
            font_size = self._fit_size(
                text, font_family, target_w, target_h, bold, italic
            )
            x, y = bx, by
            log.info("Auto-fit text → size=%dpx for bbox %dx%d",
                     font_size, target_w, target_h)

        # Create text layer (same size, transparent)
        txt_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        font = self._load_font(font_family, font_size, bold, italic)
        rgba_color = self._parse_color(color, opacity)
        rgba_stroke = self._parse_color(stroke_color, 1.0) if stroke_width > 0 else None

        kwargs = {
            "font":  font,
            "fill":  rgba_color,
            "align": align,
        }
        if stroke_width > 0 and rgba_stroke:
            kwargs["stroke_width"] = stroke_width
            kwargs["stroke_fill"]  = rgba_stroke

        # When fitting to bbox, anchor with PIL's "lt" (left/top) for predictable placement
        if fit_bbox:
            kwargs["anchor"] = "lt"

        try:
            draw.multiline_text((x, y), text, **kwargs)
        except (TypeError, ValueError):
            # PIL's multiline_text doesn't support anchor in some versions
            kwargs.pop("anchor", None)
            draw.multiline_text((x, y), text, **kwargs)

        # Composite
        out = Image.alpha_composite(base, txt_layer)
        out.convert("RGBA").save(output_path, "PNG")
        log.info("Text overlay saved → %s (size=%dpx at %d,%d)", output_path, font_size, x, y)

    def _fit_size(
        self, text: str, font_family: str,
        target_w: int, target_h: int,
        bold: bool, italic: bool,
        margin: float = 0.95,
    ) -> int:
        """
        Binary-search the largest font_size where the rendered text fits
        within (target_w, target_h). Accounts for actual font metrics
        rather than nominal em-size.
        """
        if not text:
            return max(12, target_h)

        text = text.strip("\n")
        # Search space: 6px → 4× target_h (or wider if text is short)
        low, high = 6, max(target_h * 4, 400)
        best = max(8, target_h)

        budget_w = target_w * margin
        budget_h = target_h * margin

        for _ in range(40):  # cap iterations
            if low > high:
                break
            mid = (low + high) // 2
            try:
                font = self._load_font(font_family, mid, bold, italic)
                bbox = font.getbbox(text) if "\n" not in text else None
                if bbox is None:
                    # Multiline — use textbbox via a throwaway draw
                    tmp = Image.new("RGBA", (1, 1))
                    d = ImageDraw.Draw(tmp)
                    bbox = d.multiline_textbbox((0, 0), text, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = target_w, target_h

            if w <= budget_w and h <= budget_h:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        return best

    def list_fonts(self) -> list:
        """Return sorted list of available font names."""
        return sorted(self._get_font_map().keys())

    # ── Font loading ─────────────────────────────────────────
    def _load_font(
        self, family: str, size: int, bold: bool, italic: bool
    ) -> ImageFont.FreeTypeFont:
        key = (family, size, bold, italic)
        if key in self._font_cache:
            return self._font_cache[key]

        font = self._find_font(family, bold, italic, size)
        self._font_cache[key] = font
        return font

    def _find_font(
        self, family: str, bold: bool, italic: bool, size: int
    ) -> ImageFont.FreeTypeFont:
        # Match against ALL faces (family + weight + slant), so we can pick the
        # correct Regular / Bold / Italic file instead of whichever face merely
        # shares the family name. fontconfig weight: regular≈80, bold≈200;
        # slant: roman=0, italic/oblique>0.
        candidates = []
        if family:
            family_lower = family.lower()
            for fam, weight, slant, path in self._get_font_faces():
                if family_lower not in fam:
                    continue
                is_bold   = weight >= 180
                is_italic = slant  >= 100
                is_light  = weight <= 50
                score = 0.0
                # exact family name match beats substring match
                if fam == family_lower:
                    score += 1
                # Weight: reward requested weight, penalise the wrong one.
                if bold:
                    score += 3 if is_bold else -2
                else:
                    score += -3 if is_bold else 1
                    if is_light:
                        score -= 1
                # Slant: strongly penalise unwanted italic.
                if italic:
                    score += 3 if is_italic else -2
                else:
                    score += -4 if is_italic else 1
                candidates.append((score, str(path)))
            candidates.sort(key=lambda t: -t[0])

        if candidates:
            try:
                return ImageFont.truetype(candidates[0][1], size)
            except Exception:
                pass

        # Fallback to common fonts
        fallbacks = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        for fb in fallbacks:
            if Path(fb).exists():
                try:
                    return ImageFont.truetype(fb, size)
                except Exception:
                    continue

        log.warning("No suitable font found — using Pillow default.")
        return ImageFont.load_default(size=max(size, 10))

    def _get_font_map(self) -> dict:
        """Return {name_lower: path} for all .ttf/.otf on the system."""
        if self._font_map is not None:
            return self._font_map

        font_map = {}

        # Use fc-list if available (fast)
        try:
            out = subprocess.check_output(
                ["fc-list", "--format=%{family[0]}:%{file}\n"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode(errors="ignore")
            for line in out.strip().splitlines():
                if ":" in line:
                    name, path = line.split(":", 1)
                    font_map[name.strip()] = Path(path.strip())
        except Exception:
            # Manual scan fallback
            for d in FONT_DIRS:
                if d.exists():
                    for p in d.rglob("*.[ot]tf"):
                        font_map[p.stem] = p

        self._font_map = font_map
        return font_map

    def _get_font_faces(self) -> list:
        """
        Return [(family_lower, weight, slant, Path), …] for every installed
        face. Unlike _get_font_map (one entry per family), this keeps each
        weight/style as a separate face so _find_font can pick Regular vs Bold
        vs Italic correctly.
        """
        if getattr(self, "_font_faces", None) is not None:
            return self._font_faces

        faces = []
        try:
            out = subprocess.check_output(
                ["fc-list", "--format=%{family[0]}\t%{weight}\t%{slant}\t%{file}\n"],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode(errors="ignore")
            for line in out.strip().splitlines():
                parts = line.split("\t")
                if len(parts) != 4:
                    continue
                fam, weight, slant, path = parts
                try:
                    w = int(weight) if weight.strip() else 80
                except ValueError:
                    w = 80
                try:
                    s = int(slant) if slant.strip() else 0
                except ValueError:
                    s = 0
                faces.append((fam.strip().lower(), w, s, Path(path.strip())))
        except Exception:
            # Fallback: derive weight/slant from the filename of each face.
            for d in FONT_DIRS:
                if not d.exists():
                    continue
                for p in d.rglob("*.[ot]tf"):
                    nl = p.stem.lower()
                    w = 200 if any(x in nl for x in ("bold", "black", "heavy")) else 80
                    s = 100 if ("italic" in nl or "oblique" in nl) else 0
                    faces.append((nl, w, s, p))

        self._font_faces = faces
        return faces

    # ── Color parsing ─────────────────────────────────────────
    @staticmethod
    def _parse_color(color: str, opacity: float) -> Tuple[int, int, int, int]:
        """Parse #rrggbb / #rrggbbaa / rgb(…) / rgba(…) → (r,g,b,a)."""
        color = color.strip()
        alpha = int(opacity * 255)

        # Hex
        m = re.match(r"^#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?$", color)
        if m:
            r = int(m.group(1)[0:2], 16)
            g = int(m.group(1)[2:4], 16)
            b = int(m.group(1)[4:6], 16)
            if m.group(2):
                alpha = int(m.group(2), 16)
            return (r, g, b, alpha)

        # rgb(r,g,b) / rgba(r,g,b,a)
        m = re.match(r"rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\s*\)", color)
        if m:
            r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if m.group(4):
                alpha = int(float(m.group(4)) * 255)
            return (r, g, b, alpha)

        # Named fallback
        try:
            from PIL import ImageColor
            r, g, b = ImageColor.getrgb(color)
            return (r, g, b, alpha)
        except Exception:
            return (0, 0, 0, alpha)
