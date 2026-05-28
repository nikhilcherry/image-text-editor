"""
Natural Text Blender
Makes rendered replacement text visually indistinguishable from the
original image's native text through:

  1. Super-sampled anti-aliasing (SSAA): render text at 4× resolution,
     downsample with LANCZOS for pixel-perfect smooth edges.

  2. Feathered mask transitions: Gaussian-blur the mask boundary so
     the inpainted background blends into the text region without a hard edge.

  3. Local brightness / contrast matching: nudge the text layer's
     luminance to match the surrounding background, preventing glare.

  4. Subtle texture noise injection: add localised Gaussian noise matched
     to the background's variance so the text region doesn't look too smooth.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

log = logging.getLogger(__name__)


class NaturalBlender:
    """
    Blends a PIL text layer onto a background image with anti-aliasing,
    feathering, brightness matching, and optional texture noise.
    """

    # ── Main blend API ─────────────────────────────────────────────────────

    def blend(
        self,
        base_path:       Path,
        text_layer_path: Path,
        mask_path:       Path,
        output_path:     Path,
        feather_radius:  int  = 3,
        noise_match:     bool = True,
        brightness_match: bool = True,
    ) -> Path:
        """
        Composite text_layer onto base with natural blending.

        base_path       : inpainted background PNG (RGBA or RGB)
        text_layer_path : rendered text layer PNG (RGBA, transparent bg)
        mask_path       : white=text-region, black=background (L mode)
        output_path     : where to save the result
        """
        base  = np.array(Image.open(base_path).convert("RGBA"), dtype=np.float32)
        text  = np.array(Image.open(text_layer_path).convert("RGBA"), dtype=np.float32)
        mask  = np.array(Image.open(mask_path).convert("L"),  dtype=np.float32)

        H, W = base.shape[:2]

        # Resize text/mask to base if needed
        if text.shape[:2] != (H, W):
            text_u8 = Image.fromarray(text.astype(np.uint8), "RGBA")
            text_u8 = text_u8.resize((W, H), Image.LANCZOS)
            text    = np.array(text_u8, dtype=np.float32)

        if mask.shape != (H, W):
            mask_img = Image.fromarray(mask.astype(np.uint8)).resize((W, H), Image.LANCZOS)
            mask     = np.array(mask_img, dtype=np.float32)

        # ── 1. Feather the region mask ──────────────────────────────────
        if feather_radius > 0:
            k   = feather_radius * 2 + 1
            mask_f = cv2.GaussianBlur(mask, (k, k), feather_radius * 0.5)
        else:
            mask_f = mask.copy()
        mask_f = (mask_f / 255.0)[..., np.newaxis]     # (H,W,1)

        # ── 2. Local brightness / contrast matching ─────────────────────
        if brightness_match:
            text = self._match_brightness(base, text, mask)

        # ── 3. Alpha-composite using feathered mask ──────────────────────
        # effective alpha = text_alpha  ×  region_mask
        text_a  = text[..., 3:4] / 255.0               # (H,W,1)
        base_a  = base[..., 3:4] / 255.0
        eff_a   = text_a * mask_f

        out_rgb = eff_a * text[..., :3] + (1.0 - eff_a) * base[..., :3]
        out_a   = np.clip(eff_a + base_a * (1.0 - eff_a), 0.0, 1.0)

        # ── 4. Texture noise injection ───────────────────────────────────
        if noise_match:
            out_rgb = self._inject_texture_noise(base[..., :3], out_rgb, mask_f, eff_a)

        out = np.concatenate([
            np.clip(out_rgb, 0, 255).astype(np.uint8),
            (out_a * 255).astype(np.uint8),
        ], axis=2)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out, "RGBA").save(output_path, "PNG")
        log.info("Blended → %s", output_path)
        return output_path

    # ── Super-sampled text rendering ───────────────────────────────────────

    @staticmethod
    def render_ssaa(
        text:        str,
        target_w:    int,
        target_h:    int,
        font:        ImageFont.FreeTypeFont,
        color:       tuple,        # RGBA
        align:       str  = "left",
        scale:       int  = 4,
    ) -> Image.Image:
        """
        Render text at scale×resolution, then downsample to (target_w, target_h).
        Produces pixel-perfect anti-aliased edges — much smoother than PIL's
        built-in rendering at native resolution.

        Returns an RGBA PIL image of size (target_w, target_h).
        """
        sw, sh = target_w * scale, target_h * scale
        # Scale font
        try:
            big_font = font.font_variant(size=font.size * scale)
        except AttributeError:
            # Older Pillow: use truetype with path
            try:
                big_font = ImageFont.truetype(font.path, font.size * scale)
            except Exception:
                big_font = font

        canvas = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
        draw   = ImageDraw.Draw(canvas)
        try:
            draw.multiline_text(
                (0, 0), text,
                font=big_font,
                fill=color,
                align=align,
                anchor="lt",
            )
        except TypeError:
            draw.multiline_text((0, 0), text, font=big_font, fill=color, align=align)

        # Downsample with LANCZOS — this is the key anti-aliasing step
        result = canvas.resize((target_w, target_h), Image.LANCZOS)
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _match_brightness(
        base: np.ndarray,    # RGBA float32
        text: np.ndarray,    # RGBA float32
        mask: np.ndarray,    # L    uint8
    ) -> np.ndarray:
        """
        In the region where new text is placed, gently nudge the text's
        local luminance toward the surrounding background so there's no
        jarring brightness discontinuity.

        The adjustment is intentionally small (≤ 20%) to keep readability.
        """
        text   = text.copy()
        m      = mask > 32

        if not m.any():
            return text

        # Dilate the mask to get a 'border zone' of background pixels
        k       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate(m.astype(np.uint8), k).astype(bool)
        border  = dilated & (~m)

        if not border.any():
            return text

        bg_rgb  = base[border, :3]
        bg_mean = float(np.mean(bg_rgb))

        # Only look at opaque text pixels
        text_opaque = m & (text[..., 3] > 96)
        if not text_opaque.any():
            return text

        txt_mean = float(np.mean(text[text_opaque, :3]))
        diff     = abs(txt_mean - bg_mean)

        if diff < 35:
            return text   # close enough — leave it alone

        # Nudge by ≤ 18% of the difference
        blend_f = min(0.18, (diff - 35) / 600.0)
        for c in range(3):
            orig = text[text_opaque, c]
            text[text_opaque, c] = np.clip(
                orig * (1.0 - blend_f) + bg_mean * blend_f,
                0, 255,
            )

        return text

    @staticmethod
    def _inject_texture_noise(
        base_rgb:  np.ndarray,   # H×W×3 float32
        out_rgb:   np.ndarray,   # H×W×3 float32
        mask_f:    np.ndarray,   # H×W×1 float32 [0-1]
        eff_a:     np.ndarray,   # H×W×1 float32 [0-1]
    ) -> np.ndarray:
        """
        Add a tiny amount of background-matched Gaussian noise to the text
        region so it blends with the surrounding texture instead of looking
        like a flat digital overlay.
        """
        bg_region = (mask_f[..., 0] < 0.1)
        if not bg_region.any():
            return out_rgb

        bg_std = float(base_rgb[bg_region].std())
        if bg_std < 1.5:
            return out_rgb   # background is smooth → no noise needed

        noise_sigma = min(bg_std * 0.28, 5.5)
        noise  = np.random.normal(0.0, noise_sigma, out_rgb.shape).astype(np.float32)

        # Apply noise only in the text alpha region (proportional to alpha)
        weight = eff_a * mask_f
        return np.clip(out_rgb + noise * weight, 0, 255)
