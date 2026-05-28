"""
One-off targeted replacement for a single known text region.

Used when the automatic detector over-segments (e.g. busy watercolor art):
we pass an EXPLICIT bounding box, inpaint just that band, sample the original
font, and re-type the corrected string fitted to the same box + position.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inpaint import IOPaintClient
from font_sampler import FontSampler
from text_overlay import TextOverlay
from blender import NaturalBlender


def run(src_path, out_path, text_bbox, mask_bbox, new_text, model="lama",
        font_override=None, bold_override=None):
    src_path, out_path = Path(src_path), Path(out_path)
    work = Path("/tmp/fix_one"); work.mkdir(parents=True, exist_ok=True)

    img = Image.open(src_path).convert("RGBA")
    W, H = img.size
    src = work / "source.png"; img.save(src)

    # 1. Build mask over the padded text band
    mx0, my0, mx1, my1 = mask_bbox
    m = np.zeros((H, W), dtype=np.uint8)
    m[my0:my1, mx0:mx1] = 255
    mask = work / "mask.png"; Image.fromarray(m).save(mask)

    # 2. Inpaint — erase text, reconstruct watercolor behind it
    inpainted = work / "inpainted.png"
    IOPaintClient("http://127.0.0.1:8080").inpaint(
        image_path=src, mask_path=mask, output_path=inpainted, model=model,
        prompt="seamless watercolor background, soft pastel texture, no text",
        negative_prompt="text, letters, words, watermark",
    )

    # 3. Sample original font from the PRE-inpainted text region
    tb = text_bbox
    tm = np.zeros((H, W), dtype=np.uint8); tm[tb[1]:tb[3], tb[0]:tb[2]] = 255
    tmask = work / "text_mask.png"; Image.fromarray(tm).save(tmask)
    sampler = FontSampler()
    sample = sampler.sample(source_path=src, mask_path=tmask, sample_out=work / "sample.png")
    print("Sampled:", {k: sample[k] for k in ("color", "font_size", "bold", "italic", "is_serif", "best_family", "ocr_text")})

    # Allow manual overrides when auto-detection misreads typeface/weight
    family = font_override or sample["best_family"]
    bold   = sample["bold"] if bold_override is None else bold_override

    # 4. Render corrected text, fitted to the exact original bbox + position
    text_layer = work / "text_layer.png"
    TextOverlay().add_text(
        source_path=inpainted, output_path=text_layer, text=new_text,
        x=tb[0], y=tb[1], font_size=sample["font_size"],
        font_family=family, color=sample["color"],
        bold=bold, italic=sample["italic"],
        fit_bbox=(tb[0], tb[1], tb[2], tb[3]),
    )

    # 5. Natural blend onto inpainted base
    NaturalBlender().blend(
        base_path=inpainted, text_layer_path=text_layer, mask_path=tmask,
        output_path=out_path, feather_radius=2,
        noise_match=True, brightness_match=True,
    )
    print(f"✓ Saved → {out_path}")
    return sample


def _auto_bbox(src_path, dark=45, pad=14):
    """Find a single solid-dark text band by row-density (for art where the
    automatic detector over-segments). Returns (text_bbox, mask_bbox)."""
    img = np.array(Image.open(src_path).convert("RGB"))
    H, W = img.shape[:2]
    d = (img[:, :, 0] < dark) & (img[:, :, 1] < dark) & (img[:, :, 2] < dark)
    rows = d.sum(axis=1)
    band = np.where(rows > W * 0.05)[0]      # rows that are >5% solid-dark
    if band.size == 0:
        raise SystemExit("No solid-text band found — pass --bbox manually.")
    y0b, y1b = band.min(), band.max()
    sub = np.zeros_like(d); sub[y0b:y1b + 1, :] = d[y0b:y1b + 1, :]
    ys, xs = np.where(sub)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    tb = [int(x0), int(y0), int(x1), int(y1)]
    mb = [max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)]
    return tb, [int(v) for v in mb]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Replace one known text region in an image.")
    ap.add_argument("src", help="source image path")
    ap.add_argument("new_text", help="replacement text")
    ap.add_argument("-o", "--out", help="output path (default: <src>_edited.png)")
    ap.add_argument("--bbox", help="text bbox 'x0,y0,x1,y1' (default: auto-detect dark band)")
    ap.add_argument("--font", help="force a font family, e.g. 'Liberation Serif'")
    ap.add_argument("--bold", action="store_true", help="force bold")
    ap.add_argument("--model", default="lama", help="inpaint model (lama/mat/zits)")
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out) if a.out else src.with_name(src.stem + "_edited.png")
    if a.bbox:
        tb = [int(v) for v in a.bbox.split(",")]
        img = Image.open(src); W, H = img.size
        mb = [max(0, tb[0] - 14), max(0, tb[1] - 14), min(W, tb[2] + 14), min(H, tb[3] + 14)]
    else:
        tb, mb = _auto_bbox(src)
        print(f"auto text bbox: {tb}")

    run(src, out, tb, mb, a.new_text, model=a.model,
        font_override=a.font, bold_override=(True if a.bold else None))
