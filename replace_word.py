#!/usr/bin/env python3
"""
replace_word.py — simple "find a word, replace it" tool.

You give an image and say WHAT to find and WHAT to replace it with. The tool
locates the word automatically (Gemini vision OCR), erases just that text,
rebuilds the background, and re-types the new word in the same place/font.
No manual masking or coordinates.

Examples
--------
  # Interactive — it asks you what to replace and with what:
  python replace_word.py ~/Desktop/poster.png

  # One-shot:
  python replace_word.py ~/Desktop/tre.png --find tre --replace tree

  # Several at once:
  python replace_word.py id.png -f WILLIAM -r NIKHIL -f 789 -r 123

  # Choose inpaint model / force a font:
  python replace_word.py img.png -f cat -r dog --model sd-v1-5-inpainting.ckpt --font "Liberation Sans"

Requires GEMINI_API_KEY in .env  (get one at https://aistudio.google.com/apikey)
and a running IOPaint server (./run.sh).
"""
import argparse
import logging
import sys
from pathlib import Path

# load .env + make app/ importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "app"))
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env")
    print("[INFO] Loaded .env")
except ImportError:
    pass

from word_replace import WordReplacer        # noqa: E402
from gemini_ocr import GeminiOCR             # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def _prompt_pairs() -> dict:
    """Interactive: keep asking find/replace until the user is done."""
    pairs = {}
    print("\nEnter the words to replace (blank 'find' to finish):")
    while True:
        find = input("  Find word:     ").strip()
        if not find:
            break
        repl = input("  Replace with:  ").strip()
        pairs[find] = repl
    return pairs


def main():
    ap = argparse.ArgumentParser(
        description="Find a word in an image and replace it (auto-detected).")
    ap.add_argument("image", help="path to the image")
    ap.add_argument("-f", "--find", action="append", default=[],
                    help="word/phrase to find (repeatable)")
    ap.add_argument("-r", "--replace", action="append", default=[],
                    help="replacement for the matching --find (repeatable)")
    ap.add_argument("-o", "--out", help="output path (default <name>_edited.png)")
    ap.add_argument("--model", default="mat",
                    help="inpaint model: mat (default) / lama / zits / <sd ckpt>")
    ap.add_argument("--prompt", default="",
                    help="background prompt (mainly for SD models)")
    ap.add_argument("--font", default="", help="force a font family")
    ap.add_argument("--bold", action="store_true", help="force bold")
    ap.add_argument("--no-open", action="store_true",
                    help="don't auto-open the result when done")
    args = ap.parse_args()

    img = Path(args.image).expanduser()
    if not img.exists():
        sys.exit(f"No such image: {img}")

    if not GeminiOCR().available():
        sys.exit("GEMINI_API_KEY not set. Add it to .env "
                 "(get one at https://aistudio.google.com/apikey).")

    # Build the find→replace map
    if args.find:
        if len(args.replace) != len(args.find):
            sys.exit("Each --find needs a matching --replace.")
        pairs = dict(zip(args.find, args.replace))
    else:
        pairs = _prompt_pairs()
    if not pairs:
        sys.exit("Nothing to replace.")

    print(f"\n[INFO] Replacing in {img.name}: {pairs}\n")
    res = WordReplacer().replace(
        image_path=img,
        replacements=pairs,
        out_path=Path(args.out) if args.out else None,
        model=args.model,
        prompt=args.prompt,
        font_override=args.font,
        bold_override=(True if args.bold else None),
    )

    print("\n" + "=" * 50)
    if res.get("status") == "success":
        print(f"  ✓ Saved → {res['output']}")
        print(f"  Replaced: {res['replaced']}")
        for r in res["regions"]:
            print(f"    • {r['find']!r} → {r['replace']!r}  @ {r['bbox']}")
        if res["not_found"]:
            print(f"  Not found: {res['not_found']}")
        if not args.no_open:
            _open_file(res["output"])
    elif res.get("status") == "no_match":
        print("  ✗ None of the words were found.")
        print(f"  Detected words: {res.get('detected')}")
    print("=" * 50)


def _open_file(path):
    """Open the finished image in the system's default viewer."""
    import platform
    import subprocess
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif platform.system() == "Windows":
            import os
            os.startfile(str(path))             # noqa: pylint
        else:                                    # Linux / *nix
            subprocess.Popen(["xdg-open", str(path)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  ↗ Opened {path}")
    except Exception as e:
        print(f"  (could not auto-open: {e})")


if __name__ == "__main__":
    main()
