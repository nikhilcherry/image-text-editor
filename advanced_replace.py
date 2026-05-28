#!/usr/bin/env python3
"""
Advanced Text Replacement CLI
==============================
Standalone command-line tool that runs the full pipeline on images
located in ~/Desktop (or a specified folder) WITHOUT the web UI.

Usage:
  python advanced_replace.py "New Text"
  python advanced_replace.py "New Text" --folder ~/Desktop/photos
  python advanced_replace.py --map '{"Hello":"Hola","World":"Mundo"}' --model mat
  python advanced_replace.py "SALE" --font-size 72 --color "#FF0000"

Outputs:
  • <original>_edited.png  saved next to each source image
  • ~/Desktop/text_replacement_log.json  appended with run summary

Requirements:
  • Python ≥ 3.10 with venv_app activated  (cd image-text-editor && source venv_app/bin/activate)
  • IOPaint server running on :8080         (./run.sh)
  • GROQ_API_KEY in .env for best detection (optional but recommended)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add app/ to path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent / "app"))

logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Load .env (Groq API key, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
    log.info("Loaded .env")
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Text Replacement — replaces all text in images automatically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "replacement_text",
        nargs="?",
        default=None,
        help='Replacement text applied to ALL detected regions (e.g. "New Title")',
    )
    parser.add_argument(
        "--folder", "-f",
        default=str(Path.home() / "Desktop"),
        help="Folder to scan for images (default: ~/Desktop)",
    )
    parser.add_argument(
        "--map", "-m",
        default=None,
        help='JSON replacement map: \'{"Original":"Replacement"}\'. '
             'Overrides replacement_text if provided.',
    )
    parser.add_argument(
        "--model",
        default="lama",
        choices=["lama", "mat", "zits", "sd-inpainting"],
        help="IOPaint model to use (default: lama)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Inpainting prompt (only relevant for sd-inpainting model)",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Override font size in pixels (default: auto-detect from image)",
    )
    parser.add_argument(
        "--color",
        default=None,
        help='Override text color as hex (e.g. "#FF0000"). Default: auto-detect.',
    )
    parser.add_argument(
        "--iopaint-url",
        default="http://127.0.0.1:8080",
        help="IOPaint server URL (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--log",
        default=str(Path.home() / "Desktop" / "text_replacement_log.json"),
        help="Path to the output log file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan folder and show what would be processed — don't modify anything",
    )

    args = parser.parse_args()

    # ── Build replacement map ──────────────────────────────────────────
    replacement_map: dict = {}

    if args.map:
        try:
            replacement_map = json.loads(args.map)
        except json.JSONDecodeError as e:
            log.error("Invalid --map JSON: %s", e)
            sys.exit(1)

    if args.replacement_text:
        replacement_map["*"] = args.replacement_text

    if not replacement_map:
        log.error(
            "Provide either a replacement_text positional argument or --map. "
            "Example: python advanced_replace.py \"My New Text\""
        )
        sys.exit(1)

    # ── Resolve folder ─────────────────────────────────────────────────
    folder = Path(args.folder).expanduser()
    if not folder.exists():
        log.error("Folder not found: %s", folder)
        sys.exit(1)

    # ── Import pipeline (deferred so --help works without deps) ───────
    try:
        from batch_processor import BatchProcessor, ALLOWED_EXT
    except ImportError as e:
        log.error(
            "Could not import pipeline modules: %s\n"
            "Make sure you have activated venv_app:\n"
            "  cd image-text-editor && source venv_app/bin/activate",
            e,
        )
        sys.exit(1)

    # ── Dry-run: just list files ───────────────────────────────────────
    if args.dry_run:
        images = [
            p for p in sorted(folder.iterdir())
            if p.is_file()
            and p.suffix.lower() in ALLOWED_EXT
            and not p.stem.endswith("_edited")
        ]
        if not images:
            print(f"No processable images found in {folder}")
        else:
            print(f"Would process {len(images)} image(s) in {folder}:")
            for img in images:
                print(f"  • {img.name}  ({img.stat().st_size // 1024} KB)")
            print(f"\nReplacement map: {replacement_map}")
            print(f"Model: {args.model}")
        sys.exit(0)

    # ── Run batch ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Advanced Text Replacement")
    log.info("  Folder  : %s", folder)
    log.info("  Map     : %s", replacement_map)
    log.info("  Model   : %s", args.model)
    log.info("  Log     : %s", args.log)
    log.info("=" * 60)

    def progress(name: str, frac: float):
        pct = int(frac * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct:3d}%  {name:<40}", end="", flush=True)
        if name == "done":
            print()

    bp = BatchProcessor(
        iopaint_url = args.iopaint_url,
        log_path    = Path(args.log),
    )

    try:
        summary = bp.process_folder(
            folder             = folder,
            replacement_map    = replacement_map,
            model              = args.model,
            prompt             = args.prompt,
            font_size_override = args.font_size,
            color_override     = args.color,
            progress_cb        = progress,
        )
    except KeyboardInterrupt:
        log.warning("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        log.error("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)

    # ── Print summary ──────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Processed : {summary['processed']}")
    print(f"  Succeeded : {summary['succeeded']}")
    print(f"  No text   : {summary.get('no_text', 0)}")
    print(f"  Failed    : {summary['failed']}")
    print(f"  Log file  : {args.log}")
    print("=" * 60)

    for f in summary["files"]:
        icon = "✓" if f["status"] == "success" else ("—" if f["status"] == "no_text" else "✗")
        name = Path(f["file"]).name
        out  = Path(f.get("output_path", "")).name if f.get("output_path") else ""
        note = f"→ {out}  ({f.get('replaced', 0)}/{f.get('regions', 0)} regions)" if out else (f.get("error", ""))
        print(f"  {icon} {name:<40} {note}")

    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
