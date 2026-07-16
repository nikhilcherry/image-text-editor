"""Adds app/ to sys.path so its flat sibling-import modules
(e.g. `from text_overlay import TextOverlay` inside font_sampler.py)
resolve the same way they do when the app is run directly."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
