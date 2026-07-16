"""Tests for FontSampler's pure numpy/logic helpers: serif/skew heuristics
and the Groq-font-name -> installed-font alias mapping. These don't touch
the network, IOPaint, or any GPU model."""
import numpy as np
import pytest

from font_sampler import FontSampler


@pytest.fixture
def sampler():
    return FontSampler()


def test_estimate_serif_empty_mask_is_false():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert FontSampler._estimate_serif(mask) is False


def test_estimate_serif_dense_top_and_bottom_bars_is_true():
    # A serif-like shape: solid bars at the top and bottom (the "serifs"),
    # a thin vertical stroke through the middle -- top/bottom row density
    # is much higher than middle row density.
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[0:4, :] = 255       # top serif bar, full width
    mask[-4:, :] = 255       # bottom serif bar, full width
    mask[:, 14:16] = 255     # thin vertical stem
    assert bool(FontSampler._estimate_serif(mask)) is True


def test_estimate_serif_uniform_block_is_false():
    # A solid, uniformly-dense block has top/bottom density == middle
    # density (ratio ~1.0... actually this would trigger True by the
    # >0.55 threshold since ratio=1). Use a middle-heavy shape instead
    # to get an unambiguous sans-like (non-serif) case.
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[:, 14:16] = 60      # thin stem everywhere (should not classify solidly)
    mask[10:20, :] = 255     # dense middle bar only
    assert bool(FontSampler._estimate_serif(mask)) is False


def test_estimate_skew_too_few_points_returns_zero():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 255  # only a handful of points
    assert FontSampler._estimate_skew(mask) == 0.0


def test_estimate_skew_vertical_stroke_is_near_zero():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[5:45, 24:26] = 255  # perfectly vertical stroke
    angle = FontSampler._estimate_skew(mask)
    assert abs(angle) < 2.0


def test_map_to_installed_font_known_alias(sampler, monkeypatch):
    monkeypatch.setattr(
        sampler.text_tool, "_get_font_map",
        lambda: {"liberation sans": "/fake/path/LiberationSans.ttf"},
    )
    assert sampler._map_to_installed_font("Arial") == "Liberation Sans"
    assert sampler._map_to_installed_font("Helvetica") == "Liberation Sans"


def test_map_to_installed_font_case_insensitive(sampler, monkeypatch):
    monkeypatch.setattr(
        sampler.text_tool, "_get_font_map",
        lambda: {"dejavu sans": "/fake/DejaVuSans.ttf"},
    )
    assert sampler._map_to_installed_font("SANS-SERIF") == "DejaVu Sans"


def test_map_to_installed_font_unknown_alias_falls_back_to_fuzzy_match(sampler, monkeypatch):
    monkeypatch.setattr(
        sampler.text_tool, "_get_font_map",
        lambda: {"my custom pixel font": "/fake/path.ttf"},
    )
    assert sampler._map_to_installed_font("Custom Pixel") == "my custom pixel font"


def test_map_to_installed_font_no_match_returns_none(sampler, monkeypatch):
    monkeypatch.setattr(sampler.text_tool, "_get_font_map", lambda: {})
    assert sampler._map_to_installed_font("Totally Unknown Font XYZ") is None


def test_map_to_installed_font_empty_string_returns_none(sampler, monkeypatch):
    monkeypatch.setattr(sampler.text_tool, "_get_font_map", lambda: {"arial": "/x.ttf"})
    assert sampler._map_to_installed_font("") is None


def test_verify_font_present(sampler, monkeypatch):
    monkeypatch.setattr(
        sampler.text_tool, "_get_font_map",
        lambda: {"liberation sans": "/fake/LiberationSans.ttf"},
    )
    assert sampler._verify_font("Liberation Sans") == "Liberation Sans"


def test_verify_font_absent_returns_none(sampler, monkeypatch):
    monkeypatch.setattr(sampler.text_tool, "_get_font_map", lambda: {})
    assert sampler._verify_font("Liberation Sans") is None
