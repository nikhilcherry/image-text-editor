"""Tests for TextOverlay._parse_color — pure string parsing, no font/image I/O."""
from text_overlay import TextOverlay


def _parse(color, opacity=1.0):
    return TextOverlay._parse_color(color, opacity)


def test_hex6():
    assert _parse("#ff8800") == (255, 136, 0, 255)


def test_hex6_lowercase_and_uppercase_mixed():
    assert _parse("#Ff8800") == (255, 136, 0, 255)


def test_hex8_alpha_overrides_opacity():
    # the 4th byte in #rrggbbaa should win over the separate opacity arg
    assert _parse("#ff880080", opacity=1.0) == (255, 136, 0, 128)


def test_hex6_uses_opacity_arg_for_alpha():
    assert _parse("#000000", opacity=0.5) == (0, 0, 0, 127)


def test_rgb_function():
    assert _parse("rgb(10, 20, 30)") == (10, 20, 30, 255)


def test_rgba_function_with_float_alpha():
    assert _parse("rgba(10, 20, 30, 0.5)") == (10, 20, 30, 127)


def test_rgba_function_without_alpha_falls_back_to_opacity():
    assert _parse("rgba(1, 2, 3)", opacity=1.0) == (1, 2, 3, 255)


def test_named_color():
    assert _parse("red") == (255, 0, 0, 255)


def test_whitespace_is_stripped():
    assert _parse("  #ff0000  ") == (255, 0, 0, 255)


def test_unparseable_color_falls_back_to_black():
    assert _parse("not-a-real-color-xyz") == (0, 0, 0, 255)
