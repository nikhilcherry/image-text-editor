"""Tests for AutoTextDetector._merge_boxes — pure box-merging geometry,
no OCR/CV model calls involved."""
from auto_detector import AutoTextDetector


def _merge(boxes, W=1000, H=1000, gap=8):
    return AutoTextDetector._merge_boxes(boxes, W, H, gap)


def test_empty_input():
    assert _merge([]) == []


def test_single_box_passes_through():
    assert _merge([(10, 10, 50, 50)]) == [(10, 10, 50, 50)]


def test_overlapping_boxes_merge():
    result = _merge([(10, 10, 50, 50), (40, 40, 80, 80)])
    assert result == [(10, 10, 80, 80)]


def test_boxes_within_gap_merge():
    # 5px apart horizontally, gap=8 -> should merge
    result = _merge([(0, 0, 20, 20), (25, 0, 45, 20)], gap=8)
    assert result == [(0, 0, 45, 20)]


def test_boxes_beyond_gap_do_not_merge():
    # 20px apart horizontally, gap=8 -> should NOT merge
    result = _merge([(0, 0, 20, 20), (40, 0, 60, 20)], gap=8)
    assert sorted(result) == [(0, 0, 20, 20), (40, 0, 60, 20)]


def test_chain_merge_transitively():
    # A touches B, B touches C, A does not directly touch C -- all three
    # must end up in one merged box (this is what the "changed" re-loop
    # in _merge_boxes is for).
    result = _merge([(0, 0, 20, 20), (18, 0, 38, 20), (36, 0, 56, 20)], gap=2)
    assert result == [(0, 0, 56, 20)]


def test_result_clipped_to_image_bounds():
    result = _merge([(-10, -10, 30, 30)], W=20, H=20)
    assert result == [(0, 0, 20, 20)]


def test_degenerate_tiny_box_dropped_after_clipping():
    # entirely outside the image -> clips to a <=2px sliver and is dropped
    result = _merge([(-10, -10, -9, -9)], W=100, H=100)
    assert result == []
