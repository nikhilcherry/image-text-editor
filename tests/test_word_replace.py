"""Tests for WordReplacer._match_targets -- pure word-matching logic,
no OCR/CV/inpaint calls involved."""
from word_replace import WordReplacer


def _match(words, find):
    return WordReplacer._match_targets(words, find)


def _w(text, bbox):
    return {"text": text, "bbox": bbox}


def test_exact_match():
    words = [_w("hello", [0, 0, 1, 1]), _w("world", [2, 2, 3, 3])]
    assert _match(words, "hello") == [[0, 0, 1, 1]]


def test_exact_match_is_case_and_whitespace_insensitive():
    words = [_w("  Hello  ", [0, 0, 1, 1])]
    assert _match(words, "hello") == [[0, 0, 1, 1]]


def test_substring_forward_matches_punctuation_glued_word():
    # search "image" against an OCR'd word "image!" (trailing punctuation)
    words = [_w("image!", [0, 0, 1, 1])]
    assert _match(words, "image") == [[0, 0, 1, 1]]


def test_substring_reverse_matches_longer_ocr_word_fragment():
    # OCR under-split "imagetext" into a single "imag" token
    words = [_w("imag", [0, 0, 1, 1])]
    assert _match(words, "imagetext") == [[0, 0, 1, 1]]


def test_short_ocr_word_does_not_spuriously_match_via_reverse_substring():
    # a short, common OCR token ("a") must not match just because it
    # happens to be a substring of an unrelated search term ("cat") --
    # this would replace the wrong word in the image.
    words = [_w("a", [0, 0, 1, 1]), _w("dog", [2, 2, 3, 3])]
    assert _match(words, "cat") == []


def test_two_letter_ocr_word_does_not_spuriously_match():
    words = [_w("in", [0, 0, 1, 1])]
    assert _match(words, "inspection") == []


def test_no_match_returns_empty_list():
    words = [_w("hello", [0, 0, 1, 1])]
    assert _match(words, "goodbye") == []


def test_multiword_phrase_match_across_consecutive_boxes():
    words = [
        _w("the", [0, 0, 10, 10]),
        _w("quick", [10, 0, 20, 10]),
        _w("fox", [20, 0, 30, 10]),
    ]
    assert _match(words, "the quick fox") == [[0, 0, 30, 10]]
