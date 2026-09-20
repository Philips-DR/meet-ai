"""Tests for the deterministic core of the notes layer.

No model is called here. These are the functions that decide whether a claim survives, so
they are the ones that must be pinned down: if span resolution is wrong, an invented quote
gets a real-looking timestamp and the whole trustworthiness argument collapses.
"""

import json

import pytest

from meetnotes.lexicon import apply_lexicon, load_lexicon
from meetnotes.spans import TimelineIndex, fold_text
from meetnotes.timeline import UnsupportedTimeline, load_timeline


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "Okay, so this is the background.", "speaker": 0},
    {"start": 4.0, "end": 9.0, "text": "We should move billing to the new provider.", "speaker": 1},
    {"start": 9.0, "end": 12.0, "text": "Before the quarter closes.", "speaker": 1},
]


# ---------------------------------------------------------------------------
# Span resolution
# ---------------------------------------------------------------------------

def test_exact_quote_resolves_to_its_segment():
    span = TimelineIndex(SEGMENTS).find("We should move billing to the new provider.")
    assert (span.start, span.end) == (4.0, 9.0)
    assert span.speaker == 1


def test_punctuation_and_case_do_not_prevent_a_match():
    """The model reproduces meaning faithfully and punctuation loosely. Folding is what
    keeps a true quote from being rejected over a comma."""
    span = TimelineIndex(SEGMENTS).find("okay so THIS is the background")
    assert span is not None
    assert span.start == 0.0


def test_quote_spanning_two_segments_takes_the_outer_bounds():
    span = TimelineIndex(SEGMENTS).find("the new provider. Before the quarter closes.")
    assert (span.start, span.end) == (4.0, 12.0)


def test_an_invented_quote_does_not_resolve():
    """The load-bearing test. This is how a hallucinated claim gets caught: there is
    nothing to find, so no span exists, so the claim is dropped."""
    assert TimelineIndex(SEGMENTS).find("We agreed to double the budget.") is None


def test_a_quote_crossing_speakers_has_no_single_speaker():
    """Guessing one would be exactly the false precision this design avoids."""
    span = TimelineIndex(SEGMENTS).find("the background. We should move billing")
    assert span is not None
    assert span.speaker is None


def test_an_empty_quote_resolves_to_nothing():
    assert TimelineIndex(SEGMENTS).find("   ") is None


def test_folding_is_idempotent():
    once = fold_text("Okay, so -- THIS is it!")
    assert fold_text(once) == once


def test_index_over_an_empty_timeline_finds_nothing():
    assert TimelineIndex([]).find("anything") is None


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

def test_alias_is_replaced_with_the_canonical_spelling():
    corrected = apply_lexicon(
        [{"start": 0.0, "end": 1.0, "text": "the aya data team"}],
        {"AyaData": ["aya data"]},
    )
    assert corrected[0]["text"] == "the AyaData team"


def test_replacement_is_case_insensitive():
    corrected = apply_lexicon(
        [{"start": 0.0, "end": 1.0, "text": "AYA DATA and Aya Data"}],
        {"AyaData": ["aya data"]},
    )
    assert corrected[0]["text"] == "AyaData and AyaData"


def test_the_longest_alias_wins():
    """With both listed, the shorter would otherwise match first and leave debris."""
    corrected = apply_lexicon(
        [{"start": 0.0, "end": 1.0, "text": "aya data"}],
        {"AyaData": ["aya", "aya data"]},
    )
    assert corrected[0]["text"] == "AyaData"


def test_correction_never_moves_a_span_boundary():
    """The hard rule. Every offset downstream is derived from these numbers."""
    original = [{"start": 3.5, "end": 9.25, "text": "lidar", "speaker": 2}]
    corrected = apply_lexicon(original, {"LiDAR": ["lidar"]})
    assert corrected[0]["start"] == 3.5
    assert corrected[0]["end"] == 9.25
    assert corrected[0]["speaker"] == 2


def test_an_empty_lexicon_changes_nothing():
    assert apply_lexicon(SEGMENTS, {}) == SEGMENTS


def test_partial_words_are_not_corrected():
    corrected = apply_lexicon(
        [{"start": 0.0, "end": 1.0, "text": "lidarsomething"}],
        {"LiDAR": ["lidar"]},
    )
    assert corrected[0]["text"] == "lidarsomething"


def test_lexicon_file_round_trips(tmp_path):
    path = tmp_path / "lex.json"
    path.write_text(json.dumps({"AyaData": ["aya data"]}), encoding="utf-8")
    assert load_lexicon(path) == {"AyaData": ["aya data"]}


# ---------------------------------------------------------------------------
# Timeline loading
# ---------------------------------------------------------------------------

def test_a_supported_timeline_loads(tmp_path):
    path = tmp_path / "t.timeline.json"
    path.write_text(json.dumps({"version": 1, "segments": SEGMENTS}), encoding="utf-8")
    assert len(load_timeline(path)["segments"]) == 3


def test_an_unknown_version_is_refused_rather_than_misread(tmp_path):
    path = tmp_path / "t.timeline.json"
    path.write_text(json.dumps({"version": 99, "segments": []}), encoding="utf-8")
    with pytest.raises(UnsupportedTimeline):
        load_timeline(path)


def test_a_timeline_without_segments_is_refused(tmp_path):
    path = tmp_path / "t.timeline.json"
    path.write_text(json.dumps({"version": 1}), encoding="utf-8")
    with pytest.raises(UnsupportedTimeline):
        load_timeline(path)
