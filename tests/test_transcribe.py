"""Tests for the deterministic half of the pipeline.

Everything covered here is a pure function over plain data: no audio is decoded, no model
is loaded, no network is touched. That is the point — these are the places where a silent
arithmetic error would corrupt every timestamp downstream, and they are cheap to pin down.
"""

from transcribe import escape_block_start, plan_chunks, to_paragraphs
from diarize import SpeakerLookup, split_sentences


# ---------------------------------------------------------------------------
# plan_chunks — the offset arithmetic everything else inherits
# ---------------------------------------------------------------------------

def test_short_file_is_one_chunk_covering_everything():
    assert plan_chunks(600.0, 900.0, []) == [(0.0, 600.0)]


def test_chunks_are_contiguous_and_cover_the_whole_file():
    """The invariant that matters most: a gap or an overlap here silently drops or
    duplicates speech, and every timestamp after it lands in the wrong place."""
    chunks = plan_chunks(3600.0, 600.0, [])
    assert chunks[0][0] == 0.0
    assert chunks[-1][1] == 3600.0
    for (_, end), (next_start, _) in zip(chunks, chunks[1:]):
        assert end == next_start


def test_cut_lands_on_the_midpoint_of_a_nearby_silence():
    # Silence at 590-610 straddles the 600s target; its midpoint is 600.0.
    chunks = plan_chunks(1200.0, 600.0, [(596.0, 604.0)])
    assert chunks[0] == (0.0, 600.0)


def test_cut_prefers_the_silence_closest_to_the_target():
    chunks = plan_chunks(1200.0, 600.0, [(560.0, 562.0), (598.0, 602.0)])
    assert chunks[0][1] == 600.0


def test_falls_back_to_a_hard_cut_when_no_silence_is_near():
    """No silence within BOUNDARY_SEARCH of the target: cut exactly on it rather than
    wandering arbitrarily far to find quiet."""
    chunks = plan_chunks(1200.0, 600.0, [(100.0, 120.0)])
    assert chunks[0] == (0.0, 600.0)


def test_ignores_silences_too_close_to_the_chunk_start():
    """A silence just after the previous cut would produce a near-empty chunk."""
    chunks = plan_chunks(1200.0, 600.0, [(20.0, 24.0)])
    assert chunks[0][1] == 600.0


# ---------------------------------------------------------------------------
# SpeakerLookup — who was talking during a span
# ---------------------------------------------------------------------------

TURNS = [
    {"start": 0.0, "end": 10.0, "speaker": 0},
    {"start": 10.0, "end": 20.0, "speaker": 1},
    {"start": 30.0, "end": 40.0, "speaker": 2},
]


def test_span_inside_one_turn_gets_that_speaker():
    assert SpeakerLookup(TURNS).at(2.0, 3.0) == 0


def test_span_straddling_two_turns_goes_to_the_larger_overlap():
    # 9.0-13.0 is 1s of speaker 0 and 3s of speaker 1.
    assert SpeakerLookup(TURNS).at(9.0, 13.0) == 1


def test_span_in_a_gap_falls_back_to_the_nearest_turn():
    """A word landing between turns still belongs to someone — silence in the
    diarizer's output is not evidence that nobody spoke."""
    assert SpeakerLookup(TURNS).at(24.0, 25.0) is not None


def test_no_turns_at_all_yields_no_speaker():
    assert SpeakerLookup([]).at(1.0, 2.0) is None


# ---------------------------------------------------------------------------
# split_sentences — the unit speaker voting happens over
# ---------------------------------------------------------------------------

def words(*text):
    return [{"word": w, "start": float(i), "end": float(i) + 1} for i, w in enumerate(text)]


def test_splits_on_sentence_ending_punctuation():
    assert len(split_sentences(words("Hello", " there.", " Next", " one."))) == 2


def test_trailing_fragment_without_punctuation_is_kept():
    """Dropping it would lose the last words of every recording that ends mid-sentence."""
    sentences = split_sentences(words("Done.", " And", " then"))
    assert len(sentences) == 2
    assert "".join(w["word"] for w in sentences[-1]) == " And then"


def test_question_and_exclamation_end_sentences_too():
    assert len(split_sentences(words("Really?", " Yes!", " Ok."))) == 3


# ---------------------------------------------------------------------------
# escape_block_start — a spoken line must stay a spoken line
# ---------------------------------------------------------------------------

def test_spoken_number_does_not_become_a_list_item():
    # Real transcript lines: "15. For every." / "160. That's one more."
    assert escape_block_start("160. That's one more.") == "160\\. That's one more."


def test_spoken_dash_does_not_become_a_bullet():
    assert escape_block_start("- so anyway") == "\\- so anyway"


def test_spoken_hash_does_not_become_a_heading():
    assert escape_block_start("# one") == "\\# one"
    assert escape_block_start("### three") == "##\\# three"


def test_angle_bracket_does_not_become_a_blockquote():
    assert escape_block_start("> quoted") == "\\> quoted"


def test_ordinary_speech_is_left_exactly_alone():
    assert escape_block_start("Okay, so this is the background.") == "Okay, so this is the background."


def test_a_number_mid_sentence_is_not_escaped():
    assert escape_block_start("He said 15. Then he left.") == "He said 15. Then he left."


def test_a_bare_number_without_a_following_space_is_not_a_list():
    assert escape_block_start("15.5 percent") == "15.5 percent"


# ---------------------------------------------------------------------------
# to_paragraphs — grouping
# ---------------------------------------------------------------------------

def test_a_speaker_change_always_starts_a_new_paragraph():
    """However tight the timing: running two people together reads worse than a short line."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "one", "speaker": 0},
        {"start": 1.0, "end": 2.0, "text": "two", "speaker": 1},
    ]
    assert len(to_paragraphs(segments)) == 2


def test_a_long_pause_starts_a_new_paragraph():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "one"},
        {"start": 5.0, "end": 6.0, "text": "two"},
    ]
    assert len(to_paragraphs(segments)) == 2
