"""Tests for verification and rendering.

The first test in this file is the one the whole milestone rests on: a claim the model
invented must not survive. Everything else is presentation.
"""

from meetnotes.extract import RawNotes, transcript_for_model
from meetnotes.render import render_markdown
from meetnotes.spans import TimelineIndex
from meetnotes.verify import verify


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "We should move billing to the new provider.", "speaker": 0},
    {"start": 4.0, "end": 8.0, "text": "Kwame will send the contract on Friday.", "speaker": 1},
]
INDEX = TimelineIndex(SEGMENTS)


def test_an_invented_claim_is_dropped():
    """The load-bearing test of M4. The model said it; the transcript does not support it;
    it never reaches the page."""
    raw = RawNotes(
        summary="",
        decisions=[{"text": "Budget doubled", "quote": "we agreed to double the budget"}],
        actions=[],
        questions=[],
    )
    notes = verify(raw, INDEX)
    assert notes.decisions == []
    assert len(notes.dropped) == 1
    assert notes.dropped[0]["kind"] == "decision"


def test_a_supported_claim_survives_and_carries_a_real_span():
    raw = RawNotes(
        summary="Billing.",
        decisions=[{"text": "Move billing", "quote": "move billing to the new provider"}],
        actions=[],
        questions=[],
    )
    notes = verify(raw, INDEX)
    assert len(notes.decisions) == 1
    assert notes.decisions[0].span.start == 0.0
    assert notes.dropped == []


def test_claim_count_covers_every_section():
    raw = RawNotes(
        summary="",
        decisions=[{"text": "d", "quote": "move billing"}],
        actions=[{"text": "a", "quote": "send the contract", "owner": "Kwame"}],
        questions=[{"text": "q", "quote": "on Friday"}],
    )
    assert verify(raw, INDEX).claim_count == 3


def test_a_spoken_owner_is_kept():
    """An owner named in the speech is evidence. An owner inferred from a speaker label
    would not be, which is why the prompt forbids it."""
    raw = RawNotes(
        summary="",
        decisions=[],
        actions=[{"text": "Send contract", "quote": "send the contract on Friday", "owner": "Kwame"}],
        questions=[],
    )
    assert verify(raw, INDEX).actions[0].owner == "Kwame"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _rendered(raw):
    return render_markdown(verify(raw, INDEX), "Test — notes", "test.m4a", 8.0)


def test_rendered_notes_carry_the_quote_and_a_timestamp():
    markdown = _rendered(RawNotes(
        summary="A meeting.",
        decisions=[{"text": "Move billing", "quote": "move billing to the new provider"}],
        actions=[], questions=[],
    ))
    assert "move billing to the new provider" in markdown
    assert "00:00:00" in markdown


def test_an_empty_section_says_so_rather_than_vanishing():
    """A meeting that decided nothing is a finding. An omitted section reads as a bug."""
    markdown = _rendered(RawNotes(summary="Chat.", decisions=[], actions=[], questions=[]))
    assert "No decisions were recorded." in markdown


def test_claim_text_that_starts_with_a_number_is_escaped():
    """Same gotcha as the transcript renderer: a claim beginning '15.' must not become
    the fifteenth item of a list once docu-ai parses it."""
    markdown = _rendered(RawNotes(
        summary="",
        decisions=[{"text": "160. was the figure", "quote": "move billing"}],
        actions=[], questions=[],
    ))
    assert "160\\. was the figure" in markdown


def test_the_model_never_sees_a_timestamp():
    """If it cannot see a number, it cannot echo one. This is the mechanism, so it gets
    a test rather than a comment."""
    prompt = transcript_for_model(SEGMENTS)
    assert "0.0" not in prompt
    assert "00:00" not in prompt
    assert "We should move billing" in prompt


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

def test_bedrock_ids_get_the_anthropic_prefix():
    from meetnotes.client import ModelConfig

    assert ModelConfig(region="us-east-1").bedrock_model_id == "anthropic.claude-opus-5"


def test_an_already_prefixed_id_is_not_prefixed_twice():
    from meetnotes.client import ModelConfig

    config = ModelConfig(region="us-east-1", model="anthropic.claude-opus-5")
    assert config.bedrock_model_id == "anthropic.claude-opus-5"
