"""Tests for minutes: verification first, because it is the reason minutes can be trusted
enough to circulate; then the shape a secretary expects."""

import datetime as dt

from meetnotes.extract import RawMinutes
from meetnotes.operations import MAX_READ_CHARS, _paragraphs
from meetnotes.render import minutes_title, recorded_at, render_minutes
from meetnotes.spans import TimelineIndex
from meetnotes.verify import verify_minutes

SEGMENTS = [
    {"start": 0.0, "end": 5.0, "text": "We observe a minute silence for our departed members.", "speaker": None},
    {"start": 5.0, "end": 9.0, "text": "Welcome to the third quarter meeting of the Tema branch.", "speaker": None},
    {"start": 60.0, "end": 66.0, "text": "I move for the acceptance of the minutes. Seconded by Godwin.", "speaker": None},
    {"start": 120.0, "end": 126.0, "text": "Medical claims are taking three months to be paid.", "speaker": None},
    {"start": 130.0, "end": 136.0, "text": "Jonah, please write to the authority about it.", "speaker": None},
]
INDEX = TimelineIndex(SEGMENTS)


def claim(text, quote, owner=None):
    return {"text": text, "quote": quote, **({"owner": owner} if owner is not None else {})}


def raw(**overrides):
    base = dict(
        meeting="third quarter meeting of the Tema branch",
        meeting_quote="third quarter meeting of the Tema branch",
        opening=[claim("A minute's silence was observed.", "we observe a minute silence")],
        items=[
            {"heading": "Medical claims",
             "discussion": [claim("Claims were slow.", "taking three months to be paid")],
             "resolutions": [],
             "actions": [claim("Write to the authority.", "please write to the authority", "Jonah")]},
            {"heading": "Minutes of the previous meeting",
             "discussion": [],
             "resolutions": [claim("The minutes were accepted, seconded by Godwin.",
                                   "I move for the acceptance of the minutes. Seconded by Godwin.")],
             "actions": []},
        ],
        closing=[],
    )
    return RawMinutes(**{**base, **overrides})


def test_an_invented_point_never_reaches_the_minutes():
    minutes = verify_minutes(raw(closing=[claim("Next meeting on 4 November.", "we meet again on the fourth of November")]), INDEX)
    assert minutes.closing == []
    assert minutes.dropped == [{"kind": "closing", "text": "Next meeting on 4 November.",
                                "quote": "we meet again on the fourth of November"}]


def test_an_item_with_no_evidence_is_dropped_whole_heading_and_all():
    invented = {"heading": "Election of officers",
                "discussion": [claim("New officers were elected.", "we elected new officers")],
                "resolutions": [], "actions": []}
    minutes = verify_minutes(raw(items=[*raw().items, invented]), INDEX)
    assert "Election of officers" not in [i.heading for i in minutes.items]
    assert {"kind": "item", "text": "Election of officers", "quote": ""} in minutes.dropped


def test_items_follow_the_order_the_meeting_took_them_not_the_models():
    minutes = verify_minutes(raw(), INDEX)
    assert [i.heading for i in minutes.items] == ["Minutes of the previous meeting", "Medical claims"]


def test_the_meeting_name_is_a_claim_like_any_other():
    """The title is the first thing a reader sees; a confident wrong one is worse than none."""
    unsupported = verify_minutes(raw(meeting="annual general meeting", meeting_quote="our annual general meeting"), INDEX)
    assert unsupported.meeting == ""
    assert minutes_title(unsupported, "20260812_102457.m4a") == "Minutes of the meeting of 12 August 2026"
    assert minutes_title(verify_minutes(raw(), INDEX), "x.m4a") == \
        "Minutes of the third quarter meeting of the Tema branch"


def test_the_date_comes_from_the_file_name_or_not_at_all():
    assert recorded_at("20260812_102457.m4a") == dt.datetime(2026, 8, 12, 10, 24, 57)
    assert recorded_at("board meeting.m4a") is None
    assert recorded_at("20261399_999999.m4a") is None


def test_rendered_minutes_read_like_minutes():
    text = render_minutes(verify_minutes(raw(), INDEX), "20260812_102457.m4a", 7510.0)
    assert "# Minutes of the third quarter meeting of the Tema branch" in text
    assert "**Date:** Wednesday 12 August 2026, recording began 10:24" in text
    # Attendance cannot come from a recording: a visible gap, never an inferred list.
    assert "To be completed by the secretary" in text
    assert "- Present:" in text
    assert "- The minutes were accepted, seconded by Godwin. (00:01:00)" in text
    assert "- Write to the authority. — **Jonah** (00:02:10)" in text
    # No quote under each line -- the timestamp is the citation; the quote lives in the JSON.
    assert "> “" not in text


def test_a_numbered_heading_loses_its_number():
    """'## 1. Medical claims' is a heading, but its text starts like a list item, and that is
    exactly what docu-ai's residue lint looks for at the start of a paragraph."""
    numbered = [{**raw().items[0], "heading": "1. Medical claims"}]
    text = render_minutes(verify_minutes(raw(items=numbered), INDEX), "x.m4a", 10.0)
    assert "## Medical claims" in text


def test_reading_groups_by_speaker_and_minute_with_one_timestamp_each():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "one", "speaker": 0},
        {"start": 2.0, "end": 4.0, "text": "two", "speaker": 0},
        {"start": 4.0, "end": 6.0, "text": "three", "speaker": 1},
        {"start": 70.0, "end": 72.0, "text": "four", "speaker": 1},
    ]
    assert _paragraphs(segments) == [
        (0.0, "[00:00:00] Speaker 1: one two"),
        (4.0, "[00:00:04] Speaker 2: three"),
        (70.0, "[00:01:10] Speaker 2: four"),
    ]
    assert MAX_READ_CHARS > 100_000  # a two-hour meeting must fit in one read


def test_points_within_a_section_follow_the_recording_too():
    item = {"heading": "Medical claims",
            "discussion": [claim("Write to the authority.", "please write to the authority"),
                           claim("Claims were slow.", "taking three months to be paid")],
            "resolutions": [], "actions": []}
    minutes = verify_minutes(raw(items=[item]), INDEX)
    assert [c.text for c in minutes.items[0].discussion] == ["Claims were slow.", "Write to the authority."]
