"""Render verified notes as Markdown, and as JSON for anything that needs the spans.

The Markdown is what a person reads and what docu-ai compiles. The JSON is what a lint or
a future UI reads -- it keeps the machine-readable offsets that the Markdown can only show
as a human-facing timestamp.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from meetnotes.verify import VerifiedClaim, VerifiedItem, VerifiedMinutes, VerifiedNotes

# Imported rather than reimplemented: one definition of how a timestamp is formatted and
# of which characters have to be escaped so speech does not parse as markdown structure.
from transcribe import escape_block_start, hhmmss

NOTES_VERSION = 1


def _speaker_label(claim: VerifiedClaim) -> str:
    """Attribution is deliberately hedged -- diarization is not reliable enough to assert."""
    if claim.span.speaker is None:
        return ""
    return f"Speaker {claim.span.speaker + 1}, "


def _claim_lines(claim: VerifiedClaim) -> list[str]:
    owner = f" **({claim.owner})**" if claim.owner else ""
    return [
        f"- {escape_block_start(claim.text)}{owner}",
        f"  > “{claim.quote}” — {_speaker_label(claim)}{hhmmss(claim.span.start)}",
        "",
    ]


def _section(title: str, claims: list[VerifiedClaim], empty: str) -> list[str]:
    lines = [f"## {title}", ""]
    if not claims:
        # Saying nothing was settled is information. Omitting the section would read as
        # an oversight instead of a finding.
        return [*lines, empty, ""]
    for claim in claims:
        lines += _claim_lines(claim)
    return lines


def render_markdown(notes: VerifiedNotes, title: str, source: str, duration: float) -> str:
    lines = [
        "---",
        f'source: "{source}"',
        f"duration: {hhmmss(duration)}",
        f"generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "---",
        "",
        f"# {title}",
        "",
        "## Summary",
        "",
        escape_block_start(notes.summary),
        "",
    ]
    lines += _section("Decisions", notes.decisions, "No decisions were recorded.")
    lines += _section("Action items", notes.actions, "No action items were recorded.")
    lines += _section("Open questions", notes.questions, "No open questions were recorded.")
    return "\n".join(lines).rstrip() + "\n"


def _claim_json(claim: VerifiedClaim) -> dict:
    return {
        "text": claim.text,
        "quote": claim.quote,
        "owner": claim.owner,
        "start": claim.span.start,
        "end": claim.span.end,
        "speaker": claim.span.speaker,
    }


def write_notes(out_path: Path, notes: VerifiedNotes, title: str, source: str,
                duration: float) -> tuple[Path, Path]:
    """Write the Markdown and the span-carrying JSON beside each other."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_markdown(notes, title, source, duration), encoding="utf-8"
    )

    # out_path is already "<stem>.notes.md", so swapping the extension alone gives
    # "<stem>.notes.json". Appending ".notes.json" would yield "<stem>.notes.notes.json".
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "version": NOTES_VERSION,
                "source": source,
                "title": title,
                "meeting": notes.meeting,
                "summary": notes.summary,
                "decisions": [_claim_json(c) for c in notes.decisions],
                "actions": [_claim_json(c) for c in notes.actions],
                "questions": [_claim_json(c) for c in notes.questions],
                "dropped": notes.dropped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path, json_path


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------

MINUTES_VERSION = 1

# Phone and recorder apps name files by when they started: 20260812_102457.m4a.
_RECORDED_AT = re.compile(r"(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")


def recorded_at(source: str) -> dt.datetime | None:
    """When the recording began, if its file name says so. Never guessed otherwise."""
    match = _RECORDED_AT.search(Path(source).stem)
    if not match:
        return None
    try:
        return dt.datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def _titled(kind: str, meeting: str, source: str) -> str:
    """"Minutes of the third quarter meeting of ...", falling back to the recording's date,
    then its file name. Never a bare file stem when anything better is known."""
    if meeting:
        return f"{kind} of the {meeting}"
    when = recorded_at(source)
    return f"{kind} of the meeting of {when:%-d %B %Y}" if when else f"{kind} — {Path(source).stem}"


def minutes_title(minutes: VerifiedMinutes, source: str) -> str:
    return _titled("Minutes", minutes.meeting, source)


def notes_title(notes: VerifiedNotes, source: str) -> str:
    return _titled("Notes", notes.meeting, source)


def _minute_line(claim: VerifiedClaim) -> str:
    """Minutes read as minutes: no quote under every line, just where it can be heard. The
    quote is kept in the JSON, and the timestamp is enough to find it in the recording."""
    owner = f" — **{claim.owner}**" if claim.owner else ""
    return f"- {escape_block_start(claim.text)}{owner} ({hhmmss(claim.span.start)})"


def _labelled(label: str, claims: list[VerifiedClaim]) -> list[str]:
    if not claims:
        return []
    return [f"**{label}**", "", *(_minute_line(c) for c in claims), ""]


def _item_lines(item: VerifiedItem) -> list[str]:
    # A numbered heading would read as a list marker to anything that parses it; the order
    # of the sections already numbers them.
    heading = re.sub(r"^\s*\d+[.)]\s*", "", item.heading)
    lines = [f"## {escape_block_start(heading)}", ""]
    lines += [*(_minute_line(c) for c in item.discussion), ""] if item.discussion else []
    lines += _labelled("Resolved", item.resolutions)
    lines += _labelled("Action", item.actions)
    return lines


def render_minutes(minutes: VerifiedMinutes, source: str, duration: float) -> str:
    when = recorded_at(source)
    details = []
    if when:
        details.append(f"**Date:** {when:%A %-d %B %Y}, recording began {when:%H:%M}")
    details.append(f"**Length of recording:** {hhmmss(duration)}")

    lines = [
        "---",
        f'source: "{source}"',
        f"duration: {hhmmss(duration)}",
        f"generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "---",
        "",
        f"# {minutes_title(minutes, source)}",
        "",
        "  \n".join(details),
        "",
        "## Attendance",
        "",
        # A recording cannot say who was in the room. Leaving a visible gap is honest;
        # a list inferred from voices would be the one invented part of the document.
        "Not recorded in the transcript. To be completed by the secretary:",
        "",
        "- Present:",
        "- Apologies:",
        "",
    ]
    if minutes.opening:
        lines += ["## Opening", "", *(_minute_line(c) for c in minutes.opening), ""]
    for item in minutes.items:
        lines += _item_lines(item)
    if minutes.closing:
        lines += ["## Closing", "", *(_minute_line(c) for c in minutes.closing), ""]
    lines += [
        "---",
        "",
        "*Each point is followed by the time in the recording where it can be heard. Points "
        "that could not be matched word for word to the transcript were left out. Names are "
        "spelled as the transcription heard them and should be checked before circulation.*",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _item_json(item: VerifiedItem) -> dict:
    return {
        "heading": item.heading,
        "discussion": [_claim_json(c) for c in item.discussion],
        "resolutions": [_claim_json(c) for c in item.resolutions],
        "actions": [_claim_json(c) for c in item.actions],
    }


def write_minutes(out_path: Path, minutes: VerifiedMinutes, source: str,
                  duration: float) -> tuple[Path, Path]:
    """Write the Markdown and the span-carrying JSON beside each other."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_minutes(minutes, source, duration), encoding="utf-8")
    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "version": MINUTES_VERSION,
                "source": source,
                "title": minutes_title(minutes, source),
                "meeting": minutes.meeting,
                "opening": [_claim_json(c) for c in minutes.opening],
                "items": [_item_json(i) for i in minutes.items],
                "closing": [_claim_json(c) for c in minutes.closing],
                "dropped": minutes.dropped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path, json_path
