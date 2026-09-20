"""Render verified notes as Markdown, and as JSON for anything that needs the spans.

The Markdown is what a person reads and what docu-ai compiles. The JSON is what a lint or
a future UI reads -- it keeps the machine-readable offsets that the Markdown can only show
as a human-facing timestamp.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from meetnotes.verify import VerifiedClaim, VerifiedNotes

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

    json_path = out_path.with_suffix(".notes.json")
    json_path.write_text(
        json.dumps(
            {
                "version": NOTES_VERSION,
                "source": source,
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
