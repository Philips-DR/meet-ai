"""Layer 6 -- resolve every claim to a real span, and drop the ones that do not.

Reads only. Nothing here rewrites what the model said; a claim either has evidence in the
timeline or it does not reach the page. The dropped list is kept rather than discarded
because a run that drops claims is telling you something about the prompt or the model,
and silently swallowing that would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meetnotes.extract import RawMinutes, RawNotes
from meetnotes.spans import Span, TimelineIndex


@dataclass(frozen=True)
class VerifiedClaim:
    """A claim whose quote was found verbatim in the timeline."""

    text: str
    quote: str
    span: Span
    owner: str = ""


@dataclass
class VerifiedNotes:
    summary: str
    decisions: list[VerifiedClaim] = field(default_factory=list)
    actions: list[VerifiedClaim] = field(default_factory=list)
    questions: list[VerifiedClaim] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    @property
    def claim_count(self) -> int:
        return len(self.decisions) + len(self.actions) + len(self.questions)


def _verify_one(claim: dict, index: TimelineIndex) -> VerifiedClaim | None:
    span = index.find(str(claim.get("quote", "")))
    if span is None:
        return None
    return VerifiedClaim(
        text=str(claim.get("text", "")).strip(),
        quote=str(claim.get("quote", "")).strip(),
        span=span,
        owner=str(claim.get("owner", "")).strip(),
    )


def verify(raw: RawNotes, index: TimelineIndex) -> VerifiedNotes:
    """Keep the claims the transcript actually supports; record the rest as dropped."""
    notes = VerifiedNotes(summary=raw.summary.strip())

    for kind, claims in (
        ("decision", raw.decisions),
        ("action", raw.actions),
        ("question", raw.questions),
    ):
        target = getattr(notes, f"{kind}s")
        for claim in claims:
            verified = _verify_one(claim, index)
            if verified is None:
                notes.dropped.append({"kind": kind, **claim})
            else:
                target.append(verified)

    return notes


# ---------------------------------------------------------------------------
# Minutes
# ---------------------------------------------------------------------------

@dataclass
class VerifiedItem:
    heading: str
    discussion: list[VerifiedClaim] = field(default_factory=list)
    resolutions: list[VerifiedClaim] = field(default_factory=list)
    actions: list[VerifiedClaim] = field(default_factory=list)

    @property
    def claims(self) -> list[VerifiedClaim]:
        return [*self.discussion, *self.resolutions, *self.actions]

    @property
    def start(self) -> float:
        return min(claim.span.start for claim in self.claims)


@dataclass
class VerifiedMinutes:
    meeting: str
    opening: list[VerifiedClaim] = field(default_factory=list)
    items: list[VerifiedItem] = field(default_factory=list)
    closing: list[VerifiedClaim] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)

    @property
    def claim_count(self) -> int:
        return len(self.opening) + len(self.closing) + sum(len(i.claims) for i in self.items)


def _keep(claims: list[dict], kind: str, index: TimelineIndex, dropped: list[dict],
          heading: str | None = None) -> list[VerifiedClaim]:
    kept: list[VerifiedClaim] = []
    for claim in claims:
        verified = _verify_one(claim, index)
        if verified is not None:
            kept.append(verified)
        else:
            dropped.append({"kind": kind, **({"item": heading} if heading else {}), **claim})
    return kept


def verify_minutes(raw: RawMinutes, index: TimelineIndex) -> VerifiedMinutes:
    """The same rule as notes, applied to every point in the minutes.

    Two things are minutes-specific. The meeting's name is a claim like any other: it is
    kept only if the quote naming it resolves, because a confident wrong title is the
    first thing a reader sees. And items are put in the order the meeting took them --
    by where their evidence sits in the recording -- rather than trusting the model's
    order; an item left with no evidence at all is dropped whole, heading included.
    """
    minutes = VerifiedMinutes(meeting="")
    if raw.meeting.strip() and index.find(raw.meeting_quote) is not None:
        minutes.meeting = raw.meeting.strip()
    elif raw.meeting.strip():
        minutes.dropped.append({"kind": "meeting", "text": raw.meeting, "quote": raw.meeting_quote})

    minutes.opening = _keep(raw.opening, "opening", index, minutes.dropped)
    minutes.closing = _keep(raw.closing, "closing", index, minutes.dropped)

    for item in raw.items:
        heading = str(item.get("heading", "")).strip() or "Other matters"
        verified = VerifiedItem(
            heading=heading,
            discussion=_keep(list(item.get("discussion", [])), "discussion", index, minutes.dropped, heading),
            resolutions=_keep(list(item.get("resolutions", [])), "resolution", index, minutes.dropped, heading),
            actions=_keep(list(item.get("actions", [])), "action", index, minutes.dropped, heading),
        )
        if verified.claims:
            minutes.items.append(verified)
        else:
            minutes.dropped.append({"kind": "item", "text": heading, "quote": ""})

    # Within a section too, the recording sets the order: the model groups well but lists
    # points out of sequence (found live -- a presentation read 00:39:48, then 00:34:20).
    by_time = lambda claim: claim.span.start  # noqa: E731
    for claims in (minutes.opening, minutes.closing):
        claims.sort(key=by_time)
    for item in minutes.items:
        for claims in (item.discussion, item.resolutions, item.actions):
            claims.sort(key=by_time)
    minutes.items.sort(key=lambda i: i.start)
    return minutes
