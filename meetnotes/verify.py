"""Layer 6 -- resolve every claim to a real span, and drop the ones that do not.

Reads only. Nothing here rewrites what the model said; a claim either has evidence in the
timeline or it does not reach the page. The dropped list is kept rather than discarded
because a run that drops claims is telling you something about the prompt or the model,
and silently swallowing that would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from meetnotes.extract import RawNotes
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
