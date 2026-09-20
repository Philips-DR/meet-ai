"""Resolve a verbatim quote back to a real span in the timeline.

This module is the reason a model is allowed anywhere near the notes.

The model emits QUOTES, never offsets. Only this file turns a quote into a (start, end)
pair, by searching the timeline for the text. A fabricated quote therefore cannot carry a
plausible-looking timestamp -- there is nothing to find, resolution fails, and the claim is
dropped before it can reach the page. The model produces words; only arithmetic produces
numbers. That is the same rule as "never put a model in emit/", applied to time.

Matching is done over a folded form (lowercase, alphanumerics, single spaces) so that
punctuation, capitalisation and smart quotes cannot cause a true quote to miss. Every
character of the folded haystack remembers which segment it came from, so a match maps
back to real segments and therefore to real offsets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Span:
    """Where a quote actually occurs in the recording."""

    start: float
    end: float
    speaker: int | None
    first_segment: int
    last_segment: int


def _fold(pieces: Iterable[tuple[str, int | None]]) -> Iterator[tuple[str, int | None]]:
    """Normalise text to lowercase alphanumerics and single spaces, keeping ownership.

    One implementation, used for both the haystack and the needle -- two separate
    normalisers would drift apart and quietly stop matching.
    """
    previous_was_space = True  # also suppresses a leading space
    for text, owner in pieces:
        for char in text:
            lowered = char.lower()
            if lowered.isalnum():
                yield lowered, owner
                previous_was_space = False
            elif not previous_was_space:
                yield " ", owner
                previous_was_space = True


def fold_text(text: str) -> str:
    """The folded form of a standalone string, for matching against a TimelineIndex."""
    return "".join(char for char, _ in _fold([(text, None)])).strip()


class TimelineIndex:
    """A searchable, folded view of a timeline's text.

    Built once per timeline, then queried per claim.
    """

    def __init__(self, segments: list[dict]) -> None:
        self.segments = segments
        folded = list(_fold((segment.get("text", ""), i) for i, segment in enumerate(segments)))
        self._haystack = "".join(char for char, _ in folded)
        self._owners = [owner for _, owner in folded]

    def find(self, quote: str) -> Span | None:
        """The span where this quote occurs, or None if it does not occur at all.

        None is the whole point: it is how an invented quote gets caught.
        """
        needle = fold_text(quote)
        if not needle:
            return None

        position = self._haystack.find(needle)
        if position < 0:
            return None

        first = self._owners[position]
        last = self._owners[position + len(needle) - 1]
        if first is None or last is None:  # pragma: no cover - owners are always set here
            return None

        speakers = {
            self.segments[i].get("speaker")
            for i in range(first, last + 1)
        }
        # A quote spanning two people has no single speaker, and guessing one would be
        # exactly the false precision this design exists to avoid.
        speaker = speakers.pop() if len(speakers) == 1 else None

        return Span(
            start=float(self.segments[first]["start"]),
            end=float(self.segments[last]["end"]),
            speaker=speaker,
            first_segment=first,
            last_segment=last,
        )
