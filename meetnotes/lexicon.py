"""Layer 4 -- correction, done deterministically.

A lexicon maps a canonical spelling to the ways an ASR model mangles it:

    {"AyaData": ["aya data", "ayadata"], "LiDAR": ["lidar", "lie dar"]}

Deliberately NOT a model pass. A dictionary substitution is exact, instant, reviewable and
testable, and it cannot invent a correction that was never asked for. The model seam stays
open above this for the cases a dictionary genuinely cannot reach, but nothing should be
handed to a model that arithmetic or a lookup already does correctly.

The one hard rule: correction may change the text inside a span. It may never move a span
boundary, because every offset downstream is derived from those boundaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_lexicon(path: Path) -> dict[str, list[str]]:
    """Read a lexicon file: {canonical: [alias, ...]}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("a lexicon must be an object of {canonical: [alias, ...]}")
    return {str(k): [str(a) for a in v] for k, v in data.items()}


def _pattern(lexicon: dict[str, list[str]]) -> re.Pattern | None:
    """One alternation over every alias, longest first.

    Longest-first matters: with "aya" and "aya data" both listed, the shorter alias would
    otherwise win and leave "AyaData data" behind.
    """
    aliases: list[tuple[str, str]] = []
    for canonical, alias_list in lexicon.items():
        for alias in [*alias_list, canonical]:
            if alias:
                aliases.append((alias, canonical))
    if not aliases:
        return None

    aliases.sort(key=lambda pair: len(pair[0]), reverse=True)
    joined = "|".join(re.escape(alias) for alias, _ in aliases)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def apply_lexicon(segments: list[dict], lexicon: dict[str, list[str]]) -> list[dict]:
    """Return segments with corrected text and untouched timings."""
    pattern = _pattern(lexicon)
    if pattern is None:
        return list(segments)

    canonical_for = {
        alias.lower(): canonical
        for canonical, aliases in lexicon.items()
        for alias in [*aliases, canonical]
    }

    def replace(match: re.Match) -> str:
        return canonical_for.get(match.group(0).lower(), match.group(0))

    # Rebuilt rather than mutated, and start/end are copied across verbatim.
    return [
        {**segment, "text": pattern.sub(replace, segment.get("text", ""))}
        for segment in segments
    ]
