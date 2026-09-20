"""Load a timeline and refuse one we do not understand."""

from __future__ import annotations

import json
from pathlib import Path

# The shapes this half of the tool knows how to read. transcribe.py stamps the version it
# wrote; a file from the future is refused rather than misread, which is the entire reason
# the version field exists.
SUPPORTED_VERSIONS = frozenset({1})


class UnsupportedTimeline(Exception):
    """A timeline file this build cannot safely interpret."""


def load_timeline(path: Path) -> dict:
    """Read a timeline JSON file, checking its version before trusting its contents."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    version = data.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise UnsupportedTimeline(
            f"timeline version {version!r} is not one this build reads "
            f"({sorted(SUPPORTED_VERSIONS)}). Re-run ./transcribe, or use a newer meet-ai."
        )

    if not isinstance(data.get("segments"), list):
        raise UnsupportedTimeline("timeline has no segments list")

    return data


def timeline_for(markdown_path: Path) -> Path:
    """The timeline that sits beside a given transcript. Mirrors transcribe.timeline_path."""
    return Path(markdown_path).with_suffix(".timeline.json")
