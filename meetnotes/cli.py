"""./notes -- turn a timeline into notes somebody will act on.

Consumes what ./transcribe produced. Never decodes audio, never loads a Whisper model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from meetnotes.client import DEFAULT_MODEL, ModelConfig
from meetnotes.lexicon import apply_lexicon, load_lexicon
from meetnotes.render import write_notes
from meetnotes.spans import TimelineIndex
from meetnotes.timeline import UnsupportedTimeline, load_timeline, timeline_for


def resolve_timeline(target: Path) -> Path:
    """Accept either a timeline or the transcript that sits beside one."""
    if target.name.endswith(".timeline.json"):
        return target
    if target.suffix == ".md":
        return timeline_for(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Turn a meet-ai timeline into meeting notes with verifiable quotes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", help="A .timeline.json, or the .md transcript beside one")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Where notes land. Default: beside the timeline")
    parser.add_argument("-l", "--lexicon", default=None,
                        help="JSON of {canonical: [alias, ...]} to correct names and jargon")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"),
                        help="AWS region for Bedrock. Default: $AWS_REGION")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"),
                        help="AWS profile. Default: $AWS_PROFILE, else the default profile")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="How hard the model works on it")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show what would be sent and spend nothing. No model call, "
                             "no files written")
    args = parser.parse_args(argv)

    timeline_path = resolve_timeline(Path(args.target))
    if not timeline_path.exists():
        print(f"!  no timeline at {timeline_path}", file=sys.stderr)
        print("   Transcripts written before timelines existed have none; rerun "
              "./transcribe with -f to produce one.", file=sys.stderr)
        return 1

    try:
        timeline = load_timeline(timeline_path)
    except UnsupportedTimeline as exc:
        print(f"!  {exc}", file=sys.stderr)
        return 1

    segments = timeline["segments"]
    source = timeline.get("source", timeline_path.stem)
    duration = float(timeline.get("duration", 0.0))

    corrections = 0
    if args.lexicon:
        lexicon = load_lexicon(Path(args.lexicon))
        corrected = apply_lexicon(segments, lexicon)
        corrections = sum(
            1 for before, after in zip(segments, corrected)
            if before.get("text") != after.get("text")
        )
        segments = corrected

    title = f"{Path(source).stem} — notes"
    out_dir = Path(args.output_dir) if args.output_dir else timeline_path.parent
    out_path = out_dir / f"{Path(source).stem}.notes.md"

    # The preview every tool in this suite ships: everything up to the model call, and
    # nothing that costs money or writes a file.
    if args.dry_run:
        from meetnotes.extract import transcript_for_model

        prompt = transcript_for_model(segments)
        print(f"·  {timeline_path.name}")
        print(f"   {len(segments)} segments, {len(prompt.split())} words, "
              f"~{len(prompt) // 4} tokens to send")
        if args.lexicon:
            print(f"   lexicon corrected {corrections} segment(s)")
        print(f"   would write {out_path} (dry run — nothing sent, nothing written)")
        return 0

    if not args.region:
        print("!  no AWS region. Pass --region or set AWS_REGION.", file=sys.stderr)
        return 1

    from meetnotes.extract import extract_notes
    from meetnotes.verify import verify

    config = ModelConfig(region=args.region, model=args.model, profile=args.profile)
    print(f"→  {timeline_path.name}  ({len(segments)} segments, {args.model}, "
          f"effort {args.effort})")
    if corrections:
        print(f"   lexicon corrected {corrections} segment(s)")

    raw = extract_notes(segments, config, effort=args.effort)
    notes = verify(raw, TimelineIndex(segments))

    md_path, json_path = write_notes(out_path, notes, title, source, duration)

    print(f"✓  {md_path}")
    print(f"   {json_path.name}  ({notes.claim_count} verified claims)")
    if notes.dropped:
        # Never silent: a dropped claim is the lint working, and also a signal about the
        # prompt or the model that is worth seeing.
        print(f"   {len(notes.dropped)} claim(s) dropped — quote not found in the "
              f"transcript")
    return 0


if __name__ == "__main__":
    sys.exit(main())
