"""./notes -- turn a timeline into notes somebody will act on, or minutes to circulate.

The human front door. Consumes what ./transcribe produced; never decodes audio, never
loads a Whisper model. Formats operations.py's results as prose; the MCP door formats the
same results as JSON. Neither wraps the other.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from meetnotes.client import BEDROCK, PROVIDERS, ModelConfig, resolve_provider
from meetnotes.operations import (
    OperationError,
    generate_minutes,
    generate_notes,
    preview_minutes,
    preview_notes,
)
from meetnotes.timeline import UnsupportedTimeline, resolve_timeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn a meet-ai timeline into meeting notes with verifiable quotes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", help="A .timeline.json, or the .md transcript beside one")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Where notes land. Default: beside the timeline")
    parser.add_argument("-l", "--lexicon", default=None,
                        help="JSON of {canonical: [alias, ...]} to correct names and jargon")
    parser.add_argument("--provider", default=None, choices=list(PROVIDERS),
                        help="Where to reach Claude. Default: anthropic when "
                             "$ANTHROPIC_API_KEY is set, else bedrock")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION"),
                        help="AWS region. Bedrock only. Default: $AWS_REGION")
    parser.add_argument("--profile", default=os.environ.get("AWS_PROFILE"),
                        help="AWS profile. Bedrock only. Default: $AWS_PROFILE")
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"),
                        help="Anthropic API key. Default: $ANTHROPIC_API_KEY, which the "
                             "SDK also reads on its own")
    parser.add_argument("--model", default=None,
                        help="Model id. Default: the current Claude model for the provider")
    parser.add_argument("--effort", default=None,
                        choices=["low", "medium", "high", "max"],
                        help="How hard the model works on it. Default: high for notes, "
                             "medium for minutes. xhigh is omitted "
                             "deliberately: it arrived with Opus 4.7 and 400s on 4.6")
    parser.add_argument("--minutes", action="store_true",
                        help="Write formal minutes instead of notes: opening, each agenda "
                             "item, closing, with attendance left for the secretary")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show what would be sent and spend nothing. No model call, "
                             "no files written")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    timeline_path = resolve_timeline(Path(args.target))
    out_dir = Path(args.output_dir) if args.output_dir else None
    lexicon = Path(args.lexicon) if args.lexicon else None

    try:
        if args.dry_run:
            preview = (preview_minutes if args.minutes else preview_notes)(
                timeline_path, out_dir=out_dir, lexicon_path=lexicon)
            print(f"·  {Path(preview.timeline).name}")
            print(f"   {preview.segments} segments, {preview.words} words, "
                  f"~{preview.approx_tokens} tokens to send")
            if lexicon:
                print(f"   lexicon corrected {preview.lexicon_corrections} segment(s)")
            print(f"   would write {preview.would_write} "
                  f"(dry run — nothing sent, nothing written)")
            return 0

        provider = resolve_provider(args.provider)
        if provider == BEDROCK and not args.region:
            print("!  no AWS region. Pass --region, set AWS_REGION, or switch provider "
                  "with --provider anthropic.", file=sys.stderr)
            return 1

        config = ModelConfig(
            provider=provider,
            model=args.model,
            region=args.region,
            profile=args.profile,
            api_key=args.api_key,
        )
        print(f"→  {timeline_path.name}  ({provider}, {config.resolved_model}, "
              f"effort {args.effort or 'default'})")

        if args.minutes:
            minutes = generate_minutes(
                timeline_path, config, out_dir=out_dir, lexicon_path=lexicon, effort=args.effort
            )
            print(f"✓  {minutes.markdown}")
            print(f"   {minutes.title}")
            print(f"   {len(minutes.items)} agenda items, {minutes.verified_claims} verified points "
                  f"({minutes.output_tokens} output tokens, {minutes.seconds:.0f}s)")
            if minutes.dropped_claims:
                print(f"   {minutes.dropped_claims} point(s) dropped — quote not found in "
                      f"the transcript")
            return 0

        result = generate_notes(
            timeline_path, config, out_dir=out_dir, lexicon_path=lexicon,
            effort=args.effort or "high",
        )
        print(f"✓  {result.markdown}")
        print(f"   {Path(result.json).name}  ({result.verified_claims} verified claims: "
              f"{result.decisions} decisions, {result.actions} actions, "
              f"{result.questions} questions)")
        if result.dropped_claims:
            # Never silent: a dropped claim is the lint working, and a signal about the
            # prompt or the model that is worth seeing.
            print(f"   {result.dropped_claims} claim(s) dropped — quote not found in "
                  f"the transcript")
        return 0

    except UnsupportedTimeline as error:
        print(f"!  {error}", file=sys.stderr)
        return 1
    except OperationError as error:
        print(f"!  {error}", file=sys.stderr)
        return 1
    except TypeError as error:
        # The SDK raises a bare TypeError when it can find no credentials at all. Left
        # alone that surfaces as a traceback, which says nothing about what to do next.
        if "authentication" not in str(error).lower():
            raise
        print("!  no Anthropic credentials. Pass --api-key, set ANTHROPIC_API_KEY, or "
              "use --provider bedrock.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
