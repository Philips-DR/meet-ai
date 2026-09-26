"""./record -- the human front door for capture.

With no command it records in the foreground until Ctrl-C, which is the terminal case.
`start` and `stop` are the same session split in two, which is the case a model needs and
the one that survives a closed terminal.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from meetnotes.operations import (
    OperationError,
    list_audio_sources,
    recording_status,
    start_recording,
    stop_recording,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO = Path(os.environ.get("MEET_AI_AUDIO", ROOT / "audio"))


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _report_stopped(session: dict) -> None:
    print(f"✓  {session['audio']}  ({_clock(session['duration_seconds'])}, "
          f"peak {session['max_volume_db']} dB)")
    for warning in session.get("warnings", []):
        print(f"   !  {warning}")
    print(f"   transcribe it:  ./transcribe {session['audio']}")
    print("   with speakers:  ./transcribe -d -s <people in the room> ...")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a meeting. Consent first.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "start", "stop", "status", "sources"],
                        help="run records in the foreground until Ctrl-C")
    parser.add_argument("-a", "--audio-dir", default=str(DEFAULT_AUDIO))
    parser.add_argument("--source", default=None,
                        help="Capture source. Default: the system's default microphone")
    parser.add_argument("--consent", default=None,
                        help='Who agreed, and how -- e.g. "verbal, all present". '
                             "Stored with the recording")
    parser.add_argument("--name", default=None, help="Name the recording instead of timestamping it")
    parser.add_argument("--session", default=None, help="Stop a specific session")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    audio_dir = Path(args.audio_dir)

    try:
        if args.command == "sources":
            for source in list_audio_sources():
                mark = "  (default)" if source["default"] else ""
                print(f"   {source['kind']:10}  {source['name']}{mark}")
            return 0

        if args.command == "status":
            current = recording_status(audio_dir)
            if current is None:
                print("   nothing is recording")
            elif current["state"] == "orphaned":
                print(f"   !  {current['id']} was interrupted with "
                      f"{_clock(current['recorded_seconds'])} on disk. Recover it: ./record stop")
            else:
                print(f"   recording {current['id']}  {_clock(current['recorded_seconds'])}  "
                      f"from {current['source']}")
            return 0

        if args.command == "stop":
            _report_stopped(stop_recording(audio_dir, args.session))
            return 0

        session = start_recording(audio_dir, args.source, args.consent, args.name)
        print(f"●  recording {session['id']} from {session['source']}")
        if not args.consent:
            print("   !  no consent recorded. Everyone being recorded should know.")

        if args.command == "start":
            print("   stop it with:  ./record stop")
            return 0

        print("   Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1)
                current = recording_status(audio_dir)
                if current is None or current["state"] != "recording":
                    print("\n   !  the recorder stopped on its own")
                    break
                print(f"\r   {_clock(current['recorded_seconds'])}", end="", flush=True)
        except KeyboardInterrupt:
            pass
        print()
        _report_stopped(stop_recording(audio_dir, session["id"]))
        return 0

    except OperationError as error:
        print(f"!  {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
