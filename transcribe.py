#!/usr/bin/env python3
"""Transcribe audio files to Markdown using faster-whisper.

Drop audio in audio/ and run ./transcribe — Markdown lands in transcripts/.

By default each file gets a single clean pass, which is the most accurate thing
to do: Whisper already works in 30-second windows internally, so slicing the
audio up beforehand gains nothing and only risks cutting mid-word.

For very long recordings, --chunk-minutes N splits the audio at natural silences
and caches each finished chunk, so an interrupted 3-hour job resumes instead of
starting over. That is a resumability feature, not an accuracy one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "audio"
OUT_DIR = ROOT / "transcripts"
CACHE_DIR = ROOT / ".cache"

AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4b", ".aiff", ".amr",
}

# Start a new paragraph when the speaker pauses this long (seconds), or when
# the current one gets unwieldy.
PARAGRAPH_GAP = 2.0
PARAGRAPH_MAX_CHARS = 700

# How far from an exact chunk boundary we'll wander to find a silence to cut on.
BOUNDARY_SEARCH = 45.0

# In "auto" mode, files longer than this get chunked so a crash can't cost the
# whole run. Shorter files take the single clean pass.
AUTO_CHUNK_ABOVE_MINUTES = 45.0
AUTO_CHUNK_SIZE_MINUTES = 10.0


def hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def find_audio(paths: list[str]) -> list[Path]:
    """Expand the given files/dirs into a sorted list of audio files."""
    targets = [Path(p) for p in paths] if paths else [AUDIO_DIR]
    found: list[Path] = []
    for target in targets:
        if target.is_dir():
            found += [
                p for p in sorted(target.rglob("*"))
                if p.is_file() and p.suffix.lower() in AUDIO_EXTS
            ]
        elif target.is_file():
            found.append(target)
        else:
            print(f"!  not found: {target}", file=sys.stderr)
    seen: set[Path] = set()
    return [p for p in found if not (p in seen or seen.add(p))]


def probe_duration(path: Path) -> float:
    proc = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError(f"could not read duration ({proc.stderr.strip()[:200]})")


# --------------------------------------------------------------------------
# Chunk planning — cut on silence, not mid-word
# --------------------------------------------------------------------------

SILENCE_RE = re.compile(r"silence_(start|end):\s*(-?[\d.]+)")


def find_silences(path: Path, noise_db: int, min_silence: float) -> list[tuple[float, float]]:
    """Scan the file with ffmpeg's silencedetect. Returns (start, end) spans."""
    proc = run([
        "ffmpeg", "-nostdin", "-i", str(path),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-",
    ])
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for kind, value in SILENCE_RE.findall(proc.stderr):
        if kind == "start":
            start = float(value)
        elif start is not None:
            spans.append((start, float(value)))
            start = None
    return spans


def plan_chunks(duration: float, chunk_len: float,
                silences: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pick cut points near each chunk boundary, preferring the middle of a silence."""
    if duration <= chunk_len:
        return [(0.0, duration)]

    cuts: list[float] = []
    position = 0.0
    while duration - position > chunk_len:
        target = position + chunk_len
        # Candidate silences whose midpoint is within reach of the target and
        # safely ahead of where this chunk started.
        best = None
        for sil_start, sil_end in silences:
            mid = (sil_start + sil_end) / 2
            if abs(mid - target) > BOUNDARY_SEARCH or mid <= position + 30:
                continue
            if best is None or abs(mid - target) < abs(best - target):
                best = mid
        cut = best if best is not None else target  # hard cut if nothing quiet nearby
        cuts.append(cut)
        position = cut

    bounds = [0.0] + cuts + [duration]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def extract_chunk(path: Path, start: float, end: float, dest: Path) -> None:
    """Decode one span to 16 kHz mono WAV — Whisper's native input format."""
    proc = run([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest),
    ])
    if proc.returncode != 0 or not dest.exists():
        raise RuntimeError(f"ffmpeg failed to extract chunk: {proc.stderr.strip()[:200]}")


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def cache_key(path: Path, args, chunk_minutes: float) -> str:
    """Identity of this job — changes if the audio or the decode settings change."""
    stat = path.stat()
    raw = "|".join(str(x) for x in [
        path.resolve(), stat.st_size, int(stat.st_mtime), args.model,
        args.language, args.beam_size, chunk_minutes, args.no_vad,
        args.compute_type, args.prompt, args.condition,
        # Diarized runs store word timings; a cache built without them can't be
        # reused for one, or speakers would silently fall back to segment level.
        args.diarize,
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def transcribe_span(model, wav: Path, args, offset: float,
                    total: float, started: float) -> tuple[list[dict], str]:
    """Transcribe one span; timestamps are shifted into whole-file time."""
    segments, info = model.transcribe(
        str(wav),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        vad_parameters={"min_silence_duration_ms": 500},
        initial_prompt=args.prompt,
        # Feeding previous text back in improves punctuation and keeps names
        # spelled consistently, but can send the model into a repetition loop
        # on long audio — hence opt-in.
        condition_on_previous_text=args.condition,
        # Only needed for diarization, and it costs extra time, so don't ask
        # for word timings unless we're going to use them.
        word_timestamps=args.diarize,
    )

    out: list[dict] = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            entry = {
                "start": seg.start + offset,
                "end": seg.end + offset,
                "text": text,
            }
            if args.diarize and seg.words:
                entry["words"] = [
                    {"start": w.start + offset, "end": w.end + offset, "word": w.word}
                    for w in seg.words
                ]
            out.append(entry)
        show_progress(seg.end + offset, total, started)
    return out, info.language


def show_progress(position: float, total: float, started: float) -> None:
    """Overwrite a single stderr line with decode progress."""
    elapsed = time.monotonic() - started
    speed = position / elapsed if elapsed > 0 else 0
    pct = min(100, position / total * 100) if total else 0
    eta = (total - position) / speed if speed > 0 else 0
    print(
        f"\r   {pct:5.1f}%  {hhmmss(position)}/{hhmmss(total)}  "
        f"{speed:.1f}x  eta {hhmmss(eta)}    ",
        end="", file=sys.stderr, flush=True,
    )


# --------------------------------------------------------------------------
# Markdown assembly
# --------------------------------------------------------------------------

# Markdown block syntax a spoken sentence can trip over by accident. A transcript line is
# *text*: "160. That's one more." is a sentence, not the 160th item of a list, and "- so anyway"
# is not a bullet. Without escaping, CommonMark reads them as structure and a downstream
# renderer faithfully builds the numbered list the speaker never meant. Found by running
# docu-ai over a real 2-hour transcript (2026-09-20): three spoken numbers became list items.
_BLOCK_START_RE = re.compile(r"""
    ^(
        \#{1,6}(?=\s|$)       # ATX heading
      | >                      # blockquote
      | [-+*](?=\s|$)          # bullet list
      | \d{1,9}[.)](?=\s|$)    # ordered list
    )
""", re.VERBOSE)


def escape_block_start(text: str) -> str:
    """Escape leading block syntax so a spoken line stays a spoken line.

    Escaping the marker's *last* character is what breaks the construct, and it is the only
    form that works for every case: "15\\." is text, while "\\15." would render the backslash
    literally, since a backslash only escapes punctuation.
    """
    match = _BLOCK_START_RE.match(text)
    if not match:
        return text
    marker = match.group(1)
    return f"{marker[:-1]}\\{marker[-1]}{text[len(marker):]}"


def to_paragraphs(segments: list[dict]) -> list[list[dict]]:
    """Group segments into readable paragraphs on pauses, length and speaker."""
    paragraphs: list[list[dict]] = []
    current: list[dict] = []
    for seg in segments:
        if current:
            gap = seg["start"] - current[-1]["end"]
            too_long = sum(len(s["text"]) for s in current) > PARAGRAPH_MAX_CHARS
            # A change of speaker always starts a new paragraph, however tight
            # the timing — running two people together is worse than a short line.
            speaker_changed = seg.get("speaker") != current[-1].get("speaker")
            if gap >= PARAGRAPH_GAP or too_long or speaker_changed:
                paragraphs.append(current)
                current = []
        current.append(seg)
    if current:
        paragraphs.append(current)
    return paragraphs


def render_markdown(source: Path, segments: list[dict], duration: float,
                    language: str, model_name: str, timestamps: bool,
                    speakers: int = 0) -> str:
    """Build the Markdown document, with YAML frontmatter for Obsidian etc."""
    title = source.stem.replace("_", " ").replace("-", " ").strip()
    lines = [
        "---",
        f'source: "{source.name}"',
        f"duration: {hhmmss(duration)}",
        f"language: {language}",
        f"model: {model_name}",
    ]
    if speakers:
        lines.append(f"speakers: {speakers}")
    lines += [
        f"transcribed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "---",
        "",
        f"# {title}",
        "",
    ]

    previous_speaker = object()  # sentinel: never equal to a real speaker
    for para in to_paragraphs(segments):
        text = " ".join(s["text"] for s in para).strip()
        if not text:
            continue

        speaker = para[0].get("speaker")
        # Only where the text begins a line: mid-line it is already inert, and an
        # escape there would show up as a stray backslash.
        body = escape_block_start(text)
        if speakers and speaker != previous_speaker:
            from diarize import speaker_name
            stamp = f" *[{hhmmss(para[0]['start'])}]*" if timestamps else ""
            lines.append(f"**{speaker_name(speaker)}**{stamp}")
            lines.append("")
            previous_speaker = speaker
            lines.append(body)
        elif timestamps and not speakers:
            lines.append(f"**[{hhmmss(para[0]['start'])}]** {text}")
        else:
            lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Timeline — the spans the Markdown was rendered from
# --------------------------------------------------------------------------

# Bump when the shape of a timeline file changes, so readers can refuse one
# they don't understand rather than misinterpreting it.
TIMELINE_VERSION = 1


def timeline_path(out_path: Path) -> Path:
    """Where the timeline lands for a given Markdown output. One definition."""
    return out_path.with_suffix(".timeline.json")


def write_timeline(out_path: Path, source: Path, segments: list[dict],
                   duration: float, language: str, model_name: str,
                   speakers: int) -> Path:
    """Persist the segments the Markdown was rendered from, as JSON beside it.

    Written on every run, never optional. The Markdown is a *view* of this file,
    not the other way round: each rendered sentence has a span here, so a quote
    lifted from a summary can be checked back against the audio at a real offset.
    Without it a quote is a claim; with it a quote is verifiable.

    Note these are the *post-processing* segments — after diarization has split
    and relabelled them — precisely because those are what the prose corresponds
    to. Persisting the pre-diarization ones would not line up with the text.
    """
    payload = {
        "version": TIMELINE_VERSION,
        "source": source.name,
        "duration": duration,
        "language": language,
        "model": model_name,
        "speakers": speakers,
        "transcribed": dt.datetime.now().isoformat(timespec="seconds"),
        "segments": segments,
    }
    path = timeline_path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Per-file driver
# --------------------------------------------------------------------------

def transcribe_file(model, path: Path, args) -> Path | None:
    out_path = Path(args.output_dir) / f"{path.stem}.md"
    if out_path.exists() and not args.force:
        # A transcript written before timelines existed has no spans beside it,
        # and there is no way to rebuild them without decoding the audio again.
        missing = "" if timeline_path(out_path).exists() else "  (no timeline — rerun with -f)"
        print(f"·  skip (already done): {out_path.name}{missing}")
        return None

    duration = probe_duration(path)

    # "auto": chunk only when the file is long enough that losing the run would
    # actually hurt. Cutting at silences costs ~nothing unless --condition is on.
    if args.chunk_minutes is None:
        chunk_minutes = (AUTO_CHUNK_SIZE_MINUTES
                         if duration > AUTO_CHUNK_ABOVE_MINUTES * 60
                         else float("inf"))
    else:
        chunk_minutes = args.chunk_minutes
    chunk_len = chunk_minutes * 60

    print(f"→  {path.name}  ({hhmmss(duration)})")

    # Default path: one clean pass over the original file. No re-encoding, no
    # seams — chunking buys resumability, not accuracy, so it stays opt-in.
    if duration <= chunk_len:
        started = time.monotonic()
        all_segments, language = transcribe_span(
            model, path, args, 0.0, duration, started
        )
        return finish(path, out_path, all_segments, duration, language, args, started)

    work = CACHE_DIR / f"{path.stem}-{cache_key(path, args, chunk_minutes)}"
    work.mkdir(parents=True, exist_ok=True)

    # Plan the cuts once and reuse them across resumed runs.
    plan_file = work / "chunks.json"
    if plan_file.exists():
        chunks = [tuple(c) for c in json.loads(plan_file.read_text())]
    else:
        print("   scanning for silences to cut on...", file=sys.stderr)
        silences = find_silences(path, args.silence_db, args.silence_duration)
        chunks = plan_chunks(duration, chunk_len, silences)
        plan_file.write_text(json.dumps(chunks))

    print(f"   {len(chunks)} chunks of ~{chunk_minutes:g} min (resumable)")

    all_segments: list[dict] = []
    language = args.language or "unknown"
    started = time.monotonic()

    for i, (start, end) in enumerate(chunks):
        part_file = work / f"chunk_{i:04d}.json"
        if part_file.exists():
            cached = json.loads(part_file.read_text())
            all_segments += cached["segments"]
            language = cached.get("language", language)
            print(
                f"\r   chunk {i + 1}/{len(chunks)}  {hhmmss(start)}–{hhmmss(end)}  "
                f"(cached)            ", file=sys.stderr,
            )
            continue

        wav = work / f"chunk_{i:04d}.wav"
        extract_chunk(path, start, end, wav)
        try:
            segs, lang = transcribe_span(model, wav, args, start, duration, started)
        finally:
            wav.unlink(missing_ok=True)

        # Written only after the chunk fully succeeds, so the cache is never partial.
        part_file.write_text(json.dumps({"language": lang, "segments": segs}))
        all_segments += segs
        language = lang

    if not args.keep_cache:
        shutil.rmtree(work, ignore_errors=True)

    return finish(path, out_path, all_segments, duration, language, args, started)


def run_diarization(path: Path, args) -> list[dict]:
    """Diarize the whole file, caching the result — it costs as much as decoding.

    Always run on the complete audio, never per chunk: speaker clusters are only
    comparable within a single clustering pass, so chunk-by-chunk diarization
    would make "Speaker 1" mean a different person in every chunk.
    """
    import diarize as diar

    key = hashlib.sha256(
        f"{path.resolve()}|{path.stat().st_size}|{args.speakers}|"
        f"{args.diarize_threshold}".encode()
    ).hexdigest()[:16]
    cache_file = CACHE_DIR / f"diarization-{path.stem}-{key}.json"

    if cache_file.exists():
        print("   speakers: using cached diarization")
        return json.loads(cache_file.read_text())

    turns = diar.diarize(path, num_speakers=args.speakers,
                         threshold=args.diarize_threshold)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(turns))
    return turns


def finish(path: Path, out_path: Path, segments: list[dict], duration: float,
           language: str, args, started: float) -> Path:
    """Render, write and report — shared by the single-pass and chunked paths."""
    print(file=sys.stderr)  # end the progress line

    speakers = 0
    if args.diarize:
        import diarize as diar
        turns = run_diarization(path, args)
        segments = diar.assign_speakers(segments, turns)
        speakers = len({t["speaker"] for t in turns})

    markdown = render_markdown(
        path, segments, duration, language, args.model, args.timestamps, speakers
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    write_timeline(out_path, path, segments, duration, language, args.model, speakers)

    took = time.monotonic() - started
    words = sum(len(s["text"].split()) for s in segments)
    shown = out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path
    extra = f", {speakers} speakers" if speakers else ""
    print(f"✓  {shown}  ({language}, ~{words} words{extra}, took {hhmmss(took)})")
    print(f"   {timeline_path(shown)}  ({len(segments)} spans)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to Markdown with Whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="*",
                        help="Audio files or directories. Default: everything in audio/")
    parser.add_argument("-m", "--model",
                        default=os.environ.get("WHISPER_MODEL", "distil-large-v3"),
                        help="distil-large-v3 (default, English only) | tiny | base | "
                             "small | medium | large-v3. Use a multilingual model "
                             "(small/medium/large-v3) for non-English audio")
    parser.add_argument("-l", "--language", default=os.environ.get("WHISPER_LANG") or None,
                        help="Language code like en, fr, ar. Default: auto-detect")
    parser.add_argument("-o", "--output-dir", default=str(OUT_DIR), help="Where .md files go")
    parser.add_argument("-t", "--timestamps", action="store_true",
                        help="Prefix each paragraph with its start time")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Re-transcribe even if the .md already exists")
    parser.add_argument("--compute-type",
                        default=os.environ.get("WHISPER_COMPUTE", "int8_float32"),
                        help="int8_float32 (default; same speed as int8, more accurate) "
                             "| int8 | float32")
    parser.add_argument("-p", "--prompt", default=None,
                        help="Vocabulary hint: names, jargon and spellings to expect")
    parser.add_argument("--condition", action="store_true",
                        help="Feed previous text back as context — better flow, "
                             "small risk of repetition loops")
    parser.add_argument("-c", "--chunk-minutes", default="auto",
                        help=f"auto = single pass under {AUTO_CHUNK_ABOVE_MINUTES:g} min, "
                             f"{AUTO_CHUNK_SIZE_MINUTES:g}-min resumable chunks above it. "
                             "0 = always single pass. N = always N-minute chunks")
    parser.add_argument("-d", "--diarize", action="store_true",
                        help="Label who is speaking. Roughly doubles the runtime")
    parser.add_argument("-s", "--speakers", type=int, default=0,
                        help="Number of people in the recording. Strongly "
                             "recommended with --diarize; 0 auto-detects, which "
                             "tends to over-split")
    parser.add_argument("--diarize-threshold", type=float, default=0.8,
                        help="Auto-detect sensitivity. Lower finds more speakers")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Higher is slightly more accurate and slower")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads. 0 = auto")
    parser.add_argument("--no-vad", action="store_true",
                        help="Disable silence trimming (VAD)")
    parser.add_argument("--silence-db", type=int, default=-30,
                        help="Loudness below this counts as silence when picking cuts")
    parser.add_argument("--silence-duration", type=float, default=0.4,
                        help="Minimum silence length to cut on, seconds")
    parser.add_argument("--keep-cache", action="store_true",
                        help="Keep per-chunk JSON after finishing")
    args = parser.parse_args()

    if str(args.chunk_minutes).lower() == "auto":
        args.chunk_minutes = None  # decided per file, once we know its duration
    else:
        try:
            args.chunk_minutes = float(args.chunk_minutes)
        except ValueError:
            print(f"--chunk-minutes: expected a number or 'auto', "
                  f"got {args.chunk_minutes!r}", file=sys.stderr)
            return 1
        if args.chunk_minutes <= 0:
            args.chunk_minutes = float("inf")  # single pass over the whole file

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        print("ffmpeg/ffprobe not found — install ffmpeg first.", file=sys.stderr)
        return 1

    # The distil and .en models only do English. Fed anything else they don't
    # fail — they emit confident nonsense — so say so loudly up front.
    english_only = args.model.endswith(".en") or "distil" in args.model
    if english_only:
        if args.language and args.language != "en":
            print(f"!  '{args.model}' is English-only but --language is "
                  f"'{args.language}'.\n"
                  f"   Use a multilingual model instead: -m small (or medium, "
                  f"large-v3).", file=sys.stderr)
            return 1
        args.language = "en"  # skip detection; it can only be English anyway

    files = find_audio(args.paths)
    if not files:
        where = ", ".join(args.paths) if args.paths else str(AUDIO_DIR)
        print(f"No audio files found in: {where}")
        print(f"Drop some audio into {AUDIO_DIR}/ and run this again.")
        return 1

    from faster_whisper import WhisperModel

    print(f"Loading model '{args.model}' (CPU, {args.compute_type})...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type,
                         cpu_threads=args.threads)

    print(f"{len(files)} file(s) to transcribe.\n")
    done = 0
    for path in files:
        try:
            if transcribe_file(model, path, args):
                done += 1
        except KeyboardInterrupt:
            print("\nInterrupted — finished chunks are cached, rerun to resume.",
                  file=sys.stderr)
            return 130
        except Exception as exc:  # keep going through a batch
            print(f"\n✗  {path.name}: {exc}", file=sys.stderr)

    print(f"\nDone. {done} transcript(s) written to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
