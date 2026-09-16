"""Speaker diarization — works out who spoke when.

Uses sherpa-onnx (ONNX Runtime) rather than pyannote.audio, which would drag in
PyTorch and a gated HuggingFace token for the same job. Models live in models/.

Whisper says *what* was said; this says *who* said it. The two are matched up by
overlapping their timelines.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
SEGMENTATION = MODELS / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING = MODELS / "campplus_en.onnx"


def models_present() -> bool:
    return SEGMENTATION.exists() and EMBEDDING.exists()


def load_audio(path: Path, sample_rate: int):
    """Decode any audio file to mono float32 at the rate the models expect."""
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ar", str(sample_rate), "-ac", "1", "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio: "
                           f"{proc.stderr.decode()[:200]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def diarize(path: Path, num_speakers: int = 0, threshold: float = 0.5,
            progress: bool = True) -> list[dict]:
    """Return [{start, end, speaker}] covering the whole file.

    num_speakers: exact count if you know it, 0 to detect automatically.
    threshold:    used only when detecting — lower splits speakers more eagerly.
    """
    import sherpa_onnx

    if not models_present():
        raise RuntimeError(
            f"Diarization models missing. Expected:\n  {SEGMENTATION}\n  {EMBEDDING}"
        )

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(SEGMENTATION),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING)),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=int(num_speakers) if num_speakers else -1,
            threshold=float(threshold),
        ),
        min_duration_on=0.3,   # ignore speech blips shorter than this
        min_duration_off=0.5,  # gaps shorter than this don't split a turn
    )
    if not config.validate():
        raise RuntimeError("sherpa-onnx rejected the diarization config")

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples = load_audio(path, sd.sample_rate)

    def on_progress(done: int, total: int) -> int:
        if progress:
            print(f"\r   diarizing {done * 100 // max(total, 1):3d}%   ",
                  end="", file=sys.stderr, flush=True)
        return 0

    result = sd.process(samples, callback=on_progress if progress else None)
    if progress:
        print(file=sys.stderr)

    turns = [
        {"start": s.start, "end": s.end, "speaker": s.speaker}
        for s in result.sort_by_start_time()
    ]
    return renumber_by_speech_time(turns)


def renumber_by_speech_time(turns: list[dict]) -> list[dict]:
    """Relabel speakers so 0 is the one who talks most.

    The clusterer numbers speakers arbitrarily. Ordering by airtime makes the
    labels stable and means "Speaker 1" is usually the chair or interviewer.
    """
    import collections

    totals = collections.Counter()
    for turn in turns:
        totals[turn["speaker"]] += turn["end"] - turn["start"]
    ranking = {old: new for new, (old, _) in enumerate(totals.most_common())}
    return [{**t, "speaker": ranking[t["speaker"]]} for t in turns]


# Longest plausible speaker turn; bounds how far back we scan for overlaps.
MAX_TURN_SECONDS = 120.0


class SpeakerLookup:
    """Finds who was speaking during a given time span.

    Turns are sorted by start time, so a bisect plus a bounded backward scan
    beats rescanning the whole list for every word.
    """

    def __init__(self, turns: list[dict]):
        self.turns = turns
        self.starts = [t["start"] for t in turns]

    def at(self, start: float, end: float):
        import bisect

        hi = bisect.bisect_left(self.starts, end)
        best_speaker, best_overlap = None, 0.0
        nearest_speaker, nearest_gap = None, float("inf")

        i = hi - 1
        while i >= 0 and self.starts[i] > start - MAX_TURN_SECONDS:
            turn = self.turns[i]
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            if overlap > best_overlap:
                best_speaker, best_overlap = turn["speaker"], overlap
            if overlap <= 0:
                gap = max(turn["start"] - end, start - turn["end"])
                if gap < nearest_gap:
                    nearest_speaker, nearest_gap = turn["speaker"], gap
            i -= 1

        # A word landing in a gap between turns still belongs to someone.
        return best_speaker if best_speaker is not None else nearest_speaker


def assign_speakers(segments: list[dict], turns: list[dict]) -> list[dict]:
    """Attach a speaker to each piece of transcript.

    Whisper splits on pauses, the diarizer splits on who is talking, and the two
    rarely agree — a single Whisper segment can span three people in a lively
    meeting. So when word timestamps are available we label each *word* and cut
    the segment wherever the speaker changes. Without them we can only label the
    segment as a whole, which loses every mid-segment handover.
    """
    if not turns:
        return segments

    lookup = SpeakerLookup(turns)
    out: list[dict] = []

    for seg in segments:
        words = seg.get("words")
        if not words:
            out.append({**seg, "speaker": lookup.at(seg["start"], seg["end"])})
            continue

        for sentence in split_sentences(words):
            text = "".join(w["word"] for w in sentence).strip()
            if text:
                out.append({
                    "start": sentence[0]["start"],
                    "end": sentence[-1]["end"],
                    "text": text,
                    "speaker": dominant_speaker(sentence, lookup),
                })
    return out


def split_sentences(words: list[dict]) -> list[list[dict]]:
    """Break a segment's words at sentence-ending punctuation."""
    sentences: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        current.append(word)
        if word["word"].rstrip().endswith((".", "?", "!")):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def dominant_speaker(words: list[dict], lookup: "SpeakerLookup"):
    """Whoever holds the most speaking time across these words.

    Voting over the whole sentence rather than labelling each word keeps a
    sentence attributed to one person. Pure word-level labels are jittery
    mid-sentence and shred single sentences across three speakers, which reads
    far worse than the occasional whole sentence pinned on the wrong person.
    """
    tally: dict = {}
    for word in words:
        speaker = lookup.at(word["start"], word["end"])
        tally[speaker] = tally.get(speaker, 0.0) + (word["end"] - word["start"])
    return max(tally, key=tally.get) if tally else None


def speaker_name(index: int | None) -> str:
    return "Unknown" if index is None else f"Speaker {index + 1}"


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found on PATH")
