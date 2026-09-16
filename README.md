# Whisper → Markdown

Drop audio in `audio/`, run one command, get a Markdown transcript in `transcripts/`.

Runs fully offline on CPU via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
No PyTorch, no GPU, no API calls — the audio never leaves the machine.

## Usage

```bash
./transcribe                      # everything in audio/ that isn't done yet
./transcribe meeting.mp3          # one specific file
./transcribe -t                   # with [hh:mm:ss] timestamps per paragraph
./transcribe -d -s 4              # label speakers, 4 people in the room
./transcribe -m small -l fr       # non-English needs a multilingual model
./transcribe -f                   # redo files that already have transcripts
```

Already-transcribed files are skipped, so re-running after adding new audio only
processes the new ones.

## Output

`audio/board_meeting.mp3` → `transcripts/board_meeting.md`:

```markdown
---
source: "board_meeting.mp3"
duration: 01:12:40
language: en
model: distil-large-v3
transcribed: 2026-09-15 21:30
---

# board meeting

First paragraph of speech, grouped so it reads like prose rather than
one line per subtitle cue...

Next paragraph, split where the speaker paused.
```

The YAML frontmatter means it drops straight into Obsidian or any notes tool.

## Accuracy

These are the knobs that actually matter, in order of impact.

**1. Model size.** The single biggest lever.

The default is **`distil-large-v3`** — accuracy approaching `large-v3` while
running *faster* than `small`. It is **English only**.

Measured on this machine (4-core i7-8665U, `int8_float32`, real meeting audio):

| Model | Disk | Speed | 1 hour of audio takes | Use when |
|-----------|-------|-------|----------------------|----------|
| `tiny` | 75 MB | ~8× | ~8 min | rough notes, keyword search |
| `base` | 145 MB | ~4× | ~15 min | quick drafts |
| `small` | 464 MB | 0.87× *(measured)* | ~1 h 10 min | multilingual fallback |
| `medium` | 1.5 GB | ~0.3× | ~3 h 30 min | non-English you care about |
| `large-v3` | 3.1 GB | ~0.15× | ~7 h | impractical on this CPU |
| `distil-large-v3` | 1.5 GB | **1.9× *(measured)*** | **~32 min** | **default** — English |

Counter-intuitively `distil-large-v3` is more than twice as fast as `small`
despite being three times the size. Whisper's bottleneck is the *decoder*, which
runs sequentially one token at a time; distil keeps the full 32-layer encoder but
cuts the decoder to 2 layers. So it reads the audio with `large-v3`'s quality and
writes the text far faster.

**Non-English audio**: switch to a multilingual model with `-m small` (or
`medium`). The English-only models don't degrade gracefully on other languages —
they emit fluent nonsense — so the script refuses to run if you pair one with a
non-English `--language` rather than letting it produce a confident wrong
transcript.

There is also a newer `distil-large-v3.5` if you want to try it (`-m
distil-large-v3.5`); it downloads on first use and is reported to improve on v3.

**2. Precision.** Default is `int8_float32`, chosen after measuring on this
machine (`small`, 2 min of audio, 4-core i7-8665U):

| compute type | measured speed | note |
|--------------|----------------|------|
| `int8` | 0.87× | fully quantised, and no faster than the default |
| `int8_float32` | 0.88× | **default** — int8 weights, float32 activations |
| `float32` | 0.54× | 1.6× slower |

`int8_float32` is the default because it costs nothing: identical throughput to
plain `int8`, while keeping activations in float32, which generally preserves
accuracy better than full int8 quantisation. Speed here was measured on this
machine; the accuracy claim is the standard CTranslate2 guidance, not something
benchmarked locally.

If a recording matters and you have time to spare, `--compute-type float32` is
the conservative choice.

**3. Vocabulary hints.** Whisper guesses at proper nouns and jargon. Telling it
what to expect fixes them:

```bash
./transcribe -p "Aya Data, Accra, Kwame, LiDAR, annotation pipeline"
```

**4. Pin the language.** Auto-detection occasionally guesses wrong on the first
few seconds and then transcribes the whole file in the wrong language. If you
know it, say so: `-l en`.

**5. Context carry-over.** `--condition` feeds previous text back to the model,
improving punctuation and keeping names spelled consistently — at some risk of
the model getting stuck in a repetition loop on long audio. Off by default.

## Long recordings

This is handled automatically. Files **under 45 minutes** get one clean single
pass. Files **over 45 minutes** are split into 10-minute resumable chunks.

Why chunk at all: transcription runs at roughly realtime on this CPU, so a
3-hour recording is ~3 hours of compute. If the laptop sleeps or the process
dies at hour three, a single pass loses everything. Chunking caches each
finished chunk under `.cache/` — interrupt it, re-run the same command, and it
picks up from the last completed chunk.

Why it's safe: cuts land on **natural silences** near each 10-minute mark, never
at a fixed offset that would slice mid-word. And because `condition_on_previous_text`
is off by default, Whisper already doesn't carry text context across its internal
30-second windows — so chunking costs essentially nothing in accuracy.

The one exception: if you turn on `--condition`, context *does* flow between
windows, and chunk boundaries then break it. Use a single pass in that case.

```bash
./transcribe -c 0      # force single pass, whatever the length
./transcribe -c 5      # force 5-minute chunks
./transcribe           # auto (recommended)
```

## Speaker labels

```bash
./transcribe -d -s 4 meeting.m4a     # 4 people in the room
./transcribe -d meeting.m4a          # auto-detect (worse — see below)
```

Output becomes:

```markdown
**Speaker 1**

So for a fraud claim, the fraud master is here.

**Speaker 2**

Is that your new title?
```

Speakers are numbered by **airtime**, so Speaker 1 talks most — usually the
chair. The numbering is stable across runs; the clusterer's own labels are not.

**Always pass `-s` with the real headcount.** Auto-detection over-splits badly —
on a 3-minute sample with 4 people it found 18 clusters, most of them
sub-5-second slivers of cross-talk. Pinning the count fixes it.

Diarization runs on the **whole file**, never per chunk: speaker clusters are
only comparable within one clustering pass, so chunking it would make "Speaker 1"
a different person in each chunk. It's cached separately under `.cache/`, because
it costs about as much time as the transcription itself — budget roughly **double**
the runtime with `-d`.

### How good is it?

Honest answer: **serviceable, not reliable.** Who-said-what is inferred from voice
similarity, and a single room mic with people talking over each other is the hard
case. Expect the gist to be right and individual attributions to be wrong often
enough that you should not quote from it without checking the audio.

Two design choices push the errors somewhere tolerable:

- **Speakers are voted per sentence, not per word.** Word-level labels are jittery
  mid-sentence and shred one sentence across three speakers. On the test sample
  that produced 22 mid-sentence speaker switches in 3 minutes; sentence voting cut
  it to 1. A whole sentence occasionally pinned on the wrong person reads far
  better than a sentence chopped into thirds.
- **Word timings still drive the vote**, so genuine turn-taking is caught — the
  labels just land on sentence boundaries.

It needs `--diarize`-specific word timestamps, so a diarized run cannot reuse a
plain run's cache; it re-transcribes.

### Models

`models/` holds two ONNX files (~37 MB total), downloaded once:

| File | Purpose |
|---|---|
| `sherpa-onnx-pyannote-segmentation-3-0/` | finds speech regions and overlaps |
| `campplus_en.onnx` | voice embeddings for clustering |

This uses `sherpa-onnx` on ONNX Runtime rather than `pyannote.audio`, which would
pull in PyTorch (~2.5 GB) and a gated HuggingFace token for the same result.

## Setup

Already done, but to rebuild from scratch:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./fetch-models.sh          # only if you want --diarize
```

Requires `ffmpeg` on PATH. Whisper models download themselves on first use and
are cached in `~/.cache/huggingface/`; the diarization models are not on PyPI,
hence the fetch script.

## Layout

```
audio/         drop audio here (mp3, wav, m4a, flac, mp4, mkv, ...)
transcripts/   Markdown output
transcribe     wrapper — runs the venv Python for you
transcribe.py  the actual pipeline
diarize.py     speaker labelling (optional, --diarize)
models/        ONNX diarization models (~37 MB)
.cache/        per-chunk resume data, only used with --chunk-minutes
```
