# meet-ai

Drop audio in `audio/`, run one command, get a Markdown transcript in `transcripts/`.
Run a second command and get meeting notes in which every claim cites the moment it was said.

Transcription runs fully offline on CPU via [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
No PyTorch, no GPU, no API calls — the audio never leaves the machine. Only the notes step talks to
a model, and only ever sees text.

Three ways in, all over the same code: `./transcribe` and `./notes` for a person, and `./mcp` for a
model.

> `plan.md` holds the decisions and what is left to build. `CLAUDE.md` holds the engineering rules
> and the measured facts behind the defaults below.

## Transcribing

```bash
./transcribe                      # everything in audio/ that isn't done yet
./transcribe meeting.mp3          # one specific file
./transcribe -t                   # with [hh:mm:ss] timestamps per paragraph
./transcribe -d -s 4              # label speakers, 4 people in the room
./transcribe -m small -l fr       # non-English needs a multilingual model
./transcribe -p "AyaData, Accra, Kwame, LiDAR"   # vocabulary hints
./transcribe -f                   # redo files that already have transcripts
```

Already-transcribed files are skipped, so re-running after adding new audio only processes the new
ones.

`audio/board_meeting.mp3` produces **two** files:

- `transcripts/board_meeting.md` — YAML frontmatter and prose, ready for Obsidian or docu-ai
- `transcripts/board_meeting.timeline.json` — the spans everything else is checked against

The timeline is written on every run, never optionally. The Markdown is a view of it.

### Choosing a model

| Model | Disk | Speed | 1 hour of audio | Use when |
|---|---|---|---|---|
| `tiny` | 75 MB | ~8× | ~8 min | rough notes, keyword search |
| `base` | 145 MB | ~4× | ~15 min | quick drafts |
| `small` | 464 MB | 0.87× | ~1 h 10 min | multilingual fallback |
| `medium` | 1.5 GB | ~0.3× | ~3 h 30 min | non-English you care about |
| `large-v3` | 3.1 GB | ~0.15× | ~7 h | impractical on this CPU |
| `distil-large-v3` | 1.5 GB | **1.9×** | **~32 min** | **default** — English only |

Measured on this machine; CLAUDE.md has the conditions and the reason distil is faster despite being
larger. For non-English use `-m small` or `-m medium` — pairing an English-only model with a
non-English `--language` is refused rather than allowed to produce fluent nonsense.

If a recording matters and you have time, `--compute-type float32` is the conservative choice.

### Long recordings

Automatic. Under 45 minutes gets one clean pass; over that, 10-minute resumable chunks cached under
`.cache/`, so an interrupted 3-hour job picks up where it stopped. Cuts land on natural silences,
never mid-word.

```bash
./transcribe -c 0      # force a single pass, whatever the length
./transcribe -c 5      # force 5-minute chunks
```

The one exception: `--condition` makes context flow between windows, and chunk boundaries break it.
Use a single pass in that case.

### Speaker labels

```bash
./transcribe -d -s 4 meeting.m4a     # 4 people in the room
```

**Always pass `-s` with the real headcount** — auto-detection over-splits badly. Speakers are
numbered by airtime, so Speaker 1 talks most. Budget roughly double the runtime with `-d`.

**Honest answer on quality: serviceable, not reliable.** Who-said-what is inferred from voice
similarity, and a single room mic with people talking over each other is the hard case. Expect the
gist to be right and individual attributions to be wrong often enough that you should not quote from
it without checking the audio. Everything downstream is written to that standard — see CLAUDE.md.

## Notes

```bash
./notes transcripts/meeting.md              # notes beside the transcript
./notes transcripts/meeting.md -n           # dry run — nothing sent, nothing written
./notes transcripts/meeting.md -l lex.json  # fix names and jargon first
```

Output is a `.notes.md` you read and a `.notes.json` carrying machine-readable spans:

```markdown
## Decisions

- Billing moves to the new provider before the quarter closes.
  > "so we're agreed, we cut over before the quarter closes" — Speaker 2, 00:42:15
```

**Why the quotes can be trusted.** The model never sees a timestamp, so it has nothing to echo. It
must quote verbatim, and the *search* for that quote in the timeline is what attaches a real offset.
A quote that does not appear resolves to nothing and the claim is dropped before you read it. A
dropped claim is reported, not swallowed.

**A span proves *what* was said and *when*, not *who*.** Notes name an owner only when a name is
actually spoken — never because of a speaker label.

This step needs a timeline. Transcripts produced before timelines existed have none; re-run with
`-f`.

### Where the model comes from

Bedrock by default, or the Anthropic API when `ANTHROPIC_API_KEY` is set — `--provider` forces
either. Credentials are passed in, never discovered: `--region`, `--profile`, `--api-key`,
`--model`.

## For a model to call

```bash
./mcp                            # MCP server on stdio
```

Four tools, the same operations the CLIs use: `list_recordings`, `preview_notes`, `generate_notes`,
`transcribe`. The two that spend nothing are marked read-only so a caller can tell at a glance which
are safe to run unattended.

`transcribe` is capped at 15 minutes of audio — decoding takes roughly as long as the recording, so
longer files are for `./transcribe` in a terminal. Paths and credentials are injected:
`MEET_AI_AUDIO`, `MEET_AI_TRANSCRIPTS`, `MEET_AI_PROVIDER`, `MEET_AI_MODEL`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./fetch-models.sh          # only if you want --diarize
```

Requires `ffmpeg` on PATH. Whisper models download themselves on first use into
`~/.cache/huggingface/`; the diarization models are not on PyPI, hence the fetch script — two ONNX
files, ~37 MB, using sherpa-onnx rather than pyannote.audio to avoid PyTorch and a gated token.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

No model, no audio, no network.

## Layout

```
audio/         drop audio here (mp3, wav, m4a, flac, mp4, mkv, ...)
transcripts/   Markdown output and the timeline beside it
transcribe     wrapper — audio to transcript
notes          wrapper — timeline to notes
mcp            wrapper — the MCP front door, on stdio
transcribe.py  layers 1-3: chunk planning, decoding, Markdown
diarize.py     speaker labelling (optional, --diarize)
meetnotes/     operations.py (what this tool can be asked to do, as data), cli.py and
               mcp_server.py (the two front doors), extract.py (the one model call),
               spans.py (quote → offset, no model), verify.py, render.py, lexicon.py
tests/         no model, no audio, no network
models/        ONNX diarization models (~37 MB)
.cache/        per-chunk resume data and cached diarization
```
