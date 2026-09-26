# meet-ai — engineering rules

`plan.md` holds the locked decisions and milestone sequence. **This file holds the engineering
rules.** When the two disagree, this file wins and `plan.md` should be corrected. `README.md` is
for someone using the tool; everything here is for someone changing it.

## Commands

    .venv/bin/python -m pytest tests/ -q   # must pass with no model, no audio, no network
    ./transcribe <file>                    # audio → transcript + timeline
    ./notes <timeline> [-n]                # timeline → notes. -n is the dry run
    ./mcp                                  # MCP front door, on stdio

## Mental model — read this before touching spans.py or the notes layer

**A transcript is not text. It is a timeline of tokens, each carrying a speaker and an absolute
offset into the audio.**

Flatten it to a string and everything downstream becomes unverifiable: you cannot cite, correct a
speaker label, jump to the audio, or tell a real quote from an invented one. Every design decision
below follows from that one fact.

This is why the timeline JSON is written on **every** run, not only under `--keep-cache`. The
Markdown is a *view* of that file, not the other way round.

## Architecture — layers and hard boundaries

    audio → [1] capture    → raw audio + session metadata   devices; knows nothing about ASR
          → [2] segment    → chunks with absolute offsets   pure, deterministic
          → [3] transcribe → Timeline                       provider-swappable; the only network
          → [4] resolve    → corrected Timeline             lexicon — deterministic today
          → [5] render     → markdown + spans               MODEL SEAM: the one model call
          → [6] verify     → every span resolves            reads only

- **Never let a model touch a timestamp.** It will produce a plausible one, the citation will point
  at the wrong moment, and nothing will fail loudly. Offsets are arithmetic, and arithmetic is
  layer 2's job.
- **Layers 1–3 and 6 are model-free.** Layer 4 is deterministic too (see below); only layer 5 calls
  a model.
- **meet-ai never talks to Google Docs.** Output is markdown plus a timeline JSON. docu-ai compiles
  it. Neither tool knows the other's internals.
- **Layer 6 never mutates.** It drops claims; it does not rewrite them.
- **Layer 1 is single-source today.** One microphone; the two-channel mode that would make one
  speaker's identity exact on calls is not built.

## The headline requirement, expressed as a mechanism

**Nothing in the notes is invented.**

The model is shown the transcript **without timestamps**, so it has no number to echo. Every claim
must carry a quote copied verbatim. `spans.py` — not the model — searches the timeline for that
quote and attaches the real offset. A fabricated quote does not resolve, and the claim is dropped
before anyone reads it. The model produces words; only arithmetic produces numbers.

**A dropped claim is reported, never swallowed.** It is the check working, and also a signal about
the prompt worth seeing.

**What a span proves is *what* was said and *when*, not *who*.** Attribution rests on diarization,
which is serviceable and not reliable (below), so notes name an owner only when a name is actually
spoken — never because of a speaker label. A quote crossing two speakers resolves with `speaker:
None` rather than guessing.

### The seam fix that made it work

**Every segment boundary is a word boundary.** The first live run dropped a genuine quote: segment
text was folded into the search haystack with no separator, so a segment ending `"...goes to this"`
and the next beginning `"department, like how..."` folded to `thisdepartment` and a true quote
crossing the seam looked invented. Whisper splits on pauses, which land mid-phrase constantly, so
this would have silently rejected legitimate claims in every recording.

## Two front doors, one implementation

`meetnotes/operations.py` holds what meet-ai can be asked to do, as data in and data out. `./notes`
formats those results as prose; `meetnotes/mcp_server.py` returns them as JSON. **Neither door wraps
the other** — a door that delegates inherits the other's output format, exit codes and assumptions
about who is reading, and they drift the moment either changes for its own audience.

- **Nothing in `operations.py` writes to stdout.** `transcribe.py` predates it and prints progress
  as it goes, so the one operation that calls into it captures that. On stdio **stdout IS the
  protocol stream** and one stray line corrupts the session. There is a test asserting a tool call
  writes nothing to stdout — a real hazard here, not a hypothetical one.
- **`preview_notes` spends nothing and needs no credentials.** It is the approval surface, annotated
  `readOnlyHint` so a caller can tell at a glance which tools are safe to run unattended.
- **Synchronous transcription does not fit request/response.** Decoding takes roughly as long as the
  recording, so the MCP door caps audio at `MAX_SYNC_AUDIO_SECONDS` and refuses past it, pointing at
  `./transcribe`. Lifting the cap means a background job with a status operation — a feature, not a
  bigger number.
- **Paths and credentials are injected.** `ServerPaths` carries the directories; `ModelConfig`
  carries the provider, model, region, profile and key. A tool that discovers either cannot be
  pointed at a second user's data without a rewrite.

## Measured facts

Numbers here were measured on this machine (4-core i7-8665U, `int8_float32`, real meeting audio).
Re-measure the same way before changing anything that depends on them.

**Model size is the biggest accuracy lever, and `distil-large-v3` is counter-intuitively the fast
one** — 1.9× realtime versus `small`'s 0.87×, despite being three times the size. Whisper's
bottleneck is the *decoder*, which runs sequentially one token at a time; distil keeps the full
32-layer encoder and cuts the decoder to 2 layers, so it reads with `large-v3`'s quality and writes
far faster. `float32` measured 0.54×; `int8` measured 0.87×, i.e. no faster than the
`int8_float32` default, which is why the default keeps activations in float32 for free.

**English-only models do not degrade gracefully — they emit fluent nonsense.** So the script
*refuses* to run when an English-only model is paired with a non-English `--language`, rather than
producing a confident wrong transcript.

**Chunking is for resumability, not accuracy.** Cuts land on the midpoint of a real silence found by
ffmpeg's `silencedetect`, searched up to `BOUNDARY_SEARCH` seconds around each target — so nothing
falls mid-word and there is nothing to de-duplicate at a seam. Because `condition_on_previous_text`
is off by default, Whisper already carries no context across its internal 30-second windows, so
chunking costs essentially nothing. **The exception: `--condition` makes context flow, and chunk
boundaries then break it.**

**Timestamps are shifted into whole-file time at the moment a segment is created**, never carried
across a boundary afterwards.

## Capture — layer 1

**Recording is a session, not a long-running call.** It ends when a person says stop, which no
request/response call can model at any duration. `start` and `stop` both return at once; the audio
is written by a detached ffmpeg nobody waits on; the state between them lives on disk. This is what
lets the assistant say "record this meeting". It needed no job machinery — transcription still does.

- **Raw PCM, with the session JSON as its header.** A WAV records its length in a header finalised
  only on a clean exit, so a killed recorder can leave a file claiming zero samples: a two-hour
  meeting that decodes as silence. Raw PCM has no header to corrupt — every byte is audio, and the
  rate and channels needed to read it are in the session file. Crash-safe by construction rather
  than by recovery code. FLAC is written only on a clean stop. The test for this kills the recorder
  with SIGKILL and recovers the meeting.
- **`-flush_packets 1` is load-bearing, measured.** Without it ffmpeg writes in 256 KiB blocks —
  about 2.7 s at 48 kHz mono, over 8 s at 16 kHz. The file grows in jumps, `start()` waited 2.3 s
  for the first one, and a crash loses whatever sat in the buffer, which is precisely the loss raw
  PCM was chosen to avoid. With it, `start()` returns in 0.1 s and the file tracks the clock.
- **Confirm audio is flowing before claiming to record.** A bad device makes ffmpeg exit at once;
  `start()` watches for that and fails immediately rather than letting a "recording" that captured
  nothing be discovered after the meeting.
- **Never signal a pid without checking it is still our recorder** — `/proc/<pid>/cmdline` must be
  ffmpeg writing this exact file, and not a zombie. A pid from before a reboot can be anything.
- **In-progress audio lives in `.recording/` with a `.pcm.part` extension.** `find_audio()` globs
  recursively, and a half-finished recording it could see would be transcribed half-finished.
- **The recorder runs in its own session** so a terminal's Ctrl-C does not reach it directly;
  `stop()` interrupts it deliberately, which is what gives ffmpeg the chance to flush.
- **48 kHz, lossless.** Whisper resamples to 16 kHz anyway, so 16 k would lose nothing today — and
  foreclose re-running with a better model later. Roughly 350 MB for two hours.
- **A silent recording is flagged on stop**, from `volumedetect` in the same pass as the FLAC
  encode so a long recording is read once.
- **Capture does not improve accuracy** for a single microphone in a room. Say so wherever it is
  described, because the natural assumption is that it would.

## Diarization — what it is honestly worth

**Serviceable, not reliable.** Who-said-what is inferred from voice similarity, and a single room
mic with people talking over each other is the hard case. Expect the gist right and individual
attributions wrong often enough that nothing should be quoted from it without checking the audio.
Everything downstream is written to that standard.

- **Diarization runs on the whole file, never per chunk.** Speaker clusters are only comparable
  within one clustering pass, so chunking would make "Speaker 1" a different person in each chunk.
- **Speakers are voted per sentence, not per word.** Word-level labels are jittery mid-sentence:
  on a 3-minute sample that produced 22 mid-sentence speaker switches, and sentence voting cut it to
  1. Word timings still drive the vote, so genuine turn-taking is caught.
- **Always pass the real headcount.** Auto-detection over-splits badly — 18 clusters on a 3-minute
  sample with 4 people, most of them sub-5-second slivers of cross-talk.
- Speakers are numbered by airtime, so Speaker 1 talks most. The clusterer's own labels are not
  stable across runs; this numbering is.
- A diarized run cannot reuse a plain run's cache: it needs word timestamps, so it re-transcribes.

## Markdown output

**A spoken line must stay a spoken line.** `escape_block_start` escapes leading block syntax, because
CommonMark reads `"160. That's one more."` as the 160th item of a list. Found by running docu-ai over
a real 2-hour transcript: three spoken numbers became list items.

Escaping the marker's **last** character is the only form that works universally — `15\.` is text,
while `\15.` renders the backslash, since a backslash escapes punctuation and not digits.

## Reaching a model

Bedrock by default, the first-party Anthropic API when `ANTHROPIC_API_KEY` is set. Probed live
against this account on 2026-09-20:

- **`claude-opus-5`, `opus-4-8`, `opus-4-7` and `sonnet-5` all return 403 "not available for this
  account".** Opus 4.6 is the most capable id that answers, and is the Bedrock default until access
  is granted in the console. Raise it the moment that changes.
- **This account reaches Claude only through cross-region inference profiles** (`us.anthropic...`).
  A bare on-demand id returns "isn't supported with on-demand throughput", and
  `AnthropicBedrockMantle` — the recommended client for new code — returns 404 for every profile id.
  Hence `AnthropicBedrock`. Revisit if Mantle gains profile support.
- **A profile id already contains `anthropic.` without starting with it**, so a `startswith` check
  produces `anthropic.us.anthropic...`. The bug was total, not cosmetic.
- **Opus 4.6 needs `thinking` set explicitly** — omitting it means no thinking at all, unlike Opus 5
  — and rejects `effort: "xhigh"`, which arrived with 4.7.

## Layer 4 is deterministic on purpose

The plan reserved it as a model seam; it is a dictionary substitution instead. A lexicon lookup is
exact, instant, reviewable, and cannot invent a correction nobody asked for. **Nothing should be
handed to a model that arithmetic or a lookup already does correctly.** The seam stays open above it
for the cases a dictionary genuinely cannot reach.

Correction may change the text inside a span. It may **never** move a span boundary.

## Testing

No model, no audio, no network — ever. `requirements-dev.txt` pulls in nothing heavier than pytest.

The load-bearing tests, in order: a claim whose quote is absent from the transcript is dropped; a
quote crossing a segment seam still resolves; chunks are contiguous and cover the whole file; a tool
call writes nothing to stdout.

**Name fake state so it cannot collide with the API surface the fake also presents.** `self.messages`
shadows `messages()`; both surface as "'dict' object is not callable" a long way from the cause.

## Don'ts

- Don't let a model near an offset, an index, or a segment id.
- Don't make the Markdown the source of truth. The timeline is.
- Don't assert who said something on the strength of a speaker label.
- Don't print to stdout from anything the MCP door can reach.
