# meet-ai — decisions and milestones

`CLAUDE.md` holds the engineering rules. **This file holds the locked decisions and the milestone
sequence.** When the two disagree, CLAUDE.md wins and this file gets corrected.

## What this is

Audio in, a transcript and a timeline out, and notes in which every claim cites the moment it was
said. One half of a suite: meet-ai emits markdown, docu-ai compiles it into a Google Doc, and the
assistant decides when to call either. Neither tool knows the other's internals.

## Locked decisions

| Decision | Choice | Why |
| --- | --- | --- |
| Transcription | faster-whisper, local, CPU | The audio never leaves the machine. No PyTorch, no GPU, no API call |
| Default model | `distil-large-v3` | Measured 1.9× realtime, near `large-v3` quality. English only |
| Diarization | sherpa-onnx on ONNX Runtime | pyannote.audio wants PyTorch (~2.5 GB) and a gated HF token for the same result |
| Chunking | Resumability only, cut at silences | Accuracy cost is ~nil with `condition_on_previous_text` off |
| The timeline | Written every run, never optional | The Markdown is a view of it. Without it, no claim can be checked |
| Layer 4 correction | Deterministic lexicon, not a model | A lookup is exact and cannot invent a correction. The seam stays open above it |
| Notes model | Claude, one call, spans resolved locally | The model writes words; only arithmetic writes offsets |
| Provider | Bedrock, or the API when a key is set | Injected, never discovered |
| Front doors | CLI and MCP, side by side on `operations.py` | Neither wraps the other |

## Milestones

| | Deliverable | Status |
| --- | --- | --- |
| M0 | A transcript through `docu-ai build` | Done — found two format gaps, both fixed |
| M1 | Persist the timeline beside the markdown | Done |
| M2 | Tests over the pure functions | Done |
| M3 | Layer 1: two-channel capture | **Not started — the only missing layer** |
| M4 | Notes that carry spans, and the lint | Done |
| M5 | MCP front door beside the CLI | Done |

### M3, the one that is left

Layer 1 does not exist. Audio arrives in `audio/` by hand, recorded some other way.

The prize is not convenience. On PipeWire the microphone and the system monitor source can be
captured separately, and then **speaker identity for one participant is exact rather than
probabilistic** — diarization only has to solve the other side. Given that diarization is the
weakest link in the whole pipeline (CLAUDE.md is blunt about it), removing half the problem by
changing how the recording is made is worth more than any tuning.

Also belongs here: making consent a first-class option rather than something bolted on, since
recording meetings has a surface that grows the moment this is used with anyone else.

## Deliberately not built

- **A background job runner.** The MCP door caps synchronous transcription at fifteen minutes of
  audio and points longer files at the CLI. Lifting that cap means a job with a status operation,
  which is a feature rather than a bigger number.
- **Hosted diarization.** It would likely fix attribution, at the cost of sending client audio off
  the machine. That trade is not obviously right, and it is the one open question that changes what
  a span is allowed to claim.
- **A model in layer 4.** See CLAUDE.md.

## Open questions

- Is local diarization good enough, or does attribution need a hosted provider? Transcription is
  settled; this is the weak link.
- Where should the store live once there is more than one user? Paths are caller-chosen already, so
  this is a default rather than a rewrite.
