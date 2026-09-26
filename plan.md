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
| M3a | Layer 1: single-source capture as a session | Done — crash-safe, on both doors |
| M3b | Layer 1: two-channel capture for calls | Not started |
| M4 | Notes that carry spans, and the lint | Done |
| M5 | MCP front door beside the CLI | Done |
| M6 | Transcription as a background job | Done — removes the 15-minute limit for real meetings |

### M3b, what is left of layer 1

Single-source capture exists and is a convenience, not an accuracy gain: one microphone in a room
records everyone mixed, and the real recording in this repo — mono, two hours, nine speakers — is
exactly that case.

Two-channel capture would change that **only on calls.** On PipeWire the microphone and the system
monitor are separate streams; recording them as two files, transcribing each and merging by offset
would make one participant's identity exact and leave diarization only the rest. It helps nothing
in a room, where everyone shares the same air.

For in-person meetings the lever is **speaker enrollment** instead: a short voice sample per person,
so clusters come back as names and attribution stops being a guess for anyone enrolled. That is the
more valuable next step for the recordings this tool is actually used on.

## Deliberately not built

- **Recording on the MCP door as a blocking call.** It is a session instead; see CLAUDE.md.

- **Hosted diarization.** It would likely fix attribution, at the cost of sending client audio off
  the machine. That trade is not obviously right, and it is the one open question that changes what
  a span is allowed to claim.
- **A model in layer 4.** See CLAUDE.md.

## Open questions

- Is local diarization good enough, or does attribution need a hosted provider? Transcription is
  settled; this is the weak link.
- Where should the store live once there is more than one user? Paths are caller-chosen already, so
  this is a default rather than a rewrite.
