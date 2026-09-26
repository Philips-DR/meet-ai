"""Layer 5 -- ask a model what happened, in a shape that cannot lie about when.

The contract with the model is narrow on purpose:

  * it sees the transcript WITHOUT timestamps, so it has no number to echo;
  * every claim it makes must carry a quote copied verbatim from that transcript;
  * spans.py, not the model, turns each quote into a real offset.

A claim whose quote cannot be found in the timeline is dropped. That is the mechanism the
whole milestone rests on, and it is why the prompt spends most of its words on quoting
rather than on summarising.
"""

from __future__ import annotations

from dataclasses import dataclass

from meetnotes.client import ModelConfig, build_client

# Generous, and a cap rather than a target: adaptive thinking counts against it, and being
# truncated mid-answer costs a whole retry.
MAX_TOKENS = 32000

_CLAIM_PROPERTIES = {
    "text": {"type": "string", "description": "The claim, stated plainly in your own words."},
    "quote": {
        "type": "string",
        "description": (
            "A span of the transcript, copied EXACTLY as it appears, that evidences the "
            "claim. Copy it character for character. Do not paraphrase, join separate "
            "parts, or tidy up grammar."
        ),
    },
}

# Written out rather than generated from a Pydantic model: structured outputs want one
# flat schema with no $ref indirection, and an explicit schema is also the thing a reviewer
# can actually read.
NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "What this meeting was about and what came of it. A short paragraph.",
        },
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": dict(_CLAIM_PROPERTIES),
                "required": ["text", "quote"],
                "additionalProperties": False,
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **_CLAIM_PROPERTIES,
                    "owner": {
                        "type": "string",
                        "description": (
                            "Who owns this, ONLY if a name is spoken in the transcript "
                            "itself. Empty string otherwise. Never infer an owner from a "
                            "speaker label."
                        ),
                    },
                },
                "required": ["text", "quote", "owner"],
                "additionalProperties": False,
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": dict(_CLAIM_PROPERTIES),
                "required": ["text", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "decisions", "actions", "questions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are reading a transcript of a real meeting and writing notes somebody will act on.

The rule that matters more than any other: every decision, action and open question you \
report must carry a `quote` copied EXACTLY from the transcript text you were given. \
Character for character. Do not paraphrase inside the quote, do not join two separate \
pieces of speech into one quote, do not fix grammar, and do not add punctuation that is \
not there.

Any claim whose quote does not appear verbatim in the transcript will be discarded \
automatically before anyone reads it. A short, exact quote is worth more than a long, \
tidied one.

Two further constraints:

Speaker labels in this transcript come from automatic voice clustering and are frequently \
wrong. Never attribute an action to a person because of a speaker label. Fill in `owner` \
only when a name is actually spoken -- "Kwame will send the contract" -- and leave it \
empty otherwise.

Report only what was actually settled. A meeting that decided nothing has an empty \
decisions list, and saying so is far more useful than inventing a decision to fill it.\
"""


@dataclass(frozen=True)
class RawNotes:
    """What the model returned, before any of it has been verified."""

    summary: str
    decisions: list[dict]
    actions: list[dict]
    questions: list[dict]

    @classmethod
    def from_json(cls, data: dict) -> "RawNotes":
        return cls(
            summary=str(data.get("summary", "")),
            decisions=list(data.get("decisions", [])),
            actions=list(data.get("actions", [])),
            questions=list(data.get("questions", [])),
        )


def transcript_for_model(segments: list[dict]) -> str:
    """Render the timeline as plain speech, deliberately without timestamps.

    Withholding the numbers is not tidiness -- it is what makes a fabricated offset
    impossible rather than merely detectable. The model has nothing to copy.
    """
    lines: list[str] = []
    previous_speaker = object()
    for segment in segments:
        speaker = segment.get("speaker")
        if speaker != previous_speaker:
            lines.append("")
            lines.append(f"[Speaker {speaker + 1}]" if speaker is not None else "[Unknown]")
            previous_speaker = speaker
        lines.append(segment.get("text", ""))
    return "\n".join(lines).strip()


def _structured_call(segments: list[dict], config: ModelConfig, effort: str,
                     system: str, schema: dict) -> dict:
    """One model call over the whole transcript, answered in `schema`. Streamed, because
    it is a long one."""
    import json

    client = build_client(config)
    transcript = transcript_for_model(segments)

    with client.messages.stream(
        model=config.resolved_model,
        max_tokens=MAX_TOKENS,
        system=system,
        # Explicit, not omitted: on Opus 4.6 leaving `thinking` out means no thinking at
        # all. (On Opus 5 it defaults to adaptive, so this line is harmless there too.)
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
    ) as stream:
        message = stream.get_final_message()

    # output_config.format guarantees the first text block is valid JSON of this shape.
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def extract_notes(segments: list[dict], config: ModelConfig, effort: str = "high") -> RawNotes:
    return RawNotes.from_json(_structured_call(segments, config, effort, SYSTEM_PROMPT, NOTES_SCHEMA))


# ---------------------------------------------------------------------------
# Minutes -- the same contract, in the shape a secretary circulates
# ---------------------------------------------------------------------------

def _claims(description: str, with_owner: bool = False) -> dict:
    properties = dict(_CLAIM_PROPERTIES)
    required = ["text", "quote"]
    if with_owner:
        properties["owner"] = NOTES_SCHEMA["properties"]["actions"]["items"]["properties"]["owner"]
        required.append("owner")
    return {
        "type": "array",
        "description": description,
        "items": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


MINUTES_SCHEMA = {
    "type": "object",
    "properties": {
        "meeting": {
            "type": "string",
            "description": (
                "Which meeting this was, as the transcript itself names it, phrased to "
                "follow the words 'Minutes of the' -- e.g. 'third quarter meeting of the "
                "Retirees Association, Tema branch'. Empty string if it is never stated."
            ),
        },
        "meeting_quote": {
            "type": "string",
            "description": (
                "An exact quote from the transcript that names the meeting. Empty string "
                "when `meeting` is empty."
            ),
        },
        "opening": _claims(
            "How the meeting opened: prayers, a minute's silence, announcements, the "
            "chair's welcome. In the order they happened."
        ),
        "items": {
            "type": "array",
            "description": "The agenda items taken, in the order they were taken.",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {
                        "type": "string",
                        "description": (
                            "A short agenda-style heading, e.g. 'Minutes of the previous "
                            "meeting', 'Medical claims', 'Any other business'. No numbering."
                        ),
                    },
                    "discussion": _claims(
                        "What was reported, raised or discussed under this item, in formal "
                        "minutes style: past tense, third person."
                    ),
                    "resolutions": _claims(
                        "What was resolved, agreed or accepted. Name who moved or seconded "
                        "only if that name is spoken in the transcript."
                    ),
                    "actions": _claims("What somebody is to do as a result.", with_owner=True),
                },
                "required": ["heading", "discussion", "resolutions", "actions"],
                "additionalProperties": False,
            },
        },
        "closing": _claims("How the meeting closed: date of the next meeting, closing prayer."),
    },
    "required": ["meeting", "meeting_quote", "opening", "items", "closing"],
    "additionalProperties": False,
}

MINUTES_PROMPT = """\
You are the secretary, writing the formal minutes of a real meeting from its transcript. \
The minutes will be circulated to members and read back for adoption at the next meeting, \
so they must be accurate above all else.

Write in the register of minutes: past tense, third person, plain and formal -- "The \
Chairman informed members that...", "Members were reminded to...", "It was agreed that...". \
Group what was said under the agenda items the meeting actually took, in the order it took \
them. Vendor presentations, announcements and any other business are items too.

The rule that matters more than any other: every point you record must carry a `quote` \
copied EXACTLY from the transcript text you were given. Character for character. Do not \
paraphrase inside the quote, do not join separate pieces of speech into one quote, do not \
fix grammar, and do not add punctuation that is not there. Any point whose quote does not \
appear verbatim in the transcript is discarded automatically before anyone reads it, so a \
short exact quote is worth more than a long tidied one.

Names: write a person's name only when it is spoken in the transcript. Speaker labels come \
from automatic voice clustering and are frequently wrong -- never use one to decide who said \
something, who moved a motion or who owns an action. When the transcript does not say who, \
write "a member" or leave the owner empty. Nobody's attendance can be known from a \
recording; do not list who was present.

Transcription errors: the transcript was produced by speech recognition and mishears names \
and places. In your own `text` you may correct an obvious mishearing when the correct form \
is certain from context, but a `quote` is always copied exactly as the transcript has it.

Record only what actually happened. An item where nothing was resolved has an empty \
resolutions list; do not invent a resolution to fill it.\
"""


@dataclass(frozen=True)
class RawMinutes:
    """What the model returned for minutes, before any of it has been verified."""

    meeting: str
    meeting_quote: str
    opening: list[dict]
    items: list[dict]
    closing: list[dict]

    @classmethod
    def from_json(cls, data: dict) -> "RawMinutes":
        return cls(
            meeting=str(data.get("meeting", "")),
            meeting_quote=str(data.get("meeting_quote", "")),
            opening=list(data.get("opening", [])),
            items=list(data.get("items", [])),
            closing=list(data.get("closing", [])),
        )


def extract_minutes(segments: list[dict], config: ModelConfig, effort: str = "high") -> RawMinutes:
    return RawMinutes.from_json(_structured_call(segments, config, effort, MINUTES_PROMPT, MINUTES_SCHEMA))
