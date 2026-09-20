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


def extract_notes(segments: list[dict], config: ModelConfig, effort: str = "high") -> RawNotes:
    """One model call over the whole transcript. Streamed, because it is a long one."""
    import json

    client = build_client(config)
    transcript = transcript_for_model(segments)

    with client.messages.stream(
        model=config.bedrock_model_id,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": NOTES_SCHEMA}},
        messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
    ) as stream:
        message = stream.get_final_message()

    # output_config.format guarantees the first text block is valid JSON of this shape.
    text = next(block.text for block in message.content if block.type == "text")
    return RawNotes.from_json(json.loads(text))
