"""meet-ai's second front door: the same operations the CLIs expose, reachable by a model.

Both doors sit directly on operations.py; neither wraps the other. A model calling
generate_notes and a person typing ./notes run identical code and differ only in how the
result is formatted -- JSON here, prose there.

Everything this door needs is injected: the model config AND the directories it may look
in. A tool that decides for itself where its data lives cannot be pointed at a second
user's data later without being rewritten, and a tool that resolves its own credentials
forces every other tool to agree on where identity lives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from meetnotes.client import ModelConfig, resolve_provider
from meetnotes.operations import (
    MAX_SYNC_AUDIO_SECONDS,
    OperationError,
    as_dict,
    generate_notes,
    list_recordings,
    preview_notes,
    transcribe_audio,
)

VERSION = "0.1.0"


@dataclass(frozen=True)
class ServerPaths:
    """Where this server may read and write. Supplied by the caller, never assumed."""

    audio_dir: Path
    transcript_dir: Path


def _failed(error: Exception) -> dict:
    return {"error": str(error)}


def create_meet_ai_server(paths: ServerPaths, config: ModelConfig) -> MCPServer:
    server = MCPServer(name="meet-ai", version=VERSION)

    @server.tool(
        name="list_recordings",
        title="List audio and transcripts",
        description=(
            "Show what audio files and transcripts are on disk, and for each transcript "
            "whether a timeline sits beside it. A transcript with no timeline cannot have "
            "notes made from it and must be re-transcribed first."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def _list_recordings() -> dict:
        try:
            return as_dict(list_recordings(paths.audio_dir, paths.transcript_dir))
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="preview_notes",
        title="Preview what would be sent for notes",
        description=(
            "Report the size of the transcript that would be sent to the model and where "
            "the notes would be written, without calling a model or writing anything. "
            "Costs nothing and needs no credentials, so it is the safe thing to run first."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def _preview_notes(timeline: str, lexicon: str | None = None) -> dict:
        try:
            return as_dict(
                preview_notes(Path(timeline), lexicon_path=Path(lexicon) if lexicon else None)
            )
        except (OperationError, OSError, ValueError) as error:
            return _failed(error)

    @server.tool(
        name="generate_notes",
        title="Write meeting notes from a timeline",
        description=(
            "Turn a transcript timeline into notes: a summary, decisions, action items and "
            "open questions. Every claim carries a quote that was found verbatim in the "
            "transcript, with the real audio offset attached; a claim whose quote cannot be "
            "found is dropped rather than reported. Writes a markdown file and a JSON file "
            "carrying the spans, and calls a model, which costs money."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=True
        ),
    )
    def _generate_notes(timeline: str, lexicon: str | None = None, effort: str = "high") -> dict:
        try:
            return as_dict(
                generate_notes(
                    Path(timeline),
                    config,
                    lexicon_path=Path(lexicon) if lexicon else None,
                    effort=effort,
                )
            )
        except (OperationError, OSError, ValueError) as error:
            return _failed(error)

    @server.tool(
        name="transcribe",
        title="Transcribe an audio file",
        description=(
            "Decode one audio file into a transcript and a timeline, optionally labelling "
            f"speakers. Runs on this machine with no audio leaving it. Capped at "
            f"{MAX_SYNC_AUDIO_SECONDS // 60} minutes of audio: decoding takes roughly as "
            "long as the recording, so anything longer must be run from a terminal with "
            "./transcribe instead."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        ),
    )
    def _transcribe(
        audio: str,
        diarize: bool = False,
        speakers: int = 0,
        language: str | None = None,
        vocabulary: str | None = None,
        force: bool = False,
    ) -> dict:
        try:
            return as_dict(
                transcribe_audio(
                    Path(audio),
                    paths.transcript_dir,
                    diarize=diarize,
                    speakers=speakers,
                    language=language,
                    prompt=vocabulary,
                    force=force,
                )
            )
        except (OperationError, OSError, ValueError) as error:
            return _failed(error)

    return server


def main() -> None:
    """The one place this entry point reaches for the environment, named so it is visible."""
    root = Path(os.environ.get("MEET_AI_ROOT", Path(__file__).resolve().parent.parent))
    paths = ServerPaths(
        audio_dir=Path(os.environ.get("MEET_AI_AUDIO", root / "audio")),
        transcript_dir=Path(os.environ.get("MEET_AI_TRANSCRIPTS", root / "transcripts")),
    )
    config = ModelConfig(
        provider=resolve_provider(os.environ.get("MEET_AI_PROVIDER")),
        model=os.environ.get("MEET_AI_MODEL"),
        region=os.environ.get("AWS_REGION"),
        profile=os.environ.get("AWS_PROFILE"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    create_meet_ai_server(paths, config).run(transport="stdio")


if __name__ == "__main__":
    main()
