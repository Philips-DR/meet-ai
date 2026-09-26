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
    list_audio_sources,
    list_recordings,
    preview_notes,
    recording_status,
    start_recording,
    start_transcription,
    stop_recording,
    transcribe_audio,
    transcription_status,
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
            f"{MAX_SYNC_AUDIO_SECONDS // 60} minutes of audio and returns the result in one "
            "call. For anything longer -- any real meeting -- use start_transcription."
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

    @server.tool(
        name="list_audio_sources",
        title="List microphones and output monitors",
        description=(
            "Every capture source this machine exposes. Microphones capture the room; a "
            "'.monitor' source captures what the speakers are playing, which on a call is "
            "everyone except the user. Reads only."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def _list_audio_sources() -> dict:
        try:
            return {"sources": list_audio_sources()}
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="recording_status",
        title="Is anything recording?",
        description=(
            "The unfinished recording session, if there is one, and how much audio it has "
            "on disk. A session whose recorder died is reported as 'orphaned' -- its audio "
            "is still there and stop_recording will recover it. Reads only."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def _recording_status() -> dict:
        try:
            return {"session": recording_status(paths.audio_dir)}
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="start_recording",
        title="Start recording a meeting",
        description=(
            "Begin recording from a microphone, in the background, and return at once. "
            "Nothing blocks while the meeting happens; call stop_recording when it ends. "
            "Pass `consent` describing who agreed to be recorded and how -- it is stored "
            "with the audio. Refuses if a recording is already running."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        ),
    )
    def _start_recording(source: str | None = None, consent: str | None = None,
                         name: str | None = None) -> dict:
        try:
            return start_recording(paths.audio_dir, source=source, consent=consent, name=name)
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="stop_recording",
        title="Stop recording, and save the audio",
        description=(
            "End the running recording and write it as lossless FLAC, warning if it came "
            "out silent. Also recovers an interrupted session from what reached disk. "
            "Returns the audio path, ready to hand to start_transcription."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        ),
    )
    def _stop_recording(session_id: str | None = None) -> dict:
        try:
            return stop_recording(paths.audio_dir, session_id)
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="start_transcription",
        title="Transcribe a recording in the background",
        description=(
            "Start transcribing an audio file of any length and return at once with a job "
            "id. The decode runs in the background -- roughly half the recording's length, "
            "about double that with speaker labels -- and nothing waits on it. Check on it "
            "with transcription_status; when it is done, the timeline it reports can go "
            "straight to preview_notes and generate_notes. For speaker labels pass diarize "
            "and the real number of people present: auto-detection over-splits badly. One "
            "job runs at a time."
        ),
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, open_world_hint=False
        ),
    )
    def _start_transcription(audio: str, diarize: bool = False, speakers: int = 0,
                             language: str | None = None, vocabulary: str | None = None,
                             force: bool = False) -> dict:
        try:
            return start_transcription(Path(audio), paths.transcript_dir, diarize=diarize,
                                       speakers=speakers, language=language,
                                       vocabulary=vocabulary, force=force)
        except OperationError as error:
            return _failed(error)

    @server.tool(
        name="transcription_status",
        title="How is a transcription getting on?",
        description=(
            "State of a transcription job -- running, done, failed, or interrupted -- with "
            "its phase, percent and time remaining while it runs, and the transcript and "
            "timeline paths once it is done. Pass a job id, or omit it for the most recent. "
            "An interrupted job can be started again and resumes where it stopped for long "
            "files. Reads only."
        ),
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
    )
    def _transcription_status(job_id: str | None = None) -> dict:
        try:
            return {"job": transcription_status(paths.transcript_dir, job_id)}
        except OperationError as error:
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
