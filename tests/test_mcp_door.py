"""Tests for the MCP front door.

No network, no model, no credentials. The door is driven directly through list_tools() and
call_tool(), which is the same path a real session takes once the transport is stripped
away.
"""

import asyncio
import contextlib
import io
import json
from pathlib import Path

import pytest

from meetnotes.client import ModelConfig
from meetnotes.mcp_server import ServerPaths, create_meet_ai_server
from meetnotes.operations import MAX_SYNC_AUDIO_SECONDS, OperationError, transcribe_audio


SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "We should move billing to the new provider.", "speaker": 0},
    {"start": 4.0, "end": 8.0, "text": "Kwame will send the contract on Friday.", "speaker": 1},
]


@pytest.fixture
def workspace(tmp_path):
    audio = tmp_path / "audio"
    transcripts = tmp_path / "transcripts"
    audio.mkdir()
    transcripts.mkdir()
    (transcripts / "meeting.md").write_text("# meeting\n\nbody\n", encoding="utf-8")
    (transcripts / "meeting.timeline.json").write_text(
        json.dumps({"version": 1, "source": "meeting.m4a", "duration": 8.0, "segments": SEGMENTS}),
        encoding="utf-8",
    )
    (transcripts / "orphan.md").write_text("# orphan\n\nno timeline\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def server(workspace):
    return create_meet_ai_server(
        ServerPaths(audio_dir=workspace / "audio", transcript_dir=workspace / "transcripts"),
        ModelConfig(region="us-east-1"),
    )


def call(server, name, **arguments) -> dict:
    result = asyncio.run(server.call_tool(name, arguments))
    text = "".join(block.text for block in result.content if block.type == "text")
    return json.loads(text)


def test_the_door_advertises_exactly_its_operations(server):
    names = sorted(tool.name for tool in asyncio.run(server.list_tools()))
    assert names == [
        "generate_notes", "list_audio_sources", "list_recordings", "preview_notes",
        "recording_status", "start_recording", "stop_recording", "transcribe",
    ]


def test_the_operations_that_spend_nothing_are_marked_read_only(server):
    read_only = {
        tool.name: tool.annotations.read_only_hint
        for tool in asyncio.run(server.list_tools())
    }
    assert read_only["preview_notes"] is True
    assert read_only["list_recordings"] is True
    assert read_only["list_audio_sources"] is True
    assert read_only["recording_status"] is True
    assert read_only["generate_notes"] is False
    assert read_only["transcribe"] is False
    # Starting a recording turns on a microphone; it must pass through an approval gate.
    assert read_only["start_recording"] is False
    assert read_only["stop_recording"] is False


def test_listing_reports_which_transcripts_can_actually_have_notes(server):
    """A transcript with no timeline beside it is unusable, and saying so here is what
    stops a caller discovering that only after a failed call."""
    listed = call(server, "list_recordings")
    by_name = {Path(t["markdown"]).name: t for t in listed["transcripts"]}
    assert by_name["meeting.md"]["timeline"] is not None
    assert by_name["orphan.md"]["timeline"] is None


def test_preview_runs_without_any_credentials(server, workspace):
    preview = call(
        server, "preview_notes", timeline=str(workspace / "transcripts" / "meeting.timeline.json")
    )
    assert preview["segments"] == 2
    assert preview["approx_tokens"] > 0
    assert preview["would_write"].endswith("meeting.notes.md")


def test_a_missing_timeline_comes_back_as_an_error_not_an_exception(server, workspace):
    """A door that raises takes the session down with it."""
    result = call(server, "preview_notes", timeline=str(workspace / "nope.timeline.json"))
    assert "error" in result
    assert "no timeline" in result["error"]


def test_nothing_is_written_to_stdout_while_a_tool_runs(server, workspace):
    """On stdio, stdout IS the protocol stream: one stray line of progress corrupts it.
    transcribe.py predates this door and prints as it goes, so this is a real hazard and
    not a hypothetical one."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        call(server, "list_recordings")
        call(
            server,
            "preview_notes",
            timeline=str(workspace / "transcripts" / "meeting.timeline.json"),
        )
    assert sink.getvalue() == ""


def test_long_audio_is_refused_before_anything_is_decoded(tmp_path, monkeypatch):
    """Decoding takes roughly as long as the recording, so a two-hour file would hang the
    call for two hours and then time out with nothing to show. Refusing is the honest
    answer until there is a background job to hand it to."""
    import transcribe

    audio = tmp_path / "long.wav"
    audio.write_bytes(b"not really audio")
    monkeypatch.setattr(transcribe, "probe_duration", lambda _p: MAX_SYNC_AUDIO_SECONDS + 1)

    with pytest.raises(OperationError) as excinfo:
        transcribe_audio(audio, tmp_path)
    assert "./transcribe" in str(excinfo.value)


def test_short_audio_passes_the_duration_gate(tmp_path, monkeypatch):
    """The gate must not be so blunt that a voice note cannot be transcribed. Stops before
    the model loads -- proving the gate opened, without decoding anything."""
    import transcribe

    audio = tmp_path / "short.wav"
    audio.write_bytes(b"not really audio")
    monkeypatch.setattr(transcribe, "probe_duration", lambda _p: 60.0)
    monkeypatch.setattr(
        transcribe, "transcribe_file", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("reached decode"))
    )

    with pytest.raises(RuntimeError, match="reached decode"):
        transcribe_audio(audio, tmp_path)
