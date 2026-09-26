"""Tests for layer 1.

These use real ffmpeg with a generated tone instead of a microphone -- no device, no room
audio, no network. The one that matters most is the orphan test: it kills the recorder
outright and checks the meeting survives, which is the entire reason for recording raw PCM.
"""

import json
import os
import shutil
import signal
import time
from datetime import datetime

import pytest

import capture
from transcribe import find_audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

TONE = ["-re", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
SILENCE = ["-re", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]


def record_for(audio_dir, seconds, input_args=TONE, **kwargs):
    session = capture.start(audio_dir, input_args=input_args, **kwargs)
    time.sleep(seconds)
    return session


# ---------------------------------------------------------------------------
# Pure
# ---------------------------------------------------------------------------

def test_a_session_is_named_for_when_it_started():
    assert capture.session_id(now=datetime(2026, 9, 26, 15, 30, 12)) == "2026-09-26-153012"


def test_a_given_name_becomes_a_filename_safe_slug():
    assert capture.session_id("Claims Review / Q3!") == "claims-review-q3"


def test_a_name_with_nothing_usable_falls_back_to_the_time():
    assert capture.session_id("!!!", now=datetime(2026, 1, 2, 3, 4, 5)) == "2026-01-02-030405"


def test_a_pid_that_is_not_our_recorder_is_never_treated_as_one():
    """A pid from before a reboot can belong to anything. Signalling it because a stale
    session file named it would be a bug with no upper bound on its damage."""
    assert capture.is_our_recorder(None, "/x.pcm.part") is False
    assert capture.is_our_recorder(os.getpid(), "/x.pcm.part") is False  # pytest, not ffmpeg
    assert capture.is_our_recorder(2 ** 22 + 7, "/x.pcm.part") is False  # no such process


def test_a_recording_in_progress_is_invisible_to_transcription(tmp_path):
    """find_audio() globs recursively. A half-finished recording it could see would be
    transcribed half-finished."""
    state = tmp_path / ".recording"
    state.mkdir()
    (state / "x.pcm.part").write_bytes(b"\0" * 1000)
    (tmp_path / "done.flac").write_bytes(b"fLaC")
    assert [p.name for p in find_audio([str(tmp_path)])] == ["done.flac"]


# ---------------------------------------------------------------------------
# Start and stop
# ---------------------------------------------------------------------------

def test_a_recording_becomes_lossless_audio_and_cleans_up_after_itself(tmp_path):
    record_for(tmp_path, 1.5, consent="test")
    done = capture.stop(tmp_path)

    assert done.state == "stopped"
    assert done.audio.endswith(".flac")
    assert 1.0 < done.duration_seconds < 4.0
    leftovers = sorted(p.name for p in (tmp_path / ".recording").iterdir())
    assert leftovers == []  # no part, no log, no in-progress session


def test_the_finished_session_is_kept_beside_its_audio(tmp_path):
    record_for(tmp_path, 1.0, consent="verbal, all present")
    done = capture.stop(tmp_path)
    saved = json.loads((tmp_path / f"{done.id}.session.json").read_text())
    assert saved["consent"] == "verbal, all present"
    assert saved["rate"] == 48000


def test_start_returns_promptly_rather_than_waiting_on_a_buffer(tmp_path):
    """Measured: without per-packet flushing ffmpeg writes in 256 KiB blocks and start()
    waited over two seconds for the first. The same buffer is what a crash would lose."""
    began = time.monotonic()
    capture.start(tmp_path, input_args=TONE)
    try:
        assert time.monotonic() - began < 1.5
    finally:
        capture.stop(tmp_path)


def test_audio_on_disk_tracks_the_clock(tmp_path):
    session = record_for(tmp_path, 2.0)
    try:
        assert capture.elapsed_seconds(session) > 1.5
    finally:
        capture.stop(tmp_path)


def test_a_second_recording_is_refused_while_one_runs(tmp_path):
    record_for(tmp_path, 0.3)
    try:
        with pytest.raises(capture.CaptureError, match="still recording"):
            capture.start(tmp_path, input_args=TONE, name="another")
    finally:
        capture.stop(tmp_path)


def test_a_source_that_cannot_open_fails_at_once_and_leaves_nothing(tmp_path):
    """A 'recording' that captured nothing is otherwise only discovered after the meeting."""
    with pytest.raises(capture.CaptureError, match="exited immediately"):
        capture.start(tmp_path, input_args=["-f", "lavfi", "-i", "no_such_filter"])
    assert capture.active(tmp_path) is None
    assert not list((tmp_path / ".recording").glob("*.part"))


def test_a_silent_recording_is_flagged(tmp_path):
    """A muted or wrong microphone is the classic way to lose a meeting."""
    record_for(tmp_path, 1.0, input_args=SILENCE)
    assert any("silent" in w for w in capture.stop(tmp_path).warnings)


def test_a_missing_consent_note_is_recorded_as_missing(tmp_path):
    record_for(tmp_path, 0.3)
    assert any("consent" in w for w in capture.stop(tmp_path).warnings)


def test_stopping_when_nothing_is_recording_says_so(tmp_path):
    with pytest.raises(capture.CaptureError, match="nothing is recording"):
        capture.stop(tmp_path)


# ---------------------------------------------------------------------------
# Crash safety -- the reason for raw PCM
# ---------------------------------------------------------------------------

def test_a_killed_recorder_leaves_a_recoverable_meeting(tmp_path):
    """SIGKILL gives ffmpeg no chance to finalise anything. A WAV here could decode as
    silence, because its header only records the length on a clean exit. Raw PCM has no
    header to corrupt: the audio is simply there."""
    session = record_for(tmp_path, 1.5)
    os.kill(session.pid, signal.SIGKILL)
    time.sleep(0.3)
    capture._reap(session.pid)

    interrupted = capture.active(tmp_path)
    assert interrupted is not None
    assert interrupted.state == "orphaned"

    recovered = capture.stop(tmp_path)
    assert recovered.state == "stopped"
    assert recovered.duration_seconds > 1.0
    assert any("recovered" in w for w in recovered.warnings)
