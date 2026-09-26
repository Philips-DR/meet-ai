"""Tests for transcription jobs.

A stand-in replaces transcribe.py: it prints progress in the same format and writes the same
outputs, in a second rather than an hour. What is under test is the job machinery -- detach,
report, finish, fail, recover -- not the decoder.
"""

import os
import signal
import sys
import textwrap
import time

import pytest

import jobs


def stand_in(tmp_path, body: str) -> list[str]:
    script = tmp_path / "fake_transcribe.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, "-u", str(script)]


SUCCEEDS = """
    import sys, time, json, pathlib
    audio, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[3])
    print("Loading model 'distil-large-v3' (CPU, int8_float32)...", flush=True)
    for p in (10.0, 55.5, 99.9):
        print(f"\\r   {p:5.1f}%  00:00:01/00:00:02  1.9x  eta 00:00:01    ", end="", file=sys.stderr, flush=True)
        time.sleep(0.4)
    (out / f"{audio.stem}.md").write_text("# t\\n")
    (out / f"{audio.stem}.timeline.json").write_text(json.dumps({"version": 1, "segments": []}))
    print("\\nDone. 1 transcript(s) written", flush=True)
"""

CRASHES = """
    import sys
    print("Loading model...", flush=True)
    raise SystemExit("\\u2717  audio.m4a: ffmpeg could not decode")
"""

HANGS = """
    import sys, time
    print("Loading model...", flush=True)
    print("\\r   20.0%  00:00:01/00:00:05  1.9x  eta 00:00:04    ", end="", file=sys.stderr, flush=True)
    time.sleep(60)
"""


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "meeting.flac"
    path.write_bytes(b"fLaC")
    return path


def wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return False


def job_command(tmp_path, body, audio):
    """The stand-in receives the same arguments transcribe.py would, so it can find them."""
    return stand_in(tmp_path, body) + [str(audio), "-o", str(tmp_path)]


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------

def test_the_latest_progress_line_is_the_current_one():
    """Progress is written with carriage returns, so the log is one long run of
    overwritten updates. Reading the first would report 12% forever."""
    log = "\r   12.4%  00:01/00:10  1.9x  eta 00:04:40  \r   45.2%  00:04/00:10  1.9x  eta 00:02:53  "
    assert jobs.read_progress(log) == ("transcribing", 45.2, "00:02:53")


def test_diarization_is_reported_as_its_own_phase():
    """It runs after decoding and costs about as much again, so a job at "100%" of
    transcription is only half done if speakers were asked for."""
    log = "\r   99.9%  00:10/00:10  1.9x  eta 00:00:00  \r   diarizing  37%   "
    assert jobs.read_progress(log) == ("diarizing", 37.0, None)


def test_a_job_that_has_not_printed_yet_is_starting():
    assert jobs.read_progress("") == ("starting", None, None)


def test_a_pid_that_is_not_our_transcriber_is_never_trusted():
    assert jobs.is_our_transcriber(None, "/a.flac") is False
    assert jobs.is_our_transcriber(os.getpid(), "/a.flac") is False
    assert jobs.is_our_transcriber(2 ** 22 + 7, "/a.flac") is False


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_returns_promptly_and_the_job_finishes_on_its_own(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(jobs, "is_our_transcriber", lambda pid, a: _alive(pid))
    began = time.monotonic()
    job = jobs.start(audio, tmp_path, command=job_command(tmp_path, SUCCEEDS, audio))
    assert time.monotonic() - began < 3.0  # returned while the "decode" still had work left

    assert wait_until(lambda: jobs.get(tmp_path, job.id).state == "done")
    done = jobs.get(tmp_path, job.id)
    assert done.percent == 100.0
    assert done.timeline.endswith("meeting.timeline.json")


def test_progress_is_visible_while_it_runs(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(jobs, "is_our_transcriber", lambda pid, a: _alive(pid))
    job = jobs.start(audio, tmp_path, command=job_command(tmp_path, HANGS, audio))
    try:
        assert wait_until(lambda: jobs.get(tmp_path, job.id).percent == 20.0)
        running = jobs.get(tmp_path, job.id)
        assert running.state == "running"
        assert running.eta == "00:00:04"
    finally:
        os.kill(job.pid, signal.SIGKILL)


def test_a_crashing_transcriber_reports_failed_with_the_reason(tmp_path, audio, monkeypatch):
    monkeypatch.setattr(jobs, "is_our_transcriber", lambda pid, a: _alive(pid))
    job = jobs.start(audio, tmp_path, command=job_command(tmp_path, CRASHES, audio))
    assert wait_until(lambda: jobs.get(tmp_path, job.id).state == "failed")
    assert "could not decode" in jobs.get(tmp_path, job.id).detail


def test_a_killed_transcriber_is_interrupted_not_failed(tmp_path, audio, monkeypatch):
    """Killed, or the machine slept. That is different from failing -- restarting resumes
    long files from their last finished chunk -- so it must not read as an error."""
    monkeypatch.setattr(jobs, "is_our_transcriber", lambda pid, a: _alive(pid))
    job = jobs.start(audio, tmp_path, command=job_command(tmp_path, HANGS, audio))
    os.kill(job.pid, signal.SIGKILL)
    assert wait_until(lambda: jobs.get(tmp_path, job.id).state == "interrupted")
    assert "resumes" in jobs.get(tmp_path, job.id).detail


def test_only_one_job_runs_at_a_time(tmp_path, audio, monkeypatch):
    """A decode already uses every core. A second would slow both and finish neither sooner."""
    monkeypatch.setattr(jobs, "is_our_transcriber", lambda pid, a: _alive(pid))
    job = jobs.start(audio, tmp_path, command=job_command(tmp_path, HANGS, audio))
    try:
        with pytest.raises(jobs.JobError, match="One at a time"):
            jobs.start(audio, tmp_path, command=job_command(tmp_path, HANGS, audio))
    finally:
        os.kill(job.pid, signal.SIGKILL)


def test_a_missing_audio_file_is_refused_before_anything_starts(tmp_path):
    with pytest.raises(jobs.JobError, match="no audio file"):
        jobs.start(tmp_path / "nope.flac", tmp_path)


def test_an_unknown_job_id_is_reported(tmp_path):
    with pytest.raises(jobs.JobError, match="no transcription job"):
        jobs.get(tmp_path, "nope")


def test_the_real_command_runs_transcribe_py_with_the_requested_options(tmp_path, audio, monkeypatch):
    """Without a stand-in, the job must invoke the real pipeline with the flags asked for."""
    captured = {}

    class FakePopen:
        def __init__(self, invocation, **_kwargs):
            captured["invocation"] = invocation
            self.pid = 0

        def poll(self):
            return 0

    monkeypatch.setattr(jobs.subprocess, "Popen", FakePopen)
    jobs.start(audio, tmp_path, diarize=True, speakers=9, vocabulary="AyaData, Kwame")
    invocation = captured["invocation"]
    assert invocation[-1].endswith("transcribe.py") is False
    assert any(part.endswith("transcribe.py") for part in invocation)
    assert ["-d", "-s", "9"] == invocation[invocation.index("-d"):invocation.index("-d") + 3]
    assert "AyaData, Kwame" in invocation


def _alive(pid):
    """Stand-ins are not called transcribe.py, so tests check liveness directly."""
    if not pid:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[-1].split()[0]
    except OSError:
        return False
    return state != "Z"
