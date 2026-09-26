"""Layer 1 -- capture. Getting the audio in the first place.

Recording is a SESSION, not a long-running call: it ends when a person says stop, which no
request/response call can model at any duration. So `start` and `stop` are both instant,
the recording itself is a detached ffmpeg process nobody waits on, and the state between
them lives on disk. That is what lets the assistant say "record this meeting" without a
chat turn blocking for two hours.

**Raw PCM, with the session JSON as its header.** A WAV file carries its own length in a
header that is only finalised when the writer exits cleanly -- kill the process and the
header can claim zero samples, leaving a two-hour meeting that decodes as silence. Raw PCM
has no header to corrupt: every byte on disk is audio, and the sample rate and channel
count needed to read it are in the session file. Crash-safe by construction rather than by
recovery code. It becomes FLAC -- lossless, about half the size -- only on a clean stop.

What this layer does NOT buy is accuracy. One microphone in a room is one microphone in a
room: nine people still go into it mixed, and diarization still has to pull them apart.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# 48 kHz, not 16. Whisper resamples to 16 kHz anyway, so 16 k would lose nothing TODAY --
# but it would foreclose re-running with a better model later, or simply listening back.
DEFAULT_RATE = 48000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le

# In-progress audio lives in a hidden directory under the audio dir, with an extension
# transcribe.py does not recognise. find_audio() globs recursively, so a recording in
# progress would otherwise be picked up and transcribed half-finished.
STATE_DIR = ".recording"
PART_SUFFIX = ".pcm.part"

STOP_GRACE_SECONDS = 10
START_CONFIRM_SECONDS = 3.0

# Below this the recording is treated as silent. A muted or wrong microphone is the classic
# way to lose a meeting, and it is invisible until someone tries to transcribe it.
SILENT_BELOW_DB = -60.0


class CaptureError(Exception):
    """Something the caller can act on, phrased for whoever is reading."""


@dataclass
class Session:
    id: str
    source: str
    started_at: str
    rate: int
    channels: int
    part: str
    pid: int | None = None
    consent: str | None = None
    state: str = "recording"  # recording | orphaned | stopped
    audio: str | None = None
    duration_seconds: float | None = None
    max_volume_db: float | None = None
    stopped_at: str | None = None
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Devices
# ---------------------------------------------------------------------------

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def list_sources() -> list[dict]:
    """Every capture source PipeWire exposes, microphones and output monitors alike.

    A `.monitor` source is what a speaker is playing -- on a call, everyone but you. It is
    listed so it can be chosen, but only the microphone is the default.
    """
    proc = _run(["pactl", "list", "short", "sources"])
    if proc.returncode != 0:
        raise CaptureError(f"could not list audio sources: {proc.stderr.strip()[:200]}")
    default = default_source()
    out: list[dict] = []
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        out.append({
            "name": name,
            "kind": "monitor" if name.endswith(".monitor") else "microphone",
            "default": name == default,
        })
    return out


def default_source() -> str:
    proc = _run(["pactl", "get-default-source"])
    name = proc.stdout.strip()
    if proc.returncode != 0 or not name:
        raise CaptureError("no default audio source. Pass one explicitly; see `./record sources`.")
    return name


def pulse_input(source: str) -> list[str]:
    """PipeWire, through its Pulse compatibility layer. ffmpeg is already a requirement."""
    return ["-f", "pulse", "-i", source]


# ---------------------------------------------------------------------------
# Session files
# ---------------------------------------------------------------------------

def _state_dir(audio_dir: Path) -> Path:
    path = Path(audio_dir) / STATE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _session_file(audio_dir: Path, session_id: str) -> Path:
    return _state_dir(audio_dir) / f"{session_id}.json"


def _save(audio_dir: Path, session: Session) -> None:
    _session_file(audio_dir, session.id).write_text(
        json.dumps(asdict(session), indent=2), encoding="utf-8"
    )


def _load(path: Path) -> Session | None:
    try:
        return Session(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def session_id(name: str | None = None, now: datetime | None = None) -> str:
    """A filename-safe id: the time it started, or a slug of a name you gave it."""
    if name:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        if slug:
            return slug
    return (now or datetime.now()).strftime("%Y-%m-%d-%H%M%S")


# ---------------------------------------------------------------------------
# Process safety
# ---------------------------------------------------------------------------

def is_our_recorder(pid: int | None, part: str) -> bool:
    """Is this pid still the ffmpeg writing this exact file?

    Checked before every signal. A pid recorded before a reboot can belong to anything by
    now, and interrupting an unrelated process because a stale session file named it would
    be a bug with no upper bound on its damage.
    """
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    # Field 3 of /proc/<pid>/stat is the state; "Z" is a zombie that has already exited.
    if stat.rsplit(")", 1)[-1].split()[0] == "Z":
        return False
    args = [a.decode(errors="replace") for a in cmdline]
    return bool(args) and Path(args[0]).name == "ffmpeg" and part in args


def _reap(pid: int | None) -> None:
    """Collect the exit status if the recorder is our own child, so it does not linger as a
    zombie inside a long-lived parent such as the MCP server."""
    if not pid:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


# ---------------------------------------------------------------------------
# Start, status, stop
# ---------------------------------------------------------------------------

def active(audio_dir: Path) -> Session | None:
    """The unfinished session, if there is one.

    A session whose recorder has vanished -- a crash, a reboot, a killed terminal -- comes
    back as `orphaned` rather than disappearing. Its audio is still on disk, and `stop` will
    recover it.
    """
    state = Path(audio_dir) / STATE_DIR
    if not state.is_dir():
        return None
    sessions = [s for s in (_load(p) for p in sorted(state.glob("*.json"))) if s]
    unfinished = [s for s in sessions if s.state in ("recording", "orphaned")]
    if not unfinished:
        return None
    current = unfinished[-1]
    if current.state == "recording" and not is_our_recorder(current.pid, current.part):
        _reap(current.pid)
        current.state = "orphaned"
        _save(audio_dir, current)
    return current


def start(audio_dir: Path, source: str | None = None, consent: str | None = None,
          name: str | None = None, rate: int = DEFAULT_RATE,
          input_args: list[str] | None = None) -> Session:
    """Begin recording in the background, and return as soon as audio is known to be flowing.

    `input_args` exists so the tests can substitute a generated tone for a real device.
    """
    current = active(audio_dir)
    if current is not None:
        verb = "is still recording" if current.state == "recording" else "was interrupted"
        raise CaptureError(
            f"session {current.id} {verb}. Stop it first: `./record stop`"
        )

    chosen = source or (None if input_args else default_source())
    sid = session_id(name)
    state = _state_dir(audio_dir)
    part = state / f"{sid}{PART_SUFFIX}"
    log = state / f"{sid}.log"
    if part.exists() or (Path(audio_dir) / f"{sid}.flac").exists():
        raise CaptureError(f"a recording named {sid} already exists. Choose another name.")

    command = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        *(input_args or pulse_input(chosen)),
        "-ac", str(CHANNELS), "-ar", str(rate),
        # Without this ffmpeg buffers output in 256 KiB blocks -- about 2.7 s at 48 kHz mono,
        # over 8 s at 16 kHz. The file then grows in jumps, and a crash or SIGKILL loses
        # whatever sat in the buffer, which is exactly the loss raw PCM was chosen to avoid.
        # Measured before and after, not assumed.
        "-flush_packets", "1",
        "-f", "s16le", str(part),
    ]
    with open(log, "w", encoding="utf-8") as log_handle:
        # A new session puts the recorder outside the terminal's process group, so Ctrl-C
        # in the terminal does not reach it directly -- stop() interrupts it deliberately,
        # which is what gives ffmpeg the chance to flush.
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=log_handle, start_new_session=True,
        )

    session = Session(
        id=sid, source=chosen or "(test input)", rate=rate, channels=CHANNELS,
        part=str(part), pid=process.pid, consent=consent,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )

    # Confirm audio is actually arriving before claiming to record. A wrong device name or
    # a busy source makes ffmpeg exit immediately, and a "recording" that captured nothing
    # is only discovered when the meeting is over.
    deadline = time.monotonic() + START_CONFIRM_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = log.read_text(encoding="utf-8").strip()[:300]
            part.unlink(missing_ok=True)
            raise CaptureError(f"recorder exited immediately: {detail or 'no detail'}")
        if part.exists() and part.stat().st_size > 0:
            break
        time.sleep(0.1)

    if not consent:
        session.warnings.append("no consent recorded for this session")
    _save(audio_dir, session)
    return session


def _interrupt(session: Session) -> None:
    """Ask ffmpeg to finish, then insist. Raw PCM means even the insisting loses nothing."""
    if not is_our_recorder(session.pid, session.part):
        return
    os.kill(session.pid, signal.SIGINT)
    deadline = time.monotonic() + STOP_GRACE_SECONDS
    while time.monotonic() < deadline and is_our_recorder(session.pid, session.part):
        _reap(session.pid)
        time.sleep(0.1)
    if is_our_recorder(session.pid, session.part):
        os.kill(session.pid, signal.SIGKILL)
        time.sleep(0.2)
    _reap(session.pid)


def _parse_max_volume(stderr: str) -> float | None:
    match = re.search(r"max_volume:\s*(-?[\d.]+|-inf)\s*dB", stderr)
    if not match:
        return None
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


def stop(audio_dir: Path, session_id_: str | None = None) -> Session:
    """End a session: stop the recorder, write lossless FLAC, check it is not silent.

    Also the recovery path. An orphaned session -- recorder gone, audio still on disk -- is
    finished exactly the same way, which is the point of recording raw PCM.
    """
    if session_id_:
        session = _load(_session_file(audio_dir, session_id_))
        if session is None:
            raise CaptureError(f"no session {session_id_}")
    else:
        session = active(audio_dir)
        if session is None:
            raise CaptureError("nothing is recording")

    recovered = session.state == "orphaned" or not is_our_recorder(session.pid, session.part)
    _interrupt(session)

    part = Path(session.part)
    frame = session.channels * BYTES_PER_SAMPLE
    size = part.stat().st_size if part.exists() else 0
    if size < frame:
        _session_file(audio_dir, session.id).unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        raise CaptureError(f"session {session.id} captured no audio")

    final = Path(audio_dir) / f"{session.id}.flac"
    # One pass: encode to FLAC and measure loudness together, so a two-hour recording is
    # read once rather than twice.
    proc = _run([
        "ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-y",
        "-f", "s16le", "-ar", str(session.rate), "-ac", str(session.channels),
        "-i", str(part), "-af", "volumedetect", "-c:a", "flac", str(final),
    ])
    if proc.returncode != 0 or not final.exists():
        # Keep the raw audio. A failed encode must never be the thing that loses a meeting.
        raise CaptureError(
            f"could not write {final.name}; the raw audio is kept at {part}. "
            f"{proc.stderr.strip()[-200:]}"
        )

    session.audio = str(final)
    session.duration_seconds = round(size / (session.rate * frame), 2)
    session.max_volume_db = _parse_max_volume(proc.stderr)
    session.stopped_at = datetime.now().isoformat(timespec="seconds")
    session.state = "stopped"
    if recovered:
        session.warnings.append("recorder had already stopped; audio recovered from disk")
    if session.max_volume_db is not None and session.max_volume_db < SILENT_BELOW_DB:
        session.warnings.append(
            f"recording is silent (peak {session.max_volume_db} dB). "
            f"Check the microphone is not muted and is the right source."
        )

    # The finished session file sits beside the audio it describes; the in-progress files go.
    (Path(audio_dir) / f"{session.id}.session.json").write_text(
        json.dumps(asdict(session), indent=2), encoding="utf-8"
    )
    part.unlink(missing_ok=True)
    (Path(session.part).with_suffix("").with_suffix(".log")).unlink(missing_ok=True)
    (_state_dir(audio_dir) / f"{session.id}.log").unlink(missing_ok=True)
    _session_file(audio_dir, session.id).unlink(missing_ok=True)
    return session


def elapsed_seconds(session: Session) -> float:
    """How much audio is on disk, from its size -- the recorder's own truth, not the clock."""
    part = Path(session.part)
    if not part.exists():
        return 0.0
    return part.stat().st_size / (session.rate * session.channels * BYTES_PER_SAMPLE)
