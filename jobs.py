"""Transcription as a background job.

A synchronous transcription call has to stay open for the whole decode -- about an hour for
a two-hour meeting -- and no request/response protocol survives that. So the MCP door used
to cap transcription at fifteen minutes and send anything longer to a terminal, which broke
"record this meeting and write it up" at its second step for every real meeting.

A job returns at once with an id; the decode runs as a detached `transcribe.py` process;
`status` reads its progress from the log. Nothing blocks, so nothing needs a cap.

Same shape as a recording session in capture.py, on purpose: a detached process nobody waits
on, state on disk, and a pid that is checked against /proc before anyone trusts it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ".jobs"
START_CONFIRM_SECONDS = 2.0

# transcribe.py's own progress line, and diarize.py's. Parsed rather than re-implemented:
# the log is the process's own account of where it is.
DECODE_PROGRESS = re.compile(r"(\d+\.\d)%\s+\S+/\S+\s+[\d.]+x\s+eta\s+(\S+)")
DIARIZE_PROGRESS = re.compile(r"diarizing\s+(\d+)%")


class JobError(Exception):
    """Something the caller can act on, phrased for whoever is reading."""


@dataclass
class Job:
    id: str
    audio: str
    output_dir: str
    markdown: str
    timeline: str
    log: str
    started_at: str
    pid: int | None = None
    arguments: list[str] = field(default_factory=list)
    state: str = "running"  # running | done | failed | interrupted
    phase: str = ""
    percent: float | None = None
    eta: str | None = None
    detail: str = ""


def _jobs_dir(output_dir: Path) -> Path:
    path = Path(output_dir) / JOBS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_file(output_dir: Path, job_id: str) -> Path:
    return _jobs_dir(output_dir) / f"{job_id}.json"


def _save(job: Job) -> None:
    _job_file(Path(job.output_dir), job.id).write_text(
        json.dumps(asdict(job), indent=2), encoding="utf-8"
    )


def _load(path: Path) -> Job | None:
    try:
        return Job(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def is_our_transcriber(pid: int | None, audio: str) -> bool:
    """Is this pid still the transcribe.py working on this exact file?

    A pid from before a reboot can belong to anything; a job must never report "running"
    because some unrelated process inherited the number.
    """
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    if stat.rsplit(")", 1)[-1].split()[0] == "Z":
        return False
    args = [a.decode(errors="replace") for a in cmdline]
    return any(a.endswith("transcribe.py") for a in args) and audio in args


def _reap(pid: int | None) -> None:
    if not pid:
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def read_progress(log_text: str) -> tuple[str, float | None, str | None]:
    """Where the job is, from its own log: (phase, percent, eta).

    Progress lines are written with carriage returns, so the whole log is one long line of
    overwritten updates; the LAST match of each pattern is the current one.
    """
    diarize = DIARIZE_PROGRESS.findall(log_text)
    decode = DECODE_PROGRESS.findall(log_text)
    # Diarization runs after decoding, so its presence means decoding is finished.
    if diarize:
        return "diarizing", float(diarize[-1]), None
    if decode:
        percent, eta = decode[-1]
        return "transcribing", float(percent), eta
    if "Loading model" in log_text:
        return "loading model", None, None
    return "starting", None, None


def _log_tail(path: Path, limit: int = 400) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line for line in text.replace("\r", "\n").splitlines() if line.strip()]
    return "\n".join(lines[-6:])[-limit:]


def refresh(job: Job) -> Job:
    """Bring a job's state up to date from its process and its log."""
    if job.state in ("done", "failed"):
        return job

    log = Path(job.log)
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    job.phase, job.percent, job.eta = read_progress(text)

    if is_our_transcriber(job.pid, job.audio):
        job.state = "running"
        return job

    _reap(job.pid)
    if Path(job.timeline).exists() and "Done." in text:
        job.state, job.phase, job.percent, job.eta = "done", "done", 100.0, None
        job.detail = ""
    elif "Done." in text or "✗" in text or "Traceback" in text:
        job.state = "failed"
        job.detail = _log_tail(log)
    else:
        # Gone without finishing or failing: killed, or the machine slept or rebooted.
        job.state = "interrupted"
        job.detail = ("the transcriber stopped before finishing. Starting the same job "
                      "again resumes from the last completed chunk for long files.")
    _save(job)
    return job


def current(output_dir: Path) -> Job | None:
    """The most recent job that has not finished, refreshed."""
    folder = Path(output_dir) / JOBS_DIR
    if not folder.is_dir():
        return None
    jobs = [j for j in (_load(p) for p in sorted(folder.glob("*.json"))) if j]
    for job in reversed(jobs):
        refreshed = refresh(job)
        if refreshed.state == "running":
            return refreshed
    return None


def get(output_dir: Path, job_id: str) -> Job:
    job = _load(_job_file(output_dir, job_id))
    if job is None:
        raise JobError(f"no transcription job {job_id}")
    return refresh(job)


def latest(output_dir: Path) -> Job | None:
    folder = Path(output_dir) / JOBS_DIR
    if not folder.is_dir():
        return None
    jobs = [j for j in (_load(p) for p in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime)) if j]
    return refresh(jobs[-1]) if jobs else None


def start(audio: Path, output_dir: Path, diarize: bool = False, speakers: int = 0,
          language: str | None = None, vocabulary: str | None = None, force: bool = False,
          command: list[str] | None = None) -> Job:
    """Launch a transcription in the background and return at once.

    One at a time. A decode already pins every core this machine has; a second would halve
    the speed of both and finish neither sooner.

    `command` exists so tests can substitute a stand-in for transcribe.py.
    """
    audio = Path(audio).resolve()
    if not audio.exists():
        raise JobError(f"no audio file at {audio}")

    running = current(output_dir)
    if running is not None:
        raise JobError(
            f"job {running.id} is still transcribing {Path(running.audio).name}"
            + (f" ({running.percent:.0f}%)" if running.percent is not None else "")
            + ". One at a time: a second would slow both down."
        )

    arguments = [str(audio), "-o", str(Path(output_dir).resolve())]
    if diarize:
        arguments.append("-d")
    if speakers:
        arguments += ["-s", str(speakers)]
    if language:
        arguments += ["-l", language]
    if vocabulary:
        arguments += ["-p", vocabulary]
    if force:
        arguments.append("-f")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^a-z0-9]+", "-", audio.stem.lower()).strip("-")[:40]
    folder = _jobs_dir(output_dir)
    log = folder / f"{job_id}.log"
    stem = audio.stem
    out = Path(output_dir).resolve()

    invocation = command or [sys.executable, "-u", str(ROOT / "transcribe.py"), *arguments]
    with open(log, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            invocation, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )

    job = Job(
        id=job_id, audio=str(audio), output_dir=str(out),
        markdown=str(out / f"{stem}.md"), timeline=str(out / f"{stem}.timeline.json"),
        log=str(log), pid=process.pid, arguments=arguments,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    _save(job)

    # A bad argument or a missing dependency makes transcribe.py exit at once. Say so now,
    # rather than letting the caller poll a job that was dead before its id was returned.
    deadline = time.monotonic() + START_CONFIRM_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return refresh(job)
        time.sleep(0.1)
    return refresh(job)
