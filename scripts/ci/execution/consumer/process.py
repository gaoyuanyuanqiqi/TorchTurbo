"""Run child processes with live logs and deadline enforcement."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _terminate(process: subprocess.Popen[str], sig: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_stream(command: list[str], *, cwd: Path, environment: dict[str, str], log_path: Path, timeout_seconds: int) -> tuple[int, bool]:
    """Run a command, stream its output, and return ``(exit_code, timed_out)``."""
    print("+ " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = threading.Event()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

        def terminate() -> None:
            timed_out.set()
            _terminate(process, signal.SIGTERM)
            timer = threading.Timer(10, lambda: _terminate(process, signal.SIGKILL))
            timer.daemon = True
            timer.start()

        timer = threading.Timer(timeout_seconds, terminate)
        timer.daemon = True
        timer.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            rc = process.wait()
        except KeyboardInterrupt:
            _terminate(process, signal.SIGTERM)
            process.wait()
            raise
        finally:
            timer.cancel()
    return (124, True) if timed_out.is_set() else (rc, False)


def timeout_budget(deadline: float, requested_seconds: int) -> tuple[int, bool]:
    """Return the timeout allowed by both the task and remaining job budget."""
    remaining = max(0, int(deadline - time.monotonic()))
    return min(requested_seconds, remaining), remaining < requested_seconds


CLEANUP_TIMEOUT_SECONDS = 20
TRAINING_OUTPUT_DIR = Path("/tmp/tblf-sft")
TRAIN_PROCS = "run_lf\\.py|run_swift\\.py|torchrun|deepspeed"
TMP_PATTERNS = ("tt_*.yaml", "tt_pytest_files.txt", "train.log", "after.txt", "pytest.log", "pytest.xml", "turbo.log")


def _cleanup_run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, timeout=CLEANUP_TIMEOUT_SECONDS)


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


@contextmanager
def _cleanup_step(lines: list[str], name: str) -> Iterator[None]:
    try:
        yield
    except (OSError, subprocess.SubprocessError) as exc:
        lines.append(f"::warning::{name} cleanup failed: {exc}")


def cleanup_environment(log_path: Path) -> None:
    """Best-effort cleanup after an NPU job, including failed or timed out runs."""
    lines = ["=== env cleanup ==="]
    with _cleanup_step(lines, "process"):
        left = _cleanup_run(["pgrep", "-af", TRAIN_PROCS])
        if left.returncode == 0 and left.stdout.strip():
            lines += ["--- leftover processes ---", left.stdout.strip()]
            _cleanup_run(["pkill", "-9", "-f", TRAIN_PROCS])
        else:
            lines.append("no leftover training process")

    with _cleanup_step(lines, "npu-smi"):
        if shutil.which("npu-smi"):
            mem = _cleanup_run(["npu-smi", "info", "-t", "memory"])
            if mem.stdout:
                lines += ["--- npu-smi memory ---", *mem.stdout.splitlines()[:20]]

    with _cleanup_step(lines, "temporary files"):
        for pattern in TMP_PATTERNS:
            for path in Path("/tmp").glob(pattern):
                _remove(path)

    with _cleanup_step(lines, "training output"):
        if TRAINING_OUTPUT_DIR.is_dir():
            if (TRAINING_OUTPUT_DIR / ".tt-complete").is_file():
                lines.append(f"checkpoint complete marker found, keeping {TRAINING_OUTPUT_DIR}")
            else:
                lines.append(f"no completeness marker, clearing partial output in {TRAINING_OUTPUT_DIR}")
                for child in TRAINING_OUTPUT_DIR.iterdir():
                    _remove(child)
    lines.append("=== env cleanup done ===")

    text = "\n".join(lines) + "\n"
    print(text, end="")
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(text)
    except OSError as exc:
        print(f"::warning::could not write cleanup log: {exc}", file=sys.stderr)
