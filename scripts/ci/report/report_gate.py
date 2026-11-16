#!/usr/bin/env python3
"""Write CI queue/run timing to the Job Summary.

Timing is observability only. Pass/fail decisions belong to gate.py, which
compares the Plan against the uploaded JobResult artifacts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _summary(markdown: str) -> None:
    path = _env("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as output:
            output.write(markdown)
    else:
        print(markdown)


def timing() -> int:
    """Report queue and execution time without changing the CI verdict."""
    repo, run_id = _env("GITHUB_REPOSITORY"), _env("GITHUB_RUN_ID")
    if not (repo and run_id):
        print("::warning::missing GITHUB_REPOSITORY / GITHUB_RUN_ID; skip timing")
        return 0

    proc = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/runs/{run_id}",
            "--jq",
            r'"\(.created_at) \(.run_started_at)"',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    stamps = proc.stdout.strip()
    if proc.returncode != 0 or not stamps:
        print("::warning::cannot read run timestamps (actions:read?)")
        return 0

    def parse(value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    created_text, started_text = stamps.split()
    created, started = parse(created_text), parse(started_text)
    now = datetime.now(timezone.utc)
    _summary(
        "\n| Timing | Seconds |\n|---|---:|\n"
        f"| queue | {int((started - created).total_seconds())} |\n"
        f"| run | {int((now - started).total_seconds())} |\n"
        f"| end-to-end | {int((now - created).total_seconds())} |\n"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["timing"])
    args = parser.parse_args(argv)
    if args.command == "timing":
        return timing()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
