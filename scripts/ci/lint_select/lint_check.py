#!/usr/bin/env python3
"""Run generic linters first, then TorchTurbo-specific contracts.

The generic stage is delegated to Ruff, shellcheck, actionlint, and hadolint.
The custom stage only contains rules that need repository context: job graph
wiring, test selection, model mapping, and pytest markers.

Usage:
    lint_check.py                  run every check
    lint_check.py --only NAME      run one check (repeatable)
    lint_check.py --list           list check names

Exit codes: 0 clean, 1 problems found, 2 bad usage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable

import lint_common_check
import lint_turbo_check
from lint_common_check import Check, Problem

CHECKS = lint_common_check.CHECKS + lint_turbo_check.CHECKS
CHECK_BY_NAME = {check.name: check for check in CHECKS}
ALIASES = {
    "map": "model_map",
}


def _selected(names: list[str]) -> list[Check]:
    if not names:
        return list(CHECKS)
    resolved = [ALIASES.get(name, name) for name in names]
    unknown = [name for name in resolved if name not in CHECK_BY_NAME]
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)}")
    return [CHECK_BY_NAME[name] for name in resolved]


def _summary(rows: list[tuple[str, str, int]]) -> str:
    lines = ["## Lint", "", "| check | result | problems |", "|---|---|---:|"]
    for name, result, count in rows:
        lines.append(f"| `{name}` | {result} | {count} |")
    return "\n".join(lines) + "\n"


def run(checks: Iterable[Check], summary_path: str | None) -> int:
    started = time.monotonic()
    total = 0
    rows = []
    failed = []
    for check in checks:
        try:
            problems = check.function()
        except Exception as exc:
            problems = [Problem("<check>", f"check crashed: {type(exc).__name__}: {exc}")]
        total += len(problems)
        rows.append((check.name, "fail" if problems else "ok", len(problems)))
        if not problems:
            print(f"{check.name}: ok")
            continue
        failed.append(check.name)
        print(f"{check.name}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem.render()}")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                fields = [f"title={check.name}", f"file={problem.path}"]
                if problem.line:
                    fields.append(f"line={problem.line}")
                message = problem.message.replace("\n", " ")
                print(f"::error {','.join(fields)}::{message}")

    elapsed = time.monotonic() - started
    destination = summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with open(destination, "a", encoding="utf-8") as output:
            output.write(_summary(rows))

    if total:
        print(f"\nFAILED: {', '.join(failed)} ({total} problem(s), {elapsed:.1f}s)")
        return 1
    print(f"\nLint OK ({len(rows)} checks, {elapsed:.1f}s)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run TorchTurbo static checks")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run one check (repeatable); accepts the legacy 'map' alias")
    parser.add_argument("--list", action="store_true", help="list check names and exit")
    parser.add_argument("--summary", default=None,
                        help="write a Markdown summary to this file")
    args = parser.parse_args(argv)

    if args.list:
        for check in CHECKS:
            print(f"{check.name}\t{check.description}")
        return 0
    try:
        checks = _selected(args.only)
    except ValueError as exc:
        parser.error(str(exc))
    return run(checks, args.summary)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
