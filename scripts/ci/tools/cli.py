"""Shared helpers for CI command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping

from .. import repository_root
from .json_io import write_json


def common_parser() -> argparse.ArgumentParser:
    """Return a parser for options shared by CI command-line entry points."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT", ""))
    return parser


def repository_path(value: str | None) -> Path:
    return Path(value).resolve() if value else repository_root()


def read_lines(path: str) -> list[str]:
    if path == "-":
        return sys.stdin.read().splitlines()
    return Path(path).read_text(encoding="utf-8").splitlines()


def write_json_output(path: str, value: object, *, emit_stdout: bool = True) -> None:
    if path:
        write_json(Path(path), value)
    elif emit_stdout:
        print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def append_github_output(values: Mapping[str, object], path: str = "") -> None:
    path = path or os.environ.get("GITHUB_OUTPUT", "")
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")
