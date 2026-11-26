#!/usr/bin/env python3
"""Classify changed files into the CI change kind."""

from __future__ import annotations

import argparse
import sys
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable

from .cli import read_lines

DOC_DIRS = frozenset({"docs"})
DOC_NAMES = frozenset({"LICENSE", "NOTICE"})
DOC_SUFFIXES = frozenset({".adoc", ".md", ".rst"})
IMAGE_ROOT = "docker"


def normalize_repo_path(path: str) -> str:
    """Return a repository-relative path with POSIX separators."""
    return path.strip().replace("\\", "/").removeprefix("./")


def _parsed_path(path: str) -> PurePosixPath | None:
    normalized = normalize_repo_path(path)
    return PurePosixPath(normalized) if normalized else None


def is_docs_path(path: str) -> bool:
    """Return whether one repository-relative path is documentation-only."""
    parsed = _parsed_path(path)
    if parsed is None:
        return False
    in_docs = bool(parsed.parts) and parsed.parts[0] in DOC_DIRS
    return in_docs or parsed.name in DOC_NAMES or parsed.suffix.lower() in DOC_SUFFIXES


def is_dockerfile_path(path: str) -> bool:
    """Return whether one path is a Dockerfile that triggers an image build."""
    parsed = _parsed_path(path)
    if parsed is None or not parsed.parts:
        return False
    return parsed.parts[0] == IMAGE_ROOT and parsed.name.startswith("Dockerfile")


class ChangeKind(str, Enum):
    """Change kind selected from repository changes."""

    CODE = "code"
    DOCS_ONLY = "docs_only"
    IMAGE_ONLY = "image_only"
    MIXED = "mixed"


def classify_changes(paths: Iterable[str]) -> ChangeKind:
    """Select docs, code, image-only or mixed changes from paths."""
    values = [path.strip() for path in paths if path.strip()]
    if not values:
        return ChangeKind.CODE

    has_image = any(is_dockerfile_path(value) for value in values)
    has_code = any(not is_dockerfile_path(value) and not is_docs_path(value)
                   for value in values)
    if has_image:
        return ChangeKind.MIXED if has_code else ChangeKind.IMAGE_ONLY
    return ChangeKind.CODE if has_code else ChangeKind.DOCS_ONLY


def main(argv: list[str] | None = None) -> int:
    """Print one change kind for workflow consumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed-files", default="-",
                        help="newline-separated changed paths; '-' reads stdin")
    args = parser.parse_args(argv)
    try:
        paths = read_lines(args.changed_files)
    except OSError as exc:
        print(f"cannot read changed-file list: {exc}", file=sys.stderr)
        return 2
    print(classify_changes(paths).value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
