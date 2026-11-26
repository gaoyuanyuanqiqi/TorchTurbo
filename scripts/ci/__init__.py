"""TorchTurbo CI tooling package."""

from __future__ import annotations

from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """Return the repository root containing this package."""
    anchor = (start or Path(__file__)).resolve()
    candidates = (anchor, *anchor.parents) if anchor.is_dir() else anchor.parents
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError(f"cannot find repository root from {anchor}")
