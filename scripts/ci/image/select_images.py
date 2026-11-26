"""Select container images affected by CI changes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..tools.changed_files import is_dockerfile_path, normalize_repo_path
from .image_config import ImageError, ImageSpec, load_specs


class ImageSelectionError(ImageError):
    """Raised when an image selection request is invalid."""


def select_images(
    root: Path,
    *,
    image_ids: str = "",
    changed_files: Iterable[str] | None = None,
) -> list[ImageSpec]:
    """Select container images by ID or by added/modified Dockerfile paths."""
    if bool(image_ids) == (changed_files is not None):
        raise ImageSelectionError("use exactly one of image IDs or changed files")
    specs = load_specs(root)
    if image_ids.strip().lower() == "all":
        return specs
    if image_ids:
        requested = tuple(dict.fromkeys(
            part for part in re.split(r"[,\s]+", image_ids.strip()) if part
        ))
        if not requested:
            raise ImageSelectionError("image selection is empty; use 'all' or image IDs")
        by_id = {spec.image_id: spec for spec in specs}
        unknown = [image_id for image_id in requested if image_id not in by_id]
        if unknown:
            raise ImageSelectionError(f"unknown image ID(s): {', '.join(unknown)}")
        return sorted((by_id[image_id] for image_id in requested), key=lambda spec: spec.image_id)

    changed = {normalize_repo_path(value) for value in changed_files or ()}
    changed.discard("")
    by_dockerfile = {spec.dockerfile: spec for spec in specs}
    unmatched = sorted(path for path in changed
                       if is_dockerfile_path(path) and path not in by_dockerfile)
    if unmatched:
        raise ImageSelectionError(
            "changed Dockerfiles are not mapped to resources.images.*.build.dockerfile "
            "in scripts/ci/config/ci.yml: "
            + ", ".join(unmatched)
        )
    selected = {by_dockerfile[path].image_id: by_dockerfile[path]
                for path in changed if path in by_dockerfile}
    return sorted(selected.values(), key=lambda spec: spec.image_id)
