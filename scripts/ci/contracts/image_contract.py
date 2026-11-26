"""Image contracts shared by the execution and image domains."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import Contract, ContractError

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IMAGE_KINDS = frozenset({"host", "container"})


def _require_non_empty_text(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field} must be a non-empty string")


def _require_sha256_digest(value: object, field: str) -> None:
    _require_non_empty_text(value, field)
    if not _SHA256_DIGEST.fullmatch(value):
        raise ContractError(f"{field} must be sha256:<64 lowercase hex>")


@dataclass(frozen=True)
class ResolvedImage(Contract):
    """An immutable image reference ready for the execution domain."""

    image_id: str  # Logical image ID, for example npu-a3-cann90.
    target: str    # Hardware target, for example npu-a3.
    kind: str      # Runtime kind: host or container.
    ref: str       # Executable ref: repository + sha256 digest for containers.

    def __post_init__(self) -> None:
        _require_non_empty_text(self.image_id, "image_id")
        _require_non_empty_text(self.target, "target")
        _require_non_empty_text(self.ref, "ref")
        if self.kind not in _IMAGE_KINDS:
            raise ContractError(f"kind must be host or container: {self.kind!r}")
        if self.kind == "container":
            repository, separator, digest = self.ref.rpartition("@")
            if not separator or not repository:
                raise ContractError(
                    f"container ref must be repository@sha256:<64 hex>: {self.ref!r}"
                )
            _require_sha256_digest(digest, "ref digest")

    @property
    def digest(self) -> str:
        """Return the OCI digest encoded in ref; host images have none."""
        return "" if self.kind == "host" else self.ref.rpartition("@")[2]


@dataclass(frozen=True)
class ImageBuildResult(Contract):
    """Facts produced by building or reusing a container image."""

    image: ResolvedImage  # Immutable image produced by the build.
    content_hash: str  # Full sha256 of Dockerfile, platform, and revision.
    built: bool  # True when the image was built; False when reused.
    elapsed_seconds: int = 0  # Time spent in the actual build.
    alias: str = ""  # Stable alias touched by promotion.

    def __post_init__(self) -> None:
        if self.image.kind != "container":
            raise ContractError("ImageBuildResult requires a container image")
        _require_sha256_digest(self.content_hash, "content_hash")
        if not isinstance(self.built, bool):
            raise ContractError("built must be a boolean")
