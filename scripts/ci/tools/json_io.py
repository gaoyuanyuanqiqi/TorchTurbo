"""JSON serialization, decoding, and atomic file IO for CI artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import types
from dataclasses import MISSING, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints


class JsonDecodeError(ValueError):
    """Raised when JSON cannot be decoded into the requested contract type."""


def canonical_json(value: object) -> str:
    """Return deterministic JSON for hashing and artifact comparison."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_digest(value: object) -> str:
    """Return a sha256 digest over deterministic JSON content."""
    payload = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    """Write JSON atomically so readers never observe a partial artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def decode(cls: type, value: object, path: str = "value") -> Any:
    """Decode JSON data into a dataclass using its runtime type annotations."""
    return _decode(cls, value, path)


def _fail(path: str, message: str) -> None:
    raise JsonDecodeError(f"{path} {message}")


@lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    return get_type_hints(cls)


def _decode(tp: Any, value: Any, path: str) -> Any:
    if tp is str:
        if not isinstance(value, str):
            _fail(path, "must be a string")
        return value
    if tp is int:
        if not isinstance(value, int) or isinstance(value, bool):
            _fail(path, "must be an integer")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            _fail(path, "must be a boolean")
        return value
    origin = get_origin(tp)
    args = get_args(tp)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for candidate in (item for item in args if item is not type(None)):
            try:
                return _decode(candidate, value, path)
            except (JsonDecodeError, TypeError, ValueError):
                continue
        _fail(path, "has an invalid value")
    if origin is tuple:
        if not isinstance(value, list):
            _fail(path, "must be a list")
        item_type = args[0] if args else Any
        return tuple(
            _decode(item_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if origin is dict:
        if not isinstance(value, dict):
            _fail(path, "must be a mapping")
        return value
    if isinstance(tp, type) and is_dataclass(tp):
        return _decode_dataclass(tp, value, path)
    return value


def _decode_dataclass(cls: type, value: Any, path: str) -> Any:
    if not isinstance(value, dict):
        _fail(path, "must be a mapping")
    cls_fields = fields(cls)
    allowed = {item.name for item in cls_fields}
    defaults = {
        item.name for item in cls_fields
        if item.default is not MISSING or item.default_factory is not MISSING
    }
    missing = sorted(allowed - set(value) - defaults)
    unknown = sorted(set(value) - allowed)
    if missing:
        _fail(path, f"is missing fields: {', '.join(missing)}")
    if unknown:
        _fail(path, f"has unknown fields: {', '.join(unknown)}")

    hints = _hints(cls)
    try:
        return cls(**{
            item.name: _decode(hints[item.name], value[item.name], f"{path}.{item.name}")
            for item in cls_fields
            if item.name in value
        })
    except (KeyError, TypeError) as exc:
        raise JsonDecodeError(f"{path} cannot be decoded as {cls.__name__}") from exc
