"""Resolve model and dataset inputs for an integration case."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

CI_ROOT_ENV = "CI_ASSET_ROOT"
DOWNLOAD_ENV = "CI_MODEL_DATASET_DOWNLOAD"
SOURCES = {"local", "ci", "huggingface", "modelscope"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_MODEL_WEIGHTS = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


class ModelDatasetError(ValueError):
    """Raised when a model, tokenizer, or dataset cannot be resolved."""


def allow_download_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether a job may download missing remote inputs."""
    values = os.environ if environ is None else environ
    override = values.get(DOWNLOAD_ENV)
    if override is not None:
        return override.strip().lower() in _TRUE_VALUES
    return values.get("GITHUB_EVENT_NAME") in {"schedule", "workflow_dispatch"}


def resolve_model_dataset_args(
    args: Mapping[str, Any],
    *,
    framework: str,
    root: Path,
    allow_download: bool = False,
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, str], ...]]:
    """Return a copy of ``args`` with model and dataset paths made local."""
    if framework not in {"swift", "llamafactory"}:
        raise ModelDatasetError(f"unsupported integration framework {framework!r}")
    values = os.environ if environ is None else environ
    resolved = copy.deepcopy(dict(args))
    evidence: list[dict[str, str]] = []

    fields = (
        ("model" if framework == "swift" else "model_name_or_path", "model"),
        ("tokenizer", "tokenizer"),
        ("dataset" if framework == "swift" else "dataset_dir", "dataset"),
    )
    for field, role in fields:
        if field in resolved:
            resolved[field] = _resolve_value(
                resolved[field],
                field=field,
                role=role,
                root=root,
                allow_download=allow_download,
                environ=values,
                evidence=evidence,
            )
    return resolved, tuple(evidence)


def _resolve_value(
    value: Any, *, field: str, role: str, root: Path,
    allow_download: bool, environ: Mapping[str, str],
    evidence: list[dict[str, str]],
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        errors: list[str] = []
        candidates = _candidate_specs(value, field=field)
        for index, candidate in enumerate(candidates):
            try:
                path, source, revision = _resolve_spec(
                    candidate,
                    field=field,
                    role=role,
                    root=root,
                    allow_download=allow_download,
                    environ=environ,
                )
                _validate(path, role=role, spec=candidate, field=field)
            except ModelDatasetError as exc:
                errors.append(str(exc))
                continue
            evidence.append({
                "field": field,
                "source": source,
                "path": str(path),
                **({"revision": revision} if revision else {}),
                **({"fallback": "true"} if index else {}),
            })
            return str(path)
        raise ModelDatasetError(
            f"{field}: all configured sources failed: " + " | ".join(errors)
        )
    if isinstance(value, list):
        return [
            _resolve_value(
                item,
                field=f"{field}[{index}]",
                role=role,
                root=root,
                allow_download=allow_download,
                environ=environ,
                evidence=evidence,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, str):
        path = _local_path(value, root=root, field=field)
        _validate(path, role=role, spec={}, field=field)
        evidence.append({"field": field, "source": "local", "path": str(path)})
        return str(path)
    raise ModelDatasetError(
        f"{field}: expected a path, null, mapping, or list; got {type(value).__name__}"
    )


def _candidate_specs(spec: Mapping[str, Any], *, field: str) -> list[Mapping[str, Any]]:
    primary = dict(spec)
    fallback = primary.pop("fallback", None)
    if fallback is None:
        return [primary]
    if not isinstance(fallback, Mapping):
        raise ModelDatasetError(f"{field}.fallback must be a mapping")
    if "fallback" in fallback:
        raise ModelDatasetError(f"{field}.fallback cannot contain another fallback")
    return [primary, dict(fallback)]


def _resolve_spec(
    spec: Mapping[str, Any], *, field: str, role: str, root: Path,
    allow_download: bool, environ: Mapping[str, str],
) -> tuple[Path, str, str | None]:
    raw_source = spec.get("source")
    if not isinstance(raw_source, str) or not raw_source:
        raise ModelDatasetError(f"{field}.source is required")
    source = raw_source.lower()
    if source not in SOURCES:
        raise ModelDatasetError(
            f"{field}.source must be one of {sorted(SOURCES)}; got {raw_source!r}"
        )
    if source == "local":
        raw_path = spec.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ModelDatasetError(f"{field}.path is required for source=local")
        path = _local_path(raw_path, root=root, field=field)
    elif source == "ci":
        path = _ci_path(spec, field=field, environ=environ)
    elif source == "huggingface":
        path = _huggingface_path(
            spec,
            field=field,
            role=role,
            allow_download=allow_download,
            environ=environ,
        )
    else:
        path = _modelscope_path(
            spec,
            field=field,
            role=role,
            allow_download=allow_download,
            environ=environ,
        )
    revision = None if source == "local" else _revision(spec, field)
    return path, source, revision


def _local_path(value: str, *, root: Path, field: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelDatasetError(f"{field}: path does not exist: {path}") from exc


def _ci_path(
    spec: Mapping[str, Any], *, field: str, environ: Mapping[str, str]
) -> Path:
    raw_root = environ.get(CI_ROOT_ENV)
    if not raw_root:
        raise ModelDatasetError(f"{field}: {CI_ROOT_ENV} is required for source=ci")
    root = Path(raw_root).expanduser().resolve()
    raw_name = spec.get("name")
    if not isinstance(raw_name, str) or not raw_name:
        raise ModelDatasetError(f"{field}.name is required for source=ci")
    relative = Path(raw_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ModelDatasetError(f"{field}.name must stay below {CI_ROOT_ENV}: {raw_name!r}")
    return _local_path(str(root / relative), root=root, field=field)


def _revision(spec: Mapping[str, Any], field: str) -> str | None:
    revision = spec.get("revision")
    if revision is None:
        return None
    if not isinstance(revision, str) or not revision:
        raise ModelDatasetError(f"{field}.revision must be a non-empty string")
    return revision


def _remote_args(spec: Mapping[str, Any], *, field: str) -> tuple[str, str, str | None]:
    repo = spec.get("repo")
    if not isinstance(repo, str) or not repo:
        raise ModelDatasetError(f"{field}.repo is required")
    revision = _revision(spec, field)
    if revision is None:
        raise ModelDatasetError(f"{field}.revision is required for a remote source")
    raw_file = spec.get("file")
    if raw_file is not None and (not isinstance(raw_file, str) or not raw_file):
        raise ModelDatasetError(f"{field}.file must be a non-empty string")
    return repo, revision, raw_file


def _remote_file(result: Path, *, filename: str | None, field: str, repo: str) -> Path:
    if filename is None:
        return result
    candidate = result / filename
    if candidate.is_file():
        return candidate
    if result.is_file():
        return result
    raise ModelDatasetError(f"{field}: {filename!r} is absent from {repo!r}")


def _remote_error(
    field: str, provider: str, repo: str, revision: str,
    allow_download: bool, exc: Exception,
) -> ModelDatasetError:
    mode = "download" if allow_download else "cache"
    return ModelDatasetError(
        f"{field}: cannot resolve {provider} {repo!r}@{revision} "
        f"from {mode}: {exc}"
    )


def _huggingface_path(
    spec: Mapping[str, Any], *, field: str, role: str,
    allow_download: bool, environ: Mapping[str, str],
) -> Path:
    repo, revision, filename = _remote_args(spec, field=field)
    repo_type = "dataset" if role == "dataset" else "model"
    try:
        from huggingface_hub import snapshot_download

        result = Path(snapshot_download(
            repo_id=repo,
            repo_type=repo_type,
            revision=revision,
            cache_dir=environ.get("HF_HUB_CACHE") or None,
            allow_patterns=[filename] if filename else None,
            local_files_only=not allow_download,
        ))
    except Exception as exc:  # noqa: BLE001 - add input context to provider errors
        raise _remote_error(
            field, f"Hugging Face {repo_type}", repo, revision, allow_download, exc
        ) from exc
    return _remote_file(result, filename=filename, field=field, repo=repo)


def _modelscope_path(
    spec: Mapping[str, Any], *, field: str, role: str,
    allow_download: bool, environ: Mapping[str, str],
) -> Path:
    repo, revision, filename = _remote_args(spec, field=field)
    kwargs: dict[str, Any] = {
        "model_id": repo,
        "revision": revision,
        "local_files_only": not allow_download,
    }
    cache_dir = environ.get("MODELSCOPE_CACHE")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if role == "dataset":
        kwargs["repo_type"] = "dataset"
    if filename:
        kwargs["allow_file_pattern"] = filename
    try:
        from modelscope import snapshot_download

        result = Path(snapshot_download(**kwargs))
    except Exception as exc:  # noqa: BLE001 - add input context to provider errors
        raise _remote_error(
            field, "ModelScope", repo, revision, allow_download, exc
        ) from exc
    return _remote_file(result, filename=filename, field=field, repo=repo)


def _validate(path: Path, *, role: str, spec: Mapping[str, Any], field: str) -> None:
    if role == "model":
        _validate_model(path, spec=spec, field=field)
    elif role == "tokenizer":
        if not path.is_dir():
            raise ModelDatasetError(f"{field}: tokenizer must be a directory: {path}")
    elif not path.exists():
        raise ModelDatasetError(f"{field}: dataset does not exist: {path}")


def _validate_model(path: Path, *, spec: Mapping[str, Any], field: str) -> None:
    if not path.is_dir():
        raise ModelDatasetError(f"{field}: model must be a directory: {path}")
    required = spec.get("required_files")
    if required is not None:
        names = [required] if isinstance(required, str) else required
        if not isinstance(names, list) or not all(
            isinstance(item, str) and item for item in names
        ):
            raise ModelDatasetError(f"{field}.required_files must be a list of paths")
        missing = [path / name for name in names if not (path / name).is_file()]
        if missing:
            raise ModelDatasetError(f"{field}: required model file is missing: {missing[0]}")
        return
    if not (path / "config.json").is_file():
        raise ModelDatasetError(f"{field}: config.json is missing from {path}")
    if any((path / name).is_file() for name in _MODEL_WEIGHTS):
        return
    if any(path.glob("*.safetensors")) or any(path.glob("*.bin")):
        return
    raise ModelDatasetError(f"{field}: no model weights found in {path}")
