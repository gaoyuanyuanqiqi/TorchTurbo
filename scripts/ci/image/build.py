"""Build, resolve, and verify an image described by ImagePlan."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .plan import ImagePlan
from .registry import DOCKER_LIST_V2, DOCKER_V2, Manifest, Registry, RegistryError


class BuildError(RuntimeError):
    """Raised when a candidate image cannot be built or verified."""


@dataclass(frozen=True)
class ImageResult:
    logical_image: str
    alias: str
    cann_version: str
    content_tag: str
    digest: str
    ref: str
    built: bool
    cache_hit: bool
    platform: str
    media_type: str
    elapsed_seconds: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "logical_image": self.logical_image,
            "alias": self.alias,
            "cann_version": self.cann_version,
            "content_tag": self.content_tag,
            "digest": self.digest,
            "ref": self.ref,
            "built": self.built,
            "cache_hit": self.cache_hit,
            "platform": self.platform,
            "media_type": self.media_type,
            "elapsed_seconds": self.elapsed_seconds,
            "reason": self.reason,
        }


def verify_manifest(manifest: Manifest) -> None:
    if manifest.media_type not in {DOCKER_V2, DOCKER_LIST_V2}:
        raise RegistryError(
            "registry returned unsupported manifest media type "
            f"{manifest.media_type!r}; expected Docker v2s2"
        )


def max_jobs(spec: str = "auto", *, cores: int | None = None, mem_gb: int | None = None) -> int:
    if spec and spec != "auto":
        return max(1, int(spec))
    cores = cores if cores is not None else (os.cpu_count() or 2)
    if mem_gb is None:
        mem_gb = 0
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    mem_gb = int(int(line.split()[1]) / 1048576)
                    break
        except OSError:
            mem_gb = 0
    return max(1, min(cores, (mem_gb or 1) // 3 or 1))


def build_command(
    plan: ImagePlan,
    *,
    should_push: bool,
    use_cache: bool,
    builder: str = "",
    jobs: int = 1,
    metadata_file: str = "/tmp/build-meta.json",
    dockerfile: str | None = None,
) -> list[str]:
    command = ["docker", "buildx", "build"]
    if builder:
        command += ["--builder", builder]
    command += [
        "--progress=plain",
        f"--platform={plan.platform}",
        f"--file={dockerfile or plan.dockerfile}",
        f"--build-arg=MAX_JOBS={jobs}",
    ]
    for key, value in sorted(plan.build_args.items()):
        command.append(f"--build-arg={key}={value}")
    command += ["--provenance=false", "--sbom=false"]
    if use_cache:
        scope = f"torchturbo-{plan.alias}"
        command += [
            f"--cache-from=type=gha,scope={scope}",
            f"--cache-to=type=gha,mode=min,scope={scope}",
        ]
    if should_push:
        command += [
            "--tag",
            plan.content_ref,
            f"--metadata-file={metadata_file}",
            "--output=type=image,push=true,oci-mediatypes=false",
        ]
    else:
        command.append("--output=type=cacheonly")
    command.append(plan.context)
    return command


def temporary_build_dockerfile(
    plan: ImagePlan, root: Path
) -> tuple[str, Path | None]:
    """Return a Dockerfile that scopes the GitCode WAF UA workaround to CI builds."""
    path = root / plan.dockerfile
    source = path.read_text(encoding="utf-8")
    user_agent = plan.build_args.get("GIT_HTTP_USER_AGENT", "")
    if not user_agent:
        return plan.dockerfile, None
    if "GIT_HTTP_USER_AGENT" in source:
        return plan.dockerfile, None
    lines = source.splitlines(keepends=True)
    injected: list[str] = []
    found_from = False
    for line in lines:
        injected.append(line)
        if line.lstrip().upper().startswith("FROM "):
            injected.append("ARG GIT_HTTP_USER_AGENT\n")
            injected.append("ENV GIT_HTTP_USER_AGENT=${GIT_HTTP_USER_AGENT}\n")
            found_from = True
    if not found_from:
        raise BuildError(f"dockerfile has no FROM instruction: {plan.dockerfile}")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        prefix="torchturbo-ci-Dockerfile-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write("".join(injected))
        temporary = Path(handle.name)
    return str(temporary), temporary


def _metadata_digest(path: str) -> str:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(value.get("containerimage.digest", ""))


def _run_build(
    command: list[str], *, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run docker buildx while streaming every line to the caller's log."""
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - Popen requested a pipe
        raise BuildError("failed to capture docker buildx output")
    lines: list[str] = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return subprocess.CompletedProcess(
        command, process.wait(), "".join(lines), ""
    )


def _inspect_digest(ref: str) -> str:
    process = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", ref,
         "--format", "{{.Manifest.Digest}}"],
        capture_output=True,
        text=True,
    )
    return process.stdout.strip() if process.returncode == 0 else ""


def _result_from_manifest(
    plan: ImagePlan,
    manifest: Manifest,
    *,
    built: bool,
    reason: str,
    elapsed_seconds: int = 0,
) -> ImageResult:
    return ImageResult(
        logical_image=plan.logical_image,
        alias=plan.alias,
        cann_version=plan.cann_version,
        content_tag=plan.content_tag,
        digest=manifest.digest,
        ref=f"{plan.repository}@{manifest.digest}",
        built=built,
        cache_hit=not built,
        platform=plan.platform,
        media_type=manifest.media_type,
        elapsed_seconds=elapsed_seconds,
        reason=reason,
    )


def resolve_formal(plan: ImagePlan, registry: Registry) -> ImageResult:
    alias = registry.require(plan.alias)
    content = registry.require(plan.content_tag)
    verify_manifest(alias)
    verify_manifest(content)
    if alias.digest != content.digest:
        raise RegistryError(
            f"image alias is stale: {plan.alias}={alias.digest}, "
            f"current content {plan.content_tag}={content.digest}"
        )
    return _result_from_manifest(plan, alias, built=False, reason="formal")


def resolve_candidate(plan: ImagePlan, registry: Registry) -> ImageResult:
    """Resolve the immutable content tag without requiring a promoted alias."""
    content = registry.require(plan.content_tag)
    verify_manifest(content)
    return _result_from_manifest(plan, content, built=False, reason="candidate")


def ensure_candidate(
    plan: ImagePlan,
    registry: Registry,
    root: Path,
    *,
    should_push: bool = True,
    use_cache: bool = True,
    builder: str = "",
    jobs: int = 1,
    run: Callable[..., subprocess.CompletedProcess[str]] = _run_build,
) -> ImageResult:
    existing = registry.manifest(plan.content_tag)
    if existing is not None:
        verify_manifest(existing)
        return _result_from_manifest(
            plan, existing, built=False, reason="existing_candidate"
        )
    if not should_push:
        raise BuildError(
            f"candidate {plan.content_ref} is absent and pushing is disabled; "
            "run the Ascend Candidate workflow with build_if_missing first"
        )
    metadata_file = "/tmp/build-meta.json"
    dockerfile, temporary = temporary_build_dockerfile(plan, root)
    try:
        command = build_command(
            plan,
            should_push=True,
            use_cache=use_cache,
            builder=builder,
            jobs=jobs,
            metadata_file=metadata_file,
            dockerfile=dockerfile,
        )
        started = time.monotonic()
        process = run(command, cwd=root)
        elapsed = int(time.monotonic() - started)
        log = process.stdout or ""
        Path("/tmp/build.log").write_text(log, encoding="utf-8")
        if process.returncode != 0:
            raise BuildError(
                f"image build failed (exit {process.returncode}); see /tmp/build.log"
            )
        digest = _metadata_digest(metadata_file)
        if not digest:
            digest = _inspect_digest(plan.content_ref)
        if not digest.startswith("sha256:"):
            raise BuildError(f"build did not publish a digest for {plan.content_ref}")
        manifest = registry.require(plan.content_tag)
        verify_manifest(manifest)
        if manifest.digest != digest:
            raise BuildError(
                f"built digest {digest} differs from registry {manifest.digest} "
                f"for {plan.content_tag}"
            )
        return _result_from_manifest(
            plan, manifest, built=True, reason="candidate", elapsed_seconds=elapsed
        )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
