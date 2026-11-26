"""Compute image content tags, resolve bindings, build, and promote."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from ..contracts.image_contract import ImageBuildResult, ResolvedImage
from ..tools.json_io import content_digest
from .image_config import ImageError, ImageSpec, load_image
from .registry import Manifest, Registry, RegistryError, verify_manifest

BUILD_LOG = Path("/tmp/build.log")
_GIT_USER_AGENT_PATCH = (
    "ARG GIT_HTTP_USER_AGENT\n"
    "ENV GIT_HTTP_USER_AGENT=${GIT_HTTP_USER_AGENT}\n"
)
_FROM_LINE = re.compile(r"^([ \t]*FROM[^\n]*)(?:\n|\Z)", re.IGNORECASE | re.MULTILINE)

class BuildError(RuntimeError):
    """Raised when an image cannot be built or verified."""


@dataclass(frozen=True)
class ImagePlan:
    spec: ImageSpec
    content_hash: str

    @property
    def image_id(self) -> str:
        return self.spec.image_id

    @property
    def content_tag(self) -> str:
        prefix = self.spec.alias.removesuffix("-base")
        return f"{prefix}-{self.content_hash.removeprefix('sha256:')[:12]}"

    @property
    def content_ref(self) -> str:
        return f"{self.spec.repository}:{self.content_tag}"


def get_content_hash(root: Path, spec: ImageSpec) -> str:
    """Hash the Dockerfile, platform and manual revision of one image."""
    path = root / spec.dockerfile
    try:
        dockerfile = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ImageError(f"cannot read dockerfile {spec.dockerfile}: {exc}") from exc
    return content_digest({
        "dockerfile": dockerfile,
        "platform": spec.platform,
        "revision": spec.revision,
    })


def get_image_plan(root: Path, image_id: str) -> ImagePlan:
    spec = load_image(root, image_id)
    return ImagePlan(spec=spec, content_hash=get_content_hash(root, spec))


def _resolved_image(plan: ImagePlan, digest: str) -> ResolvedImage:
    return ResolvedImage(
        image_id=plan.image_id,
        target=plan.spec.target,
        kind="container",
        ref=f"{plan.spec.repository}@{digest}",
    )


def _resolved_image_from_manifest(plan: ImagePlan, manifest: Manifest) -> ResolvedImage:
    verify_manifest(manifest)
    return _resolved_image(plan, manifest.digest)


def _image_outcome(
    plan: ImagePlan,
    image: ResolvedImage,
    *,
    built: bool,
    elapsed_seconds: int = 0,
    alias: str = "",
) -> ImageBuildResult:
    return ImageBuildResult(image=image, content_hash=plan.content_hash,
                            built=built, elapsed_seconds=elapsed_seconds,
                            alias=alias)


def resolved_image_for_content_tag(plan: ImagePlan, content_tag: str) -> ResolvedImage:
    """Resolve one content tag to an immutable image."""
    registry = Registry(plan.spec.repository)
    return _resolved_image_from_manifest(plan, registry.require_manifest(content_tag))


def check_build_environment() -> None:
    """Check whether the Docker build environment is usable."""
    if not shutil.which("docker"):
        raise BuildError("docker is not available")
    version = subprocess.run(
        ["docker", "version", "--format", "client={{.Client.Version}}"],
        capture_output=True, text=True,
    )
    if version.returncode != 0:
        raise BuildError((version.stderr or version.stdout).strip() or "docker is not usable")
    buildx = subprocess.run(["docker", "buildx", "version"], capture_output=True, text=True)
    if buildx.returncode != 0:
        raise BuildError("docker buildx is not available")


def _memory_gb() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    match = re.search(r"^MemTotal:\s+(\d+) kB$", text, re.MULTILINE)
    return int(match.group(1)) // 1048576 if match else 0


def max_jobs(
    spec: str = "auto",
    *,
    cores: int | None = None,
    mem_gb: int | None = None,
) -> int:
    if spec and spec != "auto":
        try:
            return max(1, int(spec))
        except ValueError as exc:
            raise BuildError(f"invalid max jobs value: {spec!r}") from exc
    cores = cores if cores is not None else (os.cpu_count() or 2)
    mem_gb = mem_gb if mem_gb is not None else _memory_gb()
    return max(1, min(cores, max(1, mem_gb // 3)))


def build_command(
    plan: ImagePlan,
    *,
    use_cache: bool,
    builder: str = "",
    jobs: int = 1,
    dockerfile: Path | None = None,
) -> list[str]:
    spec = plan.spec
    cache_args: list[str] = []
    if use_cache:
        scope = f"torchturbo-{spec.alias}"
        cache_args = [f"--cache-from=type=gha,scope={scope}",
                      f"--cache-to=type=gha,mode=min,scope={scope}"]
    build_args = [f"--build-arg={key}={value}"
                  for key, value in sorted(spec.ci_build_args.items())]
    return [
        "docker", "buildx", "build",
        *(["--builder", builder] if builder else []),
        "--progress=plain", f"--platform={spec.platform}",
        f"--file={dockerfile or spec.dockerfile}",
        f"--build-arg=MAX_JOBS={jobs}", *build_args,
        "--provenance=false", "--sbom=false", *cache_args,
        "--tag", plan.content_ref,
        "--output=type=image,push=true,oci-mediatypes=false", spec.context,
    ]


# Compiling the FLA library requires cloning its GitCode repository, but GitCode
# rejects requests from the git version bundled in the CANN base image. Patch
# only the build Dockerfile so the image content identity stays unchanged.
def patch_dockerfile(plan: ImagePlan, root: Path) -> tuple[Path, Path | None]:
    """Patch the build Dockerfile with a GitCode-compatible user agent."""
    original = root / plan.spec.dockerfile
    source = original.read_text(encoding="utf-8")
    user_agent = plan.spec.ci_build_args.get("GIT_HTTP_USER_AGENT", "")
    if not user_agent or "GIT_HTTP_USER_AGENT" in source:
        return original, None
    patched, count = _FROM_LINE.subn(rf"\1\n{_GIT_USER_AGENT_PATCH}", source)
    if not count:
        raise BuildError(f"dockerfile has no FROM instruction: {plan.spec.dockerfile}")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="torchturbo-ci-Dockerfile-",
        suffix=".tmp", delete=False,
    ) as handle:
        handle.write(patched)
        temporary = Path(handle.name)
    return temporary, temporary


def _run_build(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - Popen requested a pipe
        raise BuildError("failed to capture docker buildx output")
    lines = []
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    return subprocess.CompletedProcess(command, process.wait(), "".join(lines), "")


def build_image(
    plan: ImagePlan,
    root: Path,
    *,
    use_cache: bool = True,
    builder: str = "",
    jobs: int = 1,
) -> int:
    dockerfile, temporary = patch_dockerfile(plan, root)
    try:
        command = build_command(plan, use_cache=use_cache, builder=builder,
                                jobs=jobs, dockerfile=dockerfile)
        started = time.monotonic()
        process = _run_build(command, cwd=root)
        elapsed = int(time.monotonic() - started)
        log = process.stdout or ""
        BUILD_LOG.write_text(log, encoding="utf-8")
        if process.returncode != 0:
            raise BuildError(f"image build failed (exit {process.returncode}); see {BUILD_LOG}")
        return elapsed
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def get_or_build_image(
    root: Path,
    image_id: str,
    *,
    use_cache: bool = True,
    builder: str = "",
    jobs: str = "auto",
) -> ImageBuildResult:
    """Return an immutable content image, building it only when it is absent."""
    plan = get_image_plan(root, image_id)
    registry = Registry(plan.spec.repository)
    manifest = registry.find_manifest(plan.content_tag)
    if manifest is not None:
        return _image_outcome(
            plan, _resolved_image_from_manifest(plan, manifest), built=False
        )

    check_build_environment()
    elapsed = build_image(plan, root, use_cache=use_cache, builder=builder,
                          jobs=max_jobs(jobs))
    image = _resolved_image_from_manifest(
        plan, registry.require_manifest(plan.content_tag)
    )
    return _image_outcome(plan, image, built=True, elapsed_seconds=elapsed)


def resolve_image(root: Path, image_id: str) -> ResolvedImage:
    """Resolve one logical image to an immutable image."""
    plan = get_image_plan(root, image_id)
    return resolved_image_for_content_tag(plan, plan.content_tag)


def promote_image(root: Path, image_id: str) -> ImageBuildResult:
    """Move the stable publish alias to the tested content image."""
    plan = get_image_plan(root, image_id)
    registry = Registry(plan.spec.repository)
    expected_digest = _resolved_image_from_manifest(
        plan, registry.require_manifest(plan.content_tag)).digest
    check_build_environment()
    registry.promote_to_alias(plan.spec.alias, expected_digest)
    alias_image = _resolved_image_from_manifest(
        plan,
        registry.require_manifest(plan.spec.alias),
    )
    if alias_image.digest != expected_digest:
        raise RegistryError(
            f"promote verification failed: {plan.spec.alias}={alias_image.digest}, "
            f"expected {expected_digest}"
        )

    return _image_outcome(plan, alias_image, built=False, alias=plan.spec.alias)


def sync_alias(root: Path, image_id: str) -> ImageBuildResult:
    """Point the stable alias at the current content image when they differ."""
    plan = get_image_plan(root, image_id)
    registry = Registry(plan.spec.repository)
    content_image = _resolved_image_from_manifest(
        plan, registry.require_manifest(plan.content_tag)
    )
    alias_manifest = registry.find_manifest(plan.spec.alias)
    if alias_manifest is not None:
        alias_image = _resolved_image_from_manifest(plan, alias_manifest)
        if alias_image.digest == content_image.digest:
            return _image_outcome(
                plan, alias_image, built=False, alias=plan.spec.alias
            )
    return promote_image(root, image_id)
