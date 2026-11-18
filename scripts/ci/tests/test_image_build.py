from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ci.image.build import (
    BuildError,
    _run_build,
    build_command,
    ensure_candidate,
    max_jobs,
    resolve_candidate,
    resolve_formal,
    temporary_build_dockerfile,
    verify_manifest,
)
from scripts.ci.image.plan import plan_image
from scripts.ci.image.registry import (
    DOCKER_V2,
    Manifest,
    Registry,
    RegistryError,
)


class FakeRegistry:
    def __init__(self, manifests: dict[str, Manifest]) -> None:
        self.manifests = manifests

    def manifest(self, reference: str) -> Manifest | None:
        return self.manifests.get(reference)

    def require(self, reference: str) -> Manifest:
        value = self.manifest(reference)
        if value is None:
            raise RegistryError(f"missing {reference}")
        return value


def _manifest(reference: str, digest: str = "sha256:abc", media_type: str = DOCKER_V2) -> Manifest:
    return Manifest(reference=reference, digest=digest, media_type=media_type, raw={})


def test_build_command_uses_content_tag_not_alias(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")
    command = build_command(plan, should_push=True, use_cache=False, jobs=4)

    assert "--tag" in command
    assert plan.content_ref in command
    assert not any(value.endswith(f":{plan.alias}") for value in command)
    assert "--output=type=image,push=true,oci-mediatypes=false" in command
    assert "--build-arg=MAX_JOBS=4" in command
    assert "--build-arg=GIT_HTTP_USER_AGENT=git/2.56.1" in command


def test_temporary_dockerfile_injects_git_user_agent(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")

    dockerfile, temporary = temporary_build_dockerfile(plan, repo_root)

    try:
        text = Path(dockerfile).read_text(encoding="utf-8")
        assert text.count("ARG GIT_HTTP_USER_AGENT") == text.count("FROM ")
        assert text.count("ENV GIT_HTTP_USER_AGENT=${GIT_HTTP_USER_AGENT}") == text.count("FROM ")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def test_max_jobs_respects_memory_limit() -> None:
    assert max_jobs("auto", cores=8, mem_gb=15) == 5
    assert max_jobs("auto", cores=16, mem_gb=0) == 1
    assert max_jobs("6", cores=1, mem_gb=1) == 6


def test_verify_manifest_rejects_oci() -> None:
    with pytest.raises(RegistryError, match="Docker v2s2"):
        verify_manifest(_manifest("x", media_type="application/vnd.oci.image.manifest.v1+json"))


def test_resolve_formal_requires_alias_and_content_to_match(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")
    registry = FakeRegistry({
        plan.alias: _manifest(plan.alias, "sha256:abc"),
        plan.content_tag: _manifest(plan.content_tag, "sha256:def"),
    })
    with pytest.raises(RegistryError, match="alias is stale"):
        resolve_formal(plan, registry)

    registry.manifests[plan.content_tag] = _manifest(plan.content_tag, "sha256:abc")
    result = resolve_formal(plan, registry)
    assert result.ref == f"{plan.repository}@sha256:abc"
    assert result.built is False


def test_resolve_candidate_does_not_require_a_promoted_alias(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")
    registry = FakeRegistry({plan.content_tag: _manifest(plan.content_tag)})

    result = resolve_candidate(plan, registry)

    assert result.ref == f"{plan.repository}@sha256:abc"
    assert result.reason == "candidate"
    assert result.built is False

    with pytest.raises(RegistryError, match="missing"):
        resolve_candidate(plan, FakeRegistry({plan.alias: _manifest(plan.alias)}))


def test_existing_candidate_does_not_build(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")
    registry = FakeRegistry({plan.content_tag: _manifest(plan.content_tag)})

    def fail_run(*args, **kwargs):
        raise AssertionError("build should not run")

    result = ensure_candidate(
        plan,
        registry,
        repo_root,
        run=fail_run,
    )
    assert result.cache_hit is True
    assert result.reason == "existing_candidate"


def test_build_without_push_and_missing_candidate_fails(repo_root: Path) -> None:
    plan = plan_image(repo_root, "npu-default")
    with pytest.raises(BuildError, match="pushing is disabled"):
        ensure_candidate(plan, FakeRegistry({}), repo_root, should_push=False)


def test_blocked_plan_reuses_existing_candidate(repo_root: Path) -> None:
    plan = plan_image(
        repo_root,
        "npu-default",
        policy="build_if_missing",
        build_allowed=False,
    )
    registry = FakeRegistry({plan.content_tag: _manifest(plan.content_tag)})

    result = ensure_candidate(
        plan,
        registry,
        repo_root,
        should_push=False,
    )

    assert plan.action == "blocked"
    assert result.reason == "existing_candidate"
    assert result.built is False


def test_registry_promote_builds_retag_command() -> None:
    observed: list[list[str]] = []

    def fake_run(command, **kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    registry = Registry("registry.example/team/image")
    registry.promote("a3-base", "sha256:abc", run=fake_run)
    assert observed == [[
        "docker",
        "buildx",
        "imagetools",
        "create",
        "-t",
        "registry.example/team/image:a3-base",
        "registry.example/team/image@sha256:abc",
    ]]


def test_run_build_streams_and_captures_output(tmp_path: Path, capsys) -> None:
    result = _run_build(
        [sys.executable, "-c", "print('line one'); print('line two')"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["line one", "line two"]
    assert capsys.readouterr().out.splitlines() == ["line one", "line two"]
