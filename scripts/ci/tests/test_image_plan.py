from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci.image.cli import main
from scripts.ci.image.build_image import get_image_plan
from scripts.ci.image.image_config import (
    ImageError,
    RuntimeImage,
    load_image,
    load_runtime_images,
    load_specs,
    split_reference,
)
from scripts.ci.image.select_images import select_images


def _config(name: str = "npu-a3-cann90") -> dict:
    return {
        "resources": {
            "targets": {
                "cpu": {"device": "cpu"},
                "npu-a3": {"device": "npu"},
            },
            "images": {
                "cpu-default": {
                    "target": "cpu",
                    "kind": "host",
                    "ref": "python:3.11",
                },
                name: {
                    "target": "npu-a3",
                    "kind": "container",
                    "build": {
                        "dockerfile": "docker/npu/Dockerfile_aarch64",
                        "platform": "linux/arm64",
                        "revision": 1,
                        "ci_build_args": {
                            "GIT_HTTP_USER_AGENT": "git/2.56.1",
                        },
                    },
                    "publish": {
                        "ref": "registry.example/team/image:cann9.0.0-a3-base",
                    },
                },
            },
        },
    }


def _write_repo(root: Path, name: str = "npu-a3-cann90") -> Path:
    config = root / "scripts/ci/config/ci.yml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(_config(name), sort_keys=False),
        encoding="utf-8",
    )
    context = root / "docker/npu"
    context.mkdir(parents=True)
    (context / "Dockerfile_aarch64").write_text(
        "FROM base:one\n", encoding="utf-8"
    )
    return config


def test_runtime_images_expose_host_and_unresolved_container(tmp_path: Path) -> None:
    _write_repo(tmp_path)

    runtime = load_runtime_images(tmp_path)

    assert runtime["cpu-default"] == RuntimeImage(
        image_id="cpu-default",
        target="cpu",
        kind="host",
        ref="python:3.11",
    )
    assert runtime["npu-a3-cann90"] == RuntimeImage(
        image_id="npu-a3-cann90",
        target="npu-a3",
        kind="container",
    )


def test_build_hash_tracks_only_dockerfile_platform_and_revision(tmp_path: Path) -> None:
    config = _write_repo(tmp_path)
    first = get_image_plan(tmp_path, "npu-a3-cann90")

    (tmp_path / "docker/npu/setup.sh").write_text("echo ignored\n", encoding="utf-8")
    assert get_image_plan(tmp_path, "npu-a3-cann90").content_hash == first.content_hash

    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    image = document["resources"]["images"]["npu-a3-cann90"]
    image["build"]["ci_build_args"]["PIP_INDEX_URL"] = "https://example.invalid"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert get_image_plan(tmp_path, "npu-a3-cann90").content_hash == first.content_hash

    image["build"]["revision"] = 2
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    revision_change = get_image_plan(tmp_path, "npu-a3-cann90")
    assert revision_change.content_hash != first.content_hash

    image["build"]["platform"] = "linux/amd64"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    assert get_image_plan(tmp_path, "npu-a3-cann90").content_hash != (
        revision_change.content_hash
    )

    (tmp_path / "docker/npu/Dockerfile_aarch64").write_text(
        "FROM base:two\n", encoding="utf-8"
    )
    assert get_image_plan(tmp_path, "npu-a3-cann90").content_hash != (
        revision_change.content_hash
    )


def test_build_hash_is_a_full_sha256_and_context_is_derived(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    plan = get_image_plan(tmp_path, "npu-a3-cann90")

    assert plan.content_hash.startswith("sha256:")
    assert len(plan.content_hash) == len("sha256:") + 64
    assert plan.spec.context == "docker/npu"
    assert plan.content_tag.startswith("cann9.0.0-a3-")
    assert plan.content_ref.startswith("registry.example/team/image:cann9.0.0-a3-")


def test_publish_alias_does_not_change_the_build_hash(tmp_path: Path) -> None:
    config = _write_repo(tmp_path)
    first = get_image_plan(tmp_path, "npu-a3-cann90")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["resources"]["images"]["npu-a3-cann90"]["publish"]["ref"] = (
        "registry.example/team/image:cann9.0.0-a3-next"
    )
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    changed = get_image_plan(tmp_path, "npu-a3-cann90")
    assert changed.content_hash == first.content_hash
    assert changed.content_ref != first.content_ref


def test_one_dockerfile_cannot_map_to_two_images(tmp_path: Path) -> None:
    config = _write_repo(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["resources"]["images"]["npu-a3-cann92"] = {
        "target": "npu-a3",
        "kind": "container",
        "build": {
            "dockerfile": "docker/npu/Dockerfile_aarch64",
            "platform": "linux/arm64",
        },
        "publish": {
            "ref": "registry.example/team/image:cann9.2.0-a3-base",
        },
    }
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ImageError, match="maps to both"):
        load_specs(tmp_path)


def test_load_image_rejects_unknown_id(tmp_path: Path) -> None:
    _write_repo(tmp_path)

    with pytest.raises(ImageError, match="unknown image ID"):
        load_image(tmp_path, "missing")


def test_dockerfile_must_stay_under_the_image_trigger_root(tmp_path: Path) -> None:
    config = _write_repo(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["resources"]["images"]["npu-a3-cann90"]["build"]["dockerfile"] = (
        "scripts/Dockerfile_bad"
    )
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ImageError, match="docker/\\*\\*/Dockerfile"):
        load_specs(tmp_path)


def test_unknown_container_recipe_fields_are_rejected(tmp_path: Path) -> None:
    config = _write_repo(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["resources"]["images"]["npu-a3-cann90"]["build"]["context"] = "docker/npu"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ImageError, match="unknown field"):
        load_specs(tmp_path)


def test_named_image_selection_and_cli_matrix(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    specs = select_images(tmp_path, image_ids="npu-a3-cann90")
    assert [spec.image_id for spec in specs] == ["npu-a3-cann90"]

    output = tmp_path / "matrix.json"
    github_output = tmp_path / "github-output"
    assert main([
        "plan",
        "--root", str(tmp_path),
        "--images", "all",
        "--output", str(output),
        "--github-output", str(github_output),
    ]) == 0
    assert '"image_id": "npu-a3-cann90"' in output.read_text(encoding="utf-8")
    assert "image_count=1" in github_output.read_text(encoding="utf-8")


def test_split_reference_supports_registry_ports() -> None:
    assert split_reference("registry.example:5000/team/image:tag") == (
        "registry.example:5000/team/image", "tag"
    )
    with pytest.raises(ImageError, match="explicit tag"):
        split_reference("registry.example/team/image")
