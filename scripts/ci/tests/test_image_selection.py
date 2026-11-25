from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.ci.image.select_images import ImageSelectionError, select_images


def _write_two_image_repo(root: Path) -> Path:
    config = root / "scripts/ci/config/ci.yml"
    config.parent.mkdir(parents=True)
    document = {
        "resources": {
            "targets": {
                "npu-a3": {"device": "npu"},
                "npu-a5": {"device": "npu"},
            },
            "images": {
                "npu-a3-cann90": {
                    "target": "npu-a3",
                    "kind": "container",
                    "build": {
                        "dockerfile": "docker/npu/Dockerfile_aarch64",
                        "platform": "linux/arm64",
                    },
                    "publish": {
                        "ref": "registry.example/team/image:cann9.0.0-a3-base",
                    },
                },
                "npu-a5-cann90": {
                    "target": "npu-a5",
                    "kind": "container",
                    "build": {
                        "dockerfile": "docker/npu-a5/Dockerfile_a5",
                        "platform": "linux/arm64",
                    },
                    "publish": {
                        "ref": "registry.example/team/image:cann9.0.0-a5-base",
                    },
                },
            },
        },
    }
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    for name, dockerfile in (
        ("npu", "Dockerfile_aarch64"),
        ("npu-a5", "Dockerfile_a5"),
    ):
        path = root / "docker" / name / dockerfile
        path.parent.mkdir(parents=True)
        path.write_text("FROM base:one\n", encoding="utf-8")
    return config


def test_changed_files_select_only_mapped_images(tmp_path: Path) -> None:
    _write_two_image_repo(tmp_path)
    selected = select_images(
        tmp_path, changed_files=["docker/npu/Dockerfile_aarch64"]
    )

    assert [item.image_id for item in selected] == ["npu-a3-cann90"]
    assert selected[0].target == "npu-a3"


def test_changed_images_reject_unmapped_dockerfile(tmp_path: Path) -> None:
    _write_two_image_repo(tmp_path)
    with pytest.raises(ImageSelectionError, match="not mapped"):
        select_images(tmp_path, changed_files=["docker/other/Dockerfile_unknown"])


def test_non_dockerfile_changes_select_no_images(tmp_path: Path) -> None:
    _write_two_image_repo(tmp_path)
    assert select_images(tmp_path, changed_files=["docs/ci.md"]) == []
    assert select_images(tmp_path, changed_files=["docker/npu/setup.sh"]) == []
    assert select_images(tmp_path, changed_files=["scripts/ci/config/ci.yml"]) == []


def test_named_images_can_select_all_or_explicit_ids(tmp_path: Path) -> None:
    _write_two_image_repo(tmp_path)
    assert [item.image_id for item in select_images(tmp_path, image_ids="all")] == [
        "npu-a3-cann90", "npu-a5-cann90"
    ]
    assert [
        item.image_id
        for item in select_images(tmp_path, image_ids="npu-a5-cann90")
    ] == ["npu-a5-cann90"]
    with pytest.raises(ImageSelectionError, match="unknown image ID"):
        select_images(tmp_path, image_ids="missing")


def test_multiple_changed_dockerfiles_produce_multiple_images(tmp_path: Path) -> None:
    _write_two_image_repo(tmp_path)
    both = select_images(
        tmp_path,
        changed_files=[
            "docker/npu/Dockerfile_aarch64",
            "docker/npu-a5/Dockerfile_a5",
        ],
    )

    assert [item.image_id for item in both] == [
        "npu-a3-cann90", "npu-a5-cann90"
    ]
    assert {item.image_id: item.target for item in both} == {
        "npu-a3-cann90": "npu-a3",
        "npu-a5-cann90": "npu-a5",
    }


def test_new_image_with_new_dockerfile_is_selected(tmp_path: Path) -> None:
    config = _write_two_image_repo(tmp_path)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["resources"]["images"]["npu-a3-cann92"] = {
        "target": "npu-a3",
        "kind": "container",
        "build": {
            "dockerfile": "docker/npu-cann92/Dockerfile",
            "platform": "linux/arm64",
        },
        "publish": {
            "ref": "registry.example/team/image:cann9.2.0-a3-base",
        },
    }
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    dockerfile = tmp_path / "docker/npu-cann92/Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM base:new\n", encoding="utf-8")

    selected = select_images(
        tmp_path,
        changed_files=[
            "scripts/ci/config/ci.yml",
            "docker/npu-cann92/Dockerfile",
        ],
    )
    assert [item.image_id for item in selected] == ["npu-a3-cann92"]


def test_deleted_dockerfile_not_in_added_or_modified_list_is_a_no_op(
    tmp_path: Path,
) -> None:
    _write_two_image_repo(tmp_path)
    assert select_images(tmp_path, changed_files=[]) == []
