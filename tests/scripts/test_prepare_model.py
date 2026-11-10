from __future__ import annotations

import fnmatch
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare_model = _load_script(
    "prepare_model",
    REPO_ROOT / "scripts/ci/e2e_tests/prepare_model.py",
)


def _write_model(root: Path, layers: int = 2) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": {
            "num_hidden_layers": layers,
            "layer_types": ["full_attention"] * layers,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"num_hidden_layers": 3},
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"second")
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.weight":
                "model-00001-of-00002.safetensors",
            "model.language_model.embed_tokens.weight":
                "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00001-of-00002.safetensors",
            "model.language_model.layers.1.self_attn.q_proj.weight":
                "model-00002-of-00002.safetensors",
            "model.language_model.mtp.layers.0.self_attn.q_proj.weight":
                "model-00002-of-00002.safetensors",
        },
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def _write_trimmed_model(root: Path, layers: int = 1) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "text_config": {
            "num_hidden_layers": layers,
            "layer_types": ["full_attention"] * layers,
        }
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    index = {
        "weight_map": {
            "model.language_model.layers.0.self_attn.q_proj.weight":
                "model-00001-of-00001.safetensors",
            "model.language_model.embed_tokens.weight":
                "model-00001-of-00001.safetensors",
        }
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )


def test_should_keep_tensor_handles_nested_layers():
    main = "model.language_model.layers.7.self_attn.q_proj.weight"
    mtp = "model.language_model.mtp.layers.0.self_attn.q_proj.weight"
    vision = "model.visual.blocks.3.attn.qkv.weight"

    assert prepare_model.should_keep_tensor("model.language_model.layers.5.w", 6)
    assert not prepare_model.should_keep_tensor(main, 6)
    assert not prepare_model.should_keep_tensor(mtp, 6)
    assert prepare_model.should_keep_tensor(vision, 6)
    assert prepare_model.should_keep_tensor("lm_head.weight", 6)


def test_trim_model_config_updates_nested_text_config():
    config = {
        "text_config": {
            "num_hidden_layers": 8,
            "layer_types": ["full_attention"] * 8,
            "compress_ratios": [1] * 8,
            "mtp_num_hidden_layers": 1,
        },
        "vision_config": {"num_hidden_layers": 3},
    }

    assert prepare_model.trim_model_config(config, 2) == 8
    assert config["text_config"]["num_hidden_layers"] == 2
    assert config["text_config"]["layer_types"] == ["full_attention"] * 2
    assert config["text_config"]["compress_ratios"] == [1, 1]
    assert config["text_config"]["mtp_num_hidden_layers"] == 0
    assert config["vision_config"]["num_hidden_layers"] == 3


def test_trim_model_config_rejects_invalid_layer_group():
    config = {"num_hidden_layers": 20, "layer_group_size": 5}

    with pytest.raises(SystemExit, match="multiple of 5"):
        prepare_model.trim_model_config(config, 6)


def test_validate_model_dir_accepts_full_model(tmp_path):
    _write_model(tmp_path)

    assert prepare_model.validate_model_dir(tmp_path) is not None


def test_validate_model_dir_rejects_missing_shard(tmp_path):
    _write_model(tmp_path)
    (tmp_path / "model-00002-of-00002.safetensors").unlink()

    assert prepare_model.validate_model_dir(tmp_path) is None


def test_create_trimmed_model_keeps_only_the_first_layers(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _write_model(remote)

    def fake_download(repo, directory, patterns, attempts):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path in remote.iterdir():
            if patterns is None or any(fnmatch.fnmatch(path.name, p) for p in patterns):
                shutil.copy2(path, target / path.name)

    monkeypatch.setattr(prepare_model, "download_repo", fake_download)
    target = tmp_path / "trimmed"
    prepare_model.create_trimmed_model("org/model", target, 1, 1)

    index = json.loads((target / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
    assert index["metadata"]["tt_trimmed_layers"] == 1
    assert index["metadata"]["tt_original_layers"] == 2
    assert prepare_model.validate_model_dir(target, layers=1) is not None


def test_read_model_plan_rejects_more_than_sixteen_cards(tmp_path, monkeypatch):
    run_dir = tmp_path / "scripts" / "runs" / "demo"
    run_dir.mkdir(parents=True)
    (run_dir / "install_demo_deps.sh").write_text("", encoding="utf-8")
    (run_dir / "train_demo.sh").write_text("", encoding="utf-8")
    model_map = {
        "models": {
            "demo": {
                "run_dir": "demo",
                "download": {"hf_repo": "org/model"},
                "install": {"lf": "install_demo_deps.sh"},
                "train": {"lf": "train_demo.sh"},
                "params": {"cards": 32, "sp": 1, "ep": 1},
            }
        }
    }
    map_path = tmp_path / "model_map.json"
    map_path.write_text(json.dumps(model_map), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match=r"1\.\.16"):
        prepare_model.read_model_plan(str(map_path), "demo", "lf")


def test_layers_must_be_non_negative(tmp_path):
    with pytest.raises(SystemExit, match=">= 0"):
        prepare_model.prepare_model_dir(
            "org/model", "", -1, str(tmp_path), str(tmp_path), 1
        )
    with pytest.raises(SystemExit, match=">= 0"):
        prepare_model.validate_model_dir(tmp_path, -1)


def test_prepare_model_dir_uses_a_valid_hint(tmp_path):
    hint = tmp_path / "hint"
    _write_model(hint)

    result = prepare_model.prepare_model_dir(
        "org/model", str(hint), 0, str(tmp_path / "cache"), str(tmp_path / "hf"), 1
    )

    assert result["MODEL"] == str(hint)
    assert result["HIT"] == "yes"


def test_find_cached_model_requires_matching_trim_marker(tmp_path):
    cache_root = tmp_path / "cache"
    target = cache_root / "model-L1"
    _write_trimmed_model(target)
    marker = "v2 repo=org/model layers=1"
    args = ("org/model", str(tmp_path / "hf"), str(cache_root), target)

    assert prepare_model.find_cached_model(*args, "v2 repo=org/model layers=2", 1) is None

    (target / prepare_model.MARKER).write_text(marker + "\n", encoding="utf-8")
    found = prepare_model.find_cached_model(*args, marker, 1)
    assert found is not None
    assert found[0] == target


def test_prepare_model_dir_locks_only_trimmed_models(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    remote.mkdir()
    _write_model(remote)

    def fake_download(repo, directory, patterns, attempts):
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        for path in remote.iterdir():
            if patterns is None or any(fnmatch.fnmatch(path.name, p) for p in patterns):
                shutil.copy2(path, target / path.name)

    monkeypatch.setattr(prepare_model, "download_repo", fake_download)
    cache_root = tmp_path / "cache"

    prepare_model.prepare_model_dir(
        "org/model", "", 0, str(cache_root), str(tmp_path / "hf"), 1
    )
    assert not (cache_root / "org" / ".model.lock").exists()

    prepare_model.prepare_model_dir(
        "org/model", "", 1, str(cache_root), str(tmp_path / "hf"), 1
    )
    assert (cache_root / ".model-L1.lock").exists()
