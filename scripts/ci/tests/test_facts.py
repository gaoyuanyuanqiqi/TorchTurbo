from __future__ import annotations

import json
from pathlib import Path

from scripts.ci.execution.producer.facts import (
    INTEGRATION_RUNNER_SCRIPT,
    INTEGRATION_SUITE_DIRECTORY,
    _integration_facts,
    _pytest_facts,
)


def _scan(tmp_path: Path, source: str):
    return _scan_path(
        tmp_path,
        "tests/unit_tests/ops/test_example.py",
        source,
    )


def _scan_path(tmp_path: Path, relative: str, source: str):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    facts, errors = _pytest_facts(tmp_path)
    return facts[0], errors


def test_unmarked_test_defaults_to_cpu(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "def test_plain():\n    assert 1 + 1 == 2\n",
    )

    assert errors == []
    assert fact.devices == ("cpu",)
    assert fact.ranks_by_device == {"cpu": 1}


def test_explicit_multi_marker_sets_device_and_minimum_rank(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "import pytest\n\n@pytest.mark.multi_npu\n"
        "def test_npu():\n    pass\n",
    )

    assert errors == []
    assert fact.devices == ("npu",)
    assert fact.ranks_by_device == {"npu": 2}


def test_unmarked_cpu_multiprocess_rank_is_detected(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "def test_plain():\n    mp.spawn(worker, nprocs=4)\n",
    )

    assert errors == []
    assert fact.devices == ("cpu",)
    assert fact.ranks_by_device == {"cpu": 4}


def test_module_marker_list_sets_devices_and_ranks(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "import pytest\n\n"
        "pytestmark = [pytest.mark.cpu, pytest.mark.npu]\n\n"
        "def test_npu():\n    pass\n",
    )

    assert errors == []
    assert fact.devices == ("cpu", "npu")
    assert fact.ranks_by_device == {"cpu": 1, "npu": 1}


def test_parametrized_world_size_sets_rank(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.parametrize('world_size', [2, 4])\n"
        "def test_distributed(world_size):\n    pass\n",
    )

    assert errors == []
    assert fact.devices == ("cpu",)
    assert fact.ranks_by_device == {"cpu": 4}


def test_mixed_marked_and_unmarked_tests(tmp_path: Path) -> None:
    fact, errors = _scan(
        tmp_path,
        "import pytest\n\n"
        "def test_cpu():\n    pass\n\n"
        "@pytest.mark.multi_npu\n"
        "def test_npu():\n    pass\n",
    )

    assert errors == []
    assert fact.devices == ("cpu", "npu")
    assert fact.ranks_by_device == {"cpu": 1, "npu": 2}


def test_model_path_sets_suite_and_family_without_import(tmp_path: Path) -> None:
    fact, errors = _scan_path(
        tmp_path,
        "tests/unit_tests/models/qwen3_5/test_example.py",
        "def test_plain():\n    pass\n",
    )

    assert errors == []
    assert fact.suite == "model"
    assert fact.model_families == ("qwen3_5",)


def test_engine_model_import_does_not_set_family(tmp_path: Path) -> None:
    fact, errors = _scan_path(
        tmp_path,
        "tests/unit_tests/engine/test_example.py",
        "from torchturbo.models.qwen3_5 import model\n\n"
        "def test_plain():\n    pass\n",
    )

    assert errors == []
    assert fact.suite == "core"
    assert fact.model_families == ()


def test_adapter_path_sets_framework_without_import(tmp_path: Path) -> None:
    swift, errors = _scan_path(
        tmp_path,
        "tests/unit_tests/adapters/msswift/test_example.py",
        "def test_plain():\n    pass\n",
    )

    assert errors == []
    assert swift.suite == "adapter"
    assert swift.frameworks == ("swift",)


def test_adapter_path_takes_precedence_over_import(tmp_path: Path) -> None:
    fact, errors = _scan_path(
        tmp_path,
        "tests/unit_tests/adapters/msswift/test_example.py",
        "import llamafactory\n\n"
        "def test_plain():\n    pass\n",
    )

    assert errors == []
    assert fact.suite == "adapter"
    assert fact.frameworks == ("swift",)


def test_model_import_does_not_set_framework(tmp_path: Path) -> None:
    fact, errors = _scan_path(
        tmp_path,
        "tests/unit_tests/models/qwen3_5/test_example.py",
        "from llamafactory.data import loader\n\n"
        "def test_plain():\n    pass\n",
    )

    assert errors == []
    assert fact.suite == "model"
    assert fact.frameworks == ()


def _write_integration_fixture(root: Path, manifest: dict) -> None:
    suite = root / INTEGRATION_SUITE_DIRECTORY / "base_suite.py"
    suite.parent.mkdir(parents=True)
    suite.write_text("TEST_CASES = []\n", encoding="utf-8")
    runner = root / INTEGRATION_RUNNER_SCRIPT
    runner.parent.mkdir(parents=True)
    runner.write_text(
        f"print({json.dumps(manifest)!r})\n",
        encoding="utf-8",
    )


def test_missing_integration_runner_is_diagnostic(tmp_path: Path) -> None:
    suite = tmp_path / INTEGRATION_SUITE_DIRECTORY / "base_suite.py"
    suite.parent.mkdir(parents=True)
    suite.write_text("TEST_CASES = []\n", encoding="utf-8")

    facts, issues = _integration_facts(tmp_path)

    assert facts == []
    assert len(issues) == 1
    assert "manifest failed" in issues[0].message


def test_compare_false_does_not_require_golden(tmp_path: Path) -> None:
    _write_integration_fixture(tmp_path, {
        "schema_version": 1,
        "suite": "base_suite",
        "cases": [{
            "case_id": "case",
            "framework": "swift",
            "platform": "cpu",
            "ranks": 1,
            "compare": False,
            "update_golden": False,
            "config": "config.yaml",
        }],
    })
    (tmp_path / "config.yaml").write_text("value: 1\n", encoding="utf-8")

    facts, issues = _integration_facts(tmp_path)

    assert [issue.message for issue in issues] == [
        "CI integration cases must compare"
    ]
    assert [fact.case_id for fact in facts] == ["case"]


def test_compare_requires_existing_golden(tmp_path: Path) -> None:
    _write_integration_fixture(tmp_path, {
        "schema_version": 1,
        "suite": "base_suite",
        "cases": [{
            "case_id": "case",
            "framework": "swift",
            "platform": "cpu",
            "ranks": 1,
            "compare": True,
            "update_golden": False,
            "config": "config.yaml",
            "golden": "golden.yaml",
        }],
    })

    _, issues = _integration_facts(tmp_path)

    assert [issue.message for issue in issues] == ["golden file is missing"]
