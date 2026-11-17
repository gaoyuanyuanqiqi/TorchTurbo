from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.producer.scan import scan_tests


def _scan(tmp_path: Path, source: str):
    path = tmp_path / "tests" / "unit_tests" / "ops" / "test_example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return scan_tests(tmp_path, set())


def test_unmarked_cpu_test_defaults_to_cpu(tmp_path: Path) -> None:
    facts, errors = _scan(tmp_path, "def test_plain():\n    assert 1 + 1 == 2\n")

    assert errors == []
    assert facts["tests/unit_tests/ops/test_example.py"].devices == frozenset({"cpu"})


@pytest.mark.parametrize(
    ("source", "hint"),
    [
        ("def test_npu():\n    tensor.npu()\n", "@pytest.mark.npu"),
        ("def test_gpu():\n    tensor.cuda()\n", "@pytest.mark.gpu"),
        ("def test_multi():\n    require_npus(2)\n", "@pytest.mark.multi_npu"),
    ],
)
def test_accelerator_evidence_requires_explicit_marker(
    tmp_path: Path, source: str, hint: str
) -> None:
    _, errors = _scan(tmp_path, source)

    assert len(errors) == 1
    assert hint in errors[0].message


def test_unmarked_cpu_multiprocess_test_defaults_to_cpu(tmp_path: Path) -> None:
    facts, errors = _scan(
        tmp_path,
        "def test_plain():\n    mp.spawn(worker, nprocs=2)\n",
    )

    assert errors == []
    fact = facts["tests/unit_tests/ops/test_example.py"]
    assert fact.devices == frozenset({"cpu"})
    assert fact.processes_for("cpu") == 2
