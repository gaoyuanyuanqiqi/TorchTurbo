from __future__ import annotations

from pathlib import Path

from scripts.ci.tools.lint import check_device_markers


def _write_test(root: Path, source: str) -> None:
    path = root / "tests" / "unit_tests" / "test_example.py"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")


def test_npu_module_requires_marker(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "import pytest\n\npytest.importorskip('torch_npu')\n",
    )

    problems = check_device_markers(tmp_path)

    assert [problem.message for problem in problems] == [
        "NPU-specific module requires pytestmark = pytest.mark.npu"
    ]


def test_npu_marker_allows_module_signal(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "import pytest\n\n"
        "pytestmark = pytest.mark.npu\n"
        "pytest.importorskip('torch_npu')\n",
    )

    assert check_device_markers(tmp_path) == []


def test_cpu_marker_allows_module_signal(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "import pytest\n\n"
        "pytestmark = pytest.mark.cpu\n"
        "pytest.importorskip('torch_npu')\n",
    )

    assert check_device_markers(tmp_path) == []


def test_function_scope_signal_is_not_flagged(tmp_path: Path) -> None:
    _write_test(
        tmp_path,
        "import pytest\n\n"
        "def test_npu():\n"
        "    pytest.importorskip('torch_npu')\n",
    )

    assert check_device_markers(tmp_path) == []
