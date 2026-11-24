import pytest


pytestmark = pytest.mark.npu


def test_npu_marker_probe() -> None:
    assert True
