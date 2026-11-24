import pytest


pytestmark = pytest.mark.cpu


def test_cpu_marker_probe() -> None:
    assert True
