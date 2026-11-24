import pytest


pytestmark = pytest.mark.v4_unknown_marker


def test_unregistered_marker_probe() -> None:
    assert True
