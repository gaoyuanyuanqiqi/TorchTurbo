import pytest


pytestmark = pytest.mark.cpu


def test_failure_probe() -> None:
    assert False, "V4 P0 expected failure probe"
