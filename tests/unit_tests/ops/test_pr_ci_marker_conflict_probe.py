import pytest


pytestmark = [pytest.mark.cpu, pytest.mark.npu]


def test_pr_ci_marker_conflict() -> None:
    assert True
