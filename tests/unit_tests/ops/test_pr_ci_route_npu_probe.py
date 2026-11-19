import pytest


pytestmark = pytest.mark.npu


def test_pr_ci_explicit_npu_route() -> None:
    assert True
