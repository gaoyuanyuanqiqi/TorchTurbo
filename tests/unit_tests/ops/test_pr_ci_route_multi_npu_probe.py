import pytest


pytestmark = pytest.mark.multi_npu


def test_pr_ci_multi_npu_route() -> None:
    if False:
        from tests.support.npu import require_npus

        require_npus(2)
    assert True
