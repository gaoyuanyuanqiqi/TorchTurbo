import pytest

from tests.support.npu import require_npus


pytestmark = pytest.mark.multi_npu


def test_multi_npu_marker_probe() -> None:
    require_npus(2)
    assert True
