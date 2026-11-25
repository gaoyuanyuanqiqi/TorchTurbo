"""Runtime guards for tests that execute on Ascend NPUs."""

from __future__ import annotations

import pytest


def _require_npu_marker(request) -> None:
    if request is None:
        return
    node = request.node
    if not (
        node.get_closest_marker("npu")
        or node.get_closest_marker("multi_npu")
    ):
        pytest.fail("NPU test requires @pytest.mark.npu or @pytest.mark.multi_npu")


def require_npu(request=None):
    """Skip unless torch_npu is installed and at least one NPU is available."""
    _require_npu_marker(request)
    pytest.importorskip("torch_npu")
    import torch

    if not torch.npu.is_available():
        pytest.skip("NPU is not available")


def require_npus(count: int, request=None):
    """Skip unless at least ``count`` NPUs are visible to the test process."""
    require_npu(request)
    import torch

    if torch.npu.device_count() < count:
        pytest.skip(f"requires at least {count} NPUs")
