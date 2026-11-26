"""Build device availability checks for the job venv."""

from __future__ import annotations

from pathlib import Path

_PROBE_CODE = {
    "cpu": "import torch",
    "npu": (
        "import torch, torch_npu; "
        "assert torch.npu.is_available() and torch.npu.device_count() > 0"
    ),
    "gpu": (
        "import torch; "
        "assert torch.cuda.is_available() and torch.cuda.device_count() > 0"
    ),
}


def probe_command(python: Path, device: str) -> list[str]:
    return [str(python), "-c", _PROBE_CODE[device]]
