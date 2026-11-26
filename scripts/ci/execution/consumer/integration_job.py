"""Integration Job command construction and artifact handling."""

from __future__ import annotations

import shutil
from pathlib import Path

from ...contracts.planner_contract import IntegrationSpec, Job
from .runtime.venv_manager import Venv

TRANSIENT_DIRS = ("training", "initial_checkpoint", "tensorboard")


def command(
    venv: Venv,
    spec: IntegrationSpec,
    job: Job,
    *,
    root: Path,
    output_dir: Path,
) -> list[str]:
    return [
        str(venv.python),
        str(root / spec.runner_script),
        "--suite",
        spec.suite,
        "--device",
        job.runtime.device,
        "--ngpu",
        str(job.runtime.ranks),
        "--test_name",
        spec.case_id,
        "--skip-install",
        "--forbid-update-golden",
        "--output-dir",
        str(output_dir / "integration"),
        *spec.runner_args,
    ]


def prune_artifacts(output_dir: Path) -> None:
    """Drop transient model/event output before uploading CI artifacts."""
    integration = output_dir / "integration"
    if not integration.is_dir():
        return
    for name in TRANSIENT_DIRS:
        for path in integration.glob(f"*/*/{name}"):
            if not path.is_dir():
                continue
            shutil.rmtree(path, ignore_errors=True)
            print(f"Pruned transient CI artifact: {path}", flush=True)
