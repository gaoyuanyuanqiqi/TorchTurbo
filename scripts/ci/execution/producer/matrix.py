"""Build GitHub Actions matrices from execution plans."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ...contracts.planner_contract import ExecutionPlan, Job, PytestSpec
from .config import CiConfig
from .planner import slugify


def matrix_outputs(
    plan: ExecutionPlan, config: CiConfig
) -> dict[str, list[dict[str, Any]]]:
    """Build platform matrix entries without embedding full job payloads."""
    output = {f"{device}_jobs": [] for device in ("cpu", "npu", "gpu")}
    output["container_jobs"] = []
    images_by_target = Counter(image.target for image in config.images.values())
    for job in plan.jobs:
        runner = config.runner_named(job.runtime.runner)
        item = {
            "job_id": job.id,
            "display_name": _display_name(job, images_by_target),
            "image_ref": job.runtime.image.ref,
            "device": job.runtime.device,
            "target": job.runtime.target,
            "runner": runner.runner,
            "container_options": runner.container_options,
            "timeout_minutes": job.timeout_minutes,
        }
        output[f"{job.runtime.device}_jobs"].append(item)
        if job.runtime.device != "cpu":
            output["container_jobs"].append(item)
    return output


def github_outputs(plan: ExecutionPlan, config: CiConfig) -> dict[str, str]:
    """Return matrix values and counts ready for GITHUB_OUTPUT."""
    values: dict[str, str] = {}
    for key, items in matrix_outputs(plan, config).items():
        values[key] = json.dumps(
            items, ensure_ascii=False, separators=(",", ":")
        )
        values[f"{key[:-1]}_count"] = str(len(items))
    return values


def _display_name(job: Job, images_by_target: Counter[str]) -> str:
    """Return a compact human-facing name without changing the job contract."""
    if job.kind != "pytest":
        return job.id
    spec = next(
        (task.spec for task in job.tasks if isinstance(task.spec, PytestSpec)),
        None,
    )
    if spec is None:
        return job.id
    image_suffix = (
        "" if images_by_target[job.runtime.target] == 1
        else f"-{job.runtime.image.image_id}"
    )
    return slugify(
        f"pytest-{spec.suite}-{job.runtime.target}{image_suffix}-{job.runtime.venv.name}"
    )
