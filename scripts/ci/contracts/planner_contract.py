"""Execution plan contracts emitted by the CI planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..tools.json_io import JsonDecodeError, content_digest, decode, read_json
from .base import (
    DEVICES,
    Contract,
    ContractError,
    Diagnostic,
    GitSource,
    SCHEMA_VERSION,
)
from .image_contract import ResolvedImage

JOB_KINDS = ("pytest", "integration")
PLANNER_OUTPUT_STATUSES = ("ready", "failed")
CHANGE_MODES = ("code", "docs_only", "image_only", "mixed")


@dataclass(frozen=True)
class VenvSpec(Contract):
    name: str
    python: str
    requirements: tuple[str, ...]
    overrides: tuple[str, ...]
    extends: str | None


@dataclass(frozen=True)
class Runtime(Contract):
    device: str
    target: str
    ranks: int
    runner: str
    image: ResolvedImage
    venv: VenvSpec


@dataclass(frozen=True)
class PytestSpec(Contract):
    suite: str
    test_files: tuple[str, ...]
    device_marker: str


@dataclass(frozen=True)
class IntegrationSpec(Contract):
    suite: str
    case_id: str
    framework: str
    runner_script: str
    runner_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class TestTask(Contract):
    id: str
    timeout_minutes: int
    spec: PytestSpec | IntegrationSpec


@dataclass(frozen=True)
class Job(Contract):
    id: str
    kind: str
    allowed_skip_reasons: tuple[str, ...]
    timeout_minutes: int
    runtime: Runtime
    tasks: tuple[TestTask, ...]


@dataclass(frozen=True)
class ExecutionPlan(Contract):
    plan_id: str
    ci_type: str
    change_mode: str
    git_source: GitSource
    jobs: tuple[Job, ...]


@dataclass(frozen=True)
class PlannerOutput(Contract):
    schema_version: int
    status: str
    errors: tuple[Diagnostic, ...] = ()
    plan: ExecutionPlan | None = None


def plan_digest(plan: ExecutionPlan) -> str:
    """Return the digest that must match ``plan.plan_id``."""
    return content_digest({
        key: value for key, value in plan.to_dict().items()
        if key != "plan_id"
    })


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path} {message}")


def _validate_plan(plan: ExecutionPlan) -> None:
    expected = plan_digest(plan)
    if plan.plan_id != expected:
        _fail("plan.plan_id", f"does not match contents; expected {expected}")
    if plan.change_mode not in CHANGE_MODES:
        _fail("plan.change_mode", f"has unknown mode {plan.change_mode!r}")
    for index, job in enumerate(plan.jobs):
        if job.kind not in JOB_KINDS:
            _fail(f"plan.jobs[{index}].kind", f"has unknown kind {job.kind!r}")
        if job.timeout_minutes < 1:
            _fail(f"plan.jobs[{index}].timeout_minutes", "must be >= 1")
        if job.runtime.device not in DEVICES:
            _fail(f"plan.jobs[{index}].runtime.device",
                  f"has unknown device {job.runtime.device!r}")
        if job.runtime.image.target != job.runtime.target:
            _fail(f"plan.jobs[{index}].runtime.image.target",
                  "must match runtime target")
        expected_spec = PytestSpec if job.kind == "pytest" else IntegrationSpec
        for task_index, task in enumerate(job.tasks):
            path = f"plan.jobs[{index}].tasks[{task_index}].spec"
            if not isinstance(task.spec, expected_spec):
                _fail(path, f"does not match {job.kind} kind")


def _validate_output(output: PlannerOutput) -> None:
    if output.schema_version != SCHEMA_VERSION:
        _fail("planner_output.schema_version",
              f"is {output.schema_version}, expected {SCHEMA_VERSION}")
    if output.status not in PLANNER_OUTPUT_STATUSES:
        _fail("planner_output.status", f"has unknown status {output.status!r}")
    if output.status == "ready" and (output.plan is None or output.errors):
        _fail("planner_output", "status=ready requires a plan and no errors")
    if output.status == "failed" and not output.errors:
        _fail("planner_output", "status=failed requires at least one error")
    if output.plan is not None:
        _validate_plan(output.plan)


def load_planner_output(path: Path) -> PlannerOutput:
    """Load and verify a planner output, including failed plans."""
    try:
        output = decode(PlannerOutput, read_json(path), "planner output")
    except (OSError, JsonDecodeError) as exc:
        raise ContractError(f"cannot read planner output {path}: {exc}") from exc
    _validate_output(output)
    return output
