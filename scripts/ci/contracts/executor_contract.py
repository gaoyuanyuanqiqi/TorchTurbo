"""Job result contracts emitted by the CI executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..tools.json_io import JsonDecodeError, decode, read_json
from .base import Contract, ContractError, SCHEMA_VERSION
from .planner_contract import JOB_KINDS, Runtime

JOB_RESULT_FILENAME = "job-result.json"
JOB_STATUSES = ("passed", "failed", "skipped", "cancelled", "infra_error")
TASK_STATUSES = ("passed", "failed", "cancelled", "infra_error")


@dataclass(frozen=True)
class TaskEvidence(Contract):
    tests_total: int = 0  # Number of pytest tests collected.
    tests_failed: int = 0  # Number of failed pytest tests.
    tests_errors: int = 0  # Number of pytest collection or runtime errors.
    tests_skipped: int = 0  # Number of skipped pytest tests.


@dataclass(frozen=True)
class Failure(Contract):
    code: str  # Machine-readable failure code.
    message: str  # Human-readable failure detail.


@dataclass(frozen=True)
class TaskResult(Contract):
    id: str  # Planned test task ID.
    status: str  # Task status reported by the executor.
    exit_code: int  # Process exit code for the task command.
    evidence: TaskEvidence | None = None  # Actual pytest evidence for this task.
    failure: Failure | None = None  # Task-level failure detail.


@dataclass(frozen=True)
class JobResult(Contract):
    schema_version: int  # Contract schema version.
    plan_id: str  # ExecutionPlan ID this result belongs to.
    job_id: str  # Planned job ID this result belongs to.
    kind: str  # Job kind: pytest or integration.
    status: str  # Final job status.
    duration_seconds: int  # Total execution time in seconds.
    runtime: Runtime  # Runtime copied from the planned job.
    tasks: tuple[TaskResult, ...]  # Per-task execution results.
    skip_reason: str | None = None  # Allowed skip reason for skipped jobs.
    skip_message: str | None = None  # Human-readable skip detail.
    failure: Failure | None = None  # Failure outside any planned task.
    artifact_paths: dict[str, str] = field(default_factory=dict)  # Artifact paths.


def _fail(path: str, message: str) -> None:
    raise ContractError(f"{path} {message}")


def _validate_failure(failure: Failure, path: str) -> None:
    for name, value in (("code", failure.code), ("message", failure.message)):
        if not value.strip():
            _fail(f"{path}.{name}", "must not be empty")


def _validate_task(kind: str, task: TaskResult) -> None:
    path = f"job_result.tasks.{task.id}"
    if task.status not in TASK_STATUSES:
        _fail(f"{path}.status", f"has unknown status {task.status!r}")
    if task.evidence is not None and any(value < 0 for value in (
        task.evidence.tests_total,
        task.evidence.tests_failed,
        task.evidence.tests_errors,
        task.evidence.tests_skipped,
    )):
        _fail(f"{path}.evidence", "must be >= 0")
    if task.failure is not None:
        _validate_failure(task.failure, f"{path}.failure")
    if task.status == "passed":
        if task.failure is not None:
            _fail(f"{path}.failure", "must be empty for passed tasks")
        if kind == "pytest" and task.evidence is None:
            _fail(f"{path}.evidence", "passed pytest tasks require evidence")
    elif task.failure is None:
        _fail(f"{path}.failure", "non-passed tasks require failure detail")


def _validate_job_result(result: JobResult) -> None:
    if result.schema_version != SCHEMA_VERSION:
        _fail("job_result.schema_version", f"expected {SCHEMA_VERSION}")
    if result.kind not in JOB_KINDS:
        _fail("job_result.kind", f"has unknown kind {result.kind!r}")
    if result.status not in JOB_STATUSES:
        _fail("job_result.status", f"has unknown status {result.status!r}")
    if result.status == "skipped":
        if result.skip_reason is None:
            _fail("job_result", "status=skipped requires skip.reason")
        if result.tasks:
            _fail("job_result", "skipped jobs must not contain task results")
    elif result.skip_reason is not None or result.skip_message is not None:
        _fail("job_result", "skip fields are only valid for skipped status")
    for name, path in result.artifact_paths.items():
        if not name or not isinstance(path, str) or not path:
            _fail("job_result.artifact_paths", "must be a non-empty string")

    if result.status in {"passed", "skipped"} and result.failure is not None:
        _fail("job_result.failure", "must be empty for passed or skipped jobs")
    if result.failure is not None:
        _validate_failure(result.failure, "job_result.failure")
    for task in result.tasks:
        _validate_task(result.kind, task)
    has_failure = result.failure is not None or any(
        task.failure is not None for task in result.tasks
    )
    if (
        result.status in {"failed", "cancelled", "infra_error"}
        and not has_failure
    ):
        _fail("job_result", "non-passed jobs require failure detail")


def load_job_result(path: Path) -> JobResult:
    """Load one JobResult JSON file."""
    try:
        result = decode(JobResult, read_json(path), "job result")
    except (OSError, JsonDecodeError) as exc:
        raise ContractError(f"cannot read job result {path}: {exc}") from exc
    _validate_job_result(result)
    return result
