#!/usr/bin/env python3
"""Execute one planned CI job and write a self-contained JobResult."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ... import repository_root
from ...contracts.base import ContractError, SCHEMA_VERSION
from ...contracts.executor_contract import (
    JOB_RESULT_FILENAME,
    Failure,
    JobResult,
    TaskEvidence,
    TaskResult,
)
from ...contracts.planner_contract import (
    ExecutionPlan,
    IntegrationSpec,
    Job,
    PytestSpec,
    TestTask,
    load_planner_output,
)
from ...tools.json_io import canonical_json, write_json
from . import integration_job, pytest_job
from .process import cleanup_environment, run_stream, timeout_budget
from .runtime.device_probe import probe_command
from .runtime.venv_manager import Venv, VenvManager


class ExecutorError(ValueError):
    """Raised when a requested job cannot be executed."""


PROBE_TIMEOUT_SECONDS = 300


def _task_result(
    task: TestTask,
    status: str,
    exit_code: int,
    *,
    evidence: TaskEvidence | None = None,
    failure: Failure | None = None,
) -> TaskResult:
    return TaskResult(
        id=task.id,
        status=status,
        exit_code=exit_code,
        evidence=evidence,
        failure=failure,
    )


def _job_result(
    plan: ExecutionPlan,
    job: Job,
    *,
    status: str,
    duration_seconds: int,
    tasks: tuple[TaskResult, ...] = (),
    skip_reason: str | None = None,
    skip_message: str | None = None,
    failure: Failure | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> JobResult:
    return JobResult(
        schema_version=SCHEMA_VERSION,
        plan_id=plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status=status,
        duration_seconds=duration_seconds,
        runtime=job.runtime,
        tasks=tasks,
        skip_reason=skip_reason,
        skip_message=skip_message,
        failure=failure,
        artifact_paths=artifact_paths or {},
    )


def _provisional_result(
    plan: ExecutionPlan,
    job: Job,
    *,
    log_path: Path,
) -> JobResult:
    return _job_result(
        plan,
        job,
        status="infra_error",
        duration_seconds=0,
        failure=Failure(
            code="executor-incomplete",
            message="executor did not finalize JobResult",
        ),
        artifact_paths={"log": log_path.name} if log_path.is_file() else {},
    )


def _write_result(result: JobResult, output_dir: Path) -> None:
    write_json(output_dir / JOB_RESULT_FILENAME, result.to_dict())


def _load_ready_plan(path: Path) -> ExecutionPlan:
    try:
        output = load_planner_output(path)
    except ContractError as exc:
        raise ExecutorError(str(exc)) from exc
    if output.status != "ready" or output.plan is None:
        details = "; ".join(issue.message for issue in output.errors)
        raise ExecutorError(f"planner output is not ready: {details}")
    return output.plan


def _job(plan: ExecutionPlan, job_id: str) -> Job:
    for job in plan.jobs:
        if job.id == job_id:
            return job
    raise ExecutorError(f"job {job_id!r} is not in plan {plan.plan_id}")


def _probe_environment(
    venv: Venv,
    job: Job,
    *,
    root: Path,
    log_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool]:
    return run_stream(
        probe_command(venv.python, job.runtime.device),
        cwd=root,
        environment=venv.environment(),
        log_path=log_path,
        timeout_seconds=timeout_seconds,
    )


def _prepare_environment(
    job: Job,
    *,
    root: Path,
    venv_root: Path,
    log_path: Path,
    deadline: float,
) -> tuple[Venv | None, str, Failure | None]:
    """Install the planned venv and probe the planned runtime."""
    try:
        manager = VenvManager(
            root, venv_root, job.runtime.device, log_path, job.runtime.venv
        )
        venv = manager.ensure()
        probe_limit, probe_limited = timeout_budget(deadline, PROBE_TIMEOUT_SECONDS)
        if probe_limit == 0:
            return venv, "failed", _job_timeout_failure(job)
        probe_rc, probe_timeout = _probe_environment(
            venv,
            job,
            root=root,
            log_path=log_path,
            timeout_seconds=probe_limit,
        )
        if probe_rc == 0:
            return venv, "passed", None
        if probe_timeout and probe_limited:
            return venv, "failed", _job_timeout_failure(job)
        return venv, "infra_error", Failure(
            code="environment-probe-timeout" if probe_timeout else "environment-probe",
            message=f"environment probe failed with rc={probe_rc}",
        )
    except Exception as exc:  # noqa: BLE001 - every setup failure must produce a result
        return None, "infra_error", Failure(
            code="environment-setup",
            message=f"{type(exc).__name__}: {exc}",
        )


def _task_command(
    venv: Venv,
    job: Job,
    task: TestTask,
    *,
    index: int,
    root: Path,
    output_dir: Path,
) -> tuple[list[str], dict[str, str], Path | None]:
    if isinstance(task.spec, PytestSpec):
        junit = pytest_job.report_path(output_dir, index)
        return (
            pytest_job.command(venv, task.spec, junit),
            venv.environment(),
            junit,
        )
    if isinstance(task.spec, IntegrationSpec):
        return (
            integration_job.command(
                venv,
                task.spec,
                job,
                root=root,
                output_dir=output_dir,
            ),
            venv.environment(),
            None,
        )
    raise ExecutorError(f"unsupported task spec for {task.id}")


def _task_failure(
    job: Job,
    task: TestTask,
    *,
    exit_code: int,
    timed_out: bool,
    deadline: float,
) -> Failure:
    if timed_out and time.monotonic() >= deadline:
        return _job_timeout_failure(job)
    code = "task-timeout" if timed_out else "task-failed"
    message = (f"task timed out after {task.timeout_minutes} minutes"
               if timed_out else f"task exited with code {exit_code}")
    return Failure(code=code, message=message)


def _execute_tasks(
    venv: Venv,
    job: Job,
    *,
    root: Path,
    output_dir: Path,
    log_path: Path,
    deadline: float,
) -> tuple[str, list[TaskResult], list[Path]]:
    status: str = "passed"
    results: list[TaskResult] = []
    junit_paths: list[Path] = []
    for index, task in enumerate(job.tasks):
        limit, _ = timeout_budget(deadline, task.timeout_minutes * 60)
        if limit == 0:
            results.append(_task_result(
                task, "failed", 124, failure=_job_timeout_failure(job)
            ))
            return "failed", results, junit_paths
        task_started = time.monotonic()
        junit: Path | None = None
        task_status = "passed"
        task_failure: Failure | None = None
        try:
            command, environment, junit = _task_command(
                venv, job, task, index=index, root=root, output_dir=output_dir
            )
            if junit is not None:
                junit_paths.append(junit)
            exit_code, timed_out = run_stream(
                command,
                cwd=root,
                environment=environment,
                log_path=log_path,
                timeout_seconds=limit,
            )
            if exit_code != 0:
                task_status = "failed"
                task_failure = _task_failure(
                    job,
                    task,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    deadline=deadline,
                )
        except KeyboardInterrupt:
            task_status, exit_code = "cancelled", 130
            task_failure = Failure(code="cancelled", message="cancelled")
        except Exception as exc:  # noqa: BLE001 - report task infrastructure failure
            task_status, exit_code = "infra_error", 2
            task_failure = Failure(
                code="task-setup",
                message=f"{type(exc).__name__}: {exc}",
            )
        evidence = pytest_job.evidence(junit) if junit is not None and junit.is_file() else None
        results.append(_task_result(
            task, task_status, exit_code, evidence=evidence, failure=task_failure
        ))
        if task_status != "passed":
            status = task_status
            elapsed = int(time.monotonic() - task_started)
            print(f"task {task.id} finished in {elapsed}s with status={task_status}",
                  flush=True)
            break
    return status, results, junit_paths


def _job_timeout_failure(job: Job) -> Failure:
    return Failure(
        code="job-timeout",
        message=f"job budget exhausted after {job.timeout_minutes} minutes",
    )


def _write_versions(venv: Venv, output: Path, root: Path) -> None:
    code = (
        "import importlib.metadata as m\n"
        "for name in ('torch', 'transformers', 'peft', 'numpy', "
        "'llamafactory', 'swift'):\n"
        "    try: print(f'{name} = {m.version(name)}')\n"
        "    except m.PackageNotFoundError: pass\n"
    )
    result = subprocess.run(
        [str(venv.python), "-c", code],
        cwd=root,
        env=venv.environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout.strip() + "\n", encoding="utf-8")


def execute_job(
    plan: ExecutionPlan,
    job_id: str,
    *,
    output_dir: Path,
    root: Path | None = None,
) -> JobResult:
    """Execute one Job from a ready ExecutionPlan."""
    job = _job(plan, job_id)
    root = root or repository_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "turbo-ci.log"
    merged_junit = output_dir / "pytest.xml"
    versions_path = output_dir / "versions.txt"
    venv_root = Path(os.environ.get(
        "CI_VENV_ROOT", Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "turbo-ci-venv"
    ))
    log_path.write_text("", encoding="utf-8")
    started = time.monotonic()
    provisional = _provisional_result(plan, job, log_path=log_path)
    _write_result(provisional, output_dir)
    deadline = started + job.timeout_minutes * 60
    task_results: list[TaskResult] = []
    junit_paths: list[Path] = []

    venv, status, failure = _prepare_environment(
        job,
        root=root,
        venv_root=venv_root,
        log_path=log_path,
        deadline=deadline,
    )
    if status == "passed" and venv is not None:
        status, task_results, junit_paths = _execute_tasks(
            venv,
            job,
            root=root,
            output_dir=output_dir,
            log_path=log_path,
            deadline=deadline,
        )
    if venv is not None and job.runtime.device == "npu":
        cleanup_environment(log_path)

    integration_job.prune_artifacts(output_dir)
    pytest_job.merge_reports(junit_paths, merged_junit)
    if venv is not None:
        try:
            _write_versions(venv, versions_path, root)
        except OSError as exc:
            print(f"::warning::could not write versions: {exc}", file=sys.stderr)

    artifact_paths = {"log": log_path.name}
    if job.kind == "pytest":
        artifact_paths["junit"] = merged_junit.name
    if job.kind == "integration" and (output_dir / "integration").is_dir():
        artifact_paths["integration"] = "integration"
    if versions_path.is_file():
        artifact_paths["versions"] = versions_path.name
    result = _job_result(
        plan,
        job,
        status=status,
        duration_seconds=int(time.monotonic() - started),
        tasks=tuple(task_results),
        failure=failure,
        artifact_paths=artifact_paths,
    )
    _write_result(result, output_dir)
    return result


def record_skip(
    plan: ExecutionPlan,
    job_id: str,
    reason: str,
    *,
    output_dir: Path,
    message: str = "",
) -> JobResult:
    """Write a structured skipped result for a platform-level skip."""
    job = _job(plan, job_id)
    if reason not in job.allowed_skip_reasons:
        raise ExecutorError(f"job {job.id!r} does not allow skip reason {reason!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _job_result(
        plan,
        job,
        status="skipped",
        duration_seconds=0,
        skip_reason=reason,
        skip_message=message or reason,
    )
    _write_result(result, output_dir)
    return result


def render_summary(result: JobResult) -> str:
    lines = [
        f"## CI Job: {result.status}",
        "",
        f"- job: `{result.job_id}`",
        f"- plan: `{result.plan_id}`",
        f"- duration: {result.duration_seconds}s",
    ]
    if result.kind == "pytest":
        evidence = [task.evidence for task in result.tasks if task.evidence is not None]
        lines.append(
            f"- pytest: {sum(item.tests_total for item in evidence)} tests, "
            f"{sum(item.tests_failed for item in evidence)} failed, "
            f"{sum(item.tests_errors for item in evidence)} errors, "
            f"{sum(item.tests_skipped for item in evidence)} skipped"
        )
    elif result.kind == "integration" and "integration" in result.artifact_paths:
        lines.append(f"- integration: {result.artifact_paths['integration']}")
    failure = result.failure
    task_id = ""
    if failure is None:
        failed_task = next((task for task in result.tasks if task.failure is not None), None)
        if failed_task is not None:
            failure = failed_task.failure
            task_id = failed_task.id
    if failure is not None:
        task = f" ({task_id})" if task_id else ""
        lines.append(f"- failure: `{failure.code}`{task}: {failure.message}")
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="ExecutionPlan or ready PlannerOutput JSON")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", default="", help="JobResult JSON path")
    parser.add_argument("--root", default=None)
    parser.add_argument("--summary", default="")
    parser.add_argument("--record-skip", default="", metavar="REASON",
                        help="write a skipped JobResult instead of executing")
    parser.add_argument("--skip-message", default="")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else repository_root()
    try:
        plan = _load_ready_plan(Path(args.plan))
        output_dir = (Path(args.output).resolve().parent if args.output
                      else Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "ci-job")
        if args.record_skip:
            result = record_skip(
                plan,
                args.job_id,
                args.record_skip,
                output_dir=output_dir,
                message=args.skip_message,
            )
        else:
            result = execute_job(plan, args.job_id, output_dir=output_dir, root=root)
    except (ContractError, ExecutorError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2
    if args.output:
        write_json(Path(args.output), result.to_dict())
    else:
        print(canonical_json(result.to_dict()))
    if args.summary:
        with Path(args.summary).open("a", encoding="utf-8") as output:
            output.write(render_summary(result))
    return 0 if result.status in {"passed", "skipped"} else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
