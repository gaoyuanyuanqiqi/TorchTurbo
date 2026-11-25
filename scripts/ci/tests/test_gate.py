from __future__ import annotations

from dataclasses import replace

from scripts.ci.contracts.base import SCHEMA_VERSION, Diagnostic
from scripts.ci.contracts.executor_contract import (
    Failure,
    JobResult,
    TaskEvidence,
    TaskResult,
)
from scripts.ci.contracts.planner_contract import PytestSpec, load_planner_output
from scripts.ci.execution.finalization import gate
from scripts.ci.execution.finalization.gate import decide_plan, decide_planner_output
from scripts.ci.execution.producer.cli import main
from scripts.ci.execution.producer.planner import (
    build_plan,
    build_execution_plan,
    build_planner_output,
)
from scripts.ci.tools.json_io import read_json, write_json


def _result_for(plan, job, status: str = "passed") -> JobResult:
    tasks = tuple(
        TaskResult(
            id=task.id,
            status="passed" if status == "passed" else "failed",
            exit_code=0 if status == "passed" else 1,
            evidence=(
                TaskEvidence(tests_total=1)
                if job.kind == "pytest"
                else None
            ),
            failure=(
                Failure(
                    code="task-failed",
                    message="task failed",
                )
                if status != "passed"
                else None
            ),
        )
        for task in job.tasks
    )
    return JobResult(
        schema_version=SCHEMA_VERSION,
        plan_id=plan.plan_id,
        job_id=job.id,
        kind=job.kind,
        status=status,
        duration_seconds=60,
        runtime=job.runtime,
        tasks=tasks,
        artifact_paths=(
            {"integration": "integration"}
            if job.kind == "integration"
            else {}
        ),
    )


def _results(plan) -> list[JobResult]:
    return [_result_for(plan, job) for job in plan.jobs]


def _plan_with_allowed_skip(plan, job_id: str, reason: str):
    jobs = tuple(
        replace(job, allowed_skip_reasons=(reason,)) if job.id == job_id else job
        for job in plan.jobs
    )
    return build_execution_plan(
        ci_type=plan.ci_type,
        change_mode=plan.change_mode,
        git_source=plan.git_source,
        jobs=jobs,
    )


def test_all_jobs_pass(pr_plan) -> None:
    decision = decide_plan(pr_plan, _results(pr_plan))
    assert decision.status == "passed"
    assert decision.summary["expected_jobs"] == 6
    assert decision.summary["passed"] == 6
    assert decision.failures == ()


def test_docs_only_plan_passes_without_results(config, facts, source) -> None:
    plan = build_plan(config, facts, "pull_request", source, change_mode="docs_only")
    decision = decide_plan(plan, [])
    assert decision.status == "passed"
    assert decision.summary == {
        "expected_jobs": 0,
        "reported_jobs": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "missing": 0,
        "unexpected": 0,
    }
    assert decision.failures == ()


def test_missing_result_fails(pr_plan) -> None:
    decision = decide_plan(pr_plan, _results(pr_plan)[:-1])
    assert decision.status == "failed"
    assert "missing-result" in {failure["code"] for failure in decision.failures}


def test_incomplete_executor_result_is_infra_error(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(
        i for i, job in enumerate(pr_plan.jobs)
        if job.id == "pytest-core-npu-npu-a3-cann90-default"
    )
    results[index] = replace(
        results[index],
        status="infra_error",
        tasks=(),
        failure=Failure(
            code="executor-incomplete",
            message="executor did not finalize JobResult",
        ),
    )
    decision = decide_plan(pr_plan, results)
    codes = {failure["code"] for failure in decision.failures}
    assert decision.status == "failed"
    assert "job-infra_error" in codes
    assert "missing-result" not in codes
    assert decision.summary["failed"] == 1


def test_duplicate_result_fails(pr_plan) -> None:
    results = _results(pr_plan)
    results.append(results[0])
    decision = decide_plan(pr_plan, results)
    assert decision.status == "failed"
    assert "duplicate-result" in {failure["code"] for failure in decision.failures}


def test_extra_result_fails(pr_plan) -> None:
    results = _results(pr_plan)
    foreign = replace(results[0], job_id="unexpected-job")
    results.append(foreign)
    decision = decide_plan(pr_plan, results)
    assert "unexpected-result" in {failure["code"] for failure in decision.failures}


def test_skip_reason_must_be_allowed(pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.runtime.device == "npu")
    allowed = _plan_with_allowed_skip(pr_plan, job.id, "maintenance")
    results = _results(allowed)
    index = next(i for i, result in enumerate(results) if result.job_id == job.id)
    results[index] = replace(
        results[index],
        status="skipped",
        tasks=(),
        failure=None,
        skip_reason="maintenance",
        skip_message="window",
    )
    assert decide_plan(allowed, results).status == "passed"

    results[index] = replace(results[index], skip_reason="free-form")
    decision = decide_plan(allowed, results)
    assert decision.status == "failed"
    assert "skip-reason" in {failure["code"] for failure in decision.failures}


def test_skipped_result_with_tasks_fails(pr_plan) -> None:
    job = next(item for item in pr_plan.jobs if item.runtime.device == "npu")
    allowed = _plan_with_allowed_skip(pr_plan, job.id, "maintenance")
    results = _results(allowed)
    index = next(i for i, result in enumerate(results) if result.job_id == job.id)
    results[index] = replace(
        results[index],
        status="skipped",
        failure=None,
        skip_reason="maintenance",
        skip_message="window",
    )
    decision = decide_plan(allowed, results)
    assert "skip-tasks" in {failure["code"] for failure in decision.failures}


def test_failed_result_without_detail_fails(pr_plan) -> None:
    results = _results(pr_plan)
    results[0] = replace(results[0], status="failed", tasks=(), failure=None)
    decision = decide_plan(pr_plan, results)
    assert "failed-detail" in {failure["code"] for failure in decision.failures}


def test_pytest_without_tests_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    tasks = tuple(
        replace(
            task,
            evidence=replace(task.evidence, tests_total=0)
            if task.evidence is not None else None,
        )
        for task in results[index].tasks
    )
    results[index] = replace(
        results[index],
        tasks=tasks,
    )
    decision = decide_plan(pr_plan, results)
    assert "pytest-no-tests" in {failure["code"] for failure in decision.failures}


def test_pytest_all_skipped_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    tasks = tuple(
        replace(
            task,
            evidence=replace(
                task.evidence,
                tests_total=2,
                tests_skipped=2,
            )
            if task.evidence is not None else None,
        )
        for task in results[index].tasks
    )
    results[index] = replace(results[index], tasks=tasks)

    decision = decide_plan(pr_plan, results)

    assert "pytest-all-skipped" in {failure["code"] for failure in decision.failures}


def test_pytest_failures_fail(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "pytest")
    tasks = tuple(
        replace(
            task,
            evidence=replace(task.evidence, tests_failed=1)
            if task.evidence is not None else None,
        )
        for task in results[index].tasks
    )
    results[index] = replace(
        results[index],
        tasks=tasks,
    )
    decision = decide_plan(pr_plan, results)
    assert "pytest-failures" in {failure["code"] for failure in decision.failures}


def test_integration_without_artifact_fails(pr_plan) -> None:
    results = _results(pr_plan)
    index = next(i for i, result in enumerate(results) if result.kind == "integration")
    results[index] = replace(
        results[index],
        artifact_paths={},
    )
    decision = decide_plan(pr_plan, results)
    assert "integration-artifact" in {failure["code"] for failure in decision.failures}


def _plan_with_jobs(pr_plan, jobs):
    return build_execution_plan(
        ci_type=pr_plan.ci_type,
        change_mode=pr_plan.change_mode,
        git_source=pr_plan.git_source,
        jobs=jobs,
    )


def _route_job(source_job, source_task, runtime, suffix: str):
    return replace(
        source_job,
        id=f"{source_job.id}-{suffix}",
        runtime=runtime,
        tasks=(replace(source_task, id=f"{source_job.id}-{suffix}/run"),),
    )


def test_same_pytest_file_on_distinct_targets_is_allowed(pr_plan) -> None:
    source_job = next(
        job for job in pr_plan.jobs
        if job.id == "pytest-core-npu-npu-a3-cann90-default"
    )
    source_task = source_job.tasks[0]
    assert isinstance(source_task.spec, PytestSpec)

    def target_job(target: str, suffix: str):
        runtime = replace(
            source_job.runtime,
            target=target,
            image=replace(
                source_job.runtime.image,
                image_id=f"{target}-image",
                target=target,
            ),
        )
        return _route_job(source_job, source_task, runtime, suffix)

    plan = _plan_with_jobs(
        pr_plan,
        (
            target_job("npu-a3", "a3"),
            target_job("npu-a5", "a5"),
        ),
    )
    assert decide_plan(plan, _results(plan)).status == "passed"


def test_same_pytest_file_on_distinct_images_is_allowed(pr_plan) -> None:
    source_job = next(
        job for job in pr_plan.jobs
        if job.id == "pytest-core-npu-npu-a3-cann90-default"
    )
    source_task = source_job.tasks[0]
    assert isinstance(source_task.spec, PytestSpec)

    def image_job(image_id: str, suffix: str):
        runtime = replace(
            source_job.runtime,
            image=replace(source_job.runtime.image, image_id=image_id),
        )
        return _route_job(source_job, source_task, runtime, suffix)

    plan = _plan_with_jobs(
        pr_plan,
        (
            image_job("npu-a3-cann90-image", "cann90"),
            image_job("npu-a3-cann92-image", "cann92"),
        ),
    )
    assert decide_plan(plan, _results(plan)).status == "passed"


def test_same_pytest_file_on_same_image_route_fails(pr_plan) -> None:
    source_job = next(
        job for job in pr_plan.jobs
        if job.id == "pytest-core-npu-npu-a3-cann90-default"
    )
    source_task = source_job.tasks[0]
    assert isinstance(source_task.spec, PytestSpec)

    def image_job(suffix: str):
        runtime = replace(
            source_job.runtime,
            image=replace(
                source_job.runtime.image,
                image_id="npu-a3-cann90-image",
            ),
        )
        return _route_job(source_job, source_task, runtime, suffix)

    plan = _plan_with_jobs(
        pr_plan,
        (
            image_job("first"),
            image_job("second"),
        ),
    )
    decision = decide_plan(plan, _results(plan))
    assert decision.status == "failed"
    assert "duplicate-route" in {failure["code"] for failure in decision.failures}


def test_failed_planner_output_fails(pr_plan) -> None:
    output = build_planner_output(errors=())
    decision = decide_planner_output(output)
    assert decision.status == "failed"
    assert decision.failures[0]["code"] == "planner-output-missing-plan"


def test_planner_issue_path_is_not_a_task_id(pr_plan) -> None:
    output = build_planner_output(
        errors=(Diagnostic(
            code="lint",
            location="scripts/ci/x.py",
            message="bad",
        ),),
    )
    failure = decide_planner_output(output).failures[0]
    assert failure["path"] == "scripts/ci/x.py"
    assert "task_id" not in failure


def test_failed_output_cli_reports_lint_failure(tmp_path) -> None:
    path = tmp_path / "plan.json"
    assert main([
        "failed-output",
        "--output", str(path),
        "--code", "lint-failed",
        "--location", "scripts/ci",
        "--message", "lint failed; see Lint-Plan job",
    ]) == 0
    output = load_planner_output(path)
    assert output.status == "failed"
    assert output.plan is None
    decision = decide_planner_output(output)
    assert decision.status == "failed"
    assert decision.failures[0]["code"] == "planner-output-lint-failed"


def _write_cli_inputs(tmp_path, plan, results):
    plan_path = tmp_path / "plan.json"
    results_dir = tmp_path / "results"
    write_json(plan_path, build_planner_output(
        plan=plan,
    ).to_dict())
    for result in results:
        write_json(results_dir / result.job_id / "job-result.json", result.to_dict())
    write_json(results_dir / "diagnostics" / "environment.json", {"torch_ok": True})
    return plan_path, results_dir


def test_cli_aggregates_results_and_writes_decision(tmp_path, capsys, pr_plan) -> None:
    plan_path, results_dir = _write_cli_inputs(tmp_path, pr_plan, _results(pr_plan))
    decision_path = tmp_path / "decision.json"
    summary_path = tmp_path / "summary.md"

    assert gate.main([
        "--plan", str(plan_path),
        "--results", str(results_dir),
        "--output", str(decision_path),
        "--summary", str(summary_path),
    ]) == 0

    value = read_json(decision_path)
    assert value["status"] == "passed"
    assert value["summary"]["reported_jobs"] == len(pr_plan.jobs)
    assert "## CI Gate: passed" in summary_path.read_text(encoding="utf-8")
    assert capsys.readouterr().out == ""


def test_cli_failed_result_returns_one(tmp_path, pr_plan) -> None:
    results = _results(pr_plan)
    results[0] = _result_for(pr_plan, pr_plan.jobs[0], status="failed")
    plan_path, results_dir = _write_cli_inputs(tmp_path, pr_plan, results)
    decision_path = tmp_path / "decision.json"
    assert gate.main([
        "--plan", str(plan_path),
        "--results", str(results_dir),
        "--output", str(decision_path),
    ]) == 1
    value = read_json(decision_path)
    assert value["status"] == "failed"
    assert "job-failed" in {item["code"] for item in value["failures"]}
