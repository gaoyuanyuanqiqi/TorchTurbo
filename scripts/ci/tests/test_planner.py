from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

from catalog import Catalog
from contracts import ImageRef, Plan, load_plan_artifact, plan_digest, write_json
from planner import (
    artifact_for,
    build,
    main,
    matrix_outputs,
    source_for,
    validate_plan,
)
from policy import Policy


def _with_plan_id(plan: Plan) -> Plan:
    return replace(plan, plan_id=plan_digest(plan))


def test_pull_request_scope(catalog: Catalog, pr_policy: Policy, pr_plan: Plan) -> None:
    assert Counter(job.kind for job in pr_plan.jobs) == {
        "pytest": 2,
        "model_e2e": 2,
    }
    assert pr_plan.coverage["pytest"] == {
        "core": {"mode": "all", "files": 86, "routes": 88},
        "model": {"mode": "none", "files": 0, "routes": 0},
        "adapter": {"mode": "none", "files": 0, "routes": 0},
    }
    assert pr_plan.coverage["model_e2e"] == {
        "mode": "explicit",
        "cases": ["qwen3_5-lf-text-sft", "bailing_hybrid-lf-flash-sft"],
    }
    assert pr_plan.coverage["jobs"] == 4
    assert validate_plan(pr_plan, catalog, pr_policy) == []

    pytest_suites = {
        step.spec.suite
        for job in pr_plan.jobs
        if job.kind == "pytest"
        for step in job.steps
    }
    assert pytest_suites == {"core"}


def test_nightly_scope(catalog: Catalog, nightly_policy: Policy, nightly_plan: Plan) -> None:
    assert Counter(job.kind for job in nightly_plan.jobs) == {
        "pytest": 9,
        "model_e2e": 11,
    }
    assert {
        step.spec.suite
        for job in nightly_plan.jobs
        if job.kind == "pytest"
        for step in job.steps
    } == {"core", "model", "adapter"}
    assert len(nightly_plan.coverage["model_e2e"]["cases"]) == 11
    assert nightly_plan.coverage["jobs"] == 20
    assert validate_plan(nightly_plan, catalog, nightly_policy) == []


def test_cli_writes_failed_artifact_for_precondition(tmp_path: Path, repo_root: Path) -> None:
    output = tmp_path / "plan.json"
    assert main([
        "--root", str(repo_root),
        "--output", str(output),
        "--fail-code", "lint",
        "--fail-path", "scripts/ci/lint_select/lint_check.py",
        "--fail-message", "lint failed",
    ]) == 1

    artifact = load_plan_artifact(output)
    assert artifact.status == "failed"
    assert artifact.plan is None
    assert artifact.errors[0].code == "lint"


def test_plan_is_deterministic(catalog: Catalog, pr_policy: Policy, source) -> None:
    first = build(catalog, pr_policy, source)
    second = build(catalog, pr_policy, source)
    assert first.to_dict() == second.to_dict()
    assert first.plan_id == second.plan_id

    changed_source = source_for(
        catalog.root,
        repository=source.repository,
        ref=source.ref,
        sha="fedcba9876543210fedcba9876543210fedcba98",
    )
    changed = build(catalog, pr_policy, changed_source)
    assert changed.plan_id != first.plan_id


def test_matrix_adapter_only_exposes_platform_fields(
    catalog: Catalog, pr_plan: Plan
) -> None:
    matrices = matrix_outputs(pr_plan, catalog)
    assert {key: len(value) for key, value in matrices.items()} == {
        "cpu_jobs": 1,
        "npu_jobs": 3,
        "gpu_jobs": 0,
    }
    expected_fields = {
        "job_id", "image_ref", "runner_variable", "runner", "timeout_minutes",
    }
    for entries in matrices.values():
        for entry in entries:
            assert set(entry) == expected_fields
            assert "job_json" not in entry


def test_cli_matrix_outputs_match_workflow_contract(
    tmp_path: Path, repo_root: Path
) -> None:
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "github-output"
    assert main([
        "--root", str(repo_root),
        "--policy", "pull_request",
        "--output", str(plan_path),
        "--github-output", str(output_path),
    ]) == 0

    keys = {
        line.partition("=")[0]
        for line in output_path.read_text(encoding="utf-8").splitlines()
    }
    assert {
        "cpu_jobs", "cpu_job_count",
        "npu_jobs", "npu_job_count",
        "gpu_jobs", "gpu_job_count",
    } <= keys


def test_planner_failure_artifact_round_trip(
    tmp_path: Path, catalog: Catalog, source
) -> None:
    artifact = artifact_for(catalog, "missing-policy", source, ())
    assert artifact.status == "failed"
    assert artifact.plan is None
    assert [issue.code for issue in artifact.errors] == ["invalid-policy"]

    path = tmp_path / "plan-artifact.json"
    write_json(path, artifact.to_dict())
    loaded = load_plan_artifact(path)
    assert loaded.to_dict() == artifact.to_dict()


def test_ready_artifact_round_trip(tmp_path: Path, catalog: Catalog, source) -> None:
    artifact = artifact_for(catalog, "pull_request", source, ())
    assert artifact.status == "ready"
    assert artifact.plan is not None

    path = tmp_path / "plan-artifact.json"
    write_json(path, artifact.to_dict())
    loaded = load_plan_artifact(path)
    assert loaded.status == "ready"
    assert loaded.plan is not None
    assert loaded.plan.to_dict() == artifact.plan.to_dict()


def test_missing_job_is_rejected(catalog: Catalog, pr_policy: Policy, pr_plan: Plan) -> None:
    missing = "pytest-core-npu-npu-default-default"
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(job for job in pr_plan.jobs if job.id != missing),
    ))
    codes = {issue.code for issue in validate_plan(broken, catalog, pr_policy)}
    assert "missing-test" in codes


def test_tampered_image_is_rejected(catalog: Catalog, pr_policy: Policy, pr_plan: Plan) -> None:
    job = pr_plan.jobs[0]
    bad_image = ImageRef(name=job.runtime.image.name, kind=job.runtime.image.kind, ref="bad:tag")
    bad_runtime = replace(job.runtime, image=bad_image)
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=(replace(job, runtime=bad_runtime), *pr_plan.jobs[1:]),
    ))
    messages = [issue.message for issue in validate_plan(broken, catalog, pr_policy)]
    assert "image does not match Catalog" in messages


def test_tampered_venv_is_rejected(catalog: Catalog, pr_policy: Policy, pr_plan: Plan) -> None:
    job = pr_plan.jobs[0]
    bad_venv = replace(job.runtime.venv, digest="sha256:bad")
    bad_runtime = replace(job.runtime, venv=bad_venv)
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=(replace(job, runtime=bad_runtime), *pr_plan.jobs[1:]),
    ))
    messages = [issue.message for issue in validate_plan(broken, catalog, pr_policy)]
    assert "venv does not match Catalog" in messages


def test_tampered_coverage_is_rejected(catalog: Catalog, pr_policy: Policy, pr_plan: Plan) -> None:
    coverage = {
        **pr_plan.coverage,
        "pytest": {
            **pr_plan.coverage["pytest"],
            "core": {"mode": "all", "files": 1, "routes": 1},
        },
    }
    broken = _with_plan_id(replace(pr_plan, coverage=coverage))
    messages = [issue.message for issue in validate_plan(broken, catalog, pr_policy)]
    assert "expected {'mode': 'all', 'files': 86, 'routes': 88}" in messages


def test_test_outside_catalog_is_rejected(
    catalog: Catalog, pr_policy: Policy, pr_plan: Plan
) -> None:
    pytest_job = next(job for job in pr_plan.jobs if job.kind == "pytest")
    step = pytest_job.steps[0]
    bad_spec = replace(step.spec, files=(*step.spec.files, "tests/not-in-catalog.py"))
    bad_step = replace(step, spec=bad_spec)
    bad_job = replace(pytest_job, steps=(bad_step, *pytest_job.steps[1:]))
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(bad_job if job.id == pytest_job.id else job for job in pr_plan.jobs),
    ))
    messages = [issue.message for issue in validate_plan(broken, catalog, pr_policy)]
    assert "plan contains a test outside Catalog" in messages


def test_tampered_e2e_spec_is_rejected(
    catalog: Catalog, pr_policy: Policy, pr_plan: Plan
) -> None:
    e2e_job = next(job for job in pr_plan.jobs if job.kind == "model_e2e")
    step = e2e_job.steps[0]
    bad_spec = replace(step.spec, layers=step.spec.layers + 1)
    bad_step = replace(step, spec=bad_spec)
    bad_job = replace(e2e_job, steps=(bad_step,))
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(bad_job if job.id == e2e_job.id else job for job in pr_plan.jobs),
    ))
    messages = [issue.message for issue in validate_plan(broken, catalog, pr_policy)]
    assert "E2E spec does not match Catalog" in messages
