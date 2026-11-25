from __future__ import annotations

import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.ci.contracts.base import Diagnostic
from scripts.ci.contracts.image_contract import ResolvedImage
from scripts.ci.contracts.planner_contract import (
    ExecutionPlan,
    IntegrationSpec,
    PytestSpec,
    load_planner_output,
    plan_digest,
)
from scripts.ci.execution.producer.cli import main
from scripts.ci.execution.producer import cli as producer_cli
from scripts.ci.execution.producer.config import (
    CiConfig,
    CiRunner,
    CiTarget,
    ScopeSelection,
)
from scripts.ci.execution.producer.facts import TestFacts
from scripts.ci.execution.producer.planner import (
    PlanError,
    _marker,
    _task_group,
    build_plan,
    planner_output_for,
    requested_images,
)
from scripts.ci.execution.producer.matrix import matrix_outputs
from scripts.ci.image.image_config import RuntimeImage
from scripts.ci.tools.json_io import write_json

DIGEST = "sha256:" + "b" * 64
REF = f"registry.example/team/image@{DIGEST}"


def _with_plan_id(plan: ExecutionPlan) -> ExecutionPlan:
    return replace(plan, plan_id=plan_digest(plan))


def test_cpu_marker_includes_unmarked_tests() -> None:
    assert _marker("cpu") == (
        "cpu or (not npu and not multi_npu and not gpu and not multi_gpu)"
    )
    assert _marker("npu") == "npu or multi_npu"


def test_cpu_marker_selects_unmarked_tests(tmp_path: Path) -> None:
    probe = tmp_path / "test_marker_probe.py"
    probe.write_text(
        "import pytest\n\n"
        "def test_unmarked():\n    pass\n\n"
        "@pytest.mark.cpu\n"
        "def test_cpu():\n    pass\n\n"
        "@pytest.mark.npu\n"
        "def test_npu():\n    pass\n\n"
        "@pytest.mark.multi_npu\n"
        "def test_multi_npu():\n    pass\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            _marker("cpu"),
            str(probe),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed, 2 deselected" in result.stdout


def test_task_group_is_derived_from_fact(facts: TestFacts) -> None:
    by_path = {fact.path: fact for fact in facts.pytest}
    assert _task_group(
        by_path["tests/unit_tests/engine/test_core.py"]
    ) == "layer:tests/unit_tests/engine"
    assert _task_group(
        by_path["tests/unit_tests/engine/test_engine_config_loading.py"]
    ) == "layer:tests/unit_tests/engine"
    assert _task_group(
        by_path["tests/unit_tests/models/qwen3_5/test_mtp.py"]
    ) == "family:qwen3_5"


def _assert_matches(
    plan: ExecutionPlan, config: CiConfig, facts: TestFacts, **selection
) -> None:
    output = planner_output_for(
        config,
        facts,
        ci_type=plan.ci_type,
        change_mode=plan.change_mode,
        git_source=plan.git_source,
        **selection,
    )
    assert output.status == "ready"
    assert output.plan is not None
    assert output.plan.to_dict() == plan.to_dict()


def test_pull_request_scope(
    config: CiConfig, facts: TestFacts, pr_plan: ExecutionPlan
) -> None:
    assert Counter(job.kind for job in pr_plan.jobs) == {
        "pytest": 4,
        "integration": 2,
    }
    assert pr_plan.ci_type == "pull_request"
    assert pr_plan.change_mode == "code"
    _assert_matches(pr_plan, config, facts)
    assert [job.id for job in pr_plan.jobs] == [
        "integration-base-suite-qwen3-5-moe-sft-2cpu",
        "integration-base-suite-qwen3-5-moe-sft-2npu",
        "pytest-adapter-cpu-cpu-default-lf",
        "pytest-adapter-cpu-cpu-default-swift",
        "pytest-core-cpu-cpu-default-default",
        "pytest-core-npu-npu-a3-cann90-default",
    ]
    for job in pr_plan.jobs:
        assert job.runtime.image.ref
        if job.runtime.image.kind == "container":
            image_digest = job.runtime.image.ref.rpartition("@")[2]
            assert image_digest.startswith("sha256:")
        if job.runtime.device == "cpu":
            assert job.runtime.runner == "cpu/8"
        else:
            assert job.runtime.runner == "npu-a3/2"
        if job.kind == "integration":
            assert len(job.tasks) == 1
            assert isinstance(job.tasks[0].spec, IntegrationSpec)


def test_docs_only_scope_has_no_jobs(
    config: CiConfig, facts: TestFacts, source
) -> None:
    plan = build_plan(config, facts, "pull_request", source, change_mode="docs_only")
    assert plan.jobs == ()
    assert plan.change_mode == "docs_only"
    _assert_matches(plan, config, facts)
    matrices = matrix_outputs(plan, config)
    assert {key: len(value) for key, value in matrices.items()} == {
        "cpu_jobs": 0,
        "npu_jobs": 0,
        "gpu_jobs": 0,
        "container_jobs": 0,
    }


@pytest.mark.parametrize("change_mode", ["image_only", "mixed"])
def test_non_code_change_modes_have_no_jobs(
    config: CiConfig, facts: TestFacts, source, change_mode: str
) -> None:
    plan = build_plan(config, facts, "pull_request", source, change_mode=change_mode)
    assert plan.jobs == ()
    assert plan.change_mode == change_mode
    _assert_matches(plan, config, facts)


def test_nightly_scope(
    config: CiConfig, facts: TestFacts, nightly_plan: ExecutionPlan
) -> None:
    assert Counter(job.kind for job in nightly_plan.jobs) == {
        "pytest": 10,
        "integration": 3,
    }
    pytest_suites = {
        task.spec.suite
        for job in nightly_plan.jobs
        if job.kind == "pytest"
        for task in job.tasks
        if isinstance(task.spec, PytestSpec)
    }
    assert pytest_suites == {"core", "model", "adapter"}
    integration_cases = {
        (task.spec.suite, task.spec.case_id)
        for job in nightly_plan.jobs
        if job.kind == "integration"
        for task in job.tasks
        if isinstance(task.spec, IntegrationSpec)
    }
    assert integration_cases == {
        ("base_suite", "bailing_v3/sft_16npu"),
        ("base_suite", "qwen3_5_moe/sft_2cpu"),
        ("base_suite", "qwen3_5_moe/sft_2npu"),
    }
    _assert_matches(nightly_plan, config, facts)


def test_cpu_target_slice_runs_pr_cpu_jobs(
    config: CiConfig, facts: TestFacts, source
) -> None:
    output = planner_output_for(
        config,
        facts,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
        targets="cpu",
    )
    assert output.status == "ready"
    assert output.plan is not None
    plan = output.plan
    assert [job.id for job in plan.jobs] == [
        "integration-base-suite-qwen3-5-moe-sft-2cpu",
        "pytest-adapter-cpu-cpu-default-lf",
        "pytest-adapter-cpu-cpu-default-swift",
        "pytest-core-cpu-cpu-default-default",
    ]
    assert [item["display_name"] for item in matrix_outputs(plan, config)["cpu_jobs"]] == [
        "integration-base-suite-qwen3-5-moe-sft-2cpu",
        "pytest-adapter-cpu-lf",
        "pytest-adapter-cpu-swift",
        "pytest-core-cpu-default",
    ]
    assert {job.runtime.device for job in plan.jobs} == {"cpu"}
    _assert_matches(plan, config, facts, targets="cpu")


def test_cpu_target_slice_ignores_unbound_other_targets(
    config: CiConfig, facts: TestFacts, source
) -> None:
    image = replace(config.images["npu-a3-cann90"], ref="")
    unbound = replace(config, images={**config.images, image.image_id: image})

    output = planner_output_for(
        unbound,
        facts,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
        targets="cpu",
    )

    assert output.status == "ready"
    assert output.plan is not None
    assert {job.runtime.device for job in output.plan.jobs} == {"cpu"}


def test_npu_target_slice_runs_pr_npu_jobs(
    config: CiConfig, facts: TestFacts, source
) -> None:
    output = planner_output_for(
        config,
        facts,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
        targets="npu-a3",
    )
    assert output.plan is not None
    plan = output.plan
    assert [job.id for job in plan.jobs] == [
        "integration-base-suite-qwen3-5-moe-sft-2npu",
        "pytest-core-npu-npu-a3-cann90-default",
    ]
    assert {job.runtime.device for job in plan.jobs} == {"npu"}
    assert all(job.runtime.runner == "npu-a3/2" for job in plan.jobs)
    _assert_matches(plan, config, facts, targets="npu-a3")


def test_requested_images_only_returns_container_bindings(config: CiConfig) -> None:
    assert requested_images(config, "pull_request") == ("npu-a3-cann90",)
    assert requested_images(
        config, "pull_request", image_override="npu-a3-cann90"
    ) == ("npu-a3-cann90",)
    assert requested_images(config, "pull_request", targets="cpu") == ()


def test_missing_container_binding_fails(
    config: CiConfig, facts: TestFacts, source
) -> None:
    image = replace(config.images["npu-a3-cann90"], ref="")
    unbound = replace(config, images={**config.images, image.image_id: image})
    with pytest.raises(PlanError, match="missing ResolvedImage"):
        build_plan(unbound, facts, "pull_request", source)


def _add_target(
    config: CiConfig,
    *,
    target: CiTarget,
    image: RuntimeImage,
    runner: CiRunner,
    test_areas: tuple[str, ...],
) -> CiConfig:
    pr = config.scope("pull_request")
    scope = replace(
        pr,
        selections=(
            *pr.selections,
            ScopeSelection(image=image.image_id, test_areas=test_areas),
        ),
    )
    return replace(
        config,
        targets={**config.targets, target.name: target},
        images={**config.images, image.image_id: image},
        runners=(*config.runners, runner),
        scopes={**config.scopes, pr.name: scope},
    )


def test_a5_extends_target_image_runner_and_scope(
    config: CiConfig, facts: TestFacts, source
) -> None:
    target = CiTarget(name="npu-a5", device="npu")
    image = RuntimeImage(
        image_id="npu-a5-cann90",
        target=target.name,
        kind="container",
        ref=REF,
    )
    runner = CiRunner(
        target=target.name,
        rank_capacity=2,
        runner_variable="CI_NPU_A5_2_RUNNER",
        runner="linux-aarch64-a5-800i-2",
    )
    candidate = _add_target(
        config,
        target=target,
        image=image,
        runner=runner,
        test_areas=(
            "pytest/core",
            "integration/base_suite/qwen3_5_moe/sft_2npu",
        ),
    )
    current = candidate.scope("pull_request")
    candidate = replace(
        candidate,
        scopes={
            **candidate.scopes,
            current.name: replace(
                current,
                selections=(
                    current.selections[0],
                    replace(current.selections[1], test_areas=("pytest/core",)),
                    current.selections[2],
                ),
            ),
        },
    )
    a5_facts = replace(
        facts,
        integration=tuple(
            replace(case, targets=("npu-a5",))
            if case.case_id == "qwen3_5_moe/sft_2npu" else case
            for case in facts.integration
        ),
    )

    plan = build_plan(candidate, a5_facts, "pull_request", source)
    job = next(
        item for item in plan.jobs
        if item.id == "integration-base-suite-qwen3-5-moe-sft-2npu"
    )
    assert job.runtime.target == "npu-a5"
    assert job.runtime.image.image_id == "npu-a5-cann90"
    assert job.runtime.runner == "npu-a5/2"
    _assert_matches(plan, candidate, a5_facts)


def test_gpu_target_routes_to_the_container_matrix(
    config: CiConfig, facts: TestFacts, source
) -> None:
    target = CiTarget(name="gpu-h100", device="gpu")
    image = RuntimeImage(
        image_id="gpu-h100-default",
        target=target.name,
        kind="container",
        ref=REF,
    )
    runner = CiRunner(
        target=target.name,
        rank_capacity=1,
        runner_variable="CI_GPU_H100_1_RUNNER",
        runner="linux-x86_64-h100-1",
        container_options="--user root --gpus all",
    )
    candidate = _add_target(
        config,
        target=target,
        image=image,
        runner=runner,
        test_areas=("pytest/core",),
    )

    plan = build_plan(candidate, facts, "pull_request", source)
    gpu_jobs = [job for job in plan.jobs if job.runtime.device == "gpu"]
    assert gpu_jobs
    assert all(job.runtime.target == "gpu-h100" for job in gpu_jobs)
    matrices = matrix_outputs(plan, candidate)
    assert any(item["device"] == "gpu" for item in matrices["container_jobs"])
    _assert_matches(plan, candidate, facts)
    assert matrices["gpu_jobs"][0]["container_options"] == (
        "--user root --gpus all"
    )


def test_image_override_selects_the_candidate_before_scope(
    config: CiConfig, facts: TestFacts, source
) -> None:
    target = config.targets["npu-a3"]
    candidate_image = replace(
        config.images["npu-a3-cann90"],
        image_id="npu-a3-cann92",
        ref=REF,
    )
    candidate = replace(
        config,
        images={**config.images, candidate_image.image_id: candidate_image},
    )

    plan = build_plan(
        candidate,
        facts,
        "pull_request",
        source,
        image_override=candidate_image.image_id,
    )
    unit_jobs = [
        job for job in plan.jobs
        if job.runtime.target == target.name and job.kind == "pytest"
    ]
    assert unit_jobs
    assert all(
        job.runtime.image.image_id == candidate_image.image_id
        for job in unit_jobs
    )
    _assert_matches(
        plan,
        candidate,
        facts,
        image_override=candidate_image.image_id,
    )


def test_two_images_on_the_same_target_are_separate_jobs(
    config: CiConfig, facts: TestFacts, source
) -> None:
    target = config.targets["npu-a3"]
    alternative = RuntimeImage(
        image_id="npu-a3-cann92",
        target=target.name,
        kind="container",
        ref=REF,
    )
    pr_scope = config.scope("pull_request")
    scope = replace(
        pr_scope,
        selections=(
            *pr_scope.selections,
            ScopeSelection(
                image=alternative.image_id, test_areas=("pytest/core",)
            ),
        ),
    )
    candidate = replace(
        config,
        images={**config.images, alternative.image_id: alternative},
        scopes={**config.scopes, pr_scope.name: scope},
    )

    plan = build_plan(candidate, facts, "pull_request", source)
    npu_core = [
        job for job in plan.jobs
        if job.kind == "pytest" and job.runtime.target == target.name
    ]
    assert {job.runtime.image.image_id for job in npu_core} == {
        "npu-a3-cann90", "npu-a3-cann92"
    }
    assert len({job.id for job in npu_core}) == 2
    _assert_matches(plan, candidate, facts)


def test_matrix_exposes_runner_fields(
    config: CiConfig, facts: TestFacts, source
) -> None:
    output = planner_output_for(
        config,
        facts,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
    )
    assert output.plan is not None
    matrices = matrix_outputs(output.plan, config)
    assert matrices["cpu_jobs"][0]["display_name"] == (
        "integration-base-suite-qwen3-5-moe-sft-2cpu"
    )
    assert matrices["npu_jobs"][0]["display_name"] == (
        "integration-base-suite-qwen3-5-moe-sft-2npu"
    )
    assert "npu-a3/2" in {job.runtime.runner for job in output.plan.jobs}


def test_explicit_unknown_integration_case_is_rejected(
    config: CiConfig, facts: TestFacts, source
) -> None:
    scope = config.scope("pull_request")
    broken_scope = replace(
        scope,
        selections=(
            replace(
                scope.selections[0],
                test_areas=("integration/base_suite/missing/case",),
            ),
            *scope.selections[1:],
        ),
    )
    broken = replace(config, scopes={**config.scopes, "pull_request": broken_scope})
    with pytest.raises(PlanError, match="unknown integration case"):
        build_plan(broken, facts, "pull_request", source)


def test_planner_failure_output_round_trip(
    tmp_path: Path, config: CiConfig, source
) -> None:
    output = planner_output_for(
        config,
        None,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
        issues=(Diagnostic(
            code="missing-facts",
            location="tests",
            message="facts missing",
        ),),
    )
    assert output.status == "failed"
    assert output.plan is None

    path = tmp_path / "planner-output.json"
    write_json(path, output.to_dict())
    loaded = load_planner_output(path)
    assert loaded.to_dict() == output.to_dict()


def test_ready_output_round_trip(
    tmp_path: Path, config: CiConfig, facts: TestFacts, source
) -> None:
    output = planner_output_for(
        config,
        facts,
        ci_type="pull_request",
        change_mode="code",
        git_source=source,
    )
    assert output.status == "ready"
    assert output.plan is not None

    path = tmp_path / "planner-output.json"
    write_json(path, output.to_dict())
    loaded = load_planner_output(path)
    assert loaded.status == "ready"
    assert loaded.plan is not None
    assert loaded.plan.to_dict() == output.plan.to_dict()


def test_missing_job_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: ExecutionPlan
) -> None:
    missing = "pytest-core-npu-npu-a3-cann90-default"
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(job for job in pr_plan.jobs if job.id != missing),
    ))
    assert broken.to_dict() != build_plan(
        config, facts, "pull_request", pr_plan.git_source
    ).to_dict()


def test_tampered_integration_spec_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: ExecutionPlan
) -> None:
    job = next(job for job in pr_plan.jobs if job.kind == "integration")
    spec = job.tasks[0].spec
    assert isinstance(spec, IntegrationSpec)
    bad_spec = replace(spec, case_id="tampered")
    bad_job = replace(job, tasks=(replace(job.tasks[0], spec=bad_spec),))
    broken = _with_plan_id(replace(
        pr_plan,
        jobs=tuple(bad_job if item.id == job.id else item for item in pr_plan.jobs),
    ))
    assert broken.to_dict() != build_plan(
        config, facts, "pull_request", pr_plan.git_source
    ).to_dict()


def test_integration_case_requires_framework_venv(
    config: CiConfig, facts: TestFacts, source
) -> None:
    broken = replace(
        facts,
        integration=tuple(
            replace(case, framework="missing")
            if case.case_id == "qwen3_5_moe/sft_2cpu" else case
            for case in facts.integration
        ),
    )

    with pytest.raises(PlanError, match="has no venv mapping"):
        build_plan(config, broken, "pull_request", source)


def test_integration_case_requires_known_target(
    config: CiConfig, facts: TestFacts, source
) -> None:
    broken = replace(
        facts,
        integration=tuple(
            replace(case, targets=("missing",))
            if case.case_id == "qwen3_5_moe/sft_2cpu" else case
            for case in facts.integration
        ),
    )

    with pytest.raises(PlanError, match="unknown target 'missing'"):
        build_plan(config, broken, "pull_request", source)


def test_tampered_change_mode_is_rejected(
    config: CiConfig, facts: TestFacts, pr_plan: ExecutionPlan
) -> None:
    broken = _with_plan_id(replace(pr_plan, change_mode="docs_only"))
    expected = build_plan(
        config, facts, "pull_request", pr_plan.git_source, change_mode="docs_only"
    )
    assert broken.to_dict() != expected.to_dict()


def test_cli_uses_override_and_writes_matrix(
    tmp_path: Path, config: CiConfig, monkeypatch
) -> None:
    output = tmp_path / "plan.json"
    matrix = tmp_path / "matrix.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(matrix))
    assert main([
        "plan",
        "--root", str(config.root),
        "--scope", "pull_request",
        "--image", "npu-a3-cann90",
        "--image-ref", f"npu-a3-cann90={REF}",
        "--output", str(output),
    ]) == 0
    loaded_output = load_planner_output(output)
    assert loaded_output.plan is not None
    npu_jobs = [
        job for job in loaded_output.plan.jobs
        if job.runtime.target == "npu-a3"
    ]
    assert npu_jobs
    assert all(
        job.runtime.image.image_id == "npu-a3-cann90" for job in npu_jobs
    )
    assert "cpu_job_count=0" in matrix.read_text(encoding="utf-8")
    assert "npu_job_count=2" in matrix.read_text(encoding="utf-8")
    assert "container_job_count=2" in matrix.read_text(encoding="utf-8")


def test_cli_resolves_container_images_inside_plan(
    tmp_path: Path, repo_root: Path, monkeypatch
) -> None:
    output = tmp_path / "plan.json"
    calls: list[str] = []

    def resolve(root, image_id):
        calls.append(image_id)
        return ResolvedImage(
            image_id=image_id,
            target="npu-a3",
            kind="container",
            ref=REF,
        )

    monkeypatch.setattr(producer_cli, "resolve_image", resolve)
    assert main([
        "plan",
        "--root", str(repo_root),
        "--scope", "pull_request",
        "--targets", "npu-a3",
        "--output", str(output),
    ]) == 0
    loaded_output = load_planner_output(output)
    assert loaded_output.plan is not None
    assert calls == ["npu-a3-cann90"]
    assert {job.runtime.image.ref for job in loaded_output.plan.jobs} == {REF}
