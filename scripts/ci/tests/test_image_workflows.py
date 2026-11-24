from __future__ import annotations

from pathlib import Path

import yaml


def _workflow(root: Path, name: str) -> dict:
    return yaml.safe_load((root / ".github/workflows" / name).read_text(encoding="utf-8"))


def _triggers(document: dict) -> dict:
    value = document.get(True, document.get("on"))
    assert isinstance(value, dict)
    return value


def _step(job: dict, step_id: str) -> dict:
    return next(step for step in job["steps"] if step.get("id") == step_id)


def test_image_workflows_use_only_the_black_box_contract(repo_root: Path) -> None:
    for name in (
        "ascend-pr.yml",
        "ascend-image-pr.yml",
        "ascend-image-pipeline.yml",
        "ascend-image-promote.yml",
    ):
        dumped = yaml.safe_dump(_workflow(repo_root, name))
        assert "content_tag" not in dumped
        assert "get_image_plan" not in dumped
        assert "scripts/ci/image/images.yml" not in dumped
        for command in (
            "compute-image-refs",
            "resolve-image-digest",
            "build-image",
            "promote-alias",
            "check-image-env",
        ):
            assert command not in dumped


def test_normal_pr_workflow_plans_runtime_images(repo_root: Path) -> None:
    document = _workflow(repo_root, "ascend-pr.yml")
    jobs = document["jobs"]
    dumped = yaml.safe_dump(document)
    triggers = _triggers(document)

    assert set(jobs) == {"plan", "cpu", "accelerator", "gate"}
    assert jobs["gate"]["needs"] == ["plan", "cpu", "accelerator"]
    assert jobs["cpu"]["name"] == "CPU (${{ matrix.display_name }})"
    assert jobs["accelerator"]["name"] == "Accelerator (${{ matrix.display_name }})"
    assert triggers["pull_request"]["paths-ignore"] == [
        "docker/**/Dockerfile*",
    ]
    assert triggers["workflow_call"]["inputs"]["image_id"]["default"] == ""
    assert triggers["workflow_call"]["inputs"]["targets"]["default"] == ""
    assert "scripts.ci.image.cli build " not in dumped
    assert "scripts.ci.image.cli promote " not in dumped
    assert "SWR_USERNAME" not in dumped

    plan_steps = jobs["plan"]["steps"]
    step_names = [step.get("name", "") for step in plan_steps]
    assert step_names.index("classify changes") > next(
        index for index, step in enumerate(plan_steps)
        if str(step.get("uses", "")).startswith("actions/setup-python")
    )
    assert step_names.index("install CI dependencies") > step_names.index(
        "classify changes"
    )
    assert step_names.index("lint") > step_names.index("install CI dependencies")
    assert step_names.index("generate test plan") > step_names.index("lint")
    assert "resolve runtime images" not in step_names

    classify = _step(jobs["plan"], "changes")
    assert "scripts.ci.tools.changed_files" in classify["run"]
    assert "--diff-filter=AM" in classify["run"]
    assert "printf 'change_mode=%s\\nscope=%s\\n'" in classify["run"]
    assert "ci-changed-files.txt" not in classify["run"]
    assert "delegated" not in classify["run"]

    publish = _step(jobs["plan"], "plan")
    assert "--image-ref" in publish["run"]
    assert 'ARGS+=(--image "$CI_IMAGE_ID")' in publish["run"]
    assert 'ARGS+=(--image-ref "$CI_IMAGE_ID=$CI_IMAGE_REF")' in publish["run"]
    assert '--target-job "$CI_TARGET_JOB"' in publish["run"]
    assert '--targets "$CI_TARGETS"' in publish["run"]
    assert '--github-output "$GITHUB_OUTPUT"' in publish["run"]
    assert "matrix-output" not in publish["run"]
    assert "--fail-code" not in publish["run"]
    assert 'echo "::error::lint failed"' in publish["run"]
    assert "producer.cli failed-output" in publish["run"]
    assert "--code lint-failed" in publish["run"]
    assert "ci-image-refs" not in dumped
    assert "scripts.ci.image.cli resolve" not in dumped
    assert "scripts.ci.execution.producer.cli images" not in dumped
    for job_name in ("cpu", "accelerator"):
        job = jobs[job_name]
        step_names = [step.get("name", "") for step in job["steps"]]
        assert "verify result" not in step_names
        assert "enforce result" not in step_names
        execute = next(
            step for step in job["steps"]
            if step.get("name") == "execute"
        )
        assert "set +e" not in execute["run"]
        assert "executor.exit_code" not in execute["run"]
        upload = next(
            step for step in job["steps"]
            if step.get("name") == "upload result"
        )
        assert upload["with"]["if-no-files-found"] == "error"
        assert "defaults" not in job


def test_normal_pr_gate_downloads_once_and_timing_is_non_blocking(
    repo_root: Path,
) -> None:
    gate = _workflow(repo_root, "ascend-pr.yml")["jobs"]["gate"]
    steps = gate["steps"]
    downloads = [
        step for step in steps
        if step.get("uses") == "actions/download-artifact@v4"
    ]

    assert len(downloads) == 1
    assert downloads[0]["with"] == {
        "pattern": "ci-*",
        "path": "${{ runner.temp }}/ci-artifacts",
    }
    assert downloads[0]["continue-on-error"] is True

    timing = next(step for step in steps if step.get("name") == "timing")
    judge = next(step for step in steps if step.get("name") == "judge")
    assert timing["continue-on-error"] is True
    assert steps.index(timing) < steps.index(judge)
    assert judge["env"]["PLAN"] == (
        "${{ runner.temp }}/ci-artifacts/ci-plan/plan.json"
    )
    assert judge["env"]["RESULTS"] == "${{ runner.temp }}/ci-artifacts"
    assert "set -euo pipefail" in judge["run"]
    assert 'mkdir -p "$RESULTS"' in judge["run"]
    assert "dirname" not in judge["run"]


def test_image_pr_separates_change_kind_image_selection_and_cpu_preflight(
    repo_root: Path,
) -> None:
    document = _workflow(repo_root, "ascend-image-pr.yml")
    triggers = _triggers(document)
    jobs = document["jobs"]

    assert triggers["pull_request_target"]["paths"] == ["docker/**/Dockerfile*"]
    assert set(jobs) == {"plan", "cpu-tests", "build_and_test_images", "gate"}
    assert jobs["plan"]["outputs"]["image_jobs"].startswith("${{ steps.images")
    assert jobs["plan"]["outputs"]["cpu_jobs"].startswith("${{ steps.cpu")

    select = _step(jobs["plan"], "images")
    cpu = _step(jobs["plan"], "cpu")
    assert "--diff-filter=AM" in select["run"]
    assert "scripts.ci.tools.changed_files" in select["run"]
    assert "scripts.ci.image.cli plan" in select["run"]
    assert "effective_kind=no_image" in select["run"]
    assert "case \"${image_count:-0}:$kind\" in" in select["run"]
    assert "*:image_only|*:mixed" in select["run"]
    assert all(step.get("id") != "changes" for step in jobs["plan"]["steps"])
    assert cpu["if"] == "steps.images.outputs.change_kind == 'mixed'"
    assert "--targets cpu" in cpu["run"]
    assert jobs["plan"]["outputs"]["cpu_jobs"].endswith("|| '[]' }}")
    assert jobs["plan"]["outputs"]["cpu_job_count"].endswith("|| '0' }}")

    assert jobs["cpu-tests"]["needs"] == "plan"
    assert "defaults" not in jobs["cpu-tests"]
    assert jobs["cpu-tests"]["if"] == (
        "needs.plan.outputs.change_kind == 'mixed' && "
        "needs.plan.outputs.cpu_job_count != '0'"
    )
    assert jobs["build_and_test_images"]["uses"] == (
        "./.github/workflows/ascend-image-pipeline.yml"
    )
    assert jobs["build_and_test_images"]["if"].find(
        "needs.cpu-tests.result == 'skipped'"
    ) >= 0
    assert "!cancelled()" in jobs["build_and_test_images"]["if"]
    assert "always()" not in jobs["build_and_test_images"]["if"]
    assert jobs["build_and_test_images"]["strategy"]["matrix"] == (
        "${{ fromJSON(needs.plan.outputs.image_jobs) }}"
    )
    assert jobs["build_and_test_images"]["with"]["image_id"] == (
        "${{ matrix.image_id }}"
    )
    assert set(jobs["build_and_test_images"]["with"]) == {
        "image_id",
        "target",
        "checkout_ref",
    }
    assert jobs["gate"]["name"] == "Image gate"
    evaluate = next(
        step for step in jobs["gate"]["steps"]
        if step.get("name") == "evaluate image PR pipeline"
    )
    assert "fail()" in evaluate["run"]
    assert "case \"$CHANGE_KIND\" in" in evaluate["run"]
    assert "CPU tests must be skipped" not in evaluate["run"]
    assert "image build must be skipped" not in evaluate["run"]
    judge = next(
        step for step in jobs["gate"]["steps"]
        if step.get("name") == "judge CPU Plan results"
    )
    assert "[ ! -f" not in judge["run"]
    assert "promote" not in jobs


def test_image_pr_approval_precedes_pr_checkout(repo_root: Path) -> None:
    document = _workflow(repo_root, "ascend-image-pr.yml")
    steps = document["jobs"]["plan"]["steps"]
    approval = next(
        step for step in steps if step.get("name") == "Check fork image PR approval"
    )
    checkout = next(
        step for step in steps if step.get("uses") == "actions/checkout@v4"
    )

    assert steps.index(approval) < steps.index(checkout)
    assert "head.repo.full_name != github.repository" in approval["if"]
    assert "ci:image:build" in approval["if"]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["allow-unsafe-pr-checkout"] is True
    cpu_checkout = next(
        step for step in document["jobs"]["cpu-tests"]["steps"]
        if step.get("uses") == "actions/checkout@v4"
    )
    assert cpu_checkout["with"]["allow-unsafe-pr-checkout"] is True


def test_image_pipeline_builds_one_candidate_and_reuses_pr_scope(
    repo_root: Path,
) -> None:
    document = _workflow(repo_root, "ascend-image-pipeline.yml")
    triggers = _triggers(document)
    jobs = document["jobs"]
    dumped = yaml.safe_dump(document)

    assert set(jobs) == {"build", "container-tests", "gate"}
    assert triggers["workflow_call"]["inputs"]["image_id"]["required"] is True
    assert triggers["workflow_call"]["inputs"]["run_tests"]["default"] is True
    assert "workflow_dispatch" not in triggers
    assert triggers["workflow_call"]["inputs"]["checkout_ref"]["default"] == ""
    assert document["jobs"]["build"]["steps"][0]["with"]["ref"].startswith(
        "${{ inputs.checkout_ref"
    )
    expected_pr_checkout = (
        "${{ startsWith(inputs.checkout_ref, 'refs/pull/') }}"
    )
    for job_name in ("build", "container-tests"):
        checkout = next(
            step for step in jobs[job_name]["steps"]
            if step.get("uses") == "actions/checkout@v4"
        )
        assert checkout["with"]["allow-unsafe-pr-checkout"] == expected_pr_checkout
    assert "scripts.ci.image.cli build" in dumped
    assert "scripts.ci.image.cli resolve" not in dumped
    assert "scripts.ci.image.cli promote " not in dumped
    assert "scripts.ci.image.cli check-envs" not in dumped
    assert "scripts.ci.tools.lint" not in dumped
    assert "docker/login-action@v3" in dumped
    assert "docker/setup-buildx-action@v3" in dumped

    build = jobs["build"]
    assert set(build["outputs"]) == {
        "ref", "container_jobs", "container_job_count"
    }
    build_step = next(
        step for step in build["steps"] if step.get("id") == "result"
    )
    assert '--image "$IMAGE_ID"' in build_step["run"]
    plan = next(
        step for step in build["steps"]
        if step.get("name") == "plan candidate tests"
    )
    assert '--image "$IMAGE_ID"' in plan["run"]
    assert '--image-ref "$IMAGE_ID=$IMAGE_REF"' in plan["run"]
    assert "--targets \"$TARGETS\"" in plan["run"]
    assert "--github-output \"$GITHUB_OUTPUT\"" in plan["run"]
    assert "ci-matrix-output" not in plan["run"]
    assert "mkdir -p" not in plan["run"]
    assert "${{ steps.result.outputs.ref }}" in yaml.safe_dump(build)
    assert "${{ needs.build.outputs.ref }}" in yaml.safe_dump(
        jobs["container-tests"]["container"]
    )
    assert jobs["container-tests"]["strategy"]["max-parallel"] == (
        "${{ fromJSON(vars.CI_CONTAINER_MAX_PARALLEL || '2') }}"
    )
    assert jobs["gate"]["needs"] == "container-tests"
    assert "ci-plan-${{ inputs.image_id }}" in dumped
    assert "ci-result-${{ inputs.image_id }}" in dumped
    assert "npu-default" not in dumped


def test_manual_image_wraps_the_reusable_candidate_pipeline(repo_root: Path) -> None:
    document = _workflow(repo_root, "ascend-manual-image.yml")
    triggers = _triggers(document)
    job = document["jobs"]["image"]

    assert set(triggers) == {"workflow_dispatch"}
    assert triggers["workflow_dispatch"]["inputs"]["image_id"]["default"] == (
        "npu-a3-cann90"
    )
    assert triggers["workflow_dispatch"]["inputs"]["run_tests"]["default"] is False
    assert job["uses"] == "./.github/workflows/ascend-image-pipeline.yml"
    assert job["with"] == {
        "image_id": "${{ inputs.image_id }}",
        "target": "${{ inputs.target }}",
        "checkout_ref": "${{ inputs.checkout_ref }}",
        "max_jobs": "${{ inputs.max_jobs }}",
        "use_cache": "${{ inputs.use_cache }}",
        "runner": "${{ inputs.runner }}",
        "run_tests": "${{ inputs.run_tests }}",
    }
    assert job["secrets"] == {
        "SWR_USERNAME": "${{ secrets.SWR_USERNAME }}",
        "SWR_PASSWORD": "${{ secrets.SWR_PASSWORD }}",
    }


def test_image_promote_selects_affected_images_and_fans_out(repo_root: Path) -> None:
    document = _workflow(repo_root, "ascend-image-promote.yml")
    triggers = _triggers(document)
    jobs = document["jobs"]
    discovered = jobs["discover"]
    select = discovered["steps"][-1]
    promotions = jobs["promotions"]
    gate = jobs["gate"]

    assert triggers["push"]["branches"] == ["main"]
    assert triggers["push"]["paths"] == ["docker/**/Dockerfile*"]
    assert triggers["workflow_dispatch"]["inputs"]["images"]["default"] == "all"
    assert document["concurrency"] == {
        "group": "image-promote",
        "cancel-in-progress": False,
    }
    assert "refs/heads/main" in discovered["steps"][0]["run"]
    assert "scripts.ci.image.cli plan" in select["run"]
    assert "github.event.before" in select["env"]["BEFORE_SHA"]
    assert "--diff-filter=AM" in select["run"]
    assert promotions["strategy"]["matrix"] == (
        "${{ fromJSON(needs.discover.outputs.image_jobs) }}"
    )
    promote = next(step for step in promotions["steps"] if step.get("id") == "promote")
    assert '--image "$IMAGE_ID"' in promote["run"]
    assert "scripts.ci.image.cli promote" in promote["run"]
    assert "scripts.ci.image.cli resolve" not in yaml.safe_dump(promotions)
    assert "scripts.ci.image.cli check-envs" not in yaml.safe_dump(promotions)
    assert set(gate["needs"]) == {"discover", "promotions"}


def test_ci_yaml_uses_one_environment_inline_model(repo_root: Path) -> None:
    document = yaml.safe_load(
        (repo_root / "scripts/ci/config/ci.yml").read_text(encoding="utf-8")
    )

    assert "schema_version" not in document
    assert list(document["scopes"]) == ["pull_request", "nightly"]
    assert "environments" not in document
    assert "lanes" not in document["resources"]
    assert "image_pr" not in document["scopes"]
    assert document["resources"]["images"]["npu-a3-cann90"]["build"] == {
        "dockerfile": "docker/npu/Dockerfile_aarch64",
        "platform": "linux/arm64",
        "revision": 1,
        "ci_build_args": {"GIT_HTTP_USER_AGENT": "git/2.56.1"},
    }
    assert {
        selection["environment"]["image"]
        for selection in document["scopes"]["pull_request"]
    } == {"cpu-default", "npu-a3-cann90"}


def test_nightly_syncs_aliases_then_reuses_the_same_graph(repo_root: Path) -> None:
    document = _workflow(repo_root, "ascend-nightly.yml")
    jobs = document["jobs"]
    sync = jobs["sync-aliases"]
    job = jobs["full-tests"]

    assert list(jobs) == ["sync-aliases", "full-tests"]
    assert sync["if"] == "github.ref == 'refs/heads/main'"
    assert sync["continue-on-error"] is True
    assert sync["concurrency"] == {
        "group": "image-promote",
        "cancel-in-progress": False,
    }
    dumped_sync = yaml.safe_dump(sync)
    assert "SWR_USERNAME" in dumped_sync
    assert "docker login" in dumped_sync
    assert "sync-aliases" in dumped_sync
    assert "--best-effort" in dumped_sync
    assert job["needs"] == "sync-aliases"
    assert job["if"] == "always()"
    assert job["uses"] == "./.github/workflows/ascend-pr.yml"
    assert job["with"] == {"mode": "full"}
    dumped = yaml.safe_dump(document)
    assert "scripts.ci.image.cli build" not in dumped
    assert "scripts.ci.image.cli promote " not in dumped


def test_manual_test_workflow_can_pair_an_image_ref_with_an_image_id(
    repo_root: Path,
) -> None:
    primary = _workflow(repo_root, "ascend-pr.yml")
    document = _workflow(repo_root, "ascend-manual-test.yml")
    job = document["jobs"]["manual"]

    assert "workflow_dispatch" not in _triggers(primary)
    assert "image_id" in _triggers(document)["workflow_dispatch"]["inputs"]
    assert job["with"]["image_id"] == "${{ inputs.image_id }}"
