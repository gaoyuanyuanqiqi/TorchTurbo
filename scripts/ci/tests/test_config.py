from __future__ import annotations

from collections import Counter
from dataclasses import replace

import pytest

from scripts.ci.execution.producer.config import (
    CiConfig,
    CiConfigError,
    CiRunner,
    CiTarget,
    ScopeSelection,
    _load_scopes,
    area_selected,
    bind_image_refs,
)
from scripts.ci.execution.producer.facts import TestFacts
from scripts.ci.image.image_config import RuntimeImage

DIGEST_REF = "registry.example/team/image@sha256:" + "a" * 64


def test_config_resources(config: CiConfig) -> None:
    assert config.targets["cpu"].device == "cpu"
    assert config.targets["npu-a3"].device == "npu"
    assert set(config.images) == {"cpu-default", "npu-a3-cann90"}
    assert config.images["cpu-default"].kind == "host"
    assert config.images["cpu-default"].ref == "python:3.11"
    assert config.images["npu-a3-cann90"].kind == "container"
    assert config.images["npu-a3-cann90"].ref.startswith(
        "test.invalid/npu-a3-cann90@sha256:"
    )
    assert set(config.venvs) == {
        "default",
        "qwen4_exp",
        "lf",
        "swift",
        "qwen4_exp_swift",
    }
    assert config.pytest_areas.job_timeout_minutes == {
        "cpu": 40, "npu": 90, "gpu": 90,
    }
    assert config.pytest_areas.step_timeout_minutes == {
        "cpu": 40, "npu": 90, "gpu": 90,
    }
    suite = config.integration_suites["base_suite"]
    assert suite.job_timeout_minutes == {"cpu": 40, "npu": 90, "gpu": 90}
    assert suite.step_timeout_minutes == {"cpu": 40, "npu": 90, "gpu": 90}
    override = suite.overrides["bailing_v3/sft_16npu"]
    assert override.job_timeout_minutes is not None
    assert suite.job_timeout_minutes_for(
        "npu", "bailing_v3/sft_16npu"
    ) == override.job_timeout_minutes
    assert suite.job_timeout_minutes_for("npu", "qwen3_5_moe/sft_2npu") == 90


def test_runner_selection_uses_the_smallest_enough_capacity(config: CiConfig) -> None:
    assert config.runner_for("npu-a3", 1).name == "npu-a3/2"
    assert config.runner_for("npu-a3", 2).name == "npu-a3/2"
    assert config.runner_for("npu-a3", 3).name == "npu-a3/16"
    assert config.runner_for("cpu", 8).name == "cpu/8"
    with pytest.raises(CiConfigError, match="no runner"):
        config.runner_for("cpu", 9)


def test_scope_selects_test_areas_and_images(config: CiConfig) -> None:
    pr = config.scope("pull_request")
    assert pr.selections == (
        ScopeSelection(
            image="cpu-default",
            test_areas=(
                "pytest/core",
                "pytest/adapter",
                "integration/base_suite/qwen3_5_moe/sft_2cpu",
            ),
        ),
        ScopeSelection(
            image="npu-a3-cann90",
            test_areas=(
                "pytest/core",
                "integration/base_suite/qwen3_5_moe/sft_2npu",
            ),
        ),
    )

    nightly = config.scope("nightly")
    assert nightly.selections[0].test_areas is None
    assert nightly.selections[1].image == "npu-a3-cann90"


def test_scope_cross_references_are_checked(config: CiConfig) -> None:
    issues = []
    _load_scopes(
        {
            "pull_request": [
                {"environment": {"image": "cpu-default"}},
                {"environment": {"image": "missing"}},
                {"environment": {"image": "cpu-default"}},
            ],
            "nightly": [{"environment": {"image": "cpu-default"}}],
        },
        config.images,
        {"base_suite"},
        issues,
    )

    messages = [issue.message for issue in issues]
    assert "unknown image" in messages
    assert "duplicate image selection" in messages


def test_scope_test_areas_are_validated(config: CiConfig) -> None:
    issues = []
    _load_scopes(
        {
            "pull_request": [{
                "environment": {"image": "cpu-default"},
                "test_areas": [
                    "pytest/core",
                    "pytest/missing",
                    "integration/missing_suite/case",
                ],
            }],
            "nightly": [{
                "environment": {"image": "cpu-default"},
                "test_areas": [],
            }],
        },
        config.images,
        set(config.integration_suites),
        issues,
    )

    messages = [issue.message for issue in issues]
    assert "unknown TestArea 'pytest/missing'" in messages
    assert "unknown TestArea 'integration/missing_suite/case'" in messages
    assert "must not be empty" in messages


def test_unknown_target_fails_closed(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="unknown target"):
        config.target_for("npu-a5")


def test_a5_extends_by_target_image_runner_and_scope(config: CiConfig) -> None:
    target = CiTarget(name="npu-a5", device="npu")
    image = RuntimeImage(
        image_id="npu-a5-cann90",
        target=target.name,
        kind="container",
        ref="",
    )
    runner = CiRunner(
        target=target.name,
        rank_capacity=2,
        runner_variable="CI_NPU_A5_2_RUNNER",
        runner="linux-aarch64-a5-800i-2",
    )
    pr = config.scope("pull_request")
    scope = replace(
        pr,
        selections=(
            *pr.selections,
            ScopeSelection(
                image=image.image_id,
                test_areas=(
                    "pytest/core",
                    "integration/base_suite/qwen3_5_moe/sft_2npu",
                ),
            ),
        ),
    )
    candidate = replace(
        config,
        targets={**config.targets, target.name: target},
        images={**config.images, image.image_id: image},
        runners=(*config.runners, runner),
        scopes={**config.scopes, pr.name: scope},
    )

    assert candidate.scope("pull_request").selections[-1].image == "npu-a5-cann90"
    assert candidate.runner_for("npu-a5", 2) == runner


def test_venv_inheritance_is_explicit(config: CiConfig) -> None:
    venv = config.venvs["qwen4_exp_swift"]
    assert venv.extends is None
    assert venv.requirements == ("scripts/ci/config/envs/swift.txt",)
    assert venv.overrides == ("datasets==5.0.1",)


def test_area_selection_semantics(config: CiConfig) -> None:
    pr = config.scope("pull_request")
    nightly = config.scope("nightly")
    assert area_selected(pr.selections[0].test_areas, "pytest/adapter")
    assert not area_selected(pr.selections[1].test_areas, "pytest/adapter")
    assert area_selected(nightly.selections[0].test_areas, "pytest/model")
    assert not area_selected(nightly.selections[1].test_areas, "pytest/adapter")
    assert area_selected(
        nightly.selections[1].test_areas,
        "integration/base_suite/bailing_v3/sft_16npu",
    )
    assert not area_selected((), "pytest/core")


def test_pytest_venv_resolution(config: CiConfig) -> None:
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/qwen4_exp/test_model.py",
        ("qwen4_exp",),
        (),
    )
    assert (venv, issues) == ("qwen4_exp", [])
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/qwen4_exp/test_swift.py",
        ("qwen4_exp",),
        ("swift",),
    )
    assert (venv, issues) == ("qwen4_exp_swift", [])
    venv, issues = config.resolve_pytest_venv(
        "tests/unit_tests/models/bailing_moe_v3_vl/test_lf_image_parity.py",
        ("bailing_moe_v3_vl",),
        ("lf",),
    )
    assert (venv, issues) == ("lf", [])


def test_unknown_scope_is_rejected(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="unknown CI scope"):
        config.scope("missing")


def test_image_overrides_must_reference_defined_digest_pinned_images(config: CiConfig) -> None:
    with pytest.raises(CiConfigError, match="no image definition"):
        bind_image_refs(config, {"missing": DIGEST_REF})
    with pytest.raises(CiConfigError, match="digest-pinned"):
        bind_image_refs(config, {"npu-a3-cann90": "registry.example/team/image:tag"})


def test_facts_snapshot(facts: TestFacts) -> None:
    assert Counter(fact.suite for fact in facts.pytest) | Counter(
        {"total": len(facts.pytest), "integration": len(facts.integration)}
    ) == {
        "core": 94,
        "model": 56,
        "adapter": 12,
        "total": 162,
        "integration": 3,
    }
    assert [case.case_id for case in facts.integration] == [
        "bailing_v3/sft_16npu",
        "qwen3_5_moe/sft_2cpu",
        "qwen3_5_moe/sft_2npu",
    ]


def test_model_venv_distribution(config: CiConfig, facts: TestFacts) -> None:
    distribution = Counter()
    for fact in facts.pytest:
        if fact.suite != "model":
            continue
        venv, issues = config.resolve_pytest_venv(
            fact.path, fact.model_families, fact.frameworks
        )
        assert not issues
        distribution[venv] += 1
    assert distribution == {
        "default": 36,
        "qwen4_exp": 16,
        "lf": 3,
        "swift": 1,
    }
