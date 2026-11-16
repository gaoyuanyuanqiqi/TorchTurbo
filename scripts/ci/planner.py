#!/usr/bin/env python3
"""Compile Catalog and Policy into a self-contained CI PlanArtifact."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from catalog import (
    Catalog,
    CatalogError,
    CatalogImage,
    CatalogLane,
    ModelE2ECase,
    TestEntry,
    load as load_catalog,
)
from contracts import (
    SCHEMA_VERSION,
    CommandSpec,
    Issue,
    Job,
    ModelE2ESpec,
    Plan,
    PlanArtifact,
    PytestSpec,
    Runtime,
    Source,
    Step,
    build_plan,
    build_plan_artifact,
    canonical_json,
    digest,
    plan_digest,
    write_json,
)
from policy import Policy, PolicyError, load as load_policy, validate as validate_policy


class PlanError(ValueError):
    """Raised when selected work cannot be compiled into a valid plan."""

    def __init__(self, issues: Iterable[Issue]) -> None:
        self.issues = tuple(issues)
        super().__init__("; ".join(issue.message for issue in self.issues))


@dataclass
class _PytestGroup:
    suite: str
    device: str
    image: CatalogImage
    venv_name: str
    entries: list[TestEntry] = field(default_factory=list)
    max_ranks: int = 1


def _issue(path: str, message: str, code: str = "invalid-plan") -> Issue:
    return Issue(code=code, path=path, message=message)


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return value or "job"


def _marker(device: str) -> str:
    if device == "cpu":
        return "cpu"
    return f"{device} or multi_{device}"


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_for(root: Path, *, repository: str = "", ref: str = "", sha: str = "") -> Source:
    remote = _git(root, "config", "--get", "remote.origin.url")
    return Source(
        repository=repository or remote,
        ref=ref or _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        sha=sha or _git(root, "rev-parse", "HEAD"),
    )


def _runtime(catalog: Catalog, device: str, ranks: int, image: CatalogImage,
             venv_name: str, lane: CatalogLane) -> Runtime:
    return Runtime(
        device=device,
        ranks=ranks,
        lane=lane.name,
        image=image.as_ref(),
        venv=catalog.venvs[venv_name],
    )


def _pytest_job_id(suite: str, device: str, image: str, venv: str) -> str:
    return _slug(f"pytest-{suite}-{device}-{image}-{venv}")


def _build_pytest_jobs(catalog: Catalog, policy: Policy,
                       issues: list[Issue]) -> tuple[list[Job], dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], _PytestGroup] = {}
    coverage = {
        suite: {"mode": policy.pytest[suite], "files": 0, "routes": 0}
        for suite in ("core", "model", "adapter")
    }
    selected_files = {suite: set() for suite in coverage}
    selected_routes = {suite: set() for suite in coverage}

    for entry in catalog.tests:
        if not policy.selects_suite(entry.suite):
            continue
        selected_files[entry.suite].add(entry.path)
        route_errors: list[Issue] = []
        route_count = 0
        for device in entry.devices:
            ranks = entry.ranks_by_device.get(device, 1)
            try:
                image = catalog.image_for(device, entry.image_name)
                catalog.lane_for(device, image.name, ranks)
            except CatalogError as exc:
                route_errors.extend(exc.issues)
                continue
            route_count += 1
            selected_routes[entry.suite].add((entry.path, device))
            key = (entry.suite, device, image.name, entry.venv_name)
            group = groups.setdefault(
                key,
                _PytestGroup(
                    suite=entry.suite,
                    device=device,
                    image=image,
                    venv_name=entry.venv_name,
                ),
            )
            group.entries.append(entry)
            group.max_ranks = max(group.max_ranks, ranks)
        if route_count == 0:
            issues.extend(route_errors or [
                _issue(entry.path, "selected test has no executable route")
            ])

    jobs: list[Job] = []
    seen_ids: set[str] = set()
    for key in sorted(groups):
        group = groups[key]
        lane = catalog.lane_for(group.device, group.image.name, group.max_ranks)
        job_id = _pytest_job_id(*key)
        if job_id in seen_ids:
            issues.append(_issue(job_id, "duplicate pytest job id"))
            continue
        seen_ids.add(job_id)
        steps_by_key: dict[tuple[str, str], list[TestEntry]] = {}
        for entry in group.entries:
            steps_by_key.setdefault((entry.process_group, _marker(group.device)), []).append(entry)
        steps: list[Step] = []
        for (process_group, marker), entries in sorted(steps_by_key.items()):
            files = tuple(sorted({entry.path for entry in entries}))
            ranks = max(entry.ranks_by_device.get(group.device, 1) for entry in entries)
            frameworks = tuple(sorted({value for entry in entries for value in entry.frameworks}))
            families = tuple(sorted({value for entry in entries for value in entry.model_families}))
            tags = tuple(sorted({value for entry in entries for value in entry.tags}))
            steps.append(Step(
                id=f"{job_id}/{_slug(process_group)}-{_slug(marker)}",
                kind="pytest",
                ranks=ranks,
                timeout_seconds=catalog.pytest.step_timeout_seconds[group.device],
                spec=PytestSpec(
                    suite=group.suite,
                    frameworks=frameworks,
                    model_families=families,
                    marker=marker,
                    process_group=process_group,
                    tags=tags,
                    files=files,
                ),
            ))
        jobs.append(Job(
            id=job_id,
            kind="pytest",
            allowed_skip_reasons=policy.allow_skip.get(group.device, ()),
            timeout_minutes=catalog.pytest.job_timeout_minutes[group.device],
            runtime=_runtime(
                catalog,
                group.device,
                group.max_ranks,
                group.image,
                group.venv_name,
                lane,
            ),
            steps=tuple(steps),
        ))

    for suite, suite_coverage in coverage.items():
        suite_coverage["files"] = len(selected_files[suite])
        suite_coverage["routes"] = len(selected_routes[suite])
        if policy.pytest[suite] == "all" and not selected_files[suite]:
            issues.append(_issue(
                f"policies.{policy.name}.pytest.{suite}",
                "selected suite contains no tests",
            ))
    return jobs, coverage


def _model_spec(case: ModelE2ECase, defaults: Any) -> ModelE2ESpec:
    return ModelE2ESpec(
        case_id=case.case_id,
        model=case.model,
        framework=case.framework,
        run_dir=case.run_dir,
        workdir=case.workdir,
        install=CommandSpec(script=case.install_script, args=case.install_args),
        train=CommandSpec(script=case.train_script, args=case.train_args),
        hf_repo=case.hf_repo,
        layers=case.layers,
        dataset=case.dataset,
        ranks=case.ranks,
        parallelism_sp=case.sp,
        parallelism_ep=case.ep,
        seq_len=case.seq_len,
        finetuning_type=case.finetuning_type,
        fetch_timeout_seconds=defaults.fetch_timeout_seconds,
        case_timeout_seconds=case.timeout_seconds,
    )


def _build_model_e2e_jobs(catalog: Catalog, policy: Policy,
                          issues: list[Issue]) -> tuple[list[Job], dict[str, Any]]:
    selected = [case for case in catalog.cases if policy.selects_case(case.case_id)]
    jobs: list[Job] = []
    for case in selected:
        try:
            image = catalog.image_for(case.device, case.image_name)
            lane = catalog.lane_for(case.device, image.name, case.ranks)
        except CatalogError as exc:
            issues.extend(exc.issues)
            continue
        job_id = _slug(f"model-e2e-{case.case_id}")
        jobs.append(Job(
            id=job_id,
            kind="model_e2e",
            allowed_skip_reasons=policy.allow_skip.get(case.device, ()),
            timeout_minutes=catalog.model_e2e_defaults.job_timeout_minutes,
            runtime=_runtime(
                catalog,
                case.device,
                case.ranks,
                image,
                case.venv_name,
                lane,
            ),
            steps=(
                Step(
                    id=f"{job_id}/run",
                    kind="model_e2e",
                    ranks=case.ranks,
                    timeout_seconds=case.timeout_seconds,
                    spec=_model_spec(case, catalog.model_e2e_defaults),
                ),
            ),
        ))
    if policy.model_e2e_mode == "all" and not selected:
        issues.append(_issue(
            f"policies.{policy.name}.model_e2e",
            "all mode selected no E2E cases",
        ))
    return jobs, {
        "mode": policy.model_e2e_mode,
        "cases": [case.case_id for case in selected],
    }


def _plan_inputs(catalog: Catalog, policy: Policy, change_scope: str) -> dict[str, str]:
    return {
        "catalog_digest": catalog.digest,
        "policy_digest": policy.digest,
        "image_refs_digest": digest({
            name: image.ref for name, image in sorted(catalog.images.items())
        }),
        "change_scope": change_scope,
    }


def build(
    catalog: Catalog,
    policy: Policy,
    source: Source,
    *,
    change_scope: str = "code",
) -> Plan:
    """Build a ready plan or raise PlanError with structured issues."""
    if change_scope == "docs_only":
        return _build_docs_only_plan(catalog, policy, source)
    if change_scope != "code":
        raise ValueError(f"unknown change scope {change_scope!r}")
    issues = validate_policy(policy, catalog)
    inputs = _plan_inputs(catalog, policy, change_scope)
    pytest_jobs, pytest_coverage = _build_pytest_jobs(catalog, policy, issues)
    e2e_jobs, e2e_coverage = _build_model_e2e_jobs(catalog, policy, issues)
    jobs = sorted((*pytest_jobs, *e2e_jobs), key=lambda job: job.id)
    if issues:
        raise PlanError(issues)
    coverage = {
        "pytest": pytest_coverage,
        "model_e2e": e2e_coverage,
        "jobs": len(jobs),
    }
    return build_plan(
        policy=policy.name,
        source=source,
        inputs=inputs,
        coverage=coverage,
        jobs=jobs,
    )


def _build_docs_only_plan(catalog: Catalog, policy: Policy, source: Source) -> Plan:
    issues = validate_policy(policy, catalog)
    if issues:
        raise PlanError(issues)
    inputs = _plan_inputs(catalog, policy, "docs_only")
    coverage = {
        "pytest": {
            suite: {"mode": "none", "files": 0, "routes": 0}
            for suite in ("core", "model", "adapter")
        },
        "model_e2e": {"mode": "none", "cases": []},
        "jobs": 0,
    }
    return build_plan(
        policy=policy.name,
        source=source,
        inputs=inputs,
        coverage=coverage,
        jobs=(),
    )


def _target_plan(plan: Plan, target_job: str) -> Plan:
    """Reduce a validated Plan to one job for an explicit manual run."""
    jobs = tuple(job for job in plan.jobs if job.id == target_job)
    if not jobs:
        raise PlanError([_issue(
            target_job,
            f"target job {target_job!r} is not in the full plan",
            "unknown-job",
        )])

    pytest_files = {suite: set() for suite in ("core", "model", "adapter")}
    pytest_routes = {suite: 0 for suite in ("core", "model", "adapter")}
    e2e_cases: list[str] = []
    for job in jobs:
        for step in job.steps:
            if isinstance(step.spec, PytestSpec):
                pytest_files[step.spec.suite].update(step.spec.files)
                pytest_routes[step.spec.suite] += len(step.spec.files)
            elif isinstance(step.spec, ModelE2ESpec):
                e2e_cases.append(step.spec.case_id)

    pytest_coverage = {
        suite: {
            "mode": "explicit" if files else "none",
            "files": len(files),
            "routes": pytest_routes[suite],
        }
        for suite, files in pytest_files.items()
    }
    coverage = {
        "pytest": pytest_coverage,
        "model_e2e": {
            "mode": "explicit" if e2e_cases else "none",
            "cases": sorted(e2e_cases),
        },
        "jobs": len(jobs),
    }
    inputs = {**plan.inputs, "target_job": target_job}
    return build_plan(
        policy=plan.policy,
        source=plan.source,
        inputs=inputs,
        coverage=coverage,
        jobs=jobs,
    )


def validate_plan(plan: Plan, catalog: Catalog, policy: Policy) -> list[Issue]:
    """Validate a plan against the catalog and policy that produced it."""
    issues: list[Issue] = []
    change_scope = plan.inputs.get("change_scope", "code")
    if plan.schema_version != SCHEMA_VERSION:
        issues.append(_issue("plan.schema_version", f"expected {SCHEMA_VERSION}"))
    if plan.policy != policy.name:
        issues.append(_issue("plan.policy", f"expected {policy.name!r}"))
    expected_plan_id = plan_digest(plan)
    if plan.plan_id != expected_plan_id:
        issues.append(_issue(
            "plan.plan_id",
            f"does not match plan contents; expected {expected_plan_id}",
        ))
    expected_inputs = {
        "catalog_digest": catalog.digest,
        "policy_digest": policy.digest,
        "image_refs_digest": digest({
            name: image.ref for name, image in sorted(catalog.images.items())
        }),
        "change_scope": change_scope,
    }
    if plan.inputs != expected_inputs:
        issues.append(_issue("plan.inputs", "inputs do not match Catalog and Policy"))
    if change_scope == "docs_only":
        if plan.jobs:
            issues.append(_issue("plan.jobs", "docs-only plan must not contain jobs"))
        expected_coverage = {
            "pytest": {
                suite: {"mode": "none", "files": 0, "routes": 0}
                for suite in ("core", "model", "adapter")
            },
            "model_e2e": {"mode": "none", "cases": []},
            "jobs": 0,
        }
        if plan.coverage != expected_coverage:
            issues.append(_issue("plan.coverage", "docs-only plan must have zero coverage"))
        return issues
    if change_scope != "code":
        issues.append(_issue("plan.inputs.change_scope", f"unknown scope {change_scope!r}"))
        return issues

    entries_by_path = {entry.path: entry for entry in catalog.tests}
    cases_by_id = {case.case_id: case for case in catalog.cases}
    expected_tests: set[tuple[str, str]] = set()
    expected_files = {suite: set() for suite in ("core", "model", "adapter")}
    expected_routes = {suite: set() for suite in ("core", "model", "adapter")}
    expected_cases: set[str] = set()
    for entry in catalog.tests:
        if not policy.selects_suite(entry.suite):
            continue
        expected_files[entry.suite].add(entry.path)
        for device in entry.devices:
            ranks = entry.ranks_by_device.get(device, 1)
            try:
                image = catalog.image_for(device, entry.image_name)
                catalog.lane_for(device, image.name, ranks)
            except CatalogError:
                continue
            expected_tests.add((entry.path, device))
            expected_routes[entry.suite].add((entry.path, device))
    for case in catalog.cases:
        if policy.selects_case(case.case_id):
            expected_cases.add(case.case_id)

    ids: set[str] = set()
    step_ids: set[str] = set()
    seen_tests: set[tuple[str, str]] = set()
    seen_cases: set[str] = set()
    previous_job_id = ""
    for job_index, job in enumerate(plan.jobs):
        job_path = f"plan.jobs[{job_index}]"
        if job.id < previous_job_id:
            issues.append(_issue(job_path, "jobs are not sorted by id"))
        previous_job_id = job.id
        if job.id in ids:
            issues.append(_issue(job.id, "duplicate job id"))
        ids.add(job.id)
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", job.id):
            issues.append(_issue(job.id, "job id must use lowercase letters, numbers, and hyphens"))
        if job.kind not in ("pytest", "model_e2e"):
            issues.append(_issue(job.id, f"unknown job kind {job.kind!r}"))
        if not job.steps:
            issues.append(_issue(job.id, "job has no steps"))

        lane = catalog.lanes.get(job.runtime.lane)
        if lane is None:
            issues.append(_issue(job.id, "unknown lane"))
        else:
            if lane.device != job.runtime.device:
                issues.append(_issue(job.id, "lane device mismatch"))
            if lane.capacity_slots < job.runtime.ranks:
                issues.append(_issue(job.id, "lane capacity is smaller than ranks"))
        image = catalog.images.get(job.runtime.image.name)
        if image is None:
            issues.append(_issue(job.id, "unknown image"))
        elif image.as_ref().to_dict() != job.runtime.image.to_dict():
            issues.append(_issue(job.id, "image does not match Catalog"))
        venv = catalog.venvs.get(job.runtime.venv.name)
        if venv is None:
            issues.append(_issue(job.id, "unknown venv"))
        elif venv.to_dict() != job.runtime.venv.to_dict():
            issues.append(_issue(job.id, "venv does not match Catalog"))
        expected_skip = set(policy.allow_skip.get(job.runtime.device, ()))
        if set(job.allowed_skip_reasons) != expected_skip:
            issues.append(_issue(job.id, "allowed skip reasons do not match Policy"))
        max_step_ranks = max((step.ranks for step in job.steps), default=0)
        if job.runtime.ranks < max_step_ranks:
            issues.append(_issue(job.id, "runtime ranks are smaller than step ranks"))

        job_tests: set[tuple[str, str]] = set()
        job_cases: set[str] = set()
        for step in job.steps:
            if step.id in step_ids:
                issues.append(_issue(step.id, "duplicate step id"))
            step_ids.add(step.id)
            if step.kind != job.kind:
                issues.append(_issue(step.id, "step kind does not match job kind"))
            if step.ranks > job.runtime.ranks:
                issues.append(_issue(step.id, "step ranks exceed runtime ranks"))
            if step.kind == "pytest":
                spec = step.spec
                if not isinstance(spec, PytestSpec):
                    issues.append(_issue(step.id, "pytest step has wrong spec type"))
                    continue
                if spec.suite not in ("core", "model", "adapter"):
                    issues.append(_issue(step.id, f"unknown pytest suite {spec.suite!r}"))
                if spec.marker != _marker(job.runtime.device):
                    issues.append(_issue(step.id, "pytest marker does not match runtime device"))
                if not spec.files:
                    issues.append(_issue(step.id, "pytest step has no files"))
                step_entries: list[TestEntry] = []
                for path in spec.files:
                    route = (path, job.runtime.device)
                    entry = entries_by_path.get(path)
                    if entry is None:
                        issues.append(_issue(path, "plan contains a test outside Catalog"))
                        continue
                    step_entries.append(entry)
                    if entry.suite != spec.suite:
                        issues.append(_issue(path, "pytest suite does not match Catalog"))
                    if route in seen_tests:
                        issues.append(_issue(path, f"duplicate test route for {job.runtime.device}"))
                    seen_tests.add(route)
                    job_tests.add(route)
                if step_entries:
                    if any(entry.process_group != spec.process_group for entry in step_entries):
                        issues.append(_issue(step.id, "pytest process group does not match Catalog"))
                    expected_spec = PytestSpec(
                        suite=spec.suite,
                        frameworks=tuple(sorted({
                            value for entry in step_entries for value in entry.frameworks
                        })),
                        model_families=tuple(sorted({
                            value for entry in step_entries for value in entry.model_families
                        })),
                        marker=spec.marker,
                        process_group=spec.process_group,
                        tags=tuple(sorted({
                            value for entry in step_entries for value in entry.tags
                        })),
                        files=tuple(sorted({entry.path for entry in step_entries})),
                    )
                    if spec.to_dict() != expected_spec.to_dict():
                        issues.append(_issue(step.id, "pytest spec does not match Catalog"))
            elif step.kind == "model_e2e":
                spec = step.spec
                if not isinstance(spec, ModelE2ESpec):
                    issues.append(_issue(step.id, "model_e2e step has wrong spec type"))
                    continue
                if len(job.steps) != 1:
                    issues.append(_issue(job.id, "model_e2e job must contain exactly one step"))
                if spec.ranks != job.runtime.ranks or step.ranks != job.runtime.ranks:
                    issues.append(_issue(step.id, "model_e2e ranks do not match runtime"))
                case = cases_by_id.get(spec.case_id)
                if case is None:
                    issues.append(_issue(step.id, f"unknown E2E case {spec.case_id!r}"))
                    continue
                expected_spec = _model_spec(case, catalog.model_e2e_defaults)
                if spec.to_dict() != expected_spec.to_dict():
                    issues.append(_issue(step.id, "E2E spec does not match Catalog"))
                if job.runtime.device != case.device:
                    issues.append(_issue(job.id, "E2E device does not match Catalog"))
                if job.runtime.venv.name != case.venv_name:
                    issues.append(_issue(job.id, "E2E venv does not match Catalog"))
                if spec.case_id in seen_cases:
                    issues.append(_issue(spec.case_id, "duplicate E2E case"))
                seen_cases.add(spec.case_id)
                job_cases.add(spec.case_id)
            else:
                issues.append(_issue(step.id, f"unknown step kind {step.kind!r}"))
        if job.kind == "pytest" and job_cases:
            issues.append(_issue(job.id, "pytest job contains a model_e2e step"))
        if job.kind == "model_e2e" and job_tests:
            issues.append(_issue(job.id, "model_e2e job contains a pytest step"))

    missing_tests = sorted(expected_tests - seen_tests)
    for path, device in missing_tests:
        issues.append(_issue(path, f"selected test route is missing for {device}", "missing-test"))
    extra_tests = sorted(seen_tests - expected_tests)
    for path, device in extra_tests:
        issues.append(_issue(path, f"plan contains an unselected test route for {device}", "extra-test"))
    missing_cases = sorted(expected_cases - seen_cases)
    for case_id in missing_cases:
        issues.append(_issue(case_id, "selected E2E case is missing", "missing-case"))
    extra_cases = sorted(seen_cases - expected_cases)
    for case_id in extra_cases:
        issues.append(_issue(case_id, "plan contains an unselected E2E case", "extra-case"))

    coverage = plan.coverage
    if coverage.get("jobs") != len(plan.jobs):
        issues.append(_issue("plan.coverage.jobs", "job count does not match plan.jobs"))
    pytest_coverage = coverage.get("pytest")
    if not isinstance(pytest_coverage, dict) or set(pytest_coverage) != {"core", "model", "adapter"}:
        issues.append(_issue("plan.coverage.pytest", "must contain core, model, and adapter"))
    else:
        for suite in ("core", "model", "adapter"):
            value = pytest_coverage[suite]
            if not isinstance(value, dict):
                issues.append(_issue(f"plan.coverage.pytest.{suite}", "must be a mapping"))
                continue
            expected = {
                "mode": policy.pytest[suite],
                "files": len(expected_files[suite]),
                "routes": len(expected_routes[suite]),
            }
            if value != expected:
                issues.append(_issue(
                    f"plan.coverage.pytest.{suite}",
                    f"expected {expected}",
                ))
    e2e_coverage = coverage.get("model_e2e")
    expected_e2e = {
        "mode": policy.model_e2e_mode,
        "cases": [case.case_id for case in catalog.cases if policy.selects_case(case.case_id)],
    }
    if e2e_coverage != expected_e2e:
        issues.append(_issue("plan.coverage.model_e2e", f"expected {expected_e2e}"))
    return issues


def artifact_for(
    catalog: Catalog | None,
    policy_name: str,
    source: Source,
    issues: Iterable[Issue],
    target_job: str = "",
    change_scope: str = "code",
) -> PlanArtifact:
    values = tuple(issues)
    if catalog is None or values:
        return build_plan_artifact(policy=policy_name, source=source, errors=values)
    try:
        policy = load_policy(catalog.root / "scripts" / "ci" / "policies.yml", policy_name)
        plan = build(catalog, policy, source, change_scope=change_scope)
        plan_issues = validate_plan(plan, catalog, policy)
        if not plan_issues and target_job:
            plan = _target_plan(plan, target_job)
    except (PlanError, PolicyError) as exc:
        return build_plan_artifact(
            policy=policy_name,
            source=source,
            errors=exc.issues,
        )
    if plan_issues:
        return build_plan_artifact(policy=policy_name, source=source, errors=plan_issues)
    return build_plan_artifact(policy=policy_name, source=source, plan=plan)


def _matrix_entry(job: Job, catalog: Catalog) -> dict[str, Any]:
    lane = catalog.lanes[job.runtime.lane]
    return {
        "job_id": job.id,
        "image_ref": job.runtime.image.ref,
        "runner_variable": lane.runner_variable,
        "runner": lane.runner,
        "timeout_minutes": job.timeout_minutes,
    }


def matrix_outputs(plan: Plan, catalog: Catalog) -> dict[str, list[dict[str, Any]]]:
    """Build platform matrix entries without embedding full job payloads."""
    output = {f"{device}_jobs": [] for device in ("cpu", "npu", "gpu")}
    for job in plan.jobs:
        entry = _matrix_entry(job, catalog)
        output[f"{job.runtime.device}_jobs"].append(entry)
    return output


def _write_github_output(path: Path, key: str, value: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"{key}={value}\n")


def _render(issues: Iterable[Issue]) -> None:
    for issue in issues:
        print(f"{issue.path}: {issue.message}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--policy", default="pull_request")
    parser.add_argument("--output", default="")
    parser.add_argument("--github-output", default="")
    parser.add_argument("--image-ref", action="append", default=[], metavar="NAME=REF")
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-ref", default="")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--fail-code", default="planner-precondition")
    parser.add_argument("--fail-path", default="planner")
    parser.add_argument("--fail-message", default="")
    parser.add_argument(
        "--change-scope",
        choices=("code", "docs_only"),
        default="code",
        help="docs_only emits a valid zero-job plan",
    )
    parser.add_argument(
        "--target-job", default="",
        help="reduce a validated plan to one exact job for a manual run",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    if args.verify and args.target_job:
        parser.error("--target-job cannot be combined with --verify")
    if args.target_job and args.change_scope == "docs_only":
        parser.error("--target-job cannot be combined with a docs-only plan")

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    source = source_for(
        root,
        repository=args.source_repository,
        ref=args.source_ref,
        sha=args.source_sha,
    )
    image_refs: dict[str, str] = {}
    for value in args.image_ref:
        name, separator, ref = value.partition("=")
        if not separator or not name or not ref:
            parser.error(f"bad --image-ref {value!r}; expected NAME=REF")
        image_refs[name] = ref

    if args.fail_message:
        issue = Issue(
            code=args.fail_code,
            path=args.fail_path,
            message=args.fail_message,
        )
        artifact = artifact_for(None, args.policy, source, (issue,))
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(artifact.errors)
        return 1

    try:
        catalog = load_catalog(root, image_refs=image_refs)
    except CatalogError as exc:
        artifact = artifact_for(None, args.policy, source, exc.issues)
        if args.output:
            write_json(Path(args.output), artifact.to_dict())
        _render(exc.issues)
        return 1

    if args.verify:
        failures: list[Issue] = []
        for policy_name in ("pull_request", "nightly"):
            try:
                policy = load_policy(root / "scripts" / "ci" / "policies.yml", policy_name)
                plan = build(catalog, policy, source)
                docs_plan = build(catalog, policy, source, change_scope="docs_only")
                failures.extend(validate_plan(plan, catalog, policy))
                failures.extend(validate_plan(docs_plan, catalog, policy))
                print(f"{policy_name}: {len(plan.jobs)} jobs / {len(catalog.tests)} tests")
                print(f"{policy_name} docs-only: {len(docs_plan.jobs)} jobs")
            except (PolicyError, PlanError) as exc:
                failures.extend(exc.issues)
        if failures:
            _render(failures)
            return 1
        print(f"planner OK ({len(catalog.cases)} E2E cases / {len(catalog.tests)} tests)")
        return 0

    artifact = artifact_for(
        catalog,
        args.policy,
        source,
        (),
        target_job=args.target_job,
        change_scope=args.change_scope,
    )
    if args.output:
        write_json(Path(args.output), artifact.to_dict())
    if artifact.status != "ready" or artifact.plan is None:
        _render(artifact.errors)
        return 1
    if args.github_output:
        output_path = Path(args.github_output)
        for key, value in matrix_outputs(artifact.plan, catalog).items():
            _write_github_output(
                output_path,
                key,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            )
            _write_github_output(output_path, f"{key[:-1]}_count", str(len(value)))
    if not args.output:
        print(canonical_json(artifact.plan.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

