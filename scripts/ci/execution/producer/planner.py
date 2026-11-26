#!/usr/bin/env python3
"""Compile CI configuration and test facts into an execution plan."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from ...contracts.base import (
    SUITES,
    Diagnostic,
    DiagnosticError,
    GitSource,
    SCHEMA_VERSION,
)
from ...contracts.image_contract import ResolvedImage
from ...contracts.planner_contract import (
    CHANGE_MODES,
    ExecutionPlan,
    IntegrationSpec,
    Job,
    PlannerOutput,
    PytestSpec,
    Runtime,
    TestTask,
    plan_digest,
)
from ...image.image_config import RuntimeImage
from .config import (
    CiConfig,
    CiConfigError,
    CiRunner,
    CiTarget,
    IntegrationSuite,
    TestProfile,
    area_selected,
)
from .facts import (
    INTEGRATION_RUNNER_SCRIPT,
    IntegrationFact,
    PytestFact,
    TestFacts,
    test_parts,
)


class PlanError(DiagnosticError):
    """Raised when selected work cannot be compiled into a valid plan."""


@dataclass(frozen=True)
class PlanOptions:
    """Invocation options that select and shape one execution plan."""

    ci_type: str
    git_source: GitSource
    change_mode: str = "code"
    image_filter: str = ""
    target_filter: str = ""
    job_filter: str = ""


@dataclass(frozen=True)
class ImageSelection:
    index: int
    image: RuntimeImage
    test_areas: tuple[str, ...] | None


@dataclass
class _PytestGroup:
    suite: str
    device: str
    target: CiTarget
    image: RuntimeImage
    venv_name: str
    entries: dict[str, PytestFact] = field(default_factory=dict)
    max_ranks: int = 1


@dataclass
class _PlanState:
    """Mutable planning state shared by pytest and integration builders."""

    config: CiConfig
    facts: TestFacts
    profile: TestProfile
    selections: tuple[ImageSelection, ...]
    issues: list[Diagnostic]
    used_areas: set[tuple[int, str]] = field(default_factory=set)
    used_selections: set[int] = field(default_factory=set)

    def use_area(self, selection: ImageSelection, area_id: str) -> None:
        self.used_areas.add((selection.index, area_id))
        self.used_selections.add(selection.index)


def _issue(path: str, message: str, code: str = "invalid-plan") -> Diagnostic:
    """Create a planner diagnostic without raising."""
    return Diagnostic(code=code, location=path, message=message)


def slugify(value: str) -> str:
    """Return a lowercase identifier safe for Job IDs."""
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "job"


def _marker(device: str) -> str:
    """Return the pytest marker expression for one device lane."""
    if device == "cpu":
        return "cpu or (not npu and not multi_npu and not gpu and not multi_gpu)"
    return f"{device} or multi_{device}"


def _task_group(fact: PytestFact) -> str:
    """Group test files that should execute in the same test task."""
    if fact.model_families:
        return f"family:{fact.model_families[0]}"

    parts = test_parts(fact.path)
    if parts[:1] == ("models",) and len(parts) == 2:
        return "layer:tests/models"
    return f"layer:{'/'.join(Path(fact.path).parts[:-1])}"


def _claim_job_id(base: str, seen: set[str], suffix: str) -> str:
    """Return a unique, stable Job ID within one plan."""
    if base not in seen:
        seen.add(base)
        return base
    candidate = slugify(f"{base}-{suffix}")
    index = 2
    while candidate in seen:
        candidate = slugify(f"{base}-{suffix}-{index}")
        index += 1
    seen.add(candidate)
    return candidate


def _values(value: str) -> tuple[str, ...]:
    """Split a comma/space-separated filter without duplicating values."""
    return tuple(dict.fromkeys(
        part for part in re.split(r"[,\s]+", value.strip()) if part
    ))


def _git(root: Path, *args: str) -> str:
    """Run a read-only git command and return empty text on failure."""
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def source_for(
    root: Path, *, repository: str = "", ref: str = "", sha: str = ""
) -> GitSource:
    """Resolve the source identity used by Plan metadata."""
    return GitSource(
        repository=repository or _git(root, "config", "--get", "remote.origin.url"),
        git_ref=ref or _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        commit_sha=sha or _git(root, "rev-parse", "HEAD"),
    )


def _active_selections(
    config: CiConfig,
    profile: TestProfile,
    image_filter: str,
    target_filter: str,
    issues: list[Diagnostic],
) -> tuple[ImageSelection, ...]:
    """Resolve profile selections after image and target filters."""
    if image_filter and image_filter not in config.images:
        issues.append(_issue(
            "image_filter", f"unknown image {image_filter!r}", "unknown-image"
        ))
        return ()
    selected: list[ImageSelection] = []
    for index, selection in enumerate(profile.selections):
        if image_filter and selection.image != image_filter:
            continue
        configured = config.images[selection.image]
        selected.append(ImageSelection(index, configured, selection.test_areas))
    if image_filter and not selected:
        issues.append(_issue(
            "image_filter",
            f"test profile {profile.name!r} does not select image {image_filter!r}",
            "unused-image-filter",
        ))
    target_values = set(_values(target_filter))
    return tuple(
        selection for selection in selected
        if not target_values or selection.image.target in target_values
    )


def requested_images(
    config: CiConfig,
    options: PlanOptions,
) -> tuple[str, ...]:
    """Return the container image IDs selected by a test profile."""
    try:
        profile = config.test_profile(options.ci_type)
    except CiConfigError as exc:
        raise PlanError(exc.issues) from exc
    issues: list[Diagnostic] = []
    selections = _active_selections(
        config, profile, options.image_filter, options.target_filter, issues
    )
    if issues:
        raise PlanError(issues)
    return tuple(sorted(
        selection.image.image_id for selection in selections
        if selection.test_areas != () and selection.image.kind == "container"
    ))


def _runtime(
    config: CiConfig,
    device: str,
    target: str,
    ranks: int,
    image: RuntimeImage,
    venv_name: str,
    runner: CiRunner,
) -> Runtime:
    """Expand configuration references into the runtime contract."""
    return Runtime(
        device=device, target=target, ranks=ranks, runner=runner.name,
        image=ResolvedImage(
            image_id=image.image_id, target=image.target,
            kind=image.kind, ref=image.ref,
        ),
        venv=config.venvs[venv_name],
    )


def _pytest_job(
    state: _PlanState, group: _PytestGroup, job_ids: set[str]
) -> Job | None:
    """Build one pytest Job or record why it cannot be built."""
    try:
        runner = state.config.runner_for(group.target.name, group.max_ranks)
    except CiConfigError as exc:
        state.issues.extend(exc.issues)
        return None
    job_timeout = state.config.pytest_areas.job_timeout_minutes.get(group.device)
    step_timeout = state.config.pytest_areas.step_timeout_minutes.get(group.device)
    if job_timeout is None or step_timeout is None:
        state.issues.append(_issue(
            f"test_areas.unit_tests.defaults.{group.device}",
            "missing pytest timeout for selected device",
        ))
        return None
    job_id = _claim_job_id(
        slugify(
            f"pytest-{group.suite}-{group.device}-"
            f"{group.image.image_id}-{group.venv_name}"
        ),
        job_ids,
        runner.name,
    )
    tasks_by_key: dict[tuple[str, str], list[PytestFact]] = {}
    for entry in group.entries.values():
        key = (_task_group(entry), _marker(group.device))
        tasks_by_key.setdefault(key, []).append(entry)
    label = "cpu" if group.device == "cpu" else f"{group.device}-or-multi-{group.device}"
    tasks = [
        TestTask(
            id=f"{job_id}/{slugify(task_group)}-{slugify(label)}",
            timeout_minutes=step_timeout,
            spec=PytestSpec(
                suite=group.suite, device_marker=marker,
                test_files=tuple(sorted({entry.path for entry in entries})),
            ),
        )
        for (task_group, marker), entries in sorted(tasks_by_key.items())
    ]
    return Job(
        id=job_id, kind="pytest", allowed_skip_reasons=(),
        timeout_minutes=job_timeout,
        runtime=_runtime(
            state.config, group.device, group.target.name, group.max_ranks,
            group.image, group.venv_name, runner,
        ),
        tasks=tuple(tasks),
    )


def _pytest_jobs(state: _PlanState) -> list[Job]:
    """Group selected pytest facts into Jobs."""
    groups: dict[tuple[str, ...], _PytestGroup] = {}
    selected_suites = {
        suite for suite in SUITES
        if any(area_selected(s.test_areas, f"pytest/{suite}") for s in state.selections)
    }
    suites_with_files = {
        fact.suite for fact in state.facts.pytest if fact.suite in selected_suites
    }
    venvs: dict[str, str] = {}
    for selection in state.selections:
        target = state.config.target_for(selection.image.target)
        for fact in state.facts.pytest:
            if not area_selected(selection.test_areas, f"pytest/{fact.suite}"):
                continue
            if target.device not in fact.devices:
                continue
            if fact.path not in venvs:
                venv_name, venv_issues = state.config.resolve_pytest_venv(
                    fact.path, fact.model_families, fact.frameworks
                )
                state.issues.extend(venv_issues)
                venvs[fact.path] = venv_name
            venv_name = venvs[fact.path]
            if not venv_name:
                continue
            key = (fact.suite, target.device, target.name,
                   selection.image.image_id, venv_name)
            group = groups.setdefault(
                key,
                _PytestGroup(
                    suite=fact.suite, device=target.device, target=target,
                    image=selection.image, venv_name=venv_name,
                ),
            )
            group.entries.setdefault(fact.path, fact)
            group.max_ranks = max(
                group.max_ranks, fact.ranks_by_device.get(target.device, 1)
            )
            state.use_area(selection, f"pytest/{fact.suite}")

    jobs: list[Job] = []
    job_ids: set[str] = set()
    for key in sorted(groups):
        job = _pytest_job(state, groups[key], job_ids)
        if job is not None:
            jobs.append(job)
    for suite in selected_suites - suites_with_files:
        state.issues.append(_issue(
            f"test_profiles.{state.profile.name}.test_areas",
            "selected suite contains no tests",
            "empty-suite",
        ))
    return jobs


def _integration_resources(
    config: CiConfig, case: IntegrationFact, issues: list[Diagnostic]
) -> IntegrationSuite | None:
    """Validate the suite and CI resources a selected case resolves to."""
    where = f"integration/{case.suite}/{case.case_id}"
    valid = True
    if not config.device_is_configured(case.device):
        issues.append(_issue(
            f"{where}.device", f"no CI target for device {case.device!r}",
            "missing-integration-target",
        ))
        valid = False
    try:
        suite = config.integration_suite(case.suite)
    except CiConfigError as exc:
        issues.extend(exc.issues)
        return None
    if case.framework not in suite.framework_venvs:
        issues.append(_issue(
            f"test_areas.integration_tests.suites.{case.suite}.framework_venvs.{case.framework}",
            f"framework {case.framework!r} has no venv mapping",
            "missing-framework-venv",
        ))
        valid = False
    return suite if valid else None


def _selected_integration_cases(
    state: _PlanState,
) -> list[tuple[ImageSelection, IntegrationFact]]:
    """Return integration cases selected by the active image selections."""
    cases = {(case.suite, case.case_id): case for case in state.facts.integration}
    selected: list[tuple[ImageSelection, IntegrationFact]] = []
    for selection in state.selections:
        if selection.test_areas is None or "integration/all" in selection.test_areas:
            selected.extend((selection, case) for case in state.facts.integration)
            continue
        for area_id in selection.test_areas:
            if not area_id.startswith("integration/"):
                continue
            parts = area_id.split("/", 2)
            case = cases.get((parts[1], parts[2])) if len(parts) == 3 else None
            if case is None:
                state.issues.append(_issue(
                    f"test_profiles.{state.profile.name}.test_areas.integration_tests",
                    f"unknown integration case in TestArea {area_id!r}",
                    "unknown-case",
                ))
            else:
                selected.append((selection, case))

    available_suites = {case.suite for case in state.facts.integration}
    if any(
        selection.test_areas is None or "integration/all" in selection.test_areas
        for selection in state.selections
    ):
        for suite_name in state.config.integration_suites:
            if suite_name not in available_suites:
                state.issues.append(_issue(
                    f"test_profiles.{state.profile.name}.test_areas.integration_tests",
                    f"suite {suite_name!r} contains no integration cases",
                    "empty-case-suite",
                ))
    return selected


def _integration_job(
    state: _PlanState,
    selection: ImageSelection,
    case: IntegrationFact,
    resources: dict[tuple[str, str], IntegrationSuite | None],
    seen: set[tuple[str, str, str, str]],
    job_ids: set[str],
) -> Job | None:
    """Build one integration Job or record why it cannot be built."""
    case_key = (case.suite, case.case_id)
    if case_key not in resources:
        resources[case_key] = _integration_resources(state.config, case, state.issues)
    suite = resources[case_key]
    if suite is None:
        return None
    target = state.config.target_for(selection.image.target)
    if case.device != target.device:
        return None
    try:
        runner = state.config.runner_for(target.name, case.ranks)
    except CiConfigError as exc:
        state.issues.extend(exc.issues)
        return None
    key = (case.suite, case.case_id, target.name, selection.image.image_id)
    if key in seen:
        return None
    seen.add(key)
    try:
        job_timeout = suite.job_timeout_minutes_for(case.device, case.case_id)
        step_timeout = suite.step_timeout_minutes_for(case.device, case.case_id)
    except KeyError:
        state.issues.append(_issue(
            f"test_areas.integration_tests.suites.{case.suite}.defaults.{case.device}",
            "missing integration timeout for selected device",
            "missing-timeout",
        ))
        return None
    state.use_area(selection, f"integration/{case.suite}/{case.case_id}")
    job_id = _claim_job_id(
        slugify(f"integration-{case.suite}-{case.case_id}"),
        job_ids,
        f"{target.name}-{selection.image.image_id}-{runner.name}",
    )
    return Job(
        id=job_id, kind="integration", allowed_skip_reasons=(),
        timeout_minutes=job_timeout,
        runtime=_runtime(
            state.config, case.device, target.name, case.ranks,
            selection.image, suite.framework_venvs[case.framework], runner,
        ),
        tasks=(TestTask(
            id=f"{job_id}/run", timeout_minutes=step_timeout,
            spec=IntegrationSpec(
                suite=case.suite, case_id=case.case_id,
                framework=case.framework,
                runner_script=INTEGRATION_RUNNER_SCRIPT,
            ),
        ),),
    )


def _integration_jobs(state: _PlanState) -> list[Job]:
    """Build all selected integration Jobs."""
    resources: dict[tuple[str, str], IntegrationSuite | None] = {}
    seen: set[tuple[str, str, str, str]] = set()
    job_ids: set[str] = set()
    jobs: list[Job] = []
    for selection, case in _selected_integration_cases(state):
        job = _integration_job(state, selection, case, resources, seen, job_ids)
        if job is not None:
            jobs.append(job)
    return jobs


def build_execution_plan(
    *,
    ci_type: str,
    change_mode: str,
    git_source: GitSource,
    jobs: Iterable[Job],
) -> ExecutionPlan:
    """Build an execution plan and bind its content-addressed ID."""
    plan = ExecutionPlan(
        plan_id="", ci_type=ci_type, change_mode=change_mode,
        git_source=git_source, jobs=tuple(jobs),
    )
    return replace(plan, plan_id=plan_digest(plan))


def _filter_jobs(
    jobs: Iterable[Job], options: PlanOptions
) -> tuple[Job, ...]:
    """Apply the final target/job filter to a Job sequence."""
    source_jobs = tuple(jobs)
    if not options.target_filter and not options.job_filter:
        return source_jobs
    job_values = _values(options.job_filter)
    target_values = _values(options.target_filter)
    if job_values and target_values:
        raise PlanError([_issue(
            "selection", "job_filter and target_filter cannot be combined"
        )])
    if job_values:
        wanted = set(job_values)
        unknown = sorted(wanted - {job.id for job in source_jobs})
        if unknown:
            raise PlanError([
                _issue(
                    value,
                    f"target job {value!r} is not in the full plan",
                    "unknown-job",
                )
                for value in unknown
            ])
        return tuple(job for job in source_jobs if job.id in wanted)

    wanted = set(target_values)
    unknown = sorted(wanted - {job.runtime.target for job in source_jobs})
    if unknown:
        raise PlanError([
            _issue(
                value,
                f"target {value!r} has no jobs in the full plan",
                "unknown-target",
            )
            for value in unknown
        ])
    return tuple(job for job in source_jobs if job.runtime.target in wanted)


def build_plan(
    config: CiConfig,
    facts: TestFacts,
    options: PlanOptions,
) -> ExecutionPlan:
    """Build a ready plan or raise PlanError with structured issues."""
    if options.change_mode not in CHANGE_MODES:
        raise ValueError(f"unknown change mode {options.change_mode!r}")
    if options.target_filter and options.job_filter:
        raise PlanError([_issue(
            "selection", "target_filter and job_filter cannot be combined"
        )])
    try:
        profile = config.test_profile(options.ci_type)
    except CiConfigError as exc:
        raise PlanError(exc.issues) from exc
    jobs: tuple[Job, ...] = ()
    if options.change_mode != "code":
        return build_execution_plan(
            ci_type=profile.name,
            change_mode=options.change_mode,
            git_source=options.git_source,
            jobs=_filter_jobs((), options),
        )
    issues: list[Diagnostic] = []
    selections = _active_selections(
        config, profile, options.image_filter, options.target_filter, issues
    )
    if issues:
        raise PlanError(issues)
    issues.extend(
        _issue(
            f"resources.images.{selection.image.image_id}.ref",
            "missing ResolvedImage for container image",
            "missing-resolved-image",
        )
        for selection in selections
        if selection.test_areas != () and not selection.image.ref
    )
    if issues:
        raise PlanError(issues)
    state = _PlanState(config, facts, profile, selections, issues)
    jobs = tuple(sorted(
        (*_pytest_jobs(state), *_integration_jobs(state)),
        key=lambda job: job.id,
    ))
    for selection in selections:
        path = f"test_profiles.{profile.name}[{selection.index}]"
        if selection.test_areas != () and selection.index not in state.used_selections:
            issues.append(_issue(
                path,
                "selected image produced no jobs",
                "empty-selection",
            ))
        for area_id in selection.test_areas or ():
            if area_id.endswith("/all"):
                continue
            if (selection.index, area_id) not in state.used_areas:
                issues.append(_issue(
                    path,
                    f"TestArea {area_id!r} produced no jobs",
                    "empty-test-area",
                ))
    if options.image_filter and not any(
        job.runtime.image.image_id == options.image_filter for job in jobs
    ):
        issues.append(_issue(
            "image_filter",
            f"image {options.image_filter!r} was not selected by any job",
            "unused-image-filter",
        ))
    if issues:
        raise PlanError(issues)
    return build_execution_plan(
        ci_type=profile.name,
        change_mode=options.change_mode,
        git_source=options.git_source,
        jobs=_filter_jobs(jobs, options),
    )


def build_planner_output(
    *,
    errors: Iterable[Diagnostic] = (),
    plan: ExecutionPlan | None = None,
) -> PlannerOutput:
    """Build portable planner output for success or failure."""
    values = tuple(errors)
    if plan is None and not values:
        values = (_issue("planner", "planner produced no plan", "missing-plan"),)
    return PlannerOutput(
        schema_version=SCHEMA_VERSION,
        errors=values,
        plan=plan,
        status="ready" if plan is not None and not values else "failed",
    )


def planner_output_for(
    config: CiConfig,
    facts: TestFacts,
    options: PlanOptions,
    *,
    issues: Iterable[Diagnostic] = (),
) -> PlannerOutput:
    """Return portable planner output; convert PlanError into diagnostics."""
    values = tuple(issues)
    if values:
        return build_planner_output(errors=values)
    try:
        plan = build_plan(config, facts, options)
    except PlanError as exc:
        return build_planner_output(errors=exc.issues)
    return build_planner_output(plan=plan)
