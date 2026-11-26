#!/usr/bin/env python3
"""Load and validate the single CI configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from ...contracts.base import DEVICES, SUITES, Diagnostic, DiagnosticError
from ...contracts.planner_contract import VenvSpec
from ...image.image_config import ImageError, RuntimeImage, load_runtime_images

CONFIG_PATH = Path("scripts/ci/config/ci.yml")
PROFILE_NAMES = ("pull_request", "nightly")


class CiConfigError(DiagnosticError):
    """Raised when CI configuration or a cross-reference is invalid."""


@dataclass(frozen=True)
class CiTarget:
    name: str
    device: str


@dataclass(frozen=True)
class CiRunner:
    target: str
    rank_capacity: int
    runner: str
    container_options: str = "--user root"

    @property
    def name(self) -> str:
        return f"{self.target}/{self.rank_capacity}"


@dataclass(frozen=True)
class PytestAreas:
    job_timeout_minutes: dict[str, int]
    step_timeout_minutes: dict[str, int]
    model_venvs: dict[str, str]
    framework_venvs: dict[str, str]
    combo_venvs: dict[str, str]
    overrides: dict[str, str]


@dataclass(frozen=True)
class IntegrationOverride:
    job_timeout_minutes: int | None = None
    step_timeout_minutes: int | None = None


@dataclass(frozen=True)
class IntegrationSuite:
    name: str
    job_timeout_minutes: dict[str, int]
    step_timeout_minutes: dict[str, int]
    framework_venvs: dict[str, str]
    overrides: dict[str, IntegrationOverride]

    def job_timeout_minutes_for(self, device: str, case_id: str) -> int:
        override = self.overrides.get(case_id)
        if override is not None and override.job_timeout_minutes is not None:
            return override.job_timeout_minutes
        return self.job_timeout_minutes[device]

    def step_timeout_minutes_for(self, device: str, case_id: str) -> int:
        override = self.overrides.get(case_id)
        if override is not None and override.step_timeout_minutes is not None:
            return override.step_timeout_minutes
        return self.step_timeout_minutes[device]


@dataclass(frozen=True)
class ProfileSelection:
    image: str
    test_areas: tuple[str, ...] | None


def area_selected(test_areas: tuple[str, ...] | None, area_id: str) -> bool:
    if test_areas is None:
        return True
    group = area_id.split("/", 1)[0]
    return area_id in test_areas or f"{group}/all" in test_areas


@dataclass(frozen=True)
class TestProfile:
    name: str
    selections: tuple[ProfileSelection, ...]


@dataclass(frozen=True)
class CiConfig:
    root: Path
    targets: dict[str, CiTarget]
    images: dict[str, RuntimeImage]
    venvs: dict[str, VenvSpec]
    runners: tuple[CiRunner, ...]
    pytest_areas: PytestAreas
    integration_suites: dict[str, IntegrationSuite]
    test_profiles: dict[str, TestProfile]

    def target_for(self, name: str) -> CiTarget:
        target = self.targets.get(name)
        if target is None:
            raise CiConfigError([_issue(name, f"unknown target {name!r}")])
        return target

    def runner_for(self, target: str, ranks: int) -> CiRunner:
        candidates = [
            runner for runner in self.runners
            if runner.target == target and runner.rank_capacity >= ranks
        ]
        if not candidates:
            raise CiConfigError([_issue(
                target,
                f"target {target!r} has no runner with rank_capacity >= {ranks}",
            )])
        return min(candidates, key=lambda runner: runner.rank_capacity)

    def runner_named(self, name: str) -> CiRunner:
        for runner in self.runners:
            if runner.name == name:
                return runner
        raise CiConfigError([_issue(name, f"unknown runner {name!r}")])

    def test_profile(self, name: str) -> TestProfile:
        profile = self.test_profiles.get(name)
        if profile is None:
            raise CiConfigError([_issue(name, f"unknown test profile {name!r}")])
        return profile

    def device_is_configured(self, device: str) -> bool:
        return any(target.device == device for target in self.targets.values())

    def integration_suite(self, name: str) -> IntegrationSuite:
        suite = self.integration_suites.get(name)
        if suite is None:
            raise CiConfigError([_issue(name, f"unknown integration suite {name!r}")])
        return suite

    def resolve_pytest_venv(
        self, path: str, families: Iterable[str], frameworks: Iterable[str]
    ) -> tuple[str, list[Diagnostic]]:
        override = self.pytest_areas.overrides.get(path)
        if override is not None:
            return override, []
        family_set = set(families)
        framework_set = set(frameworks)
        if len(framework_set) > 1:
            return "", [_issue(
                path,
                "a multi-framework test needs an explicit override",
                "multi-framework-venv",
            )]
        family_venvs = {
            self.pytest_areas.model_venvs.get(family, "default")
            for family in family_set
        }
        if len(family_venvs) > 1:
            return "", [_issue(
                path,
                "a test cannot use multiple model-family venvs",
                "multi-family-venv",
            )]
        family_venv = next(iter(family_venvs), "default")
        framework = next(iter(framework_set), "")
        if family_set and framework and family_venv != "default":
            family = next(iter(family_set))
            combo = self.pytest_areas.combo_venvs.get(f"{family}+{framework}")
            if combo is None:
                return "", [_issue(
                    path,
                    f"no venv for family={family} framework={framework}",
                    "missing-combo-venv",
                )]
            return combo, []
        if framework:
            venv = self.pytest_areas.framework_venvs.get(framework)
            if venv is None:
                return "", [_issue(
                    path,
                    f"no venv for framework={framework}",
                    "missing-framework-venv",
                )]
            return venv, []
        return family_venv, []


def _issue(path: str, message: str, code: str = "invalid-ci-config") -> Diagnostic:
    return Diagnostic(code=code, location=path, message=message)


def _map(value: Any, path: str, issues: list[Diagnostic]) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(_issue(path, "must be a mapping"))
        return {}
    return value


def _strings(value: Any, path: str, issues: list[Diagnostic]) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)
    issues.append(_issue(path, "must be a string or list of strings"))
    return ()


def _int(value: Any, path: str, issues: list[Diagnostic], minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        issues.append(_issue(path, f"must be an integer >= {minimum}"))
        return minimum
    return value


def _int_list(value: Any, path: str, issues: list[Diagnostic]) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        issues.append(_issue(path, "must be a non-empty list"))
        return ()
    return tuple(sorted({
        _int(item, path, issues, 1) for item in value
    }))


def _timeouts(value: Any, path: str, issues: list[Diagnostic]) -> dict[str, int]:
    result: dict[str, int] = {}
    for device, timeout in _map(value, path, issues).items():
        if device not in DEVICES:
            issues.append(_issue(f"{path}.{device}", "unknown device"))
        else:
            result[device] = _int(timeout, f"{path}.{device}", issues, 1)
    return result


def _timeout_defaults(
    value: Any, path: str, issues: list[Diagnostic]
) -> tuple[dict[str, int], dict[str, int]]:
    defaults = _map(value, path, issues)
    return (
        _timeouts(
            defaults.get("job_timeout_minutes"),
            f"{path}.job_timeout_minutes",
            issues,
        ),
        _timeouts(
            defaults.get("step_timeout_minutes"),
            f"{path}.step_timeout_minutes",
            issues,
        ),
    )


def _venv_map(
    value: Any, path: str, venvs: dict[str, VenvSpec], issues: list[Diagnostic]
) -> dict[str, str]:
    mapping = _map(value, path, issues)
    for name, venv in mapping.items():
        if str(venv) not in venvs:
            issues.append(_issue(f"{path}.{name}", f"unknown venv {venv!r}"))
    return {name: str(venv) for name, venv in mapping.items()}


def _optional_int(
    value: Any, path: str, issues: list[Diagnostic]
) -> int | None:
    return _int(value, path, issues, 1) if value is not None else None


def _integration_overrides(
    value: Any, path: str, issues: list[Diagnostic]
) -> dict[str, IntegrationOverride]:
    overrides: dict[str, IntegrationOverride] = {}
    for case_id, item in _map(value, path, issues).items():
        where = f"{path}.{case_id}"
        override = _map(item, where, issues)
        overrides[case_id] = IntegrationOverride(
            job_timeout_minutes=_optional_int(
                override.get("job_timeout_minutes"),
                f"{where}.job_timeout_minutes",
                issues,
            ),
            step_timeout_minutes=_optional_int(
                override.get("step_timeout_minutes"),
                f"{where}.step_timeout_minutes",
                issues,
            ),
        )
    return overrides


def _load_targets(raw: Any, issues: list[Diagnostic]) -> dict[str, CiTarget]:
    targets: dict[str, CiTarget] = {}
    for device, names in _map(raw, "resources.devices", issues).items():
        path = f"resources.devices.{device}"
        if device not in DEVICES:
            issues.append(_issue(path, "unknown device"))
        if not isinstance(names, list) or not names:
            issues.append(_issue(path, "must be a non-empty target list"))
            continue
        for name in names:
            target = str(name)
            if target in targets:
                issues.append(_issue(path, f"duplicate target {target!r}"))
                continue
            targets[target] = CiTarget(target, device)
    return targets


def bind_image_refs(config: CiConfig, refs: dict[str, str]) -> CiConfig:
    """Return a config with runtime image refs replaced by pinned values."""
    if not refs:
        return config
    issues = [
        _issue(f"resources.images.{name}", "no image definition")
        for name in sorted(set(refs) - set(config.images))
    ]
    images: dict[str, RuntimeImage] = {}
    for name, image in config.images.items():
        ref = refs.get(name) or image.ref
        digest = ref.rpartition("@")[2]
        if image.kind == "container" and not digest.startswith("sha256:"):
            issues.append(_issue(
                f"resources.images.{name}.ref", "must be digest-pinned"
            ))
        images[name] = replace(image, ref=ref)
    if issues:
        raise CiConfigError(issues)
    return replace(config, images=images)


def _load_venvs(
    raw: Any, root: Path, issues: list[Diagnostic]
) -> dict[str, VenvSpec]:
    venvs: dict[str, VenvSpec] = {}
    for name, value in _map(raw, "resources.venvs", issues).items():
        path = f"resources.venvs.{name}"
        row = _map(value, path, issues)
        requirements = _strings(row.get("requirements"), f"{path}.requirements", issues)
        overrides = _strings(row.get("overrides"), f"{path}.overrides", issues)
        for relative in requirements:
            if not (root / relative).is_file():
                issues.append(_issue(path, f"missing file {relative!r}"))
        python = str(row.get("python") or "")
        if not python:
            issues.append(_issue(f"{path}.python", "python version is required"))
        venvs[name] = VenvSpec(
            name=name,
            python=python,
            requirements=requirements,
            overrides=overrides,
            extends=None,
        )
    if "default" not in venvs:
        issues.append(_issue("resources.venvs", "default venv is required"))
    return venvs


def _load_runners(
    raw: Any, targets: dict[str, CiTarget], issues: list[Diagnostic]
) -> tuple[CiRunner, ...]:
    rows = _map(raw, "resources.runners", issues)
    runners: list[CiRunner] = []
    for target, value in rows.items():
        path = f"resources.runners.{target}"
        row = _map(value, path, issues)
        capacities = _int_list(row.get("capacities"), f"{path}.capacities", issues)
        label = str(row.get("label") or "")
        prefix = str(row.get("label_prefix") or "")
        if target not in targets:
            issues.append(_issue(path, "unknown target"))
            continue
        if bool(label) == bool(prefix):
            issues.append(_issue(path, "set exactly one of label or label_prefix"))
            continue
        if label and len(capacities) != 1:
            issues.append(_issue(f"{path}.label", "requires exactly one capacity"))
            continue
        for capacity in capacities:
            runners.append(CiRunner(
                target,
                capacity,
                label or f"{prefix}-{capacity}",
                str(row.get("container_options") or "--user root"),
            ))
    return tuple(sorted(runners, key=lambda item: (item.target, item.rank_capacity)))


def _load_pytest(
    raw: Any, venvs: dict[str, VenvSpec], root: Path, issues: list[Diagnostic]
) -> PytestAreas:
    row = _map(raw, "test_areas.unit_tests", issues)
    job_timeout_minutes, step_timeout_minutes = _timeout_defaults(
        row.get("defaults"),
        "test_areas.unit_tests.defaults",
        issues,
    )
    model_venvs = _venv_map(
        row.get("model_venvs"), "test_areas.unit_tests.model_venvs", venvs, issues
    )
    framework_venvs = _venv_map(
        row.get("framework_venvs"),
        "test_areas.unit_tests.framework_venvs",
        venvs,
        issues,
    )
    combo_venvs = _venv_map(
        row.get("combo_venvs"), "test_areas.unit_tests.combo_venvs", venvs, issues
    )
    overrides: dict[str, str] = {}
    for path, value in _map(
        row.get("overrides"), "test_areas.unit_tests.overrides", issues
    ).items():
        where = f"test_areas.unit_tests.overrides.{path}"
        override = _map(value, where, issues)
        venv = str(override.get("venv") or "")
        if venv not in venvs:
            issues.append(_issue(f"{where}.venv", "unknown venv"))
        if not (root / path).is_file():
            issues.append(_issue(where, "test file is missing"))
        overrides[path] = venv
    return PytestAreas(
        job_timeout_minutes=job_timeout_minutes,
        step_timeout_minutes=step_timeout_minutes,
        model_venvs=model_venvs,
        framework_venvs=framework_venvs,
        combo_venvs=combo_venvs,
        overrides=overrides,
    )


def _load_integration(
    raw: Any, venvs: dict[str, VenvSpec], issues: list[Diagnostic]
) -> dict[str, IntegrationSuite]:
    row = _map(raw, "test_areas.integration_tests", issues)
    suites: dict[str, IntegrationSuite] = {}
    for name, value in _map(
        row.get("suites"), "test_areas.integration_tests.suites", issues
    ).items():
        path = f"test_areas.integration_tests.suites.{name}"
        suite = _map(value, path, issues)
        job_timeout_minutes, step_timeout_minutes = _timeout_defaults(
            suite.get("defaults"),
            f"{path}.defaults",
            issues,
        )
        framework_venvs = _venv_map(
            suite.get("framework_venvs"),
            f"{path}.framework_venvs",
            venvs,
            issues,
        )
        overrides = _integration_overrides(
            suite.get("overrides"), f"{path}.overrides", issues
        )
        suites[name] = IntegrationSuite(
            name=name,
            job_timeout_minutes=job_timeout_minutes,
            step_timeout_minutes=step_timeout_minutes,
            framework_venvs=framework_venvs,
            overrides=overrides,
        )
    return suites


def _test_area_group(
    raw: Any,
    path: str,
    group: str,
    suites: set[str],
    issues: list[Diagnostic],
) -> tuple[str, ...]:
    if raw is None:
        issues.append(_issue(path, "must be explicit: use [all] or []"))
        return ()
    values = _strings(raw, path, issues)
    if not values:
        return ()
    if "all" in values:
        if len(values) != 1:
            issues.append(_issue(path, "[all] cannot be combined with TestAreas"))
        return ("all",)
    for value in values:
        if group == "unit_tests":
            if value not in SUITES:
                issues.append(_issue(path, f"unknown TestArea {value!r}"))
        else:
            suite, separator, case = value.partition("/")
            if not separator or not case or suite not in suites:
                issues.append(_issue(path, f"unknown TestArea {value!r}"))
    return values


def _profile_test_areas(
    raw: Any,
    path: str,
    suites: set[str],
    issues: list[Diagnostic],
) -> tuple[str, ...]:
    row = _map(raw, path, issues)
    groups = {
        "unit_tests": "pytest",
        "integration_tests": "integration",
    }
    for name in set(row) - set(groups):
        issues.append(_issue(f"{path}.{name}", "unknown TestArea group"))
    selected: list[str] = []
    for group, prefix in groups.items():
        values = _test_area_group(
            row.get(group),
            f"{path}.{group}",
            group,
            suites,
            issues,
        )
        selected.extend(f"{prefix}/{value}" for value in values)
    return tuple(selected)


def _load_test_profiles(
    raw: Any,
    images: dict[str, RuntimeImage],
    suites: set[str],
    issues: list[Diagnostic],
) -> dict[str, TestProfile]:
    profiles: dict[str, TestProfile] = {}
    for name, value in _map(raw, "test_profiles", issues).items():
        path = f"test_profiles.{name}"
        if not isinstance(value, list):
            issues.append(_issue(path, "must be a list of selections"))
            value = []
        if not value:
            issues.append(_issue(path, "at least one selection is required"))
        selections: list[ProfileSelection] = []
        seen: set[str] = set()
        for index, item in enumerate(value):
            where = f"{path}[{index}]"
            selection = _map(item, where, issues)
            image = str(selection.get("image") or "")
            if image not in images:
                issues.append(_issue(f"{where}.image", "unknown image"))
            if image in seen:
                issues.append(_issue(where, "duplicate image selection"))
            seen.add(image)
            selections.append(ProfileSelection(
                image,
                _profile_test_areas(
                    selection.get("test_areas"),
                    f"{where}.test_areas",
                    suites,
                    issues,
                ),
            ))
        profiles[name] = TestProfile(name, tuple(selections))
    for name in PROFILE_NAMES:
        if name not in profiles:
            issues.append(_issue(
                "test_profiles", f"required test profile {name!r} is missing"
            ))
    return profiles


def load_config(root: Path) -> CiConfig:
    """Load ci.yml and validate all configuration-local cross references."""
    issues: list[Diagnostic] = []
    path = root / CONFIG_PATH
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CiConfigError([_issue(CONFIG_PATH.as_posix(), str(exc))]) from exc
    document = _map(document, "ci", issues)
    resources = _map(document.get("resources"), "resources", issues)
    targets = _load_targets(resources.get("devices"), issues)
    try:
        images = load_runtime_images(root)
    except ImageError as exc:
        issues.append(_issue("resources.images", str(exc)))
        images = {}
    venvs = _load_venvs(resources.get("venvs"), root, issues)
    runners = _load_runners(resources.get("runners"), targets, issues)
    test_areas = _map(document.get("test_areas"), "test_areas", issues)
    pytest_areas = _load_pytest(test_areas.get("unit_tests"), venvs, root, issues)
    integration_suites = _load_integration(
        test_areas.get("integration_tests"), venvs, issues
    )
    test_profiles = _load_test_profiles(
        document.get("test_profiles"), images, set(integration_suites), issues
    )
    if issues:
        raise CiConfigError(issues)
    return CiConfig(
        root=root,
        targets=targets,
        images=images,
        venvs=venvs,
        runners=runners,
        pytest_areas=pytest_areas,
        integration_suites=integration_suites,
        test_profiles=test_profiles,
    )
