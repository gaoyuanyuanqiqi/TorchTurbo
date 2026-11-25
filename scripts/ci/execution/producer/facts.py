#!/usr/bin/env python3
"""Derive test facts from pytest markers and integration suite manifests."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import CiConfig
from ...contracts.base import DEVICES, Diagnostic, DiagnosticError

PYTEST_ROOT = Path("tests/unit_tests")
MANIFEST_SCHEMA_VERSION = 1
DEVICE_MARKERS = frozenset({"cpu", "npu", "gpu"})
MULTI_MARKERS = {"multi_npu": "npu", "multi_gpu": "gpu"}
REQUIRED_MARKERS = ("cpu", "npu", "multi_npu", "gpu", "multi_gpu", "integration")
RANK_PARAMETERS = frozenset({"world_size", "nprocs", "num_processes", "rank_size"})
ADAPTER_FRAMEWORKS = {
    "llamafactory_v0": "lf",
    "msswift": "swift",
}


class FactsError(DiagnosticError):
    """Raised when repository test facts are missing or inconsistent."""


@dataclass(frozen=True)
class PytestFact:
    path: str
    suite: str
    frameworks: tuple[str, ...]
    model_families: tuple[str, ...]
    ranks_by_device: Mapping[str, int]

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(sorted(self.ranks_by_device))


@dataclass(frozen=True)
class IntegrationFact:
    suite: str
    case_id: str
    framework: str
    platform: str
    targets: tuple[str, ...]
    ranks: int


@dataclass(frozen=True)
class TestFacts:
    __test__ = False

    pytest: tuple[PytestFact, ...]
    integration: tuple[IntegrationFact, ...]


def _issue(path: str, message: str, code: str = "invalid-test-facts") -> Diagnostic:
    return Diagnostic(code=code, location=path, message=message)


def test_parts(path: str) -> tuple[str, ...]:
    parts = Path(path).parts
    if parts[:2] == ("tests", "unit_tests"):
        return parts[2:]
    return parts[1:] if parts[:1] == ("tests",) else parts


def _classify_path(
    path: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    parts = test_parts(path)

    if parts[:1] == ("adapters",):
        framework = ADAPTER_FRAMEWORKS.get(parts[1]) if len(parts) > 1 else None
        return "adapter", (framework,) if framework else (), ()

    if (
        parts[:1] == ("models",)
        and len(parts) >= 3
        and parts[1] != "modules"
    ):
        return "model", (), (parts[1],)

    return "core", (), ()


def _marker_names(node: ast.AST) -> set[str]:
    values = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else [node]
    names: set[str] = set()
    for value in values:
        if isinstance(value, ast.Call):
            value = value.func
        if isinstance(value, ast.Attribute):
            names.add(value.attr)
    return names


def _module_markers(module: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in targets
        ):
            names |= _marker_names(value)
    return frozenset(names)


def _test_functions(
    module: ast.Module,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    ]


def _parametrized_rank(call: ast.Call) -> int:
    try:
        parameters = ast.literal_eval(call.args[0])
        values = ast.literal_eval(call.args[1])
    except (ValueError, SyntaxError):
        return 1
    if isinstance(parameters, str):
        parameters = [part.strip() for part in parameters.split(",")]
    if not isinstance(parameters, (list, tuple)) or not isinstance(values, (list, tuple)):
        return 1
    indexes = [
        index for index, name in enumerate(parameters)
        if isinstance(name, str) and name in RANK_PARAMETERS
    ]
    required = 1
    for row in values:
        row = row if isinstance(row, (list, tuple)) else (row,)
        for index in indexes:
            if index < len(row) and isinstance(row[index], int):
                required = max(required, row[index])
    return required


def _max_rank(tree: ast.AST) -> int:
    required = 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else ""
        )
        if name == "parametrize" and len(node.args) >= 2:
            required = max(required, _parametrized_rank(node))
            continue
        if name not in {"spawn", "init_process_group", "require_npus"}:
            continue
        if name == "require_npus":
            value = node.args[0] if node.args else None
            size = value.value if isinstance(value, ast.Constant) else None
        else:
            keyword = "world_size" if name == "init_process_group" else "nprocs"
            value = next(
                (
                    keyword_node.value.value
                    for keyword_node in node.keywords
                    if keyword_node.arg == keyword
                    and isinstance(keyword_node.value, ast.Constant)
                    and isinstance(keyword_node.value.value, int)
                ),
                None,
            )
            size = value
        required = max(required, size if isinstance(size, int) else 2)
    return required


def _test_rank_requirements(
    test: ast.FunctionDef | ast.AsyncFunctionDef,
    module_markers: frozenset[str],
) -> Mapping[str, int]:
    markers = set(module_markers)
    for decorator in test.decorator_list:
        markers |= _marker_names(decorator)

    multi = markers & MULTI_MARKERS.keys()
    devices = set(markers & DEVICE_MARKERS)
    devices.update(MULTI_MARKERS[name] for name in multi)
    devices = devices or {"cpu"}
    rank = max(_max_rank(test), 2 if multi else 1)
    return {device: rank for device in devices}


def _device_ranks(
    tests: list[ast.FunctionDef | ast.AsyncFunctionDef],
    module_markers: frozenset[str],
    module: ast.Module,
) -> Mapping[str, int]:
    """Aggregate per-test rank requirements by device."""
    ranks: dict[str, int] = {}

    for test in tests:
        for device, rank in _test_rank_requirements(test, module_markers).items():
            ranks[device] = max(ranks.get(device, 1), rank)

    if len(ranks) == 1:
        device = next(iter(ranks))
        ranks[device] = max(ranks[device], _max_rank(module))

    return {device: ranks[device] for device in sorted(ranks)}


def _pytest_facts(root: Path) -> tuple[list[PytestFact], list[Diagnostic]]:
    facts: list[PytestFact] = []
    issues: list[Diagnostic] = []
    for path in sorted((root / PYTEST_ROOT).rglob("test_*.py")):
        relative_path = path.relative_to(root).as_posix()
        try:
            ast_tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError) as exc:
            issues.append(_issue(relative_path, f"cannot parse test file: {exc}"))
            continue

        tests = _test_functions(ast_tree)
        if not tests:
            continue

        suite, frameworks, model_families = _classify_path(relative_path)
        facts.append(PytestFact(
            path=relative_path,
            suite=suite,
            frameworks=frameworks,
            model_families=model_families,
            ranks_by_device=_device_ranks(tests, _module_markers(ast_tree), ast_tree),
        ))
    return facts, issues


def _manifest(
    root: Path, suite_name: str, runner: str
) -> tuple[dict[str, Any] | None, Diagnostic | None]:
    result = subprocess.run(
        [sys.executable, str(root / runner), "--suite", suite_name, "--list-cases", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        message = f"manifest failed with rc={result.returncode}: {detail}"
    else:
        try:
            payload = json.loads(result.stdout)
        except ValueError as exc:
            message = f"manifest is not JSON: {exc}"
        else:
            if isinstance(payload, dict):
                return payload, None
            message = "manifest must be a mapping"
    return None, _issue(
        runner, f"integration suite {suite_name!r} {message}", "integration-manifest"
    )


def _check_repo_file(
    root: Path, value: str, where: str, issues: list[Diagnostic]
) -> None:
    if not value:
        issues.append(_issue(where, "path is required"))
        return
    path = Path(value)
    if path.is_absolute():
        issues.append(_issue(where, "path must be repository-relative"))
        return
    absolute = (root / path).resolve()
    try:
        absolute.relative_to(root.resolve())
    except ValueError:
        issues.append(_issue(where, "path escapes the repository"))
        return
    if not absolute.is_file():
        issues.append(_issue(where, f"file is missing: {value}"))


def _targets(
    item: dict[str, Any],
    platform: str,
    config: CiConfig,
    where: str,
    issues: list[Diagnostic],
) -> tuple[str, ...]:
    single = item.get("target")
    multiple = item.get("targets")
    if single is not None and multiple is not None:
        issues.append(_issue(where, "declare target or targets, not both"))
    single_valid = isinstance(single, str) and bool(single)
    multiple_valid = isinstance(multiple, list) and bool(multiple) and all(
        isinstance(value, str) and value for value in multiple
    )
    if single is not None and not single_valid:
        issues.append(_issue(where, "target must be a non-empty string"))
    if multiple is not None and not multiple_valid:
        issues.append(_issue(where, "targets must be non-empty strings"))
    values = (single,) if single_valid else tuple(multiple) if multiple_valid else ()
    for name in values:
        target = config.targets.get(name)
        if target is None:
            issues.append(_issue(where, f"unknown target {name!r}"))
        elif target.device != platform:
            issues.append(_issue(where, f"target {name!r} is for {target.device}, not {platform}"))
    return values


def _integration_facts(
    root: Path, config: CiConfig
) -> tuple[list[IntegrationFact], list[Diagnostic]]:
    issues: list[Diagnostic] = []
    facts: list[IntegrationFact] = []
    seen: set[tuple[str, str]] = set()
    for suite_name, suite in sorted(config.integration_suites.items()):
        payload, error = _manifest(root, suite_name, suite.runner)
        if error is not None:
            issues.append(error)
            continue
        assert payload is not None
        expected = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "suite": suite_name,
        }
        for key, value in expected.items():
            actual = payload.get(key)
            if actual != value:
                issues.append(_issue(
                    suite.runner,
                    f"manifest {key} is {actual!r}, expected {value!r}",
                    "integration-manifest",
                ))
        cases = payload.get("cases")
        if not isinstance(cases, list):
            issues.append(_issue(
                suite.runner,
                f"suite {suite_name!r} manifest cases must be a list",
                "integration-manifest",
            ))
            continue
        for index, item in enumerate(cases):
            where = f"{suite.runner}.cases[{index}]"
            if not isinstance(item, dict):
                issues.append(_issue(where, "case must be a mapping"))
                continue
            case_id = item.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                issues.append(_issue(f"{where}.case_id", "case_id is required"))
                continue
            key = (suite_name, case_id)
            if key in seen:
                issues.append(_issue(f"{where}.case_id", f"duplicate case {case_id!r}"))
            seen.add(key)
            framework = item.get("framework")
            if not isinstance(framework, str) or framework not in suite.framework_venvs:
                issues.append(_issue(
                    f"{where}.framework", f"framework {framework!r} has no venv mapping"
                ))
                framework = ""
            platform = item.get("platform")
            if not isinstance(platform, str) or platform not in DEVICES:
                issues.append(_issue(f"{where}.platform", "unknown platform"))
                platform = ""
            elif not config.device_is_configured(platform):
                issues.append(_issue(
                    f"{where}.platform", f"no CI target for {platform!r}"
                ))
            targets = (
                _targets(item, platform, config, f"{where}.targets", issues)
                if platform else ()
            )
            ranks = item.get("ranks")
            if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
                issues.append(_issue(f"{where}.ranks", "ranks must be an integer >= 1"))
                ranks = 1
            compare = item.get("compare")
            if compare is not True:
                issues.append(_issue(
                    f"{where}.compare", "CI integration cases must compare"
                ))
            if item.get("update_golden") is not False:
                issues.append(_issue(
                    f"{where}.update_golden", "CI must not update golden values"
                ))
            _check_repo_file(
                root, str(item.get("config") or ""), f"{where}.config", issues
            )
            if compare is True:
                _check_repo_file(
                    root, str(item.get("golden") or ""), f"{where}.golden", issues
                )
            facts.append(IntegrationFact(
                suite=suite_name,
                case_id=case_id,
                framework=framework,
                platform=platform,
                targets=targets,
                ranks=ranks,
            ))
    return facts, issues


def load_facts(root: Path, config: CiConfig) -> TestFacts:
    """Load all configured pytest and integration facts."""
    pytest, pytest_issues = _pytest_facts(root)
    integration, integration_issues = _integration_facts(root, config)
    issues = [*pytest_issues, *integration_issues]
    if not pytest:
        issues.append(_issue(PYTEST_ROOT.as_posix(), "no pytest facts found"))
    if issues:
        raise FactsError(issues)
    return TestFacts(
        pytest=tuple(pytest),
        integration=tuple(
            sorted(integration, key=lambda item: (item.suite, item.case_id))
        ),
    )
