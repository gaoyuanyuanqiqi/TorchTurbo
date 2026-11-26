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

from ...contracts.base import Diagnostic, DiagnosticError

PYTEST_ROOT = Path("tests/unit_tests")
INTEGRATION_ROOT = Path("tests/integration_tests")
INTEGRATION_RUNNER_SCRIPT = (INTEGRATION_ROOT / "scripts/run_tests.py").as_posix()
INTEGRATION_SUITE_DIRECTORY = INTEGRATION_ROOT / "test_suite"
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
    device: str
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


def _load_suite_cases(
    root: Path, suite_name: str
) -> tuple[list[Any] | None, Diagnostic | None]:
    """Execute the integration runner and return one suite's manifest cases."""
    runner_path = INTEGRATION_RUNNER_SCRIPT
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(root / runner_path),
                "--suite", suite_name,
                "--list-cases",
                "--json",
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return None, _issue(
            runner_path,
            f"integration suite {suite_name!r} manifest cannot start: {exc}",
            "integration-manifest",
        )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        return None, _issue(
            runner_path,
            f"integration suite {suite_name!r} manifest failed "
            f"with rc={completed.returncode}: {detail}",
            "integration-manifest",
        )
    try:
        manifest = json.loads(completed.stdout)
    except ValueError as exc:
        return None, _issue(
            runner_path,
            f"integration suite {suite_name!r} manifest is not JSON: {exc}",
            "integration-manifest",
        )
    if not isinstance(manifest, dict):
        return None, _issue(
            runner_path,
            f"integration suite {suite_name!r} manifest must be a mapping",
            "integration-manifest",
        )
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return None, _issue(
            runner_path,
            f"integration suite {suite_name!r} manifest cases must be a list",
            "integration-manifest",
        )
    return cases, None


def _integration_case_fact(
    root: Path,
    suite_name: str,
    case_payload: Any,
    case_index: int,
) -> tuple[IntegrationFact | None, list[Diagnostic]]:
    """Validate one manifest case and convert it into an IntegrationFact."""
    case_location = f"integration/{suite_name}/cases[{case_index}]"
    if not isinstance(case_payload, dict):
        return None, [_issue(case_location, "case must be a mapping")]

    case_id = case_payload.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        return None, [_issue(
            f"{case_location}.case_id", "case_id is required"
        )]

    issues: list[Diagnostic] = []
    ranks = case_payload.get("ranks")
    if not isinstance(ranks, int) or isinstance(ranks, bool) or ranks < 1:
        issues.append(_issue(
            f"{case_location}.ranks", "ranks must be an integer >= 1"
        ))
        ranks = 1

    compares_results = case_payload.get("compare") is True
    if not compares_results:
        issues.append(_issue(
            f"{case_location}.compare", "CI integration cases must compare"
        ))
    if case_payload.get("update_golden") is not False:
        issues.append(_issue(
            f"{case_location}.update_golden",
            "CI must not update golden values",
        ))

    if compares_results:
        golden_path = case_payload.get("golden")
        if not isinstance(golden_path, str) or not (root / golden_path).is_file():
            issues.append(_issue(
                f"{case_location}.golden", "golden file is missing"
            ))

    return IntegrationFact(
        suite=suite_name,
        case_id=case_id,
        framework=str(case_payload.get("framework") or ""),
        device=str(case_payload.get("device") or ""),
        ranks=ranks,
    ), issues


def _integration_facts(
    root: Path,
) -> tuple[list[IntegrationFact], list[Diagnostic]]:
    """Discover integration suites from test_suite and load their manifests."""
    suite_files = sorted(
        (root / INTEGRATION_SUITE_DIRECTORY).glob("*_suite.py")
    )
    facts: list[IntegrationFact] = []
    issues: list[Diagnostic] = []
    for suite_file in suite_files:
        suite_name = suite_file.stem
        case_payloads, manifest_issue = _load_suite_cases(root, suite_name)
        if manifest_issue is not None:
            issues.append(manifest_issue)
            continue
        assert case_payloads is not None

        for case_index, case_payload in enumerate(case_payloads):
            fact, case_issues = _integration_case_fact(
                root,
                suite_name,
                case_payload,
                case_index,
            )
            issues.extend(case_issues)
            if fact is not None:
                facts.append(fact)
    return facts, issues


def load_facts(root: Path) -> TestFacts:
    """Load pytest and integration facts from repository content."""
    pytest, pytest_issues = _pytest_facts(root)
    integration, integration_issues = _integration_facts(root)
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
