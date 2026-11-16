#!/usr/bin/env python3
"""TorchTurbo-specific contracts that generic linters cannot express."""

from __future__ import annotations

import ast
import configparser
import re
import sys
from pathlib import Path

from lint_common_check import CI, ROOT, WORKFLOWS, Check, Problem, relative, text

BUILTIN_MARKERS = frozenset({
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "usefixtures",
    "xfail",
})

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # reported by the checks that need it
    yaml = None


def _yaml_document(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed "
                           "(pip install -r requirements-lint.txt)")

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate key {key!r}")
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    doc = yaml.load(text(path), Loader=UniqueKeyLoader)
    if not isinstance(doc, dict):
        raise ValueError("top level is not a mapping")
    return doc


def _job_needs(job: dict) -> list[str]:
    needs = (job or {}).get("needs") or []
    if isinstance(needs, str):
        return [needs]
    return list(needs)


def _job_run_text(job: dict) -> str:
    chunks = []
    for step in (job or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("run"):
            chunks.append(str(step["run"]))
    return "\n".join(chunks)


def _job_closure(jobs: dict[str, dict], name: str) -> set[str]:
    """Return every job reachable from `name` through `needs` edges."""
    seen: set[str] = set()
    pending = list(_job_needs(jobs.get(name, {})))
    while pending:
        current = pending.pop()
        if current in seen or current not in jobs:
            continue
        seen.add(current)
        pending.extend(_job_needs(jobs[current]))
    return seen


def check_contract() -> list[Problem]:
    """Cross-job wiring that no YAML parser or actionlint can see."""
    if yaml is None:
        return [Problem(".github/workflows", "PyYAML is required for contract checks")]
    problems = []
    docs = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        rel = relative(path)
        try:
            docs[path] = _yaml_document(path)
        except Exception as exc:
            problems.append(Problem(rel, f"cannot inspect contract: {exc}"))
    if problems:
        return problems

    for path, doc in docs.items():
        rel = relative(path)
        jobs = {str(name): job for name, job in (doc.get("jobs") or {}).items()}
        plan_jobs = [name for name, job in jobs.items()
                     if "planner.py" in _job_run_text(job)]
        if plan_jobs:
            problems.extend(_check_planner_v2(rel, jobs, yaml.safe_dump(doc, allow_unicode=True)))
            continue
    return problems


def _check_planner_v2(rel: str, jobs: dict, dumped: str) -> list[Problem]:
    """Validate the Planner -> Executor -> Gate topology."""
    problems: list[Problem] = []
    planners = [name for name, job in jobs.items()
                if "planner.py" in _job_run_text(job)]
    if len(planners) != 1:
        return [Problem(rel, f"expected one Planner job; found {sorted(planners)}")]
    planner_name = planners[0]
    planner = jobs[planner_name]
    planner_body = _job_run_text(planner)
    if "lint_check.py" not in planner_body:
        problems.append(Problem(rel, "Planner job must also run lint_check.py"))
    declared = set((planner.get("outputs") or {}).keys())
    output_pattern = rf"needs\.{re.escape(planner_name)}\.outputs\.([A-Za-z0-9_-]+)"
    used = set(re.findall(output_pattern, dumped))
    for name in sorted(used - declared):
        problems.append(Problem(
            rel,
            f"reads needs.{planner_name}.outputs.{name} but the Planner does not declare it",
        ))
    for name in sorted(declared - used):
        problems.append(Problem(rel, f"Planner output {name!r} has no consumer"))

    needed = {dependency for job in jobs.values() for dependency in _job_needs(job)}
    terminals = sorted(set(jobs) - needed)
    lanes = sorted({
        name for name, job in jobs.items()
        if planner_name in _job_needs(job)
    } - set(terminals))
    if not lanes:
        problems.append(Problem(rel, "Planner must feed at least one Executor lane"))
    for name in lanes:
        body = _job_run_text(jobs[name])
        if "executor.py" not in body:
            problems.append(Problem(rel, f"lane {name!r} does not invoke executor.py"))
    if len(terminals) != 1:
        problems.append(Problem(
            rel, f"Planner workflow must have exactly one terminal job; found {terminals}"
        ))
        return problems
    terminal = terminals[0]
    terminal_body = _job_run_text(jobs[terminal])
    if "gate.py" not in terminal_body:
        problems.append(Problem(rel, f"terminal job {terminal!r} does not invoke gate.py"))
    if "download-artifact" not in yaml.safe_dump(jobs[terminal], allow_unicode=True):
        problems.append(Problem(rel, f"terminal job {terminal!r} does not download results"))
    reachable = _job_closure(jobs, terminal)
    missing = sorted({planner_name, *lanes} - reachable)
    if missing:
        problems.append(Problem(rel, f"terminal job {terminal!r} does not reach {missing}"))
    return problems


def check_model_map() -> list[Problem]:
    """Validate model, env, lane, and E2E case facts together."""
    sys.path.insert(0, str(CI))
    from facts import load_catalog

    catalog = load_catalog(ROOT)
    return [Problem(message.split(":", 1)[0], message.split(":", 1)[-1].strip())
            for message in catalog.errors]


def _test_marker_names() -> set[str]:
    """Collect custom pytest markers used under tests/."""
    names: set[str] = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        try:
            tree = ast.parse(text(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            base = node.value
            if (isinstance(base, ast.Attribute) and base.attr == "mark"
                    and isinstance(base.value, ast.Name) and base.value.id == "pytest"):
                names.add(node.attr)
    return names - BUILTIN_MARKERS


def check_markers() -> list[Problem]:
    """Every marker used by tests is registered in pytest.ini."""
    ini = ROOT / "pytest.ini"
    pyproject = ROOT / "pyproject.toml"
    problems = []

    if "[tool.pytest.ini_options]" in text(pyproject):
        problems.append(Problem("pyproject.toml", "pytest config belongs in pytest.ini"))
    if not ini.exists():
        return problems + [Problem("pytest.ini", "missing test configuration")]

    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ini, encoding="utf-8")
    configured = set()
    for line in (parser.get("pytest", "markers", fallback="") or "").splitlines():
        if ":" in line:
            configured.add(line.split(":", 1)[0].strip())

    sys.path.insert(0, str(CI))
    from facts import REQUIRED_MARKERS

    markers = _test_marker_names()
    for name in sorted(markers - configured):
        problems.append(Problem("pytest.ini", f"marker {name!r} is used but not registered"))
    for name in sorted(configured - set(REQUIRED_MARKERS)):
        problems.append(Problem("pytest.ini", f"obsolete marker {name!r}"))
    return problems


CHECKS = (
    Check("contract", "planner/lint/lane topology and output contracts", check_contract),
    Check("model_map", "model, env, lane, and E2E case facts", check_model_map),
    Check("markers", "registered pytest marker contract", check_markers),
)
