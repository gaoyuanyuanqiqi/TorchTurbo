#!/usr/bin/env python3
"""TorchTurbo-specific contracts that generic linters cannot express."""

from __future__ import annotations

import ast
import configparser
import json
import re
import sys
from pathlib import Path

from lint_common_check import CI, ROOT, WORKFLOWS, Check, Problem, relative, run, text

MODEL_MAP_SECTIONS = ("download", "install", "train", "params")
MODEL_FRAMEWORKS = ("lf", "swift")
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
        planner_names = [name for name, job in jobs.items()
                         if "select_tests.py" in _job_run_text(job)]
        if not planner_names:
            continue
        if len(planner_names) > 1:
            problems.append(Problem(rel, "expected one planner job; found "
                                         f"{sorted(planner_names)}"))
            continue
        planner_name = planner_names[0]
        planner = jobs[planner_name]
        planner_body = _job_run_text(planner)
        dumped = yaml.safe_dump(doc, allow_unicode=True)

        planner_yaml = yaml.safe_dump(planner, allow_unicode=True)
        if "--changed-from git" in planner_body and "fetch-depth: 0" not in planner_yaml:
            problems.append(Problem(
                rel, f"{planner_name} uses git history without fetch-depth: 0"
            ))

        declared = set((planner.get("outputs") or {}).keys())
        output_pattern = rf"needs\.{re.escape(planner_name)}\.outputs\.([A-Za-z0-9_-]+)"
        used = set(re.findall(output_pattern, dumped))
        for name in sorted(used - declared):
            problems.append(Problem(
                rel,
                f"reads needs.{planner_name}.outputs.{name} but the planner does not declare it",
            ))
        for name in sorted(declared - used):
            problems.append(Problem(rel, f"planner output {name!r} has no consumer"))

        lint_jobs = {name for name, job in jobs.items()
                     if "lint_check.py" in _job_run_text(job)}
        if planner_name not in lint_jobs:
            problems.append(Problem(rel, "planner job must also run lint_check.py"))
        elif planner_body.index("lint_check.py") > planner_body.index("select_tests.py"):
            problems.append(Problem(rel, "select_tests.py runs before lint_check.py"))
        if "select_tests.py --verify" in planner_body:
            problems.append(Problem(rel, "lint_check.py already runs the selector self-test"))
        extra_lint_jobs = sorted(lint_jobs - {planner_name})
        if extra_lint_jobs:
            problems.append(Problem(rel, "lint must run only in the planner job; found in "
                                         f"{extra_lint_jobs}"))

        needed = {dependency for job in jobs.values() for dependency in _job_needs(job)}
        terminals = sorted(set(jobs) - needed)
        lanes = sorted({name for name, job in jobs.items()
                        if planner_name in _job_needs(job)} - set(terminals))
        if not lanes:
            problems.append(Problem(rel, "planner must feed at least one lane"))
        for name in lanes:
            lane_dependencies = set(_job_needs(jobs[name]))
            chained = sorted(lane_dependencies & set(lanes))
            if chained:
                problems.append(Problem(rel, f"lane {name!r} depends on lanes {chained}"))

        if len(terminals) != 1:
            problems.append(Problem(
                rel, "planner workflow must have exactly one terminal job; "
                     f"found {terminals}"
            ))
            continue
        terminal = terminals[0]
        reachable = _job_closure(jobs, terminal)
        missing = sorted({planner_name, *lanes} - reachable)
        if missing:
            problems.append(Problem(rel, f"terminal job {terminal!r} does not reach {missing}"))
    return problems


def check_selector() -> list[Problem]:
    """Run the selector's own rule assertions."""
    selector = CI / "lint_select" / "select_tests.py"
    proc = run([sys.executable, str(selector), "--verify"], timeout=180)
    if proc.returncode == 0:
        return []
    output = (proc.stdout + proc.stderr).strip().splitlines()
    detail = output[-1] if output else f"exit {proc.returncode}"
    return [Problem(relative(selector), f"selector self-test failed: {detail}")]


def _registered_models() -> set[str]:
    """Collect model names from real @register_model decorators."""
    registered: set[str] = set()
    for path in sorted((ROOT / "src" / "torchturbo" / "models").rglob("*.py")):
        try:
            tree = ast.parse(text(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                name = decorator.func
                if not ((isinstance(name, ast.Name) and name.id == "register_model")
                        or (isinstance(name, ast.Attribute) and name.attr == "register_model")):
                    continue
                values = decorator.args[:1] + [
                    keyword.value for keyword in decorator.keywords if keyword.arg == "name"
                ]
                for value in values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        registered.add(value.value)
    return registered


def check_model_map() -> list[Problem]:
    """Every registered model must be represented in model_map.json."""
    registered = _registered_models()

    path = WORKFLOWS / "model_map.json"
    if not path.exists():
        return [Problem(relative(path), "model map is missing")]
    try:
        doc = json.loads(text(path))
    except ValueError as exc:
        return [Problem(relative(path), f"invalid JSON: {exc}")]
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        return [Problem(relative(path), "missing top-level models mapping")]

    problems = []
    for name in sorted(registered - set(models)):
        problems.append(Problem(relative(path), f"registered model {name!r} is not in the map"))
    for name in sorted(set(models) - registered):
        problems.append(Problem(relative(path), f"model {name!r} is not registered"))
    for name, row in sorted(models.items()):
        if not isinstance(row, dict):
            problems.append(Problem(relative(path), f"{name}: row is not a mapping"))
            continue
        if not isinstance(row.get("run_dir"), str):
            problems.append(Problem(relative(path), f"{name}: missing 'run_dir'"))
        for section in MODEL_MAP_SECTIONS:
            if section in row and not isinstance(row[section], dict):
                problems.append(Problem(relative(path),
                                        f"{name}: {section!r} must be a mapping"))
        # Script names are declared per framework, so a typo has to be caught
        # here rather than on an 8-card pod.
        for section in ("install", "train"):
            for framework in MODEL_FRAMEWORKS:
                if framework not in (row.get(section) or {}):
                    continue
                run_dir = Path("scripts") / "runs" / str(row.get("run_dir", "-"))
                for script in str(row[section][framework]).split(","):
                    if script != "-" and not (ROOT / run_dir / script).is_file():
                        problems.append(Problem(
                            relative(path),
                            f"{name}: {section}.{framework} names a missing script: "
                            f"{run_dir / script}"))
        # cards/sp/ep are the torchrun size, declared rather than inferred from
        # the runner label, so a runnable model without a valid plan cannot run.
        if str((row.get("download") or {}).get("e2e", "-")) != "-":
            params = row.get("params") or {}
            cards, sp, ep = (params.get(k) for k in ("cards", "sp", "ep"))
            if not all(isinstance(v, int) and v >= 1 for v in (cards, sp, ep)):
                problems.append(Problem(
                    relative(path), f"{name}: e2e model needs integer params.cards/sp/ep, "
                    f"got cards={cards!r} sp={sp!r} ep={ep!r}"))
            elif cards % sp or (cards // sp) % ep:
                problems.append(Problem(
                    relative(path), f"{name}: cards={cards} cannot host sp={sp} ep={ep}"))
    return problems


def _module_marker_names(path: Path) -> set[str]:
    """Collect string keys/values from module-level names ending in MARKERS."""
    tree = ast.parse(text(path))
    names: set[str] = set()
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        if value is None or not any(isinstance(t, ast.Name) and t.id.endswith("MARKERS")
                                    for t in targets):
            continue
        while isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
                and value.func.id in ("frozenset", "set") and len(value.args) == 1:
            value = value.args[0]
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(literal, dict):
            names.update(key for key in literal if isinstance(key, str))
        elif isinstance(literal, (set, frozenset, list, tuple)):
            names.update(item for item in literal if isinstance(item, str))
    return names


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
    """Every marker used by the selector or tests is registered in pytest.ini."""
    ini = ROOT / "pytest.ini"
    pyproject = ROOT / "pyproject.toml"
    selector = CI / "lint_select" / "select_tests.py"
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

    markers = _module_marker_names(selector) | _test_marker_names()
    for name in sorted(markers - configured):
        problems.append(Problem("pytest.ini", f"marker {name!r} is used but not registered"))
    return problems


CHECKS = (
    Check("contract", "planner/lint/lane topology and output contracts", check_contract),
    Check("selector", "selector self-test and precision guards", check_selector),
    Check("model_map", "registered models match model_map.json", check_model_map),
    Check("markers", "registered pytest marker contract", check_markers),
)
