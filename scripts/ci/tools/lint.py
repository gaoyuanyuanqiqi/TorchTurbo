#!/usr/bin/env python3
"""Run generic linters and repository-specific CI contracts."""

from __future__ import annotations

import argparse
import ast
import configparser
import os
import platform
import re
import sys
import time
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
WORKFLOWS = ROOT / ".github" / "workflows"
CI = ROOT / "scripts" / "ci"
TOOL_DIR = Path(os.environ.get("TORCHTURBO_LINT_TOOLS", "/tmp/torchturbo-lint-tools"))

ACTIONLINT_VERSION = "1.7.12"
HADOLINT_VERSION = "2.15.1"
SHELLCHECK_VERSION = "0.11.0"
# ruff and PyYAML are not self-bootstrapped; their pinned versions live in
# scripts/ci/config/requirements-lint.txt alone, so this file cannot drift
# from the workflow.
REQUIREMENTS = "scripts/ci/config/requirements-lint.txt"

# These are build-policy choices, not accidental lint debt.
HADOLINT_IGNORES = ("DL3041", "DL3042", "DL3003", "DL3059")
# actionlint 1.7.12 predates this checkout input. The action itself documents
# and enforces the input, so ignore only this known metadata gap.
ACTIONLINT_IGNORES = (
    r'input "allow-unsafe-pr-checkout" is not defined in action '
    r'"actions/checkout@v4"',
)

try:
    import yaml  # type: ignore[import-not-found]
except ImportError:  # reported by the checks that need it
    yaml = None


@dataclass(frozen=True)
class Problem:
    path: str
    message: str
    line: int = 0

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    function: Callable[[], list[Problem]]


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _arch() -> str:
    machine = platform.machine().lower()
    arch = ARCHES.get(machine)
    if arch is None:
        raise RuntimeError(f"unsupported architecture: {machine}")
    return arch



PROBE = {"actionlint": "-version", "hadolint": "--version", "shellcheck": "--version"}
ARCHES = {"aarch64": "arm64", "arm64": "arm64",
          "x86_64": "amd64", "amd64": "amd64"}
SOURCE_ENTRYPOINTS = {"push", "pull_request", "pull_request_target", "merge_group", "schedule"}


def _tool_path(name: str, version: str | None = None) -> Path | None:
    candidates = [os.environ.get(f"{name.upper()}_BIN"), TOOL_DIR / name,
                  Path("/tmp") / name, shutil.which(name)]
    for index, candidate in enumerate(candidates):
        path = Path(candidate) if candidate else None
        if path is None or not path.is_file() or not os.access(path, os.X_OK):
            continue
        # An explicit *_BIN override is honoured as-is; everything else must
        # report the pinned version or the download below supplies it instead.
        if index == 0 or version is None:
            return path
        try:
            probe = run([str(path), PROBE[name]], timeout=30)
            reported = probe.stdout + probe.stderr
        except (OSError, subprocess.SubprocessError):
            reported = ""
        if version in reported:
            return path
    return None


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "torchturbo-ci-lint"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, \
                open(temporary, "wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"download produced no file: {url}")
    destination.chmod(0o755)


def _extract_binary(archive: Path, target: Path, member_name: str) -> Path:
    with tarfile.open(archive, "r:*") as bundle:
        member = next((m for m in bundle.getmembers()
                       if Path(m.name).name == member_name), None)
        if member is None:
            raise RuntimeError(f"{archive.name} has no {member_name} binary")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"cannot read {member_name} from {archive.name}")
        temporary = target.with_suffix(".part")
        with open(temporary, "wb") as output:
            shutil.copyfileobj(source, output)
    temporary.replace(target)
    target.chmod(0o755)
    return target


def _ensure_tool(name: str, version: str, url: str, *, arch: str = "",
                 suffix: str = "", member: str = "") -> Path:
    found = _tool_path(name, version)
    if found:
        return found
    target = TOOL_DIR / name
    if not suffix:
        _download(url, target)
        return target
    archive = TOOL_DIR / f"{name}-{version}-{arch}{suffix}"
    _download(url, archive)
    return _extract_binary(archive, target, member or name)


def _ensure_actionlint() -> Path:
    arch = _arch()
    version = ACTIONLINT_VERSION
    url = (f"https://github.com/rhysd/actionlint/releases/download/v{version}/"
           f"actionlint_{version}_linux_{arch}.tar.gz")
    return _ensure_tool("actionlint", version, url, arch=arch, suffix=".tar.gz")


def _ensure_hadolint() -> Path:
    arch = "arm64" if _arch() == "arm64" else "x86_64"
    url = (f"https://github.com/hadolint/hadolint/releases/download/"
           f"v{HADOLINT_VERSION}/hadolint-linux-{arch}")
    return _ensure_tool("hadolint", HADOLINT_VERSION, url)


def _ensure_shellcheck() -> Path:
    arch = "aarch64" if _arch() == "arm64" else "x86_64"
    version = SHELLCHECK_VERSION
    url = (f"https://github.com/koalaman/shellcheck/releases/download/v{version}/"
           f"shellcheck-v{version}.linux.{arch}.tar.xz")
    return _ensure_tool("shellcheck", version, url, arch=arch, suffix=".tar.xz")


def _ensure_ruff() -> Path:
    found = _tool_path("ruff")
    if found:
        return found
    raise RuntimeError(f"ruff is not installed (pip install -r {REQUIREMENTS})")


def _diagnostics(cmd: list[str], fallback: str, timeout: int = 120) -> list[Problem]:
    """Run a tool and parse `path:line[:column]: message` diagnostics."""
    proc = run(cmd, timeout=timeout)
    if proc.returncode == 0:
        return []
    problems = []
    for line in proc.stdout.splitlines():
        match = re.match(r"^(.*?):(\d+):(?:\d+: )?(.*)$", line)
        if match:
            problems.append(Problem(relative(Path(match.group(1))),
                                    match.group(3), int(match.group(2))))
    if not problems:
        detail = (proc.stdout + proc.stderr).strip() or f"exit {proc.returncode}"
        problems.append(Problem(fallback, f"{Path(cmd[0]).name} failed: {detail.splitlines()[-1]}"))
    return problems


def check_ruff() -> list[Problem]:
    """Run Ruff over repository Python, with stronger rules on CI code."""
    try:
        tool = _ensure_ruff()
    except Exception as exc:
        return [Problem("scripts/ci", str(exc))]
    problems = _diagnostics(
        [str(tool), "check", "--no-cache", "--output-format", "concise",
         "--select", "E9", "src", "scripts", "tests"],
        "scripts/ci",
    )
    problems += _diagnostics(
        [str(tool), "check", "--no-cache", "--output-format", "concise",
         "--select", "E4,E7,F", "--ignore", "E501", "scripts/ci"],
        "scripts/ci",
    )
    return problems


def check_shellcheck() -> list[Problem]:
    """Shellcheck every shell script and bin wrapper under scripts/ci."""
    scripts = {str(path) for path in CI.rglob("*.sh")}
    runtime_bin = CI / "execution" / "consumer" / "runtime" / "bin"
    scripts.update(str(path) for path in runtime_bin.glob("*") if path.is_file())
    scripts = sorted(scripts)
    if not scripts:
        return []
    try:
        tool = _ensure_shellcheck()
    except Exception as exc:
        return [Problem("scripts/ci", f"cannot obtain shellcheck: {exc}")]
    return _diagnostics([str(tool), "-x", "-f", "gcc", *scripts], "scripts/ci")


def check_actionlint() -> list[Problem]:
    """Lint workflow files with a pinned actionlint."""
    try:
        tool = _ensure_actionlint()
    except Exception as exc:
        return [Problem(".github/workflows", f"cannot obtain actionlint: {exc}")]
    command = [str(tool), "-no-color", "-oneline"]
    for pattern in ACTIONLINT_IGNORES:
        command += ["-ignore", pattern]
    return _diagnostics(command, ".github/workflows", timeout=180)


def check_hadolint() -> list[Problem]:
    """Lint Dockerfiles, failing on new warnings after the explicit allowlist."""
    try:
        tool = _ensure_hadolint()
    except Exception as exc:
        return [Problem("docker", f"cannot obtain hadolint: {exc}")]
    problems = []
    for dockerfile in sorted((ROOT / "docker").rglob("Dockerfile*")):
        cmd = ([str(tool), "--format", "gnu", "--failure-threshold", "warning"]
               + [arg for rule in HADOLINT_IGNORES for arg in ("--ignore", rule)]
               + [str(dockerfile)])
        problems += _diagnostics(cmd, relative(dockerfile))
    return problems

BUILTIN_MARKERS = frozenset({
    "filterwarnings",
    "parametrize",
    "skip",
    "skipif",
    "usefixtures",
    "xfail",
})

def _yaml_document(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError(
            f"PyYAML is not installed (pip install -r {REQUIREMENTS})"
        )

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
    return [needs] if isinstance(needs, str) else list(needs)


def _job_run_text(job: dict) -> str:
    return "\n".join(
        str(step["run"]) for step in (job or {}).get("steps") or []
        if isinstance(step, dict) and step.get("run")
    )


def _trigger_names(document: dict) -> set[str]:
    value = document.get(True, document.get("on")) or {}
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, dict)):
        return {str(item) for item in value}
    return set()


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
        if any("scripts.ci.execution.producer.cli" in _job_run_text(job)
               for job in jobs.values()):
            problems.extend(_check_planner_v2(
                rel,
                jobs,
                yaml.safe_dump(doc, allow_unicode=True),
                requires_lint=bool(_trigger_names(doc) & SOURCE_ENTRYPOINTS),
            ))
            continue
    return problems


def _check_planner_v2(
    rel: str,
    jobs: dict,
    dumped: str,
    *,
    requires_lint: bool,
) -> list[Problem]:
    """Validate the Planner -> Executor -> Gate topology."""
    problems: list[Problem] = []
    planners = [name for name, job in jobs.items()
                if "scripts.ci.execution.producer.cli" in _job_run_text(job)]
    if len(planners) != 1:
        return [Problem(rel, f"expected one Planner job; found {sorted(planners)}")]
    planner_name = planners[0]
    planner = jobs[planner_name]
    if requires_lint and not any(
        marker in _job_run_text(planner)
        for marker in ("scripts/ci/tools/lint.py", "scripts.ci.tools.lint")
    ):
        problems.append(Problem(rel, "Planner job must also run scripts/ci/tools/lint.py"))
    declared = set((planner.get("outputs") or {}).keys())
    output_pattern = rf"needs\.{re.escape(planner_name)}\.outputs\.([A-Za-z0-9_-]+)"
    used = set(re.findall(output_pattern, dumped))
    problems.extend(Problem(
        rel, f"reads needs.{planner_name}.outputs.{name} "
             "but the Planner does not declare it")
        for name in sorted(used - declared))
    problems.extend(Problem(rel, f"Planner output {name!r} has no consumer")
                    for name in sorted(declared - used))

    needed = {dependency for job in jobs.values() for dependency in _job_needs(job)}
    terminals = sorted(set(jobs) - needed)
    direct_children = sorted({name for name, job in jobs.items()
                              if planner_name in _job_needs(job)} - set(terminals))
    lanes = sorted(name for name in direct_children
                   if "scripts.ci.execution.consumer.executor" in _job_run_text(jobs[name]))
    if not lanes:
        problems.append(Problem(rel, "Planner must feed at least one Executor lane"))
    lane_pattern = rf"needs\.{re.escape(planner_name)}\.outputs\.(?:cpu|npu|gpu)_jobs"
    problems.extend(Problem(rel, f"lane {name!r} does not invoke the executor")
                    for name in sorted(set(direct_children) - set(lanes))
                    if re.search(lane_pattern, yaml.safe_dump(jobs[name], allow_unicode=True)))
    if len(terminals) != 1:
        problems.append(Problem(
            rel, f"Planner workflow must have exactly one terminal job; found {terminals}"
        ))
        return problems
    terminal = terminals[0]
    if "scripts.ci.execution.finalization.gate" not in _job_run_text(jobs[terminal]):
        problems.append(Problem(rel, f"terminal job {terminal!r} does not invoke the gate"))
    if "download-artifact" not in yaml.safe_dump(jobs[terminal], allow_unicode=True):
        problems.append(Problem(rel, f"terminal job {terminal!r} does not download results"))
    reachable = _job_closure(jobs, terminal)
    missing = sorted({planner_name, *lanes} - reachable)
    if missing:
        problems.append(Problem(rel, f"terminal job {terminal!r} does not reach {missing}"))
    return problems


def check_config() -> list[Problem]:
    """Validate ci.yml and every test fact derived from the repository."""
    from ..execution.producer.config import CiConfigError, load_config
    from ..execution.producer.facts import FactsError, load_facts

    try:
        load_config(ROOT)
        load_facts(ROOT)
    except (CiConfigError, FactsError) as exc:
        return [Problem(issue.location, issue.message) for issue in exc.issues]
    return []


NPU_MARKERS = frozenset({"npu", "multi_npu"})


def _marker_names(node: ast.AST) -> set[str]:
    values = node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) else [node]
    names: set[str] = set()
    for value in values:
        if isinstance(value, ast.Call):
            value = value.func
        if isinstance(value, ast.Attribute):
            names.add(value.attr)
    return names


def _module_markers(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
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
    return names


def _module_scope_nodes(node: ast.AST) -> Iterable[ast.AST]:
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _module_scope_nodes(child)


def _npu_signal_line(tree: ast.Module) -> int:
    for top in tree.body:
        for node in _module_scope_nodes(top):
            if isinstance(node, ast.Import) and any(
                alias.name == "torch_npu" for alias in node.names
            ):
                return node.lineno
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("torch_npu")
            ):
                return node.lineno
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else (
                    func.id if isinstance(func, ast.Name) else ""
                )
                if name in {"require_npu", "require_npus"}:
                    return node.lineno
                if (
                    name == "importorskip"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == "torch_npu"
                ):
                    return node.lineno
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "npu"
                and isinstance(node.value, ast.Name)
                and node.value.id == "torch"
            ):
                return node.lineno
    return 0


def check_device_markers(root: Path = ROOT) -> list[Problem]:
    """Require NPU modules with module-level signals to declare a device marker."""
    problems: list[Problem] = []
    for path in sorted((root / "tests" / "unit_tests").rglob("test_*.py")):
        try:
            tree = ast.parse(text(path))
        except SyntaxError:
            continue
        markers = _module_markers(tree)
        if markers & NPU_MARKERS or "cpu" in markers:
            continue
        line = _npu_signal_line(tree)
        if line:
            problems.append(Problem(
                path.relative_to(root).as_posix(),
                "NPU-specific module requires pytestmark = pytest.mark.npu",
                line,
            ))
    return problems


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
    if parser.get("pytest", "testpaths", fallback="") != "tests/unit_tests":
        problems.append(Problem(
            "pytest.ini", "testpaths must be tests/unit_tests; integration runs through its runner"))
    configured = {
        line.split(":", 1)[0].strip()
        for line in (parser.get("pytest", "markers", fallback="") or "").splitlines()
        if ":" in line
    }

    from ..execution.producer.facts import REQUIRED_MARKERS

    markers = _test_marker_names()
    problems.extend(Problem("pytest.ini", f"marker {name!r} is used but not registered")
                    for name in sorted(markers - configured))
    problems.extend(Problem("pytest.ini", f"obsolete marker {name!r}")
                    for name in sorted(configured - set(REQUIRED_MARKERS)))
    return problems

CHECKS = (
    Check("ruff", "Python syntax and CI-side Ruff rules", check_ruff),
    Check("shellcheck", "shell scripts under scripts/ci", check_shellcheck),
    Check("actionlint", "GitHub Actions workflow syntax", check_actionlint),
    Check("hadolint", "Dockerfile warnings", check_hadolint),
    Check("contract", "planner/lint/lane topology and output contracts", check_contract),
    Check("config", "CI config, pytest facts, and integration manifest contract", check_config),
    Check("markers", "registered pytest marker contract", check_markers),
    Check("device-markers", "NPU modules declare a device marker", check_device_markers),
)
CHECK_BY_NAME = {check.name: check for check in CHECKS}

def _selected(names: list[str]) -> list[Check]:
    if not names:
        return list(CHECKS)
    unknown = [name for name in names if name not in CHECK_BY_NAME]
    if unknown:
        raise ValueError(f"unknown check(s): {', '.join(unknown)}")
    return [CHECK_BY_NAME[name] for name in names]


def _summary(rows: list[tuple[str, str, int]]) -> str:
    lines = ["## Lint", "", "| check | result | problems |", "|---|---|---:|"]
    for name, result, count in rows:
        lines.append(f"| `{name}` | {result} | {count} |")
    return "\n".join(lines) + "\n"


def run_checks(checks: Iterable[Check], summary_path: str | None) -> int:
    started = time.monotonic()
    total = 0
    rows = []
    failed = []
    for check in checks:
        try:
            problems = check.function()
        except Exception as exc:
            problems = [Problem("<check>", f"check crashed: {type(exc).__name__}: {exc}")]
        total += len(problems)
        rows.append((check.name, "fail" if problems else "ok", len(problems)))
        if not problems:
            print(f"{check.name}: ok")
            continue
        failed.append(check.name)
        print(f"{check.name}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem.render()}")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                fields = f"title={check.name},file={problem.path}"
                if problem.line:
                    fields += f",line={problem.line}"
                message = problem.message.replace("\n", " ")
                print(f"::error {fields}::{message}")

    elapsed = time.monotonic() - started
    destination = summary_path or os.environ.get("GITHUB_STEP_SUMMARY")
    if destination:
        with open(destination, "a", encoding="utf-8") as output:
            output.write(_summary(rows))

    if total:
        print(f"\nFAILED: {', '.join(failed)} ({total} problem(s), {elapsed:.1f}s)")
        return 1
    print(f"\nLint OK ({len(rows)} checks, {elapsed:.1f}s)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run TorchTurbo static checks")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run one check (repeatable)")
    parser.add_argument("--list", action="store_true", help="list check names and exit")
    parser.add_argument("--summary", default=None, help="write a Markdown summary")
    args = parser.parse_args(argv)

    if args.list:
        for check in CHECKS:
            print(f"{check.name}\t{check.description}")
        return 0
    try:
        checks = _selected(args.only)
    except ValueError as exc:
        parser.error(str(exc))
    return run_checks(checks, args.summary)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
