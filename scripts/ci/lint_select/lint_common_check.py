#!/usr/bin/env python3
"""Paths, result types, and generic lint checks for CI.

Ruff, shellcheck, actionlint, and hadolint are the only implementations here.
TorchTurbo-specific contracts live in lint_turbo_check.py.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
WORKFLOWS = ROOT / ".github" / "workflows"
CI = ROOT / "scripts" / "ci"
TOOL_DIR = Path(os.environ.get("TORCHTURBO_LINT_TOOLS", "/tmp/torchturbo-lint-tools"))

ACTIONLINT_VERSION = "1.7.12"
HADOLINT_VERSION = "2.15.1"
SHELLCHECK_VERSION = "0.11.0"
# ruff and PyYAML are not self-bootstrapped; their pinned versions live in
# requirements-lint.txt alone, so this file cannot drift from the workflow.
REQUIREMENTS = "requirements-lint.txt"

# These are build-policy choices, not accidental lint debt.
HADOLINT_IGNORES = ("DL3041", "DL3042", "DL3003", "DL3059")


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
    if machine in ("aarch64", "arm64"):
        return "arm64"
    if machine in ("x86_64", "amd64"):
        return "amd64"
    raise RuntimeError(f"unsupported architecture: {machine}")



PROBE = {"actionlint": "-version", "hadolint": "--version", "shellcheck": "--version"}


def _tool_path(name: str, version: str | None = None) -> Path | None:
    candidates = [
        os.environ.get(f"{name.upper()}_BIN"),
        TOOL_DIR / name,
        Path("/tmp") / name,
        shutil.which(name),
    ]
    for index, candidate in enumerate(candidates):
        if not candidate or not Path(candidate).is_file():
            continue
        if not os.access(candidate, os.X_OK):
            continue
        # An explicit *_BIN override is honoured as-is; everything else must
        # report the pinned version or the download below supplies it instead.
        if index == 0 or version is None:
            return Path(candidate)
        try:
            probe = run([str(candidate), PROBE[name]], timeout=30)
            reported = probe.stdout + probe.stderr
        except (OSError, subprocess.SubprocessError):
            reported = ""
        if version in reported:
            return Path(candidate)
    return None


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "torchturbo-ci-lint"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(temporary, "wb") as output:
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


def _ensure_actionlint() -> Path:
    found = _tool_path("actionlint", ACTIONLINT_VERSION)
    if found:
        return found
    arch = _arch()
    version = ACTIONLINT_VERSION
    url = (f"https://github.com/rhysd/actionlint/releases/download/v{version}/"
           f"actionlint_{version}_linux_{arch}.tar.gz")
    archive = TOOL_DIR / f"actionlint-{version}-{arch}.tar.gz"
    _download(url, archive)
    return _extract_binary(archive, TOOL_DIR / "actionlint", "actionlint")


def _ensure_hadolint() -> Path:
    found = _tool_path("hadolint", HADOLINT_VERSION)
    if found:
        return found
    arch = "arm64" if _arch() == "arm64" else "x86_64"
    url = (f"https://github.com/hadolint/hadolint/releases/download/"
           f"v{HADOLINT_VERSION}/hadolint-linux-{arch}")
    target = TOOL_DIR / "hadolint"
    _download(url, target)
    return target


def _ensure_shellcheck() -> Path:
    found = _tool_path("shellcheck", SHELLCHECK_VERSION)
    if found:
        return found
    arch = "aarch64" if _arch() == "arm64" else "x86_64"
    version = SHELLCHECK_VERSION
    url = (f"https://github.com/koalaman/shellcheck/releases/download/v{version}/"
           f"shellcheck-v{version}.linux.{arch}.tar.xz")
    archive = TOOL_DIR / f"shellcheck-{version}-{arch}.tar.xz"
    _download(url, archive)
    return _extract_binary(archive, TOOL_DIR / "shellcheck", "shellcheck")


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
    """Shellcheck every shell script under scripts/ci."""
    scripts = sorted(str(path) for path in CI.rglob("*.sh"))
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
    return _diagnostics([str(tool), "-no-color", "-oneline"], ".github/workflows", timeout=180)


def check_hadolint() -> list[Problem]:
    """Lint Dockerfiles, failing on new warnings after the explicit allowlist."""
    try:
        tool = _ensure_hadolint()
    except Exception as exc:
        return [Problem("docker", f"cannot obtain hadolint: {exc}")]
    problems = []
    for dockerfile in sorted((ROOT / "docker").rglob("Dockerfile*")):
        cmd = [str(tool), "--format", "gnu", "--failure-threshold", "warning"]
        for rule in HADOLINT_IGNORES:
            cmd += ["--ignore", rule]
        cmd.append(str(dockerfile))
        problems += _diagnostics(cmd, relative(dockerfile))
    return problems


CHECKS = (
    Check("ruff", "Python syntax and CI-side Ruff rules", check_ruff),
    Check("shellcheck", "shell scripts under scripts/ci", check_shellcheck),
    Check("actionlint", "GitHub Actions workflow syntax", check_actionlint),
    Check("hadolint", "Dockerfile warnings", check_hadolint),
)
