"""pytest Job command construction, JUnit merging, and test evidence."""

from __future__ import annotations

import os
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path

from ...contracts.executor_contract import TaskEvidence
from ...contracts.planner_contract import PytestSpec
from .runtime.venv_manager import Venv


def report_path(output_dir: Path, index: int) -> Path:
    path = output_dir / "junit" / f"task-{index:02d}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def command(venv: Venv, spec: PytestSpec, junit: Path) -> list[str]:
    args = shlex.split(os.environ.get("PYTEST_ARGS", "-v --durations=20"))
    return [
        str(venv.python),
        "-m",
        "pytest",
        *args,
        "-m",
        spec.device_marker,
        "--continue-on-collection-errors",
        f"--junitxml={junit}",
        *spec.test_files,
    ]


def _suites(path: Path) -> list[ET.Element]:
    if not path.is_file():
        return []
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    return [root] if root.tag == "testsuite" else root.findall("testsuite")


def merge_reports(paths: list[Path], output: Path) -> None:
    merged = ET.Element("testsuites")
    for path in paths:
        merged.extend(_suites(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(output, encoding="utf-8", xml_declaration=True)


def evidence(path: Path) -> TaskEvidence:
    suites = _suites(path)

    def count(key: str) -> int:
        return sum(int(suite.get(key, 0)) for suite in suites)

    return TaskEvidence(
        tests_total=count("tests"),
        tests_failed=count("failures"),
        tests_errors=count("errors"),
        tests_skipped=count("skipped"),
    )
