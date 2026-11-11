#!/usr/bin/env python3
"""Execute one plan job in its environment-specific venvs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from venv_manager import Venv, VenvManager


def _runner_temp() -> Path:
    return Path(os.environ.get("RUNNER_TEMP", "/tmp"))


def _log(message: str) -> None:
    print(message, flush=True)


def _report(command: str, result_json: Path, *args: str) -> None:
    script = Path(__file__).resolve().parent / "report" / "report_result.py"
    subprocess.run(
        [sys.executable, str(script), command, "--file", str(result_json), *args],
        check=False,
    )


def _phase(result_json: Path, name: str, seconds: int, rc: int) -> None:
    _report("phase", result_json, "--name", name,
            "--seconds", str(seconds), "--rc", str(rc))


def _stream(command: list[str], *, cwd: Path, environment: dict[str, str],
            log_path: Path) -> int:
    _log("+ " + " ".join(command))
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return process.wait()


def _pytest_command(venv: Venv, step: dict[str, Any], junit: Path) -> list[str]:
    args = shlex.split(os.environ.get("PYTEST_ARGS", "-v --durations=20"))
    return [
        str(venv.python), "-m", "pytest", *args, "-m", step["marker"],
        "--continue-on-collection-errors", f"--junitxml={junit}", *step["files"],
    ]


def _merge_junit(paths: list[Path], output: Path) -> None:
    suites = []
    for path in paths:
        if not path.is_file():
            continue
        root = ET.parse(path).getroot()
        suites.extend([root] if root.tag == "testsuite" else root.findall("testsuite"))
    merged = ET.Element("testsuites")
    for suite in suites:
        merged.append(suite)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged).write(output, encoding="utf-8", xml_declaration=True)


def _write_versions(venvs: dict[str, Venv], output: Path, root: Path) -> None:
    code = (
        "import importlib.metadata as m\n"
        "for name in ('torch', 'transformers', 'peft', 'numpy', "
        "'llamafactory', 'swift'):\n"
        "    try: print(f'{name} = {m.version(name)}')\n"
        "    except m.PackageNotFoundError: pass\n"
    )
    lines = []
    for env_name, venv in sorted(venvs.items()):
        lines.append(f"[{env_name}]")
        result = subprocess.run(
            [str(venv.python), "-c", code], cwd=root, env=venv.environment(),
            capture_output=True, text=True, check=False,
        )
        lines.append(result.stdout.strip() or "(no versions)")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _execute(job: dict[str, Any], root: Path) -> int:
    if job.get("device") != "cpu":
        _log(f"::error::run_job currently accepts CPU jobs only: {job.get('id')}")
        return 2
    steps = job.get("steps") or []
    if not steps:
        _log(f"::error::job has no steps: {job.get('id')}")
        return 2

    result_json = Path(os.environ.get("RESULT_JSON", _runner_temp() / "ci-result.json"))
    log_path = Path(os.environ.get("CI_JOB_LOG", _runner_temp() / "turbo-ci.log"))
    merged_junit = Path(os.environ.get("JUNIT_OUT", _runner_temp() / "pytest.xml"))
    versions = Path(os.environ.get("VERSIONS_OUT", _runner_temp() / "versions.txt"))
    venv_root = Path(os.environ.get(
        "CI_VENV_ROOT", _runner_temp() / "turbo-ci-venv"
    ))
    runner_label = os.environ.get("RUNNER_LABEL", str(job.get("runner", "")))
    _report(
        "init", result_json, "kind=pytest", "model_type=pytest-cpu-unit",
        "mode=cpu", f"runner_label={runner_label}", f"image={job.get('image', '')}",
        f"job_id={job.get('id', '')}", f"test_path={job.get('id', '')}",
        f"pytest_args={os.environ.get('PYTEST_ARGS', '-v --durations=20')}",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    manager = VenvManager(root, venv_root, "cpu", log_path)
    venvs: dict[str, Venv] = {}
    junit_paths: list[Path] = []
    rc = 0
    for index, step in enumerate(steps):
        if step.get("kind") != "pytest":
            _log(f"::error::unsupported CPU step kind: {step.get('kind')}")
            rc = 2
            break
        env_name = str(step["env"])
        venv = venvs.get(env_name)
        if venv is None:
            started = time.monotonic()
            try:
                venv = manager.ensure(env_name)
                venvs[env_name] = venv
            except Exception as exc:  # noqa: BLE001 - report the install failure
                _log(f"::error::environment {env_name} failed: {exc}")
                _phase(result_json, f"install {env_name}", int(time.monotonic() - started), 1)
                rc = 1
                break
            _phase(result_json, f"install {env_name}", int(time.monotonic() - started), 0)

        junit = _runner_temp() / "turbo-junit" / f"step-{index:02d}.xml"
        junit.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        step_rc = _stream(
            _pytest_command(venv, step, junit), cwd=root,
            environment=venv.environment(), log_path=log_path,
        )
        elapsed = int(time.monotonic() - started)
        _phase(result_json, f"pytest {env_name} {step.get('process', '')}",
               elapsed, step_rc)
        junit_paths.append(junit)
        if step_rc != 0:
            rc = step_rc
            break

    _merge_junit(junit_paths, merged_junit)
    _write_versions(venvs, versions, root)
    return rc


def _load_job(args: argparse.Namespace) -> dict[str, Any]:
    if args.job_json:
        return json.loads(args.job_json)
    if not args.plan or not args.job_id:
        raise ValueError("provide --job-json or both --plan and --job-id")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    for job in plan.get("jobs", []):
        if job.get("id") == args.job_id:
            return job
    raise ValueError(f"job {args.job_id!r} is not in {args.plan}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-json", default="", help="one matrix job object")
    parser.add_argument("--plan", default="", help="plan JSON")
    parser.add_argument("--job-id", default="", help="job id inside --plan")
    parser.add_argument("--root", default=None)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    try:
        job = _load_job(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _log(f"::error::{exc}")
        return 2
    return _execute(job, root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

