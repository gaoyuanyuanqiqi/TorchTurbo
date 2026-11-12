#!/usr/bin/env python3
"""Compile one manual unit lane with PR CI's environment profiles."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEVICE_BY_LANE = {"cpu-unit": "cpu", "npu-unit": "npu"}


def _selection(device: str, scope: str, test_path: str,
               changed_files: list[str]) -> dict[str, Any]:
    """Build the minimal selection consumed by plan.py."""
    lane = {"run": False}
    if scope == "all" and not test_path:
        lane = {"run": True, "files": [], "scope": "all"}
    return {
        "mode": "manual",
        "changed": changed_files,
        "lanes": {f"{device}_unit": lane},
    }


def _write_output(path: str, values: dict[str, str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as output:
        for key, value in values.items():
            output.write(f"{key}={value}\n")


def _run_plan(
    root: Path,
    plan_path: Path,
    selection_path: Path,
    device: str,
    scope: str,
    test_path: str,
    changed_files: list[str],
) -> int:
    selection_path.write_text(
        json.dumps(_selection(device, scope, test_path, changed_files)),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        str(root / "scripts/ci/plan.py"),
        "--root",
        str(root),
        "--selection",
        str(selection_path),
        "--output",
        str(plan_path),
    ]
    if test_path:
        command.extend(["--test-path", test_path])
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        command.extend(["--summary-file", summary])
    return subprocess.run(command, cwd=root, check=False).returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", required=True, choices=sorted(DEVICE_BY_LANE))
    parser.add_argument("--scope", required=True, choices=("smoke", "all"))
    parser.add_argument(
        "--test-path", default="", help="space-separated paths or node ids"
    )
    parser.add_argument(
        "--smoke", default="", help="smoke paths used when test-path is empty"
    )
    parser.add_argument(
        "--changed-files", default="", help="newline-separated changed paths"
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--plan", default="/tmp/manual-plan.json")
    parser.add_argument("--github-output", default="")
    args = parser.parse_args(argv)

    root = (
        Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    )
    device = DEVICE_BY_LANE[args.lane]
    test_path = args.test_path.strip()
    if not test_path and args.scope == "smoke":
        test_path = args.smoke.strip()
        if not test_path:
            print("::error::smoke scope needs --smoke paths", file=sys.stderr)
            return 2
    changed_files: list[str] = []
    if args.changed_files:
        changed_files = [
            line.strip()
            for line in Path(args.changed_files).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    plan_path = Path(args.plan)
    selection_path = plan_path.with_name(f"{plan_path.name}.selection.json")
    rc = _run_plan(
        root, plan_path, selection_path, device, args.scope, test_path, changed_files
    )
    if rc != 0:
        return rc

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    jobs = [job for job in plan.get("jobs", []) if job.get("device") == device]
    if len(jobs) != 1:
        print(
            f"::error::manual {args.lane} produced {len(jobs)} {device} jobs; "
            "expected one. Check that the selected tests run on this lane.",
            file=sys.stderr,
        )
        return 1

    job = jobs[0]
    _write_output(args.github_output, {"job_id": str(job["id"])})
    print(f"manual job: {job['id']}")
    for step in job.get("steps", []):
        if step.get("kind") == "pytest":
            print(
                f"  env={step['env']} process={step.get('process', '')} "
                f"files={len(step.get('files', []))}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
