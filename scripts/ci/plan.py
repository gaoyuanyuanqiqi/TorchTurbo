#!/usr/bin/env python3
"""Compile CI selection into jobs, environments, and process-isolated groups."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from facts import CiCatalog, PlanError, TestFacts, load_catalog, scan_tests


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_test_path(root: Path, selection: dict[str, Any], value: str,
                     test_facts: dict[str, TestFacts]) -> tuple[
                         dict[str, Any], dict[str, list[str]], list[PlanError]]:
    """Replace unit lane file lists with an explicit manual test path."""
    if not value or value == "-":
        return selection, {}, []
    selectors: dict[str, list[str]] = {}
    errors: list[PlanError] = []
    lanes = dict(selection.get("lanes") or {})
    for token in value.split():
        base = token.split("::", 1)[0]
        candidate = (root / base).resolve()
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError:
            errors.append(PlanError(token, "test path is outside the repository"))
            continue
        if candidate.is_dir():
            paths = sorted(path for path in test_facts if path.startswith(relative + "/"))
            paths = [path for path in paths if (root / path).is_file()]
        elif candidate.is_file():
            paths = [relative]
        else:
            errors.append(PlanError(token, "test path does not exist"))
            continue
        for path in paths:
            facts = test_facts.get(path)
            if facts is None:
                errors.append(PlanError(path, "manual test path has no marker facts"))
                continue
            selected = token if candidate.is_file() else path
            for device in facts.devices:
                lanes.setdefault(f"{device}_unit", {"run": False, "files": []})
                lanes[f"{device}_unit"]["run"] = True
                lanes[f"{device}_unit"].setdefault("files", []).append(path)
            selectors.setdefault(path, []).append(selected)
    updated = dict(selection)
    updated["lanes"] = lanes
    return updated, selectors, errors


def _rewrite_test_path_files(plan: dict[str, Any],
                             selectors: dict[str, list[str]]) -> None:
    for job in plan["jobs"]:
        for step in job["steps"]:
            if step["kind"] == "pytest":
                step["files"] = sorted({
                    selected
                    for path in step["files"]
                    for selected in selectors.get(path, [path])
                })


def _marker_expression(device: str) -> str:
    return "cpu" if device == "cpu" else f"{device} or multi_{device}"


def _add_test(job: dict[str, Any], facts: TestFacts, device: str) -> None:
    key = (facts.env, facts.process)
    processes = facts.processes_for(device)
    group = job["tests"].setdefault(key, {
        "kind": "pytest",
        "env": facts.env,
        "device": device,
        "marker": _marker_expression(device),
        "process": facts.process,
        "processes": processes,
        "files": [],
    })
    group["processes"] = max(group["processes"], processes)
    group["files"].append(facts.path)


def _job(catalog: CiCatalog, image: str, device: str, stage: str,
         required_slots: int, errors: list[PlanError], path: str,
         work_id: str = "") -> dict[str, Any] | None:
    lane = catalog.lane_for(image, device, required_slots)
    if lane is None:
        errors.append(PlanError(
            path,
            f"no lane for image={image} device={device} slots={required_slots}",
        ))
        return None
    return {
        "id": f"{lane.key}/{work_id}" if work_id else lane.key,
        "image": image,
        "device": device,
        "stage": stage,
        "resource": lane.resource,
        "slots": lane.slots,
        "required_slots": required_slots,
        "runner_variable": lane.runner_variable,
        "runner": lane.runner,
        "tests": {},
        "e2e": defaultdict(list),
    }


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [job["tests"][key] for key in sorted(job["tests"])]
    for group in steps:
        group["files"] = sorted(set(group["files"]))
    for env in sorted(job["e2e"]):
        steps.append({
            "kind": "e2e",
            "env": env,
            "cases": sorted(job["e2e"][env], key=lambda case: case["case_id"]),
        })
    return steps


def _lane_test_files(lane: dict[str, Any], test_facts: dict[str, TestFacts],
                     device: str, errors: list[PlanError], path: str) -> set[str]:
    """Resolve a lane payload; an empty files list means the whole device lane."""
    if not lane.get("run"):
        return set()
    files = set(lane.get("files") or [])
    if files:
        return files
    if lane.get("scope") == "all":
        return {test_path for test_path, facts in test_facts.items()
                if device in facts.devices}
    errors.append(PlanError(path, f"{device} lane is enabled without files or scope=all"))
    return set()


def compile_plan(catalog: CiCatalog, test_facts: dict[str, TestFacts],
                 selection: dict[str, Any]) -> tuple[dict[str, Any], list[PlanError]]:
    errors: list[PlanError] = []
    # Unit and each E2E case get independent jobs to isolate pod state.
    work: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"tests": [], "cases": []}
    )
    lanes = selection.get("lanes") or {}
    for device, lane_name in (("cpu", "cpu_unit"), ("npu", "npu_unit")):
        files = _lane_test_files(lanes.get(lane_name) or {}, test_facts, device,
                                 errors, lane_name)
        for path in sorted(files):
            facts = test_facts.get(path)
            if facts is None:
                errors.append(PlanError(path, "selected test has no marker facts"))
                continue
            if facts.env not in catalog.envs:
                errors.append(PlanError(path, f"unknown env {facts.env!r}"))
            if device not in facts.devices:
                errors.append(PlanError(
                    path, f"selected test does not require {device}"
                ))
                continue
            work[(facts.image, device, "unit", "")]["tests"].append(facts)

    for selected in selection.get("e2e_cases") or []:
        case_id = str(selected.get("case_id") or "")
        case = catalog.case(case_id)
        if case is None:
            errors.append(PlanError("selection", f"unknown E2E case {case_id!r}"))
            continue
        canonical = case.as_selection(str(selected.get("reason") or ""))
        for field in ("model_type", "framework", "env", "layers", "cards", "sp", "ep",
                      "triggers"):
            if field in selected and selected[field] != canonical[field]:
                errors.append(PlanError(
                    case_id, f"selection {field}={selected[field]!r} "
                    f"does not match catalog {canonical[field]!r}"
                ))
        work[(case.image, "npu", "e2e", case_id)]["cases"].append(canonical)

    jobs: list[dict[str, Any]] = []
    for (image, device, stage, work_id), selected_work in sorted(work.items()):
        required_slots = max(
            [facts.processes_for(device) for facts in selected_work["tests"]]
            + [int(case["cards"]) for case in selected_work["cases"]]
            + [1]
        )
        path = (selected_work["tests"][0].path if selected_work["tests"]
                else selected_work["cases"][0]["case_id"])
        job = _job(
            catalog, image, device, stage, required_slots, errors, path, work_id
        )
        if job is None:
            continue
        for facts in selected_work["tests"]:
            _add_test(job, facts, device)
        for case in selected_work["cases"]:
            job["e2e"][case["env"]].append(case)
        jobs.append(job)

    id_counts = Counter(job["id"] for job in jobs)
    for job in jobs:
        if id_counts[job["id"]] > 1:
            job["id"] = f"{job['id']}-{job['stage']}"

    output_jobs = []
    for job in sorted(jobs, key=lambda item: item["id"]):
        output_jobs.append({
            "id": job["id"],
            "image": job["image"],
            "device": job["device"],
            "stage": job["stage"],
            "resource": job["resource"],
            "slots": job["slots"],
            "required_slots": job["required_slots"],
            "runner_variable": job["runner_variable"],
            "runner": job["runner"],
            "steps": _steps(job),
        })
    return {"schema": 1, "mode": selection.get("mode", "pr"), "jobs": output_jobs}, errors


def _summary(plan: dict[str, Any]) -> str:
    lines = ["## Plan", "",
             "| job | required | runner | env | work |",
             "|---|---:|---|---|---|"]
    for job in plan["jobs"]:
        for step in job["steps"]:
            if step["kind"] == "pytest":
                work = f"pytest {len(step['files'])} file(s) / {step['process']}"
            else:
                work = "e2e " + ", ".join(case["case_id"] for case in step["cases"])
            lines.append(
                f"| `{job['id']}` | {job['required_slots']} | `{job['runner']}` | "
                f"`{step['env']}` | {work} |"
            )
    if len(lines) == 4:
        lines.append("| - | - | - | - | no work selected |")
    return "\n".join(lines) + "\n"


def cpu_jobs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return CPU jobs in the compact shape consumed by an Actions matrix."""
    return [job for job in plan["jobs"] if job["device"] == "cpu"]


def device_jobs(plan: dict[str, Any], device: str) -> list[dict[str, Any]]:
    """Return accelerator jobs for one device."""
    return [job for job in plan["jobs"] if job["device"] == device]


def _write_github_output(path: Path, key: str, value: str) -> None:
    delimiter = f"EOF_{uuid.uuid4().hex}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{key}<<{delimiter}\n{value}\n{delimiter}\n")


def _render_errors(errors: list[PlanError]) -> None:
    for error in errors:
        print(f"{error.path}: {error.message}", file=sys.stderr)


def verify(catalog: CiCatalog, test_facts: dict[str, TestFacts],
           errors: list[PlanError]) -> list[PlanError]:
    failures = list(errors)
    cpu = next((facts for facts in test_facts.values() if "cpu" in facts.devices), None)
    npu = next((facts for facts in test_facts.values() if "npu" in facts.devices), None)
    e2e = next((case for case in catalog.cases if "pr" in case.triggers), None)
    if cpu is None:
        failures.append(PlanError("tests", "no CPU test facts found"))
    if npu is None:
        failures.append(PlanError("tests", "no NPU test facts found"))
    sample = {
        "mode": "pr",
        "lanes": {
            "cpu_unit": {"run": cpu is not None, "files": [cpu.path] if cpu else []},
            "npu_unit": {"run": npu is not None, "files": [npu.path] if npu else []},
        },
        "e2e_cases": [e2e.as_selection("verify")] if e2e else [],
    }
    sample_plan, plan_errors = compile_plan(catalog, test_facts, sample)
    failures.extend(plan_errors)
    sample_cpu_jobs = cpu_jobs(sample_plan)
    if sample_cpu_jobs:
        try:
            json.dumps(sample_cpu_jobs, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            failures.append(PlanError("verify", f"host matrix is not JSON serializable: {exc}"))
    else:
        failures.append(PlanError("verify", "sample CPU selection produced no host job"))
    sample_npu_jobs = device_jobs(sample_plan, "npu")
    stages = sorted(job["stage"] for job in sample_npu_jobs)
    if stages != ["e2e", "unit"]:
        failures.append(PlanError(
            "verify", f"NPU unit and E2E must produce two jobs, got {stages}"
        ))
    if len({job["id"] for job in sample_npu_jobs}) != len(sample_npu_jobs):
        failures.append(PlanError("verify", "NPU jobs must have unique ids"))
    if len(catalog.cases) >= 2:
        cases = [case.as_selection("verify") for case in catalog.cases[:2]]
        e2e_plan, e2e_errors = compile_plan(
            catalog, test_facts, {"e2e_cases": cases}
        )
        failures.extend(e2e_errors)
        e2e_jobs = [job for job in device_jobs(e2e_plan, "npu")
                    if job["stage"] == "e2e"]
        if len(e2e_jobs) != 2:
            failures.append(PlanError(
                "verify", "each E2E case must produce its own job"
            ))
        if len({job["id"] for job in e2e_jobs}) != len(e2e_jobs):
            failures.append(PlanError("verify", "E2E case jobs must have unique ids"))
        for job in e2e_jobs:
            cases = sum(len(step.get("cases", [])) for step in job["steps"])
            if cases != 1:
                failures.append(PlanError(
                    "verify", f"E2E job {job['id']} must contain exactly one case, got {cases}"
                ))
    mixed_jobs = [
        job["id"] for job in sample_plan["jobs"]
        if {step["kind"] for step in job["steps"]} == {"pytest", "e2e"}
    ]
    if mixed_jobs:
        failures.append(PlanError(
            "verify", f"unit and E2E work must use separate jobs: {mixed_jobs}"
        ))
    if cpu is not None:
        _, manual_files, manual_errors = _apply_test_path(
            catalog.root, sample, cpu.path, test_facts
        )
        failures.extend(manual_errors)
        if not manual_files:
            failures.append(PlanError("verify", "manual CPU test path produced no files"))
    failures.extend(_verify_rejections(catalog, test_facts, cpu, npu))
    return failures


def _verify_rejections(catalog: CiCatalog, test_facts: dict[str, TestFacts],
                       cpu: TestFacts | None, npu: TestFacts | None) -> list[PlanError]:
    """Prove malformed selections fail instead of producing a partial plan."""
    if cpu is None:
        return []
    failures: list[PlanError] = []

    def compile_with(facts: TestFacts, selection: dict[str, Any]) -> list[PlanError]:
        return compile_plan(catalog, {facts.path: facts}, selection)[1]

    def lane(files: list[str]) -> dict[str, Any]:
        return {"mode": "pr", "lanes": {
            "cpu_unit": {"run": True, "files": files},
            "npu_unit": {"run": False},
        }}

    def expect(name: str, run_errors: list[PlanError], message: str) -> None:
        if not any(message in error.message for error in run_errors):
            failures.append(PlanError("verify", f"negative case did not fail: {name}"))

    expect("unknown env", compile_with(replace(cpu, env="missing-env"), lane([cpu.path])),
           "unknown env")
    expect("unknown image", compile_with(replace(cpu, image="missing-image"),
                                         lane([cpu.path])), "no lane for image")
    expect("missing device", compile_with(replace(cpu, devices=frozenset()),
                                          lane([cpu.path])), "does not require cpu")
    expect("unknown E2E case", compile_plan(
        catalog, test_facts, {"e2e_cases": [{"case_id": "missing-case"}]}
    )[1], "unknown E2E case")
    if catalog.cases:
        stale = {"case_id": catalog.cases[0].case_id, "cards": 999}
        expect("stale E2E facts", compile_plan(
            catalog, test_facts, {"e2e_cases": [stale]}
        )[1], "does not match catalog")

    lane_errors: list[PlanError] = []
    if _job(catalog, cpu.image, "cpu", "unit", 99, lane_errors, "verify") is not None:
        failures.append(PlanError("verify", "negative case accepted an unavailable lane"))
    expect("unavailable lane", lane_errors, "no lane for")

    if npu is not None:
        whole_lane = {"lanes": {
            "cpu_unit": {"run": False},
            "npu_unit": {"run": True, "files": [], "scope": "all"},
        }}
        plan, whole_errors = compile_plan(catalog, test_facts, whole_lane)
        if whole_errors:
            failures.extend(whole_errors)
        elif not any(step["kind"] == "pytest"
                     for job in plan["jobs"] for step in job["steps"]):
            failures.append(PlanError("verify", "whole-lane NPU selection produced no tests"))
        else:
            npu_jobs = device_jobs(plan, "npu")
            stages = [job["stage"] for job in npu_jobs]
            if len(npu_jobs) != 1 or stages != ["unit"]:
                failures.append(PlanError(
                    "verify", "whole-lane NPU selection must produce one unit job"
                ))
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--selection", default=None, help="selector JSON")
    parser.add_argument("--output", default=None, help="write plan JSON here")
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--github-output", default=None,
                        help="append CPU/NPU/GPU matrix JSON to a GitHub output file")
    parser.add_argument("--test-path", default="",
                        help="manual CPU test path; '-' keeps the selector output")
    parser.add_argument("--verify", action="store_true", help="validate facts only")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[2]
    catalog = load_catalog(root)
    test_facts, test_errors = scan_tests(catalog)
    failures = [PlanError(error.split(":", 1)[0], error.split(":", 1)[-1].strip())
                for error in catalog.errors] + test_errors
    if args.verify:
        failures = verify(catalog, test_facts, failures)
    if failures:
        _render_errors(failures)
        return 1
    if args.verify and not args.selection:
        print(f"plan facts OK ({len(catalog.cases)} E2E cases / {len(test_facts)} tests)")
        return 0
    if not args.selection:
        parser.error("--selection is required unless --verify is used")

    selection = _load_selection(Path(args.selection))
    selection, selectors, path_errors = _apply_test_path(
        root, selection, args.test_path, test_facts
    )
    if path_errors:
        _render_errors(path_errors)
        return 1
    plan, plan_errors = compile_plan(catalog, test_facts, selection)
    if plan_errors:
        _render_errors(plan_errors)
        return 1
    _rewrite_test_path_files(plan, selectors)
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as handle:
            handle.write(_summary(plan))
    if args.github_output:
        output = Path(args.github_output)
        for key, jobs in (("cpu", cpu_jobs(plan)),
                          ("npu", device_jobs(plan, "npu"))):
            value = json.dumps(jobs, ensure_ascii=False, separators=(",", ":"))
            _write_github_output(output, f"{key}_jobs", value)
            _write_github_output(output, f"{key}_job_count", str(len(jobs)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
