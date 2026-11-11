#!/usr/bin/env python3
"""Compile CI selection into jobs, environments, and process-isolated groups."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from facts import CiCatalog, PlanError, TestFacts, load_catalog, scan_tests


def _load_selection(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _add_test(job: dict[str, Any], facts: TestFacts, device: str,
              rank: int) -> None:
    key = (facts.env, facts.process)
    group = job["tests"].setdefault(key, {
        "kind": "pytest",
        "env": facts.env,
        "device": device,
        "marker": device,
        "process": facts.process,
        "rank": rank,
        "files": [],
    })
    group["rank"] = max(group["rank"], rank)
    group["files"].append(facts.path)


def _job(catalog: CiCatalog, image: str, device: str, slots: int,
         errors: list[PlanError], path: str) -> dict[str, Any] | None:
    lane = catalog.lane_for(image, device, slots)
    if lane is None:
        errors.append(PlanError(path, f"no lane for image={image} device={device} slots={slots}"))
        return None
    return {
        "id": lane.key,
        "image": image,
        "device": device,
        "resource": lane.resource,
        "slots": lane.slots,
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
        rank_field = "cpu_ranks" if device == "cpu" else "npu_ranks"
        return {test_path for test_path, facts in test_facts.items()
                if getattr(facts, rank_field)}
    errors.append(PlanError(path, f"{device} lane is enabled without files or scope=all"))
    return set()


def compile_plan(catalog: CiCatalog, test_facts: dict[str, TestFacts],
                 selection: dict[str, Any]) -> tuple[dict[str, Any], list[PlanError]]:
    errors: list[PlanError] = []
    jobs: dict[str, dict[str, Any]] = {}
    lanes = selection.get("lanes") or {}
    for device, lane_name, rank_field in (
        ("cpu", "cpu_unit", "cpu_ranks"),
        ("npu", "npu_unit", "npu_ranks"),
    ):
        files = _lane_test_files(lanes.get(lane_name) or {}, test_facts, device,
                                 errors, lane_name)
        for path in sorted(files):
            facts = test_facts.get(path)
            if facts is None:
                errors.append(PlanError(path, "selected test has no marker facts"))
                continue
            if facts.env not in catalog.envs:
                errors.append(PlanError(path, f"unknown env {facts.env!r}"))
            ranks = getattr(facts, rank_field)
            if not ranks:
                errors.append(PlanError(path, "selected test has no rank for its lane"))
                continue
            rank = max(ranks)
            job = _job(catalog, facts.image, device, rank, errors, path)
            if job is None:
                continue
            jobs.setdefault(job["id"], job)
            _add_test(jobs[job["id"]], facts, device, rank)

    for selected in selection.get("e2e_cases") or []:
        case_id = str(selected.get("case_id") or "")
        case = catalog.case(case_id)
        if case is None:
            errors.append(PlanError("selection", f"unknown E2E case {case_id!r}"))
            continue
        canonical = case.as_selection(str(selected.get("reason") or ""))
        for field in ("model_type", "framework", "env", "layers", "cards", "sp", "ep", "tier"):
            if field in selected and selected[field] != canonical[field]:
                errors.append(PlanError(
                    case_id, f"selection {field}={selected[field]!r} "
                    f"does not match catalog {canonical[field]!r}"
                ))
        job = _job(catalog, case.image, "npu", case.cards, errors, case.case_id)
        if job is None:
            continue
        jobs.setdefault(job["id"], job)
        jobs[job["id"]]["e2e"][case.env].append(canonical)

    output_jobs = []
    for job_id in sorted(jobs):
        job = jobs[job_id]
        output_jobs.append({
            "id": job_id,
            "image": job["image"],
            "device": job["device"],
            "resource": job["resource"],
            "slots": job["slots"],
            "runner_variable": job["runner_variable"],
            "runner": job["runner"],
            "steps": _steps(job),
        })
    return {"schema": 1, "mode": selection.get("mode", "pr"), "jobs": output_jobs}, errors


def _summary(plan: dict[str, Any]) -> str:
    lines = ["## Plan (shadow)", "", "| job | runner | env | work |", "|---|---|---|---|"]
    for job in plan["jobs"]:
        for step in job["steps"]:
            if step["kind"] == "pytest":
                work = f"pytest {len(step['files'])} file(s) / {step['process']}"
            else:
                work = "e2e " + ", ".join(case["case_id"] for case in step["cases"])
            lines.append(
                f"| `{job['id']}` | `{job['runner']}` | `{step['env']}` | {work} |"
            )
    if len(lines) == 4:
        lines.append("| - | - | - | no work selected |")
    return "\n".join(lines) + "\n"


def _render_errors(errors: list[PlanError]) -> None:
    for error in errors:
        print(f"{error.path}: {error.message}", file=sys.stderr)


def verify(catalog: CiCatalog, test_facts: dict[str, TestFacts],
           errors: list[PlanError]) -> list[PlanError]:
    failures = list(errors)
    cpu = next((facts for facts in test_facts.values() if facts.cpu_ranks), None)
    npu = next((facts for facts in test_facts.values() if facts.npu_ranks), None)
    e2e = next((case for case in catalog.cases if case.tier == "pr"), None)
    if cpu is None:
        failures.append(PlanError("tests", "no CPU test facts found"))
    if npu is None:
        failures.append(PlanError("tests", "no NPU test facts found"))
    sample = {
        "mode": "pr",
        "lanes": {
            "cpu_unit": {"files": [cpu.path] if cpu else []},
            "npu_unit": {"files": [npu.path] if npu else []},
        },
        "e2e_cases": [e2e.as_selection("verify")] if e2e else [],
    }
    _, plan_errors = compile_plan(catalog, test_facts, sample)
    failures.extend(plan_errors)
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
    expect("missing rank", compile_with(replace(cpu, cpu_ranks=frozenset()),
                                        lane([cpu.path])), "no rank")
    expect("unknown E2E case", compile_plan(
        catalog, test_facts, {"e2e_cases": [{"case_id": "missing-case"}]}
    )[1], "unknown E2E case")
    if catalog.cases:
        stale = {"case_id": catalog.cases[0].case_id, "cards": 999}
        expect("stale E2E facts", compile_plan(
            catalog, test_facts, {"e2e_cases": [stale]}
        )[1], "does not match catalog")

    lane_errors: list[PlanError] = []
    if _job(catalog, cpu.image, "cpu", 99, lane_errors, "verify") is not None:
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
    return failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--selection", default=None, help="selector JSON")
    parser.add_argument("--output", default=None, help="write plan JSON here")
    parser.add_argument("--summary-file", default=None)
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
    plan, plan_errors = compile_plan(catalog, test_facts, selection)
    if plan_errors:
        _render_errors(plan_errors)
        return 1
    text = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as handle:
            handle.write(_summary(plan))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
