#!/usr/bin/env python3
"""End-to-end self-test for the CI decision layer (lint + select).

Runs the static checks, the selector's own rule assertions, then a scenario
matrix that drives `select_tests.py` with real repository paths and asserts:

  * lane switches and mode for a known change set;
  * every selected file exists and carries its lane's pytest marker;
  * the selector's outputs cover what the workflows declare and consume;
  * the plan summary renders the image it was given.

Deliberately stdlib-only and read-only: it invokes the scripts as a user would
and never imports them, so a broken CLI surface fails here rather than in CI.

Local only: no workflow runs this file. Run it before pushing a change to the
CI scripts, or from a `pre-commit` hook.

Usage: python3 scripts/ci/local_lint_select_test.py [--verbose] [--only SUBSTRING]
Exit:  0 all pass / 1 any failure
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
CI = ROOT / "scripts" / "ci"
SELECTOR = CI / "lint_select" / "select_tests.py"
LINT = CI / "lint_select" / "lint_check.py"
WORKFLOWS = ROOT / ".github" / "workflows"
MODEL_MAP = WORKFLOWS / "model_map.json"

# Marker contract: the lane a file lands in must match the marker it declares.
CPU_MARK = re.compile(r"pytest\.mark\.cpu\b")
NPU_MARK = re.compile(r"pytest\.mark\.(?:multi_npu|npu)\b")

LANE_KEYS = ("cpu_unit", "npu_unit", "npu_e2e")

FAILS: list[str] = []
VERBOSE = False


# ---------------------------------------------------------------- plumbing
def run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if VERBOSE and proc.stderr.strip():
        print(proc.stderr.rstrip())
    return proc


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{'  ' + detail if detail and not ok else ''}")
    if not ok:
        FAILS.append(f"{name}{': ' + detail if detail else ''}")
    return ok


def select(files: list[str], labels: str = "", mode: str | None = None,
           image: str = "-") -> dict:
    """Run the selector the way ci-pr.yml does and return its GITHUB_OUTPUT."""
    out, summary = Path(tempfile.mktemp()), Path(tempfile.mktemp())
    cmd = [sys.executable, str(SELECTOR), "--github-output", str(out),
           "--summary-file", str(summary), "--image-ref", image,
           "--output", os.devnull]
    for path in files:
        cmd += ["--changed-file", path]
    if mode:
        cmd += ["--mode", mode, "--labels", labels]
    proc = run(cmd)
    got: dict = {"_rc": proc.returncode, "_stderr": proc.stderr}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                got[key] = value
    got["_summary"] = summary.read_text(encoding="utf-8") if summary.exists() else ""
    out.unlink(missing_ok=True)
    summary.unlink(missing_ok=True)
    return got


def tail(text: str) -> str:
    """Last non-empty line, for a one-line failure hint."""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def flag(got: dict, lane: str) -> bool:
    return got.get(f"{lane}_run") == "true"


def listed(got: dict, lane: str) -> list[str]:
    try:
        return json.loads(got.get(f"{lane}_files", "[]"))
    except json.JSONDecodeError:
        return []


def cases(got: dict) -> list[dict]:
    try:
        return json.loads(got.get("e2e_cases", "[]"))
    except json.JSONDecodeError:
        return []


def cases_with_trigger(trigger: str) -> int:
    rows = json.loads(MODEL_MAP.read_text(encoding="utf-8")).get("models", {})
    return sum(
        1
        for row in rows.values()
        for case in row.get("cases") or []
        if trigger in (case.get("triggers") or [])
    )


def models_with_trigger(trigger: str) -> list[str]:
    rows = json.loads(MODEL_MAP.read_text(encoding="utf-8")).get("models", {})
    picked = [
        (case["id"], model)
        for model, row in rows.items()
        for case in row.get("cases") or []
        if trigger in (case.get("triggers") or [])
    ]
    return [model for _, model in sorted(picked)]


# ---------------------------------------------------------------- stages
def step_lint() -> None:
    print("\n[1/5] static checks")
    proc = run([sys.executable, str(LINT)])
    ok = check("lint_check.py exits 0", proc.returncode == 0, tail(proc.stdout or proc.stderr))
    # `CHECKS` grows; assert the run finished, not which checks exist.
    check("lint run is clean", "Lint OK" in proc.stdout, proc.stdout[-200:])
    if VERBOSE or not ok:
        print("    " + proc.stdout.strip().replace("\n", "\n    "))


def step_verify() -> None:
    print("\n[2/5] selector rule assertions")
    proc = run([sys.executable, str(SELECTOR), "--verify"])
    passes = len(re.findall(r"^  ok    ", proc.stdout, re.M))
    check("--verify exits 0", proc.returncode == 0)
    check(f"--verify ran >= 30 assertions (got {passes})", passes >= 30)
    check("--verify ends with PASS", proc.stdout.rstrip().endswith("PASS"))


SCENARIOS = [
    # Docs-only: nothing to run, and that must be explicit rather than a crash.
    dict(name="docs only", files=["docs/ci/README.md"], expect="precise",
         cpu=False, npu=False, e2e=False, e2e_count=0),
    # A test file belongs to the lane its marker names, and only that lane.
    dict(name="cpu test file", files=["tests/engine/test_model_fqn.py"], expect="precise",
         cpu=True, npu=False, e2e=False, cpu_files=["tests/engine/test_model_fqn.py"]),
    dict(name="npu test file", files=["tests/engine/test_muon_offload_distributed.py"],
         expect="precise", cpu=False, npu=True, e2e=False,
         npu_files=["tests/engine/test_muon_offload_distributed.py"]),
    # Mixed file: both halves of the same file go to their own lane.
    dict(name="mixed cpu+npu file", files=["tests/ops/test_npu_cross_entropy_loss.py"],
         expect="precise", cpu=True, npu=True, e2e=False,
         cpu_files=["tests/ops/test_npu_cross_entropy_loss.py"],
         npu_files=["tests/ops/test_npu_cross_entropy_loss.py"]),
    # Ordinary source change: precise, both unit lanes, cheapest E2E case.
    dict(name="engine core", files=["src/torchturbo/engine/core.py"], expect="precise",
         cpu=True, npu=True, e2e=True, e2e_models=["qwen3_5"]),
    # Blast radius above MAX_SELECTED_TESTS -> the selector must widen to all.
    dict(name="ops impl (too many tests)", files=["src/torchturbo/ops/ops.py"],
         expect="all", cpu=True, npu=True, e2e=True,
         npu_scope="all", full_reason="too-many-tests"),
    # Editing a family directory picks that family, with its layer trim.
    dict(name="moe family dir", files=["src/torchturbo/models/qwen3_5_moe/__init__.py"],
         expect="precise", cpu=True, npu=True, e2e=True,
         e2e_cases=[{"model_type": "qwen3_5_moe", "layers": 6}]),
    dict(name="run config", files=["scripts/runs/qwen3_5_dense/lf_text_sft.yaml"],
         expect="precise", cpu=False, npu=False, e2e=True, e2e_models=["qwen3_5"]),
    dict(name="shared model layer", files=["src/torchturbo/models/modules/attention/__init__.py"],
         expect="all", cpu=True, npu=True, e2e=True,
         full_reason="too-many-tests",
         e2e_cases=[{"model_type": "qwen3_5", "layers": 0}]),
    # CI scripts are the decision layer itself: CPU goes wide, with one tiny E2E smoke.
    dict(name="ci script", files=["scripts/ci/lint_select/select_tests.py"], expect="all",
         cpu=True, npu=False, e2e=True, e2e_models=["bailing_hybrid"],
         full_reason="ci-config-changed"),
    dict(name="workflow config", files=[".github/workflows/ci-pr.yml"], expect="all",
         cpu=True, npu=False, e2e=True, e2e_models=["bailing_hybrid"],
         full_reason="ci-config-changed"),
    dict(name="shared env deps", files=["scripts/ci/envs/common.txt"], expect="all",
         cpu=True, npu=False, e2e=True, e2e_models=["bailing_hybrid"],
         full_reason="ci-config-changed"),
    dict(name="E2E map", files=[".github/workflows/model_map.json"], expect="all",
         cpu=True, npu=False, e2e=True, e2e_models=models_with_trigger("pr"),
         full_reason="ci-config-changed"),
    # Dependency change: nothing can be trusted to install.
    dict(name="pyproject", files=["pyproject.toml"], expect="all",
         cpu=True, npu=True, e2e=True, full_reason="deps-changed"),
    # Image recipe: NPU unit runs the whole lane (empty list == whole lane).
    dict(name="docker recipe", files=["docker/npu/Dockerfile_aarch64"], expect="precise",
         cpu=False, npu=True, e2e=True, npu_scope="all"),
    # ci:skip-npu closes every NPU lane, E2E included.
    dict(name="label ci:skip-npu", files=["src/torchturbo/engine/core.py"],
         labels="ci:skip-npu", expect="precise", cpu=True, npu=False, e2e=False),
    dict(name="label ci:full", files=["src/torchturbo/engine/core.py"],
         labels="ci:full", expect="all", cpu=True, npu=True, e2e=True, e2e_all=True),
]


def assert_scenario(case: dict) -> None:
    name, files = case["name"], case["files"]
    missing = [f for f in files if not (ROOT / f).exists()]
    if not check(f"{name}: paths exist", not missing, str(missing)):
        return

    got = select(files, labels=case.get("labels", ""), mode="pr")
    if not check(f"{name}: selector ran", got["_rc"] == 0, tail(got["_stderr"])):
        return

    check(f"{name}: result mode={case['expect']}", got.get("mode") == case["expect"],
          f"got {got.get('mode')}")
    check(f"{name}: cpu_unit_run={case['cpu']}", flag(got, "cpu_unit") == case["cpu"],
          f"got {got.get('cpu_unit_run')}")
    check(f"{name}: npu_unit_run={case['npu']}", flag(got, "npu_unit") == case["npu"],
          f"got {got.get('npu_unit_run')}")
    check(f"{name}: npu_e2e_run={case['e2e']}", flag(got, "npu_e2e") == case["e2e"],
          f"got {got.get('npu_e2e_run')}")

    if "cpu_files" in case:
        check(f"{name}: CPU file list", listed(got, "cpu_unit") == case["cpu_files"],
              f"got {listed(got, 'cpu_unit')}")
    if "npu_files" in case:
        check(f"{name}: NPU file list", listed(got, "npu_unit") == case["npu_files"],
              f"got {listed(got, 'npu_unit')}")
    check(f"{name}: counts agree with flags",
          bool(listed(got, "cpu_unit")) == flag(got, "cpu_unit"),
          "cpu run flag and file list disagree")

    # Empty NPU list with run=true means "the whole lane", never "nothing".
    if case.get("npu_scope") == "all":
        check(f"{name}: NPU whole lane", listed(got, "npu_unit") == [] and flag(got, "npu_unit"),
              f"got {len(listed(got, 'npu_unit'))} file(s)")

    if case.get("full_reason"):
        check(f"{name}: full reason {case['full_reason']}",
              case["full_reason"] in got.get("_stderr", "") or
              case["full_reason"] in explain(files, case),
              "reason not reported")

    picked = cases(got)
    if case.get("e2e_all"):
        expected = cases_with_trigger("nightly")
        check(f"{name}: every nightly case ({expected})",
              len(picked) == expected, f"got {len(picked)}")
    elif "e2e_count" in case:
        check(f"{name}: e2e count={case['e2e_count']}", len(picked) == case["e2e_count"],
              f"got {[c['model_type'] for c in picked]}")
    if "e2e_models" in case:
        check(f"{name}: e2e models {case['e2e_models']}",
              [c["model_type"] for c in picked] == case["e2e_models"],
              f"got {[c['model_type'] for c in picked]}")
    if "e2e_cases" in case:
        got_pairs = [{"model_type": c["model_type"], "layers": c["layers"]} for c in picked]
        check(f"{name}: e2e cases", got_pairs == case["e2e_cases"], f"got {got_pairs}")

    check_markers(name, listed(got, "cpu_unit"), CPU_MARK, "cpu")
    check_markers(name, listed(got, "npu_unit"), NPU_MARK, "npu")


def explain(files: list[str], case: dict) -> str:
    """Stdout of --explain, where `full run because:` is printed."""
    cmd = [sys.executable, str(SELECTOR), "--explain", "--output", os.devnull,
           "--mode", "pr", "--labels", case.get("labels", "")]
    for path in files:
        cmd += ["--changed-file", path]
    return run(cmd).stdout


def check_markers(scenario: str, files: list[str], mark: re.Pattern, lane: str) -> None:
    """A selected file must exist and declare the marker of the lane it is in."""
    absent = [f for f in files if not (ROOT / f).exists()]
    if not check(f"{scenario}: {lane} files exist", not absent, str(absent[:3])):
        return
    wrong = [f for f in files
             if not mark.search((ROOT / f).read_text(encoding="utf-8", errors="replace"))]
    check(f"{scenario}: {lane} marker agrees", not wrong, str(wrong[:3]))


def step_scenarios(only: str) -> None:
    print("\n[3/5] scenario matrix")
    for case in SCENARIOS:
        if only and only not in case["name"]:
            continue
        print(f"\n  -- {case['name']}")
        assert_scenario(case)


def step_contract() -> None:
    print("\n[4/5] selector <-> workflow contract")
    text = (WORKFLOWS / "ci-pr.yml").read_text(encoding="utf-8")
    block = job_block(text, "select")
    declared = set(re.findall(r"^      ([a-z0-9_]+): \$\{\{ steps\.", block, re.M))
    # Outputs the selector itself must fill; `image` and `failed_step` come from
    # other steps in the same job, so they are counted separately.
    from_selector = set(re.findall(r"^      ([a-z0-9_]+): \$\{\{ steps\.select\.", block, re.M))
    used = set(re.findall(r"needs\.select\.outputs\.([a-z0-9_]+)", text))
    produced = set(select(["pyproject.toml"])) - {"_rc", "_stderr", "_summary"}

    check("select declares its outputs", bool(declared), "no outputs block found")
    check("every selector output is produced", from_selector <= produced,
          str(sorted(from_selector - produced)))
    check("every consumed output is declared", used <= declared, str(sorted(used - declared)))
    check("every device matrix is declared",
          {"cpu_jobs", "cpu_job_count", "npu_jobs", "npu_job_count"}
          <= declared, str(sorted(declared)))
    check("lint shares the select job", "lint_check.py" in block)
    check("lint runs before the planner",
          block.find("lint_check.py") < block.find("select_tests.py")
          if "lint_check.py" in block and "select_tests.py" in block else False)
    check("selector self-test is owned by lint_check",
          "select_tests.py --verify" not in block)


def job_block(text: str, job: str) -> str:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line == f"  {job}:"), None)
    if start is None:
        return ""
    end = next((i for i in range(start + 1, len(lines))
                if re.match(r"^  [A-Za-z0-9_-]+:\s*$", lines[i])), len(lines))
    return "\n".join(lines[start:end])


def step_summary() -> None:
    print("\n[5/5] plan summary")
    image = "swr.example/probe:tag-1"
    got = select(["src/torchturbo/engine/core.py"], image=image)
    rows = [line for line in got["_summary"].splitlines() if line.startswith("| image")]
    check("summary renders the image", rows == [f"| image | `{image}` |"], str(rows))
    check("summary has a plan table", "## Plan" in got["_summary"])
    check("summary lists lanes in order",
          re.search(r"\| lanes \(cpu_unit/npu_unit/npu_e2e\) \| "
                    r"`true / true / true` \|",
                    got["_summary"]) is not None,
          "lane row missing or mislabelled")


def main(argv: list[str]) -> int:
    global VERBOSE
    ap = argparse.ArgumentParser(description="Self-test the CI lint + select layer")
    ap.add_argument("--verbose", action="store_true", help="also print child stderr")
    ap.add_argument("--only", default="", help="run only scenarios whose name contains this")
    args = ap.parse_args(argv)
    VERBOSE = args.verbose

    print(f"repo: {ROOT}")
    step_lint()
    step_verify()
    step_scenarios(args.only)
    if not args.only:
        step_contract()
        step_summary()

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)})")
        for item in FAILS:
            print(f"  - {item}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
