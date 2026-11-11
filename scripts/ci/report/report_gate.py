#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""门禁判定与计时 —— PR 与 nightly 共用一份。

两边的规则本来几乎一样，但抄成了两段 bash，于是「fork PR 的 L2 算不算合法跳过」
只在一个文件里修对过。收进这里之后两边不可能再分叉。

    report_gate.py timing   排队 / 执行 / 端到端，写进 Job Summary（不参与判定）
    report_gate.py judge    逐泳道判定，退出码即门禁结论
    report_gate.py failed-step  把 select job 各 step 的 outcome 折叠成一个词

判定只用两条规则：

    required  跑了就必须成功。skipped 说明上游已经挂了 —— **包括 select 自己失败**。
              这一条是整套设计的核心：让「慢慢把 L2 关掉」必须显式改代码，
              而不是某个配置漂了之后无人察觉。
    optional  跳过可以，但必须是因为「本来就该跳过」。原因不对就是
              skipped-unknown，判红。

fork PR 单独处理：secrets 不递给 fork，L2 必然起不来。那是设计，不是故障，
所以认成合法跳过并写明理由（禁止 pull_request_target：secrets + fork 代码 = RCE）。

退出码：0 通过，1 未通过。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

OK, SKIP, BAD = "✅ 通过", "⏭ 跳过", "❌"


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _has_label(tag: str) -> bool:
    """标签判定只在这里（与 lint_select/select_tests.py 的 flags_from_event 分工相同：
    选什么归它，算不算过归这里）。"""
    tags = {t.strip() for t in _env("LABELS").split(",") if t.strip()}
    return tag in tags


def _skip_why() -> str:
    """Explain a skipped lane from the failed step inside the select job."""
    return {
        "lint": "静态检查未通过，按设计未启动",
        "image": "镜像解析未通过，按设计未启动",
        "select": "计划未算出，按设计未启动",
        "cancelled": "run 被取消，按设计未启动",
    }.get(_env("SEL_FAILED_STEP"), "计划未算出，按设计未启动")


# select job 里各 step 的 outcome -> 失败阶段。顺序 = step 执行顺序：早挂的那个是根因，
# 后面的失败通常只是它的连带。这里的词必须与 _skip_why() 的键对得上，且两者同处一
# 文件——分开写的话，改了一边另一边只会默默说不出原因。
FAILED_STEPS = (
    ("lint", ("LINT_OUTCOME",)),
    ("image", ("IMAGE_OUTCOME",)),
    ("select", ("SELECT_OUTCOME",)),
)


def cmd_failed_step() -> int:
    """把 select job 各 step 的 outcome 折叠成一个词，写进 $GITHUB_OUTPUT。

    GitHub 只暴露整个 job 的 `needs.select.result`，所以「哪一步挂了」必须由这个
    job 自己发布，否则 judge 只能给一个模糊结论。
    """
    if _env("JOB_STATUS") == "cancelled":
        failed = "cancelled"
    else:
        failed = next((name for name, keys in FAILED_STEPS
                       if any(_env(k) and _env(k) != "success" for k in keys)), "none")
    with open(_env("GITHUB_OUTPUT") or os.devnull, "a", encoding="utf-8") as fh:
        fh.write(f"failed_step={failed}\n")
    print(f"failed_step: {failed}")
    return 0


def _summary(markdown: str) -> None:
    path = _env("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(markdown)
    else:
        print(markdown)


# ---------------------------------------------------------------- timing
def timing() -> int:
    """排队才是隐形杀手：执行 8 min 但排队 40 min，作者感受是 48 min。
    所以 queue 与 wall 分开报，且都不参与判定 —— 它们是扩容证据，不是失败。"""
    repo, run_id = _env("GITHUB_REPOSITORY"), _env("GITHUB_RUN_ID")
    if not (repo and run_id):
        print("::warning::缺 GITHUB_REPOSITORY / GITHUB_RUN_ID，跳过计时")
        return 0

    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}",
         "--jq", r'"\(.created_at) \(.run_started_at)"'],
        capture_output=True, text=True,
    )
    stamps = proc.stdout.strip()
    if proc.returncode != 0 or not stamps:
        print("::warning::拿不到 run 时间戳（token 缺 actions:read？）")
        return 0

    def parse(text: str) -> datetime:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))

    created_s, started_s = stamps.split()
    created, started = parse(created_s), parse(started_s)
    now = datetime.now(timezone.utc)
    _summary(
        "\n| 计时 | 秒 |\n|---|---:|\n"
        f"| queue（排队） | {int((started - created).total_seconds())} |\n"
        f"| wall（执行） | {int((now - started).total_seconds())} |\n"
        f"| 端到端 | {int((now - created).total_seconds())} |\n"
    )
    return 0


# ---------------------------------------------------------------- judge
# One row per lane: (summary label, env suffix, why it was not selected).
# A new device adds rows here; nothing below repeats the lane list.
LANES = (
    ("CPU-unit", "CPU_UNIT", "未选中 CPU 测试"),
    ("NPU-unit", "NPU_UNIT", "未选中需要真卡的测试文件"),
    ("NPU-e2e", "NPU_E2E", "无匹配 E2E 用例"),
)
LANE_KEYS = tuple(key for _, key, _ in LANES)
# The lane that runs training cases; `SKIP_REASON` is published by it alone.
E2E_LANE = "NPU_E2E"


class Report:
    def __init__(self) -> None:
        self.rows: list[str] = []
        self.failed = False

    def required(self, name: str, result: str) -> None:
        if result == "success":
            self.rows.append(f"| {name} | {OK} |  |")
        elif result == "skipped":
            self.rows.append(f"| {name} | {BAD} 未执行（预期应执行） |  |")
            self.failed = True
        else:
            self.rows.append(f"| {name} | {BAD} {result} |  |")
            self.failed = True

    def optional(self, name: str, result: str, *, expect_skip: bool, why: str) -> None:
        if result == "success":
            self.rows.append(f"| {name} | {OK} |  |")
        elif result == "skipped" and expect_skip:
            self.rows.append(f"| {name} | {SKIP} | {why} |")
        elif result == "skipped":
            # 「不知道为什么没跑」比「跑失败了」更危险：它会安静地变成常态。
            self.rows.append(f"| {name} | {BAD} skipped-unknown |  |")
            self.failed = True
        else:
            self.rows.append(f"| {name} | {BAD} {result} |  |")
            self.failed = True

    def note(self, name: str, value: str) -> None:
        self.rows.append(f"| {name} | {value} |  |")

    def render(self, title: str, summary: str) -> None:
        _summary(
            f"## {title}\n\n| 泳道 | 结果 | 备注 |\n|---|---|---|\n"
            + "\n".join(self.rows)
            + f"\n\n选择：`{summary or 'unknown'}`\n"
        )


def _lane_why(key: str, fallback: str, sel_bad: bool) -> str:
    """Why a lane did not run: plan failure, an override label, or no match."""
    if sel_bad:
        return _skip_why()
    # Lanes are named <DEVICE>_<depth>, so the device comes from the key itself.
    device = key.split("_")[0].lower()
    if _has_label(f"ci:skip-{device}"):
        return f"ci:skip-{device} 标签，按设计未启动"
    return fallback


def judge_pr() -> int:
    rep = Report()
    # lint 与 select 共用一个 job：只看 R_SEL 分不清倒在哪一步，所以用
    # failed_step 把 job 结果还原成「lint 过没过」，再分别判定。
    results = {k: _env(f"R_{k}") for k in ("SEL",) + LANE_KEYS}
    results["LINT"] = "failure" if _env("SEL_FAILED_STEP") == "lint" else "success"
    # 决策层给的开关：「这个泳道本次该不该跑」。
    # 拿不到（旧调用方式 / 手敲测试）时按 true 处理，也就是宁可错判红。
    want = {k: _env(f"{k}_RUN", "true") for k in LANE_KEYS}

    rep.required("lint", results["LINT"])

    # 静态检查是整条链的闸：它没过就不算计划（select 没跑完），后面每条泳道
    # 因此一定 skipped。那是「按设计没跑」，不是「不知道为什么没跑」——
    # 后者才需要报警，前者报警只会把真正的原因埋掉。
    lint_bad = results["LINT"] != "success"

    if lint_bad:
        rep.rows.append(f"| select | {SKIP} | lint 未通过，按设计未启动 |")
        for name, _, _ in LANES:
            rep.rows.append(f"| {name} | {SKIP} | lint 未通过，按设计未启动 |")
    else:
        sel_bad = results["SEL"] != "success"

        # 每条泳道独立判定：并行拓扑下「上游红了」不再是跳过的理由，
        # 跳过只可能是「自己没被选中」或「计划没算出来」。
        if _env("IS_FORK") == "true":
            # fork PR 的 NPU 泳道不执行是设计，不是故障。
            if results["CPU_UNIT"] == "skipped" and want["CPU_UNIT"] == "false":
                rep.rows.append(f"| CPU-unit | {SKIP} | 未选中 CPU 测试 |")
            else:
                rep.required("CPU-unit", results["CPU_UNIT"])
            for name, key, _ in LANES[1:]:
                res = results[key]
                if res == "skipped":
                    rep.rows.append(f"| {name} | {SKIP} | fork PR（secrets 不递） |")
                elif res == "success":
                    rep.rows.append(f"| {name} | {OK} |  |")
                else:
                    rep.rows.append(f"| {name} | {BAD} {res} |  |")
                    rep.failed = True
        else:
            for name, key, why in LANES:
                rep.optional(
                    name, results[key],
                    expect_skip=(want[key] != "true") or sel_bad,
                    why=_lane_why(key, why, sel_bad),
                )

    # 跑起来了但自己宣告跳过（模型获取超时）：记 skipped-reason，
    # 默认判绿，只有 ci:e2e:strict 才判红。
    skip_reason = _env("SKIP_REASON")
    if skip_reason and results[E2E_LANE] == "success":
        rep.note("npu-e2e 跳过原因", f"⚠️ {skip_reason}")
        if _has_label("ci:e2e:strict"):
            rep.failed = True

    # select 是计划本身：没算出来就没有「跑不跑」可言，直接判红。
    # 静态检查红时它 skipped 是设计内，已经在上面记过，不重复。
    if not lint_bad:
        sel = results["SEL"]
        if sel == "success":
            rep.note("select", OK)
        else:
            rep.note("select", f"{BAD} {sel} - {_skip_why()}")
            rep.failed = True

    rep.render("CI 结果", _env("SUMMARY"))
    # Lint is the real failure when select never ran; a select-failed fallback
    # would point people at the wrong stage.
    fallback = "lint-failed" if lint_bad else "select-failed"
    print(f"::notice title=ci::{_env('SUMMARY') or fallback}")
    if rep.failed:
        print("::error::门禁未通过")
        return 1
    print("门禁通过")
    return 0


def judge_full() -> int:
    rep = Report()
    req = {k: _env(f"R_{k}") for k in ("SEL",) + LANE_KEYS}
    req["LINT"] = "failure" if _env("SEL_FAILED_STEP") == "lint" else "success"
    want = {k: _env(f"{k}_RUN", "true") for k in LANE_KEYS}

    rep.required("lint", req["LINT"])

    # lint 红就不算计划：后面全 skipped 是设计内，一句话说清楚即可。
    lint_bad = req["LINT"] != "success"
    if lint_bad:
        rep.rows.append(f"| select | {SKIP} | lint 未通过，按设计未启动 |")
        for name, _, _ in LANES:
            rep.rows.append(f"| {name} | {SKIP} | lint 未通过，按设计未启动 |")
        rep.render("main 全量结果", _env("SUMMARY"))
        print("::error::全量未通过")
        return 1

    rep.required("select", req["SEL"])

    # 全量模式下每条泳道都应该跑（force-all）；但保留 lane 判断，
    # 万一将来全量也做了裁剪，这里不用再改。
    sel_bad = req["SEL"] != "success"
    for name, key, why in LANES:
        rep.optional(
            name, req[key],
            expect_skip=(want[key] != "true") or sel_bad,
            why=_lane_why(key, why, sel_bad),
        )

    rep.render("main 全量结果", _env("SUMMARY"))
    print(f"::notice title=ci-full::{_env('SUMMARY') or 'select-failed'}")
    if rep.failed:
        print("::error::全量未通过")
        return 1
    print("全量通过")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="门禁判定与计时")
    ap.add_argument("command", choices=["timing", "judge", "failed-step"])
    ap.add_argument("--mode", choices=["pr", "full"], default="pr")
    args = ap.parse_args(argv)

    if args.command == "failed-step":
        return cmd_failed_step()
    if args.command == "timing":
        return timing()
    return judge_pr() if args.mode == "pr" else judge_full()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
