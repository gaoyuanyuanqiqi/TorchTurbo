#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""卡数 -> sp/ep，写出派生的 engine config。

回答的问题是「这次训练用几路切分」。仓库里那份 yaml 保持原样，派生副本写到
`--out`，这样同一份配置能被不同卡数复用而不会互相覆盖。

两件必须一起做的事（所以它们在一个文件里）：

* 缩小 sp/ep：当 cards 不能整除时，只在仓库配置上硬改会在建进程组时崩，
  所以要按 cards 收敛到能整除的值。
* 裁层后关 MTP：`fetch_model.py trim` 会删掉第 `num_hidden_layers` 层，
  而 MTP 正占这个索引；加载是 strict=False，不关就会静默留一个随机头。

输出 `SP=` / `EP=` / `TT_CONFIG=`，可被 `source`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def log(msg: str) -> None:
    print(f"[plan] {msg}", file=sys.stderr, flush=True)


def narrow(sp: int, ep: int, world: int) -> tuple[int, int]:
    """Shrink sp/ep until both divide ``world`` and ep fits inside one replica."""
    sp = max(1, int(sp))
    ep = max(1, int(ep))

    while sp > 1 and world % sp != 0:
        sp //= 2
    max_ep = max(1, world // sp)
    if ep > max_ep:
        ep = max_ep
    while ep > 1 and world % ep != 0:
        ep //= 2
    return sp, ep


def derive(
    *,
    source: Path | None,
    sp_in: str,
    ep_in: str,
    world: int,
    layers: int,
    out: Path,
) -> tuple[int, int, int, int]:
    """Write the derived config and return (old_sp, old_ep, sp, ep)."""
    doc: dict = {}
    if source is not None and source.is_file():
        doc = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    engine = doc.setdefault("engine_config", {}) or {}
    plan = engine.setdefault("parallel_plan", {}) or {}

    old_sp = int(plan.get("sp_size", 1) or 1)
    old_ep = int(plan.get("ep_size", 1) or 1)

    sp = old_sp if sp_in == "auto" else int(sp_in)
    ep = old_ep if ep_in == "auto" else int(ep_in)
    sp, ep = narrow(sp, ep, world)

    plan["sp_size"], plan["ep_size"] = sp, ep

    # See the module docstring: after trimming, the MTP layer's weights are gone
    # and the loader would keep a random head instead of failing.
    if layers > 0:
        mtp = engine.setdefault("mtp", {}) or {}
        mtp["enabled"] = False
        engine["mtp"] = mtp

    doc["engine_config"] = engine
    out.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return old_sp, old_ep, sp, ep


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="卡数 -> sp/ep")
    ap.add_argument("--output", default="", help="KEY=value 写到这个文件（默认 stdout）")
    ap.add_argument("--yaml", default="", help="source engine config; may be empty")
    ap.add_argument("--cards", type=int, required=True)
    ap.add_argument("--sp", default="auto")
    ap.add_argument("--ep", default="auto")
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    source = Path(args.yaml) if args.yaml else None
    out = Path(args.out)
    old_sp, old_ep, sp, ep = derive(
        source=source, sp_in=str(args.sp), ep_in=str(args.ep),
        world=args.cards, layers=args.layers, out=out,
    )
    log(
        f"parallel: sp={old_sp} ep={old_ep} -> sp={sp} ep={ep} "
        f"(cards={args.cards}, source={args.yaml or '<none>'})"
    )
    with open(args.output or "/dev/stdout", "a", encoding="utf-8") as handle:
        handle.write(f"SP={sp}\nEP={ep}\nTT_CONFIG={out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
