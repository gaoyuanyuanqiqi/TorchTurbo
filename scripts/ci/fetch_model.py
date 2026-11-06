#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型获取：这个目录能不能用、从哪儿找、拿一份来。

原先是三个文件（`model_cache.sh` + `hf_fetch.py` + `flash_trim.py`），但它们
回答的是同一个问题 —— **「给我一个可用的模型目录」**。拆开之后 run_train.sh 为了
一件事要认三个文件名、三种参数风格（bash 长选项 / python 长选项），而 probe
与 fetch 之间「什么算完整」的判据还得两边各写一遍。

    probe   这个目录完整吗（只看 config.json + index 承诺的分片 + 真实字节）
    find    在平台用过的所有位置里找一个完整的（含 marker 校验）
    mark    记下这个目录是按什么配方建的
    fetch   下载整个仓库（带重试）
    trim    只下前 N 层需要的分片，再裁剪 index 与 config

卡数 -> sp/ep 是另一个问题，在 `plan_parallel.py` 里。

约定：人看的日志走 stderr，机器读的 `KEY=value` 走 `--output`（可 source）。

几个踩过的坑，都写在这里而不是散在调用方：

* 目录大小说明不了什么（实测 inode size 22~499 字节），hub 风格的
  `snapshots/<sha>/` 全是符号链接，所以必须 `du -sbL`。
* index 是完整性的唯一权威：它列出的分片缺一个，说明上次下载被杀，
  此时「目录看起来有 30 GB」照样会让 CheckpointInfo 在很久以后报
  FileNotFoundError，看起来和下载毫无关系。
* 按层裁剪要改三处（下哪些分片 / 裁 index / 裁 config），漏一处就在不同的
  地方失败，所以它们必须在一个文件里。
* 裁剪之后 MTP 必须关掉而不是重编号：MTP 占 `layers[num_hidden_layers]`，
  裁完那个索引会被别的层占走，而加载是 strict=False，会静默留一个随机初始
  化的头。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

from pathlib import Path

MIN_BYTES = 3_000_000_000
MARKER_NAME = ".tt-cache-marker"
INDEX = "model.safetensors.index.json"

# 加载器决定读哪些分片之前必须有的东西：config、index、tokenizer。
# `*.py` 覆盖 trust_remote_code 的模型。
SMALL_FILES = [
    "config.json",
    INDEX,
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer*.json",
    "*.py",
]


def log(msg: str) -> None:
    print(f"[model] {msg}", file=sys.stderr, flush=True)


def out(**kv: object) -> None:
    path = os.environ.get("MODEL_OUT")
    line = "".join(f"{k}={v}\n" for k, v in kv.items())
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    else:
        sys.stdout.write(line)
        sys.stdout.flush()


# ---------------------------------------------------------------- probe
def _dir_bytes(path: Path) -> int:
    """`du -sbL`：-L 是必须的，否则统计的是符号链接本身而不是文件。"""
    proc = subprocess.run(["du", "-sbL", str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.split()[0])
    except (IndexError, ValueError):
        return 0


def probe(directory: str, min_bytes: int = MIN_BYTES) -> tuple[int, int] | None:
    """完整就返回 (bytes, shards)，否则 None 并说明原因。"""
    root = Path(directory)
    if not (root / "config.json").is_file():
        log(f"probe {root}: no config.json")
        return None

    total = _dir_bytes(root)
    if total < min_bytes:
        log(f"probe {root}: {total} bytes < {min_bytes} (incomplete)")
        return None

    shards = len(list(root.glob("*.safetensors"))) + len(
        list(root.glob("*/*.safetensors"))
    )
    if shards < 1:
        log(f"probe {root}: no *.safetensors under it")
        return None

    index = root / INDEX
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, ValueError, KeyError) as exc:
            log(f"probe {root}: index unreadable: {type(exc).__name__}: {exc}")
            return None
        missing = sorted(
            {s for s in weight_map.values() if not (root / s).is_file()}
        )
        if missing:
            log(f"probe {root}: index lists {len(missing)} shard(s) not on disk:")
            for name in missing[:5]:
                log(f"  {name}")
            return None

    return total, shards


# ---------------------------------------------------------------- find
def cmd_find(repo: str, hint: str, home: str, min_bytes: int, marker: str) -> int:
    """按「从最明确到最隐式」的顺序找，第一个完整的算命中。

    marker 不一致说明目录是按别的配方建的（层数不同 / 仓库不同），
    继续用会加载错形状的模型，所以即使 probe 通过也当没命中。
    """
    escaped = repo.replace(".", "___")      # modelscope 的转义习惯

    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint))
    candidates += [
        Path(f"/root/.cache/models/{repo}"),
        Path(f"/root/.cache/models/{escaped}"),
        Path(f"/root/.cache/modelscope/hub/models/{repo}"),
        Path(f"/root/.cache/modelscope/hub/models/{escaped}"),
        Path(f"/root/.cache/modelscope/hub/{repo}"),
    ]
    candidates += sorted(Path(home).glob(f"hub/models--{repo.replace('/', '--')}/snapshots/*"))

    for cand in candidates:
        if not cand.is_dir():
            continue
        if marker:
            stamp = cand / MARKER_NAME
            if not stamp.is_file():
                log(f"stale {cand} (no marker; expected '{marker}')")
                continue
            if stamp.read_text(encoding="utf-8").strip() != marker:
                log(f"stale {cand} (marker: {stamp.read_text(encoding='utf-8').strip()})")
                continue
        got = probe(str(cand), min_bytes)
        if got is None:
            log(f"miss {cand}")
            continue
        log(f"HIT  {cand}  bytes={got[0]} shards={got[1]}")
        out(MODEL=cand, BYTES=got[0], SHARDS=got[1], HIT="yes")
        return 0

    log(f"no usable cache for {repo}")
    out(HIT="no")
    return 0


# ---------------------------------------------------------------- mark
def cmd_mark(directory: str, text: str) -> int:
    (Path(directory) / MARKER_NAME).write_text(text + "\n", encoding="utf-8")
    log(f"marked {directory} as '{text}'")
    return 0


# ---------------------------------------------------------------- fetch
def _snapshot_download(
    repo: str, directory: str, patterns: list[str] | None, attempts: int, backoff: int
) -> str:
    """`snapshot_download` 直到成功或次数用完。

    huggingface_hub 不重试大部分失败：`http_stream_backoff` 的
    `retry_on_exceptions=()`、`retry_on_status_codes=(408, 429)`，而 hf-mirror
    常返回 521（源站不可达）。一次 20 分钟 NPU 时间的 job 不值得赌一次尝试。
    """
    from huggingface_hub import snapshot_download

    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        print(f"--- download attempt {attempt}/{attempts} ---", flush=True)
        try:
            path = snapshot_download(
                repo_id=repo, local_dir=directory, allow_patterns=patterns
            )
        except Exception as exc:                 # noqa: BLE001 - 任何失败都值得再试
            last = exc
            print(f"attempt {attempt} failed: {type(exc).__name__}: {exc}", flush=True)
            if attempt < attempts:
                wait = attempt * backoff
                print(f"retrying in {wait}s", flush=True)
                time.sleep(wait)
            continue
        print(f"downloaded {repo} -> {path}", flush=True)
        return path
    raise SystemExit(f"download {repo} failed after {attempts} attempts") from last


def cmd_fetch(repo: str, directory: str, patterns: str, attempts: int, backoff: int) -> int:
    Path(directory).mkdir(parents=True, exist_ok=True)
    allow = [p.strip() for p in patterns.split(",") if p.strip()] or None
    _snapshot_download(repo, directory, allow, attempts, backoff)
    return 0


# ---------------------------------------------------------------- trim
def _layer_index(tensor_name: str) -> int | None:
    """`model.layers.7.mlp...` → 7；其余 → None。

    只算 model.layers 的直接子项。MTP 层也是这个列表的子项，所以它会跟着
    被裁掉 —— 这正是 MTP 必须关掉而不是挪位置的原因。
    """
    parts = tensor_name.split(".")
    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
        if parts[2].isdigit():
            return int(parts[2])
    return None


def plan_shards(weight_map: dict[str, str], layers: int) -> set[str]:
    """含第 <layers 层（或不属于任何层）的张量的分片。"""
    need: set[str] = set()
    for name, shard in weight_map.items():
        index = _layer_index(name)
        if index is not None and index >= layers:
            continue
        need.add(shard)
    return need


def cmd_trim(repo: str, directory: str, layers: int, attempts: int) -> int:
    if layers < 1:
        return int(bool(print("::error::--layers 必须 >= 1；要全量请用 fetch", file=sys.stderr)))

    dst = Path(directory)
    dst.mkdir(parents=True, exist_ok=True)
    _snapshot_download(repo, str(dst), SMALL_FILES, attempts, 30)

    index_path = dst / INDEX
    if not index_path.is_file():
        raise SystemExit(
            f"{index_path} 不存在；这个模型没有 safetensors index，无法按层裁剪"
        )
    doc = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = doc["weight_map"]
    need = plan_shards(weight_map, layers)
    print(f"{layers} layers -> need {len(need)}/{len(set(weight_map.values()))} shards")

    _snapshot_download(repo, str(dst), sorted(need), attempts, 30)

    doc["weight_map"] = {k: v for k, v in weight_map.items() if v in need}
    doc.setdefault("metadata", {})["tt_trimmed_layers"] = layers
    index_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"index: kept {len(doc['weight_map'])}/{len(weight_map)} entries")

    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["num_hidden_layers"] = layers
    cfg["num_nextn_predict_layers"] = 0
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"config: num_hidden_layers={layers} num_nextn_predict_layers=0")
    return 0


# ---------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="模型获取")
    # `--output` 两边都能写：习惯上会写在子命令后面，但也可以放最前面。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output", default="", help="KEY=value 写到这个文件（默认 stdout）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", parents=[common])
    p.add_argument("--dir", required=True)
    p.add_argument("--min-bytes", type=int, default=MIN_BYTES)

    f = sub.add_parser("find", parents=[common])
    f.add_argument("--repo", required=True)
    f.add_argument("--hint", default="")
    f.add_argument("--home", default=os.environ.get("HF_HOME", "/root/.cache/huggingface"))
    f.add_argument("--min-bytes", type=int, default=MIN_BYTES)
    f.add_argument("--marker", default="")

    m = sub.add_parser("mark", parents=[common])
    m.add_argument("--dir", required=True)
    m.add_argument("--text", required=True)

    g = sub.add_parser("fetch", parents=[common])
    g.add_argument("--repo", required=True)
    g.add_argument("--dir", required=True)
    g.add_argument("--patterns", default="", help="逗号分隔；空 = 整个仓库")
    g.add_argument("--attempts", type=int, default=3)
    g.add_argument("--backoff", type=int, default=30)

    t = sub.add_parser("trim", parents=[common])
    t.add_argument("--repo", required=True)
    t.add_argument("--dir", required=True)
    t.add_argument("--layers", type=int, required=True)
    t.add_argument("--attempts", type=int, default=3)

    ap.add_argument("--output", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    # 子命令上的优先（它覆盖了 dest 同名属性）。
    if getattr(args, "output", ""):
        os.environ["MODEL_OUT"] = args.output

    if args.cmd == "probe":
        got = probe(args.dir, args.min_bytes)
        if got is None:
            return 1
        print(f"{got[0]} {got[1]}")
        return 0
    if args.cmd == "find":
        return cmd_find(args.repo, args.hint, args.home, args.min_bytes, args.marker)
    if args.cmd == "mark":
        return cmd_mark(args.dir, args.text)
    if args.cmd == "fetch":
        return cmd_fetch(args.repo, args.dir, args.patterns, args.attempts, args.backoff)
    if args.cmd == "trim":
        return cmd_trim(args.repo, args.dir, args.layers, args.attempts)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
