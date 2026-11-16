#!/usr/bin/env python3
"""Accumulate model-E2E facts in one JSON file and render its job summary.

The summary used to be assembled from ``steps.<id>.outputs``, which is empty
whenever a step fails -- so the run that most needed the numbers printed

    ##[notice]sfs=? cache_hit=? shards=? size=? train_s=n/a train_rc=n/a

because step ⑤ exited non-zero and the outputs were never published. Writing to a
file instead means every step contributes whatever it learned before dying, and the
summary is complete on the failing path too, which is the only path anyone reads.

Subcommands:

    init    --file F key=value ...        create/overwrite with static facts
    set     --file F key=value ...        merge facts (creates the file if needed)
    phase   --file F --name N --seconds S --rc R
    report  --file F [--train-log ...]    render Markdown on stdout

Values are stored as strings; ``phase`` keeps an ordered list so the table reads in
execution order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KIND_TRAIN = "train"

def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pairs(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            print(f"[result] ignoring '{item}': not key=value", file=sys.stderr)
            continue
        out[key] = value
    return out


def _fmt_seconds(value) -> str:
    try:
        seconds = int(float(value))
    except Exception:
        return "n/a"
    return f"{seconds // 60} min {seconds % 60:02d} s"


def _tail(path: Path, lines: int) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def _getter(doc: dict):
    def get(key: str, default: str = "?") -> str:
        value = doc.get(key, "")
        return value if value not in ("", None) else default

    return get


def _phase_block(phases: list[dict]) -> tuple[list[str], int]:
    total = 0
    for phase in phases:
        try:
            total += int(float(phase.get("seconds", 0)))
        except Exception:
            pass

    lines = ["", "| phase | seconds | human | rc |", "|---|---|---|---|"]
    for phase in phases:
        lines.append(
            f"| {phase.get('name', '?')} | `{phase.get('seconds', 'n/a')}` | "
            f"{_fmt_seconds(phase.get('seconds'))} | `{phase.get('rc', 'n/a')}` |"
        )
    lines.append(f"| **total** | `{total}` | **{_fmt_seconds(total)}** | |")
    return lines, total


def _render_train(doc: dict, *, log: Path, versions: Path) -> str:
    get = _getter(doc)
    phases = doc.get("phases", [])

    lines = [
        f"## {get('model_type')} / {get('framework')} on `{get('runner_label')}`",
        "",
        "| item | value |",
        "|---|---|",
        f"| image | `{get('image')}` |",
        f"| run_dir | `{get('run_dir')}` |",
        f"| install / train | `{get('install_script')}` / `{get('train_script')}` |",
        f"| model cache | **{get('cache_hit')}** (sfs `{get('sfs_uuid')}`) |",
        f"| layers | `{get('layers', '0')}` (0 = 不裁剪) |",
        f"| dataset | `{get('dataset')}` |",
        f"| shards / size | {get('shards')} / {get('model_size')} |",
        f"| parallel (sp/ep) | `{get('sp')}/{get('ep')}` |",
        f"| output_dir | `{get('output_dir')}` |",
    ]
    phase_lines, _ = _phase_block(phases)
    lines += phase_lines
    lines += [
        "",
        "### Versions after install",
        "",
        "```",
        _tail(versions, 40) or "(probe failed)",
        "```",
        "",
        "### 训练尾部",
        "",
        "```",
        _tail(log, 40) or "(no train log)",
        "```",
    ]
    return "\n".join(lines) + "\n"


def render(doc: dict, *, log: Path, versions: Path) -> str:
    return _render_train(doc, log=log, versions=versions)


def _main_phase(doc: dict) -> dict:
    needle = "train"
    found = {}
    for phase in doc.get("phases", []):
        if needle in (phase.get("name") or ""):
            found = phase
    return found


def notice(doc: dict, model_type: str) -> str:
    """The one-line annotation that survives when the log is scrolled away.

    ``rc`` is read from the recorded phase rather than from an environment
    variable, so a failing job still reports the numbers that led to it.
    """
    phase = _main_phase(doc)
    rc = phase.get("rc", "n/a")

    fields = [
        f"sfs={doc.get('sfs_uuid', '?')}",
        f"cache_hit={doc.get('cache_hit', '?')}",
        f"shards={doc.get('shards', '?')}",
        f"size={doc.get('model_size', '?')}",
        f"layers={doc.get('layers', '?')}",
        f"sp_ep={doc.get('sp', '?')}/{doc.get('ep', '?')}",
        f"train_s={phase.get('seconds', 'n/a')}",
        f"train_rc={rc}",
    ]
    lines = [f"::notice title=smoke {model_type}::{' '.join(fields)}"]
    if str(rc) not in ("0", "None", ""):
        lines.append(
            f"::error title=train failed::rc={rc}; "
            "see the train tail in the job summary"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=["init", "set", "phase", "report", "notice"],
    )
    parser.add_argument("--file", required=True, help="事实文件")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--seconds", default="")
    parser.add_argument("--rc", default="")
    parser.add_argument("--train-log", default="/tmp/train.log",
                        help="训练主日志")
    parser.add_argument("--versions", default="/tmp/after.txt")
    parser.add_argument("values", nargs="*")
    # Python 3.10 / 3.11 的 argparse 不允许「选项之后跟位置参数」：
    # `init --file X k=v` 会把 k=v 判成 unrecognized arguments（3.12 起才放宽）。
    # 而 CI 里两个 python 都可能是 3.11（镜像与 GH 托管机），所以这里不靠版本：
    # 用 parse_known_args 兜住尾巴，只认 k=v，其他（比如拼错的 --flag）照样报错，
    # 免得选项被静默吞掉。
    args, extra = parser.parse_known_args(argv)
    for token in extra:
        if "=" in token and not token.startswith("-"):
            args.values.append(token)
        else:
            parser.error(f"unrecognized argument: {token}")

    path = Path(args.file)

    if args.command == "set":
        doc = _load(path)
        doc.update(_pairs(args.values))
        _save(path, doc)
        return 0

    if args.command == "phase":
        doc = _load(path)
        doc.setdefault("phases", []).append(
            {"name": args.name, "seconds": args.seconds, "rc": args.rc}
        )
        _save(path, doc)
        return 0

    if args.command == "notice":
        print(notice(_load(path), args.model_type or str(_load(path).get("model_type", "?"))))
        return 0

    if args.command == "init":
        _save(path, _pairs(args.values))
        return 0

    doc = _load(path)
    doc.setdefault("phases", [])
    sys.stdout.write(
        render(
            doc,
            log=Path(args.train_log),
            versions=Path(args.versions),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
