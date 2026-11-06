#!/usr/bin/env python3
"""Accumulate CI facts in one JSON file and render the job summary from it.

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
import xml.etree.ElementTree as ET
from pathlib import Path

DIAG_PREFIX = "[ttdiag]"

# Two job shapes share this file. `kind` decides which summary layout is used, so a
# pytest run never prints a table of "?" for model-cache columns it has no concept of.
KIND_TRAIN = "train"
KIND_PYTEST = "pytest"

# 映射表的字段顺序。run_train.sh 的 cmd_prepare 按这个顺序 set -- 分区，
# 所以只能往后追加，不能插队。
_MAP_FIELDS = ("run_dir", "lf", "swift", "hf_repo", "e2e")


def _read_map(path: str) -> dict[str, dict[str, str]]:
    """读 model_map.json -> {model_type: {run_dir, lf, swift, hf_repo, e2e}}。

    缺字段按 "-" 补齐。读不了直接抛：调用方（map-row / map-keys /
    migrate-map）要么据此报错退出，要么把异常渲染成摘要，都不应该
    静默降成空表 —— 空表看起来就像「这个模型不存在」。
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    models = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"{path} 缺顶层 models 对象")
    return {
        name: {k: str(row.get(k, "-") or "-") for k in _MAP_FIELDS}
        for name, row in models.items() if isinstance(row, dict)
    }


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


def _grep(path: Path, needle: str, limit: int) -> str:
    if not path.is_file():
        return ""
    hits = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if needle in line:
                hits.append(line)
                if len(hits) >= limit:
                    break
    except Exception:
        return ""
    return "\n".join(hits)


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


def _render_train(doc: dict, *, log: Path, traceback: Path, versions: Path) -> str:
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
        "### model_type → 启动脚本映射（model_type / run_dir / lf / swift / hf_repo）",
        "",
        "```",
        doc.get("model_map", "(map missing)").rstrip() or "(map missing)",
        "```",
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
        "",
        "### Python 异常（excepthook 捕获，torchrun 转发会截断）",
        "",
        "```",
        _tail(traceback, 60) or "(none)",
        "```",
        "",
        "### loss 诊断（NaN 出在哪一层）",
        "",
        "```",
        _grep(log, DIAG_PREFIX, 400) or "(none)",
        "```",
    ]
    return "\n".join(lines) + "\n"


def _render_pytest(doc: dict, *, log: Path, versions: Path, junit: Path) -> str:
    get = _getter(doc)
    phases = doc.get("phases", [])

    lines = [
        f"## pytest `{get('test_path')}` on `{get('runner_label')}`",
        "",
        "| item | value |",
        "|---|---|",
        f"| image | `{get('image')}` |",
        f"| target | `{get('test_path')}` |",
        f"| args | `{get('pytest_args')}` |",
    ]
    stats = _junit_stats(junit)
    if stats is None:
        lines.append("| tests | (no JUnit report) |")
    else:
        lines.append(
            f"| tests | {stats['total']} total, **{stats['failed'] + stats['errors']} failed**, "
            f"{stats['passed']} passed, {stats['skipped']} skipped |"
        )
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
        "### 测试尾部",
        "",
        "```",
        _tail(log, 60) or "(no pytest log)",
        "```",
    ]
    if stats is not None and stats["failed_cases"]:
        lines += ["", "### Failed cases", ""]
        lines += [f"- `{name}`" for name in stats["failed_cases"]]
    return "\n".join(lines) + "\n"


def _junit_stats(path: Path) -> dict | None:
    """JUnit 统计。缺文件或解析失败都返回 None —— 汇总是报告步骤，
    永远不能反过来掩盖真实的测试结果。"""
    if not path.is_file():
        return None
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError):
        return None

    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    names = [
        f"{case.get('classname', '')}::{case.get('name', '')}"
        for suite in suites
        for case in suite.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    return {
        "total": total,
        "passed": total - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "seconds": sum(float(s.get("time", 0) or 0) for s in suites),
        "failed_cases": names,
    }


def render(
    doc: dict, *, log: Path, traceback: Path, versions: Path, junit: Path
) -> str:
    if doc.get("kind") == KIND_PYTEST:
        return _render_pytest(doc, log=log, versions=versions, junit=junit)
    return _render_train(doc, log=log, traceback=traceback, versions=versions)


def _main_phase(doc: dict) -> dict:
    needle = "pytest" if doc.get("kind") == KIND_PYTEST else "train"
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

    if doc.get("kind") == KIND_PYTEST:
        stats = _junit_stats(Path(doc.get("junit", "/tmp/pytest.xml")))
        fields = [
            f"target={doc.get('test_path', '?')}",
            f"rc={rc}",
            f"tests={stats['total'] if stats else '?'}",
            f"failed={(stats['failed'] + stats['errors']) if stats else '?'}",
            f"t={phase.get('seconds', 'n/a')}s",
        ]
        lines = [f"::notice title=pytest {model_type}::{' '.join(fields)}"]
        if str(rc) not in ("0", "None", ""):
            lines.append(
                f"::error title=pytest failed::rc={rc}; "
                "see the failed cases in the job summary"
            )
        return "\n".join(lines)

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
        choices=["init", "set", "phase", "report", "notice", "migrate-map",
                 "junit", "map-row", "map-keys"],
    )
    parser.add_argument("--file", default="", help="事实文件；除 junit 外都必填")
    parser.add_argument("--model-type", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--seconds", default="")
    parser.add_argument("--rc", default="")
    parser.add_argument("--train-log", default="/tmp/train.log",
                        help="主日志：训练用 train.log，pytest 用 pytest.log")
    parser.add_argument("--traceback", default="/tmp/tt_traceback.log")
    parser.add_argument("--versions", default="/tmp/after.txt")
    parser.add_argument("--junit", default="/tmp/pytest.xml")
    parser.add_argument("--map", default=".github/workflows/model_map.json")
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

    if args.command == "junit":
        # 在日志里留一行可 grep 的计数。缺文件不是错误：测试可能根本没跑到。
        stats = _junit_stats(Path(args.junit))
        if stats is None:
            print(f"junit: 没有可读的报告（{args.junit}）")
            return 0
        print(
            "junit: total={total} passed={passed} failed={failed} "
            "errors={errors} skipped={skipped} {seconds:.1f}s".format(**stats)
        )
        for name in stats["failed_cases"][:20]:
            print(f"  failed: {name}")
        return 0

    # ---- 映射表查询 -------------------------------------------------------
    # run_train.sh 以前用 awk 取第 2/3/4/5 列。改成 JSON 之后用 awk 就得现写
    # 一个 JSON 解析器，所以挪到这里：shell 只负责把值读进变量。
    # 这两个不需要 --file（它们不碰事实文件），所以放在必填校验之前。
    if args.command == "map-row":
        if not args.model_type:
            parser.error("map-row 需要 --model-type")
        try:
            row = _read_map(args.map).get(args.model_type)
        except (OSError, ValueError) as exc:
            print(f"::error::读不了映射表 {args.map}: {exc}", file=sys.stderr)
            return 2
        if row is None:
            return 1
        print(" ".join(row[k] for k in _MAP_FIELDS))
        return 0

    if args.command == "map-keys":
        try:
            models = _read_map(args.map)
        except (OSError, ValueError) as exc:
            print(f"::error::读不了映射表 {args.map}: {exc}", file=sys.stderr)
            return 2
        for name in models:
            print(name)
        return 0

    if not args.file:
        parser.error(f"{args.command} 需要 --file")

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

    if args.command == "migrate-map":
        # The summary embeds the map so a run is self-describing. Reading it here
        # keeps the workflow from shelling out to grep/sed.
        doc = _load(path)
        try:
            models = _read_map(args.map)
            doc["model_map"] = "\n".join(
                f"{name:<22} " + " ".join(str(row[k]) for k in _MAP_FIELDS)
                for name, row in models.items()
            )
        except Exception as exc:                       # noqa: BLE001
            doc["model_map"] = f"(map unreadable: {exc})"
        _save(path, doc)
        return 0

    doc = _load(path)
    doc.setdefault("phases", [])
    sys.stdout.write(
        render(
            doc,
            log=Path(args.train_log),
            traceback=Path(args.traceback),
            versions=Path(args.versions),
            junit=Path(args.junit),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
