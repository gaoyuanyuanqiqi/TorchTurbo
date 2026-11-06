#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""「这个测试文件要不要真 NPU 卡」的唯一判据。

原来这条规则在三个地方各写了一遍（tests/conftest.py、select_plan.py、
run_tests.sh），任意一份漂了都不会报错，只会**静默换掉跑的集合**。
现在只留这一份，另外三处都调它（check_static.py 的 npu-rule 项会拦重复实现）。

## 为什么要区分「真设备」和「假 torch_npu」

原先的判据是「文件里出现过 torch_npu」，实测两个方向都错：

* 假阳性：`test_turbo_vs_base_precision.py` 只在**注释**里提了一句 torch_npu，
  它是纯 CPU 精度测试，却被拖上 2 卡 pod；
  `test_attention_npu.py` / `test_npu_matmul_add.py` 全程 monkeypatch 假模块，
  零真实 kernel，同样占卡。
* 假阴性：`test_varlen_metadata.py` 要 `fla_npu`、`test_muon_offload_distributed.py`
  要 `torch.npu`，都不含 `torch_npu` 这个串 → CPU 泳道（缺依赖）skip、
  NPU 泳道（glob 过滤）进不去，**两条泳道都不跑**。

## 规则

判定前先去掉注释（保留字符串 —— `importorskip("torch_npu")` 是在字符串里）。
命中下列任一即为「要真设备」：

| 特征 | 含义 |
|---|---|
| `import torch_npu` / `from torch_npu import` | 真模块 |
| `importorskip("torch_npu")` | 真模块或整文件跳过 |
| `torch.npu.` | 真设备 API（`device_count` 之类） |
| `fla_npu` | AscendC 融合算子库 |
| `device="npu"` | 直接把张量放到卡上 |

**不**命中（历史上被误判为 NPU 的写法）：

| 特征 | 为什么不算 |
|---|---|
| `sys.modules["torch_npu"] = ...` | 假模块注入，CPU 可跑 |
| `TurboDevice.NPU` | 枚举值，`FakeRuntime` 里也这么写 |
| `is_npu_available` | 常见用法是 monkeypatch 成 False 来强制走 CPU |
"""

from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

# 命中即「要真设备」。
_DEVICE_PATTERNS = (
    re.compile(r"\bimport\s+torch_npu\b"),
    re.compile(r"\bfrom\s+torch_npu\b"),
    re.compile(r"""importorskip\(\s*["']torch_npu["']"""),
    re.compile(r"\btorch\.npu\."),
    re.compile(r"\bfla_npu\b"),
    re.compile(r"""device\s*=\s*["']npu["']"""),
)

CACHE: dict[Path, bool] = {}


def source_without_comments(path: Path) -> str:
    """去掉注释、保留字符串。

    不能用「按行 grep -v '#'」，字符串里的 `#` 会被误删；
    也不能连字符串一起去掉，`importorskip("torch_npu")` 就藏在字符串里。
    tokenize 是唯一同时满足这两点的做法。tokenize 失败时退回原文
    （宁可多划一个文件进 NPU 泳道，也不能漏跑）。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        lines = raw.splitlines(keepends=True)
        # 只把注释那段字符挖成空格，其余**原样保留**。
        # 不能用 untokenize：它会插空格，把 `torch.npu.` 变成
        # `torch . npu .`、把 `importorskip("torch_npu")` 变成
        # `importorskip ( "torch_npu" )`，两个正则就都匹配不上了（实测过）。
        spans = [
            (tok.start, tok.end)
            for tok in tokenize.generate_tokens(io.StringIO(raw).readline)
            if tok.type == tokenize.COMMENT
        ]
        chars = [list(line) for line in lines]
        # 从后往前挖，前面的列号才不会失效。
        for (row0, col0), (row1, col1) in reversed(spans):
            if row0 == row1:
                chars[row0 - 1][col0:col1] = " " * (col1 - col0)
        return "".join("".join(line) for line in chars)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return raw


def needs_device(path: Path) -> bool:
    """这个测试文件是否必须跑在有真实 NPU 的机器上。"""
    hit = CACHE.get(path)
    if hit is None:
        try:
            text = source_without_comments(path)
        except OSError:
            text = ""
        hit = any(pattern.search(text) for pattern in _DEVICE_PATTERNS)
        CACHE[path] = hit
    return hit


def filter_device(paths: list[Path]) -> list[Path]:
    return [path for path in paths if needs_device(path)]


def main(argv: list[str]) -> int:
    """命令行入口，供 run_tests.sh 使用。

    ``--filter``：从 stdin 读文件列表，输出仍要留在 NPU 泳道的那些。
    ``--explain``：逐条打印判定结果与命中的特征。
    """
    if not argv or argv[0] not in ("--filter", "--explain"):
        print(__doc__)
        return 2

    paths = [Path(line.strip()) for line in sys.stdin if line.strip()]
    if argv[0] == "--filter":
        print("\n".join(str(path) for path in filter_device(paths)))
        return 0

    for path in paths:
        text = source_without_comments(path) if path.exists() else ""
        hits = [pattern.pattern for pattern in _DEVICE_PATTERNS if pattern.search(text)]
        print(f"{'DEVICE' if hits else 'cpu   '}  {path}  {' '.join(hits)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
