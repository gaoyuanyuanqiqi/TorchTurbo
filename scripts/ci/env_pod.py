#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pod 环境的体检、防破坏与复原。

原先三个文件（`verify_env.py` + `torch_guard.sh` + `npu_cleanup.sh`）其实管的是
同一件事 —— **这台 pod 现在是什么状态**。合并之后一条命令就能看全：

    verify   进来的第一件事：镜像对不对、卡在不在、每张卡能不能算
    capture  装依赖之前记一份 torch 的现状
    check    装完对比：pip 有没有把自建的 NPU wheel 换成 PyPI 那份
    cleanup  超时/取消之后收尾：残留进程、显存、/tmp、半个 checkpoint

为什么要 `check`：L1 镜像里的 torch 是自己编的（版本号带 `nomimalloc`），
两个安装步骤都跑 pip，而 pip 会为了满足某个包的元数据从索引里解一份新 torch
盖掉它。`--no-build-isolation` 只是缓解。症状极不显眼：`import torch` 还能用，
`torch.npu` 没了，所有张量悄悄退回 CPU —— pytest 立刻发现，训练不会，只是慢 100 倍。
所以它必须让 job 失败，而不是警告。

`cleanup` 的四件事，每一件都能让「下一个 job」以完全无关的方式失败：
训练子进程占着 /dev/davinci*、显存没释放、SFS 上半个 checkpoint、/tmp 里上
一次的派生配置。每一步都独立、都不致命 —— 收尾自己失败不该把成功的 run 判成失败。
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# 只打印，不在这里 import。
INFORMATIONAL_MODULES = ("fla_npu", "fla")


def warn(message: str) -> None:
    print(f"::warning::{message}")


def error(message: str) -> None:
    print(f"::error::{message}")


def log(message: str) -> None:
    print(f"[env] {message}", file=sys.stderr)


# ---------------------------------------------------------------- verify
def _report_versions() -> None:
    import torch
    import torch_npu
    import triton

    print("python:", sys.version.split()[0], "| machine:", platform.machine())
    print("torch :", torch.__version__, "| torch_npu:", torch_npu.__version__)
    # pip 元数据报的是原版 triton 版本（3.5.0），因为 triton-ascend 是直接覆盖
    # triton/ 下的文件而不改 dist metadata。运行时版本才是可信信号，所以照打。
    print("triton:", triton.__version__, "(runtime; pip metadata differs by design)")
    for name in INFORMATIONAL_MODULES:
        try:
            print(f"{name}: {__import__(name).__version__}")
        except Exception as exc:                 # noqa: BLE001 - 只是信息
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")


def _check_build_tag() -> None:
    """区分自建 wheel 与 PyPI 那份。`nomimalloc` 是硬编码进
    PYTORCH_BUILD_VERSION 的，能证明来自 image-ci，不能证明 mimalloc 真被去掉。"""
    import torch

    if "nomimalloc" not in torch.__version__:
        warn("版本号里没有 nomimalloc：镜像可能不是 image-ci 构建的")


def _check_devices() -> int:
    import torch
    import torch_npu  # noqa: F401 - 注册 npu 后端

    available = torch.npu.is_available()
    count = torch.npu.device_count()
    print("npu   :", available, "| devices:", count)
    assert available, "torch.npu.is_available() == False，本 job 没分到 NPU"
    assert count > 0, "torch.npu.device_count() == 0，本 job 没分到 NPU"
    return count


def _check_matmul(count: int) -> None:
    import torch

    for index in range(count):
        dev = f"npu:{index}"
        a = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
        b = torch.randn(1024, 1024, device=dev, dtype=torch.float16)
        err = ((a @ b).float().cpu() - a.float().cpu() @ b.float().cpu()).abs().max().item()
        print(f"{dev} matmul ok, max_abs_err={err:.4f}")
        assert err < 1.0, f"{dev} matmul 误差异常 err={err}"


def cmd_verify() -> int:
    """镜像里的栈对不对 + 每张卡真算一次。

    只有真算一次 matmul 才能区分「没设备」和「有设备但栈坏了」。
    """
    _report_versions()
    _check_build_tag()
    _check_matmul(_check_devices())
    return 0


# ---------------------------------------------------------------- capture / check
_PROBE = """
import json, sys
doc = {}
try:
    import torch
except Exception as exc:
    doc["torch_ok"] = False
    doc["torch_error"] = f"{type(exc).__name__}: {exc}"
else:
    doc["torch_ok"] = True
    doc["torch_version"] = torch.__version__
    doc["torch_file"] = torch.__file__
    try:
        import torch_npu
        doc["npu_version"] = getattr(torch_npu, "__version__", "unknown")
        doc["npu_file"] = torch_npu.__file__
    except Exception as exc:
        doc["npu_error"] = f"{type(exc).__name__}: {exc}"
json.dump(doc, sys.stdout)
"""


def _probe() -> dict:
    """记录现状。缺 torch 是一种「观察」，不是中止理由 —— L2 镜像本来就可能没有。"""
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout or "{}")
    except ValueError:
        return {"torch_ok": False, "torch_error": "probe produced no output"}


def _describe(doc: dict) -> str:
    if not doc.get("torch_ok"):
        return f"torch 不可用（{doc.get('torch_error', 'not importable')}）"
    return (
        f"torch {doc.get('torch_version', '?')} @ {doc.get('torch_file', '?')}; "
        f"torch_npu {doc.get('npu_version', 'MISSING')}"
    )


def cmd_capture(path: str) -> int:
    doc = _probe()
    Path(path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    log(f"captured: {_describe(doc)}")
    return 0


def cmd_check(path: str) -> int:
    """装完依赖回头看 torch 还在不在、是不是原来那个。"""
    saved = Path(path)
    if not saved.is_file():
        log(f"没有 capture 文件 {saved}")
        return 2
    before = json.loads(saved.read_text(encoding="utf-8"))
    after = _probe()

    if not before.get("torch_ok") and not after.get("torch_ok"):
        log("装之前装之后都没有 torch，无可比对")
        return 0
    if before.get("torch_ok") and not after.get("torch_ok"):
        error(f"装完之后 torch 不可 import：{after.get('torch_error', 'unknown')}")
        return 1
    if (
        before.get("torch_version") != after.get("torch_version")
        or before.get("torch_file") != after.get("torch_file")
    ):
        error(
            f"torch 被替换：{before.get('torch_version')} ({before.get('torch_file')}) "
            f"-> {after.get('torch_version')} ({after.get('torch_file')})"
        )
        error("NPU build 已经不在了；在安装脚本里钉住元凶，或加显式的 torch 约束")
        return 1
    if before.get("npu_version") and not after.get("npu_version"):
        error(f"装完之后 torch_npu 不见了（原来是 {before['npu_version']}）")
        return 1

    log(f"ok: torch {after.get('torch_version')}；torch_npu {after.get('npu_version', 'n/a')}")
    return 0


# ---------------------------------------------------------------- cleanup
TRAIN_PROCS = "run_lf\\.py|run_swift\\.py|torchrun|deepspeed"
TMP_PATTERNS = ("tt_*.yaml", "plan.out", "train_meta.out", "cache.out")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def cmd_cleanup(output_dir: str) -> int:
    print("=== env cleanup ===")

    left = _run(["pgrep", "-af", TRAIN_PROCS])
    if left.returncode == 0 and left.stdout.strip():
        print("--- 残留进程 ---")
        print(left.stdout.strip())
        # 按命令杀整组，避免只留下 torchrun 的 worker。
        _run(["pkill", "-9", "-f", TRAIN_PROCS])
    else:
        print("no leftover training process")

    # 显存取证。清理后如果还有人占着，这里能看出来 —— 但不去 kill 别人的 job。
    if shutil.which("npu-smi"):
        mem = _run(["npu-smi", "info", "-t", "memory"])
        if mem.stdout:
            print("--- npu-smi memory ---")
            print("\n".join(mem.stdout.splitlines()[:20]))

    # 只删本 CI 写的那几个具名文件，不碰 /tmp 其他内容。
    for pattern in TMP_PATTERNS:
        for path in Path("/tmp").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    # 半写 checkpoint：有完成标记就整份保留，否则会把上一次的完整结果一起删掉，
    # 下次缓存探测就永远 MISS。
    out = Path(output_dir)
    if out.is_dir():
        if (out / ".tt-complete").is_file():
            print(f"checkpoint complete marker found, keeping {out}")
        else:
            print(f"no completeness marker, clearing partial output in {out}")
            for child in out.iterdir():
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(
                    missing_ok=True
                )

    print("=== env cleanup done ===")
    return 0


# ---------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="pod 环境：体检 / 防破坏 / 复原")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="镜像里的栈对不对 + 每张卡真算一次 matmul")

    cap = sub.add_parser("capture", help="装依赖之前记一份 torch 现状")
    cap.add_argument("--file", required=True)

    chk = sub.add_parser("check", help="装完对比，torch 被换掉就失败")
    chk.add_argument("--file", required=True)

    cln = sub.add_parser("cleanup", help="超时/取消之后的收尾")
    cln.add_argument("--output-dir", default="/tmp/tblf-sft")

    args = ap.parse_args(argv)
    if args.cmd == "verify":
        return cmd_verify()
    if args.cmd == "capture":
        return cmd_capture(args.file)
    if args.cmd == "check":
        return cmd_check(args.file)
    return cmd_cleanup(args.output_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
