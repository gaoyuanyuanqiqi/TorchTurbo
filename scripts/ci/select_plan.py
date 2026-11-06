#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 PR 改动的文件，算出要跑哪些测试、要不要起 NPU、跑哪个 E2E 画像。

三层选取，互相独立，任一命中即算选中：

  A 反向 import 闭包
      改动的模块 <- 谁 import 它 <- 这些模块又被哪些测试 import。
      这是主路径：改 engine/config.py 会带出所有间接依赖它的测试。

  B 叶子名匹配
      cross_entropy_loss.py -> test_cross_entropy_loss.py。
      兜底 A 的断链（测试直接调函数、没有 import 模块）。

  C 目录兜底
      src/torchturbo/engine/x.py -> tests/engine/。
      只对「A 和 B 都为空」的单个文件生效，避免把整个目录拖进来。

只依赖标准库；435 个文件约 0.4 s，所以每次 PR 现算，不维护静态映射表
（映射表一定会随重构腐烂，而腐烂是静默的）。

用法：
    select_plan.py --changed-file a.py --changed-file b.py
    git diff --name-only base...HEAD | select_plan.py --changed-from -
    select_plan.py --changed-file x.py --explain      # 打印每个文件命中了哪层
    select_plan.py --audit                            # 覆盖体检（不进 PR 门禁）
    select_plan.py --self-test                        # 断言算法没退化
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

PKG = "torchturbo"
CHEAPEST_MODEL = "qwen3_5"   # 2B，族内小模型，不减层就能跑

# 大 PR 直接全量：改 1000 个文件的 PR 往往是大重构，正是全量最能抓问题的时候，
# 而且选择器在这种规模下本来也会选中绝大部分，降级省不了钱。
MAX_PRECISE_FILES = 300


# --------------------------------------------------------------------------
# 索引：src 模块图 + 测试对模块的依赖
# --------------------------------------------------------------------------
class Index:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.src: dict[str, Path] = {}
        self.deps: dict[str, set[str]] = {}
        self.tests: dict[str, set[str]] = {}   # 测试相对路径 -> 它 import 的 torchturbo 模块
        self.cases: dict[str, int] = {}         # 测试相对路径 -> 静态 test_ 函数数
        self.test_files: list[str] = []
        self._build()

    # ---- 建图 ------------------------------------------------------------
    def _build(self) -> None:
        src_root = self.root / "src"
        for path in sorted((src_root / PKG).rglob("*.py")):
            self.src[self._dotted(path, src_root)] = path

        for name, path in self.src.items():
            self.deps[name] = self._imports(path, name)

        tests_root = self.root / "tests"
        for path in sorted(tests_root.rglob("test_*.py")):
            rel = path.relative_to(self.root).as_posix()
            self.test_files.append(rel)
            tree = self._parse(path)
            self.tests[rel] = self._imports_from(tree, path, None)
            # 静态计数：pytest 的实际用例数还要乘 parametrize 展开，这里只用来
            # 报「选中率」，不需要精确；选测/执行单位始终是文件。
            self.cases[rel] = sum(
                1 for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test")
            )

    @staticmethod
    def _dotted(path: Path, base: Path) -> str:
        parts = list(path.relative_to(base).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _resolve(self, cand: str) -> str | None:
        """把 import 的字面量削到「真实存在的模块」上。

        `from torchturbo.engine.config import Config` 里 Config 是符号不是模块，
        逐级回退直到命中，命中即停。
        """
        parts = cand.split(".")
        for i in range(len(parts), 0, -1):
            key = ".".join(parts[:i])
            if key in self.src:
                return key
        return None

    def _pkg_of(self, modname: str, path: Path) -> str:
        if path.name == "__init__.py":
            return modname
        return modname.rsplit(".", 1)[0] if "." in modname else ""

    def _imports(self, path: Path, modname: str | None) -> set[str]:
        return self._imports_from(self._parse(path), path, modname)

    @staticmethod
    def _parse(path: Path) -> ast.AST:
        try:
            return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return ast.Module(body=[], type_ignores=[])

    def _imports_from(self, tree: ast.AST, path: Path, modname: str | None) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    hit = self._resolve(alias.name)
                    if hit:
                        found.add(hit)

            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # 相对 import：先退到包，再按 level 逐级上浮
                    if modname is None:
                        continue
                    pkg = self._pkg_of(modname, path).split(".")
                    up = node.level - 1
                    if up:
                        pkg = pkg[:-up] if up <= len(pkg) else []
                    base = ".".join(pkg + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                if not base:
                    continue
                for alias in node.names:
                    hit = self._resolve(f"{base}.{alias.name}") or self._resolve(base)
                    if hit:
                        found.add(hit)
        return found

    # ---- 派生 ------------------------------------------------------------
    def reverse_deps(self) -> dict[str, set[str]]:
        rev: dict[str, set[str]] = {name: set() for name in self.src}
        for name, deps in self.deps.items():
            for dep in deps:
                rev.setdefault(dep, set()).add(name)
        return rev

    def dependents(self, seeds: set[str]) -> set[str]:
        """seeds 及所有（间接）依赖 seeds 的模块。"""
        rev = self.reverse_deps()
        seen: set[str] = set()
        stack = [s for s in seeds if s in self.src]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(rev.get(cur, ()))
        return seen

    def leaf_owners(self) -> dict[str, int]:
        """叶子名 -> 有多少个模块用它。只有唯一时才敢做 B 层名字匹配。"""
        counts: dict[str, int] = {}
        for name in self.src:
            leaf = name.rsplit(".", 1)[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
        return counts
    def dangling_tests(self, deleted: set[str]) -> set[str]:
        """删掉的模块 -> 还在 import 它（或其子模块）的测试。

        `self.src` 扫的是**删除之后**的树，所以删一个模块时 A 层从种子出发
        什么都找不到。而删模块恰恰是最容易让一片测试在 import 期就崩的改动。
        这里反向查：某个测试的种子是删除模块的前缀，就说明它 import 过。

        `import torchturbo.registry.core` 在文件消失后会被 `_resolve` 削到
        `torchturbo.registry`，所以按前缀匹配才找得回来。
        """
        if not deleted:
            return set()
        out: set[str] = set()
        for test, seeds in self.tests.items():
            for seed in seeds:
                if any(d == seed or d.startswith(seed + ".") for d in deleted):
                    out.add(test)
                    break
        return out

# --------------------------------------------------------------------------
# 画像映射
# --------------------------------------------------------------------------
def load_model_map(root: Path) -> dict[str, dict[str, str]]:
    """model_map.json -> {model_type: {run_dir, lf, swift, hf_repo, e2e}}。

    缺字段按 "-" 补齐，这样旧表也能跑。读不动就当空表 —— 调用方
    （pick_e2e）会把空表当成「没有画像」，不会静默改变选测。
    """
    path = root / ".github" / "workflows" / "model_map.json"
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"::warning::读不了 {path}: {exc}；按空表处理", file=sys.stderr)
        return {}
    raw = doc.get("models") if isinstance(doc, dict) else None
    if not isinstance(raw, dict):
        return {}
    fields = ("run_dir", "lf", "swift", "hf_repo", "e2e")
    return {
        name: {k: str(row.get(k, "-") or "-") for k in fields}
        for name, row in raw.items() if isinstance(row, dict)
    }


def pick_e2e(changed: list[str], models: dict[str, dict[str, str]]) -> list[dict]:
    """改动路径 -> E2E 画像，去重后按优先级排序。

    族目录变更优先于「最便宜画像」：改 qwen3_5_moe 就该跑 qwen3_5_moe，
    而不是拿 2B 的 qwen3_5 顶替。
    """
    picks: list[tuple[int, str, str]] = []   # (优先级, model_type, 依据)

    def add(priority: int, model: str | None, reason: str) -> None:
        if not model or model not in models:
            return
        if models[model]["e2e"] == "-":
            return
        picks.append((priority, model, reason))

    for p in changed:
        if p.startswith("src/torchturbo/models/"):
            seg = p[len("src/torchturbo/models/"):].split("/")[0]
            if seg.endswith(".py") or seg == "modules":
                add(1, CHEAPEST_MODEL, f"{p}（通用模型层）")
            else:
                add(0, seg, f"{p}（族目录）")

        elif p.startswith("src/torchturbo/adapters/"):
            # swift 适配层只有 MoE 那边有启动脚本；lf 走最便宜的 dense。
            add(1, "qwen3_5_moe" if "swift" in p else "bailing_hybrid", p)

        elif p.startswith("src/torchturbo/"):
            pkg = p[len("src/torchturbo/"):].split("/")[0]
            if pkg in ("ops", "engine", "parallel"):
                add(1, CHEAPEST_MODEL, f"{p}（{pkg} 变更）")
            # runtime / utils / registry 等：不影响训练能否起来，不起 E2E

        elif p.startswith("scripts/runs/"):
            run_dir = p[len("scripts/runs/"):].split("/")[0]
            for name, row in models.items():
                if row["run_dir"] == run_dir:
                    add(0, name, f"{p}（run_dir={run_dir}）")
                    break

        elif p.startswith("docker/npu/"):
            add(1, CHEAPEST_MODEL, f"{p}（镜像配方变更）")

        elif p == "pyproject.toml":
            # 依赖变了，训练随时可能起不来 —— 这是最该跑 E2E 的信号。
            add(1, CHEAPEST_MODEL, "pyproject.toml（依赖变更）")

        # .github/**  与 scripts/ci/**：CI 自身改动不烧 NPU
        # tests/**：测试改了不会让训练崩
        # docs/**、*.md：不参与

    picks.sort(key=lambda x: x[0])
    out: list[dict] = []
    seen: set[str] = set()
    for _, model, reason in picks:
        if model in seen:
            continue
        seen.add(model)
        spec = models[model]["e2e"]
        layers = 0
        if spec.startswith("trim:"):
            layers = int(spec.split(":", 1)[1])
        out.append({"model_type": model, "layers": layers, "e2e": spec, "reason": reason})
    return out


# 曾经有一个 `wants_npu(changed)` 在这里，按路径前缀猜「要不要起 NPU」。
# 它和选中的文件是两套判断，会出现「ops 改了但选中的全是 CPU 文件却起了 NPU」，
# 也会漏掉「改了别的目录但确实选中了 NPU 测试」。现在改成从 `targets` 分类结果
# 反推（见 select() 里的 lanes），一份数据得出一个结论。


# --------------------------------------------------------------------------
# 选择
# --------------------------------------------------------------------------
def _tokens(stem: str) -> set[str]:
    return {t for t in stem.replace("-", "_").split("_") if t and t != "test"}


# 每个文件只读一次：select() 里会反复问「这个测试要不要卡」。
_NPU_CACHE: dict[Path, bool] = {}


def _load_npu_rule():
    """加载唯一判据模块。

    规则写在 `npu_rule.py` 一份里，conftest / run_tests.sh / 这里都调它。
    原先三处各写一遍「文件里含 torch_npu」—— 那份规则把注释里提了一句的
    CPU 测试拖上了卡，又把要 `fla_npu` / `torch.npu` 的文件漏在两条泳道之外。
    按路径加载而不是 import：scripts/ci 不是包，直接 import 会依赖 sys.path[0]。
    """
    spec = importlib.util.spec_from_file_location(
        "ci_npu_rule", Path(__file__).with_name("npu_rule.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_NPU_RULE = _load_npu_rule()


def has_torch_npu(path: Path) -> bool:
    """这个测试文件是否必须跑在真 NPU 机器上。

    判据本体在 `scripts/ci/npu_rule.py`（去注释、保留字符串，认
    torch_npu / torch.npu / fla_npu / device="npu"）。这里只做缓存，
    避免 select() 反复读同一个文件。
    """
    hit = _NPU_CACHE.get(path)
    if hit is None:
        hit = _NPU_RULE.needs_device(path)
        _NPU_CACHE[path] = hit
    return hit


def mirror_test_dir(root: Path, rel: str) -> str | None:
    """src 相对路径 -> tests 下最深的已存在镜像目录。

    `models/qwen3_5_moe/model.py` -> `tests/models/qwen3_5_moe`，而不是退到
    `tests/models`（那会把整个 models 目录拖进来，等于没选）。
    逐级变短，取第一个真实存在的目录。
    """
    parts = rel.split("/")[:-1]              # 去掉文件名
    for n in range(len(parts), 0, -1):
        cand = "tests/" + "/".join(parts[:n])
        if (root / cand).is_dir():
            return cand
    return None


def select(index: Index, changed: list[str], force_all: bool = False,
           e2e_all: bool = False, e2e_force: bool = False,
           no_npu: bool = False) -> dict:
    changed = sorted(set(p for p in changed if p))
    total_tests = len(index.test_files)

    valves: list[str] = []
    if force_all:
        valves.append("force-all")
    if not changed:
        valves.append("no-changed-files")
    if len(changed) > MAX_PRECISE_FILES:
        valves.append(f"too-many-files({len(changed)})")
    # 依赖变了就跑全量：装不上就全红，挑用例没有意义。
    if "pyproject.toml" in changed:
        valves.append("deps-changed")
    # 改 CI 自身（workflow / 选择器 / 运行脚本）也跑全量 CPU：
    # 这些脚本就是「怎么跑测试」本身，改错了会静默少跑，只有全量能看出来。
    # 但它们不烧 NPU（E2E 与 L2a 由下面的路径规则另行判断）。
    if any(p.startswith((".github/", "scripts/ci/")) for p in changed):
        valves.append("ci-config-changed")

    src_changed = [p for p in changed if p.startswith(f"src/{PKG}/") and p.endswith(".py")]
    changed_modules: set[str] = set()
    for p in src_changed:
        mod = Index._dotted(index.root / p, index.root / "src")
        if mod in index.src:
            changed_modules.add(mod)

    # ---- A：反向 import 闭包 -------------------------------------------
    affected = index.dependents(changed_modules)
    a_tests = {t for t, seeds in index.tests.items() if seeds & affected}

    # ---- B：叶子名匹配 --------------------------------------------------
    owners = index.leaf_owners()
    b_tests: set[str] = set()
    leaf_hits: dict[str, list[str]] = {}
    for mod in changed_modules:
        leaf = mod.rsplit(".", 1)[-1]
        want = _tokens(leaf)
        if not want:
            continue
        # 同名模块不止一个时（native/npu 各一份 cross_entropy_loss.py），
        # 额外要求「父包名」也出现在测试名或它的目录里，否则宁可交给 C 层兜。
        parent = mod.rsplit(".", 2)[-2] if mod.count(".") >= 2 else ""
        ambiguous = owners.get(leaf, 0) > 1

        hit = []
        for t in index.test_files:
            haystack = _tokens(Path(t).stem) | _tokens("/".join(Path(t).parts[:-1]))
            if not want <= haystack:
                continue
            if ambiguous and (not parent or parent not in haystack):
                continue
            hit.append(t)
        if hit:
            leaf_hits[mod] = hit
            b_tests.update(hit)

    # ---- C：目录兜底（只补 A/B 都为空的文件） ---------------------------
    c_tests: set[str] = set()
    for p in src_changed:
        rel = p[len(f"src/{PKG}/"):]
        probe = mirror_test_dir(index.root, rel)
        if probe is None:
            continue
        covered = any(t.startswith(probe + "/") for t in a_tests | b_tests)
        if not covered:
            c_tests.update(t for t in index.test_files if t.startswith(probe + "/"))

    # ---- 直接改测试 / 改启动脚本 ----------------------------------------
    direct = [p for p in changed if p.startswith("tests/") and Path(p).name.startswith("test_")]
    script_tests: set[str] = set()
    if any(p.startswith("scripts/runs/") for p in changed):
        script_tests = {t for t in index.test_files if t.startswith("tests/scripts/")}

    targets = sorted(set(direct) | a_tests | b_tests | c_tests | script_tests)

    if valves:
        mode = "all"
    elif c_tests and not (a_tests or b_tests or direct or script_tests):
        mode = "fallback"
    else:
        mode = "precise"

    # 全量就是真的全量：调用方不该再需要知道「all = 用 --shard 覆盖」。
    if mode == "all":
        targets = sorted(index.test_files)

    # ---- D：被删掉的源码 --------------------------------------------------
    # `changed_modules` 只收「删完之后仍然存在」的模块，所以删文件时 A/B 层
    # 什么都选不到。这里把「还在 import 它」的测试补回来。
    deleted_modules = {
        Index._dotted(index.root / p, index.root / "src")
        for p in src_changed
        if not (index.root / p).exists()
    }
    dangling = index.dangling_tests(deleted_modules)
    if dangling:
        targets = sorted(set(targets) | dangling)

    # ---- 按「要不要真卡」切成两条泳道 -------------------------------------
    # 判断方式与 tests/conftest.py 一致（源码里出现 torch_npu）。
    # 这样「要不要起 NPU」不再靠猜路径，而是看实际选中的文件 —— 一份数据
    # 得出一个结论，不会出现「ops 改了但选中的全是 CPU 文件却起了 NPU」。
    npu_targets = [t for t in targets if has_torch_npu(index.root / t)]
    npu_set = set(npu_targets)
    cpu_targets = [t for t in targets if t not in npu_set]

    models = load_model_map(index.root)
    e2e = pick_e2e(changed, models)
    if e2e_force and not e2e:
        # ci:e2e：不管路径怎么说，至少把最便宜的画像跑一遍。
        e2e = pick_e2e(["pyproject.toml"], models)
    if e2e_all:
        # ci:full / main 全量：所有有画像的族都跑，调用方负责串行。
        e2e = [{"model_type": name, "layers": int(row["e2e"].split(":", 1)[1])
                if row["e2e"].startswith("trim:") else 0,
                "e2e": row["e2e"], "reason": "全量（ci:full）"}
               for name, row in models.items() if row["e2e"] != "-"]

    # 分片数：空跑/小改动不分片，全量按 30 个文件一片，上限 5 片。
    files_for_shard = len(cpu_targets)
    shards = 1 if files_for_shard <= 12 else min(5, math.ceil(files_for_shard / 30))

    # ---- lanes：决策的唯一产出 --------------------------------------------
    # 每个泳道都带 why，缺 why 的门禁判红 —— 「不知道为什么没跑」比「跑了但
    # 失败」更危险，它会安静地变成常态。yaml 只读这里的 run，不做任何判断。
    image_changed = any(p.startswith("docker/npu/") for p in changed)
    cpu_unit_run = len(cpu_targets) > 0
    npu_unit_run = not no_npu and (bool(npu_targets) or image_changed or mode == "all")
    npu_model_run = bool(e2e)

    lanes = {
        "cpu_unit": {
            "run": cpu_unit_run,
            "why": (f"选中 {len(cpu_targets)} 个 CPU 文件"
                    if cpu_unit_run else "没有选中任何 CPU 测试"),
            "files": cpu_targets,
            "shards": shards,
        },
        "npu_unit": {
            "run": npu_unit_run,
            # files 为空 = 泳道内全跑：算子/镜像的影响是全仓的，
            # 只跑「选中的那几个」等于没验到真正的暴露面。
            "files": [],
            "selected": len(npu_targets),
            "why": (
                "ci:skip-npu 标签" if no_npu else
                f"命中 {len(npu_targets)} 个需要真卡的测试文件（泳道内全跑）"
                if npu_targets else
                "镜像配方变更（泳道内全跑）" if image_changed else
                "全量模式（泳道内全跑）" if mode == "all" else
                "没有选中任何需要真卡的测试文件"
            ),
        },
        "npu_model": {
            "run": npu_model_run,
            "why": (f"{len(e2e)} 个画像：" + ",".join(e["model_type"] for e in e2e)
                    if npu_model_run else "没有匹配的端到端画像"),
            "profiles": e2e,
        },
    }

    reasons = {
        p: {"A": len(a_tests) if p in src_changed else 0,
            "B": len(leaf_hits.get(Index._dotted(index.root / p, index.root / "src"), []))
                 if p in src_changed else 0,
            "C": len(c_tests) if p in src_changed else 0}
        for p in changed
    }

    return {
        "schema": 1,
        "mode": mode,
        "valves": valves,
        "targets": targets,
        "shards": shards,
        # npu 保留为「L2a 要不要跑」的同义词，避免调用方跟着改。
        "npu": npu_unit_run,
        "cpu_files": cpu_targets,
        "npu_files": npu_targets,
        "e2e": e2e,
        "lanes": lanes,
        "counts": {
            "files": len(changed),
            "tests": len(targets),
            "total": total_tests,
            "cases": sum(index.cases.get(t, 0) for t in targets),
            "cases_total": sum(index.cases.values()),
            "pct": round(100.0 * len(targets) / total_tests, 1) if total_tests else 0.0,
        },
        "reasons": reasons,
        "layers": {"A": sorted(a_tests), "B": sorted(b_tests),
                   "C": sorted(c_tests), "direct": sorted(direct),
                   "dangling": sorted(dangling)},
    }


# --------------------------------------------------------------------------
# 体检（不进 PR 门禁，看趋势用）
# --------------------------------------------------------------------------
def audit(index: Index) -> dict:
    touched: set[str] = set()
    for seeds in index.tests.values():
        touched |= seeds

    owners = index.leaf_owners()
    for mod in index.src:
        leaf = mod.rsplit(".", 1)[-1]
        if owners.get(leaf, 0) != 1:
            continue
        want = _tokens(leaf)
        if want and any(want <= _tokens(Path(t).stem) for t in index.test_files):
            touched.add(mod)

    # C 层：A/B 都空时靠镜像目录兜。它定位的是「目录」，精度低于用例级，
    # 所以单独计一个数，不和 A/B 混。
    mirrored: set[str] = set()
    for mod, path in index.src.items():
        rel = path.relative_to(index.root / "src").as_posix()
        rel = rel[len(PKG) + 1:] if rel.startswith(PKG + "/") else rel
        if mirror_test_dir(index.root, rel):
            mirrored.add(mod)

    orphans = [t for t, seeds in index.tests.items() if not seeds]
    return {
        "modules": len(index.src),
        "test_files": len(index.test_files),
        "reachable": len(touched),
        "mirrored": len(mirrored),
        # A/B 精确定位率：算法能直接把「模块」对到「测试」的比例。
        "precision": round(100.0 * len(touched) / len(index.src), 1) if index.src else 0.0,
        # A/B/C 合计：能定位到测试范围（目录级也算）的比例。选测率的下界。
        "precision_dir": round(100.0 * len(touched | mirrored) / len(index.src), 1) if index.src else 0.0,
        "orphans": len(orphans),
        "orphan_files": orphans[:20],
        "untested": len(set(index.src) - touched),
        "untested_sample": sorted(set(index.src) - touched)[:20],
    }


# --------------------------------------------------------------------------
# 自检：把「算法没退化」写成断言
# --------------------------------------------------------------------------
def self_test(root: Path) -> int:
    index = Index(root)
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if cond else 'FAIL'}  {name}{('  -> ' + detail) if detail and not cond else ''}")
        if not cond:
            failures.append(name)

    print("select_plan.py --self-test")

    # ---- 泳道开关（yaml 只读这四个值） ------------------------------------
    def lane_runs(r: dict) -> dict[str, bool]:
        return {k: r["lanes"][k]["run"] for k in ("cpu_unit", "npu_unit", "npu_model")}

    r = select(index, ["docs/ci/README.md"])
    check("文档 -> 一条泳道都不起",
          lane_runs(r) == {"cpu_unit": False, "npu_unit": False, "npu_model": False},
          str(lane_runs(r)))

    r = select(index, ["src/torchturbo/ops/cross_entropy_loss.py"])
    check("ops 算子 -> 起 NPU", r["npu"])
    check("ops 算子 -> 选中用例非空", len(r["targets"]) > 0, str(r["counts"]))
    check("ops 算子 -> E2E 画像 = qwen3_5",
          [e["model_type"] for e in r["e2e"]] == [CHEAPEST_MODEL], str(r["e2e"]))
    check("ops 算子 -> NPU 泳道内全跑（files 为空）", r["lanes"]["npu_unit"]["files"] == [])
    check("ops 算子 -> npu-unit 确实被选中", r["lanes"]["npu_unit"]["selected"] > 0,
          str(r["lanes"]["npu_unit"]["selected"]))

    r = select(index, ["src/torchturbo/engine/core.py"])
    check("engine -> 精准模式", r["mode"] == "precise", r["mode"])
    # engine.core 被包门面 torchturbo.engine 直接 import，所以凡是
    # `from torchturbo.engine import ...` 的测试都会被带出来。
    check("engine -> 选中面有规模", r["counts"]["tests"] >= 20, str(r["counts"]))

    r = select(index, ["src/torchturbo/models/qwen3_5/model.py"])
    check("族目录 -> 该族画像", [e["model_type"] for e in r["e2e"]] == ["qwen3_5"], str(r["e2e"]))
    check("族目录 -> 不起 npu-unit", not r["lanes"]["npu_unit"]["run"], str(lane_runs(r)))
    check("族目录 -> 起 npu-model", r["lanes"]["npu_model"]["run"], str(lane_runs(r)))

    r = select(index, ["src/torchturbo/models/qwen3_5_moe/model.py"])
    check("MoE 族 -> 减层画像",
          [e["layers"] for e in r["e2e"]] == [6], str(r["e2e"]))

    r = select(index, ["docker/npu/Dockerfile_aarch64"])
    check("镜像配方 -> 起 npu-unit（选不中测试也要跑）", r["lanes"]["npu_unit"]["run"], str(lane_runs(r)))

    r = select(index, ["scripts/ci/select_plan.py"])
    check("CI 自身改动 -> 全量 CPU", r["mode"] == "all", r["mode"])
    check("CI 自身改动 -> 不起 npu-model", r["e2e"] == [], str(r["e2e"]))

    r = select(index, ["README.md"])
    check("文档 -> 0 用例 0 泳道",
          r["counts"]["tests"] == 0 and not r["npu"] and not r["e2e"], str(r["counts"]))

    r = select(index, ["pyproject.toml"])
    check("pyproject -> 全量", r["mode"] == "all", r["mode"])
    check("pyproject -> 起 npu-model", [e["model_type"] for e in r["e2e"]] == [CHEAPEST_MODEL], str(r["e2e"]))
    check("pyproject -> 三条泳道全跑", all(lane_runs(r).values()), str(lane_runs(r)))
    check("pyproject -> cpu-unit 只给 CPU 文件",
          all(not has_torch_npu(index.root / f) for f in r["lanes"]["cpu_unit"]["files"]))

    r = select(index, [f"src/torchturbo/engine/f{i}.py" for i in range(400)])
    check("大 PR -> 全量（不是降级）", r["mode"] == "all", str(r["valves"]))

    # ---- 删除文件：dangling import 兜底 ----------------------------------
    # `Index.src` 扫的是删完之后的树，所以删除场景必须靠名字反查。
    dangling = index.dangling_tests({"torchturbo.registry.registry"})
    check("删模块 -> 找到 import 过它的测试", len(dangling) > 0, str(len(dangling)))

    a = audit(index)
    print(f"  info  精准定位 {a['precision']}%  含目录兜底 {a['precision_dir']}%  "
          f"（{a['modules']} 模块 / {a['test_files']} 测试文件）")
    check("精准定位率 >= 65%（回归护栏）", a["precision"] >= 65.0, f"{a['precision']}%")
    check("含目录兜底 >= 90%", a["precision_dir"] >= 90.0, f"{a['precision_dir']}%")

    # ---- 事件输入 -> 四个开关 ---------------------------------------------
    # 这段规则原先分散在 workflow 的 bash case 里，yaml 里加一个标签就等于多
    # 一份实现。这里把每个组合钉死。
    def flags(mode: str, labels: str = "", with_e2e: bool = False,
              force_all: bool = False) -> tuple:
        f = flags_from_event(mode, labels, with_e2e, force_all)
        return tuple(f[k] for k in ("force_all", "e2e_all", "e2e_force", "no_npu"))

    check("full -> 全量单测，不带 E2E", flags("full") == (True, False, False, False),
          str(flags("full")))
    check("full + with_e2e -> 全部画像", flags("full", with_e2e=True) == (True, True, False, False),
          str(flags("full", with_e2e=True)))
    check("pr 无标签无输入 -> 什么都不要", flags("pr") == (False, False, False, False),
          str(flags("pr")))
    check("ci:full 标签 -> 全量 + 全部画像",
          flags("pr", "ci:full") == (True, True, False, False), str(flags("pr", "ci:full")))
    check("ci:e2e 标签 -> 只跑最便宜画像",
          flags("pr", "ci:e2e") == (False, False, True, False), str(flags("pr", "ci:e2e")))
    check("ci:skip-npu 标签 -> 不起 NPU",
          flags("pr", "ci:skip-npu") == (False, False, False, True),
          str(flags("pr", "ci:skip-npu")))
    check("标签里的空格不影响判定",
          flags("pr", "bug, ci:e2e ,docs") == (False, False, True, False),
          str(flags("pr", "bug, ci:e2e ,docs")))
    check("pr + force_all 不隐含 E2E（8 卡必须显式要）",
          flags("pr", force_all=True) == (True, False, False, False),
          str(flags("pr", force_all=True)))
    check("pr + with_e2e -> 一个画像",
          flags("pr", with_e2e=True) == (False, False, True, False),
          str(flags("pr", with_e2e=True)))

    print(f"\n{'PASS' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 0 if not failures else 1


# --------------------------------------------------------------------------
def changed_from_git(base_ref: str = "origin/main", diff_filter: str = "ACDMT") -> list[str]:
    """算本次 PR 改了哪些文件。

    基线取 merge-base（PR 分叉点），不是 main 的最新提交：main 上别人合进来的
    改动不该算进这次 PR 的选测范围，否则 PR 会随着 main 前进越来越"全量"。

    两个参数是刻意的：

    * `--no-renames` 让改名同时给出「旧路径删除 + 新路径新增」两条。只报新路径
      的话，「把代码移走」看起来等于没删任何东西，被移走的模块的 import 者就没人
      管了。DeepSpeed 的 `ci/check_paths.py` 是同一个理由。
    * `--diff-filter=ACDMT` **包含 D（删除）**。默认的 `d` 是小写 = **排除**
      删除，那样「删掉一个模块」在选测里完全不可见 —— 而这恰恰最容易让一整片
      测试在 import 期就崩。

    任何一步失败都返回空列表 —— 空列表会被 select() 当成「信息不足」直接全量
    （fail-open），比猜一个错的基线安全。
    """
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], capture_output=True, text=True)

    def give_up(why: str) -> list[str]:
        print(f"::warning::changed_files: {why}；退回全量", file=sys.stderr)
        return []

    if git("rev-parse", "--git-dir").returncode != 0:
        return give_up("不在 git 仓库里")

    # PR 里 base 分支可能还没 fetch 到，先补一次（浅克隆也能用）。
    if git("rev-parse", "--verify", "-q", base_ref).returncode != 0:
        git("fetch", "--no-tags", "--depth=200", "origin", base_ref.split("/", 1)[-1])

    base = git("merge-base", "HEAD", base_ref).stdout.strip() \
        or git("rev-parse", base_ref).stdout.strip()
    if not base:
        return give_up(f"拿不到基线 {base_ref}")

    print(f"changed_files: base={base} ref={base_ref}", file=sys.stderr)
    diff = git("diff", "--name-only", "--no-renames",
               f"--diff-filter={diff_filter}", f"{base}...HEAD")
    if diff.returncode != 0:
        return give_up(f"git diff 失败：{diff.stderr.strip()[:200]}")
    return [line for line in diff.stdout.splitlines() if line.strip()]


def read_changed(args) -> list[str]:
    files: list[str] = []
    if args.changed_from == "-":
        files += [line.strip() for line in sys.stdin if line.strip()]
    elif args.changed_from == "git":
        files += changed_from_git(args.base_ref)
    elif args.changed_from:
        files += [line.strip() for line in Path(args.changed_from).read_text(encoding="utf-8").splitlines() if line.strip()]
    files += args.changed_file or []
    return sorted(set(files))


def flags_from_event(mode: str, labels: str, with_e2e: bool,
                     force_all: bool = False) -> dict[str, bool]:
    """事件输入 -> select() 的四个开关。**规则只在这里。**

    原先这段是 workflow 里的一个 bash `case`,于是「标签什么意思」这件事同时
    存在于 yaml 和文档里，两边加一个标签就漂了。放到这里之后有一条自测序列
    （见 self_test）能把它钉死。

    mode=full  : 全量单测；with_e2e 才连 E2E 一起。
    mode=pr    : 标签优先（ci:full > ci:e2e > ci:skip-npu），再看 dispatch 的
                 force_all / with_e2e。PR 事件本身永远不会给出 with_e2e，
                 所以「PR 免费拿 E2E」这件事在结构上不可能发生。
    """
    want = {"force_all": False, "e2e_all": False,
            "e2e_force": False, "no_npu": False}

    if mode == "full":
        want["force_all"] = True
        want["e2e_all"] = bool(with_e2e)
        return want

    tags = {t.strip() for t in labels.split(",") if t.strip()}
    if "ci:full" in tags:
        want["force_all"] = True
        want["e2e_all"] = True
    elif "ci:e2e" in tags:
        want["e2e_force"] = True
    elif "ci:skip-npu" in tags:
        want["no_npu"] = True

    # force_all 的意思是「全部单测」，不是「连 E2E 也全跑」：8 卡那条泳道
    # 才是最贵的，保持 opt-in。
    if force_all:
        want["force_all"] = True
    if with_e2e:
        want["e2e_force"] = True
    return want


def render_plan(result: dict, mode: str, image: str,
                changed_files: list[str], limit: int = 100) -> str:
    """Job Summary 里的计划表。放在这里是因为它渲染的就是 select() 的产出。"""
    counts = result["counts"]
    lanes = result["lanes"]
    runs = " / ".join(str(lanes[k]["run"]).lower()
                      for k in ("cpu_unit", "npu_unit", "npu_model"))
    e2e = ", ".join(p["model_type"] for p in result["e2e"]) or "-"
    head = "\n".join([
        "## Plan",
        "",
        "| item | value |",
        "|---|---|---|",
        f"| mode | `{mode}` |",
        f"| lanes (cpu/npu/model) | `{runs}` |",
        f"| e2e | `{e2e}` |",
        f"| changed | {counts['files']} files |",
        f"| selected | {counts['tests']}/{counts['total']} files "
        f"({counts['pct']}%), ~{counts['cases']} cases |",
        f"| image | `{image}` |",
        f"| valves | {', '.join(result['valves']) or '-'} |",
    ])
    shown = changed_files[:limit]
    more = len(changed_files) - len(shown)
    body = "\n".join(shown) + (f"\n... (+{more} more)" if more > 0 else "")
    return (head
            + "\n\n<details><summary>changed files</summary>\n\n```\n"
            + body + "\n```\n</details>\n")


def _bool(raw: str | bool) -> bool:
    """'true' / 'false'（GitHub 的表达式求值结果）-> bool。

    yaml 里写 `--with-e2e "$WITH_E2E"` 比 `$( [ ... ] && echo --with-e2e )`
    短且不会因为空展开把参数错位。
    """
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description="按改动文件选出测试 / NPU / E2E 画像")
    ap.add_argument("--root", default=str(here.parents[2]),
                    help="仓库根目录（默认从脚本位置推）")
    ap.add_argument("--changed-file", action="append", default=[],
                    help="改动文件，可重复")
    ap.add_argument("--changed-from", default=None,
                    help="从文件读改动列表；'-' = stdin，'git' = 自己算 merge-base diff")
    ap.add_argument("--base-ref", default="origin/main",
                    help="--changed-from git 时的基线（默认 origin/main）")
    ap.add_argument("--dump-changed", default=None,
                    help="把算出的改动清单另存一份（供 summary / 阀门判断用）")
    ap.add_argument("--force-all", nargs="?", const=True, default=False, type=_bool,
                    help="强制全量（ci:full）；可显式给 true/false")
    ap.add_argument("--e2e-all", action="store_true", help="所有有画像的族都跑 E2E")
    ap.add_argument("--e2e-force", action="store_true", help="路径没命中时也跑最便宜画像")
    ap.add_argument("--no-npu", action="store_true", help="强制不起 NPU（ci:skip-npu）")
    # ---- 事件输入的解读（取代 workflow 里的 bash case）--------------------
    ap.add_argument("--mode", choices=("pr", "full"), default=None,
                    help="pr = 按改动选测；full = 全量。给了就从这里算四个开关")
    ap.add_argument("--labels", default="",
                    help="逗号分隔的 PR 标签；只认 ci:full / ci:e2e / ci:skip-npu")
    ap.add_argument("--with-e2e", nargs="?", const=True, default=False, type=_bool,
                    help="pr = 最便宜的一个画像；full = 全部画像")
    ap.add_argument("--summary-file", default=None,
                    help="把计划表追加到这个文件（GITHUB_STEP_SUMMARY）")
    ap.add_argument("--image-ref", default="-",
                    help="只用于 --summary-file 里的展示")
    ap.add_argument("--explain", action="store_true", help="打印每个文件的命中情况")
    ap.add_argument("--audit", action="store_true", help="覆盖体检")
    ap.add_argument("--self-test", action="store_true", help="断言算法没退化")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="只输出第 I 片（0 起）的测试文件")
    ap.add_argument("--output", default=None, help="JSON 写到文件")
    ap.add_argument("--github-output", default=None,
                    help="同时追加 key=value 到这个文件（GITHUB_OUTPUT）")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    if not (root / "src" / PKG).is_dir():
        print(f"::error::找不到 {root}/src/{PKG}，用 --root 指定仓库根", file=sys.stderr)
        return 2

    if args.self_test:
        return self_test(root)

    index = Index(root)

    if args.audit:
        print(json.dumps(audit(index), ensure_ascii=False, indent=2))
        return 0

    changed = read_changed(args)
    if args.dump_changed:
        Path(args.dump_changed).write_text(
            "\n".join(changed) + ("\n" if changed else ""), encoding="utf-8"
        )

    # --mode 给了就以它为准（规则集中在 flags_from_event）；否则用逐个
    # --force-all / --e2e-* 参数（ci-full 之外的调用方、以及自测都在用）。
    if args.mode:
        flags = flags_from_event(args.mode, args.labels, args.with_e2e,
                                 args.force_all)
    else:
        flags = {"force_all": args.force_all, "e2e_all": args.e2e_all,
                 "e2e_force": args.e2e_force, "no_npu": args.no_npu}
    result = select(index, changed, **flags)

    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        # 分片只对 L1（CPU）有意义：NPU 泳道是泳道内全跑。
        files = result["cpu_files"]
        # 轮转切分：按文件而不是按用例，保证每片之间用例数大致均衡。
        result["targets"] = [f for k, f in enumerate(files) if k % n == i]
        result["shard"] = f"{i}/{n}"

    if args.explain:
        print(f"mode={result['mode']} files={result['counts']['files']} "
              f"files-hit={result['counts']['tests']}/{result['counts']['total']} "
              f"({result['counts']['pct']}%) cases~{result['counts']['cases']} "
              f"shards={result['shards']}")
        for name, lane in result["lanes"].items():
            flag = "RUN " if lane["run"] else "skip"
            print(f"  {flag} {name:<4} {lane['why']}")
        if result["valves"]:
            print(f"阀门: {', '.join(result['valves'])}")
        for p in changed:
            r = result["reasons"][p]
            print(f"  {p}  A={r['A']} B={r['B']} C={r['C']}")
        for e in result["e2e"]:
            print(f"  E2E {e['model_type']} ({e['e2e']}, {e['layers']} 层) <- {e['reason']}")

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)

    if args.summary_file:
        with open(args.summary_file, "a", encoding="utf-8") as fh:
            fh.write(render_plan(result, args.mode or result["mode"],
                                 args.image_ref, changed))

    if args.github_output:
        lanes = result["lanes"]
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"mode={result['mode']}\n")
            fh.write(f"shards={result['shards']}\n")
            fh.write("shard_list=" + json.dumps(
                [f"{i}/{result['shards']}" for i in range(result['shards'])]
            ) + "\n")
            # 泳道开关：yaml 只读这三个值，不做任何判断。
            for name in ("cpu_unit", "npu_unit", "npu_model"):
                fh.write(f"{name}_run={str(lanes[name]['run']).lower()}\n")
            # 这个名字是为了区分「cpu-unit 真的拿到了文件」与「拿到全量」。
            # 因为 --github-output 一次只能写一行，用 JSON 数组传过去。
            fh.write("cpu_unit_files="
                     + json.dumps(lanes["cpu_unit"]["files"], ensure_ascii=False) + "\n")
            fh.write(f"npu={str(result['npu']).lower()}\n")
            fh.write(f"targets={json.dumps(result['targets'], ensure_ascii=False)}\n")
            fh.write(f"e2e={json.dumps(result['e2e'], ensure_ascii=False)}\n")
            # 保底：run_tests.sh 在不给 TEST_FILES 时自己 glob 全量，这里保留原语义。
            fh.write(f"test_files={json.dumps(index.test_files, ensure_ascii=False)}\n")
            fh.write(f"summary={result['mode']} {result['counts']['files']}f "
                     f"{result['counts']['tests']}/{result['counts']['total']}f "
                     f"{result['counts']['pct']}% cases~{result['counts']['cases']}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except BrokenPipeError:
        # `select_plan.py --explain | head` 会提前关掉管道；别让它刷一堆噪音。
        sys.stdout = None
        sys.exit(0)
