"""Lint 静态检查：不跑模型、不起 NPU，几十秒内把「结构性错误」挡住。

子命令（默认 static）：

    static            六项静态检查
    static --only X   只跑其中一项（训练侧只用 map）
    hadolint [Docker] Dockerfile 检查，二进制而不是镜像：hadolint/hadolint
                      只有 amd64，而 CI_CPU_RUNNER 默认是 arm64 托管机
    stage             把「倒在哪一步」写成 step output（select 的收尾）

六项：

1. 语法编译      —— src / scripts / tests 全部 py_compile
2. workflow 解析 —— .github/workflows/*.yml 能被 YAML 解析，且有 on / jobs
3. 映射表漂移    —— 代码里 @register_model 的名字必须都在 model_map.json 里
4. 决策层契约    —— workflow 读的 `needs.select.outputs.X` 必须真的被声明。
   这一条踩过：`select_plan.py` 算了 `targets` 却没人读，`select` 输出的是
   全部 144 个文件，选测静默失效了好几天。现在只要有一边拼错就红。
5. 选择器防篡改  —— `scripts/ci/**` 改动必须走全量（`ci-config-changed` 阀门）。
   否则一个 PR 可以把自己的选择器改成「永远选 0 个测试」并且全绿。

第 3 项原本写在训练 workflow 的 ② 里，跑到那里才发现要白等一轮装依赖；
放到 Lint 才能在 20 s 内报出来。训练侧仍保留同样的检查（那里是最终防线）。
"""

from __future__ import annotations

import glob
import json
import os
import platform
import py_compile
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / ".github" / "workflows" / "model_map.json"
PKG = "torchturbo"

# 注册了但刻意不进映射表的模型：它们没有独立启动脚本、也不参与 CI。
# 放进映射表反而会让选测把它们算进 E2E，而它们跑不起来。
NOT_IN_CI = {"qwen3_5_moe_text"}


def check_compile() -> list[str]:
    bad: list[str] = []
    for base in ("src", "scripts", "tests"):
        for path in sorted((ROOT / base).rglob("*.py")):
            try:
                py_compile.compile(str(path), doraise=True, cfile="/tmp/tt_pyc")
            except py_compile.PyCompileError as exc:
                bad.append(f"{path.relative_to(ROOT)}: {exc.msg.strip().splitlines()[-1]}")
    print(f"compile: {len(bad)} 个文件编译失败")
    return bad


def check_workflows() -> list[str]:
    try:
        import yaml  # noqa: PLC0415 - 可选依赖，缺了就降级
    except ImportError:
        print("::warning::没装 PyYAML，workflow 只做结构检查")
        yaml = None

    bad: list[str] = []
    paths = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    for path in paths:
        rel = path.relative_to(ROOT)
        # 命名规范：一律 `ci-<scope>.yml`，小写 kebab。
        # 不统一的时候会出一堆前缀，谁都分不清哪个是入口、哪个只是被调用的实现。
        if not re.fullmatch(r"ci-[a-z0-9]+(-[a-z0-9]+)*\.yml", path.name):
            bad.append(f"{rel}: 文件名不符合 ci-<scope>.yml（小写 kebab）")
        if yaml is not None:
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:                       # noqa: BLE001
                bad.append(f"{rel}: YAML 解析失败: {type(exc).__name__}: {exc}")
                continue
            if not isinstance(doc, dict):
                bad.append(f"{rel}: 顶层不是 mapping")
                continue
            # YAML 1.1 会把裸 on: 解析成 True，两种都接受。
            if not any(k in doc for k in ("on", True)):
                bad.append(f"{rel}: 缺 on:")
            if "jobs" not in doc:
                bad.append(f"{rel}: 缺 jobs")
        else:
            text = path.read_text(encoding="utf-8")
            if "jobs:" not in text:
                bad.append(f"{rel}: 缺 jobs")
    print(f"workflows: {len(paths)} 个文件，{len(bad)} 个有问题")
    return bad


def check_model_map() -> list[str]:
    registered: set[str] = set()
    for path in sorted((ROOT / "src" / PKG / "models").glob("*/model.py")):
        src = path.read_text(encoding="utf-8")
        registered.update(re.findall(r'@register_model\(\s*["\']([^"\']+)["\']', src))
    registered -= NOT_IN_CI

    bad: list[str] = []
    mapped: set[str] = set()
    if MAP.exists():
        try:
            doc = json.loads(MAP.read_text(encoding="utf-8"))
        except ValueError as exc:
            # JSON 坏了比漏收录更麻烦：整个映射表都读不出来，而下游
            # load_model_map() 会静默降成空表，选测看起来「正常」但不起 E2E。
            return [f"model_map.json 解析失败: {exc}"]
        models = doc.get("models") if isinstance(doc, dict) else None
        if not isinstance(models, dict):
            return ["model_map.json 缺顶层 models 对象"]
        mapped = set(models)
        fields = ("run_dir", "lf", "swift", "hf_repo", "e2e")
        for name, row in models.items():
            if not isinstance(row, dict):
                bad.append(f"model_map.json: {name} 不是对象")
                continue
            missing = [k for k in fields if k not in row]
            if missing:
                bad.append(f"model_map.json: {name} 缺字段 {missing}")
            spec = str(row.get("e2e", "-"))
            if spec != "-" and spec != "full" and not re.fullmatch(r"trim:\d+", spec):
                bad.append(f"model_map.json: {name} 的 e2e='{spec}' 不合法")
    else:
        return [f"找不到映射表 {MAP.relative_to(ROOT)}"]

    missing = sorted(registered - mapped)
    unknown = sorted(mapped - registered)
    print(f"model_map: registered={len(registered)} mapped={len(mapped)}")
    if unknown:
        # 只警告：先删代码后删表是正常顺序。
        print("::warning::映射表里有未注册的名字: " + ", ".join(unknown))
    if missing:
        bad.append(f"model_map 未收录: {', '.join(missing)}")
    return bad


def check_contract() -> list[str]:
    """workflow 读的 select 输出必须真的被声明；决策层必须能算出来。

    这一类错「能跑、能绿、只是少跑了东西」，比直接报错难发现得多：
    拼错一个 output 名，GitHub 给空字符串，`if: ... == 'true'` 永假 →
    job 静默不跑，gate 又把它当成「不需要」。
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        print("contract: 跳过（没装 PyYAML）")
        return []

    bad: list[str] = []
    # lint 不再是独立泳道（它是 select 里的几个 step），所以不再列 lint_run。
    needed = ("cpu_unit_run", "npu_unit_run", "npu_model_run",
              "cpu_unit_files")

    # 1. 决策层真的会写这些 key 吗 —— 直接跑一次，别猜源码长什么样。
    #    这是唯一能同时挡住「忘了写」和「写错名字」的办法。
    out = Path("/tmp/tt_lane_probe.txt")
    out.unlink(missing_ok=True)
    rc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ci" / "select_plan.py"),
         "--changed-file", "pyproject.toml", "--github-output", str(out)],
        capture_output=True, text=True,
    ).returncode
    if rc != 0:
        bad.append(f"select_plan.py 跑不通（rc={rc}），决策层不可用")
    else:
        written = {
            line.split("=", 1)[0]
            for line in out.read_text(encoding="utf-8").splitlines() if "=" in line
        }
        for key in needed:
            if key not in written:
                bad.append(f"select_plan.py 没有产出 {key}（工作流会拿到空字符串 → 静默不跑）")

    # 2. 每个 workflow：读到的 needs.select.outputs.X 必须已声明
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text) or {}
        select = (doc.get("jobs") or {}).get("select") or {}
        declared = set((select.get("outputs") or {}).keys())

        used = set(re.findall(r"needs\.select\.outputs\.([A-Za-z0-9_]+)", text))
        for name in sorted(used - declared):
            bad.append(f"{rel}: 读了 needs.select.outputs.{name}，但 select 没声明它")

        # merge-base 需要历史；fetch-depth: 1 会让 diff 失败 → 静默退回全量。
        if "--changed-from git" in text and "fetch-depth: 0" not in text:
            bad.append(f"{rel}: 用了 --changed-from git 但 checkout 没设 fetch-depth: 0")

        # 3. 静态检查与算计划必须**在同一个 job 里**，且静态在前。
        #    拆成两个 job 不会报错，只会白占一个 runner（它们本来就严格
        #    串行）；顺序写反则会拿一个没检查过的选择器算计划。
        jobs = doc.get("jobs") or {}
        dump = {n: yaml.safe_dump(j, allow_unicode=True) for n, j in jobs.items()}
        for name, body in dump.items():
            if "check_static.py" not in body:
                continue
            if "select_plan.py" not in body:
                bad.append(f"{rel}: {name} 只跑静态检查、不算计划 —— 两者应同 job")
            elif body.index("check_static.py") > body.index("select_plan.py"):
                bad.append(f"{rel}: {name} 里 select_plan.py 排在 check_static.py 之前")

        # 4. 计划只能有一个地方算：两处各算一次，必然慢慢分叉。
        planners = [n for n, body in dump.items() if "--changed-from git" in body]
        if len(planners) > 1:
            bad.append(f"{rel}: 有多个 job 在算计划 {planners}")

        # 5. 三条泳道必须**并行**：都只依赖 select，谁也不依赖谁。
        #    串起来不会报错，但 L1 红会白等，且 L2a/L2b 的 if 要多写一层
        #    「上游是否挂了」的判断——那是这轮特意去掉的东西。
        lanes = ("l1_cpu_unit", "l2_npu_unit", "l3_npu_model")
        if "select" in jobs and set(lanes) <= set(jobs):
            for lane in lanes:
                deps = jobs[lane].get("needs") or []
                if isinstance(deps, str):
                    deps = [deps]
                if list(deps) != ["select"]:
                    bad.append(
                        f"{rel}: {lane} 的 needs 应为 [select]（三条泳道并行），实为 {list(deps)}"
                    )
                # 矩阵 = 一张卡片变多个 job，图上一条泳道会炸成一堆盒子。
                if "strategy" in jobs[lane]:
                    bad.append(f"{rel}: {lane} 不能用 strategy/matrix（每条泳道只能有一个 job）")
                # `uses:` 的 job 在图上一律显示成 `调用方名 / 被调方名`，两段
                # 合不成一段 —— 所以泳道必须内联。
                if "uses" in jobs[lane]:
                    bad.append(
                        f"{rel}: {lane} 不能是可复用 workflow"
                        "（图上会变成 `调用方 / 被调方` 两段名字）"
                    )

        # 6. select 必须直连 gate：否则它失败时 gate 要多等三条已 skipped 的边。
        if "gate" in jobs and "select" in jobs:
            gate_deps = jobs["gate"].get("needs") or []
            if isinstance(gate_deps, str):
                gate_deps = [gate_deps]
            if "select" not in gate_deps:
                bad.append(f"{rel}: gate 的 needs 缺 select（提前结束通路断了）")

        # 6. 事件语义不许回到 yaml。标签 -> 开关的换算只在
        #    select_plan.py 的 flags_from_event() 里；yaml 里再写一遍就是两份
        #    实现，加一个标签必然漏改一边，而漏改不会报错，只会少跑。
        bad += _check_no_event_semantics(rel, text)

        # 7. 「超过 10 行的 bash 一律下沉到 scripts/ci」——这条容易被慢慢侵蚀，
        #    所以留一条硬线。超了就把逻辑搬进脚本，那里能本地跑、过 shellcheck。
        bad += _check_run_blocks(rel, jobs)

    print(f"contract: {len(needed)} 个 lane 输出 / {len(bad)} 个问题")
    return bad


RUN_BLOCK_LIMIT = 14      # 含 set -uo pipefail 之类的骨架行


def _check_no_event_semantics(rel: str, text: str) -> list[str]:
    """标签名不该出现在 workflow 的**逻辑**里 —— 它只属于 select_plan.py。

    `description:` 里的提及是文档（给人看的手动触发说明），不算判定。
    """
    bad: list[str] = []
    for tag in ("ci:full", "ci:e2e", "ci:skip-npu"):
        for line in text.splitlines():
            body = line.split("#", 1)[0]          # 注释里提标签是允许的
            if "description:" in body:
                continue
            if tag in body:
                bad.append(f"{rel}: workflow 里出现标签判定 `{tag}`，"
                           "应交给 select_plan.py 的 flags_from_event")
                break
    return bad


def _check_run_blocks(rel: str, jobs: dict) -> list[str]:
    """yaml 里留声明，逻辑下沉到 scripts/ci。"""
    bad: list[str] = []
    for job_name, job in jobs.items():
        for step in (job or {}).get("steps") or []:
            lines = len((step.get("run") or "").splitlines())
            if lines > RUN_BLOCK_LIMIT:
                bad.append(
                    f"{rel}: {job_name} 的 `{step.get('name')}` 有 {lines} 行 bash"
                    f"（上限 {RUN_BLOCK_LIMIT}），请下沉到 scripts/ci"
                )
    return bad


def check_selector_guard() -> list[str]:
    """选择器改了自己必须触发全量，否则 PR 能靠改选择器「过」。

    `ci-config-changed` 是唯一拦住这个的口子：它把 `.github/**` 与
    `scripts/ci/**` 的改动一律当全量。
    """
    src = (ROOT / "scripts" / "ci" / "select_plan.py").read_text(encoding="utf-8")
    bad: list[str] = []
    if "ci-config-changed" not in src:
        bad.append("select_plan.py 丢了 ci-config-changed 阀门：改选择器不再触发全量")
    for prefix in (".github/", "scripts/ci/"):
        if prefix not in src:
            bad.append(f"select_plan.py 的 ci-config-changed 没覆盖 {prefix}")
    print(f"selector: {len(bad)} 个问题")
    return bad


def check_npu_rule() -> list[str]:
    """「要不要真卡」只能有一份判据，否则漂了不会报错。

    这条规则原先在 tests/conftest.py、select_plan.py、run_tests.sh 各写了一遍
    （都是 `"torch_npu" in ...`）。三份独立实现不会互相报错，只会
    **静默换掉跑的集合**：实测它把仅有注释提及 torch_npu 的 CPU 测试拖上了卡，
    又把要 `fla_npu` / `torch.npu` 的文件漏在两条泳道之外。

    现在判据只在 `scripts/ci/npu_rule.py`，另外三处必须调它。这里扫一遍，
    谁把字符串判据写回去了就红。
    """
    bad: list[str] = []
    rule = ROOT / "scripts" / "ci" / "npu_rule.py"
    if not rule.exists():
        return ["scripts/ci/npu_rule.py 不见了：三处泳道判据会重新各写一遍"]

    consumers = {
        "tests/conftest.py": "npu_rule",
        "scripts/ci/select_plan.py": "npu_rule",
        "scripts/ci/run_tests.sh": "npu_rule.py",
    }
    for rel, marker in consumers.items():
        path = ROOT / rel
        if not path.exists():
            bad.append(f"{rel}: 文件不存在")
            continue
        text = path.read_text(encoding="utf-8")
        if marker not in text:
            bad.append(f"{rel}: 没有调 npu_rule，判据可能又写了一份")
        # 字符串判据只允许出现在 npu_rule.py 里。
        if '"torch_npu" in' in text or "'torch_npu' in" in text:
            bad.append(f"{rel}: 又出现了字符串判据 `torch_npu in ...`，请改调 npu_rule")

    # 规则本身要对得上一份基准：这 5 个例子就是当初判错的用例。
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("ci_npu_rule_check", rule)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        # 注释里提了一句 torch_npu → 纯 CPU，不能占卡
        "tests/models/deepseek_v4/test_turbo_vs_base_precision.py": False,
        # 全程 monkeypatch 假模块
        "tests/ops/test_npu_matmul_add.py": False,
        "tests/models/test_attention_npu.py": False,
        # 只认 torch_npu 会漏掉这两个，它们两条泳道都不跑
        "tests/ops/providers/npu/test_varlen_metadata.py": True,
        "tests/engine/test_muon_offload_distributed.py": True,
        # 标准情况
        "tests/ops/npu/test_fused_rms_norm.py": True,
    }
    for rel, want in expected.items():
        path = ROOT / rel
        if not path.exists():
            bad.append(f"{rel}: 基准用例不存在了，请更新 npu-rule 的样本")
            continue
        got = module.needs_device(path)
        if got != want:
            bad.append(f"{rel}: 判成 {got}，期望 {want}")

    print(f"npu-rule: {len(consumers)} 个调用方 / {len(bad)} 个问题")
    return bad


# 动词在前，读者第一眼就知道这个文件「做什么」。跟 transformers 的
# check_*.py / pytorch 的 win-test.sh 同一个套路。
NAME_PREFIXES = (
    "check", "select", "report", "fetch", "plan",
    "build", "install", "run", "supervise", "env", "npu",
)


def check_naming() -> list[str]:
    """文件名要能自解释，而且不能有没人调用的死脚本。

    两个理由：

    * 文件名是读者唯一一定看到的东西。`lane`、`checks` 这种词是项目黑话，
      三个月后连作者也要点进去才知道它干什么。
    * 下沉到这里的脚本一旦没人调用，就是悄悄腐化的开始 —— 它不在任何 CI 路径
      上，改坏了不会红。这条检查把「删掉」变成默认动作。
    """
    bad: list[str] = []
    scripts = sorted(
        p for p in (ROOT / "scripts" / "ci").iterdir()
        if p.suffix in {".py", ".sh"}
    )

    for path in scripts:
        if path.stem.split("_")[0] not in NAME_PREFIXES:
            bad.append(
                f"scripts/ci/{path.name}: 前缀必须是 "
                + " / ".join(f"{p}_" for p in NAME_PREFIXES)
            )

    # 被谁引用：workflow、其它脚本、或 tests（npu_rule 就是 conftest.py 调的）。
    sources: dict[Path, str] = {}
    for pattern, base in (
        (".github/workflows/*.yml", ROOT),
        ("scripts/ci/*.py", ROOT),
        ("scripts/ci/*.sh", ROOT),
        ("tests/**/*.py", ROOT),
    ):
        for f in base.glob(pattern):
            sources[f] = f.read_text(encoding="utf-8", errors="ignore")

    for path in scripts:
        blob = "\n".join(text for f, text in sources.items() if f != path)
        if path.name not in blob:
            bad.append(f"scripts/ci/{path.name}: 没有任何调用方，删除或接进 CI")

    print(f"naming: {len(scripts)} 个脚本 / {len(bad)} 个问题")
    return bad


def check_interpreter() -> list[str]:
    """`.py` 只能用 python3 调，`.sh` 才能用 bash。

    改名时最容易漏的错：文件从 `lint.sh` 变成 `check_static.py`，调用处只换了
    文件名、没换解释器 —— `bash x.py` 在 YAML 里合法、actionlint 也不管，只能
    等 CI 真跑到那一步才红，而且报的是看不懂的 bash 语法错。
    """
    bad: list[str] = []
    # `bash xxx.py` / `sh xxx.py` / `source xxx.py`
    wrong = re.compile(
        r"(?:^|[\s;&|(])(?:bash|sh|source|\.)\s+[\w\"'$/{}.\-]*?([\w\-]+\.py)\b",
        re.MULTILINE,
    )
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        rel = path.relative_to(ROOT).as_posix()
        for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for m in wrong.finditer(line):
                bad.append(
                    f"{rel}:{num}: 用 bash/sh 调 {m.group(1)}，应改用 python3"
                )
    print(f"interpreter: {len(bad)} 个问题")
    return bad


CHECKS = {
    "compile": check_compile,
    "workflows": check_workflows,
    "map": check_model_map,
    "contract": check_contract,
    "selector": check_selector_guard,
    "npu-rule": check_npu_rule,
    "naming": check_naming,
    "interpreter": check_interpreter,
}


# ---------------------------------------------------------------- hadolint
HADOLINT_VERSION = os.environ.get("HADOLINT_VERSION", "v2.15.1")


def cmd_hadolint(dockerfile: str) -> int:
    """二进制而不是 docker run：镜像只有 amd64，runner 默认 arm64。"""
    arch = {"aarch64": "arm64", "arm64": "arm64",
            "x86_64": "x86_64", "amd64": "x86_64"}.get(platform.machine())
    if not arch:
        print(f"::error::unsupported arch {platform.machine()}")
        return 1
    url = (f"https://github.com/hadolint/hadolint/releases/download/"
           f"{HADOLINT_VERSION}/hadolint-linux-{arch}")
    bin_path = "/tmp/hadolint"
    try:
        urllib.request.urlretrieve(url, bin_path)
    except Exception as exc:                       # noqa: BLE001
        print(f"::error::下载 hadolint 失败（{arch}）：{exc}")
        return 1
    os.chmod(bin_path, 0o755)
    # --failure-threshold error：只拦解析类问题，不拦「建议钉版本」这类风格告警。
    return subprocess.run(
        [bin_path, "--failure-threshold", "error", dockerfile], check=False
    ).returncode


# ---------------------------------------------------------------- stage
def cmd_stage() -> int:
    """select 的收尾。必须 always()：失败之后 job status 才是 failure，
    这一步正是用来分辨「静态挂了」和「计划挂了」。"""
    def env(name: str) -> str:
        return os.environ.get(name, "")

    if env("JOB_STATUS") == "cancelled":
        stage = "cancelled"
    elif env("STATIC_OUTCOME") != "success":
        stage = "static"
    elif env("PICK_OUTCOME") != "success":
        stage = "select"
    else:
        stage = "ok"

    path = env("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"stage={stage}\n")
    print(f"stage: {stage}")
    return 0


def main(argv: list[str]) -> int:
    if argv and argv[0] == "hadolint":
        return cmd_hadolint(argv[1] if len(argv) > 1
                            else "docker/npu/Dockerfile_aarch64")
    if argv and argv[0] == "stage":
        return cmd_stage()

    argv = [a for a in argv if a != "static"]      # `static` 是默认，可写可不写
    # 训练侧只需要 map 那一项（它不跑全仓编译），其余调用方不带参数跑全部。
    only = ""
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]

    picked = [CHECKS[only]] if only in CHECKS else list(CHECKS.values())
    problems = [p for check in picked for p in check()]
    if not problems:
        print("Lint OK")
        return 0
    for p in problems:
        print(f"::error::{p}")
    print(json.dumps({"problems": len(problems)}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
