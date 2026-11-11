# CI 设计：镜像与资源决定 job，所有 Python 依赖进 venv

**一句话**：job 只看 `(image, resource)`；unit 和 e2e 的 Python 依赖都装进
各自的 venv，在 job 内按 step 顺序执行。依赖版本不产生 lane，不产生 job，
也不产生 marker。

---

## 选型：主流做法与 TorchTurbo 的选择

主流训练框架通常采用 **job / container 级别隔离依赖**：

```text
一个 job = 一套硬件 + 一套 ABI + 一套 Python 依赖
一个 job 内跑多个模型测试
```

模型只是测试参数，不是环境边界。不同 CUDA / torch / transformers 版本才
拆镜像或 job；模型数量增长不会直接增加 job 数。

TorchTurbo 的情况不同：

- 11 个模型族，当前测试集中在 `tests/models/<family>/`；
- 同时存在 default 5.5.4、Swift 5.9.0、qwen4_exp 5.22 三套 Python 依赖；
- LF / Swift 是两套真实训练框架，不能简单共用一个全局环境；
- 8 卡 NPU runner 稀缺，按 env 拆 job 会让同一批 e2e 排队多次。

如果完全照搬主流做 job-per-env，会得到：

```text
npu8 / default
npu8 / swift
npu8 / qwen4_exp
```

这不是 11 个模型 job，但也意味着三倍的 8 卡排队时间。TorchTurbo 的最终
选择是：

```text
image   隔离 ABI（CANN / torch / torch_npu / 编译扩展）
resource 隔离硬件（CPU / NPU-2 / NPU-8）
venv    隔离纯 Python 依赖（transformers / peft / LF / Swift）
process 隔离模型族的进程级全局状态
```

| 差异 | 主流训练框架 | TorchTurbo |
|---|---|---|
| CUDA / CANN / torch ABI | image / job | image / job |
| GPU / NPU 卡数 | job | job |
| Python 依赖版本 | 通常一个 job 一套 | 同一 job 内多个 venv |
| 不同模型 | 测试参数 | 测试参数，不建模型 lane |
| 进程全局状态 | 测试内处理 | 按模型族分进程 |

这不是发明新调度模型，而是针对 TorchTurbo“模型多、Python pin 多、8 卡
资源少”的折中：比 per-model 镜像轻，比 job-per-env 省机器。

唯一的回退条件：如果某个 e2e 在 venv 下已验证不可行，只对该类 e2e 回退到
job-per-env；不允许整个 CI 退回全局 pip 或按模型建 lane。

---

## 0. 最终边界

CI 只保留四个正交维度：

| 差异 | 放在哪 | 为什么 |
|---|---|---|
| CANN、torch、torch_npu、编译扩展 ABI | **image → job** | 不能靠 venv 切换 |
| CPU / NPU 卡数 | **resource → job** | 机器不可虚拟化 |
| transformers、peft、LF、Swift 等纯 Python 依赖 | **env → venv step** | venv 可隔离，30–60 秒 |
| 进程级全局状态 | **isolation unit → process** | 按模型族分进程即可 |
| 用例是否需要 NPU / 多卡 | **pytestmark** | 一行代码，最便宜 |

核心规则只有一条：

```text
能进 venv 的差异，绝不升级成 job；
不能进 venv 的差异，才升级成 image 或 resource。
```

因此 **e2e 也使用 venv**，不再把“e2e 依赖环境”放进 job key。这样 unit、
integration、e2e 只要镜像和资源相同，就可以在同一个 job 内顺序执行，
不会因为 LF / Swift / qwen4_exp 的 Python 版本不同而拆机器。

---

## 1. L0：事实层

### 1.1 测试标记

`pytest.ini` 只保留 6 个 marker：

```ini
markers =
    cpu          # 不需要加速器
    npu          # 需要至少 1 张 Ascend
    rank_2       # 需要 2 个进程 / 2 张卡
    rank_4       # 需要 4 个进程 / 4 张卡
    rank_8       # 需要 8 个进程 / 8 张卡
    integration  # 需要真实框架、权重、数据集或网络
```

规则：

1. `cpu` / `npu` 必填且只能有一个；
2. `rank_N` 默认 `1`，只在 N > 1 时写；
3. `integration` 只表示外部资产或真实框架依赖，不表示版本；
4. 删除 `multi_npu`、`npu_4/8/16`、`slow`、`cuda`、`xpu`、`mps`；
5. 有 `mp.spawn` / `init_process_group` 且最大 world size > 1 时，必须
   标 `rank_N`，否则 lint fail。

`rank_N` 是**最低资源需求**，不是 job 名。调度时选择能容纳它的最小 lane。

### 1.2 依赖环境

新增 `scripts/ci/envs.toml`。它是唯一的 Python 依赖事实源：

```toml
[envs.default]
python = "3.11"
requirements = "scripts/ci/envs/default.txt"
constraints = "auto"

[envs.qwen4_exp]
python = "3.11"
requirements = "scripts/ci/envs/qwen4_exp.txt"
constraints = "auto"

[envs.lf]
python = "3.11"
requirements = "scripts/ci/envs/lf.txt"
constraints = "auto"

[envs.swift]
python = "3.11"
requirements = "scripts/ci/envs/swift.txt"
constraints = "auto"

[envs.qwen4_exp_swift]
python = "3.11"
extends = "swift"
requirements = "scripts/ci/envs/qwen4_exp_swift.txt"
constraints = "auto"
```

`constraints = "auto"` 表示安装前根据当前 lane 生成约束：

- NPU lane：从镜像里的 torch / torch_npu / numpy 版本生成，防止 pip 替换；
- CPU host lane：约束 CPU torch，版本固定为 `2.10.0`。

```text
# NPU lane 自动生成
torch==2.10.0
torch_npu==<image-version>
numpy==<image-version>
```

`envs.toml` 中的 requirements 必须是可复现的纯 Python 依赖集合。LF / Swift
原来的 `install_*deps.sh` 可以保留为生成脚本，但 CI 执行时不再直接运行
任意 shell，也不在全局 Python 里 `pip install`。

每个 env 的缓存键：

```text
sha256(python + requirements + constraints + image digest)
```

env 名称稳定（`default` / `qwen4_exp` / `lf` / `swift`），依赖变化只让缓存失效，
不产生新的 job。

### 1.3 模型到 env

新增 `scripts/ci/model_env.toml`，只服务 unit / integration 测试，放模型族与
env / image 的映射：

```toml
[models.qwen4_exp]
env = "qwen4_exp"
image = "default"      # 只有 ABI 不同才覆盖
```

未声明的模型使用 `env = "default"`、`image = "default"`，不需要重复登记。

E2E 不使用这张表选 env：同一个模型可能同时有 LF 和 Swift，qwen4_exp+Swift 的
transformers pin 也可能与普通 Swift 不同，因此 e2e 的 env 必须写在 case 上。

env 判定看**显式 import 闭包**，不看目录名：

```text
families = 文件中显式 import 的 torchturbo.models.<family>
if families 为空                  -> env = default
elif 所有 family 的 env 相同      -> 该 env
else                              -> lint fail：拆文件
```

`integration` 文件另加直接 import 判定：

```text
import llamafactory -> lf
import swift/msswift -> swift
两者都有             -> lint fail：拆文件
动态 import 不在静态闭包     -> 新增时补显式 test env override
```

跨框架 parity 测试可写显式 override，但必须在同一条记录旁说明覆盖损失：

```toml
[tests."tests/models/bailing_moe_v3_vl/test_lf_image_parity.py"]
env = "lf"  # Swift baseline skips when swift is absent
```

实测：162 个测试文件中，模型目录外有 77 个；其中 26 个显式 import
`torchturbo.models`，3 个显式 import qwen4_exp，跨两个 pin 的文件为 0。因此
import 闭包足够轻，不需要为每个模型建 lane。

进程隔离单元不能只看路径：模型目录外的文件也可能 import deepseek_v4 等
带全局 patch 的模型。规则是：

```text
tests/models/<family>/...          -> family
tests/models/test_*.py             -> layer:<tests/models>
其他文件且 import 了 model family  -> import:<family>
其他文件且没有 model import        -> 测试层目录
```

这样不会把 `tests/engine` 中 import deepseek_v4 的文件和普通 engine 测试
放进同一个进程。

### 1.4 镜像与 lane

新增 `scripts/ci/lanes.yml`。lane 是唯一能声明 runner 的地方：

```yaml
lanes:
  default/cpu-1:
    runner: ubuntu-22.04-arm
    image: default
    device: cpu
    slots: 1
  default/cpu-4:
    runner: ubuntu-22.04-arm
    image: default
    device: cpu
    slots: 4
  default/cpu-8:
    runner: ubuntu-22.04-arm
    image: default
    device: cpu
    slots: 8
  default/npu-2:
    runner: linux-aarch64-a3-800i-2
    image: default
    device: npu
    slots: 2
  default/npu-8:
    runner: linux-aarch64-a3-800i-8
    image: default
    device: npu
    slots: 8
```

以后加 CANN / torch 镜像，只追加 `image/device/slots` 不同的 lane；workflow
本身不改。找不到满足 `(image, device, slots)` 的 lane，plan 直接 fail。
`image` 是别名，plan 开始时用现有 `build_image.py resolve` 逐个解析成真实
引用；新增镜像不改变 workflow。当前 `resolve` 只支持一个 artifact，落地
多镜像时要增加 alias -> artifact/tag 的映射；这是 image CI 的扩展点，不是
plan 的例外。

### 1.5 E2E case

`.github/workflows/model_map.json` 继续声明模型到 e2e case 的映射，但 case
只保留执行所需事实：

```json
{
  "models": {
    "qwen3_5_moe": {
      "run_dir": "qwen3_5_moe",
      "cases": [
        {
          "id": "qwen3_5_moe-lf-text-sft",
          "framework": "lf",
          "env": "lf",
          "image": "default",
          "install": "install_lf_deps.sh",
          "train": "lf_text_sft.sh",
          "download": {"hf_repo": "Qwen/Qwen3.5-35B-A3B", "layers": 6},
          "dataset": "gsm8k",
          "cards": 8, "sp": 2, "ep": 4,
          "tier": "pr"
        },
        {
          "id": "qwen3_5_moe-swift-mllm-sft",
          "framework": "swift",
          "env": "swift",
          "image": "default",
          "install": "install_swift_deps.sh",
          "train": "swift_mllm_sft.sh",
          "download": {"hf_repo": "Qwen/Qwen3.5-35B-A3B", "layers": 6},
          "dataset": "gsm8k",
          "cards": 8, "sp": 2, "ep": 4,
          "tier": "nightly"
        }
      ]
    }
  }
}
```

`cards`、`sp`、`ep` 是硬需求；`cards` 映射到 resource，case `env` 映射到
venv，`framework` 只用于影响面选择和报告。`run_dir` 为空、`install` /
`train` 脚本不存在、`env` / `image` / `tier` 不合法、`download.hf_repo`
缺失或 `(model, framework)` 有多个 `tier=pr` 的记录必须 lint fail。

`tier` 只有两个值：

```text
pr        PR 的代表用例；每个 (model, framework) 最多一个
nightly   只在 nightly / ci:full 运行
```

PR 的选择规则不是“数组第一个”：

1. 先按 changed paths 找受影响的 model / framework；
2. 受影响的 case 即使标了 `nightly`，PR 也优先运行它；
3. 没有精确命中 case 时，只为每个受影响的 model 选一个 `tier=pr` 的
   最便宜代表；
4. 模型共享代码同时影响多个框架时，PR 只跑一个代表；nightly 跑全部；
5. 没有 `tier=pr` 代表的模型允许存在（nightly-only 或未就绪），但 PR
   只会在精确命中该模型 / 框架时运行它的 case。

影响面还要细化到 case / env，不能只落到 model：

```text
scripts/runs/<run_dir>/<train>.sh  -> 该 train 对应的 case
scripts/runs/<run_dir>/<other>     -> 该 run_dir 的一个 PR 代表
scripts/runs/<run_dir>/            -> 该 run_dir 的所有 case，按 tier 选
src/torchturbo/adapters/<framework>/ -> 该 framework 的一个 pr 代表
scripts/ci/envs/<env>.txt          -> 使用该 env 的一个 pr 代表
scripts/ci/envs.toml | common.txt  -> 全部 env 的一个最便宜代表
.github/workflows/model_map.json   -> E2E 的一个最便宜代表
src/torchturbo/models/<family>/    -> 该 model 的一个 pr 代表
```

环境文件变化不能悄悄只跑一个模型而不报告；日报里要写清楚“env 变化，
PR 只跑代表，nightly 跑全量”。

这样 PR 是 smoke，nightly 是 full matrix，但 PR 不会因为数组顺序变化而
漏掉真正改到的 Swift/LF 用例。

---

## 2. L1：plan 编译层

新增 `scripts/ci/plan.py`，纯函数，无副作用。**不要继续膨胀现有的
`select_tests.py`**；它继续只负责“改了什么 → 选中哪些测试”，plan 负责
“选中的测试 → 怎么跑”。

```text
for f in selected_files:
    resource = smallest_lane(markers(f))       # cpu/npu + rank_N
    image    = image_of(import_closure(f))     # 默认 default
    env      = env_of(import_closure(f))       # 默认 default
    process  = isolation_unit(f)               # family / import:<family> / layer

for case in selected_e2e:
    resource = smallest_lane(device=npu, slots=case.cards)
    image    = image_of(case)
    env      = case.env
    process  = case.id

jobs = group_by((image, resource)):
    steps = group_by(env):
        pytest groups ordered by isolation unit
        e2e cases ordered by case id
```

plan 输出 JSON，workflow 只消费它：

```json
{
  "jobs": [{
    "id": "default/npu-8",
    "runner": "linux-aarch64-a3-800i-8",
    "image": "default",
    "resource": "npu-8",
    "steps": [
      {"env": "default", "kind": "pytest", "device": "npu", "rank": 2,
       "process": "family:qwen3_5", "files": ["tests/models/qwen3_5/test_model.py"]},
      {"env": "lf", "kind": "e2e", "cases": [{"case_id": "qwen3_5_moe-lf-text-sft"}]},
      {"env": "swift", "kind": "e2e", "cases": [{"case_id": "qwen3_5_moe-swift-mllm-sft"}]}
    ]
  }]
}
```

调度规则：

1. `rank_2` 优先落 `npu-2`，没有 2 卡 lane 时才落 `npu-8`；
2. `cards=8` 只能落 `npu-8`；
3. 同 `(image, resource)` 的 unit / integration / e2e 进同一个 job；
4. job 内先 unit，再 integration，最后 e2e；失败立即停止；
5. selected 为空时不生成 job。

这样 job 数只由 `(image, resource)` 决定，与模型数、env 数、框架数无关。
唯一的例外是 nightly 的 e2e 分片：nightly 可以把每个 e2e case 作为同一
lane 下的独立 job，以便并行和单独超时；分片不改变 image、resource 或 env。

---

## 3. L2：执行层

### 3.1 workflow 只保留两类 runner

```text
plan              select + lint + 生成 plan.json
run-host          matrix: cpu lane（无 container）
run-container     matrix: npu lane（container.image 来自 lane）
gate              校验 plan、结果、skip 白名单
```

两个 run job 都是通用模板：

```yaml
strategy:
  matrix:
    include: ${{ fromJSON(needs.plan.outputs.host_jobs) }}
runs-on: ${{ matrix.runner }}
# container 版本再加：
container:
  image: ${{ matrix.image }}
```

每个 matrix job 只执行一个 `scripts/ci/run_job.py --job-id <id>`。脚本按
plan 里的 `steps` 顺序执行，不再为 cpu / npu / e2e 写三套 YAML。

### 3.2 venv 执行

```text
job = (image, resource)
  └─ step = env
       ├─ create/reuse venv
       ├─ install requirements + constraints
       ├─ verify torch / torch_npu / numpy unchanged
       └─ run pytest groups 或 e2e cases
```

venv 关键参数：

```text
NPU:  python -m venv --system-site-packages <cache>/venv-<hash>
CPU:  python -m venv <cache>/venv-<hash>
export VIRTUAL_ENV=<cache>/venv-<hash>
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
pip install -r <requirements> -c <constraints>
pip install --no-build-isolation --no-deps -e .
```

- NPU lane 的 torch / torch_npu / CANN 从基础镜像继承；
- CPU host lane 没有基础镜像，必须在 venv 内安装 `torch==2.10.0`；
- 每个 env 显式安装自己的 transformers / peft / LF / Swift；
- NPU lane 安装后调用 `env_pod.py check`，torch / torch_npu / numpy 被替换
  即 fail；CPU lane 使用同一脚本的 CPU 分支，只校验 torch 版本，不能执行
  NPU 的 `/dev/davinci*` 和 `torch_npu` 检查；
- venv 按内容 hash 放在专用 cache 目录，PR 之间复用；只创建 / 复用，不在
  CI 中执行 `rm -rf`。缓存缺失时新建目录，不做清理式回收。

依赖版本和安装顺序以 `scripts/runs/deps/*.sh` 为准。训练脚本有意采用
“先安装框架、再覆盖某个依赖”的顺序时，CI 必须复现同一顺序，不能把
requirements 当成一次 pip 解析事务。例如 Swift 先安装 `ms_swift`，随后
按训练脚本覆盖为 `datasets==5.0.1`；该组合的包元数据并不互相满足，但
它是实际训练环境。

### 3.3 e2e 也在 venv 内跑

e2e 脚本中的 `pip`、`python`、`swift`、`llamafactory` 都由 `PATH` 指向
venv。对 `torchrun` 增加一个 5 行 wrapper：

```bash
#!/usr/bin/env bash
exec "$VIRTUAL_ENV/bin/python" -m torch.distributed.run "$@"
```

这样 LF 的 `torchrun` 不会回退到系统 Python；Swift 的 `${PYTHON}` 也会
自然使用 venv。CANN 环境仍由基础镜像和训练脚本提供，不进入 venv。

禁止在 CI 中修改全局 `site-packages`、`~/.cache`、`/root/.cache` 或任何
宿主共享目录。需要清理的输出必须落在 `$RUNNER_TEMP`。

### 3.4 进程隔离

同一 env 内按 `tests/models/<family>` 分进程执行。原因不是版本，而是业务
代码存在进程级全局副作用：

```python
# src/torchturbo/models/deepseek_v4/model.py:167
MoELayer.forward = deepseek_moe_forward   # 类级覆盖，无 restore
```

实测：`tests/models/test_moe_probability_weighting.py` 单独跑 35/35 通过，
与上述文件同进程跑 31 个失败。按模型族隔离是 CI 的兜底，不要求业务代码
在本次重构中修改。

---

## 4. L3：判定层

plan 阶段 fail：

1. 测试缺 `cpu` / `npu`，或同时有两个 device marker；
2. `rank_N` 不合法，或多于一个；
3. 多进程测试没有 rank 标记；
4. import 闭包跨 env，或动态 import 无法归类；
5. `integration` 同时 import LF 和 Swift，且没有显式 test env override；
6. 模型 / e2e 引用了不存在的 env、image、lane；
7. e2e 缺 `cards`、`sp`、`ep`、`train`；
8. 同一个 job 内出现重复 isolation unit。

执行阶段 fail：

1. env 安装失败；
2. torch / torch_npu / numpy 被 pip 替换；
3. 任一 pytest group 或 e2e case 失败；
4. job 超时，或 NPU 子进程未清理；
5. skip 原因不在该 lane 白名单。

聚合 gate 只读 plan 和每个 job 的结果，不再重新判断硬件或依赖。

---

## 5. 直接重构顺序

### PR 1：事实和计划（不改变执行）

- 加 `envs.toml`、`model_env.toml`、`lanes.yml`；
- 加 `plan.py` 和 lint；
- 替换 `pick_e2e` 的旧 `run_dir/e2e` 解析，改为 case 级 `env/tier/framework`；
- 补 device / rank 标记，迁移旧 `multi_npu`；LF+Swift parity 文件暂用显式
  env override，Swift baseline 缺失时保持 skip；
- 现有 workflow 不变，plan 以 shadow 模式输出摘要，验证选中集合。

### PR 2：CPU 通用 runner

- 加 `run_job.py`、venv 安装和 `torchrun` wrapper；
- `env_pod.py` 增加 CPU 分支，CPU lane 不依赖 torch_npu / davinci 节点；
- `cpu-unit-tests` 改为 matrix + plan；
- 保留 NPU 两条旧 job，形成可回退路径。

### PR 3：NPU 与 e2e 合并

- `npu-unit-tests` / `npu-e2e-tests` 改为 `run-container` matrix；
- e2e install 脚本迁到 env requirements，在 venv 内执行；
- 删除 workflow 里写死的 `FRAMEWORK: lf`，framework 从 case 读取；
- 多镜像 alias 接入 `build_image.py resolve`，每个 image 单独解析 artifact；
- 删除旧 `l1/l2/l3` 分支逻辑和固定 lane 输出；
- gate 改为消费 plan + 结果汇总。

每个 PR 都能独立回退，不需要一次性切换整条流水线。

---

## 6. 代码量

按当前落地结果估算，最终净增约 **1,200–1,500 行**；其中大部分是事实校验和
失败路径，不是调度框架：

| 项 | 量 |
|---|---:|
| `facts.py` + `plan.py` + lint / self-check | 850–1,000 |
| `run_job.py` + venv 工具 | 150–250 |
| workflow matrix 重构 | 80–150 |
| `envs.toml` / `model_env.toml` / `lanes.yml` / requirements | 60–100 |
| marker 迁移 | device / rank 标记补齐 + 3 个旧标记迁移 |

不新增服务、数据库、调度器或 DSL；只新增一个纯函数计划和两个通用模板。

---

## 7. 为什么能持续演进

| 变化 | 改动 |
|---|---|
| 新增测试 | 加 marker；plan 自动分配 |
| 新增模型、沿用版本 | 只加 `tests/models/<family>/` |
| 模型换 transformers | `model_env.toml` 加 1 行，env 自动切 venv |
| 新框架 | 加 env requirements；不新增 job 类型 |
| 新 e2e case | 改 `model_map.json`；不改 workflow |
| 新 NPU 规格 | `lanes.yml` 加一条；plan 自动选最小 lane |
| 新 CANN / torch 镜像 | 加 image 和 lane；workflow 不改 |
| 依赖升级 | 改 requirements，缓存键变化，不影响 job 图 |

CI 的 job 图只随镜像和硬件增长；测试、模型、框架和 Python 依赖的增长都
被吸收在 plan、venv 或 process 内。

---

## 8. 非目标

- 不按模型建 lane、profile 或镜像；
- 不把依赖版本写成 marker；
- 不做跨 transformers 版本的兼容层；
- 不让 CI 在全局 Python 里安装 LF / Swift；
- 不在本次改造中修改业务代码。
