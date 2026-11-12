# CI 设计：事实决定资源，设备工作集决定 job

**一句话**：测试的设备事实决定 lane，本次选中工作的最大资源需求决定 lane 容量；
unit 和每个 E2E case 都使用独立 job。marker 只做 pytest 选测和事实一致性校验，
不作为调度事实的唯一来源。

---

## 0. 选择

主流训练框架通常用粗粒度资源池：

```text
cpu / npu / multi-npu / gpu / multi-gpu
```

测试内部通过 `mp.spawn(nprocs=N)`、`world_size`、启动器参数或 recipe 决定精确
进程数。TorchTurbo 沿用这个方向，但增加两项约束：

- 代码事实先于 marker：不能只靠人工标签判断设备；
- 同一设备、同一阶段的工作先聚合，再选择容量 lane，不能按 `rank_N` 拆多个 job。

TorchTurbo 需要在同一 job 内隔离多套 Python 依赖，因此保留 venv：

```text
image   隔离 CANN / torch / torch_npu / 编译扩展 ABI
device  隔离 CPU / NPU；GPU 事实保留，暂不调度
slot    选择能容纳本次工作集的容量
venv    隔离 transformers / peft / LF / Swift
process 隔离模型族的进程级全局状态
```

依赖版本不产生 job；模型数量不产生 job；`rank_N` 不产生 job。

---

## 1. 事实层

### 1.1 marker

目标 marker 只有以下五类设备标记，另加 `integration`：

```ini
markers =
    cpu
    npu
    multi_npu
    gpu
    multi_gpu
    integration
```

Pytest marker 必须是合法 Python 标识符，代码中使用 `multi_npu` / `multi_gpu`；
文档和展示可以写 `multi-npu` / `multi-gpu`。

语义：

| marker | 事实含义 |
|---|---|
| `cpu` | 不执行加速器代码 |
| `npu` | 至少 1 张 NPU，单进程 |
| `multi_npu` | 至少 2 张 NPU，具体 `processes` 来自事实 |
| `gpu` | 至少 1 张 GPU，单进程 |
| `multi_gpu` | 至少 2 张 GPU，具体 `processes` 来自事实 |

规则：

1. marker 必须由代码事实生成或校验，不能直接相信历史标签；
2. `multi_npu` / `multi_gpu` 只表示多设备，不携带 2/4/8 的 job 语义；
3. CPU 多进程不新增 `multi_cpu`，仍标 `cpu`，进程数写进事实；
4. 一个测试同时支持 NPU/GPU 时，可同时声明对应设备 marker；
5. 无法从代码证明设备的测试进入 `unknown`，必须有显式事实覆盖，不能回退到旧标签。

### 1.2 事实来源

事实扫描不导入测试模块，按以下证据收集：

| 事实 | 强证据 |
|---|---|
| CPU | `CpuRuntime` 继承、`device="cpu"`、纯 CPU tensor |
| NPU | `torch.npu`、`.npu()`、`.to("npu")`、HCCL、`fla_npu` |
| GPU | `torch.cuda`、`.cuda()`、`.to("cuda")`、NCCL |
| 多进程 | `mp.spawn(nprocs=N)`、`world_size=N`、`init_device_mesh((N,))` |
| 动态设备 | `get_device_runtime()`、`runtime.device` |
| 外部资产 | `integration` |

证据分为：

```text
executes  测试实际分配或执行 tensor 的设备
configures 只把 Runtime 传给 dispatcher，不实际执行设备代码
guards    is_available()/skipif，仅表示能力检查
launches  mp.spawn/init_process_group/recipe cards
```

只有 `executes` 和 `launches` 参与资源判定；`configures` 不能把一个纯配置
测试升级成 GPU/NPU job。例如 `SimpleNamespace(device=torch.device("cuda"))`
只是 mock 配置，不计入 GPU 证据。`guards`（`is_available()` / `skipif`）不能
替代设备事实。

动态设备测试在 NPU lane 上按 NPU 执行。事实层记录 GPU 证据，但当前没有 GPU
runner，因此不生成 GPU lane 或 GPU job。

### 1.3 当前测量

按当前代码事实：

- CPU unit 共 150 个测试文件，按一个 `cpu` job 聚合；最大的进程事实为 8。
- NPU unit 共 18 个测试文件，最大需要 2 卡。
- GPU unit 当前有 1 个测试文件，仅保留事实，暂不进入 CI 调度。
- `tests/engine/backends/fsdp2/test_adamw_swap_checkpoint.py` 和
  `test_cpu_offload_optimizer.py` 使用 `pin_memory=True`，需要 NPU 提供 pinned
  allocator，应归 NPU。
- e2e case 的卡数来自 `.github/workflows/model_map.json`；当前声明的 case 均为
  `cards=8`，某次 CI 可能只选中其中一部分。
- 没有 NPU unit 用例需要 4 卡或 8 卡；4/8 卡只来自 E2E `cards`。

这些数字是事实样例，不是永久常量。plan 每次根据实际选中集合重新计算。

### 1.4 E2E triggers

每个 e2e case 用 `triggers` 声明参与的阶段：

```json
"triggers": ["pr", "nightly"]
```

- 缺省或空列表表示不自动触发；
- PR 选择所有路径命中且包含 `pr` 的 case，不再只取一个代表；
- full/nightly 选择所有包含 `nightly` 的 case；
- 同一模型/框架可以声明多个 `pr` case，成本由配置者通过 trigger 控制。

---

## 2. lane 与 job

### 2.1 lane

`scripts/ci/lanes.yml` 是 runner 的唯一事实源：

```yaml
lanes:
  default/cpu:
    image: default
    device: cpu
    slots: 8          # internal process capacity; never shown as cpu-8
    resource: cpu
    runner_variable: CI_CPU_RUNNER
    runner: ubuntu-22.04-arm

  default/npu-2:
    image: default
    device: npu
    slots: 2
    resource: npu-2
    runner_variable: CI_NPU_RUNNER
    runner: linux-aarch64-a3-800i-2

  default/npu-8:
    image: default
    device: npu
    slots: 8
    resource: npu-8
    runner_variable: CI_E2E_RUNNER
    runner: linux-aarch64-a3-800i-8
```

规则：

- CPU 的 lane/job 名称固定为 `default/cpu`，没有数字后缀；
- NPU 的容量是实际硬件属性，lane 名称保留容量；
- 同一 `(image, device, runner)` 不允许只靠 `slots` 再拆一条 lane；
- GPU 当前只保留 marker 和事实；接入 runner 后才新增 `gpu-1` / `gpu-N` lane，
  规则与 NPU 相同。

### 2.2 工作集聚合

plan 不使用单条测试的 rank 直接选 job，而是按 `(image, device, stage, work_id)`
聚合。unit 的 `work_id` 为空；E2E 的 `work_id` 是 case id：

```text
work[image, device, unit, ""] = {
    pytest groups: key=(env, process isolation unit)
}
work[image, device, e2e, case_id] = {
    e2e cases: key=env
}
```

每个工作集独立计算容量并选择能容纳它的最小 lane：

```text
unit required_slots = max(所有 unit 的 processes)
e2e  required_slots = max(所有 e2e 的 cards)

required_slots <= 2  -> default/npu-2
required_slots <= 8  -> default/npu-8
required_slots > 8   -> plan fail
```

因此 PR 同时选中 NPU unit 和两个 `cards=8` 的 e2e 时，会输出三个 matrix job：

```text
default/npu-2                              stage=unit
default/npu-8/qwen3_5-lf-text-sft         stage=e2e
default/npu-8/qwen3_5_moe-lf-text-sft     stage=e2e
```

所有 job 都只依赖 `select`，由 workflow matrix 启动。每个 E2E case 是独立的
matrix job，也就是独立的 pod；NPU matrix 使用 `max-parallel: 2`，最多同时占用
两个 NPU job。NPU unit 可能占用其中一个槽位，但模型之间不会串行进入同一个 pod，
模型训练状态也不会跨 case 泄漏。

不同 image 仍可生成不同 job，因为镜像是 ABI 边界。

---

## 3. CPU / NPU

### 3.1 CPU

- 所有 CPU unit 进入 `default/cpu` 一个 job；
- `processes=2/4/8` 只影响 job 内进程数和执行时间；
- CPU 不为 `processes` 选择不同 lane；
- env 差异由 venv 隔离，process isolation unit 决定 pytest 分组。

### 3.2 NPU

- `npu` / `multi_npu` 统一进入 NPU 工作集；
- unit 使用一个 job；每个 E2E case 使用一个独立 job，各自选 `npu-2` 或 `npu-8`；
- 只选 unit 且最大 `processes=2` 时用 `npu-2`；
- 选中任意 `cards=8` 的 e2e 时用 `npu-8`；
- e2e 的 `cards/sp/ep` 来自 `model_map.json`，不从 marker 或 runner 标签推断；
- NPU matrix 最多并发 2 个 job；每个 E2E job 运行一个 case 并使用独立 pod 状态。

pytest 选测：

```text
pytest -m "npu or multi_npu"
```

### 3.3 GPU

- 当前没有可用 GPU runner，因此不声明 `gpu_unit`、不生成 GPU job；
- `gpu` / `multi_gpu` marker 和事实扫描保留，避免把 GPU 用例误归到 CPU/NPU；
- 接入 runner 后扩展 selector 和 workflow，复用同一聚合规则：

```text
pytest -m "gpu or multi_gpu"
```

---

## 4. plan 输出

plan 只输出工作，不输出“设备 + rank”组合：

```json
{
  "jobs": [
    {
      "id": "default/npu-2",
      "image": "default",
      "device": "npu",
      "stage": "unit",
      "resource": "npu-2",
      "slots": 2,
      "required_slots": 2,
      "runner": "linux-aarch64-a3-800i-2",
      "steps": [
        {"kind": "pytest", "env": "default", "marker": "npu",
         "process": "layer:tests/ops/npu", "processes": 2,
         "files": ["tests/ops/npu/test_fused_rms_norm.py"]}
      ]
    },
    {
      "id": "default/npu-8/qwen3_5-lf-text-sft",
      "image": "default",
      "device": "npu",
      "stage": "e2e",
      "resource": "npu-8",
      "slots": 8,
      "required_slots": 8,
      "runner": "linux-aarch64-a3-800i-8",
      "steps": [
        {"kind": "e2e", "env": "lf",
         "cases": [{"case_id": "qwen3_5-lf-text-sft"}]}
      ]
    }
  ]
}
```

输出给 workflow 的矩阵：

```text
cpu_jobs  / cpu_job_count    CPU jobs
npu_jobs  / npu_job_count    NPU jobs
```

选中为空时不生成 job；worker 不启动。

---

## 5. 执行层

### 5.1 通用 runner

```text
select        lint + select + plan，输出 cpu_jobs / npu_jobs
l1_cpu_unit   matrix: CPU jobs, host runner
l2_npu        matrix: NPU jobs, container image from plan
gate          按 CPU/NPU 结果判定
```

workflow 不再固定写 `npu-unit-tests` 和 `npu-e2e-tests`。NPU unit/e2e 是同一
matrix 中的独立 plan job；每个 E2E case 一个 job，`max-parallel: 2`。

### 5.2 venv

每个 env 在 job 内创建或复用 venv：

```text
job = (image, device, stage, work_id)
  └─ step = env
       ├─ create/reuse venv
       ├─ install env requirements + image constraints
       ├─ verify torch / torch_npu / numpy unchanged
       └─ run pytest group or e2e cases
```

- NPU venv 使用 `--system-site-packages`，继承镜像中的 CANN/torch/torch_npu；
- CPU venv 自行安装固定 torch；
- 不在全局 Python 安装 LF/Swift，不修改共享 cache；
- e2e 的 `pip`、`python`、`torchrun` 通过 `PATH` 指向当前 venv；
- 同一 env 的多个 pytest group 复用 venv。

### 5.3 失败与超时

- 同一 job 内 step 失败立即停止；unit 失败不会阻塞独立的 e2e job；
- 每个 e2e case 独占 job，模型训练不会污染下一个模型；
- 每个 e2e job 使用独立超时和 cleanup；
- e2e 的 `fetch_timeout` / `case_timeout` 统一由 `lanes.yml` 的 `e2e_defaults` 提供；
- NPU job 超时或取消后必须执行 environment cleanup，释放卡和子进程；
- 容量不足是 plan fail，不允许静默使用更大/更小 lane。

---

## 6. 判定层

plan 阶段 fail：

1. 测试没有设备事实，或 marker 与事实不一致；
2. 多进程事实没有 `multi_npu` / `multi_gpu`；
3. `multi_npu`/`multi_gpu` 声明与 `processes` 不一致；
4. 同一测试同时声明 CPU 和加速器且没有显式事实覆盖；
5. env/image/case/lane 不存在；
6. 最大 lane 仍容纳不下 `required_slots`；
7. e2e 缺 `cards/sp/ep/install/train`；
8. 同一 job 内 isolation unit 冲突。

执行阶段 fail：

1. env 安装失败；
2. torch / torch_npu / numpy 被替换；
3. 任一 pytest group 或 e2e case 失败；
4. job 超时或 NPU 子进程未清理；
5. 模型获取超时；
6. skip 原因不在 lane 白名单。

gate 只读每个 device job 的结果，不再分别判断 NPU unit 和 E2E。fork PR 的
NPU job 不执行是设计内跳过；`ci:e2e:strict` 仍由统一 NPU job 的 skip reason 判定。

---

## 7. 迁移顺序

1. **事实扫描 shadow**：生成 device/processes/evidence 和旧 marker 差异报告；
2. **重新标记**：按事实迁移到 `cpu` / `npu` / `multi_npu` / `gpu` / `multi_gpu`；
3. **lint**：marker 与事实双向校验，禁止遗留 `rank_N` 继续作为调度依据；
4. **plan**：按 `(image, device, stage)` 聚合，再按 required_slots 选 lane；
5. **workflow**：CPU/NPU 改通用 matrix，删除固定 unit/e2e job；
6. **删除旧语义**：删除遗留 `rank_N` 的 job 语义，只保留进程事实；
7. **GPU（暂缓）**：接入 runner 后再新增 `gpu_unit`、GPU lane 和 matrix；
   当前不改变 CPU/NPU 计划。

每个阶段可独立回退；不允许回退到按 rank 拆 job 或全局 pip。

---

## 8. 演进规则

| 变化 | 改动 |
|---|---|
| 新增测试 | 事实扫描 + marker；不改 workflow |
| 新增模型 | 加模型目录和 e2e case；不建模型 lane |
| 换 transformers | 改 env requirements；不建 job |
| 新增 NPU 规格 | lanes.yml 加容量 lane；plan 自动选最小可容纳 lane |
| 新增 CANN / torch 镜像 | 加 image 和 lane；workflow 不改 |
| 新增 GPU runner | 加 gpu lane；复用同一聚合和执行模板 |
| 新增 / 删除 E2E case | 改 model_map.json；每个选中 case 自动生成独立 job |
| e2e 卡数变化 | 改 model_map.json；plan 自动重算 NPU 容量 |

CI 的 job 图随 image、device、stage 和选中的 E2E case 增长，不随模型源码、env、
框架或 `rank_N` 增长。PR/nightly trigger 控制这些 E2E job 的数量。
