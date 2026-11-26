# TorchTurbo CI 设计说明书

- 版本：V5
- 适用分支：`ci/v5-dev-baseline`
- 目标：让 PR、nightly 和手动入口复用同一套“用例发现 -> 计划 -> 执行 -> 判定”流程。

## 1. 先理解 CI 在做什么

CI 不是直接“把所有测试跑一遍”，而是：

```text
这次改了什么
  -> 应该跑哪些测试
  -> 测试需要什么设备和镜像
  -> 生成执行计划
  -> 按计划执行
  -> 检查计划是否全部完成
```

整个流程由五部分组成：

```text
变更分类 -> Facts -> Planner -> 消费者 -> Gate
```

| 阶段 | 作用 |
|---|---|
| Facts | 发现仓库里有哪些测试，以及测试需要什么设备 |
| Planner | 结合用例、配置、资源和镜像，生成执行计划 |
| 消费者 | 按计划准备环境并执行测试 |
| Gate | 检查计划中的测试是否真正执行并产出证据 |

## 2. 用例分类与执行范围

| 类别 | 子类 | 验证内容 | 主要设备 | PR | Nightly |
|---|---|---|---|---|---|
| `unit_tests` | `core` | 通用引擎、算子、运行时、registry | CPU / NPU | ✅ | ✅ |
| `unit_tests` | `adapter` | LLaMA-Factory、ms-swift 适配 | CPU | ✅ | ✅ |
| `unit_tests` | `model` | 具体模型实现和模型特有逻辑 | CPU / NPU | ❌ | ✅ |
| `integration_tests` | `qwen3_5_moe/sft_2cpu` | Qwen3.5-MoE 2-rank CPU SFT | CPU | ✅ | ✅ |
| `integration_tests` | `qwen3_5_moe/sft_2npu` | Qwen3.5-MoE 2-rank NPU SFT | NPU | ✅ | ✅ |
| `integration_tests` | `bailing_v3/sft_16npu` | Bailing V3 16-rank NPU SFT | NPU | ❌ | ✅ |

- `PR`：`Ascend-PR-CI`
- `Nightly`：`Ascend-Nightly-CI`
- `✅`：该入口会执行；`❌`：该入口默认不执行
- Dockerfile 变更由 `Ascend-Image-PR` 单独处理，不在此表内


## 3. 从测试事实到执行计划

### Facts：有哪些测试，需要什么

Facts 从测试代码中提取静态事实，不表示测试结果。

```text
tests/unit_tests/test_*.py
  -> suite / device / ranks / framework / model_family

tests/integration_tests
  -> suite / case_id / device / ranks / framework
```

Facts 回答：

- `unit_tests`  一个test_xxx.py对应一条PytestFact 
  - `suite`：测试所属的 `core`、`adapter` 或 `model`
  - `device`：测试需要的 `cpu`、`npu` 或 `gpu`
  - `path`：测试文件路径
  - `ranks_by_device`：每种设备需要的最大 rank
  - `frameworks`：是否依赖 `lf` 或 `swift`
  - `model_families`：测试所属的模型族
- `integration_tests`
  - `suite`：integration suite 名称
  - `case_id`：suite 内的具体 case
  - `device`：case 需要的 `cpu`、`npu` 或 `gpu`
  - `ranks`：case 需要的 rank 数
  - `framework`：case 使用的 `swift` 或 `llamafactory`

Facts 不知道实际 runner，也不知道应该使用哪个 image 或 venv。

### Test Profile：这次跑哪些区域，用哪个镜像

`test_profiles` 是 `ci.yml` 中的运行配置：

```yaml
test_profiles:
  pull_request:
    - image: cpu-default
      test_areas:
        unit_tests: [core, adapter]
        integration_tests: [base_suite/qwen3_5_moe/sft_2cpu]
    - image: npu-a3-cann90
      test_areas:
        unit_tests: [core]
        integration_tests: [base_suite/qwen3_5_moe/sft_2npu]

```

它回答：

- PR 或 Nightly 使用哪个 profile
- 选中哪些 unit test / integration test 区域 和指定需要使用的镜像

它不直接指定 runner、rank 容量或 venv；这些由 Planner 继续推导。

### Planner：生成可执行计划

Planner 的核心接口：

```python
build_plan(config: CiConfig, facts: TestFacts, options: PlanOptions)
```

输入只有三类：

| 输入 | 来源 | 作用 |
|---|---|---|
| `Facts` | `facts.py` | 仓库里有哪些测试，以及测试需要什么 |
| `CiConfig` | `ci.yml` | 资源、用例区域、超时和 PR/Nightly 范围 |
| `PlanOptions` | CLI / workflow | `ci_type`、`change_mode`、`git_source` 和调试筛选条件 |

Planner 的匹配过程：

```text
ci_type -> TestProfile -> ProfileSelection(image, test_areas)
  -> image 得到 target/device
  -> test_areas + device 过滤 Facts
  -> ranks 选 runner，model/framework 选 venv
  -> 聚合成 Job 和 Task
  -> 输出 ExecutionPlan
```

- `unit_tests`：按 `suite + device + target + image + venv` 聚合 Job。
- `integration_tests`：按 `suite + case_id + target + image` 聚合 Job。
- 每个 Job 固定 `runtime(device, target, ranks, runner, image, venv)` 和 `tasks`。
- 生成的 `ExecutionPlan` 绑定 `plan_id`，消费者只能执行，不能重新选择资源。

Planner 会拒绝空选择、缺失镜像 digest、缺少 runner/timeout/venv 等无法执行的计划。
`change_mode != code` 时不生成测试 Job。

一句话概括：Facts 是事实，Profile 是选择，Planner 是匹配并生成计划。

### 实际组合

| 用例 | device | target | image | runner | venv | ranks |
|---|---|---|---|---|---|---:|
| core CPU unit test | cpu | cpu | cpu-default | cpu/8 | default | 4 |
| core NPU unit test | npu | npu-a3 | npu-a3-cann90 | npu-a3/2 | default | 2 |
| adapter lf unit test | cpu | cpu | cpu-default | cpu/8 | lf | 1 |
| qwen4_exp CPU model test | cpu | cpu | cpu-default | cpu/8 | qwen4_exp | 4 |
| qwen4_exp NPU model test | npu | npu-a3 | npu-a3-cann90 | npu-a3/2 | qwen4_exp | 2 |
| Qwen3.5-MoE 2-NPU integration | npu | npu-a3 | npu-a3-cann90 | npu-a3/2 | lf | 2 |
| Bailing V3 16-NPU integration | npu | npu-a3 | npu-a3-cann90 | npu-a3/16 | swift | 16 |

Plan 生成后，消费者不再选择设备、镜像、runner 或 venv。

## 4. 计划消费与 Gate 判定

消费者是 Plan 的唯一执行者。一个消费者进程只消费一个 `job_id`，在独立 runner 上
准备环境、执行 Task，并产出可被 Gate 校验的 `JobResult`。

核心原则：Plan 不可变，执行结果自包含，执行证据可校验。

### 消费模型

```text
ci-plan / PlannerOutput
  -> 加载并校验 Plan
  -> 按 job_id 定位唯一 Job
  -> 按 runtime 准备环境
  -> 顺序执行 Job.tasks
  -> ci-result-<job_id> / JobResult
  -> Gate
```

### 核心点

1. **加载边界**
   - 只接受 `status=ready` 的 `PlannerOutput`。
   - 校验 `plan_id`、Job 类型和 Task 类型。
   - `job_id` 必须存在于 Plan 中，否则消费者不执行任何测试。

2. **环境边界**
   - 消费者不再选择 image、runner 或 venv。
   - `runtime.device` 和 `runtime.venv` 决定实际执行环境，`runner/image` 原样写回结果。
   - venv 可以复用内容寻址缓存，但必须与 Plan 指定的环境一致。
   - 测试前先探测设备；环境准备或探测失败直接记 `infra_error`。

3. **Task 边界**
   - pytest Task 使用 Plan 中的 device marker 和测试文件。
   - integration Task 使用 Plan 中的 suite、case、framework 和 ranks。
   - Task 按顺序执行，同时受 Task 超时和 Job 总超时约束。
   - 遇到第一个非 `passed` Task 后停止后续 Task。

4. **结果边界**
   - 执行前先写 provisional 结果，避免进程崩溃后完全没有结果。
   - 执行结束覆盖为最终 `JobResult`，包含 `plan_id`、`job_id`、runtime、Task 状态、失败原因和 artifact 路径。
   - 日志、JUnit、integration artifact 和 versions 都作为 Gate 证据保留。
   - 失败、超时、取消和 infra error 也必须写结果。

5. **Skip 边界**
   - 平台级跳过不进入测试流程。
   - 只允许 Plan 中声明的 `skip_reason`，并写结构化的 skipped `JobResult`。

### Gate 原则

Gate 不重新运行测试，只校验：

- Plan 中的 Job 是否恰好对应一份结果，不能缺失、重复或越界。
- `plan_id`、Job 类型、runtime 和 Task 集合是否一致。
- pytest 是否真正收集并执行用例，是否存在全部 skip、失败或 collection error。
- integration 是否产出 artifact，skip 原因是否合法。

**Job 进程返回 0 不等于 CI 通过；Plan、结果和证据全部对齐后，Gate 才通过。**

## 5. 测试合同

- pytest 文件只能放在 `tests/unit_tests`。
- integration 用例放在 `tests/integration_tests`。
- CPU 测试默认可不写 marker。
- NPU/GPU 测试必须显式写设备 marker。
- 多 rank 测试必须能被 Facts 识别出 rank。
- 测试路径、model family、adapter framework 决定 venv。
- 纯基础组件测试不应直接依赖具体模型。
- 模型测试应放在 `tests/unit_tests/models/<family>`，使用对应 venv（若已配置）。

## 6. 镜像流程

```text
Dockerfile
  -> content hash
  -> content tag
  -> buildx build
  -> immutable digest
  -> container tests
  -> Gate
  -> post-merge promote alias
```

- content tag 由 Dockerfile、platform、revision 计算。
- 容器测试使用 digest-pinned image ref。
- candidate 验证与 stable alias promotion 分离。
- `Ascend-Image-Promote` 负责 merge 后或手动移动 alias。

## 7. fork PR 和手动入口

- fork Image PR 需要 `ci:image:build` 标签。
- 当前标签是 PR 级信任，不绑定 commit；后续 commit 会继承该标签。
- 标签不等于代码 review，构建仍会使用 SWR secret。
- `Ascend-Manual-Test` 可以精确执行一个 Plan Job。
- `Ascend-Manual-Image` 可以手动构建或复用镜像。

## 8. 当前约束

- 当前 CI 不包含 replay PR 生成工具；replay 分支与上游 PR 的 diff 对齐由外部流程负责。
- 不允许 `test_*.py` 放在 `tests/unit_tests` 之外。
- 模型 + 基础组件的跨模块测试需要拆分或显式指定 venv，不能默认由 default venv 承担。
- GPU 已有设备和部分路由模型，但当前没有实际 GPU runner/test profile。
- actionlint/hadolint/shellcheck 仍可能在运行时下载，网络抖动会造成 lint infra failure。
