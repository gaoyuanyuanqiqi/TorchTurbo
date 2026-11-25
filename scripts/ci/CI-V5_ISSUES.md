# TorchTurbo CI V5 问题清单

- 基线：`ci/v5-dev-baseline`
- 来源：`torchturbo-fork-pr-test-report.md` 记录 001–009，以及 #43–#52 的复验结果
- 目标：修复 V4 暴露的“测试计划了但没执行”“执行了但 Gate 看不出来”“设备调用未分流”等问题
- 原则：CPU 默认、NPU/GPU 显式声明；计划文件必须有执行证据；设备可用性由 CI lane 强校验

## 1. 问题总览

| ID | 优先级 | 问题 | 直接后果 |
|---|---|---|---|
| V5-01 | P0 | Facts 默认 CPU，但 Executor 用 `pytest -m cpu` | 未标记测试被静默 deselect |
| V5-02 | P0 | Gate 只检查 task 级 `tests_total > 0` | 计划文件未执行仍可绿灯 |
| V5-03 | P0 | suite 分类受 model import 影响 | engine 测试被误分到 model，PR 不执行 |
| V5-04 | P0 | NPU/GPU/multi marker 没有强制合同 | NPU 测试漏标后可能跑错 lane 或完全漏跑 |
| V5-05 | P1 | distributed 测试缺少 torchrun/多 rank 执行模式 | 依赖 `WORLD_SIZE` 的测试无法真实执行 |
| V5-06 | P1 | 测试扫描根和 PR scope 不完整 | `tests/npu`、model 测试等形成覆盖盲区 |
| V5-07 | P1 | 通用路径直接调用 `torch.npu` / `torch.cuda` | CPU lane 出现设备不存在错误 |
| V5-08 | P1 | fork image PR 审批标签未绑定 head SHA | 新 commit 可能复用旧审批 |
| V5-09 | P2 | actionlint 运行时下载不是自包含的 | 网络/DNS 抖动导致 lint 假失败 |
| V5-10 | P2 | replay PR scope 和测试质量缺少自动校验 | 混入无关变更、测试 fixture 错误未被发现 |

## 2. 详细问题

### V5-01 默认 CPU 语义与 `-m cpu` 不一致

- 现象：测试没有任何设备 marker 时，Facts 把它归为 CPU；Executor 执行 `pytest -m cpu` 时，pytest 又把它 deselect。
- 证据：
  - #49 engine task：`460 collected / 7 deselected / 453 selected`
  - #51 engine task：`502 collected / 49 deselected / 453 selected`
  - #52 ops task：新增插件用例未出现在 JUnit
- 根因：
  - `scripts/ci/execution/producer/facts.py:228`：`test_devices = test_devices or {"cpu"}`
  - `scripts/ci/execution/consumer/pytest_job.py:28`：按 `spec.device_marker` 执行 `-m`
- 影响：测试计划包含文件，但实际执行 0 个 case；同一 task 中只要还有其他用例，exit code 仍为 0。
- V5 修复：
  - 保持“无 marker 默认 CPU”的开发者友好语义。
  - CPU lane 必须显式包含未标记测试，例如自动补 `cpu` marker，或使用排除非 CPU 设备的 marker 表达式。
- 验收：新增一个普通 CPU 测试文件、不写 marker，CPU lane 必须实际执行它。

### V5-02 Gate 不校验计划文件的实际覆盖

- 现象：Gate 只检查 task 是否有测试结果，不检查计划中的每个测试文件是否至少产生一个 JUnit case。
- 证据：#49、#51、#52 均出现“计划文件存在，但新增用例未执行”，Gate 仍 passed。
- 根因：`scripts/ci/execution/finalization/gate.py:124` 只检查 `evidence.tests_total == 0`。
- 影响：部分文件执行、部分文件未执行时，Gate 无法发现覆盖率缺口。
- V5 修复：
  - Executor 记录计划文件到 JUnit classname 的覆盖关系。
  - Gate 校验每个计划 pytest 文件至少有一个 testcase。
  - 对确实允许 zero-test 的文件建立显式白名单，而不是默认放过。
- 验收：在一个已通过的 task 中混入一个未标记的新文件，Gate 必须失败。

### V5-03 suite 分类误把 core 测试归为 model

- 现象：#50 的 `engine/backends/fsdp2/test_backend_plan.py`、`engine/test_engine_config_loading.py`、`models/qwen3_5/test_vision_balance.py` 被分类为 `model`，没有进入 PR 的 `pytest/core` scope。
- 证据：PR Plan 中没有这些测试文件；本地直接运行分别为 `68 passed` 和 `119 passed, 3 xfailed`。
- 根因：`scripts/ci/execution/producer/facts.py:85` 在非 `models/` 目录下，仍会因导入 model family 返回 `model`。
- 影响：PR 修改了 engine/backend 代码，但对应的测试只在 nightly model scope 运行。
- V5 修复：
  - suite 分类改为路径优先：`engine/**`、`ops/**`、`runtime/**` 固定 core。
  - `models/<family>/**` 才按 family 归 model。
  - model family 仅用于 venv 解析，不改变 PR/nightly suite 归属。
- 验收：#50 重新生成 Plan 时包含 `test_backend_plan.py`，且不产生 model Job。

### V5-04 NPU/GPU/multi marker 合同没有强制

- 现象：NPU-only 测试未标记时会被 Facts 默认成 CPU，或者在 NPU lane 被漏选。
- 证据：
  - #43 `test_bailing_vl_get_rope_index.py` 无 marker，CPU lane `exit_code=5`
  - #44 `test_lightning_indexer_qwen38.py` 无 marker，CPU lane `exit_code=5`
  - #45 distributed 测试 4 个用例全部 deselect
  - #52 新增插件用例未进入 JUnit
- 根因：`pytest.ini` 的 `--strict-markers` 只保证 marker 名称已注册，不要求测试必须声明设备。
- V5 修复：
  - CPU 作为默认，不要求普通 CPU 测试写 marker。
  - NPU/GPU/multi 测试必须显式标记。
  - lint 检查设备专用导入/调用：`torch_npu`、`torch.npu`、`torch.cuda`、`device="npu"`、`device="cuda"` 等必须匹配对应 marker。
  - 混合 CPU/NPU 文件按测试函数标记，必要时拆分文件。
- 验收：新增一个无 marker 的 NPU 测试，lint 或 Plan 必须失败；新增无 marker 的 CPU 测试必须正常执行。

### V5-05 distributed 测试缺少真实多 rank 执行

- 现象：依赖 `torchrun` / `WORLD_SIZE` 的测试被当作普通 pytest task 执行，或者被直接 deselect。
- 证据：#45 `tests/unit_tests/models/test_mega_moe_distributed.py` 要求 2/4 NPU rank，当前 Run 没有真实执行。
- 影响：distributed 测试绿灯没有覆盖 rank 初始化、通信、分片和聚合路径。
- V5 修复：
  - `multi_npu` 且 rank > 1 的 task 使用 torchrun 或专用 launcher。
  - Plan 明确记录 rank 数和执行模式。
  - 对不能以 torchrun 运行的测试给出显式错误，不静默 deselect。
- 验收：至少一个 2-rank NPU 测试和一个 16-rank integration case 在 JUnit/日志中能证明实际 rank 数。

### V5-06 测试扫描根和 PR scope 不完整

- 现象：`tests/npu/test_mega_moe_planner.py` 不在 `PYTEST_ROOT=tests/unit_tests` 下，Planner 完全看不到；#49 的 `deepseek_v41` 测试只在 model scope 或根本不执行。
- 影响：新增测试文件可能永远不进入任何 PR Plan。
- V5 修复：
  - 明确 CI 支持的测试根目录清单。
  - 新测试文件如果不在扫描范围内，必须在 lint 阶段报错。
  - 对 `tests/npu/**` 定义独立 test area，或删除/迁移该目录。
- 验收：新增文件放在未扫描目录时，CI 明确失败并提示正确位置。

### V5-07 通用路径直接访问特殊设备

- 现象：PR48 在 CPU lane 执行时报 `AttributeError: module 'torch' has no attribute 'npu'`。
- 证据：`tests/unit_tests/engine/test_core.py::TestEngineSetup::test_forward_backward_batch_runs_forward_and_backward`，调用 `torch.npu.empty_cache()`。
- 根因：PR48 在 `engine/core.py:1322` 无条件调用 `torch.npu.empty_cache()`，没有走 device/runtime 抽象。
- V5 修复：
  - 通用 engine 代码只调用 device runtime 抽象，例如 `empty_cache()`。
  - NPU/CUDA 专用逻辑放在对应 backend 或 provider 中。
  - CPU lane 的测试环境不安装 `torch_npu`，用于及时发现未分流调用。
- 验收：CPU lane 执行通用 engine 测试不再依赖 `torch.npu`；NPU lane 仍有对应的缓存释放测试。

### V5-08 fork image PR 审批未绑定 head SHA

- 现象：`ci:image:build` 标签可以复用于后续 commit。
- 风险：维护者批准了旧 commit 后，fork PR 再推送新代码，可能继续获得 image build 权限。
- V5 修复：
  - 标签绑定审批时的 head SHA。
  - 新 commit 到达后要求重新审批，或在 workflow 中校验审批 SHA。
- 验收：push 新 commit 后旧审批失效，Image PR 必须重新获得标签审批。

### V5-09 actionlint 获取不是自包含的

- 现象：#52 首次 lint 失败：`cannot obtain actionlint: No address associated with hostname`；同一时段其他 run 正常。
- 影响：网络/DNS 抖动导致 PR 假失败，且失败原因与代码无关。
- V5 修复：
  - 缓存或预安装 actionlint 二进制。
  - lint 入口固定版本，避免运行时临时下载。
  - 对工具获取失败给出明确的 infra-error，而不是普通 lint error。
- 验收：断网或 DNS 异常时 lint 能明确报告工具不可用；正常缓存命中时不再访问网络。

### V5-10 replay PR 范围和测试质量缺少校验

- 现象：
  - #52 diff 为 45 个文件，上游 #85 为 14 个，混入 #82–#84 和 main 漂移内容。
  - #49 `deepseek_v41/test_runtime_contracts.py` 直接运行时出现 2 个 `unit_kl_scale` fixture 错误。
- 影响：PR 失败归因困难；未执行测试中的 fixture/import 错误可能长期隐藏。
- V5 修复：
  - replay PR 增加“与上游 PR changed files 对比”的范围检查。
  - 新增/修改的测试文件至少执行一次 collect-only smoke test。
  - Plan 对新增测试文件默认纳入执行，不依赖作者额外登记。
- 验收：replay PR 不能静默混入无关文件；新增测试文件出现 fixture/import 错误时 CI 直接失败。

## 3. V4 已解决的问题

| 问题 | 状态 |
|---|---|
| fork image PR 在 `pull_request_target` 下 checkout 失败 | V4 已通过 `allow-unsafe-pr-checkout` 修复并云端验证 |
| Dockerfile-only PR 同时触发普通 PR | V4 已通过 `paths-ignore` 修复 |
| actionlint 对 checkout 新输入误报 | V4 已加单点 ignore |
| PR52 actionlint DNS 失败 | 重跑 attempt 2 成功，确认是瞬时基础设施问题 |

## 4. V5 验收清单

- [ ] 无 marker 的 CPU 测试会实际执行
- [ ] 无 marker 的 NPU/GPU 测试会在 lint 或 Plan 阶段失败
- [ ] 计划中的每个 pytest 文件至少产生一个 JUnit case
- [ ] `engine/**`、`ops/**` 不再被 model import 误分类
- [ ] `tests/npu/**` 要么纳入扫描，要么明确禁止
- [ ] distributed 测试使用真实 torchrun/多 rank 执行
- [ ] 通用代码不直接访问 `torch.npu` / `torch.cuda`
- [ ] fork image PR 审批与 head SHA 绑定
- [ ] actionlint 获取不依赖运行时网络
- [ ] replay PR 范围与上游 PR 对齐

## 5. 建议复测用例

| 场景 | 复测 PR / 行为 | 期望 |
|---|---|---|
| 无 marker CPU 测试 | 新增普通 CPU 测试 | 实际执行并产出 JUnit |
| 无 marker NPU 测试 | 新增 NPU 调用但漏 marker | lint/Plan 失败 |
| 部分文件 deselect | 一个 task 中混入未选中文件 | Gate 失败 |
| engine/model 分类 | 重新跑 #50 | engine 测试进入 PR Plan |
| DP rebalance | 重新跑 #51 | 49 个新用例全部执行 |
| 插件测试 | 重新跑 #52 | `test_plugins` / `test_plugins_npu` 出现在 JUnit |
| 通用 device 调用 | 重新跑 #48 | CPU lane 不再访问 `torch.npu` |
| 分布式测试 | 运行 multi_npu probe | 使用真实多 rank 执行 |

