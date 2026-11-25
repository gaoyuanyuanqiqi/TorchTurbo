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
| V5-05 | P1（已关闭） | distributed 测试采用 `mp.spawn` 自启动多 rank，不依赖 torchrun | 无；保留 rank 识别约束 |
| V5-06 | P1 | 测试扫描根和 PR scope 不完整 | `tests/npu`、model 测试等形成覆盖盲区 |
| V5-07 | P1（部分修复） | 通用路径直接调用 `torch.npu` / `torch.cuda` | CI 在 CPU lane 正确暴露业务代码设备抽象泄漏 |
| V5-08 | P1（已澄清） | `ci:image:build` 绑定 PR，不绑定 commit | 后续 commit 继承 PR 级信任，符合当前策略 |
| V5-09 | P2 | actionlint 运行时下载不是自包含的 | 网络/DNS 抖动导致 lint 假失败 |
| V5-10 | P2 | 新增/修改测试缺少收集与执行校验 | fixture/import 错误可能被绿灯掩盖 |

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

### V5-05 distributed 测试使用 `mp.spawn`（已关闭）

- 结论：不是问题。项目测试不依赖 torchrun；pytest 进程直接运行在目标 runner 上，测试内部通过 `mp.spawn` 启动多 rank worker。
- 当前实现：
  - `facts.py` 识别 `mp.spawn(nprocs=N)`、`require_npus(N)` 和 `world_size` 参数，生成 `ranks_by_device`。
  - `planner.py` 按 `ranks` 选择 `rank_capacity >= N` 的 runner。
  - `pytest_job.py` 保持普通 `pytest` 启动，多进程由测试自身负责。
  - integration 用例由 `run_tests.py` 调用 `torch.distributed.run`，不受此问题影响。
- 保留约束：
  - `multi_npu` 测试必须使用 `mp.spawn`，且 `nprocs/rank` 能被 Facts 静态识别。
  - 不要求 Executor 支持 torchrun；torchrun 不作为 unit test 的运行方式。
- 当前证据：
  - `tests/unit_tests/engine/test_muon_offload_distributed.py` 使用 `mp.spawn(nprocs=2)`。
  - `tests/unit_tests/models/qwen4_exp/test_qsa_distributed.py` 使用 `require_npus(2)` + `mp.spawn(nprocs=2)`。
  - #45 引用的 `test_mega_moe_distributed.py` 当前工作树中不存在，原证据失效。
- 验收：`multi_npu` 测试实际启动 N 个 worker，Plan 中 `ranks >= N`，日志或 JUnit 能证明多 rank 执行。

### V5-06 测试扫描根和 PR scope 不完整

- 现象：`tests/npu/test_mega_moe_planner.py` 不在 `PYTEST_ROOT=tests/unit_tests` 下，Planner 完全看不到；#49 的 `deepseek_v41` 测试只在 model scope 或根本不执行。
- 影响：新增测试文件可能永远不进入任何 PR Plan。
- V5 修复：
  - 明确 CI 支持的测试根目录清单。
  - 新测试文件如果不在扫描范围内，必须在 lint 阶段报错。
  - 对 `tests/npu/**` 定义独立 test area，或删除/迁移该目录。
- 验收：新增文件放在未扫描目录时，CI 明确失败并提示正确位置。

### V5-07 通用路径直接访问特殊设备

- 结论：这是 CI 正确发现业务代码问题，不是 CI 误报。CPU lane 不安装 `torch_npu`，正好能暴露通用路径中的设备专用调用。
- 原问题：PR48 在 CPU lane 执行时报 `AttributeError: module 'torch' has no attribute 'npu'`；业务代码在 `engine/core.py` 中直接调用 `torch.npu.empty_cache()`，没有走 device/runtime 抽象。
- 测试归属：`tests/unit_tests/engine/test_core.py` 使用 `CpuRuntime` 和 CPU tensor，按实际行为属于 CPU 用例。不能只依赖 `pytest.mark.cpu` 判断，也不能因为这个 marker 曾不准确就否定 CI 发现的问题。
- 当前状态：
  - 原始无条件 `torch.npu.empty_cache()` 已移除。
  - `engine/core.py` 的 `_sync_device()` 仍直接判断 `torch.npu` / `torch.cuda`，但有 `hasattr` 和 `is_available()` 保护，当前不再导致 CPU lane 崩溃。
  - 这属于残留的设备抽象泄漏，不是 CI 故障；原文档中的测试证据已过时。
- 修复方向：
  - 通用 engine 代码统一调用 `device_runtime.empty_cache()` / `device_runtime.synchronize()`。
  - NPU/CUDA 专用实现只保留在对应 runtime/backend/provider 中。
  - marker 与测试实际行为不一致时单独校正 marker，不影响本次 CI 判断。
- 验收：CPU lane 执行通用 engine 测试不出现 `torch.npu` / `torch.cuda` 依赖；NPU lane 仍有对应的同步和缓存释放覆盖。

### V5-08 fork image PR 标签审批语义（已澄清）

- 当前语义：`ci:image:build` 是 **PR 级信任标签**，绑定 PR，不绑定 head SHA。
- 当前行为：
  - 维护者给 PR 加标签后，PR 后续的 Dockerfile commit 仍满足审批条件。
  - 新 commit 不会自动撤销标签，也不会要求维护者重新打标签。
  - 构建使用 SWR secret；候选镜像只推 content tag，stable alias 仍由独立的 promote workflow 控制。
- 结论：按“标签代表信任整个 PR/贡献者”的当前策略，这不是实现 bug，也不要求改成 head SHA 绑定。
- 使用约束：
  - 只给可信 contributor/PR 添加 `ci:image:build`。
  - 标签不替代代码 review，维护者仍需 review 后续改动。
  - 如果未来改为“按 commit 审批”，应使用 PR review approval、environment approval 或 `/image-build <sha>`，不要继续复用持久 label。
- 验收：文档明确 `ci:image:build` 为 PR 级语义；新 commit 继承标签符合预期，不会误称为 commit 级审批。

### V5-09 actionlint 获取不是自包含的

- 现象：#52 首次 lint 失败：`cannot obtain actionlint: No address associated with hostname`；同一时段其他 run 正常。
- 影响：网络/DNS 抖动导致 PR 假失败，且失败原因与代码无关。
- V5 修复：
  - 缓存或预安装 actionlint 二进制。
  - lint 入口固定版本，避免运行时临时下载。
  - 对工具获取失败给出明确的 infra-error，而不是普通 lint error。
- 验收：断网或 DNS 异常时 lint 能明确报告工具不可用；正常缓存命中时不再访问网络。

### V5-10 新增/修改测试缺少收集与执行校验

- 结论：CI 只处理当前 PR 的 `base..head`。Replay 分支是否与上游 PR 文件一致，属于 replay 构造流程的问题，不是当前 CI 的问题。
- 现象：#49 `deepseek_v41/test_runtime_contracts.py` 直接运行时出现 2 个 `unit_kl_scale` fixture 错误；如果文件未进入 Plan 或未实际执行，CI 仍可能绿灯。
- 影响：新增/修改测试的 import、collection、fixture 错误可能长期隐藏；测试文件存在不等于测试被验证。
- V5 修复：
  - 新增/修改的测试文件默认进入 Plan，不依赖作者额外登记。
  - 对新增/修改文件执行 collect-only，捕获 import/collection 错误。
  - 涉及 fixture 的测试必须在实际执行中覆盖；Gate 不把未产生 case 的文件当成已验证。
  - replay PR 与上游 PR 的 diff 对齐由 replay 构造工具负责，不纳入 CI 验收。
- 验收：PR 新增/修改的测试文件要么产生 JUnit case，要么 CI 明确报告未执行或收集失败。

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
- [x] distributed 测试使用 `mp.spawn` 自行启动真实多 rank，不要求 torchrun
- [ ] 通用代码不直接访问 `torch.npu` / `torch.cuda`
- [x] fork image PR 标签语义明确为 PR 级信任，不要求绑定 head SHA
- [ ] actionlint 获取不依赖运行时网络
- [ ] 新增/修改的测试文件至少被 collect 或实际执行

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
| 分布式测试 | 运行 multi_npu probe | 测试内部通过 `mp.spawn` 启动 N 个 worker，Plan 中 `ranks >= N` |
| 新增测试文件 | 新增一个测试文件 | 自动进入 Plan，并完成 collect 或实际执行 |

