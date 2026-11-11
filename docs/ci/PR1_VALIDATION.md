# PR1 验证记录：facts / plan shadow

## 状态

- 分支：`ci/refactor-facts-plan`
- 基线：`ci/npu-pipeline` (`d367203`)
- PR1 只新增 shadow `plan` 校验，不切换现有 CI 执行路径。
- 未修改 `src/` 业务代码，未提交、未推送。
- 已修正既有测试错误、放开 Qwen4 全量 CPU 用例，并修复对应的测试夹具兼容问题。

## PR1 边界

PR1 把 CI 差异收敛为四个正交维度：

```text
(image, resource) -> job
env               -> venv
isolation unit    -> process
pytestmark        -> device / rank
```

具体实现：

- `scripts/ci/facts.py`：env、lane、E2E case、test requirement 的唯一事实源；用 AST 提取模型/framework import 和 device/rank marker。
- `scripts/ci/plan.py`：将 selector 输出编译为 `(image, resource)` job，并在 job 内按 env 和进程隔离单元分组。
- `scripts/ci/envs.toml` + `scripts/ci/envs/*.txt`：Python 依赖只进入 venv，不再参与 job key。
- `scripts/ci/lanes.yml`：runner、image、device、slots 的唯一声明位置。
- `scripts/ci/model_env.toml`：unit/integration 测试的模型族到 env/image 映射。
- `.github/workflows/model_map.json`：E2E 从 model 级参数改为 case 级 schema，共 10 个 case。
- `scripts/ci/e2e_tests/prepare_model.py`、`run_train.sh`：按 case 解析 env/framework/train/install/dataset/layers。
- `select_tests.py`、`lint_turbo_check.py`：E2E 选择委托 facts，lint 校验 model_map/env/lane/marker/plan。
- `pytest.ini`：只保留 `cpu`、`npu`、`rank_2`、`rank_4`、`rank_8`、`integration`。
- 相关测试补齐 device/rank marker；两个依赖 pinned memory 的 optimizer 用例改归 NPU lane。

改动量（不含本记录）：73 个 tracked 文件，`+471 / -429`；新增 `facts.py`、`plan.py`、env/lane 配置和 `DESIGN.md`。

## Selector 与 plan 校验

以下命令均通过：

```bash
python3 scripts/ci/lint_select/select_tests.py --verify
python3 scripts/ci/plan.py --verify
python3 scripts/ci/local_lint_select_test.py
```

结果：

- selector：`PASS`，35 项负向/正向断言通过。
- selector 精度：mirror `100.0%`，strict `99.6%`，265 modules / 162 test files。
- 场景矩阵：20 组场景全部通过，包括 docs、CPU/NPU、mixed、ops full lane、engine、run config、shared env、E2E map、pyproject、labels。
- selector/workflow contract：全部通过。
- plan facts：`10 E2E cases / 162 tests`，无事实错误。

关键影响路径复核：

| changed path | 结果 |
|---|---|
| `scripts/runs/qwen3_5_dense/lf_text_sft.yaml` | 只运行 `qwen3_5-lf-text-sft`，不跑 CPU/NPU unit |
| `scripts/ci/envs/common.txt` | CPU 全量（150 files）+ 最便宜 E2E 代表 `bailing_hybrid-lf-sft` |
| `.github/workflows/model_map.json` | CPU 全量（150 files）+ 一个 E2E smoke，不跑 NPU unit |

最终全量 plan：

```text
CPU: 4 jobs / 53 pytest steps / 150 files
NPU: 2 jobs / 8 steps / 18 files / 1 E2E case
```

`test_adamw_swap_checkpoint.py` 和 `test_cpu_offload_optimizer.py` 已只出现在 NPU lane，不再由纯 CPU runner 执行。

## CPU venv 约定验证

本机：x86_64、Python 3.11.15、`torch 2.10.0+cpu`。按 `envs.toml` 的版本，并按 `scripts/runs/deps/*.sh` 的安装顺序创建了 5 个独立 venv：

| env | 关键版本 | 安装结果 |
|---|---|---|
| `default` | transformers 5.5.4 / peft 0.18.1 | OK |
| `lf` | transformers 5.5.4 / llamafactory 0.9.5 / datasets 4.0.0 | OK |
| `swift` | transformers 5.9.0 / ms-swift 4.4.2 / datasets 5.0.1 | OK（与 scripts/runs 一致） |
| `qwen4_exp` | transformers 5.9.0 / peft 0.18.1 | OK |
| `qwen4_exp_swift` | swift + transformers 5.9.0 / datasets 5.0.1 | OK（与 scripts/runs 一致） |

安装验证发现并处理了三个 CI 事实问题：

1. Swift 依赖以 `scripts/runs/deps/swift_deps.sh` 为准，保留 `datasets==5.0.1`。`ms-swift 4.4.2` 的元数据声明 `datasets<4.8.5`，因此不能把该文件当成单次 `pip install -r` 的完整约束；CI 安装顺序必须与训练脚本一致：先安装 `ms_swift==4.4.2`，再覆盖安装 `datasets==5.0.1`。在 `5.0.1` 下修复 Swift 原生参数测试后为 `182 passed / 117 skipped`，qwen4_exp_swift 为 `17 passed`。
2. `common.txt` 缺少 `dill`，导致 `test_transformers_utils.py` 在 collection 阶段整模块 skip 并以 pytest exit 5 结束；补入 `dill` 后 6 项测试通过。
3. 两个 optimizer smoke 文件使用 `pin_memory=True`，纯 CPU host 没有 pinned allocator；按既有分类从 `cpu` 改为 `npu`。

执行方式为每个 plan step 独立进程：

```bash
<env-venv>/bin/python -m pytest -q -m cpu <plan step files>
```

## CPU 全量结果

最终计划共 53 个 CPU group：

| env | groups | tests | passed | failed | collection error | skipped |
|---|---:|---:|---:|---:|---:|---:|
| `default` | 39 | 2242 | 2190 | 0 | 0 | 52 |
| `lf` | 4 | 354 | 91 | 0 | 0 | 263 |
| `swift` | 4 | 299 | 182 | 0 | 0 | 117 |
| `qwen4_exp` | 5 | 228 | 226 | 0 | 0 | 2 |
| `qwen4_exp_swift` | 1 | 17 | 17 | 0 | 0 | 0 |
| **合计** | **53** | **3140** | **2706** | **0** | **0** | **434** |

结论：53/53 个 group 成功，CPU 全量用例全部通过。

### 已关闭的失败项

1. `tests/models/deepseek_v4/test_ops_precision.py::test_packed_compress_metadata`
   - `ratio=5` 时 `capacity = seqlen // ratio = 12 // 5 = 2`。
   - 测试沿用了 `ratio=4` 的 3 列 expected，多期待了一列 padding。
   - 已改为 `[[1, 2], [3, 0]]` / `[[0, 7], [2, 0]]`；该文件 `5 passed`，所在 group `101 passed / 1 skipped`。

2. `tests/adapters/msswift/test_config.py` 3 项
   - ms-swift 在原生参数测试中默认 `fp16=True`，而 engine 明确只支持 `bf16/fp32`。
   - 适配层对 `fp16` 的拒绝行为已有独立测试覆盖，不属于需要放宽的业务契约。
   - 3 个原生参数测试显式设置 `bf16=True / fp16=False`；该文件 `59 passed`。

3. `tests/scripts/test_dequant.py`
   - collection error：`ModuleNotFoundError: No module named 'torchturbo.scripts'`。
   - `scripts/dequant` 没有 `__init__.py`，脚本和测试都以目录在 `sys.path`、模块名直接为 `dequantize` 的方式加载；`torchturbo.scripts` 命名空间不存在。
   - 已统一改为 `from dequantize import ...` / `from converter import Converter`，覆盖 `converter.py`、`bailing.py`、`deepseekv4.py`、`deepseekv4_1.py`、`kimik26.py`。
   - `test_dequant.py` 为 `20 passed`，完整 `tests/scripts` group 为 `47 passed`。

4. Qwen4 全量 CPU 用例
   - 移除 18 个文件中的 `Qwen4 tests temporarily skipped` 标记。
   - Qwen4 使用与 `scripts/runs` 一致的 transformers 5.9.0；测试 fake 保持 `(kernels, process_fn)` 契约，不做跨版本兼容。
   - Qwen4 conftest 为 CPU 用例初始化 ops dispatcher；手工构造的 engine 测试夹具补齐 `plugin_manager`。
   - qwen4_exp CPU 结果：qwen4_exp 主组 `199 passed / 2 skipped`，rank_2/rank_4 与 import 组全部通过。

### Skip 说明

- qwen4_exp 只剩 2 个 `requires MPS` 的参数 skip。
- LF/Swift 的 380 个 skip 来自需要真实模型、数据或框架资产的 integration/parity 用例。
- 其余 skip 为各模型 smoke 的环境条件 skip。

## 结论

- selector 行为符合约定，关键路径与全量场景矩阵均通过。
- 5 套 venv 均按 `scripts/runs/deps/*.sh` 的版本与安装顺序复现；缺失的 `dill`、误归 CPU 的 pinned-memory 用例已修正。
- CPU 全量 53/53 个 group 通过，无 failure、无 collection error。
- PR1 继续保持 shadow 模式，不切换正式执行路径。
