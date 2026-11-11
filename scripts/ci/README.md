# CI 辅助脚本

原来塞在 workflow heredoc 里的逻辑都搬到了这里。每个文件都能在本机直接跑——
这正是目的：之前的写法只能靠 push 上去、等一个 NPU pod 才能验证。

目录按 CI 阶段划分，公共能力单独下沉：

```text
scripts/ci/
  lint_select/   lint 与 select 阶段
  unit_tests/    CPU / NPU 单测共用的安装与 pytest runner
  e2e_tests/     NPU 模型用例：取模型、并行策略、训练监督
  image/         基础镜像构建
  common/        pod 环境
  report/        结果汇总与门禁判定
  local_*        本机自检脚本，不进 CI
```

`unit_tests/` 同时服务 CPU 和 NPU 两个 job：两者只差 `MODE`、runner 和
pytest marker，runner 本身必须共用，否则两条泳道会慢慢跑偏。

通用规则交给 Ruff（Python）、`shellcheck`、`actionlint` 和 `hadolint`；自定义
检查只保留通用工具看不到的仓库契约：job 依赖图、选测器自检、模型映射和 pytest
marker。

`local_*` 表示「只在本机跑」，不会出现在任何 workflow 里。

**按阶段划分，一个入口回答一个阶段的问题**：

| 路径 | 回答的问题 | 子命令 |
|---|---|---|
| `lint_select/select_tests.py` | 这次改动该跑什么？ | `--explain` / `--coverage` / `--verify` / `--changed-from git` / `--mode pr\|full --labels …` |
| `lint_select/lint_check.py` | 静态体检总入口；通用检查先跑，Turbo 契约后跑 | 无参数 = 全跑 / `--only NAME` |
| `unit_tests/run_tests.sh` | 单测泳道怎么装、怎么跑？ | `install` / `test`（`MODE=cpu` 用 `-m cpu`，`MODE=npu` 用 `-m npu`） |
| `e2e_tests/prepare_model.py` | 模型和派生配置是否完成预处理？ | `plan` / `fetch` / `probe` / `cases` |
| `e2e_tests/run_train.sh` | 一次训练冒烟分几步？ | `resolve` / `install` / `run` / `report` / `lane` |
| `common/env_pod.py` | 这台 pod 现在是什么状态？ | `probe` / `check` / `cleanup` |
| `image/build_image.py` | 基础镜像该不该建 / 怎么建 / 发什么引用 / SWR 上传链路是否可用 | `env` / `tag` / `build` / `verify` / `selftest` / `publish` / `resolve` / `hadolint` / `summary` / `cleanup` |
| `report/report_gate.py` | 这些结果算不算通过？ | `timing` / `judge --mode pr\|full` / `failed-step` |
| `report/report_result.py` | 事实 → 人看的摘要 | `init` / `set` / `phase` / `report` / `notice` / `junit` / `migrate-map` |

### lint：通用工具先行，Turbo 契约后置

每个检查返回 `list[Problem]`，问题带 `path` / `line` / `message`。驱动器按顺序
跑完**全部**检查再汇总，所以前一项失败不会掩盖后面的结果；本地和 CI 调的是同一个
入口。检查之间不 import 本仓代码，只有少数跨文件契约会主动调用脚本。

`lint_check.py` 只负责 CLI 和汇总；`lint_common_check.py` 依次跑 Ruff、shellcheck、
actionlint、hadolint；`lint_turbo_check.py` 再跑 `contract`、`selector`、`model_map`、
`markers`。通用规则不在这里重新实现。

`CHECKS` 是 `Check(name, description, function)` 的元组。加一个 Turbo 检查 = 写函数 +
在 `lint_turbo_check.py` 的 `CHECKS` 里加一行：

```python
def check_tabs() -> list[Problem]:
    return [Problem("src", "tab in indentation", line)
            for line in range(1, 2)]


# Then add one row to the CHECKS tuple:
Check("tabs", "indentation uses spaces", check_tabs)
```

`--list` 和 `--only` 都从 `CHECKS` 读；`--only` 可重复，`map` 是 `model_map`
的兼容别名（训练侧仍会调用 `--only map`）。

```bash
python3 scripts/ci/lint_select/lint_check.py --only ruff
```

输出是编译器格式，可以直接点进编辑器；GitHub Actions 里同时发 annotation：

```text
src/torchturbo/utils/torch_utils.py:133: invalid syntax
```

退出码：`0` 通过 / `1` 有问题 / `2` 参数用错或工具下载失败。

外部工具（actionlint / hadolint / shellcheck）优先走 PATH、`/tmp` 或
`TORCHTURBO_LINT_TOOLS`；缺失时按文件里固定的版本从 release 下载，不使用
`curl | bash`。Ruff 由 CI 的 pip 步骤固定版本安装。

### 本机自检

`local_lint_select_test.py` 不是 CI 阶段，任何 workflow 都不调用它。跑一遍等于把
`lint`、`select --verify`、13 个真实改动场景和 job 图契约全验一次：

```bash
python3 scripts/ci/local_lint_select_test.py --only docker
```

模型 job 分为 `environment check -> training orchestration -> environment cleanup`。
训练编排内部按模型训练用例执行 `resolve -> install -> preprocess -> train -> report`；
`lane` 是训练编排入口，不是独立阶段。

被下面这些 workflow 调用，`.github/workflows` 里的入口只负责声明输入：

```
ci-pr.yml          唯一入口  select -> {cpu-unit-tests, npu-unit-tests, npu-e2e-tests} -> gate
                    mode=pr（PR 门禁） / mode=full（全量，不选测）
ci-image.yml        构建基础镜像（**只手动 dispatch**）
model_map.json      模型 -> 启动脚本 / 权重 / 端到端用例（被上面两个读）
pytest.ini          测试根 / 导入路径 / marker 契约（与 select_tests.py 交叉校验）
```

三条泳道**内联**在 `ci-pr.yml` 里，不是可复用 workflow：`uses:` 的 job 一律
显示成 `调用方名 / 被调方名`，图上的泳道名就只能是两段。

yaml 里只留「输什么、调哪个脚本」，超过 10 行的 bash 一律下沉到这里。
除语法高亮之外的好处是 **shellcheck / actionlint 能看**，而且改一行能在本机验。

### 合并与不合并的边界

合并前先问一句：**它们是不是在回答同一个问题？**

| 合并 | 原来的问题 |
|---|---|
| `model_cache.sh` + `hf_fetch.py` + `flash_trim.py` → `prepare_model.py fetch` | 三个都在回答「模型能不能用」，拆开后 probe 与 fetch 对「什么算完整」各写一遍 |
| `plan_parallel.py` → `prepare_model.py plan` | 卡数与 sp/ep 由 `model_map.json` 声明，不再从 runner 标签或 yaml 推断 |
| `supervise_train.sh` → `run_train.sh` 的 `run_supervised()` | 监督逻辑只服务一次训练启动，独立文件增加参数与临时 `KEY=value` 往返 |
| `verify_env.py` + `torch_guard.sh` + `npu_cleanup.sh` → `env_pod.py` | 三个都在回答「pod 现在什么状态」（体检 / 防破坏 / 复原） |
| `run_pytest.sh` + `junit_merge.py` → `run_tests.sh` | 后者只被前者调用，而且只用了它的单分片路径 |
| `lint_checks.py` + `lint.sh` → `lint_check.py` + 两个检查模块 | 通用工具和仓库契约分开，入口只做汇总 |

反过来，下面这些**不合并**，因为它们回答的是不同的问题：

* `select_tests.py`（选什么）与 `report_gate.py`（算不算过）—— 选错和判错是两类故障，
  合一起会让门禁自己变成黑盒。
* `report_result.py`（事实 → 文字）—— 其他脚本都向它写事实，它不该反过来依赖任何人。
* `install_deps.sh` 已并入 `run_tests.sh install`：workflow 仍用两个 step，
  装挂了与测试挂了仍然分得开。

**事件语义也只放一处**：`ci:full` / `ci:e2e` / `ci:skip-<device>` 与
`mode=pr|full` 的换算在 `select_tests.py` 的 `flags_from_event()` 里，`--verify`
把每个组合钉了一遍。放进 yaml 的话，「标签什么意思」会同时存在于 yaml 和文档，
加一个标签就是两份实现 —— 而两份实现漂了不会报错，只会少跑。

## 约定

* 人看的日志走 **stderr**，机器读的 `KEY=value` 走 `--output`。调用方把它追加到
  `$GITHUB_OUTPUT`，或者直接 `source`。
* 事实写进**文件**，不写进 step outputs。step outputs 在步骤失败时是空的，
  而失败的 run 恰恰是唯一会被人翻出来看的那种。
* 跳过必须带原因。`select_tests.py` 的 skip 原因与 gate 的 `skipped-reason`
  是同一套语义：长期「无声跳过」会腐蚀 gate 的可信度。

## 本机校验

```bash
python3 scripts/ci/lint_select/lint_check.py           # 全项静态体检
python3 scripts/ci/lint_select/lint_check.py --only contract # 只查 job 图与输出契约
python3 scripts/ci/lint_select/lint_check.py --only shellcheck   # 只查 shell
python3 scripts/ci/lint_select/select_tests.py --verify   # 选测规则没退化（CI 也跑这条，失败判红）
python3 scripts/ci/lint_select/select_tests.py --coverage # 覆盖报告（只出 JSON，判绿）

bash -n scripts/ci/unit_tests/*.sh scripts/ci/e2e_tests/*.sh

# 看某个改动会被怎么选
python3 scripts/ci/lint_select/select_tests.py --changed-file src/torchturbo/engine/core.py --explain

# 装依赖前后 torch 有没有被换掉
python3 scripts/ci/common/env_pod.py probe   --file /tmp/t.json
python3 scripts/ci/common/env_pod.py check   --file /tmp/t.json

# 全量计划，不需要网络
python3 scripts/ci/lint_select/select_tests.py --force-all --output /tmp/s.json

# 解析模型对应的启动脚本 / 依赖脚本
python3 scripts/ci/e2e_tests/prepare_model.py plan --map .github/workflows/model_map.json \
  --model-type qwen3_5 --framework lf

# 模型目录完不完整（不需要网络；--layers N 同时校验减层结果）
python3 scripts/ci/e2e_tests/prepare_model.py probe --dir /root/.cache/models/<repo> \
  [--layers N]

# 全量小模型 / 减层大模型（0 = 全量；N = 只保留前 N 层）
python3 scripts/ci/e2e_tests/prepare_model.py fetch --repo Qwen/Qwen3.5-2B --layers 0
python3 scripts/ci/e2e_tests/prepare_model.py fetch --repo Qwen/Qwen3.5-35B-A3B --layers 6

# junit 计数
python3 scripts/ci/report/report_result.py junit --file /tmp/r.json --junit /tmp/pytest.xml

# 门禁判定（手写 env 模拟各种结果）
R_SEL=success SEL_FAILED_STEP=none R_CPU_UNIT=success R_NPU_UNIT=skipped R_NPU_E2E=skipped \
  WANT_NPU=false WANT_E2E='[]' python3 scripts/ci/report/report_gate.py judge --mode pr
```
