# CI 辅助脚本

原来塞在 workflow heredoc 里的逻辑都搬到了这里。每个文件都能在本机直接跑——
这正是目的：之前的写法只能靠 push 上去、等一个 NPU pod 才能验证。

**命名统一为「动词在前」**：`check_*` / `select_*` / `report_*` / `fetch_*` /
`plan_*` / `build_*` / `install_*` / `run_*` / `supervise_*`，外加状态类
`env_pod.py` / `npu_rule.py`。读者看文件名就知道它做什么，不用点进去。
跟 transformers 的 `check_repo.py`、pytorch 的 `win-test.sh` 同一个套路。

> 前缀不是自觉遵守的约定，是 `check_static.py --only naming` 的硬检查；
> 同一个检查还会报「没有任何调用方」的死脚本，让「删掉」成为默认动作。

**按职责划分，一个文件回答一个问题**：

| 文件 | 回答的问题 | 子命令 |
|---|---|---|
| `select_plan.py` | 这次改动该跑什么？ | `--explain` / `--audit` / `--self-test` / `--changed-from git` / `--mode pr\|full --labels …` |
| `check_static.py` | 静态体检（语法 / workflow / 映射表 / 契约 / 镜像 / 阶段） | `--only compile\|workflows\|map\|contract\|selector\|npu-rule\|naming` / `hadolint [Dockerfile]` / `stage` |
| `install_deps.sh` | 依赖怎么装？ | `install`（`MODE=cpu\|npu`） |
| `run_tests.sh` | 这个泳道的测试怎么跑？ | 环境变量驱动（`MODE` / `SHARD` / …） |
| `fetch_model.py` | 模型目录能用吗？没有就弄一份 | `probe` / `find` / `mark` / `fetch` / `trim` |
| `plan_parallel.py` | 卡数 -> sp/ep？ | `--yaml` / `--cards` / `--sp` / `--ep` / `--out` |
| `run_train.sh` | 一次训练冒烟分几步？ | `prepare` / `probe` / `install` / `run` / `report` / `lane` |
| `supervise_train.sh` | 训练进程有没有卡死 / 会不会留下 | — |
| `env_pod.py` | 这台 pod 现在是什么状态？ | `verify` / `capture` / `check` / `cleanup` |
| `build_image.py` | 基础镜像该不该建 / 怎么建 / 发什么引用 | `env` / `tag` / `build` / `verify` / `publish` / `resolve` / `summary` / `cleanup` |
| `report_gate.py` | 这些结果算不算通过？ | `timing` / `judge --mode pr\|full` |
| `report_result.py` | 事实 → 人看的摘要 | `init` / `set` / `phase` / `report` / `notice` / `junit` / `migrate-map` |
| `npu_rule.py` | 这个测试文件要不要真 NPU 卡？ | `--filter` / `--explain` |
| `diag/` | 异常与 NaN 从哪来 | `sitecustomize` 钩子 |

被下面这些 workflow 调用，`.github/workflows` 里的入口只负责声明输入：

```
ci-pr.yml          唯一入口  select -> {cpu-unit-tests, npu-unit-tests, npu-model-tests} -> gate
                    mode=pr（PR 门禁） / mode=full（全量，不选测）
ci-image.yml        构建基础镜像（**只手动 dispatch**）
model_map.json      模型 -> 启动脚本 / 权重 / 端到端画像（被上面两个读）
```

三条泳道**内联**在 `ci-pr.yml` 里，不是可复用 workflow：`uses:` 的 job 一律
显示成 `调用方名 / 被调方名`，图上的泳道名就只能是两段。

yaml 里只留「输什么、调哪个脚本」，超过 10 行的 bash 一律下沉到这里。
除语法高亮之外的好处是 **shellcheck / actionlint 能看**，而且改一行能在本机验。

### 为什么把三个文件合成一个

合并前先问一句：**它们是不是在回答同一个问题？**

| 合并 | 原来的问题 |
|---|---|
| `model_cache.sh` + `hf_fetch.py` + `flash_trim.py` → `fetch_model.py` | 三个都在回答「模型能不能用」，拆开后 probe 与 fetch 对「什么算完整」各写一遍 |
| `verify_env.py` + `torch_guard.sh` + `npu_cleanup.sh` → `env_pod.py` | 三个都在回答「pod 现在什么状态」（体检 / 防破坏 / 复原） |
| `run_pytest.sh` + `junit_merge.py` → `run_tests.sh` | 后者只被前者调用，而且只用了它的单分片路径 |
| `check_static.py` + `lint.sh` → 现在这一个 `check_static.py` | 两者都是「静态体检 + 工具入口」，`lint.sh` 只有 `hadolint` 和 `stage` 两个薄封装 |

反过来，下面这些**不合并**，因为它们回答的是不同的问题：

* `select_plan.py`（选什么）与 `report_gate.py`（算不算过）—— 选错和判错是两类故障，
  合一起会让门禁自己变成黑盒。
* `fetch_model.py`（模型目录）与 `plan_parallel.py`（卡数 -> sp/ep）—— 后者是纯函数式
  计算，前者要下网、改磁盘。合一起就是一开始那个“一个文件两个问题”的老毛病。
* `run_train.sh`（编排）与 `supervise_train.sh`（进程监督）—— 后者带 trap 语义，
  混进编排里会让人不敢改它。
* `report_result.py`（事实 → 文字）—— 其他脚本都向它写事实，它不该反过来依赖任何人。
* `install_deps.sh`（装依赖）与 `run_tests.sh`（跑测试）—— 装挂了和测试挂了是两类故障，
  日志里分得开才查得动。

**事件语义也只放一处**：`ci:full` / `ci:e2e` / `ci:skip-npu` 三个标签与
`mode=pr|full` 的换算在 `select_plan.py` 的 `flags_from_event()` 里，`--self-test`
把每个组合钉了一遍。放进 yaml 的话，「标签什么意思」会同时存在于 yaml 和文档，
加一个标签就是两份实现 —— 而两份实现漂了不会报错，只会少跑。

## 约定

* 人看的日志走 **stderr**，机器读的 `KEY=value` 走 `--output`。调用方把它追加到
  `$GITHUB_OUTPUT`，或者直接 `source`。
* 事实写进**文件**，不写进 step outputs。step outputs 在步骤失败时是空的，
  而失败的 run 恰恰是唯一会被人翻出来看的那种。
* 跳过必须带原因。`select_plan.py` 的 skip 原因与 gate 的 `skipped-reason`
  是同一套语义：长期「无声跳过」会腐蚀 gate 的可信度。

## 本机校验

```bash
python3 scripts/ci/check_static.py          # 全项静态体检（含命名约定）
python3 scripts/ci/check_static.py --only naming   # 只查文件名 / 死脚本
python3 scripts/ci/select_plan.py --self-test  # 选测算法没退化
python3 scripts/ci/select_plan.py --audit      # 覆盖体检

bash -n scripts/ci/*.sh

# 看某个改动会被怎么选
python3 scripts/ci/select_plan.py --changed-file src/torchturbo/engine/core.py --explain

# 装依赖前后 torch 有没有被换掉
python3 scripts/ci/env_pod.py capture --file /tmp/t.json
python3 scripts/ci/env_pod.py check   --file /tmp/t.json

# 分片计划，不需要网络
python3 scripts/ci/select_plan.py --force-all --shard 0/5 --output /tmp/s.json

# 8 卡下的并行策略
python3 scripts/ci/plan_parallel.py --yaml scripts/runs/bailing_v3/sft.yaml \
  --cards 8 --out /tmp/tt.yaml --output /tmp/plan.out && cat /tmp/plan.out

# 模型目录完不完整（不需要网络）
python3 scripts/ci/fetch_model.py probe --dir /root/.cache/models/<repo>

# junit 计数
python3 scripts/ci/report_result.py junit --file /tmp/r.json --junit /tmp/pytest.xml

# 门禁判定（手写 env 模拟各种结果）
R_SEL=success SEL_STAGE=ok R_CPU_UNIT=success R_NPU_UNIT=skipped R_NPU_MODEL=skipped \
  WANT_NPU=false WANT_E2E='[]' python3 scripts/ci/report_gate.py judge --mode pr
```
