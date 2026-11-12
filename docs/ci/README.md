# CI 体系

**三个 workflow，一个必过检查（`gate`）。** 改动的路径决定跑哪几条泳道，其余一律跳过——
但**跳过必须带原因**，否则门禁判红。

```
.github/workflows/
  ci-pr.yml          唯一入口：mode=pr 选测门禁 / mode=full 全量
  ci-manual.yml      手动单泳道（无 lint / 无 select / 无 gate，排查用）
  ci-image.yml        构建基础镜像（只手动）
  model_map.json      模型 -> 下载 / 安装 / 启动 / 训练参数

pull_request ──► ci-pr.yml
                   ├─ select   静态检查（Ruff / shellcheck / actionlint / hadolint
                   │           / 映射表 / job 图契约 / 选择器防篡改）
                   │           然后算改动 → 出计划（泳道开关 + 文件清单）
                   │           两者**同一个 job、同一个 runner**
                   ├─ cpu-unit-tests    CPU pytest（选中的文件，不分片）
                   ├─ npu-unit-tests    NPU pytest（定向跑选中文件；全量触发整条泳道）
                   ├─ npu-e2e-tests     E2E：按模型训练用例逐个跑一步
                   └─ gate     唯一 required check

手动 ─────────────► ci-pr.yml（mode 选 pr / full） / ci-image.yml
单泳道排查 ───────► ci-manual.yml（dispatch 或 ci:manual:<lane> 标签）
```

测试配置在仓库根的 `pytest.ini`：测试根、导入路径、marker 契约。**不放进
`pyproject.toml`** —— pytest 只读一个配置文件且 `pytest.ini` 优先，写在 pyproject
里的那段会被静默忽略，`lint_check.py` 的 `markers` 项会直接报错。

脚本目录与阶段对应：

| 阶段 | 脚本目录 |
|---|---|
| lint-select | `scripts/ci/lint_select/` |
| CPU / NPU 单测 | `scripts/ci/unit_tests/`（同一套 runner，`MODE` 区分） |
| NPU E2E 用例 | `scripts/ci/e2e_tests/` |
| 镜像构建 | `scripts/ci/image/` |
| 跨阶段能力 | `scripts/ci/common/`、`scripts/ci/report/` |

**每条泳道在图上只有一个盒子，而且名字就是泳道名。** 两个约束都是结构性的：

- 可复用 workflow 会把被调文件的**所有** job 展开成图上的盒子，而 `uses:` 的
  job 一律显示成 `调用方名 / 被调方名`，两段合不成一段 —— 所以泳道必须**内联**。
- `strategy.matrix` 会把一个 job 乘以条目数 —— CPU **不分片**；每个选中的 E2E
  case 是一个独立 matrix entry，最多同时运行 2 个。
- job key 带 `l1_` / `l2_` / `l3_` 前缀：图上的同级 job 按 key 排序，这是
  cpu → npu-unit → npu-e2e 从上到下的唯一保证。

## 拓扑：静态检查 + 计划算好，然后三条泳道并行

```
                            ┌─► cpu-unit-tests
select（静态检查 → 算计划） ─┼─► npu-unit-tests         ─► gate
                            └─► npu-e2e-tests
```

三条泳道**同为 `needs: select`**，谁也不依赖谁，同时启动。`gate` 的 `needs`
里第一个就是 `select` —— 这是「提前结束」通路：select 没过（静态检查挂 / 选择器
算不出计划）时，三条泳道的 `if` 全为假、一张卡都不开，而 gate 因为
`if: always()` 仍会等到那三条 `skipped` 落地后立刻给结论。**整链 ~20 s 结束**，
不是 ~14 min。

### 静态检查与算计划为什么在**同一个 job** 里

它们是严格串行的两个步骤：静态没过就不该算计划。拆成两个 job 只会：

- 白占一个 runner 名额（`CI_CPU_RUNNER` 是稀缺资源）；
- 多一次 `checkout`，而且计划那边还必须是 `fetch-depth: 0` 的全量拉取；
- 多一段 `needs` + `if` 的样板，每个条件都是一处可能写错的地方。

合成一个 job 之后，「静态没过就不算计划」是 **step 的天然语义**：前一 step 失败，
后面的直接不执行。一行 `needs` 都不用写。

代价只有一处：job 红了只看 `needs.select.result` 分不清倒在哪一步。所以最后
一步用 `if: always()` 把 `failed_step`（`none` / `lint` / `image` / `select` /
`cancelled`）写进 job 输出，`report_gate.py` 用它把失败归因到具体步骤。

| failed_step | 含义 |
|---|---|
| `lint` | `lint_check.py` 的任一项未通过（Ruff / shellcheck / actionlint / hadolint 或 Turbo 契约） |
| `image` | 基础镜像解析未通过 |
| `select` | 选测计划没算出来 |
| `none` | 没有步骤失败 |
| `cancelled` | run 被取消 |

`lint_check.py` 是唯一的 lint 入口：它先跑完所有检查、再统一报错，所以一个检查
失败不会掩盖后面的问题。Ruff 由 workflow 以固定版本安装；actionlint / hadolint /
shellcheck 的版本固定在 `lint_common_check.py` 里，从版本化 release 下载，不再
`curl | bash`。这几项都是秒级工具，因此**无条件跑**，不复制一份 diff 判断来决定
要不要跑。通用检查通过后，`lint_turbo_check.py` 再校验 job 依赖图、选择器、模型映射
和 marker 契约。

`lint_check.py` 的 `contract` 检查按脚本调用和 `needs` 依赖图识别 planner、lane 和
唯一 terminal job，不依赖 job 名字。写错会当场报错：

1. 跑 `select_tests.py` 的 planner job 必须**同时**跑 `lint_check.py`（别拆成两个 job）；
2. 同一个 job 里 `lint_check.py` 必须**排在** `select_tests.py` 之前；
3. 每个 workflow 只能有一个 planner；
4. planner 声明和下游读取的 `needs.<planner>.outputs.*` 必须双向匹配；
5. lane 必须依赖 planner，且不能互相串行；
6. workflow 必须落到唯一 terminal job，并能沿 `needs` 到达 planner 和所有 lane。

### 为什么是并行而不是串行

串行（`cpu-unit → npu-unit → npu-e2e`）看起来更省卡，但它只省下
「CPU 红时本来就不该开的卡」，而本仓 cpu-unit 目前有 37 个历史红的基线 —— 串行的实际效果是让每个 PR 都白等
~90 s，却省不到什么。并行后：

| 指标 | 串行 | 并行 |
|---|---|---|
| 最坏墙钟（实测） | 16+90+119+605 ≈ **830 s** | 16+605 ≈ **621 s** |
| 总卡·秒 | 5078 | **5078（完全相同）** |
| 峰值并发 | ≤ 8 卡 | 10 卡（2+8） |
| cpu-unit 红时 | NPU 卡不开 | NPU 已开，**白烧 5078 卡·秒** |

也就是说并行**不增加总卡消耗**，只抬高峰值并发。代价是 CPU 泳道红时 NPU 会白跑——
这是已知取舍，接受它是因为墙钟收益（~3.5 min/PR）更值钱。

注意模型 E2E 按 case 拆成独立 job/pod，`max-parallel: 2` 限制 NPU 峰值并发；
NPU unit 可能占用其中一个槽位，但模型之间不会共享 pod，训练状态也不会跨 case
泄漏。实际能否同时启动仍由 NPU runner 数量决定。每个 job 的宏观流程都是环境
检查、训练、环境清理。

### gate 的判定

并行之后不再有「上游红了所以下游 skipped」这种因果，所以 `report_gate.py` 里也**不再
有** `cpu_bad` / `npu_bad` 这类传递判断。跳过只剩两种原因：

| 情形 | 结论 |
|---|---|
| 自己的 `*_run=false` | ⏭ 绿（写明原因） |
| 自己的 `*_run=true` 却是 skipped | ❌ `skipped-unknown`（真问题） |
| select 没过（静态 / 计划） | ❌ 红，三条泳道记「计划未算出，按设计未启动」 |

**当前（调试期）所有 workflow 都只手动触发**：`push` / `schedule` 已全部注释掉，
写在哪里、怎么恢复都留在原处。恢复后应该是：

```
PR ──────────────► ci-pr.yml（select[静态+计划] -> {三条测试泳道} -> gate）
每晚 ─────────────► ci-pr.yml mode=full（全量，不选测）
每晚 ─────────────► ci-image.yml（基础镜像，内容没变就跳过）
手动排查 ─────────► ci-pr.yml mode=pr + force_all（跑全部 CPU/NPU 单测）
```


### 为什么全量和 PR 是同一个文件

两者的差别只有三行：select 里是否 `--force-all`、gate 用 `judge --mode pr|full`、
以及 `concurrency`。早先拆成两个文件的结果是 90% 的行重复写两追，
而“全量忘了同步 PR 的新泳道”是一种只能靠 review 发现的错。

现在用一个 `CI_MODE` 环境变量分叉，静态检查（`lint_check.py`）同时扫这一个文件，
两边不可能漂。`mode` 只影响选测与判定，泳道体完全共用。

### 为什么泳道要内联

图上的泳道名只能是**一段**，而可复用 workflow 的 job 一律显示成
`调用方名 / 被调方名`（早先是 `1. cpu-unit / run`），GitHub 没有合并成一段的选项。
所以三条泳道直接写在这个文件里：job key 用 `l1_` / `l2_` / `l3_` 前缀保证顺序，
`name` 就是泳道名。

## 决策层：一个计划，每条泳道一个开关

分支判断**只允许存在一处**，在 `scripts/ci/lint_select/select_tests.py` 里。它输出一份计划：

```
cpu_unit_run=true   npu_unit_run=true   npu_e2e_run=false
cpu_unit_files=["tests/ops/cpu/test_x.py", ...]
npu_unit_files=["tests/ops/npu/test_x.py", ...]
e2e_cases=[{"model_type": "qwen3_5", "layers": 0, "profile": "full", "reason": "..."}]
```

yaml 里**只读布尔值，不做任何判断**。没有 `if: contains(github.event.pull_request.labels...)`，
没有 `if: github.event_name == 'push'`，也没有 `if: needs.x.outputs.y != '[]'`。

向下游传「跑不跑」，不传「为什么跑」：一旦 yaml 也要理解路径语义，就出现了
第二份判断逻辑，两份必然漂，而且漂了不会报错——只会少跑。

三条泳道用**同一张模板**，只换中间那一行：

```yaml
if: |
  always() && needs.select.result == 'success' &&
  needs.select.outputs.npu_unit_run == 'true'
```

`always()` 是为了**不让 `needs` 的隐式跳过替我们决策**：select 失败时三条泳道
本该全部 skipped，但 gate 还要跑起来给结论；而只要写了 `always()`，就必须自己
显式拦住「计划没算出来」这一种，所以第一行永远是
`needs.select.result == 'success'`。

`report_gate.py` 拿同样的开关做判定：该跑却 `skipped` 是 `skipped-unknown`（红），
不该跑而 `skipped` 是 ⏭（绿）。两边用**同一组变量名**，否则摘要会给出
和实际执行相反的解释。

### 决策层契约由 select 里的静态检查看守

`lint_check.py` 的 `selector` 项会**真的跑一次** `select_tests.py --verify`，
断言选测规则没有退化；`contract` 项再逐个 workflow 校验
`needs.<planner>.outputs.X` 都已声明、lint 与算计划仍在同一个 job 且顺序正确。

这类 bug 的特点就是「能跑、能绿」：拼错一个输出名 → GitHub 给空字符串 →
`== 'true'` 永假 → job 静默不跑 → gate 把它当成「不需要」。实测中它让选测
失效了好几天才被发现。

`selector` 项守另一头：`scripts/ci/**` 的改动必须触发全量。否则一个 PR
可以把自己的选择器改成「永远选 0 个测试」并且全绿——那就是自己给自己发通行证。

## 三条泳道怎么被选中

表中 ✅ = 该泳道会跑；「—」= `*_run=false`，job 直接 skipped（带原因）。
定向改动时 npu-unit 跑选中的 NPU 文件；镜像、依赖或全量触发时跑整条泳道
（`npu_unit_files=[]`，由 `run_tests.sh` 用 `-m npu` 选择；混合文件在 CPU/NPU
两条泳道都会出现，由 marker 表达式拆分）。

设备用例的契约只有一层：测试文件里的显式正向 marker。纯 CPU 文件使用
`pytestmark = pytest.mark.cpu`；混合文件在 CPU 用例上加 `cpu`，在 NPU 用例上加
`npu`；纯 NPU 文件使用 `pytestmark = pytest.mark.npu`。至少两卡再
`multi_npu`（必须同时有 `npu`），4/8/16 卡分别用 `npu_4` / `npu_8` / `npu_16`。
CPU 泳道执行 `-m cpu`，NPU 泳道执行 `-m npu`。`cuda` / `xpu` / `mps` 已预留，
但没有配置对应泳道前 lint 会直接失败，不会静默落进 CPU。没有 `torch_npu` 或卡数
不足时，由 `pytest.importorskip` / `tests/support/npu.py` 在运行时 skip。

| 改动 | 静态 | cpu-unit | npu-unit | npu-e2e |
|---|:--:|:--:|---|:--:|
| `src/torchturbo/ops/**` | ✅ | 选中的 CPU 文件 | 选中的 NPU 文件；全量触发时整条泳道 | 最便宜的用例 |
| `ops/providers/<device>/<op>.py` | ✅ | 同名算子的测试 + `test_dispatch.py` | 同名算子的 NPU 测试；全量触发时整条泳道 | 最便宜的用例 |
| `models/<族>/{lf,swift}_template.py`、`mm_plugin.py` | ✅ | 负责发现它的适配器测试 | — | lf/swift 对应用例 |
| `src/torchturbo/models/<族>/**` | ✅ | 选中的 CPU 文件 | — | 该族用例 |
| `src/torchturbo/{engine,models/modules,parallel}/**` | ✅ | 选中的 CPU 文件 | — | 最便宜的用例 |
| `src/torchturbo/adapters/**` | ✅ | 选中的 CPU 文件 | — | lf/swift 对应用例 |
| `src/torchturbo/runtime/**` | ✅ | 选中的 CPU 文件 | — | — |
| `docker/npu/**` | ✅ | 选中的 CPU 文件 | **✅ 全跑** | 最便宜的用例 |
| `pyproject.toml` | ✅ | **全量** | **✅ 全跑** | 最便宜的用例 |
| `.github/workflows/**`、`scripts/ci/**`（`model_map.json` 除外） | ✅ | **全量** | — | `bailing_hybrid` tiny 兜底 |
| `.github/workflows/model_map.json` | ✅ | **全量** | — | `bailing_hybrid` tiny 兜底 |
| `scripts/runs/<dir>/**` | ✅ | 源码里真引用它的测试 | — | 该 `run_dir` 的模型 |
| 其它 `scripts/**` | ✅ | 源码里真引用它的测试 | — | — |
| `tests/**` | ✅ | affected | — | — |
| `docs/**`、`*.md` | ✅ | — | — | — |

cpu-unit 只跑选中的文件，所以「一个改文档的 PR」连 CPU 泳道都不开——静态那 15 s
就是它的全部成本。

**只改注释 / 文档串的 `.py` 改动会被忽略**，按「没改」处理。判据是两边剥掉注释与
文档串之后的 AST 是否完全相同（`code_only()`），不是看 diff 文本——后者会被字符串
里的 `#` 骗到。这条对标 transformers 的 `diff_is_docstring_only()`：改一句注释拉起
一条 NPU 泳道是本仓库最大的一类白跑。

它和「拿不到改动」必须分开：`git diff` 为空但有 merge-base 失败的可能时仍然全量
（fail-open）；而「diff 非空、只是全是注释」是明确结论，直接空选。两者在
`--explain` 里都有明确输出（`full run because: no-changed-files` vs
`cpu_unit skip  comment / docstring-only change`）。

`scripts/**` 不按目录名映射。这些脚本是用 `importlib` 按路径加载的，import 图里
根本没有边，所以只能查测试源码的文本：命中完整路径或文件名才算相关。早先这里是
「`scripts/runs/**` → `tests/scripts/**`」，但后者实际测的是
`scripts/{checkpoint,dataset,dequant}` —— 纯属 `scripts` 这个词撞车，改一个启动
脚本会白白拉进三个无关文件（其中一个还在收集阶段就报错）。
`src/**` 的补充只有五类，每类都有明确的因果（括号里就是 `--explain` 里的键名）：

- **reverse_deps（反向 import 闭包）**：改 `a.py`，凡（间接）import 它的测试都跑。
- **mirror（同深度镜像目录）**：`tests/<与 src 完全同层级>` 存在时，那层目录就是本次
  改动的单元。**只认同深度，绝不上浮** —— 上浮是早先最大的误选来源：本仓测试目录是
  扁平的，`models/qwen3_5_moe_text/model.py` 会浮到 `tests/models/`（72 个测试），
  `engine/metrics/mfu.py` 会浮到 `tests/engine/`（38 个），
  `ops/providers/native/swiglu.py` 会浮到 `tests/ops/providers/`（3 个 GDN 用例，
  它们其实测的是 `_fla_gdn.py`）。同深度下这些目录都不存在，噪声归零。
- **包门面（算在 reverse_deps 里）**：改 `<pkg>/__init__.py` 展开成「它底下所有
  子模块」。`import pkg.sub` 会先执行包的 init，所以 init 的改动影响每一个 import
  子模块的人——这条边 import 图里不存在（transformers 对 init 用的是同一套处理）。
- **convention（约定接线）**：静态图里没有边的动态发现——provider 由
  `pkgutil.walk_packages` 枚举、模板插件由 `importlib.import_module` 按名字加载。
  按命名约定接：`ops/providers/<device>/<op>.py` → 同名算子的测试 + 发现机制自己的
  测试（文件名带 `dispatch` / `providers`）；`models/<族>/{lf,swift}_template.py`、
  `mm_plugin.py` → 负责发现它的那套适配器的测试。规则表在 `select_tests.py` 的
  `PROVIDER_RE` / `DISCOVERER_RULES`。
- **path_ref（路径反查）**：`spec_from_file_location` 按路径加载（`test_cpu_resident.py`
  就是这么测的）、以及按路径加载的启动器；只能查测试源码里有没有写死这个路径。
- **test_tree（测试树自身）**：局部 `conftest.py` 由 pytest 按目录自动加载、没有
  import 边，按**目录作用域**接；其余测试 helper（`tests/models/flash_attention_fakes.py`
  这种）沿测试树的反向 import 图走。根 `tests/conftest.py` 已删除，NPU 分类只读
  测试文件里的显式 marker。

另有一个事后兜底 `deleted`：被删掉的模块已经不在图里，按名字找回还在 import 它的
测试。

五层都接不上就**空选**。空是可见的（`lanes.*.why` 会写「没有选中」），拖一层无关
用例进来是不可见的。

两张表对应两个脚本：路径 → 用例是 `scripts/ci/lint_select/select_tests.py`，路径 → 用例在
`select_tests.py` 的 `pick_e2e()` 里，用例本身写在 `model_map.json` 的 `e2e` 字段。

改 CI 自身为什么跑全量 CPU：这些脚本就是「怎么跑测试」本身，改错了会**静默少跑**，
只有全量能看出来。同时只启动 `bailing_hybrid-lf-sft` 这一个 tiny E2E 用例做真实
训练链路兜底；若同一 PR 还改了具体模型源码，则以具体模型用例为准，不叠加兜底。

## 镜像：谁生产、谁消费

| | 做法 | 为什么 |
|---|---|---|
| 触发 | **只手动 dispatch**（`schedule` 已注释掉） | 一次 4~5 h / 几十 GB；镜像变了是最难排查的问题，先不给它自动改的机会 |
| 要不要建 | 内容指纹 tag 已存在 → 跳过 | 没改动时整轮 2~3 分钟，而不是 5 小时 |
| tag | `base-<年周>-c<docker/npu 内容哈希 8 位>` | 同名同内容，不靠外部状态就能判断「建过没」 |
| 年周 | `2026w45` | 每周强制重建一次，才能发现 cann / pip 上游漂移 |
| PR 用哪个镜像 | `base-image-ref` artifact → 固定链接 `…:base-latest` | 读最近一次构建的不可变 tag；「PR 悄悄换了一个镜像」是最难排查的一类问题 |

构建成功后会把 `{ref, tag, digest}` 写成 `base-image-ref` artifact（保留 90 天），
PR / 全量的 `select` 用 `--repo "$GITHUB_REPOSITORY"` 读最近一份。解析顺序：
**人工覆盖 → artifact → 固定链接**。artifact 过期或从未构建过时回退到固定链接
并告警——kubelet 的 `IfNotPresent` 只比 tag 字符串，内容变了不会重拉。
要退回「只吃固定链接」，把 `resolve` 那行的 `--repo` 删掉即可。
**整条链路不需要任何长期凭据** —— `gh variable set` 对 `GITHUB_TOKEN` 是 403，
所以才走 artifact。

需要立刻重建（比如怀疑基础镜像坏了）：dispatch `ci-image.yml` 并勾 **force_rebuild**。
想临时切镜像排查：设仓库变量 `BASE_IMAGE_TAG`，它优先于 artifact。

## 全量闸门（`full_reasons`）

任一命中即全量（`mode=all`），`--explain` 会打印 `full run because: ...`：

| 原因（JSON 里的值） | 触发条件 | 管在哪 |
|---|---|---|
| `no-changed-files` | 改动文件为空（`merge-base` 失败、浅克隆没历史……） | `select()` |
| `force-all` | `ci:full` 标签 / dispatch 的 `force_all` / `mode=full` | `flags_from_event()` → `select()` |
| `too-many-files(N)` | 改动 **> 300 文件**（`MAX_CHANGED_FILES`） | `select()` |
| `too-many-tests(N)` | 选中的测试 **> 90 个文件**（`MAX_SELECTED_TESTS`） | `select()` |
| `deps-changed` | 动了 `pyproject.toml`：装不上就全红，挑用例没意义 | `select()` |
| `ci-config-changed` | 动了 `.github/**` 或 `scripts/ci/**`：改错了会静默少跑 | `select()` |

另有两条不在 `select()` 里：选择器异常 / 非 0 退出由 workflow 兜底，`mode=full`
只是把 `force-all` 传进来。`push` / `schedule` 当前都不触发 CI。

`too-many-files` 的方向是审过的：**大 PR 该跑更多测试，不是更少**。改 1000 个文件
的 PR 往往是大重构，正是全量最能抓问题的时候；全量不分片，CPU 也就 5~8 min。

`too-many-tests` 同理：命中面超过六成时，「精选」已经不比全量便宜，而全量的语义是
自明的。一份「看起来挑过」的不完整集合，没人能判断它漏了什么。阈值没压更低，是因为
经 `torchturbo.engine` 门面的 reverse_deps 会合法地拉出 60+ 个文件，那不是噪声；真正
该升级的是 `runtime/**`、`logging.py`、`ops/base.py` 这类全局基座。

`ci-config-changed` 只把 **CPU** 扩成全量；普通 CI 控制文件另起一个
`bailing_hybrid-lf-sft` E2E 兜底。NPU unit 要不要起，看的是「这次选中的文件里有
没有真卡测试」，与 `mode` 无关。

## 标签

| 标签 | 效果 |
|---|---|
| `ci:full` | 三条泳道全量，E2E 跑全部用例 |
| `ci:e2e` | 路径没命中时也强制跑 NPU 泳道（起 NPU + 最便宜的用例） |
| `ci:e2e:strict` | E2E 显式跳过时**判红**；模型获取超时始终判红 |
| `ci:skip-npu` | 强制关掉所有 NPU 泳道（含 E2E），hotfix 用；将来是 `ci:skip-<device>` |

标签的语义在 `scripts/ci/lint_select/select_tests.py` 的 `flags_from_event()` 里，**不在 yaml**
里：yaml 只把标签拼成字符串传过去。这样 `--verify` 能把每种组合钉死，
而「加一个标签」只需要改一处，不会出现 yaml 和文档两套说法。

## fork PR 的 NPU 泳道不自动跑

fork PR 拿不到 secrets，而两条 NPU 泳道都需要 SWR 凭据。更关键的是：
**把 secrets 递到 fork 的代码上等于 RCE**，`pull_request_target` 就是这么出事的，
本仓库明确不用。

所以 fork PR 的 NPU 泳道被跳过，并在 Summary 里写 `⏭ 跳过 | fork PR（secrets 不递）`。
需要跑的话，由 maintainer 把 commit 推进本仓分支：

```bash
git fetch origin pull/123/head:ci/pr-123
git push origin ci/pr-123
gh workflow run ci-pr.yml -F ref=ci/pr-123
```

或者直接 dispatch `ci-pr.yml`：`mode=pr` + `force_all` 跑全部单测，
`mode=full` 连 E2E 一起跑。

## 时间预算

不追求极限，只要求**不失控**。

| 档位 | 阈值 | 违反时 |
|---|---:|---|
| 目标墙钟 | ≤ 25 min | `::warning::` |
| 告警线 | > 35 min | 告警 |
| 硬上限 | 45 min（`timeout-minutes`） | job 被杀，gate 红 |

`queue`（排队）与 `wall`（执行）分开报：只看总和无法区分「该扩容」和「该优化步骤」。

超时只负责**拦住卡死**，不负责卡住正常但较慢的运行——所以 step 级超时给到实测值的
3~4 倍，而不是贴着实测值。

超时之后一定有收尾（`if: always()` → `scripts/ci/common/env_pod.py cleanup`）：被杀掉的训练会
留下占卡的子进程、没释放的显存和半个 checkpoint，下个 job 会以完全无关的方式失败。

## NPU 池并发：matrix 上限 + runner 排队，不要用 job 级 `concurrency`

第一版在泳道文件里按 runner 标签分过组
（`ci-tests-<label>`，`cancel-in-progress: false`），想的是"让手动跑和 PR 排同一个队"。
实测**这个做法会吃掉并发**：

> 同一个 concurrency group 里，GitHub 只保留**一个** pending 的 job；
> 新排进来的会把前一个 pending 的 cancel 掉（`cancel-in-progress: false` 只保证
> 不打断 *正在跑* 的那个）。

全量第一次 run 就是这么挂的：同时在队的 job 只剩最后一个，其余的都被
`cancelled`。

所以现在三个泳道文件里**没有任何 `concurrency`**。NPU matrix 用 `max-parallel: 2`
限制上界；实际并发仍由 runner 数量决定，self-hosted runner 忙着的时候，下一个 job
天然停在 `queued`。要给某个池进一步限流，应该去 ARC / runner group 那一层做。

workflow 级 `concurrency` 现在只留在一个入口文件里（`ci-pr.yml`），并且按
模式分 group：PR 用 `pr-<number>`（新 push 顶掉旧 run），全量用 `full-<run_id>`
（每次唯一，永远排不到队）。

## `timeout-minutes` 必须包 `fromJSON`

`workflow_dispatch` 里声明成 `type: number` 的输入，实际传进 `inputs` 的是**字符串**
（`"30"`，不是 `30`）。直接写

```yaml
timeout-minutes: ${{ inputs.timeout_minutes || 60 }}
```

在 dispatch 触发时求值结果是字符串 `"30"`，而 `timeout-minutes` 只接受 number。
GitHub 的反应不是报错，而是**把这个 job 从 run 里整个丢掉**：run 页面只看得到
另一个被 skip 的 job，结论是 failure，没有任何提示。很容易被误判成"runner 没起来"。

写法必须是：

```yaml
timeout-minutes: ${{ fromJSON(inputs.timeout_minutes || 60) }}
```

`fromJSON` 对 number 和字符串两种输入都成立（number 输入在 `workflow_call` 下
仍然走这条路，实测同样正常），所以两条触发链可以用同一份表达式。

这个 bug 让泳道文件的**手动入口**完全不可用 —— 提交
`f2acd5d` 之后的三次 dispatch 全是这个原因。定位过程：二分探针（`applied` 过的
`zz-probe*.yml` 已删除），`p`/`q` 两个不带 `timeout-minutes` 的 job 正常创建，
而带表达式的 `r` 直接消失。

## 需要配的东西

| 名称 | 类型 | 必需 | 用途 |
|---|---|:--:|---|
| `SWR_USERNAME` / `SWR_PASSWORD` | secret | ✅ | 拉私有基础镜像；image CI 推送 |
| `BASE_IMAGE_TAG` | variable | 可选 | **人工覆盖**镜像引用，平时不用设（不设就用固定链接） |
| `CI_CPU_RUNNER` | variable | 可选 | 默认 `ubuntu-22.04-arm` |
| `CI_NPU_RUNNER` | variable | 可选 | 默认 `linux-aarch64-a3-800i-2` |
| `CI_E2E_RUNNER` | variable | 可选 | 默认 `linux-aarch64-a3-800i-8` |

端到端用几张卡、sp/ep 各是多少，写在 `model_map.json` 每个 case 的
`cards` / `sp` / `ep` 里。E2E 的 `fetch_timeout` / `case_timeout` 是执行策略，
统一写在 `scripts/ci/lanes.yml` 的 `e2e_defaults` 中，默认分别是 1800s / 7200s；
需要手工排查时可用同名环境变量覆盖。

**不从 runner 标签推断**：真实标签带位置后缀（`...-8-cn12-001`），取末段会得到
`001`，于是 8 卡 pod 只起 1 个进程却全绿。`cards` 必须能整除 `sp`，`cards / sp`
必须能整除 `ep`，否则拒绝启动；当前上限是 16 卡，常规 E2E 用 8 卡。

小模型下载全量权重（`download.layers=0`）；大模型按 `download.layers=N` 只保留
前 N 层，同时改写嵌套的 `text_config`、逐层列表和 MTP 开关，并裁剪索引指向的
分片。两种模式共用同一个 `probe`，以 config、index 和索引声明的分片文件
一致作为“可用”的判据。减层目录用 `.tt-cache-marker` 标记内容，并在同级
`.<目录名>.lock` 上加文件锁，避免同一缓存目标被并发裁剪；全量模型不加锁。
`N` 取模型的最小不重复结构，而不是固定值：Qwen3.5-35B-A3B 的完整周期是
4 层（3 层 linear attention + 1 层 full attention）；Ling-3.0-flash 的
`layer_group_size=6` 且前 2 层为 dense，因此需要 6 层才能覆盖全部结构。

## 本地怎么验

```bash
python3 scripts/ci/lint_select/lint_check.py             # 静态体检（含选择器自检）
python3 scripts/ci/lint_select/lint_check.py --list      # 看有哪些检查
python3 scripts/ci/lint_select/lint_check.py --only shellcheck
python3 scripts/ci/lint_select/select_tests.py --verify   # 单独跑选测规则断言
python3 scripts/ci/lint_select/select_tests.py --coverage # 覆盖报告（只出 JSON，判绿）
bash scripts/ci/unit_tests/run_tests.sh                   # 真在本机跑一遍（MODE=cpu）
python3 scripts/ci/report/report_gate.py judge --mode pr  # 用手写的 env 验判定
python3 scripts/ci/image/build_image.py tag --dry-run     # 看当前内容的镜像 tag
python3 scripts/ci/image/build_image.py selftest --auth-only  # 验 SWR 凭证；去掉 --auth-only 会真实上传再删除

# 某个改动会被怎么选（含泳道开关和文件列表）
python3 scripts/ci/lint_select/select_tests.py --changed-file src/torchturbo/engine/core.py --explain

# 装完依赖 torch 有没有被 pip 换掉
python3 scripts/ci/common/env_pod.py probe   --file /tmp/t.json
python3 scripts/ci/common/env_pod.py check   --file /tmp/t.json
```

验泳道判定最快的方法（不跑 CI，只看 gate 怎么说）：

```bash
run() { env R_SEL=$1 SEL_FAILED_STEP=$2 R_CPU_UNIT=$3 R_NPU_UNIT=$4 R_NPU_E2E=$5 \
  CPU_UNIT_RUN=$6 NPU_UNIT_RUN=$7 NPU_E2E_RUN=$8 SUMMARY=t \
  python3 scripts/ci/report/report_gate.py judge --mode pr; }

run success none     skipped skipped skipped false false false  # 改文档：什么都不跑
run success none     success skipped skipped true  false false  # 改模型：cpu-unit
run success none     skipped success success false true  true   # 改 ops：两条 NPU
run success none     success success success true  true  true   # 全绿
run failure lint     skipped skipped skipped true  true  true   # 静态倒：其余按设计未启动
run failure image    skipped skipped skipped true  true  true   # 镜像解析倒
run failure select   skipped skipped skipped true  true  true   # 计划没算出
run failure cancelled skipped skipped skipped true true  true   # run 被取消
```

## 一个刻意的取舍

cpu-unit-tests 的**裸机**泳道不用 `pip install -e ".[dev]"`，而是先装
「可用版本」的 torch，再 `--no-deps` 装本包。

原因：`pyproject` 钉的是 `torch>=2.10.0`（NPU 镜像里那套），GH 托管机只能从 PyPI 拿
arm64 CPU wheel。钉死了装不上，放开了又和镜像不一致——所以裸机泳道只承诺
「纯 python 逻辑能跑」，精度相关的验证交给容器泳道。

## 实测记录

`tmp_record/` 里是当初定这些数字用的原始日志（镜像拉取 7.4 MiB/s、训练 163 s 等）。
它们原来放在 `.github/workflows/` 下，改个 md 就会触发整轮 CI，所以搬到了这里。
