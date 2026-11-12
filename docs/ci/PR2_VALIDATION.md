# PR2 验证记录：CPU 通用 runner

## 范围

- `plan.py` 输出 CPU `cpu_jobs` matrix。
- `ascend-pr.yml` 的 `cpu-unit-tests` 改为 matrix，每个 entry 执行一个 plan job。
- `run_job.py` 按 step 顺序创建 / 复用 venv，逐进程运行 pytest 并汇总 JUnit。
- `venv_manager.py` 以依赖内容和 Python 版本生成 cache key；不删除共享 cache。
- `env_pod.py` 增加 `--device cpu`，CPU 分支不访问 davinci 或 torch_npu。
- `scripts/ci/bin/torchrun` 固定使用当前 `VIRTUAL_ENV` 的 Python。
- `envs.toml` 支持 `overrides`，Swift 的 `datasets==5.0.1` 在 `ms_swift` 之后安装。
- 旧 `npu-unit-tests` / `npu-e2e-tests` 保持原样，作为回滚路径。

## 已执行校验

```bash
python3 scripts/ci/lint_select/lint_check.py
python3 scripts/ci/plan.py --verify
python3 scripts/ci/local_lint_select_test.py
```

结果：

- lint 9 项通过；
- `plan facts OK (10 E2E cases / 162 tests)`；
- selector 的 5 个自检阶段全部通过；
- 全量 selection 生成 4 个 CPU matrix entry，全部 `device=cpu`；
- `cpu_jobs` 序列化后约 14 KB，远低于 step output 上限；
- `workflow_dispatch.test_path` 已验证文件、目录和 node id：node id 原样传给
  pytest；NPU-only 路径产生空 CPU matrix，不误跑 CPU lane；
- `env_pod.py probe/check --device cpu` 在本机 `torch 2.10.0+cpu` 下通过；
- `run_job.py` 的 job 解析、pytest 命令和 JUnit 合并 helper 自检通过；
- 用现有解释器替换 venv manager，执行 `tests/detector/test_detector.py`：
  `9 passed`，JUnit、phase 事实和 versions 输出均成功生成；
- `bash -n scripts/ci/bin/torchrun` 通过。

## 未执行项

未在本机完成全量 venv 安装和 CPU suite。当前机器没有 CI 使用的 Python 3.11
基础环境，`torch==2.10.0` 下载约 915 MB；为避免重复占用本地环境，安装验证
交给 PR CI。完整验证以 Actions 的 `cpu-unit-tests (<resource>)` matrix 为准。

## 预期 CI 行为

- `select` 先生成 selection 和 plan，再输出 `cpu_jobs`。
- `cpu-unit-tests` 只为 plan 中的 `(image, cpu resource)` 生成 box。
- job 内按 env 建 venv；同一 env 的多个 process-isolated pytest step 复用 venv。
- 任一 step 失败立即停止，后续 step 不继续占用 runner。
- venv 只在单个 job 内复用，不做跨 run 的 venv 缓存。
- NPU 两条旧 job 的入口、输出和 gate 语义不变。

