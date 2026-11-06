#!/usr/bin/env bash
# 装依赖。两条单测泳道只有这一处不同：裸机要自己补 torch，容器里那套
# NPU wheel 必须在装完之后验一遍来源。
#
# 子命令：
#   install   装 + 记 phase（MODE=cpu|npu 决定走哪条）
#
# 入参（环境变量）：
#   MODE                cpu | npu
#   TRANSFORMERS_SPEC   可选的额外版本约束（空 = 用镜像自带的）
#   RESULT_JSON         事实文件
#   TORCH_BEFORE        npu 泳道的 torch 快照（env_pod.py capture 产出）
#
# 为什么这些不能留在 yaml 里：`PIPESTATUS`、`tee` 与 `set -o pipefail` 的
# 组合很容易写错一步就把失败吞掉，而失败吞掉的后果是「装了一半也继续跑」，
# 几十个测试全红在一个和依赖无关的地方。

set -uo pipefail

CI_SCRIPTS="${CI_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RESULT_JSON="${RESULT_JSON:-/tmp/ci-result.json}"
MODE="${MODE:-cpu}"
TRANSFORMERS_SPEC="${TRANSFORMERS_SPEC:-}"
TORCH_BEFORE="${TORCH_BEFORE:-/tmp/torch-before.json}"
LOG="${LOG:-/tmp/turbo.log}"

result() { python3 "$CI_SCRIPTS/report_result.py" "$@"; }

# 跑一条 pip，日志 tee 出去（失败时是唯一线索），rc 走 PIPESTATUS。
pip_run() {                         # pip_run <首次?> <参数...>
  local first="$1"; shift
  if [ "$first" = yes ]; then
    python -m pip install "$@" 2>&1 | tee "$LOG"
  else
    python -m pip install "$@" 2>&1 | tee -a "$LOG"
  fi
  return "${PIPESTATUS[0]}"
}

# transformers 是传递依赖进来的：不显式打一行，版本漂了只会表现成一批红测试。
print_versions() {
  python -c "import transformers; print('transformers =', transformers.__version__)" \
    || echo "::warning::transformers 没装上"
}

cmd_install() {
  local start rc=0
  start=$(date +%s)

  if [ "$MODE" = "npu" ]; then
    # 容器里 torch 已经是自建的 NPU wheel。
    # --no-build-isolation：[build-system].requires 里有 torch，不加这行 pip
    # 会为了跑 setuptools 再下一份完整的 torch。
    pip_run yes --no-cache-dir --no-build-isolation -e ".[test]" || rc=$?
    print_versions
    if [ "$rc" -eq 0 ]; then
      # pip 解依赖时可能一声不响地把自建 wheel 换成 PyPI 的。
      python3 "$CI_SCRIPTS/env_pod.py" check --file "$TORCH_BEFORE" || rc=1
    fi
  else
    # 裸机：pyproject 钉的 torch 只有 NPU 镜像里才有，GH 托管机装不上，
    # 所以先装一个「能用的」torch，再装本包。
    pip_run yes torch || rc=$?
    if [ "$rc" -eq 0 ]; then
      # 与容器泳道用同一套 extras，否则同一条测试在两条泳道上结论会不同。
      pip_run no --no-build-isolation -e ".[test]" || rc=$?
    fi
    if [ "$rc" -eq 0 ] && [ -n "$TRANSFORMERS_SPEC" ]; then
      pip_run no "$TRANSFORMERS_SPEC" || rc=$?
    fi
    print_versions
  fi

  result phase --file "$RESULT_JSON" --name 'install' \
    --seconds "$(( $(date +%s) - start ))" --rc "$rc" || true
  echo "::notice::install mode=${MODE} rc=${rc} in $(( $(date +%s) - start ))s"
  return "$rc"
}

case "${1:-}" in
  install) cmd_install ;;
  *)
    echo "用法：$0 {install}" >&2
    exit 2
    ;;
esac
