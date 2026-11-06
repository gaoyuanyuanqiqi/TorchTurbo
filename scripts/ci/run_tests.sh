#!/usr/bin/env bash
# pytest 泳道的全部逻辑：挑文件 → 分片 → marker 过滤 → 跑 → 出统计。
# CPU（裸机）与 NPU（container）两条泳道共用这一份 —— 它们唯一真正的差别是
# 「在哪台机器上跑」，跑什么、怎么切、怎么汇总必须完全一致，两份 yaml 迟早跑偏。
#
# 环境变量（由 workflow 传入）：
#   MODE        cpu | npu          cpu 加 -m "not npu"，npu 加 -m npu
#   TEST_FILES  JSON 数组；空 = 跑全部
#   TEST_PATH   单个 path / node id；非空则忽略 TEST_FILES 与 SHARD
#               `-` = 显式要「整条泳道」（dispatch 的默认值是具体路径，
#               传空串会被 GitHub 当成未提供而回落默认值，所以需要这个哨兵）
#   SHARD       "0/3" 形式；空 = 不切
#   PYTEST_ARGS 额外参数（按空格拆是有意的）
#   JUNIT_OUT   junit 输出路径
#   RESULT_JSON report_result.py 的结果文件
set -uo pipefail

CI_SCRIPTS="${CI_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODE="${MODE:-cpu}"
TEST_FILES="${TEST_FILES:-}"
TEST_PATH="${TEST_PATH:-}"
# `-` = 显式请求整条泳道。workflow_dispatch 的空字符串输入会被当成
# 「未提供」而回落成默认值（实测：传 "" 实际拿到 tests/ops/npu，
# 只跑了 1 个文件），所以手动入口需要一个不可能和真实路径冲突的哨兵。
if [ "$TEST_PATH" = "-" ]; then
  TEST_PATH=""
fi
SHARD="${SHARD:-}"
PYTEST_ARGS="${PYTEST_ARGS:--v --durations=20}"
JUNIT_OUT="${JUNIT_OUT:-/tmp/pytest.xml}"
RESULT_JSON="${RESULT_JSON:-/tmp/ci-result.json}"

python3 "$CI_SCRIPTS/report_result.py" init --file "$RESULT_JSON" \
  kind=pytest model_type="pytest-${MODE}" mode="$MODE" shard="${SHARD:--}"

# ---- 决定跑哪些文件 ---------------------------------------------------------
# 两件事必须在这里做完：分片（按文件轮转，片间用例数大致均衡）和
# 「文件列表为空就全量」（选择器没能给出结论时 fail-open）。
# 手动排查给的是 TEST_PATH（可能是 node id），它优先于一切。
if [ -n "$TEST_PATH" ]; then
  printf '%s\n' "$TEST_PATH" > /tmp/tt_pytest_files.txt
else
python3 - "$MODE" "$TEST_FILES" "$SHARD" >/tmp/tt_pytest_files.txt <<'PY'
import glob, json, subprocess, sys

mode, raw, shard = sys.argv[1], sys.argv[2], sys.argv[3]
files = json.loads(raw) if raw.strip() else []
if not files:
    files = sorted(glob.glob("tests/**/test_*.py", recursive=True))
    if mode == "npu":
        # npu 泳道只收「必须真跑在卡上」的文件。判据不在这里写：
        # 调 scripts/ci/npu_rule.py（与 conftest.py、select_plan.py 同一份），
        # 否则三处会慢慢漂，漂了也不报错，只会静默换掉跑的集合。
        # 全量 glob 后不筛会把整个 tests/ 导一遍，缺 peft / PIL 这类
        # 可选依赖的模块会变成 collection error，看起来像 NPU 坏了。
        out = subprocess.run(
            [sys.executable, "scripts/ci/npu_rule.py", "--filter"],
            input="\n".join(files), capture_output=True, text=True, check=True,
        ).stdout
        files = [line for line in out.splitlines() if line.strip()]
if shard:
    i, n = (int(x) for x in shard.split("/"))
    files = [f for k, f in enumerate(files) if k % n == i]
print("\n".join(files))
PY
fi

mapfile -t FILES < /tmp/tt_pytest_files.txt
if [ "${#FILES[@]}" -eq 0 ]; then
  echo "::error::没有可跑的测试文件（选择器和全量兜底都为空）"
  python3 "$CI_SCRIPTS/report_result.py" phase --file "$RESULT_JSON" \
    --name '④ pytest' --seconds 0 --rc 1
  exit 1
fi

echo "mode=${MODE} shard=${SHARD:--} files=${#FILES[@]}"
printf '  %s\n' "${FILES[@]}" | head -30

# ---- marker 过滤 ------------------------------------------------------------
# cpu：排除需要 torch_npu 的；npu：只要那些。
# 显式 TEST_PATH 时不适用：那是「我就要跑这一个」，marker 分流会把一个纯 CPU
# 文件在 npu 泳道里 deselect 干净，结果 rc=5 —— 看起来像泳道坏了，其实只是过滤。
if [ -n "$TEST_PATH" ]; then
  MARK=()
elif [ "$MODE" = "npu" ]; then
  MARK=(-m npu)
else
  MARK=(-m "not npu")
fi

echo "--- collected ---"
# shellcheck disable=SC2086  # PYTEST_ARGS 故意不进数组，要按空格拆
python -m pytest "${FILES[@]}" "${MARK[@]}" ${PYTEST_ARGS} --collect-only -q 2>&1 \
  | tail -20 || true

echo "--- run ---"
START=$(date +%s)
set +e
# shellcheck disable=SC2086
python -m pytest "${FILES[@]}" "${MARK[@]}" ${PYTEST_ARGS} \
  --junitxml="$JUNIT_OUT" 2>&1 | tee /tmp/pytest.log
RC=${PIPESTATUS[0]}
ELAPSED=$(( $(date +%s) - START ))

python3 "$CI_SCRIPTS/report_result.py" phase --file "$RESULT_JSON" \
  --name "④ pytest (${MODE})" --seconds "$ELAPSED" --rc "$RC"
echo "::notice::pytest mode=${MODE} rc=${RC} in ${ELAPSED}s"

echo "--- junit ---"
python3 "$CI_SCRIPTS/report_result.py" junit --junit "$JUNIT_OUT" || true

echo "rc=${RC}" >> "${GITHUB_OUTPUT:-/dev/null}"
exit "$RC"
